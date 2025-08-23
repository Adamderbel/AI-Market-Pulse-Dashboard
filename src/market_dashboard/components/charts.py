"""
Chart generation utilities for Market Dashboard.
Contains functions for creating various chart types.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any
import logging

from ..utils.formatting import get_chart_colors
from ..utils.data_processing import normalized_series
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.settings import DEFAULT_CHART_HEIGHT, DEFAULT_CHART_TEMPLATE

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generates various chart types for market data visualization."""
    
    def __init__(self):
        """Initialize chart generator with default settings."""
        self.colors = get_chart_colors()
        self.default_height = DEFAULT_CHART_HEIGHT
        self.template = DEFAULT_CHART_TEMPLATE
    
    def create_price_volume_chart(
        self,
        df: pd.DataFrame,
        symbol: str,
        period_label: str,
        show_ma: bool = True
    ) -> go.Figure:
        """
        Create a price and volume chart with candlesticks and moving averages.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Asset symbol
            period_label: Period label for title
            show_ma: Whether to show moving averages
            
        Returns:
            Plotly figure object
        """
        try:
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.7, 0.3], vertical_spacing=0.08
            )
            
            # Candlestick chart (fallback to line if OHLC not available)
            has_ohlc = all(col in df.columns for col in ["open", "high", "low", "close"])
            
            if has_ohlc:
                fig.add_trace(go.Candlestick(
                    x=df["date"],
                    open=df["open"], high=df["high"],
                    low=df["low"], close=df["close"],
                    name="Price",
                    increasing=dict(line=dict(color=self.colors["positive"])),
                    decreasing=dict(line=dict(color=self.colors["negative"]))
                ), row=1, col=1)
            else:
                # Fallback to line chart if only close price available
                y_series = df["close"] if "close" in df.columns else pd.Series([], dtype=float)
                fig.add_trace(go.Scatter(
                    x=df["date"],
                    y=y_series,
                    mode="lines",
                    name="Price",
                    line=dict(width=2, color=self.colors["positive"])
                ), row=1, col=1)
            
            # Moving averages
            if show_ma and "close" in df.columns and len(df) >= 10:
                self._add_moving_averages(fig, df)
            
            # Volume chart
            if "volume" in df.columns and df["volume"].notna().any():
                self._add_volume_chart(fig, df, has_ohlc)
            
            # Update layout
            fig.update_layout(
                title=f"{symbol.upper()} Price & Volume ({period_label})",
                template=self.template,
                height=self.default_height,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            
            # Update axes
            fig.update_xaxes(
                title_text="Date",
                showspikes=True, spikemode="across", spikesnap="cursor",
                rangeslider=dict(visible=True),
                row=1, col=1
            )
            fig.update_yaxes(title_text="Price (USD)", tickprefix="$", tickformat=",.2f", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price/volume chart: {e}")
            return go.Figure().update_layout(title="Error rendering chart")
    
    def _add_moving_averages(self, fig: go.Figure, df: pd.DataFrame) -> None:
        """Add moving averages to the chart."""
        # 20-period MA
        ma20 = df["close"].rolling(window=20, min_periods=5).mean()
        if ma20.notna().sum() > 0:
            fig.add_trace(go.Scatter(
                x=df["date"], y=ma20, mode="lines",
                name="MA(20)", line=dict(color=self.colors["ma20"], width=1.8)
            ), row=1, col=1)
        
        # 50-period MA
        ma50 = df["close"].rolling(window=50, min_periods=10).mean()
        if ma50.notna().sum() > 0:
            fig.add_trace(go.Scatter(
                x=df["date"], y=ma50, mode="lines",
                name="MA(50)", line=dict(color=self.colors["ma50"], width=1.8, dash="dash")
            ), row=1, col=1)
    
    def _add_volume_chart(self, fig: go.Figure, df: pd.DataFrame, has_ohlc: bool) -> None:
        """Add volume chart to the figure."""
        if has_ohlc and {"open", "close"}.issubset(df.columns):
            # Color volume bars based on price direction
            vol_colors = [
                self.colors["volume_up"] if (pd.notna(c) and pd.notna(o) and c >= o) 
                else self.colors["volume_down"]
                for c, o in zip(df["close"], df["open"])
            ]
        else:
            # Default color if we can't determine direction
            vol_colors = [self.colors["volume_neutral"]] * len(df)
        
        fig.add_trace(go.Bar(
            x=df["date"], y=df["volume"],
            name="Volume", marker=dict(color=vol_colors)
        ), row=2, col=1)
    
    def create_comparison_chart(
        self,
        df_all: pd.DataFrame,
        symbols: List[str],
        period: str,
        date_range: str,
        normalize: bool = True
    ) -> go.Figure:
        """
        Create a multi-asset comparison chart.
        
        Args:
            df_all: DataFrame with all market data
            symbols: List of symbols to compare
            period: Time period
            date_range: Date range
            normalize: Whether to normalize prices to 100
            
        Returns:
            Plotly figure object
        """
        try:
            from ..utils.data_processing import filter_date, resample_data
            
            fig = go.Figure()
            
            for sym in symbols:
                df_sym = df_all[df_all["symbol"] == sym].copy()
                df_sym = filter_date(df_sym, date_range)
                df_sym = resample_data(df_sym, period)
                
                if df_sym.empty:
                    continue
                
                if normalize:
                    y_vals = normalized_series(df_sym["close"])
                else:
                    y_vals = df_sym["close"]
                
                fig.add_trace(go.Scatter(
                    x=df_sym["date"], y=y_vals, mode='lines',
                    name=sym.upper(),
                    line=dict(width=2),
                    hoverinfo="x+y+name"
                ))
            
            period_label = {"D": "Daily", "W": "Weekly", "M": "Monthly"}[period]
            y_title = "Normalized Price (%)" if normalize else "Price (USD)"
            title_suffix = " - Normalized to 100" if normalize else ""
            
            fig.update_layout(
                title=f"Stock Performance Comparison ({period_label}){title_suffix}",
                xaxis_title="Date",
                yaxis_title=y_title,
                template=self.template,
                height=500,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            return go.Figure().update_layout(title="Error rendering comparison chart")
    
    def create_correlation_heatmap(self, returns_df: pd.DataFrame) -> go.Figure:
        """
        Create a correlation heatmap for asset returns.
        
        Args:
            returns_df: DataFrame with asset returns
            
        Returns:
            Plotly figure object
        """
        try:
            if returns_df.shape[1] < 2:
                return go.Figure().update_layout(title="Correlation Heatmap (not enough data)")
            
            corr = returns_df.corr()
            labels = [c.upper() for c in corr.columns]
            
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=labels,
                y=labels,
                colorscale="RdBu",
                zmin=-1, zmax=1, zmid=0,
                colorbar=dict(title="Correlation"),
                text=corr.round(2).values,
                texttemplate="%{text}",
                showscale=True
            ))
            
            fig.update_layout(
                title="Correlation Heatmap of Returns",
                height=450,
                template=self.template
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return go.Figure().update_layout(title="Error rendering heatmap")
    
    def create_returns_boxplot(self, returns_df: pd.DataFrame, period_label: str) -> go.Figure:
        """
        Create a boxplot of returns distribution.
        
        Args:
            returns_df: DataFrame with asset returns
            period_label: Period label for title
            
        Returns:
            Plotly figure object
        """
        try:
            if returns_df.empty or returns_df.shape[1] < 1:
                return go.Figure().update_layout(title="Returns Distribution (not enough data)")
            
            box_traces = []
            for col in returns_df.columns:
                series_pct = (returns_df[col].dropna() * 100.0)
                if not series_pct.empty:
                    box_traces.append(go.Box(
                        y=series_pct,
                        name=col.upper(),
                        boxmean='sd',
                        marker_color=self.colors["neutral"]
                    ))
            
            fig = go.Figure(data=box_traces)
            fig.update_layout(
                title=f"Distribution of {period_label} Returns (%)",
                yaxis_title="Return (%)",
                template=self.template,
                height=450,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating returns boxplot: {e}")
            return go.Figure().update_layout(title="Error rendering boxplot")
