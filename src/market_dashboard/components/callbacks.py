"""
Callback functions for Market Dashboard.
Contains all Dash callback definitions.
"""
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
import logging

from ..data import DatabaseManager
from ..ai import InsightsGenerator, OllamaClient
from ..utils.data_processing import resample_data, filter_date
from ..utils.formatting import pct_change_str, colorize_pct
from .charts import ChartGenerator
from .layouts import create_landing_layout, create_dashboard_layout
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.settings import DB_PATH, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TEMPERATURE

logger = logging.getLogger(__name__)


def register_callbacks(app, assets):
    """
    Register all callbacks for the Dash app.
    
    Args:
        app: Dash application instance
        assets: List of available asset symbols
    """
    # Initialize components
    db_manager = DatabaseManager(DB_PATH)
    ollama_client = OllamaClient(host=OLLAMA_HOST, model=OLLAMA_MODEL, temperature=OLLAMA_TEMPERATURE)
    insights_generator = InsightsGenerator(ollama_client)
    chart_generator = ChartGenerator()
    
    # Load data once
    df_all = db_manager.load_market_data()
    
    # Router callback
    @app.callback(
        Output("page-content-router", "children"),
        Input("url", "pathname")
    )
    def display_page(pathname):
        if pathname in ("/", None):
            return create_landing_layout()
        if pathname == "/dashboard":
            return create_dashboard_layout(assets)
        return create_landing_layout()
    
    # Tab visibility callback
    @app.callback(
        [Output('single-section', 'style'),
         Output('multi-section', 'style')],
        Input('dashboard-tabs', 'value')
    )
    def toggle_sections(tab):
        if tab == 'single-stock':
            return {"display": "block"}, {"display": "none"}
        return {"display": "none"}, {"display": "block"}
    
    # Show multi content only after valid selection
    @app.callback(
        Output("multi-content", "style"),
        Input("multi-asset-dropdown", "value")
    )
    def show_multi_content(symbols):
        if symbols and len(symbols) >= 2:
            return {"display": "block"}
        return {"display": "none"}
    
    # Single stock dashboard callback
    @app.callback(
        [Output("single-kpi-cards", "children"),
         Output("single-price-chart", "figure"),
         Output("single-ai-insights", "children")],
        [Input("single-asset-dropdown", "value"),
         Input("single-period-dropdown", "value"),
         Input("single-date-range", "value")]
    )
    def update_single_dashboard(symbol, period, date_range):
        try:
            # Validate inputs
            if period not in {"D", "W", "M"}:
                period = "D"
            if date_range not in {"30d", "90d", "ytd", "1y", "max"}:
                date_range = "1y"
            
            if not symbol or df_all.empty:
                return (
                    [dbc.Col(dbc.Card(dbc.CardBody("Select a stock to view data.")), width=12)],
                    go.Figure().update_layout(title="No data"),
                    "Select a stock to generate insights."
                )
            
            # Filter and process data
            df_daily = df_all[df_all["symbol"] == symbol].copy()
            if df_daily.empty:
                return (
                    [dbc.Col(dbc.Card(dbc.CardBody("No data available.")), width=12)],
                    go.Figure().update_layout(title="No data"),
                    "No data available for insights."
                )
            
            df_daily = filter_date(df_daily, date_range)
            df = resample_data(df_daily, period)
            
            if df.empty:
                return (
                    [dbc.Col(dbc.Card(dbc.CardBody("No data available for selection.")), width=12)],
                    go.Figure().update_layout(title="No data"),
                    "No data available for insights."
                )
            
            # Create KPI cards
            kpis = create_kpi_cards(df, period)
            
            # Create chart
            period_label = {"D": "Daily", "W": "Weekly", "M": "Monthly"}.get(period, period)
            fig = chart_generator.create_price_volume_chart(df, symbol, period_label)
            
            # Generate insights
            insights = insights_generator.generate_market_insights(df, symbol, period)
            
            return kpis, fig, insights
            
        except Exception as e:
            logger.exception(f"Single stock dashboard error: {e}")
            fallback_fig = go.Figure().update_layout(title="Error rendering chart")
            return (
                [dbc.Col(dbc.Card(dbc.CardBody("An error occurred while rendering this view.")), width=12)],
                fallback_fig,
                "Insights unavailable due to an error."
            )
    
    # Download CSV callback
    @app.callback(
        Output("download-dataframe-csv", "data"),
        Input("btn_csv", "n_clicks"),
        [State("single-asset-dropdown", "value"),
         State("single-period-dropdown", "value"),
         State("single-date-range", "value")],
        prevent_initial_call=True
    )
    def download_csv(n_clicks, symbol, period, date_range):
        if not symbol or df_all.empty:
            return None
        
        df_symbol = df_all[df_all["symbol"] == symbol].copy()
        df_symbol = filter_date(df_symbol, date_range)
        df_symbol = resample_data(df_symbol, period)
        
        if df_symbol.empty:
            return None
        
        return dcc.send_data_frame(df_symbol.to_csv, f"{symbol}_{period}_{date_range}.csv", index=False)

    # Multi-stock comparison callback
    @app.callback(
        [Output("comparison-chart", "figure"),
         Output("comparison-heatmap", "figure"),
         Output("comparison-boxplot", "figure"),
         Output("comparison-insights-container", "children")],
        [Input("multi-asset-dropdown", "value"),
         Input("multi-period-dropdown", "value"),
         Input("multi-date-range", "value"),
         Input("multi-options", "value")],
    )
    def update_comparison_dashboard(symbols, period, date_range, multi_opts):
        try:
            # Validate inputs
            if not symbols or len(symbols) < 2 or df_all.empty:
                empty_fig = go.Figure().update_layout(title="Select at least 2 stocks to compare")
                default_card = dbc.Card([
                    dbc.CardHeader("Comparative Insights"),
                    dbc.CardBody("Select at least 2 stocks to generate insights.")
                ])
                return empty_fig, empty_fig, empty_fig, default_card

            # Build comparison chart
            normalize = (multi_opts is not None) and ("norm" in set(multi_opts))
            comp_fig = chart_generator.create_comparison_chart(df_all, symbols, period, date_range, normalize)

            # Build correlation matrix for heatmap
            returns_matrix = {}
            for sym in symbols:
                df_sym = df_all[df_all["symbol"] == sym].copy()
                df_sym = filter_date(df_sym, date_range)
                df_sym = resample_data(df_sym, period)
                if not df_sym.empty:
                    returns_matrix[sym] = df_sym.set_index("date")["close"].pct_change()

            # Create charts
            returns_df = pd.DataFrame(returns_matrix).dropna(how="all").dropna(axis=1, how="all")
            heatmap_fig = chart_generator.create_correlation_heatmap(returns_df)

            period_label = {"D": "Daily", "W": "Weekly", "M": "Monthly"}[period]
            boxplot_fig = chart_generator.create_returns_boxplot(returns_df, period_label)

            # Generate insights
            insights = insights_generator.generate_comparative_insights(df_all, symbols, period)
            insights_card = dbc.Card([
                dbc.CardHeader("Comparative Insights"),
                dbc.CardBody(insights)
            ])

            return comp_fig, heatmap_fig, boxplot_fig, insights_card

        except Exception as e:
            logger.exception(f"Multi-stock dashboard error: {e}")
            empty_fig = go.Figure().update_layout(title="Error rendering chart")
            error_card = dbc.Card([
                dbc.CardHeader("Comparative Insights"),
                dbc.CardBody("Error generating insights.")
            ])
            return empty_fig, empty_fig, empty_fig, error_card


def create_kpi_cards(df, period):
    """Create KPI cards for single stock view."""
    unit = {"D": "day", "W": "week", "M": "month"}.get(period, "period")
    
    latest = df.iloc[-1]
    prev_short = df.iloc[-2] if len(df) >= 2 else None
    prev_medium = df.iloc[-6] if len(df) >= 6 else None
    prev_long = df.iloc[-21] if len(df) >= 21 else None
    
    kpi_short = pct_change_str(latest['close'], prev_short['close']) if prev_short is not None else "–"
    kpi_med = pct_change_str(latest['close'], prev_medium['close']) if prev_medium is not None else "–"
    kpi_long = pct_change_str(latest['close'], prev_long['close']) if prev_long is not None else "–"
    
    return [
        dbc.Col(dbc.Card([
            dbc.CardHeader("Close Price"),
            dbc.CardBody(f"${latest['close']:,.2f}")
        ], color="light", className="shadow-sm"), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader(f"1 {unit.capitalize()} Change"),
            dbc.CardBody(html.Span(kpi_short, style=colorize_pct(kpi_short)))
        ], color="light", className="shadow-sm"), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader(f"5 {unit.capitalize()}s Change"),
            dbc.CardBody(html.Span(kpi_med, style=colorize_pct(kpi_med)))
        ], color="light", className="shadow-sm"), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader(f"20 {unit.capitalize()}s Change"),
            dbc.CardBody(html.Span(kpi_long, style=colorize_pct(kpi_long)))
        ], color="light", className="shadow-sm"), width=3),
    ]
