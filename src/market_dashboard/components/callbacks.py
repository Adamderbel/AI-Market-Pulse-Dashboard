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
         Output('multi-section', 'style'),
         Output('forecast-section', 'style')],
        Input('dashboard-tabs', 'value')
    )
    def toggle_sections(tab):
        if tab == 'single-stock':
            return {"display": "block"}, {"display": "none"}, {"display": "none"}
        elif tab == 'multi-stock':
            return {"display": "none"}, {"display": "block"}, {"display": "none"}
        elif tab == 'forecasting':
            return {"display": "none"}, {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}, {"display": "none"}
    
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

    # Forecasting callbacks
    @app.callback(
        [Output("forecast-summary-cards", "children"),
         Output("forecast-chart", "figure"),
         Output("forecast-insights", "children")],
        Input("btn-forecast", "n_clicks"),
        [State("forecast-asset-dropdown", "value"),
         State("forecast-period-dropdown", "value"),
         State("forecast-model-dropdown", "value")],
        prevent_initial_call=True
    )
    def update_forecast(n_clicks, symbol, days_ahead, model_type):
        try:
            if not symbol or df_all.empty:
                return (
                    [dbc.Col(dbc.Card(dbc.CardBody("Select a stock to generate forecast.")), width=12)],
                    go.Figure().update_layout(title="No data"),
                    "Select a stock and click 'Forecast' to generate predictions."
                )

            # Get data for the selected symbol
            df_symbol = df_all[df_all["symbol"] == symbol].copy()
            if df_symbol.empty or len(df_symbol) < 60:
                return (
                    [dbc.Col(dbc.Card(dbc.CardBody("Insufficient data for forecasting.")), width=12)],
                    go.Figure().update_layout(title="Insufficient data"),
                    "Need at least 60 days of data for reliable forecasting."
                )

            # Import forecasting utilities
            from ..utils.forecasting import StockForecaster, simple_trend_forecast

            # Generate forecast based on model type
            if model_type == "trend":
                # Use simple trend forecast
                forecast_data = simple_trend_forecast(df_symbol, days_ahead)
                forecast_chart = chart_generator.create_forecast_chart(
                    df_symbol, symbol, forecast_data, days_ahead
                )

                # Create summary cards for trend forecast
                current_price = forecast_data['current_price']
                forecast_price = forecast_data['trend_forecast']
                change_pct = ((forecast_price - current_price) / current_price) * 100

                summary_cards = [
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Current Price"),
                        dbc.CardBody(f"${current_price:.2f}")
                    ], color="light"), width=3),

                    dbc.Col(dbc.Card([
                        dbc.CardHeader(f"Forecast ({days_ahead}d)"),
                        dbc.CardBody(f"${forecast_price:.2f}")
                    ], color="light"), width=3),

                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Expected Change"),
                        dbc.CardBody(f"{change_pct:+.2f}%")
                    ], color="light"), width=3),

                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Confidence"),
                        dbc.CardBody(f"{forecast_data['confidence']:.0f}%")
                    ], color="light"), width=3),
                ]

                insights = f"""
                **Trend Analysis for {symbol.upper()}:**

                Based on recent price movements, the trend is **{forecast_data['trend_direction']}**.

                **Forecast Summary:**
                - Current Price: ${current_price:.2f}
                - {days_ahead}-Day Target: ${forecast_price:.2f}
                - Expected Change: {change_pct:+.2f}%
                - Confidence Level: {forecast_data['confidence']:.0f}%

                **Range Estimate:**
                - Lower Bound: ${forecast_data['trend_lower']:.2f}
                - Upper Bound: ${forecast_data['trend_upper']:.2f}

                *Note: This is a simple trend-based forecast. Market conditions can change rapidly.*
                """

            else:
                # Use machine learning models
                forecaster = StockForecaster()

                try:
                    # Train the models
                    metrics = forecaster.train(df_symbol, days_ahead)

                    # Generate predictions for the selected period
                    predictions = forecaster.predict(df_symbol, days_ahead)

                    # Select the requested model prediction and ensure chart consistency
                    if model_type in predictions:
                        forecast_price = predictions[model_type]
                        # Create a focused prediction dict for the chart to show the selected model
                        chart_predictions = {
                            model_type: predictions[model_type],
                            f'{model_type}_lower': predictions.get(f'{model_type}_lower', predictions[model_type] * 0.95),
                            f'{model_type}_upper': predictions.get(f'{model_type}_upper', predictions[model_type] * 1.05)
                        }
                    elif 'ensemble' in predictions:
                        forecast_price = predictions['ensemble']
                        chart_predictions = {
                            'ensemble': predictions['ensemble'],
                            'ensemble_lower': predictions.get('ensemble_lower', predictions['ensemble'] * 0.95),
                            'ensemble_upper': predictions.get('ensemble_upper', predictions['ensemble'] * 1.05)
                        }
                    else:
                        # Fallback to any available prediction
                        available_models = [k for k in predictions.keys() if not k.endswith('_lower') and not k.endswith('_upper')]
                        if available_models:
                            forecast_price = predictions[available_models[0]]
                            chart_predictions = {available_models[0]: forecast_price}
                        else:
                            forecast_price = df_symbol['close'].iloc[-1]
                            chart_predictions = {'current': forecast_price}

                    # Create forecast chart with consistent prediction
                    forecast_chart = chart_generator.create_forecast_chart(
                        df_symbol, symbol, chart_predictions, days_ahead
                    )

                    # Create summary cards with multiple predictions
                    current_price = df_symbol['close'].iloc[-1]
                    change_pct = ((forecast_price - current_price) / current_price) * 100

                    # Get model accuracy
                    model_key = model_type if model_type in metrics else 'linear'
                    if model_key in metrics and 'mape' in metrics[model_key]:
                        accuracy = max(0, 100 - metrics[model_key]['mape'])
                    else:
                        accuracy = 75  # Default accuracy

                    # Create clean summary cards
                    summary_cards = [
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Current Price"),
                            dbc.CardBody(f"${current_price:.2f}")
                        ], color="light"), width=3),

                        dbc.Col(dbc.Card([
                            dbc.CardHeader(f"Forecast ({days_ahead} day{'s' if days_ahead > 1 else ''})"),
                            dbc.CardBody(f"${forecast_price:.2f}")
                        ], color="light"), width=3),

                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Expected Change"),
                            dbc.CardBody(f"{change_pct:+.2f}%")
                        ], color="light"), width=3),

                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Model Accuracy"),
                            dbc.CardBody(f"{accuracy:.1f}%")
                        ], color="light"), width=3),
                    ]

                    # Generate AI-powered insights using Ollama
                    try:
                        # Prepare data for AI analysis
                        direction = "upward" if change_pct > 0 else "downward" if change_pct < 0 else "sideways"
                        confidence_level = "high" if accuracy > 80 else "moderate" if accuracy > 60 else "low"

                        # Get recent price data for context
                        recent_data = df_symbol.tail(10)
                        price_trend = "increasing" if recent_data['close'].iloc[-1] > recent_data['close'].iloc[0] else "decreasing"
                        volatility = recent_data['close'].pct_change().std() * 100

                        # Prepare model performance summary (Prophet removed)
                        working_models = []
                        for model_name in ['linear', 'arima']:
                            if model_name in metrics and 'mape' in metrics[model_name]:
                                mape_val = metrics[model_name]['mape']
                                if isinstance(mape_val, (int, float)):
                                    model_accuracy = max(0, 100 - mape_val)
                                    working_models.append(f"{model_name}: {model_accuracy:.0f}% accuracy")

                        # Create simplified prompt for Ollama
                        prompt = f"""
Analyze this stock forecast for {symbol.upper()}:

- Current Price: ${current_price:.2f}
- Predicted Price: ${forecast_price:.2f} ({change_pct:+.2f}% in {days_ahead} days)
- Model: {model_type.replace('_', ' ').title()} with {accuracy:.1f}% accuracy
- Recent trend: {price_trend}

Provide a brief 2-3 sentence analysis of what happened in the recent period and what this prediction means. Keep it concise and professional.
"""

                        # Generate insights using Ollama
                        insights = insights_generator.generate_market_insights(
                            df_symbol, symbol, "custom"
                        )

                        # If Ollama fails, use the custom prompt
                        if "Ollama server is not available" in insights or "Unable to generate insights" in insights:
                            try:
                                # Try direct Ollama call
                                ai_insights = ollama_client.chat(
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": "You are a professional financial analyst. Provide a brief 2-3 sentence analysis of the stock forecast. Be concise and professional."
                                        },
                                        {"role": "user", "content": prompt}
                                    ],
                                    temperature=0.3,
                                    max_tokens=150
                                )
                                # Format in the requested style
                                insights = f"""**🤖 AI-Powered Forecast Analysis for {symbol.upper()}**

**Prediction:** ${current_price:.2f} → ${forecast_price:.2f} ({change_pct:+.2f}% in {days_ahead} day{'s' if days_ahead > 1 else ''})

**Model:** {model_type.replace('_', ' ').title()} with {accuracy:.1f}% accuracy

{ai_insights}"""
                            except Exception as e:
                                logger.warning(f"Ollama direct call failed: {e}")
                                # Fallback to simple analysis
                                insights = f"""
**� {symbol.upper()} Forecast Analysis**

**Prediction Summary:**
The {model_type.replace('_', ' ').title()} model predicts {symbol.upper()} will move from ${current_price:.2f} to ${forecast_price:.2f} over {days_ahead} days ({change_pct:+.2f}%).

**Analysis:**
- **Movement:** {direction.title()} trend with {"significant" if abs(change_pct) > 3 else "moderate" if abs(change_pct) > 1 else "minimal"} change
- **Confidence:** {confidence_level.title()} ({accuracy:.1f}% model accuracy)
- **Risk Level:** {"High" if abs(change_pct) > 5 else "Moderate" if abs(change_pct) > 2 else "Low"} volatility expected

**Key Takeaway:**
{symbol.upper()} shows a {direction} trend with {confidence_level} confidence. Consider this alongside other market factors.

*Note: AI analysis unavailable - using statistical summary.*
"""
                        else:
                            # Format the AI insights in the requested style
                            insights = f"""**🤖 AI-Powered Forecast Analysis for {symbol.upper()}**

**Prediction:** ${current_price:.2f} → ${forecast_price:.2f} ({change_pct:+.2f}% in {days_ahead} day{'s' if days_ahead > 1 else ''})

**Model:** {model_type.replace('_', ' ').title()} with {accuracy:.1f}% accuracy

{insights}"""

                    except Exception as e:
                        logger.error(f"Error generating AI insights: {e}")
                        # Simple fallback
                        insights = f"""
**� {symbol.upper()} Forecast Summary**

**Prediction:** ${current_price:.2f} → ${forecast_price:.2f} ({change_pct:+.2f}% over {days_ahead} days)

**Analysis:** The {model_type.replace('_', ' ').title()} model suggests a {direction} movement with {accuracy:.1f}% historical accuracy.

**Risk Level:** {"High" if abs(change_pct) > 5 else "Moderate" if abs(change_pct) > 2 else "Low"} volatility expected.

*AI analysis temporarily unavailable.*
"""

                except Exception as e:
                    logger.error(f"ML forecasting failed: {e}")
                    # Fallback to simple trend forecast
                    forecast_data = simple_trend_forecast(df_symbol, days_ahead)
                    forecast_chart = chart_generator.create_forecast_chart(
                        df_symbol, symbol, forecast_data, days_ahead
                    )

                    current_price = forecast_data['current_price']
                    forecast_price = forecast_data['trend_forecast']
                    change_pct = ((forecast_price - current_price) / current_price) * 100

                    summary_cards = [
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Current Price"),
                            dbc.CardBody(f"${current_price:.2f}")
                        ], color="light"), width=3),

                        dbc.Col(dbc.Card([
                            dbc.CardHeader(f"Trend Forecast ({days_ahead}d)"),
                            dbc.CardBody(f"${forecast_price:.2f}")
                        ], color="light"), width=3),

                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Expected Change"),
                            dbc.CardBody(f"{change_pct:+.2f}%")
                        ], color="light"), width=3),

                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Method"),
                            dbc.CardBody("Trend Analysis")
                        ], color="light"), width=3),
                    ]

                    insights = f"""
                    **Fallback Trend Forecast for {symbol.upper()}:**

                    ML models unavailable, using trend analysis.

                    **Forecast Summary:**
                    - Current Price: ${current_price:.2f}
                    - {days_ahead}-Day Target: ${forecast_price:.2f}
                    - Expected Change: {change_pct:+.2f}%

                    *Note: This is a simplified forecast. For better accuracy, ensure sufficient historical data.*
                    """

            return summary_cards, forecast_chart, insights

        except Exception as e:
            logger.exception(f"Forecasting error: {e}")
            fallback_fig = go.Figure().update_layout(title="Error generating forecast")
            return (
                [dbc.Col(dbc.Card(dbc.CardBody("An error occurred while generating the forecast.")), width=12)],
                fallback_fig,
                "Unable to generate forecast due to an error."
            )


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
