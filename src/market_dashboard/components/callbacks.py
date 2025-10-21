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
from ..ai.openrouter_client import InsightsGenerator
from ..ai.openrouter_client_impl import OpenRouterClient
from ..utils.data_processing import resample_data, filter_date
from ..utils.formatting import pct_change_str, colorize_pct
from .charts import ChartGenerator
from .layouts import create_landing_layout, create_dashboard_layout
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.settings import DB_PATH, OPENROUTER_API_KEY


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
    openrouter_client = OpenRouterClient(api_key=OPENROUTER_API_KEY)
    insights_generator = InsightsGenerator(openrouter_client)
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
            # Landing page as default
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
        Output("single-results-container", "children"),
        [Input("btn-analyze-single", "n_clicks")],
        [State("single-asset-dropdown", "value"),
         State("single-date-range", "value")],
        prevent_initial_call=True
    )
    def update_single_dashboard(n_clicks, symbol, date_range):
        try:
            # Check if button was clicked
            if not n_clicks:
                return html.Div()

            # Always use daily period
            period = "D"

            # Validate inputs
            if date_range not in {"30d", "90d", "ytd", "1y", "max"}:
                date_range = "1y"

            if not symbol or df_all is None or df_all.empty:
                return html.Div(
                    dbc.Card(dbc.CardBody("Select a stock to view data."))
                )

            # Filter and process data
            df_daily = df_all[df_all["symbol"] == symbol].copy()
            if df_daily.empty:
                return html.Div(
                    dbc.Card(dbc.CardBody("No data available."))
                )

            df_daily = filter_date(df_daily, date_range)
            df = resample_data(df_daily, period)

            if df is None or df.empty:
                return html.Div(
                    dbc.Card(dbc.CardBody("No data available for selection."))
                )

            # Create KPI cards
            kpis = create_kpi_cards(df, period)

            # Create chart
            period_label = {"D": "Daily", "W": "Weekly", "M": "Monthly"}.get(period, period)
            fig = chart_generator.create_price_volume_chart(df, symbol, period_label)

            # Generate insights (synchronous call)
            try:
                insights = insights_generator.generate_market_insights(df, symbol, period)
            except Exception as e:
                logger.warning(f"AI insights generation failed, falling back to simple message: {e}")
                insights = "AI insights unavailable."

            # Create the complete single stock layout
            single_content = html.Div([
                dbc.Row(kpis, className="mb-3"),
                dbc.Row([dbc.Col(dcc.Graph(figure=fig), width=12)]),
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader("AI Insights"),
                            dbc.CardBody(dcc.Markdown(insights))
                        ]),
                        width=12
                    )
                ], className="mt-3")
            ])

            return single_content

        except Exception as e:
            logger.exception(f"Single stock dashboard error: {e}")
            error_content = html.Div([
                dbc.Alert(
                    "An error occurred while generating the analysis. Please try again.",
                    color="danger"
                )
            ])
            return error_content

    # Download CSV callback
    @app.callback(
        Output("download-dataframe-csv", "data"),
        Input("btn_csv", "n_clicks"),
        [State("single-asset-dropdown", "value"),
         State("single-date-range", "value")],
        prevent_initial_call=True
    )
    def download_csv(n_clicks, symbol, date_range):
        if not symbol or df_all is None or df_all.empty:
            return None

        # Always use daily period
        period = "D"

        df_symbol = df_all[df_all["symbol"] == symbol].copy()
        df_symbol = filter_date(df_symbol, date_range)
        df_symbol = resample_data(df_symbol, period)

        if df_symbol is None or df_symbol.empty:
            return None

        return dcc.send_data_frame(df_symbol.to_csv, f"{symbol}_{period}_{date_range}.csv", index=False)

    # Multi-stock comparison callback
    @app.callback(
        Output("multi-results-container", "children"),
        [Input("btn-compare-multi", "n_clicks")],
        [State("multi-asset-dropdown", "value"),
         State("multi-date-range", "value")],
        prevent_initial_call=True
    )
    def update_comparison_dashboard(n_clicks, symbols, date_range):
        try:
            # Check if button was clicked
            if not n_clicks:
                return html.Div()

            # Always use daily period and normalize to 100
            period = "D"
            multi_opts = ["norm"]  # Always normalize

            # Validate inputs
            if not symbols or len(symbols) < 2 or df_all is None or df_all.empty:
                error_content = html.Div([
                    dbc.Alert(
                        "Please select at least 2 stocks to generate comparison.",
                        color="warning"
                    )
                ])
                return error_content

            # Build comparison chart
            normalize = (multi_opts is not None) and ("norm" in set(multi_opts))
            comp_fig = chart_generator.create_comparison_chart(df_all, symbols, period, date_range, normalize)

            # Build correlation matrix for heatmap
            returns_matrix = {}
            for sym in symbols:
                df_sym = df_all[df_all["symbol"] == sym].copy()
                df_sym = filter_date(df_sym, date_range)
                df_sym = resample_data(df_sym, period)
                if df_sym is not None and not df_sym.empty:
                    returns_matrix[sym] = df_sym.set_index("date")["close"].pct_change()

            # Create charts
            returns_df = pd.DataFrame(returns_matrix).dropna(how="all").dropna(axis=1, how="all")
            heatmap_fig = chart_generator.create_correlation_heatmap(returns_df)

            period_label = {"D": "Daily", "W": "Weekly", "M": "Monthly"}[period]
            boxplot_fig = chart_generator.create_returns_boxplot(returns_df, period_label)

            # Generate insights (synchronous call)
            try:
                insights_text = insights_generator.generate_comparative_insights(df_all, symbols, period)
            except Exception as e:
                logger.warning(f"Comparative AI insights generation failed: {e}")
                insights_text = "Comparative insights unavailable."

            insights_card = dbc.Card([
                dbc.CardHeader("Comparative Insights"),
                dbc.CardBody(dcc.Markdown(insights_text))
            ])

            # Create the complete multi-stock layout
            multi_content = html.Div([
                dbc.Row([
                    dbc.Col(
                        dcc.Checklist(
                            id="multi-options",
                            options=[{"label": "Normalize to 100", "value": "norm"}],
                            value=["norm"],
                            inline=True
                        ),
                        width=12
                    )
                ], className="mb-2"),
                insights_card,
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=comp_fig), width=12),
                    dbc.Col(dcc.Graph(figure=heatmap_fig), width=6),
                    dbc.Col(dcc.Graph(figure=boxplot_fig), width=6),
                ], className="mt-3")
            ])

            return multi_content

        except Exception as e:
            logger.exception(f"Multi-stock dashboard error: {e}")
            error_content = html.Div([
                dbc.Alert(
                    "An error occurred while generating the comparison. Please try again.",
                    color="danger"
                )
            ])
            return error_content

    # Forecasting callbacks with persistence
    @app.callback(
        [Output("forecast-results-container", "children"),
         Output("forecast-results-store", "data")],
        Input("btn-forecast", "n_clicks"),
        [State("forecast-asset-dropdown", "value"),
         State("forecast-period-dropdown", "value"),
         State("forecast-model-dropdown", "value"),
         State("forecast-results-store", "data")],
        prevent_initial_call=True
    )
    def update_forecast(n_clicks, symbol, days_ahead, model_type, stored_results):
        try:
            # If no button click, return stored results or empty
            if not n_clicks:
                if stored_results:
                    return stored_results.get("content", html.Div()), stored_results
                else:
                    return html.Div(), {}

            if not symbol or df_all is None or df_all.empty:
                empty_content = html.Div([
                    dbc.Alert("Please select a stock to generate forecast.", color="warning")
                ])
                return empty_content, {}

            # Get data for the selected symbol
            df_symbol = df_all[df_all["symbol"] == symbol].copy()
            if df_symbol is None or df_symbol.empty or len(df_symbol) < 60:
                empty_content = html.Div([
                    dbc.Col(dbc.Card(dbc.CardBody("Insufficient data for forecasting.")), width=12)
                ])
                # Two outputs: container content and stored data (empty)
                return empty_content, {}

            # Import forecasting utilities
            from ..utils.forecasting import StockForecaster, simple_trend_forecast

            # Default placeholders
            forecast_chart = None
            summary_cards = []
            insights = "Forecast insights unavailable."

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
                        dbc.CardBody(f"{forecast_data.get('confidence', 0):.0f}%")
                    ], color="light"), width=3),
                ]

                insights = f"""
**Trend Analysis for {symbol.upper()}:**

Based on recent price movements, the trend is **{forecast_data.get('trend_direction', 'unknown')}**.

**Forecast Summary:**
- Current Price: ${current_price:.2f}
- {days_ahead}-Day Target: ${forecast_price:.2f}
- Expected Change: {change_pct:+.2f}%

**Range Estimate:**
- Lower Bound: ${forecast_data.get('trend_lower', 0):.2f}
- Upper Bound: ${forecast_data.get('trend_upper', 0):.2f}

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

                    # Generate AI-powered insights using OpenRouter (synchronous)
                    try:
                        # Try using a synchronous insights generation method if available
                        try:
                            ai_insights = insights_generator.generate_forecast_insights(
                                df_symbol, symbol, current_price, forecast_price,
                                days_ahead, model_type, accuracy
                            )
                        except AttributeError:
                            # Fallback: use synchronous wrapper to avoid coroutine misuse
                            ai_insights = openrouter_client.chat_sync(
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "You are a professional financial analyst. Provide a brief 2-3 sentence analysis of the stock forecast. Be concise and professional."
                                    },
                                    {"role": "user", "content": f"Analyze {symbol.upper()}: Current ${current_price:.2f}, Predicted ${forecast_price:.2f} ({change_pct:+.2f}% in {days_ahead} days). Model: {model_type} ({accuracy:.1f}% accuracy)."}
                                ],
                                temperature=0.3,
                                max_tokens=150
                            )

                        # Format the forecast insights in the requested style
                        insights = f"""**🤖 AI-Powered Forecast Analysis for {symbol.upper()}**

**Prediction:** ${current_price:.2f} → ${forecast_price:.2f} ({change_pct:+.2f}% in {days_ahead} day{'s' if days_ahead > 1 else ''})

**Model:** {model_type.replace('_', ' ').title()} with {accuracy:.1f}% accuracy

{ai_insights}"""
                    except Exception as e:
                        logger.warning(f"OpenRouter API call failed or synchronous call not available: {e}")
                        # Fallback to simple analysis
                        direction = "upward" if change_pct > 0 else "downward" if change_pct < 0 else "sideways"
                        confidence_level = "high" if accuracy > 80 else "moderate" if accuracy > 60 else "low"
                        insights = f"""
**{symbol.upper()} Forecast Analysis**

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

            # Create the complete forecast layout
            forecast_content = html.Div([
                dbc.Row(summary_cards, className="mb-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=forecast_chart), width=12)
                ]),
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader("Forecast Analysis"),
                            dbc.CardBody(dcc.Markdown(insights))
                        ]),
                        width=12
                    )
                ], className="mt-3")
            ])

            # Store the results for persistence
            stored_data = {
                "content": forecast_content,
                "symbol": symbol,
                "days_ahead": days_ahead,
                "model_type": model_type,
                "timestamp": pd.Timestamp.now().isoformat()
            }

            return forecast_content, stored_data

        except Exception as e:
            logger.exception(f"Forecasting error: {e}")
            error_content = html.Div([
                dbc.Alert(
                    "An error occurred while generating the forecast. Please try again.",
                    color="danger"
                )
            ])
            return error_content, {}

    # Callback to restore forecast results when switching tabs
    @app.callback(
        Output("forecast-results-container", "children", allow_duplicate=True),
        Input("forecast-results-store", "data"),
        prevent_initial_call=True
    )
    def restore_forecast_results(stored_data):
        """Restore forecast results from storage when switching tabs."""
        if stored_data and "content" in stored_data:
            return stored_data["content"]
        return html.Div()


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
