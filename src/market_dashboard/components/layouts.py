"""
Layout components for Market Dashboard.
Contains all UI layout definitions.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.settings import PERIOD_OPTIONS, DATE_RANGE_OPTIONS


def create_chatbot_ui() -> html.Div:
    """
    Chatbot UI:
    - Floating toggle button
    - Chat container with header (minimize/close), messages, typing indicator, and input group
    - Session store initialized with a greeting from the bot
    Styles are provided via Dash assets at `src/market_dashboard/assets/chat.css`.
    """

    return html.Div([

        # Session store with initial greeting from bot
        dcc.Store(id="chat-store", storage_type="session", data={
            "session_id": None,
            "messages": [
                {"role": "bot", "content": "Hello! How can I assist you today?"}
            ]
        }),

        # Chat container
        html.Div([
            # Header
            html.Div([
                html.H5("AI Assistant"),
                html.Div([
                    dbc.Button("-", id="minimize-chat", size="sm", outline=True, color="light", className="me-1"),
                    dbc.Button("×", id="close-chat", size="sm", outline=True, color="light"),
                ])
            ], className="chat-header"),

            # Messages
            html.Div(id="chat-messages", className="chat-messages"),

            # Input area
            html.Div([
                html.Div([
                    html.Span(), html.Span(), html.Span()
                ], id="typing-indicator", className="typing-indicator"),
                html.Div([
                    dcc.Input(id="user-input", type="text", placeholder="Type your message...", debounce=False),
                    dbc.Button("Send", id="send-button", color="primary")
                ], className="input-group")
            ], className="chat-input-container"),
        ], id="chat-container", className="chat-container visible"),

        # Floating toggle button
        html.Button(children="💬", id="chat-toggle", className="btn btn-primary rounded-circle")
    ])
def create_landing_layout() -> dbc.Container:
    """
    Create the landing page layout.
    
    Returns:
        Landing page layout component
    """
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("AI Market Pulse", className="display-4 mt-5 mb-3 text-center"),
                html.P(
                    "Interactive market dashboard with AI insights. Compare assets, explore trends, and export data.",
                    className="lead text-center"
                ),
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col(html.Div("• Candlesticks with volume and moving averages"), width=12),
                    dbc.Col(html.Div("• Multi-asset comparison, correlation heatmap, and return distributions"), width=12),
                    dbc.Col(html.Div("• AI-generated insights for single and multiple assets"), width=12),
                ], className="mb-4 text-center"),
                html.Div(
                    dbc.Button("Start Analysis", color="primary", size="lg", href="/dashboard", className="px-4"),
                    className="text-center"
                ),
                html.Div(html.Small("Tip: Use filters to tailor your view."), className="text-center mt-3 text-muted")
            ], width=12)
        ])
    ], fluid=True)


def create_single_stock_layout(assets: list) -> dbc.Container:
    """
    Create the single stock analysis layout.
    
    Args:
        assets: List of available asset symbols
        
    Returns:
        Single stock layout component
    """
    return dbc.Container([
        html.H3("Single Stock Analysis", className="mt-3 mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Select Stock:", className="fw-bold"),
                dcc.Dropdown(
                    id="single-asset-dropdown",
                    options=[{"label": s.upper(), "value": s} for s in assets],
                    value=assets[0] if len(assets) > 0 else None,
                    placeholder="Choose a stock...",
                    persistence=True,
                    persistence_type="session"
                )
            ], width=4),
            dbc.Col([
                html.Label("Date Range:", className="fw-bold"),
                dcc.Dropdown(
                    id="single-date-range",
                    options=DATE_RANGE_OPTIONS,
                    value="1y",
                    persistence=True,
                    persistence_type="session"
                )
            ], width=3),
            dbc.Col([
                html.Label("Action:", className="fw-bold"),
                html.Br(),
                dbc.Button(
                    "Analyze Stock",
                    id="btn-analyze-single",
                    color="primary",
                    className="w-100"
                )
            ], width=2),
            dbc.Col([
                html.Label("Export:", className="fw-bold"),
                html.Br(),
                dbc.Button(" CSV", id="btn_csv", color="secondary", size="lg", className="w-100"),
                dcc.Download(id="download-dataframe-csv")
            ], width=2)
        ], className="mb-4"),

        # Loading spinner for results
        dcc.Loading(
            id="loading-single-stock",
            type="circle",
            children=html.Div(id="single-results-container")
        )
    ], fluid=True)


def create_multi_stock_layout(assets: list) -> dbc.Container:
    """
    Create the multi-stock comparison layout.
    
    Args:
        assets: List of available asset symbols
        
    Returns:
        Multi-stock layout component
    """
    return dbc.Container([
        html.H3("Multi-Stock Comparison", className="mt-3 mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Select Stocks to Compare:", className="fw-bold"),
                dcc.Dropdown(
                    id="multi-asset-dropdown",
                    options=[{"label": s.upper(), "value": s} for s in assets],
                    value=[],
                    multi=True,
                    placeholder="Choose 2+ stocks...",
                    persistence=True,
                    persistence_type="session"
                )
            ], width=5),
            dbc.Col([
                html.Label("Date Range:", className="fw-bold"),
                dcc.Dropdown(
                    id="multi-date-range",
                    options=DATE_RANGE_OPTIONS,
                    value="1y",
                    persistence=True,
                    persistence_type="session"
                )
            ], width=4),
            dbc.Col([
                html.Label("Action:", className="fw-bold"),
                html.Br(),
                dbc.Button(
                    "Compare Stocks",
                    id="btn-compare-multi",
                    color="primary",
                    className="w-100"
                )
            ], width=3),
        ], className="mb-4"),

        # Loading spinner for results
        dcc.Loading(
            id="loading-multi",
            type="circle",
            children=html.Div(id="multi-results-container")
        )
    ], fluid=True)


def create_forecast_layout(assets: list) -> dbc.Container:
    """
    Create the stock forecasting layout.
    
    Args:
        assets: List of available asset symbols

    Returns:
        Forecast layout component
    """
    return dbc.Container([
        html.H3(" Stock Price Forecasting", className="mt-3 mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Select Stock to Forecast:"),
                dcc.Dropdown(
                    id="forecast-asset-dropdown",
                    options=[{"label": s.upper(), "value": s} for s in assets],
                    value=assets[0] if len(assets) > 0 else None,
                    placeholder="Choose a stock...",
                    persistence=True,
                    persistence_type="session"
                )
            ], width=4),
            dbc.Col([
                html.Label("Forecast Period:"),
                dcc.Dropdown(
                    id="forecast-period-dropdown",
                    options=[
                        {"label": "1 Day", "value": 1},
                        {"label": "3 Days", "value": 3},
                        {"label": "7 Days", "value": 7},
                        {"label": "10 Days", "value": 10}
                    ],
                    value=7,
                    persistence=True,
                    persistence_type="session"
                )
            ], width=3),
            dbc.Col([
                html.Label("Model Type:"),
                dcc.Dropdown(
                    id="forecast-model-dropdown",
                    options=[
                        {"label": "Ensemble (Recommended)", "value": "ensemble"},
                        {"label": "ARIMA (Statistical)", "value": "arima"},
                        {"label": "Linear Regression", "value": "linear"},
                        {"label": "Simple Trend", "value": "trend"}
                    ],
                    value="ensemble",
                    persistence=True,
                    persistence_type="session"
                )
            ], width=3),
            dbc.Col([
                html.Label("Generate Forecast:"),
                dbc.Button("Forecast", id="btn-forecast", color="primary", className="w-100"),
            ], width=2)
        ], className="mb-4"),

        # Loading spinner for forecast results
        dcc.Loading(
            id="loading-forecast",
            type="circle",
            children=html.Div(id="forecast-results-container")
        ),

        # Store for persisting forecast results
        dcc.Store(id="forecast-results-store", storage_type="session")
    ], fluid=True)


def create_dashboard_layout(assets: list) -> dbc.Container:
    """
    Create the main dashboard layout with tabs.

    Args:
        assets: List of available asset symbols

    Returns:
        Dashboard layout component
    """
    return dbc.Container([
        html.H2("AI Market Pulse Dashboard", className="mt-3 mb-3"),
        dcc.Tabs(id="dashboard-tabs", value='single-stock', children=[
            dcc.Tab(label='Single Stock Analysis', value='single-stock'),
            dcc.Tab(label='Multi-Stock Comparison', value='multi-stock'),
            dcc.Tab(label='Price Forecasting', value='forecasting'),
        ]),
        html.Div(id="single-section", children=create_single_stock_layout(assets)),
        html.Div(id="multi-section", children=create_multi_stock_layout(assets), style={"display": "none"}),
        html.Div(id="forecast-section", children=create_forecast_layout(assets), style={"display": "none"})
    ], fluid=True)


def create_app_layout() -> html.Div:
    """
    Create the root app layout with routing.
    
    Returns:
        Root layout component
    """
    return html.Div([
        dcc.Location(id="url"),
        html.Div(id="page-content-router"),
        # Global chatbot UI
        create_chatbot_ui(),
    ])