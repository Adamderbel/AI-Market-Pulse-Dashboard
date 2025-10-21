"""
Main application file for Market Dashboard.
Creates and configures the Dash application.
"""
import sys
from pathlib import Path

# Add config to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dash import Dash
import dash_bootstrap_components as dbc
import logging

from .data import DatabaseManager
from .components import register_callbacks, create_app_layout
from .utils.logging_config import setup_logging
from config.settings import (
    DB_PATH, DASH_HOST, DASH_PORT, DASH_DEBUG, LOG_LEVEL
)

# Setup logging
setup_logging(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def create_app():
    """
    Create and configure the Dash application.
    
    Returns:
        Configured Dash app instance
    """
    # Initialize Dash app
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.LUX],
        suppress_callback_exceptions=True,
        title="AI Market Pulse Dashboard"
    )
    
    # Load available assets
    db_manager = DatabaseManager(DB_PATH)
    assets = db_manager.get_available_symbols()
    
    if not assets:
        logger.warning("No assets found in database. Please run data fetching and loading scripts first.")
        assets = []
    else:
        logger.info(f"Loaded {len(assets)} assets: {', '.join(assets)}")
    
    # Set app layout
    app.layout = create_app_layout()
    
    # Register callbacks
    register_callbacks(app, assets)
    
    # Store the server for Gunicorn
    server = app.server
    
    return app


def run_app():
    """Run the Dash application."""
    app = create_app()
    
    logger.info(f"Starting Market Dashboard on {DASH_HOST}:{DASH_PORT}")
    logger.info(f"Debug mode: {DASH_DEBUG}")
    
    app.run(
        debug=DASH_DEBUG,
        host=DASH_HOST,
        port=DASH_PORT
    )


# For Gunicorn
app = create_app()
server = app.server

if __name__ == "__main__":
    run_app()