"""
Configuration management for Market Dashboard application.
Centralizes all configuration values with environment variable support.
"""
import os
from typing import List, Optional
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "saved_data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Database configuration
DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "market.db"))

# Data fetching configuration
STOCK_TICKERS = [
    "AAPL", "TSLA", "MSFT", "SPY", "GOOGL", 
    "AMZN", "NVDA", "META", "QQQ"
]
LOOKBACK_DAYS_STOCKS = int(os.getenv("LOOKBACK_DAYS_STOCKS", "730"))  # 2 years

# Ollama AI configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30.0"))
OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "2"))

# Dashboard configuration
DASH_HOST = os.getenv("DASH_HOST", "0.0.0.0")
DASH_PORT = int(os.getenv("DASH_PORT", "8050"))
DASH_DEBUG = os.getenv("DASH_DEBUG", "True").lower() in ("true", "1", "yes")

# Chart configuration
DEFAULT_CHART_HEIGHT = 650
DEFAULT_CHART_TEMPLATE = "plotly_white"

# Period and date range options
PERIOD_OPTIONS = [
    {"label": "Daily", "value": "D"},
    {"label": "Weekly", "value": "W"},
    {"label": "Monthly", "value": "M"}
]

DATE_RANGE_OPTIONS = [
    {"label": "Last 30 Days", "value": "30d"},
    {"label": "Last 90 Days", "value": "90d"},
    {"label": "Year to Date", "value": "ytd"},
    {"label": "1 Year", "value": "1y"},
    {"label": "Max", "value": "max"},
]

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Ensure data directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
