#!/usr/bin/env python3
"""
Data fetching script for Market Dashboard.
Fetches market data and saves to processed files.
"""
import sys
import os
from pathlib import Path

# Add src and project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from market_dashboard.data import DataFetcher
from market_dashboard.utils.logging_config import setup_logging
from config.settings import STOCK_TICKERS, LOOKBACK_DAYS_STOCKS, DATA_RAW_DIR, DATA_PROCESSED_DIR, LOG_LEVEL

def main():
    """Main function to fetch and process market data."""
    # Setup logging
    setup_logging(level=LOG_LEVEL)
    
    # Initialize data fetcher
    fetcher = DataFetcher(
        save_dir_raw=str(DATA_RAW_DIR),
        save_dir_processed=str(DATA_PROCESSED_DIR)
    )
    
    print("Fetching market data...")
    print(f"Tickers: {', '.join(STOCK_TICKERS)}")
    print(f"Lookback days: {LOOKBACK_DAYS_STOCKS}")
    
    # Fetch and process data
    raw_data, processed_data = fetcher.fetch_and_process_stocks(
        tickers=STOCK_TICKERS,
        lookback_days=LOOKBACK_DAYS_STOCKS
    )
    
    if processed_data.empty:
        print("No data fetched. Check your internet connection and try again.")
        return 1
    
    # Print summary
    fetcher.print_summary(processed_data)
    print("Data fetching complete.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
