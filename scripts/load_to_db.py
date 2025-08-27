#!/usr/bin/env python3
"""
Database loading script for Market Dashboard.
Loads processed data files into SQLite database.
"""
import sys
from pathlib import Path

# Add src and project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from market_dashboard.data import DataLoader
from market_dashboard.utils.logging_config import setup_logging
from config.settings import DB_PATH, DATA_PROCESSED_DIR, LOG_LEVEL

def main():
    """Main function to load data to database."""
    # Setup logging
    setup_logging(level=LOG_LEVEL)
    
    # Initialize data loader
    loader = DataLoader(
        db_path=DB_PATH,
        data_processed_dir=str(DATA_PROCESSED_DIR)
    )
    
    print("Loading data to database...")
    print(f"Database: {DB_PATH}")
    print(f"Data directory: {DATA_PROCESSED_DIR}")
    
    # Get data summary
    summary = loader.get_data_summary()
    print("\nData files summary:")
    for name, info in summary.items():
        if info.get("exists"):
            if "error" in info:
                print(f"  {name}: ERROR - {info['error']}")
            else:
                print(f"  {name}: {info['rows']} rows, {info['symbols']} symbols, {info['date_range'][0]} to {info['date_range'][1]}")
        else:
            print(f"  {name}: NOT FOUND")
    
    # Load data
    success = loader.load_all_data()
    
    if success:
        print("Data loading complete.")
        return 0
    else:
        print("Data loading failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
