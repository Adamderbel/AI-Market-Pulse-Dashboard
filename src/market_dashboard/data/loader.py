"""
Data loading utilities for Market Dashboard.
Handles loading data from files to database.
"""
import sqlite3
import pandas as pd
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading data from files to database."""
    
    def __init__(self, db_path: str, data_processed_dir: str = "data/processed"):
        """
        Initialize data loader.
        
        Args:
            db_path: Path to SQLite database
            data_processed_dir: Directory containing processed data files
        """
        self.db_path = db_path
        self.data_processed_dir = data_processed_dir
        self.all_file = os.path.join(data_processed_dir, "market_all.parquet")
        self.latest_file = os.path.join(data_processed_dir, "market_latest.parquet")
    
    def append_parquet_to_sqlite(self, parquet_file: str, table_name: str, conn: sqlite3.Connection) -> bool:
        """
        Append data from parquet file to SQLite table.
        Only inserts new rows based on date comparison.
        
        Args:
            parquet_file: Path to parquet file
            table_name: Target table name
            conn: SQLite connection
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(parquet_file):
                logger.warning(f"Skipping: {parquet_file} not found")
                return False
            
            df = pd.read_parquet(parquet_file)
            if df.empty:
                logger.warning(f"Skipping empty dataframe for '{table_name}' from {parquet_file}")
                return False
            
            # Ensure date column is datetime
            df["date"] = pd.to_datetime(df["date"])
            
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Get max date per symbol to avoid duplicates
                existing = pd.read_sql(
                    f"SELECT symbol, MAX(date) as max_date FROM {table_name} GROUP BY symbol", 
                    conn
                )
                existing["max_date"] = pd.to_datetime(existing["max_date"])
                
                # Filter only new rows
                new_rows = []
                for symbol, group in df.groupby("symbol"):
                    cutoff_rows = existing.loc[existing["symbol"] == symbol, "max_date"]
                    if not cutoff_rows.empty:
                        cutoff_date = cutoff_rows.iloc[0]
                        group = group[group["date"] > cutoff_date]
                    new_rows.append(group)
                
                df_new = pd.concat(new_rows, ignore_index=True) if new_rows else pd.DataFrame()
            else:
                df_new = df
            
            if df_new.empty:
                logger.info(f"No new rows to insert into {table_name}")
                return True
            
            # Insert new data
            df_new.to_sql(table_name, conn, if_exists="append", index=False)
            logger.info(f"Inserted {len(df_new)} new rows into '{table_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error loading {parquet_file} to {table_name}: {e}")
            return False
    
    def load_all_data(self) -> bool:
        """
        Load all processed data files to database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                success = True
                
                # Load market_all data
                if not self.append_parquet_to_sqlite(self.all_file, "market_all", conn):
                    success = False
                
                # Load market_latest data
                if not self.append_parquet_to_sqlite(self.latest_file, "market_latest", conn):
                    success = False
                
                return success
                
        except Exception as e:
            logger.error(f"Error loading data to database: {e}")
            return False
    
    def replace_table_data(self, parquet_file: str, table_name: str) -> bool:
        """
        Replace entire table with data from parquet file.
        
        Args:
            parquet_file: Path to parquet file
            table_name: Target table name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(parquet_file):
                logger.error(f"File not found: {parquet_file}")
                return False
            
            df = pd.read_parquet(parquet_file)
            if df.empty:
                logger.warning(f"Empty dataframe from {parquet_file}")
                return False
            
            # Ensure date column is datetime
            df["date"] = pd.to_datetime(df["date"])
            
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                logger.info(f"Replaced {table_name} with {len(df)} rows")
                return True
                
        except Exception as e:
            logger.error(f"Error replacing table {table_name}: {e}")
            return False
    
    def get_data_summary(self) -> dict:
        """
        Get summary of available data files.
        
        Returns:
            Dictionary with file information
        """
        summary = {}
        
        for file_path, name in [(self.all_file, "market_all"), (self.latest_file, "market_latest")]:
            if os.path.exists(file_path):
                try:
                    df = pd.read_parquet(file_path)
                    summary[name] = {
                        "exists": True,
                        "rows": len(df),
                        "symbols": df["symbol"].nunique() if "symbol" in df.columns else 0,
                        "date_range": (
                            df["date"].min().strftime("%Y-%m-%d") if "date" in df.columns and not df.empty else None,
                            df["date"].max().strftime("%Y-%m-%d") if "date" in df.columns and not df.empty else None
                        ) if "date" in df.columns and not df.empty else (None, None)
                    }
                except Exception as e:
                    summary[name] = {"exists": True, "error": str(e)}
            else:
                summary[name] = {"exists": False}
        
        return summary
