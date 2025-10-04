"""
Database management for Market Dashboard.
Handles SQLite database connections and operations.
"""
import sqlite3
import pandas as pd
import logging
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, db_path: str):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Yields:
            sqlite3.Connection: Database connection
        """
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def load_market_data(self) -> pd.DataFrame:
        """
        Load all market data from database.
        
        Returns:
            DataFrame with market data
        """
        try:
            with self.get_connection() as conn:
                df = pd.read_sql("SELECT * FROM market_all", conn)
                if df.empty:
                    return df
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values(["symbol", "date"])
                return df
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return pd.DataFrame()
    
    def load_latest_data(self) -> pd.DataFrame:
        """
        Load latest market data from database.
        
        Returns:
            DataFrame with latest market data
        """
        try:
            with self.get_connection() as conn:
                df = pd.read_sql("SELECT * FROM market_latest", conn)
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                return df
        except Exception as e:
            logger.error(f"Error loading latest data: {e}")
            return pd.DataFrame()
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols in database.
        
        Returns:
            List of symbol strings
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT symbol FROM market_all ORDER BY symbol")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return []
    
    def get_symbol_data(self, symbol: str) -> pd.DataFrame:
        """
        Get data for a specific symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with symbol data
        """
        try:
            with self.get_connection() as conn:
                query = "SELECT * FROM market_all WHERE symbol = ? ORDER BY date"
                df = pd.read_sql(query, conn, params=[symbol])
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                return df
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def insert_data(self, df: pd.DataFrame, table_name: str, if_exists: str = "append") -> bool:
        """
        Insert data into database table.
        
        Args:
            df: DataFrame to insert
            table_name: Target table name
            if_exists: What to do if table exists ('append', 'replace', 'fail')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if df.empty:
                logger.warning(f"No data to insert into {table_name}")
                return False
            
            with self.get_connection() as conn:
                df.to_sql(table_name, conn, if_exists=if_exists, index=False)
                logger.info(f"Inserted {len(df)} rows into {table_name}")
                return True
        except Exception as e:
            logger.error(f"Error inserting data into {table_name}: {e}")
            return False
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if table exists in database.
        
        Args:
            table_name: Name of table to check
            
        Returns:
            True if table exists, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    [table_name]
                )
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a table.
        
        Args:
            table_name: Name of table
            
        Returns:
            Dictionary with table information
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                # Get date range
                cursor.execute(f"SELECT MIN(date), MAX(date) FROM {table_name}")
                date_range = cursor.fetchone()
                
                # Get unique symbols
                cursor.execute(f"SELECT COUNT(DISTINCT symbol) FROM {table_name}")
                symbol_count = cursor.fetchone()[0]
                
                return {
                    "row_count": row_count,
                    "date_range": date_range,
                    "symbol_count": symbol_count
                }
        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            return {}
            
    def initialize_database(self):
        """
        Initialize database with required tables if they don't exist.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create market_all table if it doesn't exist
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_all (
                    date TEXT,
                    symbol TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (date, symbol)
                )
                ''')
                
                # Create market_latest table if it doesn't exist
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_latest (
                    symbol TEXT PRIMARY KEY,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Create indices for better query performance
                cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_market_all_symbol 
                ON market_all(symbol, date DESC)
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
