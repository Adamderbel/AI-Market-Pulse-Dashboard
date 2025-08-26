"""
Data fetching utilities for Market Dashboard.
Handles fetching data from external sources like Yahoo Finance.
"""
import os
from datetime import datetime, timedelta, timezone
from typing import List
import pandas as pd
import yfinance as yf
import logging

from ..utils.data_processing import finalize_dataframe, add_trend_features

logger = logging.getLogger(__name__)


class DataFetcher:
    """Handles fetching market data from external sources."""
    
    def __init__(self, save_dir_raw: str = "data/raw", save_dir_processed: str = "data/processed"):
        """
        Initialize data fetcher.
        
        Args:
            save_dir_raw: Directory for raw data files
            save_dir_processed: Directory for processed data files
        """
        self.save_dir_raw = save_dir_raw
        self.save_dir_processed = save_dir_processed
        
        # Ensure directories exist
        os.makedirs(save_dir_raw, exist_ok=True)
        os.makedirs(save_dir_processed, exist_ok=True)
    
    def fetch_stocks_yf(self, tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            tickers: List of stock symbols
            start: Start date
            end: End date
            
        Returns:
            DataFrame with stock data
        """
        try:
            logger.info(f"Fetching data for {len(tickers)} tickers from Yahoo Finance")
            
            # Download data in one go for speed
            data = yf.download(
                tickers=" ".join(tickers),
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            
            # Normalize data structure
            frames = []
            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        # Single ticker case
                        df_ticker = data.reset_index().rename(columns=str.lower)
                    else:
                        # Multiple tickers case
                        df_ticker = data[ticker].reset_index().rename(columns=str.lower)
                    
                    df_ticker["symbol"] = ticker
                    df_ticker["source"] = "yfinance"
                    df_ticker = df_ticker.rename(columns={"Date": "date"})
                    
                    # Select required columns
                    required_cols = ["date", "symbol", "open", "high", "low", "close", "volume", "source"]
                    available_cols = [col for col in required_cols if col in df_ticker.columns]
                    frames.append(df_ticker[available_cols])
                    
                except Exception as e:
                    logger.warning(f"Failed to process {ticker}: {e}")
                    continue
            
            if not frames:
                logger.warning("No data fetched for any ticker")
                return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume", "source"])
            
            result = pd.concat(frames, ignore_index=True)
            result["asset_type"] = "stock"
            
            logger.info(f"Successfully fetched {len(result)} rows of data")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume", "source"])
    
    def fetch_and_process_stocks(self, tickers: List[str], lookback_days: int = 730) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch and process stock data with features.
        
        Args:
            tickers: List of stock symbols
            lookback_days: Number of days to look back
            
        Returns:
            Tuple of (raw_data, processed_data)
        """
        today = datetime.now(timezone.utc)
        start_date = (today - timedelta(days=lookback_days)).date()
        # Add 1 day to end_date because Yahoo Finance end parameter is exclusive
        end_date = (today + timedelta(days=1)).date()

        logger.info(f"Fetching stocks from {start_date} to {end_date} (end date exclusive)")
        
        # Fetch raw data
        stocks_df = self.fetch_stocks_yf(
            tickers,
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.min.time())
        )
        
        if stocks_df.empty:
            logger.warning("No stock data fetched")
            return pd.DataFrame(), pd.DataFrame()
        
        # Finalize raw data
        stocks_df = finalize_dataframe(stocks_df, asset_type="stock")
        
        # Save raw data
        raw_path = os.path.join(self.save_dir_raw, "stocks.parquet")
        stocks_df.to_parquet(raw_path, index=False)
        logger.info(f"Saved raw data: {raw_path}")
        
        # Add features and process
        stocks_df["close"] = pd.to_numeric(stocks_df["close"], errors="coerce")
        stocks_df["volume"] = pd.to_numeric(stocks_df["volume"], errors="coerce")
        
        processed_df = add_trend_features(stocks_df)
        
        # Save processed data
        processed_path_all = os.path.join(self.save_dir_processed, "market_all.parquet")
        processed_df.to_parquet(processed_path_all, index=False)
        
        # Save latest data
        latest_df = processed_df.sort_values("date").groupby(["asset_type", "symbol"]).tail(1)
        processed_path_latest = os.path.join(self.save_dir_processed, "market_latest.parquet")
        latest_df.to_parquet(processed_path_latest, index=False)
        
        logger.info(f"Saved processed data: {processed_path_all}, {processed_path_latest}")
        
        return stocks_df, processed_df
    
    def print_summary(self, df: pd.DataFrame) -> None:
        """
        Print summary of fetched data.
        
        Args:
            df: DataFrame to summarize
        """
        if df.empty:
            print("No data to summarize")
            return
        
        def fmt_pct(x):
            return "–" if pd.isna(x) else f"{x*100:,.2f}%"
        
        latest = df.sort_values("date").groupby(["asset_type", "symbol"]).tail(1)
        
        print("\nToday's snapshot (latest row per asset):")
        for _, row in latest.sort_values(["asset_type", "symbol"]).iterrows():
            print(f"{row['asset_type']:6} | {row['symbol']:8} | "
                  f"close={row['close']:>10,.2f} | "
                  f"1d={fmt_pct(row.get('ret_1d'))} | "
                  f"7d={fmt_pct(row.get('ret_7d'))} | "
                  f"30d={fmt_pct(row.get('ret_30d'))} | "
                  f"vol_spike={row.get('vol_spike', pd.NA)}")
        
        print(f"\nTotal rows: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Symbols: {', '.join(sorted(df['symbol'].unique()))}")
