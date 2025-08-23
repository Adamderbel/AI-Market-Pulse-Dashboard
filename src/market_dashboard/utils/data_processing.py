"""
Data processing utilities for Market Dashboard.
Contains functions for data resampling, filtering, and transformation.
"""
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def resample_data(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Resample data to D/W/M periods.
    Aggregates only columns that exist to avoid KeyErrors.
    
    Args:
        df: DataFrame with OHLCV data
        period: Resampling period ('D', 'W', 'M')
        
    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df
    if period == "D":
        return df.copy()

    rule = "W" if period == "W" else "M"  # weekly or month-end

    # Build aggregation map based on available columns
    agg_map = {}
    for col, func in (
        ("open", "first"),
        ("high", "max"),
        ("low", "min"),
        ("close", "last"),
        ("volume", "sum"),
    ):
        if col in df.columns:
            agg_map[col] = func

    out = (
        df.set_index("date")
          .resample(rule)
          .agg(agg_map)
          .reset_index()
    )

    # Drop rows with NaN in present price columns (if any)
    critical = [c for c in ["open", "high", "low", "close"] if c in out.columns]
    if critical:
        out = out.dropna(subset=critical)

    if out.empty:
        return out

    # Preserve metadata columns
    if "symbol" in df.columns:
        out["symbol"] = df["symbol"].iloc[0]
    if "asset_type" in df.columns:
        out["asset_type"] = df["asset_type"].iloc[0]
    
    return out


def filter_date(df: pd.DataFrame, range_val: str) -> pd.DataFrame:
    """
    Filter DataFrame by relative date range anchored to the dataset's latest date.
    
    Args:
        df: DataFrame with date column
        range_val: Date range ('30d', '90d', 'ytd', '1y', 'max')
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    latest = df["date"].max()
    
    if range_val == "30d":
        cutoff = latest - pd.Timedelta(days=30)
    elif range_val == "90d":
        cutoff = latest - pd.Timedelta(days=90)
    elif range_val == "ytd":
        cutoff = pd.Timestamp(year=latest.year, month=1, day=1)
    elif range_val == "1y":
        cutoff = latest - pd.Timedelta(days=365)
    else:
        return df  # "max"
    
    return df[df["date"] >= cutoff]


def normalized_series(close_series: pd.Series) -> pd.Series:
    """
    Normalize a price series to start at 100.
    
    Args:
        close_series: Series of closing prices
        
    Returns:
        Normalized series starting at 100
    """
    base = close_series.iloc[0] if len(close_series) > 0 else None
    if pd.notna(base) and float(base) != 0.0:
        return (close_series / float(base)) * 100.0
    return pd.Series(100.0, index=close_series.index)


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical analysis features to the DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional technical features
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["asset_type", "symbol", "date"])

    def _add_features(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        
        # Returns
        g["ret_1d"] = g["close"].pct_change(1)
        g["ret_7d"] = g["close"].pct_change(7)
        g["ret_30d"] = g["close"].pct_change(30)
        
        # Simple Moving Averages
        g["sma20"] = g["close"].rolling(window=20, min_periods=1).mean()
        g["sma50"] = g["close"].rolling(window=50, min_periods=1).mean()
        
        # Volume features
        if "volume" in g.columns:
            g["vol_sma20"] = g["volume"].rolling(window=20, min_periods=1).mean()
            g["vol_spike"] = (g["volume"] / g["vol_sma20"]).where(g["vol_sma20"] > 0)
        
        return g

    return df.groupby(["asset_type", "symbol"], group_keys=False).apply(_add_features)


def finalize_dataframe(df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    """
    Ensure consistent DataFrame schema and clean data.
    
    Args:
        df: Raw DataFrame
        asset_type: Type of asset ('stock', 'crypto', etc.)
        
    Returns:
        Cleaned DataFrame with consistent schema
    """
    # Ensure consistent schema
    required_cols = ["date", "symbol", "open", "high", "low", "close", "volume", "source", "asset_type"]
    df = df.copy()
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA
    
    df["asset_type"] = asset_type
    
    # Clean and sort
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])
    
    return df[required_cols]
