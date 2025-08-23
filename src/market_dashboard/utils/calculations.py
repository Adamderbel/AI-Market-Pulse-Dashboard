"""
Calculation utilities for Market Dashboard.
Contains functions for financial calculations and technical indicators.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple


def simple_moving_average(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        series: Price series
        window: Window size for moving average
        min_periods: Minimum periods required for calculation
        
    Returns:
        Moving average series
    """
    if min_periods is None:
        min_periods = max(1, window // 2)
    
    return series.rolling(window=window, min_periods=min_periods).mean()


def percentage_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate percentage change over specified periods.
    
    Args:
        series: Price series
        periods: Number of periods for change calculation
        
    Returns:
        Percentage change series
    """
    return series.pct_change(periods=periods)


def volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """
    Calculate rolling volatility.
    
    Args:
        returns: Returns series
        window: Rolling window size
        annualize: Whether to annualize volatility
        
    Returns:
        Volatility series
    """
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(252)  # Assuming 252 trading days per year
    return vol


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, window: int = 252) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Risk-free rate (annualized)
        window: Rolling window size
        
    Returns:
        Sharpe ratio series
    """
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    rolling_mean = excess_returns.rolling(window=window).mean() * 252
    rolling_std = returns.rolling(window=window).std() * np.sqrt(252)
    
    return rolling_mean / rolling_std


def max_drawdown(price_series: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown and its dates.
    
    Args:
        price_series: Price series
        
    Returns:
        Tuple of (max_drawdown, peak_date, trough_date)
    """
    # Calculate running maximum
    running_max = price_series.expanding().max()
    
    # Calculate drawdown
    drawdown = (price_series - running_max) / running_max
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    # Find peak date (last peak before max drawdown)
    peak_date = running_max.loc[:max_dd_date].idxmax()
    
    return max_dd, peak_date, max_dd_date


def correlation_matrix(data: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple assets.
    
    Args:
        data: DataFrame with asset returns
        method: Correlation method ('pearson', 'kendall', 'spearman')
        
    Returns:
        Correlation matrix
    """
    return data.corr(method=method)


def beta_calculation(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate beta coefficient relative to market.
    
    Args:
        asset_returns: Asset return series
        market_returns: Market return series
        
    Returns:
        Beta coefficient
    """
    # Align series and remove NaN values
    aligned_data = pd.DataFrame({
        'asset': asset_returns,
        'market': market_returns
    }).dropna()
    
    if len(aligned_data) < 2:
        return np.nan
    
    covariance = aligned_data['asset'].cov(aligned_data['market'])
    market_variance = aligned_data['market'].var()
    
    if market_variance == 0:
        return np.nan
    
    return covariance / market_variance


def rsi(price_series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        price_series: Price series
        window: Window size for RSI calculation
        
    Returns:
        RSI series
    """
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def bollinger_bands(price_series: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        price_series: Price series
        window: Window size for moving average
        num_std: Number of standard deviations for bands
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle_band = simple_moving_average(price_series, window)
    std = price_series.rolling(window=window).std()
    
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band
