"""
Formatting utilities for Market Dashboard.
Contains functions for data formatting, styling, and display.
"""
import pandas as pd
from typing import Dict, Any


def pct_change_str(new: float, old: float) -> str:
    """
    Calculate and format percentage change as string.
    
    Args:
        new: New value
        old: Old value
        
    Returns:
        Formatted percentage change string
    """
    try:
        if pd.notna(old) and pd.notna(new) and float(old) != 0.0:
            val = ((float(new) - float(old)) / float(old)) * 100
            return f"{val:.2f}%"
    except Exception:
        pass
    return "–"


def colorize_pct(pct_text: str) -> Dict[str, str]:
    """
    Return color style for percentage text.
    
    Args:
        pct_text: Percentage text (e.g., "+5.23%" or "-2.15%")
        
    Returns:
        Dictionary with color style
    """
    if isinstance(pct_text, str) and pct_text != "–":
        if pct_text.strip().startswith("-"):
            return {"color": "#D9534F"}  # red
        else:
            return {"color": "#5CB85C"}  # green
    return {}


def format_currency(value: float, prefix: str = "$", decimals: int = 2) -> str:
    """
    Format a numeric value as currency.
    
    Args:
        value: Numeric value to format
        prefix: Currency prefix (default: "$")
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    try:
        return f"{prefix}{value:,.{decimals}f}"
    except (ValueError, TypeError):
        return f"{prefix}0.00"


def format_volume(volume: float) -> str:
    """
    Format volume with appropriate suffixes (K, M, B).
    
    Args:
        volume: Volume value
        
    Returns:
        Formatted volume string
    """
    try:
        if pd.isna(volume):
            return "–"
        
        volume = float(volume)
        if volume >= 1_000_000_000:
            return f"{volume / 1_000_000_000:.1f}B"
        elif volume >= 1_000_000:
            return f"{volume / 1_000_000:.1f}M"
        elif volume >= 1_000:
            return f"{volume / 1_000:.1f}K"
        else:
            return f"{volume:.0f}"
    except (ValueError, TypeError):
        return "–"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as percentage.
    
    Args:
        value: Decimal value (e.g., 0.05 for 5%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    try:
        if pd.isna(value):
            return "–"
        return f"{value * 100:.{decimals}f}%"
    except (ValueError, TypeError):
        return "–"


def get_chart_colors() -> Dict[str, str]:
    """
    Get consistent color scheme for charts.
    
    Returns:
        Dictionary of color mappings
    """
    return {
        "positive": "#5CB85C",  # green
        "negative": "#D9534F",  # red
        "neutral": "#5BC0DE",   # blue
        "volume_up": "rgba(76,175,80,0.45)",
        "volume_down": "rgba(244,67,54,0.45)",
        "volume_neutral": "rgba(0,123,255,0.35)",
        "ma20": "#1f77b4",
        "ma50": "#9467bd"
    }
