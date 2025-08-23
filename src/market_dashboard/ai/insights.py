"""
AI-powered market insights generation.
Provides market analysis using local Ollama models.
"""
from typing import List
import pandas as pd
import logging

from .ollama_client import OllamaClient, OllamaError

logger = logging.getLogger(__name__)


class InsightsGenerator:
    """Generates AI-powered market insights."""
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize insights generator.
        
        Args:
            ollama_client: Configured Ollama client
        """
        self.ollama_client = ollama_client
    
    def generate_market_insights(self, df: pd.DataFrame, symbol: str, period: str = "D") -> str:
        """
        Generate concise AI insights for a given asset and period.
        
        Args:
            df: DataFrame with market data
            symbol: Asset symbol
            period: Time period ('D', 'W', 'M')
            
        Returns:
            Human-readable insights string
        """
        try:
            # Basic validation
            if df is None or df.empty:
                return "Not enough data to generate insights."
            
            # Ensure proper datetime and sort
            if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            
            recent_data = (
                df.dropna(subset=["date", "open", "high", "low", "close"])
                  .sort_values("date")
                  .tail(10)
            )
            
            if recent_data.empty:
                return "Not enough data to generate insights."
            
            # Check if Ollama server is available
            if not self.ollama_client.is_server_available():
                return "Ollama server is not available. Please make sure Ollama is running."
            
            # Prepare data for the prompt
            price_changes = []
            last_close = float(recent_data.iloc[-1]["close"])
            
            # Look back up to 5 prior points if available
            for i in range(1, min(6, len(recent_data))):
                prev_close = float(recent_data.iloc[-i]["close"])
                if prev_close != 0:
                    change = ((last_close - prev_close) / prev_close) * 100.0
                    price_changes.append(f"{i} period(s) ago: {change:.2f}%")
            
            latest_row = recent_data.iloc[-1]
            latest_date = pd.to_datetime(latest_row["date"], errors="coerce")
            latest_date_str = latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "N/A"
            
            # Create detailed prompt
            prompt = f"""
Provide a concise but detailed market analysis for {symbol.upper()} over the recent {period} interval in 3-5 sentences.
Cover: trend/momentum and magnitude of recent moves; volatility/gaps/spikes; notable levels (recent highs/lows, round numbers) and moving average context;
volume vs typical activity; and finish with one clear risk or watch item to monitor next.

Recent change snapshots: {', '.join(price_changes) if price_changes else 'Insufficient history for change computations.'}

Latest data:
Date: {latest_date_str}
Open: {float(latest_row['open']):.2f}
High: {float(latest_row['high']):.2f}
Low: {float(latest_row['low']):.2f}
Close: {float(latest_row['close']):.2f}
Volume: {int(latest_row['volume']) if pd.notna(latest_row.get('volume')) else 0}
"""
            
            # Generate insights using chat API
            content = self.ollama_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional market analyst. Write 3-5 crisp sentences. "
                            "Explain trend/momentum, volatility, key levels/MA context, notable volume, "
                            "and one actionable risk/watch item. Avoid filler and disclaimers."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=320
            )
            
            return content
            
        except OllamaError as e:
            logger.error(f"Ollama error: {e}")
            return "Unable to generate insights: Ollama service unavailable."
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return f"Unable to generate insights at this time: {str(e)}"
    
    def generate_comparative_insights(self, df: pd.DataFrame, symbols: List[str], period: str = "D") -> str:
        """
        Generate brief comparative AI insights for multiple assets.
        
        Args:
            df: DataFrame with market data
            symbols: List of asset symbols
            period: Time period ('D', 'W', 'M')
            
        Returns:
            Human-readable comparative insights string
        """
        try:
            if df is None or df.empty or not symbols or len(symbols) < 2:
                return "Insufficient data for comparative analysis."
            
            # Ensure proper datetime and sort
            if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            
            comparisons: List[str] = []
            
            for symbol in symbols:
                symbol_df = (
                    df[df["symbol"] == symbol]
                    .dropna(subset=["date", "close"])
                    .sort_values("date")
                )
                
                if len(symbol_df) >= 2:
                    last = symbol_df.iloc[-1]
                    prev = symbol_df.iloc[-2]
                    prev_close = float(prev["close"])
                    last_close = float(last["close"])
                    change = ((last_close - prev_close) / prev_close) * 100.0 if prev_close != 0 else 0.0
                    comparisons.append(f"{symbol.upper()}: {last_close:.2f} ({change:+.2f}%)")
            
            if len(comparisons) < 2:
                return "Insufficient data for comparative analysis."
            
            # Check Ollama availability
            if not self.ollama_client.is_server_available():
                return "Ollama server is not available. Please make sure Ollama is running."
            
            prompt = f"""
Compare the following assets (period={period}): {', '.join(comparisons)}.

Write 3-5 sentences covering:
- Relative performance ranking (leaders/laggards) and magnitude of dispersion.
- Correlation/cluster behavior (who moves together; any diversifiers).
- Volatility or drawdown/outlier observations.
- A succinct takeaway on positioning or risk to monitor next.
"""
            
            content = self.ollama_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional market analyst. Provide 3-5 precise sentences. "
                            "Rank assets, describe dispersion and correlation patterns, note any volatility/drawdowns, "
                            "and end with one clear, concise takeaway. Avoid filler and disclaimers."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=320
            )
            
            return content
            
        except OllamaError as e:
            logger.error(f"Ollama error: {e}")
            return "Unable to generate comparative insights: Ollama service unavailable."
        except Exception as e:
            logger.error(f"Error generating comparative insights: {e}")
            return f"Unable to generate comparative insights: {str(e)}"
