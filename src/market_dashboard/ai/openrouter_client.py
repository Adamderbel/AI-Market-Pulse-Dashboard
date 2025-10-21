"""
AI-powered market insights generation.
Provides market analysis using OpenRouter's Mistral 7B model.
"""
import os
from typing import List, Optional, Dict, Any
import pandas as pd
import logging
import asyncio

from .openrouter_client_impl import OpenRouterClient, OpenRouterError

logger = logging.getLogger(__name__)


class InsightsGenerator:
    """Generates AI-powered market insights using OpenRouter."""

    def __init__(self, openrouter_client: Optional[OpenRouterClient] = None, api_key: Optional[str] = None):
        """
        Initialize insights generator.

        Args:
            openrouter_client: Optional pre-configured OpenRouter client
            api_key: OpenRouter API key (required if client not provided)
        """
        if openrouter_client is None and api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenRouter API key is required. Either pass openrouter_client or api_key parameter or set OPENROUTER_API_KEY env var."
                )
        self.openrouter_client = openrouter_client or OpenRouterClient(api_key=api_key)
        self._should_close_client = openrouter_client is None

    async def generate_market_insights_async(self, df: pd.DataFrame, symbol: str, period: str = "D") -> str:
        """Generate concise AI insights for a given asset and period asynchronously."""
        try:
            if df is None or df.empty:
                return "Not enough data to generate insights."

            # Ensure proper datetime
            if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"], errors="coerce")

            recent_data = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").tail(10)
            if recent_data.empty:
                return "Not enough data to generate insights."

            price_changes = []
            last_close = float(recent_data.iloc[-1]["close"])
            for i in range(1, min(6, len(recent_data))):
                prev_close = float(recent_data.iloc[-i]["close"])
                if prev_close != 0:
                    change = ((last_close - prev_close) / prev_close) * 100.0
                    price_changes.append(f"{i} period(s) ago: {change:.2f}%")

            latest_row = recent_data.iloc[-1]
            latest_date = pd.to_datetime(latest_row["date"], errors="coerce")
            latest_date_str = latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "N/A"

            prompt = f"""
Provide a concise market analysis for {symbol.upper()} over the recent {period} interval in 3-5 sentences.
Cover: trend/momentum, volatility/gaps/spikes, notable levels (highs/lows, round numbers), moving average context,
volume vs typical activity, and one clear risk or watch item.

Recent change snapshots: {', '.join(price_changes) if price_changes else 'Insufficient history for change computations.'}

Latest data:
Date: {latest_date_str}
Open: {float(latest_row['open']):.2f}
High: {float(latest_row['high']):.2f}
Low: {float(latest_row['low']):.2f}
Close: {float(latest_row['close']):.2f}
Volume: {int(latest_row['volume']) if pd.notna(latest_row.get('volume')) else 0}
"""

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional market analyst. Write 3-5 crisp sentences. "
                        "Explain trend/momentum, volatility, key levels/MA context, notable volume, "
                        "and one actionable risk/watch item. Avoid filler and disclaimers."
                    )
                },
                {"role": "user", "content": prompt.strip()}
            ]

            content = await self.openrouter_client.chat(messages=messages, temperature=0.2, max_tokens=320)
            return content.strip()

        except OpenRouterError as e:
            logger.error(f"OpenRouter error: {e}")
            return "Unable to generate insights at the moment."
        except Exception as e:
            logger.exception("Error generating market insights")
            return f"Unexpected error: {str(e)}"

    def generate_market_insights(self, df: pd.DataFrame, symbol: str, period: str = "D") -> str:
        """Synchronous wrapper."""
        return asyncio.run(self.generate_market_insights_async(df, symbol, period))

    async def generate_comparative_insights_async(self, df: pd.DataFrame, symbols: List[str], period: str = "D") -> str:
        """Generate comparative insights asynchronously."""
        try:
            if df is None or df.empty or len(symbols) < 2:
                return "Insufficient data for comparative analysis."

            comparisons = []
            for symbol in symbols:
                symbol_df = df[df["symbol"] == symbol].dropna(subset=["date", "close"]).sort_values("date")
                if len(symbol_df) >= 2:
                    last = symbol_df.iloc[-1]
                    prev = symbol_df.iloc[-2]
                    change = ((last["close"] - prev["close"]) / prev["close"]) * 100 if prev["close"] != 0 else 0
                    comparisons.append(f"{symbol.upper()}: {last['close']:.2f} ({change:+.2f}%)")

            if len(comparisons) < 2:
                return "Insufficient data for comparative analysis."

            prompt = f"""
Compare the following assets (period={period}): {', '.join(comparisons)}.

Write 3-5 sentences covering:
- Relative performance ranking
- Correlation/cluster behavior
- Volatility or drawdown observations
- One clear takeaway on positioning or risk
"""

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional market analyst. Provide 3-5 precise sentences. "
                        "Rank assets, describe dispersion and correlation patterns, note volatility/drawdowns, "
                        "and end with one clear, concise takeaway."
                    )
                },
                {"role": "user", "content": prompt.strip()}
            ]

            content = await self.openrouter_client.chat(messages=messages, temperature=0.2, max_tokens=320)
            return content.strip()

        except OpenRouterError as e:
            logger.error(f"OpenRouter error in comparative insights: {e}")
            return "Unable to generate comparative insights at the moment."
        except Exception as e:
            logger.exception("Error generating comparative insights")
            return f"Unexpected error: {str(e)}"

    def generate_comparative_insights(self, df: pd.DataFrame, symbols: List[str], period: str = "D") -> str:
        """Synchronous wrapper."""
        return asyncio.run(self.generate_comparative_insights_async(df, symbols, period))

    async def generate_forecast_insights_async(
        self, df: pd.DataFrame, symbol: str, current_price: float, forecast_price: float,
        days_ahead: int, model_name: str, model_accuracy: float
    ) -> str:
        """Generate AI-powered forecast insights asynchronously."""
        try:
            if df is None or df.empty:
                return "Insufficient data to generate forecast insights."

            price_change = forecast_price - current_price
            price_change_pct = (price_change / current_price) * 100 if current_price != 0 else 0

            recent_data = df.tail(10)
            if len(recent_data) >= 2:
                recent_trend = "increasing" if recent_data['close'].iloc[-1] > recent_data['close'].iloc[0] else "decreasing"
                volatility = recent_data['close'].pct_change().std() * 100
            else:
                recent_trend = "stable"
                volatility = 0

            # Determine investment advice
            def get_investment_advice():
                if price_change_pct > 3 and model_accuracy > 80:
                    return "Strong Buy - High confidence upward movement expected"
                elif price_change_pct > 1 and model_accuracy > 70:
                    return "Buy - Moderate upward potential"
                elif price_change_pct > 0 and volatility < 5:
                    return "Hold/Light Buy - Modest gains expected"
                elif price_change_pct < -3 and model_accuracy > 80:
                    return "Sell/Avoid - High confidence downward movement"
                elif price_change_pct < -1 and model_accuracy > 70:
                    return "Hold/Reduce - Downward pressure anticipated"
                elif model_accuracy < 60 or abs(price_change_pct) < 1:
                    return "Hold - Low conviction forecast"
                else:
                    return "Hold - Mixed signals"

            investment_advice = get_investment_advice()

            try:
                prompt = f"""
Analyze this stock forecast for {symbol.upper()}:

Current Price: ${current_price:.2f}
Forecast Price: ${forecast_price:.2f} ({price_change_pct:+.2f}%)
Time Horizon: {days_ahead} days
Model: {model_name}
Model Accuracy: {model_accuracy:.1f}%
Recent Trend: {recent_trend}
Volatility: {volatility:.2f}%

Provide a concise 2-3 sentence analysis of this forecast, including:
1. The predicted price movement and its significance
2. The confidence level based on model accuracy
3. Any notable patterns or considerations from the recent trend
4. The investment advice: {investment_advice}

Keep the analysis professional and focused on key insights.
"""

                messages = [
                    {
                        "role": "system",
                        "content": "You are a professional financial analyst. Provide a clear, concise analysis of stock forecasts."
                    },
                    {"role": "user", "content": prompt}
                ]

                response = await self.openrouter_client.chat(
                    messages=messages,
                    temperature=0.3,
                    max_tokens=300
                )
                return response if response else ""
            except Exception as e:
                logger.error(f"Error in generate_forecast_insights_async: {str(e)}")
                return f"Unable to generate forecast insights: {str(e)}\n\nInvestment Recommendation: {investment_advice}"

        except OpenRouterError as e:
            logger.error(f"OpenRouter error in forecast insights: {e}")
            return f"Forecast: {symbol.upper()} expected change {price_change_pct:+.2f}%, Investment Advice: {investment_advice}"
        except Exception as e:
            logger.exception("Unexpected error in generate_forecast_insights_async")
            return f"Unexpected error: {str(e)}"

    def generate_forecast_insights(
        self, df: pd.DataFrame, symbol: str, current_price: float, forecast_price: float,
        days_ahead: int, model_name: str, model_accuracy: float
    ) -> str:
        """Synchronous wrapper for forecast insights."""
        return asyncio.run(self.generate_forecast_insights_async(df, symbol, current_price, forecast_price, days_ahead, model_name, model_accuracy))
