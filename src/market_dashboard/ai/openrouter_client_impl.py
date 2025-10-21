"""
OpenRouter API client for interacting with various AI models.
"""
import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
import httpx

logger = logging.getLogger(__name__)

class OpenRouterError(Exception):
    """Custom exception for OpenRouter API errors."""
    pass

class OpenRouterClient:
    """Client for interacting with the OpenRouter API."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If not provided, will try to get from OPENROUTER_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Either pass api_key parameter or set OPENROUTER_API_KEY env var.")
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "mistralai/mistral-7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate a chat completion using OpenRouter.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: The model to use (default: mistralai/mistral-7b-instruct:free).
            temperature: Controls randomness (0.0 to 1.0).
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            The generated text content.
            
        Raises:
            OpenRouterError: If there's an error with the API request.
        """
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            
            # Create a fresh AsyncClient per call to avoid cross-event-loop issues
            async with httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/Adamderbel/AI-Market-Pulse-Dashboard",
                    "X-Title": "AI Market Pulse Dashboard",
                },
                timeout=30.0,
            ) as client:
                response = await client.post(
                    "/chat/completions",
                    json=payload,
                )
            
            response.raise_for_status()
            result = response.json()
            
            if "choices" not in result or not result["choices"]:
                raise OpenRouterError("No choices in response")
                
            return result["choices"][0]["message"]["content"]
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(error_msg)
            raise OpenRouterError(error_msg) from e
            
        except (json.JSONDecodeError, KeyError) as e:
            error_msg = f"Error parsing response: {str(e)}"
            logger.error(error_msg)
            raise OpenRouterError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            raise OpenRouterError(error_msg) from e

    def chat_sync(
        self,
        messages: List[Dict[str, str]],
        model: str = "mistralai/mistral-7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> str:
        """Synchronous wrapper around chat for use in non-async contexts (e.g., Dash callbacks)."""
        return asyncio.run(self.chat(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs))

# For backward compatibility
OpenRouterClient = OpenRouterClient
