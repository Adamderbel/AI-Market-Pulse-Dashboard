"""
Ollama client for AI-powered market insights.
Handles communication with local Ollama server.
"""
import os
import time
from typing import Any, Dict, List, Optional
import logging
import httpx

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Custom exception for Ollama-related errors."""
    pass


class OllamaClient:
    """Client for communicating with Ollama server."""
    
    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: float = 30.0,
        retries: int = 2
    ):
        """
        Initialize Ollama client.
        
        Args:
            host: Ollama server host URL
            model: Default model to use
            temperature: Default temperature for generation
            timeout: Request timeout in seconds
            retries: Number of retry attempts
        """
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "mistral:latest")
        self.temperature = temperature if temperature is not None else float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
        self.timeout = timeout
        self.retries = retries
    
    def is_server_available(self, timeout: float = 2.0) -> bool:
        """
        Check if Ollama server is available.
        
        Args:
            timeout: Timeout for availability check
            
        Returns:
            True if server is available, False otherwise
        """
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(f"{self.host}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama server not available: {e}")
            return False
    
    def _post_json(
        self,
        path: str,
        payload: Dict[str, Any],
        timeout: Optional[float] = None,
        retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Make POST request to Ollama API.
        
        Args:
            path: API endpoint path
            payload: Request payload
            timeout: Request timeout
            retries: Number of retry attempts
            
        Returns:
            Response JSON data
            
        Raises:
            OllamaError: If request fails
        """
        url = f"{self.host}{path}"
        timeout = timeout or self.timeout
        retries = retries if retries is not None else self.retries
        last_exc: Optional[Exception] = None
        
        # Quick server availability check
        if not self.is_server_available(timeout=2.0):
            raise OllamaError(f"Ollama server not reachable at {self.host}")
        
        for attempt in range(retries):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(url, json=payload)
                
                if response.status_code != 200:
                    raise OllamaError(f"HTTP {response.status_code}: {response.text[:500]}")
                
                return response.json()
                
            except Exception as e:
                last_exc = e
                if attempt == retries - 1:
                    break
                
                # Exponential backoff
                wait_time = 1.0 * (2 ** attempt)
                logger.debug(f"Retry {attempt + 1}/{retries} after {wait_time}s")
                time.sleep(wait_time)
        
        raise OllamaError(f"Failed to call Ollama at {url}: {last_exc}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate response using chat API.
        
        Args:
            messages: List of message dictionaries
            model: Model to use (overrides default)
            temperature: Temperature for generation (overrides default)
            max_tokens: Maximum tokens to generate
            options: Additional options
            
        Returns:
            Generated response text
            
        Raises:
            OllamaError: If generation fails
        """
        opts = dict(options or {})
        
        if temperature is None:
            temperature = self.temperature
        opts["temperature"] = temperature
        
        if max_tokens is not None:
            opts["num_predict"] = max_tokens
        
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": opts,
        }
        
        try:
            data = self._post_json("/api/chat", payload)
            content = data["message"]["content"]
            
            if not isinstance(content, str):
                raise TypeError("Invalid content type from Ollama")
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            raise OllamaError(f"Failed to generate response: {e}")
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate response using generate API.
        
        Args:
            prompt: Input prompt
            model: Model to use (overrides default)
            temperature: Temperature for generation (overrides default)
            max_tokens: Maximum tokens to generate
            options: Additional options
            
        Returns:
            Generated response text
            
        Raises:
            OllamaError: If generation fails
        """
        opts = dict(options or {})
        
        if temperature is None:
            temperature = self.temperature
        opts["temperature"] = temperature
        
        if max_tokens is not None:
            opts["num_predict"] = max_tokens
        
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": opts,
        }
        
        try:
            data = self._post_json("/api/generate", payload)
            content = data["response"]
            
            if not isinstance(content, str):
                raise TypeError("Invalid response type from Ollama")
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"Ollama generate error: {e}")
            raise OllamaError(f"Failed to generate response: {e}")
