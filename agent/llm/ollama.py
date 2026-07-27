import json
import logging
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.llm.base import BaseLLM, LLMProviderError

class OllamaLLM(BaseLLM):
    """Ollama local offline LLM provider implementation using httpx."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2", max_tokens: int = 1000) -> None:
        """Initialize the Ollama LLM provider.

        Args:
            base_url (str): Base URL of the Ollama server (default: http://localhost:11434).
            model (str): Ollama model name (default: llama3.2).
            max_tokens (int): Maximum output tokens (default: 1000).
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.max_tokens = max_tokens
        self.client = httpx.Client(timeout=60.0)
        self.logger = logging.getLogger(__name__)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using local Ollama instance.

        Args:
            system_prompt (str): System prompt.
            user_prompt (str): User prompt.

        Raises:
            LLMProviderError: If the Ollama service returns an error or is unreachable.

        Returns:
            str: Generated text content.
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {"num_predict": self.max_tokens}
        }
        try:
            response = self.client.post(url, json=payload)
            if response.status_code != 200:
                raise LLMProviderError(f"Ollama API returned non-200 status code: {response.status_code}")
            
            data = response.json()
            content = data.get("message", {}).get("content")
            if content is None:
                raise LLMProviderError("Invalid JSON structure in response from Ollama API.")
            return str(content)
        except Exception as e:
            self.logger.error(f"Ollama generation failure: {str(e)}")
            raise LLMProviderError(f"Failed to generate message using Ollama API: {str(e)}", cause=e)

    def health_check(self) -> bool:
        """Verify local Ollama connectivity.

        Returns:
            bool: True if reachable, False otherwise.
        """
        url = f"{self.base_url}/api/tags"
        try:
            response = self.client.get(url)
            is_healthy = response.status_code == 200
            self.logger.debug(f"Ollama health check status: {is_healthy}")
            return is_healthy
        except Exception as e:
            self.logger.debug(f"Ollama health check failed: {str(e)}")
            return False

    def __del__(self) -> None:
        """Destructor to ensure HTTP client is closed safely."""
        try:
            self.client.close()
        except Exception:
            pass
