import logging
from google.genai import Client
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.llm.base import BaseLLM, LLMProviderError

class GeminiLLM(BaseLLM):
    """Google Gemini LLM provider implementation using the google-genai SDK."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", max_tokens: int = 1000) -> None:
        """Initialize the Gemini LLM provider.

        Args:
            api_key (str): Google Gemini API Key.
            model (str): Gemini model name (default: gemini-1.5-flash).
            max_tokens (int): Maximum output tokens (default: 1000).
        """
        self.client = Client(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Google Gemini.

        Args:
            system_prompt (str): Instructions governing model behavior.
            user_prompt (str): The primary user query.

        Raises:
            LLMProviderError: If Google Gemini API fails.

        Returns:
            str: Generated text content.
        """
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        try:
            # We can also pass config options if needed, but simple generate_content matches the spec.
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
            )
            if not response or not response.text:
                raise LLMProviderError("Empty response returned from Gemini API.")
            return response.text
        except Exception as e:
            self.logger.error(f"Gemini generation failure: {str(e)}")
            raise LLMProviderError(f"Failed to generate message using Gemini API: {str(e)}", cause=e)

    def health_check(self) -> bool:
        """Verify Gemini connectivity with a ping request.

        Returns:
            bool: True if reachable, False otherwise.
        """
        try:
            # Low token query to check model availability and API key validity
            response = self.client.models.generate_content(
                model=self.model,
                contents="ping",
            )
            is_healthy = bool(response and response.text)
            self.logger.debug(f"Gemini health check status: {is_healthy}")
            return is_healthy
        except Exception as e:
            self.logger.debug(f"Gemini health check failed: {str(e)}")
            return False
