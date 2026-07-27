import logging
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.llm.base import BaseLLM, LLMProviderError

class GroqLLM(BaseLLM):
    """Groq LLM provider implementation using the official groq SDK."""

    def __init__(self, api_key: str, model: str = "llama3-8b-8192", max_tokens: int = 1000) -> None:
        """Initialize the Groq LLM provider.

        Args:
            api_key (str): Groq API Key.
            model (str): Groq model name (default: llama3-8b-8192).
            max_tokens (int): Maximum output tokens (default: 1000).
        """
        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Groq.

        Args:
            system_prompt (str): System prompt.
            user_prompt (str): User prompt.

        Raises:
            LLMProviderError: If Groq API fails.

        Returns:
            str: Generated text content.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            content = response.choices[0].message.content
            if not content:
                raise LLMProviderError("Empty response returned from Groq API.")
            return content
        except Exception as e:
            self.logger.error(f"Groq generation failure: {str(e)}")
            raise LLMProviderError(f"Failed to generate message using Groq API: {str(e)}", cause=e)

    def health_check(self) -> bool:
        """Verify Groq connectivity with a ping request.

        Returns:
            bool: True if reachable, False otherwise.
        """
        try:
            # Low token query to check model availability and API key validity
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=10,
                messages=[
                    {"role": "system", "content": "ping"},
                    {"role": "user", "content": "ping"}
                ]
            )
            is_healthy = bool(response and response.choices and response.choices[0].message.content)
            self.logger.debug(f"Groq health check status: {is_healthy}")
            return is_healthy
        except Exception as e:
            self.logger.debug(f"Groq health check failed: {str(e)}")
            return False
