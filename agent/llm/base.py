from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

class LLMProviderError(Exception):
    """Custom exception raised when an LLM provider API call fails."""

    def __init__(self, message: str, cause: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause

class BaseLLM(ABC):
    """Abstract Base Class representing a generic Large Language Model provider."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response using the configured LLM provider.

        Args:
            system_prompt (str): Instructions governing the model behavior.
            user_prompt (str): The primary input request for the model.

        Raises:
            LLMProviderError: If the provider API call fails after retries.

        Returns:
            str: The generated text response.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Verify if the configured LLM provider is reachable.

        Returns:
            bool: True if the provider is healthy and reachable, False otherwise.
        """
        pass
