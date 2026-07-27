import os
import logging
from agent.llm.base import BaseLLM, LLMProviderError
from agent.llm.gemini import GeminiLLM
from agent.llm.groq import GroqLLM
from agent.llm.ollama import OllamaLLM

def create_llm_provider(config: dict) -> BaseLLM:
    """Create and configure an LLM provider based on the settings configuration.

    Args:
        config (dict): Configuration dictionary containing provider settings.

    Raises:
        LLMProviderError: If config is invalid or env key is missing.

    Returns:
        BaseLLM: Concretely instantiated LLM provider.
    """
    logger = logging.getLogger(__name__)
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "").lower().strip()

    if provider == "gemini":
        api_key_env = llm_config.get("api_key_env", "GEMINI_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise LLMProviderError("GEMINI_API_KEY not set in environment")
        
        provider_instance = GeminiLLM(
            api_key=api_key,
            model=llm_config.get("gemini_model", "gemini-1.5-flash"),
            max_tokens=llm_config.get("max_tokens", 1000)
        )

    elif provider == "groq":
        api_key_env = llm_config.get("api_key_env", "GROQ_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise LLMProviderError("GROQ_API_KEY not set in environment")
        
        provider_instance = GroqLLM(
            api_key=api_key,
            model=llm_config.get("groq_model", "llama3-8b-8192"),
            max_tokens=llm_config.get("max_tokens", 1000)
        )

    elif provider == "ollama":
        provider_instance = OllamaLLM(
            base_url=llm_config.get("ollama_base_url", "http://localhost:11434"),
            model=llm_config.get("ollama_model", "llama3.2"),
            max_tokens=llm_config.get("max_tokens", 1000)
        )

    else:
        raise LLMProviderError(
            f"Unknown LLM provider: '{provider}'. Must be one of: gemini, groq, ollama"
        )

    logger.info(f"Successfully initialized LLM provider: {provider_instance.__class__.__name__}")
    return provider_instance

def get_llm_health(llm: BaseLLM) -> dict:
    """Check the connection health status of the LLM provider.

    Args:
        llm (BaseLLM): An instantiated LLM provider class.

    Returns:
        dict: A health report dictionary containing provider name, status, and health boolean.
    """
    try:
        is_healthy = llm.health_check()
        return {
            "provider": type(llm).__name__,
            "healthy": is_healthy,
            "status": "ok" if is_healthy else "unreachable"
        }
    except Exception as e:
        return {
            "provider": type(llm).__name__,
            "healthy": False,
            "status": f"error: {str(e)}"
        }
