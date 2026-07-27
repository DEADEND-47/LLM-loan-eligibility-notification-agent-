import logging
from typing import Optional

class LanguageSelector:
    """Detects and validates customer language preferences, providing safe defaults."""

    def __init__(self, default_lang: str = "English") -> None:
        """Initialize LanguageSelector.

        Args:
            default_lang (str): Configured fallback language (default: English).
        """
        self.default_lang = default_lang
        self.logger = logging.getLogger(__name__)
        self.supported_languages = [
            "English", "Hindi", "Tamil", "Bengali",
            "Telugu", "Marathi", "Gujarati", "Kannada"
        ]

    def detect_language(self, prospect_id: int, row_dict: dict) -> str:
        """Determine the notification language preference from customer row values.

        Args:
            prospect_id (int): Customer prospect ID.
            row_dict (dict): Dictionary representing raw data row fields.

        Returns:
            str: Resolved language name (title case).
        """
        if "preferred_language" in row_dict:
            value = row_dict["preferred_language"]
            if value is not None and isinstance(value, str) and value.strip():
                normalized = value.strip().title()
                if normalized in self.supported_languages:
                    self.logger.debug(f"Language '{normalized}' selected for PROSPECTID {prospect_id}")
                    return normalized
                else:
                    self.logger.warning(
                        f"Unsupported language '{value}' for PROSPECTID {prospect_id}. "
                        f"Falling back to {self.default_lang}"
                    )
                    return self.default_lang
            else:
                self.logger.debug(
                    f"Invalid or empty language preference for PROSPECTID {prospect_id}. "
                    f"Using default: {self.default_lang}"
                )
                return self.default_lang
        
        self.logger.debug(
            f"No language preference found for PROSPECTID {prospect_id}. "
            f"Using default: {self.default_lang}"
        )
        return self.default_lang

    def is_supported(self, language: str) -> bool:
        """Check if the provided language is in the bank's supported list.

        Args:
            language (str): Name of the language.

        Returns:
            bool: True if supported, False otherwise.
        """
        if not language or not isinstance(language, str):
            return False
        return language.strip().title() in self.supported_languages

    def get_supported_languages(self) -> list:
        """Return the copy of the supported languages list.

        Returns:
            list: Supported languages.
        """
        return list(self.supported_languages)
