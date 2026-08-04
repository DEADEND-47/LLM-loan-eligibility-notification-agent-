import logging
from typing import Optional

class ExplainabilityLayer:
    """Parses raw risk decisions into user-friendly banking bullet points."""

    def __init__(self) -> None:
        """Initialize the ExplainabilityLayer with standardized tier labeling."""
        self.logger = logging.getLogger(__name__)
        self.tier_labels = {
            "P1": "Premium tier — lowest available rate (8.5%)",
            "P2": "Standard tier — competitive rate (10.5%)"
        }

    def enrich_reason(self, reason_str: str, credit_score: Optional[float], tier: str) -> str:
        """Parse raw underwriting reason lists into compliance-friendly bullet points.

        Args:
            reason_str (str): Semicolon-delimited credit justification string.
            credit_score (Optional[float]): Credit Health Score.
            tier (str): Pre-approved classification tier.

        Returns:
            str: Multi-line, bulleted text string explaining approval parameters.
        """
        bullets = []

        if credit_score is not None:
            bullets.append(f"- Credit Health Score: {credit_score:.0f}")

        if tier in self.tier_labels:
            bullets.append(f"- {self.tier_labels[tier]}")
        else:
            self.logger.warning(f"Unknown tier: {tier}")

        if not reason_str:
            self.logger.warning("Empty reason string received")
        else:
            parts = reason_str.split(";")
            for part in parts:
                clean_part = part.strip()
                if clean_part:
                    bullets.append(f"- {clean_part}")

        return "\n".join(bullets)

    def parse_reasons(self, reason_str: str) -> list:
        """Utility to split semicolon separated lists of credit risk parameters.

        Args:
            reason_str (str): Raw string.

        Returns:
            list: List of stripped strings.
        """
        if not reason_str:
            return []
        
        parts = reason_str.split(";")
        return [part.strip() for part in parts if part.strip()]

    def format_for_display(self, reason_str: str, credit_score: Optional[float], tier: str) -> dict:
        """Structure credit decision justifications for API endpoints or log exports.

        Args:
            reason_str (str): Semicolon-delimited credit justification string.
            credit_score (Optional[float]): Credit Health Score.
            tier (str): Pre-approved classification tier.

        Returns:
            dict: Structured explainability payload.
        """
        enriched = self.enrich_reason(reason_str, credit_score, tier)
        parsed = self.parse_reasons(reason_str)
        tier_label = self.tier_labels.get(tier, "Unknown tier")
        credit_score_display = f"{credit_score:.0f}" if credit_score is not None else "Not available"

        return {
            "enriched_text": enriched,
            "reason_list": parsed,
            "tier_label": tier_label,
            "credit_score_display": credit_score_display
        }
