import logging
import traceback
from typing import Optional
from agent.llm.base import BaseLLM, LLMProviderError

def format_inr(amount: float) -> str:
    """Format a numeric value into the Indian numbering style (e.g., ₹2,50,00,000).

    Args:
        amount (float): Numeric monetary amount.

    Returns:
        str: Formatted currency string prefixing the Indian Rupee symbol.
    """
    is_negative = amount < 0
    abs_amount = abs(amount)
    
    amount_int = int(round(abs_amount))
    amount_str = str(amount_int)

    if len(amount_str) <= 3:
        result = amount_str
    else:
        last_three = amount_str[-3:]
        remaining = amount_str[:-3]
        
        groups = []
        while len(remaining) > 2:
            groups.append(remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.append(remaining)
            
        groups.reverse()
        result = ",".join(groups) + "," + last_three

    if is_negative:
        return f"-₹{result}"
    return f"₹{result}"

class MessageGenerator:
    """Uses the configured LLM provider to construct personalized, localized loan notification documents."""

    def __init__(self, llm: BaseLLM, offer_validity_days: int = 30) -> None:
        """Initialize the MessageGenerator.

        Args:
            llm (BaseLLM): An instantiated LLM provider.
            offer_validity_days (int): Days the credit offer remains valid (default: 30).
        """
        self.llm = llm
        self.offer_validity_days = offer_validity_days
        self.logger = logging.getLogger(__name__)

    def _build_system_prompt(self, language: str) -> str:
        """Build the system instructions guiding the model tone and criteria rules.

        Args:
            language (str): The target language.

        Returns:
            str: Formatting instructions.
        """
        return (
            "You are a professional loan notification officer at a leading Indian bank. Your job is to write "
            "a clean, modern, and professional WhatsApp message informing a customer about their loan pre-approval.\n\n"
            "Style and Format Rules:\n"
            "- Always be professional, warm, and concise. Tone down corporate jargon.\n"
            "- Keep the total length around 80-150 words.\n"
            "- Use emojis sparingly and appropriately for a business context (e.g. 📢, ✅, 📋).\n"
            "- Ensure it looks good on WhatsApp by using clean line breaks and bullet points.\n"
            "- Do not include duplicate sections or repeat the details multiple times.\n"
            "- Do not write a long letter format. Keep paragraphs very short.\n"
            f"- Write in {language}."
        )

    def _build_user_prompt(self, eligibility_dict: dict, notification_id: str, language: str, reason_text: str) -> str:
        """Construct the prompt representing customer specific variables to input to the LLM.

        Args:
            eligibility_dict (dict): Dictionary with customer details.
            notification_id (str): Generated transaction key.
            language (str): Target translation language.
            reason_text (str): Enriched text bullets.

        Returns:
            str: User prompt string.
        """
        loan_amount_str = format_inr(eligibility_dict["loan_amount"])
        monthly_emi_str = format_inr(eligibility_dict["monthly_emi"])
        total_interest_str = format_inr(eligibility_dict["total_interest"])
        total_payable_str = format_inr(eligibility_dict["total_payable"])
        
        credit_score = eligibility_dict.get("credit_score")
        credit_score_str = f"{credit_score:.0f}" if credit_score is not None else "Not available"

        return (
            f"Write a WhatsApp message with the following details:\n\n"
            f"Notification ID: {notification_id}\n"
            f"Customer Prospect ID: {eligibility_dict['prospect_id']}\n"
            f"Tier: {eligibility_dict['tier']}\n"
            f"Approved Loan Amount: {loan_amount_str}\n"
            f"Interest Rate: {eligibility_dict['interest_rate']}% p.a.\n"
            f"Tenure: {eligibility_dict['tenure_years']} years\n"
            f"Monthly EMI: {monthly_emi_str}\n"
            f"Total Interest Payable: {total_interest_str}\n"
            f"Total Amount Payable: {total_payable_str}\n"
            f"Repayment Method: {eligibility_dict['repayment_method']}\n"
            f"Offer Validity: {self.offer_validity_days} days\n"
            f"Credit Health Score: {credit_score_str}\n"
            f"Approval Highlights (Use 2-3 points from this verbatim where possible):\n{reason_text}\n\n"
            f"Structure the message exactly as follows, with no extra sections:\n"
            f"1. Warm Greeting (Dear customer / Dear Mr./Ms. and eligibility statement)\n"
            f"2. Pre-approved loan details:\n"
            f"   - Approved Amount: {loan_amount_str}\n"
            f"   - Interest Rate: {eligibility_dict['interest_rate']}% p.a.\n"
            f"   - Monthly EMI: {monthly_emi_str}\n"
            f"3. Approval Highlights (2-3 brief bullet points based on the highlights above)\n"
            f"4. Next Steps (Simple instructions on how to claim using Notification ID: {notification_id} in branch or net banking within {self.offer_validity_days} days)\n"
            f"5. Short closing disclaimer (e.g. terms apply)."
        )

    def generate(self, eligibility_dict: dict, notification_id: str, language: str = "English", enriched_reason: str = "") -> str:
        """Compose the notification using the selected LLM provider.

        Args:
            eligibility_dict (dict): Underwriting result values.
            notification_id (str): Unique notification ID.
            language (str): Selected translation language (default: English).
            enriched_reason (str): Structured justification text (optional).

        Raises:
            LLMProviderError: If the model API call fails after retries.

        Returns:
            str: Generated notification string.
        """
        # Resolve explanation text source
        reason_text = enriched_reason if enriched_reason else eligibility_dict.get("reason", "")
        
        system_prompt = self._build_system_prompt(language)
        user_prompt = self._build_user_prompt(eligibility_dict, notification_id, language, reason_text)

        try:
            message = self.llm.generate(system_prompt, user_prompt)
            self.logger.info(f"Notification message generated successfully for {notification_id}")
            return message
        except LLMProviderError as e:
            self.logger.critical(
                f"LLM generation failed for {notification_id}. Traceback:\n{traceback.format_exc()}"
            )
            raise e

    def generate_preview(self, message: str, max_chars: int = 200) -> str:
        """Create a single-line summary preview of the notification text for audit logs.

        Args:
            message (str): Full notification message text.
            max_chars (int): Maximum preview length (default: 200).

        Returns:
            str: Cleaned single-line summary string.
        """
        if not message:
            return ""
        
        # Replace newlines with spaces for log file storage formats
        single_line = message.replace("\n", " ").strip()
        if len(single_line) > max_chars:
            return single_line[:max_chars].strip() + "..."
        return single_line
