import logging
import traceback
import re
from typing import Optional

# Dynamic import to generate repayment schedules for excel sources
try:
    from dashboard import pipeline as pl
except ImportError:
    pl = None

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

def format_compact_inr(amount: float) -> str:
    """Format monetary values into compact Indian numbering style (e.g., ₹3.2L or ₹76K).

    Args:
        amount (float): Numeric monetary amount.

    Returns:
        str: Formatted compact currency string.
    """
    is_negative = amount < 0
    v = abs(amount)
    if v >= 100000:
        formatted = f"{v / 100000:.1f}L"
        if formatted.endswith(".0L"):
            formatted = formatted.replace(".0L", "L")
    elif v >= 1000:
        formatted = f"{v / 1000:.0f}K"
    else:
        formatted = f"{v:.0f}"
        
    if is_negative:
        return f"-₹{formatted}"
    return f"₹{formatted}"

def clean_for_whatsapp(text: str) -> str:
    # Remove markdown bold/italic that WhatsApp doesn't render
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

class MessageGenerator:
    """Uses the configured LLM provider to construct personalized, localized loan notification documents."""

    def __init__(self, llm, offer_validity_days: int = 30) -> None:
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
            "You are a professional loan notification assistant. Your job is to format a pre-approved loan message.\n\n"
            "CRITICAL RULES:\n"
            "1. You MUST generate the message in the EXACT template structure provided, with NO deviations, NO extra commentary, and NO greetings outside the template.\n"
            "2. Do NOT use markdown formatting (no bold '**', no italic '_', no bullet '*' characters outside the bullet list dashes). WhatsApp handles formatting differently.\n"
            "3. Keep the message under 300 words.\n"
            f"4. If the target language is NOT English, translate the entire message body into {language}, but KEEP emoji icons, number values, and placeholders/IDs as-is."
        )

    def _build_user_prompt(
        self,
        customer_name: str,
        loan_amount_lakhs: str,
        interest_rate_str: str,
        tenure_years: int,
        monthly_emi_str: str,
        repayment_type: str,
        risk_category: str,
        credit_score_range: str,
        repayment_snapshot: str,
        offer_validity_days: int
    ) -> str:
        """Construct the prompt representing customer specific variables to input to the LLM.

        Returns:
            str: User prompt string.
        """
        return (
            "Generate the WhatsApp loan notification using the exact values and template structure below:\n\n"
            "Values to inject:\n"
            f"- Customer Name: {customer_name}\n"
            f"- Loan Amount: {loan_amount_lakhs}\n"
            f"- Interest Rate: {interest_rate_str}\n"
            f"- Tenure: {tenure_years} years\n"
            f"- Monthly EMI: {monthly_emi_str}\n"
            f"- Repayment Type: {repayment_type}\n"
            f"- Risk Category: {risk_category}\n"
            f"- Credit Score Range: {credit_score_range}\n"
            f"- Repayment Snapshot: {repayment_snapshot}\n"
            f"- Validity Days: {offer_validity_days}\n\n"
            "STRUCTURE TEMPLATE (You MUST copy this structure exactly line-by-line):\n"
            "Dear [Customer Name],\n\n"
            "Great news! Based on your credit profile, you have been pre-approved for a personal loan.\n\n"
            "🏦 *Loan Offer Details*\n"
            "- Loan Amount: ₹[Loan Amount]\n"
            "- Interest Rate: [Interest Rate] p.a.\n"
            "- Tenure: [Tenure] years\n"
            "- Monthly EMI: [Monthly EMI]\n"
            "- Repayment Type: [Repayment Type]\n\n"
            "📊 *Your Credit Summary*\n"
            "- Risk Category: [Risk Category]\n"
            "- Credit Score Range: [Credit Score Range]\n\n"
            "💳 *Repayment Snapshot*\n"
            "[Repayment Snapshot]\n\n"
            "✅ This offer is valid for [Validity Days] days. Contact your nearest branch or reply to this message to proceed.\n\n"
            "Regards,\n"
            "FinRisk Lending Team"
        )

    def generate(self, eligibility_dict: dict, notification_id: str, language: str = "English", enriched_reason: str = "") -> str:
        """Compose the notification using the selected LLM provider.

        Args:
            eligibility_dict (dict): Underwriting result values.
            notification_id (str): Unique notification ID.
            language (str): Selected translation language (default: English).
            enriched_reason (str): Structured justification text (optional).

        Returns:
            str: Generated notification string.
        """
        # Resolve explanation text source
        reason_text = enriched_reason if enriched_reason else eligibility_dict.get("reason", "")
        
        # Generate repayment summary from schedule
        schedule = eligibility_dict.get("repayment_schedule")
        if not schedule and pl is not None:
            try:
                schedule = pl.repayment_schedule(
                    principal=float(eligibility_dict["loan_amount"]),
                    annual_rate_pct=float(eligibility_dict["interest_rate"]),
                    tenure_years=float(eligibility_dict["tenure_years"])
                )
            except Exception as e:
                self.logger.error(f"Failed to generate dynamic schedule in generator: {e}")
        
        # Build Repayment Snapshot
        tenure = int(eligibility_dict["tenure_years"])
        if schedule and len(schedule) > 0:
            try:
                y1_row = schedule[0]
                y1_principal_paid = y1_row.get("principal_paid", 0)
                repayment_snapshot = f"Clear {format_compact_inr(y1_principal_paid)} in Year 1 of {tenure} — your biggest milestone."
            except Exception as e:
                self.logger.error(f"Failed to format repayment snapshot: {e}")
                repayment_snapshot = f"Clear {format_compact_inr(eligibility_dict['loan_amount'])} in {tenure} years — your biggest milestone."
        else:
            repayment_snapshot = f"Clear {format_compact_inr(eligibility_dict['loan_amount'])} in {tenure} years — your biggest milestone."
            
        # Format variables
        customer_name = f"Customer #{eligibility_dict['prospect_id']}"
        loan_amount_lakhs = f"{eligibility_dict['loan_amount'] / 100000:.2f} Lakhs"
        interest_rate_str = f"{eligibility_dict['interest_rate']:.1f}%"
        monthly_emi_str = format_inr(eligibility_dict['monthly_emi'])
        
        repayment_method = eligibility_dict.get('repayment_method', '')
        if "step-up" in repayment_method.lower():
            repayment_type = "Step-Up EMI"
        else:
            repayment_type = "Fixed Monthly EMI"
            
        risk_cat_map = {"P1": "Best Risk", "P2": "Good Risk", "P3": "Moderate Risk", "P4": "High Risk"}
        risk_category = risk_cat_map.get(eligibility_dict["tier"], "Moderate Risk")
        
        score_range_map = {"P1": "750+", "P2": "700-749", "P3": "650-699", "P4": "<650"}
        credit_score_range = score_range_map.get(eligibility_dict["tier"], "650-699")

        system_prompt = self._build_system_prompt(language)
        user_prompt = self._build_user_prompt(
            customer_name=customer_name,
            loan_amount_lakhs=loan_amount_lakhs,
            interest_rate_str=interest_rate_str,
            tenure_years=tenure,
            monthly_emi_str=monthly_emi_str,
            repayment_type=repayment_type,
            risk_category=risk_category,
            credit_score_range=credit_score_range,
            repayment_snapshot=repayment_snapshot,
            offer_validity_days=self.offer_validity_days
        )

        try:
            message = self.llm.generate(system_prompt, user_prompt)
            self.logger.info(f"Notification message generated successfully for {notification_id}")
            cleaned_message = clean_for_whatsapp(message)
            self.logger.debug(f"Generated message body:\n{cleaned_message}")
            return cleaned_message
        except Exception as e:
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
        
        single_line = message.replace("\n", " ").strip()
        if len(single_line) > max_chars:
            return single_line[:max_chars].strip() + "..."
        return single_line
