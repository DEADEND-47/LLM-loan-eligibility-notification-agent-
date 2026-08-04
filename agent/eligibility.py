import logging
import math
from typing import Optional, Dict, Any

class EligibilityChecker:
    """Evaluates customer record eligibility tiers and parses individual financial fields."""

    def __init__(self, config: dict) -> None:
        """Initialize the EligibilityChecker.

        Args:
            config (dict): Global configuration settings dictionary.
        """
        self.approved_tiers = ["P1", "P2", "P3"]
        self.declined_tier = "P4"
        self.logger = logging.getLogger(__name__)

    def check(self, row: Dict[str, Any]) -> Optional[dict]:
        """Verify credit flag eligibility and construct a formatted prospect dictionary.

        Args:
            row (Dict[str, Any]): A single row dictionary representing a customer record.

        Returns:
            Optional[dict]: Cleaned eligibility dict if approved, otherwise None.
        """
        approved_flag = row.get("Approved_Flag")

        if approved_flag not in self.approved_tiers and approved_flag != self.declined_tier:
            self.logger.warning(
                f"Unknown Approved_Flag value: {approved_flag} for PROSPECTID {row.get('PROSPECTID')}"
            )
            return None

        if approved_flag == self.declined_tier:
            return None

        # Parse Credit Health Score (handling Nulls / NaNs gracefully)
        value = row.get("Credit_Health_Score")
        if value is None or (isinstance(value, float) and math.isnan(value)):
            credit_score = None
        else:
            try:
                credit_score = float(value)
            except (ValueError, TypeError):
                credit_score = None

        # Parse Income-to-TradeLine Ratio
        ratio_val = row.get("Income_TL_Ratio", 0)
        try:
            income_tl_ratio = float(ratio_val) if ratio_val else 0.0
        except (ValueError, TypeError):
            income_tl_ratio = 0.0

        is_income_outlier = income_tl_ratio > 100000

        # Parse Net Monthly Income
        income_val = row.get("NETMONTHLYINCOME", 0)
        try:
            net_monthly_income = float(income_val) if income_val is not None else 0.0
        except (ValueError, TypeError):
            net_monthly_income = 0.0

        try:
            return {
                "prospect_id": int(row["PROSPECTID"]),
                "tier": str(approved_flag),
                "loan_amount": float(row["Recommended_Loan_Amount"]),
                "interest_rate": float(row["Interest_Rate_Pct"]),
                "tenure_years": int(row["Tenure_Years"]),
                "repayment_method": str(row["Repayment_Method"]),
                "total_interest": float(row["Total_Interest_Payable"]),
                "total_payable": float(row["Total_Amount_Payable"]),
                "monthly_emi": float(row["Monthly_EMI"]),
                "reason": str(row["Reason_For_Approval"]),
                "credit_score": credit_score,
                "income_tl_ratio": income_tl_ratio,
                "is_income_outlier": is_income_outlier,
                "contact_no": str(row.get("Contact_No", "")),
                "repayment_schedule": row.get("repayment_schedule", []),
                "net_monthly_income": net_monthly_income
            }
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(
                f"Failed parsing eligibility keys for PROSPECTID {row.get('PROSPECTID')}: {str(e)}"
            )
            return None

    def is_eligible(self, row: Dict[str, Any]) -> bool:
        """Helper to quickly check if a record falls into an approved tier.

        Args:
            row (Dict[str, Any]): Row data.

        Returns:
            bool: True if eligible, False otherwise.
        """
        approved_flag = row.get("Approved_Flag")
        return approved_flag in self.approved_tiers

def validate_loan_offer(applicant_dict: Dict[str, Any]) -> tuple[bool, str]:
    """Validate that the given applicant's loan details satisfy all underwriting business rules.

    Args:
        applicant_dict (Dict[str, Any]): Parsed applicant details dictionary.

    Returns:
        tuple[bool, str]: (is_valid, reason) where is_valid is True if all rules pass, 
                          else False and reason is the rejection remark.
    """
    # 1. Risk Tier filter: Only pass P1, P2, P3. Reject P4 immediately.
    tier = applicant_dict.get("tier")
    if tier not in ["P1", "P2", "P3"]:
        return False, "Risk tier P4 is rejected"

    # 2. Minimum loan amount filter: loan_amount must be > 50000.
    loan_amount = applicant_dict.get("loan_amount")
    if loan_amount is None or loan_amount <= 50000:
        return False, "Loan amount below minimum threshold"

    # 3. Minimum EMI filter: monthly_emi must be > 0.
    monthly_emi = applicant_dict.get("monthly_emi")
    if monthly_emi is None or monthly_emi <= 0:
        return False, "EMI calculation failed or income too low"

    # 4. FOIR check: monthly_emi must be <= 30% of the applicant's net monthly income.
    net_monthly_income = applicant_dict.get("net_monthly_income", 0.0)
    if net_monthly_income <= 0.0:
        return False, "EMI exceeds FOIR limit"
    
    # Calculate FOIR ratio
    foir = monthly_emi / net_monthly_income
    # Avoid small float precision edge cases by rounding to 4 decimals (e.g. 30.0001% vs 30%)
    if round(foir, 4) > 0.30:
        return False, "EMI exceeds FOIR limit"

    # 5. Income floor check: net_monthly_income must be >= 5657.
    if net_monthly_income < 5657:
        return False, "Income below minimum threshold"

    return True, ""
