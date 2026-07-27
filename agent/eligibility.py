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
        self.approved_tiers = config["eligibility"]["approved_tiers"]
        self.declined_tier = config["eligibility"]["declined_tier"]
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
                "is_income_outlier": is_income_outlier
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
