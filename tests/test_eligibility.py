import pytest
import math
from agent.eligibility import EligibilityChecker

@pytest.fixture
def config() -> dict:
    """Fixture returning configuration mapping for eligibility testing."""
    return {
        "eligibility": {
            "approved_tiers": ["P1", "P2"],
            "declined_tier": "P3"
        }
    }

@pytest.fixture
def checker(config) -> EligibilityChecker:
    """Fixture initializing EligibilityChecker."""
    return EligibilityChecker(config)

@pytest.fixture
def p1_row() -> dict:
    """Fixture representing a pre-approved Premium (P1) customer row."""
    return {
        "PROSPECTID": 101,
        "Approved_Flag": "P1",
        "Risk_Tier": "P1",
        "Recommended_Loan_Amount": 1200000.0,
        "Interest_Rate_Pct": 8.5,
        "Tenure_Years": 5,
        "Repayment_Method": "EMI Fixed Monthly",
        "Total_Interest_Payable": 270000.0,
        "Total_Amount_Payable": 1470000.0,
        "Monthly_EMI": 24500.0,
        "Reason_For_Approval": "Strong credit score; No missed payments",
        "Credit_Health_Score": 380.0,
        "Income_TL_Ratio": 25000.0
    }

@pytest.fixture
def p2_row(p1_row) -> dict:
    """Fixture representing a standard (P2) pre-approved customer row."""
    row = p1_row.copy()
    row["Approved_Flag"] = "P2"
    row["Interest_Rate_Pct"] = 10.5
    row["Tenure_Years"] = 4
    return row

@pytest.fixture
def p3_row(p1_row) -> dict:
    """Fixture representing a declined (P3) customer row."""
    row = p1_row.copy()
    row["Approved_Flag"] = "P3"
    return row

@pytest.fixture
def null_credit_row(p1_row) -> dict:
    """Fixture representing a customer with missing credit health score."""
    row = p1_row.copy()
    row["Credit_Health_Score"] = float('nan')
    return row

@pytest.fixture
def outlier_income_row(p1_row) -> dict:
    """Fixture representing a customer with outlier trade line ratios."""
    row = p1_row.copy()
    row["Income_TL_Ratio"] = 999999.0
    return row

def test_p1_eligible(checker, p1_row):
    """Verify check parses tier P1 parameters correctly."""
    result = checker.check(p1_row)
    assert result is not None
    assert result["tier"] == "P1"
    assert result["interest_rate"] == 8.5
    assert result["prospect_id"] == 101

def test_p2_eligible(checker, p2_row):
    """Verify check parses tier P2 parameters correctly."""
    result = checker.check(p2_row)
    assert result is not None
    assert result["tier"] == "P2"
    assert result["interest_rate"] == 10.5

def test_p3_declined(checker, p3_row):
    """Verify check returns None for tier P3 declined rows."""
    result = checker.check(p3_row)
    assert result is None

def test_null_credit_score(checker, null_credit_row):
    """Verify NaN credit scores are converted to None."""
    result = checker.check(null_credit_row)
    assert result is not None
    assert result["credit_score"] is None

def test_income_outlier_flagged(checker, outlier_income_row):
    """Verify extreme income ratios are flagged as outliers."""
    result = checker.check(outlier_income_row)
    assert result is not None
    assert result["is_income_outlier"] is True

def test_is_eligible_true(checker, p1_row):
    """Verify helper is_eligible works for pre-approvals."""
    assert checker.is_eligible(p1_row) is True

def test_is_eligible_false(checker, p3_row):
    """Verify helper is_eligible returns False for declines."""
    assert checker.is_eligible(p3_row) is False
