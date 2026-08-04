import pytest
from agent.explainer import ExplainabilityLayer

@pytest.fixture
def explainer() -> ExplainabilityLayer:
    """Fixture returning a fresh instance of ExplainabilityLayer."""
    return ExplainabilityLayer()

def test_enrich_reason_with_score(explainer):
    """Verify that credit score, tier labels, and credit factors are concatenated correctly."""
    reason = "Strong credit score; No missed payments"
    result = explainer.enrich_reason(reason, 380.0, "P1")
    
    assert "Credit Health Score: 380" in result
    assert "Premium tier" in result
    assert "Strong credit score" in result
    assert "No missed payments" in result
    assert result.count("-") == 4  # Score, tier, score factor, payment factor

def test_enrich_reason_without_score(explainer):
    """Verify that formatting skips credit scores gracefully if they are absent."""
    result = explainer.enrich_reason("Stable income", None, "P2")
    
    assert "Credit Health Score" not in result
    assert "Standard tier" in result
    assert "Stable income" in result

def test_enrich_reason_empty_string(explainer):
    """Verify empty raw reasons still format and return tier tags."""
    result = explainer.enrich_reason("", None, "P1")
    assert isinstance(result, str)
    assert "Premium tier" in result

def test_parse_reasons_multiple(explainer):
    """Verify splitting of semicolon-separated parameters into individual items."""
    reasons = explainer.parse_reasons("Factor one; Factor two; Factor three")
    assert len(reasons) == 3
    assert "Factor one" in reasons
    assert "Factor two" in reasons
    assert "Factor three" in reasons

def test_parse_reasons_empty(explainer):
    """Verify empty string parses return empty lists."""
    result = explainer.parse_reasons("")
    assert result == []

def test_parse_reasons_none(explainer):
    """Verify None parses return empty lists."""
    result = explainer.parse_reasons(None)
    assert result == []

def test_format_for_display(explainer):
    """Verify format_for_display packages structured metadata keys correctly."""
    result = explainer.format_for_display("Strong score; Good history", 350.0, "P1")
    
    assert "enriched_text" in result
    assert "reason_list" in result
    assert "tier_label" in result
    assert "credit_score_display" in result
    assert result["credit_score_display"] == "350"
    assert len(result["reason_list"]) == 2

def test_format_for_display_no_score(explainer):
    """Verify missing credit score maps to compliant default labels."""
    result = explainer.format_for_display("Some reason", None, "P2")
    assert result["credit_score_display"] == "Not available"
