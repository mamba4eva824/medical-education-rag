"""Tests for the response validation guardrails."""

from src.generation.guardrails import validate_response


class TestGuardrails:
    """Tests for validate_response()."""

    def test_blocks_medical_advice(self, sample_sources):
        """Prohibited phrases like 'self-diagnose' are caught."""
        result = validate_response(
            "You should self-diagnose this condition.", sample_sources
        )
        assert result["within_scope"] is False
        assert result["passed"] is False

    def test_detects_hallucinated_citations(self, sample_sources):
        """Out-of-range [N] citations are flagged."""
        # 3 sources provided, but citing [5]
        result = validate_response(
            "This is supported by source [5] and research.", sample_sources
        )
        assert result["no_hallucinated_citations"] is False
        assert result["passed"] is False

    def test_passes_valid_response(self, sample_sources):
        """Clean responses with proper citations pass all checks."""
        response = (
            "Heart failure symptoms include edema [1] and fatigue. "
            "Hypertension is managed with ACE inhibitors [2]. "
            "Blood glucose monitoring is important for diabetes [3]."
        )
        result = validate_response(response, sample_sources)
        assert result["has_citations"] is True
        assert result["within_scope"] is True
        assert result["not_empty"] is True
        assert result["source_grounded"] is True
        assert result["no_hallucinated_citations"] is True
        assert result["passed"] is True

    def test_detects_empty_response(self, sample_sources):
        """Empty or whitespace-only responses are caught."""
        result = validate_response("   ", sample_sources)
        assert result["not_empty"] is False
        assert result["passed"] is False

    def test_detects_missing_citations(self, sample_sources):
        """Responses without [N] citation markers are flagged."""
        result = validate_response(
            "Heart failure symptoms include edema and fatigue from sources.",
            sample_sources,
        )
        assert result["has_citations"] is False
        assert result["passed"] is False

    def test_returns_dict(self, sample_sources):
        """validate_response always returns a plain dict."""
        result = validate_response("test [1]", sample_sources)
        assert isinstance(result, dict)
        assert "passed" in result
        assert "within_scope" in result
