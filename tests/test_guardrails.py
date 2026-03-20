"""Tests for the response guardrails module."""

from src.generation.guardrails import validate_response


def _make_sources(texts: list[str]) -> list[dict]:
    """Helper to build source dicts in the expected shape."""
    return [{"doc": {"text": t}} for t in texts]


class TestNotEmpty:
    def test_empty_string_fails(self):
        result = validate_response("", _make_sources(["some text"]))
        assert result["not_empty"] is False
        assert result["passed"] is False

    def test_whitespace_only_fails(self):
        result = validate_response("   ", _make_sources(["some text"]))
        assert result["not_empty"] is False

    def test_non_empty_passes(self):
        result = validate_response("Some answer [1]", _make_sources(["Some answer"]))
        assert result["not_empty"] is True


class TestHasCitations:
    def test_with_valid_citations(self):
        sources = _make_sources(["Heart failure info", "Diabetes info"])
        result = validate_response("Answer based on [1] and [2].", sources)
        assert result["has_citations"] is True

    def test_without_citations(self):
        sources = _make_sources(["Heart failure info"])
        result = validate_response("Answer without any references.", sources)
        assert result["has_citations"] is False


class TestWithinScope:
    def test_catches_self_diagnose(self):
        sources = _make_sources(["Heart failure symptoms include edema and fatigue."])
        result = validate_response(
            "You should self-diagnose this condition.", sources,
        )
        assert result["within_scope"] is False

    def test_catches_self_medicate(self):
        sources = _make_sources(["Treatment info"])
        result = validate_response("You can self-medicate with aspirin.", sources)
        assert result["within_scope"] is False

    def test_educational_content_passes(self):
        sources = _make_sources(["Heart failure is a chronic condition."])
        result = validate_response(
            "Heart failure is a chronic condition [1].", sources,
        )
        assert result["within_scope"] is True


class TestSourceGrounded:
    def test_above_threshold(self):
        source_text = "Heart failure symptoms include edema fatigue and shortness of breath"
        response = "Heart failure symptoms include edema and fatigue [1]."
        result = validate_response(response, _make_sources([source_text]))
        assert result["source_grounded"] is True

    def test_below_threshold(self):
        source_text = "Completely unrelated topic about astronomy and stars"
        response = "The patient should monitor blood pressure regularly [1]."
        result = validate_response(response, _make_sources([source_text]))
        assert result["source_grounded"] is False

    def test_no_citations_means_not_grounded(self):
        result = validate_response(
            "Answer without citations.", _make_sources(["some text"]),
        )
        assert result["source_grounded"] is False


class TestNoHallucinatedCitations:
    def test_valid_citations(self):
        sources = _make_sources(["text1", "text2"])
        result = validate_response("See [1] and [2].", sources)
        assert result["no_hallucinated_citations"] is True

    def test_hallucinated_citation(self):
        sources = _make_sources(["text1", "text2"])
        result = validate_response("See [5] for details.", sources)
        assert result["no_hallucinated_citations"] is False

    def test_no_citations_is_valid(self):
        result = validate_response("No refs here.", _make_sources(["text"]))
        assert result["no_hallucinated_citations"] is True


class TestAllPass:
    def test_well_formed_response_passes(self):
        source_text = "Heart failure is a chronic condition causing edema and fatigue"
        response = "Heart failure is a chronic condition causing edema and fatigue [1]."
        result = validate_response(response, _make_sources([source_text]))
        assert result["passed"] is True
        assert all([
            result["not_empty"],
            result["has_citations"],
            result["within_scope"],
            result["source_grounded"],
            result["no_hallucinated_citations"],
        ])
