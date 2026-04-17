"""Response validation guardrails for medical education RAG pipeline.

Checks: has_citations, within_scope, not_empty, source_grounded,
no_hallucinated_citations.
"""

import re
import logging

logger = logging.getLogger(__name__)

PROHIBITED_ADVICE = [
    "self-diagnose",
    "self-medicate",
    "stop taking your medication",
    "do not see a doctor",
    "skip your appointment",
    "ignore your symptoms",
    "replace your doctor",
    "no need for medical attention",
]


def _extract_source_text(source: dict) -> str:
    """Extract text from a source dict, handling both wrapped and flat formats."""
    if "doc" in source and isinstance(source["doc"], dict):
        return source["doc"].get("text", "")
    return source.get("text", "")


def _check_has_citations(response: str) -> bool:
    """Check if the response contains citation markers like [1], [2], etc."""
    return bool(re.search(r"\[\d+\]", response))


def _check_within_scope(response: str) -> bool:
    """Check that the response contains no prohibited medical advice phrases."""
    response_lower = response.lower()
    for phrase in PROHIBITED_ADVICE:
        if phrase in response_lower:
            logger.warning("Prohibited phrase detected: %s", phrase)
            return False
    return True


def _check_not_empty(response: str) -> bool:
    """Check that the response has substantive content."""
    stripped = response.strip()
    return len(stripped) > 10


def _verify_claims(response: str, sources: list[dict]) -> bool:
    """Check source grounding via token overlap (>30% threshold)."""
    if not sources:
        return False

    response_tokens = set(re.findall(r"\w+", response.lower()))
    if not response_tokens:
        return False

    all_source_tokens = set()
    for source in sources:
        text = _extract_source_text(source)
        all_source_tokens.update(re.findall(r"\w+", text.lower()))

    if not all_source_tokens:
        return False

    overlap = response_tokens & all_source_tokens
    overlap_ratio = len(overlap) / len(response_tokens)

    return overlap_ratio >= 0.3


def _check_citation_range(response: str, num_sources: int) -> bool:
    """Check that all [N] citation references are within the valid source range."""
    citations = re.findall(r"\[(\d+)\]", response)
    if not citations:
        return True

    for cite in citations:
        if int(cite) < 1 or int(cite) > num_sources:
            logger.warning("Hallucinated citation [%s] with %d sources", cite, num_sources)
            return False
    return True


def validate_response(response: str, sources: list[dict]) -> dict:
    """Validate a generated response against guardrail checks.

    Args:
        response: The generated answer text.
        sources: List of source chunk dicts (wrapped or flat format).

    Returns:
        Dict with bool values for each check and an overall 'passed' key.
    """
    has_citations = _check_has_citations(response)
    within_scope = _check_within_scope(response)
    not_empty = _check_not_empty(response)
    source_grounded = _verify_claims(response, sources)
    no_hallucinated_citations = _check_citation_range(response, len(sources))

    passed = all([has_citations, within_scope, not_empty, source_grounded, no_hallucinated_citations])

    result = {
        "has_citations": has_citations,
        "within_scope": within_scope,
        "not_empty": not_empty,
        "source_grounded": source_grounded,
        "no_hallucinated_citations": no_hallucinated_citations,
        "passed": passed,
    }

    if not passed:
        failed = [k for k, v in result.items() if k != "passed" and not v]
        logger.warning("Guardrail checks failed: %s", failed)

    return result
