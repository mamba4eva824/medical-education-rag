"""Response validation guardrails for the medical education RAG pipeline.

Implements five safety checks to ensure generated answers are grounded,
properly cited, and within the educational scope.
"""

import re

PROHIBITED_PHRASES = [
    "self-diagnose",
    "self-medicate",
    "stop taking",
    "start taking",
    "prescribe",
    "i recommend you take",
    "you should take",
    "discontinue your",
]

SOURCE_GROUNDING_THRESHOLD = 0.30


def _extract_citation_indices(text: str) -> list[int]:
    """Extract all [N] citation indices from text."""
    return [int(m) for m in re.findall(r"\[(\d+)\]", text)]


def _token_set(text: str) -> set[str]:
    """Lowercase whitespace-split token set."""
    return set(text.lower().split())


def _check_not_empty(response: str) -> bool:
    return bool(response and response.strip())


def _check_has_citations(response: str, n_sources: int) -> bool:
    indices = _extract_citation_indices(response)
    return any(1 <= idx <= n_sources for idx in indices)


def _check_within_scope(response: str) -> bool:
    lower = response.lower()
    return not any(phrase in lower for phrase in PROHIBITED_PHRASES)


def _extract_citing_sentence(response: str, citation: str) -> str:
    """Extract the sentence containing a citation marker."""
    # Split on sentence boundaries, find the one with the citation
    sentences = re.split(r"(?<=[.!?])\s+", response)
    for sentence in sentences:
        if citation in sentence:
            return sentence
    return response


def _check_source_grounded(response: str, sources: list[dict]) -> bool:
    """Verify each citation's surrounding sentence overlaps >= 30% with the source."""
    indices = _extract_citation_indices(response)
    if not indices:
        return False

    valid_checks = 0
    for idx in set(indices):
        if idx < 1 or idx > len(sources):
            continue
        source_text = sources[idx - 1].get("doc", sources[idx - 1]).get("text", "")
        source_tokens = _token_set(source_text)
        # Compare the citing sentence, not the whole response
        citing_sentence = _extract_citing_sentence(response, f"[{idx}]")
        sentence_tokens = _token_set(citing_sentence)
        if not sentence_tokens:
            return False
        overlap = len(sentence_tokens & source_tokens) / len(sentence_tokens)
        if overlap < SOURCE_GROUNDING_THRESHOLD:
            return False
        valid_checks += 1
    return valid_checks > 0


def _check_no_hallucinated_citations(response: str, n_sources: int) -> bool:
    indices = _extract_citation_indices(response)
    if not indices:
        return True
    return all(1 <= idx <= n_sources for idx in indices)


def validate_response(response: str, sources: list[dict]) -> dict:
    """Run all five guardrail checks on a generated response.

    Args:
        response: The LLM-generated answer text.
        sources: List of source dicts, each with shape {"doc": {"text": "..."}}.

    Returns:
        Dict with boolean results for each check and an overall ``passed`` flag.
    """
    n_sources = len(sources)

    not_empty = _check_not_empty(response)
    has_citations = _check_has_citations(response, n_sources)
    within_scope = _check_within_scope(response)
    source_grounded = _check_source_grounded(response, sources)
    no_hallucinated_citations = _check_no_hallucinated_citations(response, n_sources)

    passed = all([
        not_empty,
        has_citations,
        within_scope,
        source_grounded,
        no_hallucinated_citations,
    ])

    return {
        "not_empty": not_empty,
        "has_citations": has_citations,
        "within_scope": within_scope,
        "source_grounded": source_grounded,
        "no_hallucinated_citations": no_hallucinated_citations,
        "passed": passed,
    }
