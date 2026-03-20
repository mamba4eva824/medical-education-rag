"""LLM-powered query expansion for medical education search."""

import logging

from src.generation.llm_client import LLMClient

logger = logging.getLogger(__name__)


def expand_query(query: str, client: LLMClient | None = None) -> list[str]:
    """Generate alternative search queries using Claude.

    Args:
        query: The original search query.
        client: LLMClient instance. Creates a default one if not provided.

    Returns:
        List of queries: [original] + [up to 3 alternatives].
    """
    if client is None:
        client = LLMClient()

    prompt = (
        f'Generate 3 alternative search queries for medical education: '
        f'"{query}". Return only the queries, one per line. '
        f'Do not number them or add any other text.'
    )

    try:
        response = client.complete(
            prompt=prompt,
            system="You are a medical search query expander.",
            temperature=0.7,
            max_tokens=200,
        )
        expanded = [
            q.strip() for q in response.strip().split("\n") if q.strip()
        ]
        return [query] + expanded[:3]
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}. Using original query.")
        return [query]
