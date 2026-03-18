"""Content recommendation engine built on top of a VectorStore."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.embeddings.vector_store import VectorStore


class ContentRecommender:
    """Wraps a VectorStore to provide content recommendations."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def get_similar(
        self,
        document_text: str,
        n: int = 5,
        specialty_filter: str | None = None,
    ) -> list[dict]:
        """Find content similar to the given text.

        Args:
            document_text: The source text to find similar content for.
            n: Number of similar items to return.
            specialty_filter: Optional qtype value to filter results.

        Returns:
            A list of result dicts from the vector store, excluding the
            source document if it appears in the results.
        """
        filter_dict = {"qtype": specialty_filter} if specialty_filter else None
        results = self.vector_store.search(
            query=document_text,
            top_k=n + 1,
            filter=filter_dict,
        )
        # Remove the source document if it appears in results
        return [
            r for r in results if r.get("text", "") != document_text[:500]
        ][:n]

    def recommend_study_path(
        self,
        weak_topics: list[str],
        n_per_topic: int = 3,
    ) -> dict[str, list[dict]]:
        """Given topics a learner struggles with, recommend content for each.

        Args:
            weak_topics: A list of topic strings the learner is weak in.
            n_per_topic: Number of recommendations per topic.

        Returns:
            A dict mapping each topic string to a list of recommended
            content dicts.
        """
        recommendations: dict[str, list[dict]] = {}
        for topic in weak_topics:
            recommendations[topic] = self.get_similar(topic, n=n_per_topic)
        return recommendations
