"""Cross-encoder reranker for two-stage retrieval."""

from sentence_transformers import CrossEncoder


class Reranker:
    """Reranks candidate chunks using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = 5,
    ) -> list[dict]:
        """Score each (query, passage) pair and return top_n sorted by score.

        Args:
            query: The search query.
            candidates: List of chunk dicts, each must have a 'text' key.
            top_n: Number of top results to return.

        Returns:
            List of dicts with 'doc' and 'rerank_score' keys.
        """
        if not candidates:
            return []

        pairs = [[query, doc["text"]] for doc in candidates]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            {"doc": doc, "rerank_score": float(score)}
            for doc, score in ranked[:top_n]
        ]
