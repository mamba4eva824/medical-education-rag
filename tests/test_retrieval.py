"""Tests for the reranker module."""

from unittest.mock import MagicMock, patch


@patch("src.retrieval.reranker.CrossEncoder")
class TestReranker:
    def _make_reranker(self, mock_ce_cls):
        from src.retrieval.reranker import Reranker

        mock_model = MagicMock()
        mock_ce_cls.return_value = mock_model
        reranker = Reranker()
        return reranker, mock_model

    def test_rerank_returns_top_n(self, mock_ce_cls):
        reranker, mock_model = self._make_reranker(mock_ce_cls)
        mock_model.predict.return_value = [0.9, 0.1, 0.5]

        candidates = [
            {"text": "doc A", "chunk_id": "a"},
            {"text": "doc B", "chunk_id": "b"},
            {"text": "doc C", "chunk_id": "c"},
        ]
        results = reranker.rerank("query", candidates, top_n=2)
        assert len(results) == 2

    def test_rerank_sorted_descending(self, mock_ce_cls):
        reranker, mock_model = self._make_reranker(mock_ce_cls)
        mock_model.predict.return_value = [0.1, 0.9, 0.5]

        candidates = [
            {"text": "doc A", "chunk_id": "a"},
            {"text": "doc B", "chunk_id": "b"},
            {"text": "doc C", "chunk_id": "c"},
        ]
        results = reranker.rerank("query", candidates, top_n=3)
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_empty_candidates(self, mock_ce_cls):
        reranker, _ = self._make_reranker(mock_ce_cls)
        results = reranker.rerank("query", [], top_n=5)
        assert results == []

    def test_rerank_preserves_doc(self, mock_ce_cls):
        reranker, mock_model = self._make_reranker(mock_ce_cls)
        mock_model.predict.return_value = [0.8]

        candidates = [{"text": "doc A", "chunk_id": "a", "qtype": "general"}]
        results = reranker.rerank("query", candidates, top_n=1)
        assert results[0]["doc"]["text"] == "doc A"
        assert results[0]["doc"]["qtype"] == "general"
