"""Tests for the cross-encoder reranker."""

from unittest.mock import MagicMock, patch

import numpy as np


class TestReranker:
    """Tests for the Reranker class."""

    def _make_reranker(self):
        """Create a Reranker with a mocked CrossEncoder."""
        with patch("src.retrieval.reranker.CrossEncoder") as MockCE:
            mock_model = MagicMock()
            MockCE.return_value = mock_model
            from src.retrieval.reranker import Reranker
            reranker = Reranker()
            reranker.model = mock_model
            return reranker

    def test_rerank_improves_order(self):
        """Medical content ranks above irrelevant content."""
        reranker = self._make_reranker()

        candidates = [
            {"text": "The weather today is sunny and warm.", "chunk_id": "irr1"},
            {"text": "Heart failure symptoms include edema and fatigue.", "chunk_id": "med1"},
            {"text": "Python is a programming language.", "chunk_id": "irr2"},
        ]

        # Mock: medical content gets highest score
        reranker.model.predict.return_value = np.array([0.1, 0.9, 0.05])

        results = reranker.rerank("heart failure symptoms", candidates, top_n=2)

        assert len(results) == 2
        assert results[0]["doc"]["chunk_id"] == "med1"
        assert results[0]["rerank_score"] > results[1]["rerank_score"]

    def test_empty_candidates(self):
        """Handles empty input gracefully."""
        reranker = self._make_reranker()
        results = reranker.rerank("any query", [], top_n=5)
        assert results == []

    def test_rerank_returns_correct_structure(self):
        """Each result has 'doc' and 'rerank_score' keys."""
        reranker = self._make_reranker()

        candidates = [
            {"text": "Diabetes requires monitoring.", "chunk_id": "d1"},
        ]
        reranker.model.predict.return_value = np.array([0.8])

        results = reranker.rerank("diabetes", candidates, top_n=1)

        assert len(results) == 1
        assert "doc" in results[0]
        assert "rerank_score" in results[0]
        assert isinstance(results[0]["rerank_score"], float)
