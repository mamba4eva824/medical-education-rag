"""Tests for retrieval strategies and RAGPipeline mode switching."""

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.strategies import (
    DenseRetrievalStrategy,
    HybridRetrievalStrategy,
    RetrievalResult,
)


# --- Fixtures ---

@pytest.fixture
def chunk_lookup():
    """Shared chunk lookup simulating parquet data."""
    return {
        "chunk_a": {
            "chunk_id": "chunk_a",
            "text": "Full text of chunk A about heart failure symptoms including edema.",
            "question": "What are heart failure symptoms?",
            "qtype": "symptoms",
            "source": "medquad",
            "chunk_index": 0,
            "total_chunks": 1,
        },
        "chunk_b": {
            "chunk_id": "chunk_b",
            "text": "Full text of chunk B about hypertension treatment with ACE inhibitors.",
            "question": "How is hypertension treated?",
            "qtype": "treatment",
            "source": "medquad",
            "chunk_index": 0,
            "total_chunks": 1,
        },
    }


# --- DenseRetrievalStrategy Tests ---

class TestDenseRetrievalStrategy:
    """Tests for dense-only retrieval."""

    def test_returns_retrieval_result(self, chunk_lookup):
        """Strategy returns a RetrievalResult dataclass."""
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"chunk_id": "chunk_a", "score": 0.95, "text": "Truncated..."},
        ]
        strategy = DenseRetrievalStrategy(
            vector_store=mock_store, chunk_lookup=chunk_lookup,
        )
        result = strategy.retrieve("heart failure", top_k=5)
        assert isinstance(result, RetrievalResult)

    def test_resolves_full_text_from_lookup(self, chunk_lookup):
        """Truncated Pinecone text is replaced with full parquet text."""
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"chunk_id": "chunk_a", "score": 0.95, "text": "Truncated..."},
        ]
        strategy = DenseRetrievalStrategy(
            vector_store=mock_store, chunk_lookup=chunk_lookup,
        )
        result = strategy.retrieve("heart failure", top_k=5)
        assert "edema" in result.candidates[0]["text"]
        assert result.candidates[0]["text"] != "Truncated..."

    def test_preserves_cosine_score(self, chunk_lookup):
        """Dense score from Pinecone is preserved after text resolution."""
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"chunk_id": "chunk_a", "score": 0.92, "text": "Truncated..."},
        ]
        strategy = DenseRetrievalStrategy(
            vector_store=mock_store, chunk_lookup=chunk_lookup,
        )
        result = strategy.retrieve("heart failure", top_k=5)
        assert result.candidates[0]["score"] == 0.92

    def test_no_query_expansion(self, chunk_lookup):
        """Simple strategy returns only the original query."""
        mock_store = MagicMock()
        mock_store.search.return_value = []
        strategy = DenseRetrievalStrategy(
            vector_store=mock_store, chunk_lookup=chunk_lookup,
        )
        result = strategy.retrieve("heart failure", top_k=5)
        assert result.expanded_queries == ["heart failure"]
        assert result.api_calls["llm_calls"] == 0
        assert result.api_calls["pinecone_calls"] == 1

    def test_unknown_chunk_id_uses_pinecone_data(self, chunk_lookup):
        """Chunks not in lookup fall back to Pinecone metadata."""
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"chunk_id": "unknown_id", "score": 0.8, "text": "Pinecone text"},
        ]
        strategy = DenseRetrievalStrategy(
            vector_store=mock_store, chunk_lookup=chunk_lookup,
        )
        result = strategy.retrieve("query", top_k=5)
        assert result.candidates[0]["text"] == "Pinecone text"

    def test_strategy_name(self, chunk_lookup):
        """Strategy name is 'simple'."""
        mock_store = MagicMock()
        strategy = DenseRetrievalStrategy(
            vector_store=mock_store, chunk_lookup=chunk_lookup,
        )
        assert strategy.name == "simple"


# --- HybridRetrievalStrategy Tests ---

class TestHybridRetrievalStrategy:
    """Tests for hybrid retrieval with query expansion."""

    def test_calls_expand_and_search(self):
        """Hybrid strategy expands query and searches for each variant."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [
            {"chunk_id": "c1", "text": "result", "rrf_score": 0.5},
        ]
        mock_llm = MagicMock()

        strategy = HybridRetrievalStrategy(
            searcher=mock_searcher, llm_client=mock_llm,
        )

        with patch("src.retrieval.strategies.expand_query") as mock_expand:
            mock_expand.return_value = ["original", "alt1", "alt2"]
            result = strategy.retrieve("original", top_k=10)

        assert mock_searcher.search.call_count == 3
        assert result.expanded_queries == ["original", "alt1", "alt2"]
        assert result.api_calls["llm_calls"] == 1
        assert result.api_calls["pinecone_calls"] == 3

    def test_deduplicates_by_chunk_id(self):
        """Duplicate chunk_ids across expanded queries are removed."""
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [
            {"chunk_id": "same_id", "text": "duplicate", "rrf_score": 0.5},
        ]
        mock_llm = MagicMock()

        strategy = HybridRetrievalStrategy(
            searcher=mock_searcher, llm_client=mock_llm,
        )

        with patch("src.retrieval.strategies.expand_query") as mock_expand:
            mock_expand.return_value = ["q1", "q2", "q3"]
            result = strategy.retrieve("q1", top_k=10)

        # 3 searches returning the same chunk_id -> 1 unique candidate
        assert len(result.candidates) == 1

    def test_strategy_name(self):
        """Strategy name is 'full'."""
        mock_searcher = MagicMock()
        mock_llm = MagicMock()
        strategy = HybridRetrievalStrategy(
            searcher=mock_searcher, llm_client=mock_llm,
        )
        assert strategy.name == "full"


# --- RAGPipeline Integration Tests ---

class TestPipelineModes:
    """Tests for RAGPipeline using different strategies."""

    def test_pipeline_returns_enriched_result(self):
        """Pipeline result includes pipeline_mode, timing, and api_calls."""
        from src.generation.rag_chain import RAGPipeline

        mock_strategy = MagicMock()
        mock_strategy.name = "full"
        mock_strategy.retrieve.return_value = RetrievalResult(
            candidates=[
                {"chunk_id": "c1", "text": "Heart failure causes edema."},
            ],
            expanded_queries=["heart failure"],
            api_calls={"llm_calls": 1, "pinecone_calls": 4},
        )

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {
                "doc": {"chunk_id": "c1", "text": "Heart failure causes edema."},
                "rerank_score": 0.95,
            },
        ]

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Edema is a symptom [1]."

        with patch("src.generation.rag_chain.validate_response") as mock_val:
            mock_val.return_value = {
                "has_citations": True,
                "within_scope": True,
                "not_empty": True,
                "source_grounded": True,
                "no_hallucinated_citations": True,
                "passed": True,
            }

            pipeline = RAGPipeline(
                strategy=mock_strategy,
                reranker=mock_reranker,
                llm_client=mock_llm,
            )
            result = pipeline.answer("heart failure symptoms")

        assert result["pipeline_mode"] == "full"
        assert "retrieval_s" in result["timing"]
        assert "rerank_s" in result["timing"]
        assert "generation_s" in result["timing"]
        assert result["api_calls"]["llm_calls"] == 1
        assert result["api_calls"]["reranker_calls"] == 1

    def test_pipeline_result_schema_compatible(self):
        """Both strategy modes produce results with the same keys."""
        from src.generation.rag_chain import RAGPipeline

        expected_keys = {
            "answer", "sources", "scores", "validation",
            "expanded_queries", "pipeline_mode", "timing", "api_calls",
        }

        for mode_name in ("full", "simple"):
            mock_strategy = MagicMock()
            mock_strategy.name = mode_name
            mock_strategy.retrieve.return_value = RetrievalResult(
                candidates=[{"chunk_id": "c1", "text": "Test text."}],
                expanded_queries=["query"],
                api_calls={"llm_calls": 0, "pinecone_calls": 1},
            )

            mock_reranker = MagicMock()
            mock_reranker.rerank.return_value = [
                {"doc": {"chunk_id": "c1", "text": "Test text."}, "rerank_score": 0.9},
            ]

            mock_llm = MagicMock()
            mock_llm.complete.return_value = "Answer [1]."

            with patch("src.generation.rag_chain.validate_response") as mock_val:
                mock_val.return_value = {
                    "has_citations": True, "within_scope": True,
                    "not_empty": True, "source_grounded": True,
                    "no_hallucinated_citations": True, "passed": True,
                }
                pipeline = RAGPipeline(
                    strategy=mock_strategy,
                    reranker=mock_reranker,
                    llm_client=mock_llm,
                )
                result = pipeline.answer("test query")

            assert set(result.keys()) == expected_keys, (
                f"Mode '{mode_name}' missing keys: {expected_keys - set(result.keys())}"
            )
