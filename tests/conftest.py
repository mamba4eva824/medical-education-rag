"""Shared test fixtures for the medical education RAG test suite."""

import pytest


@pytest.fixture
def sample_sources():
    """Reranker-output-format sources for guardrail and pipeline tests."""
    return [
        {"doc": {"text": "Heart failure symptoms include edema and fatigue."}},
        {"doc": {"text": "Hypertension management includes ACE inhibitors."}},
        {"doc": {"text": "Diabetes mellitus requires blood glucose monitoring."}},
    ]


@pytest.fixture
def mock_pipeline_result():
    """A mock result dict matching RAGPipeline.answer() output."""
    return {
        "answer": "Heart failure symptoms include edema [1] and fatigue [2].",
        "sources": [
            {"text": "Heart failure symptoms include edema and fatigue.", "source": "medquad", "qtype": "cardiology", "chunk_id": "abc123"},
            {"text": "Hypertension management includes ACE inhibitors.", "source": "medquad", "qtype": "cardiology", "chunk_id": "def456"},
        ],
        "scores": [0.95, 0.87],
        "validation": {
            "has_citations": True,
            "within_scope": True,
            "not_empty": True,
            "source_grounded": True,
            "no_hallucinated_citations": True,
            "passed": True,
        },
        "expanded_queries": ["heart failure symptoms"],
    }
