"""Tests for FastAPI endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.models import (
    QueryRequest,
    QueryResponse,
    RecommendRequest,
    RecommendResponse,
    Source,
    ValidationResult,
)


@pytest.fixture
def mock_pipeline():
    mock = MagicMock()
    mock.answer.return_value = {
        "answer": "Heart failure is a chronic condition [1].",
        "sources": [
            {"text": "Heart failure info", "source": "medquad", "qtype": "general"},
        ],
        "scores": [0.95],
        "validation": {
            "has_citations": True,
            "within_scope": True,
            "source_grounded": True,
            "not_empty": True,
            "no_hallucinated_citations": True,
            "passed": True,
        },
        "expanded_queries": ["heart failure"],
    }
    return mock


@pytest.fixture
def mock_recommender():
    mock = MagicMock()
    mock.get_similar.return_value = [
        {"text": "Related content", "score": 0.85},
    ]
    return mock


@pytest.fixture
def client(mock_pipeline, mock_recommender):
    """Create a test client with mocked globals, bypassing lifespan."""
    import src.api.main as api_module

    # Inject mocks directly, bypassing the lifespan that loads real models
    original_pipeline = api_module.pipeline
    original_recommender = api_module.recommender
    api_module.pipeline = mock_pipeline
    api_module.recommender = mock_recommender

    # Create a plain app without lifespan to avoid model loading
    test_app = FastAPI()

    # Re-register the route handlers on the test app
    from src.api.main import ask, health, recommend

    test_app.post("/ask", response_model=QueryResponse)(ask)
    test_app.post("/recommend", response_model=RecommendResponse)(recommend)
    test_app.get("/health")(health)

    with TestClient(test_app) as c:
        yield c

    # Restore originals
    api_module.pipeline = original_pipeline
    api_module.recommender = original_recommender


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestAskEndpoint:
    def test_ask_returns_200(self, client):
        response = client.post("/ask", json={"question": "What is heart failure?"})
        assert response.status_code == 200

    def test_ask_includes_answer(self, client):
        response = client.post("/ask", json={"question": "What is heart failure?"})
        data = response.json()
        assert "answer" in data
        assert "Heart failure" in data["answer"]

    def test_ask_includes_latency(self, client):
        response = client.post("/ask", json={"question": "What is heart failure?"})
        data = response.json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_ask_includes_validation(self, client):
        response = client.post("/ask", json={"question": "What is heart failure?"})
        data = response.json()
        assert "validation" in data
        assert data["validation"]["passed"] is True

    def test_ask_rejects_short_question(self, client):
        response = client.post("/ask", json={"question": "Hi"})
        assert response.status_code == 422


class TestRecommendEndpoint:
    def test_recommend_returns_200(self, client):
        response = client.post(
            "/recommend", json={"document_text": "Heart failure overview"},
        )
        assert response.status_code == 200

    def test_recommend_returns_list(self, client):
        response = client.post(
            "/recommend", json={"document_text": "Heart failure overview"},
        )
        data = response.json()
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)
