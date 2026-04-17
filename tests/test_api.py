"""Tests for FastAPI endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(mock_pipeline_result):
    """Create a test client with mocked pipeline and recommender."""
    # Mock the pipeline (used for both "full" and "simple" modes)
    mock_pipe = MagicMock()
    mock_pipe.answer.return_value = mock_pipeline_result

    # Mock the recommender
    mock_rec = MagicMock()
    mock_rec.get_similar.return_value = [{"text": "rec1", "score": 0.9}]

    # Patch globals before importing app
    mock_pipelines = {"full": mock_pipe, "simple": mock_pipe}
    with patch("src.api.main.pipelines", mock_pipelines), \
         patch("src.api.main.recommender", mock_rec):
        from src.api.main import app
        # Override lifespan to avoid loading real models
        app.router.lifespan_context = _noop_lifespan
        yield TestClient(app)


from contextlib import asynccontextmanager

@asynccontextmanager
async def _noop_lifespan(app):
    yield


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        """Health endpoint returns 200 with status info."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestAskEndpoint:
    """Tests for POST /ask."""

    def test_ask_returns_valid_response(self, client):
        """Ask endpoint returns a valid QueryResponse."""
        response = client.post(
            "/ask",
            json={"question": "What are heart failure symptoms?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "validation" in data
        assert "latency_ms" in data

    def test_ask_validates_input(self, client):
        """Rejects questions that are too short."""
        response = client.post("/ask", json={"question": "ab"})
        assert response.status_code == 422

    def test_ask_validation_fields(self, client):
        """Response validation includes all 6 fields."""
        response = client.post(
            "/ask",
            json={"question": "What causes hypertension?"},
        )
        assert response.status_code == 200
        validation = response.json()["validation"]
        assert "has_citations" in validation
        assert "within_scope" in validation
        assert "not_empty" in validation
        assert "source_grounded" in validation
        assert "no_hallucinated_citations" in validation
        assert "passed" in validation
