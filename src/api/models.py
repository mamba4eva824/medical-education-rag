"""Pydantic v2 schemas for the medical education RAG API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for querying the RAG system."""

    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    specialty: str | None = None


class Source(BaseModel):
    """A single retrieved source document."""

    text: str
    source: str
    specialty: str
    relevance_score: float


class ValidationResult(BaseModel):
    """Validation outcome for a generated answer."""

    has_citations: bool
    within_scope: bool
    not_empty: bool
    source_grounded: bool
    no_hallucinated_citations: bool
    passed: bool


class QueryResponse(BaseModel):
    """Response schema for a RAG query."""

    answer: str
    sources: list[Source]
    validation: ValidationResult
    latency_ms: float


class RecommendRequest(BaseModel):
    """Request schema for document recommendations."""

    document_text: str
    n: int = Field(default=5)
    specialty: str | None = None


class RecommendResponse(BaseModel):
    """Response schema for document recommendations."""

    recommendations: list[dict]
