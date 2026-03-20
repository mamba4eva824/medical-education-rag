"""FastAPI application for the Medical Education RAG API."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.models import (
    QueryRequest,
    QueryResponse,
    RecommendRequest,
    RecommendResponse,
    Source,
    ValidationResult,
)

logger = logging.getLogger(__name__)

# Global references set during startup
pipeline = None
recommender = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and initialize pipeline on startup."""
    global pipeline, recommender
    logger.info("Starting up — loading models and data...")

    from src.embeddings.recommender import ContentRecommender
    from src.embeddings.vector_store import VectorStore
    from src.generation.llm_client import LLMClient
    from src.generation.rag_chain import RAGPipeline
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.reranker import Reranker

    store = VectorStore()
    searcher = HybridSearcher(vector_store=store)
    reranker = Reranker()
    llm = LLMClient()

    pipeline = RAGPipeline(
        searcher=searcher, reranker=reranker, llm_client=llm,
    )
    recommender = ContentRecommender(vector_store=store)

    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Medical Education RAG API",
    description="Semantic search, Q&A, and content recommendations",
    lifespan=lifespan,
)


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """Answer a medical education question using RAG."""
    start = time.time()
    try:
        result = pipeline.answer(
            query=request.question, top_k=request.top_k,
        )
        latency = time.time() - start

        sources = [
            Source(
                text=s.get("text", "")[:500],
                source=s.get("source", "medquad"),
                specialty=s.get("qtype", "general"),
                relevance_score=score,
            )
            for s, score in zip(result["sources"], result["scores"])
        ]

        validation = ValidationResult(**result["validation"])

        logger.info(
            f"query={request.question[:50]} latency={latency:.3f}s "
            f"passed={validation.passed}"
        )

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            validation=validation,
            latency_ms=latency * 1000,
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """Get content recommendations based on similarity."""
    results = recommender.get_similar(
        request.document_text,
        n=request.n,
        specialty_filter=request.specialty,
    )
    return RecommendResponse(recommendations=results)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
    }
