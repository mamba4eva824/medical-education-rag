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
pipelines: dict = {}
recommender = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and initialize pipelines on startup."""
    global pipelines, recommender
    logger.info("Starting up — loading models and data...")

    from src.embeddings.recommender import ContentRecommender
    from src.embeddings.vector_store import VectorStore
    from src.generation.llm_client import LLMClient
    from src.generation.rag_chain import RAGPipeline
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.reranker import Reranker
    from src.retrieval.strategies import (
        DenseRetrievalStrategy,
        HybridRetrievalStrategy,
    )

    store = VectorStore()
    searcher = HybridSearcher(vector_store=store)
    reranker = Reranker()
    llm = LLMClient()

    # Build strategies — DenseRetrievalStrategy shares parquet data
    # already loaded by HybridSearcher (no duplicate load)
    chunk_lookup = {
        m["chunk_id"]: m for m in searcher.chunk_metadata
    }
    hybrid_strategy = HybridRetrievalStrategy(searcher=searcher, llm_client=llm)
    dense_strategy = DenseRetrievalStrategy(
        vector_store=store, chunk_lookup=chunk_lookup,
    )

    pipelines["full"] = RAGPipeline(
        strategy=hybrid_strategy, reranker=reranker, llm_client=llm,
    )
    pipelines["simple"] = RAGPipeline(
        strategy=dense_strategy, reranker=reranker, llm_client=llm,
    )

    recommender = ContentRecommender(vector_store=store)

    logger.info("Startup complete — full and simple pipelines ready.")
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
        pipe = pipelines.get(request.mode)
        if pipe is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown mode '{request.mode}'. Use 'full' or 'simple'.",
            )

        result = pipe.answer(
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
            f"mode={request.mode} query={request.question[:50]} "
            f"latency={latency:.3f}s passed={validation.passed}"
        )

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            validation=validation,
            latency_ms=latency * 1000,
            pipeline_mode=result["pipeline_mode"],
            timing=result["timing"],
            api_calls=result["api_calls"],
        )
    except HTTPException:
        raise
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
        "pipeline_loaded": len(pipelines) > 0,
    }
