"""Pluggable retrieval strategies for the RAG pipeline.

Defines a RetrievalStrategy Protocol and two concrete implementations:
- HybridRetrievalStrategy: query expansion + BM25/dense RRF fusion
- DenseRetrievalStrategy: Pinecone dense-only with full-text resolution
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol

from src.embeddings.vector_store import VectorStore
from src.generation.llm_client import LLMClient
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.query_expander import expand_query

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Standardized output from any retrieval strategy."""

    candidates: list[dict]
    expanded_queries: list[str]
    api_calls: dict = field(default_factory=dict)


class RetrievalStrategy(Protocol):
    """Protocol for pluggable retrieval strategies."""

    name: str

    def retrieve(self, query: str, top_k: int = 20) -> RetrievalResult: ...


class HybridRetrievalStrategy:
    """Query expansion + BM25/dense hybrid search with RRF fusion.

    Wraps the existing HybridSearcher and query expander. Each query
    triggers 1 LLM call (expansion) + N Pinecone calls (one per
    expanded query) + N local BM25 searches.
    """

    name = "full"

    def __init__(self, searcher: HybridSearcher, llm_client: LLMClient):
        self.searcher = searcher
        self.llm = llm_client

    def retrieve(self, query: str, top_k: int = 20) -> RetrievalResult:
        queries = expand_query(query, client=self.llm)

        all_candidates: list[dict] = []
        for q in queries:
            results = self.searcher.search(q, top_k=top_k)
            all_candidates.extend(results)

        # Deduplicate by chunk_id, keep first occurrence
        seen: set[str] = set()
        unique: list[dict] = []
        for c in all_candidates:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                unique.append(c)

        return RetrievalResult(
            candidates=unique,
            expanded_queries=queries,
            api_calls={
                "llm_calls": 1,
                "pinecone_calls": len(queries),
            },
        )


class DenseRetrievalStrategy:
    """Dense-only retrieval via Pinecone with full-text resolution.

    Bypasses query expansion and BM25. Uses a shared chunk lookup
    to resolve full text (Pinecone metadata truncates to 500 chars).
    """

    name = "simple"

    def __init__(
        self,
        vector_store: VectorStore,
        chunk_lookup: dict[str, dict],
    ):
        self.vector_store = vector_store
        self.chunk_lookup = chunk_lookup

    def retrieve(self, query: str, top_k: int = 20) -> RetrievalResult:
        results = self.vector_store.search(query, top_k=top_k)

        # Resolve full text from shared parquet data
        resolved: list[dict] = []
        for r in results:
            chunk_id = r["chunk_id"]
            if chunk_id in self.chunk_lookup:
                full = self.chunk_lookup[chunk_id].copy()
                full["score"] = r["score"]
                resolved.append(full)
            else:
                resolved.append(r)

        return RetrievalResult(
            candidates=resolved,
            expanded_queries=[query],
            api_calls={
                "llm_calls": 0,
                "pinecone_calls": 1,
            },
        )
