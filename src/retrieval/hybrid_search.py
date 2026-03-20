"""Hybrid search combining BM25 (sparse) and Pinecone (dense) with RRF fusion."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.embeddings.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Combines BM25 sparse retrieval with Pinecone dense retrieval."""

    def __init__(
        self,
        vector_store: VectorStore,
        chunks_path: str | Path = "data/processed/medical_chunks.parquet",
    ):
        self.vector_store = vector_store

        # Load full chunk texts for BM25 (Pinecone truncates to 500 chars)
        chunks_df = pd.read_parquet(chunks_path)
        self.chunk_ids = chunks_df["chunk_id"].tolist()
        self.chunk_texts = chunks_df["text"].tolist()
        self.chunk_metadata = chunks_df.to_dict("records")

        # Build BM25 index on tokenized texts
        tokenized = [text.lower().split() for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized)

        # Lookup: chunk_id -> index in corpus
        self.id_to_idx = {cid: i for i, cid in enumerate(self.chunk_ids)}

        logger.info(
            f"HybridSearcher initialized with {len(self.chunk_texts)} chunks"
        )

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Run hybrid search and return fused results.

        Args:
            query: The search query.
            top_k: Number of candidates to retrieve from each source.

        Returns:
            List of chunk dicts sorted by RRF score (descending).
        """
        # Sparse: BM25
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[::-1][:top_k]
        sparse_results = [
            (self.chunk_ids[idx], bm25_scores[idx]) for idx in bm25_top
        ]

        # Dense: Pinecone
        dense_results = self.vector_store.search(query, top_k=top_k)

        return self._rrf_combine(sparse_results, dense_results, k=60)

    def _rrf_combine(
        self,
        sparse_results: list[tuple[str, float]],
        dense_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """Combine sparse and dense results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) across both result lists.
        """
        scores: dict[str, float] = {}

        # Sparse contributions
        for rank, (chunk_id, _) in enumerate(sparse_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

        # Dense contributions
        for rank, result in enumerate(dense_results):
            chunk_id = result["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)

        # Sort by RRF score, attach full metadata
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for chunk_id, rrf_score in ranked:
            idx = self.id_to_idx.get(chunk_id)
            if idx is not None:
                meta = self.chunk_metadata[idx].copy()
                meta["rrf_score"] = rrf_score
                results.append(meta)

        return results
