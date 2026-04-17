"""End-to-end RAG pipeline for medical education Q&A."""

import logging
import time

from src.generation.guardrails import validate_response
from src.generation.llm_client import LLMClient
from src.generation.prompts import EDUCATION_QA_PROMPT
from src.retrieval.reranker import Reranker
from src.retrieval.strategies import RetrievalStrategy

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-augmented generation pipeline for medical education.

    Uses a pluggable RetrievalStrategy to decouple the pipeline from
    any specific retrieval approach (hybrid, dense-only, etc.).
    """

    def __init__(
        self,
        strategy: RetrievalStrategy,
        reranker: Reranker,
        llm_client: LLMClient,
    ):
        self.strategy = strategy
        self.reranker = reranker
        self.llm = llm_client

    def answer(
        self,
        query: str,
        top_k: int = 20,
        rerank_n: int = 5,
    ) -> dict:
        """Run the RAG pipeline using the configured retrieval strategy.

        Flow: retrieve (strategy-dependent) -> rerank -> generate -> validate.

        Returns:
            Dict with keys: answer, sources, scores, validation,
            expanded_queries, pipeline_mode, timing, api_calls.
        """
        timing: dict[str, float] = {}

        # 1. Retrieve via strategy
        t0 = time.perf_counter()
        retrieval_result = self.strategy.retrieve(query, top_k=top_k)
        timing["retrieval_s"] = time.perf_counter() - t0

        candidates = retrieval_result.candidates

        # 2. Rerank
        t0 = time.perf_counter()
        reranked = self.reranker.rerank(query, candidates, top_n=rerank_n)
        timing["rerank_s"] = time.perf_counter() - t0

        # 3. Build context with numbered citations
        context = "\n---\n".join(
            f'[{i + 1}] {r["doc"]["text"]}'
            for i, r in enumerate(reranked)
        )

        # 4. Generate
        t0 = time.perf_counter()
        prompt = EDUCATION_QA_PROMPT.format(
            context=context, question=query,
        )
        response = self.llm.complete(prompt)
        timing["generation_s"] = time.perf_counter() - t0

        # 5. Build result
        sources = [r["doc"] for r in reranked]
        scores = [r["rerank_score"] for r in reranked]

        validation = validate_response(response, reranked)

        # Add reranker call to api_calls
        api_calls = {**retrieval_result.api_calls, "reranker_calls": 1}

        return {
            "answer": response,
            "sources": sources,
            "scores": scores,
            "validation": validation,
            "expanded_queries": retrieval_result.expanded_queries,
            "pipeline_mode": self.strategy.name,
            "timing": timing,
            "api_calls": api_calls,
        }
