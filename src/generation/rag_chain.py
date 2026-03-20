"""End-to-end RAG pipeline for medical education Q&A."""

import logging

from src.generation.guardrails import validate_response
from src.generation.llm_client import LLMClient
from src.generation.prompts import EDUCATION_QA_PROMPT
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.query_expander import expand_query
from src.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-augmented generation pipeline for medical education."""

    def __init__(
        self,
        searcher: HybridSearcher,
        reranker: Reranker,
        llm_client: LLMClient,
    ):
        self.searcher = searcher
        self.reranker = reranker
        self.llm = llm_client

    def answer(
        self,
        query: str,
        top_k: int = 20,
        rerank_n: int = 5,
    ) -> dict:
        """Run the full RAG pipeline.

        Flow: expand -> retrieve -> dedup -> rerank -> generate -> return.

        Returns:
            Dict with keys: answer, sources, scores, validation,
            expanded_queries.
        """
        # 1. Expand query
        queries = expand_query(query, client=self.llm)

        # 2. Retrieve across expanded queries
        all_candidates: list[dict] = []
        for q in queries:
            results = self.searcher.search(q, top_k=top_k)
            all_candidates.extend(results)

        # 3. Deduplicate by chunk_id (keep first occurrence)
        seen: set[str] = set()
        unique: list[dict] = []
        for c in all_candidates:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                unique.append(c)

        # 4. Rerank
        reranked = self.reranker.rerank(query, unique, top_n=rerank_n)

        # 5. Build context with numbered citations
        context = "\n---\n".join(
            f'[{i + 1}] {r["doc"]["text"]}'
            for i, r in enumerate(reranked)
        )

        # 6. Generate
        prompt = EDUCATION_QA_PROMPT.format(
            context=context, question=query,
        )
        response = self.llm.complete(prompt)

        # 7. Validate response with guardrails
        validation = validate_response(response, reranked)

        # 8. Build result
        sources = [r["doc"] for r in reranked]
        scores = [r["rerank_score"] for r in reranked]

        return {
            "answer": response,
            "sources": sources,
            "scores": scores,
            "validation": validation,
            "expanded_queries": queries,
        }
