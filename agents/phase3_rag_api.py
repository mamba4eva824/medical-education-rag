#!/usr/bin/env python3
"""
Phase 3 Agent: RAG Architecture, Retrieval & API
==================================================
Job Responsibility: Apply RAG architectures, agentic workflows, prompt engineering
strategies, and LLM orchestration patterns + Develop backend services and APIs

Validates:
- Reranker (cross-encoder)
- Retrieval quality predictor (regression model)
- Hybrid search (BM25 + dense)
- Query expander (LLM-powered)
- RAG chain (end-to-end pipeline)
- Prompt templates
- LLM client abstraction
- FastAPI application with endpoints
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import PhaseAgent


class Phase3Agent(PhaseAgent):
    phase_name = "Phase 3: RAG Architecture, Retrieval & API"

    def run(self):
        # --- File existence ---
        files = {
            "reranker.py": "src/retrieval/reranker.py",
            "quality_predictor.py": "src/retrieval/quality_predictor.py",
            "hybrid_search.py": "src/retrieval/hybrid_search.py",
            "query_expander.py": "src/retrieval/query_expander.py",
            "rag_chain.py": "src/generation/rag_chain.py",
            "prompts.py": "src/generation/prompts.py",
            "llm_client.py": "src/generation/llm_client.py",
            "FastAPI main.py": "src/api/main.py",
            "API models.py": "src/api/models.py",
        }
        for name, path in files.items():
            self.check(f"{name} exists", self.file_has_content, path)

        # --- Import checks ---
        modules = [
            "src.retrieval.reranker",
            "src.retrieval.quality_predictor",
            "src.retrieval.hybrid_search",
            "src.retrieval.query_expander",
            "src.generation.rag_chain",
            "src.generation.prompts",
            "src.generation.llm_client",
            "src.api.main",
            "src.api.models",
        ]
        for mod in modules:
            self.check(f"{mod.split('.')[-1]} imports cleanly", self.module_imports, mod)

        # --- Interface checks ---
        self.check("Reranker has rerank() method", self._test_reranker_interface)
        self.check("RetrievalQualityPredictor has required methods", self._test_quality_predictor_interface)
        self.check("HybridSearcher has search() method", self._test_hybrid_interface)
        self.check("RAGPipeline has answer() method", self._test_rag_interface)
        self.check("Prompt templates are defined", self._test_prompts)
        self.check("FastAPI has required endpoints", self._test_api_endpoints)
        self.check("Pydantic schemas are complete", self._test_pydantic_schemas)

    def _test_reranker_interface(self) -> tuple[bool, str]:
        try:
            from src.retrieval.reranker import Reranker
            if not hasattr(Reranker, 'rerank'):
                return False, "Missing rerank() method"
            return True, "Reranker has rerank() method"
        except Exception as e:
            return False, str(e)

    def _test_quality_predictor_interface(self) -> tuple[bool, str]:
        try:
            from src.retrieval.quality_predictor import RetrievalQualityPredictor
            required = ['extract_features', 'predict', 'train_and_log']
            missing = [m for m in required if not hasattr(RetrievalQualityPredictor, m)]
            if missing:
                return False, f"Missing methods: {missing}"
            return True, "RetrievalQualityPredictor has extract_features(), predict(), train_and_log()"
        except Exception as e:
            return False, str(e)

    def _test_hybrid_interface(self) -> tuple[bool, str]:
        try:
            from src.retrieval.hybrid_search import HybridSearcher
            required = ['search', '_rrf_combine']
            missing = [m for m in required if not hasattr(HybridSearcher, m)]
            if missing:
                return False, f"Missing methods: {missing}"
            return True, "HybridSearcher has search() and _rrf_combine()"
        except Exception as e:
            return False, str(e)

    def _test_rag_interface(self) -> tuple[bool, str]:
        try:
            from src.generation.rag_chain import RAGPipeline
            if not hasattr(RAGPipeline, 'answer'):
                return False, "Missing answer() method"
            return True, "RAGPipeline has answer() method"
        except Exception as e:
            return False, str(e)

    def _test_prompts(self) -> tuple[bool, str]:
        try:
            from src.generation.prompts import (
                EDUCATION_QA_PROMPT,
                STUDY_GUIDE_PROMPT,
                SUMMARIZATION_PROMPT,
            )
            prompts = {
                'EDUCATION_QA_PROMPT': EDUCATION_QA_PROMPT,
                'STUDY_GUIDE_PROMPT': STUDY_GUIDE_PROMPT,
                'SUMMARIZATION_PROMPT': SUMMARIZATION_PROMPT,
            }
            for name, p in prompts.items():
                if not p or len(p) < 50:
                    return False, f"{name} is too short or empty"
                if '{context}' not in p:
                    return False, f"{name} missing {{context}} placeholder"
            return True, "All 3 prompt templates defined with {context} placeholder"
        except ImportError as e:
            return False, str(e)

    def _test_api_endpoints(self) -> tuple[bool, str]:
        try:
            from src.api.main import app
            routes = [r.path for r in app.routes]
            required = ['/ask', '/recommend', '/health']
            missing = [r for r in required if r not in routes]
            if missing:
                return False, f"Missing endpoints: {missing}"
            return True, f"Found all endpoints: {required}"
        except Exception as e:
            return False, str(e)

    def _test_pydantic_schemas(self) -> tuple[bool, str]:
        try:
            from src.api.models import (
                QueryRequest, QueryResponse,
                RecommendRequest, RecommendResponse,
                Source, ValidationResult,
            )
            schemas = ['QueryRequest', 'QueryResponse', 'RecommendRequest',
                       'RecommendResponse', 'Source', 'ValidationResult']
            return True, f"All {len(schemas)} Pydantic schemas defined"
        except ImportError as e:
            return False, str(e)


if __name__ == "__main__":
    agent = Phase3Agent()
    report = agent.execute()
    sys.exit(0 if report.all_passed else 1)
