# Medical Education RAG Pipeline — Claude Code Instructions

## Project Context
This is an interview preparation project for the Applied AI Engineer role at Vanderbilt University School of Medicine. Every code decision must map to a specific job responsibility under **AI Application Development** or **ML Engineering & Production Systems**.

## Development Phases
This project follows a 5-week lifecycle. Each phase has an automated agent script in `agents/` and specific instructions below.

### Current Phase Checklist
Before starting work, check which phase we're in by looking at what exists:
- No `src/ingestion/*.py` files (only `__init__.py`) → **Phase 1: Data Ingestion**
- No `src/embeddings/vector_store.py` → **Phase 2: Embeddings & Search**
- No `src/retrieval/reranker.py` → **Phase 3: RAG & API**
- No `src/generation/guardrails.py` → **Phase 4: Guardrails, Prediction & Tests**
- No `databricks/db_01_delta_tables.py` content → **Phase 5: Databricks Porting**

---

## Phase 1: Data Ingestion & Chunking (Week 1)

**Job Responsibility:** Design and build AI-powered features including semantic search, content recommendations, and LLM-based tools

**Dataset:** `keivalya/MedQuad-MedicalQnADataset` (16,407 Q&A pairs, 16 qtypes, from NIH)

**Files to create:**
- `src/ingestion/medical_loader.py` — `MedQuADLoader` class, loads from HuggingFace, holds out 500 eval pairs
- `src/ingestion/chunker.py` — `MedicalChunker` class, Q&A-aware adaptive chunking
- `notebooks/01_data_ingestion.ipynb` — End-to-end pipeline notebook

**Conventions:**
- `MedQuADLoader.load()` returns `(documents, eval_pairs, test_pairs)` — 500 eval + 200 test held-out Q&A pairs
- Chunk metadata schema: `{chunk_id, text, question, qtype, source, chunk_index, total_chunks}`
- Chunking strategy: all chunks have format `"Q: {question}\n\nA: {answer_or_fragment}"`; answers ≤800 chars → single chunk; >800 chars → split answer at 800 chars then prepend question prefix to each
- Default max_chunk_size=800 (answer portion), leaving room for question prefix under model token limits
- Chunk IDs are MD5 hashes: `hashlib.md5((question + text).encode()).hexdigest()[:12]`
- Save chunks to `data/processed/medical_chunks.parquet`
- Save eval pairs to `data/processed/eval_queries.parquet`
- Save test pairs to `data/processed/test_queries.parquet` (held out for final unbiased metrics only)
- Use `python-dotenv` for all API keys — never hardcode

**Validation agent:** Run `python agents/phase1_ingestion.py` to verify all loaders work and chunks are saved.

---

## Phase 2: Embeddings, Vector Store & Recommendations (Week 2)

**Job Responsibility:** Evaluate vendor versus open-source AI products based on performance, cost, and reliability

**Files to create:**
- `src/embeddings/vector_store.py` — Pinecone serverless index wrapper
- `src/embeddings/recommender.py` — ContentRecommender using same embedding infra
- `notebooks/02_embedding_comparison.ipynb` — Compare 3 HuggingFace models with MLflow
- `notebooks/02b_build_vector_store.ipynb` — Build production index

**Models (all from HuggingFace, auto-download):**
- `sentence-transformers/all-MiniLM-L6-v2` (80MB, baseline)
- `sentence-transformers/all-mpnet-base-v2` (420MB, quality)
- `pritamdeka/S-PubMedBert-MS-MARCO` (420MB, domain-specific)

**Conventions:**
- MLflow tracking URI: `mlruns` (local directory)
- Experiment name: `embedding_comparison`
- Log params: model name, type, size, embedding_dim
- Log metrics: precision_at_5, mrr, encoding_time_sec
- Pinecone index name: `medical-education-chunks`
- Pinecone API key loaded from `PINECONE_API_KEY` env var

**Validation agent:** Run `python agents/phase2_embeddings.py` to verify vector store and MLflow runs.

---

## Phase 3: RAG Architecture, Retrieval & API (Week 3)

**Job Responsibility:** Apply RAG architectures, agentic workflows, prompt engineering strategies, and LLM orchestration patterns + Develop backend services and APIs

**Files to create:**
- `src/retrieval/reranker.py` — CrossEncoder(`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `src/retrieval/quality_predictor.py` — RetrievalQualityPredictor (StandardScaler → GBR), predicts relevance 0–1
- `src/retrieval/hybrid_search.py` — BM25 + dense with RRF fusion
- `src/retrieval/query_expander.py` — Groq Llama 3.1 query expansion
- `src/generation/rag_chain.py` — RAGPipeline: expand → retrieve → dedup → rerank → score quality → generate → validate
- `src/generation/prompts.py` — EDUCATION_QA_PROMPT, STUDY_GUIDE_PROMPT, SUMMARIZATION_PROMPT
- `src/generation/llm_client.py` — Groq client wrapper
- `src/api/main.py` — FastAPI with /ask, /recommend, /health
- `src/api/models.py` — Pydantic schemas
- `notebooks/03_retrieval_experiments.ipynb` — Compare retrieval strategies with MLflow

**Conventions:**
- Reranker default model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- RRF k parameter: 60
- Query expansion: 3 alternative queries via Groq
- Quality predictor: GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
- Quality predictor features: cosine_similarity, bm25_score, token_overlap, chunk_length, specialty_match
- RAG pipeline returns: `{answer, sources, scores, predicted_quality, validation, expanded_queries}`
- FastAPI runs on port 8000
- All API responses include latency_ms

**Validation agent:** Run `python agents/phase3_rag_api.py` to verify RAG chain and API endpoints.

---

## Phase 4: Guardrails, Prediction & Tests (Week 4)

**Job Responsibility:** Ensure responsible AI practices + Predictive modeling for learner outcomes + MLOps practices

**Files to create:**
- `src/generation/guardrails.py` — validate_response with 5 checks
- `src/prediction/at_risk_model.py` — AtRiskPipeline (StandardScaler → GBM classification)
- `src/api/monitoring.py` — QueryMetrics dataclass
- `notebooks/05_predictive_model.ipynb` — At-risk classifier training + SHAP explanations
- `notebooks/05b_quality_predictor.ipynb` — Retrieval quality regressor training + SHAP
- `tests/test_retrieval.py` — Reranker tests
- `tests/test_quality_predictor.py` — Quality predictor regression tests
- `tests/test_guardrails.py` — Guardrail validation tests
- `tests/test_api.py` — FastAPI endpoint tests

**Conventions:**
- Guardrail checks: has_citations, within_scope, not_empty, source_grounded, no_hallucinated_citations
- Source grounding threshold: 30% token overlap
- At-risk model: GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1), trained on synthetic data (500 students, np.random.seed(42))
- Quality predictor: trained via cross-encoder distillation — 500 queries × 20 retrieved chunks = ~10K training examples, cross-encoder scores as regression target
- Quality predictor features: cosine_similarity, bm25_score, token_overlap, chunk_length, qtype_match
- Tests use pytest, run with `pytest tests/ -v`

**Validation agent:** Run `python agents/phase4_guardrails_tests.py` to verify guardrails, model, and test suite.

---

## Phase 5: Databricks Porting (Week 5)

**Job Responsibility:** Build and maintain ML pipelines in Databricks + Full lifecycle management

**Files to create:**
- `notebooks/04_export_for_databricks.ipynb` — Export CSVs for upload
- `databricks/db_01_delta_tables.py` — Create Delta tables
- `databricks/db_02_mlflow_experiments.py` — Re-run embedding comparison
- `databricks/db_03_model_registry.py` — Register at-risk model

**Conventions:**
- Export to `data/exports/*.csv`
- Databricks notebooks use `%pip install` for dependencies
- MLflow experiment path: `/Users/you@email.com/embedding_comparison`
- Registered model name: `learner_at_risk_classifier`

**Validation agent:** Run `python agents/phase5_databricks.py` to verify exports and notebook structure.

---

## Bug Fixing Protocol

**IMPORTANT:** When a bug is reported, do NOT start by trying to fix it. Follow this sequence:

1. **Reproduce first:** Write a test that reproduces the bug and confirms it fails.
2. **Delegate the fix:** Use subagents to attempt the fix.
3. **Prove it:** The fix is only valid when the failing test now passes.

This ensures every bug fix is verified and prevents regressions.

---

## General Conventions

- **Imports:** Use absolute imports from project root (e.g., `from src.ingestion.chunker import DocumentChunker`)
- **Type hints:** Use Python 3.11+ syntax (`str | None` not `Optional[str]`)
- **Error handling:** Log errors with `logging` module, raise HTTPException in API layer
- **Environment variables:** Always load via `python-dotenv`, never hardcode keys
- **MLflow:** Every experiment gets tracked — params, metrics, and model artifacts
- **Git:** Commit after each phase completion with descriptive message

## Running

```bash
source .venv/bin/activate
pip install -e ".[dev]"              # Install deps
pytest tests/ -v                     # Run tests
mlflow ui --port 5000                # View experiments
uvicorn src.api.main:app --reload    # Start API
```
