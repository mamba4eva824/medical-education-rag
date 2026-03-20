# Medical Education RAG Pipeline

A retrieval-augmented generation (RAG) pipeline for medical education, built as a portfolio project for the Applied AI Engineer role at Vanderbilt University School of Medicine.

The system ingests authoritative medical Q&A content, chunks it for semantic retrieval, and will power an intelligent tutoring chatbot with citation-backed answers, content recommendations, and guardrails appropriate for medical education.

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Ingestion and Chunking | Complete |
| 2 | Embeddings, Vector Store, and Recommendations | Complete |
| 3 | RAG Architecture, Retrieval ML, and API | Complete |
| 4 | Guardrails, Predictive Models, and Tests | Complete |
| 5 | Databricks Porting | Planned |

---

## Architecture

```
MedQuAD (NIH)                 Embedding Models              LLM (Groq/Claude)
     |                              |                              |
     v                              v                              v
 MedQuADLoader ──> MedicalChunker ──> Pinecone ──> HybridSearch ──> RAGPipeline
     |                                    |              |              |
     v                                    v              v              v
 eval/test split              VectorStore + BM25    Reranker      Guardrails
                                                                      |
                                                                      v
                                                              FastAPI (/ask, /recommend)
```

---

## What Has Been Built

### Phase 1: Data Ingestion and Chunking

**Dataset:** MedQuAD -- 16,407 medical Q&A pairs from 12 NIH institutes, loaded via HuggingFace.

**Data pipeline:**
- Loads and deduplicates the dataset
- Produces a three-way stratified split: 15,117 documents for indexing, 500 eval pairs for development, 200 test pairs for final metrics
- Applies Q&A-aware adaptive chunking: short answers (under 1,000 characters) stay intact as single chunks with the question embedded in the text; long answers split at paragraph boundaries with the original question preserved in metadata
- Outputs 30,198 retrieval-ready chunks with structured metadata (question, topic category, source, position)

**Key design decisions:**
- Three-way split prevents data leakage: the quality predictor (Phase 4) trains on eval queries, so test queries remain sealed for unbiased final metrics
- Chunk IDs are deterministic MD5 hashes of question + text, ensuring reproducibility across runs
- Every chunk carries its original question as metadata, enabling hybrid retrieval against both question and answer text

### Phase 2: Embeddings, Vector Store, and Recommendations

**Embedding models evaluated:** Three HuggingFace models compared using local cosine similarity on a 2,000-chunk sample, with metrics logged to MLflow.

| Model | P@5 | MRR | Encoding Time | Dimensions |
|-------|-----|-----|---------------|------------|
| all-MiniLM-L6-v2 | 0.070 | 0.117 | 6.4s | 384 |
| pritamdeka/S-PubMedBert-MS-MARCO | 0.062 | 0.126 | 21.3s | 768 |
| all-mpnet-base-v2 | 0.090 | 0.169 | 183.7s | 768 |

**Vector store:** PubMedBert selected for domain specialization in medical content. 10,763 vectors indexed in Pinecone serverless (30% sample, cosine metric).

**Components built:**
- `VectorStore` -- Pinecone wrapper with batch upsert, retry logic, and post-upsert count verification
- `ContentRecommender` -- similarity search and personalized study path recommendations
- MLflow experiment tracking with 3 model comparison runs

### Phase 3: RAG Architecture, Retrieval, and API

**Pipeline:** End-to-end retrieval-augmented generation with query expansion, hybrid search, cross-encoder reranking, and citation-backed answer generation.

**Retrieval strategy:**
- Query expansion via Claude Haiku generates 3 alternative phrasings per question
- Hybrid search combines BM25 (keyword) + Pinecone (semantic) with Reciprocal Rank Fusion (k=60)
- Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) narrows ~40 candidates to top 5
- Quality predictor estimates retrieval relevance from 5 lightweight features

**Generation:**
- Claude Haiku 4.5 generates answers with numbered citations referencing retrieved sources
- Three prompt templates: education Q&A, study guide, summarization
- Validation layer checks citations, scope, and source grounding

**API:**
- FastAPI with `/ask`, `/recommend`, and `/health` endpoints
- Pydantic request/response schemas with auto-generated OpenAPI docs
- All responses include `latency_ms` for monitoring

### Phase 4: Guardrails, Predictive Models, and Tests

**Guardrails:** Five independent safety checks on every generated response:

| Check | Purpose |
|-------|---------|
| `not_empty` | Response has substance |
| `has_citations` | References numbered sources |
| `within_scope` | No prohibited medical advice (8 blocked phrases) |
| `source_grounded` | Per-citation sentence has 30% token overlap with source |
| `no_hallucinated_citations` | No references beyond actual source count |

**Predictive models:**
- **At-risk learner classifier** — GradientBoostingClassifier (StandardScaler pipeline) trained on synthetic engagement data (500 students, 6 features). SHAP explanations show per-feature contributions for stakeholder trust.
- **Retrieval quality predictor** — GradientBoostingRegressor trained via cross-encoder distillation (~10K examples). Predicts relevance in <1ms vs ~200ms for the full cross-encoder.

**Monitoring:** `QueryMetrics` dataclass tracks latency, guardrail pass rates, and empty response rates for production alerting.

**Test suite:** 32 pytest tests across 4 files — guardrails (15), API endpoints (8), quality predictor (5), reranker (4). All tests run offline with mocked external services.

**Evaluation:** `scripts/run_eval_queries.py` runs end-to-end through BM25 + Anthropic on eval and test sets, reporting per-qtype guardrail pass rates and answer overlap.

---

## Project Structure

```
medical-education-rag/
|-- src/
|   |-- ingestion/
|   |   |-- medical_loader.py       # MedQuAD loader with 3-way stratified split
|   |   |-- chunker.py              # Q&A-aware adaptive chunking
|   |-- embeddings/
|   |   |-- vector_store.py         # Pinecone serverless wrapper
|   |   |-- recommender.py          # Content recommendation engine
|   |-- retrieval/
|   |   |-- reranker.py             # Cross-encoder reranker (ms-marco-MiniLM)
|   |   |-- hybrid_search.py        # BM25 + Pinecone with RRF fusion
|   |   |-- query_expander.py       # LLM-powered query diversification
|   |   |-- quality_predictor.py    # Lightweight relevance scorer (distilled)
|   |-- generation/
|   |   |-- rag_chain.py            # End-to-end RAG pipeline orchestration
|   |   |-- llm_client.py           # Claude Haiku 4.5 wrapper
|   |   |-- prompts.py              # Education-specific prompt templates
|   |   |-- guardrails.py           # 5-check response validation
|   |-- prediction/
|   |   |-- at_risk_model.py        # GBM at-risk learner classifier
|   |-- api/
|   |   |-- main.py                 # FastAPI with /ask, /recommend, /health
|   |   |-- models.py               # Pydantic request/response schemas
|   |   |-- monitoring.py           # QueryMetrics latency and quality tracking
|
|-- notebooks/
|   |-- 01_data_ingestion.ipynb      # Data loading and chunking pipeline
|   |-- 02_embedding_comparison.ipynb  # Model comparison with MLflow
|   |-- 02b_build_vector_store.ipynb   # Build Pinecone index
|   |-- 03_retrieval_experiments.ipynb # Retrieval strategy comparison
|   |-- 05_predictive_model.ipynb      # At-risk classifier + SHAP
|   |-- 05b_quality_predictor.ipynb    # Cross-encoder distillation + SHAP
|
|-- scripts/
|   |-- run_ingestion.py             # Data ingestion automation
|   |-- run_embedding_comparison.py  # Embedding model evaluation
|   |-- run_build_index.py           # Pinecone index builder
|   |-- run_eval_queries.py          # End-to-end eval/test set runner
|
|-- tests/
|   |-- test_guardrails.py           # 15 guardrail validation tests
|   |-- test_api.py                  # 8 FastAPI endpoint tests
|   |-- test_quality_predictor.py    # 5 quality predictor tests
|   |-- test_retrieval.py            # 4 reranker tests
|
|-- agents/                          # Phase validation agents and workflow prompts
|-- commands/                        # Claude Code slash commands (GSD, RALF)
|-- data/processed/                  # Parquet output files
|-- docs/                            # Executive reports and interview prep
```

---

## Setup

### Prerequisites

- Python 3.11 or later
- pip

### Installation

```bash
git clone https://github.com/mamba4eva824/medical-education-rag.git
cd medical-education-rag

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configuration

Copy the environment template and add your API keys:

```bash
cp .env.example .env
```

Required keys (for Phases 2+):

| Key | Source | Purpose |
|-----|--------|---------|
| GROQ_API_KEY | groq.com | LLM inference for query expansion and generation |
| ANTHROPIC_API_KEY | console.anthropic.com | Optional Claude backend |
| HF_TOKEN | huggingface.co | Optional for gated model access |
| PINECONE_API_KEY | pinecone.io | Vector database for semantic search |

### Running the Pipeline

The processed data files are included in the repository. To regenerate from scratch:

```bash
python scripts/run_ingestion.py
```

### Validation

Each phase has an automated validation agent:

```bash
# Check Phase 1 (13 checks)
python agents/phase1_ingestion.py

# Check Phase 2 (9 checks)
python agents/phase2_embeddings.py

# Check Phase 3
python agents/phase3_rag_api.py

# Check Phase 4
python agents/phase4_guardrails_tests.py

# Run test suite
pytest tests/ -v

# Check all phases
python agents/run_all.py
```

---

## Responsibility Mapping

This project is structured around the two core pillars of the Applied AI Engineer role.

### AI Application Development

| Responsibility | Project Component |
|----------------|-------------------|
| Semantic search and content recommendations | Vector store with filtered retrieval (Phase 2) |
| RAG architectures and LLM orchestration | Full RAG chain with query expansion (Phase 3) |
| Backend services and APIs | FastAPI with /ask, /recommend, /health (Phase 3) |
| Vendor vs. open-source evaluation | Embedding model comparison with MLflow (Phase 2) |
| Responsible AI and guardrails | Content filtering and citation validation (Phase 4) |

### ML Engineering and Production Systems

| Responsibility | Project Component |
|----------------|-------------------|
| ML pipelines in Databricks | Delta tables and ported experiments (Phase 5) |
| Deployment with monitoring | Query metrics, latency tracking, error handling (Phase 4) |
| MLOps practices | Version control, pytest suite, MLflow tracking (Phase 4) |
| Predictive modeling | At-risk learner classifier and retrieval quality predictor (Phase 4) |
| Full lifecycle management | Model Registry stages in Databricks (Phase 5) |

---

## Development Workflow

This project uses two Claude Code slash commands for structured development:

- `/gsd` -- Planning workflow: audit, acceptance criteria, alternatives analysis, challenger review, user approval
- `/ralf` -- Execution workflow: implement, verify (automated gates), review (semantic check), learn, complete

Phase validation agents run automatically during the verify step to ensure correctness before marking tasks complete.

---

## License

This project is for educational and interview demonstration purposes.
