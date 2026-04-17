# Medical Education RAG Pipeline

A retrieval-augmented generation (RAG) pipeline for medical education, built as a portfolio project for the Applied AI Engineer role at Vanderbilt University School of Medicine.

The system ingests authoritative medical Q&A content, chunks it for semantic retrieval, and powers an intelligent tutoring chatbot with citation-backed answers, content recommendations, and guardrails appropriate for medical education.

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Ingestion and Chunking | Complete |
| 2 | Embeddings, Vector Store, and Recommendations | Complete |
| 3 | RAG Architecture, Retrieval ML, and API | Complete |
| 4 | Guardrails, Predictive Models, and Tests | Complete |
| 5 | A/B Pipeline Comparison and Evaluation | Complete |
| 6 | Databricks Porting | Planned |

---

## Architecture

```
MedQuAD (NIH)                 Embedding Models              LLM (Claude Haiku 4.5)
     |                              |                              |
     v                              v                              v
 MedQuADLoader ──> MedicalChunker ──> Pinecone ──> RetrievalStrategy ──> RAGPipeline
     |                                    |              |                    |
     v                                    v              v                    v
 eval/test split              VectorStore + BM25    Reranker             Guardrails
                                                                             |
                                                                             v
                                                                     FastAPI (/ask, /recommend)
                                                                             |
                                                                             v
                                                                     A/B Eval Harness + MLflow
```

### Strategy Pattern (A/B Pipeline)

The pipeline uses a pluggable `RetrievalStrategy` to support two modes:

| | Full Pipeline | Simple Pipeline |
|---|---|---|
| Query expansion | Claude generates 3 alternatives | None |
| Retrieval | BM25 + Pinecone dense, RRF fusion | Pinecone dense only |
| LLM calls/query | 2 (expand + generate) | 1 (generate only) |
| Pinecone calls/query | 4 (one per expanded query) | 1 |
| Reranking | Cross-encoder (same) | Cross-encoder (same) |
| Mean latency | ~6s | ~3.5s |

Switch modes at runtime via the API's `mode` parameter.

---

## Retrieval Accuracy

Evaluated on two test sets to measure different capabilities. See [docs/retrieval_accuracy_changelog.md](docs/retrieval_accuracy_changelog.md) for the full evolution.

### Indexed Eval (answers in vector DB) — tests retrieval quality

| Pipeline | Precision@5 | MRR | Answer Overlap | Guardrail Pass |
|----------|:-----------:|:---:|:--------------:|:--------------:|
| **Full** | **0.380**   | **0.800** | **50.0%** | 100% |
| Simple   | 0.220       | 0.433 | 26.3%          | 90%  |

### Held-Out Eval (answers not in vector DB) — tests graceful degradation

| Pipeline | Precision@5 | MRR | Answer Overlap | Guardrail Pass |
|----------|:-----------:|:---:|:--------------:|:--------------:|
| Full     | 0.120       | 0.237 | 24.8%        | 100% |
| Simple   | 0.160       | 0.258 | 20.1%        | 100% |

**Key finding**: When content exists in the index, MRR=0.800 means the correct chunk ranks in position 1-2. The full pipeline earns its extra cost with +73% precision and +85% MRR over simple. On held-out questions (content not indexed), both pipelines perform similarly — the accuracy gap is entirely explained by content availability.

---

## What Has Been Built

### Phase 1: Data Ingestion and Chunking

**Dataset:** MedQuAD -- 16,407 medical Q&A pairs from 12 NIH institutes, loaded via HuggingFace.

- Three-way stratified split: 15,117 documents for indexing, 500 eval pairs for development, 200 test pairs for final metrics
- Q&A-aware adaptive chunking: short answers stay intact, long answers split at paragraph boundaries with the original question preserved
- 35,886 retrieval-ready chunks with structured metadata

### Phase 2: Embeddings, Vector Store, and Recommendations

Three HuggingFace embedding models compared with MLflow tracking. PubMedBert selected for domain specialization in medical content.

| Model | P@5 | MRR | Dimensions |
|-------|-----|-----|------------|
| all-MiniLM-L6-v2 | 0.070 | 0.117 | 384 |
| pritamdeka/S-PubMedBert-MS-MARCO | 0.062 | 0.126 | 768 |
| all-mpnet-base-v2 | 0.090 | 0.169 | 768 |

35,886 vectors indexed in Pinecone serverless (cosine metric, AWS us-east-1).

### Phase 3: RAG Architecture, Retrieval, and API

- **Hybrid search**: BM25 sparse + Pinecone dense retrieval with Reciprocal Rank Fusion (k=60)
- **Cross-encoder reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` for two-stage retrieval
- **Query expansion**: Claude Haiku generates 3 alternative queries per search
- **FastAPI**: `/ask`, `/recommend`, `/health` endpoints with latency tracking
- **Guardrails**: Citation validation, scope checking, source grounding (30% token overlap threshold)

### Phase 4: Guardrails, Predictive Models, and Tests

- **5-point guardrail validation**: has_citations, within_scope, not_empty, source_grounded, no_hallucinated_citations
- **At-risk learner prediction**: GradientBoostingClassifier on synthetic student data with SHAP explanations
- **Retrieval quality prediction**: GradientBoostingRegressor trained via cross-encoder distillation
- **Query monitoring**: p50/p95 latency tracking, guardrail failure rate, empty result rate
- **Test suite**: 28 tests passing (pytest) — API, guardrails, reranker, quality predictor, strategies

### Phase 5: A/B Pipeline Comparison and Evaluation

- **Strategy pattern**: `RetrievalStrategy` Protocol with `HybridRetrievalStrategy` and `DenseRetrievalStrategy`
- **Per-component timing**: retrieval, reranking, and generation measured separately
- **Evaluation harness**: precision@5, MRR, answer token overlap, guardrail pass rate, MLflow logging
- **Dual eval methodology**: indexed eval (retrieval quality) and held-out eval (graceful degradation)
- **Shared utilities**: `token_overlap` and `STOP_WORDS` extracted to `src/utils/text.py`

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
|   |   |-- strategies.py           # RetrievalStrategy Protocol + implementations
|   |   |-- hybrid_search.py        # BM25 + dense retrieval with RRF fusion
|   |   |-- reranker.py             # Cross-encoder reranking
|   |   |-- query_expander.py       # LLM-powered query expansion
|   |   |-- quality_predictor.py    # Retrieval quality regression model
|   |-- generation/
|   |   |-- rag_chain.py            # RAGPipeline with strategy pattern
|   |   |-- llm_client.py           # Claude Haiku 4.5 wrapper
|   |   |-- prompts.py              # Prompt templates
|   |   |-- guardrails.py           # 5-point response validation
|   |-- evaluation/
|   |   |-- eval_harness.py         # A/B evaluation with MLflow logging
|   |-- prediction/
|   |   |-- at_risk_model.py        # At-risk learner classifier
|   |-- api/
|   |   |-- main.py                 # FastAPI with mode switching
|   |   |-- models.py               # Pydantic request/response schemas
|   |   |-- monitoring.py           # Query metrics collection
|   |-- utils/
|   |   |-- text.py                 # Shared token overlap utilities
|
|-- notebooks/
|   |-- 01_data_ingestion.ipynb
|   |-- 02_embedding_comparison.ipynb
|   |-- 02b_build_vector_store.ipynb
|   |-- 05_predictive_model.ipynb
|   |-- 05b_quality_predictor.ipynb
|   |-- 06_ab_pipeline_comparison.ipynb
|
|-- scripts/
|   |-- run_ingestion.py             # Data ingestion automation
|   |-- run_embedding_comparison.py  # Embedding model evaluation
|   |-- run_build_index.py           # Pinecone index builder
|   |-- run_ab_evaluation.py         # A/B pipeline comparison CLI
|
|-- tests/                           # 28 tests: API, guardrails, strategies, models
|-- agents/                          # Phase validation agents
|-- docs/                            # Executive reports, accuracy changelog
|-- data/processed/                  # Parquet output files
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

| Key | Source | Purpose |
|-----|--------|---------|
| ANTHROPIC_API_KEY | console.anthropic.com | LLM inference (Claude Haiku 4.5) |
| PINECONE_API_KEY | pinecone.io | Vector database for semantic search |
| HF_TOKEN | huggingface.co | Optional for gated model access |

### Running

```bash
# Start the API
uvicorn src.api.main:app --reload --port 8000

# Run tests
pytest tests/ -v

# Run A/B evaluation
python scripts/run_ab_evaluation.py --n-queries 10

# View MLflow experiments
mlflow ui --port 5000
```

### API Usage

```bash
# Full pipeline (query expansion + hybrid search)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of heart failure?", "mode": "full"}'

# Simple pipeline (dense-only, faster)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of heart failure?", "mode": "simple"}'
```

---

## Responsibility Mapping

This project is structured around the two core pillars of the Applied AI Engineer role.

### AI Application Development

| Responsibility | Project Component |
|----------------|-------------------|
| Semantic search and content recommendations | Vector store with filtered retrieval, ContentRecommender |
| RAG architectures and LLM orchestration | Strategy pattern with pluggable retrieval, full RAG chain |
| Backend services and APIs | FastAPI with mode switching, latency tracking |
| Vendor vs. open-source evaluation | Embedding model comparison + A/B pipeline comparison with MLflow |
| Responsible AI and guardrails | 5-point validation, citation checking, scope enforcement |

### ML Engineering and Production Systems

| Responsibility | Project Component |
|----------------|-------------------|
| ML pipelines in Databricks | Delta tables and ported experiments (Phase 6) |
| Deployment with monitoring | Query metrics, latency tracking, error handling |
| MLOps practices | 28 pytest tests, MLflow tracking, reproducible evaluation |
| Predictive modeling | At-risk learner classifier, retrieval quality predictor, SHAP explanations |
| Full lifecycle management | Model Registry stages in Databricks (Phase 6) |

---

## Development Workflow

This project uses two Claude Code slash commands for structured development:

- `/gsd` -- Planning workflow: audit, acceptance criteria, alternatives analysis, challenger review, user approval
- `/ralf` -- Execution workflow: implement, verify (automated gates), review (semantic check), learn, complete

---

## License

This project is for educational and interview demonstration purposes.
