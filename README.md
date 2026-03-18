# Medical Education RAG Pipeline

A retrieval-augmented generation (RAG) pipeline for medical education, built as a portfolio project for the Applied AI Engineer role at Vanderbilt University School of Medicine.

The system ingests authoritative medical Q&A content, chunks it for semantic retrieval, and will power an intelligent tutoring chatbot with citation-backed answers, content recommendations, and guardrails appropriate for medical education.

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Ingestion and Chunking | Complete |
| 2 | Embeddings, Vector Store, and Recommendations | Planned |
| 3 | RAG Architecture, Retrieval ML, and API | Planned |
| 4 | Guardrails, Predictive Models, and Tests | Planned |
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

---

## Project Structure

```
medical-education-rag/
|-- src/
|   |-- ingestion/
|   |   |-- medical_loader.py       # MedQuAD loader with 3-way stratified split
|   |   |-- chunker.py              # Q&A-aware adaptive chunking
|   |-- embeddings/                  # Phase 2: vector store, recommender
|   |-- retrieval/                   # Phase 3: reranker, hybrid search, query expansion
|   |-- generation/                  # Phase 3-4: RAG chain, prompts, guardrails
|   |-- prediction/                  # Phase 4: at-risk learner model
|   |-- api/                         # Phase 3: FastAPI application
|
|-- notebooks/
|   |-- 01_data_ingestion.ipynb      # Data loading and chunking pipeline
|
|-- scripts/
|   |-- run_ingestion.py             # CLI pipeline automation
|
|-- agents/                          # Phase validation agents and workflow prompts
|-- commands/                        # Claude Code slash commands (GSD, RALF)
|-- data/processed/                  # Parquet output files
|-- docs/                            # Executive reports and interview prep
|-- tests/                           # Phase 4: pytest suite
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
