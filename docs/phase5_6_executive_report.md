# Phase 5-6: A/B Pipeline Comparison & Azure Databricks Porting — Executive Report

## Job Responsibilities Addressed

**AI Application Development:** *RAG architectures, agentic workflows, prompt engineering strategies, and LLM orchestration patterns*

**ML Engineering & Production Systems:** *Build and maintain ML pipelines in Databricks, full lifecycle management, MLOps practices*

---

## What Was Built

### Phase 5: A/B Pipeline Comparison Framework

A pluggable retrieval architecture using the Strategy pattern, with an evaluation harness that quantifies the cost-vs-quality tradeoff between two pipeline modes.

### Phase 6: Azure Databricks Porting

The local ML pipeline ported to Azure Databricks with Unity Catalog, Delta Lake tables, managed MLflow experiments, and model registration — the same stack Vanderbilt uses.

---

## Architecture: Strategy Pattern

The monolithic RAG pipeline was refactored into a composable architecture where the retrieval mechanism is pluggable:

```
RetrievalStrategy (Protocol)
  ├── HybridRetrievalStrategy  — query expansion + BM25/dense RRF fusion
  └── DenseRetrievalStrategy   — Pinecone dense-only with full-text resolution

RAGPipeline(strategy, reranker, llm_client)
  └── answer(query) → delegates to strategy → rerank → generate → validate
```

This allows runtime mode switching via the API's `mode` parameter — no code changes, no redeployment.

### Why the Strategy Pattern

| Alternative Considered | Why Rejected |
|------------------------|--------------|
| Mode flag in `answer()` method | Violates single-responsibility, grows into spaghetti with 3+ modes |
| Separate pipeline classes | Duplicates reranking/generation/validation logic |
| **Strategy pattern** | Clean separation, open for extension, each strategy owns its retrieval concern |

---

## A/B Evaluation Results

### Key Discovery: Eval Design Matters More Than Model Tuning

Initial evaluation showed near-zero retrieval precision (P@5=0.044). Investigation revealed that **84% of eval questions had their answers completely removed from the index** during the train/eval split. The pipeline was being tested on questions it had no content for.

| Eval Version | Change | P@5 |
|---|---|---|
| v0.1 | Baseline (50% threshold, held-out) | 0.044 |
| v0.2 | Lowered threshold to 30% | 0.116 |
| v0.3 | Diagnosed eval design flaw | 0.175 (in-index subset) |
| v0.4 | **Created indexed eval set** | **0.476** |

The 4.1x improvement came from fixing the evaluation methodology, not the pipeline.

### Indexed Eval (answers in vector DB) — Retrieval Quality

| Pipeline | Precision@5 | MRR | Answer Overlap | Latency |
|----------|:-----------:|:---:|:--------------:|:-------:|
| **Full** | **0.380** | **0.800** | **50.0%** | 6.7s |
| Simple | 0.220 | 0.433 | 26.3% | 3.6s |

### Held-Out Eval (answers not in vector DB) — Graceful Degradation

| Pipeline | Precision@5 | MRR | Answer Overlap | Latency |
|----------|:-----------:|:---:|:--------------:|:-------:|
| Full | 0.120 | 0.237 | 24.8% | 6.1s |
| Simple | 0.160 | 0.258 | 20.1% | 3.6s |

### What This Proves

1. **The full pipeline justifies its cost when content exists.** MRR=0.800 means the right chunk ranks in position 1-2. The +73% precision over simple comes from query expansion catching synonym mismatches and BM25 catching exact medical terms.

2. **Both pipelines degrade similarly on unknown topics.** When content isn't indexed, neither pipeline can fabricate relevant chunks — guardrails catch this at 100% pass rate.

3. **Short-answer RAG is effectively solved** (MRR=0.952 on single-chunk questions). The remaining challenge is long multi-chunk answers where retrieval fragmentation limits coverage.

---

## Azure Databricks Deployment

### Infrastructure Provisioned

| Resource | Details |
|----------|---------|
| **Resource Group** | `medical-education-rag-rg` (East US) |
| **Workspace** | `medical-education-rag-dbx` (Premium SKU) |
| **Unity Catalog** | `medical_education_rag_dbx` (managed catalog) |
| **Schema** | `rag_data` |
| **Volume** | `raw_data` (managed, for CSV uploads) |
| **Cluster** | `medical-education-rag` (Standard_DS3_v2, single node, 30-min auto-terminate) |
| **Workspace URL** | `adb-7405618539948883.3.azuredatabricks.net` |

### Data Pipeline

```
Local parquet files
  → Export to CSV (notebooks/04_export_for_databricks.ipynb)
    → Upload to Unity Catalog Volume via REST API
      → Spark reads CSVs → Delta tables (db_01_delta_tables.py)
        → MLflow experiments (db_02_mlflow_experiments.py)
          → Model Registry (db_03_model_registry.py)
```

### Delta Tables Created

| Table | Rows | Purpose |
|-------|------|---------|
| `medical_chunks` | 35,886 | Retrieval-ready chunks with metadata |
| `eval_queries` | 500 | Development evaluation Q&A pairs |
| `test_queries` | 200 | Sealed test set for final metrics |

### MLflow Experiments Ported

| Experiment | Runs | Key Metrics |
|-----------|------|-------------|
| `embedding_comparison` | 3 | precision@5, MRR, encoding_time per model |
| `at_risk_prediction` | 1 | F1 mean/std from 5-fold CV |

### Registered Model

| Model | Version | Artifact |
|-------|---------|----------|
| `learner_at_risk_classifier` | 1 | StandardScaler + GradientBoostingClassifier pipeline |

---

## Components Delivered

| Component | File | Purpose |
|-----------|------|---------|
| Strategy Protocol | `src/retrieval/strategies.py` | Pluggable retrieval with 2 implementations |
| Refactored Pipeline | `src/generation/rag_chain.py` | Strategy-based RAGPipeline with per-component timing |
| API Mode Switching | `src/api/main.py`, `src/api/models.py` | `mode` parameter on `/ask` endpoint |
| Evaluation Harness | `src/evaluation/eval_harness.py` | Precision@5, MRR, answer overlap, MLflow logging |
| Shared Utilities | `src/utils/text.py` | `token_overlap` extracted for reuse |
| A/B CLI | `scripts/run_ab_evaluation.py` | Command-line pipeline comparison |
| Comparison Notebook | `notebooks/06_ab_pipeline_comparison.ipynb` | Charts and cost analysis |
| Export Notebook | `notebooks/04_export_for_databricks.ipynb` | Parquet to CSV export |
| Delta Tables | `databricks/db_01_delta_tables.py` | CSV to Delta Lake |
| MLflow Experiments | `databricks/db_02_mlflow_experiments.py` | Embedding comparison on Databricks |
| Model Registry | `databricks/db_03_model_registry.py` | Train and register at-risk model |
| Strategy Tests | `tests/test_strategies.py` | 11 tests for both strategies + pipeline modes |
| Accuracy Changelog | `docs/retrieval_accuracy_changelog.md` | Evolution of accuracy metrics |

**Test suite**: 28/28 passing | **Phase 5 agent**: 10/10 checks passing

---

## How This Maps to VSTAR

| This Project | VSTAR Application |
|-------------|-------------------|
| Strategy pattern for A/B retrieval | Test different retrieval approaches for VSTAR content without code changes |
| Indexed vs held-out eval methodology | Separate retrieval quality metrics from coverage gaps in VSTAR curriculum |
| Azure Databricks with Unity Catalog | Same infrastructure Vanderbilt uses for VSTAR data |
| Delta tables for medical content | Versioned, queryable curriculum content in Databricks |
| MLflow model registry | Centralized model versioning for production AI features |
| Per-component timing in pipeline | Identify latency bottlenecks in production tutoring sessions |

---

## Interview Talking Points

### 1. Evaluation Methodology

> "I initially saw P@5 of 0.044 and thought the pipeline was broken. Instead of immediately tuning hyperparameters, I investigated why. It turned out 84% of eval questions had their answers held out of the index — I was testing whether RAG can answer questions it has no content for. Once I created a proper indexed eval set, precision jumped to 0.476. The lesson: fix the measurement before fixing the model."

**Follow-up if asked "Isn't testing on indexed data cheating?":**
> "No — it's testing the right thing. RAG retrieval should be evaluated on 'can you find content you have?' Testing on held-out content measures graceful degradation, which is a separate concern. I run both evals — indexed for retrieval quality, held-out for out-of-scope handling. Each answers a different question."

### 2. Architecture Decision (Strategy Pattern)

> "I chose the Strategy pattern because retrieval approaches change frequently in production — you might start with dense-only, add BM25, experiment with learned sparse retrieval. The pipeline shouldn't know or care which approach is used. With this architecture, adding a new retrieval strategy is one class that implements `retrieve()`. No changes to reranking, generation, or validation."

**Follow-up if asked "Why not just a mode flag?":**
> "A mode flag works for two modes. By the third mode, `answer()` becomes a routing function with growing if/elif branches. The Strategy pattern keeps each retrieval approach self-contained and testable. It's also a cleaner interview signal — it shows I think about extensibility, not just getting the current feature working."

### 3. Full vs Simple Pipeline Tradeoff

> "The full pipeline with query expansion and hybrid search delivers 73% higher precision than dense-only. But it costs 2 LLM calls and 4 Pinecone calls per query versus 1 and 1. For a 200-student medical school class doing 10 queries each during an anatomy review, the full pipeline costs about $6/day more but ensures the tutor finds the right content. For asynchronous study where latency matters less, that's worth it. For real-time quizzing where 3.5 seconds matters, you might switch to simple mode."

### 4. Azure Databricks Porting

> "I provisioned an Azure Databricks workspace with Unity Catalog, uploaded the data to a managed Volume, created Delta tables with Spark, and registered the at-risk prediction model in the Model Registry. This is the same workflow I'd follow at Vanderbilt — the data lives in Delta Lake, experiments are tracked in managed MLflow, and models are versioned in Unity Catalog. The local development environment and Databricks share the same code structure; only the I/O layer changes."

**Follow-up if asked about cost:**
> "The single-node cluster I used costs about $2/hour on Azure. The entire porting exercise — data upload, Delta table creation, experiment runs, model registration — used under $5 of compute. In production, you'd schedule this as a Databricks job that spins up a cluster, runs, and terminates — no idle cost."

### 5. End-to-End Pipeline Maturity

> "This project covers the full ML lifecycle: data ingestion with quality checks, embedding model evaluation, retrieval architecture with A/B testing, guardrails for medical content, predictive modeling with SHAP explanations, API deployment with monitoring, and Databricks porting with Unity Catalog. Every phase has automated validation agents, MLflow tracking, and a test suite. That's the level of engineering discipline I'd bring to VSTAR."

### 6. Key Numbers to Cite

- **35,886** retrieval-ready chunks from **16,407** NIH Q&A pairs
- **P@5 = 0.380**, **MRR = 0.800** on indexed content (full pipeline)
- **50.0%** answer token overlap with ground truth
- **100%** guardrail pass rate on both eval sets
- **28** automated tests, **10/10** Phase 5 validation checks
- **3 Delta tables**, **2 MLflow experiments**, **1 registered model** in Azure Databricks
- **~$5** total Azure compute cost for the entire Databricks porting

### 7. Anticipated Questions

**"What would you improve next?"**
> "Three things: (1) Metadata-based retrieval — classify incoming query type and filter Pinecone by specialty to narrow the search space. The infrastructure exists in ContentRecommender but isn't wired into the pipeline. (2) Sibling chunk expansion — when we find chunk 3/6, automatically pull adjacent chunks for fuller context. (3) Latency optimization — the full pipeline at 6.7 seconds is too slow for interactive tutoring. Async query expansion or pre-computed expansions could cut that in half."

**"How would you handle VSTAR content that isn't Q&A format?"**
> "The Strategy pattern extends naturally. I'd create a `CurriculumRetrievalStrategy` that handles lecture transcripts, assessment items, and clinical cases — each with structure-aware chunking. The downstream pipeline (reranking, generation, guardrails) doesn't change. The eval harness would need task-specific ground truth, which I'd generate with faculty input."

**"What's your experience with Databricks in production?"**
> "This project demonstrates the core workflow: Unity Catalog for governance, Delta Lake for versioned data, MLflow for experiment tracking, and the Model Registry for deployment. In production, I'd add Databricks Jobs for scheduled retraining, Workflows for the full pipeline, and monitoring via Databricks SQL dashboards tracking retrieval quality over time."
