# Medical Education RAG Pipeline — Implementation Guide

## Overview

This guide walks through building the entire project step-by-step. Every component maps to a specific job responsibility for the Applied AI Engineer role at Vanderbilt School of Medicine.

**Strategy:** Build locally (Weeks 1–4) using Claude Code + Cursor IDE, then port ML components to Databricks Free Edition (Week 5).

---

## Prerequisites & Setup

### 1. Environment Setup

```bash
cd /Users/christopherweinreich/Documents/Projects/medical_education_rag_diy

# Initialize git
git init
git add .
git commit -m "Initial project scaffolding"

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -e ".[dev]"
```

### 2. API Keys Needed

Copy `.env.example` to `.env` and fill in:

| Key | Source | Free? | Purpose |
|-----|--------|-------|---------|
| `GROQ_API_KEY` | groq.com | Yes (free tier) | LLM inference (Llama 3.1) for query expansion + generation |
| `HF_TOKEN` | huggingface.co/settings/tokens | Yes | Download gated models if needed |
| `ANTHROPIC_API_KEY` | console.anthropic.com | Paid | Optional: Claude as LLM backend |

### 3. HuggingFace Models (All Free, Downloaded Automatically)

These models are all on HuggingFace and will download automatically via `sentence-transformers`:

| Model | HF Hub ID | Size | Purpose |
|-------|-----------|------|---------|
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | 80MB | Baseline embedding model |
| MPNet | `sentence-transformers/all-mpnet-base-v2` | 420MB | Better quality embeddings |
| PubMedBert | `pritamdeka/S-PubMedBert-MS-MARCO` | 420MB | Domain-specific medical embeddings |
| Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 80MB | Reranking model |

No special downloads needed — `SentenceTransformer("model-name")` fetches them from HF Hub on first use.

---

## WEEK 1: Data Ingestion & Chunking

**Job Responsibility:** *Design and build AI-powered features including semantic search, content recommendations, and LLM-based tools*

### Step 1A: MedQuAD Data Loader

**File: `src/ingestion/medical_loader.py`**

**Dataset:** `keivalya/MedQuad-MedicalQnADataset` on HuggingFace

```python
from datasets import load_dataset
ds = load_dataset("keivalya/MedQuad-MedicalQnADataset")
```

**Dataset structure (verified):**
- **16,407 rows**, train split only
- **Columns:** `qtype` (str), `Question` (str), `Answer` (str)
- **16 qtypes:** information (4535), symptoms (2748), treatment (2442), inheritance (1446), frequency (1120), genetic changes (1087), causes (727), exams and tests (653), research (395), outlook (361), susceptibility (324), considerations (235), prevention (210), stages (77), complications (46), support groups (1)
- **Answer lengths:** Min 6 chars, Max 29K chars, Mean 1,303 chars, Median 890 chars
- **Zero nulls** — all fields populated

**What to implement:**
- `MedQuADLoader` class with `load()` method
- Returns three outputs:
  1. `documents: list[dict]` — each with `{text, question, qtype, source}`
  2. `eval_pairs: list[dict]` — 500 Q&A pairs for development evaluation (embedding comparison, quality predictor training)
  3. `test_pairs: list[dict]` — 200 Q&A pairs sealed for final unbiased metrics
- Split: ~15,200 for indexing, 500 eval, 200 test (stratified holdout)

**Interview talking point:** "I chose MedQuAD because it's authoritative NIH content in Q&A format — exactly what a medical student chatbot needs. The Q&A structure also gives me free evaluation ground truth for measuring retrieval quality."

### Step 1B: Q&A-Aware Adaptive Chunking

**File: `src/ingestion/chunker.py`**

MedQuAD answers have a median of 890 chars — most don't need splitting. The chunking strategy adapts based on answer length:

| Answer Length | ~% of Data | Strategy |
|--------------|------------|----------|
| ≤ 1000 chars | ~60% | Single chunk: `"Q: {question}\n\nA: {answer}"` |
| 1000–3000 chars | ~30% | Split answer into paragraphs, each chunk carries question in metadata |
| 3000+ chars | ~10% | Recursive split with overlap, question in metadata |

**What to implement:**
- `MedicalChunker` class with `chunk_medquad(question, answer, metadata)` method
- For short answers (≤ max_chunk_size): produce one chunk with full Q&A text
- For long answers: split into paragraph-level children using `RecursiveCharacterTextSplitter` as fallback
- Every chunk gets this metadata schema:

```python
{
    'chunk_id': 'md5hash',           # deterministic ID
    'text': 'chunk content',          # the actual text
    'question': 'original question',  # critical for retrieval — enables searching against Q and A
    'qtype': 'symptoms',             # free categorization from MedQuAD
    'source': 'medquad',
    'chunk_index': 0,                # position if answer was split
    'total_chunks': 1,               # how many chunks this answer produced
}
```

**Why this strategy:**
- Preserves Q&A relationships — a chunk about "Symptoms include fatigue" is more retrievable when the system knows it answers "What are the symptoms of heart failure?"
- The `question` metadata enables hybrid retrieval: search against both chunk text AND the original question
- Short answers stay intact (no information loss from splitting)
- Long answers split at natural paragraph boundaries

**What to log:** Total chunks, single-chunk vs multi-chunk answers, avg chunk length, chunks per qtype — save as Parquet.

### Step 1C: Run the Pipeline

**File: `notebooks/01_data_ingestion.ipynb`**

Create a notebook that:
1. Loads MedQuAD via `MedQuADLoader`
2. Holds out 500 Q&A pairs (stratified by qtype) as `eval_pairs`
3. Chunks remaining ~15,900 Q&A pairs with `MedicalChunker`
4. Saves to `data/processed/medical_chunks.parquet`
5. Saves eval pairs to `data/processed/eval_queries.parquet`
6. Saves test pairs to `data/processed/test_queries.parquet`
7. Prints summary statistics (chunk counts by qtype, single vs multi-chunk, length distribution)

**Deliverables after Week 1:**
- [ ] `src/ingestion/medical_loader.py` — MedQuAD loader with eval split
- [ ] `src/ingestion/chunker.py` — Q&A-aware adaptive chunking pipeline
- [ ] `notebooks/01_data_ingestion.ipynb` — end-to-end data pipeline notebook
- [ ] `data/processed/medical_chunks.parquet` — processed chunks with metadata
- [ ] `data/processed/eval_queries.parquet` — 500 eval Q&A pairs (used during development)
- [ ] `data/processed/test_queries.parquet` — 200 test Q&A pairs (sealed for final metrics)
- [ ] Git commit with all Week 1 code

---

## WEEK 2: Embeddings, Vector Store & Recommendations

**Job Responsibility:** *Evaluate vendor versus open-source AI products based on performance, cost, and reliability*

### Step 2A: Embedding Model Comparison

**File: `notebooks/02_embedding_comparison.ipynb`**

This is your vendor vs. open-source evaluation story. Compare 3 models from HuggingFace:

```python
from sentence_transformers import SentenceTransformer
import mlflow

models_to_test = {
    'all-MiniLM-L6-v2':                    {'type': 'open-source', 'size': '80MB'},
    'all-mpnet-base-v2':                    {'type': 'open-source', 'size': '420MB'},
    'pritamdeka/S-PubMedBert-MS-MARCO':     {'type': 'domain-specific', 'size': '420MB'},
}
```

**What to implement:**
1. Load each model via `SentenceTransformer(model_name)` — auto-downloads from HF
2. Encode all chunks, measure encoding time
3. Build a ChromaDB collection per model
4. Run test queries (prepare 20-30 medical questions)
5. Compute retrieval metrics: **Precision@5**, **MRR** (Mean Reciprocal Rank)
6. Log everything to MLflow:
   - `mlflow.log_params()` — model name, type, size, embedding dim
   - `mlflow.log_metrics()` — P@5, MRR, encoding time

**Evaluation using held-out MedQuAD pairs (from Phase 1):**
- Load the 500 eval Q&A pairs from `data/processed/eval_queries.parquet`
- The held-out answers were NOT chunked into the vector store — they're ground truth
- For each eval question, search the vector store and measure whether retrieved chunks are topically relevant (same qtype, overlapping medical concepts)
- Use the held-out answers as reference text for computing semantic similarity with retrieved chunks
- This gives you Precision@5 and MRR without manual labeling

**Interview talking point:** "PubMedBert outperformed general models on medical queries because it was pre-trained on PubMed abstracts — domain-specific models matter when your data has specialized vocabulary."

### Step 2B: ChromaDB Vector Store

**File: `src/embeddings/vector_store.py`**

**What to implement:**
- `VectorStore` class wrapping ChromaDB
- `build_index(chunks, model_name)` — encodes and inserts all chunks
- `search(query, n_results, specialty_filter)` — semantic search with optional metadata filtering
- Persistent storage to `./chroma_db/` directory

```python
import chromadb
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name='pritamdeka/S-PubMedBert-MS-MARCO',
                 persist_dir='./chroma_db'):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection('medical_chunks')
```

### Step 2C: Content Recommendation Engine

**File: `src/embeddings/recommender.py`**

**Job Responsibility:** *Design and build AI-powered features including content recommendations*

**What to implement:**
- `ContentRecommender` class that uses the same embedding infrastructure
- `get_similar(document_text, n, specialty_filter)` — find similar content
- `recommend_study_path(weak_topics, n_per_topic)` — given topics a learner struggles with, recommend content for each

**Interview talking point:** "This same recommendation engine could power personalized learning pathways in VSTAR — if a student struggles with cardiology, we surface related cardiology content."

### Step 2D: Build the Vector Store

**File: `notebooks/02b_build_vector_store.ipynb`**

Notebook that:
1. Loads processed chunks from Parquet
2. Builds the ChromaDB index using the best embedding model (likely PubMedBert)
3. Runs a few test searches to verify
4. Tests the content recommender

**Deliverables after Week 2:**
- [ ] `notebooks/02_embedding_comparison.ipynb` — model comparison with MLflow
- [ ] `src/embeddings/vector_store.py` — ChromaDB wrapper
- [ ] `src/embeddings/recommender.py` — content recommendation engine
- [ ] `notebooks/02b_build_vector_store.ipynb` — index building notebook
- [ ] `chroma_db/` — persisted vector store
- [ ] MLflow runs visible in `mlruns/` (run `mlflow ui --port 5000` to view)
- [ ] Git commit with all Week 2 code

---

## WEEK 3: RAG Architecture, Retrieval ML & API

**Job Responsibility:** *Apply RAG architectures, agentic workflows, prompt engineering strategies, and LLM orchestration patterns*

### Step 3A: Cross-Encoder Reranking

**File: `src/retrieval/reranker.py`**

Two-stage retrieval: fast vector search (top-20) → cross-encoder reranks to top-5.

The cross-encoder model auto-downloads from HuggingFace:
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
```

**What to implement:**
- `Reranker` class with `rerank(query, candidates, top_n)` method
- Scores each `(query, passage)` pair, returns sorted results with scores

### Step 3B: Retrieval Quality Predictor

**File: `src/retrieval/quality_predictor.py`**

A regression model that predicts a continuous relevance score (0–1) for each retrieved chunk, giving the pipeline a confidence signal before generation.

**Features extracted per (query, chunk) pair:**
- `cosine_similarity` — embedding distance from vector store
- `bm25_score` — sparse retrieval score from BM25
- `query_chunk_token_overlap` — percentage of query tokens found in chunk
- `chunk_length` — character count of the chunk
- `specialty_match` — binary: does the chunk's specialty match query intent?

**What to implement:**
- `RetrievalQualityPredictor` class with sklearn Pipeline (StandardScaler → GradientBoostingRegressor)
- `extract_features(query_embedding, chunk_embedding, bm25_score, chunk_length, specialty_match)` — builds feature vector from retrieval signals
- `predict(features)` — returns continuous relevance scores (0–1) for a batch of chunks
- `train_and_log(X, y, experiment_name)` — trains with 5-fold CV, logs to MLflow:
  - Params: model type, n_estimators, max_depth, learning_rate, n_features
  - Metrics: rmse, mae, r2_mean_cv5, r2_std_cv5
  - Artifact: serialized model via `mlflow.sklearn.log_model()`

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

class RetrievalQualityPredictor:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1
            ))
        ])
```

**Training data:** Use the 20–30 evaluation queries from Phase 2. For each (query, retrieved chunk) pair, assign a relevance label 0.0–1.0 (or use cross-encoder scores as soft labels). Training happens in Phase 4 via `notebooks/05b_quality_predictor.ipynb`.

**Where it fits in the RAG pipeline:** After hybrid retrieval, before generation — the predicted quality score determines whether retrieval is confident enough to generate, or whether to expand the query and retrieve more chunks.

**Interview talking point:** "The cross-encoder tells us which chunks are best. The quality predictor tells us whether retrieval *overall* is reliable enough to generate an answer — if predicted quality is low, we surface a confidence warning to the student rather than risk a bad answer."

### Step 3D: Hybrid Search (BM25 + Dense Fusion)

**File: `src/retrieval/hybrid_search.py`**

Combines sparse (BM25) and dense (vector) retrieval using Reciprocal Rank Fusion (RRF):

**What to implement:**
- `HybridSearcher` class
- `search(query, top_k)` — runs both BM25 and ChromaDB, fuses with RRF
- `_rrf_combine()` — merges ranked lists with formula: `score = Σ 1/(k + rank)`

### Step 3E: LLM-Powered Query Expansion

**File: `src/retrieval/query_expander.py`**

Uses Groq (free tier, fast inference) to expand queries:

```python
from groq import Groq
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
```

**What to implement:**
- `expand_query(query)` — sends query to Llama 3.1 via Groq, gets 3 alternative phrasings
- Returns `[original_query] + [expanded_queries]`

**Interview talking point:** "Query expansion is a simple agentic pattern — the LLM decides how to search, not just what to answer."

### Step 3F: End-to-End RAG Chain

**File: `src/generation/rag_chain.py`**

The core pipeline connecting everything:

**What to implement:**
- `RAGPipeline` class with `answer(query, top_k, rerank_n)` method
- Pipeline flow:
  1. **Expand** query via LLM
  2. **Retrieve** across expanded queries (hybrid search)
  3. **Deduplicate** by chunk_id
  4. **Rerank** with cross-encoder
  5. **Score retrieval quality** via RetrievalQualityPredictor (predicted_quality 0–1)
  6. **Build context** with numbered citations `[1], [2], ...`
  7. **Generate** answer via LLM with education prompt
  8. **Validate** with guardrails
- Returns: `{answer, sources, scores, predicted_quality, validation, expanded_queries}`

### Step 3G: Prompt Engineering

**File: `src/generation/prompts.py`**

Three prompt templates for different educational use cases:

1. `EDUCATION_QA_PROMPT` — Q&A with citations for student questions
2. `STUDY_GUIDE_PROMPT` — generates study guides from content
3. `SUMMARIZATION_PROMPT` — summarizes for board review

**Interview talking point:** "Different prompt templates serve different educational workflows — the same retrieval backend powers Q&A, study guides, and summaries."

### Step 3H: LLM Client Abstraction

**File: `src/generation/llm_client.py`**

**What to implement:**
- `LLMClient` class that wraps Groq (primary) with a consistent interface
- `complete(prompt, temperature, max_tokens)` method
- Optionally support Anthropic as a second provider for the vendor comparison story

### Step 3I: FastAPI Application

**File: `src/api/main.py`**

**Job Responsibility:** *Develop backend services and APIs that expose AI capabilities for integration*

**What to implement:**
- `POST /ask` — takes a question, returns RAG response with citations + validation
- `POST /recommend` — takes document text, returns similar content
- `GET /health` — returns status + model info
- Structured logging for every request (query, latency, pass/fail)

**File: `src/api/models.py`**

Pydantic schemas:
- `QueryRequest` — question (str), top_k (int), specialty (optional str)
- `QueryResponse` — answer, sources, validation, predicted_quality, latency_ms
- `RecommendRequest` — document_text, n, specialty
- `RecommendResponse` — recommendations list
- `Source` — text, source, specialty, relevance_score
- `ValidationResult` — has_citations, within_scope, source_grounded, passed

**Run locally:** `uvicorn src.api.main:app --reload --port 8000`
**Auto-docs:** Visit `http://localhost:8000/docs` for Swagger UI

### Step 3J: Retrieval Experiment Notebook

**File: `notebooks/03_retrieval_experiments.ipynb`**

Compare retrieval strategies with MLflow:
1. Baseline (vector only)
2. + Reranking
3. + Hybrid (BM25 + dense)
4. + Query expansion

Log P@5, MRR, nDCG@10, and latency for each. This gives you a clear "before/after" story.

**Deliverables after Week 3:**
- [ ] `src/retrieval/reranker.py` — cross-encoder reranking
- [ ] `src/retrieval/quality_predictor.py` — retrieval quality regression model
- [ ] `src/retrieval/hybrid_search.py` — BM25 + dense fusion
- [ ] `src/retrieval/query_expander.py` — LLM query expansion
- [ ] `src/generation/rag_chain.py` — end-to-end RAG pipeline
- [ ] `src/generation/prompts.py` — 3 education prompt templates
- [ ] `src/generation/llm_client.py` — LLM client abstraction
- [ ] `src/api/main.py` — FastAPI application
- [ ] `src/api/models.py` — Pydantic request/response schemas
- [ ] `notebooks/03_retrieval_experiments.ipynb` — retrieval comparison
- [ ] Git commit with all Week 3 code

---

## WEEK 4: Guardrails, Predictive Model & Tests

### Step 4A: Guardrails & Content Filtering

**File: `src/generation/guardrails.py`**

**Job Responsibility:** *Ensure responsible AI practices, including guardrails, content filtering, and transparency*

**What to implement:**
- `PROHIBITED_ADVICE` list — phrases that should never appear in educational AI output
- `validate_response(response, sources)` — returns dict of checks:
  - `has_citations` — does the response cite sources?
  - `within_scope` — no prohibited medical advice?
  - `not_empty` — response has substance?
  - `source_grounded` — response tokens overlap with source tokens (>30%)?
  - `no_hallucinated_citations` — all `[N]` references are within range?
  - `passed` — all checks pass?
- `_verify_claims()` — token overlap check between response and sources
- `_check_citation_range()` — regex to find `[N]` and verify N ≤ num_sources

**Interview talking point:** "In medical education, a wrong answer could mislead a future physician. Guardrails aren't optional — they're a core feature."

### Step 4B: At-Risk Learner Prediction Model

**File: `src/prediction/at_risk_model.py`**

**Job Responsibility:** *Contribute to predictive modeling for learner performance prediction, early intervention identification, and resource optimization*

**What to implement:**
- `AtRiskPipeline` class with sklearn Pipeline (StandardScaler → GradientBoostingClassifier)
- `create_features(raw_data)` — feature engineering:
  - `engagement_score` = weighted sum of resource_views, forum_posts, time_on_platform
  - `quiz_trend` = rolling average of quiz score changes per student
  - `attendance_x_quiz` = interaction feature
- `train_and_log(X, y, experiment_name)` — trains with 5-fold CV, logs to MLflow:
  - Params: n_estimators, max_depth, n_features
  - Metrics: f1_mean, f1_std
  - Artifact: serialized model via `mlflow.sklearn.log_model()`

### Step 4C: Retrieval Quality Predictor Training

**File: `notebooks/05b_quality_predictor.ipynb`**

Train the retrieval quality regression model using **cross-encoder distillation** — the cross-encoder is the accurate-but-slow "teacher," and the quality predictor is the fast "student."

**Training data generation (self-supervised from your own pipeline):**

1. Take 500 MedQuAD questions (from the eval set or a separate sample)
2. For each question, retrieve top-20 chunks from the vector store
3. Score each (question, chunk) pair with the cross-encoder → this is your regression target (0–1)
4. Extract features per (question, chunk) pair:
   - `cosine_similarity` — embedding distance from vector store
   - `bm25_score` — sparse retrieval score
   - `query_chunk_token_overlap` — % of query tokens found in chunk
   - `chunk_length` — character count
   - `qtype_match` — binary: does chunk's qtype match query intent?
5. Result: ~10,000 training examples (500 queries × 20 chunks each)

**Training:**
- `RetrievalQualityPredictor` with sklearn Pipeline (StandardScaler → GradientBoostingRegressor)
- 5-fold CV, log RMSE, MAE, R² to MLflow under `retrieval_quality` experiment
- SHAP explanations to show which retrieval signals drive quality predictions

**Where it fits in the RAG pipeline:**
```
predicted_quality < 0.4 → surface "low confidence" warning to student
predicted_quality ≥ 0.4 → generate answer normally
```

**Interview talking point:** "The cross-encoder tells us which chunks are best. The quality predictor tells us whether retrieval *overall* is reliable enough to generate — if quality is low, we warn the student rather than risk a bad answer. This also demonstrates both classification (at-risk) and regression (quality prediction) ML in the same system."

### Step 4D: Simulated Learner Dataset

**File: `notebooks/05_predictive_model.ipynb`**

Since we can't use real FERPA-protected student data, generate realistic synthetic data:

```python
np.random.seed(42)
n = 500
data = pd.DataFrame({
    'student_id': range(n),
    'quiz_avg': np.random.normal(75, 12, n),
    'attendance_pct': np.random.beta(8, 2, n) * 100,
    'resource_views': np.random.poisson(15, n),
    'forum_posts': np.random.poisson(3, n),
    'time_on_platform_hrs': np.random.gamma(5, 2, n),
    'prior_gpa': np.random.normal(3.2, 0.5, n).clip(0, 4),
})
data['at_risk'] = ((data['quiz_avg'] < 65) | (data['attendance_pct'] < 70)).astype(int)
```

Also add SHAP explanations:
```python
import shap
explainer = shap.TreeExplainer(pipeline.named_steps['model'])
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

**Interview talking point:** "SHAP values let advisors understand *why* a student is flagged — is it attendance? quiz trends? — so they can target the right intervention."

### Step 4E: Monitoring Module

**File: `src/api/monitoring.py`**

**Job Responsibility:** *Deploy with monitoring, logging, and error handling*

**What to implement:**
- `QueryMetrics` dataclass tracking:
  - total_queries, failed_queries, guardrail_failures, empty_results
  - latencies (deque of last 1000)
- `record(latency, passed, empty)` — record each query
- `summary()` — returns p50/p95 latency, error rates, guardrail failure rate

### Step 4F: Test Suite

**File: `tests/test_retrieval.py`**

**Job Responsibility:** *Implement MLOps practices: version control, testing, documentation, reproducibility*

**What to test:**
- `TestReranker.test_rerank_improves_order` — medical content ranks above irrelevant content
- `TestReranker.test_empty_candidates` — handles empty input gracefully

**File: `tests/test_guardrails.py`**

**What to test:**
- `test_blocks_medical_advice` — prohibited phrases are caught
- `test_detects_hallucinated_citations` — out-of-range citations flagged
- `test_passes_valid_response` — clean responses pass all checks

**File: `tests/test_api.py`**

**What to test:**
- `test_health_endpoint` — returns 200 with model info
- `test_ask_endpoint` — returns valid QueryResponse with predicted_quality field
- `test_ask_validates_input` — rejects empty/too-short questions

**File: `tests/test_quality_predictor.py`**

**What to test:**
- `test_predict_scores_in_range` — all predictions between 0.0 and 1.0
- `test_relevant_chunk_scores_higher` — relevant chunks score higher than irrelevant ones
- `test_extract_features_shape` — feature extraction returns correct dimensions
- `test_train_and_log_creates_mlflow_run` — training logs params, metrics, and model artifact

Run tests: `pytest tests/ -v`

**Deliverables after Week 4:**
- [ ] `src/generation/guardrails.py` — validation + content filtering
- [ ] `src/prediction/at_risk_model.py` — predictive model pipeline (classification)
- [ ] `src/api/monitoring.py` — query metrics tracking
- [ ] `notebooks/05_predictive_model.ipynb` — at-risk model training + SHAP
- [ ] `notebooks/05b_quality_predictor.ipynb` — retrieval quality regression training + SHAP
- [ ] `tests/test_retrieval.py` — retrieval tests
- [ ] `tests/test_guardrails.py` — guardrail tests
- [ ] `tests/test_quality_predictor.py` — quality predictor regression tests
- [ ] `tests/test_api.py` — API integration tests
- [ ] All tests passing: `pytest tests/ -v`
- [ ] Git commit with all Week 4 code

---

## WEEK 5: Port to Databricks Free Edition

**Job Responsibility:** *Build and maintain ML pipelines in Databricks*

### What to Port vs. What to Skip

| Component | Port? | Why |
|-----------|-------|-----|
| Processed datasets → Delta tables | **Yes** | Shows Delta Lake + Unity Catalog familiarity |
| MLflow experiments | **Yes** | Shows Databricks MLflow UI |
| Model Registry entry | **Yes** | Shows MLOps workflow on their platform |
| Results dashboard | **Yes** | Shows stakeholder communication |
| ChromaDB / FastAPI | **No** | Explain constraints; mention Databricks equivalents |

### Step 5A: Export Data for Upload

**File: `notebooks/04_export_for_databricks.ipynb`**

```python
# Export processed chunks as CSV for upload to Databricks
chunks_df.to_csv('data/exports/medical_chunks.csv', index=False)
learner_df.to_csv('data/exports/learner_data.csv', index=False)
```

Upload these CSVs via the Databricks UI (Catalog → Create Table → Upload).

### Step 5B: Delta Lake Tables

**File: `databricks/db_01_delta_tables.py`**

```python
# In Databricks notebook:
df = spark.read.csv('/Volumes/your_catalog/default/uploads/medical_chunks.csv',
                    header=True, inferSchema=True)
df.write.format('delta').saveAsTable('medical_education.chunks')

# Verify
display(spark.sql('''
    SELECT specialty, COUNT(*) as chunk_count,
           AVG(LENGTH(text)) as avg_length
    FROM medical_education.chunks
    GROUP BY specialty ORDER BY chunk_count DESC
'''))
```

### Step 5C: MLflow Experiments in Databricks

**File: `databricks/db_02_mlflow_experiments.py`**

Re-run the embedding comparison on a subset (200 chunks) to demonstrate MLflow in Databricks UI:

```python
%pip install sentence-transformers
import mlflow
mlflow.set_experiment('/Users/you@email.com/embedding_comparison')

for m in ['all-MiniLM-L6-v2', 'all-mpnet-base-v2']:
    with mlflow.start_run(run_name=m):
        # encode, evaluate, log metrics
```

### Step 5D: Model Registry

**File: `databricks/db_03_model_registry.py`**

Train the at-risk model on Databricks and register it:

```python
mlflow.sklearn.log_model(
    model, 'at_risk_model',
    registered_model_name='learner_at_risk_classifier'
)
```

This makes it visible in the Databricks Models UI with version history.

### Step 5E: Dashboard

Build a Databricks SQL dashboard with 4 panels:
1. **Dataset composition** — chunks by specialty (bar chart)
2. **Embedding comparison** — P@5, MRR by model (table/chart from MLflow)
3. **At-risk model performance** — F1, AUC, feature importance
4. **Retrieval quality** — before/after metrics for baseline vs enhanced pipeline

**Deliverables after Week 5:**
- [ ] `databricks/db_01_delta_tables.py`
- [ ] `databricks/db_02_mlflow_experiments.py`
- [ ] `databricks/db_03_model_registry.py`
- [ ] `notebooks/04_export_for_databricks.ipynb`
- [ ] Dashboard screenshots in repo
- [ ] Databricks workspace accessible for demo

---

## Complete File Map

```
medical-education-rag/
├── README.md                              # Project overview + demo instructions
├── IMPLEMENTATION_GUIDE.md                # This file
├── pyproject.toml                         # Dependencies + project config
├── .env.example                           # API key template
├── .gitignore
│
├── src/
│   ├── ingestion/
│   │   ├── medical_loader.py              # MedQuAD loader with eval split
│   │   └── chunker.py                     # Q&A-aware adaptive chunking
│   │
│   ├── embeddings/
│   │   ├── vector_store.py                # ChromaDB wrapper
│   │   └── recommender.py                 # Content recommendation engine
│   │
│   ├── retrieval/
│   │   ├── reranker.py                    # Cross-encoder reranking
│   │   ├── quality_predictor.py           # Retrieval quality regression model
│   │   ├── hybrid_search.py               # BM25 + dense fusion (RRF)
│   │   └── query_expander.py              # LLM-powered query expansion
│   │
│   ├── generation/
│   │   ├── rag_chain.py                   # End-to-end RAG pipeline
│   │   ├── prompts.py                     # Education prompt templates
│   │   ├── llm_client.py                  # LLM provider abstraction
│   │   └── guardrails.py                  # Response validation + content filtering
│   │
│   ├── prediction/
│   │   └── at_risk_model.py               # At-risk learner prediction pipeline
│   │
│   └── api/
│       ├── main.py                        # FastAPI application
│       ├── models.py                      # Pydantic schemas
│       └── monitoring.py                  # Query metrics + logging
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb            # Data loading + chunking pipeline
│   ├── 02_embedding_comparison.ipynb      # Model comparison with MLflow
│   ├── 02b_build_vector_store.ipynb       # Build ChromaDB index
│   ├── 03_retrieval_experiments.ipynb     # Retrieval strategy comparison
│   ├── 04_export_for_databricks.ipynb     # Export data for Databricks upload
│   ├── 05_predictive_model.ipynb          # At-risk model + SHAP
│   └── 05b_quality_predictor.ipynb        # Retrieval quality regression + SHAP
│
├── databricks/
│   ├── db_01_delta_tables.py              # Create Delta tables from CSVs
│   ├── db_02_mlflow_experiments.py        # Embedding comparison in Databricks
│   └── db_03_model_registry.py            # Register model in Databricks
│
├── tests/
│   ├── test_retrieval.py                  # Reranker + search tests
│   ├── test_quality_predictor.py          # Retrieval quality regression tests
│   ├── test_guardrails.py                 # Guardrail validation tests
│   └── test_api.py                        # FastAPI endpoint tests
│
├── data/
│   ├── raw/                               # Raw downloaded data
│   ├── processed/                         # Chunked + processed Parquet files
│   └── exports/                           # CSVs for Databricks upload
│
└── mlruns/                                # Local MLflow tracking (gitignored)
```

---

## Job Responsibility → Code Mapping

### AI Application Development

| Responsibility | What You Built | Key Files |
|---------------|----------------|-----------|
| Semantic search, content recommendations, LLM tools | ChromaDB search + ContentRecommender + RAG Q&A | `vector_store.py`, `recommender.py`, `rag_chain.py` |
| RAG architectures, agentic workflows, prompt engineering | Full RAG chain with query expansion + 3 prompts | `rag_chain.py`, `query_expander.py`, `prompts.py` |
| Backend services and APIs | FastAPI with /ask, /recommend, /health | `main.py`, `models.py` |
| Evaluate vendor vs open-source | MLflow experiment: 3 embedding models compared | `02_embedding_comparison.ipynb` |
| Responsible AI: guardrails, filtering, transparency | Guardrails module + validation in every response | `guardrails.py` |

### ML Engineering & Production Systems

| Responsibility | What You Built | Key Files |
|---------------|----------------|-----------|
| ML pipelines in Databricks | Delta tables + Spark SQL + ported experiments | `databricks/db_*.py` |
| Deploy with monitoring, logging, error handling | FastAPI + QueryMetrics + structured logging | `main.py`, `monitoring.py` |
| MLOps: version control, testing, reproducibility | Git + pytest + MLflow tracking | `tests/`, `mlruns/` |
| Performance, reliability, scalability | p50/p95 latency tracking, async API | `monitoring.py` |
| Full lifecycle: deploy, monitor, iterate, retire | Lifecycle stages documented + Model Registry | `db_03_model_registry.py` |
| Predictive modeling: learner outcomes | At-risk classifier + retrieval quality regressor with SHAP | `at_risk_model.py`, `quality_predictor.py`, notebooks 05/05b |

---

## Running the Project

### Local Development

```bash
# Activate environment
source .venv/bin/activate

# Start MLflow UI (separate terminal)
mlflow ui --port 5000

# Run the API
uvicorn src.api.main:app --reload --port 8000

# View API docs
open http://localhost:8000/docs

# Run tests
pytest tests/ -v

# View MLflow experiments
open http://localhost:5000
```

### Key Demo Commands

```bash
# Test the /ask endpoint
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of heart failure?", "top_k": 5}'

# Test the /recommend endpoint
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"document_text": "Cardiac output and ejection fraction", "n": 5}'

# Health check
curl http://localhost:8000/health
```

---

## Interview Preparation Quick Reference

**"What would you build first?"**
→ Semantic search over existing VSTAR curriculum content. High impact, lower risk than generative AI, and it builds the embedding + retrieval infrastructure everything else depends on.

**"How would you integrate with VSTAR?"**
→ FastAPI endpoints as microservices. /ask powers student Q&A, /recommend powers "suggested reading." Work with the full-stack team on auth, rate limiting, and UI integration.

**"How do you handle hallucinations?"**
→ Walk through guardrails.py: citation validation, source grounding, prohibited phrases. Layer in LLM-as-judge for nuanced checks and human review for edge cases.

**"How would you measure if AI helps students learn?"**
→ A/B testing with learning outcomes, educator feedback on retrieval quality, pre/post assessment scores. Log everything to MLflow.

**Closing statement:**
"I built the complete AI application locally for rapid iteration, then ported the ML engineering components to Databricks to validate they work on your platform. Here's my GitHub repo and here's my Databricks workspace."
