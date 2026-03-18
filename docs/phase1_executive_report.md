# Phase 1: Data Ingestion & Chunking — Executive Report

## Job Responsibility Addressed

**AI Application Development:** *Design and build AI-powered features for VSTAR and other EDI platforms, including intelligent tutoring capabilities, semantic search, content recommendations, and LLM-based tools for learners and educators*

---

## What Was Built

A data ingestion and text chunking pipeline that transforms raw medical Q&A content from the MedQuAD dataset into retrieval-ready chunks for a semantic search and RAG (Retrieval-Augmented Generation) system.

### Components Delivered

| Component | File | Purpose |
|-----------|------|---------|
| Data Loader | `src/ingestion/medical_loader.py` | Loads MedQuAD from HuggingFace, deduplicates, produces 3-way stratified split |
| Chunker | `src/ingestion/chunker.py` | Q&A-aware adaptive chunking with metadata preservation |
| Pipeline Script | `scripts/run_ingestion.py` | Reproducible CLI automation |
| Notebook | `notebooks/01_data_ingestion.ipynb` | Interactive exploration and documentation |
| Validation Agent | `agents/phase1_ingestion.py` | 13 automated checks verifying correctness |

---

## Dataset: MedQuAD

**Source:** `keivalya/MedQuad-MedicalQnADataset` on HuggingFace
**Origin:** 12 NIH institutes (NCI, NHLBI, NIDDK, etc.)
**Format:** Question-Answer pairs with topic classification

| Metric | Value |
|--------|-------|
| Total Q&A pairs | 16,407 |
| After deduplication | 15,817 |
| Medical topic categories (qtypes) | 16 |
| Answer length range | 6 – 29,046 characters |
| Answer length median | 890 characters |
| Zero null/empty fields | Confirmed |

### Why MedQuAD

1. **Authoritative content** — sourced from NIH, the gold standard for medical information
2. **Q&A format** — directly mirrors how medical students interact with a tutoring chatbot
3. **Built-in evaluation** — Q&A pairs provide natural ground truth for measuring retrieval quality without manual labeling
4. **Topic diversity** — 16 medical question types spanning symptoms, treatment, causes, genetics, prevention, and more

---

## Three-Way Data Split

| Split | Count | Percentage | Purpose |
|-------|-------|------------|---------|
| **Index set** | 15,117 Q&A pairs | 95.6% | Chunked and indexed in vector store for retrieval |
| **Eval set** | 500 Q&A pairs | 3.2% | Development evaluation: embedding model comparison, retrieval tuning, quality predictor training |
| **Test set** | 200 Q&A pairs | 1.3% | Final unbiased metrics for production readiness assessment |

**Stratification:** Both holdout sets are proportionally sampled across all 16 medical topic categories, ensuring evaluation covers the full breadth of medical content.

**Why three splits:** The eval set guides development decisions (which embedding model to use, how to tune retrieval). The quality predictor model (Phase 4) trains on eval set queries. Reporting final metrics on the same data used for development would overstate performance. The sealed test set provides honest, unbiased metrics suitable for stakeholder reporting.

---

## Chunking Strategy: Q&A-Aware Adaptive Splitting

Traditional text chunking treats all documents identically — split every N characters. This loses the Q&A relationship that makes medical content retrievable.

### Our approach adapts based on answer length:

| Answer Length | % of Data | Strategy | Result |
|--------------|-----------|----------|--------|
| ≤ 1,000 chars | 59% | Keep full Q&A as single chunk | Question embedded in chunk text improves retrieval matching |
| > 1,000 chars | 41% | Split answer at paragraph boundaries | Each fragment carries the original question as metadata |

### Output Statistics

| Metric | Value |
|--------|-------|
| Total chunks produced | 30,198 |
| Single-chunk Q&A pairs | 8,158 (59% of Q&A pairs) |
| Multi-chunk fragments | 22,040 (from 6,560 Q&A pairs) |
| Mean chunk length | 677 characters |
| Median chunk length | 760 characters |
| Max chunk length | 1,092 characters |
| Chunk size on disk | 9.8 MB (Parquet) |

### Metadata preserved per chunk

Every chunk carries structured metadata enabling filtered search, attribution, and evaluation:

```
chunk_id        — deterministic MD5 hash (reproducible across runs)
text            — the chunk content
question        — the original medical question this chunk answers
qtype           — topic category (symptoms, treatment, causes, etc.)
source          — data provenance ("medquad")
chunk_index     — position within multi-chunk answers
total_chunks    — total fragments for this Q&A pair
```

---

## Quality Assurance

### Automated Validation: 13/13 Checks Passing

The Phase 1 validation agent (`agents/phase1_ingestion.py`) runs 13 automated checks:

| Category | Checks | Status |
|----------|--------|--------|
| File existence | loader, chunker, notebook exist with meaningful content | All pass |
| Import integrity | Both modules import without errors | All pass |
| Loader interface | MedQuADLoader has load() method | Pass |
| Short Q&A chunking | Single chunk produced, question embedded in text | Pass |
| Long Q&A chunking | Multiple chunks produced, question in metadata | Pass |
| Metadata schema | All 7 required keys present on every chunk | Pass |
| Deterministic IDs | Same input always produces same chunk_id | Pass |
| Eval data | 500 eval pairs saved with question column | Pass |
| Test data | 200 test pairs saved with question column | Pass |
| Processed chunks | 30,198 chunks saved with question metadata | Pass |

### Design Decisions and Trade-offs

| Decision | Rationale |
|----------|-----------|
| MD5 hash of question + text for chunk_id | Prevents collisions when different questions share similar answer fragments (common in NIH boilerplate text) |
| `usedforsecurity=False` on MD5 | Future-proofs against Python FIPS-mode restrictions; chunk IDs are identifiers, not security artifacts |
| RecursiveCharacterTextSplitter for long answers | Battle-tested splitter handles edge cases; separator hierarchy (`\n\n` > `\n` > `. ` > `; ` > `- ` > ` `) respects medical text structure |
| 50-character chunk overlap | Preserves context at split boundaries without excessive duplication |
| Parquet output format | Columnar storage with compression; 3x smaller than CSV; preserves types |

---

## How This Maps to VSTAR

| This Project | VSTAR Application |
|-------------|-------------------|
| MedQuAD Q&A ingestion | Curriculum content ingestion from VSTAR's data lake |
| Q&A-aware chunking | Structure-aware chunking for any educational content (lectures, assessments, clinical cases) |
| Stratified eval/test split | Rigorous evaluation framework for measuring AI feature impact on learning outcomes |
| Metadata preservation | Enables filtered search by specialty, topic, content type |
| Reproducible pipeline | `scripts/run_ingestion.py` runs identically in CI/CD; `random_state=42` ensures reproducibility |

---

## Next Steps (Phase 2)

Phase 1's output feeds directly into Phase 2: Embeddings, Vector Store & Recommendations.

- The 30,198 chunks will be encoded by 3 embedding models (MiniLM, MPNet, PubMedBert) and indexed in Pinecone
- The 500 eval pairs provide ground truth for comparing embedding model retrieval quality
- The `question` metadata on every chunk enables hybrid search strategies (match against both question and answer text)
- MLflow will track all embedding experiments for the vendor vs. open-source evaluation story

---

# Interview Talking Points

## 1. Data Selection & Domain Expertise

> "I chose MedQuAD because it's sourced from 12 NIH institutes — the same authoritative content medical students rely on. The Q&A format directly mirrors how students would interact with an intelligent tutoring system in VSTAR. Unlike generic medical datasets, every record has a structured question, a comprehensive answer, and a topic classification — that structure is what makes downstream retrieval and evaluation possible."

**Follow-up if asked "Why not use more datasets?":**
> "I started with one high-quality, authoritative dataset rather than mixing multiple sources of varying quality. The architecture is domain-agnostic — the loaders return a standard document format, so adding VSTAR's curriculum content later requires writing one new loader class, not changing any downstream code."

## 2. Chunking Strategy

> "I designed a Q&A-aware chunking strategy rather than using generic fixed-size splitting. Medical Q&A pairs have natural boundaries — the question-answer relationship is critical for retrieval. Short answers stay intact as single chunks with the question embedded in the text. Long answers split at paragraph boundaries, but every fragment carries the original question as metadata. This means during retrieval, the system can match queries against both the answer text and the original question — significantly improving recall for rephrased questions."

**Follow-up if asked "Why not semantic chunking?":**
> "Semantic chunking adds latency (embedding every sentence) and makes chunks non-deterministic across runs. For medical content where answers are already well-structured by NIH authors, respecting paragraph boundaries with a fallback splitter gives us 95% of the benefit at a fraction of the complexity. I can always layer in semantic chunking later if retrieval metrics show boundary-crossing issues."

## 3. Evaluation Rigor (Three-Way Split)

> "I implemented a proper three-way split — index, eval, and test — because the eval set is used during development to compare embedding models and train the retrieval quality predictor. Reporting final metrics on the same data that influenced model selection would overstate performance. The 200-item test set stays sealed until the final demo, giving us honest, unbiased numbers to present."

**Follow-up if asked about FERPA:**
> "MedQuAD is public NIH data, so no privacy concerns here. But the pipeline architecture is the same I'd use with FERPA-protected VSTAR data — the split logic, chunking, and evaluation framework don't change. What changes is access controls, de-identification, and audit logging around the data loader."

## 4. MLOps & Reproducibility

> "Every aspect of the pipeline is reproducible. `random_state=42` ensures identical splits across runs. Chunk IDs are deterministic MD5 hashes — same input always produces the same chunk. The pipeline runs as a single script (`python scripts/run_ingestion.py`) or an interactive notebook. A 13-check validation agent verifies correctness automatically. This is the kind of MLOps discipline I'd bring to VSTAR from day one."

## 5. Production Readiness Signals

> "Even in Phase 1, I'm thinking about production. Parquet over CSV for 3x compression. Metadata on every chunk for filtered retrieval. Deduplication before splitting to prevent training-eval leakage. A validation agent that catches regressions. These aren't afterthoughts — they're built into the foundation."

## 6. Connecting to the Job Description

| JD Requirement | What I Demonstrated |
|----------------|---------------------|
| "Semantic search across educational content" | Built the data foundation — 30K searchable chunks with structured metadata |
| "Content recommendations for personalized learning" | Metadata (qtype, question) enables filtered recommendations by medical topic |
| "MLOps practices: version control, testing, reproducibility" | Deterministic pipeline, 13-check validation agent, git-tracked code |
| "Evaluate vendor vs open-source" | Three-way split with proper eval/test separation enables rigorous model comparison in Phase 2 |
| "Build and maintain ML pipelines in Databricks" | Parquet output format is Delta Lake-ready for direct upload to Databricks |
| "Predictive modeling: learner performance prediction" | Eval/test infrastructure reused for at-risk prediction model evaluation in Phase 4 |

## 7. Key Numbers to Cite

- **16,407** Q&A pairs from **12 NIH institutes**
- **30,198** retrieval-ready chunks after adaptive chunking
- **16 medical topic categories** with stratified evaluation
- **13/13** automated validation checks passing
- **3-way split**: 95.6% index / 3.2% eval / 1.3% test
- **9.8 MB** total processed data (Parquet, compressed)

## 8. Anticipated Questions

**"What would you build first at Vanderbilt?"**
> "Exactly this — a data ingestion pipeline for VSTAR's existing curriculum content. It's the foundation everything else depends on. Semantic search, recommendations, and RAG all start with well-chunked, well-indexed content. I'd work with your data engineer to connect to the Databricks data lake and build loaders for VSTAR's content formats."

**"How would you handle VSTAR content that isn't Q&A format?"**
> "The chunker is designed for extensibility. `chunk_medquad()` handles Q&A pairs. I'd add `chunk_lecture()`, `chunk_assessment()`, `chunk_clinical_case()` methods — each with structure-aware splitting appropriate to that content type. The output schema stays the same: `{chunk_id, text, metadata...}`. Downstream retrieval doesn't need to know what kind of content it's searching."

**"How does this scale?"**
> "MedQuAD is 16K pairs. VSTAR might have 100K+ content items. The pipeline processes MedQuAD in under 30 seconds locally. At 100K items, we'd batch-process in Databricks using Spark, partition the Parquet output by specialty, and scale the Pinecone index — serverless scales automatically. The architecture doesn't change — just the data volume."
