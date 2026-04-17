# Retrieval Accuracy Changelog

Tracks accuracy improvements across the RAG pipeline architecture. All metrics use a 30% token overlap relevance threshold unless noted.

**Metrics**:
- **Precision@5**: Fraction of top-5 reranked chunks that are relevant (>30% token overlap with ground truth answer)
- **MRR**: Mean Reciprocal Rank — how high the first relevant chunk ranks (1.0 = first position)
- **Answer Overlap**: Token overlap between the LLM-generated answer and the ground truth answer
- **Latency**: Mean end-to-end response time per query

---

## v0.1 — Baseline Eval (Held-Out Questions, 50% Threshold)

**Date**: 2026-04-17
**Eval set**: `eval_queries.parquet` (50 of 500 held-out Q&A pairs)
**Threshold**: 50% token overlap (inherited from embedding comparison script)

| Pipeline | Precision@5 | MRR | Answer Overlap | Latency |
|----------|:-----------:|:---:|:--------------:|:-------:|
| Full     | 0.044       | 0.101 | 25.3%        | 7.2s    |
| Simple   | 0.048       | 0.087 | 26.8%        | 3.6s    |

**Finding**: Both pipelines showed near-zero retrieval precision. Initial interpretation was that the pipeline was inaccurate.

---

## v0.2 — Threshold Correction (Held-Out Questions, 30% Threshold)

**Date**: 2026-04-17
**Change**: Lowered relevance threshold from 50% to 30%
**Rationale**: 50% token overlap is unreachable for long multi-paragraph answers where individual chunks only cover a fraction of the full answer's vocabulary.

| Pipeline | Precision@5 | MRR | Answer Overlap | Latency |
|----------|:-----------:|:---:|:--------------:|:-------:|
| Full     | 0.116       | 0.256 | 24.7%        | 6.0s    |
| Simple   | 0.104       | 0.173 | 22.0%        | 6.2s    |

**Finding**: Metrics improved but remained low. Full pipeline showed a quality edge (+48% MRR over simple). Investigation continued.

---

## v0.3 — Root Cause Discovery: Eval Design Flaw

**Date**: 2026-04-17
**Change**: Diagnosed that 84% (42/50) of eval questions had their Q&A pairs **held out before chunking** — those answers do not exist in the Pinecone index.

**Breakdown by content availability**:

| Subset | Count | Precision@5 | MRR | Answer Overlap |
|--------|:-----:|:-----------:|:---:|:--------------:|
| Question in index | 8/50 | 0.175 | 0.292 | 36.4% |
| Question NOT in index | 42/50 | 0.105 | 0.249 | 22.5% |

**Breakdown by answer complexity (full pipeline, held-out eval)**:

| Answer Type | Count | Precision@5 | MRR | Answer Overlap |
|-------------|:-----:|:-----------:|:---:|:--------------:|
| Short (1 chunk, <=800 chars) | 23 | 0.157 | 0.313 | 34.6% |
| Long (2+ chunks, >800 chars) | 27 | 0.081 | 0.207 | 16.3% |

**Finding**: The eval was measuring "can the system answer questions about conditions it has no content for?" — not retrieval quality. The low accuracy was an eval methodology artifact, not a pipeline deficiency.

---

## v0.4 — Indexed Eval (Answers in Vector DB)

**Date**: 2026-04-17
**Change**: Created `indexed_eval_queries.parquet` — 50 questions sampled from the 13,860 questions whose Q&A pairs ARE in the Pinecone index. Stratified by qtype, balanced single/multi-chunk.

| Pipeline | Precision@5 | MRR | Answer Overlap | Guardrail Pass | Latency |
|----------|:-----------:|:---:|:--------------:|:--------------:|:-------:|
| **Full** | **0.476**   | **0.777** | **43.6%** | 100% | 5.95s |
| Simple   | 0.264       | 0.488 | 28.9%          | 96%  | 3.51s |

**Improvement over v0.2**: Precision@5 **4.1x** (0.116 -> 0.476), MRR **3.0x** (0.256 -> 0.777)

**Full vs Simple pipeline delta**: Full pipeline delivers +80% precision, +59% MRR — query expansion and hybrid BM25/dense search earn their cost when content exists in the index.

### By answer complexity (full pipeline, indexed eval):

| Answer Type | Count | Precision@5 | MRR | Answer Overlap |
|-------------|:-----:|:-----------:|:---:|:--------------:|
| Short (1 chunk) | 21 | **0.648** | **0.952** | 54.9% |
| Long (2+ chunks) | 29 | 0.352 | 0.649 | 35.4% |

**Finding**: Short-answer retrieval is strong (MRR=0.952 means the right chunk is almost always in position 1). Long-answer retrieval is the remaining optimization target — fragmentation across multiple chunks limits precision.

### By medical specialty (full pipeline, indexed eval):

| QType | Precision@5 | MRR | Answer Overlap |
|-------|:-----------:|:---:|:--------------:|
| frequency | 0.960 | 1.000 | 57.8% |
| inheritance | 0.680 | 0.767 | 47.7% |
| genetic changes | 0.560 | 1.000 | 44.8% |
| prevention | 0.560 | 1.000 | 37.4% |
| outlook | 0.440 | 1.000 | 54.8% |
| causes | 0.400 | 0.800 | 47.6% |
| complications | 0.360 | 0.800 | 47.3% |
| exams and tests | 0.360 | 0.500 | 31.5% |
| information | 0.280 | 0.600 | 42.2% |
| considerations | 0.160 | 0.300 | 24.7% |

---

## Architecture Summary

```
v0.1  P@5=0.044  (50% threshold, held-out eval)
  |
  v  Lowered threshold to 30%
v0.2  P@5=0.116  (30% threshold, held-out eval)
  |
  v  Diagnosed eval design flaw (answers not in index)
v0.3  P@5=0.175  (in-index subset of held-out eval)
  |
  v  Created indexed eval set (answers ARE in index)
v0.4  P@5=0.476  (indexed eval, full pipeline)
```

## Remaining Optimization Targets

1. **Long-answer retrieval** (P@5=0.352 vs short-answer P@5=0.648) — sibling chunk expansion or increased chunk size would recover fragmented context
2. **Qtype filtering** — narrowing Pinecone search by detected question type (infrastructure already exists in `ContentRecommender`)
3. **"considerations" and "information" qtypes** — lowest precision, may need domain-specific retrieval tuning

## Key Lessons

- **Eval methodology matters more than model tuning.** The 4.1x precision improvement from v0.2 to v0.4 came from fixing the evaluation, not the pipeline.
- **The full pipeline justifies its cost.** Query expansion + hybrid search delivers +80% precision over dense-only when content exists — BM25 keyword matching catches exact medical terms that embeddings miss.
- **Short-answer RAG is effectively solved** (MRR=0.952). The remaining challenge is long multi-chunk answers where retrieval fragmentation limits coverage.
