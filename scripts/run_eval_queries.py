"""Run sample queries from eval and test sets through a local RAG pipeline.

Uses BM25 retrieval (local) + Anthropic API for generation. No Pinecone or
HuggingFace model downloads required.
"""

import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

from src.generation.guardrails import validate_response
from src.generation.llm_client import LLMClient
from src.generation.prompts import EDUCATION_QA_PROMPT


def init_local_pipeline():
    """Build a local BM25-based pipeline (no external vector store)."""
    print("Loading chunks...")
    chunks_df = pd.read_parquet("data/processed/medical_chunks.parquet")
    corpus = chunks_df["text"].tolist()
    tokenized = [doc.lower().split() for doc in corpus]

    print(f"Building BM25 index over {len(corpus)} chunks...")
    bm25 = BM25Okapi(tokenized)

    print("Initializing Anthropic LLM client...")
    llm = LLMClient()

    print("Pipeline ready.\n")
    return bm25, chunks_df, llm


def retrieve_bm25(query: str, bm25: BM25Okapi, chunks_df: pd.DataFrame,
                  top_k: int = 5) -> list[dict]:
    """Retrieve top_k chunks via BM25."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        row = chunks_df.iloc[idx]
        results.append({
            "doc": {
                "text": row["text"],
                "qtype": row.get("qtype", ""),
                "source": row.get("source", "medquad"),
                "chunk_id": row.get("chunk_id", ""),
            },
            "bm25_score": float(scores[idx]),
        })
    return results


def generate_answer(query: str, sources: list[dict], llm: LLMClient) -> str:
    """Generate answer from retrieved sources."""
    context = "\n---\n".join(
        f'[{i+1}] {s["doc"]["text"]}' for i, s in enumerate(sources)
    )
    prompt = EDUCATION_QA_PROMPT.format(context=context, question=query)
    return llm.complete(prompt)


def compute_answer_overlap(predicted: str, reference: str) -> float:
    """Token overlap between predicted and reference answer."""
    pred_tokens = set(predicted.lower().split())
    ref_tokens = set(reference.lower().split())
    if not ref_tokens:
        return 0.0
    return len(pred_tokens & ref_tokens) / len(ref_tokens)


def run_queries(bm25, chunks_df, llm, df: pd.DataFrame,
                set_name: str, n: int = 10):
    """Run n queries through the local pipeline."""
    # Sample diverse qtypes
    sample = df.groupby("qtype", group_keys=False).apply(
        lambda x: x.sample(min(len(x), max(1, n // df["qtype"].nunique())),
                           random_state=42)
    ).head(n).reset_index(drop=True)

    results = []
    print(f"{'='*70}")
    print(f"  {set_name} — Running {len(sample)} queries")
    print(f"{'='*70}\n")

    for i, row in sample.iterrows():
        question = row["question"]
        reference = row["answer"]
        qtype = row["qtype"]

        print(f"[{i+1}/{len(sample)}] ({qtype}) {question[:70]}")

        start = time.time()
        try:
            # Retrieve
            sources = retrieve_bm25(question, bm25, chunks_df, top_k=5)

            # Generate
            answer = generate_answer(question, sources, llm)
            latency = (time.time() - start) * 1000

            # Validate
            validation = validate_response(answer, sources)

            # Measure overlap with reference
            overlap = compute_answer_overlap(answer, reference)

            results.append({
                "question": question,
                "qtype": qtype,
                "latency_ms": latency,
                "answer_length": len(answer),
                "n_sources": len(sources),
                "top_bm25_score": sources[0]["bm25_score"] if sources else 0,
                "answer_overlap": overlap,
                **validation,
            })

            status = "PASS" if validation["passed"] else "FAIL"
            checks_failed = [k for k, v in validation.items()
                             if k != "passed" and v is False]
            fail_info = f" (failed: {', '.join(checks_failed)})" if checks_failed else ""
            print(f"  [{status}]{fail_info} latency={latency:.0f}ms "
                  f"overlap={overlap:.2f}")
            print(f"  Answer: {answer[:150]}...")
            print()

        except Exception as e:
            latency = (time.time() - start) * 1000
            print(f"  ERROR: {e}\n")
            results.append({
                "question": question,
                "qtype": qtype,
                "latency_ms": latency,
                "answer_length": 0,
                "n_sources": 0,
                "top_bm25_score": 0,
                "answer_overlap": 0,
                "passed": False,
                "error": str(e),
            })

    return results


def print_summary(results: list[dict], set_name: str):
    """Print aggregate metrics."""
    df = pd.DataFrame(results)
    n = len(df)

    print(f"\n{'='*70}")
    print(f"  {set_name} — Summary ({n} queries)")
    print(f"{'='*70}")
    print(f"  Guardrail pass rate:     {df['passed'].mean():.0%}")

    for check in ["not_empty", "has_citations", "within_scope",
                   "source_grounded", "no_hallucinated_citations"]:
        if check in df.columns:
            print(f"    {check:30s} {df[check].mean():.0%}")

    print(f"  Avg answer overlap:      {df['answer_overlap'].mean():.3f}")
    print(f"  Avg latency:             {df['latency_ms'].mean():.0f}ms")
    print(f"  P50 latency:             {df['latency_ms'].median():.0f}ms")
    print(f"  Avg answer length:       {df['answer_length'].mean():.0f} chars")
    print(f"  Avg top BM25 score:      {df['top_bm25_score'].mean():.2f}")
    print()

    # Per-qtype breakdown
    if "qtype" in df.columns and "passed" in df.columns:
        print("  Per-qtype results:")
        for qtype, group in df.groupby("qtype"):
            p = group["passed"].mean()
            ovlp = group["answer_overlap"].mean()
            print(f"    {qtype:25s} pass={p:.0%}  overlap={ovlp:.2f}  "
                  f"({len(group)} queries)")
    print()


if __name__ == "__main__":
    bm25, chunks_df, llm = init_local_pipeline()

    # Eval set
    eval_df = pd.read_parquet("data/processed/eval_queries.parquet")
    eval_results = run_queries(bm25, chunks_df, llm, eval_df, "EVAL SET", n=10)
    print_summary(eval_results, "EVAL SET")

    # Test set
    test_df = pd.read_parquet("data/processed/test_queries.parquet")
    test_results = run_queries(bm25, chunks_df, llm, test_df, "TEST SET", n=10)
    print_summary(test_results, "TEST SET")
