"""Build the Pinecone vector store with the best embedding model.

Run this AFTER run_embedding_comparison.py to index all chunks with
the winning model. Default: PubMedBert (pritamdeka/S-PubMedBert-MS-MARCO).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.embeddings.vector_store import VectorStore


def main(
    model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
    sample_frac: float = 1.0,
) -> None:
    chunks_df = pd.read_parquet(
        PROJECT_ROOT / "data/processed/medical_chunks.parquet"
    )
    if sample_frac < 1.0:
        chunks_df = chunks_df.sample(frac=sample_frac, random_state=42)
        print(f"Sampled {len(chunks_df)} chunks ({sample_frac:.0%} of full dataset)")
    chunks = chunks_df.to_dict("records")
    print(f"Loaded {len(chunks)} chunks")

    print(f"Building Pinecone index with model: {model_name}")
    store = VectorStore(model_name=model_name)
    total = store.build_index(chunks)
    print(f"Upserted {total} vectors to Pinecone index '{store.index_name}'")

    # Quick verification search
    results = store.search("What are the symptoms of heart failure?", top_k=3)
    print(f"\nVerification search — top 3 results:")
    for r in results:
        print(f"  [{r['score']:.3f}] {r.get('text', '')[:100]}...")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "pritamdeka/S-PubMedBert-MS-MARCO"
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    main(model, sample_frac=frac)
