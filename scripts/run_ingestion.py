"""Data ingestion pipeline: load MedQuAD, chunk, and save parquet files."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.ingestion.medical_loader import MedQuADLoader
from src.ingestion.chunker import MedicalChunker


def main() -> None:
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    loader = MedQuADLoader()
    documents, eval_pairs, test_pairs = loader.load()
    print(
        f"Documents: {len(documents)}, "
        f"Eval pairs: {len(eval_pairs)}, "
        f"Test pairs: {len(test_pairs)}"
    )

    # 2. Chunk documents
    chunker = MedicalChunker(max_chunk_size=1000)
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_medquad(
            question=doc["question"],
            answer=doc["text"],
            metadata={"qtype": doc["qtype"]},
        )
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    # 3. Save chunks
    chunks_df = pd.DataFrame(all_chunks)
    chunks_path = output_dir / "medical_chunks.parquet"
    chunks_df.to_parquet(chunks_path, index=False)
    print(f"Saved medical_chunks.parquet: {len(chunks_df)} rows")

    # 4. Save eval queries
    eval_df = pd.DataFrame(eval_pairs)
    eval_path = output_dir / "eval_queries.parquet"
    eval_df.to_parquet(eval_path, index=False)
    print(f"Saved eval_queries.parquet: {len(eval_df)} rows")

    # 5. Save test queries
    test_df = pd.DataFrame(test_pairs)
    test_path = output_dir / "test_queries.parquet"
    test_df.to_parquet(test_path, index=False)
    print(f"Saved test_queries.parquet: {len(test_df)} rows")

    # 6. Summary stats
    print("\n--- Summary ---")
    print(f"Chunks by qtype:\n{chunks_df['qtype'].value_counts()}")
    single = chunks_df[chunks_df["total_chunks"] == 1]
    multi = chunks_df[chunks_df["total_chunks"] > 1]
    print(f"\nSingle-chunk answers: {len(single)}")
    print(
        f"Multi-chunk answers: {len(multi)} (from {multi['question'].nunique()} Q&A pairs)"
    )
    print(f"\nChunk length stats:")
    print(chunks_df["text"].str.len().describe())


if __name__ == "__main__":
    main()
