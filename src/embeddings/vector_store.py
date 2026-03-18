import os
import time
import logging

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        index_name: str = "medical-education-chunks",
    ):
        self.model = SentenceTransformer(model_name)
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = index_name
        dim = self.model.get_sentence_embedding_dimension()
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self.index = self.pc.Index(index_name)

    def build_index(self, chunks: list[dict], batch_size: int = 100) -> int:
        """Encode and upsert chunks. Returns total vectors upserted."""
        texts = [c["text"] for c in chunks]
        logger.info(f"Encoding {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)

        total_upserted = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]
            vectors = []
            for chunk, emb in zip(batch_chunks, batch_embeddings):
                metadata = {
                    "question": chunk["question"],
                    "qtype": chunk["qtype"],
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                    "text": chunk["text"][:500],
                }
                vectors.append((chunk["chunk_id"], emb.tolist(), metadata))
            # Retry up to 3 times on failure
            for attempt in range(3):
                try:
                    self.index.upsert(vectors=vectors)
                    total_upserted += len(vectors)
                    break
                except Exception as e:
                    logger.warning(
                        f"Upsert batch {i // batch_size} attempt {attempt + 1} failed: {e}"
                    )
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
            if (i // batch_size) % 50 == 0:
                logger.info(f"Upserted {total_upserted}/{len(chunks)} vectors")

        # Verify count (Pinecone is eventually consistent, wait briefly)
        time.sleep(5)
        stats = self.index.describe_index_stats()
        actual = stats.get("total_vector_count", 0)
        logger.info(f"Pinecone index has {actual} vectors (expected {len(chunks)})")
        return total_upserted

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[dict]:
        """Search for similar chunks. Returns list of dicts with metadata + score."""
        embedding = self.model.encode([query])[0].tolist()
        results = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter,
        )
        return [
            {
                "chunk_id": match["id"],
                "score": match["score"],
                **match.get("metadata", {}),
            }
            for match in results["matches"]
        ]
