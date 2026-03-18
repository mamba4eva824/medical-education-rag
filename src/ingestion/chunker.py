import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter


class MedicalChunker:
    def __init__(self, max_chunk_size: int = 800):
        self.max_chunk_size = max_chunk_size
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", "; ", "- ", " "],
        )

    def chunk_medquad(self, question: str, answer: str, metadata: dict) -> list[dict]:
        qtype = metadata.get("qtype", "information")
        prefix = f"Q: {question}\n\nA: "

        # Fast path: answer fits in one chunk — no splitting needed
        if len(answer) <= self.max_chunk_size:
            text = prefix + answer
            return [
                {
                    "chunk_id": hashlib.md5(
                        (question + text).encode(), usedforsecurity=False
                    ).hexdigest()[:12],
                    "text": text,
                    "question": question,
                    "qtype": qtype,
                    "source": "medquad",
                    "chunk_index": 0,
                    "total_chunks": 1,
                }
            ]

        # Long answers: split the answer first, then prepend question prefix
        splits = self.splitter.split_text(answer)
        total = len(splits)
        chunks = []
        for i, split_text in enumerate(splits):
            text = prefix + split_text
            chunks.append(
                {
                    "chunk_id": hashlib.md5(
                        (question + text).encode(), usedforsecurity=False
                    ).hexdigest()[:12],
                    "text": text,
                    "question": question,
                    "qtype": qtype,
                    "source": "medquad",
                    "chunk_index": i,
                    "total_chunks": total,
                }
            )
        return chunks
