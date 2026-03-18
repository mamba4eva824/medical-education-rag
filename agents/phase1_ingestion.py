#!/usr/bin/env python3
"""
Phase 1 Agent: Data Ingestion & Chunking
=========================================
Job Responsibility: Design and build AI-powered features including semantic search,
content recommendations, and LLM-based tools

Validates:
- MedQuAD data loader (keivalya/MedQuad-MedicalQnADataset)
- Q&A-aware adaptive chunking pipeline
- Eval query holdout for Phase 2
- Processed data output in data/processed/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import PhaseAgent


class Phase1Agent(PhaseAgent):
    phase_name = "Phase 1: Data Ingestion & Chunking"

    def run(self):
        # --- File existence checks ---
        self.check(
            "medical_loader.py exists",
            self.file_has_content, "src/ingestion/medical_loader.py"
        )
        self.check(
            "chunker.py exists",
            self.file_has_content, "src/ingestion/chunker.py"
        )
        self.check(
            "Data ingestion notebook exists",
            self.file_has_content, "notebooks/01_data_ingestion.ipynb"
        )

        # --- Import checks ---
        self.check(
            "medical_loader imports cleanly",
            self.module_imports, "src.ingestion.medical_loader"
        )
        self.check(
            "chunker imports cleanly",
            self.module_imports, "src.ingestion.chunker"
        )

        # --- Functional checks ---
        self.check("MedQuADLoader exists with load() method", self._test_loader)
        self.check("MedicalChunker handles short Q&A (single chunk)", self._test_short_qa_chunking)
        self.check("MedicalChunker handles long Q&A (multi-chunk)", self._test_long_qa_chunking)
        self.check("Chunks have required metadata schema", self._test_chunk_metadata)
        self.check("Chunk IDs are deterministic (MD5)", self._test_chunk_ids)
        self.check("Eval queries saved for Phase 2", self._test_eval_queries)
        self.check("Test queries saved for final evaluation", self._test_test_queries)
        self.check("Processed chunks exist in data/processed/", self._test_processed_data)

    def _test_loader(self) -> tuple[bool, str]:
        try:
            from src.ingestion.medical_loader import MedQuADLoader
            if not hasattr(MedQuADLoader, 'load'):
                return False, "MedQuADLoader missing load() method"
            return True, "MedQuADLoader found with load() method"
        except ImportError as e:
            return False, str(e)

    def _test_short_qa_chunking(self) -> tuple[bool, str]:
        """Short answer (≤1000 chars) should produce a single chunk with full Q&A."""
        try:
            from src.ingestion.chunker import MedicalChunker
            chunker = MedicalChunker(max_chunk_size=1000)
            question = "What are the symptoms of heart failure?"
            answer = (
                "Symptoms include shortness of breath, fatigue, "
                "and swelling in the legs."
            )
            chunks = chunker.chunk_medquad(
                question, answer, metadata={'qtype': 'symptoms'}
            )
            if not chunks:
                return False, "Chunker returned empty list"
            if len(chunks) != 1:
                return False, f"Short answer should produce 1 chunk, got {len(chunks)}"
            if question not in chunks[0]['text']:
                return False, "Single chunk should contain the question"
            return True, "Short Q&A → 1 chunk with full Q&A text"
        except Exception as e:
            return False, str(e)

    def _test_long_qa_chunking(self) -> tuple[bool, str]:
        """Long answer (>1000 chars) should produce multiple chunks."""
        try:
            from src.ingestion.chunker import MedicalChunker
            chunker = MedicalChunker(max_chunk_size=200)
            question = "What are the treatments for diabetes?"
            answer = "Treatment paragraph one. " * 50 + "\n\n" + "Treatment paragraph two. " * 50
            chunks = chunker.chunk_medquad(
                question, answer, metadata={'qtype': 'treatment'}
            )
            if len(chunks) < 2:
                return False, f"Long answer should produce multiple chunks, got {len(chunks)}"
            # All chunks should carry the question in metadata
            for c in chunks:
                if 'question' not in c:
                    return False, "Multi-chunk: each chunk must have 'question' in metadata"
            return True, f"Long Q&A → {len(chunks)} chunks, all with question metadata"
        except Exception as e:
            return False, str(e)

    def _test_chunk_metadata(self) -> tuple[bool, str]:
        """Verify chunks have the full metadata schema."""
        try:
            from src.ingestion.chunker import MedicalChunker
            chunker = MedicalChunker()
            chunks = chunker.chunk_medquad(
                "Test question?", "Test answer.",
                metadata={'qtype': 'information'}
            )
            required_keys = {
                'chunk_id', 'text', 'question', 'qtype',
                'source', 'chunk_index', 'total_chunks'
            }
            for c in chunks:
                missing = required_keys - set(c.keys())
                if missing:
                    return False, f"Chunk missing keys: {missing}"
            return True, f"All required metadata keys present: {required_keys}"
        except Exception as e:
            return False, str(e)

    def _test_chunk_ids(self) -> tuple[bool, str]:
        try:
            from src.ingestion.chunker import MedicalChunker
            chunker = MedicalChunker()
            chunks1 = chunker.chunk_medquad(
                "Q?", "Answer text.", metadata={'qtype': 'information'}
            )
            chunks2 = chunker.chunk_medquad(
                "Q?", "Answer text.", metadata={'qtype': 'information'}
            )
            if chunks1[0]['chunk_id'] == chunks2[0]['chunk_id']:
                return True, "Same input produces same chunk_id"
            return False, "Chunk IDs are not deterministic"
        except Exception as e:
            return False, str(e)

    def _test_eval_queries(self) -> tuple[bool, str]:
        path = self.project_root / "data" / "processed" / "eval_queries.parquet"
        if not path.exists():
            return False, "Missing data/processed/eval_queries.parquet — run notebook 01"
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            if len(df) < 100:
                return False, f"Only {len(df)} eval pairs — need ~500"
            if 'Question' not in df.columns and 'question' not in df.columns:
                return False, f"Missing question column. Columns: {list(df.columns)}"
            return True, f"Found {len(df)} eval Q&A pairs"
        except Exception as e:
            return False, str(e)

    def _test_test_queries(self) -> tuple[bool, str]:
        path = self.project_root / "data" / "processed" / "test_queries.parquet"
        if not path.exists():
            return False, "Missing data/processed/test_queries.parquet — run notebook 01"
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            if len(df) < 50:
                return False, f"Only {len(df)} test pairs — need ~200"
            if 'question' not in df.columns:
                return False, f"Missing question column. Columns: {list(df.columns)}"
            return True, f"Found {len(df)} test Q&A pairs"
        except Exception as e:
            return False, str(e)

    def _test_processed_data(self) -> tuple[bool, str]:
        path = self.project_root / "data" / "processed" / "medical_chunks.parquet"
        if not path.exists():
            return False, "Missing data/processed/medical_chunks.parquet — run notebook 01"
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            if len(df) < 1000:
                return False, f"Only {len(df)} chunks — expected 15K+"
            if 'question' not in df.columns:
                return False, f"Chunks missing 'question' column. Columns: {list(df.columns)}"
            return True, f"Found {len(df)} chunks with question metadata"
        except Exception as e:
            return False, str(e)


if __name__ == "__main__":
    agent = Phase1Agent()
    report = agent.execute()
    sys.exit(0 if report.all_passed else 1)
