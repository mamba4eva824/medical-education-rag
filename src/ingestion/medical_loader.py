"""Loader for the MedQuAD medical Q&A dataset."""

from datasets import load_dataset
from sklearn.model_selection import train_test_split


class MedQuADLoader:
    """Loads and splits the MedQuAD dataset into documents, eval, and test sets."""

    DATASET_ID = "keivalya/MedQuad-MedicalQnADataset"

    def load(self) -> tuple[list[dict], list[dict], list[dict]]:
        """Load the MedQuAD dataset, deduplicate, and split three ways.

        Returns ``(documents, eval_pairs, test_pairs)`` where *documents* are
        dicts with keys ``{text, question, qtype, source}``, and both
        *eval_pairs* and *test_pairs* are dicts with keys
        ``{question, answer, qtype}``.
        """
        ds = load_dataset(self.DATASET_ID)
        rows = ds["train"].to_pandas()

        # Deduplicate on exact Answer text, keeping the first occurrence.
        rows = rows.drop_duplicates(subset="Answer", keep="first").reset_index(drop=True)

        # Separate qtypes with fewer than 5 examples (forced into documents).
        qtype_counts = rows["qtype"].value_counts()
        rare_qtypes = set(qtype_counts[qtype_counts < 5].index)

        rare_mask = rows["qtype"].isin(rare_qtypes)
        rare_rows = rows[rare_mask]
        main_rows = rows[~rare_mask]

        # Split 1: hold out 700 rows (stratified) from the main rows.
        doc_rows, holdout_rows = train_test_split(
            main_rows,
            test_size=700,
            stratify=main_rows["qtype"],
            random_state=42,
        )

        # Split 2: divide holdout into 500 eval + 200 test.
        # No stratify on the second split — the 700 holdout is already
        # stratified, and rare qtypes within it may have too few rows.
        eval_rows, test_rows = train_test_split(
            holdout_rows,
            test_size=200,
            random_state=42,
        )

        # Build document dicts (rare rows + doc_rows).
        documents: list[dict] = []
        for _, row in rare_rows.iterrows():
            documents.append({
                "text": row["Answer"],
                "question": row["Question"],
                "qtype": row["qtype"],
                "source": "medquad",
            })
        for _, row in doc_rows.iterrows():
            documents.append({
                "text": row["Answer"],
                "question": row["Question"],
                "qtype": row["qtype"],
                "source": "medquad",
            })

        def _to_pairs(df):
            return [
                {"question": r["Question"], "answer": r["Answer"], "qtype": r["qtype"]}
                for _, r in df.iterrows()
            ]

        return documents, _to_pairs(eval_rows), _to_pairs(test_rows)
