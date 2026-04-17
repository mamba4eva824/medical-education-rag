"""Shared text utilities for evaluation metrics."""

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
    "either", "neither", "each", "every", "all", "any", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "because", "if", "when", "while", "that",
    "this", "these", "those", "it", "its", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their", "what", "which", "who", "whom", "how", "where", "there",
}


def token_overlap(text_a: str, text_b: str) -> float:
    """Compute the fraction of content tokens in text_a that appear in text_b.

    Removes stop words before comparison. Returns 0.0 if text_a has no
    content tokens.
    """
    tokens_a = set(text_a.lower().split()) - STOP_WORDS
    tokens_b = set(text_b.lower().split()) - STOP_WORDS
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)
