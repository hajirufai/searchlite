"""
Relevance scoring — TF-IDF and BM25 (Okapi) implementations.
"""

import math
from searchlite.posting import PostingList


class Scorer:
    """Base scorer interface."""

    def score(self, tf: int, df: int, doc_length: int,
              avg_doc_length: float, total_docs: int) -> float:
        raise NotImplementedError


class TFIDFScorer(Scorer):
    """
    Classic TF-IDF scoring.
    score = tf(t,d) × idf(t)
    tf = 1 + log(freq) if freq > 0
    idf = log(N / df)
    """

    def score(self, tf: int, df: int, doc_length: int,
              avg_doc_length: float, total_docs: int) -> float:
        if tf <= 0 or df <= 0 or total_docs <= 0:
            return 0.0

        tf_score = 1.0 + math.log(tf)
        idf_score = math.log(total_docs / df)

        return tf_score * idf_score


class BM25Scorer(Scorer):
    """
    Okapi BM25 scoring — the industry standard for full-text search.

    score(q,d) = Σ IDF(qi) × (tf(qi,d) × (k1+1)) / (tf(qi,d) + k1 × (1 - b + b × |d|/avgdl))

    Parameters:
        k1: term frequency saturation. Higher = more weight to tf. Default 1.2
        b:  length normalization. 0 = no normalization, 1 = full. Default 0.75
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score(self, tf: int, df: int, doc_length: int,
              avg_doc_length: float, total_docs: int) -> float:
        if tf <= 0 or df <= 0 or total_docs <= 0:
            return 0.0

        # IDF with smoothing to avoid negative values
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

        # Prevent division by zero
        avg_dl = max(avg_doc_length, 1.0)

        # BM25 term frequency component
        tf_norm = (tf * (self.k1 + 1)) / (
            tf + self.k1 * (1.0 - self.b + self.b * doc_length / avg_dl)
        )

        return idf * tf_norm

    def score_multi_field(self, field_scores: list[tuple[float, float]]) -> float:
        """
        Combine scores from multiple fields with boosting.
        field_scores: list of (score, boost) tuples
        """
        total = 0.0
        for score, boost in field_scores:
            total += score * boost
        return total
