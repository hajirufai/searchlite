"""Tests for TF-IDF and BM25 scoring."""

import math
import pytest
from searchlite.scorer import TFIDFScorer, BM25Scorer


class TestTFIDFScorer:
    def setup_method(self):
        self.scorer = TFIDFScorer()

    def test_basic_score(self):
        score = self.scorer.score(tf=5, df=10, doc_length=100,
                                  avg_doc_length=100, total_docs=1000)
        assert score > 0

    def test_zero_tf(self):
        score = self.scorer.score(tf=0, df=10, doc_length=100,
                                  avg_doc_length=100, total_docs=1000)
        assert score == 0.0

    def test_zero_df(self):
        score = self.scorer.score(tf=5, df=0, doc_length=100,
                                  avg_doc_length=100, total_docs=1000)
        assert score == 0.0

    def test_higher_tf_higher_score(self):
        score1 = self.scorer.score(tf=1, df=10, doc_length=100,
                                   avg_doc_length=100, total_docs=1000)
        score2 = self.scorer.score(tf=10, df=10, doc_length=100,
                                   avg_doc_length=100, total_docs=1000)
        assert score2 > score1

    def test_rarer_term_higher_score(self):
        # Term in 10 docs vs term in 500 docs
        score_rare = self.scorer.score(tf=5, df=10, doc_length=100,
                                       avg_doc_length=100, total_docs=1000)
        score_common = self.scorer.score(tf=5, df=500, doc_length=100,
                                         avg_doc_length=100, total_docs=1000)
        assert score_rare > score_common

    def test_idf_calculation(self):
        # IDF = log(N/df) = log(1000/10) = log(100) ≈ 4.605
        score = self.scorer.score(tf=1, df=10, doc_length=100,
                                  avg_doc_length=100, total_docs=1000)
        expected_idf = math.log(1000 / 10)
        expected_tf = 1.0 + math.log(1)
        assert abs(score - expected_tf * expected_idf) < 0.001


class TestBM25Scorer:
    def setup_method(self):
        self.scorer = BM25Scorer(k1=1.2, b=0.75)

    def test_basic_score(self):
        score = self.scorer.score(tf=5, df=10, doc_length=100,
                                  avg_doc_length=100, total_docs=1000)
        assert score > 0

    def test_zero_tf(self):
        score = self.scorer.score(tf=0, df=10, doc_length=100,
                                  avg_doc_length=100, total_docs=1000)
        assert score == 0.0

    def test_higher_tf_higher_score(self):
        score1 = self.scorer.score(tf=1, df=10, doc_length=100,
                                   avg_doc_length=100, total_docs=1000)
        score2 = self.scorer.score(tf=10, df=10, doc_length=100,
                                   avg_doc_length=100, total_docs=1000)
        assert score2 > score1

    def test_tf_saturation(self):
        """BM25 should show diminishing returns for high tf."""
        scores = []
        for tf in [1, 10, 100, 1000]:
            s = self.scorer.score(tf=tf, df=10, doc_length=100,
                                  avg_doc_length=100, total_docs=1000)
            scores.append(s)

        # Each increase should be smaller than the last
        diffs = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
        assert diffs[0] > diffs[1] > diffs[2]

    def test_length_normalization(self):
        """Shorter documents should score higher (all else equal)."""
        score_short = self.scorer.score(tf=5, df=10, doc_length=50,
                                        avg_doc_length=100, total_docs=1000)
        score_long = self.scorer.score(tf=5, df=10, doc_length=200,
                                       avg_doc_length=100, total_docs=1000)
        assert score_short > score_long

    def test_no_length_normalization(self):
        """With b=0, document length shouldn't matter."""
        scorer = BM25Scorer(k1=1.2, b=0.0)
        score_short = scorer.score(tf=5, df=10, doc_length=50,
                                   avg_doc_length=100, total_docs=1000)
        score_long = scorer.score(tf=5, df=10, doc_length=200,
                                  avg_doc_length=100, total_docs=1000)
        assert abs(score_short - score_long) < 0.001

    def test_custom_parameters(self):
        scorer = BM25Scorer(k1=2.0, b=0.5)
        score = scorer.score(tf=5, df=10, doc_length=100,
                             avg_doc_length=100, total_docs=1000)
        assert score > 0

    def test_multi_field_scoring(self):
        field_scores = [(0.5, 2.0), (0.3, 1.0)]
        combined = self.scorer.score_multi_field(field_scores)
        assert combined == 0.5 * 2.0 + 0.3 * 1.0

    def test_rare_vs_common(self):
        score_rare = self.scorer.score(tf=1, df=1, doc_length=100,
                                       avg_doc_length=100, total_docs=1000)
        score_common = self.scorer.score(tf=1, df=900, doc_length=100,
                                         avg_doc_length=100, total_docs=1000)
        assert score_rare > score_common
