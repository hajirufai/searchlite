"""Tests for the text analysis pipeline."""

import pytest
from searchlite.analyzer import (
    Analyzer, AnalyzerPipeline,
    standard_analyzer, simple_analyzer, keyword_analyzer, exact_analyzer,
)
from searchlite.tokenizer import RegexTokenizer, WhitespaceTokenizer
from searchlite.normalizer import LowercaseNormalizer, AccentStripper
from searchlite.stemmer import PorterStemmer
from searchlite.stopwords import StopWordFilter


class TestAnalyzer:
    def test_default_analyzer(self):
        analyzer = Analyzer()
        tokens = analyzer.analyze("Hello World")
        terms = [t.text for t in tokens]
        assert "hello" in terms
        assert "world" in terms

    def test_with_stemmer(self):
        analyzer = Analyzer(stemmer=PorterStemmer())
        terms = analyzer.analyze_to_terms("running quickly")
        assert "run" in terms

    def test_with_stop_words(self):
        analyzer = Analyzer(stop_filter=StopWordFilter())
        terms = analyzer.analyze_to_terms("the quick brown fox")
        assert "the" not in terms
        assert "quick" in terms

    def test_full_pipeline(self):
        analyzer = Analyzer(
            tokenizer=RegexTokenizer(),
            normalizers=[LowercaseNormalizer()],
            stop_filter=StopWordFilter(),
            stemmer=PorterStemmer(),
        )
        terms = analyzer.analyze_to_terms("The dogs are running in the park")
        assert "the" not in terms
        assert "are" not in terms
        assert "dog" in terms
        assert "run" in terms
        assert "park" in terms

    def test_empty_input(self):
        analyzer = Analyzer()
        assert analyzer.analyze("") == []
        assert analyzer.analyze_to_terms("") == []

    def test_accent_stripping(self):
        analyzer = Analyzer(
            normalizers=[LowercaseNormalizer(), AccentStripper()],
            stemmer=None,  # no stemming so we can test accents cleanly
        )
        terms = analyzer.analyze_to_terms("café résumé naïve")
        assert "cafe" in terms
        assert "resume" in terms
        assert "naive" in terms


class TestStandardAnalyzer:
    def test_standard(self):
        analyzer = standard_analyzer()
        terms = analyzer.analyze_to_terms("The quick brown fox jumps over the lazy dog")
        assert "the" not in terms
        assert "over" not in terms
        assert "quick" in terms
        assert "brown" in terms
        assert "fox" in terms
        assert "jump" in terms  # stemmed from jumps

    def test_case_insensitive(self):
        analyzer = standard_analyzer()
        terms1 = analyzer.analyze_to_terms("Python")
        terms2 = analyzer.analyze_to_terms("python")
        assert terms1 == terms2


class TestSimpleAnalyzer:
    def test_simple(self):
        analyzer = simple_analyzer()
        terms = analyzer.analyze_to_terms("Hello World")
        assert terms == ["hello", "world"]


class TestAnalyzerPipeline:
    def test_builder(self):
        analyzer = (
            AnalyzerPipeline()
            .tokenizer(RegexTokenizer())
            .add_normalizer(LowercaseNormalizer())
            .stop_words()
            .stemmer(PorterStemmer())
            .build()
        )
        terms = analyzer.analyze_to_terms("The Running Dogs")
        assert "the" not in terms
        assert "run" in terms
        assert "dog" in terms

    def test_minimal_builder(self):
        analyzer = AnalyzerPipeline().build()
        terms = analyzer.analyze_to_terms("Hello")
        assert len(terms) > 0
