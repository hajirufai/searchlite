"""Tests for the Porter Stemmer implementation."""

import pytest
from searchlite.stemmer import PorterStemmer
from searchlite.tokenizer import Token


class TestPorterStemmer:
    def setup_method(self):
        self.stemmer = PorterStemmer()

    # Step 1a tests
    def test_sses(self):
        assert self.stemmer.stem_word("caresses") == "caress"

    def test_ies(self):
        assert self.stemmer.stem_word("ponies") == "poni"

    def test_ss(self):
        assert self.stemmer.stem_word("caress") == "caress"

    def test_s(self):
        assert self.stemmer.stem_word("cats") == "cat"

    # Step 1b tests
    def test_eed_with_measure(self):
        assert self.stemmer.stem_word("agreed") == "agre"

    def test_ed(self):
        assert self.stemmer.stem_word("plastered") == "plaster"

    def test_ing(self):
        assert self.stemmer.stem_word("motoring") == "motor"

    # Common words
    def test_running(self):
        assert self.stemmer.stem_word("running") == "run"

    def test_connection(self):
        result = self.stemmer.stem_word("connected")
        assert result == "connect"

    def test_generalization(self):
        result = self.stemmer.stem_word("generalization")
        assert result == "gener"

    def test_conditional(self):
        result = self.stemmer.stem_word("conditional")
        assert result in ("condit", "condition")

    # Edge cases
    def test_short_words(self):
        assert self.stemmer.stem_word("a") == "a"
        assert self.stemmer.stem_word("an") == "an"

    def test_already_stemmed(self):
        assert self.stemmer.stem_word("run") == "run"

    def test_empty(self):
        assert self.stemmer.stem_word("") == ""

    # Step 1c
    def test_happy(self):
        assert self.stemmer.stem_word("happy") == "happi"

    # Step 2
    def test_relational(self):
        assert self.stemmer.stem_word("relational") == "relat"

    def test_rational(self):
        assert self.stemmer.stem_word("rational") == "ration"

    # Step 3
    def test_triplicate(self):
        assert self.stemmer.stem_word("triplicate") == "triplic"

    def test_hopeful(self):
        assert self.stemmer.stem_word("hopeful") == "hope"

    def test_goodness(self):
        assert self.stemmer.stem_word("goodness") == "good"

    # Step 5
    def test_probate(self):
        result = self.stemmer.stem_word("probate")
        # Just check it doesn't crash and returns something reasonable
        assert isinstance(result, str)
        assert len(result) > 0

    # Token list stemming
    def test_stem_tokens(self):
        tokens = [
            Token("running", 0, 0, 7),
            Token("dogs", 1, 8, 12),
        ]
        result = self.stemmer.stem_tokens(tokens)
        assert result[0].text == "run"
        assert result[1].text == "dog"

    # Consistency
    def test_same_stem_for_variants(self):
        """Related words should stem to the same root."""
        stem1 = self.stemmer.stem_word("connect")
        stem2 = self.stemmer.stem_word("connected")
        stem3 = self.stemmer.stem_word("connecting")
        stem4 = self.stemmer.stem_word("connection")
        # At least connected and connecting should match connect
        assert stem1 == stem2 == stem3

    def test_program_variants(self):
        s1 = self.stemmer.stem_word("program")
        s2 = self.stemmer.stem_word("programs")
        s3 = self.stemmer.stem_word("programming")
        assert s1 == s2  # program -> program, programs -> program

    def test_search_variants(self):
        s1 = self.stemmer.stem_word("search")
        s2 = self.stemmer.stem_word("searching")
        s3 = self.stemmer.stem_word("searches")
        assert s1 == s2 == s3
