"""Tests for tokenizers."""

import pytest
from searchlite.tokenizer import (
    WhitespaceTokenizer, RegexTokenizer, NgramTokenizer, SentenceTokenizer, Token
)


class TestWhitespaceTokenizer:
    def setup_method(self):
        self.tok = WhitespaceTokenizer()

    def test_basic(self):
        tokens = self.tok.tokenize("hello world")
        assert [t.text for t in tokens] == ["hello", "world"]

    def test_positions(self):
        tokens = self.tok.tokenize("one two three")
        assert [t.position for t in tokens] == [0, 1, 2]

    def test_multiple_spaces(self):
        tokens = self.tok.tokenize("hello   world")
        assert [t.text for t in tokens] == ["hello", "world"]

    def test_tabs_newlines(self):
        tokens = self.tok.tokenize("hello\tworld\nfoo")
        assert [t.text for t in tokens] == ["hello", "world", "foo"]

    def test_empty_string(self):
        assert self.tok.tokenize("") == []

    def test_single_word(self):
        tokens = self.tok.tokenize("hello")
        assert len(tokens) == 1
        assert tokens[0].text == "hello"

    def test_preserves_punctuation(self):
        tokens = self.tok.tokenize("hello, world!")
        assert [t.text for t in tokens] == ["hello,", "world!"]

    def test_char_offsets(self):
        tokens = self.tok.tokenize("hello world")
        assert tokens[0].start_char == 0
        assert tokens[0].end_char == 5
        assert tokens[1].start_char == 6
        assert tokens[1].end_char == 11


class TestRegexTokenizer:
    def setup_method(self):
        self.tok = RegexTokenizer()

    def test_basic(self):
        tokens = self.tok.tokenize("hello world")
        assert [t.text for t in tokens] == ["hello", "world"]

    def test_strips_punctuation(self):
        tokens = self.tok.tokenize("hello, world! foo.")
        assert [t.text for t in tokens] == ["hello", "world", "foo"]

    def test_hyphenated_words(self):
        tokens = self.tok.tokenize("well-known machine-learning")
        assert [t.text for t in tokens] == ["well-known", "machine-learning"]

    def test_contractions(self):
        tokens = self.tok.tokenize("don't can't won't")
        assert [t.text for t in tokens] == ["don't", "can't", "won't"]

    def test_numbers(self):
        tokens = self.tok.tokenize("version 3 release 12")
        assert [t.text for t in tokens] == ["version", "3", "release", "12"]

    def test_mixed(self):
        tokens = self.tok.tokenize("Python 3.12 is great!")
        texts = [t.text for t in tokens]
        assert "Python" in texts
        assert "3" in texts
        assert "12" in texts
        assert "great" in texts

    def test_empty(self):
        assert self.tok.tokenize("") == []

    def test_only_punctuation(self):
        assert self.tok.tokenize("...!!!") == []

    def test_custom_pattern(self):
        tok = RegexTokenizer(pattern=r"\S+")
        tokens = tok.tokenize("hello world")
        assert [t.text for t in tokens] == ["hello", "world"]


class TestNgramTokenizer:
    def test_bigrams(self):
        tok = NgramTokenizer(min_n=2, max_n=2)
        tokens = tok.tokenize("abc")
        texts = [t.text for t in tokens]
        assert "ab" in texts
        assert "bc" in texts

    def test_trigrams(self):
        tok = NgramTokenizer(min_n=3, max_n=3)
        tokens = tok.tokenize("abcd")
        texts = [t.text for t in tokens]
        assert "abc" in texts
        assert "bcd" in texts

    def test_range(self):
        tok = NgramTokenizer(min_n=2, max_n=3)
        tokens = tok.tokenize("abc")
        texts = [t.text for t in tokens]
        assert "ab" in texts
        assert "bc" in texts
        assert "abc" in texts

    def test_short_input(self):
        tok = NgramTokenizer(min_n=3, max_n=3)
        tokens = tok.tokenize("ab")
        assert len(tokens) == 0


class TestSentenceTokenizer:
    def test_basic(self):
        tok = SentenceTokenizer()
        sents = tok.tokenize("Hello world. How are you? Fine thanks.")
        assert len(sents) == 3

    def test_single_sentence(self):
        tok = SentenceTokenizer()
        sents = tok.tokenize("Just one sentence")
        assert len(sents) == 1


class TestToken:
    def test_repr(self):
        t = Token("hello", 0, 0, 5)
        assert "hello" in repr(t)
        assert "pos=0" in repr(t)
