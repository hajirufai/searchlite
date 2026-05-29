"""
Text analysis pipeline — chains tokenizer, normalizers, stemmer, and filters.
Each field can use a different analyzer configuration.
"""

from searchlite.tokenizer import Token, RegexTokenizer, WhitespaceTokenizer
from searchlite.normalizer import LowercaseNormalizer, UnicodeNormalizer, AccentStripper
from searchlite.stemmer import PorterStemmer
from searchlite.stopwords import StopWordFilter, ENGLISH_STOP_WORDS


class Analyzer:
    """
    A text analysis pipeline that transforms raw text into searchable terms.
    Pipeline: tokenize → normalize → filter stop words → stem
    """

    def __init__(self, tokenizer=None, normalizers=None,
                 stop_filter=None, stemmer=None):
        self.tokenizer = tokenizer or RegexTokenizer()
        self.normalizers = normalizers or [LowercaseNormalizer()]
        self.stop_filter = stop_filter
        self.stemmer = stemmer

    def analyze(self, text: str) -> list[Token]:
        """Run text through the full analysis pipeline."""
        if not text:
            return []

        tokens = self.tokenizer.tokenize(text)

        for normalizer in self.normalizers:
            tokens = normalizer.normalize(tokens)

        if self.stop_filter:
            tokens = self.stop_filter.filter(tokens)

        if self.stemmer:
            tokens = self.stemmer.stem_tokens(tokens)

        return tokens

    def analyze_to_terms(self, text: str) -> list[str]:
        """Convenience: return just the term strings."""
        return [tok.text for tok in self.analyze(text)]


class AnalyzerPipeline:
    """Builder for constructing custom analyzers."""

    def __init__(self):
        self._tokenizer = None
        self._normalizers = []
        self._stop_filter = None
        self._stemmer = None

    def tokenizer(self, tok):
        self._tokenizer = tok
        return self

    def add_normalizer(self, norm):
        self._normalizers.append(norm)
        return self

    def stop_words(self, words=None):
        self._stop_filter = StopWordFilter(words)
        return self

    def stemmer(self, stem):
        self._stemmer = stem
        return self

    def build(self) -> Analyzer:
        return Analyzer(
            tokenizer=self._tokenizer,
            normalizers=self._normalizers if self._normalizers else None,
            stop_filter=self._stop_filter,
            stemmer=self._stemmer,
        )


# Pre-built analyzer configurations

def standard_analyzer() -> Analyzer:
    """Standard analyzer: regex tokenize → lowercase → stop words → stem."""
    return Analyzer(
        tokenizer=RegexTokenizer(),
        normalizers=[LowercaseNormalizer()],
        stop_filter=StopWordFilter(),
        stemmer=PorterStemmer(),
    )


def simple_analyzer() -> Analyzer:
    """Simple analyzer: whitespace split → lowercase. No stemming or stop words."""
    return Analyzer(
        tokenizer=WhitespaceTokenizer(),
        normalizers=[LowercaseNormalizer()],
    )


def keyword_analyzer() -> Analyzer:
    """Keyword analyzer: treats entire input as one token, lowercased."""
    return Analyzer(
        tokenizer=WhitespaceTokenizer(),
        normalizers=[LowercaseNormalizer()],
    )


def exact_analyzer() -> Analyzer:
    """Exact match analyzer: no transformation at all."""
    return Analyzer(
        tokenizer=WhitespaceTokenizer(),
        normalizers=[],
    )
