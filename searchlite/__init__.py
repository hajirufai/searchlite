"""
SearchLite — A full-text search engine built from scratch.
No Elasticsearch, no Whoosh, no Lucene. Just Python.
"""

from searchlite.engine import SearchEngine
from searchlite.schema import Schema, TextField, KeywordField, NumericField
from searchlite.document import Document
from searchlite.query import TermQuery, BoolQuery, PhraseQuery, WildcardQuery, MatchAllQuery
from searchlite.analyzer import Analyzer, AnalyzerPipeline
from searchlite.tokenizer import WhitespaceTokenizer, RegexTokenizer, NgramTokenizer
from searchlite.normalizer import LowercaseNormalizer, UnicodeNormalizer, AccentStripper
from searchlite.stemmer import PorterStemmer
from searchlite.stopwords import StopWordFilter, ENGLISH_STOP_WORDS
from searchlite.scorer import BM25Scorer, TFIDFScorer

__version__ = "1.0.0"
__author__ = "Haji Rufai"

__all__ = [
    "SearchEngine",
    "Schema", "TextField", "KeywordField", "NumericField",
    "Document",
    "TermQuery", "BoolQuery", "PhraseQuery", "WildcardQuery", "MatchAllQuery",
    "Analyzer", "AnalyzerPipeline",
    "WhitespaceTokenizer", "RegexTokenizer", "NgramTokenizer",
    "LowercaseNormalizer", "UnicodeNormalizer", "AccentStripper",
    "PorterStemmer",
    "StopWordFilter", "ENGLISH_STOP_WORDS",
    "BM25Scorer", "TFIDFScorer",
]
