"""
Stop word filtering — remove common words that add noise to search results.
"""

from searchlite.tokenizer import Token

ENGLISH_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for",
    "from", "had", "has", "have", "he", "her", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "just", "me", "my", "no", "nor",
    "not", "of", "on", "or", "our", "out", "own", "say", "she", "so",
    "some", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "up", "us",
    "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "about", "after", "again", "all", "also", "am", "any",
    "because", "been", "before", "being", "between", "both", "can",
    "could", "did", "does", "doing", "down", "during", "each", "few",
    "get", "got", "here", "herself", "him", "himself", "itself", "let",
    "more", "most", "must", "myself", "now", "off", "once", "only",
    "other", "over", "same", "should", "such", "tell", "under", "until",
    "upon", "want", "way", "well", "went", "what", "whom", "work",
})


class StopWordFilter:
    """Remove stop words from token lists."""

    def __init__(self, stop_words: frozenset[str] | None = None):
        self.stop_words = stop_words if stop_words is not None else ENGLISH_STOP_WORDS

    def filter(self, tokens: list[Token]) -> list[Token]:
        return [tok for tok in tokens if tok.text not in self.stop_words]
