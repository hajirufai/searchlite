"""
Tokenizers break text into individual tokens with position tracking.
"""

import re
from dataclasses import dataclass


@dataclass
class Token:
    """A single token with position info."""
    text: str
    position: int
    start_char: int
    end_char: int

    def __repr__(self):
        return f"Token({self.text!r}, pos={self.position})"


class WhitespaceTokenizer:
    """Split on whitespace. Simplest tokenizer."""

    def tokenize(self, text: str) -> list[Token]:
        tokens = []
        pos = 0
        for match in re.finditer(r'\S+', text):
            tokens.append(Token(
                text=match.group(),
                position=pos,
                start_char=match.start(),
                end_char=match.end(),
            ))
            pos += 1
        return tokens


class RegexTokenizer:
    """
    Word-boundary tokenizer. Extracts alphanumeric sequences,
    handles contractions and hyphenated words.
    """

    def __init__(self, pattern: str = r"[\w]+(?:[-'][\w]+)*"):
        self._pattern = re.compile(pattern)

    def tokenize(self, text: str) -> list[Token]:
        tokens = []
        pos = 0
        for match in self._pattern.finditer(text):
            tokens.append(Token(
                text=match.group(),
                position=pos,
                start_char=match.start(),
                end_char=match.end(),
            ))
            pos += 1
        return tokens


class NgramTokenizer:
    """
    Character n-gram tokenizer for fuzzy matching.
    Generates overlapping substrings of length n.
    """

    def __init__(self, min_n: int = 2, max_n: int = 3):
        self.min_n = min_n
        self.max_n = max_n

    def tokenize(self, text: str) -> list[Token]:
        tokens = []
        pos = 0
        cleaned = re.sub(r'\s+', ' ', text.strip())
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(cleaned) - n + 1):
                gram = cleaned[i:i + n]
                if gram.strip():
                    tokens.append(Token(
                        text=gram,
                        position=pos,
                        start_char=i,
                        end_char=i + n,
                    ))
                    pos += 1
        return tokens


class SentenceTokenizer:
    """Split text into sentences for snippet extraction."""

    _boundary = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    def tokenize(self, text: str) -> list[str]:
        parts = self._boundary.split(text)
        return [s.strip() for s in parts if s.strip()]
