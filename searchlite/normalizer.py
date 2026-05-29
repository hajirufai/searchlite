"""
Normalizers transform token text into canonical forms.
"""

import unicodedata
from searchlite.tokenizer import Token


class LowercaseNormalizer:
    """Convert all tokens to lowercase."""

    def normalize(self, tokens: list[Token]) -> list[Token]:
        for tok in tokens:
            tok.text = tok.text.lower()
        return tokens


class UnicodeNormalizer:
    """Normalize unicode to NFC form."""

    def __init__(self, form: str = "NFC"):
        self.form = form

    def normalize(self, tokens: list[Token]) -> list[Token]:
        for tok in tokens:
            tok.text = unicodedata.normalize(self.form, tok.text)
        return tokens


class AccentStripper:
    """Remove diacritical marks (é→e, ñ→n, ü→u)."""

    def normalize(self, tokens: list[Token]) -> list[Token]:
        for tok in tokens:
            nfkd = unicodedata.normalize("NFKD", tok.text)
            tok.text = "".join(
                ch for ch in nfkd if unicodedata.category(ch) != "Mn"
            )
        return tokens


class PunctuationStripper:
    """Remove leading/trailing punctuation from tokens."""

    def normalize(self, tokens: list[Token]) -> list[Token]:
        result = []
        for tok in tokens:
            cleaned = tok.text.strip(".,;:!?\"'()[]{}…—–-")
            if cleaned:
                tok.text = cleaned
                result.append(tok)
        return result
