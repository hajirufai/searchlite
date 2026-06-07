"""
Porter Stemmer — implemented from scratch following Martin Porter's algorithm.
Ref: https://tartarus.org/martin/PorterStemmer/def.txt

No NLTK, no external libraries. Pure Python implementation of all five steps.
"""

import re
from searchlite.tokenizer import Token


class PorterStemmer:
    """
    The Porter stemming algorithm, implemented from the original paper.
    Reduces words to their morphological root.
    """

    # Vowels and consonant helpers
    _vowels = frozenset("aeiou")

    def _is_consonant(self, word: str, i: int) -> bool:
        ch = word[i]
        if ch in self._vowels:
            return False
        if ch == "y":
            if i == 0:
                return True
            return not self._is_consonant(word, i - 1)
        return True

    def _measure(self, stem: str) -> int:
        """
        Calculate m — the 'measure' of a stem.
        m = number of VC sequences in the stem.
        [C](VC){m}[V]
        """
        cv = ""
        for i in range(len(stem)):
            cv += "C" if self._is_consonant(stem, i) else "V"

        # Count VC transitions
        m = 0
        i = 0
        while i < len(cv) and cv[i] == "C":
            i += 1
        while i < len(cv):
            if cv[i] == "V":
                i += 1
                while i < len(cv) and cv[i] == "V":
                    i += 1
                if i < len(cv):
                    m += 1
                    i += 1
                    while i < len(cv) and cv[i] == "C":
                        i += 1
                else:
                    break
            else:
                break
        return m

    def _has_vowel(self, stem: str) -> bool:
        for i in range(len(stem)):
            if not self._is_consonant(stem, i):
                return True
        return False

    def _ends_double_consonant(self, word: str) -> bool:
        if len(word) < 2:
            return False
        return (word[-1] == word[-2] and self._is_consonant(word, len(word) - 1))

    def _cvc(self, word: str) -> bool:
        """Check if word ends with consonant-vowel-consonant where last C is not w, x, y."""
        if len(word) < 3:
            return False
        if (self._is_consonant(word, len(word) - 1) and
            not self._is_consonant(word, len(word) - 2) and
            self._is_consonant(word, len(word) - 3)):
            return word[-1] not in ("w", "x", "y")
        return False

    def _replace_suffix(self, word: str, suffix: str, replacement: str) -> str:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            return stem + replacement
        return word

    def _step1a(self, word: str) -> str:
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-2]
        if word.endswith("ss"):
            return word
        if word.endswith("s"):
            return word[:-1]
        return word

    def _step1b(self, word: str) -> str:
        if word.endswith("eed"):
            stem = word[:-3]
            if self._measure(stem) > 0:
                return word[:-1]
            return word

        changed = False
        if word.endswith("ed"):
            stem = word[:-2]
            if self._has_vowel(stem):
                word = stem
                changed = True
        elif word.endswith("ing"):
            stem = word[:-3]
            if self._has_vowel(stem):
                word = stem
                changed = True

        if changed:
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                word += "e"
            elif self._ends_double_consonant(word) and word[-1] not in ("l", "s", "z"):
                word = word[:-1]
            elif self._measure(word) == 1 and self._cvc(word):
                word += "e"

        return word

    def _step1c(self, word: str) -> str:
        if word.endswith("y") and self._has_vowel(word[:-1]):
            return word[:-1] + "i"
        return word

    def _step2(self, word: str) -> str:
        mappings = [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
            ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
            ("alli", "al"), ("entli", "ent"), ("eli", "e"),
            ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
            ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
            ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
            ("iviti", "ive"), ("biliti", "ble"),
        ]
        for suffix, replacement in mappings:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 0:
                    return stem + replacement
                return word
        return word

    def _step3(self, word: str) -> str:
        mappings = [
            ("icate", "ic"), ("ative", ""), ("alize", "al"),
            ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""),
        ]
        for suffix, replacement in mappings:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 0:
                    return stem + replacement
                return word
        return word

    def _step4(self, word: str) -> str:
        suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible",
            "ant", "ement", "ment", "ent", "ion", "ou", "ism",
            "ate", "iti", "ous", "ive", "ize",
        ]
        for suffix in suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if suffix == "ion":
                    if stem and stem[-1] in ("s", "t"):
                        if self._measure(stem) > 1:
                            return stem
                else:
                    if self._measure(stem) > 1:
                        return stem
                return word
        return word

    def _step5a(self, word: str) -> str:
        if word.endswith("e"):
            stem = word[:-1]
            if self._measure(stem) > 1:
                return stem
            if self._measure(stem) == 1 and not self._cvc(stem):
                return stem
        return word

    def _step5b(self, word: str) -> str:
        if self._measure(word) > 1 and self._ends_double_consonant(word) and word[-1] == "l":
            return word[:-1]
        return word

    def stem_word(self, word: str) -> str:
        """Stem a single word using the Porter algorithm."""
        if len(word) <= 2:
            return word

        word = word.lower()

        word = self._step1a(word)
        word = self._step1b(word)
        word = self._step1c(word)
        word = self._step2(word)
        word = self._step3(word)
        word = self._step4(word)
        word = self._step5a(word)
        word = self._step5b(word)

        return word

    def stem_tokens(self, tokens: list[Token]) -> list[Token]:
        """Stem all tokens in a list."""
        for tok in tokens:
            tok.text = self.stem_word(tok.text)
        return tokens
