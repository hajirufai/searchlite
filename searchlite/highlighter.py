"""
Result highlighting — extract relevant snippets and wrap matched terms.
"""

import re


class Highlighter:
    """
    Extracts and highlights search result snippets.
    Finds the best passage containing query terms and wraps matches in tags.
    """

    def __init__(self, pre_tag: str = "<b>", post_tag: str = "</b>",
                 fragment_size: int = 150, max_fragments: int = 3,
                 separator: str = " ... "):
        self.pre_tag = pre_tag
        self.post_tag = post_tag
        self.fragment_size = fragment_size
        self.max_fragments = max_fragments
        self.separator = separator

    def highlight(self, text: str, query_terms: list[str],
                  fragment_size: int | None = None) -> str:
        """
        Extract highlighted snippets from text.
        Returns fragments with matched terms wrapped in pre/post tags.
        """
        if not text or not query_terms:
            return text[:self.fragment_size] if text else ""

        frag_size = fragment_size or self.fragment_size
        terms_lower = [t.lower() for t in query_terms if t]

        # Find all term positions in the text
        positions = []
        text_lower = text.lower()
        for term in terms_lower:
            start = 0
            while True:
                idx = text_lower.find(term, start)
                if idx == -1:
                    break
                positions.append((idx, idx + len(term), term))
                start = idx + 1

        if not positions:
            # No matches found — return start of text
            return text[:frag_size] + ("..." if len(text) > frag_size else "")

        positions.sort(key=lambda p: p[0])

        # Score windows by term density
        fragments = self._best_fragments(text, positions, frag_size)

        # Take top fragments
        fragments = fragments[:self.max_fragments]
        fragments.sort(key=lambda f: f[0])  # sort by position

        # Build highlighted snippets
        snippets = []
        for frag_start, frag_end, frag_positions in fragments:
            snippet = self._highlight_fragment(
                text, frag_start, frag_end, frag_positions
            )
            snippets.append(snippet)

        return self.separator.join(snippets)

    def _best_fragments(self, text: str, positions: list[tuple],
                        frag_size: int) -> list[tuple]:
        """Find the best text fragments containing the most query terms."""
        if not positions:
            return [(0, min(frag_size, len(text)), [])]

        scored = []
        used_ranges = []

        for anchor_start, anchor_end, _ in positions:
            # Center fragment around this match
            half = frag_size // 2
            frag_start = max(0, anchor_start - half)
            frag_end = min(len(text), frag_start + frag_size)

            # Adjust to word boundaries
            frag_start = self._snap_to_word(text, frag_start, direction="left")
            frag_end = self._snap_to_word(text, frag_end, direction="right")

            # Check overlap with already-selected fragments
            overlaps = False
            for us, ue in used_ranges:
                if frag_start < ue and frag_end > us:
                    overlaps = True
                    break
            if overlaps:
                continue

            # Count terms in this fragment
            frag_positions = [
                (s, e, t) for s, e, t in positions
                if s >= frag_start and e <= frag_end
            ]
            score = len(frag_positions)

            # Bonus for unique terms
            unique = len(set(t for _, _, t in frag_positions))
            score += unique * 2

            scored.append((score, frag_start, frag_end, frag_positions))
            used_ranges.append((frag_start, frag_end))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(s, e, p) for _, s, e, p in scored]

    def _highlight_fragment(self, text: str, frag_start: int,
                            frag_end: int, positions: list[tuple]) -> str:
        """Build a highlighted fragment string."""
        fragment = ""
        last = frag_start

        for match_start, match_end, _ in sorted(positions, key=lambda p: p[0]):
            if match_start < last:
                continue
            # Text before match
            fragment += text[last:match_start]
            # Highlighted match
            fragment += self.pre_tag + text[match_start:match_end] + self.post_tag
            last = match_end

        # Remaining text
        fragment += text[last:frag_end]

        # Add ellipsis
        prefix = "..." if frag_start > 0 else ""
        suffix = "..." if frag_end < len(text) else ""

        return prefix + fragment.strip() + suffix

    def _snap_to_word(self, text: str, pos: int, direction: str) -> int:
        """Snap a position to the nearest word boundary."""
        if direction == "left":
            while pos > 0 and text[pos - 1] not in " \n\t":
                pos -= 1
        else:
            while pos < len(text) and text[pos - 1] not in " \n\t":
                pos += 1
        return pos
