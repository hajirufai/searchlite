"""Tests for result highlighting."""

import pytest
from searchlite.highlighter import Highlighter


class TestHighlighter:
    def setup_method(self):
        self.hl = Highlighter(pre_tag="<b>", post_tag="</b>")

    def test_basic_highlight(self):
        text = "Python is a great programming language"
        result = self.hl.highlight(text, ["python"])
        assert "<b>" in result
        assert "Python" in result or "python" in result

    def test_multiple_terms(self):
        text = "Python is used for data science and machine learning"
        result = self.hl.highlight(text, ["python", "data"])
        assert "<b>" in result

    def test_no_matches(self):
        text = "Hello world"
        result = self.hl.highlight(text, ["xyz"])
        # Should return truncated text without highlights
        assert "Hello" in result
        assert "<b>" not in result

    def test_empty_text(self):
        result = self.hl.highlight("", ["python"])
        assert result == ""

    def test_empty_terms(self):
        text = "Hello world"
        result = self.hl.highlight(text, [])
        assert "Hello" in result

    def test_custom_tags(self):
        hl = Highlighter(pre_tag="[", post_tag="]")
        text = "Python is great"
        result = hl.highlight(text, ["python"])
        assert "[" in result

    def test_fragment_size(self):
        text = "A " * 500 + "Python is here" + " B" * 500
        result = self.hl.highlight(text, ["python"], fragment_size=50)
        assert len(result) < len(text)

    def test_multiple_fragments(self):
        hl = Highlighter(max_fragments=2, fragment_size=30)
        text = "Python at the start. " + "x " * 100 + "Python at the end."
        result = hl.highlight(text, ["python"])
        # Should find matches in both locations
        assert "..." in result or "Python" in result
