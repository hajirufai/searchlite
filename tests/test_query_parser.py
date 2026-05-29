"""Tests for the query parser."""

import pytest
from searchlite.query_parser import QueryParser, QueryParseError
from searchlite.query import (
    TermQuery, PhraseQuery, BoolQuery, WildcardQuery, MatchAllQuery,
)


class TestQueryParser:
    def setup_method(self):
        self.parser = QueryParser()

    def test_single_term(self):
        q = self.parser.parse("python")
        assert isinstance(q, TermQuery)
        assert q.term == "python"

    def test_two_terms_implicit_and(self):
        q = self.parser.parse("python data")
        assert isinstance(q, BoolQuery)
        assert len(q.must) == 2

    def test_explicit_and(self):
        q = self.parser.parse("python AND data")
        assert isinstance(q, BoolQuery)
        assert len(q.must) == 2

    def test_or(self):
        q = self.parser.parse("python OR java")
        assert isinstance(q, BoolQuery)
        assert len(q.should) == 2

    def test_not(self):
        q = self.parser.parse("NOT python")
        assert isinstance(q, BoolQuery)
        assert len(q.must_not) == 1

    def test_phrase(self):
        q = self.parser.parse('"machine learning"')
        assert isinstance(q, PhraseQuery)
        assert q.terms == ["machine", "learning"]

    def test_field_term(self):
        q = self.parser.parse("title:python")
        assert isinstance(q, TermQuery)
        assert q.field == "title"
        assert q.term == "python"

    def test_field_phrase(self):
        q = self.parser.parse('title:"machine learning"')
        assert isinstance(q, PhraseQuery)
        assert q.field == "title"
        assert q.terms == ["machine", "learning"]

    def test_wildcard(self):
        q = self.parser.parse("pyth*")
        assert isinstance(q, WildcardQuery)
        assert "pyth" in q.pattern

    def test_field_wildcard(self):
        q = self.parser.parse("title:pyth*")
        assert isinstance(q, WildcardQuery)
        assert q.field == "title"

    def test_grouping(self):
        q = self.parser.parse("(python OR java) AND data")
        assert isinstance(q, BoolQuery)
        assert len(q.must) == 2

    def test_complex(self):
        q = self.parser.parse('title:python AND body:"data pipeline" OR kafka')
        assert isinstance(q, (BoolQuery,))

    def test_empty_query(self):
        q = self.parser.parse("")
        assert isinstance(q, MatchAllQuery)

    def test_star_query(self):
        q = self.parser.parse("*")
        assert isinstance(q, MatchAllQuery)

    def test_boost(self):
        q = self.parser.parse("python^2.0")
        assert isinstance(q, TermQuery)
        assert q.boost == 2.0

    def test_three_terms(self):
        q = self.parser.parse("python data engineering")
        assert isinstance(q, BoolQuery)
        # Should have 3 must clauses (implicit AND)
        assert len(q.must) == 3

    def test_mixed_and_or(self):
        q = self.parser.parse("python OR java AND data")
        # AND binds tighter than OR
        assert isinstance(q, (BoolQuery,))

    def test_multiple_or(self):
        q = self.parser.parse("python OR java OR rust")
        assert isinstance(q, BoolQuery)
        assert len(q.should) == 3

    def test_nested_groups(self):
        q = self.parser.parse("(python OR java) AND (data OR ml)")
        assert isinstance(q, BoolQuery)

    def test_case_preservation_in_terms(self):
        q = self.parser.parse("Python")
        assert isinstance(q, TermQuery)
        assert q.term == "python"  # lowercased

    def test_not_with_and(self):
        q = self.parser.parse("python AND NOT java")
        assert isinstance(q, BoolQuery)
