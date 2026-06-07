"""Tests for search execution."""

import pytest
from searchlite.index import InvertedIndex
from searchlite.schema import Schema, TextField, KeywordField
from searchlite.searcher import Searcher, SearchResults
from searchlite.scorer import BM25Scorer
from searchlite.query import TermQuery, PhraseQuery, BoolQuery, WildcardQuery, MatchAllQuery


@pytest.fixture
def search_env():
    schema = Schema(
        title=TextField(boost=2.0),
        body=TextField(),
        tags=KeywordField(faceted=True),
        author=KeywordField(),
    )
    index = InvertedIndex(schema)

    # Add test documents
    index.add_document({
        "title": "Introduction to Python Programming",
        "body": "Python is a versatile programming language used in data science and machine learning",
        "tags": ["python", "programming"],
        "author": "alice",
    })
    index.add_document({
        "title": "Data Engineering with Apache Kafka",
        "body": "Apache Kafka handles real-time data feeds with high throughput and low latency",
        "tags": ["data-engineering", "kafka"],
        "author": "bob",
    })
    index.add_document({
        "title": "Machine Learning Fundamentals",
        "body": "Machine learning is a subset of artificial intelligence that provides systems the ability to learn",
        "tags": ["machine-learning", "ai"],
        "author": "alice",
    })
    index.add_document({
        "title": "SQL Query Optimization",
        "body": "Writing efficient SQL queries is crucial for database performance and data pipelines",
        "tags": ["sql", "database"],
        "author": "charlie",
    })
    index.add_document({
        "title": "Python for Data Science",
        "body": "Python is the most popular language for data science with libraries like pandas and numpy",
        "tags": ["python", "data-science"],
        "author": "alice",
    })

    searcher = Searcher(index)
    return searcher, index


class TestSearcher:
    def test_term_search(self, search_env):
        searcher, _ = search_env
        results = searcher.search(TermQuery(term="python"))
        assert results.total >= 1
        assert len(results.hits) >= 1

    def test_relevance_ordering(self, search_env):
        searcher, _ = search_env
        results = searcher.search(TermQuery(term="python"))
        if len(results.hits) >= 2:
            scores = [h.score for h in results.hits]
            assert scores == sorted(scores, reverse=True)

    def test_phrase_search(self, search_env):
        searcher, _ = search_env
        results = searcher.search(PhraseQuery(terms=["machine", "learning"]))
        assert results.total >= 1

    def test_bool_and(self, search_env):
        searcher, _ = search_env
        results = searcher.search(BoolQuery(
            must=[TermQuery(term="python"), TermQuery(term="data")]
        ))
        assert results.total >= 1
        # All results should contain both terms
        for hit in results:
            text = (hit.doc.get("title", "") + " " + hit.doc.get("body", "")).lower()
            assert "python" in text

    def test_bool_or(self, search_env):
        searcher, _ = search_env
        results = searcher.search(BoolQuery(
            should=[TermQuery(term="kafka"), TermQuery(term="sql")]
        ))
        assert results.total >= 2

    def test_bool_not(self, search_env):
        searcher, _ = search_env
        all_results = searcher.search(TermQuery(term="python"))
        not_results = searcher.search(BoolQuery(
            must=[TermQuery(term="python")],
            must_not=[TermQuery(term="science")]
        ))
        assert not_results.total <= all_results.total

    def test_wildcard_search(self, search_env):
        searcher, _ = search_env
        results = searcher.search(WildcardQuery(pattern="pyth*"))
        assert results.total >= 1

    def test_match_all(self, search_env):
        searcher, index = search_env
        results = searcher.search(MatchAllQuery())
        assert results.total == index.doc_count

    def test_field_specific(self, search_env):
        searcher, _ = search_env
        results = searcher.search(TermQuery(term="python", field="title"))
        assert results.total >= 1

    def test_pagination(self, search_env):
        searcher, _ = search_env
        page1 = searcher.search(MatchAllQuery(), limit=2, offset=0)
        page2 = searcher.search(MatchAllQuery(), limit=2, offset=2)
        assert len(page1.hits) == 2
        # Pages shouldn't overlap
        ids1 = {h.doc_id for h in page1.hits}
        ids2 = {h.doc_id for h in page2.hits}
        assert ids1.isdisjoint(ids2)

    def test_facets(self, search_env):
        searcher, _ = search_env
        results = searcher.search(MatchAllQuery(), facet_fields=["tags"])
        assert "tags" in results.facets
        assert len(results.facets["tags"]) > 0

    def test_facet_filter(self, search_env):
        from searchlite.facets import FacetFilter
        searcher, _ = search_env
        ff = FacetFilter()
        ff.add_filter("tags", ["python"])
        results = searcher.search(MatchAllQuery(), facet_filter=ff)
        for hit in results:
            assert "python" in hit.doc.get("tags", [])

    def test_no_results(self, search_env):
        searcher, _ = search_env
        results = searcher.search(TermQuery(term="xyznonexistent"))
        assert results.total == 0
        assert len(results.hits) == 0

    def test_highlight(self, search_env):
        searcher, _ = search_env
        results = searcher.search(TermQuery(term="python"))
        if results.hits:
            snippet = results.hits[0].highlight("body")
            assert isinstance(snippet, str)
            assert len(snippet) > 0

    def test_query_time(self, search_env):
        searcher, _ = search_env
        results = searcher.search(TermQuery(term="python"))
        assert results.query_time_ms >= 0

    def test_boost(self, search_env):
        searcher, _ = search_env
        results_normal = searcher.search(TermQuery(term="python"))
        results_boosted = searcher.search(TermQuery(term="python", boost=5.0))
        if results_normal.hits and results_boosted.hits:
            assert results_boosted.hits[0].score >= results_normal.hits[0].score

    def test_title_boost(self, search_env):
        """Documents with matches in title (boosted field) should rank higher."""
        searcher, _ = search_env
        results = searcher.search(TermQuery(term="python"))
        if len(results.hits) >= 2:
            # First result should have "Python" in title
            top_title = results.hits[0].doc.get("title", "").lower()
            assert "python" in top_title

    def test_search_results_repr(self, search_env):
        searcher, _ = search_env
        results = searcher.search(TermQuery(term="python"))
        assert "total=" in repr(results)

    def test_search_results_iterable(self, search_env):
        searcher, _ = search_env
        results = searcher.search(TermQuery(term="python"))
        count = sum(1 for _ in results)
        assert count == len(results)
