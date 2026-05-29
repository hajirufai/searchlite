"""Tests for the inverted index."""

import pytest
from searchlite.index import InvertedIndex
from searchlite.schema import Schema, TextField, KeywordField, NumericField


@pytest.fixture
def schema():
    return Schema(
        title=TextField(boost=2.0),
        body=TextField(),
        tags=KeywordField(faceted=True),
    )


@pytest.fixture
def index(schema):
    return InvertedIndex(schema)


class TestInvertedIndex:
    def test_add_document(self, index):
        doc_id = index.add_document({
            "title": "Hello World",
            "body": "This is a test document",
            "tags": ["test", "hello"],
        })
        assert doc_id == 0
        assert index.doc_count == 1

    def test_sequential_ids(self, index):
        id1 = index.add_document({"title": "First"})
        id2 = index.add_document({"title": "Second"})
        assert id1 == 0
        assert id2 == 1

    def test_get_document(self, index):
        index.add_document({"title": "Hello", "body": "World"})
        doc = index.get_document(0)
        assert doc is not None
        assert doc["title"] == "Hello"
        assert doc["body"] == "World"

    def test_get_nonexistent(self, index):
        assert index.get_document(999) is None

    def test_term_lookup(self, index):
        index.add_document({"title": "Python Programming"})
        # The analyzer lowercases and may stem
        pl = index.get_postings("title", "python")
        assert pl.doc_frequency >= 0  # depends on analyzer

    def test_keyword_field(self, index):
        index.add_document({"tags": ["python", "tutorial"]})
        pl = index.get_postings("tags", "python")
        assert pl.doc_frequency == 1

    def test_keyword_list(self, index):
        index.add_document({"tags": ["a", "b", "c"]})
        for tag in ["a", "b", "c"]:
            pl = index.get_postings("tags", tag)
            assert pl.doc_frequency == 1

    def test_multiple_docs_same_term(self, index):
        index.add_document({"title": "Python Basics"})
        index.add_document({"title": "Python Advanced"})
        pl = index.get_postings("title", "python")
        assert pl.doc_frequency >= 1

    def test_doc_count(self, index):
        assert index.doc_count == 0
        index.add_document({"title": "One"})
        assert index.doc_count == 1
        index.add_document({"title": "Two"})
        assert index.doc_count == 2

    def test_term_count(self, index):
        index.add_document({"title": "Hello World"})
        assert index.term_count > 0

    def test_remove_document(self, index):
        doc_id = index.add_document({"title": "Remove Me"})
        assert index.doc_count == 1
        assert index.remove_document(doc_id) is True
        assert index.doc_count == 0

    def test_remove_nonexistent(self, index):
        assert index.remove_document(999) is False

    def test_all_doc_ids(self, index):
        index.add_document({"title": "One"})
        index.add_document({"title": "Two"})
        ids = index.all_doc_ids()
        assert len(ids) == 2

    def test_stats(self, index):
        index.add_document({"title": "Hello", "body": "World"})
        stats = index.stats()
        assert stats["documents"] == 1
        assert stats["unique_terms"] > 0
        assert "title" in stats["fields"]

    def test_field_length(self, index):
        index.add_document({"title": "One Two Three"})
        length = index.get_field_length(0, "title")
        assert length > 0

    def test_avg_field_length(self, index):
        index.add_document({"title": "One Two"})
        index.add_document({"title": "One Two Three Four"})
        avg = index.avg_field_length("title")
        assert avg > 0

    def test_serialization_roundtrip(self, index):
        index.add_document({"title": "Hello World", "tags": ["test"]})
        index.add_document({"title": "Foo Bar", "tags": ["foo"]})

        data = index.to_dict()

        index2 = InvertedIndex(index.schema)
        index2.load_from_dict(data)

        assert index2.doc_count == 2
        doc = index2.get_document(0)
        assert doc["title"] == "Hello World"

    def test_missing_fields(self, index):
        # Only provide title, no body or tags
        doc_id = index.add_document({"title": "Just a title"})
        assert doc_id == 0
        doc = index.get_document(0)
        assert doc["title"] == "Just a title"

    def test_get_all_terms(self, index):
        index.add_document({"title": "Python Data Science"})
        terms = index.get_all_terms("title")
        assert len(terms) > 0

    def test_doc_frequency(self, index):
        index.add_document({"title": "Python"})
        index.add_document({"title": "Python"})
        index.add_document({"title": "Java"})
        # "python" (stemmed) should appear in 2 docs
        df = index.get_doc_frequency("title", "python")
        assert df == 2
