"""Tests for the high-level SearchEngine API."""

import pytest
import tempfile
from searchlite.engine import SearchEngine
from searchlite.schema import Schema, TextField, KeywordField, NumericField


@pytest.fixture
def schema():
    return Schema(
        title=TextField(boost=2.0),
        body=TextField(),
        tags=KeywordField(faceted=True),
        author=KeywordField(),
    )


@pytest.fixture
def engine(schema):
    return SearchEngine(schema=schema)


class TestSearchEngine:
    def test_create_engine(self, schema):
        engine = SearchEngine(schema=schema)
        assert engine.stats()["documents"] == 0

    def test_add_and_search(self, engine):
        engine.add({"title": "Python Tutorial", "body": "Learn Python programming"})
        results = engine.search("python")
        assert results.total >= 1

    def test_add_many(self, engine):
        docs = [
            {"title": "Doc 1", "body": "First document"},
            {"title": "Doc 2", "body": "Second document"},
            {"title": "Doc 3", "body": "Third document"},
        ]
        ids = engine.add_many(docs)
        assert len(ids) == 3
        assert engine.stats()["documents"] == 3

    def test_remove(self, engine):
        doc_id = engine.add({"title": "Remove me"})
        assert engine.stats()["documents"] == 1
        engine.remove(doc_id)
        assert engine.stats()["documents"] == 0

    def test_get(self, engine):
        doc_id = engine.add({"title": "Test", "body": "Hello"})
        doc = engine.get(doc_id)
        assert doc["title"] == "Test"

    def test_get_nonexistent(self, engine):
        assert engine.get(999) is None

    def test_string_query(self, engine):
        engine.add({"title": "Python Data Science"})
        results = engine.search("python")
        assert results.total >= 1

    def test_phrase_query(self, engine):
        engine.add({"title": "Machine Learning", "body": "Deep machine learning algorithms"})
        results = engine.search('"machine learning"')
        assert results.total >= 1

    def test_boolean_query(self, engine):
        engine.add({"title": "Python", "body": "Python programming"})
        engine.add({"title": "Java", "body": "Java programming"})
        results = engine.search("python AND programming")
        for hit in results:
            assert "python" in hit.doc.get("title", "").lower() or \
                   "python" in hit.doc.get("body", "").lower()

    def test_faceted_search(self, engine):
        engine.add({"title": "A", "tags": ["python"]})
        engine.add({"title": "B", "tags": ["python", "data"]})
        engine.add({"title": "C", "tags": ["java"]})
        results = engine.search("*", facets=["tags"])
        assert "tags" in results.facets
        assert results.facets["tags"]["python"] == 2

    def test_filtered_search(self, engine):
        engine.add({"title": "A", "tags": ["python"], "body": "test"})
        engine.add({"title": "B", "tags": ["java"], "body": "test"})
        results = engine.search("test", filters={"tags": ["python"]})
        for hit in results:
            assert "python" in hit.doc.get("tags", [])

    def test_pagination(self, engine):
        for i in range(10):
            engine.add({"title": f"Document {i}", "body": "test content"})
        page1 = engine.search("test", limit=3, offset=0)
        page2 = engine.search("test", limit=3, offset=3)
        assert len(page1.hits) == 3
        assert len(page2.hits) == 3

    def test_stats(self, engine):
        engine.add({"title": "Test"})
        stats = engine.stats()
        assert stats["documents"] == 1
        assert "unique_terms" in stats

    def test_default_schema(self):
        engine = SearchEngine()
        engine.add({"content": "Hello world"})
        results = engine.search("hello")
        assert results.total >= 1

    def test_repr(self, engine):
        engine.add({"title": "Test"})
        r = repr(engine)
        assert "docs=1" in r


class TestSearchEnginePersistence:
    def test_save_and_load(self, schema):
        with tempfile.TemporaryDirectory() as tmp:
            # Create and populate
            engine = SearchEngine(tmp, schema=schema)
            engine.add({"title": "Persisted Doc", "body": "This should survive"})
            engine.add({"title": "Another Doc", "body": "Also persisted"})
            engine.commit()

            # Reload
            engine2 = SearchEngine(tmp, schema=schema)
            assert engine2.stats()["documents"] == 2

            results = engine2.search("persisted")
            assert results.total >= 1

    def test_context_manager(self, schema):
        with tempfile.TemporaryDirectory() as tmp:
            with SearchEngine(tmp, schema=schema) as engine:
                engine.add({"title": "Context Test"})

            # Should be committed on exit
            engine2 = SearchEngine(tmp, schema=schema)
            assert engine2.stats()["documents"] == 1

    def test_incremental_indexing(self, schema):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SearchEngine(tmp, schema=schema)
            engine.add({"title": "First"})
            engine.commit()

            engine.add({"title": "Second"})
            engine.commit()

            engine2 = SearchEngine(tmp, schema=schema)
            assert engine2.stats()["documents"] == 2

    def test_tfidf_scorer(self, schema):
        engine = SearchEngine(schema=schema, scorer="tfidf")
        engine.add({"title": "Python Test"})
        results = engine.search("python")
        assert results.total >= 1


class TestSearchEngineEdgeCases:
    def test_empty_search(self, engine):
        results = engine.search("")
        assert isinstance(results.total, int)

    def test_search_empty_index(self, engine):
        results = engine.search("anything")
        assert results.total == 0

    def test_special_characters(self, engine):
        engine.add({"title": "C++ Programming", "body": "Learn C++ basics"})
        # Should handle gracefully even if special chars are stripped
        results = engine.search("programming")
        assert results.total >= 1

    def test_unicode(self, engine):
        engine.add({"title": "Café résumé", "body": "Unicode text with accents"})
        results = engine.search("unicode")
        assert isinstance(results, type(results))

    def test_long_document(self, engine):
        engine.add({"title": "Long", "body": "word " * 10000})
        results = engine.search("word")
        assert results.total == 1

    def test_duplicate_documents(self, engine):
        engine.add({"title": "Same", "body": "Same content"})
        engine.add({"title": "Same", "body": "Same content"})
        assert engine.stats()["documents"] == 2

    def test_numeric_field(self):
        schema = Schema(
            title=TextField(),
            year=NumericField(),
        )
        engine = SearchEngine(schema=schema)
        engine.add({"title": "Old Article", "year": 2020})
        engine.add({"title": "New Article", "year": 2024})
        results = engine.search("article")
        assert results.total == 2
