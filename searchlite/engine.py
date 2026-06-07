"""
SearchEngine — the high-level API that ties everything together.
Index documents, run queries, persist to disk.
"""

import time
from pathlib import Path

from searchlite.index import InvertedIndex
from searchlite.schema import Schema, TextField, KeywordField
from searchlite.searcher import Searcher, SearchResults
from searchlite.scorer import BM25Scorer, TFIDFScorer, Scorer
from searchlite.highlighter import Highlighter
from searchlite.query_parser import QueryParser
from searchlite.query import Query
from searchlite.storage import IndexStorage
from searchlite.facets import FacetFilter
from searchlite.utils import Timer, format_size


class SearchEngine:
    """
    Full-text search engine with indexing, querying, and persistence.

    Usage:
        engine = SearchEngine("./my_index", schema=Schema(
            title=TextField(boost=2.0),
            body=TextField(),
            tags=KeywordField(faceted=True),
        ))

        engine.add({"title": "Hello", "body": "World", "tags": ["greeting"]})
        engine.commit()

        results = engine.search("hello")
        for hit in results:
            print(hit.score, hit.doc["title"])
    """

    def __init__(self, path: str | Path | None = None,
                 schema: Schema | None = None,
                 scorer: str | Scorer = "bm25"):
        """
        Create or open a search engine.

        Args:
            path: Directory for persistent storage. None for in-memory only.
            schema: Document schema defining fields and their types.
            scorer: "bm25", "tfidf", or a custom Scorer instance.
        """
        if schema is None:
            schema = Schema(content=TextField())

        self.schema = schema
        self._index = InvertedIndex(schema)
        self._storage = IndexStorage(path) if path else None
        self._parser = QueryParser(default_fields=schema.get_searchable_fields())
        self._highlighter = Highlighter()

        if isinstance(scorer, str):
            self._scorer = BM25Scorer() if scorer == "bm25" else TFIDFScorer()
        else:
            self._scorer = scorer

        self._searcher = Searcher(self._index, self._scorer, self._highlighter)
        self._dirty = False
        self._doc_count_at_last_commit = 0

        # Try loading existing index
        if self._storage and self._storage.exists():
            self._load()

    def add(self, document: dict) -> int:
        """
        Add a document to the index.
        Returns the assigned document ID.
        """
        doc_id = self._index.add_document(document)
        self._dirty = True
        return doc_id

    def add_many(self, documents: list[dict]) -> list[int]:
        """Add multiple documents at once. Returns list of doc IDs."""
        ids = []
        for doc in documents:
            ids.append(self.add(doc))
        return ids

    def remove(self, doc_id: int) -> bool:
        """Remove a document by ID."""
        result = self._index.remove_document(doc_id)
        if result:
            self._dirty = True
        return result

    def search(self, query: str | Query, limit: int = 10, offset: int = 0,
               facets: list[str] | None = None,
               filters: dict[str, list[str]] | None = None) -> SearchResults:
        """
        Search the index.

        Args:
            query: Query string or Query object
            limit: Max results to return
            offset: Skip first N results (for pagination)
            facets: Field names to collect facet counts for
            filters: {field: [values]} to filter results

        Returns:
            SearchResults with hits, total count, timing, and facets
        """
        if isinstance(query, str):
            parsed = self._parser.parse(query)
        else:
            parsed = query

        facet_filter = None
        if filters:
            facet_filter = FacetFilter()
            for field_name, values in filters.items():
                facet_filter.add_filter(field_name, values)

        return self._searcher.search(
            parsed, limit=limit, offset=offset,
            facet_fields=facets, facet_filter=facet_filter,
        )

    def get(self, doc_id: int) -> dict | None:
        """Retrieve a document by ID."""
        doc = self._index.get_document(doc_id)
        return doc.fields if doc else None

    def commit(self):
        """Persist the current index to disk."""
        if self._storage and self._dirty:
            data = self._index.to_dict()
            self._storage.compact(data)
            self._dirty = False
            self._doc_count_at_last_commit = self._index.doc_count

    def close(self):
        """Commit and close the engine."""
        self.commit()

    def stats(self) -> dict:
        """Index statistics."""
        base = self._index.stats()
        if self._storage:
            base["index_size"] = self._storage.index_size_human()
            base["path"] = str(self._storage.path)
        base["dirty"] = self._dirty
        return base

    def _load(self):
        """Load index from disk."""
        data = self._storage.load_latest()
        if data:
            self._index.load_from_dict(data)
            self._doc_count_at_last_commit = self._index.doc_count

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return (
            f"SearchEngine(docs={self._index.doc_count}, "
            f"terms={self._index.term_count})"
        )
