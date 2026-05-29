"""
Inverted index — maps terms to posting lists with document positions.
The heart of the search engine.
"""

from collections import defaultdict
from searchlite.posting import Posting, PostingList
from searchlite.document import Document
from searchlite.schema import Schema, TextField, KeywordField, NumericField
from searchlite.analyzer import Analyzer


class InvertedIndex:
    """
    In-memory inverted index. Maps (field, term) → PostingList.
    Also stores document data, field lengths, and index statistics.
    """

    def __init__(self, schema: Schema):
        self.schema = schema
        self._index: dict[str, dict[str, PostingList]] = defaultdict(
            lambda: defaultdict(PostingList)
        )
        self._documents: dict[int, Document] = {}
        self._doc_field_lengths: dict[int, dict[str, int]] = {}
        self._next_doc_id: int = 0
        self._total_field_lengths: dict[str, int] = defaultdict(int)

    @property
    def doc_count(self) -> int:
        return len(self._documents)

    @property
    def term_count(self) -> int:
        count = 0
        for field_terms in self._index.values():
            count += len(field_terms)
        return count

    def avg_field_length(self, field_name: str) -> float:
        if self.doc_count == 0:
            return 0.0
        return self._total_field_lengths.get(field_name, 0) / self.doc_count

    def add_document(self, fields: dict) -> int:
        """
        Index a document. Analyzes each field and builds posting lists.
        Returns the assigned document ID.
        """
        doc_id = self._next_doc_id
        self._next_doc_id += 1

        doc = Document(doc_id=doc_id, fields=fields)
        self._documents[doc_id] = doc
        self._doc_field_lengths[doc_id] = {}

        for field_name, field_type in self.schema.fields.items():
            value = fields.get(field_name)
            if value is None:
                continue

            if isinstance(field_type, NumericField):
                # Numeric fields: store value directly, no text analysis
                self._index[field_name]["__numeric__"] = self._index[field_name].get(
                    "__numeric__", PostingList()
                )
                continue

            if isinstance(field_type, KeywordField):
                # Keyword fields: handle lists (tags) or single values
                values = value if isinstance(value, list) else [value]
                for val in values:
                    term = str(val).lower().strip()
                    if term:
                        posting = Posting(
                            doc_id=doc_id, term_freq=1,
                            positions=[0], field_name=field_name
                        )
                        self._index[field_name][term].add(posting)
                self._doc_field_lengths[doc_id][field_name] = len(values)
                self._total_field_lengths[field_name] += len(values)
                continue

            if isinstance(field_type, TextField):
                analyzer = field_type.get_analyzer()
                if analyzer is None:
                    continue

                tokens = analyzer.analyze(str(value))
                self._doc_field_lengths[doc_id][field_name] = len(tokens)
                self._total_field_lengths[field_name] += len(tokens)

                # Build term → positions map for this doc+field
                term_positions: dict[str, list[int]] = defaultdict(list)
                term_freqs: dict[str, int] = defaultdict(int)

                for token in tokens:
                    term_positions[token.text].append(token.position)
                    term_freqs[token.text] += 1

                for term, positions in term_positions.items():
                    posting = Posting(
                        doc_id=doc_id,
                        term_freq=term_freqs[term],
                        positions=positions,
                        field_name=field_name,
                    )
                    self._index[field_name][term].add(posting)

        return doc_id

    def get_postings(self, field_name: str, term: str) -> PostingList:
        """Get the posting list for a term in a field."""
        return self._index.get(field_name, {}).get(term, PostingList())

    def get_document(self, doc_id: int) -> Document | None:
        """Retrieve a document by ID."""
        return self._documents.get(doc_id)

    def get_field_length(self, doc_id: int, field_name: str) -> int:
        """Get the token count for a field in a document."""
        return self._doc_field_lengths.get(doc_id, {}).get(field_name, 0)

    def get_all_terms(self, field_name: str) -> list[str]:
        """Get all indexed terms for a field."""
        return list(self._index.get(field_name, {}).keys())

    def get_doc_frequency(self, field_name: str, term: str) -> int:
        """How many documents contain this term in this field."""
        pl = self.get_postings(field_name, term)
        return pl.doc_frequency

    def all_doc_ids(self) -> set[int]:
        """Set of all document IDs in the index."""
        return set(self._documents.keys())

    def remove_document(self, doc_id: int) -> bool:
        """Remove a document from the index."""
        if doc_id not in self._documents:
            return False

        # Remove from posting lists
        for field_name in list(self._index.keys()):
            for term in list(self._index[field_name].keys()):
                pl = self._index[field_name][term]
                pl._postings = [p for p in pl._postings if p.doc_id != doc_id]
                if not pl._postings:
                    del self._index[field_name][term]

        # Update field length totals
        for field_name, length in self._doc_field_lengths.get(doc_id, {}).items():
            self._total_field_lengths[field_name] -= length

        del self._documents[doc_id]
        if doc_id in self._doc_field_lengths:
            del self._doc_field_lengths[doc_id]

        return True

    def stats(self) -> dict:
        """Index statistics."""
        return {
            "documents": self.doc_count,
            "unique_terms": self.term_count,
            "fields": list(self.schema.fields.keys()),
            "avg_field_lengths": {
                f: round(self.avg_field_length(f), 1)
                for f in self.schema.get_text_fields()
            },
        }

    def to_dict(self) -> dict:
        """Serialize the index for persistence."""
        index_data = {}
        for field_name, terms in self._index.items():
            index_data[field_name] = {}
            for term, pl in terms.items():
                index_data[field_name][term] = pl.to_list()

        docs_data = {
            str(doc_id): doc.to_dict()
            for doc_id, doc in self._documents.items()
        }

        return {
            "index": index_data,
            "documents": docs_data,
            "field_lengths": {
                str(k): v for k, v in self._doc_field_lengths.items()
            },
            "total_field_lengths": dict(self._total_field_lengths),
            "next_doc_id": self._next_doc_id,
        }

    def load_from_dict(self, data: dict):
        """Restore the index from serialized data."""
        self._next_doc_id = data.get("next_doc_id", 0)

        for field_name, terms in data.get("index", {}).items():
            for term, postings_data in terms.items():
                self._index[field_name][term] = PostingList.from_list(postings_data)

        for doc_id_str, doc_data in data.get("documents", {}).items():
            doc = Document.from_dict(doc_data)
            self._documents[doc.doc_id] = doc

        for doc_id_str, lengths in data.get("field_lengths", {}).items():
            self._doc_field_lengths[int(doc_id_str)] = lengths

        self._total_field_lengths = defaultdict(int, data.get("total_field_lengths", {}))
