"""
Search execution — takes a Query AST and runs it against the inverted index
to produce ranked results.
"""

import heapq
from dataclasses import dataclass, field
from searchlite.index import InvertedIndex
from searchlite.query import (
    Query, TermQuery, PhraseQuery, BoolQuery,
    WildcardQuery, RangeQuery, MatchAllQuery,
)
from searchlite.scorer import BM25Scorer, Scorer
from searchlite.highlighter import Highlighter
from searchlite.facets import FacetCollector, FacetFilter
from searchlite.schema import TextField, KeywordField


@dataclass
class SearchHit:
    """A single search result."""
    doc_id: int
    score: float
    doc: dict
    _highlighter: Highlighter = field(repr=False, default=None)
    _query_terms: list[str] = field(repr=False, default_factory=list)

    def highlight(self, field_name: str, **kwargs) -> str:
        """Get highlighted snippet for a field."""
        text = self.doc.get(field_name, "")
        if not text or not self._highlighter:
            return str(text)
        return self._highlighter.highlight(str(text), self._query_terms, **kwargs)

    def __repr__(self):
        return f"SearchHit(id={self.doc_id}, score={self.score:.4f})"


@dataclass
class SearchResults:
    """Container for search results with pagination and facets."""
    hits: list[SearchHit]
    total: int
    query_time_ms: float = 0.0
    facets: dict = field(default_factory=dict)

    def __iter__(self):
        return iter(self.hits)

    def __len__(self):
        return len(self.hits)

    def __repr__(self):
        return f"SearchResults(total={self.total}, returned={len(self.hits)}, time={self.query_time_ms:.1f}ms)"


class Searcher:
    """
    Executes queries against the inverted index and returns ranked results.
    """

    def __init__(self, index: InvertedIndex, scorer: Scorer | None = None,
                 highlighter: Highlighter | None = None):
        self.index = index
        self.scorer = scorer or BM25Scorer()
        self.highlighter = highlighter or Highlighter()

    def search(self, query: Query, limit: int = 10, offset: int = 0,
               facet_fields: list[str] | None = None,
               facet_filter: FacetFilter | None = None) -> SearchResults:
        """
        Execute a query and return ranked, paginated results.
        """
        import time
        start = time.perf_counter()

        # Collect matching doc_ids with scores
        scored_docs = self._execute_query(query)

        # Apply facet filters
        if facet_filter and facet_filter.active:
            scored_docs = {
                doc_id: score for doc_id, score in scored_docs.items()
                if self._passes_filter(doc_id, facet_filter)
            }

        total = len(scored_docs)

        # Collect facets from matched docs
        facets = {}
        if facet_fields:
            collector = FacetCollector()
            for doc_id in scored_docs:
                doc = self.index.get_document(doc_id)
                if doc:
                    for ff in facet_fields:
                        value = doc.fields.get(ff)
                        if value is not None:
                            collector.add(ff, value)
            facets = collector.to_dict()

        # Get top-K using a heap
        if offset + limit < total:
            top_items = heapq.nlargest(
                offset + limit,
                scored_docs.items(),
                key=lambda x: x[1],
            )
        else:
            top_items = sorted(
                scored_docs.items(),
                key=lambda x: x[1],
                reverse=True,
            )

        # Apply offset
        page_items = top_items[offset:offset + limit]

        # Build query terms for highlighting
        query_terms = self._extract_query_terms(query)

        # Build hits
        hits = []
        for doc_id, score in page_items:
            doc = self.index.get_document(doc_id)
            if doc:
                hits.append(SearchHit(
                    doc_id=doc_id,
                    score=score,
                    doc=doc.fields,
                    _highlighter=self.highlighter,
                    _query_terms=query_terms,
                ))

        elapsed = (time.perf_counter() - start) * 1000

        return SearchResults(
            hits=hits,
            total=total,
            query_time_ms=round(elapsed, 2),
            facets=facets,
        )

    def _execute_query(self, query: Query) -> dict[int, float]:
        """Recursively execute a query, returning {doc_id: score}."""
        if isinstance(query, MatchAllQuery):
            return {doc_id: query.boost for doc_id in self.index.all_doc_ids()}

        if isinstance(query, TermQuery):
            return self._execute_term_query(query)

        if isinstance(query, PhraseQuery):
            return self._execute_phrase_query(query)

        if isinstance(query, BoolQuery):
            return self._execute_bool_query(query)

        if isinstance(query, WildcardQuery):
            return self._execute_wildcard_query(query)

        if isinstance(query, RangeQuery):
            return self._execute_range_query(query)

        return {}

    def _execute_term_query(self, query: TermQuery) -> dict[int, float]:
        """Score documents containing a term."""
        scores: dict[int, float] = {}
        fields = [query.field] if query.field else self.index.schema.get_searchable_fields()

        for field_name in fields:
            field_type = self.index.schema.get_field(field_name)
            if field_type is None:
                continue

            # For keyword fields, use the term directly
            if isinstance(field_type, KeywordField):
                term = query.term
            else:
                # Analyze the query term through the field's analyzer
                analyzer = field_type.get_analyzer()
                if analyzer:
                    analyzed = analyzer.analyze_to_terms(query.term)
                    term = analyzed[0] if analyzed else query.term
                else:
                    term = query.term

            pl = self.index.get_postings(field_name, term)
            if not pl:
                continue

            total_docs = self.index.doc_count
            df = pl.doc_frequency
            avg_dl = self.index.avg_field_length(field_name)
            boost = field_type.boost * query.boost

            for posting in pl:
                dl = self.index.get_field_length(posting.doc_id, field_name)
                score = self.scorer.score(posting.term_freq, df, dl, avg_dl, total_docs)
                score *= boost

                if posting.doc_id in scores:
                    scores[posting.doc_id] += score
                else:
                    scores[posting.doc_id] = score

        return scores

    def _execute_phrase_query(self, query: PhraseQuery) -> dict[int, float]:
        """Score documents containing terms in exact sequence."""
        scores: dict[int, float] = {}
        fields = [query.field] if query.field else self.index.schema.get_text_fields()

        for field_name in fields:
            field_type = self.index.schema.get_field(field_name)
            if field_type is None:
                continue

            # Analyze each phrase term
            analyzer = field_type.get_analyzer()
            analyzed_terms = []
            for t in query.terms:
                if analyzer:
                    result = analyzer.analyze_to_terms(t)
                    if result:
                        analyzed_terms.append(result[0])
                else:
                    analyzed_terms.append(t)

            if not analyzed_terms:
                continue

            # Get posting lists for each term
            posting_lists = []
            for term in analyzed_terms:
                pl = self.index.get_postings(field_name, term)
                if not pl or pl.doc_frequency == 0:
                    posting_lists = []
                    break
                posting_lists.append(pl)

            if not posting_lists:
                continue

            # Find documents containing ALL terms
            common_docs = posting_lists[0].doc_ids()
            for pl in posting_lists[1:]:
                common_docs &= pl.doc_ids()

            # Check position sequences
            for doc_id in common_docs:
                if self._check_phrase_positions(posting_lists, doc_id,
                                                 field_name, query.slop):
                    # Score based on first term's stats
                    total_docs = self.index.doc_count
                    df = posting_lists[0].doc_frequency
                    dl = self.index.get_field_length(doc_id, field_name)
                    avg_dl = self.index.avg_field_length(field_name)
                    boost = field_type.boost * query.boost

                    # Phrase matches get a boost (they're more specific)
                    phrase_boost = 2.0
                    score = self.scorer.score(1, df, dl, avg_dl, total_docs)
                    score *= boost * phrase_boost

                    if doc_id in scores:
                        scores[doc_id] += score
                    else:
                        scores[doc_id] = score

        return scores

    def _check_phrase_positions(self, posting_lists, doc_id: int,
                                 field_name: str, slop: int = 0) -> bool:
        """Check if terms appear in sequence in a document."""
        # Get positions for each term in this doc
        all_positions = []
        for pl in posting_lists:
            doc_postings = pl.get_by_doc(doc_id)
            field_postings = [p for p in doc_postings if p.field_name == field_name]
            if not field_postings:
                return False
            positions = []
            for p in field_postings:
                positions.extend(p.positions)
            all_positions.append(sorted(positions))

        if not all_positions or not all_positions[0]:
            return False

        # Check for sequential positions
        for start_pos in all_positions[0]:
            match = True
            current_pos = start_pos
            for i in range(1, len(all_positions)):
                expected = current_pos + 1
                found = False
                for pos in all_positions[i]:
                    if abs(pos - expected) <= slop:
                        current_pos = pos
                        found = True
                        break
                if not found:
                    match = False
                    break
            if match:
                return True

        return False

    def _execute_bool_query(self, query: BoolQuery) -> dict[int, float]:
        """Execute boolean combinations of sub-queries."""
        # Execute must (AND) clauses
        must_results = None
        for sub in query.must:
            sub_scores = self._execute_query(sub)
            if must_results is None:
                must_results = dict(sub_scores)
            else:
                common = set(must_results.keys()) & set(sub_scores.keys())
                must_results = {
                    doc_id: must_results[doc_id] + sub_scores[doc_id]
                    for doc_id in common
                }

        # Execute should (OR) clauses
        should_results: dict[int, float] = {}
        for sub in query.should:
            sub_scores = self._execute_query(sub)
            for doc_id, score in sub_scores.items():
                if doc_id in should_results:
                    should_results[doc_id] = max(should_results[doc_id], score)
                else:
                    should_results[doc_id] = score

        # Execute must_not (NOT) clauses
        excluded = set()
        for sub in query.must_not:
            sub_scores = self._execute_query(sub)
            excluded.update(sub_scores.keys())

        # Combine results
        if must_results is not None and should_results:
            combined = dict(must_results)
            for doc_id, score in should_results.items():
                if doc_id in combined:
                    combined[doc_id] += score
        elif must_results is not None:
            combined = must_results
        elif should_results:
            combined = should_results
        else:
            combined = {}

        # Apply exclusions
        for doc_id in excluded:
            combined.pop(doc_id, None)

        # Apply boost
        if query.boost != 1.0:
            combined = {k: v * query.boost for k, v in combined.items()}

        return combined

    def _execute_wildcard_query(self, query: WildcardQuery) -> dict[int, float]:
        """Execute prefix/wildcard matching."""
        scores: dict[int, float] = {}
        fields = [query.field] if query.field else self.index.schema.get_searchable_fields()
        prefix = query.pattern.rstrip("*").lower()

        for field_name in fields:
            all_terms = self.index.get_all_terms(field_name)
            matching_terms = [t for t in all_terms if t.startswith(prefix)]

            for term in matching_terms:
                term_query = TermQuery(term=term, field=field_name, boost=query.boost)
                term_scores = self._execute_term_query(term_query)
                for doc_id, score in term_scores.items():
                    if doc_id in scores:
                        scores[doc_id] = max(scores[doc_id], score)
                    else:
                        scores[doc_id] = score

        return scores

    def _execute_range_query(self, query: RangeQuery) -> dict[int, float]:
        """Execute range query on numeric fields."""
        scores: dict[int, float] = {}

        for doc_id in self.index.all_doc_ids():
            doc = self.index.get_document(doc_id)
            if not doc:
                continue

            value = doc.fields.get(query.field)
            if value is None:
                continue

            try:
                value = float(value)
            except (ValueError, TypeError):
                continue

            in_range = True
            if query.gte is not None and value < float(query.gte):
                in_range = False
            if query.gt is not None and value <= float(query.gt):
                in_range = False
            if query.lte is not None and value > float(query.lte):
                in_range = False
            if query.lt is not None and value >= float(query.lt):
                in_range = False

            if in_range:
                scores[doc_id] = query.boost

        return scores

    def _passes_filter(self, doc_id: int, facet_filter: FacetFilter) -> bool:
        """Check if doc passes facet filter."""
        doc = self.index.get_document(doc_id)
        if not doc:
            return False
        return facet_filter.matches(doc.fields)

    def _extract_query_terms(self, query: Query) -> list[str]:
        """Extract all search terms from a query for highlighting."""
        terms = []
        if isinstance(query, TermQuery):
            terms.append(query.term)
        elif isinstance(query, PhraseQuery):
            terms.extend(query.terms)
        elif isinstance(query, BoolQuery):
            for sub in query.must + query.should:
                terms.extend(self._extract_query_terms(sub))
        elif isinstance(query, WildcardQuery):
            terms.append(query.pattern.rstrip("*"))
        return terms
