"""
Query AST — represents parsed search queries as a tree of query nodes.
"""

from dataclasses import dataclass, field


class Query:
    """Base query node."""
    pass


@dataclass
class TermQuery(Query):
    """Match documents containing a single term."""
    term: str
    field: str | None = None
    boost: float = 1.0

    def __repr__(self):
        if self.field:
            return f"TermQuery({self.field}:{self.term})"
        return f"TermQuery({self.term})"


@dataclass
class PhraseQuery(Query):
    """Match documents containing terms in exact order (position-aware)."""
    terms: list[str]
    field: str | None = None
    boost: float = 1.0
    slop: int = 0  # allowed gap between terms

    def __repr__(self):
        phrase = " ".join(self.terms)
        if self.field:
            return f'PhraseQuery({self.field}:"{phrase}")'
        return f'PhraseQuery("{phrase}")'


@dataclass
class BoolQuery(Query):
    """
    Boolean combination of sub-queries.
    must = AND, should = OR, must_not = NOT
    """
    must: list[Query] = field(default_factory=list)
    should: list[Query] = field(default_factory=list)
    must_not: list[Query] = field(default_factory=list)
    boost: float = 1.0

    def __repr__(self):
        parts = []
        if self.must:
            parts.append(f"must={self.must}")
        if self.should:
            parts.append(f"should={self.should}")
        if self.must_not:
            parts.append(f"must_not={self.must_not}")
        return f"BoolQuery({', '.join(parts)})"


@dataclass
class WildcardQuery(Query):
    """Prefix/wildcard matching (e.g., pyth* matches python, pythonic)."""
    pattern: str
    field: str | None = None
    boost: float = 1.0

    def __repr__(self):
        return f"WildcardQuery({self.pattern})"


@dataclass
class RangeQuery(Query):
    """Numeric or string range query."""
    field: str
    gte: float | str | None = None
    lte: float | str | None = None
    gt: float | str | None = None
    lt: float | str | None = None
    boost: float = 1.0

    def __repr__(self):
        parts = []
        if self.gte is not None:
            parts.append(f">={self.gte}")
        if self.gt is not None:
            parts.append(f">{self.gt}")
        if self.lte is not None:
            parts.append(f"<={self.lte}")
        if self.lt is not None:
            parts.append(f"<{self.lt}")
        return f"RangeQuery({self.field}: {' '.join(parts)})"


@dataclass
class MatchAllQuery(Query):
    """Match every document in the index."""
    boost: float = 1.0

    def __repr__(self):
        return "MatchAllQuery()"
