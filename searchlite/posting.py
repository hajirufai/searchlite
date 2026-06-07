"""
Posting lists — the core data structure of an inverted index.
Each posting records which document a term appears in, how often, and where.
"""

from dataclasses import dataclass, field


@dataclass
class Posting:
    """
    A single posting: one term in one document.
    """
    doc_id: int
    term_freq: int = 1
    positions: list[int] = field(default_factory=list)
    field_name: str = ""

    def to_dict(self) -> dict:
        d = {"doc_id": self.doc_id, "tf": self.term_freq}
        if self.positions:
            d["pos"] = self.positions
        if self.field_name:
            d["field"] = self.field_name
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Posting":
        return cls(
            doc_id=data["doc_id"],
            term_freq=data.get("tf", 1),
            positions=data.get("pos", []),
            field_name=data.get("field", ""),
        )


class PostingList:
    """
    Sorted list of postings for a term. Supports intersection and union
    operations needed for boolean queries.
    """

    def __init__(self, postings: list[Posting] | None = None):
        self._postings: list[Posting] = sorted(
            postings or [], key=lambda p: (p.doc_id, p.field_name)
        )

    def add(self, posting: Posting):
        """Add a posting, keeping the list sorted."""
        # Check if we already have this doc_id + field combo
        for i, existing in enumerate(self._postings):
            if existing.doc_id == posting.doc_id and existing.field_name == posting.field_name:
                existing.term_freq += posting.term_freq
                existing.positions.extend(posting.positions)
                existing.positions.sort()
                return

        # Binary insert to maintain sort order
        lo, hi = 0, len(self._postings)
        key = (posting.doc_id, posting.field_name)
        while lo < hi:
            mid = (lo + hi) // 2
            mid_key = (self._postings[mid].doc_id, self._postings[mid].field_name)
            if mid_key < key:
                lo = mid + 1
            else:
                hi = mid
        self._postings.insert(lo, posting)

    def get_by_doc(self, doc_id: int) -> list[Posting]:
        """Get all postings for a specific document."""
        return [p for p in self._postings if p.doc_id == doc_id]

    @property
    def doc_frequency(self) -> int:
        """Number of unique documents containing this term."""
        return len(set(p.doc_id for p in self._postings))

    @property
    def total_frequency(self) -> int:
        """Total occurrences across all documents."""
        return sum(p.term_freq for p in self._postings)

    def doc_ids(self) -> set[int]:
        """Set of all document IDs in this posting list."""
        return set(p.doc_id for p in self._postings)

    def __iter__(self):
        return iter(self._postings)

    def __len__(self):
        return len(self._postings)

    def __repr__(self):
        return f"PostingList({len(self._postings)} postings, df={self.doc_frequency})"

    def to_list(self) -> list[dict]:
        return [p.to_dict() for p in self._postings]

    @classmethod
    def from_list(cls, data: list[dict]) -> "PostingList":
        postings = [Posting.from_dict(d) for d in data]
        return cls(postings)

    @staticmethod
    def intersect(a: "PostingList", b: "PostingList") -> set[int]:
        """AND — documents appearing in both lists."""
        return a.doc_ids() & b.doc_ids()

    @staticmethod
    def union(a: "PostingList", b: "PostingList") -> set[int]:
        """OR — documents appearing in either list."""
        return a.doc_ids() | b.doc_ids()

    @staticmethod
    def difference(a: "PostingList", b: "PostingList") -> set[int]:
        """NOT — documents in a but not in b."""
        return a.doc_ids() - b.doc_ids()
