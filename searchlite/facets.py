"""
Faceted search — count and filter results by field values.
"""

from collections import Counter, defaultdict


class FacetCollector:
    """Collects facet counts during search execution."""

    def __init__(self):
        self._counts: dict[str, Counter] = defaultdict(Counter)

    def add(self, field_name: str, value):
        """Record a value for faceting."""
        if isinstance(value, list):
            for v in value:
                self._counts[field_name][str(v)] += 1
        else:
            self._counts[field_name][str(value)] += 1

    def get_facets(self, field_name: str, limit: int = 10) -> list[tuple[str, int]]:
        """Get top facet values with counts, sorted by count descending."""
        return self._counts[field_name].most_common(limit)

    def get_all_facets(self, limit: int = 10) -> dict[str, list[tuple[str, int]]]:
        """Get facets for all fields."""
        return {
            field: counter.most_common(limit)
            for field, counter in self._counts.items()
        }

    def to_dict(self) -> dict[str, dict[str, int]]:
        """Serialize facets to a dict."""
        return {
            field: dict(counter.most_common())
            for field, counter in self._counts.items()
        }


class FacetFilter:
    """Filter documents by facet field values."""

    def __init__(self):
        self._filters: dict[str, set[str]] = {}

    def add_filter(self, field_name: str, values: list[str]):
        """Add a filter: only include docs where field matches one of the values."""
        self._filters[field_name] = set(values)

    def matches(self, doc_fields: dict) -> bool:
        """Check if a document passes all active filters."""
        for field_name, allowed in self._filters.items():
            value = doc_fields.get(field_name)
            if value is None:
                return False

            if isinstance(value, list):
                if not any(str(v) in allowed for v in value):
                    return False
            else:
                if str(value) not in allowed:
                    return False

        return True

    @property
    def active(self) -> bool:
        return bool(self._filters)
