"""Tests for faceted search."""

import pytest
from searchlite.facets import FacetCollector, FacetFilter


class TestFacetCollector:
    def test_basic_counting(self):
        fc = FacetCollector()
        fc.add("category", "python")
        fc.add("category", "python")
        fc.add("category", "java")

        facets = fc.get_facets("category")
        assert ("python", 2) in facets
        assert ("java", 1) in facets

    def test_list_values(self):
        fc = FacetCollector()
        fc.add("tags", ["python", "tutorial"])
        fc.add("tags", ["python", "advanced"])

        facets = fc.get_facets("tags")
        counts = dict(facets)
        assert counts["python"] == 2
        assert counts["tutorial"] == 1
        assert counts["advanced"] == 1

    def test_limit(self):
        fc = FacetCollector()
        for i in range(20):
            fc.add("category", f"cat_{i}")

        facets = fc.get_facets("category", limit=5)
        assert len(facets) == 5

    def test_multiple_fields(self):
        fc = FacetCollector()
        fc.add("category", "python")
        fc.add("level", "beginner")

        all_facets = fc.get_all_facets()
        assert "category" in all_facets
        assert "level" in all_facets

    def test_to_dict(self):
        fc = FacetCollector()
        fc.add("category", "python")
        d = fc.to_dict()
        assert isinstance(d, dict)
        assert "python" in d["category"]

    def test_empty(self):
        fc = FacetCollector()
        facets = fc.get_facets("nonexistent")
        assert facets == []


class TestFacetFilter:
    def test_single_filter(self):
        ff = FacetFilter()
        ff.add_filter("category", ["python"])

        assert ff.matches({"category": "python"}) is True
        assert ff.matches({"category": "java"}) is False

    def test_multiple_allowed_values(self):
        ff = FacetFilter()
        ff.add_filter("category", ["python", "java"])

        assert ff.matches({"category": "python"}) is True
        assert ff.matches({"category": "java"}) is True
        assert ff.matches({"category": "rust"}) is False

    def test_list_field(self):
        ff = FacetFilter()
        ff.add_filter("tags", ["python"])

        assert ff.matches({"tags": ["python", "tutorial"]}) is True
        assert ff.matches({"tags": ["java", "tutorial"]}) is False

    def test_missing_field(self):
        ff = FacetFilter()
        ff.add_filter("category", ["python"])

        assert ff.matches({}) is False

    def test_multiple_filters(self):
        ff = FacetFilter()
        ff.add_filter("category", ["python"])
        ff.add_filter("level", ["beginner"])

        assert ff.matches({"category": "python", "level": "beginner"}) is True
        assert ff.matches({"category": "python", "level": "advanced"}) is False

    def test_active(self):
        ff = FacetFilter()
        assert ff.active is False
        ff.add_filter("category", ["python"])
        assert ff.active is True

    def test_no_filter_matches_all(self):
        ff = FacetFilter()
        assert ff.matches({"anything": "value"}) is True
