"""Tests for persistent storage."""

import pytest
import tempfile
import json
from pathlib import Path
from searchlite.storage import IndexStorage


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestIndexStorage:
    def test_initialize(self, temp_dir):
        storage = IndexStorage(temp_dir)
        storage.initialize()
        assert (Path(temp_dir) / "segments").is_dir()

    def test_save_and_load(self, temp_dir):
        storage = IndexStorage(temp_dir)
        data = {"test": "data", "count": 42}
        seg_id = storage.save_segment(data)

        loaded = storage.load_segment(seg_id)
        assert loaded == data

    def test_load_latest(self, temp_dir):
        storage = IndexStorage(temp_dir)
        storage.save_segment({"version": 1}, segment_id="seg1")
        storage.save_segment({"version": 2}, segment_id="seg2")

        latest = storage.load_latest()
        assert latest["version"] == 2

    def test_list_segments(self, temp_dir):
        storage = IndexStorage(temp_dir)
        storage.save_segment({"a": 1}, segment_id="s1")
        storage.save_segment({"b": 2}, segment_id="s2")

        segs = storage.list_segments()
        assert len(segs) == 2

    def test_delete_segment(self, temp_dir):
        storage = IndexStorage(temp_dir)
        storage.save_segment({"a": 1}, segment_id="s1")
        assert storage.delete_segment("s1") is True
        assert storage.load_segment("s1") is None

    def test_delete_nonexistent(self, temp_dir):
        storage = IndexStorage(temp_dir)
        storage.initialize()
        assert storage.delete_segment("nope") is False

    def test_compact(self, temp_dir):
        storage = IndexStorage(temp_dir)
        storage.save_segment({"a": 1}, segment_id="s1")
        storage.save_segment({"b": 2}, segment_id="s2")

        storage.compact({"merged": True})

        segs = storage.list_segments()
        assert len(segs) == 1
        latest = storage.load_latest()
        assert latest["merged"] is True

    def test_index_size(self, temp_dir):
        storage = IndexStorage(temp_dir)
        storage.save_segment({"data": "x" * 1000})
        size = storage.index_size_bytes()
        assert size > 0

    def test_index_size_human(self, temp_dir):
        storage = IndexStorage(temp_dir)
        storage.save_segment({"data": "x" * 1000})
        human = storage.index_size_human()
        assert "B" in human or "KB" in human

    def test_exists(self, temp_dir):
        storage = IndexStorage(temp_dir)
        assert storage.exists() is False
        storage.save_segment({"test": 1})
        assert storage.exists() is True

    def test_load_nonexistent(self, temp_dir):
        storage = IndexStorage(temp_dir)
        assert storage.load_segment("nope") is None

    def test_load_latest_empty(self, temp_dir):
        storage = IndexStorage(temp_dir)
        assert storage.load_latest() is None
