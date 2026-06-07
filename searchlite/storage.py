"""
Persistent storage — write and read index data to/from disk.
Supports JSON segments for simplicity and debuggability.
"""

import json
import os
import time
from pathlib import Path


class IndexStorage:
    """
    File-based index persistence.
    Stores index segments as JSON files with metadata.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._segments_dir = self.path / "segments"
        self._meta_file = self.path / "meta.json"

    def initialize(self):
        """Create the storage directory structure."""
        self._segments_dir.mkdir(parents=True, exist_ok=True)

    def save_segment(self, index_data: dict, segment_id: str | None = None) -> str:
        """
        Write an index segment to disk.
        Returns the segment filename.
        """
        self.initialize()

        if segment_id is None:
            segment_id = f"seg_{int(time.time() * 1000)}"

        segment_file = self._segments_dir / f"{segment_id}.json"
        with open(segment_file, "w") as f:
            json.dump(index_data, f, separators=(",", ":"))

        # Update metadata
        self._update_meta(segment_id)

        return segment_id

    def load_segment(self, segment_id: str) -> dict | None:
        """Load a segment from disk."""
        segment_file = self._segments_dir / f"{segment_id}.json"
        if not segment_file.exists():
            return None

        with open(segment_file) as f:
            return json.load(f)

    def load_latest(self) -> dict | None:
        """Load the most recent segment."""
        meta = self._load_meta()
        if not meta or not meta.get("segments"):
            return None

        latest = meta["segments"][-1]
        return self.load_segment(latest["id"])

    def list_segments(self) -> list[dict]:
        """List all segments with metadata."""
        meta = self._load_meta()
        return meta.get("segments", []) if meta else []

    def delete_segment(self, segment_id: str) -> bool:
        """Remove a segment file."""
        segment_file = self._segments_dir / f"{segment_id}.json"
        if segment_file.exists():
            segment_file.unlink()

            meta = self._load_meta()
            if meta:
                meta["segments"] = [
                    s for s in meta.get("segments", [])
                    if s["id"] != segment_id
                ]
                self._save_meta(meta)
            return True
        return False

    def compact(self, merged_data: dict) -> str:
        """
        Replace all segments with a single compacted segment.
        Removes old segments and writes merged data.
        """
        # Delete existing segments
        for seg in self.list_segments():
            segment_file = self._segments_dir / f"{seg['id']}.json"
            if segment_file.exists():
                segment_file.unlink()

        # Write merged segment
        seg_id = f"compact_{int(time.time() * 1000)}"
        self._save_meta({"segments": []})
        return self.save_segment(merged_data, segment_id=seg_id)

    def index_size_bytes(self) -> int:
        """Total size of all segment files."""
        total = 0
        if self._segments_dir.exists():
            for f in self._segments_dir.iterdir():
                if f.is_file():
                    total += f.stat().st_size
        return total

    def index_size_human(self) -> str:
        """Human-readable index size."""
        size = self.index_size_bytes()
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def exists(self) -> bool:
        """Check if an index exists at this path."""
        return self._meta_file.exists()

    def _update_meta(self, segment_id: str):
        meta = self._load_meta() or {"segments": [], "created": time.time()}
        meta["segments"].append({
            "id": segment_id,
            "timestamp": time.time(),
        })
        meta["updated"] = time.time()
        self._save_meta(meta)

    def _load_meta(self) -> dict | None:
        if self._meta_file.exists():
            with open(self._meta_file) as f:
                return json.load(f)
        return None

    def _save_meta(self, meta: dict):
        self.initialize()
        with open(self._meta_file, "w") as f:
            json.dump(meta, f, indent=2)
