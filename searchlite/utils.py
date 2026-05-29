"""
Utility functions — timing, formatting, memory usage.
"""

import time
import sys
from functools import wraps


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def sizeof_deep(obj, seen=None) -> int:
    """Recursively estimate the memory usage of an object."""
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(sizeof_deep(k, seen) + sizeof_deep(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(sizeof_deep(item, seen) for item in obj)
    elif hasattr(obj, "__dict__"):
        size += sizeof_deep(obj.__dict__, seen)

    return size


class Timer:
    """Simple context manager for timing operations."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000

    def __repr__(self):
        return f"{self.label}: {self.elapsed_ms:.2f}ms"


def timed(func):
    """Decorator that prints execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  [{func.__name__}] {elapsed:.2f}ms")
        return result
    return wrapper


def truncate(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
