"""
Document model — represents an indexed document with fields and metadata.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A document stored in the index."""
    doc_id: int
    fields: dict[str, Any]
    _stored: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.fields.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.fields

    def get(self, key: str, default=None) -> Any:
        return self.fields.get(key, default)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "fields": self.fields,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        return cls(
            doc_id=data["doc_id"],
            fields=data["fields"],
        )

    def __repr__(self):
        preview = {}
        for k, v in self.fields.items():
            if isinstance(v, str) and len(v) > 50:
                preview[k] = v[:50] + "..."
            else:
                preview[k] = v
        return f"Document(id={self.doc_id}, {preview})"
