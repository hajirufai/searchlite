"""
Schema definition — describes what fields a document has and how they're indexed.
"""

from dataclasses import dataclass, field as dc_field
from searchlite.analyzer import standard_analyzer, keyword_analyzer, Analyzer


@dataclass
class FieldType:
    """Base field type."""
    stored: bool = True
    indexed: bool = True
    analyzer: Analyzer | None = None
    boost: float = 1.0
    faceted: bool = False

    def get_analyzer(self) -> Analyzer | None:
        return self.analyzer


@dataclass
class TextField(FieldType):
    """Full-text searchable field with analysis pipeline."""
    stored: bool = True
    indexed: bool = True
    analyzer: Analyzer | None = None
    boost: float = 1.0

    def __post_init__(self):
        if self.analyzer is None:
            self.analyzer = standard_analyzer()


@dataclass
class KeywordField(FieldType):
    """Exact-match field (tags, IDs, categories). No stemming or analysis."""
    stored: bool = True
    indexed: bool = True
    analyzer: Analyzer | None = None
    faceted: bool = False
    boost: float = 1.0

    def __post_init__(self):
        if self.analyzer is None:
            self.analyzer = keyword_analyzer()


@dataclass
class NumericField(FieldType):
    """Numeric field for range queries and sorting."""
    stored: bool = True
    indexed: bool = True
    sortable: bool = True
    boost: float = 1.0

    def get_analyzer(self) -> None:
        return None


class Schema:
    """
    Defines the structure of documents in the index.

    Usage:
        schema = Schema(
            title=TextField(boost=2.0),
            body=TextField(),
            tags=KeywordField(faceted=True),
            year=NumericField(),
        )
    """

    def __init__(self, **fields):
        self.fields: dict[str, FieldType] = {}
        for name, field_type in fields.items():
            if not isinstance(field_type, FieldType):
                raise TypeError(
                    f"Field '{name}' must be a FieldType instance, got {type(field_type)}"
                )
            self.fields[name] = field_type

    def get_field(self, name: str) -> FieldType | None:
        return self.fields.get(name)

    def get_text_fields(self) -> list[str]:
        return [n for n, f in self.fields.items() if isinstance(f, TextField)]

    def get_keyword_fields(self) -> list[str]:
        return [n for n, f in self.fields.items() if isinstance(f, KeywordField)]

    def get_numeric_fields(self) -> list[str]:
        return [n for n, f in self.fields.items() if isinstance(f, NumericField)]

    def get_faceted_fields(self) -> list[str]:
        return [n for n, f in self.fields.items() if f.faceted]

    def get_searchable_fields(self) -> list[str]:
        return [n for n, f in self.fields.items() if f.indexed and isinstance(f, (TextField, KeywordField))]

    def field_names(self) -> list[str]:
        return list(self.fields.keys())

    def validate_document(self, doc_fields: dict) -> list[str]:
        """Check a doc's fields against the schema. Returns list of warnings."""
        warnings = []
        for name in doc_fields:
            if name not in self.fields:
                warnings.append(f"Unknown field '{name}' not in schema")
        return warnings

    def __repr__(self):
        parts = []
        for name, ft in self.fields.items():
            parts.append(f"{name}={ft.__class__.__name__}")
        return f"Schema({', '.join(parts)})"
