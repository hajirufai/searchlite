# SearchLite

A full-text search engine built from scratch in Python. No Elasticsearch, no Whoosh, no Lucene bindings — just pure Python implementing the core algorithms that power search.

## What's inside

- **Inverted index** with term-frequency tracking and position storage
- **BM25 + TF-IDF** scoring with field boosting and length normalization
- **Porter stemmer** implemented from the original algorithm (5 steps, all rules)
- **Query parser** — recursive descent parser handling `AND`, `OR`, `NOT`, phrases, wildcards, field-specific search, grouping, and boost operators
- **Phrase queries** with positional intersection and slop support
- **Faceted search** with field counting and multi-value filtering
- **Result highlighting** with best-passage extraction
- **Persistent storage** — JSON-based segments with compaction
- **Schema system** — typed fields (text, keyword, numeric) with per-field analyzers

Zero external dependencies. Standard library only.

## Quick start

```python
from searchlite import SearchEngine, Schema, TextField, KeywordField

engine = SearchEngine(schema=Schema(
    title=TextField(boost=2.0),
    body=TextField(),
    tags=KeywordField(faceted=True),
))

engine.add({
    "title": "Building Data Pipelines",
    "body": "A guide to ETL with Python and Apache Kafka for real-time streaming",
    "tags": ["data-engineering", "python", "kafka"],
})

engine.add({
    "title": "Machine Learning Fundamentals",
    "body": "Supervised and unsupervised learning with scikit-learn and PyTorch",
    "tags": ["machine-learning", "python"],
})

# Simple term search
results = engine.search("python")

# Phrase query
results = engine.search('"data pipelines"')

# Boolean operators
results = engine.search("python AND kafka")
results = engine.search("tensorflow OR pytorch")
results = engine.search("python NOT java")

# Field-specific
results = engine.search('title:"machine learning"')

# Wildcards
results = engine.search("pyth*")

# With facets
results = engine.search("python", facets=["tags"])
print(results.facets)  # {"tags": {"python": 2, "data-engineering": 1, ...}}

# With filtering
results = engine.search("python", filters={"tags": ["machine-learning"]})

# Highlighting
for hit in results:
    print(hit.score, hit.doc["title"])
    print(hit.highlight("body", fragment_size=100))
```

## Persistence

```python
# Save to disk
engine = SearchEngine("./my_index", schema=schema)
engine.add({"title": "Persisted document", "body": "This survives restarts"})
engine.commit()

# Load later
engine = SearchEngine("./my_index", schema=schema)
results = engine.search("persisted")  # finds it
```

## Query syntax

| Query | Description |
|---|---|
| `python` | Single term |
| `python data` | Implicit AND |
| `python AND data` | Explicit AND |
| `python OR java` | Either term |
| `NOT python` | Exclude term |
| `"machine learning"` | Exact phrase |
| `title:python` | Field-specific term |
| `title:"data science"` | Field-specific phrase |
| `pyth*` | Prefix wildcard |
| `python^2.0` | Term boost |
| `(python OR java) AND data` | Grouping |

## Architecture

```
                   ┌─────────────┐
                   │  SearchEngine│
                   └──────┬──────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
        ┌─────┴─────┐ ┌──┴──┐  ┌────┴─────┐
        │  Searcher  │ │Index│  │  Storage  │
        └─────┬─────┘ └──┬──┘  └──────────┘
              │           │
    ┌─────────┼─────┐     │
    │         │     │     │
┌───┴──┐ ┌───┴──┐ ┌┴──┐  │
│Scorer│ │Parser│ │HL │  │
└──────┘ └──────┘ └───┘  │
                          │
                ┌─────────┼─────────┐
                │         │         │
          ┌─────┴──┐ ┌───┴────┐ ┌──┴───┐
          │Analyzer│ │Postings│ │Schema│
          └───┬────┘ └────────┘ └──────┘
              │
    ┌─────────┼─────────┐
    │         │         │
┌───┴────┐┌──┴───┐┌───┴────┐
│Tokenizer││Stemmer││Stopwords│
└────────┘└──────┘└────────┘
```

**Modules:**

| Module | Lines | Purpose |
|---|---|---|
| `tokenizer.py` | Regex, whitespace, n-gram, sentence tokenizers with position tracking |
| `normalizer.py` | Lowercase, Unicode NFKC, accent stripping, punctuation removal |
| `stemmer.py` | Full Porter stemmer (Steps 1a–5) |
| `stopwords.py` | English stop word list + filter |
| `analyzer.py` | Composable analysis pipeline with builder pattern |
| `schema.py` | Field types (Text, Keyword, Numeric) with per-field configuration |
| `posting.py` | Posting lists with set operations (intersect, union, difference) |
| `index.py` | Inverted index — add, remove, serialize, term/doc lookups |
| `query.py` | Query AST nodes (Term, Phrase, Bool, Wildcard, Range, MatchAll) |
| `query_parser.py` | Recursive descent parser with operator precedence |
| `scorer.py` | BM25 (k1, b tunable) and TF-IDF scorers |
| `highlighter.py` | Snippet extraction with best-passage ranking |
| `facets.py` | Facet collection and multi-field filtering |
| `storage.py` | JSON segment persistence with compaction |
| `searcher.py` | Query execution pipeline: parse → postings → score → rank |
| `engine.py` | High-level API tying everything together |

## Performance

Benchmarked on synthetic documents:

```
Index  1000 docs:  ~1.2s  (830 docs/sec)
Index  5000 docs:  ~9.1s  (548 docs/sec)

Query (1000 docs):
  "python"                    avg=0.26ms  p99=0.30ms
  "python AND machine"        avg=0.52ms  p99=0.56ms
  "docker OR kubernetes"      avg=0.58ms  p99=0.63ms
  title:python                avg=0.10ms  p99=0.16ms
  pyth*                       avg=0.31ms  p99=0.43ms
```

## Running tests

```bash
pytest tests/ -v
```

194 tests covering tokenization, stemming, analysis, indexing, scoring, query parsing, highlighting, facets, storage, search execution, and the engine API.

## Examples

```bash
python examples/basic_search.py       # Basic indexing and querying
python examples/benchmark.py          # Performance benchmarks
python examples/wikipedia_search.py   # Wikipedia article search
python examples/log_search.py         # Server log analysis
```

## License

MIT
