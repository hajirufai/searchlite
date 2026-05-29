"""
SearchLite benchmark — measure indexing speed and query throughput.
"""

import sys
import os
import time
import random
import string

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searchlite import SearchEngine, Schema, TextField, KeywordField


def generate_document(doc_id: int) -> dict:
    """Generate a synthetic document for benchmarking."""
    topics = [
        "python programming data science machine learning",
        "javascript react frontend web development",
        "java spring boot microservices backend api",
        "sql database postgresql indexing query optimization",
        "docker kubernetes devops cloud infrastructure",
        "rust systems programming memory safety performance",
        "data engineering etl pipeline streaming batch processing",
        "natural language processing transformers embeddings",
        "distributed systems consensus replication sharding",
        "computer networks tcp http websocket protocol",
    ]

    words = topics[doc_id % len(topics)].split()
    extra = " ".join(random.choices(words, k=random.randint(50, 150)))
    title = " ".join(random.choices(words, k=random.randint(3, 6))).title()
    tags = random.sample(words, min(3, len(words)))

    return {
        "title": title,
        "body": f"Article about {title.lower()}. {extra}",
        "tags": tags,
        "author": random.choice(["alice", "bob", "charlie", "diana"]),
    }


def main():
    schema = Schema(
        title=TextField(boost=2.0),
        body=TextField(),
        tags=KeywordField(faceted=True),
        author=KeywordField(),
    )

    # Benchmark indexing
    doc_counts = [100, 500, 1000, 5000]
    for n in doc_counts:
        engine = SearchEngine(schema=schema)
        docs = [generate_document(i) for i in range(n)]

        start = time.perf_counter()
        engine.add_many(docs)
        elapsed = time.perf_counter() - start

        rate = n / elapsed
        print(f"Index {n:>5} docs: {elapsed*1000:.1f}ms ({rate:.0f} docs/sec)")

    print()

    # Benchmark querying
    engine = SearchEngine(schema=schema)
    docs = [generate_document(i) for i in range(1000)]
    engine.add_many(docs)

    queries = [
        "python",
        "data engineering",
        "python AND machine",
        '"machine learning"',
        "docker OR kubernetes",
        "title:python",
        "pyth*",
        "NOT javascript",
        "(python OR java) AND data",
    ]

    print(f"Query benchmark ({engine.stats()['documents']} docs indexed):")
    for q in queries:
        # Warm up
        engine.search(q)

        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            engine.search(q)
            times.append((time.perf_counter() - start) * 1000)

        avg = sum(times) / len(times)
        p99 = sorted(times)[98]
        print(f"  {q:<35} avg={avg:.3f}ms  p99={p99:.3f}ms")

    print()

    # Persistence benchmark
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        engine = SearchEngine(tmp, schema=schema)
        docs = [generate_document(i) for i in range(1000)]
        engine.add_many(docs)

        start = time.perf_counter()
        engine.commit()
        write_time = (time.perf_counter() - start) * 1000

        # Reload
        start = time.perf_counter()
        engine2 = SearchEngine(tmp, schema=schema)
        load_time = (time.perf_counter() - start) * 1000

        print(f"Persistence (1000 docs):")
        print(f"  Write: {write_time:.1f}ms")
        print(f"  Load:  {load_time:.1f}ms")
        print(f"  Size:  {engine._storage.index_size_human()}")
        print(f"  Docs:  {engine2.stats()['documents']}")


if __name__ == "__main__":
    main()
