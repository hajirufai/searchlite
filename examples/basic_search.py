"""
Basic SearchLite example — index some documents and run queries.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searchlite import SearchEngine, Schema, TextField, KeywordField


def main():
    # Define schema
    schema = Schema(
        title=TextField(boost=2.0),
        body=TextField(),
        tags=KeywordField(faceted=True),
        author=KeywordField(),
    )

    # Create engine (in-memory for this example)
    engine = SearchEngine(schema=schema)

    # Index some documents
    documents = [
        {
            "title": "Introduction to Python Programming",
            "body": "Python is a versatile programming language used in web development, "
                    "data science, machine learning, and automation. Its clean syntax makes "
                    "it ideal for beginners and professionals alike.",
            "tags": ["python", "programming", "tutorial"],
            "author": "Haji Rufai",
        },
        {
            "title": "Building Data Pipelines with Apache Kafka",
            "body": "Apache Kafka is a distributed event streaming platform. It handles "
                    "real-time data feeds with high throughput and low latency. This guide "
                    "covers setting up producers, consumers, and topic partitioning.",
            "tags": ["data-engineering", "kafka", "streaming"],
            "author": "Haji Rufai",
        },
        {
            "title": "Machine Learning with Scikit-Learn",
            "body": "Scikit-learn provides simple and efficient tools for data mining and "
                    "machine learning in Python. Learn about classification, regression, "
                    "clustering, and dimensionality reduction.",
            "tags": ["machine-learning", "python", "data-science"],
            "author": "Jane Smith",
        },
        {
            "title": "SQL Query Optimization Techniques",
            "body": "Writing efficient SQL queries is crucial for database performance. "
                    "This article covers indexing strategies, query plans, JOIN optimization, "
                    "and common anti-patterns to avoid in production databases.",
            "tags": ["sql", "database", "performance"],
            "author": "Haji Rufai",
        },
        {
            "title": "Docker Containers for Data Engineers",
            "body": "Docker simplifies deployment by packaging applications into containers. "
                    "Data engineers use Docker to create reproducible environments for ETL "
                    "pipelines, databases, and data processing workflows.",
            "tags": ["docker", "devops", "data-engineering"],
            "author": "Jane Smith",
        },
    ]

    print("Indexing documents...")
    for doc in documents:
        engine.add(doc)
    print(f"Indexed {len(documents)} documents\n")

    # Run searches
    print("=" * 60)
    print("Search: 'python'")
    print("=" * 60)
    results = engine.search("python")
    for hit in results:
        print(f"  [{hit.score:.4f}] {hit.doc['title']}")
        print(f"           {hit.highlight('body', fragment_size=80)}")
        print()

    print("=" * 60)
    print("Search: 'data engineering'")
    print("=" * 60)
    results = engine.search("data engineering")
    for hit in results:
        print(f"  [{hit.score:.4f}] {hit.doc['title']}")
        print(f"           {hit.highlight('body', fragment_size=80)}")
        print()

    print("=" * 60)
    print('Search: \'"machine learning"\' (phrase query)')
    print("=" * 60)
    results = engine.search('"machine learning"')
    for hit in results:
        print(f"  [{hit.score:.4f}] {hit.doc['title']}")
        print()

    print("=" * 60)
    print("Search: 'python AND data' (boolean AND)")
    print("=" * 60)
    results = engine.search("python AND data")
    for hit in results:
        print(f"  [{hit.score:.4f}] {hit.doc['title']}")
        print()

    print("=" * 60)
    print("Search: 'docker OR kafka' (boolean OR)")
    print("=" * 60)
    results = engine.search("docker OR kafka")
    for hit in results:
        print(f"  [{hit.score:.4f}] {hit.doc['title']}")
        print()

    print("=" * 60)
    print("Search: 'title:python' (field-specific)")
    print("=" * 60)
    results = engine.search("title:python")
    for hit in results:
        print(f"  [{hit.score:.4f}] {hit.doc['title']}")
        print()

    print("=" * 60)
    print("Faceted search: 'python' with tag facets")
    print("=" * 60)
    results = engine.search("python", facets=["tags"])
    for hit in results:
        print(f"  [{hit.score:.4f}] {hit.doc['title']}")
    print(f"\n  Facets: {results.facets}")
    print()

    # Stats
    print("=" * 60)
    print("Index Stats")
    print("=" * 60)
    stats = engine.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
