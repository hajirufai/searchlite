"""
Wikipedia search example — index article summaries and search them.
Demonstrates real-world usage with longer text content.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searchlite import SearchEngine, Schema, TextField, KeywordField


# Sample Wikipedia-style articles
ARTICLES = [
    {
        "title": "Python (programming language)",
        "body": (
            "Python is a high-level, general-purpose programming language. Its design "
            "philosophy emphasizes code readability with the use of significant indentation. "
            "Python is dynamically typed and garbage-collected. It supports multiple programming "
            "paradigms, including structured, object-oriented and functional programming. "
            "It was created by Guido van Rossum and first released in 1991. Python consistently "
            "ranks as one of the most popular programming languages."
        ),
        "category": "programming",
    },
    {
        "title": "Apache Kafka",
        "body": (
            "Apache Kafka is a distributed event store and stream-processing platform. "
            "The project was developed by LinkedIn and donated to the Apache Software Foundation. "
            "Kafka is used for building real-time data pipelines and streaming applications. "
            "It is horizontally scalable, fault-tolerant, and runs in production in thousands "
            "of companies. Kafka stores streams of records in categories called topics."
        ),
        "category": "data-engineering",
    },
    {
        "title": "MapReduce",
        "body": (
            "MapReduce is a programming model for processing and generating big data sets with "
            "a parallel, distributed algorithm on a cluster. It was originally developed at "
            "Google. The model consists of a Map function that processes key-value pairs to "
            "generate intermediate pairs, and a Reduce function that merges all intermediate "
            "values associated with the same key. This enables distributed computing on "
            "large datasets across many machines."
        ),
        "category": "distributed-systems",
    },
    {
        "title": "PostgreSQL",
        "body": (
            "PostgreSQL is a free and open-source relational database management system "
            "emphasizing extensibility and SQL compliance. It handles workloads ranging from "
            "single-machine applications to data warehouses with many concurrent users. "
            "PostgreSQL features transactions with ACID properties, automatically updatable "
            "views, materialized views, triggers, foreign keys, and stored procedures. "
            "It is designed to handle a range of workloads, from single machines to data "
            "warehouses or web services with many concurrent users."
        ),
        "category": "database",
    },
    {
        "title": "Nairobi",
        "body": (
            "Nairobi is the capital and largest city of Kenya. The city and its surrounding "
            "area also form Nairobi County. Nairobi is the political, financial, and cultural "
            "center of Kenya. The city is known for Nairobi National Park, a large game reserve "
            "within the city limits. Nairobi is also home to the United Nations Environment "
            "Programme and many international NGOs operating in East Africa."
        ),
        "category": "geography",
    },
    {
        "title": "Machine Learning",
        "body": (
            "Machine learning is a subset of artificial intelligence that provides systems "
            "the ability to automatically learn and improve from experience without being "
            "explicitly programmed. It focuses on the development of computer programs that "
            "can access data and use it to learn for themselves. The process begins with "
            "observations or data, such as examples, direct experience, or instruction. "
            "Common approaches include supervised learning, unsupervised learning, and "
            "reinforcement learning."
        ),
        "category": "ai",
    },
]


def main():
    schema = Schema(
        title=TextField(boost=3.0),
        body=TextField(),
        category=KeywordField(faceted=True),
    )

    engine = SearchEngine(schema=schema)

    print("Indexing Wikipedia articles...")
    for article in ARTICLES:
        engine.add(article)
    print(f"Indexed {len(ARTICLES)} articles\n")

    # Demonstrate various search capabilities
    searches = [
        ("python programming", "Simple multi-term search"),
        ('"data pipelines"', "Phrase query"),
        ("database OR distributed", "Boolean OR"),
        ("python AND learning", "Boolean AND"),
        ("NOT python", "Boolean NOT"),
        ("title:kafka", "Field-specific search"),
        ("kenya nairobi", "Geographic search"),
    ]

    for query, description in searches:
        print(f"{'─' * 60}")
        print(f"Query: {query}  ({description})")
        print(f"{'─' * 60}")
        results = engine.search(query, facets=["category"])

        if not results.hits:
            print("  No results found.\n")
            continue

        for hit in results:
            print(f"  Score: {hit.score:.4f}")
            print(f"  Title: {hit.doc['title']}")
            print(f"  Snippet: {hit.highlight('body', fragment_size=100)}")
            print()

        if results.facets:
            cats = results.facets.get("category", {})
            if cats:
                print(f"  Categories: {cats}")
        print()


if __name__ == "__main__":
    main()
