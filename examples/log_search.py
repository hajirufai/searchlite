"""
Log search example — index server logs and search by patterns.
Demonstrates keyword fields, filtering, and faceted search.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searchlite import SearchEngine, Schema, TextField, KeywordField


# Simulated server log entries
LOG_ENTRIES = [
    {"message": "GET /api/users 200 OK response time 45ms", "level": "info", "service": "api-gateway"},
    {"message": "POST /api/auth/login 200 OK user authenticated successfully", "level": "info", "service": "auth-service"},
    {"message": "GET /api/orders 500 Internal Server Error database connection timeout", "level": "error", "service": "order-service"},
    {"message": "Connection pool exhausted waiting for available connection", "level": "error", "service": "order-service"},
    {"message": "GET /api/products 200 OK cached response served in 2ms", "level": "info", "service": "product-service"},
    {"message": "WARNING disk usage at 85 percent on volume data-01", "level": "warning", "service": "monitoring"},
    {"message": "POST /api/payments 201 Created payment processed successfully", "level": "info", "service": "payment-service"},
    {"message": "SSL certificate expiring in 7 days for api.example.com", "level": "warning", "service": "monitoring"},
    {"message": "GET /api/users/123 404 Not Found user does not exist", "level": "warning", "service": "api-gateway"},
    {"message": "Memory usage spike detected 92 percent RSS on worker-3", "level": "error", "service": "monitoring"},
    {"message": "Rate limit exceeded for IP 192.168.1.100 throttling requests", "level": "warning", "service": "api-gateway"},
    {"message": "Database migration completed successfully 15 tables updated", "level": "info", "service": "order-service"},
]


def main():
    schema = Schema(
        message=TextField(),
        level=KeywordField(faceted=True),
        service=KeywordField(faceted=True),
    )

    engine = SearchEngine(schema=schema)

    print("Indexing log entries...")
    for entry in LOG_ENTRIES:
        engine.add(entry)
    print(f"Indexed {len(LOG_ENTRIES)} log entries\n")

    # Search for errors
    print("=" * 60)
    print("Search: 'error timeout' — finding connection issues")
    print("=" * 60)
    results = engine.search("error timeout", facets=["level", "service"])
    for hit in results:
        print(f"  [{hit.doc['level']:>7}] [{hit.doc['service']}] {hit.doc['message']}")
    print(f"  Facets: {results.facets}\n")

    # Filter by service
    print("=" * 60)
    print("Search: all entries, filtered to 'monitoring' service")
    print("=" * 60)
    results = engine.search("*", filters={"service": ["monitoring"]})
    for hit in results:
        print(f"  [{hit.doc['level']:>7}] {hit.doc['message']}")
    print()

    # Search with facets
    print("=" * 60)
    print("Search: 'api' — with service and level facets")
    print("=" * 60)
    results = engine.search("api", facets=["service", "level"])
    for hit in results:
        print(f"  [{hit.doc['level']:>7}] [{hit.doc['service']}] {hit.doc['message']}")
    print(f"\n  Service breakdown: {results.facets.get('service', {})}")
    print(f"  Level breakdown:   {results.facets.get('level', {})}")


if __name__ == "__main__":
    main()
