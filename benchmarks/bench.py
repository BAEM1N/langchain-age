"""Neo4j vs AGE fair benchmark.

Measures identical operations on both databases under the same conditions.
No tricks, no AGE-specific optimisations — pure Cypher on both sides.

Usage:
    python benchmarks/bench.py
"""
from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from typing import Any, Callable

# ── Connections ──────────────────────────────────────────────────

AGE_DSN = "host=localhost port=5433 dbname=langchain_age user=langchain password=langchain"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "testpassword"
GRAPH_NAME = "bench_graph"


def get_age():
    from langchain_age import AGEGraph
    return AGEGraph(AGE_DSN, GRAPH_NAME, refresh_schema=False)


def get_neo4j():
    from langchain_neo4j import Neo4jGraph
    return Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)


# ── Benchmark harness ───────────────────────────────────────────

def bench(fn: Callable, iterations: int = 50) -> dict[str, float]:
    """Run fn() `iterations` times, return timing stats in ms."""
    # Warmup
    for _ in range(3):
        fn()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)

    return {
        "p50": statistics.median(times),
        "p99": sorted(times)[int(len(times) * 0.99)],
        "mean": statistics.mean(times),
        "min": min(times),
        "max": max(times),
    }


def fmt(stats: dict[str, float]) -> str:
    return f"p50={stats['p50']:.1f}ms  p99={stats['p99']:.1f}ms  mean={stats['mean']:.1f}ms"


def run_test(name: str, neo4j_fn: Callable, age_fn: Callable, iterations: int = 50):
    print(f"\n{'─' * 60}")
    print(f"  {name} ({iterations} iterations)")
    print(f"{'─' * 60}")
    neo4j_stats = bench(neo4j_fn, iterations)
    age_stats = bench(age_fn, iterations)
    ratio = age_stats["p50"] / neo4j_stats["p50"] if neo4j_stats["p50"] > 0 else 0
    print(f"  Neo4j: {fmt(neo4j_stats)}")
    print(f"  AGE:   {fmt(age_stats)}")
    if ratio > 1:
        print(f"  → Neo4j {ratio:.1f}x faster")
    else:
        print(f"  → AGE {1/ratio:.1f}x faster")
    return {"name": name, "neo4j": neo4j_stats, "age": age_stats}


# ── Setup: seed identical data ──────────────────────────────────

def seed_data(neo4j, age, node_count: int = 1000):
    """Insert identical graph data into both databases."""
    print(f"\nSeeding {node_count} nodes + relationships...")

    # Clean
    neo4j.query("MATCH (n) DETACH DELETE n")
    try:
        age.query("MATCH (n) DETACH DELETE n")
    except Exception:
        pass

    # Nodes — batch 100 at a time
    batch_size = 100
    for start in range(0, node_count, batch_size):
        end = min(start + batch_size, node_count)
        # Neo4j: UNWIND
        neo4j.query(
            "UNWIND range($s, $e) AS i "
            "CREATE (:Node {idx: i, name: 'node_' + toString(i)})",
            params={"s": start, "e": end - 1},
        )
        # AGE: UNWIND (also supported)
        items = ", ".join(f"{{idx: {i}, name: 'node_{i}'}}" for i in range(start, end))
        age.query(f"UNWIND [{items}] AS row CREATE (:Node {{idx: row.idx, name: row.name}})")

    # Relationships: each node i → (i*7+13) % node_count
    for start in range(0, node_count, batch_size):
        end = min(start + batch_size, node_count)
        neo4j.query(
            "UNWIND range($s, $e) AS i "
            "MATCH (a:Node {idx: i}), (b:Node {idx: (i * 7 + 13) % $n}) "
            "CREATE (a)-[:LINK]->(b)",
            params={"s": start, "e": end - 1, "n": node_count},
        )
        items = ", ".join(str(i) for i in range(start, end))
        age.query(
            f"UNWIND [{items}] AS i "
            f"MATCH (a:Node {{idx: i}}), (b:Node {{idx: (i * 7 + 13) % {node_count}}}) "
            f"CREATE (a)-[:LINK]->(b)"
        )

    # Second relationship set for branching
    for start in range(0, node_count, batch_size):
        end = min(start + batch_size, node_count)
        neo4j.query(
            "UNWIND range($s, $e) AS i "
            "MATCH (a:Node {idx: i}), (b:Node {idx: (i * 13 + 7) % $n}) "
            "CREATE (a)-[:LINK]->(b)",
            params={"s": start, "e": end - 1, "n": node_count},
        )
        items = ", ".join(str(i) for i in range(start, end))
        age.query(
            f"UNWIND [{items}] AS i "
            f"MATCH (a:Node {{idx: i}}), (b:Node {{idx: (i * 13 + 7) % {node_count}}}) "
            f"CREATE (a)-[:LINK]->(b)"
        )

    # Verify
    neo4j_count = neo4j.query("MATCH (n:Node) RETURN count(n) AS c")[0]["c"]
    age_count = age.query("MATCH (n:Node) RETURN count(n) AS c")[0]["c"]
    print(f"  Neo4j: {neo4j_count} nodes")
    print(f"  AGE:   {age_count} nodes")
    assert neo4j_count == age_count, "Node counts don't match!"


# ── Tests ───────────────────────────────────────────────────────

def main():
    neo4j = get_neo4j()
    age = get_age()

    print("=" * 60)
    print("  Neo4j vs AGE — Fair Benchmark")
    print("  Same Cypher, same data, same hardware")
    print("=" * 60)

    seed_data(neo4j, age, node_count=1000)

    results = []

    # 1. Point lookup
    results.append(run_test(
        "Point lookup (MATCH by property)",
        lambda: neo4j.query("MATCH (n:Node {idx: 42}) RETURN n.name AS name"),
        lambda: age.query("MATCH (n:Node {idx: 42}) RETURN n.name AS name"),
    ))

    # 2. 1-hop traversal
    results.append(run_test(
        "1-hop traversal",
        lambda: neo4j.query("MATCH (a:Node {idx: 0})-[:LINK]->(b) RETURN count(b) AS c"),
        lambda: age.query("MATCH (a:Node {idx: 0})-[:LINK]->(b) RETURN count(b) AS c"),
    ))

    # 3. 3-hop traversal
    results.append(run_test(
        "3-hop traversal",
        lambda: neo4j.query("MATCH (a:Node {idx: 0})-[:LINK*3]->(b) RETURN count(DISTINCT b) AS c"),
        lambda: age.query("MATCH (a:Node {idx: 0})-[:LINK*3]->(b) RETURN count(DISTINCT b) AS c"),
        iterations=30,
    ))

    # 4. 6-hop traversal
    results.append(run_test(
        "6-hop traversal",
        lambda: neo4j.query("MATCH (a:Node {idx: 0})-[:LINK*6]->(b) RETURN count(DISTINCT b) AS c"),
        lambda: age.query("MATCH (a:Node {idx: 0})-[:LINK*6]->(b) RETURN count(DISTINCT b) AS c"),
        iterations=20,
    ))

    # 5. Aggregation
    results.append(run_test(
        "Full count aggregation",
        lambda: neo4j.query("MATCH (n:Node) RETURN count(n) AS c"),
        lambda: age.query("MATCH (n:Node) RETURN count(n) AS c"),
    ))

    # 6. Node creation (single)
    counter = {"n": 10000}

    def neo4j_create():
        counter["n"] += 1
        neo4j.query(f"CREATE (:Temp {{idx: {counter['n']}}})")

    def age_create():
        counter["n"] += 1
        age.query(f"CREATE (:Temp {{idx: {counter['n']}}})")

    results.append(run_test("Single node CREATE", neo4j_create, age_create, iterations=100))

    # 7. Batch create (100 nodes)
    results.append(run_test(
        "Batch CREATE (100 nodes)",
        lambda: neo4j.query("UNWIND range(0, 99) AS i CREATE (:BatchTemp {idx: i})"),
        lambda: age.query(
            "UNWIND [" + ", ".join(f"{{idx: {i}}}" for i in range(100)) + "] AS row "
            "CREATE (:BatchTemp {idx: row.idx})"
        ),
        iterations=20,
    ))

    # 8. Schema introspection
    results.append(run_test(
        "Schema refresh",
        lambda: neo4j.refresh_schema(),
        lambda: age.refresh_schema(),
        iterations=10,
    ))

    # Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"{'Test':<35} {'Neo4j p50':>12} {'AGE p50':>12} {'Winner':>10}")
    print("─" * 69)
    for r in results:
        n = r["neo4j"]["p50"]
        a = r["age"]["p50"]
        winner = "Neo4j" if n < a else "AGE"
        ratio = max(n, a) / min(n, a) if min(n, a) > 0 else 0
        print(f"{r['name']:<35} {n:>9.1f}ms {a:>9.1f}ms {winner:>6} {ratio:.1f}x")

    # Cleanup
    neo4j.query("MATCH (n) DETACH DELETE n")
    age.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
