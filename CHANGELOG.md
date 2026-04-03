# Changelog

## [0.1.1] - 2026-04-03

Initial stable release. Public API baseline established.

### Highlights
- LangChain integration for Apache AGE (graph) + pgvector (vector) on PostgreSQL
- Drop-in replacement for `langchain-neo4j` — same `query()`, `similarity_search()`, `from_existing_graph()`, `GraphCypherQAChain` API
- Three install modes: `[graph]`, `[vector]`, `[all]`
- 118 tests (65 unit + 53 integration), ruff 0 errors, mypy 0 errors
- Bilingual documentation (English + Korean)

### Since 0.0.6

### Added
- `traverse()` — WITH RECURSIVE deep hop traversal (10–22x faster than Cypher `*N`)
- `create_property_index()` — B-tree/GIN index on node properties
- mogrify-based pseudo parameter binding (`query("WHERE n.name = %s", ("Alice",))`)
- UNWIND batch pattern in `add_graph_documents()` (1 Cypher call per label)
- `refresh_schema()` via `ag_catalog` direct SQL (eliminates N+1 queries)
- Error-specific retry logic (SerializationFailure, DeadlockDetected, ConnectionFailure)
- `__del__` cleanup on garbage collection

## [0.0.5] - 2026-04-02

### Added
- Three-mode architecture: `pip install langchain-age[graph]`, `[vector]`, `[all]`
- Lazy imports via `__getattr__` — each mode works without the other's dependencies
- `similarity_search_with_relevance_scores()` with per-strategy score normalisation
- `embeddings` property (LangChain VectorStore convention)

### Removed
- `PostgresChatMessageHistory` — use `langgraph-checkpoint-postgres` instead

## [0.0.4] - 2026-04-02

### Changed
- Security hardening: `validate_sql_identifier()`, `escape_cypher_identifier()`, `escape_cypher_string()`
- Batch INSERT via `executemany()` with `batch_size=1000`
- `max_marginal_relevance_search()` reuses stored embeddings (no extra API call)
- Double-quoted SQL identifiers throughout
- OpenCypher-standard `''` string escaping (replaces non-standard `\'`)
- Context-manager support (`close()`, `__enter__`/`__exit__`)
- `_build_filter_clause()` depth guard (MAX_DEPTH=10)

## [0.0.3] - 2026-04-02

### Fixed
- PG18 compatibility: bypass `age_sdk.cypher()` which generates `cypher(NULL,NULL)`
- AGE Cypher reserved word handling via backtick quoting (25 words tested)
- LangChain v1 migration (`RunnableSerializable`, Pydantic v2)

## [0.0.2] - 2026-04-01

### Added
- apache-age-python SDK integration (psycopg3-based)
- Hybrid search (vector + full-text via RRF)
- MongoDB-style metadata filter operators

## [0.0.1] - 2026-04-01

### Added
- Initial release: AGEGraph, AGEVector, AGEGraphCypherQAChain
