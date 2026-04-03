# langchain-age — Agent Instructions

This file helps AI coding agents (Claude Code, Cursor, Copilot, Codex, etc.)
understand and use this library effectively.

## What This Library Does

`langchain-age` is a LangChain integration for **Apache AGE** (graph) and
**pgvector** (vector) on PostgreSQL. It is a drop-in replacement for
`langchain-neo4j` — same API, runs on PostgreSQL instead of Neo4j.

## Core Classes

### AGEGraph (Graph Store)
```python
from langchain_age import AGEGraph

graph = AGEGraph(
    "host=localhost port=5433 dbname=langchain_age user=langchain password=langchain",
    graph_name="my_graph",
)

# Execute Cypher (same as Neo4j)
graph.query("CREATE (:Person {name: 'Alice', age: 30})")
results = graph.query("MATCH (n:Person) RETURN n.name AS name")

# Parameter binding (mogrify-based, safe)
graph.query("MATCH (n) WHERE n.name = %s RETURN n", params=("Alice",))

# Deep traversal (10-22x faster than Cypher *N)
results = graph.traverse(
    start_label="Person", start_filter={"name": "Alice"},
    edge_label="KNOWS", max_depth=6,
)

# Schema
graph.refresh_schema()
print(graph.schema)

# GraphDocument upsert (UNWIND batch)
graph.add_graph_documents([graph_doc], include_source=True)

# Always close or use context manager
graph.close()
```

### AGEVector (Vector Store)
```python
from langchain_age import AGEVector, DistanceStrategy

store = AGEVector(
    connection_string="host=localhost port=5433 ...",
    embedding_function=embeddings,
    collection_name="my_docs",
    distance_strategy=DistanceStrategy.COSINE,  # or EUCLIDEAN, MAX_INNER_PRODUCT
)

# Add documents
store.add_texts(["text1", "text2"], metadatas=[{"key": "val"}, {}])

# Search
docs = store.similarity_search("query", k=5)
docs_with_scores = store.similarity_search_with_relevance_scores("query", k=5)

# Metadata filters (MongoDB-style)
docs = store.similarity_search("query", filter={"category": {"$in": ["a", "b"]}})

# Hybrid search (vector + full-text RRF)
from langchain_age import SearchType
store = AGEVector(..., search_type=SearchType.HYBRID)

# As LangChain retriever
retriever = store.as_retriever(search_kwargs={"k": 5})

# From existing graph nodes
store = AGEVector.from_existing_graph(
    embedding=embeddings, connection_string="...",
    graph_name="kg", node_label="Document",
    text_node_properties=["title", "content"],
)
```

### AGEGraphCypherQAChain
```python
from langchain_age import AGEGraph, AGEGraphCypherQAChain

chain = AGEGraphCypherQAChain.from_llm(
    llm, graph=graph,
    allow_dangerous_requests=True,  # REQUIRED
)
answer = chain.run("Who does Alice know?")
```

## Important: AGE-Specific Cypher Rules

When generating Cypher for AGE:
1. **No APOC** — AGE does not support APOC procedures
2. **Always alias RETURN values** — `RETURN n.name AS name`, not `RETURN n.name`
3. **Backtick reserved words** — `n.\`desc\``, `n.\`order\`` (25 reserved words)
4. **No `$param` binding** — Use `%s` with `params=` or `escape_cypher_string()`
5. **UNWIND works** — `UNWIND [{...}, {...}] AS row CREATE ...`
6. **No `elementId()`** — Use `id(n)` or return the node itself

## Security Utilities

```python
from langchain_age.utils.cypher import (
    escape_cypher_identifier,   # Backtick-quote: "desc" → "`desc`"
    escape_cypher_string,       # String escape: "it's" → "it''s"
    validate_sql_identifier,    # Table name check: raises ValueError if unsafe
    validate_cypher,            # Lightweight Cypher syntax check
)
```

## Project Structure

```
langchain_age/
├── __init__.py                  # Lazy imports
├── graphs/age_graph.py          # AGEGraph (GraphStore)
├── vectorstores/age_vector.py   # AGEVector (VectorStore)
├── chains/graph_cypher_qa_chain.py  # QA chain
├── utils/
│   ├── cypher.py                # SQL wrapping, escaping, validation
│   └── agtype.py                # Vertex/Edge/Path → dict conversion
└── _vendor/age/                 # Vendored apache-age-python SDK
```

## Database Setup

```bash
cd docker && docker compose up -d
# PostgreSQL 18 + Apache AGE 1.7.0 + pgvector + pg_trgm
# DSN: host=localhost port=5433 dbname=langchain_age user=langchain password=langchain
```

## Testing

```bash
pytest tests/unit/                # 65 tests, no DB
LANGCHAIN_AGE_TEST_DSN="..." pytest tests/integration/  # 53 tests
ruff check langchain_age/ tests/
mypy langchain_age/
```

## Common Patterns

### RAG with graph context
```python
# 1. Vector search
docs = store.similarity_search("query", k=3)

# 2. Expand via graph
for doc in docs:
    neighbors = graph.query(
        "MATCH (n)-[r]->(m) WHERE id(n) = %s RETURN type(r) AS rel, m.name AS name",
        params=(doc.metadata["age_node_id"],),
    )
```

### LangGraph on same DB
```python
from langgraph.store.postgres import PostgresStore
with PostgresStore.from_conn_string(same_dsn) as store:
    store.setup()  # Coexists with AGE + pgvector tables
```
