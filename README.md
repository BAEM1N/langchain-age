# langchain-age

**LangChain integration for [Apache AGE](https://age.apache.org/) (graph) + [pgvector](https://github.com/pgvector/pgvector) (vector) on PostgreSQL.**

Drop-in replacement for `langchain-neo4j` — same API, runs on PostgreSQL instead of Neo4j.

```python
from langchain_age import AGEGraph, AGEVector, AGEGraphCypherQAChain
```

## Installation

Three install modes depending on what you need:

```bash
# Graph only (Apache AGE)
pip install "langchain-age[graph]"

# Vector only (pgvector)
pip install "langchain-age[vector]"

# Everything (Graph + Vector)
pip install "langchain-age[all]"
```

> **Note**: The Apache AGE Python driver is installed from GitHub (the PyPI version is outdated).
> This is the [official SDK](https://github.com/apache/age/tree/master/drivers/python) maintained by the Apache AGE project.

## Quick Start

### 1. Start the database

```bash
cd docker
docker compose up -d
```

Single container: PostgreSQL 18 + Apache AGE 1.7.0 + pgvector.

### 2. Graph mode

```python
from langchain_age import AGEGraph

graph = AGEGraph(
    "host=localhost port=5433 dbname=langchain_age user=langchain password=langchain",
    graph_name="my_graph",
)

graph.query("CREATE (:Person {name: 'Alice', age: 30})")
results = graph.query("MATCH (n:Person) RETURN n.name AS name")
# [{'name': 'Alice'}]
```

### 3. Vector mode

```python
from langchain_age import AGEVector, DistanceStrategy

store = AGEVector(
    connection_string="host=localhost port=5433 dbname=langchain_age user=langchain password=langchain",
    embedding_function=my_embeddings,
    collection_name="docs",
    distance_strategy=DistanceStrategy.COSINE,
)

store.add_texts(["Apache AGE adds Cypher to PostgreSQL."])
results = store.similarity_search("graph database", k=5)
```

### 4. Graph + Vector (GraphRAG)

```python
# Vectorise existing graph nodes
store = AGEVector.from_existing_graph(
    embedding=my_embeddings,
    connection_string="...",
    graph_name="my_graph",
    node_label="Document",
    text_node_properties=["title", "content"],
)
```

### 5. Cypher QA Chain

```python
from langchain_age import AGEGraph, AGEGraphCypherQAChain
from langchain_openai import ChatOpenAI

chain = AGEGraphCypherQAChain.from_llm(
    ChatOpenAI(model="gpt-4o-mini"),
    graph=AGEGraph("...", "movies"),
    allow_dangerous_requests=True,
)
answer = chain.run("Which movies did Tom Hanks act in?")
```

### 6. Long-term Memory & Checkpointing

Uses the same PostgreSQL instance via [langgraph-checkpoint-postgres](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.postgres.PostgresSaver):

```python
from langgraph.store.postgres import PostgresStore

with PostgresStore.from_conn_string("host=localhost port=5433 ...") as store:
    store.setup()
    store.put(("users", "123"), "prefs", {"theme": "dark"})
```

AGE graph tables, pgvector tables, and LangGraph store tables coexist in the same database.

## Features

| Component | Class | Description |
|-----------|-------|-------------|
| **Graph** | `AGEGraph` | GraphStore backed by Apache AGE. Cypher execution, schema introspection, GraphDocument upserts. |
| **Vector** | `AGEVector` | VectorStore backed by pgvector. Cosine/L2/IP distance, HNSW & IVFFlat indexes, hybrid search (vector + full-text via RRF), MMR. |
| **Chain** | `AGEGraphCypherQAChain` | LLM generates Cypher, executes against AGE, returns natural-language answer. |

### Security

- SQL identifier validation at construction (`validate_sql_identifier`)
- Cypher identifier backtick-quoting for all 25 AGE reserved words (`escape_cypher_identifier`)
- OpenCypher-standard string escaping with `''` doubling (`escape_cypher_string`)
- `allow_dangerous_requests` gate on the QA chain
- Double-quoted SQL table/index names throughout

### langchain-neo4j API Compatibility

| Feature | langchain-neo4j | langchain-age |
|---------|----------------|---------------|
| `from_existing_graph()` | `Neo4jVector.from_existing_graph()` | `AGEVector.from_existing_graph()` |
| `from_existing_index()` | `Neo4jVector.from_existing_index()` | `AGEVector.from_existing_index()` |
| `similarity_search_with_relevance_scores()` | Yes | Yes |
| `as_retriever()` | Yes | Yes |
| Hybrid search | Lucene full-text | PostgreSQL tsvector + RRF |
| `include_types` / `exclude_types` | Yes | Yes |
| `add_graph_documents()` | Yes | Yes |
| Context manager | Yes | Yes |
| Batch insert | `UNWIND ... IN TRANSACTIONS OF 1000 ROWS` | `executemany` with `batch_size=1000` |

## AGE vs Neo4j

| | Neo4j | Apache AGE |
|---|---|---|
| Cypher execution | Bolt protocol | SQL-wrapped: `SELECT * FROM cypher(...)` |
| Connection | `neo4j://` | PostgreSQL DSN |
| Vector search | Native index | pgvector extension |
| APOC | Available | Not available |
| Data types | Native graph | `agtype` (JSON superset) |
| Parameterised Cypher | Yes (`$param`) | Not supported (values inlined) |

`langchain-age` handles SQL wrapping automatically — you write plain Cypher.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [01_graph.ipynb](notebooks/01_graph.ipynb) | AGEGraph: Cypher CRUD, schema, GraphDocument |
| [02_vector.ipynb](notebooks/02_vector.ipynb) | AGEVector: similarity, hybrid, MMR, filters, HNSW |
| [03_graph_vector.ipynb](notebooks/03_graph_vector.ipynb) | GraphRAG: from_existing_graph, QA chain |
| [04_neo4j_vs_age.ipynb](notebooks/04_neo4j_vs_age.ipynb) | **Neo4j vs AGE 병렬 비교** — 동일 작업을 나란히 실행 |

01은 API 키 불필요. 02~04는 OpenAI API 키를 `getpass`로 입력.

## Running Tests

```bash
# Unit tests (no DB required) — 65 tests
pytest tests/unit/

# Integration tests (requires Docker container) — 43 tests
export LANGCHAIN_AGE_TEST_DSN="host=localhost port=5433 dbname=langchain_age user=langchain password=langchain"
pytest tests/integration/
```

## Project Structure

```
langchain-age/
├── docker/                          # PG18 + AGE + pgvector container
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── init/01_init_extensions.sql
├── langchain_age/
│   ├── __init__.py                  # Lazy imports (3-mode support)
│   ├── graphs/
│   │   └── age_graph.py             # AGEGraph (GraphStore)
│   ├── vectorstores/
│   │   └── age_vector.py            # AGEVector (VectorStore)
│   ├── chains/
│   │   └── graph_cypher_qa_chain.py # AGEGraphCypherQAChain
│   └── utils/
│       ├── agtype.py                # Vertex/Edge/Path → dict conversion
│       └── cypher.py                # SQL wrapping, escaping, validation
├── tests/
│   ├── unit/                        # 65 tests, no DB
│   └── integration/                 # 43 tests, live DB
└── pyproject.toml
```

## Python Support

Tested on Python 3.10, 3.11, 3.12, 3.13, 3.14.

## License

MIT
