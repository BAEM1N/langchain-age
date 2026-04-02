# langchain-age

**LangChain integration for [Apache AGE](https://age.apache.org/) (graph) + [pgvector](https://github.com/pgvector/pgvector) (vector) on PostgreSQL.**

Mirrors the `langchain-neo4j` API so teams already familiar with the Neo4j integration can switch with minimal friction.

---

## Features

| Component | Description |
|---|---|
| `AGEGraph` | `GraphStore` backed by PostgreSQL + Apache AGE. Executes Cypher via the `cypher()` SQL wrapper, introspects schema, and supports `GraphDocument` upserts. |
| `AGEVector` | `VectorStore` backed by pgvector. Supports cosine / L2 / inner-product similarity, HNSW & IVFFlat indexing, MMR search, and optional linkage to AGE graph nodes. |
| `AGEGraphCypherQAChain` | LLM chain that auto-generates Cypher for Apache AGE, executes it, and returns a natural-language answer. Drop-in replacement for `GraphCypherQAChain`. |

---

## Quick Start

### 1. Start the database

```bash
cd docker
cp .env.example .env          # edit credentials if needed
docker compose up -d
```

This spins up a single PostgreSQL container with both **Apache AGE** and **pgvector** pre-installed.

### 2. Install the library

```bash
pip install -e ".[dev]"
```

### 3. Use AGEGraph

```python
from langchain_age import AGEGraph

graph = AGEGraph(
    connection_string="host=localhost port=5432 dbname=langchain_age user=langchain password=langchain",
    graph_name="my_graph",
)

# Run Cypher directly
graph.query("CREATE (:Person {name: 'Alice'})")
results = graph.query("MATCH (n:Person) RETURN n.name AS name")
print(results)  # [{'name': 'Alice'}]

# Inspect the schema
print(graph.schema)
```

### 4. Use AGEVector

```python
from langchain_age import AGEVector, DistanceStrategy
from langchain_openai import OpenAIEmbeddings

store = AGEVector(
    connection_string="...",
    embedding_function=OpenAIEmbeddings(),
    collection_name="my_vectors",
    distance_strategy=DistanceStrategy.COSINE,
)

store.add_texts(["Apache AGE adds Cypher to PostgreSQL.", "pgvector enables vector search."])

results = store.similarity_search("graph database", k=2)
for doc in results:
    print(doc.page_content)
```

### 5. Use AGEGraphCypherQAChain

```python
from langchain_age import AGEGraph, AGEGraphCypherQAChain
from langchain_openai import ChatOpenAI

graph = AGEGraph(connection_string="...", graph_name="movies")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = AGEGraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    allow_dangerous_requests=True,
    verbose=True,
)

answer = chain.run("Which movies did Tom Hanks act in?")
print(answer)
```

---

## AGE vs Neo4j – Key Differences

| | Neo4j | Apache AGE |
|---|---|---|
| Cypher execution | Direct Cypher protocol | Wrapped in SQL: `SELECT * FROM cypher('graph', $$ ... $$) AS (col agtype)` |
| Connection | Bolt protocol (`neo4j://`) | PostgreSQL DSN / URI |
| Vector search | Native vector index | pgvector extension |
| APOC procedures | Available | **Not available** |
| Data type | Native graph types | `agtype` (superset of JSON) |

`langchain-age` handles the SQL wrapping automatically — you write plain Cypher.

---

## Running Tests

```bash
# Unit tests (no DB required)
pytest tests/unit/

# Integration tests (requires Docker container)
export LANGCHAIN_AGE_TEST_DSN="host=localhost port=5432 dbname=langchain_age user=langchain password=langchain"
pytest tests/integration/
```

---

## Project Structure

```
langchain-age/
├── docker/                    # PostgreSQL + AGE + pgvector container
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── init/
│       └── 01_init_extensions.sql
├── langchain_age/
│   ├── __init__.py
│   ├── graphs/
│   │   └── age_graph.py       # AGEGraph (GraphStore)
│   ├── vectorstores/
│   │   └── age_vector.py      # AGEVector (VectorStore)
│   ├── chains/
│   │   └── graph_cypher_qa_chain.py  # AGEGraphCypherQAChain
│   └── utils/
│       ├── agtype.py          # agtype ↔ Python conversion
│       └── cypher.py          # Cypher wrapping & validation
├── tests/
│   ├── unit/                  # Pure Python, no DB
│   └── integration/           # Requires live DB
├── examples/
│   ├── 01_basic_graph_qa.py
│   ├── 02_vector_search.py
│   └── 03_graph_rag.py        # Combined GraphRAG pattern
└── pyproject.toml
```

---

## License

MIT
