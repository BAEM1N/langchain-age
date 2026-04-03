# Getting Started

## Prerequisites

- Python 3.10+
- Docker (for the database)
- OpenAI API key (optional, for embedding/LLM features)

## 1. Start the Database

```bash
git clone https://github.com/BAEM1N/langchain-age.git
cd langchain-age/docker
docker compose up -d
```

This starts a single PostgreSQL 18 container with:
- **Apache AGE 1.7.0** — graph engine (Cypher support)
- **pgvector** — vector similarity search
- **pg_trgm** — trigram similarity for full-text

Verify it's running:

```bash
docker compose ps
# Should show "healthy"
```

## 2. Install the Library

Choose the mode you need:

```bash
# Graph only (AGEGraph + AGEGraphCypherQAChain)
pip install "langchain-age[graph]"

# Vector only (AGEVector + pgvector)
pip install "langchain-age[vector]"

# Everything
pip install "langchain-age[all]"
```

## 3. First Graph Query

```python
from langchain_age import AGEGraph

graph = AGEGraph(
    "host=localhost port=5433 dbname=langchain_age user=langchain password=langchain",
    graph_name="quickstart",
)

# Create nodes
graph.query("CREATE (:Person {name: 'Alice', role: 'engineer'})")
graph.query("CREATE (:Person {name: 'Bob', role: 'designer'})")
graph.query(
    "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) "
    "CREATE (a)-[:WORKS_WITH]->(b)"
)

# Query
results = graph.query(
    "MATCH (a:Person)-[:WORKS_WITH]->(b:Person) "
    "RETURN a.name AS from_person, b.name AS to_person"
)
print(results)
# [{'from_person': 'Alice', 'to_person': 'Bob'}]

graph.close()
```

## 4. First Vector Search

```python
from langchain_age import AGEVector, DistanceStrategy
from langchain_openai import OpenAIEmbeddings  # pip install langchain-openai

store = AGEVector(
    connection_string="host=localhost port=5433 dbname=langchain_age user=langchain password=langchain",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="quickstart_docs",
    distance_strategy=DistanceStrategy.COSINE,
)

# Add documents
store.add_texts([
    "Apache AGE adds graph query capabilities to PostgreSQL.",
    "pgvector enables fast vector similarity search.",
    "LangChain is a framework for building LLM applications.",
])

# Search
results = store.similarity_search("graph database", k=2)
for doc in results:
    print(doc.page_content)

store.close()
```

## 5. Combined Graph + Vector

```python
from langchain_age import AGEGraph, AGEVector

# Build a knowledge graph
graph = AGEGraph(conn_str, graph_name="kb")
graph.query("CREATE (:Topic {name: 'PostgreSQL', desc: 'relational database'})")
graph.query("CREATE (:Topic {name: 'AGE', desc: 'graph extension for PG'})")
graph.query(
    "MATCH (a:Topic {name: 'AGE'}), (b:Topic {name: 'PostgreSQL'}) "
    "CREATE (a)-[:EXTENDS]->(b)"
)

# Vectorise graph nodes
store = AGEVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    connection_string=conn_str,
    graph_name="kb",
    node_label="Topic",
    text_node_properties=["name", "desc"],
)

# Vector search → then expand via graph
docs = store.similarity_search("graph query", k=1)
print(docs[0].page_content)
# "AGE graph extension for PG"
print(docs[0].metadata)
# {"age_node_id": "...", "node_label": "Topic"}
```

## Next Steps

- [Tutorial](tutorial.md) — full walkthrough of all features
- [API Reference](api-reference.md) — class/method documentation
- [notebooks/](../../notebooks/) — Jupyter notebooks with runnable examples
