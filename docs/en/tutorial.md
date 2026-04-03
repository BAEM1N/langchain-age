# Tutorial

Complete walkthrough of all langchain-age features.

## Table of Contents

1. [Graph Operations](#1-graph-operations)
2. [Vector Operations](#2-vector-operations)
3. [Hybrid Search](#3-hybrid-search)
4. [Graph + Vector (GraphRAG)](#4-graph--vector-graphrag)
5. [Cypher QA Chain](#5-cypher-qa-chain)
6. [Deep Traversal](#6-deep-traversal)
7. [Metadata Filtering](#7-metadata-filtering)
8. [Performance Tuning](#8-performance-tuning)
9. [LangGraph Integration](#9-langgraph-integration)

---

## 1. Graph Operations

### Connection

```python
from langchain_age import AGEGraph

# Basic connection
graph = AGEGraph(
    "host=localhost port=5433 dbname=langchain_age user=langchain password=langchain",
    graph_name="tutorial",
)

# Context manager (auto-close)
with AGEGraph(conn_str, "tutorial") as graph:
    graph.query("MATCH (n) RETURN count(n) AS total")
```

### CRUD

```python
# Create nodes
graph.query("CREATE (:Person {name: 'Alice', age: 30})")

# Read
results = graph.query("MATCH (n:Person) RETURN n.name AS name, n.age AS age")

# Update
graph.query("MATCH (n:Person {name: 'Alice'}) SET n.age = 31")

# Delete
graph.query("MATCH (n:Person {name: 'Alice'}) DELETE n")
```

### Parameter Binding (mogrify)

AGE does not support native `$param` binding, but langchain-age provides
safe value escaping via psycopg3's `mogrify`:

```python
# Safe — values are escaped by psycopg3
graph.query(
    "MATCH (n:Person) WHERE n.name = %s RETURN n.age AS age",
    params=("Alice",),
)

# Also safe — manual escaping
from langchain_age.utils.cypher import escape_cypher_string
name = escape_cypher_string(user_input)
graph.query(f"MATCH (n:Person {{name: '{name}'}}) RETURN n")
```

### Schema Introspection

```python
graph.refresh_schema()
print(graph.schema)
# Node labels and properties:
#   :Person {age, name}
# Relationship types and properties:
#   [:KNOWS] {since}
# Relationship patterns:
#   (:Person)-[:KNOWS]->(:Person)

# Programmatic access
schema = graph.structured_schema
print(schema["node_props"])    # {"Person": ["age", "name"]}
print(schema["relationships"]) # [{"start": "Person", "type": "KNOWS", "end": "Person"}]
```

### GraphDocument Upsert

```python
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

doc = GraphDocument(
    nodes=[
        Node(id="alice", type="Person", properties={"name": "Alice"}),
        Node(id="bob", type="Person", properties={"name": "Bob"}),
    ],
    relationships=[
        Relationship(
            source=Node(id="alice", type="Person"),
            target=Node(id="bob", type="Person"),
            type="KNOWS",
        ),
    ],
    source=Document(page_content="Alice knows Bob"),
)

# Batch insert — uses UNWIND internally
graph.add_graph_documents([doc], include_source=True)
```

---

## 2. Vector Operations

### Basic Similarity Search

```python
from langchain_age import AGEVector, DistanceStrategy
from langchain_openai import OpenAIEmbeddings

store = AGEVector(
    connection_string=conn_str,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="docs",
    distance_strategy=DistanceStrategy.COSINE,
)

store.add_texts(["PostgreSQL is powerful.", "AGE adds graph queries."])

# Similarity search
docs = store.similarity_search("database", k=2)

# With distance scores (lower = more similar)
results = store.similarity_search_with_score("database", k=2)

# With relevance scores (0-1, higher = more similar)
results = store.similarity_search_with_relevance_scores("database", k=2)
```

### MMR (Maximal Marginal Relevance)

Balances relevance and diversity. Reuses stored embeddings — no extra API call.

```python
docs = store.max_marginal_relevance_search(
    "database technology",
    k=3,          # return 3 docs
    fetch_k=10,   # consider top 10 candidates
    lambda_mult=0.5,  # 0=max diversity, 1=max relevance
)
```

### Index Management

```python
# HNSW (recommended for production)
store.create_hnsw_index(m=16, ef_construction=64)

# IVFFlat (faster build, slightly less accurate)
store.create_ivfflat_index(n_lists=100)

# Drop all indexes
store.drop_index()
```

### LangChain Retriever

```python
retriever = store.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("What is AGE?")

# Use in a RAG chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template(
    "Answer based on context:\n{context}\n\nQuestion: {question}"
)
chain = prompt | ChatOpenAI() | StrOutputParser()
```

---

## 3. Hybrid Search

Combines vector similarity and PostgreSQL full-text search via RRF
(Reciprocal Rank Fusion, k=60).

```python
from langchain_age import AGEVector, SearchType

store = AGEVector(
    connection_string=conn_str,
    embedding_function=embeddings,
    collection_name="hybrid_docs",
    search_type=SearchType.HYBRID,  # Enable hybrid mode
)

store.add_texts([
    "PostgreSQL supports JSON and full-text search.",
    "Apache AGE adds Cypher graph queries to PostgreSQL.",
    "pgvector enables vector similarity search.",
])

# Automatically combines vector + keyword matching
results = store.similarity_search("PostgreSQL graph extensions", k=3)
```

---

## 4. Graph + Vector (GraphRAG)

### Vectorise Existing Graph Nodes

```python
store = AGEVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    connection_string=conn_str,
    graph_name="tutorial",
    node_label="Person",
    text_node_properties=["name", "bio"],  # concatenated as document text
    collection_name="person_vectors",
)
```

### Vector Search → Graph Context Expansion

```python
# Step 1: Find relevant nodes via vector search
docs = store.similarity_search("engineer", k=2)

# Step 2: Expand via graph relationships
for doc in docs:
    node_label = doc.metadata["node_label"]
    # Use the graph to find related nodes
    neighbors = graph.query(
        f"MATCH (n:{node_label})-[r]->(m) "
        f"WHERE id(n) = {doc.metadata['age_node_id']} "
        f"RETURN type(r) AS rel, m.name AS neighbor"
    )
```

---

## 5. Cypher QA Chain

LLM generates Cypher → AGE executes → LLM answers in natural language.

```python
from langchain_age import AGEGraph, AGEGraphCypherQAChain
from langchain_openai import ChatOpenAI

graph = AGEGraph(conn_str, "tutorial")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = AGEGraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    allow_dangerous_requests=True,
    return_intermediate_steps=True,
    verbose=True,
)

result = chain.invoke({"query": "Who does Alice work with?"})
print(result["result"])                         # Natural language answer
print(result["intermediate_steps"][0]["query"])  # Generated Cypher
```

### Schema Filtering

```python
# Only expose certain types to the LLM
chain = AGEGraphCypherQAChain.from_llm(
    llm, graph=graph,
    include_types=["Person", "KNOWS"],   # whitelist
    # exclude_types=["InternalNode"],     # or blacklist
    allow_dangerous_requests=True,
)
```

---

## 6. Deep Traversal

`traverse()` uses PostgreSQL `WITH RECURSIVE` — 10–22x faster than
Cypher `*N` variable-length paths.

```python
# Find all nodes reachable within 6 hops
results = graph.traverse(
    start_label="Person",
    start_filter={"name": "Alice"},
    edge_label="KNOWS",
    max_depth=6,
    direction="outgoing",      # "incoming" or "both"
    return_properties=True,
)

for r in results:
    print(f"  depth={r['depth']} → {r['properties']}")
```

### Property Indexes

Speed up start-node lookup for `traverse()`:

```python
# B-tree index on a specific property
graph.create_property_index("Person", "name")

# GIN index on all properties (slower to build, supports all operators)
graph.create_property_index("Person", "name", index_type="gin")
```

---

## 7. Metadata Filtering

MongoDB-style filter operators on JSONB metadata:

```python
# Equality
store.similarity_search("query", filter={"author": "Alice"})

# Operators
store.similarity_search("query", filter={"year": {"$gte": 2024}})
store.similarity_search("query", filter={"tag": {"$in": ["ai", "ml"]}})
store.similarity_search("query", filter={"score": {"$between": [0.5, 1.0]}})
store.similarity_search("query", filter={"title": {"$ilike": "%graph%"}})

# Logical combinations
store.similarity_search("query", filter={
    "$and": [
        {"author": "Alice"},
        {"year": {"$gte": 2024}},
    ]
})
```

Supported: `$eq`, `$ne`, `$lt`, `$lte`, `$gt`, `$gte`, `$in`, `$nin`,
`$between`, `$like`, `$ilike`, `$exists`, `$and`, `$or`.

---

## 8. Performance Tuning

### Batch Size

```python
store = AGEVector(
    ...,
    batch_size=5000,  # default is 1000
)
```

### HNSW Parameters

```python
# Higher m = better recall, more memory
# Higher ef_construction = better quality, slower build
store.create_hnsw_index(m=32, ef_construction=128)
```

### Schema Refresh

`refresh_schema()` queries `ag_catalog` system tables directly (not
per-label Cypher), so it scales well even with hundreds of labels.

### Deep Traversal vs Cypher

| Pattern | Method | When to use |
|---------|--------|-------------|
| 1–3 hops | `graph.query("MATCH ...*3...")` | Simple, readable |
| 4+ hops | `graph.traverse(max_depth=N)` | 10–22x faster |

---

## 9. LangGraph Integration

langchain-age uses the same PostgreSQL instance as LangGraph's
`PostgresStore` and `PostgresSaver`. All tables coexist.

```python
from langgraph.store.postgres import PostgresStore

# Same connection string as AGEGraph / AGEVector
with PostgresStore.from_conn_string(conn_str) as store:
    store.setup()
    store.put(("users", "123"), "prefs", {"theme": "dark"})
    item = store.get(("users", "123"), "prefs")
```

No extra database needed — graph, vector, and long-term memory in one PostgreSQL.
