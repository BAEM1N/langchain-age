# GraphRAG with Just PostgreSQL — No Neo4j Required

*How to build a production-ready Graph + Vector RAG pipeline using only PostgreSQL, Apache AGE, and pgvector.*

---

## The Problem

You want GraphRAG — the pattern where an LLM retrieves context from both a knowledge graph and vector embeddings before generating an answer. It's powerful. The most-cited approach uses Neo4j.

But Neo4j means:
- **Another database** to deploy, monitor, back up, and secure
- **$15K–$100K/year** for Enterprise (HA, RBAC, online backup)
- **GPL licensing** headaches for commercial products
- **Two connection strings** — one for your app's PostgreSQL, another for Neo4j

What if your existing PostgreSQL could do both?

## The Answer: Apache AGE + pgvector

[Apache AGE](https://age.apache.org/) is a PostgreSQL extension that adds Cypher graph queries. [pgvector](https://github.com/pgvector/pgvector) adds vector similarity search. Both run inside the same PostgreSQL instance you're already using.

```
┌─────────────────────────────────────┐
│          PostgreSQL 18              │
│                                     │
│  ┌───────────┐  ┌────────────────┐  │
│  │ Apache AGE│  │   pgvector     │  │
│  │  (Cypher) │  │  (Embeddings)  │  │
│  └───────────┘  └────────────────┘  │
│                                     │
│  ┌────────────────────────────────┐  │
│  │   LangGraph Store/Checkpoint  │  │
│  └────────────────────────────────┘  │
└─────────────────────────────────────┘
     One database. One connection string.
```

## Quick Comparison

| | Neo4j + Pinecone | PostgreSQL + AGE + pgvector |
|---|---|---|
| Databases | 2 (graph + vector) | **1** |
| Licensing | GPL + proprietary | **Apache 2.0 + PostgreSQL License** |
| Cost (HA) | $15K+/year + vector DB pricing | **$0** (PG native HA) |
| LangChain integration | `langchain-neo4j` | **`langchain-age`** |
| Deployment | 2 clusters to manage | **1 PostgreSQL** |
| Backup | 2 backup pipelines | **1 pg_dump** |
| Long-term memory | Separate DB or service | **Same DB** (LangGraph PostgresStore) |

## Setup in 5 Minutes

### 1. Start the database

```bash
git clone https://github.com/BAEM1N/langchain-age.git
cd langchain-age/docker
docker compose up -d
```

One container. AGE + pgvector + pg_trgm pre-installed.

### 2. Install

```bash
pip install "langchain-age[all]" langchain-openai
```

### 3. Build a Knowledge Graph

```python
from langchain_age import AGEGraph

graph = AGEGraph(
    "host=localhost port=5433 dbname=langchain_age user=langchain password=langchain",
    graph_name="company_kg",
)

# Same Cypher as Neo4j — no new syntax to learn
graph.query("CREATE (:Person {name: 'Alice', role: 'CTO'})")
graph.query("CREATE (:Person {name: 'Bob', role: 'Engineer'})")
graph.query("CREATE (:Product {name: 'AGE', desc: 'Graph extension for PostgreSQL'})")
graph.query(
    "MATCH (a:Person {name: 'Alice'}), (p:Product {name: 'AGE'}) "
    "CREATE (a)-[:LEADS]->(p)"
)
graph.query(
    "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) "
    "CREATE (a)-[:MANAGES]->(b)"
)
```

### 4. Vectorise Graph Nodes

```python
from langchain_age import AGEVector
from langchain_openai import OpenAIEmbeddings

# One line: graph nodes → vector embeddings
store = AGEVector.from_existing_graph(
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    connection_string="host=localhost port=5433 ...",
    graph_name="company_kg",
    node_label="Person",
    text_node_properties=["name", "role"],
    collection_name="person_vectors",
)
```

### 5. GraphRAG: Vector Search + Graph Context

```python
# Step 1: Find relevant people via vector search
docs = store.similarity_search("engineering leadership", k=2)

# Step 2: Expand context via graph relationships
for doc in docs:
    label = doc.metadata["node_label"]
    neighbors = graph.query(
        f"MATCH (n:{label})-[r]->(m) RETURN type(r) AS rel, m.name AS name"
    )
    print(f"{doc.page_content}: {neighbors}")
```

### 6. LLM-Powered Cypher QA

```python
from langchain_age import AGEGraphCypherQAChain
from langchain_openai import ChatOpenAI

chain = AGEGraphCypherQAChain.from_llm(
    ChatOpenAI(model="gpt-4o-mini"),
    graph=graph,
    allow_dangerous_requests=True,
)

answer = chain.run("Who does Alice manage?")
# "Alice manages Bob, who is an Engineer."
```

## Why Not Neo4j?

### 1. You Already Have PostgreSQL

Most applications already run PostgreSQL. Adding AGE is `CREATE EXTENSION age;` — not deploying a new database cluster.

### 2. Licensing Freedom

Neo4j Community is GPL. If you embed it in a product you distribute, GPL propagates. Neo4j Enterprise requires a commercial license ($15K+/year).

Apache AGE is Apache 2.0. Do whatever you want.

### 3. Total Cost

| Scenario | Neo4j | AGE |
|----------|-------|-----|
| Dev/test | Free (Community, single node) | Free |
| Production HA | **$15K+/year** (Enterprise) or AuraDB ($65/GB/month) | **$0** (PostgreSQL Patroni/repmgr) |
| Vector search | Additional vector DB | **Included** (pgvector) |
| Long-term memory | Additional service | **Included** (LangGraph PostgresStore) |

### 4. One Backup, One Monitor, One Team

Your PostgreSQL DBA already knows how to:
- Set up streaming replication
- Run pg_dump / pg_basebackup
- Monitor with pg_stat_statements
- Manage connection pooling (PgBouncer)

No new operational expertise needed.

### 5. Performance is Fine for RAG

Neo4j's index-free adjacency wins at 6+ hop deep traversals. But RAG workloads are typically 1–3 hops: "find related documents", "expand entity context".

For these patterns, AGE is fast enough — and `langchain-age` provides a `traverse()` method using PostgreSQL's `WITH RECURSIVE` that's actually **10–22x faster** than AGE's own Cypher for deep hops.

## When Neo4j IS Better

Be honest:
- **Billions of nodes + deep graph algorithms** (PageRank, community detection) — Neo4j GDS has no equivalent in AGE
- **Enterprise support contract** — Neo4j has a sales team, SLAs, and 24/7 support
- **Mature ecosystem** — 450+ APOC procedures, Bloom visualization, GraphConnect conference

If your workload is "social network analysis on 10 billion edges with real-time community detection", use Neo4j. Pay the license fee. It's worth it.

If your workload is "RAG application that needs graph context alongside vector search", AGE on PostgreSQL is the simpler, cheaper, and more maintainable choice.

## Getting Started

```bash
pip install "langchain-age[all]"
```

- [GitHub](https://github.com/BAEM1N/langchain-age)
- [Tutorial (EN)](https://github.com/BAEM1N/langchain-age/blob/main/docs/en/tutorial.md)
- [Tutorial (KO)](https://github.com/BAEM1N/langchain-age/blob/main/docs/ko/tutorial.md)
- [Notebooks](https://github.com/BAEM1N/langchain-age/tree/main/notebooks)

---

*langchain-age is MIT licensed. Apache AGE is Apache 2.0. pgvector is PostgreSQL License. No licensing fees, no vendor lock-in.*
