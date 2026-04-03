---
name: langchain-age
description: Use when working with Apache AGE graph database, pgvector, or langchain-age library. Provides correct Cypher patterns, API usage, and AGE-specific constraints.
---

# langchain-age Skill

## When to activate
- User mentions Apache AGE, AGEGraph, AGEVector, or langchain-age
- User wants graph + vector search on PostgreSQL
- User is migrating from Neo4j to AGE
- User needs Cypher queries for AGE (different from Neo4j in subtle ways)

## Key API

```python
from langchain_age import AGEGraph, AGEVector, AGEGraphCypherQAChain, DistanceStrategy, SearchType
```

### Graph
```python
graph = AGEGraph(dsn, graph_name="kg")
graph.query("MATCH (n:Person) RETURN n.name AS name")
graph.query("MATCH (n) WHERE n.name = %s RETURN n", params=("Alice",))
graph.traverse("Person", {"name": "Alice"}, "KNOWS", max_depth=6)
graph.add_graph_documents([doc])
graph.refresh_schema()
```

### Vector
```python
store = AGEVector(connection_string=dsn, embedding_function=emb, collection_name="docs")
store.add_texts(["..."], metadatas=[{...}])
store.similarity_search("query", k=5, filter={"tag": {"$in": ["a", "b"]}})
store = AGEVector.from_existing_graph(embedding=emb, connection_string=dsn, graph_name="kg", node_label="Doc", text_node_properties=["title"])
```

### Chain
```python
chain = AGEGraphCypherQAChain.from_llm(llm, graph=graph, allow_dangerous_requests=True)
chain.run("question")
```

## AGE Cypher Rules (CRITICAL)
1. NO APOC procedures
2. ALWAYS alias returns: `RETURN n.name AS name`
3. Backtick reserved words: `` n.`desc` ``, `` n.`order` ``
4. No `$param` — use `%s` with `params=(value,)`
5. UNWIND works: `UNWIND [{...}] AS row CREATE ...`
6. No `elementId()` — use `id(n)` or return node object

## Filter Operators
`$eq $ne $lt $lte $gt $gte $in $nin $between $like $ilike $exists $and $or`

## Performance
- 1-3 hop: use `graph.query()` (Cypher)
- 4+ hop: use `graph.traverse()` (WITH RECURSIVE, 10-22x faster)
- Batch insert: automatic via UNWIND in `add_graph_documents()`
- Vector index: `store.create_hnsw_index(m=16, ef_construction=64)`

## DB Setup
```bash
cd docker && docker compose up -d
# DSN: host=localhost port=5433 dbname=langchain_age user=langchain password=langchain
```
