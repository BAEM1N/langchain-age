# API Reference

## AGEGraph

`langchain_age.graphs.age_graph.AGEGraph`

GraphStore backed by PostgreSQL + Apache AGE.

### Constructor

```python
AGEGraph(
    connection_string: str,
    graph_name: str,
    *,
    timeout: float | None = None,
    refresh_schema: bool = True,
    sanitize: bool = True,
    enhanced_schema: bool = False,
    include_types: list[str] | None = None,
    exclude_types: list[str] | None = None,
    max_retries: int = 3,
)
```

### Methods

| Method | Description |
|--------|-------------|
| `query(query, params=None)` | Execute Cypher and return list of dicts. Supports `%s` mogrify params. |
| `refresh_schema()` | Re-introspect graph via `ag_catalog` SQL. |
| `add_graph_documents(docs, include_source=False)` | Batch upsert `GraphDocument` objects using UNWIND. |
| `traverse(start_label, start_filter, edge_label, max_depth, *, direction="outgoing", return_properties=True)` | WITH RECURSIVE deep hop traversal (10–22x faster than Cypher `*N`). |
| `create_property_index(node_label, property_name, *, index_type="btree")` | Create B-tree or GIN index on node properties. |
| `create_graph()` | Create the AGE graph if not exists. |
| `drop_graph()` | Drop the graph. Irreversible. |
| `close()` | Close the connection. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `get_schema` | `str` | Human-readable schema string. |
| `get_structured_schema` | `dict` | Programmatic schema with `node_props`, `rel_props`, `relationships`. |

---

## AGEVector

`langchain_age.vectorstores.age_vector.AGEVector`

VectorStore backed by pgvector with optional AGE graph linkage.

### Constructor

```python
AGEVector(
    connection_string: str,
    embedding_function: Embeddings,
    *,
    collection_name: str = "langchain_age_vectors",
    distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
    search_type: SearchType = SearchType.VECTOR,
    pre_delete_collection: bool = False,
    relevance_score_fn: Callable[[float], float] | None = None,
    age_graph_name: str | None = None,
    retrieval_query: str | None = None,
    embedding_dimension: int | None = None,
    batch_size: int = 1_000,
)
```

### Methods

| Method | Description |
|--------|-------------|
| `add_texts(texts, metadatas=None, ids=None)` | Embed and store texts. Batch INSERT via `executemany`. |
| `add_documents(documents, ids=None)` | Embed and store `Document` objects. |
| `similarity_search(query, k=4, filter=None)` | Return k most similar documents. |
| `similarity_search_with_score(query, k=4, filter=None)` | Return docs with raw distance scores. |
| `similarity_search_with_relevance_scores(query, k=4, filter=None)` | Return docs with normalised [0,1] scores. |
| `similarity_search_by_vector(embedding, k=4, filter=None)` | Search by pre-computed vector. |
| `max_marginal_relevance_search(query, k=4, fetch_k=20, lambda_mult=0.5)` | MMR search, reuses stored embeddings. |
| `delete(ids=None)` | Delete documents by ID. |
| `get_by_ids(ids)` | Fetch documents by ID. |
| `as_retriever(**kwargs)` | Convert to LangChain Retriever. |
| `create_hnsw_index(m=16, ef_construction=64)` | Create HNSW index. |
| `create_ivfflat_index(n_lists=100)` | Create IVFFlat index. |
| `drop_index()` | Drop all vector indexes. |
| `close()` | Close the connection. |

### Class Methods

| Method | Description |
|--------|-------------|
| `from_texts(texts, embedding, **kwargs)` | Create and populate from texts. |
| `from_documents(documents, embedding, **kwargs)` | Create and populate from documents. |
| `from_existing_index(embedding, connection_string, collection_name)` | Connect to existing table. |
| `from_existing_graph(embedding, connection_string, graph_name, node_label, text_node_properties)` | Vectorise AGE graph nodes. |

---

## AGEGraphCypherQAChain

`langchain_age.chains.graph_cypher_qa_chain.AGEGraphCypherQAChain`

QA chain: LLM generates Cypher → AGE executes → LLM answers.

### Factory

```python
AGEGraphCypherQAChain.from_llm(
    llm: BaseLanguageModel,
    graph: AGEGraph,
    *,
    cypher_llm: BaseLanguageModel | None = None,
    qa_llm: BaseLanguageModel | None = None,
    include_types: list[str] | None = None,
    exclude_types: list[str] | None = None,
    validate_cypher: bool = True,
    allow_dangerous_requests: bool = False,  # MUST be True
)
```

### Execution

| Method | Description |
|--------|-------------|
| `invoke({"query": "..."})` | Returns `{"result": "...", "intermediate_steps": [...]}`. |
| `run("...")` | Convenience single-string interface. |

---

## Enums

### DistanceStrategy

| Value | Operator | Description |
|-------|----------|-------------|
| `COSINE` | `<=>` | Cosine distance (default) |
| `EUCLIDEAN` | `<->` | L2 distance |
| `MAX_INNER_PRODUCT` | `<#>` | Negative inner product |

### SearchType

| Value | Description |
|-------|-------------|
| `VECTOR` | Vector similarity only (default) |
| `HYBRID` | Vector + PostgreSQL full-text via RRF |

---

## Utility Functions

`langchain_age.utils.cypher`

| Function | Description |
|----------|-------------|
| `escape_cypher_identifier(name)` | Backtick-quote for Cypher reserved words. |
| `escape_cypher_string(value)` | `''` doubling (OpenCypher standard). |
| `validate_sql_identifier(name)` | Regex check for safe SQL identifiers. |
| `validate_cypher(query)` | Lightweight Cypher syntax check. |
| `wrap_cypher_query(graph, cypher, columns)` | Build PG18-compatible `SELECT * FROM cypher(...)`. |
