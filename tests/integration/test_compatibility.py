"""Integration tests for langchain-age LangChain compatibility.

Verifies API parity with langchain-neo4j:
  from langchain_age import AGEGraph
  from langchain_age import AGEVector
  from langchain_age import AGEGraphCypherQAChain, DistanceStrategy, SearchType

Requirements:
  - Docker container: docker/docker-compose.yml
  - DSN: LANGCHAIN_AGE_TEST_DSN env var or default below
"""
import os

import pytest

DSN = os.getenv(
    "LANGCHAIN_AGE_TEST_DSN",
    "host=localhost port=5433 dbname=langchain_age user=langchain password=langchain",
)
GRAPH = "test_compat_graph"

# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def graph():
    from langchain_age import AGEGraph

    g = AGEGraph(DSN, GRAPH, refresh_schema=False)
    yield g
    try:
        g.drop_graph()
    except Exception:
        pass


@pytest.fixture(scope="module")
def vector_store():
    from langchain_age import AGEVector, DistanceStrategy

    store = AGEVector(
        connection_string=DSN,
        embedding_function=FakeEmbeddings(),
        collection_name="test_compat_vectors",
        distance_strategy=DistanceStrategy.COSINE,
        pre_delete_collection=True,
    )
    yield store
    store._drop_table()


class FakeEmbeddings:
    """Deterministic embeddings (dim=4) — no external API required."""

    def embed_documents(self, texts):
        return [
            [float(i % 4) / 4.0 for i in range(len(t), len(t) + 4)] for t in texts
        ]

    def embed_query(self, text):
        return [0.1, 0.5, 0.8, 0.2]


# ─────────────────────────────────────────────────────────────────
# 1. AGEGraph — Neo4jGraph compatible interface
# ─────────────────────────────────────────────────────────────────


class TestAGEGraph:
    """Verify AGEGraph matches the Neo4jGraph interface."""

    def test_import_style(self):
        """from langchain_age import AGEGraph (Neo4j style)."""
        from langchain_age import AGEGraph

        assert AGEGraph is not None

    def test_query_create_and_match(self, graph):
        """graph.query() — CREATE + MATCH."""
        graph.query("MERGE (:Person {name: 'Alice', age: 30})")
        results = graph.query(
            "MATCH (n:Person {name: 'Alice'}) RETURN n.name AS name, n.age AS age"
        )
        assert len(results) >= 1
        assert results[0]["name"] == "Alice"
        assert results[0]["age"] == 30

    def test_query_relationship(self, graph):
        """Create and query relationships."""
        graph.query("MERGE (:Person {name: 'Bob'})")
        graph.query(
            "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) "
            "MERGE (a)-[:KNOWS]->(b)"
        )
        results = graph.query(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) "
            "RETURN a.name AS src, b.name AS dst"
        )
        assert any(r["src"] == "Alice" and r["dst"] == "Bob" for r in results)

    def test_refresh_schema(self, graph):
        """graph.refresh_schema() + graph.schema property."""
        graph.refresh_schema()
        assert isinstance(graph.schema, str)
        assert len(graph.schema) > 0
        assert isinstance(graph.structured_schema, dict)
        assert "node_props" in graph.structured_schema
        assert "relationships" in graph.structured_schema

    def test_get_schema_property(self, graph):
        """graph.get_schema property (same as Neo4j)."""
        schema = graph.get_schema
        assert isinstance(schema, str)

    def test_get_structured_schema_property(self, graph):
        """graph.get_structured_schema property (same as Neo4j)."""
        ss = graph.get_structured_schema
        assert isinstance(ss, dict)
        assert "node_props" in ss

    def test_add_graph_documents(self, graph):
        """graph.add_graph_documents() — LLMGraphTransformer output compatible."""
        from langchain_community.graphs.graph_document import (
            GraphDocument,
            Node,
            Relationship,
        )
        from langchain_core.documents import Document

        movie = Node(
            id="inception", type="Movie", properties={"title": "Inception"}
        )
        director = Node(
            id="nolan", type="Director", properties={"name": "Christopher Nolan"}
        )
        rel = Relationship(source=director, target=movie, type="DIRECTED")
        source = Document(
            page_content="Inception by Nolan", metadata={"source": "test"}
        )

        graph.add_graph_documents(
            [GraphDocument(nodes=[movie, director], relationships=[rel], source=source)],
            include_source=True,
        )

        results = graph.query(
            "MATCH (d:Director)-[:DIRECTED]->(m:Movie) "
            "RETURN d.name AS director, m.title AS title"
        )
        assert any(r.get("title") == "Inception" for r in results)

    def test_include_exclude_types(self, graph):
        """include_types / exclude_types filtering (same as Neo4j)."""
        from langchain_age import AGEGraph

        g2 = AGEGraph(DSN, GRAPH, refresh_schema=True, include_types=["Person"])
        assert "Person" in g2.structured_schema.get("node_props", {})
        assert "Movie" not in g2.structured_schema.get("node_props", {})

        g3 = AGEGraph(DSN, GRAPH, refresh_schema=True, exclude_types=["Person"])
        assert "Person" not in g3.structured_schema.get("node_props", {})


# ─────────────────────────────────────────────────────────────────
# 1b. Deep traversal (WITH RECURSIVE) & property index
# ─────────────────────────────────────────────────────────────────


class TestDeepTraversal:
    """WITH RECURSIVE based deep hop optimisation tests."""

    def test_traverse_outgoing(self, graph):
        """traverse() — default outgoing direction."""
        # Test data: Alice -> Bob (KNOWS)
        results = graph.traverse(
            start_label="Person",
            start_filter={"name": "Alice"},
            edge_label="KNOWS",
            max_depth=3,
        )
        assert len(results) >= 1
        assert all("depth" in r for r in results)
        assert all("node_id" in r for r in results)

    def test_traverse_returns_properties(self, graph):
        """traverse(return_properties=True) — includes node properties."""
        results = graph.traverse(
            start_label="Person",
            start_filter={"name": "Alice"},
            edge_label="KNOWS",
            max_depth=2,
            return_properties=True,
        )
        if results:
            assert "properties" in results[0]
            assert isinstance(results[0]["properties"], dict)

    def test_traverse_no_properties(self, graph):
        """traverse(return_properties=False) — lightweight traversal."""
        results = graph.traverse(
            start_label="Person",
            start_filter={"name": "Alice"},
            edge_label="KNOWS",
            max_depth=2,
            return_properties=False,
        )
        if results:
            assert "properties" not in results[0]

    def test_traverse_incoming(self, graph):
        """traverse(direction='incoming') — reverse direction."""
        results = graph.traverse(
            start_label="Person",
            start_filter={"name": "Bob"},
            edge_label="KNOWS",
            max_depth=2,
            direction="incoming",
        )
        # Alice -> Bob, so incoming from Bob should find Alice
        assert len(results) >= 1

    def test_traverse_both(self, graph):
        """traverse(direction='both') — bidirectional."""
        results = graph.traverse(
            start_label="Person",
            start_filter={"name": "Alice"},
            edge_label="KNOWS",
            max_depth=2,
            direction="both",
        )
        assert len(results) >= 1

    def test_create_property_index_btree(self, graph):
        """create_property_index() — B-tree index creation."""
        graph.create_property_index("Person", "name", index_type="btree")

    def test_create_property_index_gin(self, graph):
        """create_property_index() — GIN index creation."""
        graph.create_property_index("Person", "name", index_type="gin")


# ─────────────────────────────────────────────────────────────────
# 2. AGEVector — Neo4jVector / PGVectorStore compatible interface
# ─────────────────────────────────────────────────────────────────


class TestAGEVector:
    """Verify AGEVector matches the Neo4jVector interface."""

    def test_import_style(self):
        """from langchain_age import AGEVector, DistanceStrategy, SearchType."""
        from langchain_age import AGEVector, DistanceStrategy, SearchType

        assert AGEVector is not None
        assert DistanceStrategy.COSINE is not None
        assert SearchType.HYBRID is not None

    def test_add_documents(self, vector_store):
        """add_documents() + returned ID list."""
        from langchain_core.documents import Document

        docs = [
            Document(
                page_content="Apache AGE is a graph extension for PostgreSQL.",
                metadata={"source": "a"},
            ),
            Document(
                page_content="pgvector enables vector similarity search.",
                metadata={"source": "b"},
            ),
            Document(
                page_content="LangChain connects LLMs to tools and data.",
                metadata={"source": "c"},
            ),
        ]
        ids = vector_store.add_documents(docs)
        assert len(ids) == 3
        assert all(isinstance(i, str) for i in ids)

    def test_similarity_search(self, vector_store):
        """similarity_search(query, k) — basic interface."""
        results = vector_store.similarity_search("graph database", k=2)
        assert len(results) <= 2
        assert all(hasattr(r, "page_content") for r in results)

    def test_similarity_search_with_score(self, vector_store):
        """similarity_search_with_score() — (Document, float) tuples."""
        results = vector_store.similarity_search_with_score("vector search", k=2)
        assert len(results) <= 2
        for doc, score in results:
            assert hasattr(doc, "page_content")
            assert isinstance(score, float)

    def test_similarity_search_by_vector(self, vector_store):
        """similarity_search_by_vector() — raw embedding input."""
        embedding = [0.1, 0.5, 0.8, 0.2]
        results = vector_store.similarity_search_by_vector(embedding, k=2)
        assert len(results) <= 2

    def test_add_texts(self, vector_store):
        """add_texts() interface."""
        ids = vector_store.add_texts(
            ["Graph RAG combines graphs and vectors.", "OpenCypher runs on AGE."],
            metadatas=[{"tag": "rag"}, {"tag": "cypher"}],
        )
        assert len(ids) == 2

    def test_metadata_filter_equality(self, vector_store):
        """Simple equality metadata filter."""
        results = vector_store.similarity_search(
            "search", k=10, filter={"source": "a"}
        )
        for doc in results:
            assert doc.metadata.get("source") == "a"

    def test_metadata_filter_operators(self, vector_store):
        """$in, $like and other advanced filter operators."""
        results = vector_store.similarity_search(
            "search", k=10, filter={"source": {"$in": ["a", "b"]}}
        )
        for doc in results:
            assert doc.metadata.get("source") in ("a", "b")

    def test_delete(self, vector_store):
        """delete(ids) interface."""
        ids = vector_store.add_texts(["delete me"])
        result = vector_store.delete(ids)
        assert result is True

    def test_get_by_ids(self, vector_store):
        """get_by_ids() interface."""
        ids = vector_store.add_texts(["fetch by id test"])
        docs = vector_store.get_by_ids(ids)
        assert len(docs) == 1
        assert docs[0].page_content == "fetch by id test"

    def test_as_retriever(self, vector_store):
        """as_retriever() — LangChain Retriever conversion (VectorStore base)."""
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        assert retriever is not None
        results = retriever.invoke("graph database")
        assert isinstance(results, list)

    def test_from_texts_classmethod(self):
        """from_texts() class method."""
        from langchain_age import AGEVector

        store = AGEVector.from_texts(
            texts=["hello world", "foo bar"],
            embedding=FakeEmbeddings(),
            connection_string=DSN,
            collection_name="test_from_texts_tmp",
            pre_delete_collection=True,
        )
        results = store.similarity_search("hello", k=1)
        assert len(results) >= 1
        store._drop_table()

    def test_from_existing_index_classmethod(self):
        """from_existing_index() — reconnect to existing table."""
        from langchain_age import AGEVector

        store = AGEVector.from_existing_index(
            embedding=FakeEmbeddings(),
            connection_string=DSN,
            collection_name="test_compat_vectors",
        )
        assert store is not None

    def test_hybrid_search(self):
        """Hybrid search (SearchType.HYBRID) — vector + full-text RRF."""
        from langchain_age import AGEVector, SearchType

        store = AGEVector(
            connection_string=DSN,
            embedding_function=FakeEmbeddings(),
            collection_name="test_hybrid_tmp",
            search_type=SearchType.HYBRID,
            pre_delete_collection=True,
        )
        store.add_texts([
            "PostgreSQL is a powerful relational database.",
            "Apache AGE adds graph capabilities to PostgreSQL.",
            "pgvector enables fast vector similarity search.",
        ])
        results = store.similarity_search("graph database PostgreSQL", k=3)
        assert len(results) >= 1
        store._drop_table()

    def test_hnsw_index_creation(self, vector_store):
        """HNSW index creation interface."""
        vector_store.create_hnsw_index(m=8, ef_construction=32)


# ─────────────────────────────────────────────────────────────────
# 3. from_existing_graph
# ─────────────────────────────────────────────────────────────────


class TestFromExistingGraph:
    """Neo4jVector.from_existing_graph() equivalent pattern."""

    def test_from_existing_graph(self, graph):
        from langchain_age import AGEVector

        # "desc" is a Cypher reserved keyword — backtick quoting handles it.
        graph.query("MERGE (:Product {name: 'LangChain', desc: 'LLM framework'})")
        graph.query(
            "MERGE (:Product {name: 'pgvector', desc: 'Vector search extension'})"
        )

        store = AGEVector.from_existing_graph(
            embedding=FakeEmbeddings(),
            connection_string=DSN,
            graph_name=GRAPH,
            node_label="Product",
            text_node_properties=["name", "desc"],
            collection_name="test_from_graph_tmp",
        )
        results = store.similarity_search("LLM", k=2)
        assert len(results) >= 1
        store._drop_table()


# ─────────────────────────────────────────────────────────────────
# 4. AGEGraphCypherQAChain — GraphCypherQAChain compatible interface
# ─────────────────────────────────────────────────────────────────


class FakeLLM:
    """Dummy LLM for pipeline structure verification without API calls.

    Returns hardcoded Cypher or QA answers based on input type,
    allowing chain structure tests without an actual LLM.
    """

    def invoke(self, input, config=None, **kwargs):
        if isinstance(input, list):
            content = str(input)
            if "schema" in content.lower() or "cypher" in content.lower():
                return "MATCH (n:Person) RETURN n.name AS name LIMIT 3"
            return "Alice and Bob are people in the graph."
        if isinstance(input, dict):
            if "schema" in input:
                return "MATCH (n:Person) RETURN n.name AS name LIMIT 3"
            return "Alice and Bob are people in the graph."
        return "Alice and Bob are people in the graph."


class TestAGEGraphCypherQAChain:
    """Verify AGEGraphCypherQAChain matches the GraphCypherQAChain interface."""

    def test_import_style(self):
        """from langchain_age import AGEGraphCypherQAChain."""
        from langchain_age import AGEGraphCypherQAChain

        assert AGEGraphCypherQAChain is not None

    def test_dangerous_request_gate(self, graph):
        """allow_dangerous_requests=False raises ValueError (same as Neo4j)."""
        from langchain_age import AGEGraphCypherQAChain

        with pytest.raises(ValueError, match="allow_dangerous_requests"):
            AGEGraphCypherQAChain.from_llm(
                FakeLLM(), graph=graph, allow_dangerous_requests=False
            )

    def _make_chain(self, graph, **kwargs):
        """Build a fake chain using RunnableLambda (passes LangChain v1 Pydantic validation)."""
        from langchain_core.runnables import RunnableLambda

        from langchain_age import AGEGraphCypherQAChain

        cypher_chain = RunnableLambda(
            lambda _: "MATCH (n:Person) RETURN n.name AS name LIMIT 3"
        )
        qa_chain = RunnableLambda(
            lambda _: "Alice and Bob are people in the graph."
        )
        return AGEGraphCypherQAChain(
            graph=graph,
            cypher_generation_chain=cypher_chain,
            qa_chain=qa_chain,
            allow_dangerous_requests=True,
            **kwargs,
        )

    def test_chain_invoke(self, graph):
        """chain.invoke({'query': ...}) — basic execution flow."""
        graph.refresh_schema()
        chain = self._make_chain(graph)
        result = chain.invoke({"query": "Who are the people?"})
        assert "result" in result
        assert isinstance(result["result"], str)

    def test_chain_run(self, graph):
        """chain.run(query) — single-string interface (same as Neo4j)."""
        chain = self._make_chain(graph)
        result = chain.run("List all people")
        assert isinstance(result, str)

    def test_return_intermediate_steps(self, graph):
        """return_intermediate_steps=True — includes Cypher + context."""
        chain = self._make_chain(graph, return_intermediate_steps=True)
        result = chain.invoke({"query": "Find people"})
        assert "intermediate_steps" in result
        steps = result["intermediate_steps"]
        assert any("query" in s for s in steps)

    def test_include_exclude_types_in_chain(self, graph):
        """include_types / exclude_types — chain-level schema filter."""
        chain = self._make_chain(graph, include_types=["Person"])
        result = chain.invoke({"query": "Find people"})
        assert "result" in result

    def test_custom_prompts(self, graph):
        """cypher_prompt / qa_prompt customisation (same params as Neo4j)."""
        from langchain_age import (
            CYPHER_GENERATION_PROMPT,
            QA_PROMPT,
        )

        assert CYPHER_GENERATION_PROMPT is not None
        assert QA_PROMPT is not None
        chain = self._make_chain(graph)
        assert chain is not None


# ─────────────────────────────────────────────────────────────────
# 4b. Chain verbose/callback regression tests
# ─────────────────────────────────────────────────────────────────


class TestChainVerboseCallback:
    """Regression tests for verbose field and run_manager callback paths."""

    def _make_chain(self, graph, **kwargs):
        from langchain_core.runnables import RunnableLambda

        from langchain_age import AGEGraphCypherQAChain

        return AGEGraphCypherQAChain(
            graph=graph,
            cypher_generation_chain=RunnableLambda(
                lambda _: "MATCH (n:Person) RETURN n.name AS name LIMIT 3"
            ),
            qa_chain=RunnableLambda(lambda _: "Answer."),
            allow_dangerous_requests=True,
            **kwargs,
        )

    def test_verbose_field_exists(self, graph):
        """verbose field should be defined and default to False."""
        chain = self._make_chain(graph)
        assert hasattr(chain, "verbose")
        assert chain.verbose is False

    def test_verbose_true_does_not_crash(self, graph):
        """verbose=True should not raise AttributeError."""
        graph.refresh_schema()
        chain = self._make_chain(graph, verbose=True)
        assert chain.verbose is True
        result = chain.invoke({"query": "test"})
        assert "result" in result

    def test_verbose_false_invoke(self, graph):
        """verbose=False invoke should work identically."""
        chain = self._make_chain(graph, verbose=False)
        result = chain.invoke({"query": "test"})
        assert "result" in result


# ─────────────────────────────────────────────────────────────────
# 5. Extended VectorStore methods
# ─────────────────────────────────────────────────────────────────


class TestVectorStoreExtended:
    """Verify VectorStore methods added for interface completeness."""

    def test_embeddings_property(self, vector_store):
        """embeddings property — LangChain VectorStore standard."""
        emb = vector_store.embeddings
        assert emb is vector_store.embedding_function

    def test_similarity_search_with_relevance_scores(self, vector_store):
        """similarity_search_with_relevance_scores — normalised to [0, 1]."""
        vector_store.add_texts(["dog walks", "cat naps"], metadatas=[{}, {}])
        results = vector_store.similarity_search_with_relevance_scores("dog", k=2)
        assert len(results) >= 1
        for _doc, score in results:
            assert isinstance(score, float)

    def test_context_manager(self):
        """with AGEVector(...) as store: — context manager."""
        from langchain_age import AGEVector

        with AGEVector(
            connection_string=DSN,
            embedding_function=FakeEmbeddings(),
            collection_name="test_ctx_mgr",
            pre_delete_collection=True,
        ) as store:
            store.add_texts(["context manager test"])
            results = store.similarity_search("context", k=1)
            assert len(results) >= 1
        # After __exit__, connection is closed
        assert store._conn.closed
