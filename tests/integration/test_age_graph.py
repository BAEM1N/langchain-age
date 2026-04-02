"""Integration tests for AGEGraph.

Requires a running PostgreSQL + AGE instance.  Set the environment variable:

    LANGCHAIN_AGE_TEST_DSN=host=localhost port=5432 dbname=langchain_age user=langchain password=langchain

Or use the Docker Compose setup in ``docker/``:

    cd docker && docker compose up -d
"""
from __future__ import annotations

import os

import pytest

SKIP_REASON = "Set LANGCHAIN_AGE_TEST_DSN to run integration tests."
TEST_DSN = os.environ.get("LANGCHAIN_AGE_TEST_DSN", "")
TEST_GRAPH = "test_langchain_age"


@pytest.fixture(scope="module")
def age_graph():
    if not TEST_DSN:
        pytest.skip(SKIP_REASON)
    from langchain_age import AGEGraph

    graph = AGEGraph(TEST_DSN, TEST_GRAPH, refresh_schema=False)
    yield graph
    # Cleanup
    graph.drop_graph()


def test_create_and_query_node(age_graph):
    age_graph.query("CREATE (:TestPerson {name: 'Alice', age: 30})")
    results = age_graph.query("MATCH (n:TestPerson) RETURN n.name AS name")
    names = [r.get("name") for r in results]
    assert "Alice" in names


def test_create_relationship(age_graph):
    age_graph.query("CREATE (:Animal {name: 'Cat'})")
    age_graph.query("CREATE (:Animal {name: 'Dog'})")
    age_graph.query(
        "MATCH (a:Animal {name: 'Cat'}), (b:Animal {name: 'Dog'}) "
        "CREATE (a)-[:FRIENDS_WITH]->(b)"
    )
    results = age_graph.query(
        "MATCH (a:Animal)-[r:FRIENDS_WITH]->(b:Animal) RETURN a.name AS a, b.name AS b"
    )
    assert len(results) >= 1
    assert results[0]["a"] == "Cat"
    assert results[0]["b"] == "Dog"


def test_refresh_schema(age_graph):
    age_graph.refresh_schema()
    assert isinstance(age_graph.schema, str)
    assert isinstance(age_graph.structured_schema, dict)


def test_add_graph_documents(age_graph):
    from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
    from langchain_core.documents import Document

    node_a = Node(id="1", type="Movie", properties={"title": "Inception"})
    node_b = Node(id="2", type="Director", properties={"name": "Nolan"})
    rel = Relationship(source=node_b, target=node_a, type="DIRECTED")
    source_doc = Document(
        page_content="Inception was directed by Nolan.",
        metadata={"source": "test"},
    )
    gd = GraphDocument(nodes=[node_a, node_b], relationships=[rel], source=source_doc)

    age_graph.add_graph_documents([gd], include_source=True)

    results = age_graph.query("MATCH (d:Director)-[:DIRECTED]->(m:Movie) RETURN d.name AS director, m.title AS title")
    assert any(r.get("title") == "Inception" for r in results)
