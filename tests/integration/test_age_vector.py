"""Integration tests for AGEVector.

Requires a running PostgreSQL + pgvector instance.  Set:

    LANGCHAIN_AGE_TEST_DSN=host=localhost port=5432 dbname=langchain_age user=langchain password=langchain
"""
from __future__ import annotations

import os

import pytest

SKIP_REASON = "Set LANGCHAIN_AGE_TEST_DSN to run integration tests."
TEST_DSN = os.environ.get("LANGCHAIN_AGE_TEST_DSN", "")


class FakeEmbeddings:
    """Deterministic embeddings for testing (dim=4)."""

    def embed_documents(self, texts):
        return [[float(i % 4) / 4 for i in range(len(t), len(t) + 4)] for t in texts]

    def embed_query(self, text):
        return [0.25, 0.5, 0.75, 1.0]


@pytest.fixture(scope="module")
def age_vector():
    if not TEST_DSN:
        pytest.skip(SKIP_REASON)
    from langchain_age import AGEVector

    store = AGEVector(
        connection_string=TEST_DSN,
        embedding_function=FakeEmbeddings(),
        collection_name="test_langchain_age_vectors",
        pre_delete_collection=True,
    )
    yield store
    store._drop_table()


def test_add_and_search(age_vector):
    from langchain_core.documents import Document

    docs = [
        Document(page_content="Alice is a data scientist.", metadata={"role": "DS"}),
        Document(page_content="Bob is a software engineer.", metadata={"role": "SWE"}),
    ]
    ids = age_vector.add_documents(docs)
    assert len(ids) == 2

    results = age_vector.similarity_search("data scientist", k=2)
    assert len(results) >= 1


def test_add_texts(age_vector):
    ids = age_vector.add_texts(["Graph databases are cool.", "pgvector rocks!"])
    assert len(ids) == 2


def test_similarity_with_score(age_vector):
    results = age_vector.similarity_search_with_score("database", k=2)
    assert all(isinstance(score, float) for _, score in results)


def test_delete(age_vector):
    ids = age_vector.add_texts(["Delete me."])
    deleted = age_vector.delete(ids)
    assert deleted is True


def test_get_by_ids(age_vector):
    ids = age_vector.add_texts(["Fetch by ID."])
    docs = age_vector.get_by_ids(ids)
    assert len(docs) == 1
    assert docs[0].page_content == "Fetch by ID."
