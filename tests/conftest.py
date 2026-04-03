"""Shared fixtures for langchain-age tests."""
from __future__ import annotations

import os

import pytest

DSN = os.getenv(
    "LANGCHAIN_AGE_TEST_DSN",
    "host=localhost port=5433 dbname=langchain_age user=langchain password=langchain",
)
GRAPH = "test_compat_graph"


def pytest_collection_modifyitems(config, items):
    """Skip integration tests when LANGCHAIN_AGE_TEST_DSN is not set."""
    if os.getenv("LANGCHAIN_AGE_TEST_DSN"):
        return
    skip = pytest.mark.skip(reason="LANGCHAIN_AGE_TEST_DSN not set")
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip)
