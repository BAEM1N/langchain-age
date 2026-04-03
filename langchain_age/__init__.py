"""langchain-age: Apache AGE + pgvector integration for LangChain.

Supports three usage modes:

**Graph only** (Apache AGE)::

    from langchain_age import AGEGraph

**Vector only** (pgvector)::

    from langchain_age import AGEVector

**Graph + Vector** (both)::

    from langchain_age import AGEGraph, AGEVector
    store = AGEVector.from_existing_graph(...)

For long-term memory and checkpointing, use ``langgraph-checkpoint-postgres``
which connects to the same PostgreSQL instance::

    from langgraph.store.postgres import PostgresStore
"""
from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.0.6"

# Lazy imports — each module is only loaded when accessed, so users can
# install only the dependencies they need (e.g. pgvector without age, or
# age without pgvector).

if TYPE_CHECKING:
    from langchain_age.chains.graph_cypher_qa_chain import (
        CYPHER_GENERATION_PROMPT as CYPHER_GENERATION_PROMPT,
    )
    from langchain_age.chains.graph_cypher_qa_chain import (
        QA_PROMPT as QA_PROMPT,
    )
    from langchain_age.chains.graph_cypher_qa_chain import (
        AGEGraphCypherQAChain as AGEGraphCypherQAChain,
    )
    from langchain_age.graphs.age_graph import AGEGraph as AGEGraph
    from langchain_age.vectorstores.age_vector import (
        AGEVector as AGEVector,
    )
    from langchain_age.vectorstores.age_vector import (
        DistanceStrategy as DistanceStrategy,
    )
    from langchain_age.vectorstores.age_vector import (
        SearchType as SearchType,
    )


def __getattr__(name: str):  # noqa: C901
    if name == "AGEGraph":
        from langchain_age.graphs.age_graph import AGEGraph
        return AGEGraph
    if name == "AGEVector":
        from langchain_age.vectorstores.age_vector import AGEVector
        return AGEVector
    if name == "AGEGraphCypherQAChain":
        from langchain_age.chains.graph_cypher_qa_chain import AGEGraphCypherQAChain
        return AGEGraphCypherQAChain
    if name == "DistanceStrategy":
        from langchain_age.vectorstores.age_vector import DistanceStrategy
        return DistanceStrategy
    if name == "SearchType":
        from langchain_age.vectorstores.age_vector import SearchType
        return SearchType
    if name == "CYPHER_GENERATION_PROMPT":
        from langchain_age.chains.graph_cypher_qa_chain import CYPHER_GENERATION_PROMPT
        return CYPHER_GENERATION_PROMPT
    if name == "QA_PROMPT":
        from langchain_age.chains.graph_cypher_qa_chain import QA_PROMPT
        return QA_PROMPT
    raise AttributeError(f"module 'langchain_age' has no attribute {name!r}")


__all__ = [
    "AGEGraph",
    "AGEVector",
    "AGEGraphCypherQAChain",
    "DistanceStrategy",
    "SearchType",
    "CYPHER_GENERATION_PROMPT",
    "QA_PROMPT",
]
