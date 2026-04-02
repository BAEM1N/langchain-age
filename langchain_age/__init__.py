"""langchain-age: Apache AGE + pgvector integration for LangChain.

Supports three usage modes:

**Graph only** (Apache AGE)::

    from langchain_age import AGEGraph

**Vector only** (pgvector)::

    from langchain_age import AGEVector

**Graph + Vector** (both)::

    from langchain_age import AGEGraph, AGEVector
    store = AGEVector.from_existing_graph(...)

**Chat history** (plain PostgreSQL, no extensions required)::

    from langchain_age import PostgresChatMessageHistory
"""
from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.5.0"

# Lazy imports — each module is only loaded when accessed, so users can
# install only the dependencies they need (e.g. pgvector without age, or
# age without pgvector).

if TYPE_CHECKING:
    from langchain_age.chains.graph_cypher_qa_chain import (
        AGEGraphCypherQAChain as AGEGraphCypherQAChain,
        CYPHER_GENERATION_PROMPT as CYPHER_GENERATION_PROMPT,
        QA_PROMPT as QA_PROMPT,
    )
    from langchain_age.chat_message_histories import (
        PostgresChatMessageHistory as PostgresChatMessageHistory,
    )
    from langchain_age.graphs.age_graph import AGEGraph as AGEGraph
    from langchain_age.vectorstores.age_vector import (
        AGEVector as AGEVector,
        DistanceStrategy as DistanceStrategy,
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
    if name == "PostgresChatMessageHistory":
        from langchain_age.chat_message_histories import PostgresChatMessageHistory
        return PostgresChatMessageHistory
    raise AttributeError(f"module 'langchain_age' has no attribute {name!r}")


__all__ = [
    "AGEGraph",
    "AGEVector",
    "AGEGraphCypherQAChain",
    "DistanceStrategy",
    "SearchType",
    "CYPHER_GENERATION_PROMPT",
    "QA_PROMPT",
    "PostgresChatMessageHistory",
]
