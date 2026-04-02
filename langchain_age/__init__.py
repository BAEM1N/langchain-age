"""langchain-age: Apache AGE + pgvector integration for LangChain.

Provides:
- ``AGEGraph``              – GraphStore backed by PostgreSQL + Apache AGE
- ``AGEVector``             – VectorStore backed by pgvector
- ``AGEGraphCypherQAChain`` – QA chain that generates Cypher for AGE
- ``DistanceStrategy``      – Distance metric enum for AGEVector
"""
from langchain_age.chains.graph_cypher_qa_chain import (
    AGEGraphCypherQAChain,
    CYPHER_GENERATION_PROMPT,
    QA_PROMPT,
)
from langchain_age.graphs.age_graph import AGEGraph
from langchain_age.vectorstores.age_vector import AGEVector, DistanceStrategy

__all__ = [
    "AGEGraph",
    "AGEVector",
    "AGEGraphCypherQAChain",
    "DistanceStrategy",
    "CYPHER_GENERATION_PROMPT",
    "QA_PROMPT",
]

__version__ = "0.1.0"
