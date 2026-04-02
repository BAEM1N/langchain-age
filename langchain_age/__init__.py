"""langchain-age: Apache AGE + pgvector integration for LangChain.

Drop-in replacement pattern (mirrors langchain-neo4j):

    from langchain_age import AGEGraph
    from langchain_age import AGEVector
    from langchain_age import AGEGraphCypherQAChain

Provides:
- ``AGEGraph``              – GraphStore (PostgreSQL + Apache AGE)
- ``AGEVector``             – VectorStore (pgvector) with optional AGE linkage
- ``AGEGraphCypherQAChain`` – LLM chain that generates Cypher for AGE
- ``DistanceStrategy``      – Cosine / Euclidean / InnerProduct enum
- ``SearchType``            – Vector / Hybrid search enum
"""
from langchain_age.chains.graph_cypher_qa_chain import (
    AGEGraphCypherQAChain,
    CYPHER_GENERATION_PROMPT,
    QA_PROMPT,
)
from langchain_age.graphs.age_graph import AGEGraph
from langchain_age.vectorstores.age_vector import AGEVector, DistanceStrategy, SearchType

__all__ = [
    "AGEGraph",
    "AGEVector",
    "AGEGraphCypherQAChain",
    "DistanceStrategy",
    "SearchType",
    "CYPHER_GENERATION_PROMPT",
    "QA_PROMPT",
]

__version__ = "0.2.0"
