from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_age.chains.graph_cypher_qa_chain import AGEGraphCypherQAChain as AGEGraphCypherQAChain

def __getattr__(name: str):
    if name == "AGEGraphCypherQAChain":
        from langchain_age.chains.graph_cypher_qa_chain import AGEGraphCypherQAChain
        return AGEGraphCypherQAChain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["AGEGraphCypherQAChain"]
