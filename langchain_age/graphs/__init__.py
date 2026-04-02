from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_age.graphs.age_graph import AGEGraph as AGEGraph

def __getattr__(name: str):
    if name == "AGEGraph":
        from langchain_age.graphs.age_graph import AGEGraph
        return AGEGraph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["AGEGraph"]
