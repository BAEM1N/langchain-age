from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_age.vectorstores.age_vector import AGEVector as AGEVector

def __getattr__(name: str):
    if name == "AGEVector":
        from langchain_age.vectorstores.age_vector import AGEVector
        return AGEVector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["AGEVector"]
