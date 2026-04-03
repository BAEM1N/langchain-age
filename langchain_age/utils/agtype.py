"""Utilities for converting Apache AGE SDK objects to plain Python dicts.

With the apache-age-python SDK installed, psycopg3 automatically deserialises
agtype column values into ``Vertex``, ``Edge``, ``Path``, or plain Python
scalars via the registered ``AgeLoader``.  This module converts those SDK
objects into plain dicts / lists so they can be used in LangChain pipelines
without an AGE dependency downstream.
"""
from __future__ import annotations

import json
import re
from typing import Any

try:
    from langchain_age._vendor.age import Edge, Path, Vertex

    _AGE_SDK_AVAILABLE = True
except ImportError:
    _AGE_SDK_AVAILABLE = False
    Vertex = Edge = Path = None  # type: ignore[assignment,misc]

# Type alias for the return values of agobj_to_dict / agtype_to_python.
AGEPythonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


def agobj_to_dict(obj: Any) -> AGEPythonValue:
    """Convert an apache-age-python SDK object to a plain Python value.

    Conversion table:

    =========  =====================================================
    SDK type   Python output
    =========  =====================================================
    Vertex     ``{"id": int, "label": str, "properties": dict}``
    Edge       ``{"id": int, "label": str, "start_id": int,``
               ``"end_id": int, "properties": dict}``
    Path       ``list`` of converted entities
    list       recursively converted list
    dict       recursively converted dict
    scalar     returned as-is
    =========  =====================================================

    Args:
        obj: Value returned by psycopg3 / the AGE type loader.

    Returns:
        Plain Python value safe to serialise or pass to LangChain.
    """
    if _AGE_SDK_AVAILABLE:
        if isinstance(obj, Vertex):
            return {
                "id": obj.id,
                "label": obj.label,
                "properties": obj.properties or {},
            }
        if isinstance(obj, Edge):
            return {
                "id": obj.id,
                "label": obj.label,
                "start_id": obj.start_id,
                "end_id": obj.end_id,
                "properties": obj.properties or {},
            }
        if isinstance(obj, Path):
            return [agobj_to_dict(e) for e in obj]

    if isinstance(obj, list):
        return [agobj_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: agobj_to_dict(v) for k, v in obj.items()}
    return obj  # type: ignore[no-any-return]


def agtype_to_python(value: Any) -> AGEPythonValue:
    """Best-effort conversion for raw agtype strings (fallback path).

    When the SDK is active, psycopg3 handles conversion before this is called.
    This function is kept as a fallback for edge cases where a raw agtype
    string slips through (e.g. in metadata columns).

    Args:
        value: Raw value, possibly an agtype-encoded string.

    Returns:
        Converted Python value.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return agobj_to_dict(value)

    _SUFFIX_RE = re.compile(
        r"::(vertex|edge|path|agtype|integer|float|numeric|boolean|string)$",
        re.IGNORECASE,
    )
    cleaned = _SUFFIX_RE.sub("", value.strip())

    try:
        return json.loads(cleaned)  # type: ignore[no-any-return]
    except (json.JSONDecodeError, ValueError):
        pass
    return cleaned


def python_to_agtype(value: Any) -> str:
    """Serialise a Python value to an agtype-compatible Cypher literal.

    Uses OpenCypher-standard escaping:
    - Strings: single-quote doubling (``''``) — NOT backslash escaping.
    - Backslashes: doubled (``\\\\``) to prevent unintended Cypher escape sequences.
    - Booleans / null: lowercase literals.
    - Numbers: plain ``str()``.
    - Dicts / lists: JSON (for agtype map / list literals).

    Args:
        value: Python value to serialise.

    Returns:
        Cypher literal string (not yet wrapped in quotes for strings —
        the returned string includes the surrounding quotes).
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, (dict, list)):
        # AGE accepts JSON-compatible map / list literals in Cypher.
        return json.dumps(value)
    # Fallback: coerce to string
    escaped = str(value).replace("\\", "\\\\").replace("'", "''")
    return f"'{escaped}'"
