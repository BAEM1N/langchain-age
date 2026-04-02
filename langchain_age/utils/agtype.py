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
    from age import Edge, Path, Vertex
    _AGE_SDK_AVAILABLE = True
except ImportError:
    _AGE_SDK_AVAILABLE = False
    Vertex = Edge = Path = None  # type: ignore[assignment,misc]


def agobj_to_dict(obj: Any) -> Any:
    """Convert an apache-age-python SDK object to a plain Python value.

    Handles:
    - ``Vertex``  → ``{"id": ..., "label": ..., "properties": {...}}``
    - ``Edge``    → ``{"id": ..., "label": ..., "start_id": ..., "end_id": ..., "properties": {...}}``
    - ``Path``    → list of converted entities
    - ``list``    → recursively converted list
    - ``dict``    → recursively converted dict
    - scalars     → returned as-is
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
    return obj


def agtype_to_python(value: Any) -> Any:
    """Best-effort conversion for raw agtype strings (fallback path).

    When the SDK is active, psycopg3 handles conversion before this is
    called.  This function is kept as a fallback for edge cases where
    a raw agtype string slips through (e.g. in metadata columns).
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return agobj_to_dict(value)

    _SUFFIX_RE = re.compile(
        r"::(vertex|edge|path|agtype|integer|float|numeric|boolean|string)$",
        re.IGNORECASE,
    )
    value = _SUFFIX_RE.sub("", value.strip())

    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        pass
    return value


def python_to_agtype(value: Any) -> str:
    """Serialise a Python value to an agtype-compatible Cypher literal."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "\\'") + "'"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return f"'{value}'"
