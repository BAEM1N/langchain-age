"""Utilities for converting between Apache AGE agtype and Python objects."""
from __future__ import annotations

import json
import re
from typing import Any


def agtype_to_python(value: Any) -> Any:
    """Convert an agtype value returned from AGE into a native Python object.

    AGE returns results as agtype strings with optional type suffixes like
    ``::vertex``, ``::edge``, ``::path``. This function strips those suffixes
    and parses the JSON-like payload into a Python dict / list / scalar.

    Args:
        value: Raw value returned from psycopg2 cursor (str, int, float, etc.)

    Returns:
        Native Python object (dict, list, str, int, float, bool, or None).
    """
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    # Strip AGE graph-type suffixes produced by agtype output
    _SUFFIX_RE = re.compile(
        r"::(vertex|edge|path|agtype|integer|float|numeric|boolean|string)$",
        re.IGNORECASE,
    )
    value = _SUFFIX_RE.sub("", value.strip())

    # Try JSON decode first (handles objects, arrays, quoted strings, booleans, numbers)
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        pass

    # Bare unquoted string fallback
    return value


def python_to_agtype(value: Any) -> str:
    """Serialize a Python value to an agtype-compatible string literal.

    This is used when injecting literal values into Cypher query strings.
    Prefer parameterised queries where possible.

    Args:
        value: Python value to serialize.

    Returns:
        String representation suitable for embedding in a Cypher expression.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        # Escape single quotes inside string
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return f"'{value}'"
