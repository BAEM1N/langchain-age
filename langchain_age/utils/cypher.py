"""Utilities for building and validating AGE Cypher queries."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple


def wrap_cypher_query(
    graph_name: str,
    cypher: str,
    columns: List[Tuple[str, str]],
) -> str:
    """Wrap a Cypher statement in the AGE SQL function call.

    Apache AGE requires Cypher to be executed through the ``cypher()`` SQL
    function inside a ``SELECT … FROM`` clause::

        SELECT * FROM cypher('graph', $$ MATCH (n) RETURN n $$) AS (n agtype);

    Args:
        graph_name: Name of the AGE graph.
        cypher: Pure Cypher query string (no surrounding SQL).
        columns: List of ``(alias, type)`` pairs that describe the projected
            columns, e.g. ``[("n", "agtype"), ("r", "agtype")]``.
            Pass ``[("result", "agtype")]`` as a safe default when the column
            list is not known in advance.

    Returns:
        Complete SQL statement ready for execution with psycopg2.
    """
    if not columns:
        columns = [("result", "agtype")]

    col_defs = ", ".join(f"{alias} {dtype}" for alias, dtype in columns)
    # Sanitise graph name (alphanumeric + underscores only)
    safe_graph = re.sub(r"[^a-zA-Z0-9_]", "", graph_name)
    # Use dollar-quoting so internal single-quotes are safe
    return f"SELECT * FROM cypher('{safe_graph}', $$ {cypher} $$) AS ({col_defs});"


def validate_cypher(cypher: str) -> Optional[str]:
    """Perform lightweight validation on a Cypher query string.

    Returns an error message string if the query looks invalid, or ``None``
    if no obvious problems are detected.

    This is a best-effort check; the database is the authoritative validator.

    Args:
        cypher: Cypher query string to validate.

    Returns:
        Error string or ``None``.
    """
    cypher_stripped = cypher.strip()

    if not cypher_stripped:
        return "Cypher query is empty."

    # Disallow embedded dollar-quote markers that would break the SQL wrapper
    if "$$" in cypher_stripped:
        return "Cypher query must not contain '$$' (dollar-quote markers)."

    # Ensure at least one recognisable Cypher keyword is present
    cypher_keywords = re.compile(
        r"\b(MATCH|CREATE|MERGE|RETURN|WITH|WHERE|DELETE|SET|REMOVE|UNWIND|OPTIONAL)\b",
        re.IGNORECASE,
    )
    if not cypher_keywords.search(cypher_stripped):
        return (
            "Cypher query does not contain any recognised Cypher clause "
            "(MATCH, CREATE, MERGE, RETURN, …)."
        )

    return None


def extract_cypher_return_aliases(cypher: str) -> List[str]:
    """Extract column aliases from the RETURN clause of a Cypher query.

    Used to auto-build the column definition list for ``wrap_cypher_query``.

    Args:
        cypher: Cypher query string.

    Returns:
        List of alias strings.  Falls back to ``["result"]`` if parsing fails.
    """
    # Find the last RETURN clause (handles WITH … RETURN chains)
    match = re.search(r"\bRETURN\b(.+?)(?:\bLIMIT\b|\bSKIP\b|\bORDER\b|$)", cypher, re.IGNORECASE | re.DOTALL)
    if not match:
        return ["result"]

    return_clause = match.group(1).strip()

    aliases: List[str] = []
    for term in return_clause.split(","):
        term = term.strip()
        # Handle "expr AS alias"
        as_match = re.search(r"\bAS\s+(\w+)\s*$", term, re.IGNORECASE)
        if as_match:
            aliases.append(as_match.group(1))
        else:
            # Use last identifier-like token (e.g. "n" from "n.name" → "name")
            tokens = re.findall(r"\w+", term)
            if tokens:
                aliases.append(tokens[-1])

    return aliases if aliases else ["result"]
