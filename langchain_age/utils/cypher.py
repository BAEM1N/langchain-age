"""Utilities for building and validating AGE Cypher queries.

Design notes (compared with langchain-neo4j):
- ``escape_cypher_identifier`` mirrors Neo4j's backtick-quoting convention
  for property names, labels, and aliases that may contain reserved words.
- ``escape_cypher_string`` uses Cypher-standard single-quote doubling (``''``)
  rather than backslash escaping, which is the OpenCypher specification.
- ``validate_sql_identifier`` guards table/schema names that land in raw SQL
  f-strings (following the langchain-postgres pattern of rejecting unsafe names
  early rather than sanitising silently).
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cypher identifier / string helpers
# ---------------------------------------------------------------------------

# AGE Cypher reserved words that CANNOT be used as unquoted property accessors
# or aliases.  Tested exhaustively against AGE 1.7.0 / PG18.
CYPHER_RESERVED_WORDS: frozenset[str] = frozenset(
    [
        "all", "and", "as", "asc", "ascending", "by",
        "case", "contains", "count", "create",
        "delete", "desc", "descending", "detach", "distinct",
        "else", "end", "ends",
        "false", "filter",
        "in", "is",
        "keys",
        "labels", "limit",
        "match", "merge",
        "not", "null",
        "optional", "or", "order",
        "remove", "return",
        "set", "skip", "starts",
        "then", "true", "type",
        "union", "unique", "unwind",
        "when", "where", "with",
        "xor",
    ]
)


def escape_cypher_identifier(name: str) -> str:
    """Wrap *name* in Cypher backtick quotes, escaping embedded backticks.

    Mirrors the Neo4j pattern used throughout ``langchain-neo4j``::

        f"MATCH (n:`{node_label}`) SET n.`{prop}` = ..."

    Should be used for:
    - Node / relationship labels
    - Property names (both read and write)
    - RETURN aliases that might clash with reserved words

    Args:
        name: Raw identifier string (e.g. ``"desc"``, ``"my label"``).

    Returns:
        Backtick-quoted identifier (e.g. ``"`desc`"``, ``"`my label`"``).
    """
    return "`" + name.replace("`", "``") + "`"


def escape_cypher_string(value: str) -> str:
    """Escape a string value for safe embedding in a Cypher literal.

    Uses the OpenCypher standard: single-quote doubling (``''``).
    Backslash escaping (``\\'``) is NOT standard Cypher and should be avoided.

    Also escapes backslashes to prevent unintended Cypher escape sequences.

    Args:
        value: Raw Python string.

    Returns:
        Escaped string suitable for wrapping in single quotes in Cypher.

    Example::

        cypher = f"MERGE (n:Person {{name: '{escape_cypher_string(name)}'}}) "
    """
    return value.replace("\\", "\\\\").replace("'", "''")


# ---------------------------------------------------------------------------
# SQL identifier validation (langchain-postgres pattern)
# ---------------------------------------------------------------------------

_SAFE_SQL_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_sql_identifier(name: str, context: str = "identifier") -> str:
    """Validate that *name* is a safe bare SQL identifier.

    Follows the ``langchain-postgres`` convention of rejecting unsafe names
    early (at object construction) rather than silently sanitising them.

    Only allows letters, digits, and underscores, starting with a letter or
    underscore — identical to PostgreSQL unquoted-identifier rules.

    Args:
        name: Candidate identifier (table name, schema name, index name …).
        context: Human-readable label used in the error message.

    Returns:
        *name* unchanged if valid.

    Raises:
        ValueError: If *name* contains characters outside ``[a-zA-Z0-9_]``
            or starts with a digit.
    """
    if not _SAFE_SQL_IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Unsafe {context}: {name!r}. "
            "Only letters, digits, and underscores are allowed, "
            "and the name must start with a letter or underscore."
        )
    return name


# ---------------------------------------------------------------------------
# SQL query builder
# ---------------------------------------------------------------------------


def wrap_cypher_query(
    graph_name: str,
    cypher: str,
    columns: List[Tuple[str, str]],
) -> str:
    """Wrap a Cypher statement in the AGE ``cypher()`` SQL function call.

    Generates the PG18-compatible form::

        SELECT * FROM cypher('graph', $$ MATCH … $$) AS (col agtype);

    The ``age_prepare_cypher`` + ``cypher(NULL, NULL)`` two-phase approach
    used by ``apache-age-python`` ≤ 0.0.7 is rejected by PG18's parser
    (requires a name constant, not NULL).  This function generates the
    correct single-shot form.

    Args:
        graph_name: AGE graph name (sanitised to alphanumeric + underscore).
        cypher: Pure Cypher query string.
        columns: ``(alias, sql_type)`` pairs, e.g.
            ``[("n", "agtype"), ("r", "agtype")]``.
            Falls back to ``[("result", "agtype")]`` when empty.

    Returns:
        Complete SQL statement ready for ``psycopg`` execution.
    """
    if not columns:
        columns = [("result", "agtype")]

    col_defs = ", ".join(f"{alias} {dtype}" for alias, dtype in columns)
    # Sanitise graph name — silently strip unsafe chars to match AGE's own rules.
    safe_graph = re.sub(r"[^a-zA-Z0-9_]", "", graph_name)
    if safe_graph != graph_name:
        logger.warning(
            "Graph name %r was sanitised to %r for SQL embedding.",
            graph_name,
            safe_graph,
        )
    return f"SELECT * FROM cypher('{safe_graph}', $$ {cypher} $$) AS ({col_defs});"


# ---------------------------------------------------------------------------
# Cypher query analysis helpers
# ---------------------------------------------------------------------------


def validate_cypher(cypher: str) -> Optional[str]:
    """Lightweight validation of a Cypher query string.

    Returns an error message if the query looks invalid, or ``None`` if it
    passes all checks.  The database remains the authoritative validator.

    Checks:
    - Non-empty query
    - No embedded dollar-quote markers (``$$``) that would break the SQL wrapper
    - Presence of at least one recognised Cypher keyword

    Args:
        cypher: Cypher query string to validate.

    Returns:
        Error string, or ``None`` on success.
    """
    stripped = cypher.strip()

    if not stripped:
        return "Cypher query is empty."

    if "$$" in stripped:
        return "Cypher query must not contain '$$' (dollar-quote markers)."

    _CYPHER_KEYWORD_RE = re.compile(
        r"\b(MATCH|CREATE|MERGE|RETURN|WITH|WHERE|DELETE|SET|REMOVE|UNWIND|OPTIONAL)\b",
        re.IGNORECASE,
    )
    if not _CYPHER_KEYWORD_RE.search(stripped):
        return (
            "Cypher query does not contain any recognised Cypher clause "
            "(MATCH, CREATE, MERGE, RETURN, …)."
        )

    return None


def extract_cypher_return_aliases(cypher: str) -> List[str]:
    """Extract column aliases from the RETURN clause of a Cypher query.

    Used to auto-build the column definition list for :func:`wrap_cypher_query`.

    Handles:
    - ``RETURN expr AS alias`` → ``alias``
    - ``RETURN n.property`` → last identifier token (e.g. ``property``)
    - Multiple comma-separated items
    - LIMIT / SKIP / ORDER terminators

    Args:
        cypher: Cypher query string.

    Returns:
        List of alias strings.  Falls back to ``["result"]`` if parsing fails.
    """
    match = re.search(
        r"\bRETURN\b(.+?)(?:\bLIMIT\b|\bSKIP\b|\bORDER\b|$)",
        cypher,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ["result"]

    return_clause = match.group(1).strip()
    aliases: List[str] = []

    for term in return_clause.split(","):
        term = term.strip()
        as_match = re.search(r"\bAS\s+(\w+)\s*$", term, re.IGNORECASE)
        if as_match:
            aliases.append(as_match.group(1))
        else:
            tokens = re.findall(r"\w+", term)
            if tokens:
                aliases.append(tokens[-1])

    return aliases if aliases else ["result"]
