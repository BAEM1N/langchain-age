"""Apache AGE graph store for LangChain – backed by apache-age-python SDK.

Design notes (compared with langchain-neo4j ``Neo4jGraph``):
- Uses ``wrap_cypher_query`` instead of ``age_sdk.cypher()`` because the SDK
  generates ``cypher(NULL, NULL)`` which PG18 rejects (requires a name constant).
- Backtick-quotes all user-supplied labels and property names throughout,
  following the Neo4j convention: ``MERGE (n:`Label`) SET n.`prop` = …``.
- Uses ``''`` (single-quote doubling) for Cypher string escaping — the
  OpenCypher standard — instead of the non-standard ``\\'``.
- Exposes ``close()`` and context-manager support (``__enter__``/``__exit__``)
  matching the Neo4j driver pattern.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import age as age_sdk
    from age import setUpAge
    import psycopg
    from psycopg.client_cursor import ClientCursor
except ImportError as e:
    raise ImportError(
        "apache-age-python and psycopg are required.\n"
        "Install: pip install apache-age-python 'psycopg[binary]'"
    ) from e

from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore

from langchain_age.utils.agtype import agobj_to_dict
from langchain_age.utils.cypher import (
    escape_cypher_identifier,
    escape_cypher_string,
    extract_cypher_return_aliases,
    validate_cypher,
    wrap_cypher_query,
)


class AGEGraph(GraphStore):
    """LangChain ``GraphStore`` backed by PostgreSQL + Apache AGE.

    Mirrors the ``Neo4jGraph`` interface from *langchain-neo4j*.

    Uses the official **apache-age-python** SDK (psycopg3) for the
    ``setUpAge`` initialisation step (which registers agtype type adapters),
    but bypasses ``age_sdk.cypher()`` for query execution — the SDK generates
    ``cypher(NULL, NULL)`` which is rejected by PG18.  Instead, queries are
    built via :func:`~langchain_age.utils.cypher.wrap_cypher_query` which
    produces the PG18-compatible ``cypher('name', $$ … $$)`` form.

    Args:
        connection_string: psycopg3-compatible DSN or URI.
        graph_name: Name of the AGE graph (created automatically if absent).
        timeout: Optional statement timeout in seconds.
        refresh_schema: Load schema immediately on init.
        sanitize: Truncate large string values in query results.
        enhanced_schema: Sample property values to enrich the schema string.
        include_types: Whitelist of node/edge label types to expose in schema.
        exclude_types: Blacklist of node/edge label types to hide from schema.

    Example::

        graph = AGEGraph("host=localhost dbname=mydb user=foo password=bar",
                         graph_name="kg")
        graph.query("MATCH (n:Person) RETURN n.name AS name")

    Context-manager usage (mirrors Neo4j driver)::

        with AGEGraph(conn_str, "kg") as graph:
            graph.query("MATCH (n) RETURN count(n) AS total")
    """

    def __init__(
        self,
        connection_string: str,
        graph_name: str,
        *,
        timeout: Optional[float] = None,
        refresh_schema: bool = True,
        sanitize: bool = True,
        enhanced_schema: bool = False,
        include_types: Optional[List[str]] = None,
        exclude_types: Optional[List[str]] = None,
    ) -> None:
        self._conn_string = connection_string
        self.graph_name = graph_name
        self._timeout = timeout
        self._sanitize = sanitize
        self._enhanced_schema = enhanced_schema
        self._include_types = include_types or []
        self._exclude_types = exclude_types or []

        self._conn: psycopg.Connection = self._connect()
        self._ensure_extensions()
        self._ensure_graph()

        self.schema: str = ""
        self.structured_schema: Dict[str, Any] = {}

        if refresh_schema:
            self.refresh_schema()

    # ------------------------------------------------------------------
    # Context-manager support (mirrors Neo4j driver pattern)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        try:
            if not self._conn.closed:
                self._conn.close()
        except Exception:
            logger.debug("Exception while closing AGEGraph connection", exc_info=True)

    def __enter__(self) -> AGEGraph:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # GraphStore interface
    # ------------------------------------------------------------------

    @property
    def get_schema(self) -> str:
        return self.schema

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        return self.structured_schema

    def query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results as plain Python dicts.

        Builds a PG18-compatible ``SELECT * FROM cypher('graph', $$ … $$)``
        SQL statement directly (bypassing ``age_sdk.cypher()``), then converts
        ``Vertex`` / ``Edge`` / ``Path`` results to plain Python dicts via
        :func:`~langchain_age.utils.agtype.agobj_to_dict`.

        Args:
            query: Pure Cypher string.
            params: Not currently supported by AGE; values must be inlined
                (use :func:`~langchain_age.utils.cypher.escape_cypher_string`
                for safe inlining).

        Returns:
            List of row dicts, one per result row.

        Raises:
            ValueError: If the Cypher query fails basic validation.
            psycopg.Error: On database-level errors.
        """
        if params:
            logger.warning(
                "AGEGraph.query() received params=%r but AGE does not support "
                "parameterised Cypher — values must be inlined in the query string.",
                params,
            )

        error = validate_cypher(query)
        if error:
            raise ValueError(f"Invalid Cypher: {error}")

        aliases = extract_cypher_return_aliases(query)
        sql = wrap_cypher_query(
            self.graph_name,
            query,
            [(alias, "agtype") for alias in aliases],
        )

        with self._conn.cursor() as cur:
            if self._timeout:
                cur.execute(
                    "SET LOCAL statement_timeout = %s",
                    (int(self._timeout * 1000),),
                )
            try:
                cur.execute(sql)
                if cur.description is None:
                    self._conn.commit()
                    return []
                col_names = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
            except Exception:
                self._conn.rollback()
                raise

        self._conn.commit()

        results = []
        for row in rows:
            record: Dict[str, Any] = {}
            for col, val in zip(col_names, row):
                converted = agobj_to_dict(val)
                record[col] = self._sanitize_value(converted) if self._sanitize else converted
            results.append(record)
        return results

    def refresh_schema(self) -> None:
        """Re-introspect the AGE graph and update ``schema`` / ``structured_schema``."""
        node_labels = self._filter_labels(self._fetch_labels("v"))
        edge_labels = self._filter_labels(self._fetch_labels("e"))

        node_props: Dict[str, List[str]] = {}
        edge_props: Dict[str, List[str]] = {}

        for label in node_labels:
            node_props[label] = self._sample_props(label, "v")
        for label in edge_labels:
            edge_props[label] = self._sample_props(label, "e")

        relationships = self._fetch_relationships(edge_labels)

        self.structured_schema = {
            "node_props": node_props,
            "rel_props": edge_props,
            "relationships": relationships,
            "metadata": {"constraint": [], "index": []},
        }
        self.schema = self._build_schema_string(node_props, edge_props, relationships)

    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
    ) -> None:
        """Upsert ``GraphDocument`` objects into the AGE graph.

        All labels and property values are escaped using
        :func:`~langchain_age.utils.cypher.escape_cypher_identifier` (backtick
        quoting) and :func:`~langchain_age.utils.cypher.escape_cypher_string`
        (``''`` doubling) respectively, mirroring the Neo4j convention.
        """
        for doc in graph_documents:
            for node in doc.nodes:
                label = escape_cypher_identifier(node.type)
                props = self._props_to_cypher(node.properties or {})
                node_id = escape_cypher_string(str(node.id))
                self._run_write(
                    f"MERGE (n:{label} {{id: '{node_id}'}}) SET n += {props}"
                )

            for rel in doc.relationships:
                src_label = escape_cypher_identifier(rel.source.type)
                tgt_label = escape_cypher_identifier(rel.target.type)
                rel_label = escape_cypher_identifier(rel.type)
                props = self._props_to_cypher(rel.properties or {})
                src_id = escape_cypher_string(str(rel.source.id))
                tgt_id = escape_cypher_string(str(rel.target.id))
                self._run_write(
                    f"MATCH (a:{src_label} {{id: '{src_id}'}}), "
                    f"(b:{tgt_label} {{id: '{tgt_id}'}}) "
                    f"MERGE (a)-[r:{rel_label}]->(b) SET r += {props}"
                )

            if include_source and doc.source:
                src_id_val = escape_cypher_string(
                    doc.source.metadata.get("source", "unknown")
                )
                content = escape_cypher_string(doc.source.page_content[:500])
                self._run_write(
                    f"MERGE (s:Document {{source: '{src_id_val}'}}) "
                    f"SET s.content = '{content}'"
                )
                for node in doc.nodes:
                    node_label = escape_cypher_identifier(node.type)
                    node_id_val = escape_cypher_string(str(node.id))
                    self._run_write(
                        f"MATCH (s:Document {{source: '{src_id_val}'}}), "
                        f"(n:{node_label} {{id: '{node_id_val}'}}) "
                        f"MERGE (s)-[:MENTIONS]->(n)"
                    )

    # ------------------------------------------------------------------
    # Graph management
    # ------------------------------------------------------------------

    def create_graph(self) -> None:
        """Create the AGE graph if it does not already exist."""
        self._ensure_graph()

    def drop_graph(self) -> None:
        """Drop the AGE graph and all its data. **Irreversible.**"""
        with self._conn.cursor() as cur:
            cur.execute("SELECT drop_graph(%s, true);", (self.graph_name,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> psycopg.Connection:
        # ClientCursor is required by setUpAge (uses mogrify internally).
        conn = psycopg.connect(self._conn_string, cursor_factory=ClientCursor)
        conn.autocommit = False
        setUpAge(conn, self.graph_name)
        return conn

    def _ensure_extensions(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS age;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self._conn.commit()

    def _ensure_graph(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM ag_catalog.ag_graph WHERE name = %s;",
                (self.graph_name,),
            )
            (count,) = cur.fetchone()
            if count == 0:
                cur.execute("SELECT create_graph(%s);", (self.graph_name,))
        self._conn.commit()

    def _fetch_labels(self, kind: str) -> List[str]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT name FROM ag_catalog.ag_label
                WHERE graph = (SELECT graphid FROM ag_catalog.ag_graph WHERE name = %s)
                  AND kind = %s
                ORDER BY name;
                """,
                (self.graph_name, kind),
            )
            return [row[0] for row in cur.fetchall()]

    def _filter_labels(self, labels: List[str]) -> List[str]:
        if self._include_types:
            labels = [l for l in labels if l in self._include_types]
        if self._exclude_types:
            labels = [l for l in labels if l not in self._exclude_types]
        return labels

    def _sample_props(self, label: str, kind: str) -> List[str]:
        """Sample property keys for *label* by fetching up to 5 nodes/edges.

        Uses backtick-quoted label to handle reserved Cypher words, following
        the Neo4j pattern.  Failures are logged and return an empty list rather
        than propagating — a missing property list degrades schema quality but
        should not abort the entire ``refresh_schema`` call.
        """
        alias = "n" if kind == "v" else "r"
        escaped_label = escape_cypher_identifier(label)
        cypher = f"MATCH ({alias}:{escaped_label}) RETURN {alias} LIMIT 5"
        try:
            rows = self.query(cypher)
        except Exception as exc:
            logger.warning(
                "Failed to sample properties for label %r (%s): %s",
                label,
                "vertex" if kind == "v" else "edge",
                exc,
            )
            return []
        keys: set[str] = set()
        for row in rows:
            val = next(iter(row.values()), None)
            if isinstance(val, dict):
                props = val.get("properties", val)
                if isinstance(props, dict):
                    keys.update(props.keys())
        return sorted(keys)

    def _fetch_relationships(self, edge_labels: List[str]) -> List[Dict[str, str]]:
        triples: List[Dict[str, str]] = []
        for label in edge_labels:
            escaped = escape_cypher_identifier(label)
            try:
                rows = self.query(
                    f"MATCH (a)-[r:{escaped}]->(b) RETURN a, r, b LIMIT 5"
                )
            except Exception as exc:
                logger.warning(
                    "Failed to fetch relationships for edge label %r: %s", label, exc
                )
                continue
            for row in rows:
                a_label = self._extract_label(row.get("a"))
                b_label = self._extract_label(row.get("b"))
                triple = {"start": a_label, "type": label, "end": b_label}
                if triple not in triples:
                    triples.append(triple)
        return triples

    @staticmethod
    def _extract_label(node: Any) -> str:
        if isinstance(node, dict):
            lbl = node.get("label", node.get("labels", ""))
            if isinstance(lbl, list):
                return lbl[0] if lbl else "Unknown"
            return str(lbl) if lbl else "Unknown"
        return "Unknown"

    def _run_write(self, cypher: str) -> None:
        """Execute a write-only Cypher statement (no RETURN expected)."""
        sql = wrap_cypher_query(self.graph_name, cypher, [("v", "agtype")])
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql)
        except Exception:
            self._conn.rollback()
            raise
        self._conn.commit()

    @staticmethod
    def _props_to_cypher(props: Dict[str, Any]) -> str:
        """Serialise a Python dict to a Cypher map literal.

        - Keys are backtick-quoted (handles reserved words like ``desc``).
        - String values use ``''`` escaping (OpenCypher standard).
        - Booleans, null, and numbers are rendered as Cypher literals.

        Example::

            {"name": "Alice's cat", "age": 3}
            → {`name`: 'Alice''s cat', `age`: 3}
        """
        if not props:
            return "{}"
        pairs = []
        for k, v in props.items():
            key = escape_cypher_identifier(k)
            if isinstance(v, bool):
                pairs.append(f"{key}: {'true' if v else 'false'}")
            elif v is None:
                pairs.append(f"{key}: null")
            elif isinstance(v, (int, float)):
                pairs.append(f"{key}: {v}")
            elif isinstance(v, str):
                pairs.append(f"{key}: '{escape_cypher_string(v)}'")
            else:
                # dict / list — JSON-serialise as agtype map/list literal
                import json
                pairs.append(f"{key}: {json.dumps(v)}")
        return "{" + ", ".join(pairs) + "}"

    @staticmethod
    def _sanitize_value(value: Any, max_len: int = 1000) -> Any:
        """Truncate oversized strings and strip internal ``_`` prefixed keys."""
        if isinstance(value, str) and len(value) > max_len:
            return value[:max_len] + "…"
        if isinstance(value, dict):
            return {
                k: AGEGraph._sanitize_value(v, max_len)
                for k, v in value.items()
                if not str(k).startswith("_")
            }
        if isinstance(value, list):
            return [AGEGraph._sanitize_value(v, max_len) for v in value]
        return value

    @staticmethod
    def _build_schema_string(
        node_props: Dict[str, List[str]],
        edge_props: Dict[str, List[str]],
        relationships: List[Dict[str, str]],
    ) -> str:
        lines = ["Node labels and properties:"]
        for label, props in node_props.items():
            lines.append(
                f"  :{label} {{{', '.join(props) or '(no properties sampled)'}}}"
            )
        lines += ["", "Relationship types and properties:"]
        for label, props in edge_props.items():
            lines.append(
                f"  [:{label}] {{{', '.join(props) or '(no properties sampled)'}}}"
            )
        lines += ["", "Relationship patterns:"]
        for rel in relationships:
            lines.append(f"  (:{rel['start']})-[:{rel['type']}]->(:{rel['end']})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"AGEGraph(graph='{self.graph_name}')"
