"""Apache AGE graph store integration for LangChain."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import psycopg2
    import psycopg2.extras
except ImportError as e:
    raise ImportError(
        "psycopg2 is required for AGEGraph. "
        "Install it with: pip install psycopg2-binary"
    ) from e

from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore

from langchain_age.utils.agtype import agtype_to_python
from langchain_age.utils.cypher import (
    extract_cypher_return_aliases,
    validate_cypher,
    wrap_cypher_query,
)

# ---------------------------------------------------------------------------
# Schema introspection queries
# ---------------------------------------------------------------------------

_SCHEMA_NODE_LABELS_QUERY = """
SELECT name
FROM ag_catalog.ag_label
WHERE graph = (SELECT graphid FROM ag_catalog.ag_graph WHERE name = %s)
  AND kind = 'v'
ORDER BY name;
"""

_SCHEMA_EDGE_LABELS_QUERY = """
SELECT name
FROM ag_catalog.ag_label
WHERE graph = (SELECT graphid FROM ag_catalog.ag_graph WHERE name = %s)
  AND kind = 'e'
ORDER BY name;
"""

# Sample a few properties per label to infer schema
_SCHEMA_NODE_PROPS_QUERY = """
SELECT * FROM cypher(%s, $$
    MATCH (n:{label})
    RETURN n
    LIMIT 5
$$) AS (n agtype);
"""

_SCHEMA_EDGE_PROPS_QUERY = """
SELECT * FROM cypher(%s, $$
    MATCH ()-[r:{label}]->()
    RETURN r
    LIMIT 5
$$) AS (r agtype);
"""


class AGEGraph(GraphStore):
    """LangChain ``GraphStore`` backed by PostgreSQL + Apache AGE.

    Mirrors the interface of ``langchain_neo4j.Neo4jGraph`` so that chains
    written for Neo4j can be adapted with minimal changes.

    Args:
        connection_string: ``psycopg2``-compatible DSN, e.g.
            ``"host=localhost port=5432 dbname=mydb user=foo password=bar"``.
            Also accepts the ``postgresql://user:pass@host/db`` URI format.
        graph_name: Name of the AGE graph to use.  The graph is created
            automatically if it does not exist.
        timeout: Optional statement-level timeout in seconds.
        refresh_schema: Whether to load the graph schema immediately on init.
        sanitize: Strip keys / values that may cause issues in downstream LLM
            prompts (very large strings, binary blobs, etc.).
        enhanced_schema: Sample property values to enrich the schema string.
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
    ) -> None:
        self._conn_string = connection_string
        self.graph_name = graph_name
        self._timeout = timeout
        self._sanitize = sanitize
        self._enhanced_schema = enhanced_schema

        self._conn: psycopg2.extensions.connection = self._connect()
        self._ensure_extensions()
        self._ensure_graph()

        self.schema: str = ""
        self.structured_schema: Dict[str, Any] = {}

        if refresh_schema:
            self.refresh_schema()

    # ------------------------------------------------------------------
    # GraphStore interface
    # ------------------------------------------------------------------

    @property
    def get_schema(self) -> str:  # noqa: D102
        return self.schema

    @property
    def get_structured_schema(self) -> Dict[str, Any]:  # noqa: D102
        return self.structured_schema

    def query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query against the AGE graph.

        The raw Cypher is automatically wrapped in the SQL ``cypher()``
        function call required by Apache AGE.

        Args:
            query: Cypher query string.
            params: *Currently unused* – AGE does not support parameterised
                Cypher.  Embed literal values directly in the query string.

        Returns:
            List of dicts, one per result row.
        """
        error = validate_cypher(query)
        if error:
            raise ValueError(f"Invalid Cypher: {error}")

        aliases = extract_cypher_return_aliases(query)
        columns = [(alias, "agtype") for alias in aliases]
        sql = wrap_cypher_query(self.graph_name, query, columns)

        with self._conn.cursor() as cur:
            if self._timeout:
                cur.execute(
                    "SET LOCAL statement_timeout = %s",
                    (int(self._timeout * 1000),),
                )
            cur.execute(sql)
            if cur.description is None:
                self._conn.commit()
                return []
            col_names = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

        self._conn.commit()
        results = []
        for row in rows:
            record: Dict[str, Any] = {}
            for col, val in zip(col_names, row):
                parsed = agtype_to_python(val)
                record[col] = self._sanitize_value(parsed) if self._sanitize else parsed
            results.append(record)
        return results

    def refresh_schema(self) -> None:
        """Re-introspect the AGE graph and update ``schema`` / ``structured_schema``."""
        node_labels = self._fetch_labels("v")
        edge_labels = self._fetch_labels("e")

        node_props: Dict[str, List[str]] = {}
        edge_props: Dict[str, List[str]] = {}

        for label in node_labels:
            node_props[label] = self._sample_props(label, "v")

        for label in edge_labels:
            edge_props[label] = self._sample_props(label, "e")

        # Try to infer relationships (start_label)-[edge]->(end_label)
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

        Args:
            graph_documents: Documents produced by an LLM-based graph extractor.
            include_source: If ``True``, also create a ``Document`` node linked
                to all extracted entities via a ``MENTIONS`` relationship.
        """
        for doc in graph_documents:
            # Upsert nodes
            for node in doc.nodes:
                props = self._props_to_cypher(node.properties or {})
                cypher = (
                    f"MERGE (n:{node.type} {{id: '{node.id}'}}) "
                    f"SET n += {props}"
                )
                self._run_write_cypher(cypher)

            # Upsert edges
            for rel in doc.relationships:
                props = self._props_to_cypher(rel.properties or {})
                cypher = (
                    f"MATCH (a:{rel.source.type} {{id: '{rel.source.id}'}}), "
                    f"(b:{rel.target.type} {{id: '{rel.target.id}'}}) "
                    f"MERGE (a)-[r:{rel.type}]->(b) "
                    f"SET r += {props}"
                )
                self._run_write_cypher(cypher)

            if include_source and doc.source:
                source_id = doc.source.metadata.get("source", "unknown")
                safe_content = doc.source.page_content.replace("'", "\\'")[:500]
                cypher = (
                    f"MERGE (s:Document {{source: '{source_id}'}}) "
                    f"SET s.content = '{safe_content}'"
                )
                self._run_write_cypher(cypher)

                for node in doc.nodes:
                    cypher = (
                        f"MATCH (s:Document {{source: '{source_id}'}}), "
                        f"(n:{node.type} {{id: '{node.id}'}}) "
                        f"MERGE (s)-[:MENTIONS]->(n)"
                    )
                    self._run_write_cypher(cypher)

    # ------------------------------------------------------------------
    # Graph management helpers
    # ------------------------------------------------------------------

    def create_graph(self) -> None:
        """Create the AGE graph if it does not already exist."""
        self._ensure_graph()

    def drop_graph(self) -> None:
        """Drop the AGE graph and all its data. **Irreversible.**"""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT drop_graph(%s, true);",
                (self.graph_name,),
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> psycopg2.extensions.connection:
        conn = psycopg2.connect(self._conn_string)
        conn.autocommit = False
        # Load AGE and set search path
        with conn.cursor() as cur:
            cur.execute("LOAD 'age';")
            cur.execute("SET search_path = ag_catalog, \"$user\", public;")
        conn.commit()
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
        """Return all vertex (``'v'``) or edge (``'e'``) label names."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT name FROM ag_catalog.ag_label
                WHERE graph = (
                    SELECT graphid FROM ag_catalog.ag_graph WHERE name = %s
                ) AND kind = %s
                ORDER BY name;
                """,
                (self.graph_name, kind),
            )
            return [row[0] for row in cur.fetchall()]

    def _sample_props(self, label: str, kind: str) -> List[str]:
        """Sample up to 5 nodes/edges of the given label and collect property keys."""
        prop_alias = "n" if kind == "v" else "r"
        cypher = f"MATCH ({prop_alias}:{label}) RETURN {prop_alias} LIMIT 5"
        try:
            rows = self.query(cypher)
        except Exception:
            return []

        keys: set[str] = set()
        for row in rows:
            val = row.get(prop_alias) or row.get(list(row.keys())[0])
            if isinstance(val, dict):
                props = val.get("properties", val)
                if isinstance(props, dict):
                    keys.update(props.keys())
        return sorted(keys)

    def _fetch_relationships(self, edge_labels: List[str]) -> List[Dict[str, str]]:
        """Infer (start_label, edge_label, end_label) triples by sampling."""
        triples: List[Dict[str, str]] = []
        for label in edge_labels:
            cypher = f"MATCH (a)-[r:{label}]->(b) RETURN a, r, b LIMIT 5"
            try:
                rows = self.query(cypher)
            except Exception:
                continue
            for row in rows:
                a = row.get("a") or {}
                b = row.get("b") or {}
                a_label = self._extract_label(a)
                b_label = self._extract_label(b)
                triple = {"start": a_label, "type": label, "end": b_label}
                if triple not in triples:
                    triples.append(triple)
        return triples

    @staticmethod
    def _extract_label(node: Any) -> str:
        if isinstance(node, dict):
            labels = node.get("label", node.get("labels", ""))
            if isinstance(labels, list):
                return labels[0] if labels else "Unknown"
            return str(labels) if labels else "Unknown"
        return "Unknown"

    def _run_write_cypher(self, cypher: str) -> None:
        """Execute a write-only Cypher statement (CREATE / MERGE / SET)."""
        sql = wrap_cypher_query(self.graph_name, cypher, [("v", "agtype")])
        with self._conn.cursor() as cur:
            cur.execute(sql)
        self._conn.commit()

    @staticmethod
    def _props_to_cypher(props: Dict[str, Any]) -> str:
        """Serialize a Python dict to an AGE-compatible Cypher map literal."""
        if not props:
            return "{}"
        pairs = []
        for k, v in props.items():
            if isinstance(v, str):
                v_str = v.replace("'", "\\'")
                pairs.append(f"{k}: '{v_str}'")
            elif isinstance(v, bool):
                pairs.append(f"{k}: {'true' if v else 'false'}")
            elif v is None:
                pairs.append(f"{k}: null")
            else:
                pairs.append(f"{k}: {v}")
        return "{" + ", ".join(pairs) + "}"

    @staticmethod
    def _sanitize_value(value: Any, max_len: int = 1000) -> Any:
        """Truncate large string values to keep LLM prompts manageable."""
        if isinstance(value, str) and len(value) > max_len:
            return value[:max_len] + "…"
        if isinstance(value, dict):
            return {
                k: AGEGraph._sanitize_value(v, max_len)
                for k, v in value.items()
                if not k.startswith("_")
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
            prop_str = ", ".join(props) if props else "(no properties sampled)"
            lines.append(f"  :{label} {{{prop_str}}}")

        lines.append("")
        lines.append("Relationship types and properties:")
        for label, props in edge_props.items():
            prop_str = ", ".join(props) if props else "(no properties sampled)"
            lines.append(f"  [:{label}] {{{prop_str}}}")

        lines.append("")
        lines.append("Relationship patterns:")
        for rel in relationships:
            lines.append(
                f"  (:{rel['start']})-[:{rel['type']}]->(:{rel['end']})"
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"AGEGraph(graph='{self.graph_name}')"
