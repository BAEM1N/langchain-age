"""Apache AGE graph store for LangChain – backed by apache-age-python SDK."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

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
    extract_cypher_return_aliases,
    validate_cypher,
    wrap_cypher_query,
)


class AGEGraph(GraphStore):
    """LangChain ``GraphStore`` backed by PostgreSQL + Apache AGE.

    Uses the official **apache-age-python** SDK (psycopg3) for connections,
    which registers agtype type adapters automatically so query results come
    back as native ``Vertex`` / ``Edge`` / ``Path`` objects.

    Mirrors the ``Neo4jGraph`` interface from *langchain-neo4j*.

    Args:
        connection_string: psycopg3-compatible DSN or URI, e.g.
            ``"host=localhost port=5432 dbname=mydb user=foo password=bar"``.
        graph_name: Name of the AGE graph (created automatically if absent).
        timeout: Optional statement timeout in seconds.
        refresh_schema: Load schema immediately on init.
        sanitize: Truncate large string values in query results.
        enhanced_schema: Sample property values to enrich the schema string.
        include_types: Whitelist of node/edge label types to expose in schema.
        exclude_types: Blacklist of node/edge label types to hide from schema.
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

        The SDK wraps the Cypher in the required SQL ``cypher()`` call and
        deserialises agtype results into ``Vertex`` / ``Edge`` / ``Path``
        objects, which are then converted to plain dicts.

        Args:
            query: Pure Cypher string.
            params: Not currently supported by AGE; values must be inlined.

        Returns:
            List of row dicts, one per result row.
        """
        error = validate_cypher(query)
        if error:
            raise ValueError(f"Invalid Cypher: {error}")

        aliases = extract_cypher_return_aliases(query)

        with self._conn.cursor() as cur:
            if self._timeout:
                cur.execute(
                    "SET LOCAL statement_timeout = %s",
                    (int(self._timeout * 1000),),
                )
            try:
                # age_sdk.cypher() generates cypher(NULL,NULL) which PG18 rejects.
                # Build the SQL directly with literal graph name + dollar-quoted query.
                sql = wrap_cypher_query(
                    self.graph_name,
                    query,
                    [(alias, "agtype") for alias in aliases],
                )
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
        """Upsert ``GraphDocument`` objects into the AGE graph."""
        for doc in graph_documents:
            for node in doc.nodes:
                props = self._props_to_cypher(node.properties or {})
                self._run_write(f"MERGE (n:{node.type} {{id: '{node.id}'}}) SET n += {props}")

            for rel in doc.relationships:
                props = self._props_to_cypher(rel.properties or {})
                self._run_write(
                    f"MATCH (a:{rel.source.type} {{id: '{rel.source.id}'}}), "
                    f"(b:{rel.target.type} {{id: '{rel.target.id}'}}) "
                    f"MERGE (a)-[r:{rel.type}]->(b) SET r += {props}"
                )

            if include_source and doc.source:
                src_id = doc.source.metadata.get("source", "unknown")
                content = doc.source.page_content.replace("'", "\\'")[:500]
                self._run_write(
                    f"MERGE (s:Document {{source: '{src_id}'}}) SET s.content = '{content}'"
                )
                for node in doc.nodes:
                    self._run_write(
                        f"MATCH (s:Document {{source: '{src_id}'}}), "
                        f"(n:{node.type} {{id: '{node.id}'}}) MERGE (s)-[:MENTIONS]->(n)"
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
        # ClientCursor is required by the AGE SDK's cypher() helper (uses mogrify)
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
        alias = "n" if kind == "v" else "r"
        cypher = f"MATCH ({alias}:{label}) RETURN {alias} LIMIT 5"
        try:
            rows = self.query(cypher)
        except Exception:
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
            try:
                rows = self.query(
                    f"MATCH (a)-[r:{label}]->(b) RETURN a, r, b LIMIT 5"
                )
            except Exception:
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
        sql = wrap_cypher_query(self.graph_name, cypher, [("v", "agtype")])
        with self._conn.cursor() as cur:
            cur.execute(sql)
        self._conn.commit()

    @staticmethod
    def _props_to_cypher(props: Dict[str, Any]) -> str:
        if not props:
            return "{}"
        pairs = []
        for k, v in props.items():
            if isinstance(v, str):
                pairs.append(f"{k}: '{v.replace(chr(39), chr(92)+chr(39))}'")
            elif isinstance(v, bool):
                pairs.append(f"{k}: {'true' if v else 'false'}")
            elif v is None:
                pairs.append(f"{k}: null")
            else:
                pairs.append(f"{k}: {v}")
        return "{" + ", ".join(pairs) + "}"

    @staticmethod
    def _sanitize_value(value: Any, max_len: int = 1000) -> Any:
        if isinstance(value, str) and len(value) > max_len:
            return value[:max_len] + "…"
        if isinstance(value, dict):
            return {k: AGEGraph._sanitize_value(v, max_len) for k, v in value.items() if not str(k).startswith("_")}
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
            lines.append(f"  :{label} {{{', '.join(props) or '(no properties sampled)'}}}")
        lines += ["", "Relationship types and properties:"]
        for label, props in edge_props.items():
            lines.append(f"  [:{label}] {{{', '.join(props) or '(no properties sampled)'}}}")
        lines += ["", "Relationship patterns:"]
        for rel in relationships:
            lines.append(f"  (:{rel['start']})-[:{rel['type']}]->(:{rel['end']})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"AGEGraph(graph='{self.graph_name}')"
