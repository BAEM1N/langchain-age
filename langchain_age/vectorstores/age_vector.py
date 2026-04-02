"""pgvector-backed VectorStore with Apache AGE graph node linkage.

Design notes (compared with langchain-neo4j / langchain-postgres):

``Neo4jVector`` reference points adopted:
- Backtick-quoting for all user-supplied property names in generated Cypher.
- Context-manager / ``close()`` support.
- ``batch_size`` parameter on ``add_texts`` (1 000 rows per transaction,
  mirroring Neo4j's "IN TRANSACTIONS OF 1000 ROWS" batch pattern).

``langchain-postgres`` (PGVector) reference points adopted:
- Double-quoted SQL identifiers for table and index names.
- ``validate_sql_identifier`` guard at construction time rather than silent
  sanitisation — failing fast is safer than silently renaming a table.
- ``executemany`` for batch INSERT (replaces N individual INSERT round-trips).
- ``asyncio.get_running_loop()`` instead of the deprecated ``get_event_loop()``.

Security improvements over v0.3.x:
- ``collection_name`` is validated at ``__init__`` time.
- All index and table names are double-quoted in SQL.
- Property names in Cypher are backtick-quoted (handles reserved words).
- Cypher string values use ``''`` doubling (OpenCypher standard).
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import psycopg
    import psycopg.rows
    from pgvector.psycopg import register_vector
except ImportError as e:
    raise ImportError(
        "psycopg and pgvector are required.\n"
        "Install: pip install 'psycopg[binary]' pgvector"
    ) from e

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_age.utils.cypher import validate_sql_identifier


class DistanceStrategy(str, Enum):
    """pgvector distance operator."""

    COSINE = "<=>"
    EUCLIDEAN = "<->"
    MAX_INNER_PRODUCT = "<#>"


class SearchType(str, Enum):
    """Search mode, mirrors Neo4jVector."""

    VECTOR = "vector"
    HYBRID = "hybrid"


_DEFAULT_COLLECTION = "langchain_age_vectors"
_DEFAULT_BATCH_SIZE = 1_000


class AGEVector(VectorStore):
    """``VectorStore`` backed by pgvector with optional AGE graph linkage.

    Mirrors ``Neo4jVector`` (langchain-neo4j) and ``PGVectorStore``
    (langchain-postgres) APIs.

    Args:
        connection_string: psycopg3 DSN / URI.
        embedding_function: LangChain ``Embeddings`` instance.
        collection_name: PostgreSQL table name for vector storage.
            Validated at construction: only ``[a-zA-Z_][a-zA-Z0-9_]*``.
        distance_strategy: Similarity metric (default: cosine).
        search_type: ``SearchType.VECTOR`` (default) or ``SearchType.HYBRID``
            (vector + PostgreSQL full-text via RRF).
        pre_delete_collection: Drop & recreate the table on init.
        relevance_score_fn: Custom ``distance → score`` mapper.
        age_graph_name: AGE graph name for graph-enhanced retrieval.
        retrieval_query: Custom Cypher snippet for graph context enrichment.
        embedding_dimension: Explicit vector dimension.  When ``None``, the
            dimension is auto-detected via a sample ``embed_query`` call.
        batch_size: Number of rows per INSERT transaction in ``add_texts``
            (default 1 000, matching Neo4j's batch pattern).

    Context-manager usage::

        with AGEVector(conn, embedding, collection_name="docs") as store:
            store.add_texts(["hello", "world"])
    """

    def __init__(
        self,
        connection_string: str,
        embedding_function: Embeddings,
        *,
        collection_name: str = _DEFAULT_COLLECTION,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        search_type: SearchType = SearchType.VECTOR,
        pre_delete_collection: bool = False,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        age_graph_name: Optional[str] = None,
        retrieval_query: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        # Validate collection_name early — mirrors langchain-postgres pattern.
        validate_sql_identifier(collection_name, context="collection_name")

        self._conn_string = connection_string
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.distance_strategy = distance_strategy
        self.search_type = search_type
        self._relevance_score_fn = relevance_score_fn
        self._embedding_dimension = embedding_dimension
        self.age_graph_name = age_graph_name
        self.retrieval_query = retrieval_query
        self.batch_size = batch_size

        self._conn = self._connect()
        if pre_delete_collection:
            self._drop_table()
        self._create_table_if_not_exists()

    # ------------------------------------------------------------------
    # Context-manager support (mirrors Neo4j driver pattern)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        try:
            if not self._conn.closed:
                self._conn.close()
        except Exception:
            logger.debug("Exception while closing AGEVector connection", exc_info=True)

    def __enter__(self) -> AGEVector:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed *texts* and store them.

        Uses ``executemany`` for batch INSERT (one round-trip per
        ``batch_size`` rows instead of one per document), mirroring the
        Neo4j "IN TRANSACTIONS OF 1000 ROWS" pattern.
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        embeddings = self.embedding_function.embed_documents(texts_list)
        ids = ids or [str(uuid.uuid4()) for _ in texts_list]
        metadatas = metadatas or [{} for _ in texts_list]

        # Build parameter rows — extract age_node_id from metadata in-place.
        param_rows = []
        for doc_id, text, emb, meta in zip(ids, texts_list, embeddings, metadatas):
            meta = dict(meta)  # copy — do not mutate caller's dict
            age_node_id = meta.pop("age_node_id", None)
            param_rows.append(
                (doc_id, text, self._to_vec(emb), psycopg.types.json.Jsonb(meta), age_node_id)
            )

        # Use double-quoted table name — langchain-postgres convention.
        sql = f"""
            INSERT INTO "{self.collection_name}"
                (id, content, embedding, metadata, age_node_id)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE
                SET content     = EXCLUDED.content,
                    embedding   = EXCLUDED.embedding,
                    metadata    = EXCLUDED.metadata,
                    age_node_id = EXCLUDED.age_node_id;
        """
        try:
            with self._conn.cursor() as cur:
                # Batch by batch_size to limit transaction size.
                for start in range(0, len(param_rows), self.batch_size):
                    batch = param_rows[start : start + self.batch_size]
                    cur.executemany(sql, batch)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        return ids

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed and store ``Document`` objects."""
        return self.add_texts(
            [d.page_content for d in documents],
            metadatas=[dict(d.metadata) for d in documents],
            ids=ids,
            **kwargs,
        )

    def delete(self, ids: List[str], **kwargs: Any) -> Optional[bool]:
        """Delete documents by ID."""
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f'DELETE FROM "{self.collection_name}" WHERE id = ANY(%s);',
                    (ids,),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return True

    def get_by_ids(self, ids: List[str], **kwargs: Any) -> List[Document]:
        """Fetch documents by their IDs."""
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f'SELECT id, content, metadata FROM "{self.collection_name}" WHERE id = ANY(%s);',
                    (ids,),
                )
                rows = cur.fetchall()
        except Exception:
            self._conn.rollback()
            raise
        return [Document(page_content=r[1], metadata=r[2] or {}) for r in rows]

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return *k* most similar documents."""
        return [
            doc
            for doc, _ in self.similarity_search_with_score(query, k=k, filter=filter)
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return *k* most similar documents with distance scores."""
        embedding = self.embedding_function.embed_query(query)
        if self.search_type == SearchType.HYBRID:
            return self._hybrid_search_with_score(query, embedding, k=k, filter=filter)
        return self.similarity_search_by_vector_with_score(embedding, k=k, filter=filter)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search by pre-computed embedding vector."""
        return [
            doc
            for doc, _ in self.similarity_search_by_vector_with_score(
                embedding, k=k, filter=filter
            )
        ]

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search by vector and return documents with raw distance scores."""
        where_clause, where_params = self._build_filter_clause(filter)
        sql = f"""
            SELECT id, content, metadata, age_node_id,
                   embedding {self.distance_strategy.value} %s AS distance
            FROM "{self.collection_name}"
            {where_clause}
            ORDER BY distance
            LIMIT %s;
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, [self._to_vec(embedding)] + where_params + [k])
                rows = cur.fetchall()
        except Exception:
            self._conn.rollback()
            raise
        return self._rows_to_docs(rows)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return diverse documents via Maximal Marginal Relevance.

        Reuses the stored embedding vectors from the DB (fetched alongside the
        documents) instead of re-calling the embedding API, following the
        efficient pattern used in ``langchain-postgres``.
        """
        from langchain_core.vectorstores.utils import maximal_marginal_relevance

        query_emb = self.embedding_function.embed_query(query)
        where_clause, where_params = self._build_filter_clause(filter)

        # Fetch candidates *with* their stored embedding vectors.
        sql = f"""
            SELECT id, content, metadata, age_node_id,
                   embedding {self.distance_strategy.value} %s AS distance,
                   embedding
            FROM "{self.collection_name}"
            {where_clause}
            ORDER BY distance
            LIMIT %s;
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, [self._to_vec(query_emb)] + where_params + [fetch_k])
                rows = cur.fetchall()
        except Exception:
            self._conn.rollback()
            raise

        if not rows:
            return []

        docs_and_scores = self._rows_to_docs(
            [(r[0], r[1], r[2], r[3], r[4]) for r in rows]
        )
        docs = [doc for doc, _ in docs_and_scores]

        # Reuse stored embedding vectors (column index 5) — no extra API call.
        candidate_embeddings = [
            list(r[5]) if r[5] is not None else [0.0] * len(query_emb)
            for r in rows
        ]

        selected = maximal_marginal_relevance(
            np.array(query_emb),
            candidate_embeddings,
            lambda_mult=lambda_mult,
            k=k,
        )
        return [docs[i] for i in selected]

    # ------------------------------------------------------------------
    # Hybrid search (vector + PostgreSQL full-text via RRF)
    # ------------------------------------------------------------------

    def _hybrid_search_with_score(
        self,
        query: str,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Combine vector similarity and PostgreSQL full-text via RRF (k=60)."""
        where_clause, where_params = self._build_filter_clause(filter)
        fetch_k = k * 4
        tbl = f'"{self.collection_name}"'

        vector_sql = f"""
            SELECT id, content, metadata, age_node_id,
                   row_number() OVER (
                       ORDER BY embedding {self.distance_strategy.value} %s
                   ) AS rn_vec
            FROM {tbl}
            {where_clause}
            LIMIT %s
        """
        fts_sql = f"""
            SELECT id,
                   row_number() OVER (
                       ORDER BY ts_rank(fts, plainto_tsquery('english', %s)) DESC
                   ) AS rn_fts
            FROM {tbl}
            WHERE fts @@ plainto_tsquery('english', %s)
            {('AND ' + where_clause.lstrip('WHERE ')) if where_clause else ''}
            LIMIT %s
        """
        rrf_sql = f"""
            WITH vec AS ({vector_sql}),
                 fts AS ({fts_sql}),
                 fused AS (
                     SELECT COALESCE(vec.id, fts.id) AS id,
                            (  COALESCE(1.0 / (60 + vec.rn_vec), 0)
                             + COALESCE(1.0 / (60 + fts.rn_fts), 0)
                            ) AS score,
                            vec.content, vec.metadata, vec.age_node_id
                     FROM vec FULL OUTER JOIN fts ON vec.id = fts.id
                 )
            SELECT id, content, metadata, age_node_id, (1 - score) AS distance
            FROM fused
            WHERE content IS NOT NULL
            ORDER BY score DESC
            LIMIT %s;
        """
        params = (
            [self._to_vec(embedding)] + where_params + [fetch_k]
            + [query, query] + where_params + [fetch_k]
            + [k]
        )
        try:
            with self._conn.cursor() as cur:
                cur.execute(rrf_sql, params)
                rows = cur.fetchall()
        except Exception:
            self._conn.rollback()
            raise
        return self._rows_to_docs(rows)

    # ------------------------------------------------------------------
    # Class-method constructors (mirrors Neo4jVector / PGVectorStore)
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls: Type[AGEVector],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AGEVector:
        """Create a store and populate it with *texts*."""
        store = cls(embedding_function=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas)
        return store

    @classmethod
    def from_documents(
        cls: Type[AGEVector],
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> AGEVector:
        """Create a store and populate it with *documents*."""
        store = cls(embedding_function=embedding, **kwargs)
        store.add_documents(documents)
        return store

    @classmethod
    def from_existing_index(
        cls: Type[AGEVector],
        embedding: Embeddings,
        connection_string: str,
        collection_name: str = _DEFAULT_COLLECTION,
        **kwargs: Any,
    ) -> AGEVector:
        """Connect to an existing pgvector table (no re-insertion)."""
        return cls(
            connection_string=connection_string,
            embedding_function=embedding,
            collection_name=collection_name,
            **kwargs,
        )

    @classmethod
    def from_existing_graph(
        cls: Type[AGEVector],
        embedding: Embeddings,
        connection_string: str,
        graph_name: str,
        node_label: str,
        text_node_properties: List[str],
        embedding_node_property: str = "embedding",
        collection_name: str = _DEFAULT_COLLECTION,
        **kwargs: Any,
    ) -> AGEVector:
        """Build a vector store from existing AGE graph nodes.

        Fetches nodes of *node_label*, concatenates *text_node_properties*
        into a text string, embeds them, and stores vectors in the collection.
        Mirrors ``Neo4jVector.from_existing_graph``.

        All property names are backtick-quoted in the generated Cypher so
        that AGE Cypher reserved words (``desc``, ``asc``, ``order``, ``where``,
        ``match``, ``limit``, ``skip``, ``set``, …) are handled transparently.

        Args:
            embedding: Embeddings instance.
            connection_string: psycopg3 DSN.
            graph_name: AGE graph name.
            node_label: Node label to vectorise.
            text_node_properties: Node properties to concatenate as document text.
            embedding_node_property: Informational — not written back to AGE.
            collection_name: pgvector table name.
        """
        from langchain_age.graphs.age_graph import AGEGraph
        from langchain_age.utils.cypher import escape_cypher_identifier

        graph = AGEGraph(connection_string, graph_name, refresh_schema=False)

        # Backtick-quote every property name — handles Cypher reserved words.
        # "prop_" prefix on aliases prevents SQL reserved-word collisions in
        # the AGE column-definition list (e.g. "desc agtype" → SQL error).
        prop_returns = ", ".join(
            f"n.{escape_cypher_identifier(p)} AS prop_{p}"
            for p in text_node_properties
        )
        rows = graph.query(
            f"MATCH (n:{escape_cypher_identifier(node_label)}) "
            f"RETURN n AS node_obj, {prop_returns}"
        )

        docs: List[Document] = []
        for row in rows:
            parts = [
                str(row[f"prop_{p}"])
                for p in text_node_properties
                if row.get(f"prop_{p}") is not None
            ]
            text = " ".join(parts).strip()
            if not text:
                continue
            node_obj = row.get("node_obj") or {}
            node_id = (
                str(node_obj.get("id", "")) if isinstance(node_obj, dict) else ""
            )
            meta = {"age_node_id": node_id, "node_label": node_label}
            docs.append(Document(page_content=text, metadata=meta))

        store = cls(
            connection_string=connection_string,
            embedding_function=embedding,
            collection_name=collection_name,
            age_graph_name=graph_name,
            **kwargs,
        )
        if docs:
            store.add_documents(docs)
        return store

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Async version of ``add_texts``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.add_texts(list(texts), metadatas=metadatas, ids=ids)
        )

    async def aadd_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Async version of ``add_documents``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.add_documents(documents, ids=ids)
        )

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Async similarity search."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.similarity_search(query, k=k, filter=filter)
        )

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Async vector similarity search."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.similarity_search_by_vector(embedding, k=k, filter=filter),
        )

    async def adelete(self, ids: List[str], **kwargs: Any) -> Optional[bool]:
        """Async delete."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.delete(ids))

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def create_hnsw_index(self, m: int = 16, ef_construction: int = 64) -> None:
        """Create an HNSW index for approximate nearest-neighbour search."""
        op = self._op_class()
        idx = f'"{self.collection_name}_hnsw_idx"'
        tbl = f'"{self.collection_name}"'
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx} ON {tbl} "
                    f"USING hnsw (embedding {op}) "
                    f"WITH (m = {m}, ef_construction = {ef_construction});"
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def create_ivfflat_index(self, n_lists: int = 100) -> None:
        """Create an IVFFlat index for approximate nearest-neighbour search."""
        op = self._op_class()
        idx = f'"{self.collection_name}_ivfflat_idx"'
        tbl = f'"{self.collection_name}"'
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx} ON {tbl} "
                    f"USING ivfflat (embedding {op}) "
                    f"WITH (lists = {n_lists});"
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def drop_index(self) -> None:
        """Drop all vector indexes on this collection."""
        for suffix in ("_hnsw_idx", "_ivfflat_idx"):
            idx = f'"{self.collection_name}{suffix}"'
            try:
                with self._conn.cursor() as cur:
                    cur.execute(f"DROP INDEX IF EXISTS {idx};")
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_vec(embedding: List[float]) -> "np.ndarray":
        """Convert a Python list to numpy float32 array (required by pgvector)."""
        return np.array(embedding, dtype=np.float32)

    def _detect_dimension(self) -> Optional[int]:
        """Auto-detect embedding dimension via a sample embed_query call."""
        if self._embedding_dimension:
            return self._embedding_dimension
        try:
            sample = self.embedding_function.embed_query("test")
            return len(sample)
        except Exception as exc:
            logger.warning("Could not auto-detect embedding dimension: %s", exc)
            return None

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._conn_string)
        conn.autocommit = False
        register_vector(conn)
        return conn

    def _create_table_if_not_exists(self) -> None:
        dim = self._detect_dimension()
        vec_type = f"vector({dim})" if dim else "vector"
        tbl = f'"{self.collection_name}"'
        fts_idx = f'"{self.collection_name}_fts_idx"'
        try:
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {tbl} (
                        id          TEXT PRIMARY KEY,
                        content     TEXT NOT NULL,
                        embedding   {vec_type},
                        metadata    JSONB DEFAULT '{{}}',
                        age_node_id TEXT,
                        fts         TSVECTOR
                                    GENERATED ALWAYS AS
                                    (to_tsvector('english', content)) STORED
                    );
                """)
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {fts_idx} "
                    f"ON {tbl} USING gin(fts);"
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _drop_table(self) -> None:
        self._conn.rollback()  # clear any aborted transaction first
        tbl = f'"{self.collection_name}"'
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {tbl};")
        self._conn.commit()

    def _op_class(self) -> str:
        return {
            DistanceStrategy.COSINE: "vector_cosine_ops",
            DistanceStrategy.EUCLIDEAN: "vector_l2_ops",
            DistanceStrategy.MAX_INNER_PRODUCT: "vector_ip_ops",
        }[self.distance_strategy]

    def _rows_to_docs(self, rows: list) -> List[Tuple[Document, float]]:
        results = []
        for row_id, content, meta, age_node_id, distance in rows:
            meta = dict(meta or {})
            if age_node_id:
                meta["age_node_id"] = age_node_id
            score = (
                self._relevance_score_fn(distance)
                if self._relevance_score_fn
                else distance
            )
            results.append((Document(page_content=content, metadata=meta), score))
        return results

    @staticmethod
    def _build_filter_clause(
        filter: Optional[Dict[str, Any]],
        _depth: int = 0,
    ) -> Tuple[str, List[Any]]:
        """Build a ``WHERE`` clause supporting MongoDB-style filter operators.

        Supported operators:
        ``$eq``, ``$ne``, ``$lt``, ``$lte``, ``$gt``, ``$gte``,
        ``$in``, ``$nin``, ``$between``, ``$like``, ``$ilike``,
        ``$exists``, ``$and``, ``$or``.

        Recursion depth is capped at 10 to prevent stack overflow from
        deeply nested ``$and``/``$or`` structures.
        """
        if not filter:
            return "", []

        MAX_DEPTH = 10

        def _parse(f: Dict[str, Any], depth: int = 0) -> Tuple[str, List[Any]]:
            if depth > MAX_DEPTH:
                raise ValueError(
                    f"Filter nesting exceeds maximum depth of {MAX_DEPTH}. "
                    "Simplify the filter expression."
                )

            parts: List[str] = []
            params: List[Any] = []

            if "$and" in f:
                sub_parts, sub_params = [], []
                for sub in f["$and"]:
                    s, p = _parse(sub, depth + 1)
                    sub_parts.append(s)
                    sub_params.extend(p)
                parts.append("(" + " AND ".join(sub_parts) + ")")
                params.extend(sub_params)
                return " AND ".join(parts), params

            if "$or" in f:
                sub_parts, sub_params = [], []
                for sub in f["$or"]:
                    s, p = _parse(sub, depth + 1)
                    sub_parts.append(s)
                    sub_params.extend(p)
                parts.append("(" + " OR ".join(sub_parts) + ")")
                params.extend(sub_params)
                return " AND ".join(parts), params

            _SCALAR_OPS: Dict[str, str] = {
                "$eq":    "metadata->>%s = %s",
                "$ne":    "metadata->>%s != %s",
                "$lt":    "(metadata->>%s)::numeric < %s",
                "$lte":   "(metadata->>%s)::numeric <= %s",
                "$gt":    "(metadata->>%s)::numeric > %s",
                "$gte":   "(metadata->>%s)::numeric >= %s",
                "$like":  "metadata->>%s LIKE %s",
                "$ilike": "metadata->>%s ILIKE %s",
            }

            for key, expr in f.items():
                if isinstance(expr, dict):
                    for op, val in expr.items():
                        if op == "$in":
                            parts.append("metadata->>%s = ANY(%s)")
                            params.extend([key, list(map(str, val))])
                        elif op == "$nin":
                            parts.append("NOT (metadata->>%s = ANY(%s))")
                            params.extend([key, list(map(str, val))])
                        elif op == "$between":
                            lo, hi = val
                            parts.append(
                                "(metadata->>%s)::numeric BETWEEN %s AND %s"
                            )
                            params.extend([key, lo, hi])
                        elif op == "$exists":
                            parts.append(
                                "metadata ? %s" if val else "NOT (metadata ? %s)"
                            )
                            params.append(key)
                        elif op in _SCALAR_OPS:
                            parts.append(_SCALAR_OPS[op])
                            params.extend([key, str(val)])
                        else:
                            raise ValueError(f"Unsupported filter operator: {op!r}")
                else:
                    # Bare equality shorthand: {"key": "value"}
                    parts.append("metadata->>%s = %s")
                    params.extend([key, str(expr)])

            return " AND ".join(parts), params

        clause, params = _parse(filter)
        return f"WHERE {clause}", params

    def __repr__(self) -> str:
        return (
            f"AGEVector(collection='{self.collection_name}', "
            f"distance={self.distance_strategy.name}, "
            f"search={self.search_type.name})"
        )
