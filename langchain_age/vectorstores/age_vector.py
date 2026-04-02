"""pgvector-backed VectorStore with Apache AGE graph node linkage.

Mirrors ``Neo4jVector`` from *langchain-neo4j* and ``PGVectorStore`` from
*langchain-postgres*, combining:
- **pgvector** for fast vector similarity search
- **Apache AGE** for optional graph-context retrieval
- **PostgreSQL full-text search** for hybrid (vector + keyword) search
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

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


class AGEVector(VectorStore):
    """``VectorStore`` backed by pgvector with optional AGE graph linkage.

    Mirrors ``Neo4jVector`` (langchain-neo4j) and ``PGVectorStore``
    (langchain-postgres) APIs.

    Args:
        connection_string: psycopg3 DSN / URI.
        embedding_function: LangChain ``Embeddings`` instance.
        collection_name: PostgreSQL table name for vector storage.
        distance_strategy: Similarity metric.
        search_type: ``"vector"`` (default) or ``"hybrid"`` (vector + fulltext).
        pre_delete_collection: Drop & recreate table on init.
        relevance_score_fn: Custom distance → score mapper.
        age_graph_name: AGE graph name for graph-enhanced retrieval.
        retrieval_query: Custom Cypher snippet appended after the vector
            search to enrich results with graph context.  Must return columns
            ``text``, ``score``, and ``metadata``.
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
    ) -> None:
        self._conn_string = connection_string
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.distance_strategy = distance_strategy
        self.search_type = search_type
        self._relevance_score_fn = relevance_score_fn
        self._embedding_dimension = embedding_dimension
        self.age_graph_name = age_graph_name
        self.retrieval_query = retrieval_query

        self._conn = self._connect()
        if pre_delete_collection:
            self._drop_table()
        self._create_table_if_not_exists()

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
        """Embed *texts* and store them."""
        texts_list = list(texts)
        embeddings = self.embedding_function.embed_documents(texts_list)
        ids = ids or [str(uuid.uuid4()) for _ in texts_list]
        metadatas = metadatas or [{} for _ in texts_list]

        with self._conn.cursor() as cur:
            for doc_id, text, emb, meta in zip(ids, texts_list, embeddings, metadatas):
                age_node_id = meta.pop("age_node_id", None)
                try:
                    # fts is a GENERATED ALWAYS column — omit from INSERT list
                    cur.execute(
                        f"""
                        INSERT INTO {self.collection_name}
                            (id, content, embedding, metadata, age_node_id)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE
                            SET content     = EXCLUDED.content,
                                embedding   = EXCLUDED.embedding,
                                metadata    = EXCLUDED.metadata,
                                age_node_id = EXCLUDED.age_node_id;
                        """,
                        (doc_id, text, self._to_vec(emb), psycopg.types.json.Jsonb(meta), age_node_id),
                    )
                except Exception:
                    self._conn.rollback()
                    raise
        self._conn.commit()
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
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.collection_name} WHERE id = ANY(%s);", (ids,))
        self._conn.commit()
        return True

    def get_by_ids(self, ids: List[str], **kwargs: Any) -> List[Document]:
        """Fetch documents by their IDs."""
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, content, metadata FROM {self.collection_name} WHERE id = ANY(%s);",
                (ids,),
            )
            rows = cur.fetchall()
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
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, filter=filter)]

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
        """Search by pre-computed embedding."""
        return [doc for doc, _ in self.similarity_search_by_vector_with_score(embedding, k=k, filter=filter)]

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
            FROM {self.collection_name}
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
        """Return diverse documents via Maximal Marginal Relevance."""
        from langchain_core.vectorstores.utils import maximal_marginal_relevance
        import numpy as np

        query_emb = self.embedding_function.embed_query(query)
        candidates_and_scores = self.similarity_search_by_vector_with_score(
            query_emb, k=fetch_k, filter=filter
        )
        if not candidates_and_scores:
            return []

        docs = [doc for doc, _ in candidates_and_scores]
        candidate_embeddings = self.embedding_function.embed_documents(
            [d.page_content for d in docs]
        )
        selected = maximal_marginal_relevance(
            np.array(query_emb), candidate_embeddings, lambda_mult=lambda_mult, k=k
        )
        return [docs[i] for i in selected]

    # ------------------------------------------------------------------
    # Hybrid search (vector + PostgreSQL full-text)
    # ------------------------------------------------------------------

    def _hybrid_search_with_score(
        self,
        query: str,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Combine vector similarity and PostgreSQL full-text via RRF."""
        where_clause, where_params = self._build_filter_clause(filter)
        fetch_k = k * 4

        # Vector ranking
        vector_sql = f"""
            SELECT id, content, metadata, age_node_id,
                   row_number() OVER (ORDER BY embedding {self.distance_strategy.value} %s) AS rn_vec
            FROM {self.collection_name}
            {where_clause}
            LIMIT %s
        """
        # Full-text ranking
        fts_sql = f"""
            SELECT id,
                   row_number() OVER (ORDER BY ts_rank(fts, plainto_tsquery('english', %s)) DESC) AS rn_fts
            FROM {self.collection_name}
            WHERE fts @@ plainto_tsquery('english', %s)
            {('AND ' + where_clause.lstrip('WHERE ')) if where_clause else ''}
            LIMIT %s
        """
        # RRF fusion  (k=60 is standard RRF constant)
        rrf_sql = f"""
            WITH vec AS ({vector_sql}),
                 fts AS ({fts_sql}),
                 fused AS (
                     SELECT COALESCE(vec.id, fts.id) AS id,
                            (COALESCE(1.0/(60+vec.rn_vec), 0) + COALESCE(1.0/(60+fts.rn_fts), 0)) AS score,
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
            [self._to_vec(embedding)] + where_params + [fetch_k]   # vec subquery
            + [query, query] + where_params + [fetch_k]  # fts subquery
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
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls: Type[AGEVector],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AGEVector:
        """Create store and populate with *texts*."""
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
        """Create store and populate with *documents*."""
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

        Args:
            embedding: Embeddings instance.
            connection_string: psycopg3 DSN.
            graph_name: AGE graph name.
            node_label: Node label to vectorise.
            text_node_properties: Node properties to concatenate as text.
            embedding_node_property: Property name to store the embedding
                (stored as AGE property — informational only, actual vectors
                live in the pgvector table).
            collection_name: pgvector table name.
        """
        from langchain_age.graphs.age_graph import AGEGraph

        graph = AGEGraph(connection_string, graph_name, refresh_schema=False)

        # Backtick-quote every property name so AGE Cypher reserved words (desc, asc, order,
        # where, match, limit, skip, set, …) don't break the parser.
        # "prop_" prefix on aliases avoids SQL reserved-word collisions in the column-def list.
        prop_returns = ", ".join(
            f"n.`{p}` AS prop_{p}" for p in text_node_properties
        )
        # Return the node itself to extract internal id from properties dict
        rows = graph.query(
            f"MATCH (n:{node_label}) RETURN n AS node_obj, {prop_returns}"
        )

        docs: List[Document] = []
        for row in rows:
            parts = [str(row[f"prop_{p}"]) for p in text_node_properties if row.get(f"prop_{p}")]
            text = " ".join(parts).strip()
            if not text:
                continue
            # Extract the AGE internal node id from the Vertex object
            node_obj = row.get("node_obj") or {}
            node_id = str(node_obj.get("id", "")) if isinstance(node_obj, dict) else ""
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
    # Async interface (mirrors PGVectorStore)
    # ------------------------------------------------------------------

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Async version of ``add_texts``."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.add_texts(list(texts), metadatas=metadatas, ids=ids)
        )

    async def aadd_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Async version of ``add_documents``."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
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
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
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
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.similarity_search_by_vector(embedding, k=k, filter=filter)
        )

    async def adelete(self, ids: List[str], **kwargs: Any) -> Optional[bool]:
        """Async delete."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.delete(ids)
        )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def create_hnsw_index(self, m: int = 16, ef_construction: int = 64) -> None:
        """Create HNSW index for approximate nearest-neighbour search."""
        op = self._op_class()
        with self._conn.cursor() as cur:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.collection_name}_hnsw_idx "
                f"ON {self.collection_name} USING hnsw (embedding {op}) "
                f"WITH (m = {m}, ef_construction = {ef_construction});"
            )
        self._conn.commit()

    def create_ivfflat_index(self, n_lists: int = 100) -> None:
        """Create IVFFlat index for approximate nearest-neighbour search."""
        op = self._op_class()
        with self._conn.cursor() as cur:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.collection_name}_ivfflat_idx "
                f"ON {self.collection_name} USING ivfflat (embedding {op}) "
                f"WITH (lists = {n_lists});"
            )
        self._conn.commit()

    def drop_index(self) -> None:
        """Drop all vector indexes on this collection."""
        for suffix in ("_hnsw_idx", "_ivfflat_idx"):
            with self._conn.cursor() as cur:
                cur.execute(f"DROP INDEX IF EXISTS {self.collection_name}{suffix};")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_vec(embedding: List[float]) -> "np.ndarray":
        """Convert a Python list to numpy float32 array (required by pgvector adapter)."""
        return np.array(embedding, dtype=np.float32)

    def _detect_dimension(self) -> Optional[int]:
        """Auto-detect embedding dimension by generating a test embedding."""
        if self._embedding_dimension:
            return self._embedding_dimension
        try:
            sample = self.embedding_function.embed_query("test")
            return len(sample)
        except Exception:
            return None

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._conn_string)
        conn.autocommit = False
        register_vector(conn)
        return conn

    def _create_table_if_not_exists(self) -> None:
        dim = self._detect_dimension()
        # vector(N) enables HNSW/IVFFlat indexes; plain vector() works but prevents indexing
        vec_type = f"vector({dim})" if dim else "vector"
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.collection_name} (
                    id          TEXT PRIMARY KEY,
                    content     TEXT NOT NULL,
                    embedding   {vec_type},
                    metadata    JSONB DEFAULT '{{}}',
                    age_node_id TEXT,
                    fts         TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
                );
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.collection_name}_fts_idx
                ON {self.collection_name} USING gin(fts);
            """)
        self._conn.commit()

    def _drop_table(self) -> None:
        self._conn.rollback()  # clear any aborted transaction first
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.collection_name};")
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
            if age_node_id:
                (meta or {})["age_node_id"] = age_node_id
            score = self._relevance_score_fn(distance) if self._relevance_score_fn else distance
            results.append((Document(page_content=content, metadata=meta or {}), score))
        return results

    @staticmethod
    def _build_filter_clause(
        filter: Optional[Dict[str, Any]],
    ) -> Tuple[str, List[Any]]:
        """Build a WHERE clause supporting MongoDB-style filter operators.

        Supports: ``$eq``, ``$ne``, ``$lt``, ``$lte``, ``$gt``, ``$gte``,
        ``$in``, ``$nin``, ``$between``, ``$like``, ``$ilike``,
        ``$exists``, ``$and``, ``$or``.
        """
        if not filter:
            return "", []

        def _parse(f: Dict[str, Any]) -> Tuple[str, List[Any]]:
            parts, params = [], []

            if "$and" in f:
                sub_parts, sub_params = [], []
                for sub in f["$and"]:
                    s, p = _parse(sub)
                    sub_parts.append(s)
                    sub_params.extend(p)
                parts.append("(" + " AND ".join(sub_parts) + ")")
                params.extend(sub_params)
                return " AND ".join(parts), params

            if "$or" in f:
                sub_parts, sub_params = [], []
                for sub in f["$or"]:
                    s, p = _parse(sub)
                    sub_parts.append(s)
                    sub_params.extend(p)
                parts.append("(" + " OR ".join(sub_parts) + ")")
                params.extend(sub_params)
                return " AND ".join(parts), params

            _OPS = {
                "$eq":      ("metadata->>%s = %s",           1),
                "$ne":      ("metadata->>%s != %s",          1),
                "$lt":      ("(metadata->>%s)::numeric < %s",  1),
                "$lte":     ("(metadata->>%s)::numeric <= %s", 1),
                "$gt":      ("(metadata->>%s)::numeric > %s",  1),
                "$gte":     ("(metadata->>%s)::numeric >= %s", 1),
                "$like":    ("metadata->>%s LIKE %s",         1),
                "$ilike":   ("metadata->>%s ILIKE %s",        1),
            }

            for key, expr in f.items():
                if isinstance(expr, dict):
                    for op, val in expr.items():
                        if op == "$in":
                            parts.append(f"metadata->>%s = ANY(%s)")
                            params.extend([key, list(map(str, val))])
                        elif op == "$nin":
                            parts.append(f"NOT (metadata->>%s = ANY(%s))")
                            params.extend([key, list(map(str, val))])
                        elif op == "$between":
                            lo, hi = val
                            parts.append("(metadata->>%s)::numeric BETWEEN %s AND %s")
                            params.extend([key, lo, hi])
                        elif op == "$exists":
                            if val:
                                parts.append("metadata ? %s")
                            else:
                                parts.append("NOT (metadata ? %s)")
                            params.append(key)
                        elif op in _OPS:
                            template, _ = _OPS[op]
                            parts.append(template)
                            params.extend([key, str(val)])
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
