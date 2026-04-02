"""PGVector-backed vector store that keeps embeddings linked to AGE graph nodes."""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

try:
    import psycopg2
    import psycopg2.extras
except ImportError as e:
    raise ImportError(
        "psycopg2 is required for AGEVector. "
        "Install it with: pip install psycopg2-binary"
    ) from e

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class DistanceStrategy(str, Enum):
    """pgvector distance operator mapping."""

    COSINE = "<=>"         # cosine distance  (1 - cosine_similarity)
    EUCLIDEAN = "<->"      # L2 / Euclidean distance
    MAX_INNER_PRODUCT = "<#>"  # negative inner product (use for dot-product similarity)


_DEFAULT_COLLECTION = "langchain_age_vectors"


class AGEVector(VectorStore):
    """``VectorStore`` backed by pgvector, with optional linkage to AGE graph nodes.

    Embeddings are stored in a regular PostgreSQL table using the ``vector``
    type from the **pgvector** extension.  Optionally, each vector row can
    reference a node in an Apache AGE graph so you can combine similarity
    search with graph traversal.

    Args:
        connection_string: psycopg2 DSN or URI.
        embedding_function: LangChain ``Embeddings`` instance.
        collection_name: Name of the PostgreSQL table used for vector storage.
        distance_strategy: Distance metric for similarity search.
        pre_delete_collection: Drop and recreate the table on initialisation.
        relevance_score_fn: Custom function to map raw distance to a 0-1 score.
        age_graph_name: Optional AGE graph name.  When set, vectors are linked
            to graph nodes via an ``age_node_id`` column.
    """

    def __init__(
        self,
        connection_string: str,
        embedding_function: Embeddings,
        *,
        collection_name: str = _DEFAULT_COLLECTION,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        pre_delete_collection: bool = False,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        age_graph_name: Optional[str] = None,
    ) -> None:
        self._conn_string = connection_string
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.distance_strategy = distance_strategy
        self._relevance_score_fn = relevance_score_fn
        self.age_graph_name = age_graph_name

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
        """Embed *texts* and add them to the vector store.

        Args:
            texts: Iterable of text strings to embed.
            metadatas: Optional list of metadata dicts, one per text.
            ids: Optional list of UUIDs.  Generated if not provided.

        Returns:
            List of inserted IDs.
        """
        texts_list = list(texts)
        embeddings = self.embedding_function.embed_documents(texts_list)
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts_list]
        if not metadatas:
            metadatas = [{} for _ in texts_list]

        with self._conn.cursor() as cur:
            for doc_id, text, embedding, meta in zip(ids, texts_list, embeddings, metadatas):
                age_node_id = meta.pop("age_node_id", None)
                cur.execute(
                    f"""
                    INSERT INTO {self.collection_name}
                        (id, content, embedding, metadata, age_node_id)
                    VALUES (%s, %s, %s::vector, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                        SET content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            age_node_id = EXCLUDED.age_node_id;
                    """,
                    (doc_id, text, self._embedding_to_str(embedding), psycopg2.extras.Json(meta), age_node_id),
                )
        self._conn.commit()
        return ids

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed and store ``Document`` objects."""
        texts = [doc.page_content for doc in documents]
        metadatas = [dict(doc.metadata) for doc in documents]
        return self.add_texts(texts, metadatas=metadatas, ids=ids, **kwargs)

    def delete(self, ids: List[str], **kwargs: Any) -> Optional[bool]:
        """Delete documents by their IDs."""
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.collection_name} WHERE id = ANY(%s);",
                (ids,),
            )
        self._conn.commit()
        return True

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return the *k* most similar documents."""
        docs_and_scores = self.similarity_search_with_score(query, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return the *k* most similar documents with their distance scores."""
        embedding = self.embedding_function.embed_query(query)
        return self.similarity_search_by_vector_with_score(embedding, k=k, filter=filter, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search using a pre-computed embedding vector."""
        docs_and_scores = self.similarity_search_by_vector_with_score(embedding, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in docs_and_scores]

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
                   embedding {self.distance_strategy.value} %s::vector AS distance
            FROM {self.collection_name}
            {where_clause}
            ORDER BY distance
            LIMIT %s;
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, [self._embedding_to_str(embedding)] + where_params + [k])
            rows = cur.fetchall()

        results: List[Tuple[Document, float]] = []
        for row_id, content, meta, age_node_id, distance in rows:
            if isinstance(meta, str):
                import json
                meta = json.loads(meta)
            if age_node_id:
                meta["age_node_id"] = age_node_id
            doc = Document(page_content=content, metadata=meta or {})
            score = self._relevance_score_fn(distance) if self._relevance_score_fn else distance
            results.append((doc, score))
        return results

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return diverse documents via Maximal Marginal Relevance (MMR)."""
        from langchain_core.vectorstores.utils import maximal_marginal_relevance
        import numpy as np

        query_embedding = self.embedding_function.embed_query(query)
        candidates = self.similarity_search_by_vector_with_score(
            query_embedding, k=fetch_k, filter=filter
        )
        if not candidates:
            return []

        docs = [doc for doc, _ in candidates]
        embeddings = self.embedding_function.embed_documents([d.page_content for d in docs])
        selected_indices = maximal_marginal_relevance(
            np.array(query_embedding), embeddings, lambda_mult=lambda_mult, k=k
        )
        return [docs[i] for i in selected_indices]

    def get_by_ids(self, ids: List[str], **kwargs: Any) -> List[Document]:
        """Fetch documents by their IDs."""
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, content, metadata FROM {self.collection_name} WHERE id = ANY(%s);",
                (ids,),
            )
            rows = cur.fetchall()

        docs = []
        for _, content, meta in rows:
            if isinstance(meta, str):
                import json
                meta = json.loads(meta)
            docs.append(Document(page_content=content, metadata=meta or {}))
        return docs

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
        """Create an ``AGEVector`` store and populate it with *texts*."""
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
        """Create an ``AGEVector`` store and populate it with *documents*."""
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
        """Connect to an existing pgvector table without re-inserting data."""
        return cls(
            connection_string=connection_string,
            embedding_function=embedding,
            collection_name=collection_name,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def create_hnsw_index(
        self,
        m: int = 16,
        ef_construction: int = 64,
    ) -> None:
        """Create an HNSW index on the embedding column for fast ANN search.

        Args:
            m: Max number of connections per layer (higher = more accurate, slower build).
            ef_construction: Size of the dynamic candidate list during index build.
        """
        op_class = self._pgvector_op_class()
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.collection_name}_hnsw_idx
                ON {self.collection_name}
                USING hnsw (embedding {op_class})
                WITH (m = {m}, ef_construction = {ef_construction});
                """
            )
        self._conn.commit()

    def create_ivfflat_index(self, n_lists: int = 100) -> None:
        """Create an IVFFlat index on the embedding column.

        Args:
            n_lists: Number of clusters (higher = faster search, lower recall).
        """
        op_class = self._pgvector_op_class()
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.collection_name}_ivfflat_idx
                ON {self.collection_name}
                USING ivfflat (embedding {op_class})
                WITH (lists = {n_lists});
                """
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> psycopg2.extensions.connection:
        conn = psycopg2.connect(self._conn_string)
        conn.autocommit = False
        return conn

    def _create_table_if_not_exists(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.collection_name} (
                    id          TEXT PRIMARY KEY,
                    content     TEXT NOT NULL,
                    embedding   vector,
                    metadata    JSONB DEFAULT '{{}}',
                    age_node_id TEXT          -- optional AGE graph node reference
                );
                """
            )
        self._conn.commit()

    def _drop_table(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.collection_name};")
        self._conn.commit()

    def _pgvector_op_class(self) -> str:
        return {
            DistanceStrategy.COSINE: "vector_cosine_ops",
            DistanceStrategy.EUCLIDEAN: "vector_l2_ops",
            DistanceStrategy.MAX_INNER_PRODUCT: "vector_ip_ops",
        }[self.distance_strategy]

    @staticmethod
    def _embedding_to_str(embedding: List[float]) -> str:
        return "[" + ",".join(map(str, embedding)) + "]"

    @staticmethod
    def _build_filter_clause(filter: Optional[Dict[str, Any]]) -> Tuple[str, List[Any]]:
        """Build a WHERE clause from a simple equality filter dict."""
        if not filter:
            return "", []
        conditions = []
        params: List[Any] = []
        for key, value in filter.items():
            conditions.append(f"metadata->>%s = %s")
            params.extend([key, str(value)])
        return "WHERE " + " AND ".join(conditions), params

    def __repr__(self) -> str:
        return (
            f"AGEVector(collection='{self.collection_name}', "
            f"distance={self.distance_strategy.name})"
        )
