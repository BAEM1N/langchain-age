"""
langchain-age LangChain 호환성 통합 테스트

Neo4j와 동일한 방식으로 사용 가능한지 검증:
  from langchain_age import AGEGraph
  from langchain_age import AGEVector
  from langchain_age import AGEGraphCypherQAChain, DistanceStrategy, SearchType

DB: docker/docker-compose.yml 컨테이너 필요
DSN: LANGCHAIN_AGE_TEST_DSN 환경변수 또는 아래 기본값
"""
import os
import pytest

DSN = os.getenv(
    "LANGCHAIN_AGE_TEST_DSN",
    "host=localhost port=5433 dbname=langchain_age user=langchain password=langchain",
)
GRAPH = "test_compat_graph"

# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def graph():
    from langchain_age import AGEGraph
    g = AGEGraph(DSN, GRAPH, refresh_schema=False)
    yield g
    try:
        g.drop_graph()
    except Exception:
        pass


@pytest.fixture(scope="module")
def vector_store():
    from langchain_age import AGEVector, DistanceStrategy
    store = AGEVector(
        connection_string=DSN,
        embedding_function=FakeEmbeddings(),
        collection_name="test_compat_vectors",
        distance_strategy=DistanceStrategy.COSINE,
        pre_delete_collection=True,
    )
    yield store
    store._drop_table()


class FakeEmbeddings:
    """결정적 임베딩 (dim=4) – 외부 API 불필요"""
    def embed_documents(self, texts):
        return [[float(i % 4) / 4.0 for i in range(len(t), len(t) + 4)] for t in texts]

    def embed_query(self, text):
        return [0.1, 0.5, 0.8, 0.2]


# ─────────────────────────────────────────────────────────────────
# 1. AGEGraph — Neo4jGraph 동일 인터페이스
# ─────────────────────────────────────────────────────────────────

class TestAGEGraph:
    """Neo4jGraph와 동일한 인터페이스 검증"""

    def test_import_style(self):
        """from langchain_age import AGEGraph  (Neo4j 스타일)"""
        from langchain_age import AGEGraph
        assert AGEGraph is not None

    def test_query_create_and_match(self, graph):
        """graph.query() — CREATE + MATCH"""
        graph.query("MERGE (:Person {name: 'Alice', age: 30})")
        results = graph.query("MATCH (n:Person {name: 'Alice'}) RETURN n.name AS name, n.age AS age")
        assert len(results) >= 1
        assert results[0]["name"] == "Alice"
        assert results[0]["age"] == 30

    def test_query_relationship(self, graph):
        """관계 생성 및 조회"""
        graph.query("MERGE (:Person {name: 'Bob'})")
        graph.query(
            "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) "
            "MERGE (a)-[:KNOWS]->(b)"
        )
        results = graph.query(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name AS src, b.name AS dst"
        )
        assert any(r["src"] == "Alice" and r["dst"] == "Bob" for r in results)

    def test_refresh_schema(self, graph):
        """graph.refresh_schema() + graph.schema 프로퍼티"""
        graph.refresh_schema()
        assert isinstance(graph.schema, str)
        assert len(graph.schema) > 0
        assert isinstance(graph.structured_schema, dict)
        assert "node_props" in graph.structured_schema
        assert "relationships" in graph.structured_schema

    def test_get_schema_property(self, graph):
        """graph.get_schema 프로퍼티 (Neo4j 동일)"""
        schema = graph.get_schema
        assert isinstance(schema, str)

    def test_get_structured_schema_property(self, graph):
        """graph.get_structured_schema 프로퍼티 (Neo4j 동일)"""
        ss = graph.get_structured_schema
        assert isinstance(ss, dict)
        assert "node_props" in ss

    def test_add_graph_documents(self, graph):
        """graph.add_graph_documents() — LLMGraphTransformer 출력 호환"""
        from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
        from langchain_core.documents import Document

        movie = Node(id="inception", type="Movie", properties={"title": "Inception"})
        director = Node(id="nolan", type="Director", properties={"name": "Christopher Nolan"})
        rel = Relationship(source=director, target=movie, type="DIRECTED")
        source = Document(page_content="Inception by Nolan", metadata={"source": "test"})

        graph.add_graph_documents(
            [GraphDocument(nodes=[movie, director], relationships=[rel], source=source)],
            include_source=True,
        )

        results = graph.query(
            "MATCH (d:Director)-[:DIRECTED]->(m:Movie) "
            "RETURN d.name AS director, m.title AS title"
        )
        assert any(r.get("title") == "Inception" for r in results)

    def test_include_exclude_types(self, graph):
        """include_types / exclude_types 필터링 (Neo4j 동일 파라미터)"""
        from langchain_age import AGEGraph
        g2 = AGEGraph(DSN, GRAPH, refresh_schema=True, include_types=["Person"])
        assert "Person" in g2.structured_schema.get("node_props", {})
        assert "Movie" not in g2.structured_schema.get("node_props", {})

        g3 = AGEGraph(DSN, GRAPH, refresh_schema=True, exclude_types=["Person"])
        assert "Person" not in g3.structured_schema.get("node_props", {})


# ─────────────────────────────────────────────────────────────────
# 2. AGEVector — Neo4jVector / PGVectorStore 동일 인터페이스
# ─────────────────────────────────────────────────────────────────

class TestAGEVector:
    """Neo4jVector와 동일한 인터페이스 검증"""

    def test_import_style(self):
        """from langchain_age import AGEVector, DistanceStrategy, SearchType"""
        from langchain_age import AGEVector, DistanceStrategy, SearchType
        assert AGEVector is not None
        assert DistanceStrategy.COSINE is not None
        assert SearchType.HYBRID is not None

    def test_add_documents(self, vector_store):
        """add_documents() + 반환 ID 리스트"""
        from langchain_core.documents import Document
        docs = [
            Document(page_content="Apache AGE is a graph extension for PostgreSQL.", metadata={"source": "a"}),
            Document(page_content="pgvector enables vector similarity search.", metadata={"source": "b"}),
            Document(page_content="LangChain connects LLMs to tools and data.", metadata={"source": "c"}),
        ]
        ids = vector_store.add_documents(docs)
        assert len(ids) == 3
        assert all(isinstance(i, str) for i in ids)

    def test_similarity_search(self, vector_store):
        """similarity_search(query, k) — 기본 인터페이스"""
        results = vector_store.similarity_search("graph database", k=2)
        assert len(results) <= 2
        assert all(hasattr(r, "page_content") for r in results)

    def test_similarity_search_with_score(self, vector_store):
        """similarity_search_with_score() — (Document, float) 튜플"""
        results = vector_store.similarity_search_with_score("vector search", k=2)
        assert len(results) <= 2
        for doc, score in results:
            assert hasattr(doc, "page_content")
            assert isinstance(score, float)

    def test_similarity_search_by_vector(self, vector_store):
        """similarity_search_by_vector() — 임베딩 직접 전달"""
        embedding = [0.1, 0.5, 0.8, 0.2]
        results = vector_store.similarity_search_by_vector(embedding, k=2)
        assert len(results) <= 2

    def test_add_texts(self, vector_store):
        """add_texts() 인터페이스"""
        ids = vector_store.add_texts(
            ["Graph RAG combines graphs and vectors.", "OpenCypher runs on AGE."],
            metadatas=[{"tag": "rag"}, {"tag": "cypher"}],
        )
        assert len(ids) == 2

    def test_metadata_filter_equality(self, vector_store):
        """단순 equality 메타데이터 필터"""
        results = vector_store.similarity_search("search", k=10, filter={"source": "a"})
        for doc in results:
            assert doc.metadata.get("source") == "a"

    def test_metadata_filter_operators(self, vector_store):
        """$in, $like 등 고급 필터 연산자"""
        results = vector_store.similarity_search(
            "search", k=10, filter={"source": {"$in": ["a", "b"]}}
        )
        for doc in results:
            assert doc.metadata.get("source") in ("a", "b")

    def test_delete(self, vector_store):
        """delete(ids) 인터페이스"""
        ids = vector_store.add_texts(["delete me"])
        result = vector_store.delete(ids)
        assert result is True

    def test_get_by_ids(self, vector_store):
        """get_by_ids() 인터페이스"""
        ids = vector_store.add_texts(["fetch by id test"])
        docs = vector_store.get_by_ids(ids)
        assert len(docs) == 1
        assert docs[0].page_content == "fetch by id test"

    def test_as_retriever(self, vector_store):
        """as_retriever() — LangChain Retriever 변환 (VectorStore 상속)"""
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        assert retriever is not None
        results = retriever.invoke("graph database")
        assert isinstance(results, list)

    def test_from_texts_classmethod(self):
        """from_texts() 클래스 메서드"""
        from langchain_age import AGEVector
        store = AGEVector.from_texts(
            texts=["hello world", "foo bar"],
            embedding=FakeEmbeddings(),
            connection_string=DSN,
            collection_name="test_from_texts_tmp",
            pre_delete_collection=True,
        )
        results = store.similarity_search("hello", k=1)
        assert len(results) >= 1
        store._drop_table()

    def test_from_existing_index_classmethod(self):
        """from_existing_index() — 기존 테이블 재연결"""
        from langchain_age import AGEVector
        store = AGEVector.from_existing_index(
            embedding=FakeEmbeddings(),
            connection_string=DSN,
            collection_name="test_compat_vectors",
        )
        assert store is not None

    def test_hybrid_search(self):
        """hybrid search (SearchType.HYBRID) — vector + full-text RRF"""
        from langchain_age import AGEVector, SearchType
        store = AGEVector(
            connection_string=DSN,
            embedding_function=FakeEmbeddings(),
            collection_name="test_hybrid_tmp",
            search_type=SearchType.HYBRID,
            pre_delete_collection=True,
        )
        store.add_texts([
            "PostgreSQL is a powerful relational database.",
            "Apache AGE adds graph capabilities to PostgreSQL.",
            "pgvector enables fast vector similarity search.",
        ])
        results = store.similarity_search("graph database PostgreSQL", k=3)
        assert len(results) >= 1
        store._drop_table()

    def test_hnsw_index_creation(self, vector_store):
        """HNSW 인덱스 생성 인터페이스"""
        vector_store.create_hnsw_index(m=8, ef_construction=32)


# ─────────────────────────────────────────────────────────────────
# 3. from_existing_graph
# ─────────────────────────────────────────────────────────────────

class TestFromExistingGraph:
    """Neo4jVector.from_existing_graph() 동일 패턴"""

    def test_from_existing_graph(self, graph):
        from langchain_age import AGEVector
        # 먼저 그래프 노드 확인 (TestAGEGraph에서 생성된 것 활용)
        # "desc" is a Cypher reserved keyword — backtick quoting in from_existing_graph handles it.
        graph.query("MERGE (:Product {name: 'LangChain', desc: 'LLM framework'})")
        graph.query("MERGE (:Product {name: 'pgvector', desc: 'Vector search extension'})")

        store = AGEVector.from_existing_graph(
            embedding=FakeEmbeddings(),
            connection_string=DSN,
            graph_name=GRAPH,
            node_label="Product",
            text_node_properties=["name", "desc"],
            collection_name="test_from_graph_tmp",
        )
        results = store.similarity_search("LLM", k=2)
        assert len(results) >= 1
        store._drop_table()


# ─────────────────────────────────────────────────────────────────
# 4. AGEGraphCypherQAChain — GraphCypherQAChain 동일 인터페이스
# ─────────────────────────────────────────────────────────────────

class FakeLLM:
    """LLM 없이 체인 파이프라인 구조만 검증하는 더미.

    LangChain의 Runnable pipe ( | ) 는 실제 RunnableSequence를 만드므로
    __or__을 오버라이드하지 않고, ChatPromptTemplate | FakeLLM() 형태로
    쓰려면 LangChain의 Runnable 프로토콜을 따라야 합니다.
    여기서는 직접 chain을 구성하지 않고 invoke만 검증합니다.
    """
    def invoke(self, input, config=None, **kwargs):
        # 프롬프트 메시지 목록이 들어오면 → Cypher 반환
        if isinstance(input, list):
            # 스키마가 포함된 system message → Cypher generation 단계
            content = str(input)
            if "schema" in content.lower() or "cypher" in content.lower():
                return "MATCH (n:Person) RETURN n.name AS name LIMIT 3"
            return "Alice and Bob are people in the graph."
        if isinstance(input, dict):
            if "schema" in input:
                return "MATCH (n:Person) RETURN n.name AS name LIMIT 3"
            return "Alice and Bob are people in the graph."
        return "Alice and Bob are people in the graph."


class TestAGEGraphCypherQAChain:
    """GraphCypherQAChain 동일 인터페이스 검증"""

    def test_import_style(self):
        """from langchain_age import AGEGraphCypherQAChain"""
        from langchain_age import AGEGraphCypherQAChain
        assert AGEGraphCypherQAChain is not None

    def test_dangerous_request_gate(self, graph):
        """allow_dangerous_requests=False → ValueError (Neo4j 동일)"""
        from langchain_age import AGEGraphCypherQAChain
        with pytest.raises(ValueError, match="allow_dangerous_requests"):
            AGEGraphCypherQAChain.from_llm(
                FakeLLM(), graph=graph, allow_dangerous_requests=False
            )

    def _make_chain(self, graph, **kwargs):
        """RunnableLambda로 fake chain 구성 (LangChain v1 Pydantic 검증 통과)"""
        from langchain_age import AGEGraphCypherQAChain
        from langchain_core.runnables import RunnableLambda

        cypher_chain = RunnableLambda(
            lambda _: "MATCH (n:Person) RETURN n.name AS name LIMIT 3"
        )
        qa_chain = RunnableLambda(
            lambda _: "Alice and Bob are people in the graph."
        )
        return AGEGraphCypherQAChain(
            graph=graph,
            cypher_generation_chain=cypher_chain,
            qa_chain=qa_chain,
            allow_dangerous_requests=True,
            **kwargs,
        )

    def test_chain_invoke(self, graph):
        """chain.invoke({'query': ...}) — 기본 실행 흐름"""
        graph.refresh_schema()
        chain = self._make_chain(graph)
        result = chain.invoke({"query": "Who are the people?"})
        assert "result" in result
        assert isinstance(result["result"], str)

    def test_chain_run(self, graph):
        """chain.run(query) — 단일 문자열 인터페이스 (Neo4j 동일)"""
        chain = self._make_chain(graph)
        result = chain.run("List all people")
        assert isinstance(result, str)

    def test_return_intermediate_steps(self, graph):
        """return_intermediate_steps=True → Cypher + context 포함"""
        chain = self._make_chain(graph, return_intermediate_steps=True)
        result = chain.invoke({"query": "Find people"})
        assert "intermediate_steps" in result
        steps = result["intermediate_steps"]
        assert any("query" in s for s in steps)

    def test_include_exclude_types_in_chain(self, graph):
        """include_types / exclude_types — 체인 레벨 스키마 필터"""
        chain = self._make_chain(graph, include_types=["Person"])
        result = chain.invoke({"query": "Find people"})
        assert "result" in result

    def test_custom_prompts(self, graph):
        """cypher_prompt / qa_prompt 커스터마이징 (Neo4j 동일 파라미터)"""
        from langchain_age import AGEGraphCypherQAChain, CYPHER_GENERATION_PROMPT, QA_PROMPT
        assert CYPHER_GENERATION_PROMPT is not None
        assert QA_PROMPT is not None
        chain = self._make_chain(graph)
        assert chain is not None
