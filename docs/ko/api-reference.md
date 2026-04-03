# API 레퍼런스

## AGEGraph

`langchain_age.graphs.age_graph.AGEGraph`

PostgreSQL + Apache AGE 기반 GraphStore.

### 생성자

```python
AGEGraph(
    connection_string: str,
    graph_name: str,
    *,
    timeout: float | None = None,
    refresh_schema: bool = True,
    sanitize: bool = True,
    enhanced_schema: bool = False,
    include_types: list[str] | None = None,
    exclude_types: list[str] | None = None,
    max_retries: int = 3,
)
```

### 메서드

| 메서드 | 설명 |
|--------|------|
| `query(query, params=None)` | Cypher 실행, dict 리스트 반환. `%s` mogrify 파라미터 지원. |
| `refresh_schema()` | `ag_catalog` SQL로 스키마 재탐색. |
| `add_graph_documents(docs, include_source=False)` | UNWIND로 `GraphDocument` 배치 삽입. |
| `traverse(start_label, start_filter, edge_label, max_depth, *, direction, return_properties)` | WITH RECURSIVE 딥 홉 (Cypher `*N` 대비 10~22배). |
| `create_property_index(node_label, property_name, *, index_type="btree")` | 노드 프로퍼티에 B-tree/GIN 인덱스 생성. |
| `create_graph()` | 그래프 생성 (없으면). |
| `drop_graph()` | 그래프 삭제. 되돌릴 수 없음. |
| `close()` | 연결 닫기. |

### 프로퍼티

| 프로퍼티 | 타입 | 설명 |
|---------|------|------|
| `get_schema` | `str` | 사람이 읽을 수 있는 스키마 문자열. |
| `get_structured_schema` | `dict` | `node_props`, `rel_props`, `relationships` 포함. |

---

## AGEVector

`langchain_age.vectorstores.age_vector.AGEVector`

pgvector 기반 VectorStore + 선택적 AGE 그래프 연계.

### 생성자

```python
AGEVector(
    connection_string: str,
    embedding_function: Embeddings,
    *,
    collection_name: str = "langchain_age_vectors",
    distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
    search_type: SearchType = SearchType.VECTOR,
    pre_delete_collection: bool = False,
    relevance_score_fn: Callable[[float], float] | None = None,
    age_graph_name: str | None = None,
    retrieval_query: str | None = None,
    embedding_dimension: int | None = None,
    batch_size: int = 1_000,
)
```

### 메서드

| 메서드 | 설명 |
|--------|------|
| `add_texts(texts, metadatas=None, ids=None)` | 텍스트 임베딩 및 저장. `executemany` 배치. |
| `add_documents(documents, ids=None)` | `Document` 객체 임베딩 및 저장. |
| `similarity_search(query, k=4, filter=None)` | k개 가장 유사한 문서 반환. |
| `similarity_search_with_score(query, k=4, filter=None)` | 거리 점수 포함 반환. |
| `similarity_search_with_relevance_scores(query, k=4, filter=None)` | [0,1] 정규화 점수 반환. |
| `similarity_search_by_vector(embedding, k=4, filter=None)` | 사전 계산된 벡터로 검색. |
| `max_marginal_relevance_search(query, k=4, fetch_k=20, lambda_mult=0.5)` | MMR 검색, 저장된 임베딩 재사용. |
| `delete(ids=None)` | ID로 문서 삭제. |
| `get_by_ids(ids)` | ID로 문서 조회. |
| `as_retriever(**kwargs)` | LangChain Retriever로 변환. |
| `create_hnsw_index(m=16, ef_construction=64)` | HNSW 인덱스 생성. |
| `create_ivfflat_index(n_lists=100)` | IVFFlat 인덱스 생성. |
| `drop_index()` | 모든 벡터 인덱스 삭제. |
| `close()` | 연결 닫기. |

### 클래스 메서드

| 메서드 | 설명 |
|--------|------|
| `from_texts(texts, embedding, **kwargs)` | 텍스트로 생성 및 저장. |
| `from_documents(documents, embedding, **kwargs)` | Document로 생성 및 저장. |
| `from_existing_index(embedding, connection_string, collection_name)` | 기존 테이블에 연결. |
| `from_existing_graph(embedding, connection_string, graph_name, node_label, text_node_properties)` | AGE 그래프 노드를 벡터화. |

---

## AGEGraphCypherQAChain

`langchain_age.chains.graph_cypher_qa_chain.AGEGraphCypherQAChain`

QA 체인: LLM이 Cypher 생성 → AGE 실행 → LLM이 답변.

### 팩토리

```python
AGEGraphCypherQAChain.from_llm(
    llm: BaseLanguageModel,
    graph: AGEGraph,
    *,
    cypher_llm: BaseLanguageModel | None = None,
    qa_llm: BaseLanguageModel | None = None,
    include_types: list[str] | None = None,
    exclude_types: list[str] | None = None,
    validate_cypher: bool = True,
    allow_dangerous_requests: bool = False,  # 반드시 True
)
```

### 실행

| 메서드 | 설명 |
|--------|------|
| `invoke({"query": "..."})` | `{"result": "...", "intermediate_steps": [...]}` 반환. |
| `run("...")` | 단일 문자열 인터페이스. |

---

## Enum

### DistanceStrategy

| 값 | 연산자 | 설명 |
|----|--------|------|
| `COSINE` | `<=>` | 코사인 거리 (기본값) |
| `EUCLIDEAN` | `<->` | L2 거리 |
| `MAX_INNER_PRODUCT` | `<#>` | 음의 내적 |

### SearchType

| 값 | 설명 |
|----|------|
| `VECTOR` | 벡터 유사도만 (기본값) |
| `HYBRID` | 벡터 + PostgreSQL 전문 검색 (RRF) |

---

## 유틸리티 함수

`langchain_age.utils.cypher`

| 함수 | 설명 |
|------|------|
| `escape_cypher_identifier(name)` | Cypher 예약어 백틱 쿼팅. |
| `escape_cypher_string(value)` | `''` 더블링 (OpenCypher 표준). |
| `validate_sql_identifier(name)` | SQL 식별자 안전성 정규식 검증. |
| `validate_cypher(query)` | 경량 Cypher 문법 검증. |
| `wrap_cypher_query(graph, cypher, columns)` | PG18 호환 `SELECT * FROM cypher(...)` 빌드. |
