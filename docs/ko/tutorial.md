# 튜토리얼

langchain-age 전체 기능 상세 가이드.

## 목차

1. [그래프 기능](#1-그래프-기능)
2. [벡터 기능](#2-벡터-기능)
3. [하이브리드 검색](#3-하이브리드-검색)
4. [그래프 + 벡터 (GraphRAG)](#4-그래프--벡터-graphrag)
5. [Cypher QA 체인](#5-cypher-qa-체인)
6. [딥 트래버셜](#6-딥-트래버셜)
7. [메타데이터 필터링](#7-메타데이터-필터링)
8. [성능 튜닝](#8-성능-튜닝)
9. [LangGraph 연동](#9-langgraph-연동)

---

## 1. 그래프 기능

### 연결

```python
from langchain_age import AGEGraph

# 기본 연결
graph = AGEGraph(
    "host=localhost port=5433 dbname=langchain_age user=langchain password=langchain",
    graph_name="tutorial",
)

# 컨텍스트 매니저 (자동 정리)
with AGEGraph(conn_str, "tutorial") as graph:
    graph.query("MATCH (n) RETURN count(n) AS total")
```

### CRUD 기본 조작

```python
# 생성
graph.query("CREATE (:Person {name: 'Alice', age: 30})")

# 조회
results = graph.query("MATCH (n:Person) RETURN n.name AS name, n.age AS age")

# 수정
graph.query("MATCH (n:Person {name: 'Alice'}) SET n.age = 31")

# 삭제
graph.query("MATCH (n:Person {name: 'Alice'}) DELETE n")
```

### 파라미터 바인딩 (mogrify)

AGE는 네이티브 `$param` 바인딩을 지원하지 않지만,
psycopg3의 `mogrify`를 통해 안전한 값 이스케이핑을 제공합니다:

```python
# 안전 — psycopg3가 값을 이스케이핑
graph.query(
    "MATCH (n:Person) WHERE n.name = %s RETURN n.age AS age",
    params=("Alice",),
)

# 수동 이스케이핑도 가능
from langchain_age.utils.cypher import escape_cypher_string
name = escape_cypher_string(user_input)
graph.query(f"MATCH (n:Person {{name: '{name}'}}) RETURN n")
```

### 스키마 탐색

```python
graph.refresh_schema()
print(graph.schema)
# Node labels and properties:
#   :Person {age, name}
# Relationship types and properties:
#   [:KNOWS] {since}
# Relationship patterns:
#   (:Person)-[:KNOWS]->(:Person)

# 프로그래밍 접근
schema = graph.structured_schema
print(schema["node_props"])    # {"Person": ["age", "name"]}
print(schema["relationships"]) # [{"start": "Person", "type": "KNOWS", "end": "Person"}]
```

### GraphDocument 일괄 삽입

```python
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

doc = GraphDocument(
    nodes=[
        Node(id="alice", type="Person", properties={"name": "Alice"}),
        Node(id="bob", type="Person", properties={"name": "Bob"}),
    ],
    relationships=[
        Relationship(
            source=Node(id="alice", type="Person"),
            target=Node(id="bob", type="Person"),
            type="KNOWS",
        ),
    ],
    source=Document(page_content="Alice knows Bob"),
)

# 배치 삽입 — 내부적으로 UNWIND 사용
graph.add_graph_documents([doc], include_source=True)
```

---

## 2. 벡터 기능

### 기본 유사도 검색

```python
from langchain_age import AGEVector, DistanceStrategy
from langchain_openai import OpenAIEmbeddings

store = AGEVector(
    connection_string=conn_str,
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="docs",
    distance_strategy=DistanceStrategy.COSINE,
)

store.add_texts(["PostgreSQL은 강력합니다.", "AGE는 그래프 쿼리를 추가합니다."])

# 유사도 검색
docs = store.similarity_search("데이터베이스", k=2)

# 거리 점수 포함 (낮을수록 유사)
results = store.similarity_search_with_score("데이터베이스", k=2)

# 관련도 점수 포함 (0~1, 높을수록 유사)
results = store.similarity_search_with_relevance_scores("데이터베이스", k=2)
```

### MMR (Maximal Marginal Relevance)

관련성과 다양성의 균형. DB에 저장된 임베딩을 재사용하므로 추가 API 호출 없음.

```python
docs = store.max_marginal_relevance_search(
    "데이터베이스 기술",
    k=3,          # 3개 반환
    fetch_k=10,   # 상위 10개 후보 고려
    lambda_mult=0.5,  # 0=최대 다양성, 1=최대 관련성
)
```

### 인덱스 관리

```python
# HNSW (프로덕션 권장)
store.create_hnsw_index(m=16, ef_construction=64)

# IVFFlat (빌드 빠름, 정확도 약간 낮음)
store.create_ivfflat_index(n_lists=100)

# 모든 인덱스 삭제
store.drop_index()
```

### LangChain Retriever

```python
retriever = store.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("AGE란?")

# RAG 체인에서 사용
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template(
    "컨텍스트를 기반으로 답변:\n{context}\n\n질문: {question}"
)
chain = prompt | ChatOpenAI() | StrOutputParser()
```

---

## 3. 하이브리드 검색

벡터 유사도와 PostgreSQL 전문 검색을 RRF(Reciprocal Rank Fusion, k=60)로 결합.

```python
from langchain_age import AGEVector, SearchType

store = AGEVector(
    connection_string=conn_str,
    embedding_function=embeddings,
    collection_name="hybrid_docs",
    search_type=SearchType.HYBRID,  # 하이브리드 모드 활성화
)

store.add_texts([
    "PostgreSQL은 JSON과 전문 검색을 지원합니다.",
    "Apache AGE는 PostgreSQL에 Cypher 그래프 쿼리를 추가합니다.",
    "pgvector는 벡터 유사도 검색을 지원합니다.",
])

# 자동으로 벡터 + 키워드 매칭 결합
results = store.similarity_search("PostgreSQL 그래프 확장", k=3)
```

---

## 4. 그래프 + 벡터 (GraphRAG)

### 기존 그래프 노드 벡터화

```python
store = AGEVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    connection_string=conn_str,
    graph_name="tutorial",
    node_label="Person",
    text_node_properties=["name", "bio"],  # 텍스트로 결합
    collection_name="person_vectors",
)
```

### 벡터 검색 → 그래프 컨텍스트 확장

```python
# 1단계: 벡터 검색으로 관련 노드 찾기
docs = store.similarity_search("엔지니어", k=2)

# 2단계: 그래프에서 관계 확장
for doc in docs:
    node_label = doc.metadata["node_label"]
    neighbors = graph.query(
        f"MATCH (n:{node_label})-[r]->(m) "
        f"WHERE id(n) = {doc.metadata['age_node_id']} "
        f"RETURN type(r) AS rel, m.name AS neighbor"
    )
```

---

## 5. Cypher QA 체인

LLM이 Cypher 생성 → AGE 실행 → LLM이 자연어 답변.

```python
from langchain_age import AGEGraph, AGEGraphCypherQAChain
from langchain_openai import ChatOpenAI

graph = AGEGraph(conn_str, "tutorial")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = AGEGraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    allow_dangerous_requests=True,
    return_intermediate_steps=True,
    verbose=True,
)

result = chain.invoke({"query": "Alice와 함께 일하는 사람은?"})
print(result["result"])                         # 자연어 답변
print(result["intermediate_steps"][0]["query"])  # 생성된 Cypher
```

### 스키마 필터링

```python
# 특정 타입만 LLM에 노출
chain = AGEGraphCypherQAChain.from_llm(
    llm, graph=graph,
    include_types=["Person", "KNOWS"],   # 화이트리스트
    # exclude_types=["InternalNode"],     # 또는 블랙리스트
    allow_dangerous_requests=True,
)
```

---

## 6. 딥 트래버셜

`traverse()`는 PostgreSQL `WITH RECURSIVE` 사용 — Cypher `*N`보다 10~22배 빠름.

```python
# 6홉 이내 도달 가능한 모든 노드 탐색
results = graph.traverse(
    start_label="Person",
    start_filter={"name": "Alice"},
    edge_label="KNOWS",
    max_depth=6,
    direction="outgoing",      # "incoming" 또는 "both"
    return_properties=True,
)

for r in results:
    print(f"  depth={r['depth']} → {r['properties']}")
```

### 프로퍼티 인덱스

`traverse()` 시작 노드 조회 가속:

```python
# 특정 프로퍼티 B-tree 인덱스
graph.create_property_index("Person", "name")

# 전체 프로퍼티 GIN 인덱스 (빌드 느림, 모든 연산자 지원)
graph.create_property_index("Person", "name", index_type="gin")
```

---

## 7. 메타데이터 필터링

JSONB 메타데이터에 대한 MongoDB 스타일 필터 연산자:

```python
# 같음
store.similarity_search("쿼리", filter={"author": "Alice"})

# 비교 연산자
store.similarity_search("쿼리", filter={"year": {"$gte": 2024}})
store.similarity_search("쿼리", filter={"tag": {"$in": ["ai", "ml"]}})
store.similarity_search("쿼리", filter={"score": {"$between": [0.5, 1.0]}})
store.similarity_search("쿼리", filter={"title": {"$ilike": "%graph%"}})

# 논리 조합
store.similarity_search("쿼리", filter={
    "$and": [
        {"author": "Alice"},
        {"year": {"$gte": 2024}},
    ]
})
```

지원 연산자: `$eq`, `$ne`, `$lt`, `$lte`, `$gt`, `$gte`, `$in`, `$nin`,
`$between`, `$like`, `$ilike`, `$exists`, `$and`, `$or`

---

## 8. 성능 튜닝

### 배치 크기

```python
store = AGEVector(
    ...,
    batch_size=5000,  # 기본값 1000
)
```

### HNSW 파라미터

```python
# m 증가 = 더 나은 재현율, 더 많은 메모리
# ef_construction 증가 = 더 나은 품질, 더 느린 빌드
store.create_hnsw_index(m=32, ef_construction=128)
```

### 스키마 새로고침

`refresh_schema()`는 `ag_catalog` 시스템 테이블을 직접 조회합니다
(라벨별 Cypher가 아닌 SQL). 수백 개 라벨에서도 잘 확장됩니다.

### 딥 트래버셜 vs Cypher

| 패턴 | 방법 | 사용 시점 |
|------|------|----------|
| 1~3 홉 | `graph.query("MATCH ...*3...")` | 간단하고 읽기 좋음 |
| 4+ 홉 | `graph.traverse(max_depth=N)` | 10~22배 빠름 |

---

## 9. LangGraph 연동

langchain-age는 LangGraph의 `PostgresStore`/`PostgresSaver`와 같은
PostgreSQL 인스턴스를 사용합니다. 모든 테이블이 공존합니다.

```python
from langgraph.store.postgres import PostgresStore

# AGEGraph / AGEVector와 동일한 연결 문자열
with PostgresStore.from_conn_string(conn_str) as store:
    store.setup()
    store.put(("users", "123"), "prefs", {"theme": "dark"})
    item = store.get(("users", "123"), "prefs")
```

추가 데이터베이스 불필요 — 그래프, 벡터, 장기 메모리가 하나의 PostgreSQL에.
