# 시작 가이드

## 사전 요구사항

- Python 3.10+
- Docker (데이터베이스용)
- OpenAI API 키 (선택사항, 임베딩/LLM 기능 사용 시)

## 1. 데이터베이스 시작

```bash
git clone https://github.com/BAEM1N/langchain-age.git
cd langchain-age/docker
docker compose up -d
```

하나의 PostgreSQL 18 컨테이너에 다음이 포함됩니다:
- **Apache AGE 1.7.0** — 그래프 엔진 (Cypher 지원)
- **pgvector** — 벡터 유사도 검색
- **pg_trgm** — 전문 검색용 트리그램

정상 동작 확인:

```bash
docker compose ps
# "healthy" 표시 확인
```

## 2. 라이브러리 설치

필요한 모드 선택:

```bash
# 그래프만 (AGEGraph + AGEGraphCypherQAChain)
pip install "langchain-age[graph]"

# 벡터만 (AGEVector + pgvector)
pip install "langchain-age[vector]"

# 전부
pip install "langchain-age[all]"
```

## 3. 첫 번째 그래프 쿼리

```python
from langchain_age import AGEGraph

graph = AGEGraph(
    "host=localhost port=5433 dbname=langchain_age user=langchain password=langchain",
    graph_name="quickstart",
)

# 노드 생성
graph.query("CREATE (:Person {name: 'Alice', role: 'engineer'})")
graph.query("CREATE (:Person {name: 'Bob', role: 'designer'})")
graph.query(
    "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) "
    "CREATE (a)-[:WORKS_WITH]->(b)"
)

# 조회
results = graph.query(
    "MATCH (a:Person)-[:WORKS_WITH]->(b:Person) "
    "RETURN a.name AS from_person, b.name AS to_person"
)
print(results)
# [{'from_person': 'Alice', 'to_person': 'Bob'}]

graph.close()
```

## 4. 첫 번째 벡터 검색

```python
from langchain_age import AGEVector, DistanceStrategy
from langchain_openai import OpenAIEmbeddings  # pip install langchain-openai

store = AGEVector(
    connection_string="host=localhost port=5433 dbname=langchain_age user=langchain password=langchain",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="quickstart_docs",
    distance_strategy=DistanceStrategy.COSINE,
)

# 문서 추가
store.add_texts([
    "Apache AGE는 PostgreSQL에 그래프 쿼리 기능을 추가합니다.",
    "pgvector는 빠른 벡터 유사도 검색을 지원합니다.",
    "LangChain은 LLM 애플리케이션 구축 프레임워크입니다.",
])

# 검색
results = store.similarity_search("그래프 데이터베이스", k=2)
for doc in results:
    print(doc.page_content)

store.close()
```

## 5. 그래프 + 벡터 결합

```python
from langchain_age import AGEGraph, AGEVector

# 지식 그래프 구축
graph = AGEGraph(conn_str, graph_name="kb")
graph.query("CREATE (:Topic {name: 'PostgreSQL', desc: 'relational database'})")
graph.query("CREATE (:Topic {name: 'AGE', desc: 'graph extension for PG'})")
graph.query(
    "MATCH (a:Topic {name: 'AGE'}), (b:Topic {name: 'PostgreSQL'}) "
    "CREATE (a)-[:EXTENDS]->(b)"
)

# 그래프 노드를 벡터화
store = AGEVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    connection_string=conn_str,
    graph_name="kb",
    node_label="Topic",
    text_node_properties=["name", "desc"],
)

# 벡터 검색 → 그래프 컨텍스트 확장
docs = store.similarity_search("graph query", k=1)
print(docs[0].page_content)
# "AGE graph extension for PG"
print(docs[0].metadata)
# {"age_node_id": "...", "node_label": "Topic"}
```

## 다음 단계

- [튜토리얼](tutorial.md) — 전체 기능 상세 가이드
- [API 레퍼런스](api-reference.md) — 클래스/메서드 문서
- [notebooks/](../../notebooks/) — 실행 가능한 Jupyter 노트북 예제
