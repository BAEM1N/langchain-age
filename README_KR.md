# langchain-age

[![PyPI](https://img.shields.io/pypi/v/langchain-age)](https://pypi.org/project/langchain-age/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-age)](https://pypi.org/project/langchain-age/)
[![CI](https://github.com/BAEM1N/langchain-age/actions/workflows/ci.yml/badge.svg)](https://github.com/BAEM1N/langchain-age/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![EN](https://img.shields.io/badge/lang-English-red.svg)](README.md)

[Apache AGE](https://age.apache.org/) (그래프) + [pgvector](https://github.com/pgvector/pgvector) (벡터)를 PostgreSQL 위에서 LangChain과 통합하는 라이브러리입니다.

`langchain-neo4j`와 동일한 API — Neo4j 대신 PostgreSQL에서 동작합니다.

## 설치

```bash
pip install langchain-age            # 그래프 (AGEGraph + Cypher QA 체인)
pip install "langchain-age[all]"     # 그래프 + 벡터 (+ AGEVector)
```

Apache AGE 드라이버가 내장되어 있어 추가 설치가 필요 없습니다.

## 빠른 시작

### AGEGraph

```python
from langchain_age import AGEGraph

graph = AGEGraph(
    "host=localhost port=5433 dbname=langchain_age user=langchain password=langchain",
    graph_name="my_graph",
)

graph.query("CREATE (:Person {name: 'Alice', age: 30})")
results = graph.query("MATCH (n:Person) RETURN n.name AS name, n.age AS age")
# [{'name': 'Alice', 'age': 30}]
```

### AGEVector

```python
from langchain_age import AGEVector, DistanceStrategy
from langchain_openai import OpenAIEmbeddings

store = AGEVector(
    connection_string="host=localhost port=5433 dbname=langchain_age user=langchain password=langchain",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="my_docs",
    distance_strategy=DistanceStrategy.COSINE,
)

store.add_texts(["Apache AGE는 PostgreSQL에 Cypher를 추가합니다.", "pgvector는 벡터 검색을 지원합니다."])
docs = store.similarity_search("그래프 데이터베이스", k=2)
```

### AGEGraphCypherQAChain

```python
from langchain_age import AGEGraph, AGEGraphCypherQAChain
from langchain_openai import ChatOpenAI

chain = AGEGraphCypherQAChain.from_llm(
    ChatOpenAI(model="gpt-4o-mini"),
    graph=AGEGraph("...", "movies"),
    allow_dangerous_requests=True,
)
answer = chain.run("톰 행크스가 출연한 영화는?")
```

### 그래프 + 벡터 (GraphRAG)

```python
store = AGEVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    connection_string="...",
    graph_name="my_graph",
    node_label="Document",
    text_node_properties=["title", "content"],
)
docs = store.similarity_search("머신러닝", k=3)
```

## 기능

| 컴포넌트 | 클래스 | 설명 |
|----------|--------|------|
| **그래프** | `AGEGraph` | Cypher 실행, 스키마 탐색, `add_graph_documents()`, `traverse()` (WITH RECURSIVE, Cypher `*N` 대비 10~22배 빠름), 프로퍼티 인덱싱 |
| **벡터** | `AGEVector` | Cosine/L2/IP 유사도, HNSW & IVFFlat 인덱스, 하이브리드 검색 (벡터 + 전문 검색 RRF), MMR, 메타데이터 필터 (`$eq`, `$in`, `$between`, `$like`, `$and`, `$or`, ...) |
| **체인** | `AGEGraphCypherQAChain` | LLM이 Cypher 생성 → AGE 실행 → LLM 답변. 스키마 필터링, Cypher 검증, function response 모드 |

## 왜 Neo4j 대신 AGE인가?

| | Neo4j | Apache AGE |
|---|---|---|
| **인프라** | 별도 데이터베이스 | **기존 PostgreSQL에 확장** |
| **비용 (HA)** | 연 $15K+ (Enterprise) | **$0** (PG 네이티브 HA) |
| **라이선스** | GPL / 상용 | **Apache 2.0** |
| **벡터 검색** | Enterprise 기능 | **pgvector (무료, 같은 DB)** |
| **LangGraph 메모리** | 별도 DB 필요 | **같은 PostgreSQL** |
| **운영** | 그래프 DB 전문 지식 필요 | **기존 PG DBA로 충분** |

둘 다 Cypher를 사용합니다. `langchain-age`가 SQL 래핑을 자동 처리하므로 Neo4j와 동일한 Cypher를 작성하면 됩니다.

## 데이터베이스 설정

```bash
git clone https://github.com/BAEM1N/langchain-age.git
cd langchain-age/docker
docker compose up -d
```

단일 컨테이너: PostgreSQL 18 + Apache AGE 1.7.0 + pgvector + pg_trgm.

## 문서

| 언어 | 시작 가이드 | 튜토리얼 | API 레퍼런스 |
|------|:-:|:-:|:-:|
| English | [Link](docs/en/getting-started.md) | [Link](docs/en/tutorial.md) | [Link](docs/en/api-reference.md) |
| 한국어 | [Link](docs/ko/getting-started.md) | [Link](docs/ko/tutorial.md) | [Link](docs/ko/api-reference.md) |

### 노트북

| 노트북 | 설명 |
|--------|------|
| [01_graph.ipynb](notebooks/01_graph.ipynb) | Cypher CRUD, 스키마, GraphDocument (API 키 불필요) |
| [02_vector.ipynb](notebooks/02_vector.ipynb) | 유사도, 하이브리드, MMR, 필터, HNSW (OpenAI) |
| [03_graph_vector.ipynb](notebooks/03_graph_vector.ipynb) | GraphRAG, QA 체인 (OpenAI) |

## 테스트

```bash
pytest tests/unit/                # 65개, DB 불필요

export LANGCHAIN_AGE_TEST_DSN="host=localhost port=5433 dbname=langchain_age user=langchain password=langchain"
pytest tests/integration/         # 53개, Docker 필요
```

## 호환성

- **Python**: 3.10, 3.11, 3.12, 3.13, 3.14
- **PostgreSQL**: 18 + Apache AGE 1.7.0 + pgvector 0.8.2
- **LangChain**: v1 (`langchain-core>=1.0.0`)

### 알려진 제약사항

- AGE는 파라미터화된 Cypher (`$param`)를 지원하지 않음 — `mogrify` 기반 이스케이핑 제공
- Async 메서드는 `run_in_executor` 래핑 사용 (네이티브 `psycopg.AsyncConnection` 미지원)

## 기여하기

기여를 환영합니다. 큰 변경은 먼저 Issue를 열어주세요.

```bash
git clone https://github.com/BAEM1N/langchain-age.git
cd langchain-age
pip install -e ".[dev]"
pytest tests/unit/
ruff check langchain_age/ tests/
mypy langchain_age/
```

## 라이선스

MIT — [LICENSE](LICENSE) 참조.

내장된 Apache AGE Python 드라이버 (`langchain_age/_vendor/age/`)는 Apache 2.0 라이선스입니다.
