# langchain-age

[![PyPI](https://img.shields.io/pypi/v/langchain-age)](https://pypi.org/project/langchain-age/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-age)](https://pypi.org/project/langchain-age/)
[![CI](https://github.com/BAEM1N/langchain-age/actions/workflows/ci.yml/badge.svg)](https://github.com/BAEM1N/langchain-age/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

LangChain integration for [Apache AGE](https://age.apache.org/) (graph) + [pgvector](https://github.com/pgvector/pgvector) (vector) on PostgreSQL.

[한국어 문서 (Korean)](#한국어)

## Installation

```bash
pip install langchain-age            # Graph (AGEGraph + Cypher QA chain)
pip install "langchain-age[all]"     # Graph + Vector (+ AGEVector with pgvector)
```

Everything works out of the box — the Apache AGE driver is vendored, no extra steps needed.

## Quick Start

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

store.add_texts(["Apache AGE adds Cypher to PostgreSQL.", "pgvector enables vector search."])
docs = store.similarity_search("graph database", k=2)
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
answer = chain.run("Which movies did Tom Hanks act in?")
```

### Graph + Vector (GraphRAG)

```python
store = AGEVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    connection_string="...",
    graph_name="my_graph",
    node_label="Document",
    text_node_properties=["title", "content"],
)
docs = store.similarity_search("machine learning", k=3)
```

## Features

| Component | Class | Description |
|-----------|-------|-------------|
| **Graph** | `AGEGraph` | Cypher execution, schema introspection, `add_graph_documents()`, `traverse()` (WITH RECURSIVE, 10–22x faster than Cypher `*N`), property indexing |
| **Vector** | `AGEVector` | Cosine/L2/IP similarity, HNSW & IVFFlat indexes, hybrid search (vector + full-text via RRF), MMR, metadata filters (`$eq`, `$in`, `$between`, `$like`, `$and`, `$or`, ...) |
| **Chain** | `AGEGraphCypherQAChain` | LLM generates Cypher → AGE executes → LLM answers. Schema filtering, Cypher validation, function response mode |

## Why AGE over Neo4j?

| | Neo4j | Apache AGE |
|---|---|---|
| **Infrastructure** | Separate database | **Extension on your existing PostgreSQL** |
| **Cost (HA)** | $15K+/year (Enterprise) | **$0** (PostgreSQL native HA) |
| **License** | GPL / Commercial | **Apache 2.0** |
| **Vector search** | Enterprise feature | **pgvector (free, same DB)** |
| **LangGraph memory** | Separate DB | **Same PostgreSQL** |
| **Ops team** | Graph DB expertise needed | **Your existing PG DBA** |

Both use Cypher. `langchain-age` wraps the SQL automatically — you write the same Cypher as Neo4j.

## Database Setup

```bash
git clone https://github.com/BAEM1N/langchain-age.git
cd langchain-age/docker
docker compose up -d
```

Single container: PostgreSQL 18 + Apache AGE 1.7.0 + pgvector + pg_trgm.

## Documentation

| Language | Getting Started | Tutorial | API Reference |
|----------|:-:|:-:|:-:|
| English | [Link](docs/en/getting-started.md) | [Link](docs/en/tutorial.md) | [Link](docs/en/api-reference.md) |
| Korean | [Link](docs/ko/getting-started.md) | [Link](docs/ko/tutorial.md) | [Link](docs/ko/api-reference.md) |

### Notebooks

| Notebook | Description |
|----------|-------------|
| [01_graph.ipynb](notebooks/01_graph.ipynb) | Cypher CRUD, schema, GraphDocument (no API key needed) |
| [02_vector.ipynb](notebooks/02_vector.ipynb) | Similarity, hybrid, MMR, filters, HNSW (OpenAI) |
| [03_graph_vector.ipynb](notebooks/03_graph_vector.ipynb) | GraphRAG, QA chain (OpenAI) |

## Running Tests

```bash
pytest tests/unit/                # 65 tests, no DB required

export LANGCHAIN_AGE_TEST_DSN="host=localhost port=5433 dbname=langchain_age user=langchain password=langchain"
pytest tests/integration/         # 53 tests, requires Docker
```

## Compatibility

- **Python**: 3.10, 3.11, 3.12, 3.13, 3.14
- **PostgreSQL**: 18 + Apache AGE 1.7.0 + pgvector 0.8.2
- **LangChain**: v1 (`langchain-core>=1.0.0`)

### Known Limitations

- AGE does not support parameterised Cypher (`$param`) — `mogrify`-based escaping is provided
- Async methods use `run_in_executor` (not native `psycopg.AsyncConnection` yet)

## Contributing

Contributions welcome. Please open an issue first for major changes.

```bash
git clone https://github.com/BAEM1N/langchain-age.git
cd langchain-age
pip install -e ".[dev]"
pytest tests/unit/
ruff check langchain_age/ tests/
mypy langchain_age/
```

## License

MIT — see [LICENSE](LICENSE).

The vendored Apache AGE Python driver (`langchain_age/_vendor/age/`) is licensed under Apache 2.0.

---

<a id="한국어"></a>

# langchain-age (한국어)

[Apache AGE](https://age.apache.org/) (그래프) + [pgvector](https://github.com/pgvector/pgvector) (벡터)를 PostgreSQL 위에서 LangChain과 통합하는 라이브러리.

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

## 왜 Neo4j 대신 AGE인가?

| | Neo4j | Apache AGE |
|---|---|---|
| **인프라** | 별도 데이터베이스 | **기존 PostgreSQL에 확장** |
| **비용 (HA)** | 연 $15K+ (Enterprise) | **$0** (PG 네이티브 HA) |
| **라이선스** | GPL / 상용 | **Apache 2.0** |
| **벡터 검색** | Enterprise 기능 | **pgvector (무료, 같은 DB)** |
| **LangGraph 메모리** | 별도 DB 필요 | **같은 PostgreSQL** |

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

## 호환성

- **Python**: 3.10, 3.11, 3.12, 3.13, 3.14
- **PostgreSQL**: 18 + Apache AGE 1.7.0 + pgvector 0.8.2
- **LangChain**: v1 (`langchain-core>=1.0.0`)

## 라이선스

MIT — [LICENSE](LICENSE) 참조.

내장된 Apache AGE Python 드라이버 (`langchain_age/_vendor/age/`)는 Apache 2.0 라이선스입니다.
