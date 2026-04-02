"""Example 02 – Semantic vector search with AGEVector.

Prerequisites:
    1. Start the Docker container:
           cd docker && docker compose up -d
    2. pip install -e ".[dev]"
    3. Set OPENAI_API_KEY.

Usage:
    python examples/02_vector_search.py
"""
import os

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from langchain_age import AGEVector, DistanceStrategy

DSN = os.getenv(
    "LANGCHAIN_AGE_DSN",
    "host=localhost port=5432 dbname=langchain_age user=langchain password=langchain",
)


def main() -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 1. Create store and add documents
    store = AGEVector(
        connection_string=DSN,
        embedding_function=embeddings,
        collection_name="knowledge_base",
        distance_strategy=DistanceStrategy.COSINE,
        pre_delete_collection=True,
    )

    docs = [
        Document(page_content="Apache AGE extends PostgreSQL with graph capabilities using Cypher.", metadata={"source": "docs"}),
        Document(page_content="pgvector enables efficient vector similarity search inside PostgreSQL.", metadata={"source": "docs"}),
        Document(page_content="LangChain is a framework for building LLM-powered applications.", metadata={"source": "docs"}),
        Document(page_content="Graph databases store data as nodes and relationships.", metadata={"source": "docs"}),
        Document(page_content="Vector embeddings represent text as high-dimensional numerical vectors.", metadata={"source": "docs"}),
    ]
    store.add_documents(docs)
    print(f"Stored {len(docs)} documents.")

    # 2. Build an HNSW index for fast approximate search
    store.create_hnsw_index()
    print("HNSW index created.")

    # 3. Search
    queries = [
        "How does AGE relate to graph databases?",
        "What is vector similarity search?",
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        results = store.similarity_search_with_score(query, k=2)
        for doc, score in results:
            print(f"  [{score:.4f}] {doc.page_content[:80]}")


if __name__ == "__main__":
    main()
