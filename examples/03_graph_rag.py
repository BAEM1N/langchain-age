"""Example 03 – Graph-enhanced RAG combining AGEGraph + AGEVector.

Demonstrates a simple GraphRAG pattern:
  1. Extract entities from documents and store them in the AGE graph.
  2. Embed document chunks into AGEVector with references to graph node IDs.
  3. At query time, retrieve similar chunks, then expand context via graph traversal.

Prerequisites:
    1. Start Docker: cd docker && docker compose up -d
    2. pip install -e ".[dev]"
    3. Set OPENAI_API_KEY.

Usage:
    python examples/03_graph_rag.py
"""
import os

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_age import AGEGraph, AGEGraphCypherQAChain, AGEVector, DistanceStrategy

DSN = os.getenv(
    "LANGCHAIN_AGE_DSN",
    "host=localhost port=5432 dbname=langchain_age user=langchain password=langchain",
)
GRAPH = "graphrag_demo"


def main() -> None:
    graph = AGEGraph(DSN, GRAPH)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = AGEVector(
        connection_string=DSN,
        embedding_function=embeddings,
        collection_name="graphrag_chunks",
        distance_strategy=DistanceStrategy.COSINE,
        pre_delete_collection=True,
        age_graph_name=GRAPH,
    )

    # --- Seed graph data -------------------------------------------------------
    entities = [
        ("Company", "OpenAI", {"founded": "2015"}),
        ("Company", "Anthropic", {"founded": "2021"}),
        ("Person", "Sam Altman", {}),
        ("Person", "Dario Amodei", {}),
        ("Product", "ChatGPT", {}),
        ("Product", "Claude", {}),
    ]
    for label, name, props in entities:
        prop_str = ", ".join(f"{k}: '{v}'" for k, v in props.items())
        if prop_str:
            graph.query(f"MERGE (n:{label} {{name: '{name}', {prop_str}}})")
        else:
            graph.query(f"MERGE (n:{label} {{name: '{name}'}})")

    relations = [
        ("Person", "Sam Altman", "LEADS", "Company", "OpenAI"),
        ("Person", "Dario Amodei", "LEADS", "Company", "Anthropic"),
        ("Company", "OpenAI", "CREATED", "Product", "ChatGPT"),
        ("Company", "Anthropic", "CREATED", "Product", "Claude"),
    ]
    for src_label, src_name, rel_type, tgt_label, tgt_name in relations:
        graph.query(
            f"MATCH (a:{src_label} {{name: '{src_name}'}}), "
            f"(b:{tgt_label} {{name: '{tgt_name}'}}) "
            f"MERGE (a)-[:{rel_type}]->(b)"
        )

    # --- Embed document chunks -------------------------------------------------
    chunks = [
        Document(
            page_content="OpenAI was founded in 2015 and created ChatGPT, a widely-used AI assistant.",
            metadata={"age_node_id": "OpenAI"},
        ),
        Document(
            page_content="Anthropic, founded in 2021 by Dario Amodei, developed the Claude AI assistant.",
            metadata={"age_node_id": "Anthropic"},
        ),
        Document(
            page_content="Sam Altman leads OpenAI as its CEO and is a prominent figure in AI.",
            metadata={"age_node_id": "Sam Altman"},
        ),
    ]
    vector_store.add_documents(chunks)

    # --- Retrieve + Graph expand -----------------------------------------------
    query = "Who leads the company that created Claude?"
    print(f"\nQuery: {query}")

    similar_chunks = vector_store.similarity_search(query, k=2)
    print("\nRelevant chunks:")
    for chunk in similar_chunks:
        print(f"  [{chunk.metadata.get('age_node_id', '?')}] {chunk.page_content[:80]}")

    # Expand via graph
    entity = similar_chunks[0].metadata.get("age_node_id")
    if entity:
        expansions = graph.query(
            f"MATCH (n {{name: '{entity}'}})-[r]-(connected) "
            f"RETURN n.name AS entity, type(r) AS relation, connected.name AS connected"
        )
        print(f"\nGraph context for '{entity}':")
        for row in expansions:
            print(f"  {row.get('entity')} -[{row.get('relation')}]-> {row.get('connected')}")

    # --- Graph QA chain --------------------------------------------------------
    graph.refresh_schema()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_chain = AGEGraphCypherQAChain.from_llm(
        llm, graph=graph, verbose=True, allow_dangerous_requests=True
    )
    answer = qa_chain.run(query)
    print(f"\nFinal answer: {answer}")


if __name__ == "__main__":
    main()
