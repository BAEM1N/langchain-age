"""Example 01 – Graph QA with AGEGraphCypherQAChain.

Prerequisites:
    1. Start the Docker container:
           cd docker && docker compose up -d
    2. Install dependencies:
           pip install -e ".[dev]"
    3. Set OPENAI_API_KEY (or swap the LLM below for any LangChain-compatible model).

Usage:
    python examples/01_basic_graph_qa.py
"""
import os

from langchain_openai import ChatOpenAI

from langchain_age import AGEGraph, AGEGraphCypherQAChain

DSN = os.getenv(
    "LANGCHAIN_AGE_DSN",
    "host=localhost port=5432 dbname=langchain_age user=langchain password=langchain",
)
GRAPH = "movies"


def main() -> None:
    # 1. Connect to AGE and seed some data
    graph = AGEGraph(DSN, GRAPH)

    graph.query("MERGE (:Person {name: 'Tom Hanks'})")
    graph.query("MERGE (:Movie {title: 'Forrest Gump', year: 1994})")
    graph.query(
        "MATCH (p:Person {name: 'Tom Hanks'}), (m:Movie {title: 'Forrest Gump'}) "
        "MERGE (p)-[:ACTED_IN]->(m)"
    )
    graph.query("MERGE (:Movie {title: 'Cast Away', year: 2000})")
    graph.query(
        "MATCH (p:Person {name: 'Tom Hanks'}), (m:Movie {title: 'Cast Away'}) "
        "MERGE (p)-[:ACTED_IN]->(m)"
    )

    graph.refresh_schema()
    print("Schema:\n", graph.schema)

    # 2. Build the QA chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = AGEGraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        verbose=True,
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
    )

    # 3. Ask questions
    questions = [
        "Which movies did Tom Hanks act in?",
        "How many movies are in the database?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        result = chain.invoke({"query": q})
        print(f"A: {result['result']}")
        steps = result.get("intermediate_steps", [])
        if steps:
            print(f"   Cypher: {steps[0].get('query')}")


if __name__ == "__main__":
    main()
