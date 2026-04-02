"""Graph QA chain that generates Cypher for Apache AGE."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableSerializable

from langchain_age.graphs.age_graph import AGEGraph
from langchain_age.utils.cypher import validate_cypher


# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------

_CYPHER_GENERATION_SYSTEM = """You are an expert in Apache AGE graph database and Cypher query language.
Given a graph schema and a user question, generate a valid Cypher query to retrieve the answer.

IMPORTANT AGE-specific rules:
- Use standard openCypher syntax only.
- Do NOT wrap the query in SQL (AGEGraphCypherQAChain handles the SQL wrapping automatically).
- Use MATCH, RETURN, WHERE, CREATE, MERGE, etc. as normal Cypher clauses.
- Property access uses dot notation: n.property_name
- Always alias returned values explicitly: RETURN n.name AS name
- Do NOT use APOC procedures (not available in AGE).
- Do NOT use Neo4j-specific functions.

Graph schema:
{schema}

Generate only the Cypher query — no explanation, no markdown fences."""

_CYPHER_GENERATION_HUMAN = "Question: {question}"

CYPHER_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _CYPHER_GENERATION_SYSTEM),
        ("human", _CYPHER_GENERATION_HUMAN),
    ]
)

_QA_SYSTEM = """You are an assistant that answers questions based on data retrieved from a graph database.
Use the provided context to answer the question concisely and accurately.
If the context does not contain enough information, say so honestly.

Context from the graph:
{context}"""

_QA_HUMAN = "Question: {question}"

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _QA_SYSTEM),
        ("human", _QA_HUMAN),
    ]
)


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


class AGEGraphCypherQAChain(RunnableSerializable):
    """Question-answering chain over an Apache AGE graph using Cypher.

    Mirrors ``GraphCypherQAChain`` from *langchain-neo4j* but targets AGE.

    Pipeline:
        1. ``cypher_generation_chain``: ``(schema, question) → Cypher query``
        2. Execute Cypher against ``AGEGraph``.
        3. ``qa_chain``: ``(context, question) → natural-language answer``

    Args:
        graph: Connected :class:`~langchain_age.graphs.AGEGraph` instance.
        cypher_generation_chain: Runnable that produces a Cypher string.
        qa_chain: Runnable that produces the final answer string.
        input_key: Name of the input key carrying the user question.
        output_key: Name of the output key carrying the answer.
        top_k: Maximum number of rows to retrieve from the graph.
        return_intermediate_steps: If ``True``, include ``intermediate_steps``
            in the output dict (contains generated Cypher + raw DB results).
        return_direct: Skip the QA chain and return raw DB results directly.
        allow_dangerous_requests: Must be set to ``True`` to enable chain
            execution (safety gate, same as Neo4j chain).
    """

    graph: AGEGraph
    cypher_generation_chain: Runnable
    qa_chain: Runnable
    input_key: str = "query"
    output_key: str = "result"
    top_k: int = 10
    return_intermediate_steps: bool = False
    return_direct: bool = False
    allow_dangerous_requests: bool = False

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        graph: AGEGraph,
        *,
        cypher_llm: Optional[BaseLanguageModel] = None,
        qa_llm: Optional[BaseLanguageModel] = None,
        cypher_prompt: BasePromptTemplate = CYPHER_GENERATION_PROMPT,
        qa_prompt: BasePromptTemplate = QA_PROMPT,
        allow_dangerous_requests: bool = False,
        **kwargs: Any,
    ) -> AGEGraphCypherQAChain:
        """Convenience constructor that builds the internal chains from an LLM.

        Args:
            llm: Default LLM used for both Cypher generation and QA.
            graph: Connected :class:`~langchain_age.graphs.AGEGraph`.
            cypher_llm: Override LLM for Cypher generation.
            qa_llm: Override LLM for QA step.
            cypher_prompt: Prompt template for Cypher generation.
            qa_prompt: Prompt template for the QA step.
            allow_dangerous_requests: Safety gate – must be ``True``.
            **kwargs: Forwarded to the chain constructor.
        """
        if not allow_dangerous_requests:
            raise ValueError(
                "AGEGraphCypherQAChain can execute arbitrary Cypher queries against "
                "your database.  Set ``allow_dangerous_requests=True`` to confirm "
                "that you have verified the security implications."
            )

        cypher_generation_chain = cypher_prompt | (cypher_llm or llm) | StrOutputParser()
        qa_chain = qa_prompt | (qa_llm or llm) | StrOutputParser()

        return cls(
            graph=graph,
            cypher_generation_chain=cypher_generation_chain,
            qa_chain=qa_chain,
            allow_dangerous_requests=allow_dangerous_requests,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if not self.allow_dangerous_requests:
            raise ValueError(
                "Set ``allow_dangerous_requests=True`` to use AGEGraphCypherQAChain."
            )

        question = inputs[self.input_key]
        callbacks = run_manager.get_child() if run_manager else None

        # Step 1 – Generate Cypher
        cypher_query: str = self.cypher_generation_chain.invoke(
            {"schema": self.graph.get_schema, "question": question},
            config={"callbacks": callbacks},
        )
        cypher_query = cypher_query.strip()

        # Strip accidental markdown fences
        if cypher_query.startswith("```"):
            lines = cypher_query.splitlines()
            cypher_query = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        if run_manager:
            run_manager.on_text("Generated Cypher:", end="\n", verbose=self.verbose)
            run_manager.on_text(cypher_query, color="green", end="\n", verbose=self.verbose)

        # Validate before hitting the DB
        validation_error = validate_cypher(cypher_query)
        if validation_error:
            error_msg = f"Generated Cypher is invalid: {validation_error}"
            if run_manager:
                run_manager.on_text(error_msg, color="red", end="\n", verbose=self.verbose)
            return {self.output_key: error_msg}

        # Step 2 – Execute against AGE
        try:
            db_results = self.graph.query(cypher_query)
            db_results = db_results[: self.top_k]
        except Exception as exc:
            error_msg = f"Graph query failed: {exc}"
            if run_manager:
                run_manager.on_text(error_msg, color="red", end="\n", verbose=self.verbose)
            return {self.output_key: error_msg}

        if run_manager:
            run_manager.on_text("Graph results:", end="\n", verbose=self.verbose)
            run_manager.on_text(str(db_results), color="yellow", end="\n", verbose=self.verbose)

        if self.return_direct:
            final_answer = str(db_results)
        else:
            # Step 3 – Generate natural-language answer
            final_answer = self.qa_chain.invoke(
                {"context": db_results, "question": question},
                config={"callbacks": callbacks},
            )

        output: Dict[str, Any] = {self.output_key: final_answer}
        if self.return_intermediate_steps:
            output["intermediate_steps"] = [
                {"query": cypher_query},
                {"context": db_results},
            ]
        return output

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute the chain (Runnable interface)."""
        return self._call(input)

    def run(self, query: str, **kwargs: Any) -> str:
        """Convenience method for single-string input/output."""
        return self.invoke({self.input_key: query})[self.output_key]
