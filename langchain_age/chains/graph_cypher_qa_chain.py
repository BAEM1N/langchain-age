"""Graph QA chain that generates Cypher for Apache AGE.

Mirrors ``GraphCypherQAChain`` from *langchain-neo4j*, adopting its:
- ``allow_dangerous_requests`` safety gate
- ``return_intermediate_steps`` option
- ``return_direct`` option
- ``include_types`` / ``exclude_types`` schema filtering
- ``use_function_response`` tool-message mode
- ``validate_cypher`` option (validate generated query before execution)

Security note: The generated Cypher is executed against the database.
Only enable ``allow_dangerous_requests`` in trusted environments.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableSerializable

from langchain_age.graphs.age_graph import AGEGraph
from langchain_age.utils.cypher import validate_cypher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default prompts (mirrors Neo4j chain structure)
# ---------------------------------------------------------------------------

_CYPHER_GENERATION_SYSTEM = """You are an expert in Apache AGE graph database and Cypher query language.
Given a graph schema and a user question, generate a valid Cypher query to retrieve the answer.

IMPORTANT AGE-specific rules:
- Use standard openCypher syntax only — no APOC, no Neo4j-specific functions.
- Do NOT wrap the query in SQL (that is handled automatically).
- Always alias returned values: RETURN n.name AS name, not just RETURN n.name.
- Property names that are reserved words must be backtick-quoted: n.`desc`, n.`order`.
- Use MATCH, WHERE, RETURN, CREATE, MERGE, SET as normal Cypher clauses.

Graph schema:
{schema}

Generate ONLY the Cypher query — no explanation, no markdown fences."""

CYPHER_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _CYPHER_GENERATION_SYSTEM),
        ("human", "Question: {question}"),
    ]
)

_QA_SYSTEM = """You are an assistant that answers questions based on data from a graph database.
Use the provided context to answer concisely and accurately.
If the context is insufficient, say so.

Context from the graph:
{context}"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _QA_SYSTEM),
        ("human", "Question: {question}"),
    ]
)

_FUNCTION_RESPONSE_SYSTEM = (
    "You are an assistant that answers questions based on structured data "
    "returned from a graph database query.  The data is provided as a "
    "tool/function result.  Answer using only this data, concisely and accurately."
)


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


class AGEGraphCypherQAChain(RunnableSerializable):
    """QA chain over an Apache AGE graph using LLM-generated Cypher.

    Mirrors ``GraphCypherQAChain`` from *langchain-neo4j*.

    Pipeline:

    1. ``cypher_generation_chain`` → Cypher query string
    2. Optional ``validate_cypher`` check
    3. Execute Cypher against ``AGEGraph``
    4. ``qa_chain`` → natural-language answer

    Args:
        graph: Connected :class:`~langchain_age.graphs.AGEGraph`.
        cypher_generation_chain: Runnable producing a Cypher string.
        qa_chain: Runnable producing the final answer.
        input_key: Input dict key for the user question.
        output_key: Output dict key for the answer.
        top_k: Max DB rows to feed into the QA step.
        return_intermediate_steps: Include generated Cypher + raw DB results
            in the output dict.
        return_direct: Skip the QA step, return raw DB results as the answer.
        include_types: Whitelist of node/edge label types to expose in the
            schema string passed to the Cypher-generation prompt.
        exclude_types: Blacklist of node/edge label types to hide.
        use_function_response: Pass DB results as an LLM tool/function
            response instead of embedding them in the prompt body.
        validate_cypher: Run lightweight Cypher validation before executing;
            returns an error message instead of raising on invalid queries.
        allow_dangerous_requests: Safety gate — must be ``True`` to use.

    !!! warning "Security note"
        This chain generates and executes arbitrary Cypher queries.
        Use ``allow_dangerous_requests=True`` only in trusted environments,
        and consider restricting schema exposure via ``include_types``.
    """

    graph: AGEGraph
    cypher_generation_chain: Runnable
    qa_chain: Runnable
    input_key: str = "query"
    output_key: str = "result"
    top_k: int = 10
    verbose: bool = False
    return_intermediate_steps: bool = False
    return_direct: bool = False
    include_types: list[str] = []
    exclude_types: list[str] = []
    use_function_response: bool = False
    validate_cypher: bool = True
    allow_dangerous_requests: bool = False

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Factory (mirrors Neo4jGraph.from_llm)
    # ------------------------------------------------------------------

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        graph: AGEGraph,
        *,
        cypher_llm: BaseLanguageModel | None = None,
        qa_llm: BaseLanguageModel | None = None,
        cypher_prompt: BasePromptTemplate = CYPHER_GENERATION_PROMPT,
        qa_prompt: BasePromptTemplate = QA_PROMPT,
        include_types: list[str] | None = None,
        exclude_types: list[str] | None = None,
        validate_cypher: bool = True,
        allow_dangerous_requests: bool = False,
        **kwargs: Any,
    ) -> AGEGraphCypherQAChain:
        """Build the chain from a single LLM (or separate cypher / qa LLMs).

        Args:
            llm: Default LLM used for both Cypher generation and QA.
            graph: Connected :class:`~langchain_age.graphs.AGEGraph`.
            cypher_llm: Override LLM for the Cypher-generation step.
            qa_llm: Override LLM for the QA step.
            cypher_prompt: Prompt template for Cypher generation.
            qa_prompt: Prompt template for the QA step.
            include_types: Whitelist for schema exposure.
            exclude_types: Blacklist for schema exposure.
            validate_cypher: Enable lightweight pre-execution query validation.
            allow_dangerous_requests: Must be ``True`` to enable execution.
        """
        if not allow_dangerous_requests:
            raise ValueError(
                "AGEGraphCypherQAChain executes arbitrary Cypher against your "
                "database.  Set ``allow_dangerous_requests=True`` to confirm "
                "you understand the risk."
            )

        return cls(
            graph=graph,
            cypher_generation_chain=(
                cypher_prompt | (cypher_llm or llm) | StrOutputParser()
            ),
            qa_chain=qa_prompt | (qa_llm or llm) | StrOutputParser(),
            include_types=include_types or [],
            exclude_types=exclude_types or [],
            validate_cypher=validate_cypher,
            allow_dangerous_requests=True,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _get_schema(self) -> str:
        """Return schema filtered by include/exclude_types."""
        if not self.include_types and not self.exclude_types:
            return self.graph.get_schema

        ss = self.graph.structured_schema
        node_props = {
            k: v
            for k, v in ss.get("node_props", {}).items()
            if self._type_allowed(k)
        }
        rel_props = {
            k: v
            for k, v in ss.get("rel_props", {}).items()
            if self._type_allowed(k)
        }
        rels = [
            r
            for r in ss.get("relationships", [])
            if self._type_allowed(r["start"]) and self._type_allowed(r["end"])
        ]
        return AGEGraph._build_schema_string(node_props, rel_props, rels)

    def _type_allowed(self, label: str) -> bool:
        if self.include_types and label not in self.include_types:
            return False
        if self.exclude_types and label in self.exclude_types:
            return False
        return True

    def _call(
        self, inputs: dict[str, Any], run_manager: Any | None = None
    ) -> dict[str, Any]:
        if not self.allow_dangerous_requests:
            raise ValueError(
                "Set allow_dangerous_requests=True to use AGEGraphCypherQAChain."
            )

        question = inputs[self.input_key]
        callbacks = run_manager.get_child() if run_manager else None

        # Step 1 — Generate Cypher
        cypher_query: str = self.cypher_generation_chain.invoke(
            {"schema": self._get_schema(), "question": question},
            config={"callbacks": callbacks},
        ).strip()

        # Strip accidental markdown fences (mirrors Neo4j chain behaviour)
        if cypher_query.startswith("```"):
            cypher_query = "\n".join(
                line
                for line in cypher_query.splitlines()
                if not line.startswith("```")
            ).strip()

        logger.debug("Generated Cypher: %s", cypher_query)
        if run_manager:
            run_manager.on_text(
                "Generated Cypher:\n", end="", verbose=self.verbose
            )
            run_manager.on_text(
                cypher_query, color="green", end="\n", verbose=self.verbose
            )

        # Step 2 — Optional lightweight validation
        if self.validate_cypher:
            error = validate_cypher(cypher_query)
            if error:
                logger.warning("Generated Cypher failed validation: %s", error)
                return {
                    self.output_key: f"The generated Cypher query is invalid: {error}"
                }

        # Step 3 — Execute against AGE
        try:
            db_results = self.graph.query(cypher_query)[: self.top_k]
        except Exception as exc:
            # Log the full exception internally but return a generic message
            # to the caller — avoids leaking connection strings / schema details.
            logger.error(
                "Graph query failed for question %r: %s", question, exc, exc_info=True
            )
            return {
                self.output_key: (
                    "The graph query could not be completed. "
                    "Please rephrase your question or try again later."
                )
            }

        logger.debug("Graph results (%d rows): %s", len(db_results), db_results)
        if run_manager:
            run_manager.on_text(
                "Graph results:\n", end="", verbose=self.verbose
            )
            run_manager.on_text(
                str(db_results), color="yellow", end="\n", verbose=self.verbose
            )

        if self.return_direct:
            final: Any = str(db_results)
        elif self.use_function_response:
            final = self._function_response_answer(
                question, cypher_query, db_results, callbacks
            )
        else:
            final = self.qa_chain.invoke(
                {"context": db_results, "question": question},
                config={"callbacks": callbacks},
            )

        out: dict[str, Any] = {self.output_key: final}
        if self.return_intermediate_steps:
            out["intermediate_steps"] = [
                {"query": cypher_query},
                {"context": db_results},
            ]
        return out

    def _function_response_answer(
        self,
        question: str,
        cypher: str,
        db_results: list[dict[str, Any]],
        callbacks: Any,
    ) -> str:
        """Pass DB results as a tool/function message for models that support it."""
        import json

        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        tool_call_id = "graph_query_result"
        messages = [
            HumanMessage(content=question),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "graph_query",
                        "args": {"cypher": cypher},
                        "id": tool_call_id,
                    }
                ],
            ),
            ToolMessage(
                content=json.dumps(db_results), tool_call_id=tool_call_id
            ),
            HumanMessage(
                content=_FUNCTION_RESPONSE_SYSTEM + "\n\nAnswer the question above."
            ),
        ]
        response = self.qa_chain.invoke(messages, config={"callbacks": callbacks})
        return response if isinstance(response, str) else str(response)

    def invoke(
        self,
        input: dict[str, Any],
        config: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._call(input)

    def run(self, query: str, **kwargs: Any) -> str:
        """Convenience single-string interface."""
        return str(self.invoke({self.input_key: query})[self.output_key])
