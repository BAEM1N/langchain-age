"""Unit tests for langchain_age utility functions."""
import pytest

from langchain_age.utils.agtype import agtype_to_python, python_to_agtype
from langchain_age.utils.cypher import (
    extract_cypher_return_aliases,
    validate_cypher,
    wrap_cypher_query,
)


# ---------------------------------------------------------------------------
# agtype_to_python
# ---------------------------------------------------------------------------


class TestAgtypeToPython:
    def test_none(self):
        assert agtype_to_python(None) is None

    def test_int_passthrough(self):
        assert agtype_to_python(42) == 42

    def test_json_object_with_suffix(self):
        raw = '{"id": 1, "label": "Person", "properties": {"name": "Alice"}}::vertex'
        result = agtype_to_python(raw)
        assert isinstance(result, dict)
        assert result["properties"]["name"] == "Alice"

    def test_plain_json_object(self):
        raw = '{"key": "value"}'
        result = agtype_to_python(raw)
        assert result == {"key": "value"}

    def test_json_number(self):
        assert agtype_to_python("42::integer") == 42
        assert agtype_to_python("3.14::float") == pytest.approx(3.14)

    def test_json_boolean(self):
        assert agtype_to_python("true") is True
        assert agtype_to_python("false") is False

    def test_bare_string_fallback(self):
        assert agtype_to_python("hello world") == "hello world"


# ---------------------------------------------------------------------------
# python_to_agtype
# ---------------------------------------------------------------------------


class TestPythonToAgtype:
    def test_none(self):
        assert python_to_agtype(None) == "null"

    def test_bool_true(self):
        assert python_to_agtype(True) == "true"

    def test_bool_false(self):
        assert python_to_agtype(False) == "false"

    def test_integer(self):
        assert python_to_agtype(42) == "42"

    def test_float(self):
        assert python_to_agtype(3.14) == "3.14"

    def test_string(self):
        assert python_to_agtype("hello") == "'hello'"

    def test_string_with_single_quote(self):
        result = python_to_agtype("it's")
        assert "\\'" in result

    def test_dict(self):
        result = python_to_agtype({"a": 1})
        assert result == '{"a": 1}'


# ---------------------------------------------------------------------------
# wrap_cypher_query
# ---------------------------------------------------------------------------


class TestWrapCypherQuery:
    def test_basic_wrap(self):
        sql = wrap_cypher_query("mygraph", "MATCH (n) RETURN n", [("n", "agtype")])
        assert "SELECT * FROM cypher(" in sql
        assert "'mygraph'" in sql
        assert "$$ MATCH (n) RETURN n $$" in sql
        assert "AS (n agtype)" in sql

    def test_sanitises_graph_name(self):
        sql = wrap_cypher_query("my-graph!", "MATCH (n) RETURN n", [("n", "agtype")])
        assert "mygraph" in sql
        assert "-" not in sql.split("cypher(")[1].split(",")[0]

    def test_default_column_when_empty(self):
        sql = wrap_cypher_query("g", "MATCH (n) RETURN n", [])
        assert "(result agtype)" in sql

    def test_multiple_columns(self):
        sql = wrap_cypher_query("g", "MATCH (a)-[r]->(b) RETURN a, r, b", [
            ("a", "agtype"), ("r", "agtype"), ("b", "agtype")
        ])
        assert "(a agtype, r agtype, b agtype)" in sql


# ---------------------------------------------------------------------------
# validate_cypher
# ---------------------------------------------------------------------------


class TestValidateCypher:
    def test_valid_match(self):
        assert validate_cypher("MATCH (n) RETURN n") is None

    def test_empty_query(self):
        assert validate_cypher("") is not None
        assert validate_cypher("   ") is not None

    def test_no_cypher_keyword(self):
        assert validate_cypher("SELECT * FROM foo") is not None

    def test_dollar_quote_rejected(self):
        assert validate_cypher("MATCH (n) $$ RETURN n") is not None

    def test_create_valid(self):
        assert validate_cypher("CREATE (n:Person {name: 'Alice'})") is None


# ---------------------------------------------------------------------------
# extract_cypher_return_aliases
# ---------------------------------------------------------------------------


class TestExtractReturnAliases:
    def test_simple_alias(self):
        aliases = extract_cypher_return_aliases("MATCH (n) RETURN n")
        assert aliases == ["n"]

    def test_explicit_as_alias(self):
        aliases = extract_cypher_return_aliases("MATCH (n) RETURN n.name AS name, n.age AS age")
        assert "name" in aliases
        assert "age" in aliases

    def test_property_fallback(self):
        aliases = extract_cypher_return_aliases("MATCH (n) RETURN n.name")
        assert "name" in aliases

    def test_no_return_clause(self):
        aliases = extract_cypher_return_aliases("CREATE (n:Person)")
        assert aliases == ["result"]
