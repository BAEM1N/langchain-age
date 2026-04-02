"""Unit tests for langchain_age utility functions."""
import pytest

from langchain_age.utils.agtype import agobj_to_dict, agtype_to_python, python_to_agtype
from langchain_age.utils.cypher import (
    extract_cypher_return_aliases,
    validate_cypher,
    wrap_cypher_query,
)


# ---------------------------------------------------------------------------
# agobj_to_dict  (SDK object → plain dict)
# ---------------------------------------------------------------------------

class TestAgobjToDict:
    def test_none(self):
        assert agobj_to_dict(None) is None

    def test_passthrough_scalar(self):
        assert agobj_to_dict(42) == 42
        assert agobj_to_dict("hello") == "hello"

    def test_dict_recursive(self):
        result = agobj_to_dict({"a": {"b": 1}})
        assert result == {"a": {"b": 1}}

    def test_list_recursive(self):
        result = agobj_to_dict([1, {"x": 2}])
        assert result == [1, {"x": 2}]


# ---------------------------------------------------------------------------
# agtype_to_python  (raw string fallback)
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
        assert agtype_to_python('{"key": "value"}') == {"key": "value"}

    def test_json_number_suffix(self):
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
        assert python_to_agtype({"a": 1}) == '{"a": 1}'


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
        assert "-" not in sql.split("cypher(")[1].split(",")[0]

    def test_default_column_when_empty(self):
        assert "(result agtype)" in wrap_cypher_query("g", "MATCH (n) RETURN n", [])

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
        assert "n" in extract_cypher_return_aliases("MATCH (n) RETURN n")

    def test_explicit_as_alias(self):
        aliases = extract_cypher_return_aliases("MATCH (n) RETURN n.name AS name, n.age AS age")
        assert "name" in aliases
        assert "age" in aliases

    def test_property_fallback(self):
        assert "name" in extract_cypher_return_aliases("MATCH (n) RETURN n.name")

    def test_no_return_clause(self):
        assert extract_cypher_return_aliases("CREATE (n:Person)") == ["result"]


# ---------------------------------------------------------------------------
# AGEVector filter clause builder
# ---------------------------------------------------------------------------

class TestFilterClause:
    def _build(self, f):
        from langchain_age.vectorstores.age_vector import AGEVector
        return AGEVector._build_filter_clause(f)

    def test_empty_filter(self):
        clause, params = self._build(None)
        assert clause == ""
        assert params == []

    def test_simple_equality(self):
        clause, params = self._build({"role": "admin"})
        assert "metadata->>%s = %s" in clause
        assert "role" in params
        assert "admin" in params

    def test_eq_operator(self):
        clause, params = self._build({"age": {"$eq": 30}})
        assert "=" in clause

    def test_lt_operator(self):
        clause, params = self._build({"score": {"$lt": 0.5}})
        assert "< %s" in clause

    def test_in_operator(self):
        clause, params = self._build({"status": {"$in": ["a", "b"]}})
        assert "ANY" in clause

    def test_between_operator(self):
        clause, params = self._build({"price": {"$between": [10, 100]}})
        assert "BETWEEN" in clause
        assert 10 in params
        assert 100 in params

    def test_like_operator(self):
        clause, params = self._build({"name": {"$like": "Al%"}})
        assert "LIKE" in clause

    def test_ilike_operator(self):
        clause, params = self._build({"name": {"$ilike": "al%"}})
        assert "ILIKE" in clause

    def test_exists_operator(self):
        clause, params = self._build({"field": {"$exists": True}})
        assert "?" in clause

    def test_and_operator(self):
        clause, params = self._build({"$and": [{"a": "1"}, {"b": "2"}]})
        assert "AND" in clause

    def test_or_operator(self):
        clause, params = self._build({"$or": [{"a": "1"}, {"b": "2"}]})
        assert "OR" in clause
