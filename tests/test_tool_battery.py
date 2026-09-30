import json
from copy import deepcopy

import jsonschema
import pytest

from providers_verifier.cases.tool_battery import DATA, _wrap_schema


@pytest.mark.parametrize(
    ("suite", "line", "valid", "invalid"),
    [
        ("TestAnyOf", 4, "x", True),
        ("TestDefs", 2, {}, {"extra": True}),
        ("TestEnforcerCases", 96, "x", False),
        ("TestID", 2, "x", 1),
        ("TestReferences", 11, {"game": {"black_id": 1}, "player": {"id": 1}}, {"game": {"black_id": "x"}}),
        ("TestReferences", 12, {"root": {"value": "x", "children": [{"value": "y"}]}}, {"root": {"value": "x", "children": [{}]}}),
        ("TestReferences", 13, {"node": {"value": "x", "next": {"value": "y"}}}, {"node": {"value": "x", "next": {}}}),
        ("TestReferences", 14, {"data": [{"next": ["x"]}]}, {"data": [{"next": [1]}]}),
        ("TestSingleTypeInArray", 11, {"reference": {"name": "x"}}, {"reference": {"name": 1}}),
    ],
)
def test_wrapped_walle_references_preserve_validity(suite, line, valid, invalid):
    schema = json.loads((DATA / "walle" / suite / "valid.jsonl").read_text().splitlines()[line - 1])
    original = deepcopy(schema)
    wrapped = _wrap_schema(schema)
    jsonschema.validate(valid, schema)
    jsonschema.validate({"value": valid}, wrapped)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"value": invalid}, wrapped)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({}, wrapped)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"value": valid, "extra": None}, wrapped)
    assert schema == original
    if "$defs" in schema:
        assert wrapped["$defs"] == schema["$defs"]
    if "$id" in schema:
        assert wrapped["$id"] == schema["$id"]


@pytest.mark.parametrize("schema_id", [None, "https://example.com/case.json"])
def test_recursive_root_and_local_reference_keep_the_original_value_schema(schema_id):
    schema = {
        "$defs": {"__case_schema": {"type": "integer"}},
        "anyOf": [{"$ref": "#/$defs/__case_schema"}, {"type": "array", "items": {"$ref": "#"}}],
    }
    if schema_id is not None:
        schema["$id"] = schema_id
    original = deepcopy(schema)
    wrapped = _wrap_schema(schema)
    for value in [1, [], [1, [2]]]:
        jsonschema.validate(value, schema)
        jsonschema.validate({"value": value}, wrapped)
    for value in ["x", [1, ["x"]]]:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(value, schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"value": value}, wrapped)
    assert schema == original
    assert wrapped["$defs"]["__case_schema"] == schema["$defs"]["__case_schema"]


@pytest.mark.parametrize("schema", [{"type": "object", "properties": {"x": {"type": "string"}}}, {"properties": {"x": {"type": "string"}}}])
def test_object_roots_keep_existing_request_shape(schema):
    assert _wrap_schema(schema) == {"type": "object", **schema}


@pytest.mark.parametrize("schema", [True, False, {"type": "string", "minLength": 2}])
def test_simple_and_boolean_roots_keep_existing_value_wrapper(schema):
    assert _wrap_schema(schema) == {"type": "object", "properties": {"value": schema}, "required": ["value"], "additionalProperties": False}


@pytest.mark.parametrize("keyword", ["const", "enum", "default", "examples"])
def test_literal_root_reference_is_not_rewritten(keyword):
    literal = {"$ref": "#"}
    schema = {keyword: [literal] if keyword in ("enum", "examples") else literal}
    wrapped = _wrap_schema(schema)
    assert wrapped["properties"]["value"] == schema
    jsonschema.validate({"value": literal}, wrapped)


def test_property_named_const_still_rewrites_its_schema():
    schema = {"anyOf": [{"type": "integer"}, {"type": "object", "properties": {"const": {"$ref": "#"}}, "required": ["const"]}]}
    value = {"const": {"const": 1}}
    jsonschema.validate(value, schema)
    jsonschema.validate({"value": value}, _wrap_schema(schema))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"value": {"const": "wrong"}}, _wrap_schema(schema))
