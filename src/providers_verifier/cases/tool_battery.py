"""Tool-call battery: MiniMax's 102 labelled prompts and Moonshot's 212 walle
JSON-Schema cases, built the way the two vendor kits build them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from providers_verifier.cases import Case

DATA = Path(__file__).resolve().parents[3] / "data"

WALLE_TOOL_NAME = "submit"
WALLE_PROMPT = (
    f"Call the {WALLE_TOOL_NAME} tool exactly once with minimum runtime arguments that satisfy "
    "its parameter schema, try your best to create the arguments. Do not copy or describe the "
    "JSON Schema itself. Do not include schema keywords like type, properties, required, or "
    "additionalProperties unless the schema explicitly requires them as argument property names. "
    "If the schema defines a top-level value argument, provide the minimal valid value for it. "
    "Always include every required property. Respect minItems, minProperties, enum, const, "
    "minimum, and minLength constraints. Prefer empty arrays and empty objects only when those "
    "constraints allow them. Do not answer with plain text."
)


def minimax_cases(max_tokens: int = 4096) -> list[Case]:
    cases = []
    for i, line in enumerate((DATA / "minimax" / "sample.jsonl").read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        request: dict[str, Any] = {"messages": row["messages"], "max_tokens": max_tokens}
        if row.get("tools"):
            request["tools"] = row["tools"]
        for k in ("temperature", "top_p"):
            if k in row:
                request[k] = row[k]
        expected = row.get("expected_tool_call")
        expect: dict[str, Any] = {"kind": "tool_call" if expected else ("text" if expected is False else "any")}
        if expected is not None:
            expect["expected_tool_call"] = bool(expected)
        if row.get("check_type"):
            expect["check_type"] = row["check_type"]
        cases.append(Case(f"mm_{i:03d}", "tool_battery", request, expect, "MiniMax-Provider-Verifier sample.jsonl"))
    return cases


# The reference-preserving wrapper helpers are adapted from Moonshot AI's
# Kimi-Vendor-Verifier, tests/tool_call_json_schema/validator.py at
# 66092cf444c97356c0e11c5078c67116390615d9 (MIT, Copyright (c) 2026 Moonshot AI).
# License: data/walle/LICENSE-Kimi-Vendor-Verifier.
def _rewrite_root_refs(obj: Any, target_ref: str, *, schema_map: bool = False) -> Any:
    """Rewrite JSON Schema root self-references to *target_ref*."""
    if isinstance(obj, dict):
        rewritten: dict[str, Any] = {}
        for key, value in obj.items():
            if schema_map:
                rewritten[key] = _rewrite_root_refs(value, target_ref)
            elif key == "$ref" and value == "#":
                rewritten[key] = target_ref
            elif key in ("const", "enum", "default", "examples"):
                rewritten[key] = value
            else:
                rewritten[key] = _rewrite_root_refs(value, target_ref, schema_map=key in ("$defs", "definitions", "properties", "patternProperties", "dependentSchemas"))
        return rewritten
    if isinstance(obj, list):
        return [_rewrite_root_refs(item, target_ref) for item in obj]
    return obj


def _wrap_schema(schema: Any) -> dict[str, Any]:
    """Tool `parameters` must be an object schema; other roots become a required `value` property."""
    if isinstance(schema, dict) and schema.get("type") == "object":
        return schema
    if isinstance(schema, dict) and "properties" in schema and "type" not in schema:
        return {"type": "object", **schema}
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {"value": schema}, "required": ["value"], "additionalProperties": False}

    wrapped: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": ["value"],
        "additionalProperties": False,
    }
    defs = dict(schema.get("$defs", {}))
    def_name = "__case_schema"
    while def_name in defs:
        def_name = f"_{def_name}"
    target_ref = f"#/$defs/{def_name}"
    rewritten = _rewrite_root_refs(schema, target_ref)
    if rewritten != schema:
        defs.update(rewritten.get("$defs", {}))
        defs[def_name] = {k: v for k, v in rewritten.items() if k not in ("$defs", "$id")}
        wrapped["properties"] = {"value": {"$ref": target_ref}}
        wrapped["$defs"] = defs
    else:
        wrapped["properties"] = {"value": {k: v for k, v in schema.items() if k not in ("$defs", "$id")}}
        if "$defs" in schema:
            wrapped["$defs"] = schema["$defs"]

    if "$id" in schema:
        wrapped["$id"] = schema["$id"]
    return wrapped


def walle_cases(max_tokens: int = 2048, modes: tuple[str, ...] = ("nonstream", "stream")) -> list[Case]:
    cases = []
    for suite_dir in sorted((DATA / "walle").iterdir()):
        f = suite_dir / "valid.jsonl"
        if not f.is_file():
            continue
        for line_no, line in enumerate(f.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            schema = json.loads(line)
            params = _wrap_schema(schema)
            for mode in modes:
                request: dict[str, Any] = {
                    "messages": [{"role": "user", "content": WALLE_PROMPT}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": WALLE_TOOL_NAME, "description": "Submit minimal JSON arguments that validate against this JSON Schema.", "parameters": params, "strict": True},
                        }
                    ],
                    "tool_choice": {"type": "function", "function": {"name": WALLE_TOOL_NAME}},
                    "max_tokens": max_tokens,
                }
                if mode == "stream":
                    request["stream"] = True
                cases.append(
                    Case(
                        f"walle_{suite_dir.name}_{line_no:03d}_{mode}",
                        "tool_schema",
                        request,
                        {"kind": "tool_call", "tool_names": [WALLE_TOOL_NAME], "expected_tool_call": True, "schema_source": "tool"},
                        f"walle {suite_dir.name}:{line_no}",
                    )
                )
    return cases
