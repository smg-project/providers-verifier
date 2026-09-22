"""Field-level comparison of a target outcome against the vendor's golden outcomes.

Exact keys: status class, finish_reason, whether a tool call was triggered, the
set of tool names, argument JSON validity and schema validity, reasoning presence,
JSON validity for structured cases, stream usage presence. Text is scored, never
asserted."""

from __future__ import annotations

import difflib
import json
from collections import Counter
from typing import Any

import jsonschema


def _status_class(status: int | None) -> str:
    if status is None:
        return "none"
    return f"{status // 100}xx"


def _tool_names(rec: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted({t.get("name") or "" for t in rec.get("tool_calls") or []}))


def _majority(values: list[Any]) -> Any:
    counted = Counter(json.dumps(v, sort_keys=True) for v in values)
    top = counted.most_common(1)[0][0]
    return json.loads(top)


def args_match_schema(rec: dict[str, Any], tools: list[dict[str, Any]]) -> bool | None:
    """True if every tool call's arguments validate against the declared schema."""
    calls = rec.get("tool_calls") or []
    if not calls:
        return None
    schemas = {t["function"]["name"]: t["function"].get("parameters") or {} for t in tools}
    for call in calls:
        schema = schemas.get(call.get("name"))
        if schema is None or not call.get("arguments_json_valid"):
            return False
        try:
            jsonschema.validate(json.loads(call["arguments"]), schema)
        except jsonschema.ValidationError:
            return False
    return True


def golden_summary(goldens: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse repeated vendor recordings into the majority behaviour."""
    # records with no HTTP status are client-side failures (SDK rejected a field, connection error);
    # they say nothing about the vendor, so they only count when nothing else exists
    answered = [g for g in goldens if g.get("status") is not None]
    goldens = answered or goldens
    ok = [g for g in goldens if g.get("status") == 200]
    base = ok or goldens
    return {
        "status_class": _majority([_status_class(g.get("status")) for g in goldens]),
        "finish_reason": _majority([g.get("finish_reason") for g in base]),
        "tool_triggered": _majority([bool(g.get("tool_calls")) for g in base]),
        "tool_names": _majority([list(_tool_names(g)) for g in base]),
        "reasoning_present": _majority([bool(g.get("reasoning_present")) for g in base]),
        "stream_usage_present": _majority([g.get("stream_usage_present") for g in base]),
        "prompt_tokens": _majority([(g.get("usage") or {}).get("prompt_tokens") for g in base]),
        "n": len(goldens),
        "n_success": len(ok),
    }


def compare(case: dict[str, Any], golden: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    expect = case.get("expect") or {}
    kind = expect.get("kind", "any")
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, ok: bool | None, want: Any, got: Any) -> None:
        checks[name] = {"ok": ok, "want": want, "got": got}

    check("status_class", _status_class(target.get("status")) == golden["status_class"], golden["status_class"], _status_class(target.get("status")))
    if target.get("status") == 200 and golden["status_class"] == "2xx":
        check("finish_reason", target.get("finish_reason") == golden["finish_reason"], golden["finish_reason"], target.get("finish_reason"))
        want_trigger = expect["expected_tool_call"] if expect.get("expected_tool_call") is not None else golden["tool_triggered"]
        check("tool_triggered", bool(target.get("tool_calls")) == want_trigger, want_trigger, bool(target.get("tool_calls")))
        if want_trigger:
            check("tool_names", list(_tool_names(target)) == golden["tool_names"], golden["tool_names"], list(_tool_names(target)))
            tools = case.get("request", {}).get("tools") or []
            check("args_match_schema", args_match_schema(target, tools), True, args_match_schema(target, tools))
        if kind in ("text", "thinking") or expect.get("reasoning_present") is not None:
            check("reasoning_present", bool(target.get("reasoning_present")) == golden["reasoning_present"], golden["reasoning_present"], bool(target.get("reasoning_present")))
        if kind == "json":
            content = target.get("content") or ""
            valid = True
            try:
                obj = json.loads(content)
                if expect.get("schema"):
                    jsonschema.validate(obj, expect["schema"])
            except Exception:  # noqa: BLE001
                valid = False
            check("json_valid", valid, True, valid)
        if expect.get("stream_usage"):
            check("stream_usage_present", bool(target.get("stream_usage_present")) == bool(golden["stream_usage_present"]), golden["stream_usage_present"], target.get("stream_usage_present"))
        if expect.get("content_contains"):
            check("content_contains", expect["content_contains"] in (target.get("content") or ""), expect["content_contains"], (target.get("content") or "")[:80])
        if expect.get("content_contains_any"):
            got = (target.get("content") or "").lower()
            check("content_contains_any", any(w in got for w in expect["content_contains_any"]), expect["content_contains_any"], got[:80])
        if expect.get("content_order"):
            got = target.get("content") or ""
            positions = [(got.find(k), k) for k in expect["content_order"] if got.find(k) != -1]
            order = [k for _, k in sorted(positions)]
            check("content_order", order == expect["content_order"], expect["content_order"], order)
        if expect.get("prompt_tokens_exact") and golden.get("prompt_tokens") is not None:
            got_pt = (target.get("usage") or {}).get("prompt_tokens")
            check("prompt_tokens_match", got_pt == golden["prompt_tokens"], golden["prompt_tokens"], got_pt)
        if "contains_russian_characters_unicode" in (expect.get("check_type") or []):
            import re as _re

            has_ru = bool(_re.search(r"[\u0400-\u04FF]", target.get("content") or ""))
            check("language_following", not has_ru, True, not has_ru)
        # leak detector: think markers never belong in content
        leak = "</think>" in (target.get("content") or "") or "<think>" in (target.get("content") or "")
        check("no_think_leak", not leak, True, not leak)

    hard = [k for k, v in checks.items() if v["ok"] is False]
    return {
        "case": case["id"],
        "category": case["category"],
        "expected_tool_call": expect.get("expected_tool_call"),
        "pass": not hard,
        "failed_checks": hard,
        "checks": checks,
        "text_similarity": round(difflib.SequenceMatcher(None, (golden.get("content_sample") or ""), (target.get("content") or "")).ratio(), 3) if golden.get("content_sample") else None,
    }
