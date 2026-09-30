import pytest

from providers_verifier.core.compare import compare, golden_summary

TOOL = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
CASE = {"id": "t", "category": "tools", "request": {"tools": [TOOL]}, "expect": {"kind": "tool_call", "tool_names": ["get_weather"]}}


def golden_tool_call():
    return {"status": 200, "finish_reason": "tool_calls", "reasoning_present": True, "tool_calls": [{"name": "get_weather", "arguments": '{"city": "Beijing"}', "arguments_json_valid": True}]}


def test_identical_outcome_passes():
    summary = golden_summary([golden_tool_call(), golden_tool_call()])
    result = compare(CASE, summary, golden_tool_call())
    assert result["pass"], result["failed_checks"]


def test_missing_tool_call_and_leak_fail():
    summary = golden_summary([golden_tool_call()])
    target = {"status": 200, "finish_reason": "stop", "reasoning_present": True, "tool_calls": [], "content": "I will check</think>done"}
    result = compare(CASE, summary, target)
    assert not result["pass"]
    assert set(result["failed_checks"]) >= {"finish_reason", "tool_triggered", "no_think_leak"}


def test_schema_violation_fails():
    summary = golden_summary([golden_tool_call()])
    target = golden_tool_call()
    target["tool_calls"][0]["arguments"] = '{"town": "Beijing"}'
    result = compare(CASE, summary, target)
    assert "args_match_schema" in result["failed_checks"]


def test_majority_of_repeats_wins():
    flaky = golden_tool_call()
    flaky["tool_calls"] = []
    flaky["finish_reason"] = "stop"
    summary = golden_summary([golden_tool_call(), golden_tool_call(), flaky])
    assert summary["tool_triggered"] is True and summary["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("variation, failed_check", [("valid", None), ("missing", "tool_triggered"), ("wrong-name", "tool_names"), ("truncated", "finish_reason"), ("schema", "args_match_schema")])
def test_explicit_named_tool_overrides_inconsistent_golden(variation, failed_check):
    case = {**CASE, "expect": {**CASE["expect"], "expected_tool_call": True}}
    summary = golden_summary([{"status": 200, "finish_reason": None, "tool_calls": []}])
    target = golden_tool_call()
    if variation == "missing":
        target["tool_calls"] = []
    elif variation == "wrong-name":
        target["tool_calls"][0]["name"] = "other"
    elif variation == "truncated":
        target["finish_reason"] = "length"
    elif variation == "schema":
        target["tool_calls"][0]["arguments"] = '{}'
    result = compare(case, summary, target)
    assert result["pass"] is (failed_check is None)
    if failed_check:
        assert failed_check in result["failed_checks"]


def test_unlabelled_tool_names_still_follow_golden():
    case = {**CASE, "expect": {**CASE["expect"], "tool_names": ["other"]}}
    target = golden_tool_call()
    summary = golden_summary([target])
    assert compare(case, summary, target)["pass"]


def test_required_tool_without_declared_names_keeps_golden_finish():
    case = {**CASE, "expect": {"kind": "tool_call", "expected_tool_call": True}}
    target = {**golden_tool_call(), "finish_reason": "stop"}
    assert compare(case, golden_summary([target]), target)["pass"]
