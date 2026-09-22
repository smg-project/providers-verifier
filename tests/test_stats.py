from providers_verifier.core.junit import write_junit
from providers_verifier.core.stats import aggregate_cases, metrics, threshold_failures


def result(case, ok, tool=None, expected=None, checks=None, status=200, reasoning=False, content="x", category="tools"):
    checks = checks or {}
    return {
        "case": case,
        "category": category,
        "pass": ok,
        "expected_tool_call": expected,
        "failed_checks": [k for k, v in checks.items() if v["ok"] is False],
        "checks": checks,
        "target_record": {"status": status, "tool_calls": tool, "reasoning_present": reasoning, "content": content, "latency_ms": 100},
        "golden_summary": {"status_class": "2xx", "tool_triggered": bool(tool)},
    }


def test_majority_over_repeats():
    rs = [result("a", True), result("a", False, checks={"json_valid": {"ok": False, "want": True, "got": False}}), result("a", True)]
    agg = aggregate_cases(rs)
    assert agg["a"]["pass"] is True and agg["a"]["passes"] == 2 and agg["a"]["failed_checks"] == ["json_valid"]


def test_confusion_matrix_and_f1():
    tc = [{"name": "f", "arguments": "{}", "arguments_json_valid": True}]
    ok_schema = {"args_match_schema": {"ok": True, "want": True, "got": True}}
    rs = [
        result("tp", True, tool=tc, expected=True, checks=ok_schema),
        result("fn", False, tool=None, expected=True),
        result("fp", False, tool=tc, expected=False),
        result("tn", True, tool=None, expected=False),
    ]
    m = metrics(rs)
    assert m["tool_confusion"] == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}
    assert m["tool_trigger_match_rate"] == 0.5 and m["tool_trigger_f1"] == 0.5 and m["tool_schema_accuracy"] == 1.0
    assert "tool_trigger_f1=0.5 < 0.98" in m["threshold_failures"]


def test_error_only_reasoning_counts_empty_answers():
    rs = [result("a", True, reasoning=True, content=""), result("b", True, reasoning=True, content="ok")]
    assert metrics(rs)["error_only_reasoning_rate"] == 0.5
    assert threshold_failures({"error_only_reasoning_rate": 0.5}) == ["error_only_reasoning_rate=0.5 > 0.0"]


def test_junit_written(tmp_path):
    rs = [result("a", False, checks={"json_valid": {"ok": False, "want": True, "got": "<bad>"}}, category="structured")]
    path = tmp_path / "junit.xml"
    write_junit(path, aggregate_cases(rs), "http://t/v1")
    text = path.read_text()
    assert '<testsuite name="structured" tests="1" failures="1">' in text and "&lt;bad&gt;" in text
