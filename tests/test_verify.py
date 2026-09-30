import json
import sys
import xml.etree.ElementTree as ET

import pytest

from providers_verifier import verify
from providers_verifier.cases import Case


@pytest.fixture
def run_verify(tmp_path, monkeypatch):
    record = {"status": 200, "finish_reason": "stop", "content": "ok", "tool_calls": [], "reasoning_present": False, "latency_ms": 1}
    golden_dir = tmp_path / "golden" / "zai" / "fixture"
    golden_dir.mkdir(parents=True)
    monkeypatch.setattr(verify, "GOLDEN", tmp_path / "golden")

    def run(selected, recorded, *, empty=(), repeats=1, status=200):
        for case_id in recorded:
            (golden_dir / f"{case_id}.jsonl").write_text(json.dumps(record) + "\n")
        for case_id in empty:
            (golden_dir / f"{case_id}.jsonl").write_text("\n")
        cases = [Case(case_id, "text", {"messages": [{"role": "user", "content": "hi"}]}, {"kind": "text"}) for case_id in selected]
        monkeypatch.setattr(verify, "cases_for", lambda *args: cases)
        calls = []

        def call(*args):
            calls.append(args)
            return {**record, "status": status}

        monkeypatch.setattr(verify, "call_openai_style", call)
        report = tmp_path / "report.json"
        junit = tmp_path / "junit.xml"
        monkeypatch.setattr(sys, "argv", [
            "pv-verify", "--target", "http://127.0.0.1:1/v1", "--model", "fixture", "--golden-model", "fixture",
            "--repeats", str(repeats), "--report", str(report), "--junit", str(junit),
        ])
        code = 0
        try:
            verify.main()
        except SystemExit as exc:
            code = exc.code
        return code, json.loads(report.read_text()), ET.parse(junit).getroot(), calls

    return run


@pytest.mark.parametrize(
    ("selected", "recorded", "empty", "repeats", "executed", "missing"),
    [
        (["one"], [], [], 1, [], ["one"]),
        (["one"], [], ["one"], 1, [], ["one"]),
        (["one", "two"], ["one"], [], 1, ["one"], ["two"]),
        ([], [], [], 1, [], []),
        (["one"], ["one"], [], 0, [], []),
    ],
    ids=["missing-golden", "empty-golden", "partial-coverage", "empty-selection", "zero-repeats"],
)
def test_incomplete_coverage_fails_with_reports(run_verify, selected, recorded, empty, repeats, executed, missing):
    code, report, junit, calls = run_verify(selected, recorded, empty=empty, repeats=repeats)
    assert code == 1
    assert report["coverage"] == {"selected_cases": selected, "executed_cases": executed, "missing_golden": missing, "complete": False}
    assert sorted(report["per_case"]) == executed
    assert len(calls) == len(executed)
    assert len(report["results"]) == len(executed)
    errors = junit.findall("./testsuite/testcase/error")
    assert len(errors) == 1
    assert "coverage" in errors[0].text.lower()
    assert all(case_id in errors[0].text for case_id in missing)
    if executed:
        assert report["per_case"]["one"]["pass"] is True
        assert junit.find("./testsuite[@name='text']/testcase[@name='one']") is not None


@pytest.mark.parametrize("status, expected_code", [(200, 0), (500, 1)])
def test_complete_coverage_preserves_threshold_verdict(run_verify, status, expected_code):
    code, report, junit, calls = run_verify(["one"], ["one"], status=status)
    assert code == expected_code
    assert report["coverage"] == {"selected_cases": ["one"], "executed_cases": ["one"], "missing_golden": [], "complete": True}
    assert len(calls) == 1
    assert report["metrics"]["query_success_rate"] == (1.0 if status == 200 else 0.0)
    assert bool(report["metrics"]["threshold_failures"]) == (status != 200)
    assert junit.findall("./testsuite/testcase/error") == []
    assert len(junit.findall("./testsuite/testcase/failure")) == (0 if status == 200 else 1)
