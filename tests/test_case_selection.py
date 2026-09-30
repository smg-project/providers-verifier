from types import SimpleNamespace

import pytest

from providers_verifier import cases


@pytest.mark.parametrize("categories", [None, ["params", "tool_schema"]])
def test_diagnostic_cases_require_explicit_ids(monkeypatch, categories):
    diagnostic = [
        cases.Case("param_temperature_negative", "params", {"temperature": -0.5}),
        cases.Case("walle_TestRangeConstraints_003_nonstream", "tool_schema", {"minLength": 999999999999}),
        cases.Case("walle_TestRangeConstraints_003_stream", "tool_schema", {"minLength": 999999999999, "stream": True}),
    ]
    control = cases.Case("param_temperature_above_range", "params", {"temperature": 3})
    monkeypatch.setattr(cases, "all_cases", lambda: [control, *diagnostic])
    vendor = SimpleNamespace(extra_cases=list, supports={})
    assert cases.cases_for(vendor, categories) == [control]
    assert cases.cases_for(vendor, categories, [c.id for c in diagnostic]) == diagnostic
    assert cases.cases_for(vendor, categories, [control.id, diagnostic[0].id]) == [control, diagnostic[0]]
