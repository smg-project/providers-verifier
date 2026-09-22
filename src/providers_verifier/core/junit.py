"""Minimal JUnit XML writer: one testsuite per category, one testcase per case
(majority over repeats), failure text listing the failed checks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


def write_junit(path: str | Path, per_case: dict[str, dict[str, Any]], target: str) -> None:
    suites: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for case_id, agg in per_case.items():
        suites[agg["category"]].append((case_id, agg))
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', f'<testsuites name="providers-verifier {escape(target)}">']
    for category, items in sorted(suites.items()):
        failures = sum(1 for _, a in items if not a["pass"])
        lines.append(f'  <testsuite name="{escape(category)}" tests="{len(items)}" failures="{failures}">')
        for case_id, a in sorted(items):
            lines.append(f'    <testcase classname="{escape(category)}" name="{escape(case_id)}" time="{a.get("mean_latency_s", 0):.3f}">')
            if not a["pass"]:
                detail = "; ".join(f"{k}: want {v['want']!r} got {v['got']!r}" for k, v in a["worst_checks"].items())
                lines.append(f'      <failure message="{escape(", ".join(a["failed_checks"]))}">{escape(detail)}</failure>')
            lines.append("    </testcase>")
        lines.append("  </testsuite>")
    lines.append("</testsuites>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
