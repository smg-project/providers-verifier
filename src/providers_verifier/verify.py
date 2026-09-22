"""Replay the cases against a target OpenAI-compatible endpoint, compare each
answer with the vendor golden set, and print MiniMax-style metrics. Writes a
JSON report and optionally JUnit XML. Exit code 1 when a threshold fails."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import OpenAI

from providers_verifier.cases import Case, cases_for
from providers_verifier.core.compare import compare, golden_summary
from providers_verifier.core.junit import write_junit
from providers_verifier.core.stats import THRESHOLDS, aggregate_cases, metrics
from providers_verifier.record import GOLDEN, call_openai_style
from providers_verifier.vendors import VENDORS, get_vendor

BATTERY_CATEGORIES = ("tool_battery", "tool_schema")


def load_golden(golden_dir: Path, case: Case) -> tuple[dict[str, Any], str | None] | None:
    path = golden_dir / f"{case.id}.jsonl"
    if not path.exists():
        return None
    goldens = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not goldens:
        return None
    summary = golden_summary(goldens)
    sample = next((g.get("content") for g in goldens if g.get("content")), None)
    return summary, sample


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify a target against the vendor golden set")
    ap.add_argument("--target", required=True, help="OpenAI-compatible base URL, e.g. http://localhost:18083/v1")
    ap.add_argument("--model", required=True, help="model name on the target")
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--golden-vendor", default="zai", choices=VENDORS)
    ap.add_argument("--golden-model", default=None, help="default: the vendor's default_model")
    ap.add_argument("--repeats", type=int, default=3, help="runs per case")
    ap.add_argument("--battery-repeats", type=int, default=10, help="runs per case for tool_battery and tool_schema")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--category", action="append")
    ap.add_argument("--case", action="append")
    ap.add_argument("--separate-reasoning", action="store_true", help="add SMG's separate_reasoning=true to every request")
    ap.add_argument("--extra-body", default=None, help="JSON object merged into every request's extra_body")
    ap.add_argument("--report", default=None, help="JSON report path (default runs/verify-<model>-<utc>.json)")
    ap.add_argument("--junit", default=None, help="JUnit XML path")
    args = ap.parse_args()

    vendor = get_vendor(args.golden_vendor)
    golden_model = args.golden_model or vendor.default_model
    golden_dir = GOLDEN / vendor.name / golden_model
    client = OpenAI(api_key=args.api_key, base_url=args.target, timeout=args.timeout, max_retries=0)
    base_extra: dict[str, Any] = json.loads(args.extra_body) if args.extra_body else {}
    if args.separate_reasoning:
        base_extra["separate_reasoning"] = True

    jobs: list[tuple[Case, dict[str, Any], str | None, int]] = []
    skipped = []
    for case in cases_for(vendor, args.category, args.case):
        loaded = load_golden(golden_dir, case)
        if loaded is None:
            skipped.append(case.id)
            continue
        summary, sample = loaded
        n = args.battery_repeats if case.category in BATTERY_CATEGORIES else args.repeats
        for i in range(n):
            jobs.append((case, summary, sample, i))
    if skipped:
        print(f"{len(skipped)} cases have no golden recording and were skipped: {', '.join(skipped[:10])}{' ...' if len(skipped) > 10 else ''}")

    def run(job: tuple[Case, dict[str, Any], str | None, int]) -> dict[str, Any]:
        case, summary, sample, i = job
        # vendor-only fields still ride in extra_body so the target sees the same request the vendor did
        target = call_openai_style(client, args.model, case.request, dict(base_extra), vendor.passthrough_keys, vendor.recommended_sampling)
        cmp = compare({"id": case.id, "category": case.category, "request": case.request, "expect": case.expect}, {**summary, "content_sample": sample}, target)
        cmp.update({"repeat": i, "target_record": target, "golden_summary": summary})
        tag = "PASS" if cmp["pass"] else "FAIL " + ",".join(cmp["failed_checks"])
        print(f"{case.id:34s} #{i} {tag}", flush=True)
        return cmp

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        results = list(pool.map(run, jobs))

    per_case = aggregate_cases(results)
    m = metrics(results)
    m["cases_failed_majority"] = sorted(k for k, v in per_case.items() if not v["pass"])
    m["thresholds"] = THRESHOLDS
    print(json.dumps({k: v for k, v in m.items() if k not in ("thresholds",)}, indent=2, ensure_ascii=False))

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report = Path(args.report or f"runs/verify-{golden_model}-{stamp}.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"target": args.target, "model": args.model, "golden": str(golden_dir), "metrics": m, "per_case": per_case, "results": results}, ensure_ascii=False, indent=1))
    print("report:", report)
    if args.junit:
        write_junit(args.junit, per_case, args.target)
        print("junit:", args.junit)
    if m.get("threshold_failures"):
        print("THRESHOLDS FAILED: " + "; ".join(m["threshold_failures"]))
        sys.exit(1)


if __name__ == "__main__":
    main()
