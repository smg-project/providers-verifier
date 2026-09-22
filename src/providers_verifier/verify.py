"""Replay the cases against a target OpenAI-compatible endpoint and compare with
the vendor golden set. Prints MiniMax-style metrics and writes a JSON report."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from providers_verifier.cases import load_cases
from providers_verifier.compare import compare, golden_summary
from providers_verifier.normalize import record_from_error, record_from_response, record_from_stream
from providers_verifier.record import GOLDEN


def _call(client: OpenAI, model: str, request: dict[str, Any], extra_body: dict[str, Any], temperature: float) -> dict[str, Any]:
    req = dict(request)
    req.pop("thinking", None)
    req.setdefault("temperature", temperature)
    req.setdefault("top_p", 0.95)
    t0 = time.time()
    try:
        if req.get("stream"):
            rec = record_from_stream(list(client.chat.completions.create(model=model, extra_body=extra_body or None, **req)))
        else:
            rec = record_from_response(client.chat.completions.create(model=model, extra_body=extra_body or None, **req))
    except Exception as exc:  # noqa: BLE001
        rec = record_from_error(exc)
    rec["latency_ms"] = int((time.time() - t0) * 1000)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify a target against the vendor golden set")
    ap.add_argument("--target", required=True, help="OpenAI-compatible base URL, e.g. http://localhost:18083/v1")
    ap.add_argument("--model", required=True, help="model name on the target")
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--golden-vendor", default="zai")
    ap.add_argument("--golden-model", default="glm-5.3-flash")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--category", action="append")
    ap.add_argument("--case", action="append")
    ap.add_argument("--separate-reasoning", action="store_true", help="add SMG's separate_reasoning=true to every request")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.target, timeout=900)
    cases = load_cases(args.category, args.case)
    golden_dir = GOLDEN / args.golden_vendor / args.golden_model
    results = []
    for case in cases:
        gpath = golden_dir / f"{case.id}.jsonl"
        if not gpath.exists():
            print(f"{case.id}: no golden recording, skipped")
            continue
        goldens = [json.loads(l) for l in gpath.read_text().splitlines() if l.strip()]
        summary = golden_summary(goldens)
        sample = next((g.get("content") for g in goldens if g.get("content")), None)
        summary["content_sample"] = sample
        extra: dict[str, Any] = {}
        if case.request.get("thinking"):
            extra["thinking"] = case.request["thinking"]
        if args.separate_reasoning:
            extra["separate_reasoning"] = True
        for i in range(args.repeats):
            target = _call(client, args.model, case.request, extra, 1.0)
            cmp = compare({"id": case.id, "category": case.category, "request": case.request, "expect": case.expect}, summary, target)
            cmp["repeat"] = i
            cmp["target_record"] = target
            cmp["golden_summary"] = {k: v for k, v in summary.items() if k != "content_sample"}
            results.append(cmp)
            tag = "PASS" if cmp["pass"] else "FAIL " + ",".join(cmp["failed_checks"])
            print(f"{case.id:34s} #{i} {tag}")

    # metrics
    by_case: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_case.setdefault(r["case"], []).append(r)
    tool_cases = [r for r in results if r["golden_summary"]["tool_triggered"]]
    tp = sum(1 for r in tool_cases if r["target_record"].get("tool_calls"))
    schema_ok = sum(1 for r in tool_cases if r["checks"].get("args_match_schema", {}).get("ok"))
    leak = sum(1 for r in results if r["checks"].get("no_think_leak", {}).get("ok") is False)
    success = sum(1 for r in results if r["target_record"].get("status") == 200)
    metrics = {
        "cases": len(by_case),
        "runs": len(results),
        "query_success_rate": round(success / len(results), 4) if results else None,
        "case_pass_rate": round(sum(1 for r in results if r["pass"]) / len(results), 4) if results else None,
        "tool_trigger_match_rate": round(sum(1 for r in results if r["checks"].get("tool_triggered", {}).get("ok")) / max(1, sum(1 for r in results if "tool_triggered" in r["checks"])), 4),
        "tool_schema_accuracy": round(schema_ok / tp, 4) if tp else None,
        "think_leak_rate": round(leak / len(results), 4) if results else None,
        "failed_by_check": dict(Counter(k for r in results for k in r["failed_checks"])),
        "failing_cases": sorted({r["case"] for r in results if not r["pass"]}),
    }
    print(json.dumps(metrics, indent=2))
    report = args.report or f"runs/verify-{args.golden_model}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text(json.dumps({"target": args.target, "model": args.model, "metrics": metrics, "results": results}, ensure_ascii=False, indent=1))
    print("report:", report)


if __name__ == "__main__":
    main()
