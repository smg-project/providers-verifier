"""Aggregation over repeats, MiniMax-style metrics, and pass thresholds."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

# Thresholds in the vocabulary of MiniMax-Provider-Verifier, plus ours. A metric
# below its threshold fails the run.
THRESHOLDS: dict[str, float] = {
    "query_success_rate": 1.0,
    "tool_trigger_match_rate": 0.98,
    "tool_trigger_f1": 0.98,
    "tool_schema_accuracy": 0.98,
    "error_only_reasoning_rate_max": 0.0,
    "think_leak_rate_max": 0.0,
    "prompt_tokens_match_rate": 1.0,
    "language_following_rate": 0.4,
    "scenario_check_pass_rate": 1.0,
}


def aggregate_cases(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Majority verdict per case over repeats; keep the failed checks of the worst repeat."""
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_case[r["case"]].append(r)
    out = {}
    for case_id, runs in by_case.items():
        passes = sum(1 for r in runs if r["pass"])
        worst = min(runs, key=lambda r: (r["pass"], -len(r["failed_checks"])))
        failed = Counter(k for r in runs for k in r["failed_checks"])
        out[case_id] = {
            "category": runs[0]["category"],
            "runs": len(runs),
            "passes": passes,
            "pass": passes * 2 > len(runs),
            "pass_rate": round(passes / len(runs), 3),
            "failed_checks": sorted(failed),
            "failed_check_counts": dict(failed),
            "worst_checks": {k: v for k, v in worst["checks"].items() if v["ok"] is False},
            "mean_latency_s": round(sum(r["target_record"].get("latency_ms", 0) for r in runs) / len(runs) / 1000, 3),
        }
    return out


def metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    if not n:
        return {}
    success = sum(1 for r in results if r["target_record"].get("status") == 200)

    # tool-call confusion matrix against the expected label (case label if present, else golden majority)
    tp = fn = fp = tn = 0
    schema_ok = schema_bad = 0
    eor_checked = eor = 0
    for r in results:
        g = r["golden_summary"]
        exp = r.get("expected_tool_call")
        if exp is None:
            exp = g.get("tool_triggered") if g.get("status_class") == "2xx" else None
        t = r["target_record"]
        if t.get("status") != 200:
            continue
        actual = bool(t.get("tool_calls"))
        if exp is True:
            if actual:
                tp += 1
                ok = r["checks"].get("args_match_schema", {}).get("ok")
                if ok is True:
                    schema_ok += 1
                elif ok is False:
                    schema_bad += 1
            else:
                fn += 1
        elif exp is False:
            if actual:
                fp += 1
            else:
                tn += 1
        # error-only reasoning: reasoning but neither content nor tool call
        eor_checked += 1
        if t.get("reasoning_present") and not (t.get("content") or "").strip() and not actual:
            eor += 1

    def rate(a: int, b: int) -> float | None:
        return round(a / b, 4) if b else None

    precision = rate(tp, tp + fp)
    recall = rate(tp, tp + fn)
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision and recall else (0.0 if (tp + fp + fn) else None)

    def check_rate(name: str) -> float | None:
        rel = [r for r in results if name in r["checks"]]
        return rate(sum(1 for r in rel if r["checks"][name]["ok"]), len(rel))

    leak = sum(1 for r in results if r["checks"].get("no_think_leak", {}).get("ok") is False)
    m = {
        "runs": n,
        "cases": len({r["case"] for r in results}),
        "query_success_rate": rate(success, n),
        "case_pass_rate": rate(sum(1 for r in results if r["pass"]), n),
        "tool_confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "tool_trigger_match_rate": rate(tp + tn, tp + fn + fp + tn),
        "tool_trigger_f1": f1,
        "tool_schema_accuracy": rate(schema_ok, schema_ok + schema_bad),
        "error_only_reasoning_rate": rate(eor, eor_checked),
        "think_leak_rate": rate(leak, n),
        "prompt_tokens_match_rate": check_rate("prompt_tokens_match"),
        "language_following_rate": check_rate("language_following"),
        "scenario_check_pass_rate": check_rate("content_order"),
        "failed_by_check": dict(Counter(k for r in results for k in r["failed_checks"])),
        "by_category": {},
    }
    cats = defaultdict(list)
    for r in results:
        cats[r["category"]].append(r)
    for c, rs in sorted(cats.items()):
        m["by_category"][c] = {"runs": len(rs), "pass_rate": rate(sum(1 for r in rs if r["pass"]), len(rs))}
    m["threshold_failures"] = threshold_failures(m)
    return m


def threshold_failures(m: dict[str, Any]) -> list[str]:
    fails = []
    for key, th in THRESHOLDS.items():
        if key.endswith("_max"):
            v = m.get(key[: -len("_max")])
            if v is not None and v > th:
                fails.append(f"{key[:-4]}={v} > {th}")
        else:
            v = m.get(key)
            if v is not None and v < th:
                fails.append(f"{key}={v} < {th}")
    return fails
