"""Record the vendor's real behaviour for every case, with both the native SDK
and the OpenAI SDK, into golden/<vendor>/<model>/<case>.jsonl."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from providers_verifier.cases import Case, load_cases
from providers_verifier.normalize import record_from_error, record_from_response, record_from_stream
from providers_verifier.vendors import zai

GOLDEN = Path(__file__).resolve().parents[2] / "golden"


def _call_openai_style(client: Any, model: str, case: Case, extra_body: dict[str, Any] | None, temperature: float | None) -> dict[str, Any]:
    req = dict(case.request)
    # vendor-only fields never go through the OpenAI SDK's typed params
    req.pop("thinking", None)
    if temperature is not None:
        req["temperature"] = temperature
        req.setdefault("top_p", zai.RECOMMENDED_SAMPLING["top_p"])
    t0 = time.time()
    try:
        if req.get("stream"):
            stream = client.chat.completions.create(model=model, extra_body=extra_body or None, **req)
            rec = record_from_stream(list(stream))
        else:
            rec = record_from_response(client.chat.completions.create(model=model, extra_body=extra_body or None, **req))
    except Exception as exc:  # noqa: BLE001 - every failure is data here
        rec = record_from_error(exc)
    rec["latency_ms"] = int((time.time() - t0) * 1000)
    return rec


def record_case(case: Case, model: str, sdk: str, repeat: int, temperature: float | None) -> dict[str, Any]:
    if sdk == "zai":
        client = zai.native_client()
    else:
        client = zai.openai_client()
    extra = zai.vendor_body(case.request)
    rec = _call_openai_style(client, model, case, extra, temperature)
    rec.update({"case": case.id, "category": case.category, "model": model, "sdk": sdk, "repeat": repeat, "temperature": temperature, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return rec


def main() -> None:
    load_dotenv(dotenv_path=".env")
    ap = argparse.ArgumentParser(description="Record vendor golden outcomes")
    ap.add_argument("--vendor", default="zai", choices=["zai"])
    ap.add_argument("--model", default=zai.DEFAULT_MODEL)
    ap.add_argument("--sdk", default="zai,openai", help="comma list: zai, openai")
    ap.add_argument("--repeats", type=int, default=3, help="recordings at the vendor's recommended sampling")
    ap.add_argument("--cold", action="store_true", help="also record one pass at temperature 0.2 for stable text")
    ap.add_argument("--category", action="append")
    ap.add_argument("--case", action="append")
    args = ap.parse_args()

    cases = load_cases(args.category, args.case)
    out_dir = GOLDEN / args.vendor / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    sdks = [s.strip() for s in args.sdk.split(",") if s.strip()]
    total = 0
    for case in cases:
        path = out_dir / f"{case.id}.jsonl"
        with path.open("w") as f:
            for sdk in sdks:
                temps: list[float | None] = [zai.RECOMMENDED_SAMPLING["temperature"]] * args.repeats
                if args.cold:
                    temps.append(0.2)
                for i, temp in enumerate(temps):
                    rec = record_case(case, args.model, sdk, i, temp)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += 1
                    tag = "ERR" if rec.get("status") != 200 else (rec.get("finish_reason") or "?")
                    tools = ",".join(t["name"] or "?" for t in rec.get("tool_calls") or [])
                    print(f"{case.id:34s} {sdk:6s} #{i} {tag:12s} tools=[{tools}] reasoning={rec.get('reasoning_present')} {rec.get('latency_ms')}ms")
    print(f"recorded {total} outcomes for {len(cases)} cases into {out_dir}")


if __name__ == "__main__":
    main()
