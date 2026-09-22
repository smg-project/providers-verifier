"""Record the vendor's real behaviour for every case, with the vendor's native SDK
and the OpenAI SDK, into golden/<vendor>/<model>/<case>.jsonl."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from providers_verifier.cases import Case, cases_for
from providers_verifier.core.normalize import record_from_error, record_from_response, record_from_stream
from providers_verifier.vendors import VENDORS, get_vendor
from providers_verifier.vendors.base import VendorProfile

GOLDEN = Path(__file__).resolve().parents[2] / "golden"


def call_openai_style(client: Any, model: str, request: dict[str, Any], extra_body: dict[str, Any] | None, passthrough_keys: tuple[str, ...], sampling: dict[str, float] | None) -> dict[str, Any]:
    """One chat completion through any client with the OpenAI `chat.completions.create` shape."""
    req = dict(request)
    extra_body = dict(extra_body or {})
    # fields the SDK would reject as typed params still reach the vendor, in the body
    for key in passthrough_keys:
        if key in req:
            value = req.pop(key)
            if key == "extra_body":
                extra_body.update(value)
            else:
                extra_body.setdefault(key, value)
    if sampling:
        for k, v in sampling.items():
            req.setdefault(k, v)
    t0 = time.time()
    try:
        if req.get("stream"):
            rec = record_from_stream(list(client.chat.completions.create(model=model, extra_body=extra_body or None, **req)))
        else:
            rec = record_from_response(client.chat.completions.create(model=model, extra_body=extra_body or None, **req))
    except Exception as exc:  # noqa: BLE001 - every failure is data here
        rec = record_from_error(exc)
    rec["latency_ms"] = int((time.time() - t0) * 1000)
    return rec


def record_case(vendor: VendorProfile, case: Case, model: str, sdk: str, repeat: int, temperature: float | None) -> dict[str, Any]:
    client = vendor.native_client() if sdk == "native" else vendor.openai_client()
    sampling = None if temperature is None else {**vendor.recommended_sampling, "temperature": temperature}
    passthrough = vendor.passthrough_keys + (vendor.native_extra_body_keys if sdk == "native" else ())
    rec = call_openai_style(client, model, case.request, vendor.vendor_body(case.request), passthrough, sampling)
    rec.update(
        {"case": case.id, "category": case.category, "model": model, "sdk": sdk, "repeat": repeat, "temperature": temperature, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    )
    return rec


def main() -> None:
    load_dotenv(dotenv_path=".env")
    ap = argparse.ArgumentParser(description="Record vendor golden outcomes")
    ap.add_argument("--vendor", default="zai", choices=VENDORS)
    ap.add_argument("--model", default=None, help="vendor model id (default: the vendor's default_model)")
    ap.add_argument("--sdk", default="native,openai", help="comma list of native, openai")
    ap.add_argument("--repeats", type=int, default=3, help="recordings at the vendor's recommended sampling")
    ap.add_argument("--cold", action="store_true", help="also record one pass at temperature 0.2 for stable text")
    ap.add_argument("--concurrency", type=int, default=4, help="cases recorded in parallel")
    ap.add_argument("--category", action="append")
    ap.add_argument("--case", action="append")
    ap.add_argument("--skip-existing", action="store_true", help="keep golden files that already exist")
    args = ap.parse_args()

    vendor = get_vendor(args.vendor)
    model = args.model or vendor.default_model
    cases = cases_for(vendor, args.category, args.case)
    out_dir = GOLDEN / vendor.name / model
    out_dir.mkdir(parents=True, exist_ok=True)
    sdks = [("native" if s.strip() == vendor.name else s.strip()) for s in args.sdk.split(",") if s.strip()]
    unknown = [s for s in sdks if s not in vendor.sdks()]
    if unknown:
        raise SystemExit(f"{vendor.name} supports sdks {vendor.sdks()}, not {unknown}")
    if args.skip_existing:
        cases = [c for c in cases if not (out_dir / f"{c.id}.jsonl").exists()]

    temps: list[float | None] = [vendor.recommended_sampling["temperature"]] * args.repeats
    if args.cold:
        temps.append(0.2)

    def run(case: Case) -> int:
        recs = [record_case(vendor, case, model, sdk, i, t) for sdk in sdks for i, t in enumerate(temps)]
        (out_dir / f"{case.id}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        for rec in recs:
            tag = "ERR" if rec.get("status") != 200 else (rec.get("finish_reason") or "?")
            tools = ",".join(t["name"] or "?" for t in rec.get("tool_calls") or [])
            print(f"{case.id:34s} {rec['sdk']:6s} #{rec['repeat']} {tag:12s} tools=[{tools}] reasoning={rec.get('reasoning_present')} {rec.get('latency_ms')}ms", flush=True)
        return len(recs)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        total = sum(pool.map(run, cases))
    print(f"recorded {total} outcomes for {len(cases)} cases into {out_dir}")


if __name__ == "__main__":
    main()
