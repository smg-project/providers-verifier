# providers-verifier

Does a model served through SMG behave the same way it behaves on the vendor's
own API?

Vendors such as Kimi and MiniMax publish their own verifier kits. Vendors that do
not (z.ai's GLM today, DeepSeek next) get one here. The idea is simple:

1. **Record.** Send a large set of requests to the vendor's real API, with the
   vendor's native SDK and with the OpenAI SDK, several times each. Store what
   came back as a *golden set*.
2. **Replay.** Send the same requests to the gateway under test.
3. **Compare.** For every request, check the gateway's answer against the golden
   set and report pass rates in the same vocabulary MiniMax's verifier uses.

Everything that can be compared exactly is compared exactly: HTTP status class,
finish reason, whether a tool was called and which one, whether the arguments fit
the tool's JSON schema, whether reasoning was returned separately, whether
`<think>` text leaked into the answer, whether streamed usage arrived, and the
exact prompt token count for media requests. Free text is only checked for
required substrings and ordering, never for wording.

## Quick start

```bash
uv sync --extra dev
cp .env.example .env            # put the vendor API key here; .env is git-ignored

# 1. record the golden set from the vendor (z.ai, GLM-5.3-Flash)
uv run pv-record --vendor zai --model glm-5.3-flash --repeats 3 --cold

# 2. replay against the gateway and compare
uv run pv-verify --target http://localhost:18083/v1 --model glm-5.3-flash \
    --separate-reasoning --junit runs/junit.xml

# unit tests for the comparison and statistics code
uv run pytest
```

`pv-verify` prints a metrics block, writes a JSON report under `runs/`, writes
JUnit XML when asked, and exits non-zero if any threshold fails, so it can sit in
CI.

## What gets tested

The shared cases live in `src/providers_verifier/cases/`. They are vendor
neutral: the model name is injected at run time and vendor-only request fields
are added by the vendor profile.

| Category | Cases | What it proves |
|---|---|---|
| `text`, `thinking`, `streaming`, `structured`, `error` | 11 | Basic chat, reasoning split, streamed usage, JSON mode and JSON schema, error shape |
| `tools` | 8 | Tool calling in every `tool_choice` mode, parallel calls, tool results round-trip |
| `tool_battery` | 102 | MiniMax's labelled prompts: does the model call a tool when it should and stay quiet when it should not |
| `tool_schema` | 424 | Kimi's 212 real-world JSON schemas, forced through a strict `submit` tool, streamed and not |
| `params` | 24 | Out-of-range and malformed parameters: the gateway must answer the way the vendor does, not crash or silently accept |
| `media` | 45 | Image resolutions from 16 px to 8k, five formats, 1 to 64 images per request, base64 and URL, videos, mixed media, and the **exact prompt token count** the vendor charges |
| `longctx` | 4 | Needle retrieval at 32k, 128k, 512k and 1M prompt tokens |
| `vendor` | 12 | Vendor-specific fields (for z.ai: the `thinking` object, preserved thinking across turns, every `reasoning_effort` value, `tool_stream`, `file_url`) |

Total: 634 cases for z.ai. Tool batteries run 10 times per case by default,
everything else 3 times, and a case passes when the majority of its runs pass.

The test data under `data/` is vendored from the MIT-licensed
MiniMax-Provider-Verifier and Kimi-Vendor-Verifier repositories. See
`data/README.md` for attribution.

### Metrics and thresholds

| Metric | Threshold | Meaning |
|---|---|---|
| `query_success_rate` | 1.00 | Every request the vendor answers with 200, the gateway answers with 200 |
| `tool_trigger_match_rate` | 0.98 | Tool called exactly when expected (confusion matrix is also reported) |
| `tool_trigger_f1` | 0.98 | Precision and recall of tool triggering |
| `tool_schema_accuracy` | 0.98 | Arguments validate against the tool's JSON schema |
| `think_leak_rate` | 0.00 | No `<think>` markup in the answer text |
| `error_only_reasoning_rate` | 0.00 | No answers that contain reasoning but neither content nor a tool call |
| `prompt_tokens_match_rate` | 1.00 | Media requests are billed the same number of prompt tokens as at the vendor |
| `language_following_rate` | 0.40 | MiniMax's Russian-answer prompts are answered in Russian |
| `scenario_check_pass_rate` | 1.00 | Multi-step tool scenarios keep the vendor's content order |

Thresholds are in `src/providers_verifier/core/stats.py`.

## Repository layout

```
src/providers_verifier/
  cases/            shared, vendor-neutral cases (see table above)
  core/
    normalize.py    turns any SDK response, stream, or exception into one flat record
    compare.py      per-case checks against the golden summary
    stats.py        majority vote over repeats, metrics, thresholds
    junit.py        JUnit XML writer
  vendors/
    base.py         VendorProfile: clients, vendor-only request fields, vendor cases
    zai/            z.ai: native zai-sdk + OpenAI SDK clients, GLM cases
    deepseek/       placeholder for the next vendor
  record.py         pv-record
  verify.py         pv-verify
data/               vendored prompts, schemas and sample videos (MIT)
golden/<vendor>/<model>/<case>.jsonl   recorded vendor behaviour, committed
runs/               reports from pv-verify (git-ignored)
tests/              unit tests
```

### Adding a vendor

Create `src/providers_verifier/vendors/<name>/` with:

- `client.py`: an `openai_client()` factory, optionally a `native_client()`
  factory with the same `chat.completions.create` shape, `vendor_body(request)`
  returning the vendor-only fields to send in `extra_body`, and
  `RECOMMENDED_SAMPLING`.
- `cases.py`: cases for behaviour only that vendor has.
- `__init__.py`: `PROFILE = VendorProfile(...)` wiring the above. List the request
  keys that must not go through the OpenAI SDK's typed parameters in
  `passthrough_keys`.

Add the name to `VENDORS` in `vendors/__init__.py`. The shared cases, recording,
replay, comparison, statistics and reports need no changes. Keep one vendor per
package; a separate repository per vendor would duplicate all of `core/` and the
shared batteries for no gain.

## Notes for GLM on SMG

- z.ai's `thinking` object is not an OpenAI parameter. `pv-verify` forwards it in
  `extra_body` so SMG sees the same request the vendor saw.
- Long-context cases need the model's tokenizer to size their filler exactly. Set
  `PV_TOKENIZER_JSON=/path/to/tokenizer.json`; without it the filler is sized by
  a character estimate and the token counts will be approximate.
- The API key is read from `.env` and never written to golden files or reports.
