# providers-verifier

Record what a model vendor's real API does, then replay the same requests against
an OpenAI-compatible gateway (SMG) and compare. First vendor: z.ai (GLM-5.3-Flash).

```bash
uv sync --extra dev
cp .env.example .env            # ZAI_API_KEY=...
uv run pv-record --model glm-5.3-flash --repeats 3 --cold        # writes golden/zai/glm-5.3-flash/*.jsonl
uv run pv-verify --target http://localhost:18083/v1 --model glm-5.3-flash --separate-reasoning
uv run pytest
```

- `src/providers_verifier/cases/` — vendor-neutral cases (text, thinking, tools, structured, vision, streaming, error). `model` is injected at run time; z.ai-only fields (`thinking`, `tool_stream`) are added by `vendors/zai.py`.
- `record.py` — runs every case with the native `zai-sdk` and with the OpenAI SDK, N times at the vendor's recommended sampling (temperature 1, top_p 0.95) plus one cold pass, and stores normalized outcomes.
- `verify.py` — replays against a target, compares with `compare.py` (exact: status class, finish reason, tool trigger, tool names, argument schema validity, reasoning presence, JSON validity, stream usage; scored: text), prints metrics in the MiniMax-Provider-Verifier vocabulary, writes `runs/*.json`.

The API key never leaves `.env` (git-ignored).
