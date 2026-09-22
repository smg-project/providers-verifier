"""Long-context needle retrieval at 128k, 512k and 1M tokens. Filler is calibrated
against the model tokenizer at build time when `tokenizers` files are available,
else a chars-per-token estimate is used."""

from __future__ import annotations

import os
from functools import lru_cache

from providers_verifier.cases import Case

FILLER = (
    "The quarterly maintenance log records routine inspections of pumps, valves, and gauges across the "
    "facility. Technicians note pressure readings, replace worn gaskets, and file reports before the shift ends. "
)
NEEDLE = "The secret access code for the archive room is {code}."
QUESTION = "What is the secret access code for the archive room? Reply with the code only."


@lru_cache(maxsize=1)
def _tokens_per_filler() -> float:
    path = os.environ.get("PV_TOKENIZER_JSON")
    if path and os.path.exists(path):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(path)
        return len(tok.encode(FILLER * 50, add_special_tokens=False).ids) / 50
    return len(FILLER) / 3.6


def build_prompt(target_tokens: int, code: str) -> str:
    per = _tokens_per_filler()
    n = max(1, int(target_tokens / per))
    chunks = [FILLER] * n
    chunks.insert(n // 2, NEEDLE.format(code=code) + " ")
    return "".join(chunks) + "\n\n" + QUESTION


def longctx_cases(sizes: tuple[int, ...] = (32_000, 128_000, 512_000, 1_000_000)) -> list[Case]:
    cases = []
    for size in sizes:
        code = f"ZX-{size // 1000}-{(size * 7919) % 9973:04d}"
        cases.append(
            Case(
                f"longctx_{size // 1000}k",
                "longctx",
                {"messages": [{"role": "user", "content": build_prompt(size, code)}], "max_tokens": 1024},
                {"kind": "text", "content_contains": code, "prompt_tokens_exact": True},
                "needle in the middle; prompt_tokens must match the vendor",
            )
        )
    return cases
