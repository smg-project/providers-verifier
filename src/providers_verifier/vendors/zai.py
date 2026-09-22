"""z.ai (GLM) vendor: native `zai-sdk` client and OpenAI-compatible client, plus
the vendor-specific request fields."""

from __future__ import annotations

import os
from typing import Any

BASE_URL = "https://api.z.ai/api/paas/v4/"
DEFAULT_MODEL = "glm-5.3-flash"
RECOMMENDED_SAMPLING = {"temperature": 1.0, "top_p": 0.95}


def api_key() -> str:
    key = os.environ.get("ZAI_API_KEY")
    if not key:
        raise SystemExit("ZAI_API_KEY is not set (put it in .env)")
    return key


def native_client():
    from zai import ZaiClient

    return ZaiClient(api_key=api_key(), base_url=os.environ.get("ZAI_BASE_URL", BASE_URL))


def openai_client():
    from openai import OpenAI

    return OpenAI(api_key=api_key(), base_url=os.environ.get("ZAI_BASE_URL", BASE_URL))


def vendor_body(request: dict[str, Any]) -> dict[str, Any]:
    """Fields that ride in `extra_body` for the vendor: z.ai's `thinking` object
    (forced on for GLM-5.3, `clear_thinking: false` recommended), `tool_stream`
    for streamed tool calls. Cases may override `thinking`."""
    body: dict[str, Any] = {}
    body["thinking"] = request.get("thinking", {"type": "enabled", "clear_thinking": False})
    if request.get("stream") and request.get("tools"):
        body["tool_stream"] = True
    return body
