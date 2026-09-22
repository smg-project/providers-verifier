"""Turn a chat completion (streamed or not) or an error into one comparable record."""

from __future__ import annotations

import json
from typing import Any


def _json_valid(text: str | None) -> bool:
    if text is None:
        return False
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def record_from_response(resp: Any) -> dict[str, Any]:
    d = resp.model_dump() if hasattr(resp, "model_dump") else resp
    choice = (d.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    tool_calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        tool_calls.append({"name": fn.get("name"), "arguments": fn.get("arguments"), "arguments_json_valid": _json_valid(fn.get("arguments"))})
    usage = d.get("usage") or {}
    return {
        "status": 200,
        "finish_reason": choice.get("finish_reason"),
        "content": msg.get("content"),
        "reasoning_content": msg.get("reasoning_content"),
        "reasoning_present": bool(msg.get("reasoning_content")),
        "tool_calls": tool_calls,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        },
        "streamed": False,
    }


def record_from_stream(chunks: list[Any]) -> dict[str, Any]:
    content, reasoning, finish = [], [], None
    calls: dict[int, dict[str, Any]] = {}
    usage = None
    first_tool_chunk_has_id = None
    for ch in chunks:
        d = ch.model_dump() if hasattr(ch, "model_dump") else ch
        if d.get("usage"):
            usage = d["usage"]
        for c in d.get("choices") or []:
            delta = c.get("delta") or {}
            finish = c.get("finish_reason") or finish
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index") or 0
                slot = calls.setdefault(idx, {"name": None, "arguments": ""})
                if first_tool_chunk_has_id is None:
                    first_tool_chunk_has_id = bool(tc.get("id"))
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
    tool_calls = [{"name": v["name"], "arguments": v["arguments"], "arguments_json_valid": _json_valid(v["arguments"])} for _, v in sorted(calls.items())]
    usage = usage or {}
    return {
        "status": 200,
        "finish_reason": finish,
        "content": "".join(content) or None,
        "reasoning_content": "".join(reasoning) or None,
        "reasoning_present": bool(reasoning),
        "tool_calls": tool_calls,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        },
        "streamed": True,
        "stream_usage_present": bool(usage),
        "stream_first_tool_chunk_has_id": first_tool_chunk_has_id,
        "stream_chunks": len(chunks),
    }


def record_from_error(exc: Exception) -> dict[str, Any]:
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    message = None
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        message = err.get("message") if isinstance(err, dict) else None
    return {
        "status": status,
        "error_type": type(exc).__name__,
        "error_message": (message or str(exc))[:500],
        "finish_reason": None,
        "content": None,
        "reasoning_present": False,
        "tool_calls": [],
        "usage": {},
    }
