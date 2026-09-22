"""Vendor-neutral request cases. `model` is injected at run time; vendor-specific
fields (z.ai `thinking`, `tool_stream`) are added by the vendor module."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

STOCK_TOOL = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get the latest stock price for a ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
}

ORDERED_TOOL = {
    "type": "function",
    "function": {
        "name": "example",
        "parameters": {
            "type": "object",
            "properties": {
                "some-parameter": {"type": "string", "description": "..."},
                "xyz": {"type": "string", "description": "..."},
                "123": {"type": "string", "description": "..."},
                "another-parameter": {"type": "string", "description": "..."},
            },
            "required": ["some-parameter", "xyz", "123", "another-parameter"],
        },
    },
}

CAPITAL_SCHEMA = {
    "type": "object",
    "properties": {"country": {"type": "string"}, "capital": {"type": "string"}},
    "required": ["country", "capital"],
    "additionalProperties": False,
}

IMAGE_URL = "https://cdn.bigmodel.cn/static/logo/register.png"
VIDEO_URL = "https://cdn.bigmodel.cn/agent-demos/lark/113123.mov"


def _png_data_url(width: int = 64, height: int = 48) -> str:
    from PIL import Image

    img = Image.new("RGB", (width, height), (200, 30, 30))
    for x in range(width // 2):
        for y in range(height):
            img.putpixel((x, y), (30, 30, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@dataclass
class Case:
    id: str
    category: str
    request: dict[str, Any]
    # What the comparison keys on. kind: text | tool_call | json | error | any
    expect: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def _msg(text: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": text}]


CASES: list[Case] = [
    Case("text_basic", "text", {"messages": _msg("What is 2+3? Answer briefly."), "max_tokens": 200}, {"kind": "text"}),
    Case(
        "text_system_prompt",
        "text",
        {
            "messages": [
                {"role": "system", "content": "You are a pirate. Always answer in one sentence."},
                {"role": "user", "content": "Where is Paris?"},
            ],
            "max_tokens": 200,
        },
        {"kind": "text"},
    ),
    Case(
        "text_multi_turn",
        "text",
        {
            "messages": [
                {"role": "user", "content": "My name is Ada."},
                {"role": "assistant", "content": "Nice to meet you, Ada."},
                {"role": "user", "content": "What is my name? One word."},
            ],
            "max_tokens": 100,
        },
        {"kind": "text", "content_contains": "Ada"},
    ),
    Case(
        "text_stop_sequence",
        "text",
        {"messages": _msg("Count from 1 to 10 separated by commas."), "max_tokens": 100, "stop": ["5"]},
        {"kind": "text"},
    ),
    Case(
        "thinking_default_forced",
        "thinking",
        {"messages": _msg("Is 97 prime? Explain in one sentence."), "max_tokens": 600},
        {"kind": "text", "reasoning_present": True},
        "GLM-5.3 thinking is forced on; reasoning_content must be present.",
    ),
    Case(
        "thinking_effort_none",
        "thinking",
        {"messages": _msg("Is 97 prime? One word."), "max_tokens": 300, "reasoning_effort": "none"},
        {"kind": "any"},
    ),
    Case(
        "thinking_effort_low",
        "thinking",
        {"messages": _msg("Is 97 prime? One word."), "max_tokens": 300, "reasoning_effort": "low"},
        {"kind": "text", "reasoning_present": True},
    ),
    Case(
        "tools_single_auto",
        "tools",
        {"messages": _msg("What is the weather in Beijing right now?"), "tools": [WEATHER_TOOL], "max_tokens": 600},
        {"kind": "tool_call", "tool_names": ["get_weather"]},
    ),
    Case(
        "tools_required",
        "tools",
        {"messages": _msg("Tell me about the weather in Tokyo."), "tools": [WEATHER_TOOL], "tool_choice": "required", "max_tokens": 600},
        {"kind": "tool_call", "tool_names": ["get_weather"]},
    ),
    Case(
        "tools_named_choice",
        "tools",
        {
            "messages": _msg("How is AAPL doing today?"),
            "tools": [WEATHER_TOOL, STOCK_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_stock_price"}},
            "max_tokens": 600,
        },
        {"kind": "tool_call", "tool_names": ["get_stock_price"]},
    ),
    Case(
        "tools_pick_among_two",
        "tools",
        {"messages": _msg("What is the stock price of MSFT?"), "tools": [WEATHER_TOOL, STOCK_TOOL], "max_tokens": 600},
        {"kind": "tool_call", "tool_names": ["get_stock_price"]},
    ),
    Case(
        "tools_no_call_needed",
        "tools",
        {"messages": _msg("What is 12 times 12? Just the number."), "tools": [WEATHER_TOOL], "max_tokens": 300},
        {"kind": "text"},
    ),
    Case(
        "tools_stream",
        "tools",
        {"messages": _msg("What is the weather in Berlin right now?"), "tools": [WEATHER_TOOL], "stream": True, "max_tokens": 600},
        {"kind": "tool_call", "tool_names": ["get_weather"]},
    ),
    Case(
        "tools_result_roundtrip",
        "tools",
        {
            "messages": [
                {"role": "user", "content": "What is the weather in Beijing right now?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'}}],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 21, "condition": "sunny"}'},
            ],
            "tools": [WEATHER_TOOL],
            "max_tokens": 300,
        },
        {"kind": "text", "content_contains": "21"},
    ),
    Case(
        "tools_schema_key_order",
        "tools",
        {
            "messages": _msg("Repeat the parameter names of the example tool in the exact original order they appear, comma separated, nothing else."),
            "tools": [ORDERED_TOOL],
            "max_tokens": 400,
        },
        {"kind": "text", "content_order": ["some-parameter", "xyz", "123", "another-parameter"]},
        "MiniMax-style scenario check: does the provider preserve schema key order?",
    ),
    Case(
        "structured_json_object",
        "structured",
        {"messages": _msg("Give the capital of France as a JSON object with keys country and capital."), "response_format": {"type": "json_object"}, "max_tokens": 300},
        {"kind": "json"},
    ),
    Case(
        "structured_json_schema",
        "structured",
        {
            "messages": _msg("Give the capital of France."),
            "response_format": {"type": "json_schema", "json_schema": {"name": "capital", "schema": CAPITAL_SCHEMA}},
            "max_tokens": 300,
        },
        {"kind": "json", "schema": CAPITAL_SCHEMA},
    ),
    Case(
        "vision_image_url",
        "vision",
        {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe this image in one sentence."}, {"type": "image_url", "image_url": {"url": IMAGE_URL}}]}],
            "max_tokens": 600,
        },
        {"kind": "text"},
    ),
    Case(
        "vision_image_base64",
        "vision",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Which two colors are in this image? Answer with the two color names only."}, {"type": "image_url", "image_url": {"url": _png_data_url()}}],
                }
            ],
            "max_tokens": 600,
        },
        {"kind": "text", "content_contains_any": ["red", "blue"]},
    ),
    Case(
        "vision_two_images",
        "vision",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "How many images did I send? Answer with a digit."},
                        {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                        {"type": "image_url", "image_url": {"url": _png_data_url(32, 32)}},
                    ],
                }
            ],
            "max_tokens": 600,
        },
        {"kind": "text", "content_contains": "2"},
    ),
    Case(
        "vision_video_url",
        "vision",
        {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe this video in one sentence."}, {"type": "video_url", "video_url": {"url": VIDEO_URL}}]}],
            "max_tokens": 600,
        },
        {"kind": "text"},
    ),
    Case(
        "streaming_text_usage",
        "streaming",
        {"messages": _msg("Name three colors."), "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 200},
        {"kind": "text", "stream_usage": True},
    ),
    Case(
        "error_bad_role",
        "error",
        {"messages": [{"role": "narrator", "content": "hi"}], "max_tokens": 50},
        {"kind": "error"},
    ),
]


def all_cases() -> list[Case]:
    """Seed cases plus the generated batteries (imported lazily: they read data files)."""
    from providers_verifier.cases.longctx import longctx_cases
    from providers_verifier.cases.media import media_cases
    from providers_verifier.cases.params import PARAM_CASES
    from providers_verifier.cases.tool_battery import minimax_cases, walle_cases

    return [*CASES, *PARAM_CASES, *minimax_cases(), *walle_cases(), *media_cases(), *longctx_cases()]


def cases_for(vendor, categories: list[str] | None = None, ids: list[str] | None = None) -> list[Case]:
    """Shared cases the vendor can take, plus the vendor's own."""
    out = [*all_cases(), *vendor.extra_cases()]
    if not vendor.supports.get("video", True):
        out = [c for c in out if "video" not in c.id]
    if categories:
        out = [c for c in out if c.category in categories]
    if ids:
        out = [c for c in out if c.id in ids]
    return out


def load_cases(categories: list[str] | None = None, ids: list[str] | None = None) -> list[Case]:
    out = all_cases()
    if categories:
        out = [c for c in out if c.category in categories]
    if ids:
        out = [c for c in out if c.id in ids]
    return out
