"""z.ai-specific cases: the `thinking` object, preserved thinking, reasoning effort
values, `tool_stream`, `file_url` content blocks."""

from __future__ import annotations

from providers_verifier.cases import WEATHER_TOOL, Case, _msg

FILE_URL = "https://cdn.bigmodel.cn/static/demo/demo2.txt"


def zai_cases() -> list[Case]:
    return [
        Case(
            "zai_thinking_type_disabled",
            "vendor",
            {"messages": _msg("Is 97 prime? One word."), "max_tokens": 300, "thinking": {"type": "disabled"}},
            {"kind": "any"},
            "documented as unsupported on GLM-5.3-Flash",
        ),
        Case(
            "zai_thinking_clear_false_roundtrip",
            "vendor",
            {
                "messages": [
                    {"role": "user", "content": "Pick a number between 1 and 10 and remember it. Reply only with 'ok'."},
                    {"role": "assistant", "content": "ok", "reasoning_content": "I will pick 7 and keep it in mind."},
                    {"role": "user", "content": "Which number did you pick? Answer with the digit only."},
                ],
                "max_tokens": 200,
                "thinking": {"type": "enabled", "clear_thinking": False},
            },
            {"kind": "text", "content_contains": "7"},
            "preserved thinking",
        ),
        Case(
            "zai_thinking_clear_true_roundtrip",
            "vendor",
            {
                "messages": [
                    {"role": "user", "content": "Pick a number between 1 and 10 and remember it. Reply only with 'ok'."},
                    {"role": "assistant", "content": "ok", "reasoning_content": "I will pick 7 and keep it in mind."},
                    {"role": "user", "content": "Which number did you pick? Answer with the digit only."},
                ],
                "max_tokens": 200,
                "thinking": {"type": "enabled", "clear_thinking": True},
            },
            {"kind": "any"},
            "history reasoning dropped",
        ),
        *[
            Case(f"zai_reasoning_effort_{e}", "vendor", {"messages": _msg("Is 97 prime? One word."), "max_tokens": 400, "reasoning_effort": e}, {"kind": "any"})
            for e in ("none", "minimal", "low", "medium", "high", "xhigh", "max")
        ],
        Case(
            "zai_tool_stream_flag",
            "vendor",
            {"messages": _msg("What is the weather in Berlin right now?"), "tools": [WEATHER_TOOL], "stream": True, "tool_stream": True, "max_tokens": 600},
            {"kind": "tool_call", "tool_names": ["get_weather"]},
            "streamed tool arguments",
        ),
        Case(
            "zai_file_url",
            "vendor",
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "Summarize this file in one sentence."}, {"type": "file_url", "file_url": {"url": FILE_URL}}]}], "max_tokens": 400},
            {"kind": "any"},
            "z.ai-only content block; SMG is expected to reject",
        ),
    ]
