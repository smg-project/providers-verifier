"""Parameter policing: what the vendor does with out-of-range or unsupported
request fields. Expectations come from the recording (status class)."""

from __future__ import annotations

from providers_verifier.cases import WEATHER_TOOL, Case

_MSG = [{"role": "user", "content": "Say hi in one word."}]


def _c(id_: str, **fields):
    req = {"messages": _MSG, "max_tokens": 50, **fields}
    return Case(id_, "params", req, {"kind": "any"}, "record the vendor's status code")


PARAM_CASES: list[Case] = [
    _c("param_temperature_2_5", temperature=2.5),
    _c("param_temperature_negative", temperature=-0.5),
    _c("param_temperature_0", temperature=0),
    _c("param_top_p_1_5", top_p=1.5),
    _c("param_top_p_0", top_p=0.0),
    _c("param_n_2", n=2),
    _c("param_presence_penalty_3", presence_penalty=3.0),
    _c("param_frequency_penalty_3", frequency_penalty=3.0),
    _c("param_max_tokens_0", max_tokens=0),
    _c("param_max_tokens_negative", max_tokens=-1),
    _c("param_max_tokens_huge", max_tokens=10_000_000),
    _c("param_seed", seed=7),
    _c("param_logprobs", logprobs=True, top_logprobs=2),
    _c("param_stop_many", stop=["a", "b", "c", "d", "e", "f"]),
    _c("param_reasoning_effort_invalid", reasoning_effort="ultra"),
    _c("param_thinking_type_bogus", thinking={"type": "sometimes"}),
    _c("param_empty_messages", messages=[]),
    _c("param_unknown_field", extra_body={"foo_bar_baz": 1}),
    _c("param_tool_bad_type", tools=[{"type": "retrieval", "function": {"name": "x", "parameters": {"type": "object"}}}]),
    _c("param_tool_duplicate_names", tools=[WEATHER_TOOL, WEATHER_TOOL]),
    _c("param_tool_choice_unknown_tool", tools=[WEATHER_TOOL], tool_choice={"type": "function", "function": {"name": "nope"}}),
    _c("param_json_schema_not_object", response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": "not-an-object"}}),
    _c("param_stream_options_without_stream", stream_options={"include_usage": True}),
    _c("param_tool_name_invalid_chars", tools=[{"type": "function", "function": {"name": "bad name!", "parameters": {"type": "object"}}}]),
]
