"""z.ai (GLM) vendor profile."""

from __future__ import annotations

from providers_verifier.vendors.base import VendorProfile
from providers_verifier.vendors.zai import client
from providers_verifier.vendors.zai.cases import zai_cases

PROFILE = VendorProfile(
    name="zai",
    default_model=client.DEFAULT_MODEL,
    base_url=client.BASE_URL,
    env_key="ZAI_API_KEY",
    recommended_sampling=client.RECOMMENDED_SAMPLING,
    openai_client=client.openai_client,
    native_client=client.native_client,
    vendor_body=client.vendor_body,
    passthrough_keys=("thinking", "tool_stream", "extra_body"),
    native_extra_body_keys=("stream_options", "n", "logprobs", "top_logprobs", "presence_penalty", "frequency_penalty"),
    extra_cases=zai_cases,
    stream_fallback=True,
    notes=(
        "GLM-5.3 / 5.3-Flash: thinking forced on, clear_thinking:false recommended, tool_stream for streamed tools. "
        "Non-streaming requests are closed by the vendor after ~60-180 s of generation; long-thinking answers only complete when streamed."
    ),
)
