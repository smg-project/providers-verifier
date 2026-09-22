"""A vendor profile: how to talk to the vendor's API natively and through the
OpenAI SDK, which request fields are vendor-specific, and which cases only make
sense for that vendor. Shared cases live in `providers_verifier.cases`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from providers_verifier.cases import Case


@dataclass
class VendorProfile:
    name: str
    default_model: str
    base_url: str
    env_key: str
    recommended_sampling: dict[str, float]
    # OpenAI-SDK client factory (always available) and native SDK factory (may be None)
    openai_client: Callable[[], Any]
    native_client: Callable[[], Any] | None = None
    # vendor-only fields to send in extra_body, derived from the case request
    vendor_body: Callable[[dict[str, Any]], dict[str, Any]] = lambda request: {}
    # request keys the OpenAI SDK must not see as typed params (moved to extra_body)
    passthrough_keys: tuple[str, ...] = ()
    # request keys the native SDK rejects as typed params; sent in extra_body for native recordings
    native_extra_body_keys: tuple[str, ...] = ()
    # vendor-specific cases appended to the shared set
    extra_cases: Callable[[], list[Case]] = list
    # capability flags used to skip shared cases the vendor cannot take
    supports: dict[str, bool] = field(default_factory=lambda: {"video": True, "image": True, "stream_usage": True})
    notes: str = ""

    def sdks(self) -> list[str]:
        return ["openai"] + (["native"] if self.native_client else [])
