"""Vendor registry. Add a package under vendors/<name>/ exposing PROFILE."""

from __future__ import annotations

from importlib import import_module

from providers_verifier.vendors.base import VendorProfile

VENDORS = ("zai", "deepseek")


def get_vendor(name: str) -> VendorProfile:
    try:
        return import_module(f"providers_verifier.vendors.{name}").PROFILE
    except (ImportError, AttributeError) as exc:
        raise SystemExit(f"unknown or incomplete vendor {name!r}: {exc}") from exc
