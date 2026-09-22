"""Media matrix: image resolution tiers, formats, counts, sizes, base64 vs URL,
video formats and lengths. Images are generated on the fly; expectations are
recorded from the vendor (status, prompt_tokens, answer)."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from providers_verifier.cases import IMAGE_URL, VIDEO_URL, Case

DATA = Path(__file__).resolve().parents[3] / "data" / "media"


def _image_bytes(width: int, height: int, fmt: str = "PNG", pattern: str = "split") -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (220, 40, 40))
    d = ImageDraw.Draw(img)
    if pattern == "split":
        d.rectangle([0, 0, width // 2, height], fill=(40, 40, 220))
    elif pattern == "noise":
        import random

        rnd = random.Random(width * 7919 + height)
        px = img.load()
        for x in range(0, width, max(1, width // 256)):
            for y in range(0, height, max(1, height // 256)):
                px[x, y] = (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
    d.rectangle([width * 0.4, height * 0.4, width * 0.6, height * 0.6], fill=(40, 200, 40))
    buf = io.BytesIO()
    save_fmt = "JPEG" if fmt.upper() in ("JPG", "JPEG") else fmt.upper()
    kwargs = {"quality": 95} if save_fmt == "JPEG" else {}
    if save_fmt == "GIF":
        img = img.convert("P")
    img.save(buf, format=save_fmt, **kwargs)
    return buf.getvalue()


def _data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def image_part(width: int, height: int, fmt: str = "PNG", pattern: str = "split") -> dict[str, Any]:
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "JPG": "image/jpeg", "WEBP": "image/webp", "GIF": "image/gif", "BMP": "image/bmp"}[fmt.upper()]
    return {"type": "image_url", "image_url": {"url": _data_url(_image_bytes(width, height, fmt, pattern), mime)}}


def _video_part_from_file(name: str) -> dict[str, Any]:
    p = DATA / "videos" / name
    mime = {"mp4": "video/mp4", "mov": "video/quicktime", "mkv": "video/x-matroska", "avi": "video/x-msvideo"}[name.rsplit(".", 1)[1]]
    return {"type": "video_url", "video_url": {"url": _data_url(p.read_bytes(), mime)}}


COLOR_Q = "Which three colors are in this image? Answer with the color names only."
COUNT_Q = "How many images did I send? Answer with a digit only."


def _case(id_: str, parts: list[dict[str, Any]], question: str, expect: dict[str, Any] | None = None, max_tokens: int = 800, notes: str = "") -> Case:
    content = [{"type": "text", "text": question}, *parts]
    return Case(id_, "media", {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens}, expect or {"kind": "text"}, notes)


def media_cases() -> list[Case]:
    cases: list[Case] = []
    # resolution tiers (square and non-square), one image each
    for w, h in [(16, 16), (28, 28), (64, 64), (224, 224), (448, 448), (896, 896), (1344, 1344), (2048, 2048), (3840, 2160), (4096, 4096), (4000, 200), (200, 4000), (8192, 512)]:
        cases.append(_case(f"img_res_{w}x{h}", [image_part(w, h)], COLOR_Q, {"kind": "text", "content_contains_any": ["red", "blue", "green"], "prompt_tokens_exact": True}, notes="resolution tier"))
    # formats
    for fmt in ("PNG", "JPEG", "WEBP", "GIF", "BMP"):
        cases.append(_case(f"img_fmt_{fmt.lower()}", [image_part(512, 384, fmt)], COLOR_Q, {"kind": "text", "content_contains_any": ["red", "blue", "green"]}, notes="format"))
    # counts (all base64, small)
    for n in (1, 2, 4, 8, 16, 32, 64):
        parts = [image_part(96 + 8 * i, 96, pattern="noise") for i in range(n)]
        cases.append(_case(f"img_count_{n:02d}", parts, COUNT_Q, {"kind": "text", "content_contains": str(n), "prompt_tokens_exact": True}, notes="count tier"))
    # size tiers (noise PNG compresses badly -> large payloads)
    for label, side in (("2mb", 900), ("6mb", 1500), ("12mb", 2100)):
        cases.append(_case(f"img_size_{label}", [image_part(side, side, pattern="noise")], "Describe the image in five words.", {"kind": "any"}, notes="payload size tier"))
    # URL vs base64 of the same public image, and mixed
    cases.append(_case("img_url_https", [{"type": "image_url", "image_url": {"url": IMAGE_URL}}], "Describe this image in one sentence.", {"kind": "text", "prompt_tokens_exact": True}))
    cases.append(_case("img_mixed_url_and_base64", [{"type": "image_url", "image_url": {"url": IMAGE_URL}}, image_part(320, 240)], COUNT_Q, {"kind": "text", "content_contains": "2"}))
    # detail hint as in OpenAI
    for detail in ("low", "high", "auto"):
        p = image_part(1024, 1024)
        p["image_url"]["detail"] = detail
        cases.append(_case(f"img_detail_{detail}", [p], COLOR_Q, {"kind": "text", "prompt_tokens_exact": True}, notes="detail hint"))
    # images in system / assistant history / tool message
    cases.append(
        Case(
            "img_in_system_message",
            "media",
            {
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": "The user's avatar is attached."}, image_part(128, 128)]},
                    {"role": "user", "content": "What colors are in my avatar?"},
                ],
                "max_tokens": 300,
            },
            {"kind": "any"},
        )
    )
    cases.append(
        Case(
            "img_multiturn_followup",
            "media",
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Remember this image."}, image_part(256, 256)]},
                    {"role": "assistant", "content": "Noted."},
                    {"role": "user", "content": "Which colors were in it? Names only."},
                ],
                "max_tokens": 300,
            },
            {"kind": "text", "content_contains_any": ["red", "blue", "green"]},
        )
    )
    # videos
    cases.append(
        _case("video_url_https", [{"type": "video_url", "video_url": {"url": VIDEO_URL}}], "Describe this video in one sentence.", {"kind": "text", "prompt_tokens_exact": True}, max_tokens=600)
    )
    for name in ("real_2s.mp4", "12s_real.mp4", "test_video.mov", "test_video.mkv"):
        if (DATA / "videos" / name).is_file():
            cases.append(_case(f"video_b64_{name.replace('.', '_')}", [_video_part_from_file(name)], "Describe this video in one sentence.", {"kind": "any"}, max_tokens=600, notes="base64 video"))
    cases.append(
        _case(
            "video_plus_image", [{"type": "video_url", "video_url": {"url": VIDEO_URL}}, image_part(256, 256)], "Describe the video and name the colors in the image.", {"kind": "any"}, max_tokens=600
        )
    )
    cases.append(
        _case(
            "video_two_urls",
            [{"type": "video_url", "video_url": {"url": VIDEO_URL}}, {"type": "video_url", "video_url": {"url": VIDEO_URL}}],
            "How many videos did I send? Digit only.",
            {"kind": "any"},
            max_tokens=600,
        )
    )
    # error shapes
    cases.append(_case("img_broken_base64", [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}], COLOR_Q, {"kind": "any"}))
    cases.append(_case("img_unreachable_url", [{"type": "image_url", "image_url": {"url": "https://example.invalid/nope.png"}}], COLOR_Q, {"kind": "any"}))
    cases.append(_case("img_text_mime", [{"type": "image_url", "image_url": {"url": _data_url(b"hello", "text/plain")}}], COLOR_Q, {"kind": "any"}))
    return cases
