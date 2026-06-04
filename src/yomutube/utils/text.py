from __future__ import annotations

import html
import re
import unicodedata


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
MUSIC_RE = re.compile(r"[♪♫♬♩]+")
BRACKET_NOISE_RE = re.compile(r"^\s*[\[(（【]?(音楽|拍手|笑い|music|applause|laughter)[\])）】]?\s*$", re.I)


def normalize_text(
    text: str | None,
    *,
    unicode_form: str = "NFKC",
    remove_music_notes: bool = True,
    collapse_whitespace: bool = True,
    strip_tags: bool = True,
) -> str:
    if not text:
        return ""
    value = html.unescape(str(text))
    if strip_tags:
        value = TAG_RE.sub(" ", value)
    value = unicodedata.normalize(unicode_form, value)
    if remove_music_notes:
        value = MUSIC_RE.sub(" ", value)
    if collapse_whitespace:
        value = SPACE_RE.sub(" ", value)
    return value.strip()


def is_noise_text(text: str | None) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return True
    return BRACKET_NOISE_RE.match(normalized) is not None


def truncate_text(text: str, limit: int = 80) -> str:
    value = normalize_text(text)
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"
