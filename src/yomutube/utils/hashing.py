from __future__ import annotations

import hashlib
import re
import unicodedata


SAFE_CHARS_RE = re.compile(r"[^0-9A-Za-z._-]+")
MULTI_SEP_RE = re.compile(r"[_-]{2,}")
PATH_UNSAFE_RE = re.compile(r"[/\\\x00-\x1f\x7f]+")
MULTI_SPACE_RE = re.compile(r"\s+")


def short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def safe_slug(value: str, *, fallback: str = "untitled", max_length: int = 80) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    asciiish = normalized.encode("ascii", "ignore").decode("ascii")
    slug = SAFE_CHARS_RE.sub("_", asciiish).strip("._-")
    slug = MULTI_SEP_RE.sub("_", slug)
    if not slug:
        slug = fallback
    return slug[:max_length].rstrip("._-") or fallback


def safe_path_component(value: str | None, *, fallback: str = "untitled", max_bytes: int = 240) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    component = PATH_UNSAFE_RE.sub("／", normalized)
    component = MULTI_SPACE_RE.sub(" ", component).strip(" .")
    if not component:
        component = fallback
    return _truncate_utf8(component, max_bytes=max_bytes).strip(" .") or fallback


def _truncate_utf8(value: str, *, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    result: list[str] = []
    used = 0
    for char in value:
        size = len(char.encode("utf-8"))
        if used + size > max_bytes:
            break
        result.append(char)
        used += size
    return "".join(result)
