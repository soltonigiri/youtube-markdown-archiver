from __future__ import annotations

from yomutube.utils.text import is_noise_text, normalize_text


def normalize_subtitle_text(
    text: str | None,
    *,
    unicode_form: str = "NFKC",
    remove_music_notes: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    return normalize_text(
        text,
        unicode_form=unicode_form,
        remove_music_notes=remove_music_notes,
        collapse_whitespace=collapse_whitespace,
        strip_tags=True,
    )


def is_empty_subtitle_text(text: str | None) -> bool:
    normalized = normalize_subtitle_text(text)
    return not normalized or is_noise_text(normalized)
