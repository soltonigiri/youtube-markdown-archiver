from __future__ import annotations

import fnmatch
from typing import Any

from yomutube.config import AppConfig
from yomutube.models import SubtitleTrack


LANGUAGE_PRIORITY = ("ja.*", "en.*")
FORMAT_PRIORITY = ("vtt", "json3", "ttml", "srv3", "best")
EXCLUDED_LANGUAGES = ("live_chat",)


def _matches_language(language: str, pattern: str) -> bool:
    value = language.strip()
    candidate = value.lower()
    normalized_pattern = pattern.strip().lower()
    if fnmatch.fnmatchcase(candidate, normalized_pattern):
        return True
    if normalized_pattern.endswith(".*"):
        base = normalized_pattern[:-2]
        return candidate == base or fnmatch.fnmatchcase(candidate, f"{base}-*")
    return False


def _is_excluded(language: str, patterns: list[str] | tuple[str, ...]) -> bool:
    return any(_matches_language(language, pattern) for pattern in patterns)


def _entry_ext(entry: dict[str, Any]) -> str:
    return str(entry.get("ext") or entry.get("format") or "").strip().lower()


def _select_format(entries: list[dict[str, Any]], formats: list[str] | tuple[str, ...]) -> dict[str, Any] | None:
    if not entries:
        return None
    for fmt in formats:
        normalized = fmt.lower()
        if normalized == "best":
            for entry in entries:
                if entry.get("url") or entry.get("filepath") or entry.get("path"):
                    return entry
            return entries[0]
        for entry in entries:
            if _entry_ext(entry) == normalized:
                return entry
    return entries[0]


def _find_language(
    tracks: dict[str, Any],
    language_pattern: str,
    excluded_languages: list[str] | tuple[str, ...],
) -> tuple[str, list[dict[str, Any]]] | None:
    for language in sorted(tracks):
        if _is_excluded(language, excluded_languages):
            continue
        if not _matches_language(language, language_pattern):
            continue
        entries = tracks.get(language) or []
        if isinstance(entries, dict):
            entries = [entries]
        normalized_entries = [entry for entry in entries if isinstance(entry, dict)]
        if normalized_entries:
            return language, normalized_entries
    return None


def select_subtitle_track(info: dict[str, Any], config: AppConfig | None = None) -> SubtitleTrack | None:
    """Select one subtitle track from yt-dlp info.

    Priority is manual ja/en. Automatic captions are considered only when
    subtitles.fallback_to_auto is explicitly true. Within the chosen language,
    formats are selected as vtt, json3, ttml, srv3, then any best available item.
    """

    if not info:
        return None

    languages = tuple((config.get("ytdlp.subtitle_languages", LANGUAGE_PRIORITY) if config else None) or LANGUAGE_PRIORITY)
    formats = tuple((config.get("ytdlp.subtitle_formats", FORMAT_PRIORITY) if config else None) or FORMAT_PRIORITY)
    excluded = tuple((config.get("ytdlp.exclude_subtitle_languages", EXCLUDED_LANGUAGES) if config else EXCLUDED_LANGUAGES) or ())
    fallback_to_auto = bool(config.get("subtitles.fallback_to_auto", False) if config else False)

    sources = [("manual", "subtitles")]
    if fallback_to_auto:
        sources.append(("auto", "automatic_captions"))
    for source, info_key in sources:
        source_tracks = info.get(info_key) or {}
        if not isinstance(source_tracks, dict):
            continue
        for language_pattern in languages:
            match = _find_language(source_tracks, str(language_pattern), excluded)
            if match is None:
                continue
            language, entries = match
            selected = _select_format(entries, formats)
            if selected is None:
                continue
            raw = dict(selected)
            raw["language"] = language
            raw["source"] = source
            return SubtitleTrack(
                source=source,
                language=language,
                ext=_entry_ext(selected) or "best",
                url=selected.get("url"),
                name=selected.get("name"),
                raw=raw,
            )
    return None
