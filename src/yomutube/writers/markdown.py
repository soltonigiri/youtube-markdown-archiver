from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from yomutube.archive_assets import build_archive_asset_rows, read_jsonl_for_display
from yomutube.models import ArchiveResult, Manifest, VideoMetadata
from yomutube.state import atomic_write_text
from yomutube.utils.timecode import ms_to_timecode, seconds_to_timecode, youtube_seconds
from yomutube.writers.jsonl import write_jsonl


JST = timezone(timedelta(hours=9))
DEFAULT_SOURCE_PRIORITY = "manual_subtitle > asr > auto_subtitle"
TRANSCRIPT_SOURCES = {"manual_subtitle", "asr", "auto_subtitle"}


def write_archive(
    *,
    archive_dir: str | Path | None = None,
    metadata: VideoMetadata | dict[str, Any],
    segments: Iterable[Any],
    raw_asr: Iterable[Any] | None = None,
    raw_subtitles: Iterable[Any] | None = None,
    raw_ocr: Iterable[Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    manifest: Manifest | dict[str, Any] | None = None,
    program_version: str = "0.1.0",
    include_frontmatter: bool = True,
    include_ocr_events: bool = True,
    youtube_timestamp_links: bool = True,
    config: Any = None,
) -> ArchiveResult:
    if config is not None:
        include_frontmatter = bool(_config_get(config, "markdown.include_frontmatter", include_frontmatter))
        include_ocr_events = bool(_config_get(config, "markdown.include_ocr_events", include_ocr_events))
        youtube_timestamp_links = bool(_config_get(config, "markdown.youtube_timestamp_links", youtube_timestamp_links))
    target = _resolve_archive_dir(archive_dir, manifest)
    target.mkdir(parents=True, exist_ok=True)

    metadata_row = _to_metadata_dict(metadata)
    segment_rows = _sorted_rows(segments)
    raw_asr_rows = _sorted_rows(raw_asr or [])
    raw_subtitle_rows = _sorted_rows(raw_subtitles or [])
    raw_ocr_rows = _sorted_rows(raw_ocr or [])
    diagnostics_row = dict(diagnostics or {})
    speaker_aliases = _read_json_dict(target / "speaker_aliases.json")
    corrections = read_jsonl_for_display(target / "corrections.jsonl")
    manifest_obj = _coerce_manifest(
        manifest,
        metadata=metadata_row,
        archive_dir=target,
        diagnostics=diagnostics_row,
        program_version=program_version,
    )

    _write_json(target / "metadata.json", metadata_row)
    _ensure_plain_json(target / "speaker_aliases.json", speaker_aliases)
    _ensure_jsonl(target / "corrections.jsonl")
    if config is not None and bool(_config_get(config, "archive.write_effective_config", True)):
        _write_effective_config(target / "effective_config.yaml", config)
    write_jsonl(target / "segments.jsonl", segment_rows)
    write_jsonl(target / "raw_asr.jsonl", raw_asr_rows)
    write_jsonl(target / "raw_subtitles.jsonl", raw_subtitle_rows)
    write_jsonl(target / "raw_ocr.jsonl", raw_ocr_rows)
    _write_archive_assets(
        target,
        metadata=metadata_row,
        segments=segment_rows,
        raw_asr=raw_asr_rows,
        raw_ocr=raw_ocr_rows,
        speaker_aliases=speaker_aliases,
        config=config,
    )
    _write_json(target / "diagnostics.json", diagnostics_row)
    _write_json(target / "manifest.json", manifest_obj.to_dict())

    display_rows = _apply_correction_rows(segment_rows, corrections)
    index_text = render_index_markdown(
        metadata=metadata_row,
        segments=display_rows,
        raw_asr=raw_asr_rows,
        raw_subtitles=raw_subtitle_rows,
        raw_ocr=raw_ocr_rows,
        diagnostics=diagnostics_row,
        manifest=manifest_obj,
        speaker_aliases=speaker_aliases,
        include_frontmatter=include_frontmatter,
        include_ocr_events=include_ocr_events,
        youtube_timestamp_links=youtube_timestamp_links,
    )
    index_path = target / "index.md"
    atomic_write_text(index_path, index_text)

    return ArchiveResult(archive_dir=target, index_path=index_path, manifest=manifest_obj)


def render_index_markdown(
    *,
    metadata: dict[str, Any],
    segments: list[dict[str, Any]],
    raw_asr: list[dict[str, Any]] | None = None,
    raw_subtitles: list[dict[str, Any]] | None = None,
    raw_ocr: list[dict[str, Any]] | None = None,
    diagnostics: dict[str, Any] | None = None,
    manifest: Manifest | None = None,
    speaker_aliases: dict[str, str] | None = None,
    include_frontmatter: bool = True,
    include_ocr_events: bool = True,
    youtube_timestamp_links: bool = True,
) -> str:
    diagnostics = diagnostics or {}
    speaker_aliases = speaker_aliases or {}
    raw_asr = raw_asr or []
    raw_subtitles = raw_subtitles or []
    raw_ocr = raw_ocr or []
    title = str(metadata.get("title") or "Untitled")
    lines: list[str] = []

    if include_frontmatter:
        lines.extend(_frontmatter(metadata, segments, diagnostics, manifest))
        lines.append("")

    lines.append(f"# {title}")
    lines.append("")
    lines.extend(_metadata_section(metadata, diagnostics))
    lines.append("")
    lines.append("## Transcript")
    lines.append("")
    for segment in segments:
        if _is_ocr(segment):
            continue
        if not _should_show_segment(segment):
            continue
        rendered = _render_segment_line(
            segment,
            metadata,
            youtube_timestamp_links=youtube_timestamp_links,
            speaker_aliases=speaker_aliases,
        )
        if rendered:
            lines.append(rendered)
            lines.append("")

    if include_ocr_events:
        ocr_events = _ocr_events(raw_ocr or segments, assume_ocr=bool(raw_ocr))
        if ocr_events:
            lines.append("## OCR Events")
            lines.append("")
            lines.append("| Time | Region | Text |")
            lines.append("|---:|---|---|")
            for event in ocr_events:
                lines.append(
                    f"| {_timecode(event)} | {_escape_table(_ocr_region(event))} | {_escape_table(str(event.get('text') or ''))} |"
                )
            lines.append("")

    lines.extend(_diagnostics_section(diagnostics, segments, raw_asr, raw_subtitles, raw_ocr))
    return "\n".join(lines).rstrip() + "\n"


class MarkdownArchiveWriter:
    def __init__(
        self,
        archive_dir: str | Path,
        *,
        program_version: str = "0.1.0",
        include_frontmatter: bool = True,
        include_ocr_events: bool = True,
        youtube_timestamp_links: bool = True,
    ) -> None:
        self.archive_dir = Path(archive_dir)
        self.program_version = program_version
        self.include_frontmatter = include_frontmatter
        self.include_ocr_events = include_ocr_events
        self.youtube_timestamp_links = youtube_timestamp_links

    def write_archive(self, **kwargs: Any) -> ArchiveResult:
        kwargs.setdefault("archive_dir", self.archive_dir)
        kwargs.setdefault("program_version", self.program_version)
        kwargs.setdefault("include_frontmatter", self.include_frontmatter)
        kwargs.setdefault("include_ocr_events", self.include_ocr_events)
        kwargs.setdefault("youtube_timestamp_links", self.youtube_timestamp_links)
        return write_archive(**kwargs)


def _resolve_archive_dir(archive_dir: str | Path | None, manifest: Manifest | dict[str, Any] | None) -> Path:
    if archive_dir is not None:
        return Path(archive_dir)
    if isinstance(manifest, Manifest) and manifest.archive_path:
        return Path(manifest.archive_path)
    if isinstance(manifest, dict) and manifest.get("archive_path"):
        return Path(str(manifest["archive_path"]))
    raise ValueError("archive_dir is required when manifest.archive_path is missing")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    if "schema_version" not in data:
        data = {"schema_version": "1.0", **data}
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _ensure_plain_json(path: Path, data: dict[str, Any]) -> None:
    if path.exists():
        return
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _ensure_jsonl(path: Path) -> None:
    if not path.exists():
        atomic_write_text(path, "")


def _write_archive_assets(
    target: Path,
    *,
    metadata: dict[str, Any],
    segments: list[dict[str, Any]],
    raw_asr: list[dict[str, Any]],
    raw_ocr: list[dict[str, Any]],
    speaker_aliases: dict[str, str],
    config: Any,
) -> None:
    assets = build_archive_asset_rows(
        metadata=metadata,
        segments=segments,
        raw_asr=raw_asr,
        raw_ocr=raw_ocr,
        speaker_aliases=speaker_aliases,
    )
    jsonl_assets = [
        ("archive.write_words", "words.jsonl", assets["words"]),
        ("archive.write_speaker_turns", "speaker_turns.jsonl", assets["speaker_turns"]),
        ("archive.write_alignment", "alignment.jsonl", assets["alignment"]),
        ("archive.write_conflicts", "conflicts.jsonl", assets["conflicts"]),
        ("archive.write_topics", "topics.jsonl", assets["topics"]),
        ("archive.write_entities", "entities.jsonl", assets["entities"]),
        ("archive.write_visual_text", "visual_text.jsonl", assets["visual_text"]),
        ("archive.write_visual_text", "slides.jsonl", assets["slides"]),
    ]
    for dotted, filename, rows in jsonl_assets:
        if bool(_config_get(config, dotted, True)):
            write_jsonl(target / filename, rows)
    if bool(_config_get(config, "archive.write_quality_report", True)):
        _write_json(target / "quality.json", assets["quality"])


def _write_effective_config(path: Path, config: Any) -> None:
    data = config.data if hasattr(config, "data") else config
    if not isinstance(data, dict):
        data = {"value": str(data)}
    content = yaml.safe_dump(data, allow_unicode=True, sort_keys=True)
    atomic_write_text(path, content if content.endswith("\n") else content + "\n")


def _read_json_dict(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items() if value is not None}


def _apply_correction_rows(rows: list[dict[str, Any]], corrections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replacements = {
        str(row.get("target_segment_id")): row
        for row in corrections
        if row.get("operation") == "replace_text" and row.get("target_segment_id")
    }
    if not replacements:
        return rows
    corrected: list[dict[str, Any]] = []
    for row in rows:
        replacement = replacements.get(str(row.get("id")))
        if not replacement:
            corrected.append(row)
            continue
        metadata = dict(row.get("metadata") if isinstance(row.get("metadata"), dict) else {})
        metadata["display_correction"] = {
            "operation": replacement.get("operation"),
            "reason": replacement.get("reason"),
            "created_at": replacement.get("created_at"),
            "before": replacement.get("before"),
        }
        updated = dict(row)
        updated["text"] = str(replacement.get("after") or "")
        updated["metadata"] = metadata
        corrected.append(updated)
    return corrected


def _to_metadata_dict(metadata: VideoMetadata | dict[str, Any]) -> dict[str, Any]:
    if isinstance(metadata, VideoMetadata):
        return metadata.to_dict()
    return dict(metadata)


def _to_row(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        row = dict(value)
    elif hasattr(value, "to_dict"):
        row = dict(value.to_dict())
    elif is_dataclass(value):
        row = asdict(value)
    else:
        raise TypeError(f"unsupported archive row: {type(value).__name__}")
    row.setdefault("schema_version", "1.0")
    start_ms = int(row.get("start_ms") or 0)
    video_id = row.get("video_id")
    row.setdefault("timestamp", ms_to_timecode(start_ms, include_millis=True))
    if video_id:
        row.setdefault("youtube_timestamp_url", f"https://youtu.be/{video_id}?t={youtube_seconds(start_ms)}")
    return row


def _sorted_rows(rows: Iterable[Any]) -> list[dict[str, Any]]:
    converted = [_to_row(row) for row in rows]
    return sorted(converted, key=lambda row: (int(row.get("start_ms") or 0), str(row.get("id") or "")))


def _coerce_manifest(
    manifest: Manifest | dict[str, Any] | None,
    *,
    metadata: dict[str, Any],
    archive_dir: Path,
    diagnostics: dict[str, Any],
    program_version: str,
) -> Manifest:
    if isinstance(manifest, Manifest):
        result = Manifest.from_dict(manifest.to_dict())
    elif isinstance(manifest, dict):
        result = Manifest.from_dict(manifest)
    else:
        result = Manifest(
            program="YomuTube",
            version=program_version,
            video_id=str(metadata.get("video_id") or "unknown"),
            run_id=str(diagnostics.get("run_id") or "unknown"),
            status="done",
            steps={"markdown": "done"},
            models=_diagnostic_models(diagnostics),
            created_at=_iso_now(),
            finished_at=_iso_now(),
            archive_path=str(archive_dir),
        )
    if not result.archive_path:
        result.archive_path = str(archive_dir)
    if not result.finished_at:
        result.finished_at = _iso_now()
    return result


def _diagnostic_models(diagnostics: dict[str, Any]) -> dict[str, str]:
    models: dict[str, str] = {}
    asr = diagnostics.get("asr") if isinstance(diagnostics.get("asr"), dict) else {}
    ocr = diagnostics.get("ocr") if isinstance(diagnostics.get("ocr"), dict) else {}
    if asr:
        engine = asr.get("engine") or "asr"
        model = asr.get("model")
        models["asr"] = f"{engine}:{model}" if model else str(engine)
    if ocr:
        models["ocr"] = str(ocr.get("engine") or "ocr")
    return models


def _frontmatter(
    metadata: dict[str, Any],
    segments: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    manifest: Manifest | None,
) -> list[str]:
    program = manifest.program if manifest else "YomuTube"
    version = manifest.version if manifest else "0.1.0"
    processed_at = manifest.finished_at if manifest and manifest.finished_at else _iso_now()
    values = [
        ("program", program, False),
        ("program_version", version, False),
        ("video_id", metadata.get("video_id") or "unknown", False),
        ("title", metadata.get("title") or "Untitled", True),
        ("channel", _channel(metadata), True),
        ("channel_id", metadata.get("channel_id"), False),
        ("url", metadata.get("webpage_url") or metadata.get("url"), True),
        ("upload_date", _normalize_upload_date(metadata.get("upload_date")), False),
        ("duration", _duration(metadata), True),
        ("processed_at", processed_at, True),
        ("asr_engine", _asr_engine(diagnostics, manifest), False),
        ("asr_model", _asr_model(diagnostics, manifest), False),
        ("ocr_engine", _ocr_engine(diagnostics, manifest), False),
        ("subtitle_source", _subtitle_source(diagnostics, segments), False),
        ("language", _language(metadata, segments), False),
    ]
    lines = ["---"]
    for key, value, quote in values:
        lines.append(f"{key}: {_yaml_value(value, quote=quote)}")
    lines.append("---")
    return lines


def _metadata_section(metadata: dict[str, Any], diagnostics: dict[str, Any]) -> list[str]:
    return [
        "## Metadata",
        "",
        f"- Channel: {_channel(metadata) or 'unknown'}",
        f"- Duration: {_duration(metadata)}",
        "- Source: YouTube",
        f"- Transcript source priority: {DEFAULT_SOURCE_PRIORITY}",
        f"- OCR: {'enabled' if _nested(diagnostics, 'ocr', 'enabled') else 'disabled'}",
    ]


def _render_segment_line(
    segment: dict[str, Any],
    metadata: dict[str, Any],
    *,
    youtube_timestamp_links: bool,
    speaker_aliases: dict[str, str] | None = None,
) -> str:
    text = str(segment.get("text") or "").strip()
    if not text:
        return ""
    body_parts: list[str] = [_timestamp_link(segment, metadata, youtube_timestamp_links=youtube_timestamp_links)]
    if _is_uncertain(segment):
        body_parts.append("[?]")
    if _is_ocr(segment):
        body_parts.append(f"[画面: {_ocr_region(segment)}]")
    speaker = segment.get("speaker")
    if speaker:
        body_parts.append(f"{_speaker_label(str(speaker), speaker_aliases or {})}:")
    body_parts.append(text)
    return " ".join(body_parts)


def _speaker_label(speaker: str, aliases: dict[str, str]) -> str:
    return aliases.get(speaker) or speaker


def _timestamp_link(segment: dict[str, Any], metadata: dict[str, Any], *, youtube_timestamp_links: bool) -> str:
    timecode = _timecode(segment)
    if not youtube_timestamp_links:
        return f"[{timecode}]"
    link = _youtube_url(metadata, int(segment.get("start_ms") or 0))
    if not link:
        return f"[{timecode}]"
    return f"[{timecode}]({link})"


def _youtube_url(metadata: dict[str, Any], start_ms: int) -> str:
    seconds = youtube_seconds(start_ms)
    video_id = metadata.get("video_id")
    if video_id:
        return f"https://youtu.be/{video_id}?t={seconds}"
    url = str(metadata.get("webpage_url") or metadata.get("url") or "")
    if not url:
        return ""
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}t={seconds}"


def _should_show_segment(segment: dict[str, Any]) -> bool:
    metadata = segment.get("metadata") if isinstance(segment.get("metadata"), dict) else {}
    source = str(segment.get("primary_source") or segment.get("source") or "")
    if segment.get("conflict") and source not in TRANSCRIPT_SOURCES:
        return False
    if metadata.get("markdown_visible") is False or metadata.get("hidden") is True:
        return False
    if not str(segment.get("text") or "").strip():
        return False
    return True


def _is_ocr(segment: dict[str, Any]) -> bool:
    source = str(segment.get("source") or segment.get("primary_source") or "").lower()
    role = str(segment.get("role") or "").lower()
    return "ocr" in source or role == "ocr"


def _is_uncertain(segment: dict[str, Any]) -> bool:
    metadata = segment.get("metadata") if isinstance(segment.get("metadata"), dict) else {}
    if metadata.get("uncertain"):
        return True
    confidence = segment.get("confidence")
    if confidence is None:
        return False
    value = float(confidence)
    return 0.0 <= value < 0.5


def _ocr_events(rows: list[dict[str, Any]], *, assume_ocr: bool = False) -> list[dict[str, Any]]:
    return [row for row in _sorted_rows(rows) if (assume_ocr or _is_ocr(row)) and _should_show_segment(row)]


def _ocr_region(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(metadata.get("region") or metadata.get("roi") or row.get("role") or "unknown")


def _diagnostics_section(
    diagnostics: dict[str, Any],
    segments: list[dict[str, Any]],
    raw_asr: list[dict[str, Any]],
    raw_subtitles: list[dict[str, Any]],
    raw_ocr: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "## Processing Diagnostics",
        "",
        f"- ASR segments: {_nested(diagnostics, 'asr', 'segments', default=len(raw_asr))}",
        f"- Subtitle segments: {_nested(diagnostics, 'subtitle', 'segments', default=len(raw_subtitles))}",
        f"- OCR events: {_nested(diagnostics, 'ocr', 'deduped_events', default=len(_ocr_events(raw_ocr, assume_ocr=True)))}",
        f"- Fused segments: {_nested(diagnostics, 'fusion', 'fused_segments', default=len(segments))}",
    ]
    conflicts = _nested(diagnostics, "fusion", "conflicts")
    if conflicts:
        lines.append(f"- Conflicts: {conflicts}")
    errors = diagnostics.get("errors")
    if errors:
        lines.append(f"- Errors: {_format_errors(errors)}")
    return lines


def _format_errors(errors: Any) -> str:
    if isinstance(errors, list):
        return "; ".join(str(item) for item in errors)
    if isinstance(errors, dict):
        return "; ".join(f"{key}: {value}" for key, value in errors.items())
    return str(errors)


def _channel(metadata: dict[str, Any]) -> str:
    return str(metadata.get("channel_name") or metadata.get("channel") or metadata.get("uploader") or "")


def _duration(metadata: dict[str, Any]) -> str:
    value = metadata.get("duration_sec", metadata.get("duration", 0))
    if isinstance(value, str) and ":" in value:
        return value
    try:
        return seconds_to_timecode(float(value or 0), include_millis=False)
    except (TypeError, ValueError):
        return "00:00:00"


def _timecode(row: dict[str, Any]) -> str:
    return ms_to_timecode(int(row.get("start_ms") or 0), include_millis=True)


def _normalize_upload_date(value: Any) -> Any:
    raw = str(value or "")
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return value


def _asr_engine(diagnostics: dict[str, Any], manifest: Manifest | None) -> str | None:
    value = _nested(diagnostics, "asr", "engine")
    if value:
        return str(value)
    model = manifest.models.get("asr") if manifest else None
    return str(model).split(":", 1)[0] if model else None


def _asr_model(diagnostics: dict[str, Any], manifest: Manifest | None) -> str | None:
    value = _nested(diagnostics, "asr", "model")
    if value:
        return str(value)
    model = manifest.models.get("asr") if manifest else None
    if model and ":" in model:
        return str(model).split(":", 1)[1]
    return None


def _ocr_engine(diagnostics: dict[str, Any], manifest: Manifest | None) -> str | None:
    value = _nested(diagnostics, "ocr", "engine")
    if value:
        return str(value)
    model = manifest.models.get("ocr") if manifest else None
    return str(model) if model else None


def _subtitle_source(diagnostics: dict[str, Any], segments: list[dict[str, Any]]) -> str | None:
    source_type = _nested(diagnostics, "subtitle", "type")
    if source_type == "manual":
        return "manual_subtitle"
    if source_type == "auto":
        return "auto_subtitle"
    for segment in segments:
        source = segment.get("primary_source") or segment.get("source")
        if source in {"manual_subtitle", "auto_subtitle", "asr"}:
            return str(source)
    return None


def _language(metadata: dict[str, Any], segments: list[dict[str, Any]]) -> str | None:
    if metadata.get("language"):
        return str(metadata["language"])
    for segment in segments:
        if segment.get("language"):
            return str(segment["language"])
    return None


def _nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _config_get(config: Any, dotted: str, default: Any) -> Any:
    if isinstance(config, dict):
        return _nested(config, *dotted.split("."), default=default)
    if hasattr(config, "get"):
        try:
            return config.get(dotted, default)
        except TypeError:
            return default
    return default


def _yaml_value(value: Any, *, quote: bool) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if quote:
        return json.dumps(text, ensure_ascii=False)
    return text or "null"


def _escape_table(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|").strip()


def _iso_now() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")
