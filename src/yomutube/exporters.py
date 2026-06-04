from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from yomutube.archive_assets import read_json_for_display, read_jsonl_for_display, read_speaker_aliases
from yomutube.state import atomic_write_text, iso_now
from yomutube.utils.timecode import ms_to_timecode, youtube_seconds


@dataclass(slots=True)
class ExportResult:
    output_path: Path
    profile_path: Path
    format: str


EXPORT_EXTENSIONS = {
    "srt": "srt",
    "vtt": "vtt",
    "txt": "txt",
    "csv": "csv",
    "obsidian": "md",
    "quotes": "md",
    "topics": "md",
    "speakers": "md",
}


def export_archive(archive_dir: str | Path, format: str, *, output: str | Path | None = None) -> ExportResult:
    archive = Path(archive_dir)
    fmt = format.strip().lower()
    if fmt not in EXPORT_EXTENSIONS:
        raise ValueError(f"unsupported export format: {format}")

    metadata = _read_metadata(archive)
    segments = _transcript_segments(read_jsonl_for_display(archive / "segments.jsonl"))
    topics = read_jsonl_for_display(archive / "topics.jsonl")
    speaker_turns = read_jsonl_for_display(archive / "speaker_turns.jsonl")
    speaker_aliases = read_speaker_aliases(archive)

    if fmt == "srt":
        content = export_srt(metadata, segments, speaker_aliases=speaker_aliases)
    elif fmt == "vtt":
        content = export_vtt(metadata, segments, speaker_aliases=speaker_aliases)
    elif fmt == "txt":
        content = export_txt(metadata, segments, speaker_aliases=speaker_aliases)
    elif fmt == "csv":
        content = export_csv(metadata, segments, speaker_aliases=speaker_aliases)
    elif fmt == "obsidian":
        content = export_obsidian(metadata, segments, topics=topics, speaker_aliases=speaker_aliases)
    elif fmt == "quotes":
        content = export_quotes(metadata, segments, speaker_aliases=speaker_aliases)
    elif fmt == "topics":
        content = export_topics(metadata, topics)
    else:
        content = export_speakers(metadata, segments, speaker_turns=speaker_turns, speaker_aliases=speaker_aliases)

    exports_dir = archive / "exports"
    target = Path(output) if output is not None else exports_dir / f"{fmt}.{EXPORT_EXTENSIONS[fmt]}"
    profile_path = exports_dir / f"{fmt}.json"
    atomic_write_text(target, content if content.endswith("\n") else content + "\n")
    _write_export_profile(
        profile_path,
        archive=archive,
        output_path=target,
        format=fmt,
        ext=EXPORT_EXTENSIONS[fmt],
        segment_count=len(segments),
    )
    return ExportResult(output_path=target, profile_path=profile_path, format=fmt)


def export_srt(metadata: Mapping[str, Any], segments: Iterable[Mapping[str, Any]], *, speaker_aliases: Mapping[str, str] | None = None) -> str:
    lines: list[str] = []
    for index, segment in enumerate(_transcript_segments(segments), start=1):
        lines.append(str(index))
        lines.append(f"{_srt_time(segment.get('start_ms'))} --> {_srt_time(segment.get('end_ms'))}")
        lines.append(_display_text(segment, speaker_aliases=speaker_aliases))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_vtt(metadata: Mapping[str, Any], segments: Iterable[Mapping[str, Any]], *, speaker_aliases: Mapping[str, str] | None = None) -> str:
    lines = ["WEBVTT", ""]
    for segment in _transcript_segments(segments):
        lines.append(f"{_vtt_time(segment.get('start_ms'))} --> {_vtt_time(segment.get('end_ms'))}")
        lines.append(_display_text(segment, speaker_aliases=speaker_aliases))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_txt(metadata: Mapping[str, Any], segments: Iterable[Mapping[str, Any]], *, speaker_aliases: Mapping[str, str] | None = None) -> str:
    return "".join(
        f"[{ms_to_timecode(int(segment.get('start_ms') or 0), include_millis=True)}] {_display_text(segment, speaker_aliases=speaker_aliases)}\n"
        for segment in _transcript_segments(segments)
    )


def export_csv(metadata: Mapping[str, Any], segments: Iterable[Mapping[str, Any]], *, speaker_aliases: Mapping[str, str] | None = None) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["segment_id", "video_id", "start_ms", "end_ms", "timestamp", "speaker", "source", "text", "youtube_url"],
    )
    writer.writeheader()
    for segment in _transcript_segments(segments):
        writer.writerow(
            {
                "segment_id": segment.get("id"),
                "video_id": segment.get("video_id") or metadata.get("video_id"),
                "start_ms": segment.get("start_ms"),
                "end_ms": segment.get("end_ms"),
                "timestamp": ms_to_timecode(int(segment.get("start_ms") or 0), include_millis=True),
                "speaker": _speaker_label(segment.get("speaker"), speaker_aliases),
                "source": segment.get("primary_source") or segment.get("source"),
                "text": segment.get("text"),
                "youtube_url": _youtube_url(metadata, segment),
            }
        )
    return buffer.getvalue()


def export_obsidian(
    metadata: Mapping[str, Any],
    segments: Iterable[Mapping[str, Any]],
    *,
    topics: Iterable[Mapping[str, Any]] | None = None,
    speaker_aliases: Mapping[str, str] | None = None,
) -> str:
    lines = [
        "---",
        f"video_id: {metadata.get('video_id') or 'unknown'}",
        f'title: {json.dumps(str(metadata.get("title") or "Untitled"), ensure_ascii=False)}',
        f'channel: {json.dumps(str(metadata.get("channel_name") or metadata.get("channel") or ""), ensure_ascii=False)}',
        "source: youtube",
        "---",
        "",
        f"# {metadata.get('title') or 'Untitled'}",
        "",
    ]
    topic_rows = list(topics or [])
    if topic_rows:
        lines.extend(["## Topics", ""])
        for topic in topic_rows:
            lines.append(f"- {_markdown_time_link(metadata, topic)} {topic.get('title') or ''}".rstrip())
        lines.append("")
    lines.extend(["## Transcript", ""])
    for segment in _transcript_segments(segments):
        lines.append(f"- {_markdown_time_link(metadata, segment)} {_display_text(segment, speaker_aliases=speaker_aliases)}")
    return "\n".join(lines).rstrip() + "\n"


def export_quotes(metadata: Mapping[str, Any], segments: Iterable[Mapping[str, Any]], *, speaker_aliases: Mapping[str, str] | None = None) -> str:
    lines = [f"# Quotes: {metadata.get('title') or 'Untitled'}", ""]
    for segment in _transcript_segments(segments):
        lines.append(f"> {_markdown_time_link(metadata, segment)} {_display_text(segment, speaker_aliases=speaker_aliases)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_topics(metadata: Mapping[str, Any], topics: Iterable[Mapping[str, Any]]) -> str:
    lines = [
        f"# Topics: {metadata.get('title') or 'Untitled'}",
        "",
        "| Start | End | Title | Summary |",
        "|---:|---:|---|---|",
    ]
    for topic in topics:
        lines.append(
            f"| {ms_to_timecode(int(topic.get('start_ms') or 0), include_millis=True)} | "
            f"{ms_to_timecode(int(topic.get('end_ms') or 0), include_millis=True)} | "
            f"{_escape_table(str(topic.get('title') or ''))} | {_escape_table(str(topic.get('summary') or ''))} |"
        )
    return "\n".join(lines) + "\n"


def export_speakers(
    metadata: Mapping[str, Any],
    segments: Iterable[Mapping[str, Any]],
    *,
    speaker_turns: Iterable[Mapping[str, Any]] | None = None,
    speaker_aliases: Mapping[str, str] | None = None,
) -> str:
    stats: dict[str, dict[str, Any]] = {}
    for segment in _transcript_segments(segments):
        speaker = segment.get("speaker")
        if not speaker:
            continue
        row = stats.setdefault(str(speaker), {"count": 0, "duration_ms": 0})
        row["count"] += 1
        row["duration_ms"] += max(0, int(segment.get("end_ms") or 0) - int(segment.get("start_ms") or 0))
    turn_durations: dict[str, int] = {}
    for turn in speaker_turns or []:
        speaker = turn.get("speaker")
        if not speaker:
            continue
        key = str(speaker)
        turn_durations[key] = turn_durations.get(key, 0) + max(0, int(turn.get("end_ms") or 0) - int(turn.get("start_ms") or 0))
        stats.setdefault(key, {"count": 0, "duration_ms": 0})
    for speaker, duration_ms in turn_durations.items():
        stats[speaker]["duration_ms"] = duration_ms

    lines = [
        f"# Speakers: {metadata.get('title') or 'Untitled'}",
        "",
        "| Speaker | Alias | Segments | Duration |",
        "|---|---|---:|---:|",
    ]
    for speaker in sorted(stats):
        row = stats[speaker]
        lines.append(
            f"| {speaker} | {_speaker_label(speaker, speaker_aliases) or ''} | {row['count']} | {ms_to_timecode(row['duration_ms'], include_millis=True)} |"
        )
    return "\n".join(lines) + "\n"


def quote_at(archive_dir: str | Path, at_ms: int) -> str:
    archive = Path(archive_dir)
    metadata = _read_metadata(archive)
    speaker_aliases = read_speaker_aliases(archive)
    segments = _transcript_segments(read_jsonl_for_display(archive / "segments.jsonl"))
    if not segments:
        raise ValueError(f"archive has no transcript segments: {archive}")
    target = min(segments, key=lambda row: _distance_to_segment(row, int(at_ms)))
    return (
        f"> {_markdown_time_link(metadata, target)} {_display_text(target, speaker_aliases=speaker_aliases)}\n\n"
        f"`segment_id: {target.get('id')}`\n"
    )


def _write_export_profile(
    profile_path: Path,
    *,
    archive: Path,
    output_path: Path,
    format: str,
    ext: str,
    segment_count: int,
) -> None:
    payload = {
        "schema_version": "1.0",
        "format": format,
        "ext": ext,
        "created_at": iso_now(),
        "input_archive": str(archive),
        "output_path": str(output_path),
        "segment_count": segment_count,
        "source_files": sorted(path.name for path in archive.iterdir() if path.is_file()),
    }
    atomic_write_text(profile_path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _read_metadata(archive: Path) -> dict[str, Any]:
    data = read_json_for_display(archive / "metadata.json", default={})
    return data if isinstance(data, dict) else {}


def _transcript_segments(segments: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        row = dict(segment)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if metadata.get("hidden") is True or metadata.get("markdown_visible") is False:
            continue
        if _is_ocr(row):
            continue
        if not str(row.get("text") or "").strip():
            continue
        rows.append(row)
    return sorted(rows, key=lambda row: (int(row.get("start_ms") or 0), str(row.get("id") or "")))


def _display_text(segment: Mapping[str, Any], *, speaker_aliases: Mapping[str, str] | None = None) -> str:
    text = str(segment.get("text") or "").strip()
    speaker = _speaker_label(segment.get("speaker"), speaker_aliases)
    return f"{speaker}: {text}" if speaker else text


def _speaker_label(speaker: Any, aliases: Mapping[str, str] | None) -> str:
    if not speaker:
        return ""
    raw = str(speaker)
    return str((aliases or {}).get(raw) or raw)


def _markdown_time_link(metadata: Mapping[str, Any], segment: Mapping[str, Any]) -> str:
    start_ms = int(segment.get("start_ms") or 0)
    timecode = ms_to_timecode(start_ms, include_millis=True)
    return f"[{timecode}]({_youtube_url(metadata, segment)})"


def _youtube_url(metadata: Mapping[str, Any], segment: Mapping[str, Any]) -> str:
    start_ms = int(segment.get("start_ms") or 0)
    video_id = segment.get("video_id") or metadata.get("video_id")
    if video_id:
        return f"https://youtu.be/{video_id}?t={youtube_seconds(start_ms)}"
    url = str(metadata.get("webpage_url") or metadata.get("url") or "")
    if not url:
        return ""
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}t={youtube_seconds(start_ms)}"


def _srt_time(value: Any) -> str:
    return ms_to_timecode(int(value or 0), include_millis=True).replace(".", ",")


def _vtt_time(value: Any) -> str:
    return ms_to_timecode(int(value or 0), include_millis=True)


def _is_ocr(row: Mapping[str, Any]) -> bool:
    source = str(row.get("source") or row.get("primary_source") or "").lower()
    role = str(row.get("role") or "").lower()
    return "ocr" in source or role == "ocr"


def _distance_to_segment(segment: Mapping[str, Any], at_ms: int) -> int:
    start_ms = int(segment.get("start_ms") or 0)
    end_ms = int(segment.get("end_ms") or start_ms)
    if start_ms <= at_ms <= end_ms:
        return 0
    return min(abs(at_ms - start_ms), abs(at_ms - end_ms))


def _escape_table(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|").strip()
