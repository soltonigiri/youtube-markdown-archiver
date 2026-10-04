from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

from yomutube.models import Segment, SubtitleTrack
from yomutube.subtitle.normalizer import is_empty_subtitle_text, normalize_subtitle_text
from yomutube.utils.hashing import short_hash
from yomutube.utils.timecode import parse_timecode


TIME_VALUE_RE = r"(?:\d+:)?\d{1,2}:\d{2}(?:[,.]\d{1,3})?"
CUE_TIMING_RE = re.compile(rf"(?P<start>{TIME_VALUE_RE})\s*-->\s*(?P<end>{TIME_VALUE_RE})")
TTML_P_RE = re.compile(r"<p\b(?P<attrs>[^>]*)>(?P<text>.*?)</p>", re.I | re.S)
ATTR_RE = re.compile(r"(?P<name>begin|end|dur|start)\s*=\s*['\"](?P<value>[^'\"]+)['\"]", re.I)


class SubtitleParseError(ValueError):
    pass


def _split_blocks(content: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw_line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip("\ufeff")
        if line.strip():
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _parse_cue_blocks(content: str) -> list[tuple[int, int, str]]:
    cues: list[tuple[int, int, str]] = []
    for block in _split_blocks(content):
        if not block:
            continue
        first = block[0].strip()
        if first.upper().startswith(("NOTE", "STYLE", "REGION")):
            continue
        timing_index = next((index for index, line in enumerate(block) if "-->" in line), None)
        if timing_index is None:
            continue
        match = CUE_TIMING_RE.search(block[timing_index])
        if match is None:
            continue
        text = "\n".join(block[timing_index + 1 :])
        cues.append((parse_timecode(match.group("start")), parse_timecode(match.group("end")), text))
    return cues


def parse_vtt(content: str) -> list[tuple[int, int, str]]:
    return _parse_cue_blocks(content)


def parse_srt(content: str) -> list[tuple[int, int, str]]:
    return _parse_cue_blocks(content)


def parse_json3(content: str) -> list[tuple[int, int, str]]:
    data = json.loads(content)
    events = data.get("events") if isinstance(data, dict) else None
    if not isinstance(events, list):
        raise SubtitleParseError("json3 subtitle must contain an events list")
    cues: list[tuple[int, int, str]] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        segments = event.get("segs") or []
        if not isinstance(segments, list):
            continue
        text = "".join(str(seg.get("utf8") or "") for seg in segments if isinstance(seg, dict))
        start_ms = int(float(event.get("tStartMs") or 0))
        duration = event.get("dDurationMs")
        if duration is not None:
            end_ms = start_ms + int(float(duration))
        elif index + 1 < len(events) and isinstance(events[index + 1], dict):
            end_ms = int(float(events[index + 1].get("tStartMs") or start_ms))
        else:
            end_ms = start_ms
        cues.append((start_ms, end_ms, text))
    return cues


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_ttml_time(value: str | None) -> int | None:
    if not value:
        return None
    raw = value.strip()
    if raw.endswith("ms"):
        return round(float(raw[:-2]) if raw[:-2] else 0)
    if raw.endswith("s"):
        return round(float(raw[:-1]) * 1000)
    if raw.endswith("m"):
        return round(float(raw[:-1]) * 60_000)
    if raw.endswith("h"):
        return round(float(raw[:-1]) * 3_600_000)
    if ":" in raw:
        return parse_timecode(raw)
    return round(float(raw) * 1000)


def parse_ttml(content: str) -> list[tuple[int, int, str]]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return _parse_ttml_regex(content)
    cues: list[tuple[int, int, str]] = []
    for element in root.iter():
        if _strip_namespace(element.tag) != "p":
            continue
        start_ms = _parse_ttml_time(element.attrib.get("begin") or element.attrib.get("start"))
        if start_ms is None:
            continue
        end_ms = _parse_ttml_time(element.attrib.get("end"))
        dur_ms = _parse_ttml_time(element.attrib.get("dur"))
        if end_ms is None:
            end_ms = start_ms + dur_ms if dur_ms is not None else start_ms
        text = "".join(element.itertext())
        cues.append((start_ms, end_ms, text))
    return cues


def _parse_ttml_regex(content: str) -> list[tuple[int, int, str]]:
    cues: list[tuple[int, int, str]] = []
    for match in TTML_P_RE.finditer(content):
        attrs = {item.group("name").lower(): item.group("value") for item in ATTR_RE.finditer(match.group("attrs"))}
        start_ms = _parse_ttml_time(attrs.get("begin") or attrs.get("start"))
        if start_ms is None:
            continue
        end_ms = _parse_ttml_time(attrs.get("end"))
        dur_ms = _parse_ttml_time(attrs.get("dur"))
        if end_ms is None:
            end_ms = start_ms + dur_ms if dur_ms is not None else start_ms
        cues.append((start_ms, end_ms, match.group("text")))
    return cues


def parse_srv3(content: str) -> list[tuple[int, int, str]]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []
    cues: list[tuple[int, int, str]] = []
    for element in root.iter():
        if _strip_namespace(element.tag) != "text":
            continue
        start_raw = element.attrib.get("start")
        if start_raw is None:
            continue
        start_ms = round(float(start_raw) * 1000)
        duration_ms = round(float(element.attrib.get("dur") or 0) * 1000)
        cues.append((start_ms, start_ms + duration_ms, "".join(element.itertext())))
    return cues


def _track_value(track: SubtitleTrack | None, attr: str, default: Any = None) -> Any:
    if track is None:
        return default
    return getattr(track, attr, default)


def _segment_source(track: SubtitleTrack | None, fallback: str | None = None) -> str:
    if track is not None:
        return track.segment_source
    return fallback or "subtitle"


def _video_id(track: SubtitleTrack | None, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    raw = _track_value(track, "raw", {}) or {}
    return str(raw.get("video_id") or raw.get("id") or "unknown")


def _to_segments(
    cues: Iterable[tuple[int, int, str]],
    *,
    track: SubtitleTrack | None = None,
    video_id: str | None = None,
    language: str | None = None,
    source: str | None = None,
) -> list[Segment]:
    actual_source = _segment_source(track, source)
    actual_language = language or _track_value(track, "language")
    actual_video_id = _video_id(track, video_id)
    segments: list[Segment] = []
    for start_ms, end_ms, text in cues:
        normalized = normalize_subtitle_text(text)
        if is_empty_subtitle_text(normalized):
            continue
        end_ms = max(int(start_ms), int(end_ms))
        segment_index = len(segments) + 1
        digest = short_hash(f"{actual_video_id}:{actual_source}:{start_ms}:{end_ms}:{normalized}", 8)
        segments.append(
            Segment(
                id=f"{actual_source}_{segment_index:06d}_{digest}",
                video_id=actual_video_id,
                source=actual_source,
                start_ms=int(start_ms),
                end_ms=end_ms,
                text=normalized,
                language=actual_language,
            )
        )
    return segments


def parse_subtitle_text(
    content: str,
    ext: str,
    *,
    track: SubtitleTrack | None = None,
    video_id: str | None = None,
    language: str | None = None,
    source: str | None = None,
) -> list[Segment]:
    normalized_ext = ext.lower().lstrip(".")
    if normalized_ext == "vtt":
        cues = parse_vtt(content)
    elif normalized_ext == "srt":
        cues = parse_srt(content)
    elif normalized_ext in {"json3", "json"}:
        cues = parse_json3(content)
    elif normalized_ext in {"ttml", "dfxp"}:
        cues = parse_ttml(content)
    elif normalized_ext in {"srv3", "xml"}:
        cues = parse_srv3(content) or parse_ttml(content)
    else:
        raise SubtitleParseError(f"unsupported subtitle format: {ext}")
    return _to_segments(cues, track=track, video_id=video_id, language=language, source=source)


def parse_subtitle_file(
    track_or_path: SubtitleTrack | str | Path,
    video_id: str | SubtitleTrack | None = None,
    track: SubtitleTrack | None = None,
    *,
    language: str | None = None,
    source: str | None = None,
) -> list[Segment]:
    if isinstance(video_id, SubtitleTrack) and track is None:
        track = video_id
        video_id = None
    actual_track = track_or_path if isinstance(track_or_path, SubtitleTrack) else track
    path = actual_track.path if isinstance(track_or_path, SubtitleTrack) else Path(track_or_path)
    if path is None:
        raise ValueError("subtitle track has no path")
    ext = (actual_track.ext if actual_track is not None and actual_track.ext else path.suffix).lower().lstrip(".")
    content = Path(path).read_text(encoding="utf-8")
    video_id_value = video_id if isinstance(video_id, str) else None
    return parse_subtitle_text(content, ext, track=actual_track, video_id=video_id_value, language=language, source=source)
