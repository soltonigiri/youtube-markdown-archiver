from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from yomutube.models import Chapter, Segment
from yomutube.state import atomic_write_text, iso_now
from yomutube.utils.timecode import ms_to_timecode
from yomutube.writers.jsonl import write_jsonl


URL_RE = re.compile(r"https?://[^\s<>)\"']+")


@dataclass(slots=True)
class Correction:
    target_segment_id: str
    operation: str
    before: str
    after: str
    reason: str = "manual_correction"
    created_at: str = field(default_factory=iso_now)
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QualityReport:
    overall_score: float
    subtitle_coverage: float
    asr_avg_confidence: float | None
    conflict_rate: float
    speaker_coverage: float
    ocr_event_count: int
    timeline_gaps: list[dict[str, Any]]
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_json_for_display(path: str | Path, *, default: Any = None) -> Any:
    source = Path(path)
    if not source.exists():
        return {} if default is None else default
    try:
        with source.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {} if default is None else default


def read_jsonl_for_display(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except (OSError, UnicodeDecodeError):
        return []
    return rows


def read_speaker_aliases(archive_dir: str | Path) -> dict[str, str]:
    data = read_json_for_display(Path(archive_dir) / "speaker_aliases.json", default={})
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items() if value is not None}


def write_speaker_aliases(archive_dir: str | Path, aliases: Mapping[str, str]) -> None:
    target = Path(archive_dir) / "speaker_aliases.json"
    payload = {str(key): str(value) for key, value in aliases.items() if value is not None}
    atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def append_correction(path: str | Path, correction: Correction | Mapping[str, Any]) -> None:
    target = Path(path)
    rows = read_jsonl_for_display(target)
    payload = correction.to_dict() if isinstance(correction, Correction) else dict(correction)
    payload.setdefault("schema_version", "1.0")
    rows.append(payload)
    write_jsonl(target, rows)


def append_archive_correction(
    archive_dir: str | Path,
    *,
    target_segment_id: str,
    after: str,
    before: str = "",
    reason: str = "manual_correction",
) -> Correction:
    correction = Correction(
        target_segment_id=target_segment_id,
        operation="replace_text",
        before=before,
        after=after,
        reason=reason,
    )
    append_correction(Path(archive_dir) / "corrections.jsonl", correction)
    return correction


def apply_corrections_for_display(
    segments: Iterable[Segment],
    corrections: Iterable[Mapping[str, Any]],
) -> list[Segment]:
    replacements: dict[str, Mapping[str, Any]] = {}
    for row in corrections:
        if row.get("operation") == "replace_text" and row.get("target_segment_id"):
            replacements[str(row["target_segment_id"])] = row

    result: list[Segment] = []
    for segment in segments:
        correction = replacements.get(segment.id)
        if not correction:
            result.append(segment)
            continue
        metadata = dict(segment.metadata)
        metadata["display_correction"] = {
            "operation": correction.get("operation"),
            "reason": correction.get("reason"),
            "created_at": correction.get("created_at"),
            "before": correction.get("before"),
        }
        result.append(replace(segment, text=str(correction.get("after") or ""), metadata=metadata))
    return result


def word_rows_from_segment(segment: Segment | Mapping[str, Any]) -> list[dict[str, Any]]:
    row = _segment_dict(segment)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    words = metadata.get("words")
    if not isinstance(words, list):
        return []
    result: list[dict[str, Any]] = []
    for index, word in enumerate(words, start=1):
        if not isinstance(word, Mapping):
            continue
        text = str(word.get("word") or word.get("text") or "").strip()
        if not text:
            continue
        start_ms = _word_ms(word, "start_ms", "start", default=int(row.get("start_ms") or 0))
        end_ms = _word_ms(word, "end_ms", "end", default=start_ms)
        result.append(
            {
                "schema_version": "1.0",
                "id": f"{row.get('id')}_word_{index:04d}",
                "video_id": row.get("video_id"),
                "segment_id": row.get("id"),
                "word": text,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "speaker": row.get("speaker"),
                "confidence": word.get("probability", row.get("confidence")),
            }
        )
    return result


def word_rows_from_segments(segments: Iterable[Segment | Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        rows.extend(word_rows_from_segment(segment))
    return sorted(rows, key=lambda row: (int(row.get("start_ms") or 0), str(row.get("id") or "")))


def speaker_turn_row_from_turn(turn: Any, *, aliases: Mapping[str, str] | None = None) -> dict[str, Any]:
    aliases = aliases or {}
    row = _object_dict(turn)
    speaker = str(row.get("speaker") or "SPEAKER_UNKNOWN")
    start_ms = int(row.get("start_ms") or round(float(row.get("start", 0.0)) * 1000))
    end_ms = int(row.get("end_ms") or round(float(row.get("end", 0.0)) * 1000))
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "schema_version": "1.0",
        "speaker": speaker,
        "alias": aliases.get(speaker),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "confidence": row.get("confidence"),
        "metadata": dict(metadata),
    }


def speaker_turn_rows_from_turns(turns: Iterable[Any], *, aliases: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    return sorted(
        [speaker_turn_row_from_turn(turn, aliases=aliases) for turn in turns],
        key=lambda row: (int(row.get("start_ms") or 0), str(row.get("speaker") or "")),
    )


def speaker_turn_rows_from_segments(
    segments: Iterable[Segment | Mapping[str, Any]],
    *,
    aliases: Mapping[str, str] | None = None,
    merge_gap_ms: int = 700,
) -> list[dict[str, Any]]:
    aliases = aliases or {}
    rows: list[dict[str, Any]] = []
    for row in sorted((_segment_dict(segment) for segment in segments), key=lambda item: (int(item.get("start_ms") or 0), int(item.get("end_ms") or 0))):
        speaker = row.get("speaker")
        if not speaker:
            continue
        speaker = str(speaker)
        start_ms = int(row.get("start_ms") or 0)
        end_ms = int(row.get("end_ms") or start_ms)
        if rows and rows[-1]["speaker"] == speaker and 0 <= start_ms - int(rows[-1]["end_ms"]) <= merge_gap_ms:
            rows[-1]["end_ms"] = max(int(rows[-1]["end_ms"]), end_ms)
            rows[-1]["metadata"].setdefault("segment_ids", []).append(row.get("id"))
            continue
        rows.append(
            {
                "schema_version": "1.0",
                "speaker": speaker,
                "alias": aliases.get(speaker),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "confidence": row.get("confidence"),
                "metadata": {"segment_ids": [row.get("id")]},
            }
        )
    return rows


def build_quality_report(
    segments: Iterable[Segment | Mapping[str, Any]],
    *,
    duration_ms: int | None = None,
    duration_sec: float | None = None,
) -> QualityReport:
    rows = [_segment_dict(segment) for segment in segments]
    duration = int(duration_ms if duration_ms is not None else round(float(duration_sec or 0) * 1000))
    transcript_rows = [row for row in rows if not _is_ocr_row(row)]
    subtitle_rows = [row for row in rows if str(row.get("source") or row.get("primary_source") or "") in {"manual_subtitle", "auto_subtitle"}]
    asr_rows = [row for row in rows if str(row.get("source") or row.get("primary_source") or "") == "asr"]

    subtitle_coverage = _coverage(subtitle_rows, duration)
    asr_confidences = [_float(row.get("confidence")) for row in asr_rows if row.get("confidence") is not None]
    asr_avg_confidence = sum(asr_confidences) / len(asr_confidences) if asr_confidences else None
    conflict_rate = _ratio(sum(1 for row in rows if row.get("conflict")), len(rows))
    speaker_coverage = _ratio(sum(_duration(row) for row in transcript_rows if row.get("speaker")), sum(_duration(row) for row in transcript_rows))
    ocr_event_count = sum(1 for row in rows if _is_ocr_row(row))
    timeline_gaps = _timeline_gaps(transcript_rows, duration)

    score = 0.55
    score += subtitle_coverage * 0.15
    score += speaker_coverage * 0.10
    score += min(1.0, len(transcript_rows) / max(1.0, duration / 30_000)) * 0.10 if duration else 0.05
    score -= conflict_rate * 0.20
    score -= min(0.20, len(timeline_gaps) * 0.03)
    overall_score = round(max(0.0, min(1.0, score)), 4)

    return QualityReport(
        overall_score=overall_score,
        subtitle_coverage=round(subtitle_coverage, 4),
        asr_avg_confidence=round(asr_avg_confidence, 4) if asr_avg_confidence is not None else None,
        conflict_rate=round(conflict_rate, 4),
        speaker_coverage=round(speaker_coverage, 4),
        ocr_event_count=ocr_event_count,
        timeline_gaps=timeline_gaps,
    )


def alignment_rows_from_segments(segments: Iterable[Segment | Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        primary = _segment_dict(segment)
        alternatives = primary.get("alternatives") if isinstance(primary.get("alternatives"), list) else []
        for alternative in alternatives:
            if not isinstance(alternative, Mapping):
                continue
            rows.append(
                {
                    "schema_version": "1.0",
                    "primary_segment_id": primary.get("id"),
                    "primary_source": primary.get("primary_source") or primary.get("source"),
                    "alternative_segment_id": alternative.get("id"),
                    "alternative_source": alternative.get("source"),
                    "video_id": primary.get("video_id"),
                    "start_ms": min(int(primary.get("start_ms") or 0), int(alternative.get("start_ms") or primary.get("start_ms") or 0)),
                    "end_ms": max(int(primary.get("end_ms") or 0), int(alternative.get("end_ms") or primary.get("end_ms") or 0)),
                    "similarity": alternative.get("similarity"),
                }
            )
    return rows


def conflict_rows_from_segments(segments: Iterable[Segment | Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        row = _segment_dict(segment)
        alternatives = [dict(item) for item in row.get("alternatives", []) if isinstance(item, Mapping)]
        if not row.get("conflict") and not any(_float(item.get("similarity")) < 0.82 for item in alternatives):
            continue
        rows.append(
            {
                "schema_version": "1.0",
                "segment_id": row.get("id"),
                "video_id": row.get("video_id"),
                "start_ms": row.get("start_ms"),
                "end_ms": row.get("end_ms"),
                "source": row.get("primary_source") or row.get("source"),
                "text": row.get("text"),
                "alternatives": alternatives,
                "conflict_type": "candidate_difference",
            }
        )
    return rows


def topic_rows_from_chapters(video_id: str, chapters: Iterable[Chapter | Mapping[str, Any]], *, duration_ms: int = 0) -> list[dict[str, Any]]:
    chapter_rows = [_object_dict(chapter) for chapter in chapters]
    rows: list[dict[str, Any]] = []
    for index, chapter in enumerate(chapter_rows, start=1):
        start_ms = round(float(chapter.get("start_time") or chapter.get("start_sec") or 0) * 1000)
        if chapter.get("end_time") is not None:
            end_ms = round(float(chapter.get("end_time") or 0) * 1000)
        elif index < len(chapter_rows):
            end_ms = round(float(chapter_rows[index].get("start_time") or chapter_rows[index].get("start_sec") or 0) * 1000)
        else:
            end_ms = int(duration_ms)
        rows.append(
            {
                "schema_version": "1.0",
                "id": f"topic_{index:04d}",
                "topic_id": f"topic_{index:04d}",
                "video_id": video_id,
                "title": str(chapter.get("title") or f"Topic {index}"),
                "start_ms": start_ms,
                "end_ms": max(start_ms, end_ms),
                "summary": str(chapter.get("summary") or ""),
                "keywords": list(chapter.get("keywords") or []),
            }
        )
    return rows


def entity_rows_from_segments(
    segments: Iterable[Segment | Mapping[str, Any]],
    *,
    terms: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    term_values = [term for term in (terms or []) if str(term).strip()]
    for segment in segments:
        row = _segment_dict(segment)
        text = str(row.get("text") or "")
        for url in URL_RE.findall(text):
            _add_entity(buckets, url.rstrip(".,、。"), "url", row)
        for term in term_values:
            if str(term).casefold() in text.casefold():
                _add_entity(buckets, str(term), "term", row)
    return sorted(buckets.values(), key=lambda row: (row["kind"], row["entity"]))


def visual_text_rows_from_segments(segments: Iterable[Segment | Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        row = _segment_dict(segment)
        if not _is_ocr_row(row):
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        rows.append(
            {
                "schema_version": "1.0",
                "id": row.get("id"),
                "video_id": row.get("video_id"),
                "start_ms": row.get("start_ms"),
                "end_ms": row.get("end_ms"),
                "text": row.get("text"),
                "region": metadata.get("region") or metadata.get("roi") or row.get("role") or "unknown",
                "role": row.get("role") or metadata.get("role"),
                "confidence": row.get("confidence"),
                "metadata": dict(metadata),
            }
        )
    return rows


def slide_rows_from_visual_text(visual_rows: Iterable[Mapping[str, Any]], *, merge_gap_ms: int = 1000) -> list[dict[str, Any]]:
    rows = sorted([dict(row) for row in visual_rows], key=lambda row: (int(row.get("start_ms") or 0), int(row.get("end_ms") or 0)))
    slides: list[dict[str, Any]] = []
    for row in rows:
        start_ms = int(row.get("start_ms") or 0)
        end_ms = int(row.get("end_ms") or start_ms)
        if slides and row.get("text") == slides[-1].get("text") and row.get("region") == slides[-1].get("region") and 0 <= start_ms - int(slides[-1].get("end_ms") or 0) <= merge_gap_ms:
            slides[-1]["end_ms"] = max(int(slides[-1]["end_ms"]), end_ms)
            slides[-1]["source_ids"].append(row.get("id"))
            continue
        index = len(slides) + 1
        slides.append(
            {
                "schema_version": "1.0",
                "id": f"slide_{index:04d}",
                "video_id": row.get("video_id"),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": row.get("text"),
                "region": row.get("region"),
                "source_ids": [row.get("id")],
            }
        )
    return slides


def build_archive_asset_rows(
    *,
    metadata: Mapping[str, Any],
    segments: Iterable[Segment | Mapping[str, Any]],
    raw_asr: Iterable[Segment | Mapping[str, Any]] | None = None,
    raw_ocr: Iterable[Segment | Mapping[str, Any]] | None = None,
    speaker_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    segment_rows = list(segments)
    asr_rows = list(raw_asr or [])
    ocr_rows = list(raw_ocr or [])
    aliases = dict(speaker_aliases or {})
    duration_ms = round(float(metadata.get("duration_sec") or metadata.get("duration") or 0) * 1000)
    video_id = str(metadata.get("video_id") or "unknown")
    chapters = metadata.get("chapters") if isinstance(metadata.get("chapters"), list) else []
    visual_rows = visual_text_rows_from_segments(ocr_rows or segment_rows)
    return {
        "words": word_rows_from_segments(asr_rows),
        "speaker_turns": speaker_turn_rows_from_segments(asr_rows or segment_rows, aliases=aliases),
        "alignment": alignment_rows_from_segments(segment_rows),
        "conflicts": conflict_rows_from_segments(segment_rows),
        "topics": topic_rows_from_chapters(video_id, chapters, duration_ms=duration_ms),
        "entities": entity_rows_from_segments(segment_rows),
        "visual_text": visual_rows,
        "slides": slide_rows_from_visual_text(visual_rows),
        "quality": build_quality_report(segment_rows, duration_ms=duration_ms).to_dict(),
    }


def _word_ms(word: Mapping[str, Any], ms_key: str, sec_key: str, *, default: int) -> int:
    if word.get(ms_key) is not None:
        return int(round(float(word[ms_key])))
    if word.get(sec_key) is not None:
        return int(round(float(word[sec_key]) * 1000))
    return int(default)


def _segment_dict(segment: Segment | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(segment, Segment):
        return segment.to_dict()
    return dict(segment)


def _object_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_") and not callable(getattr(value, key))
    }


def _is_ocr_row(row: Mapping[str, Any]) -> bool:
    source = str(row.get("source") or row.get("primary_source") or "").lower()
    role = str(row.get("role") or "").lower()
    return "ocr" in source or role == "ocr"


def _coverage(rows: list[Mapping[str, Any]], duration_ms: int) -> float:
    if duration_ms <= 0:
        return 0.0
    return min(1.0, sum(_duration(row) for row in rows) / duration_ms)


def _duration(row: Mapping[str, Any]) -> int:
    return max(0, int(row.get("end_ms") or 0) - int(row.get("start_ms") or 0))


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _timeline_gaps(rows: list[Mapping[str, Any]], duration_ms: int, *, min_gap_ms: int = 5000) -> list[dict[str, Any]]:
    if duration_ms <= 0:
        return []
    sorted_rows = sorted(rows, key=lambda row: (int(row.get("start_ms") or 0), int(row.get("end_ms") or 0)))
    gaps: list[dict[str, Any]] = []
    cursor = 0
    for row in sorted_rows:
        start_ms = int(row.get("start_ms") or 0)
        if start_ms - cursor >= min_gap_ms:
            gaps.append({"start_ms": cursor, "end_ms": start_ms, "reason": "no_transcript"})
        cursor = max(cursor, int(row.get("end_ms") or start_ms))
    if duration_ms - cursor >= min_gap_ms:
        gaps.append({"start_ms": cursor, "end_ms": duration_ms, "reason": "no_transcript"})
    return gaps


def _add_entity(buckets: dict[tuple[str, str], dict[str, Any]], entity: str, kind: str, segment: Mapping[str, Any]) -> None:
    key = (entity, kind)
    row = buckets.setdefault(
        key,
        {
            "schema_version": "1.0",
            "entity": entity,
            "kind": kind,
            "type": kind,
            "count": 0,
            "occurrences": [],
        },
    )
    row["count"] += 1
    row["occurrences"].append(
        {
            "segment_id": segment.get("id"),
            "video_id": segment.get("video_id"),
            "start_ms": segment.get("start_ms"),
            "timestamp": ms_to_timecode(int(segment.get("start_ms") or 0), include_millis=True),
        }
    )
