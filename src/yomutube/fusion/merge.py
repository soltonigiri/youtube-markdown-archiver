from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from yomutube.models import Segment
from yomutube.utils.text import normalize_text

from .align import best_overlap_match, suppress_ocr_duplicates
from .scoring import DEFAULT_PRIORITY, segment_text_similarity, source_rank


def merge_adjacent_segments(
    segments: Iterable[Segment],
    *,
    merge_gap_ms: int = 700,
    max_segment_chars: int = 180,
) -> list[Segment]:
    merged: list[Segment] = []
    for segment in sorted(segments, key=lambda item: (item.start_ms, item.end_ms, item.id)):
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        combined_text = normalize_text(f"{previous.text} {segment.text}")
        can_merge = (
            previous.source == segment.source
            and previous.speaker == segment.speaker
            and previous.role == segment.role
            and previous.duplicate_of is None
            and segment.duplicate_of is None
            and not previous.conflict
            and not segment.conflict
            and not previous.alternatives
            and not segment.alternatives
            and 0 <= segment.start_ms - previous.end_ms <= merge_gap_ms
            and len(combined_text) <= max_segment_chars
        )
        if not can_merge:
            merged.append(segment)
            continue
        metadata = dict(previous.metadata)
        metadata.update(segment.metadata)
        merged[-1] = replace(
            previous,
            end_ms=max(previous.end_ms, segment.end_ms),
            text=combined_text,
            confidence=_merge_confidence(previous.confidence, segment.confidence),
            metadata=metadata,
        )
    return merged


def fuse_segments(
    *,
    manual_subtitles: list[Segment] | None = None,
    auto_subtitles: list[Segment] | None = None,
    asr_segments: list[Segment] | None = None,
    ocr_segments: list[Segment] | None = None,
    config: Any = None,
) -> list[Segment]:
    priority = _config_get(config, "fusion.primary_text_priority", DEFAULT_PRIORITY)
    overlap_threshold = float(_config_get(config, "fusion.overlap_threshold", 0.50))
    duplicate_similarity_threshold = float(_config_get(config, "fusion.duplicate_similarity_threshold", 0.82))
    merge_gap_ms = int(_config_get(config, "fusion.merge_gap_ms", 700))
    max_segment_chars = int(_config_get(config, "fusion.max_segment_chars", 180))

    transcript_candidates = list(manual_subtitles or []) + list(asr_segments or []) + list(auto_subtitles or [])
    transcript_candidates.sort(key=lambda item: (source_rank(item.source, priority), item.start_ms, item.end_ms))

    primary: list[Segment] = []
    for candidate in transcript_candidates:
        match = best_overlap_match(candidate, primary, overlap_threshold=overlap_threshold)
        if match is None:
            primary.append(_primary_copy(candidate))
            continue
        index = primary.index(match)
        similarity = segment_text_similarity(candidate, match)
        alternative = {
            "id": candidate.id,
            "source": candidate.source,
            "start_ms": candidate.start_ms,
            "end_ms": candidate.end_ms,
            "text": candidate.text,
            "similarity": similarity,
        }
        alternatives = [*match.alternatives, alternative]
        conflict = match.conflict or similarity < duplicate_similarity_threshold
        primary[index] = replace(match, alternatives=alternatives, conflict=conflict)

    primary.sort(key=lambda item: (item.start_ms, item.end_ms, source_rank(item.source, priority)))
    merged_primary = merge_adjacent_segments(
        primary,
        merge_gap_ms=merge_gap_ms,
        max_segment_chars=max_segment_chars,
    )

    ocr_with_duplicates = suppress_ocr_duplicates(
        list(ocr_segments or []),
        merged_primary,
        overlap_threshold=overlap_threshold,
        similarity_threshold=duplicate_similarity_threshold,
    )
    all_segments = merged_primary + ocr_with_duplicates
    return sorted(all_segments, key=lambda item: (item.start_ms, item.end_ms, source_rank(item.source, priority), item.id))


def _primary_copy(segment: Segment) -> Segment:
    return replace(segment, primary_source=segment.primary_source or segment.source)


def _merge_confidence(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return (left + right) / 2.0


def _config_get(config: Any, dotted: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return _dict_get(config, dotted, default)
    if hasattr(config, "get"):
        try:
            value = config.get(dotted, default)
        except TypeError:
            value = _dict_get(config, dotted, default)
        return default if value is None else value
    return default


def _dict_get(data: dict[str, Any], dotted: str, default: Any) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
