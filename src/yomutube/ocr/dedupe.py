from __future__ import annotations

import difflib
from collections.abc import Iterable

from .base import BBox, OCREvent, OCRObservation
from yomutube.utils.text import normalize_text


def text_similarity(left: str, right: str) -> float:
    lhs = normalize_text(left).lower()
    rhs = normalize_text(right).lower()
    if not lhs and not rhs:
        return 1.0
    if not lhs or not rhs:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return float(fuzz.ratio(lhs, rhs)) / 100.0
    except ImportError:
        return difflib.SequenceMatcher(None, lhs, rhs).ratio()


def bbox_iou(left: BBox | None, right: BBox | None) -> float:
    if left is None and right is None:
        return 1.0
    if left is None or right is None:
        return 0.0
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def dedupe_observations(
    observations: Iterable[OCRObservation],
    *,
    similarity_threshold: float = 0.86,
    bbox_iou_threshold: float = 0.50,
    gap_ms: int = 1500,
    min_duration_ms: int = 400,
) -> list[OCREvent]:
    events: list[OCREvent] = []
    current: OCREvent | None = None
    last_observation_ms: int | None = None

    for observation in sorted(observations, key=lambda item: (item.frame_ms, item.region, item.text)):
        if not observation.text:
            continue
        if current is None or not _same_event(
            current,
            observation,
            last_observation_ms=last_observation_ms,
            similarity_threshold=similarity_threshold,
            bbox_iou_threshold=bbox_iou_threshold,
            gap_ms=gap_ms,
        ):
            if current is not None:
                _finalize_duration(current, min_duration_ms)
                events.append(current)
            current = _new_event(len(events) + 1, observation)
        else:
            current.end_ms = max(current.end_ms, observation.frame_ms)
            current.observation_count += 1
            current.confidence = _merge_confidence(current.confidence, observation.confidence)
            if observation.confidence is not None and (
                current.confidence is None or observation.confidence >= current.confidence
            ):
                current.text = observation.text
                current.bbox = observation.bbox or current.bbox
        last_observation_ms = observation.frame_ms

    if current is not None:
        _finalize_duration(current, min_duration_ms)
        events.append(current)
    return events


def _new_event(index: int, observation: OCRObservation) -> OCREvent:
    return OCREvent(
        id=f"ocr_{index:06d}",
        video_id=observation.video_id,
        start_ms=observation.frame_ms,
        end_ms=observation.frame_ms,
        text=observation.text,
        region=observation.region,
        role=observation.role,
        bbox=observation.bbox,
        confidence=observation.confidence,
        observation_count=1,
        metadata={"engines": [observation.engine] if observation.engine else []},
    )


def _same_event(
    event: OCREvent,
    observation: OCRObservation,
    *,
    last_observation_ms: int | None,
    similarity_threshold: float,
    bbox_iou_threshold: float,
    gap_ms: int,
) -> bool:
    if event.region != observation.region or event.role != observation.role:
        return False
    if last_observation_ms is not None and observation.frame_ms - last_observation_ms > gap_ms:
        return False
    if text_similarity(event.text, observation.text) < similarity_threshold:
        return False
    if event.bbox is None or observation.bbox is None:
        return True
    return bbox_iou(event.bbox, observation.bbox) >= bbox_iou_threshold


def _merge_confidence(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return (left + right) / 2.0


def _finalize_duration(event: OCREvent, min_duration_ms: int) -> None:
    if event.end_ms - event.start_ms < min_duration_ms:
        event.end_ms = event.start_ms + min_duration_ms
