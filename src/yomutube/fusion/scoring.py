from __future__ import annotations

import difflib

from yomutube.models import Segment
from yomutube.utils.text import normalize_text

DEFAULT_PRIORITY = ["manual_subtitle", "asr", "auto_subtitle"]


def overlap_ms(left: Segment, right: Segment) -> int:
    return max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))


def time_overlap_ratio(left: Segment, right: Segment) -> float:
    overlap = overlap_ms(left, right)
    denominator = min(left.duration_ms, right.duration_ms)
    return overlap / denominator if denominator else 0.0


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


def segment_text_similarity(left: Segment, right: Segment) -> float:
    return text_similarity(left.text, right.text)


def source_rank(source: str, priority: list[str] | None = None) -> int:
    order = priority or DEFAULT_PRIORITY
    try:
        return order.index(source)
    except ValueError:
        return len(order)


def is_duplicate(
    left: Segment,
    right: Segment,
    *,
    overlap_threshold: float = 0.50,
    similarity_threshold: float = 0.82,
) -> bool:
    return (
        time_overlap_ratio(left, right) >= overlap_threshold
        and segment_text_similarity(left, right) >= similarity_threshold
    )
