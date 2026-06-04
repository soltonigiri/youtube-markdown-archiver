from __future__ import annotations

from dataclasses import replace

from yomutube.models import Segment

from .scoring import segment_text_similarity, time_overlap_ratio


def best_overlap_match(
    target: Segment,
    candidates: list[Segment],
    *,
    overlap_threshold: float = 0.50,
) -> Segment | None:
    matches = [
        candidate
        for candidate in candidates
        if time_overlap_ratio(target, candidate) >= overlap_threshold
    ]
    if not matches:
        return None
    return max(matches, key=lambda candidate: time_overlap_ratio(target, candidate))


def suppress_ocr_duplicates(
    ocr_segments: list[Segment],
    transcript_segments: list[Segment],
    *,
    overlap_threshold: float = 0.50,
    similarity_threshold: float = 0.82,
) -> list[Segment]:
    results: list[Segment] = []
    for ocr in ocr_segments:
        best: tuple[Segment, float] | None = None
        for segment in transcript_segments:
            overlap = time_overlap_ratio(ocr, segment)
            if overlap < overlap_threshold:
                continue
            similarity = segment_text_similarity(ocr, segment)
            if similarity < similarity_threshold:
                continue
            if best is None or similarity > best[1]:
                best = (segment, similarity)
        if best is None:
            results.append(ocr)
            continue
        metadata = dict(ocr.metadata)
        metadata["duplicate_similarity"] = best[1]
        results.append(replace(ocr, duplicate_of=best[0].id, metadata=metadata))
    return results
