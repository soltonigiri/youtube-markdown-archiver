from __future__ import annotations

from yomutube.fusion.merge import fuse_segments, merge_adjacent_segments
from yomutube.models import Segment


def seg(segment_id: str, source: str, start: int, end: int, text: str, speaker: str | None = None) -> Segment:
    return Segment(
        id=segment_id,
        video_id="vid",
        source=source,
        start_ms=start,
        end_ms=end,
        text=text,
        speaker=speaker,
    )


def test_fuse_segments_prefers_manual_then_asr_then_auto_and_keeps_alternatives() -> None:
    manual = seg("m1", "manual_subtitle", 1000, 3000, "ローカルで処理します")
    asr = seg("a1", "asr", 1000, 3000, "ローカルで処理します")
    auto = seg("y1", "auto_subtitle", 1000, 3000, "クラウドで処理します")

    fused = fuse_segments(manual_subtitles=[manual], asr_segments=[asr], auto_subtitles=[auto])

    assert len(fused) == 1
    assert fused[0].id == "m1"
    assert fused[0].primary_source == "manual_subtitle"
    assert [alternative["source"] for alternative in fused[0].alternatives] == ["asr", "auto_subtitle"]
    assert fused[0].conflict is True


def test_fuse_segments_suppresses_ocr_duplicates() -> None:
    asr = seg("a1", "asr", 1000, 3000, "今日はローカル処理について話します")
    ocr = seg("o1", "ocr", 1200, 2800, "今日はローカル処理について話します")

    fused = fuse_segments(asr_segments=[asr], ocr_segments=[ocr])

    ocr_result = next(segment for segment in fused if segment.source == "ocr")
    assert ocr_result.duplicate_of == "a1"
    assert "duplicate_similarity" in ocr_result.metadata


def test_merge_adjacent_segments_merges_same_source_speaker_within_gap() -> None:
    merged = merge_adjacent_segments(
        [
            seg("a1", "asr", 1000, 2000, "hello", speaker="SPEAKER_00"),
            seg("a2", "asr", 2500, 3200, "world", speaker="SPEAKER_00"),
        ],
        merge_gap_ms=700,
    )

    assert len(merged) == 1
    assert merged[0].text == "hello world"
    assert merged[0].end_ms == 3200
