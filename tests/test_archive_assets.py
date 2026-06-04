from __future__ import annotations

import json
from pathlib import Path

from yomutube.archive_assets import (
    Correction,
    alignment_rows_from_segments,
    append_correction,
    apply_corrections_for_display,
    build_quality_report,
    conflict_rows_from_segments,
    entity_rows_from_segments,
    read_json_for_display,
    read_jsonl_for_display,
    read_speaker_aliases,
    slide_rows_from_visual_text,
    speaker_turn_row_from_turn,
    speaker_turn_rows_from_segments,
    topic_rows_from_chapters,
    visual_text_rows_from_segments,
    word_rows_from_segment,
    write_speaker_aliases,
)
from yomutube.models import Chapter, Segment


def segment(
    segment_id: str,
    source: str,
    start_ms: int,
    end_ms: int,
    text: str,
    **kwargs: object,
) -> Segment:
    return Segment(
        id=segment_id,
        video_id="vid",
        source=source,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        **kwargs,
    )


def test_word_rows_from_asr_segment_metadata_words() -> None:
    asr = segment(
        "asr_001",
        "asr",
        1000,
        1800,
        "hello world",
        speaker="SPEAKER_00",
        confidence=0.7,
        metadata={
            "words": [
                {"word": " hello", "start_ms": 1010, "end_ms": 1200, "probability": 0.9},
                {"text": "world", "start": 1.21, "end": 1.6},
            ]
        },
    )

    rows = word_rows_from_segment(asr)

    assert [row["word"] for row in rows] == ["hello", "world"]
    assert rows[0]["start_ms"] == 1010
    assert rows[0]["confidence"] == 0.9
    assert rows[1]["end_ms"] == 1600
    assert rows[1]["confidence"] == 0.7
    assert rows[1]["speaker"] == "SPEAKER_00"


def test_speaker_turn_rows_from_turns_and_segments() -> None:
    turn_row = speaker_turn_row_from_turn(
        {"speaker": "SPEAKER_00", "start_ms": 0, "end_ms": 900, "confidence": 0.8},
        aliases={"SPEAKER_00": "Alice"},
    )
    assert turn_row["alias"] == "Alice"
    assert turn_row["speaker"] == "SPEAKER_00"

    rows = speaker_turn_rows_from_segments(
        [
            segment("a", "asr", 0, 1000, "one", speaker="SPEAKER_00"),
            segment("b", "asr", 1100, 2000, "two", speaker="SPEAKER_00"),
            segment("c", "asr", 2300, 3000, "three", speaker="SPEAKER_01"),
            segment("d", "asr", 3100, 3300, "ignored"),
        ],
        aliases={"SPEAKER_00": "Alice"},
        merge_gap_ms=200,
    )

    assert rows[0]["start_ms"] == 0
    assert rows[0]["end_ms"] == 2000
    assert rows[0]["alias"] == "Alice"
    assert rows[0]["metadata"]["segment_ids"] == ["a", "b"]
    assert rows[1]["speaker"] == "SPEAKER_01"


def test_aliases_and_corrections_helpers_are_display_safe(tmp_path: Path) -> None:
    assert read_speaker_aliases(tmp_path) == {}
    write_speaker_aliases(tmp_path, {"SPEAKER_00": "Alice"})
    assert read_speaker_aliases(tmp_path) == {"SPEAKER_00": "Alice"}

    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    assert read_json_for_display(tmp_path / "broken.json", default={"ok": False}) == {"ok": False}

    corrections_path = tmp_path / "corrections.jsonl"
    correction = Correction(
        target_segment_id="seg_001",
        operation="replace_text",
        before="old",
        after="new",
        reason="typo",
        created_at="2026-04-28T12:00:00+09:00",
    )
    append_correction(corrections_path, correction)
    corrections_path.write_text(
        corrections_path.read_text(encoding="utf-8") + "{broken\n" + json.dumps(correction.to_dict()) + "\n",
        encoding="utf-8",
    )

    corrections = read_jsonl_for_display(corrections_path)
    assert len(corrections) == 2

    original = segment("seg_001", "asr", 0, 1000, "old")
    displayed = apply_corrections_for_display([original], corrections)
    assert original.text == "old"
    assert displayed[0].text == "new"
    assert displayed[0].metadata["display_correction"]["reason"] == "typo"


def test_quality_alignment_and_conflict_helpers() -> None:
    segments = [
        segment("sub_001", "manual_subtitle", 0, 5000, "manual text"),
        segment("asr_001", "asr", 0, 5000, "asr text", confidence=0.8, speaker="SPEAKER_00"),
        segment(
            "fused_001",
            "fused",
            6000,
            7000,
            "primary",
            primary_source="manual_subtitle",
            conflict=True,
            alternatives=[
                {
                    "id": "asr_002",
                    "source": "asr",
                    "start_ms": 6000,
                    "end_ms": 7000,
                    "text": "alternate",
                    "similarity": 0.2,
                }
            ],
        ),
        segment("ocr_001", "ocr", 8000, 8500, "title"),
    ]

    report = build_quality_report(segments, duration_ms=10_000).to_dict()
    assert report["subtitle_coverage"] == 0.5
    assert report["asr_avg_confidence"] == 0.8
    assert report["conflict_rate"] == 0.25
    assert report["speaker_coverage"] > 0
    assert report["ocr_event_count"] == 1
    assert "overall_score" in report

    alignment_rows = alignment_rows_from_segments(segments)
    assert alignment_rows[0]["primary_segment_id"] == "fused_001"
    assert alignment_rows[0]["alternative_segment_id"] == "asr_002"

    conflict_rows = conflict_rows_from_segments(segments)
    assert conflict_rows[0]["segment_id"] == "fused_001"
    assert conflict_rows[0]["alternatives"][0]["text"] == "alternate"


def test_topic_entity_visual_text_and_slide_helpers() -> None:
    topics = topic_rows_from_chapters(
        "vid",
        [
            Chapter(title="Intro", start_time=0, end_time=10),
            Chapter(title="Deep Dive", start_time=10, end_time=None),
        ],
        duration_ms=30_000,
    )
    assert topics[-1]["end_ms"] == 30_000

    entities = entity_rows_from_segments(
        [segment("seg_001", "asr", 0, 1000, "Visit https://example.com and Python.")],
        terms=["Python"],
    )
    assert {row["kind"] for row in entities} == {"url", "term"}

    visual_rows = visual_text_rows_from_segments(
        [
            segment("ocr_001", "ocr", 0, 1000, "Slide A", confidence=0.7, metadata={"region": "slide"}),
            segment("ocr_002", "ocr", 1200, 2000, "Slide A", confidence=0.8, metadata={"region": "slide"}),
        ]
    )
    assert visual_rows[0]["region"] == "slide"

    slides = slide_rows_from_visual_text(visual_rows, merge_gap_ms=500)
    assert len(slides) == 1
    assert slides[0]["start_ms"] == 0
    assert slides[0]["end_ms"] == 2000
