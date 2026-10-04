from __future__ import annotations

import csv
import json
from pathlib import Path

from yomutube.exporters import (
    export_archive,
    export_csv,
    export_obsidian,
    export_speakers,
    export_srt,
    export_topics,
    export_txt,
    export_vtt,
    quote_at,
)


def test_pure_export_functions_render_transcript_topics_and_speakers(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path)
    metadata = _read_json(archive / "metadata.json")
    segments = _read_jsonl(archive / "segments.jsonl")
    topics = _read_jsonl(archive / "topics.jsonl")
    speaker_turns = _read_jsonl(archive / "speaker_turns.jsonl")
    speaker_aliases = _read_json(archive / "speaker_aliases.json")

    srt = export_srt(segments, speaker_aliases=speaker_aliases)
    assert "00:00:01,240 --> 00:00:03,000\nAlice: Local pipeline introduction." in srt
    assert "Slide title" not in srt

    vtt = export_vtt(segments, speaker_aliases=speaker_aliases)
    assert vtt.startswith("WEBVTT\n\n")
    assert "00:00:04.500 --> 00:00:06.500\nBob: Second point." in vtt

    txt = export_txt(segments, speaker_aliases=speaker_aliases)
    assert "[00:00:04.500] Bob: Second point." in txt
    assert "Hidden line" not in txt

    csv_text = export_csv(metadata, segments, speaker_aliases=speaker_aliases)
    csv_rows = list(csv.DictReader(csv_text.splitlines()))
    assert csv_rows[0]["segment_id"] == "seg_001"
    assert csv_rows[0]["speaker"] == "Alice"
    assert csv_rows[0]["youtube_url"] == "https://youtu.be/abc123?t=1"

    obsidian = export_obsidian(metadata, segments, topics=topics, speaker_aliases=speaker_aliases)
    assert 'title: "Example Video"' in obsidian
    assert "## Topics" in obsidian
    assert "- [00:00:00.000](https://youtu.be/abc123?t=0) Intro" in obsidian

    topic_markdown = export_topics(metadata, topics)
    assert "| 00:00:00.000 | 00:00:07.000 | Intro | Opening context |" in topic_markdown

    speaker_markdown = export_speakers(
        metadata,
        segments,
        speaker_turns=speaker_turns,
        speaker_aliases=speaker_aliases,
    )
    assert "| SPEAKER_00 | Alice | 1 | 00:00:02.000 |" in speaker_markdown


def test_export_archive_writes_default_output_and_profile(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path)

    result = export_archive(archive, "csv")

    assert result.output_path == archive / "exports" / "csv.csv"
    assert result.profile_path == archive / "exports" / "csv.json"
    assert result.output_path.exists()
    profile = json.loads(result.profile_path.read_text(encoding="utf-8"))
    assert profile["schema_version"] == "1.0"
    assert profile["format"] == "csv"
    assert profile["ext"] == "csv"
    assert profile["input_archive"] == str(archive)
    assert profile["output_path"] == str(result.output_path)
    assert profile["segment_count"] == 2
    assert "segments.jsonl" in profile["source_files"]


def test_export_archive_accepts_custom_output_and_quote_uses_nearest_segment(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path)
    output = tmp_path / "clip.md"

    result = export_archive(archive, "quotes", output=output)
    quote = quote_at(archive, 4600)

    assert result.output_path == output
    assert (archive / "exports" / "quotes.json").exists()
    assert "> [00:00:04.500](https://youtu.be/abc123?t=4) Bob: Second point." in quote
    assert "`segment_id: seg_002`" in quote


def test_export_archive_skips_broken_jsonl_lines_when_possible(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path)
    (archive / "segments.jsonl").write_text(
        json.dumps(
            {
                "id": "seg_001",
                "video_id": "abc123",
                "source": "fused",
                "start_ms": 1240,
                "end_ms": 3000,
                "text": "Good line.",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        + "{broken json\n",
        encoding="utf-8",
    )

    result = export_archive(archive, "txt")

    assert result.output_path.read_text(encoding="utf-8") == "[00:00:01.240] Good line.\n"


def _write_archive(tmp_path: Path) -> Path:
    archive = tmp_path / "archive"
    archive.mkdir()
    _write_json(
        archive / "metadata.json",
        {
            "archive_id": "yt_abc123",
            "channel_id": "UCexample",
            "channel_name": "Example Channel",
            "duration_sec": 12,
            "title": "Example Video",
            "upload_date": "2026-04-01",
            "video_id": "abc123",
            "webpage_url": "https://www.youtube.com/watch?v=abc123",
        },
    )
    _write_jsonl(
        archive / "segments.jsonl",
        [
            {
                "id": "seg_001",
                "video_id": "abc123",
                "source": "fused",
                "primary_source": "asr",
                "start_ms": 1240,
                "end_ms": 3000,
                "speaker": "SPEAKER_00",
                "text": "Local pipeline introduction.",
            },
            {
                "id": "seg_002",
                "video_id": "abc123",
                "source": "fused",
                "primary_source": "asr",
                "start_ms": 4500,
                "end_ms": 6500,
                "speaker": "SPEAKER_01",
                "text": "Second point.",
            },
            {
                "id": "ocr_001",
                "video_id": "abc123",
                "source": "ocr",
                "start_ms": 7000,
                "end_ms": 7600,
                "metadata": {"region": "slide"},
                "text": "Slide title",
            },
            {
                "id": "hidden_001",
                "video_id": "abc123",
                "source": "fused",
                "start_ms": 8000,
                "end_ms": 9000,
                "metadata": {"hidden": True},
                "text": "Hidden line",
            },
        ],
    )
    _write_jsonl(
        archive / "topics.jsonl",
        [
            {
                "id": "topic_001",
                "title": "Intro",
                "start_ms": 0,
                "end_ms": 7000,
                "summary": "Opening context",
            }
        ],
    )
    _write_jsonl(
        archive / "speaker_turns.jsonl",
        [
            {"speaker": "SPEAKER_00", "start_ms": 1000, "end_ms": 3000},
            {"speaker": "SPEAKER_01", "start_ms": 4500, "end_ms": 6500},
        ],
    )
    _write_json(archive / "speaker_aliases.json", {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})
    return archive


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
