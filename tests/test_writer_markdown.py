from __future__ import annotations

import json
from pathlib import Path

from yomutube.models import Manifest, Segment, VideoMetadata
from yomutube.writers.jsonl import read_jsonl
from yomutube.writers.markdown import write_archive


def test_write_archive_outputs_markdown_and_sidecars(tmp_path: Path) -> None:
    metadata = VideoMetadata(
        video_id="abc123",
        title="Example Video",
        channel_id="UCexample",
        channel_name="Example Channel",
        uploader="Example Uploader",
        upload_date="2026-04-01",
        duration_sec=1234,
        webpage_url="https://www.youtube.com/watch?v=abc123",
    )
    segments = [
        Segment(
            id="seg_001",
            video_id="abc123",
            source="fused",
            primary_source="asr",
            start_ms=1240,
            end_ms=3000,
            text="Local pipeline introduction.",
            language="en",
            speaker="SPEAKER_00",
        ),
        Segment(
            id="seg_002",
            video_id="abc123",
            source="ocr",
            start_ms=9300,
            end_ms=9800,
            text="YomuTube = YouTube -> Markdown Archive",
            language="en",
            metadata={"region": "slide"},
        ),
        Segment(
            id="seg_003",
            video_id="abc123",
            source="fused",
            start_ms=10000,
            end_ms=11000,
            text="Hidden conflict.",
            conflict=True,
        ),
        Segment(
            id="seg_004",
            video_id="abc123",
            source="asr",
            primary_source="asr",
            start_ms=12000,
            end_ms=13000,
            text="Visible primary conflict.",
            confidence=-0.2,
            conflict=True,
            alternatives=[
                {
                    "id": "sub_002",
                    "source": "auto_subtitle",
                    "start_ms": 12000,
                    "end_ms": 13000,
                    "text": "Visible primary conflict alternative.",
                    "similarity": 0.2,
                }
            ],
        ),
    ]
    diagnostics = {
        "subtitle": {"segments": 2},
        "asr": {"enabled": True, "engine": "faster-whisper", "model": "medium", "segments": 1},
        "ocr": {"enabled": True, "engine": "paddleocr", "deduped_events": 1},
        "fusion": {"fused_segments": 3, "conflicts": 1},
    }
    manifest = Manifest(
        program="YomuTube",
        version="0.1.0",
        video_id="abc123",
        run_id="test-run",
        status="done",
        steps={"markdown": "done"},
        models={"asr": "faster-whisper:medium", "ocr": "paddleocr"},
        created_at="2026-04-28T12:00:00+09:00",
        finished_at="2026-04-28T12:08:21+09:00",
    )

    result = write_archive(
        archive_dir=tmp_path,
        metadata=metadata,
        segments=segments,
        raw_asr=[segments[0]],
        raw_subtitles=[{"id": "sub_001", "start_ms": 1240, "end_ms": 3000, "text": "Local pipeline introduction."}],
        raw_ocr=[segments[1]],
        diagnostics=diagnostics,
        manifest=manifest,
    )

    assert result.index_path == tmp_path / "index.md"
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "segments.jsonl").exists()
    assert (tmp_path / "raw_asr.jsonl").exists()
    assert (tmp_path / "raw_subtitles.jsonl").exists()
    assert (tmp_path / "raw_ocr.jsonl").exists()
    assert (tmp_path / "speaker_aliases.json").exists()
    assert (tmp_path / "corrections.jsonl").exists()
    assert (tmp_path / "speaker_turns.jsonl").exists()
    assert (tmp_path / "conflicts.jsonl").exists()
    assert (tmp_path / "quality.json").exists()
    assert (tmp_path / "diagnostics.json").exists()
    assert (tmp_path / "manifest.json").exists()

    index = result.index_path.read_text(encoding="utf-8")
    assert "program: YomuTube" in index
    assert 'title: "Example Video"' in index
    transcript = index.split("## Transcript", 1)[1].split("## OCR Events", 1)[0]
    assert "[00:00:01.240](https://youtu.be/abc123?t=1) SPEAKER_00: Local pipeline introduction." in transcript
    assert "YomuTube = YouTube -> Markdown Archive" not in transcript
    assert "## OCR Events" in index
    assert "| 00:00:09.300 | slide | YomuTube = YouTube -> Markdown Archive |" in index
    assert "Visible primary conflict." in transcript
    assert "[?] [00:00:12.000](https://youtu.be/abc123?t=12) Visible primary conflict." not in transcript
    assert "- ASR segments: 1" in index
    assert "- Fused segments: 3" in index
    assert "Hidden conflict." not in index

    metadata_json = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata_json["video_id"] == "abc123"
    assert len(read_jsonl(tmp_path / "segments.jsonl")) == 4
    assert json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))["conflict_rate"] > 0
    assert read_jsonl(tmp_path / "speaker_turns.jsonl")[0]["speaker"] == "SPEAKER_00"
    assert "seg_004" in {row["segment_id"] for row in read_jsonl(tmp_path / "conflicts.jsonl")}
