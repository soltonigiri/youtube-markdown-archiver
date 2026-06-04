from __future__ import annotations

import json
from pathlib import Path

from yomutube.archive_browser import find_archives, search_archives, segment_matches


def _write_archive(root: Path, channel: str, name: str, *, video_id: str, title: str, segments: list[dict]) -> Path:
    archive = root / channel / name
    archive.mkdir(parents=True)
    (archive / "manifest.json").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "status": "done",
                "created_at": "2026-05-01T00:00:00+09:00",
                "finished_at": "2026-05-01T00:01:00+09:00",
                "archive_path": str(archive),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (archive / "metadata.json").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "title": title,
                "channel_id": f"UC{video_id}",
                "channel_name": channel,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (archive / "segments.jsonl").write_text(
        "".join(json.dumps(segment, ensure_ascii=False, sort_keys=True) + "\n" for segment in segments),
        encoding="utf-8",
    )
    return archive


def test_search_archives_filters_segments_and_channel(tmp_path: Path) -> None:
    _write_archive(
        tmp_path,
        "日本語 Channel",
        "jp-video",
        video_id="jp123",
        title="検索テスト",
        segments=[
            {
                "id": "seg_001",
                "source": "asr",
                "speaker": "SPEAKER_00",
                "start_ms": 0,
                "end_ms": 1000,
                "text": "日本語検索の説明です",
                "video_id": "jp123",
            },
            {
                "id": "seg_002",
                "source": "ocr",
                "start_ms": 1000,
                "end_ms": 2000,
                "text": "画面 OCR の文字です",
                "video_id": "jp123",
            },
            {
                "conflict": True,
                "id": "seg_003",
                "source": "asr",
                "speaker": "SPEAKER_01",
                "start_ms": 2000,
                "end_ms": 3000,
                "text": "競合候補です",
                "video_id": "jp123",
            },
        ],
    )
    _write_archive(
        tmp_path,
        "Other Channel",
        "other-video",
        video_id="other123",
        title="Other",
        segments=[
            {
                "id": "seg_001",
                "source": "asr",
                "speaker": "SPEAKER_00",
                "start_ms": 0,
                "end_ms": 1000,
                "text": "日本語検索の別動画です",
                "video_id": "other123",
            }
        ],
    )

    rows = search_archives(
        tmp_path,
        "検索",
        filters={"speaker": "SPEAKER_00", "source": "asr", "channel": "日本語"},
    )

    assert [row.video_id for row in rows] == ["jp123"]
    assert search_archives(tmp_path, "検索", filters={"source": "ocr"}) == []


def test_segment_matches_supports_ocr_and_conflicts_filters(tmp_path: Path) -> None:
    _write_archive(
        tmp_path,
        "Example Channel",
        "video",
        video_id="abc123",
        title="Example",
        segments=[
            {"id": "seg_001", "source": "asr", "start_ms": 0, "end_ms": 1000, "text": "spoken text"},
            {"id": "seg_002", "source": "ocr", "start_ms": 1000, "end_ms": 2000, "text": "screen text"},
            {"conflict": True, "id": "seg_003", "source": "asr", "start_ms": 2000, "end_ms": 3000, "text": "conflict text"},
        ],
    )
    archive = find_archives(tmp_path)[0]

    assert [row["id"] for row in segment_matches(archive, "", filters={"ocr": True})] == ["seg_002"]
    assert [row["id"] for row in segment_matches(archive, "", filters={"conflicts": True})] == ["seg_003"]
    assert [row.video_id for row in search_archives(tmp_path, "", filters={"ocr": True})] == ["abc123"]
    assert [row.video_id for row in search_archives(tmp_path, "", filters={"conflicts": True})] == ["abc123"]


def test_archive_browser_gracefully_skips_broken_json_and_jsonl(tmp_path: Path) -> None:
    archive = tmp_path / "Broken Channel" / "video"
    archive.mkdir(parents=True)
    (archive / "manifest.json").write_text("{broken manifest\n", encoding="utf-8")
    (archive / "metadata.json").write_text(
        '{"video_id":"broken123","title":"Broken","channel_name":"Broken Channel"}\n',
        encoding="utf-8",
    )
    (archive / "segments.jsonl").write_text(
        '{"id":"seg_001","start_ms":0,"end_ms":1000,"text":"surviving row"}\n{broken jsonl\n',
        encoding="utf-8",
    )

    archives = find_archives(tmp_path)
    rows = search_archives(tmp_path, "surviving")

    assert archives[0].video_id == "broken123"
    assert rows[0].video_id == "broken123"
    assert segment_matches(rows[0], "surviving")[0]["id"] == "seg_001"
