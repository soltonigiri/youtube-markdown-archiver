from __future__ import annotations

import sqlite3
from pathlib import Path

from yomutube.writers.sqlite_index import build_sqlite_index, make_search_filters, ngram_terms, search_index


FIXTURE_ARCHIVE = Path(__file__).parent / "fixtures" / "archive"


def test_build_sqlite_index_from_archive_fixture(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"

    result = build_sqlite_index(FIXTURE_ARCHIVE, db_path)

    assert result.archive_count == 1
    assert result.segment_count == 2
    assert db_path.exists()

    rows = search_index(db_path, "Markdown")
    assert rows
    assert rows[0]["video_id"] == "abc123"
    assert rows[0]["timestamp"] == "00:00:09.300"
    assert rows[0]["archive_id"] == "20260401_abc123_Example_Video"
    assert rows[0]["channel_id"] == "UCexample"
    assert rows[0]["primary_source"] == "ocr"
    assert rows[0]["youtube_url"] == "https://www.youtube.com/watch?v=abc123"

    channel_rows = search_index(db_path, "Channel")
    assert channel_rows
    assert channel_rows[0]["channel"] == "Example Channel"

    with sqlite3.connect(db_path) as conn:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(segments)").fetchall()
        }
    assert {"archive_id", "channel_id", "primary_source", "youtube_url"} <= columns


def test_build_sqlite_index_falls_back_to_regular_table(tmp_path: Path) -> None:
    db_path = tmp_path / "archive-fallback.db"

    result = build_sqlite_index(FIXTURE_ARCHIVE, db_path, use_fts=False)

    assert result.fts_available is False
    rows = search_index(db_path, "pipeline")
    assert len(rows) == 1
    assert rows[0]["text"] == "Local pipeline introduction."
    assert rows[0]["youtube_url"] == "https://www.youtube.com/watch?v=abc123"

    with sqlite3.connect(db_path) as conn:
        table_type = conn.execute(
            "SELECT type FROM sqlite_master WHERE name = 'segments_fts'"
        ).fetchone()[0]
    assert table_type == "table"


def test_search_index_filters_speaker_source_and_channel(tmp_path: Path) -> None:
    archive = tmp_path / "archives" / "Japanese Channel" / "video"
    archive.mkdir(parents=True)
    (archive / "metadata.json").write_text(
        """
        {
          "channel_id": "UCjp",
          "channel_name": "日本語 Channel",
          "title": "検索テスト",
          "video_id": "jp123",
          "webpage_url": "https://www.youtube.com/watch?v=jp123"
        }
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    (archive / "segments.jsonl").write_text(
        "\n".join(
            [
                '{"id":"seg_001","source":"asr","primary_source":"asr","speaker":"SPEAKER_00","start_ms":0,"end_ms":1000,"text":"日本語検索の説明です","video_id":"jp123"}',
                '{"id":"seg_002","source":"ocr","primary_source":"ocr","start_ms":1000,"end_ms":2000,"text":"画面の文字です","video_id":"jp123"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "archive.db"
    build_sqlite_index(tmp_path / "archives", db_path)

    rows = search_index(
        db_path,
        "検索",
        filters=make_search_filters(speaker="SPEAKER_00", source="asr", channel="日本語"),
    )

    assert len(rows) == 1
    assert rows[0]["segment_id"] == "seg_001"
    assert rows[0]["speaker"] == "SPEAKER_00"
    assert search_index(db_path, "検索", filters=make_search_filters(source="ocr")) == []


def test_ngram_terms_support_japanese_like_fallback(tmp_path: Path) -> None:
    archive = tmp_path / "archives" / "UCjp" / "video"
    archive.mkdir(parents=True)
    (archive / "metadata.json").write_text(
        '{"channel_name":"日本語 Channel","title":"長い題名","video_id":"jp456"}\n',
        encoding="utf-8",
    )
    (archive / "segments.jsonl").write_text(
        '{"id":"seg_001","source":"asr","start_ms":0,"end_ms":1000,"text":"これは日本語検索確認です","video_id":"jp456"}\n',
        encoding="utf-8",
    )
    db_path = tmp_path / "archive.db"
    build_sqlite_index(tmp_path / "archives", db_path)

    assert "検索" in ngram_terms("日本語検索", min_size=2, max_size=2)
    rows = search_index(db_path, "検索")

    assert len(rows) == 1
    assert rows[0]["text"] == "これは日本語検索確認です"
