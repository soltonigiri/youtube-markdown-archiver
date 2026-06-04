from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from yomutube.utils.timecode import ms_to_timecode
from yomutube.writers.jsonl import read_jsonl


@dataclass(slots=True)
class IndexResult:
    archive_dir: Path
    db_path: Path
    archive_count: int
    segment_count: int
    fts_available: bool


def build_sqlite_index(
    archive_dir: str | Path,
    output: str | Path | None = None,
    *,
    output_path: str | Path | None = None,
    use_fts: bool = True,
) -> IndexResult:
    root = Path(archive_dir).expanduser()
    target = output if output is not None else output_path
    if target is None:
        raise ValueError("output or output_path is required")
    db_path = Path(target).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    archive_paths = _find_archive_paths(root)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("DROP TABLE IF EXISTS segments_fts")
        conn.execute("DROP TABLE IF EXISTS segments")
        conn.execute("DROP TABLE IF EXISTS index_meta")
        _create_segments_table(conn)
        fts_available = _create_search_table(conn, use_fts=use_fts)
        conn.execute("CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "INSERT INTO index_meta(key, value) VALUES (?, ?)",
            ("fts_available", "1" if fts_available else "0"),
        )

        segment_count = 0
        for path in archive_paths:
            metadata = _read_metadata(path)
            for segment in read_jsonl(path / "segments.jsonl"):
                if _insert_segment(conn, path, metadata, segment, fts_available=fts_available):
                    segment_count += 1

    return IndexResult(
        archive_dir=root,
        db_path=db_path,
        archive_count=len(archive_paths),
        segment_count=segment_count,
        fts_available=fts_available,
    )


def build_index(
    archive_dir: str | Path,
    output: str | Path | None = None,
    *,
    output_path: str | Path | None = None,
    use_fts: bool = True,
) -> IndexResult:
    return build_sqlite_index(archive_dir, output, output_path=output_path, use_fts=use_fts)


def make_search_filters(
    *,
    speaker: str | None = None,
    source: str | None = None,
    channel: str | None = None,
    ocr: bool | None = None,
    conflicts: bool | None = None,
    semantic: bool | None = None,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "speaker": speaker,
            "source": source,
            "channel": channel,
            "ocr": ocr,
            "conflicts": conflicts,
            "semantic": semantic,
        }.items()
        if value not in (None, "", [], (), set())
    }


def ngram_terms(text: str, *, min_size: int = 2, max_size: int = 3) -> list[str]:
    value = "".join(str(text).split())
    if not value:
        return []
    terms: list[str] = []
    for size in range(max(1, min_size), max(min_size, max_size) + 1):
        for index in range(0, max(0, len(value) - size + 1)):
            term = value[index : index + size]
            if term not in terms:
                terms.append(term)
    return terms


def search_index(
    db_path: str | Path,
    query: str,
    *,
    limit: int = 20,
    filters: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not query.strip() and not filters:
        return []
    with sqlite3.connect(Path(db_path).expanduser()) as conn:
        conn.row_factory = sqlite3.Row
        where_sql, params = _filter_where(filters)
        if query.strip() and _db_has_fts(conn):
            try:
                sql = f"""
                    SELECT s.archive_path, s.archive_id, s.segment_id, s.video_id,
                           s.channel_id, s.title, s.channel, s.text, s.timestamp,
                           s.start_ms, s.end_ms, s.source, s.primary_source,
                           s.speaker, s.youtube_url, s.conflict
                    FROM segments_fts f
                    JOIN segments s ON s.rowid = f.rowid
                    WHERE segments_fts MATCH ?{where_sql}
                    ORDER BY s.video_id, s.start_ms
                    LIMIT ?
                    """
                rows = conn.execute(
                    sql,
                    (query, *params, int(limit)),
                ).fetchall()
                if rows:
                    return [dict(row) for row in rows]
            except sqlite3.OperationalError:
                pass
        pattern = f"%{query}%"
        segment_scoped = _segment_filters_requested(filters)
        text_sql = " AND s.text LIKE ?" if query.strip() and segment_scoped else " AND (s.text LIKE ? OR s.title LIKE ? OR s.channel LIKE ?)" if query.strip() else ""
        text_params: tuple[Any, ...] = (pattern,) if query.strip() and segment_scoped else (pattern, pattern, pattern) if query.strip() else ()
        rows = conn.execute(
            f"""
            SELECT s.archive_path, s.archive_id, s.segment_id, s.video_id, s.channel_id,
                   s.title, s.channel, s.text, s.timestamp, s.start_ms, s.end_ms, s.source,
                   s.primary_source, s.speaker, s.youtube_url, s.conflict
            FROM segments s
            WHERE 1=1{text_sql}{where_sql}
            ORDER BY video_id, start_ms
            LIMIT ?
            """,
            (*text_params, *params, int(limit)),
        ).fetchall()
        return [dict(row) for row in rows]


def _find_archive_paths(root: Path) -> list[Path]:
    if (root / "segments.jsonl").exists():
        return [root]
    if not root.exists():
        return []
    return sorted({path.parent for path in root.rglob("segments.jsonl")})


def _read_metadata(archive_path: Path) -> dict[str, Any]:
    path = archive_path / "metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _create_segments_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE segments (
          archive_path TEXT NOT NULL,
          archive_id TEXT,
          segment_id TEXT,
          video_id TEXT,
          channel_id TEXT,
          title TEXT,
          channel TEXT,
          text TEXT NOT NULL,
          timestamp TEXT NOT NULL,
          start_ms INTEGER NOT NULL,
          end_ms INTEGER,
          source TEXT,
          primary_source TEXT,
          speaker TEXT,
          youtube_url TEXT,
          conflict INTEGER NOT NULL DEFAULT 0,
          segment_json TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX idx_segments_video_time ON segments(video_id, start_ms)")
    conn.execute("CREATE INDEX idx_segments_source ON segments(source)")
    conn.execute("CREATE INDEX idx_segments_speaker ON segments(speaker)")
    conn.execute("CREATE INDEX idx_segments_channel ON segments(channel)")


def _create_search_table(conn: sqlite3.Connection, *, use_fts: bool) -> bool:
    if use_fts:
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE segments_fts USING fts5(
                  title,
                  channel,
                  text
                )
                """
            )
            return True
        except sqlite3.OperationalError:
            conn.execute("DROP TABLE IF EXISTS segments_fts")
    conn.execute(
        """
        CREATE TABLE segments_fts (
          segment_rowid INTEGER NOT NULL,
          title TEXT,
          channel TEXT,
          text TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX idx_segments_fts_text ON segments_fts(text)")
    return False


def _insert_segment(
    conn: sqlite3.Connection,
    archive_path: Path,
    metadata: dict[str, Any],
    segment: dict[str, Any],
    *,
    fts_available: bool,
) -> bool:
    text = str(segment.get("text") or "").strip()
    if not text:
        return False
    start_ms = int(segment.get("start_ms") or 0)
    end_ms = int(segment.get("end_ms") or start_ms)
    archive_id = str(metadata.get("archive_id") or archive_path.name)
    video_id = str(segment.get("video_id") or metadata.get("video_id") or "unknown")
    channel_id = metadata.get("channel_id")
    title = str(metadata.get("title") or "")
    channel = str(metadata.get("channel_name") or metadata.get("channel") or metadata.get("uploader") or "")
    source = str(segment.get("source") or segment.get("primary_source") or "")
    primary_source = str(segment.get("primary_source") or source or "")
    speaker = segment.get("speaker")
    conflict = 1 if segment.get("conflict") else 0
    youtube_url = str(
        segment.get("youtube_url")
        or metadata.get("webpage_url")
        or metadata.get("source_url")
        or metadata.get("original_url")
        or ""
    )
    timestamp = ms_to_timecode(start_ms, include_millis=True)
    cursor = conn.execute(
        """
        INSERT INTO segments(
          archive_path, archive_id, segment_id, video_id, channel_id, title,
          channel, text, timestamp, start_ms, end_ms, source, primary_source,
          speaker, youtube_url, conflict, segment_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(archive_path),
            archive_id,
            segment.get("id"),
            video_id,
            channel_id,
            title,
            channel,
            text,
            timestamp,
            start_ms,
            end_ms,
            source,
            primary_source,
            speaker,
            youtube_url,
            conflict,
            json.dumps(segment, ensure_ascii=False, sort_keys=True),
        ),
    )
    rowid = cursor.lastrowid
    if fts_available:
        conn.execute(
            """
            INSERT INTO segments_fts(rowid, title, channel, text)
            VALUES (?, ?, ?, ?)
            """,
            (rowid, title, channel, text),
        )
    else:
        conn.execute(
            """
            INSERT INTO segments_fts(segment_rowid, title, channel, text)
            VALUES (?, ?, ?, ?)
            """,
            (rowid, title, channel, text),
        )
    return True


def _db_has_fts(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT value FROM index_meta WHERE key = 'fts_available'").fetchone()
    return bool(row and row[0] == "1")


def _filter_where(filters: Mapping[str, Any] | None) -> tuple[str, list[Any]]:
    if not filters:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    speaker = filters.get("speaker")
    if speaker:
        clauses.append(" AND s.speaker LIKE ?")
        params.append(f"%{speaker}%")
    source = filters.get("source")
    if source:
        clauses.append(" AND (s.source LIKE ? OR s.primary_source LIKE ?)")
        params.extend([f"%{source}%", f"%{source}%"])
    channel = filters.get("channel")
    if channel:
        clauses.append(" AND (s.channel LIKE ? OR s.channel_id LIKE ?)")
        params.extend([f"%{channel}%", f"%{channel}%"])
    if filters.get("ocr") is not None:
        clauses.append(" AND ((s.source LIKE '%ocr%' OR s.primary_source LIKE '%ocr%') = ?)")
        params.append(1 if bool(filters.get("ocr")) else 0)
    if filters.get("conflicts") is not None:
        clauses.append(" AND s.conflict = ?")
        params.append(1 if bool(filters.get("conflicts")) else 0)
    return "".join(clauses), params


def _segment_filters_requested(filters: Mapping[str, Any] | None) -> bool:
    if not filters:
        return False
    return any(key in filters and filters.get(key) not in (None, "", [], (), set()) for key in ("speaker", "source", "ocr", "conflicts"))
