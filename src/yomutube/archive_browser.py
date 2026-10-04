from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ArchiveSummary:
    path: Path
    video_id: str
    title: str
    channel: str
    status: str
    created_at: str
    finished_at: str
    index_path: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def find_archives(root: str | Path) -> list[ArchiveSummary]:
    archive_paths = _find_archive_paths(Path(root).expanduser())
    archives = [_read_archive(path) for path in archive_paths]
    return sorted(archives, key=lambda item: (item.finished_at or item.created_at, str(item.path)), reverse=True)


def get_archive(root: str | Path, key: str) -> ArchiveSummary | None:
    for archive in find_archives(root):
        if key in {archive.video_id, archive.path.name, str(archive.path)}:
            return archive
    return None


def search_archives(
    root: str | Path,
    query: str,
    *,
    limit: int = 20,
    filters: Mapping[str, Any] | None = None,
) -> list[ArchiveSummary]:
    needle = query.casefold()
    matches: list[ArchiveSummary] = []
    for archive in find_archives(root):
        if not _archive_matches_filters(archive, filters):
            continue
        archive_match = _archive_contains(archive, needle)
        segment_match = _segments_contain(archive.path, needle, filters=filters)
        if _segment_filters_requested(filters):
            if segment_match:
                matches.append(archive)
        elif archive_match:
            matches.append(archive)
        elif segment_match:
            matches.append(archive)
        if len(matches) >= limit:
            break
    return matches


def last_archive(root: str | Path) -> ArchiveSummary | None:
    archives = find_archives(root)
    return archives[0] if archives else None


def segment_matches(
    archive: ArchiveSummary,
    query: str,
    *,
    limit: int = 5,
    filters: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    needle = query.casefold()
    rows: list[dict[str, Any]] = []
    for row in _read_jsonl(archive.path / "segments.jsonl"):
        text = str(row.get("text") or "")
        if needle in text.casefold() and _row_matches_filters(row, filters):
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def _find_archive_paths(root: Path) -> list[Path]:
    if (root / "manifest.json").exists() or (root / "segments.jsonl").exists():
        return [root]
    if not root.exists():
        return []
    paths = {path.parent for path in root.rglob("manifest.json")}
    paths.update(path.parent for path in root.rglob("segments.jsonl"))
    return sorted(paths)


def _read_archive(path: Path) -> ArchiveSummary:
    manifest = _read_json(path / "manifest.json")
    metadata = _read_json(path / "metadata.json")
    video_id = str(manifest.get("video_id") or metadata.get("video_id") or path.name)
    title = str(metadata.get("title") or manifest.get("title") or path.name)
    channel = str(metadata.get("channel_name") or metadata.get("channel") or metadata.get("uploader") or path.parent.name)
    return ArchiveSummary(
        path=path,
        video_id=video_id,
        title=title,
        channel=channel,
        status=str(manifest.get("status") or "unknown"),
        created_at=str(manifest.get("created_at") or ""),
        finished_at=str(manifest.get("finished_at") or manifest.get("updated_at") or ""),
        index_path=path / "index.md",
        manifest=manifest,
        metadata=metadata,
    )


def _archive_contains(archive: ArchiveSummary, needle: str) -> bool:
    if not needle:
        return True
    haystacks = [archive.video_id, archive.title, archive.channel, str(archive.path)]
    return any(needle in value.casefold() for value in haystacks if value)


def _segments_contain(path: Path, needle: str, *, filters: Mapping[str, Any] | None = None) -> bool:
    for row in _read_jsonl(path / "segments.jsonl"):
        if needle in str(row.get("text") or "").casefold() and _row_matches_filters(row, filters):
            return True
    if not needle and _only_conflicts_filter(filters):
        return bool(_read_jsonl(path / "conflicts.jsonl"))
    return False


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    rows.append(data)
    except (OSError, UnicodeDecodeError):
        return []
    return rows


def _archive_matches_filters(archive: ArchiveSummary, filters: Mapping[str, Any] | None) -> bool:
    channel_terms = _filter_terms(filters, "channel")
    if not channel_terms:
        return True
    values = [
        archive.channel,
        archive.metadata.get("channel"),
        archive.metadata.get("channel_name"),
        archive.metadata.get("channel_id"),
        archive.metadata.get("uploader"),
    ]
    return _contains_any(values, channel_terms)


def _row_matches_filters(row: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
    speaker_terms = _filter_terms(filters, "speaker")
    if speaker_terms and not _contains_any([row.get("speaker")], speaker_terms):
        return False

    source_terms = _filter_terms(filters, "source")
    if source_terms and not _contains_any([row.get("source"), row.get("primary_source")], source_terms):
        return False

    ocr_filter = _bool_filter(filters, "ocr")
    if ocr_filter is not None and _row_is_ocr(row) is not ocr_filter:
        return False

    conflicts_filter = _bool_filter(filters, "conflicts")
    if conflicts_filter is not None and _row_has_conflict(row) is not conflicts_filter:
        return False

    return True


def _segment_filters_requested(filters: Mapping[str, Any] | None) -> bool:
    if not filters:
        return False
    return any(key in filters and filters.get(key) not in (None, "", [], (), set()) for key in ("speaker", "source", "ocr", "conflicts"))


def _only_conflicts_filter(filters: Mapping[str, Any] | None) -> bool:
    return bool(_bool_filter(filters, "conflicts") is True and not _filter_terms(filters, "speaker") and not _filter_terms(filters, "source") and _bool_filter(filters, "ocr") is None)


def _row_is_ocr(row: Mapping[str, Any]) -> bool:
    values = [row.get("source"), row.get("primary_source")]
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        values.extend([metadata.get("source"), metadata.get("role"), metadata.get("region")])
    return _contains_any(values, ("ocr",))


def _row_has_conflict(row: Mapping[str, Any]) -> bool:
    return bool(row.get("conflict"))


def _filter_terms(filters: Mapping[str, Any] | None, key: str) -> tuple[str, ...]:
    if not filters or key not in filters:
        return ()
    value = filters.get(key)
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = [str(item) for item in value if item is not None]
    else:
        values = [str(value)]
    return tuple(item.casefold() for item in values if item.strip())


def _bool_filter(filters: Mapping[str, Any] | None, key: str) -> bool | None:
    if not filters or key not in filters or filters.get(key) is None:
        return None
    value = filters[key]
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return bool(value)


def _contains_any(values: list[Any], terms: tuple[str, ...]) -> bool:
    for value in values:
        text = str(value or "").casefold()
        if text and any(term in text for term in terms):
            return True
    return False
