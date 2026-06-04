from __future__ import annotations

import fcntl
import json
import os
import shutil
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from .config import AppConfig
from .models import Manifest, RunPaths, StepState, VideoMetadata
from .utils.hashing import safe_path_component, safe_slug, short_hash


JST = timezone(timedelta(hours=9))


def now_jst() -> datetime:
    return datetime.now(JST)


def iso_now() -> str:
    return now_jst().isoformat(timespec="seconds")


def make_run_id(video_id: str) -> str:
    return f"{now_jst().strftime('%Y%m%d_%H%M%S')}_{video_id}"


def archive_dir_for(metadata: VideoMetadata, config: AppConfig) -> Path:
    channel = _channel_component(metadata)
    title = _title_component(metadata)
    candidate = config.archive_dir / channel / title
    manifest = load_manifest(candidate)
    if manifest is None or manifest.video_id == metadata.video_id:
        return candidate
    return config.archive_dir / channel / f"{title}__{metadata.video_id}"


def existing_archive_for(metadata: VideoMetadata, config: AppConfig) -> Path | None:
    readable = archive_dir_for(metadata, config)
    if readable.exists():
        return readable

    old_base = config.archive_dir / safe_slug(metadata.channel_id or "unknown", fallback="unknown")
    old_current = old_base / metadata.video_id
    if old_current.exists():
        return old_current

    legacy_matches = []
    if old_base.exists():
        legacy_matches = sorted(old_base.glob(f"*_{metadata.video_id}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if legacy_matches:
        return legacy_matches[0]

    manifest_matches = []
    if config.archive_dir.exists():
        for manifest_path in config.archive_dir.rglob("manifest.json"):
            manifest = load_manifest(manifest_path.parent)
            if manifest and manifest.video_id == metadata.video_id:
                manifest_matches.append(manifest_path.parent)
    if manifest_matches:
        return sorted(manifest_matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return None


def _channel_component(metadata: VideoMetadata) -> str:
    value = metadata.channel_name or metadata.uploader or metadata.channel_id or "unknown"
    return safe_path_component(value, fallback=safe_slug(metadata.channel_id or "unknown", fallback="unknown"))


def _title_component(metadata: VideoMetadata) -> str:
    return safe_path_component(metadata.title, fallback=metadata.video_id)


def init_run(metadata: VideoMetadata, config: AppConfig) -> RunPaths:
    run_id = make_run_id(metadata.video_id)
    work_dir = config.work_dir / run_id
    archive_dir = archive_dir_for(metadata, config)
    work_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(run_id=run_id, work_dir=work_dir, archive_dir=archive_dir)


def new_manifest(metadata: VideoMetadata, run_paths: RunPaths, version: str) -> Manifest:
    steps = {
        "metadata": StepState(),
        "download": StepState(),
        "subtitle": StepState(),
        "asr": StepState(),
        "ocr": StepState(),
        "diarization": StepState(),
        "fusion": StepState(),
        "markdown": StepState(),
    }
    now = iso_now()
    return Manifest(
        program="YomuTube",
        version=version,
        video_id=metadata.video_id,
        run_id=run_paths.run_id,
        status="running",
        steps=steps,
        models={},
        created_at=now,
        schema_version="1.0",
        archive_id=f"yt_{metadata.video_id}",
        source_url=metadata.webpage_url,
        updated_at=now,
        archive_path=str(run_paths.archive_dir),
    )


def manifest_path(archive_dir: str | Path) -> Path:
    return Path(archive_dir) / "manifest.json"


def load_manifest(archive_dir: str | Path) -> Manifest | None:
    path = manifest_path(archive_dir)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return Manifest.from_dict(json.load(fh))


def write_manifest(archive_dir: str | Path, manifest: Manifest) -> None:
    path = manifest_path(archive_dir)
    manifest.updated_at = iso_now()
    write_json(path, manifest.to_dict())


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    content = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(target, content)


def atomic_write_text(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        _fsync_dir(target.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def should_resume(archive_dir: Path, resume: bool = True) -> bool:
    if not resume:
        return False
    manifest = load_manifest(archive_dir)
    return bool(manifest and manifest.status == "done" and (archive_dir / "index.md").exists())


def cleanup_work_dir(work_dir: str | Path, *, keep: bool = False) -> None:
    path = Path(work_dir)
    if keep or not path.exists():
        return
    shutil.rmtree(path)


def clean_work_root(work_root: str | Path) -> list[Path]:
    root = Path(work_root).expanduser()
    if not root.exists():
        return []
    removed: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
            removed.append(child)
    return removed


def stable_run_suffix(url: str) -> str:
    return short_hash(url, 10)


class ArchiveLockError(RuntimeError):
    pass


class FileLock:
    def __init__(self, path: str | Path, *, if_running: str = "fail") -> None:
        if if_running not in {"fail", "skip", "wait"}:
            raise ValueError("if_running must be one of: fail, skip, wait")
        self.path = Path(path)
        self.if_running = if_running
        self.acquired = False
        self._fh: Any = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a+", encoding="utf-8")
        flags = fcntl.LOCK_EX
        if self.if_running in {"fail", "skip"}:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(self._fh.fileno(), flags)
        except BlockingIOError as exc:
            self._fh.close()
            self._fh = None
            if self.if_running == "skip":
                return False
            raise ArchiveLockError(f"lock already held: {self.path}") from exc
        self.acquired = True
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(f"pid={os.getpid()}\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        return True

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if self.acquired:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self.acquired = False
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


class ArchiveLock(FileLock):
    def __init__(self, archive_dir: str | Path, *, if_running: str = "fail") -> None:
        super().__init__(Path(archive_dir) / ".lock", if_running=if_running)
