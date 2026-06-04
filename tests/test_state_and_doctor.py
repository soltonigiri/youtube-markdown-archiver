from __future__ import annotations

import json
import multiprocessing
import shutil
from pathlib import Path

from yomutube.config import AppConfig
from yomutube.doctor import doctor_as_json, run_doctor
from yomutube.models import StepState, VideoMetadata
from yomutube.state import (
    ArchiveLock,
    archive_dir_for,
    cleanup_work_dir,
    existing_archive_for,
    init_run,
    load_manifest,
    new_manifest,
    write_json,
    write_manifest,
)
from yomutube.writers.jsonl import read_jsonl, write_jsonl


def test_archive_dir_is_stable(tmp_path) -> None:
    config = AppConfig.load().with_overrides({"paths": {"archive_dir": str(tmp_path / "archives"), "work_dir": str(tmp_path / "work")}})
    metadata = VideoMetadata(
        video_id="abc123",
        title="全「命中95の技」が、命中95である理由を解説/ポケモン",
        channel_id="UCxxx",
        channel_name="【ポケモン解説】運動クラブ",
        uploader=None,
        upload_date="20260401",
        duration_sec=10,
        webpage_url="https://youtu.be/abc123",
    )
    assert archive_dir_for(metadata, config) == (
        tmp_path
        / "archives"
        / "【ポケモン解説】運動クラブ"
        / "全「命中95の技」が、命中95である理由を解説／ポケモン"
    )


def test_existing_archive_finds_current_and_legacy_paths(tmp_path) -> None:
    config = AppConfig.load().with_overrides({"paths": {"archive_dir": str(tmp_path / "archives"), "work_dir": str(tmp_path / "work")}})
    metadata = VideoMetadata(
        video_id="abc123",
        title="Example Video",
        channel_id="UCxxx",
        channel_name="Example",
        uploader=None,
        upload_date="20260401",
        duration_sec=10,
        webpage_url="https://youtu.be/abc123",
    )
    readable = tmp_path / "archives" / "Example" / "Example Video"
    readable.mkdir(parents=True)
    assert existing_archive_for(metadata, config) == readable

    shutil.rmtree(readable.parent)
    legacy = tmp_path / "archives" / "UCxxx" / "20260401_abc123_Example_Video"
    legacy.mkdir(parents=True)
    assert existing_archive_for(metadata, config) == legacy

    current = tmp_path / "archives" / "UCxxx" / "abc123"
    current.mkdir()
    assert existing_archive_for(metadata, config) == current


def test_manifest_roundtrip(tmp_path) -> None:
    config = AppConfig.load().with_overrides({"paths": {"archive_dir": str(tmp_path / "archives"), "work_dir": str(tmp_path / "work")}})
    metadata = VideoMetadata(
        video_id="abc123",
        title="Example Video",
        channel_id=None,
        channel_name=None,
        uploader=None,
        upload_date=None,
        duration_sec=10,
        webpage_url="https://youtu.be/abc123",
    )
    run_paths = init_run(metadata, config)
    manifest = new_manifest(metadata, run_paths, "0.1.0")
    write_manifest(run_paths.archive_dir, manifest)
    loaded = load_manifest(run_paths.archive_dir)
    assert loaded is not None
    assert loaded.video_id == "abc123"
    assert loaded.schema_version == "1.0"
    assert loaded.archive_id == "yt_abc123"
    assert loaded.source_url == "https://youtu.be/abc123"
    assert isinstance(loaded.steps["metadata"], StepState)
    assert loaded.steps["metadata"].status == "pending"
    cleanup_work_dir(run_paths.work_dir)
    assert not run_paths.work_dir.exists()


def test_manifest_loads_legacy_version_and_step_strings(tmp_path) -> None:
    archive = tmp_path / "archives" / "unknown" / "abc123"
    archive.mkdir(parents=True)
    (archive / "manifest.json").write_text(
        json.dumps(
            {
                "program": "YomuTube",
                "program_version": "0.0.9",
                "video_id": "abc123",
                "run_id": "run",
                "status": "done",
                "steps": {"metadata": "done"},
                "models": {},
                "created_at": "2026-04-29T00:00:00+09:00",
            }
        ),
        encoding="utf-8",
    )
    manifest = load_manifest(archive)
    assert manifest is not None
    assert manifest.version == "0.0.9"
    assert isinstance(manifest.steps["metadata"], StepState)
    assert manifest.steps["metadata"].status == "done"


def test_atomic_json_and_jsonl_writes(tmp_path: Path) -> None:
    json_path = tmp_path / "archive" / "metadata.json"
    write_json(json_path, {"video_id": "abc123"})
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"video_id": "abc123"}
    assert list(json_path.parent.glob("*.tmp")) == []

    jsonl_path = tmp_path / "archive" / "segments.jsonl"
    write_jsonl(jsonl_path, [{"id": "s1"}, {"id": "s2"}])
    assert read_jsonl(jsonl_path) == [{"id": "s1"}, {"id": "s2"}]
    assert list(jsonl_path.parent.glob("*.tmp")) == []


def _try_archive_lock(archive_dir: str, queue: multiprocessing.Queue) -> None:
    with ArchiveLock(archive_dir, if_running="skip") as lock:
        queue.put(lock.acquired)


def test_archive_lock_skip_when_running(tmp_path: Path) -> None:
    archive = tmp_path / "archives" / "unknown" / "abc123"
    with ArchiveLock(archive, if_running="fail") as lock:
        assert lock.acquired is True
        queue: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=_try_archive_lock, args=(str(archive), queue))
        process.start()
        process.join(5)
        assert process.exitcode == 0
        assert queue.get(timeout=1) is False


def test_doctor_json_shape(tmp_path) -> None:
    config = AppConfig.load().with_overrides({"paths": {"archive_dir": str(tmp_path / "archives"), "work_dir": str(tmp_path / "work")}})
    checks = run_doctor(config)
    payload = doctor_as_json(checks)
    assert "ffmpeg" in payload
    assert any(check.name == "path:archive_dir" for check in checks)
