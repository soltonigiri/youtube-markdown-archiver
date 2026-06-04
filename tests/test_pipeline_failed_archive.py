from __future__ import annotations

import json
from pathlib import Path

from yomutube.config import AppConfig
from yomutube.pipeline import write_failed_metadata_archive


def test_write_failed_metadata_archive_keeps_retry_context(tmp_path: Path) -> None:
    config = AppConfig(
        {
            "app": {"version": "0.1.0"},
            "paths": {
                "archive_dir": str(tmp_path / "archives"),
                "work_dir": str(tmp_path / "work"),
            },
        }
    )

    archive = write_failed_metadata_archive(
        config,
        "https://www.youtube.com/watch?v=missing",
        RuntimeError("metadata temporarily unavailable"),
    )

    assert archive.parent.name == "failed"
    request = json.loads((archive / "request.json").read_text(encoding="utf-8"))
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((archive / "diagnostics.json").read_text(encoding="utf-8"))

    assert request["source_url"] == "https://www.youtube.com/watch?v=missing"
    assert manifest["status"] == "failed"
    assert manifest["steps"]["metadata"]["error_code"] == "YT_METADATA_FAILED"
    assert diagnostics["retryable"] is True
    assert diagnostics["error_code"] == "YT_METADATA_FAILED"
