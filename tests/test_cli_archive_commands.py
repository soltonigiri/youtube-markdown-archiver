from __future__ import annotations

import json

from typer.testing import CliRunner

from yomutube.cli import app


runner = CliRunner()


def _write_archive(root, channel: str, name: str, *, video_id: str, title: str, text: str) -> None:
    archive = root / channel / name
    archive.mkdir(parents=True)
    (archive / "manifest.json").write_text(
        json.dumps(
            {
                "program": "YomuTube",
                "program_version": "0.1.0",
                "video_id": video_id,
                "run_id": "run1",
                "status": "done",
                "steps": {},
                "models": {},
                "created_at": "2026-04-29T00:00:00+09:00",
                "finished_at": "2026-04-29T00:01:00+09:00",
                "archive_path": str(archive),
            }
        ),
        encoding="utf-8",
    )
    (archive / "metadata.json").write_text(
        json.dumps({"video_id": video_id, "title": title, "channel_name": channel, "duration": 10}),
        encoding="utf-8",
    )
    (archive / "segments.jsonl").write_text(
        json.dumps({"video_id": video_id, "start_ms": 0, "end_ms": 1000, "text": text}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (archive / "index.md").write_text(f"# {title}\n", encoding="utf-8")


def test_archive_cli_list_show_search_last_open(tmp_path) -> None:
    _write_archive(tmp_path, "UCexample", "20260401_abc123_Example", video_id="abc123", title="Example", text="hello local archive")

    list_result = runner.invoke(app, ["list", "--archive-dir", str(tmp_path)])
    assert list_result.exit_code == 0
    assert "abc123" in list_result.output

    show_result = runner.invoke(app, ["show", "abc123", "--archive-dir", str(tmp_path)])
    assert show_result.exit_code == 0
    assert "Example" in show_result.output

    search_result = runner.invoke(app, ["search", "local", "--archive-dir", str(tmp_path)])
    assert search_result.exit_code == 0
    assert "hello local archive" in search_result.output

    last_result = runner.invoke(app, ["last", "--archive-dir", str(tmp_path)])
    assert last_result.exit_code == 0
    assert "abc123" in last_result.output

    open_result = runner.invoke(app, ["open", "abc123", "--archive-dir", str(tmp_path)])
    assert open_result.exit_code == 0
    assert "index.md" in open_result.output

    editor_result = runner.invoke(app, ["open", "abc123", "true", "--archive-dir", str(tmp_path)])
    assert editor_result.exit_code == 0
