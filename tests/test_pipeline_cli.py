from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import yaml
from typer.testing import CliRunner

from yomutube.cli import app
from yomutube.config import AppConfig
from yomutube.download import ytdlp_client


def test_subtitle_archive_transcribe_resume_export_and_index(tmp_path, monkeypatch) -> None:
    downloads = []
    info = {
        "id": "fixture123", "title": "Subtitle example", "channel_id": "UCexample",
        "channel": "Example", "duration": 10, "upload_date": "20260401",
        "webpage_url": "https://www.youtube.com/watch?v=fixture123",
        "automatic_captions": {"en": [{"ext": "vtt", "url": "https://example.test/captions"}]},
    }

    class FixtureYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def extract_info(self, url, download=False):
            if download:
                downloads.append(url)
                media = Path(self.options["outtmpl"]["default"]).parent
                (media / "fixture123.mp4").write_bytes(b"subtitle-only fixture")
                if self.options["writeautomaticsub"]:
                    (media / "subtitles" / "fixture123.en.vtt").write_text(
                        "WEBVTT\n\n00:00:00.000 --> 00:00:10.000\nHello archive.\n",
                        encoding="utf-8",
                    )
            return info

    monkeypatch.setattr(ytdlp_client, "import_ytdlp", lambda: SimpleNamespace(YoutubeDL=FixtureYoutubeDL))
    config = AppConfig.load().with_overrides({
        "paths": {"archive_dir": str(tmp_path / "archives"), "work_dir": str(tmp_path / "work")},
    })
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.data), encoding="utf-8")
    channels = tmp_path / "channels"
    channels.mkdir()
    (channels / "example.yaml").write_text(
        "match:\n  channel_id: UCexample\nsubtitles:\n  fallback_to_auto: true\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    def invoke(*args):
        result = runner.invoke(app, [*args, "--config", str(config_path)])
        assert result.exit_code == 0, result.output or repr(result.exception)
        return result.output

    inspected = json.loads(invoke("inspect", info["webpage_url"], "--json"))
    assert inspected["subtitle_track"]["source"] == "auto"
    args = ("transcribe", info["webpage_url"], "--max-height", "720", "--subtitle-policy", "prefer_manual")
    assert "Done:" in invoke(*args)
    archive = next((tmp_path / "archives").glob("*/*"))
    segments_before = (archive / "segments.jsonl").read_bytes()
    assert "Hello archive." in (archive / "index.md").read_text()
    manifest = json.loads((archive / "manifest.json").read_text())
    assert manifest["status"] == "done"
    assert manifest["steps"]["asr"]["status"] == "skipped"
    assert "Resumed:" in invoke(*args)
    assert len(downloads) == 1
    assert (archive / "segments.jsonl").read_bytes() == segments_before
    assert "Hello archive." in invoke("search", "Hello")
    exported = tmp_path / "captions.srt"
    invoke("export", info["id"], "--format", "srt", "--output", str(exported))
    assert "00:00:00,000 --> 00:00:10,000\nHello archive." in exported.read_text()
    database = tmp_path / "index.db"
    invoke("index", "--output", str(database))
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT text FROM segments").fetchall() == [("Hello archive.",)]

    info["duration"] = 8000
    info["automatic_captions"] = {}
    config_path.write_text(yaml.safe_dump(config.with_overrides({"ocr": {"enabled": "auto"}}).data))
    inspected = json.loads(invoke("inspect", info["webpage_url"], "--json"))
    assert inspected["planned"]["ocr"] == "skipped"
