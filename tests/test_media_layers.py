from __future__ import annotations

import subprocess

import pytest

from yomutube.config import AppConfig
from yomutube.media import audio, ffmpeg, frames


def test_ffmpeg_check_binary_raises_for_missing(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda name: None)

    with pytest.raises(ffmpeg.FFmpegDependencyError):
        ffmpeg.check_binary("ffmpeg")


def test_extract_audio_builds_expected_ffmpeg_command(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_text("media", encoding="utf-8")
    output = tmp_path / "audio.wav"
    calls: list[list[str]] = []

    monkeypatch.setattr(ffmpeg, "require_ffmpeg", lambda binary: "/usr/bin/ffmpeg")

    def fake_run(args, *, label):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(ffmpeg, "run_command", fake_run)

    config = AppConfig({"media": {"audio_sample_rate": 8000}, "performance": {"ffmpeg_threads": 2}})
    result = audio.extract_audio_wav(source, output, config)

    assert result == output
    assert calls[0][:4] == ["/usr/bin/ffmpeg", "-y", "-i", str(source)]
    assert "-vn" in calls[0]
    assert calls[0][-7:] == ["-threads", "2", "-ac", "1", "-ar", "8000", "-c:a", "pcm_s16le", str(output)][-7:]


def test_frame_sampling_can_skip_when_opencv_missing(monkeypatch) -> None:
    monkeypatch.setattr(frames, "_import_cv2", lambda: None)

    assert frames.sample_roi_frames("missing.mp4", on_missing="skip") == []
    with pytest.raises(frames.OpenCVUnavailableError):
        frames.sample_roi_frames("missing.mp4")
