from __future__ import annotations

from types import SimpleNamespace

import pytest

from yomutube.asr.base import ASRDeviceError, ASRSettings
from yomutube.asr.faster_whisper_engine import FasterWhisperEngine, select_runtime


def test_select_runtime_falls_back_to_cpu_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yomutube.asr.faster_whisper_engine.cuda_available", lambda: False)
    settings = ASRSettings(device="cuda", strict_device=False, compute_type="int8_float16")

    assert select_runtime(settings) == ("cpu", "int8", "cuda_unavailable")


def test_select_runtime_raises_on_strict_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yomutube.asr.faster_whisper_engine.cuda_available", lambda: False)
    settings = ASRSettings(device="cuda", strict_device=True)

    with pytest.raises(ASRDeviceError):
        select_runtime(settings)


def test_faster_whisper_output_is_converted_to_segments(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    class RawSegment:
        start = 1.2
        end = 2.5
        text = "  hello   world  "
        avg_logprob = -0.2
        words = None

    class Info:
        language = "en"

    class WhisperModel:
        def __init__(self, model: str, *, device: str, compute_type: str):
            self.model = model
            self.device = device
            self.compute_type = compute_type

    class BatchedInferencePipeline:
        def __init__(self, *, model: WhisperModel):
            self.model = model

        def transcribe(self, audio_path: str, **kwargs):
            return [RawSegment()], Info()

    monkeypatch.setattr("yomutube.asr.faster_whisper_engine.cuda_available", lambda: False)
    fake_module = SimpleNamespace(
        WhisperModel=WhisperModel,
        BatchedInferencePipeline=BatchedInferencePipeline,
    )
    monkeypatch.setattr("yomutube.asr.faster_whisper_engine.importlib.import_module", lambda name: fake_module)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    segments = FasterWhisperEngine(ASRSettings(device="cuda", strict_device=False)).transcribe(
        audio_path,
        video_id="vid",
    )

    assert len(segments) == 1
    assert segments[0].id == "asr_000001"
    assert segments[0].video_id == "vid"
    assert segments[0].source == "asr"
    assert segments[0].start_ms == 1200
    assert segments[0].end_ms == 2500
    assert segments[0].text == "hello world"
    assert segments[0].language == "en"
    assert segments[0].metadata["device"] == "cpu"
    assert segments[0].metadata["fallback_reason"] == "cuda_unavailable"
