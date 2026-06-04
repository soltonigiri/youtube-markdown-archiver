from __future__ import annotations

import wave

import numpy as np
import pytest

from yomutube.diarization.base import (
    DiarizationEngineUnavailable,
    DiarizationSettings,
    SpeakerTurn,
    assign_speakers_to_segments,
    diarize_audio,
)
from yomutube.diarization.pyannote_engine import PyannoteDiarizationEngine
from yomutube.models import Segment


def _write_wav(path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())


def test_diarization_off_returns_empty_list() -> None:
    assert diarize_audio("audio.wav", {"diarization": {"enabled": False}}) == []


def test_diarization_on_without_pyannote_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr("yomutube.diarization.pyannote_engine.importlib.import_module", fail_import)

    with pytest.raises(DiarizationEngineUnavailable):
        diarize_audio("audio.wav", {"diarization": {"enabled": True, "engine": "pyannote", "fallback_engine": None}})


def test_pyannote_unavailable_uses_local_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from yomutube.diarization.local_cluster_engine import LocalClusterDiarizationEngine
    from yomutube.diarization.pyannote_engine import PyannoteDiarizationEngine

    def fail_pyannote(self, audio_path, segments=None):
        raise DiarizationEngineUnavailable("gated model")

    def fallback_local(self, audio_path, segments=None):
        return [SpeakerTurn(speaker="SPEAKER_00", start_ms=0, end_ms=1000, metadata={"engine": "local_cluster"})]

    monkeypatch.setattr(PyannoteDiarizationEngine, "diarize", fail_pyannote)
    monkeypatch.setattr(LocalClusterDiarizationEngine, "diarize", fallback_local)

    turns = diarize_audio("audio.wav", {"diarization": {"enabled": True, "engine": "pyannote", "fallback_engine": "local_cluster"}})

    assert turns == [SpeakerTurn(speaker="SPEAKER_00", start_ms=0, end_ms=1000, metadata={"engine": "local_cluster"})]


def test_local_cluster_diarization_splits_synthetic_segments(tmp_path) -> None:
    sample_rate = 16000
    second = np.arange(sample_rate) / sample_rate
    speaker_a = 0.4 * np.sin(2 * np.pi * 220 * second)
    speaker_b = 0.4 * np.sin(2 * np.pi * 880 * second)
    audio = np.concatenate([speaker_a, speaker_b, speaker_a, speaker_b])
    audio_path = tmp_path / "synthetic.wav"
    _write_wav(audio_path, audio, sample_rate=sample_rate)

    segments = [
        Segment(id="asr_1", video_id="vid", source="asr", start_ms=0, end_ms=1000, text="a"),
        Segment(id="asr_2", video_id="vid", source="asr", start_ms=1000, end_ms=2000, text="b"),
        Segment(id="asr_3", video_id="vid", source="asr", start_ms=2000, end_ms=3000, text="a"),
        Segment(id="asr_4", video_id="vid", source="asr", start_ms=3000, end_ms=4000, text="b"),
    ]

    turns = diarize_audio(
        audio_path,
        {
            "diarization": {
                "enabled": True,
                "engine": "local_cluster",
                "local_cluster": {"num_speakers": 2, "min_segment_ms": 200},
            }
        },
        segments=segments,
    )

    speakers = [turn.speaker for turn in turns]
    assert speakers[0] == speakers[2]
    assert speakers[1] == speakers[3]
    assert speakers[0] != speakers[1]


def test_assign_speakers_to_segments_uses_largest_overlap() -> None:
    segments = [
        Segment(id="asr_1", video_id="vid", source="asr", start_ms=1000, end_ms=3000, text="hello"),
        Segment(id="asr_2", video_id="vid", source="asr", start_ms=3000, end_ms=4500, text="world"),
    ]
    turns = [
        SpeakerTurn(speaker="SPEAKER_00", start_ms=0, end_ms=2500),
        SpeakerTurn(speaker="SPEAKER_01", start_ms=2500, end_ms=5000),
    ]

    assigned = assign_speakers_to_segments(segments, turns)

    assert [segment.speaker for segment in assigned] == ["SPEAKER_00", "SPEAKER_01"]


def test_pyannote_diarize_output_is_converted_to_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTimeline:
        start = 1.2
        end = 3.4

    class FakeAnnotation:
        def itertracks(self, yield_label: bool = False):
            assert yield_label is True
            yield FakeTimeline(), None, "SPEAKER_07"

    class FakeOutput:
        speaker_diarization = FakeAnnotation()

    class FakePipeline:
        def __call__(self, audio_path: str, **kwargs):
            return FakeOutput()

    monkeypatch.setattr(PyannoteDiarizationEngine, "_pipeline", lambda self: FakePipeline())

    turns = PyannoteDiarizationEngine(DiarizationSettings(enabled=True)).diarize("audio.wav")

    assert turns == [SpeakerTurn(speaker="SPEAKER_07", start_ms=1200, end_ms=3400)]
