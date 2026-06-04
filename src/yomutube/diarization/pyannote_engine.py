from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from yomutube.models import Segment

from .base import DiarizationEngineUnavailable, DiarizationSettings, SpeakerTurn


class PyannoteDiarizationEngine:
    def __init__(self, settings: DiarizationSettings):
        self.settings = settings

    def diarize(self, audio_path: str | Path, *, segments: list[Segment] | None = None) -> list[SpeakerTurn]:
        pipeline = self._pipeline()
        kwargs = {
            "min_speakers": self.settings.min_speakers,
            "max_speakers": self.settings.max_speakers,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        try:
            diarization = pipeline(str(audio_path), **kwargs)
        except Exception as exc:
            raise DiarizationEngineUnavailable(f"pyannote diarization failed: {exc}") from exc
        turns: list[SpeakerTurn] = []
        annotation = _annotation_from_output(diarization)
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(
                    speaker=str(speaker),
                    start_ms=round(float(turn.start) * 1000),
                    end_ms=round(float(turn.end) * 1000),
                )
            )
        return turns

    def _pipeline(self) -> Any:
        try:
            pyannote_audio = importlib.import_module("pyannote.audio")
        except ImportError as exc:
            raise DiarizationEngineUnavailable(
                "pyannote.audio is required when diarization is enabled."
            ) from exc

        token = os.environ.get(self.settings.hf_token_env)
        try:
            try:
                pipeline = pyannote_audio.Pipeline.from_pretrained(self.settings.model, token=token)
            except TypeError:
                pipeline = pyannote_audio.Pipeline.from_pretrained(self.settings.model, use_auth_token=token)
        except Exception as exc:
            raise DiarizationEngineUnavailable(f"pyannote pipeline unavailable: {exc}") from exc

        device = self._device()
        if device is not None and hasattr(pipeline, "to"):
            pipeline.to(device)
        return pipeline

    def _device(self) -> Any:
        requested = self.settings.device.lower()
        if not requested:
            return None
        if requested == "cuda" and not self._cuda_available():
            requested = "cpu"
        try:
            torch = importlib.import_module("torch")
            return torch.device(requested)
        except ImportError:
            return requested

    @staticmethod
    def _cuda_available() -> bool:
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            return False
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False


def _annotation_from_output(output: Any) -> Any:
    if hasattr(output, "itertracks"):
        return output
    speaker_diarization = getattr(output, "speaker_diarization", None)
    if speaker_diarization is not None and hasattr(speaker_diarization, "itertracks"):
        return speaker_diarization
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    if exclusive is not None and hasattr(exclusive, "itertracks"):
        return exclusive
    raise DiarizationEngineUnavailable("pyannote output does not contain diarization tracks")
