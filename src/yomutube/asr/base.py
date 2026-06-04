from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from yomutube.config import AppConfig
from yomutube.models import Segment
from yomutube.utils.text import normalize_text


class ASRError(RuntimeError):
    """Base ASR error."""


class ASREngineUnavailable(ASRError):
    """Raised when the configured ASR engine is not installed."""


class ASRDeviceError(ASRError):
    """Raised when the requested ASR device cannot be used."""


class ASREngine(Protocol):
    def transcribe(self, audio_path: str | Path, *, video_id: str) -> list[Segment]:
        ...


@dataclass(slots=True)
class ASRSettings:
    enabled: bool = True
    engine: str = "faster-whisper"
    model: str = "medium"
    device: str = "cuda"
    strict_device: bool = False
    compute_type: str = "int8_float16"
    batch_size: int = 8
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = False
    language: str | None = None
    initial_prompt: str | None = None

    @classmethod
    def from_config(cls, config: AppConfig | dict[str, Any] | None) -> "ASRSettings":
        section = _section(config, "asr")
        language = section.get("language", "auto")
        if language == "auto":
            language = None
        return cls(
            enabled=bool(section.get("enabled", True)),
            engine=str(section.get("engine", "faster-whisper")),
            model=str(section.get("model", "medium")),
            device=str(section.get("device", "cuda")),
            strict_device=bool(section.get("strict_device", False)),
            compute_type=str(section.get("compute_type", "int8_float16")),
            batch_size=int(section.get("batch_size", 8)),
            beam_size=int(section.get("beam_size", 5)),
            vad_filter=bool(section.get("vad_filter", True)),
            word_timestamps=bool(section.get("word_timestamps", False)),
            language=language,
            initial_prompt=section.get("initial_prompt"),
        )


def _section(config: AppConfig | dict[str, Any] | None, key: str) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, AppConfig):
        value = config.get(key, {})
    else:
        value = config.get(key, {})
    return dict(value or {}) if isinstance(value, dict) else {}


def make_asr_segment(
    *,
    index: int,
    video_id: str,
    start_sec: float,
    end_sec: float,
    text: str,
    language: str | None,
    engine: str,
    model: str,
    device: str,
    compute_type: str,
    confidence: float | None = None,
    words: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Segment | None:
    normalized = normalize_text(text)
    if not normalized:
        return None
    segment_metadata: dict[str, Any] = {
        "engine": engine,
        "model": model,
        "device": device,
        "compute_type": compute_type,
    }
    if words is not None:
        segment_metadata["words"] = words
    if metadata:
        segment_metadata.update(metadata)
    return Segment(
        id=f"asr_{index:06d}",
        video_id=video_id,
        source="asr",
        start_ms=round(float(start_sec) * 1000),
        end_ms=round(float(end_sec) * 1000),
        text=normalized,
        language=language,
        confidence=confidence,
        metadata=segment_metadata,
    )


def transcribe_audio(
    audio_path: str | Path,
    config: AppConfig | dict[str, Any] | None,
    *,
    video_id: str = "unknown",
) -> list[Segment]:
    settings = ASRSettings.from_config(config)
    if not settings.enabled:
        return []
    engine_name = settings.engine.replace("_", "-").lower()
    if engine_name != "faster-whisper":
        raise ASRError(f"unsupported ASR engine: {settings.engine}")

    from .faster_whisper_engine import FasterWhisperEngine

    return FasterWhisperEngine(settings).transcribe(audio_path, video_id=video_id)
