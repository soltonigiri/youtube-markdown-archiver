from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from yomutube.config import AppConfig
from yomutube.models import Segment


class DiarizationError(RuntimeError):
    """Base diarization error."""


class DiarizationEngineUnavailable(DiarizationError):
    """Raised when diarization is enabled but the engine is unavailable."""


@dataclass(slots=True)
class SpeakerTurn:
    speaker: str
    start_ms: int
    end_ms: int
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass(slots=True)
class DiarizationSettings:
    enabled: bool = False
    engine: str = "pyannote"
    model: str = "pyannote/speaker-diarization-community-1"
    fallback_engine: str | None = "local_cluster"
    device: str = "cuda"
    min_speakers: int | None = None
    max_speakers: int | None = None
    hf_token_env: str = "HF_TOKEN"
    local_num_speakers: int = 2
    local_window_ms: int = 2000
    local_hop_ms: int = 1000
    local_min_segment_ms: int = 500

    @classmethod
    def from_config(cls, config: AppConfig | None) -> "DiarizationSettings":
        value = config.get("diarization", {}) if config else {}
        section = value if isinstance(value, dict) else {}
        local = section.get("local_cluster") if isinstance(section.get("local_cluster"), dict) else {}
        fallback = section.get("fallback_engine", "local_cluster")
        return cls(
            enabled=bool(section.get("enabled", False)),
            engine=str(section.get("engine", "pyannote")),
            model=str(section.get("model", "pyannote/speaker-diarization-community-1")),
            fallback_engine=str(fallback) if fallback else None,
            device=str(section.get("device", "cuda")),
            min_speakers=section.get("min_speakers"),
            max_speakers=section.get("max_speakers"),
            hf_token_env=str(section.get("hf_token_env", "HF_TOKEN")),
            local_num_speakers=int(local.get("num_speakers", 2)),
            local_window_ms=int(local.get("window_ms", 2000)),
            local_hop_ms=int(local.get("hop_ms", 1000)),
            local_min_segment_ms=int(local.get("min_segment_ms", 500)),
        )


def diarize_audio(
    audio_path: str | Path,
    config: AppConfig | None,
    *,
    segments: list[Segment] | None = None,
) -> list[SpeakerTurn]:
    settings = DiarizationSettings.from_config(config)
    if not settings.enabled:
        return []
    try:
        return _diarize_with_engine(settings.engine, settings, audio_path, segments=segments)
    except DiarizationEngineUnavailable:
        fallback = (settings.fallback_engine or "").strip()
        if fallback and fallback.lower() != settings.engine.lower():
            return _diarize_with_engine(fallback, settings, audio_path, segments=segments)
        raise


def _diarize_with_engine(
    engine: str,
    settings: DiarizationSettings,
    audio_path: str | Path,
    *,
    segments: list[Segment] | None = None,
) -> list[SpeakerTurn]:
    normalized = engine.lower()
    if normalized == "pyannote":
        from .pyannote_engine import PyannoteDiarizationEngine

        return PyannoteDiarizationEngine(settings).diarize(audio_path, segments=segments)
    if normalized == "local_cluster":
        from .local_cluster_engine import LocalClusterDiarizationEngine

        return LocalClusterDiarizationEngine(settings).diarize(audio_path, segments=segments)
    raise DiarizationError(f"unsupported diarization engine: {engine}")


def assign_speakers_to_segments(
    segments: list[Segment],
    turns: list[SpeakerTurn],
    *,
    min_overlap_ratio: float = 0.0,
) -> list[Segment]:
    if not turns:
        return list(segments)
    assigned: list[Segment] = []
    for segment in segments:
        best_turn = max(turns, key=lambda turn: overlap_ms(segment.start_ms, segment.end_ms, turn.start_ms, turn.end_ms))
        overlap = overlap_ms(segment.start_ms, segment.end_ms, best_turn.start_ms, best_turn.end_ms)
        ratio = overlap / segment.duration_ms if segment.duration_ms else 0.0
        if overlap > 0 and ratio >= min_overlap_ratio:
            assigned.append(replace(segment, speaker=best_turn.speaker))
        else:
            assigned.append(segment)
    return assigned


def overlap_ms(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start))
