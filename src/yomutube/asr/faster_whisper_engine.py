from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .base import ASRDeviceError, ASREngineUnavailable, ASRError, ASRSettings, make_asr_segment


def cuda_available() -> bool:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cpu_compute_type(compute_type: str) -> str:
    return "int8" if "float16" in compute_type or compute_type == "float16" else compute_type


def select_runtime(settings: ASRSettings) -> tuple[str, str, str | None]:
    requested = settings.device.lower()
    if requested == "cuda" and not cuda_available():
        if settings.strict_device:
            raise ASRDeviceError("ASR device 'cuda' was requested, but CUDA is not available.")
        return "cpu", cpu_compute_type(settings.compute_type), "cuda_unavailable"
    return requested, settings.compute_type, None


class FasterWhisperEngine:
    def __init__(self, settings: ASRSettings):
        self.settings = settings

    def transcribe(self, audio_path: str | Path, *, video_id: str) -> list:
        device, compute_type, fallback_reason = select_runtime(self.settings)
        try:
            return self._transcribe_once(
                audio_path,
                video_id=video_id,
                device=device,
                compute_type=compute_type,
                fallback_reason=fallback_reason,
            )
        except ASREngineUnavailable:
            raise
        except Exception as exc:
            if device == "cuda" and not self.settings.strict_device:
                try:
                    return self._transcribe_once(
                        audio_path,
                        video_id=video_id,
                        device="cpu",
                        compute_type=cpu_compute_type(self.settings.compute_type),
                        fallback_reason=f"cuda_runtime_error: {exc}",
                    )
                except Exception as cpu_exc:
                    raise ASRError(f"faster-whisper failed after CPU fallback: {cpu_exc}") from cpu_exc
            if device == "cuda" and self.settings.strict_device:
                raise ASRDeviceError(f"faster-whisper failed on strict CUDA device: {exc}") from exc
            raise ASRError(f"faster-whisper failed: {exc}") from exc

    def _module(self) -> Any:
        try:
            return importlib.import_module("faster_whisper")
        except ImportError as exc:
            raise ASREngineUnavailable(
                "faster-whisper is required when ASR is enabled. "
                "Install yomutube with the 'full' extra or disable ASR."
            ) from exc

    def _transcribe_once(
        self,
        audio_path: str | Path,
        *,
        video_id: str,
        device: str,
        compute_type: str,
        fallback_reason: str | None,
    ) -> list:
        module = self._module()
        model = module.WhisperModel(
            self.settings.model,
            device=device,
            compute_type=compute_type,
        )
        runner: Any = model
        pipeline_class = getattr(module, "BatchedInferencePipeline", None)
        if pipeline_class is not None and self.settings.batch_size > 1:
            runner = pipeline_class(model=model)

        kwargs: dict[str, Any] = {
            "beam_size": self.settings.beam_size,
            "vad_filter": self.settings.vad_filter,
            "word_timestamps": self.settings.word_timestamps,
            "language": self.settings.language,
            "initial_prompt": self.settings.initial_prompt,
        }
        if runner is not model:
            kwargs["batch_size"] = self.settings.batch_size
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        raw_segments, info = runner.transcribe(str(audio_path), **kwargs)
        language = getattr(info, "language", None) or self.settings.language
        metadata: dict[str, Any] = {}
        if fallback_reason:
            metadata["fallback_reason"] = fallback_reason

        segments = []
        for index, item in enumerate(raw_segments, start=1):
            words = self._words(item)
            segment = make_asr_segment(
                index=index,
                video_id=video_id,
                start_sec=getattr(item, "start", 0.0),
                end_sec=getattr(item, "end", 0.0),
                text=getattr(item, "text", ""),
                language=language,
                engine="faster-whisper",
                model=self.settings.model,
                device=device,
                compute_type=compute_type,
                confidence=getattr(item, "avg_logprob", None),
                words=words,
                metadata=metadata,
            )
            if segment is not None:
                segments.append(segment)
        return segments

    @staticmethod
    def _words(item: Any) -> list[dict[str, Any]] | None:
        raw_words = getattr(item, "words", None)
        if raw_words is None:
            return None
        words: list[dict[str, Any]] = []
        for word in raw_words:
            words.append(
                {
                    "start_ms": round(float(getattr(word, "start", 0.0)) * 1000),
                    "end_ms": round(float(getattr(word, "end", 0.0)) * 1000),
                    "word": getattr(word, "word", ""),
                    "probability": getattr(word, "probability", None),
                }
            )
        return words
