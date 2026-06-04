"""OCR engines and post-processing."""

from .base import OCREngineUnavailable, OCRError, OCREvent, OCRObservation, OCRSettings, recognize_frame

__all__ = [
    "OCREngineUnavailable",
    "OCRError",
    "OCREvent",
    "OCRObservation",
    "OCRSettings",
    "recognize_frame",
]
