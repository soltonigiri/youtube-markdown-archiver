from __future__ import annotations

import importlib
from typing import Any

from .base import OCREngineUnavailable, OCRObservation, make_observation


class TesseractEngine:
    def __init__(self, *, languages: list[str] | None = None):
        self.languages = languages or ["ja", "en"]
        try:
            self._pytesseract = importlib.import_module("pytesseract")
        except ImportError as exc:
            raise OCREngineUnavailable("pytesseract is not installed.") from exc

    def recognize(
        self,
        image: Any,
        *,
        frame_ms: int,
        video_id: str,
        region: str,
        role: str | None = None,
    ) -> list[OCRObservation]:
        try:
            output = self._pytesseract.Output.DICT
            data = self._pytesseract.image_to_data(image, lang=tesseract_lang(self.languages), output_type=output)
        except Exception as exc:
            raise OCREngineUnavailable(f"Tesseract OCR is not available: {exc}") from exc

        observations: list[OCRObservation] = []
        count = len(data.get("text", []))
        for index in range(count):
            text = data["text"][index]
            confidence = _confidence(data.get("conf", [None] * count)[index])
            if confidence is not None and confidence < 0:
                continue
            left = int(data.get("left", [0] * count)[index])
            top = int(data.get("top", [0] * count)[index])
            width = int(data.get("width", [0] * count)[index])
            height = int(data.get("height", [0] * count)[index])
            observation = make_observation(
                video_id=video_id,
                frame_ms=frame_ms,
                text=text,
                region=region,
                role=role,
                bbox=(left, top, left + width, top + height),
                confidence=confidence,
                engine="tesseract",
            )
            if observation is not None:
                observations.append(observation)
        return observations


def tesseract_lang(languages: list[str]) -> str:
    mapping = {"ja": "jpn", "jpn": "jpn", "en": "eng", "eng": "eng"}
    values = [mapping.get(lang.lower(), lang.lower()) for lang in languages]
    return "+".join(dict.fromkeys(values)) or "eng"


def _confidence(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score / 100.0 if score > 1.0 else score
