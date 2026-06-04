from __future__ import annotations

import importlib
from typing import Any, Iterable

from .base import OCREngineUnavailable, OCRObservation, make_observation


class PaddleOCREngine:
    def __init__(self, *, languages: list[str] | None = None):
        self.languages = languages or ["ja", "en"]
        try:
            module = importlib.import_module("paddleocr")
        except ImportError as exc:
            raise OCREngineUnavailable("PaddleOCR is not installed.") from exc
        lang = paddle_lang(self.languages)
        try:
            self._ocr = module.PaddleOCR(lang=lang, use_angle_cls=True)
        except TypeError:
            self._ocr = module.PaddleOCR(lang=lang)

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
            try:
                raw = self._ocr.ocr(image, cls=True)
            except TypeError:
                raw = self._ocr.ocr(image)
        except Exception as exc:
            raise OCREngineUnavailable(f"PaddleOCR failed: {exc}") from exc

        observations: list[OCRObservation] = []
        for bbox, text, confidence in iter_paddle_results(raw):
            observation = make_observation(
                video_id=video_id,
                frame_ms=frame_ms,
                text=text,
                region=region,
                role=role,
                bbox=bbox,
                confidence=confidence,
                engine="paddleocr",
            )
            if observation is not None:
                observations.append(observation)
        return observations


def paddle_lang(languages: list[str]) -> str:
    normalized = {lang.lower() for lang in languages}
    if "ja" in normalized or "jpn" in normalized or "japan" in normalized:
        return "japan"
    if "en" in normalized or "eng" in normalized:
        return "en"
    return next(iter(normalized), "en")


def iter_paddle_results(raw: Any) -> Iterable[tuple[Any, str, float | None]]:
    if raw is None:
        return
    if isinstance(raw, dict):
        texts = _first_present(raw, "rec_texts", "texts")
        scores = _first_present(raw, "rec_scores", "scores")
        boxes = _first_present(raw, "rec_boxes", "dt_polys", "boxes")
        if texts is None:
            texts = []
        if scores is None:
            scores = []
        if boxes is None:
            boxes = []
        if hasattr(texts, "tolist"):
            texts = texts.tolist()
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if hasattr(boxes, "tolist"):
            boxes = boxes.tolist()
        for index, text in enumerate(texts):
            yield boxes[index] if index < len(boxes) else None, str(text), _score(scores[index] if index < len(scores) else None)
        return
    if isinstance(raw, (list, tuple)):
        if _looks_like_ocr_item(raw):
            bbox, text, confidence = _parse_ocr_item(raw)
            yield bbox, text, confidence
            return
        for item in raw:
            yield from iter_paddle_results(item)


def _looks_like_ocr_item(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[1], (list, tuple))
        and len(value[1]) >= 1
        and isinstance(value[1][0], str)
    )


def _parse_ocr_item(value: Any) -> tuple[Any, str, float | None]:
    bbox = value[0]
    text = str(value[1][0])
    confidence = _score(value[1][1] if len(value[1]) > 1 else None)
    return bbox, text, confidence


def _score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score / 100.0 if score > 1.0 else score


def _first_present(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None
