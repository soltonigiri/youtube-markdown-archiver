from __future__ import annotations

from types import SimpleNamespace

import pytest

from yomutube.ocr.base import OCREngineUnavailable
from yomutube.ocr.base import OCRObservation
from yomutube.ocr.dedupe import bbox_iou, dedupe_observations, text_similarity
from yomutube.ocr.paddle_engine import PaddleOCREngine


def test_text_similarity_uses_stdlib_fallback_or_optional_rapidfuzz() -> None:
    assert text_similarity("字幕とASRを統合する", "字幕と ASR を統合する") >= 0.86


def test_bbox_iou() -> None:
    assert round(bbox_iou((0, 0, 100, 100), (50, 50, 150, 150)), 2) == 0.14


def test_dedupe_observations_merges_repeated_caption() -> None:
    observations = [
        OCRObservation(
            video_id="vid",
            frame_ms=1000,
            text="重要: 字幕とASRを統合する",
            region="bottom_caption",
            role="caption",
            bbox=(10, 20, 500, 80),
            confidence=0.90,
            engine="paddleocr",
        ),
        OCRObservation(
            video_id="vid",
            frame_ms=1400,
            text="重要: 字幕と ASR を統合する",
            region="bottom_caption",
            role="caption",
            bbox=(12, 20, 502, 82),
            confidence=0.86,
            engine="paddleocr",
        ),
        OCRObservation(
            video_id="vid",
            frame_ms=5000,
            text="次の話題",
            region="bottom_caption",
            role="caption",
            bbox=(10, 20, 220, 80),
            confidence=0.92,
            engine="tesseract",
        ),
    ]

    events = dedupe_observations(observations, similarity_threshold=0.86, min_duration_ms=400)

    assert len(events) == 2
    assert events[0].id == "ocr_000001"
    assert events[0].start_ms == 1000
    assert events[0].end_ms == 1400
    assert events[0].observation_count == 2
    assert events[0].to_segment().source == "ocr"


def test_paddle_runtime_error_becomes_unavailable_for_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenPaddleOCR:
        def __init__(self, **kwargs):
            pass

        def ocr(self, image, **kwargs):
            if kwargs:
                raise TypeError("cls is unsupported")
            raise NotImplementedError("runtime failure")

    monkeypatch.setitem(
        __import__("sys").modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=BrokenPaddleOCR),
    )

    engine = PaddleOCREngine(languages=["ja", "en"])
    with pytest.raises(OCREngineUnavailable, match="PaddleOCR failed"):
        engine.recognize(object(), frame_ms=0, video_id="vid", region="bottom_caption")
