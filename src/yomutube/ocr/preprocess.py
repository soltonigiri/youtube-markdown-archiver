from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class OCRRegion:
    name: str
    role: str | None = None
    enabled: bool = True
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0
    preprocess: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> "OCRRegion":
        return cls(
            name=str(data.get("name", "region")),
            role=data.get("role"),
            enabled=bool(data.get("enabled", True)),
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            w=float(data.get("w", 1.0)),
            h=float(data.get("h", 1.0)),
            preprocess=dict(data.get("preprocess") or {}),
        )


def crop_roi(frame: Any, region: OCRRegion | dict[str, Any]) -> Any:
    roi = region if isinstance(region, OCRRegion) else OCRRegion.from_config(region)
    if not hasattr(frame, "shape"):
        raise ValueError("frame must expose a shape attribute")
    height, width = frame.shape[:2]
    left = _clamp(round(roi.x * width), 0, width)
    top = _clamp(round(roi.y * height), 0, height)
    right = _clamp(round((roi.x + roi.w) * width), left, width)
    bottom = _clamp(round((roi.y + roi.h) * height), top, height)
    return frame[top:bottom, left:right]


def preprocess_roi(image: Any, options: dict[str, Any] | None = None) -> Any:
    opts = dict(options or {})
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError:
        return image

    result = image
    upscale = float(opts.get("upscale") or 1.0)
    if upscale != 1.0 and hasattr(result, "shape"):
        height, width = result.shape[:2]
        result = cv2.resize(result, (max(1, round(width * upscale)), max(1, round(height * upscale))))
    if opts.get("grayscale"):
        if hasattr(result, "shape") and len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    if opts.get("denoise"):
        if hasattr(result, "shape") and len(result.shape) == 2:
            result = cv2.fastNlMeansDenoising(result, None, 10, 7, 21)
        else:
            result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
    threshold = str(opts.get("threshold") or "none").lower()
    if threshold == "adaptive":
        gray = _ensure_gray(cv2, result)
        result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)
    elif threshold == "otsu":
        gray = _ensure_gray(cv2, result)
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result


def prepare_region(frame: Any, region: OCRRegion | dict[str, Any]) -> Any:
    roi = region if isinstance(region, OCRRegion) else OCRRegion.from_config(region)
    return preprocess_roi(crop_roi(frame, roi), roi.preprocess)


def _ensure_gray(cv2: Any, image: Any) -> Any:
    if hasattr(image, "shape") and len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))
