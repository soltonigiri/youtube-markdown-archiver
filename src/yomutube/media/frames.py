from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


class OpenCVUnavailableError(RuntimeError):
    """Raised when OpenCV frame sampling is requested but cv2 is missing."""


class FrameSamplingError(RuntimeError):
    """Raised when OpenCV cannot read the requested video."""


@dataclass(slots=True)
class FrameSample:
    timestamp_ms: int
    frame_index: int
    region_name: str
    image: Any
    x: int
    y: int
    width: int
    height: int


def _import_cv2() -> Any | None:
    try:
        return importlib.import_module("cv2")
    except ModuleNotFoundError as exc:
        if exc.name == "cv2":
            return None
        raise


def opencv_available() -> bool:
    return _import_cv2() is not None


def _normalized_regions(regions: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if regions is None:
        return [{"name": "full_frame", "enabled": True, "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}]
    normalized: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        if not region or region.get("enabled", True) is False:
            continue
        normalized.append(
            {
                "name": str(region.get("name") or f"region_{index + 1}"),
                "x": float(region.get("x", 0.0)),
                "y": float(region.get("y", 0.0)),
                "w": float(region.get("w", 1.0)),
                "h": float(region.get("h", 1.0)),
            }
        )
    return normalized


def _crop(frame: Any, region: dict[str, Any]) -> tuple[Any, int, int, int, int] | None:
    frame_height, frame_width = frame.shape[:2]
    left = max(0, min(frame_width, round(region["x"] * frame_width)))
    top = max(0, min(frame_height, round(region["y"] * frame_height)))
    right = max(left, min(frame_width, round((region["x"] + region["w"]) * frame_width)))
    bottom = max(top, min(frame_height, round((region["y"] + region["h"]) * frame_height)))
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return None
    return frame[top:bottom, left:right].copy(), left, top, width, height


def _iter_roi_frames(
    cv2: Any,
    video_path: Path,
    *,
    regions: Iterable[dict[str, Any]] | None,
    fps: float,
    start_ms: int,
    end_ms: int | None,
) -> Iterator[FrameSample]:
    if fps <= 0:
        raise ValueError("fps must be greater than 0")
    source = Path(video_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"video not found: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FrameSamplingError(f"OpenCV could not open video: {source}")
    try:
        native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if end_ms is None and native_fps > 0 and frame_count > 0:
            end_ms = round(frame_count / native_fps * 1000)
        interval_ms = round(1000 / fps)
        timestamp_ms = max(0, int(start_ms))
        selected_regions = _normalized_regions(regions)
        while end_ms is None or timestamp_ms <= end_ms:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = round(timestamp_ms / 1000 * native_fps)
            for region in selected_regions:
                crop = _crop(frame, region)
                if crop is None:
                    continue
                image, x, y, width, height = crop
                yield FrameSample(
                    timestamp_ms=timestamp_ms,
                    frame_index=frame_index,
                    region_name=region["name"],
                    image=image,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                )
            timestamp_ms += interval_ms
    finally:
        capture.release()


def iter_roi_frames(
    video_path: str | Path,
    *,
    regions: Iterable[dict[str, Any]] | None = None,
    fps: float = 2.0,
    start_ms: int = 0,
    end_ms: int | None = None,
    on_missing: str = "error",
) -> Iterator[FrameSample]:
    cv2 = _import_cv2()
    if cv2 is None:
        if on_missing == "skip":
            return iter(())
        raise OpenCVUnavailableError(
            "opencv-python is required for frame sampling. Install with: python -m pip install opencv-python"
        )
    return _iter_roi_frames(
        cv2,
        Path(video_path),
        regions=regions,
        fps=fps,
        start_ms=start_ms,
        end_ms=end_ms,
    )


def sample_roi_frames(
    video_path: str | Path,
    *,
    regions: Iterable[dict[str, Any]] | None = None,
    fps: float = 2.0,
    start_ms: int = 0,
    end_ms: int | None = None,
    on_missing: str = "error",
) -> list[FrameSample]:
    return list(
        iter_roi_frames(
            video_path,
            regions=regions,
            fps=fps,
            start_ms=start_ms,
            end_ms=end_ms,
            on_missing=on_missing,
        )
    )


sample_frames = sample_roi_frames
