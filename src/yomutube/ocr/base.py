from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from yomutube.config import AppConfig
from yomutube.models import Segment
from yomutube.utils.text import normalize_text

BBox = tuple[int, int, int, int]


class OCRError(RuntimeError):
    """Base OCR error."""


class OCREngineUnavailable(OCRError):
    """Raised when an OCR engine is not installed or cannot run."""


class OCREngine(Protocol):
    def recognize(
        self,
        image: Any,
        *,
        frame_ms: int,
        video_id: str,
        region: str,
        role: str | None = None,
    ) -> list["OCRObservation"]:
        ...


@dataclass(slots=True)
class OCRObservation:
    video_id: str
    frame_ms: int
    text: str
    region: str
    role: str | None = None
    bbox: BBox | None = None
    confidence: float | None = None
    engine: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = normalize_text(self.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OCREvent:
    id: str
    video_id: str
    start_ms: int
    end_ms: int
    text: str
    region: str
    role: str | None = None
    bbox: BBox | None = None
    confidence: float | None = None
    duplicate_of: str | None = None
    source: str = "ocr"
    observation_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = normalize_text(self.text)

    def to_segment(self) -> Segment:
        metadata = dict(self.metadata)
        metadata["region"] = self.region
        metadata["observation_count"] = self.observation_count
        if self.bbox is not None:
            metadata["bbox"] = list(self.bbox)
        return Segment(
            id=self.id,
            video_id=self.video_id,
            source="ocr",
            start_ms=self.start_ms,
            end_ms=self.end_ms,
            text=self.text,
            role=self.role,
            confidence=self.confidence,
            duplicate_of=self.duplicate_of,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OCRSettings:
    enabled: bool = True
    primary_engine: str = "tesseract"
    fallback_engine: str | None = "paddleocr"
    languages: list[str] = field(default_factory=lambda: ["ja", "en"])

    @classmethod
    def from_config(cls, config: AppConfig | None) -> "OCRSettings":
        value = config.get("ocr", {}) if config else {}
        section = value if isinstance(value, dict) else {}
        languages = section.get("languages") or ["ja", "en"]
        return cls(
            enabled=bool(section.get("enabled", True)),
            primary_engine=str(section.get("primary_engine", "tesseract")),
            fallback_engine=section.get("fallback_engine", "paddleocr"),
            languages=[str(lang) for lang in languages],
        )


def coerce_bbox(value: Any) -> BBox | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple) and len(value) == 4:
        return tuple(int(v) for v in value)  # type: ignore[return-value]
    if isinstance(value, list) and len(value) == 4 and all(not isinstance(v, (list, tuple)) for v in value):
        return tuple(int(v) for v in value)  # type: ignore[return-value]
    if isinstance(value, list) and value and all(isinstance(v, (list, tuple)) for v in value):
        xs = [int(point[0]) for point in value if len(point) >= 2]
        ys = [int(point[1]) for point in value if len(point) >= 2]
        if xs and ys:
            return min(xs), min(ys), max(xs), max(ys)
    return None


def make_observation(
    *,
    video_id: str,
    frame_ms: int,
    text: str,
    region: str,
    role: str | None,
    bbox: Any = None,
    confidence: float | None = None,
    engine: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> OCRObservation | None:
    normalized = normalize_text(text)
    if not normalized:
        return None
    return OCRObservation(
        video_id=video_id,
        frame_ms=int(frame_ms),
        text=normalized,
        region=region,
        role=role,
        bbox=coerce_bbox(bbox),
        confidence=confidence,
        engine=engine,
        metadata=metadata or {},
    )


def recognize_frame(
    image: Any,
    config: AppConfig | None,
    *,
    frame_ms: int,
    video_id: str = "unknown",
    region: str = "full_frame",
    role: str | None = None,
) -> list[OCRObservation]:
    settings = OCRSettings.from_config(config)
    if not settings.enabled:
        return []

    primary_name = settings.primary_engine.lower()
    fallback_name = (settings.fallback_engine or "").lower()
    primary_engine = _optional_engine(primary_name, settings)
    observations, _primary_available, _fallback_engine = _recognize_with_fallback(
        primary_engine,
        None,
        fallback_name if fallback_name != primary_name else "",
        settings,
        image,
        frame_ms=frame_ms,
        video_id=video_id,
        region=region,
        role=role,
    )
    return observations


def _optional_engine(name: str, settings: OCRSettings) -> OCREngine | None:
    try:
        return _engine(name, settings)
    except OCREngineUnavailable:
        return None


def _recognize_with_fallback(
    primary_engine: OCREngine | None,
    fallback_engine: OCREngine | None,
    fallback_name: str,
    settings: OCRSettings,
    image: Any,
    *,
    frame_ms: int,
    video_id: str,
    region: str,
    role: str | None,
) -> tuple[list[OCRObservation], bool, OCREngine | None]:
    if primary_engine is not None:
        try:
            return (
                primary_engine.recognize(
                    image,
                    frame_ms=frame_ms,
                    video_id=video_id,
                    region=region,
                    role=role,
                ),
                True,
                fallback_engine,
            )
        except OCREngineUnavailable:
            primary_engine = None
    if fallback_engine is None:
        fallback_engine = _optional_engine(fallback_name, settings) if fallback_name else None
    if fallback_engine is None:
        return [], primary_engine is not None, None
    try:
        return (
            fallback_engine.recognize(
                image,
                frame_ms=frame_ms,
                video_id=video_id,
                region=region,
                role=role,
            ),
            False,
            fallback_engine,
        )
    except OCREngineUnavailable:
        return [], False, None


def _engine(name: str, settings: OCRSettings) -> OCREngine:
    if name in {"paddle", "paddleocr"}:
        from .paddle_engine import PaddleOCREngine

        return PaddleOCREngine(languages=settings.languages)
    if name in {"tesseract", "pytesseract"}:
        from .tesseract_engine import TesseractEngine

        return TesseractEngine(languages=settings.languages)
    raise OCRError(f"unsupported OCR engine: {name}")


def run_ocr(video_path: str | Path, video_id: str, config: AppConfig | None) -> list[Segment]:
    settings = OCRSettings.from_config(config)
    if not settings.enabled:
        return []

    from yomutube.media.frames import iter_roi_frames
    from .dedupe import dedupe_observations

    value = config.get("ocr", {}) if config else {}
    ocr_section = value if isinstance(value, dict) else {}
    sampling = ocr_section.get("frame_sampling") if isinstance(ocr_section.get("frame_sampling"), dict) else {}
    regions = ocr_section.get("regions") if isinstance(ocr_section.get("regions"), list) else None
    region_by_name = {str(region.get("name")): region for region in regions or [] if isinstance(region, dict)}
    dedupe_config = ocr_section.get("dedupe") if isinstance(ocr_section.get("dedupe"), dict) else {}

    primary_name = settings.primary_engine.lower()
    fallback_name = (settings.fallback_engine or "").lower()
    primary_engine = _optional_engine(primary_name, settings)
    fallback_engine = None
    if fallback_name == primary_name:
        fallback_name = ""

    observations: list[OCRObservation] = []
    for sample in iter_roi_frames(
        video_path,
        regions=regions,
        fps=float(sampling.get("subtitle_roi_fps", 0.1)),
        on_missing="skip",
    ):
        region_config = region_by_name.get(sample.region_name, {})
        frame_observations, primary_available, fallback_engine = _recognize_with_fallback(
            primary_engine,
            fallback_engine,
            fallback_name,
            settings,
            sample.image,
            frame_ms=sample.timestamp_ms,
            video_id=video_id,
            region=sample.region_name,
            role=region_config.get("role") if isinstance(region_config, dict) else None,
        )
        if not primary_available:
            primary_engine = None
        observations.extend(frame_observations)

    min_confidence = float(dedupe_config.get("min_confidence", 0.45))
    filtered = [
        observation
        for observation in observations
        if observation.confidence is None or observation.confidence >= min_confidence
    ]
    events = dedupe_observations(
        filtered,
        similarity_threshold=float(dedupe_config.get("similarity_threshold", 0.86)),
        min_duration_ms=int(dedupe_config.get("min_duration_ms", 400)),
    )
    return [event.to_segment() for event in events]
