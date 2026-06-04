from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Segment:
    id: str
    video_id: str
    source: str
    start_ms: int
    end_ms: int
    text: str
    language: str | None = None
    speaker: str | None = None
    role: str | None = None
    confidence: float | None = None
    duplicate_of: str | None = None
    primary_source: str | None = None
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    conflict: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: data.get(key) for key in allowed if key in data})

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass(slots=True)
class SubtitleTrack:
    source: str
    language: str
    ext: str
    url: str | None = None
    name: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    coverage_ratio: float | None = None
    path: Path | None = None

    @property
    def segment_source(self) -> str:
        return "manual_subtitle" if self.source == "manual" else "auto_subtitle"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.path is not None:
            data["path"] = str(self.path)
        return data


@dataclass(slots=True)
class Chapter:
    title: str
    start_time: float | None = None
    end_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VideoMetadata:
    video_id: str
    title: str
    channel_id: str | None
    channel_name: str | None
    uploader: str | None
    upload_date: str | None
    duration_sec: float
    webpage_url: str
    thumbnail_path: str | None = None
    description: str | None = None
    chapters: list[Chapter] = field(default_factory=list)
    availability: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["chapters"] = [chapter.to_dict() for chapter in self.chapters]
        data["schema_version"] = "1.0"
        data["archive_id"] = f"yt_{self.video_id}"
        return data

    @classmethod
    def from_info(cls, info: dict[str, Any], thumbnail_path: str | None = None) -> "VideoMetadata":
        chapters = []
        for item in info.get("chapters") or []:
            chapters.append(
                Chapter(
                    title=str(item.get("title") or ""),
                    start_time=item.get("start_time"),
                    end_time=item.get("end_time"),
                )
            )
        return cls(
            video_id=str(info.get("id") or info.get("video_id") or "unknown"),
            title=str(info.get("title") or "Untitled"),
            channel_id=info.get("channel_id"),
            channel_name=info.get("channel") or info.get("channel_name"),
            uploader=info.get("uploader"),
            upload_date=info.get("upload_date"),
            duration_sec=float(info.get("duration") or info.get("duration_sec") or 0),
            webpage_url=str(info.get("webpage_url") or info.get("original_url") or ""),
            thumbnail_path=thumbnail_path,
            description=info.get("description"),
            chapters=chapters,
            availability=info.get("availability"),
        )


@dataclass(slots=True)
class RunPaths:
    run_id: str
    work_dir: Path
    archive_dir: Path


@dataclass(slots=True)
class StepState:
    status: str = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    output: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    retryable: bool | None = None
    input_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: Any) -> "StepState":
        if isinstance(value, StepState):
            return cls.from_value(value.to_dict())
        if isinstance(value, str):
            return cls(status=value)
        if isinstance(value, dict):
            raw_output = value.get("output")
            output = raw_output if isinstance(raw_output, dict) else {"path": str(raw_output)} if raw_output else {}
            return cls(
                status=str(value.get("status") or "pending"),
                started_at=value.get("started_at"),
                finished_at=value.get("finished_at"),
                output=output,
                error_code=value.get("error_code"),
                retryable=value.get("retryable"),
                input_hash=value.get("input_hash"),
            )
        return cls()


@dataclass(slots=True)
class Manifest:
    program: str
    version: str
    video_id: str
    run_id: str
    status: str
    steps: dict[str, StepState | str]
    models: dict[str, str]
    created_at: str
    schema_version: str = "1.0"
    archive_id: str | None = None
    source_url: str | None = None
    updated_at: str | None = None
    finished_at: str | None = None
    archive_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["steps"] = {key: StepState.from_value(value).to_dict() for key, value in self.steps.items()}
        data["program_version"] = self.version
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        version = data.get("version") or data.get("program_version") or "0.1.0"
        video_id = data.get("video_id", "unknown")
        return cls(
            program=data.get("program", "YomuTube"),
            version=version,
            video_id=video_id,
            run_id=data.get("run_id", "unknown"),
            status=data.get("status", "unknown"),
            steps={key: StepState.from_value(value) for key, value in dict(data.get("steps") or {}).items()},
            models=dict(data.get("models") or {}),
            created_at=data.get("created_at", ""),
            schema_version=data.get("schema_version", "1.0"),
            archive_id=data.get("archive_id") or f"yt_{video_id}",
            source_url=data.get("source_url"),
            updated_at=data.get("updated_at"),
            finished_at=data.get("finished_at"),
            archive_path=data.get("archive_path"),
            error=data.get("error"),
        )


@dataclass(slots=True)
class ArchiveResult:
    archive_dir: Path
    index_path: Path
    manifest: Manifest
    resumed: bool = False
