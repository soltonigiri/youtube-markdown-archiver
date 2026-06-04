from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yomutube.models import SubtitleTrack, VideoMetadata


DEFAULT_REDACT_FIELDS = {
    "requested_formats",
    "requested_downloads",
    "formats",
    "thumbnails",
    "automatic_captions",
    "subtitles",
    "http_headers",
    "headers",
    "cookies",
}

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv"}
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".opus", ".weba", ".aac", ".ogg", ".wav"}
SUBTITLE_EXTENSIONS = {".vtt", ".srt", ".json3", ".ttml", ".srv3", ".xml"}
THUMBNAIL_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}


class YtdlpDependencyError(RuntimeError):
    """Raised when yt-dlp is required but not installed."""


class YtdlpDownloadError(RuntimeError):
    """Raised when yt-dlp did not produce a usable media file."""


@dataclass(slots=True)
class MediaDownloadResult:
    media_dir: Path
    video_path: Path | None = None
    audio_path: Path | None = None
    subtitle_path: Path | None = None
    info_path: Path | None = None
    thumbnail_path: Path | None = None
    description_path: Path | None = None
    downloaded_paths: list[Path] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def video_or_audio(self) -> Path:
        path = self.video_path or self.audio_path
        if path is None:
            raise FileNotFoundError(f"downloaded media not found under {self.media_dir}")
        return path

    def to_dict(self) -> dict[str, Any]:
        return {
            "media_dir": self.media_dir,
            "video": self.video_path,
            "audio": self.audio_path,
            "subtitle": self.subtitle_path,
            "info": self.info_path,
            "thumbnail": self.thumbnail_path,
            "description": self.description_path,
            "downloaded": list(self.downloaded_paths),
        }


def import_ytdlp() -> Any:
    try:
        return importlib.import_module("yt_dlp")
    except ModuleNotFoundError as exc:
        if exc.name == "yt_dlp":
            raise YtdlpDependencyError(
                "yt-dlp is required for metadata and media download. Install with: python -m pip install yt-dlp"
            ) from exc
        raise


class YtdlpClient:
    def __init__(self, config: Any = None) -> None:
        self.config = config

    def extract_metadata(
        self,
        url: str,
        *,
        download: bool = False,
        sanitize: bool = False,
        ytdlp_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return extract_metadata(
            url,
            config=self.config,
            download=download,
            sanitize=sanitize,
            ytdlp_options=ytdlp_options,
        )

    def sanitize_info(self, info: dict[str, Any], *, metadata: VideoMetadata | None = None) -> dict[str, Any]:
        if metadata is not None:
            return metadata.to_dict()
        return sanitize_info(info, _redact_fields(self.config))

    def download_temp_media(
        self,
        url: str,
        work_dir: str | Path,
        subtitle_track: SubtitleTrack | None = None,
        *,
        ytdlp_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = download_temp_media(
            url,
            work_dir,
            config=self.config,
            subtitle_track=subtitle_track,
            ytdlp_options=ytdlp_options,
        )
        return result.to_dict()


def _config_get(config: Any, dotted: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        current: Any = config
        for part in dotted.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(dotted, default)
    return default


def _base_options(config: Any = None) -> dict[str, Any]:
    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": bool(_config_get(config, "ytdlp.no_playlist", True)),
    }
    fmt = _config_get(config, "ytdlp.format")
    if fmt:
        options["format"] = str(fmt)
    cookies_file = _config_get(config, "ytdlp.cookies_file")
    if cookies_file:
        options["cookiefile"] = str(Path(str(cookies_file)).expanduser())
    user_agent = _config_get(config, "ytdlp.user_agent")
    if user_agent:
        options["http_headers"] = {"User-Agent": str(user_agent)}
    retries = _config_get(config, "ytdlp.retries")
    if retries is not None:
        options["retries"] = int(retries)
    sleep_interval = _config_get(config, "ytdlp.sleep_interval_sec")
    if sleep_interval is not None:
        options["sleep_interval"] = float(sleep_interval)
    return options


def _redact_fields(config: Any = None) -> set[str]:
    fields = _config_get(config, "compliance.redact_infojson_fields")
    if fields is None:
        return set(DEFAULT_REDACT_FIELDS)
    return {str(field) for field in fields}


def _json_safe(value: Any, redact_fields: set[str]) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if key_str in redact_fields or key_str.startswith("__"):
                continue
            result[key_str] = _json_safe(item, redact_fields)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, redact_fields) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def sanitize_info(info: dict[str, Any], redact_fields: list[str] | set[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    fields = set(DEFAULT_REDACT_FIELDS if redact_fields is None else redact_fields)
    return _json_safe(info, fields)


def extract_metadata(
    url: str,
    config: Any = None,
    *,
    download: bool = False,
    sanitize: bool = False,
    ytdlp_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not url:
        raise ValueError("url is required")
    ytdlp = import_ytdlp()
    options = _base_options(config)
    if ytdlp_options:
        options.update(ytdlp_options)
    with ytdlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=download)
    if not isinstance(info, dict):
        raise YtdlpDownloadError("yt-dlp returned invalid metadata")
    if sanitize:
        return sanitize_info(info, _redact_fields(config))
    return info


def extract_video_metadata(url: str, config: Any = None) -> VideoMetadata:
    return VideoMetadata.from_info(extract_metadata(url, config=config, download=False))


def _subtitle_options(track: SubtitleTrack | None) -> dict[str, Any]:
    if track is None:
        return {"writesubtitles": False, "writeautomaticsub": False}
    return {
        "writesubtitles": track.source == "manual",
        "writeautomaticsub": track.source == "auto",
        "subtitleslangs": [track.language],
        "subtitlesformat": track.ext or "vtt/json3/ttml/srv3/best",
    }


def download_temp_media(
    url: str,
    work_dir: str | Path,
    config: Any = None,
    *,
    subtitle_track: SubtitleTrack | None = None,
    ytdlp_options: dict[str, Any] | None = None,
) -> MediaDownloadResult:
    if not url:
        raise ValueError("url is required")
    ytdlp = import_ytdlp()
    media_dir = Path(work_dir).expanduser() / "media"
    subtitle_dir = media_dir / "subtitles"
    subtitle_dir.mkdir(parents=True, exist_ok=True)

    options = _base_options(config)
    options.update(
        {
            "outtmpl": {
                "default": str(media_dir / "%(id)s.%(ext)s"),
                "subtitle": str(subtitle_dir / "%(id)s.%(ext)s"),
            },
            "writeinfojson": bool(_config_get(config, "ytdlp.write_info_json", True)),
            "writedescription": bool(_config_get(config, "ytdlp.write_description", True)),
            "writethumbnail": bool(_config_get(config, "ytdlp.write_thumbnail", True)),
        }
    )
    options.update(_subtitle_options(subtitle_track))
    if ytdlp_options:
        options.update(ytdlp_options)

    with ytdlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
    if not isinstance(info, dict):
        raise YtdlpDownloadError("yt-dlp returned invalid metadata after download")

    result = discover_downloaded_paths(media_dir, info=info, subtitle_track=subtitle_track)
    result.info = info
    if subtitle_track is not None and result.subtitle_path is not None:
        subtitle_track.path = result.subtitle_path
    if result.video_path is None and result.audio_path is None:
        raise YtdlpDownloadError(f"downloaded media file not found under {media_dir}")
    return result


def _iter_info_paths(info: Any) -> list[Path]:
    paths: list[Path] = []
    if isinstance(info, dict):
        for key, value in info.items():
            if key in {"filepath", "filename", "_filename"} and value:
                paths.append(Path(str(value)))
            else:
                paths.extend(_iter_info_paths(value))
    elif isinstance(info, list):
        for item in info:
            paths.extend(_iter_info_paths(item))
    return paths


def _resolve_existing_paths(media_dir: Path, info: dict[str, Any] | None) -> list[Path]:
    discovered: list[Path] = []
    if info:
        for path in _iter_info_paths(info):
            candidates = [path]
            if not path.is_absolute():
                candidates.append(media_dir / path)
            for candidate in candidates:
                if candidate.exists() and candidate.is_file():
                    discovered.append(candidate)
                    break
    discovered.extend(path for path in media_dir.rglob("*") if path.is_file())
    unique = {path.resolve(): path for path in discovered}
    return sorted(unique.values(), key=lambda item: str(item))


def _first_path(paths: list[Path], extensions: set[str], *, contains: str | None = None) -> Path | None:
    for path in paths:
        if path.suffix.lower() not in extensions:
            continue
        if contains and contains not in path.name:
            continue
        return path
    return None


def discover_downloaded_paths(
    media_dir: str | Path,
    *,
    info: dict[str, Any] | None = None,
    subtitle_track: SubtitleTrack | None = None,
) -> MediaDownloadResult:
    root = Path(media_dir).expanduser()
    paths = _resolve_existing_paths(root, info)
    video_path = _first_path(paths, VIDEO_EXTENSIONS)
    audio_path = _first_path(paths, AUDIO_EXTENSIONS)
    subtitle_path: Path | None = None
    if subtitle_track is not None:
        subtitle_path = _first_path(paths, SUBTITLE_EXTENSIONS, contains=f".{subtitle_track.ext}")
    if subtitle_path is None:
        subtitle_path = _first_path(paths, SUBTITLE_EXTENSIONS)
    info_path = next((path for path in paths if path.name.endswith(".info.json")), None)
    description_path = next((path for path in paths if path.name.endswith(".description")), None)
    thumbnail_path = _first_path(paths, THUMBNAIL_EXTENSIONS)
    return MediaDownloadResult(
        media_dir=root,
        video_path=video_path,
        audio_path=audio_path,
        subtitle_path=subtitle_path,
        info_path=info_path,
        thumbnail_path=thumbnail_path,
        description_path=description_path,
        downloaded_paths=paths,
    )
