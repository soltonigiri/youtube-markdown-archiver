from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from yomutube.config import AppConfig

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv"}
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".opus", ".weba", ".aac", ".ogg", ".wav"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


class FFmpegDependencyError(RuntimeError):
    """Raised when ffmpeg or ffprobe is required but missing."""


class FFmpegCommandError(RuntimeError):
    """Raised when ffmpeg or ffprobe exits with an error."""


def check_binary(binary: str) -> Path:
    path = Path(binary)
    if path.is_absolute() or len(path.parts) > 1:
        if path.exists() and path.is_file():
            return path
        raise FFmpegDependencyError(f"binary not found: {binary}")
    resolved = shutil.which(binary)
    if resolved is None:
        raise FFmpegDependencyError(f"{binary} is required but was not found on PATH")
    return Path(resolved)


def binary_available(binary: str) -> bool:
    try:
        check_binary(binary)
    except FFmpegDependencyError:
        return False
    return True


def require_ffmpeg(binary: str = "ffmpeg") -> Path:
    return check_binary(binary)


def require_ffprobe(binary: str = "ffprobe") -> Path:
    return check_binary(binary)


def run_command(args: list[str], *, label: str = "ffmpeg") -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise FFmpegCommandError(f"{label} failed with code {result.returncode}: {stderr}")
    return result


def ffprobe_json(input_path: str | Path, *, ffprobe_binary: str = "ffprobe") -> dict[str, Any]:
    ffprobe_path = require_ffprobe(ffprobe_binary)
    args = [
        str(ffprobe_path),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(input_path),
    ]
    result = run_command(args, label="ffprobe")
    return json.loads(result.stdout or "{}")


def find_media_files(directory: str | Path, extensions: set[str] | None = None) -> list[Path]:
    root = Path(directory).expanduser()
    if not root.exists():
        return []
    allowed = extensions or MEDIA_EXTENSIONS
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed)


def find_primary_media_file(directory: str | Path, *, prefer_video: bool = True) -> Path | None:
    files = find_media_files(directory)
    if prefer_video:
        for path in files:
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                return path
    for path in files:
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            return path
    return files[0] if files else None


def find_primary_media(directory: str | Path, *, prefer_video: bool = True) -> Path | None:
    return find_primary_media_file(directory, prefer_video=prefer_video)


def extract_audio(
    input_path: str | Path,
    output_path: str | Path | None = None,
    config: AppConfig | None = None,
    *,
    ffmpeg_binary: str = "ffmpeg",
    overwrite: bool = True,
) -> Path:
    from yomutube.media.audio import extract_audio_wav

    return extract_audio_wav(
        input_path,
        output_path,
        config,
        ffmpeg_binary=ffmpeg_binary,
        overwrite=overwrite,
    )
