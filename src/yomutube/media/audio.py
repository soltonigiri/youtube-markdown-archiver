from __future__ import annotations

from pathlib import Path

from yomutube.config import AppConfig
from yomutube.media import ffmpeg as ffmpeg_helpers


def extract_audio_wav(
    input_path: str | Path,
    output_path: str | Path | None = None,
    config: AppConfig | None = None,
    *,
    ffmpeg_binary: str = "ffmpeg",
    overwrite: bool = True,
) -> Path:
    source = Path(input_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"input media not found: {source}")
    ffmpeg_path = ffmpeg_helpers.require_ffmpeg(ffmpeg_binary)
    target = Path(output_path).expanduser() if output_path else source.parent / "audio.wav"
    target.parent.mkdir(parents=True, exist_ok=True)

    sample_rate = int(config.get("media.audio_sample_rate", 16000) if config else 16000)
    channels = int(config.get("media.audio_channels", 1) if config else 1)
    codec = str(config.get("media.audio_codec", "pcm_s16le") if config else "pcm_s16le")
    threads = config.get("performance.ffmpeg_threads") if config else None

    args = [str(ffmpeg_path), "-y" if overwrite else "-n", "-i", str(source), "-vn"]
    if threads:
        args.extend(["-threads", str(int(threads))])
    args.extend(["-ac", str(channels), "-ar", str(sample_rate), "-c:a", codec, str(target)])
    try:
        ffmpeg_helpers.run_command(args, label="ffmpeg audio extraction")
    except ffmpeg_helpers.FFmpegCommandError:
        raise
    return target


def find_media_file(directory: str | Path, *, prefer_video: bool = True) -> Path | None:
    return ffmpeg_helpers.find_primary_media_file(directory, prefer_video=prefer_video)


extract_audio = extract_audio_wav
