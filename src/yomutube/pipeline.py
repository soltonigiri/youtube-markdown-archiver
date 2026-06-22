from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .config import AppConfig
from .models import ArchiveResult, Manifest, Segment, StepState, VideoMetadata
from .state import (
    ArchiveLock,
    archive_dir_for,
    cleanup_work_dir,
    existing_archive_for,
    init_run,
    iso_now,
    load_manifest,
    new_manifest,
    should_resume,
    stable_run_suffix,
    write_json,
    write_manifest,
)


class PipelineError(RuntimeError):
    pass


def _normalize_choice(value: Any) -> str:
    return str(value).strip().lower().replace("_", "-")


def _is_enabled(value: Any, *, auto_value: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    normalized = _normalize_choice(value)
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled", "never"}:
        return False
    if normalized == "auto":
        return auto_value
    return bool(value)


def _mark(
    manifest: Manifest,
    step: str,
    status: str,
    *,
    output: str | dict[str, Any] | None = None,
    error_code: str | None = None,
    retryable: bool | None = None,
) -> None:
    current = StepState.from_value(manifest.steps.get(step))
    now = iso_now()
    current.status = status
    if status == "running" and current.started_at is None:
        current.started_at = now
    if status in {"done", "failed", "skipped"}:
        current.finished_at = now
    if output is not None:
        current.output = output if isinstance(output, dict) else {"path": output}
    if error_code is not None:
        current.error_code = error_code
    if retryable is not None:
        current.retryable = retryable
    manifest.steps[step] = current
    manifest.updated_at = now


def subtitle_coverage_ratio(segments: list[Segment], duration_sec: float) -> float:
    if not segments or duration_sec <= 0:
        return 0.0
    covered_ms = sum(max(0, segment.end_ms - segment.start_ms) for segment in segments)
    return min(1.0, covered_ms / (duration_sec * 1000))


def subtitle_quality(segments: list[Segment], metadata: VideoMetadata, config: AppConfig) -> dict[str, Any]:
    manual = [segment for segment in segments if segment.source == "manual_subtitle"]
    auto = [segment for segment in segments if segment.source == "auto_subtitle"]
    all_coverage = subtitle_coverage_ratio(segments, metadata.duration_sec)
    manual_coverage = subtitle_coverage_ratio(manual, metadata.duration_sec)
    auto_coverage = subtitle_coverage_ratio(auto, metadata.duration_sec)
    primary_threshold = float(
        config.get(
            "subtitles.min_coverage_ratio_for_primary",
            config.get("asr.good_subtitle_coverage_ratio", 0.85),
        )
    )
    available_threshold = float(
        config.get(
            "subtitles.min_coverage_ratio_for_available",
            config.get("subtitles.min_coverage_ratio", 0.25),
        )
    )
    best_coverage = max(manual_coverage, auto_coverage, all_coverage)
    return {
        "segments": len(segments),
        "manual_segments": len(manual),
        "auto_segments": len(auto),
        "coverage_ratio": all_coverage,
        "manual_coverage_ratio": manual_coverage,
        "auto_coverage_ratio": auto_coverage,
        "available": best_coverage >= available_threshold,
        "primary": best_coverage >= primary_threshold,
        "manual_primary": manual_coverage >= primary_threshold,
        "primary_threshold": primary_threshold,
        "available_threshold": available_threshold,
    }


def should_run_asr(config: AppConfig, quality: dict[str, Any]) -> bool:
    if not _is_enabled(config.get("asr.enabled", True)):
        return False
    when = _normalize_choice(config.get("asr.when", "no-good-subtitle"))
    if when == "always":
        return True
    if when == "never":
        return False
    has_subtitle = bool(quality.get("segments"))
    if when == "no-subtitle":
        return not has_subtitle
    if when == "low-coverage":
        return not bool(quality.get("available"))
    if when == "no-good-subtitle":
        return not bool(quality.get("primary"))
    return True


def should_run_ocr(config: AppConfig, metadata: VideoMetadata, quality: dict[str, Any]) -> bool:
    setting = config.get("ocr.enabled", "auto")
    normalized = _normalize_choice(setting)
    if normalized in {"on", "true", "1", "yes"} or setting is True:
        return True
    if normalized in {"off", "false", "0", "no", "never"} or setting is False:
        return False
    max_duration = config.get("ocr.auto_policy.skip_when.duration_over_sec", 7200)
    if max_duration is not None and metadata.duration_sec > float(max_duration):
        return False
    min_height = config.get("ocr.auto_policy.skip_when.low_resolution_below_height", 480)
    height = metadata.raw.get("height") if hasattr(metadata, "raw") else None
    if height is not None and min_height is not None and int(height) < int(min_height):
        return False
    return not bool(quality.get("primary"))


def processing_plan(metadata: VideoMetadata, track: Any, config: AppConfig) -> dict[str, Any]:
    pseudo_quality = {
        "segments": 1 if track else 0,
        "primary": bool(track and getattr(track, "source", "") == "manual"),
        "available": bool(track),
    }
    primary_text = "manual_subtitle" if track and track.source == "manual" else "auto_subtitle" if track else "asr"
    return {
        "archive_path": str(archive_dir_for(metadata, config)),
        "primary_text": primary_text,
        "asr": "run" if should_run_asr(config, pseudo_quality) else "skipped",
        "ocr": "run" if should_run_ocr(config, metadata, pseudo_quality) else "skipped",
    }


def _import_worker_modules() -> dict[str, Any]:
    # Imports live here so the package can be imported before optional engines are installed.
    from .asr.base import transcribe_audio
    from .diarization.base import assign_speakers_to_segments, diarize_audio
    from .download.subtitle_selector import select_subtitle_track
    from .download.ytdlp_client import YtdlpClient
    from .fusion.merge import fuse_segments
    from .media.ffmpeg import extract_audio, find_primary_media
    from .ocr.base import run_ocr
    from .subtitle.parser import parse_subtitle_file
    from .writers.markdown import write_archive

    return {
        "assign_speakers_to_segments": assign_speakers_to_segments,
        "diarize_audio": diarize_audio,
        "transcribe_audio": transcribe_audio,
        "select_subtitle_track": select_subtitle_track,
        "YtdlpClient": YtdlpClient,
        "fuse_segments": fuse_segments,
        "extract_audio": extract_audio,
        "find_primary_media": find_primary_media,
        "run_ocr": run_ocr,
        "parse_subtitle_file": parse_subtitle_file,
        "write_archive": write_archive,
    }


class YomuTubePipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def inspect(self, url: str) -> dict[str, Any]:
        modules = _import_worker_modules()
        client = modules["YtdlpClient"](self.config)
        info = client.extract_metadata(url)
        effective_config = self.config.with_channel_profile(info)
        track = modules["select_subtitle_track"](info, effective_config)
        metadata = VideoMetadata.from_info(info)
        plan = processing_plan(metadata, track, effective_config)
        warnings: list[str] = []
        if metadata.duration_sec >= 7200:
            warnings.append("Long video. OCR may take time or be skipped by auto policy.")
        return {
            "metadata": metadata.to_dict(),
            "subtitle_track": track.to_dict() if track else None,
            "has_manual_subtitles": bool(info.get("subtitles")),
            "has_auto_subtitles": bool(info.get("automatic_captions")),
            "planned": plan,
            "warnings": warnings,
            "effective_config": str(effective_config.source_path),
        }

    def transcribe(self, url: str) -> ArchiveResult:
        modules = _import_worker_modules()
        timings: dict[str, float] = {}
        diagnostics: dict[str, Any] = {"errors": [], "warnings": []}

        client = modules["YtdlpClient"](self.config)

        t0 = time.monotonic()
        try:
            info = client.extract_metadata(url)
        except Exception as exc:
            if _is_enabled(self.config.get("archive.write_failed_archive", True)):
                write_failed_metadata_archive(self.config, url, exc)
            raise
        timings["metadata_sec"] = time.monotonic() - t0
        config = self.config.with_channel_profile(info)
        metadata = VideoMetadata.from_info(info)
        existing_archive = existing_archive_for(metadata, config)
        resume = bool(config.get("runtime.resume", True))
        rerun_policy = _normalize_choice(config.get("runtime.rerun_policy", "reuse_done"))
        if existing_archive and rerun_policy == "reuse-done" and should_resume(existing_archive, resume=resume):
            manifest = load_manifest(existing_archive)
            if manifest is None:
                raise PipelineError(f"manifest missing in resumed archive: {existing_archive}")
            return ArchiveResult(existing_archive, existing_archive / "index.md", manifest, resumed=True)

        archive_base = archive_dir_for(metadata, config)
        run_paths = init_run(metadata, config)
        if rerun_policy == "new-run":
            run_paths.archive_dir = archive_base / "runs" / run_paths.run_id
            run_paths.archive_dir.mkdir(parents=True, exist_ok=True)
        manifest = new_manifest(metadata, run_paths, config.version)
        archive_lock = ArchiveLock(archive_base, if_running=_normalize_choice(config.get("runtime.if_running", "fail")))
        if not archive_lock.acquire():
            if existing_archive and (existing_archive / "index.md").exists():
                manifest = load_manifest(existing_archive)
                if manifest is not None:
                    return ArchiveResult(existing_archive, existing_archive / "index.md", manifest, resumed=True)
            raise PipelineError(f"archive is already running: {archive_base}")

        subtitle_segments: list[Segment] = []
        asr_segments: list[Segment] = []
        ocr_segments: list[Segment] = []
        diarization_turns: list[Any] = []
        fused_segments: list[Segment] = []
        media_paths: dict[str, Any] = {}

        try:
            write_manifest(run_paths.archive_dir, manifest)
            write_json(run_paths.archive_dir / "metadata.json", client.sanitize_info(info, metadata=metadata))
            _mark(manifest, "metadata", "done", output="metadata.json")
            write_manifest(run_paths.archive_dir, manifest)

            track = modules["select_subtitle_track"](info, config)
            t0 = time.monotonic()
            try:
                media_paths = client.download_temp_media(url, run_paths.work_dir, track)
                _mark(manifest, "download", "done", output={key: str(value) for key, value in media_paths.items() if value})
            except Exception as exc:
                diagnostics["warnings"].append(f"download failed; subtitle-only fallback may be used: {exc}")
                _mark(manifest, "download", "failed", error_code="YT_DOWNLOAD_FAILED", retryable=True)
                media_paths = {}
            timings["download_sec"] = time.monotonic() - t0
            write_manifest(run_paths.archive_dir, manifest)

            subtitle_file = media_paths.get("subtitle") if isinstance(media_paths, dict) else None
            if subtitle_file and Path(subtitle_file).exists() and track:
                subtitle_segments = modules["parse_subtitle_file"](subtitle_file, metadata.video_id, track)
            subtitle_info = subtitle_quality(subtitle_segments, metadata, config)
            _mark(
                manifest,
                "subtitle",
                "done" if subtitle_segments else "skipped",
                output={"segments": len(subtitle_segments), "quality": subtitle_info},
            )
            write_manifest(run_paths.archive_dir, manifest)

            primary_media = modules["find_primary_media"](run_paths.work_dir)
            audio_path = None
            asr_enabled = should_run_asr(config, subtitle_info)
            ocr_enabled = should_run_ocr(config, metadata, subtitle_info)
            if asr_enabled:
                if primary_media is None:
                    raise PipelineError("ASR is enabled but no downloaded media was found")
                audio_path = modules["extract_audio"](primary_media, run_paths.work_dir / "media" / "audio.wav", config)
                t0 = time.monotonic()
                asr_segments = modules["transcribe_audio"](audio_path, config, video_id=metadata.video_id)
                manifest.models["asr"] = f"{config.get('asr.engine', 'faster-whisper')}:{config.get('asr.model', 'large-v3')}"
                timings["asr_sec"] = time.monotonic() - t0
                _mark(manifest, "asr", "done", output={"segments": len(asr_segments), "audio": str(audio_path)})
            else:
                _mark(manifest, "asr", "skipped", output={"reason": str(config.get("asr.when", "no_good_subtitle"))})
            write_manifest(run_paths.archive_dir, manifest)

            if ocr_enabled and primary_media is not None:
                t0 = time.monotonic()
                ocr_segments = modules["run_ocr"](primary_media, metadata.video_id, config)
                timings["ocr_sec"] = time.monotonic() - t0
                _mark(manifest, "ocr", "done" if ocr_segments else "skipped", output={"segments": len(ocr_segments)})
            else:
                _mark(manifest, "ocr", "skipped", output={"reason": str(config.get("ocr.enabled", "auto"))})
            write_manifest(run_paths.archive_dir, manifest)

            diarization_enabled = _is_enabled(config.get("diarization.enabled", False), auto_value=False)
            if diarization_enabled:
                if audio_path is None:
                    raise PipelineError("diarization is enabled but audio was not extracted")
                diarization_turns = modules["diarize_audio"](audio_path, config, segments=asr_segments)
                asr_segments = modules["assign_speakers_to_segments"](asr_segments, diarization_turns)
                manifest.models["diarization"] = f"{config.get('diarization.engine', 'pyannote')}:{config.get('diarization.model', '')}"
                _mark(manifest, "diarization", "done", output={"turns": len(diarization_turns)})
            else:
                _mark(manifest, "diarization", "skipped")
            write_manifest(run_paths.archive_dir, manifest)

            t0 = time.monotonic()
            manual_subtitles = [segment for segment in subtitle_segments if segment.source == "manual_subtitle"]
            auto_subtitles = [segment for segment in subtitle_segments if segment.source == "auto_subtitle"]
            fused_segments = modules["fuse_segments"](
                manual_subtitles=manual_subtitles,
                auto_subtitles=auto_subtitles,
                asr_segments=asr_segments,
                ocr_segments=ocr_segments,
                config=config,
            )
            timings["fusion_sec"] = time.monotonic() - t0
            _mark(manifest, "fusion", "done", output={"segments": len(fused_segments)})

            diagnostics.update(
                {
                    "schema_version": "1.0",
                    "duration_sec": metadata.duration_sec,
                    "subtitle": {
                        "segments": len(subtitle_segments),
                        "selected": track.to_dict() if track else None,
                        "quality": subtitle_info,
                    },
                    "asr": {
                        "enabled": asr_enabled,
                        "when": config.get("asr.when", "no_good_subtitle"),
                        "segments": len(asr_segments),
                    },
                    "ocr": {
                        "enabled": ocr_enabled,
                        "mode": config.get("ocr.enabled", "auto"),
                        "segments": len(ocr_segments),
                    },
                    "fusion": {"fused_segments": len(fused_segments)},
                    "timings": timings,
                }
            )

            manifest.status = "done"
            manifest.finished_at = iso_now()
            _mark(manifest, "markdown", "done", output="index.md")
            result = modules["write_archive"](
                archive_dir=run_paths.archive_dir,
                metadata=metadata,
                segments=fused_segments,
                raw_asr=asr_segments,
                raw_subtitles=subtitle_segments,
                raw_ocr=ocr_segments,
                manifest=manifest,
                diagnostics=diagnostics,
                config=config,
            )
            keep_temp = bool(config.get("storage.keep_temp_media", False))
            cleanup_work_dir(run_paths.work_dir, keep=keep_temp)
            archive_lock.release()
            return result
        except Exception as exc:
            manifest.status = "failed"
            manifest.error = str(exc)
            manifest.finished_at = iso_now()
            manifest.updated_at = manifest.finished_at
            diagnostics["errors"].append(str(exc))
            diagnostics.setdefault("schema_version", "1.0")
            write_json(run_paths.archive_dir / "diagnostics.json", diagnostics)
            write_manifest(run_paths.archive_dir, manifest)
            archive_lock.release()
            raise


def write_failed_metadata_archive(config: AppConfig, url: str, exc: Exception) -> Path:
    failed_dir = config.archive_dir / "failed" / stable_run_suffix(url)
    now = iso_now()
    message = str(exc)
    request = {
        "schema_version": "1.0",
        "source_url": url,
        "created_at": now,
    }
    diagnostics = {
        "schema_version": "1.0",
        "errors": [message],
        "warnings": [],
        "source_url": url,
        "error_code": "YT_METADATA_FAILED",
        "retryable": True,
        "yt_dlp_error_excerpt": message[:800],
    }
    manifest = {
        "schema_version": "1.0",
        "program": "YomuTube",
        "version": config.version,
        "program_version": config.version,
        "video_id": "unknown",
        "archive_id": f"failed_{stable_run_suffix(url)}",
        "run_id": f"failed_{stable_run_suffix(url)}",
        "status": "failed",
        "source_url": url,
        "created_at": now,
        "updated_at": now,
        "finished_at": now,
        "archive_path": str(failed_dir),
        "models": {},
        "error": message,
        "steps": {
            "metadata": {
                "status": "failed",
                "started_at": now,
                "finished_at": now,
                "output": {},
                "error_code": "YT_METADATA_FAILED",
                "retryable": True,
                "input_hash": None,
            }
        },
    }
    write_json(failed_dir / "request.json", request)
    write_json(failed_dir / "diagnostics.json", diagnostics)
    write_json(failed_dir / "manifest.json", manifest)
    return failed_dir


def reprocess_archive(archive_dir: str | Path, *, from_step: str, config: AppConfig) -> ArchiveResult:
    from .fusion.merge import fuse_segments
    from .models import Segment, VideoMetadata
    from .state import read_json
    from .writers.jsonl import read_jsonl
    from .writers.markdown import write_archive

    archive = Path(archive_dir).expanduser()
    if from_step not in {"fusion", "markdown"}:
        raise PipelineError("reprocess currently supports --from-step fusion or markdown")
    metadata = VideoMetadata.from_info(read_json(archive / "metadata.json"))
    manifest = load_manifest(archive)
    if manifest is None:
        raise PipelineError(f"manifest not found: {archive}")
    raw_asr = [Segment.from_dict(row) for row in read_jsonl(archive / "raw_asr.jsonl")]
    raw_subtitles = [Segment.from_dict(row) for row in read_jsonl(archive / "raw_subtitles.jsonl")]
    raw_ocr = [Segment.from_dict(row) for row in read_jsonl(archive / "raw_ocr.jsonl")]
    if from_step == "fusion":
        segments = fuse_segments(
            manual_subtitles=[segment for segment in raw_subtitles if segment.source == "manual_subtitle"],
            auto_subtitles=[segment for segment in raw_subtitles if segment.source == "auto_subtitle"],
            asr_segments=raw_asr,
            ocr_segments=raw_ocr,
            config=config,
        )
    else:
        segments = [Segment.from_dict(row) for row in read_jsonl(archive / "segments.jsonl")]
    diagnostics = read_json(archive / "diagnostics.json") if (archive / "diagnostics.json").exists() else {}
    manifest.status = "done"
    manifest.finished_at = iso_now()
    return write_archive(
        archive_dir=archive,
        metadata=metadata,
        segments=segments,
        raw_asr=raw_asr,
        raw_subtitles=raw_subtitles,
        raw_ocr=raw_ocr,
        manifest=manifest,
        diagnostics=diagnostics,
        config=config,
    )
