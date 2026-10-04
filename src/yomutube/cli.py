from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .archive_browser import find_archives, get_archive, last_archive, search_archives, segment_matches
from .archive_assets import (
    append_archive_correction,
    read_json_for_display,
    read_jsonl_for_display,
    read_speaker_aliases,
    write_speaker_aliases,
)
from .config import AppConfig, ConfigError, transcribe_overrides
from .doctor import doctor_as_json, doctor_exit_code, run_doctor
from .exporters import EXPORT_EXTENSIONS, export_archive, quote_at
from .pipeline import YomuTubePipeline, reprocess_archive
from .state import clean_work_root
from .utils.timecode import parse_timecode


app = typer.Typer(help="YouTube URL を Markdown archive に変換します。", no_args_is_help=True)
console = Console()


def _load_config(
    config_path: Optional[Path],
    profile: str = "default",
    mode: str | None = None,
    overrides: dict | None = None,
) -> AppConfig:
    config = AppConfig.load(config_path, profile=profile)
    if mode:
        config = config.with_mode(mode)
    if overrides:
        config = config.with_overrides(overrides)
    return config


def _archive_root(config_path: Optional[Path], archive_dir: Optional[Path]) -> Path:
    if archive_dir is not None:
        return archive_dir.expanduser()
    return _load_config(config_path).archive_dir


def _print_archive_line(archive) -> None:
    console.print(f"{archive.video_id}\t{archive.status}\t{archive.title}\t{archive.path}")


@app.command()
def doctor(
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
    json_output: bool = typer.Option(False, "--json", help="JSON で出力"),
) -> None:
    """依存関係、CUDA、OCR 言語、出力 path を確認します。"""

    app_config = _load_config(config)
    checks = run_doctor(app_config)
    if json_output:
        print(doctor_as_json(checks))
    else:
        table = Table("Name", "Status", "Required", "Detail")
        for check in checks:
            table.add_row(check.name, check.status, "yes" if check.required else "no", check.detail or "")
        console.print(table)
    raise typer.Exit(doctor_exit_code(checks))


@app.command(name="inspect")
def inspect_url(
    url: str = typer.Argument(..., help="YouTube URL"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
    profile: str = typer.Option("default", "--profile", help="config profile"),
    json_output: bool = typer.Option(False, "--json", help="JSON で出力"),
) -> None:
    """metadata と字幕候補を取得して処理方針を表示します。"""

    app_config = _load_config(config, profile=profile)
    result = YomuTubePipeline(app_config).inspect(url)
    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    metadata = result["metadata"]
    console.print(f"Title: {metadata.get('title')}")
    console.print(f"Video ID: {metadata.get('video_id')}")
    console.print(f"Channel: {metadata.get('channel_name') or metadata.get('channel_id')}")
    console.print(f"Duration: {metadata.get('duration_sec')} sec")
    console.print(f"Subtitle: {result.get('subtitle_track') or 'none'}")


@app.command()
def transcribe(
    url: str = typer.Argument(..., help="YouTube URL"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
    profile: str = typer.Option("default", "--profile", help="config profile"),
    mode: str = typer.Option("standard", "--mode", help="quick, standard, full"),
    lang: str = typer.Option("auto", "--lang", help="ASR language"),
    asr_model: Optional[str] = typer.Option(None, "--asr-model", help="faster-whisper model"),
    asr_when: Optional[str] = typer.Option(None, "--asr-when", help="always, no-subtitle, low-coverage, no-good-subtitle, never"),
    asr_device: Optional[str] = typer.Option(None, "--asr-device", help="cuda or cpu"),
    strict_device: Optional[bool] = typer.Option(None, "--strict-device/--no-strict-device", help="指定 device が使えない場合に停止するか"),
    ocr: Optional[str] = typer.Option(None, "--ocr", help="auto, on, off"),
    diarization: Optional[str] = typer.Option(None, "--diarization", help="auto, on, off"),
    max_height: Optional[int] = typer.Option(None, "--max-height", help="OCR 用動画の最大高さ"),
    subtitle_policy: Optional[str] = typer.Option(None, "--subtitle-policy", help="字幕選択方針"),
    keep_temp_media: Optional[bool] = typer.Option(None, "--keep-temp-media/--no-keep-temp-media", help="一時 media を保持するか"),
    resume: Optional[bool] = typer.Option(None, "--resume/--no-resume", help="done manifest があれば再利用するか"),
    rerun: Optional[str] = typer.Option(None, "--rerun", help="reuse-done, overwrite, new-run"),
    if_running: Optional[str] = typer.Option(None, "--if-running", help="fail, skip, wait"),
) -> None:
    """1本の動画を Markdown archive に変換します。"""

    overrides = transcribe_overrides(
        lang=lang,
        asr_model=asr_model,
        asr_when=asr_when,
        asr_device=asr_device,
        strict_device=strict_device,
        ocr=ocr,
        diarization=diarization,
        max_height=max_height,
        subtitle_policy=subtitle_policy,
        keep_temp_media=keep_temp_media,
        resume=resume,
        rerun=rerun,
        if_running=if_running,
    )
    try:
        app_config = _load_config(config, profile=profile, mode=mode, overrides=overrides)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    result = YomuTubePipeline(app_config).transcribe(url)
    prefix = "Resumed" if result.resumed else "Done"
    console.print(f"{prefix}: {result.index_path}")


@app.command()
def batch(
    urls_file: Path = typer.Argument(..., help="1行1URLのテキストファイル"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
    profile: str = typer.Option("default", "--profile", help="config profile"),
) -> None:
    """URL リストを順に処理します。"""

    app_config = _load_config(config, profile=profile)
    pipeline = YomuTubePipeline(app_config)
    urls = [line.strip() for line in urls_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, url in enumerate(urls, 1):
        console.print(f"[{index}/{len(urls)}] {url}")
        result = pipeline.transcribe(url)
        console.print(str(result.index_path))


@app.command()
def reprocess(
    archive_id: Path = typer.Argument(..., help="既存 archive directory"),
    from_step: str = typer.Option("fusion", "--from", "--from-step", help="fusion or markdown"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
    profile: str = typer.Option("default", "--profile", help="config profile"),
) -> None:
    """既存 sidecar から後段だけ再処理します。"""

    app_config = _load_config(config, profile=profile)
    result = reprocess_archive(archive_id, from_step=from_step, config=app_config)
    console.print(f"Done: {result.index_path}")


@app.command()
def clean(
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
    work_dir: Optional[Path] = typer.Option(None, "--work-dir", help="削除する work directory"),
) -> None:
    """一時ファイルを削除します。"""

    app_config = _load_config(config)
    removed = clean_work_root(work_dir or app_config.work_dir)
    console.print(f"Removed: {len(removed)}")


@app.command(name="index")
def build_index(
    archive_dir: Optional[Path] = typer.Option(None, "--archive-dir", help="archive root"),
    output: Optional[Path] = typer.Option(None, "--output", help="SQLite DB path"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
) -> None:
    """既存 archive を検索用 SQLite index にします。"""

    from .writers.sqlite_index import build_sqlite_index

    app_config = _load_config(config)
    source = archive_dir or app_config.archive_dir
    target = output or (source / "archive.db")
    result = build_sqlite_index(source, target)
    console.print(f"Indexed: {result.segment_count} segments -> {target}")


@app.command(name="status")
@app.command(name="list")
def list_archives(
    archive_dir: Optional[Path] = typer.Option(None, "--archive-dir", help="archive root"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
    limit: int = typer.Option(50, "--limit", help="表示件数"),
) -> None:
    """作成済み archive を一覧表示します。"""

    for archive in find_archives(_archive_root(config, archive_dir))[:limit]:
        _print_archive_line(archive)


@app.command()
def show(
    video_id: str = typer.Argument(..., help="video_id または archive directory 名"),
    archive_dir: Optional[Path] = typer.Option(None, "--archive-dir", help="archive root"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
    json_output: bool = typer.Option(False, "--json", help="JSON で出力"),
) -> None:
    """archive の metadata と保存先を表示します。"""

    archive = get_archive(_archive_root(config, archive_dir), video_id)
    if archive is None:
        raise typer.BadParameter(f"archive not found: {video_id}")
    if json_output:
        print(json.dumps({"path": str(archive.path), "manifest": archive.manifest, "metadata": archive.metadata}, ensure_ascii=False, indent=2))
        return
    console.print(f"Video ID: {archive.video_id}")
    console.print(f"Title: {archive.title}")
    console.print(f"Channel: {archive.channel}")
    console.print(f"Status: {archive.status}")
    console.print(f"Path: {archive.path}")
    console.print(f"Index: {archive.index_path}")


@app.command()
def search(
    keyword: str = typer.Argument("", help="検索語"),
    archive_dir: Optional[Path] = typer.Option(None, "--archive-dir", help="archive root"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
    limit: int = typer.Option(20, "--limit", help="表示件数"),
    speaker: Optional[str] = typer.Option(None, "--speaker", help="speaker filter"),
    source: Optional[str] = typer.Option(None, "--source", help="source filter"),
    channel: Optional[str] = typer.Option(None, "--channel", help="channel filter"),
    ocr: bool = typer.Option(False, "--ocr", help="OCR segment だけ検索"),
    conflicts_only: bool = typer.Option(False, "--conflicts", help="conflict segment だけ検索"),
    semantic: bool = typer.Option(False, "--semantic", help="local semantic search"),
) -> None:
    """archive の metadata と segments.jsonl を簡易検索します。"""

    if semantic:
        console.print("Semantic search is not configured for this local archive.")
        return
    filters = {
        key: value
        for key, value in {
            "speaker": speaker,
            "source": source,
            "channel": channel,
            "ocr": True if ocr else None,
            "conflicts": True if conflicts_only else None,
        }.items()
        if value is not None
    }
    for archive in search_archives(_archive_root(config, archive_dir), keyword, limit=limit, filters=filters):
        _print_archive_line(archive)
        for row in segment_matches(archive, keyword, limit=3, filters=filters):
            console.print(f"  {row.get('start_ms', 0)}ms\t{row.get('text', '')}")


@app.command()
def last(
    archive_dir: Optional[Path] = typer.Option(None, "--archive-dir", help="archive root"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
) -> None:
    """直近の archive を表示します。"""

    archive = last_archive(_archive_root(config, archive_dir))
    if archive is None:
        raise typer.BadParameter("archive not found")
    _print_archive_line(archive)


@app.command(name="open")
def open_archive(
    video_id: str = typer.Argument(..., help="video_id または archive directory 名"),
    editor_command: Optional[str] = typer.Argument(None, help="使用する editor command"),
    archive_dir: Optional[Path] = typer.Option(None, "--archive-dir", help="archive root"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
    editor: bool = typer.Option(False, "--editor", help="$EDITOR または editor 引数で index.md を開く"),
) -> None:
    """index.md の path を表示、または editor で開きます。"""

    archive = get_archive(_archive_root(config, archive_dir), video_id)
    if archive is None:
        raise typer.BadParameter(f"archive not found: {video_id}")
    index_path = archive.index_path
    if not editor and editor_command is None:
        console.print(str(index_path))
        return
    command = editor_command or os.environ.get("EDITOR")
    if not command:
        raise typer.BadParameter("$EDITOR is not set")
    subprocess.run([command, str(index_path)], check=True)


@app.command()
def quality(
    video_id: str = typer.Argument(..., help="video_id または archive directory 名"),
    archive_dir: Optional[Path] = typer.Option(None, "--archive-dir", help="archive root"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
    json_output: bool = typer.Option(False, "--json", help="JSON で出力"),
) -> None:
    """archive の品質スコアを表示します。"""

    archive = _require_archive(config, archive_dir, video_id)
    payload = read_json_for_display(archive.path / "quality.json", default={})
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    console.print(f"Overall: {payload.get('overall_score', 'unknown')}")
    console.print(f"Subtitle coverage: {payload.get('subtitle_coverage', 'unknown')}")
    console.print(f"Conflict rate: {payload.get('conflict_rate', 'unknown')}")
    console.print(f"Speaker coverage: {payload.get('speaker_coverage', 'unknown')}")


@app.command()
def conflicts(
    video_id: str = typer.Argument(..., help="video_id または archive directory 名"),
    archive_dir: Optional[Path] = typer.Option(None, "--archive-dir", help="archive root"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
    json_output: bool = typer.Option(False, "--json", help="JSON で出力"),
    limit: int = typer.Option(50, "--limit", help="表示件数"),
) -> None:
    """subtitle / ASR の差分や怪しい箇所を表示します。"""

    archive = _require_archive(config, archive_dir, video_id)
    rows = read_jsonl_for_display(archive.path / "conflicts.jsonl")[:limit]
    if json_output:
        print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
        return
    for row in rows:
        console.print(f"{row.get('start_ms', 0)}ms\t{row.get('source', '')}\t{row.get('text', '')}")


@app.command()
def speakers(
    video_id: str = typer.Argument(..., help="video_id または archive directory 名"),
    archive_dir: Optional[Path] = typer.Option(None, "--archive-dir", help="archive root"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
    json_output: bool = typer.Option(False, "--json", help="JSON で出力"),
) -> None:
    """話者一覧と発話量を表示します。"""

    archive = _require_archive(config, archive_dir, video_id)
    aliases = read_speaker_aliases(archive.path)
    stats: dict[str, dict[str, int | str | None]] = {}
    for row in read_jsonl_for_display(archive.path / "segments.jsonl"):
        speaker = row.get("speaker")
        if not speaker:
            continue
        key = str(speaker)
        item = stats.setdefault(key, {"speaker": key, "alias": aliases.get(key), "segments": 0, "duration_ms": 0})
        item["segments"] = int(item["segments"] or 0) + 1
        item["duration_ms"] = int(item["duration_ms"] or 0) + max(0, int(row.get("end_ms") or 0) - int(row.get("start_ms") or 0))
    rows = list(stats.values())
    if json_output:
        print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
        return
    for row in rows:
        console.print(f"{row['speaker']}\t{row.get('alias') or ''}\t{row['segments']} segments\t{row['duration_ms']}ms")


@app.command(name="rename-speaker")
def rename_speaker(
    video_id: str = typer.Argument(..., help="video_id または archive directory 名"),
    speaker_id: str = typer.Argument(..., help="SPEAKER_00 など"),
    alias: str = typer.Argument(..., help="表示名"),
    archive_dir: Optional[Path] = typer.Option(None, "--archive-dir", help="archive root"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
) -> None:
    """speaker alias を更新します。"""

    archive = _require_archive(config, archive_dir, video_id)
    aliases = read_speaker_aliases(archive.path)
    aliases[speaker_id] = alias
    write_speaker_aliases(archive.path, aliases)
    console.print(f"Updated: {archive.path / 'speaker_aliases.json'}")


@app.command()
def correct(
    video_id: str = typer.Argument(..., help="video_id または archive directory 名"),
    segment_id: str = typer.Option(..., "--segment", help="修正対象 Segment ID"),
    text: str = typer.Option(..., "--text", help="修正後の本文"),
    reason: str = typer.Option("manual_correction", "--reason", help="修正理由"),
    archive_dir: Optional[Path] = typer.Option(None, "--archive-dir", help="archive root"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
) -> None:
    """corrections.jsonl に修正差分を追記します。"""

    archive = _require_archive(config, archive_dir, video_id)
    before = ""
    for row in read_jsonl_for_display(archive.path / "segments.jsonl"):
        if row.get("id") == segment_id:
            before = str(row.get("text") or "")
            break
    if not before:
        raise typer.BadParameter(f"segment not found: {segment_id}")
    append_archive_correction(archive.path, target_segment_id=segment_id, before=before, after=text, reason=reason)
    console.print(f"Updated: {archive.path / 'corrections.jsonl'}")


@app.command(name="export")
def export_command(
    video_id: str = typer.Argument(..., help="video_id または archive directory 名"),
    format: str = typer.Option(..., "--format", help="srt, vtt, txt, csv, obsidian, quotes, topics, speakers"),
    archive_dir: Optional[Path] = typer.Option(None, "--archive-dir", help="archive root"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
    output: Optional[Path] = typer.Option(None, "--output", help="出力 path"),
) -> None:
    """archive から再利用用ファイルを生成します。"""

    if format not in EXPORT_EXTENSIONS:
        choices = ", ".join(sorted(EXPORT_EXTENSIONS))
        raise typer.BadParameter(f"format must be one of: {choices}")
    archive = _require_archive(config, archive_dir, video_id)
    result = export_archive(archive.path, format, output=output)
    console.print(str(result.output_path))


@app.command()
def quote(
    video_id: str = typer.Argument(..., help="video_id または archive directory 名"),
    at: str = typer.Option(..., "--at", help="HH:MM:SS または HH:MM:SS.mmm"),
    archive_dir: Optional[Path] = typer.Option(None, "--archive-dir", help="archive root"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
) -> None:
    """指定時刻の引用 Markdown を表示します。"""

    archive = _require_archive(config, archive_dir, video_id)
    console.print(quote_at(archive.path, parse_timecode(at)))


@app.command(name="ingest-channel")
def ingest_channel(
    url: str = typer.Argument(..., help="YouTube channel URL"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
    profile: str = typer.Option("default", "--profile", help="config profile"),
    since: Optional[str] = typer.Option(None, "--since", help="YYYY-MM-DD"),
    max_results: int = typer.Option(50, "--max-results", help="最大取得件数"),
    duration_max_sec: Optional[int] = typer.Option(None, "--duration-max-sec", help="最大動画長"),
    include: Optional[str] = typer.Option(None, "--include", help="title include"),
    exclude: Optional[str] = typer.Option(None, "--exclude", help="title exclude"),
    queue: Optional[Path] = typer.Option(None, "--queue", help="queue file path"),
    dry_run: bool = typer.Option(False, "--dry-run", help="queue 生成だけ行う"),
) -> None:
    """channel の動画 URL を queue 化します。"""

    app_config = _load_config(config, profile=profile)
    urls = _collection_urls(url, app_config, max_results=max_results, since=since, duration_max_sec=duration_max_sec, include=include, exclude=exclude)
    queue_path = queue or (app_config.work_dir / "queues" / "channel.txt")
    _write_queue(queue_path, urls)
    console.print(f"Queued: {len(urls)} -> {queue_path}")
    if not dry_run:
        pipeline = YomuTubePipeline(app_config)
        for item in urls:
            pipeline.transcribe(item)


@app.command(name="ingest-playlist")
def ingest_playlist(
    url: str = typer.Argument(..., help="YouTube playlist URL"),
    config: Optional[Path] = typer.Option(None, "--config", help="設定ファイル path"),
    profile: str = typer.Option("default", "--profile", help="config profile"),
    since: Optional[str] = typer.Option(None, "--since", help="YYYY-MM-DD"),
    max_results: int = typer.Option(50, "--max-results", help="最大取得件数"),
    queue: Optional[Path] = typer.Option(None, "--queue", help="queue file path"),
    dry_run: bool = typer.Option(False, "--dry-run", help="queue 生成だけ行う"),
) -> None:
    """playlist の動画 URL を queue 化します。"""

    app_config = _load_config(config, profile=profile)
    urls = _collection_urls(url, app_config, max_results=max_results, since=since)
    queue_path = queue or (app_config.work_dir / "queues" / "playlist.txt")
    _write_queue(queue_path, urls)
    console.print(f"Queued: {len(urls)} -> {queue_path}")
    if not dry_run:
        pipeline = YomuTubePipeline(app_config)
        for item in urls:
            pipeline.transcribe(item)


def _require_archive(config: Optional[Path], archive_dir: Optional[Path], key: str):
    archive = get_archive(_archive_root(config, archive_dir), key)
    if archive is None:
        raise typer.BadParameter(f"archive not found: {key}")
    return archive


def _collection_urls(
    url: str,
    config: AppConfig,
    *,
    max_results: int,
    since: str | None = None,
    duration_max_sec: int | None = None,
    include: str | None = None,
    exclude: str | None = None,
) -> list[str]:
    from .download.ytdlp_client import import_ytdlp

    ytdlp = import_ytdlp()
    options = {"quiet": True, "extract_flat": True, "playlistend": int(max_results), "ignoreerrors": True}
    with ytdlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = info.get("entries") if isinstance(info, dict) else []
    urls: list[str] = []
    existing_ids = {archive.video_id for archive in find_archives(config.archive_dir)}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        video_id = str(entry.get("id") or "")
        title = str(entry.get("title") or "")
        upload_date = str(entry.get("upload_date") or "")
        duration = entry.get("duration")
        if video_id and video_id in existing_ids:
            continue
        if since and upload_date and _date_digits(upload_date) < _date_digits(since):
            continue
        if duration_max_sec is not None and duration is not None and int(duration) > int(duration_max_sec):
            continue
        if include and include.casefold() not in title.casefold():
            continue
        if exclude and exclude.casefold() in title.casefold():
            continue
        item_url = entry.get("url") or entry.get("webpage_url")
        if item_url and "youtube.com" not in str(item_url) and video_id:
            item_url = f"https://www.youtube.com/watch?v={video_id}"
        if item_url:
            urls.append(str(item_url))
    return urls


def _write_queue(path: Path, urls: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{url}\n" for url in urls), encoding="utf-8")


def _date_digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())[:8]
