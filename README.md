# youtube-markdown-archiver

youtube-markdown-archiver is a local Python CLI that turns YouTube metadata, subtitles, ASR output, and OCR text into searchable Markdown archives.

It is designed for research notes, video review, and personal knowledge management workflows where the durable output should be text files, JSONL artifacts, and a small SQLite index instead of downloaded media files.

## Features

- Inspect YouTube URLs and capture normalized metadata.
- Download and normalize manual or automatic subtitles when available.
- Run ASR with `faster-whisper` when subtitles are missing or weak.
- Extract visual text through OCR when enabled.
- Merge subtitle, ASR, OCR, speaker, and quality signals into archive files.
- Write Markdown, JSONL, export formats, quality reports, and SQLite search indexes.
- Keep heavy video/audio files as temporary processing inputs rather than published artifacts.

## Install

```bash
python -m pip install -e ".[dev]"
```

For real video processing, install only the feature groups you need:

```bash
python -m pip install -e ".[core,download,subtitle]"
python -m pip install -e ".[asr]"
python -m pip install -e ".[ocr]"
python -m pip install -e ".[diarization]"
python -m pip install -e ".[full]"
```

The default development install is intentionally light. Unit tests and imports do not require video downloads, ASR models, OCR engines, or diarization models.
Run `doctor` after installing the feature extras you intend to use. With only `.[dev]`, it may report missing runtime tools such as `yt-dlp` or `faster-whisper`.

## Quick Start

```bash
python -m yomutube doctor --json
python -m yomutube inspect "https://www.youtube.com/watch?v=VIDEO_ID"
python -m yomutube transcribe "https://www.youtube.com/watch?v=VIDEO_ID" --mode standard
```

By default, runtime artifacts are written under `data/`:

```text
data/
  archives/
  work/
  models/
  cache/
```

These paths are ignored by Git.

## Common Commands

```bash
python -m yomutube transcribe URL --mode quick
python -m yomutube transcribe URL --mode standard
python -m yomutube transcribe URL --mode standard --ocr on
python -m yomutube transcribe URL --mode full --asr-when always
python -m yomutube batch urls.txt
python -m yomutube index
python -m yomutube list
python -m yomutube show VIDEO_ID
python -m yomutube search "keyword"
python -m yomutube export VIDEO_ID --format srt
python -m yomutube quote VIDEO_ID --at 00:01:23
```

## Output Shape

A completed archive is organized around a manifest and text-first artifacts:

```text
data/archives/<channel>/<title>/
  manifest.json
  metadata.json
  index.md
  segments.jsonl
  quality_report.json
```

`index.md` is the human-facing reading surface. JSONL files preserve structured segments for search, export, and later reprocessing.

See `examples/sample_archive/` for a small fictional archive.

## Data And Rights

This repository does not include videos, audio files, model caches, cookies, or processed archives from real videos.

You are responsible for following YouTube's terms, the rights of each video owner, and the rules that apply to downloaded subtitles, audio, OCR text, and generated archives. Do not publish processed archives unless you have the right to publish that material.

## Development

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m yomutube index --archive-dir tests/fixtures/archive --output /tmp/yomutube-test.db
```

## License

MIT. The license covers this source code. It does not grant rights to third-party videos, subtitles, audio, OCR results, model weights, or platform content.
