# youtube-markdown-archiver

youtube-markdown-archiver は、YouTube のメタデータ、字幕、ASR 出力、OCR テキストを統合し、検索しやすい Markdown アーカイブとして保存するローカル Python CLI です。

研究メモ、動画レビュー、個人の知識管理を想定しています。動画や音声ファイルそのものではなく、Markdown、JSONL、小さな SQLite インデックスを長期的な成果物として残す設計です。

## 機能

- YouTube URL を検査し、正規化したメタデータを取得します。
- 利用可能な手動字幕または自動字幕を取得し、扱いやすい形に正規化します。
- 字幕がない、または品質が不十分な場合に `faster-whisper` で ASR を実行します。
- OCR を有効にした場合、画面内の文字情報を抽出します。
- 字幕、ASR、OCR、話者、品質情報を統合してアーカイブを生成します。
- Markdown、JSONL、エクスポート用ファイル、品質レポート、SQLite 検索インデックスを書き出します。
- 重い動画・音声ファイルは一時処理用の入力として扱い、公開成果物には含めません。

## インストール

```bash
python -m pip install -e ".[dev]"
```

実動画を処理する場合は、必要な機能だけを追加でインストールしてください。

```bash
python -m pip install -e ".[core,download,subtitle]"
python -m pip install -e ".[asr]"
python -m pip install -e ".[ocr]"
python -m pip install -e ".[diarization]"
python -m pip install -e ".[full]"
```

標準の開発用インストールは意図的に軽量です。単体テストと import 確認には、動画ダウンロード、ASR モデル、OCR エンジン、話者分離モデルは不要です。
実際に使う機能の extra を入れたあとで `doctor` を実行してください。`.[dev]` だけの状態では、`yt-dlp` や `faster-whisper` などの実行時ツール不足が報告されることがあります。

## クイックスタート

```bash
python -m yomutube doctor --json
python -m yomutube inspect "https://www.youtube.com/watch?v=VIDEO_ID"
python -m yomutube transcribe "https://www.youtube.com/watch?v=VIDEO_ID" --mode standard
```

既定では、実行時の成果物は `data/` 以下に保存されます。

```text
data/
  archives/
  work/
  models/
  cache/
```

これらのパスは Git の管理対象外です。

## よく使うコマンド

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

## 出力形式

処理済みのアーカイブは、manifest とテキスト中心の成果物で構成されます。

```text
data/archives/<channel>/<title>/
  manifest.json
  metadata.json
  index.md
  segments.jsonl
  quality_report.json
```

`index.md` は人間が読むための主要ファイルです。JSONL ファイルには、検索、エクスポート、再処理に使える構造化済みセグメントを保存します。

小さな架空サンプルとして `examples/sample_archive/` を用意しています。

## データと権利

このリポジトリには、動画、音声ファイル、モデルキャッシュ、cookie、実動画由来の処理済みアーカイブは含めていません。

YouTube の規約、各動画の権利者の権利、ダウンロードした字幕・音声・OCR テキスト・生成アーカイブに適用されるルールは、利用者自身が確認してください。公開する権利がない処理済みアーカイブを公開しないでください。

## 開発

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m yomutube index --archive-dir tests/fixtures/archive --output /tmp/yomutube-test.db
```

## ライセンス

MIT License です。このライセンスの対象は、このリポジトリ内のソースコードです。第三者の動画、字幕、音声、OCR 結果、モデルの重み、プラットフォーム上のコンテンツに対する権利を付与するものではありません。
