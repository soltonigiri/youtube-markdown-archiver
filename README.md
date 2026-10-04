# youtube-markdown-archiver

YouTubeの字幕・音声認識・画面内の文字をまとめ、検索できるMarkdownアーカイブを作るPython製CLIです。話者情報や品質レポートを保存し、SRT・VTT・CSVなどへ書き出せます。

## インストール

Python 3.10以上が必要です。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[download,asr]"
```

音声の抽出には`ffmpeg`と`ffprobe`を使います。OSのパッケージマネージャーなどでインストールし、PATHに追加してください。

必要な機能に合わせて、インストールするextraを選べます。

| extra | 機能・依存ライブラリ |
| --- | --- |
| `download` | yt-dlpによるメタデータ・字幕・動画の取得 |
| `asr` | faster-whisperによる音声認識 |
| `ocr` | OpenCVとTesseractのPythonアダプター |
| `paddleocr` | PaddleOCRとPaddlePaddle |
| `diarization` | pyannote.audioによる話者分離 |
| `full` | download・asr・ocr・diarizationの一括インストール |
| `dev` | テスト用の依存ライブラリ |

たとえば、開発と全機能の利用には`python -m pip install -e ".[dev,full]"`を使います。PaddleOCRは`full`に含まれないため、必要なら`paddleocr`も指定してください。

Tesseract OCRには`tesseract`本体と`jpn`・`eng`などの言語データが必要です。pyannoteによる話者分離には、Hugging Faceでのモデル利用条件の承認と`HF_TOKEN`の設定が必要です。`diarization.engine: local_cluster`では認証なしで話者を分類できます。

## 使い方

```bash
python -m yomutube doctor --json
python -m yomutube inspect "https://www.youtube.com/watch?v=VIDEO_ID"
python -m yomutube transcribe "https://www.youtube.com/watch?v=VIDEO_ID" --mode standard
```

`doctor`で依存ライブラリと外部コマンドを確認し、`inspect`で動画情報と字幕候補を調べます。`transcribe`は字幕を優先し、設定した条件に応じて音声認識やOCRを実行します。

```bash
python -m yomutube transcribe URL --mode quick
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

設定は`configs/default.yaml`を基準に、プロファイル・モード・CLIオプションで上書きできます。`configs/channels/*.yaml`ではチャンネル別の設定を指定できます。詳細は各コマンドの`--help`と[処理・出力仕様](仕様書.md)を参照してください。

## 出力

既定の保存先は`data/archives/<channel>/<title>/`、一時処理用の保存先は`data/work/`です。

| ファイル | 内容 |
| --- | --- |
| `index.md` | タイムスタンプ付きの本文 |
| `metadata.json` / `manifest.json` | 動画情報・処理状態 |
| `segments.jsonl` | 検索や書き出しに使う統合済みのセグメント |
| `raw_subtitles.jsonl` / `raw_asr.jsonl` / `raw_ocr.jsonl` | 取得・認識した各ソースの結果 |
| `words.jsonl` / `alignment.jsonl` | 単語・位置の対応 |
| `speaker_turns.jsonl` | 話者の区間 |
| `topics.jsonl` / `entities.jsonl` | 話題・固有表現 |
| `visual_text.jsonl` / `slides.jsonl` | 画面内の文字・スライド |
| `conflicts.jsonl` / `quality.json` | ソース間の食い違い・品質情報 |

出力するファイルは設定で選べます。[サンプル](examples/sample_archive/index.md)でMarkdownの形式を確認できます。

## 開発

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m yomutube index --archive-dir tests/fixtures/archive --output /tmp/yomutube-test.db
```

## ライセンス

コードは[MIT License](LICENSE)です。取得した動画・字幕などには、各提供元の利用条件が適用されます。
