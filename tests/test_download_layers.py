from __future__ import annotations

from yomutube.config import AppConfig
from yomutube.download.subtitle_selector import select_subtitle_track
from yomutube.download.ytdlp_client import _base_options, discover_downloaded_paths, sanitize_info


def test_default_format_prefers_ocr_readable_video_codecs() -> None:
    options = _base_options(AppConfig.load())
    fmt = str(options["format"])

    assert "vcodec*=avc1" in fmt
    assert "vcodec!*=av01" in fmt


def test_selects_manual_ja_vtt_before_auto_and_en() -> None:
    info = {
        "subtitles": {
            "en": [{"ext": "vtt", "url": "https://example.test/en.vtt"}],
            "ja": [
                {"ext": "json3", "url": "https://example.test/ja.json3"},
                {"ext": "vtt", "url": "https://example.test/ja.vtt"},
            ],
        },
        "automatic_captions": {
            "ja": [{"ext": "vtt", "url": "https://example.test/auto-ja.vtt"}],
        },
    }

    track = select_subtitle_track(info)

    assert track is not None
    assert track.source == "manual"
    assert track.language == "ja"
    assert track.ext == "vtt"
    assert track.segment_source == "manual_subtitle"


def test_ignores_auto_subtitles_by_default_when_manual_missing() -> None:
    info = {
        "subtitles": {},
        "automatic_captions": {
            "en-US": [{"ext": "ttml", "url": "https://example.test/en.ttml"}],
        },
    }

    track = select_subtitle_track(info)

    assert track is None


def test_selects_auto_en_when_explicitly_allowed() -> None:
    info = {
        "subtitles": {},
        "automatic_captions": {
            "en-US": [{"ext": "ttml", "url": "https://example.test/en.ttml"}],
        },
    }

    track = select_subtitle_track(info, AppConfig({"subtitles": {"fallback_to_auto": True}}))

    assert track is not None
    assert track.source == "auto"
    assert track.language == "en-US"
    assert track.ext == "ttml"


def test_sanitize_info_redacts_heavy_private_fields() -> None:
    info = {
        "id": "abc123",
        "title": "Example",
        "formats": [{"url": "https://example.test/media"}],
        "http_headers": {"Cookie": "secret"},
        "subtitles": {"ja": []},
        "nested": {"requested_formats": [{"url": "hidden"}], "kept": True},
    }

    sanitized = sanitize_info(info)

    assert sanitized == {"id": "abc123", "title": "Example", "nested": {"kept": True}}


def test_discovers_downloaded_paths(tmp_path) -> None:
    media_dir = tmp_path / "media"
    subtitle_dir = media_dir / "subtitles"
    subtitle_dir.mkdir(parents=True)
    video = media_dir / "abc123.mp4"
    subtitle = subtitle_dir / "abc123.vtt"
    info_json = media_dir / "abc123.info.json"
    video.write_text("video", encoding="utf-8")
    subtitle.write_text("WEBVTT", encoding="utf-8")
    info_json.write_text("{}", encoding="utf-8")

    result = discover_downloaded_paths(media_dir)

    assert result.video_path == video
    assert result.subtitle_path == subtitle
    assert result.info_path == info_json
    assert result.video_or_audio == video
