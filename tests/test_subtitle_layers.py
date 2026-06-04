from __future__ import annotations

from pathlib import Path

from yomutube.models import SubtitleTrack
from yomutube.subtitle.normalizer import normalize_subtitle_text
from yomutube.subtitle.parser import parse_subtitle_file, parse_subtitle_text


def test_normalize_subtitle_text_strips_tags_nfkc_and_spaces() -> None:
    assert normalize_subtitle_text("<b>Ｈｅｌｌｏ</b>\n  world") == "Hello world"


def test_parse_vtt_uses_track_segment_source(tmp_path: Path) -> None:
    path = tmp_path / "caption.vtt"
    path.write_text(
        """WEBVTT

00:00:01.000 --> 00:00:02.500
<c>こんにちは</c>   世界
""",
        encoding="utf-8",
    )
    track = SubtitleTrack(source="manual", language="ja", ext="vtt", path=path, raw={"video_id": "video1"})

    segments = parse_subtitle_file(track)

    assert len(segments) == 1
    assert segments[0].source == "manual_subtitle"
    assert segments[0].video_id == "video1"
    assert segments[0].start_ms == 1000
    assert segments[0].end_ms == 2500
    assert segments[0].text == "こんにちは 世界"
    assert segments[0].language == "ja"


def test_parse_srt_json3_and_ttml() -> None:
    srt_segments = parse_subtitle_text(
        "1\n00:00:01,000 --> 00:00:02,000\nSRT line\n",
        "srt",
        source="manual_subtitle",
        video_id="video1",
    )
    json3_segments = parse_subtitle_text(
        '{"events":[{"tStartMs":2000,"dDurationMs":750,"segs":[{"utf8":"JSON3"},{"utf8":" line"}]}]}',
        "json3",
        source="auto_subtitle",
        video_id="video1",
    )
    ttml_segments = parse_subtitle_text(
        '<tt><body><div><p begin="00:00:03.000" end="00:00:04.000">TTML line</p></div></body></tt>',
        "ttml",
        source="manual_subtitle",
        video_id="video1",
    )

    assert srt_segments[0].text == "SRT line"
    assert json3_segments[0].start_ms == 2000
    assert json3_segments[0].end_ms == 2750
    assert json3_segments[0].text == "JSON3 line"
    assert ttml_segments[0].start_ms == 3000
    assert ttml_segments[0].end_ms == 4000
