from __future__ import annotations

from pathlib import Path

from yomutube.config import AppConfig, bool_from_cli, deep_update, transcribe_overrides
from yomutube.utils.hashing import safe_path_component, safe_slug
from yomutube.utils.text import normalize_text
from yomutube.utils.timecode import ms_to_timecode, parse_timecode


def test_deep_update_keeps_nested_values() -> None:
    result = deep_update({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}})
    assert result == {"a": {"b": 3, "c": 2}}


def test_default_config_loads() -> None:
    config = AppConfig.load()
    assert config.version == "0.1.0"
    assert config.get("asr.model") == "medium"
    assert config.get("asr.when") == "no_good_subtitle"
    assert config.get("ocr.enabled") is False
    assert config.get("ocr.primary_engine") == "tesseract"
    assert config.get("ocr.frame_sampling.subtitle_roi_fps") == 0.1
    assert config.archive_dir == Path("data/archives")
    assert config.work_dir == Path("data/work")


def test_standard_mode_is_normal_transcribe_default() -> None:
    config = AppConfig.load().with_mode("standard")
    assert config.get("asr.when") == "no_good_subtitle"
    assert config.get("asr.model") == "medium"
    assert config.get("ocr.enabled") is False
    assert config.get("diarization.enabled") is False


def test_mode_overlay_is_separate_from_profile() -> None:
    config = AppConfig.load().with_mode("quick")
    assert config.get("asr.model") == "small"
    assert config.get("ocr.enabled") is False
    assert "standard" in config.get("modes")


def test_transcribe_overrides() -> None:
    overrides = transcribe_overrides(
        lang="ja",
        asr_model="small",
        asr_device="cuda",
        strict_device="true",
        asr_when="no-subtitle",
        ocr="auto",
        diarization="on",
        keep_temp_media="true",
        resume=False,
        rerun="overwrite",
        if_running="skip",
    )
    assert overrides["asr"]["language"] == "ja"
    assert overrides["asr"]["model"] == "small"
    assert overrides["asr"]["device"] == "cuda"
    assert overrides["asr"]["strict_device"] is True
    assert overrides["asr"]["when"] == "no_subtitle"
    assert overrides["ocr"]["enabled"] == "auto"
    assert overrides["diarization"]["enabled"] is True
    assert overrides["storage"]["keep_temp_media"] is True
    assert overrides["runtime"]["resume"] is False
    assert overrides["runtime"]["rerun_policy"] == "overwrite"
    assert overrides["runtime"]["if_running"] == "skip"


def test_bool_from_cli() -> None:
    assert bool_from_cli("on") is True
    assert bool_from_cli("false") is False


def test_timecode_roundtrip() -> None:
    assert ms_to_timecode(3_723_045) == "01:02:03.045"
    assert parse_timecode("01:02:03.045") == 3_723_045
    assert parse_timecode("02:03.5") == 123_500


def test_text_normalization_and_slug() -> None:
    assert normalize_text("<b>Ａ  B</b> ♪") == "A B"
    assert safe_slug("日本語 title!?") == "title"
    assert safe_path_component("【ポケモン解説】運動クラブ") == "【ポケモン解説】運動クラブ"
    assert safe_path_component("A/B\\C") == "A／B／C"
