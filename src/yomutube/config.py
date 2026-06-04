from __future__ import annotations

import copy
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent


def default_config_path() -> Path:
    module_path = Path(__file__).resolve()
    candidates = [
        module_path.parents[2] / "configs" / "default.yaml",
        module_path.parents[1] / "configs" / "default.yaml",
        PACKAGE_ROOT / "configs" / "default.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_CONFIG_PATH = default_config_path()


class ConfigError(ValueError):
    pass


def deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config must be a mapping: {path}")
    return data


@dataclass(slots=True)
class AppConfig:
    data: dict[str, Any]
    source_path: Path = DEFAULT_CONFIG_PATH

    @classmethod
    def load(cls, path: str | Path | None = None, profile: str | None = None) -> "AppConfig":
        source_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
        data = load_yaml(source_path)
        config = cls(data=data, source_path=source_path)
        if profile and profile != "default":
            config = config.with_profile(profile)
        return config

    def with_profile(self, profile: str) -> "AppConfig":
        profile_path = self.source_path.parent / "profiles" / f"{profile}.yaml"
        if not profile_path.exists():
            raise ConfigError(f"profile not found: {profile}")
        return AppConfig(deep_update(self.data, load_yaml(profile_path)), self.source_path)

    def with_mode(self, mode: str | None) -> "AppConfig":
        if not mode:
            return self
        mode_name = normalize_enum(mode, ALLOWED_MODES, "mode")
        modes = self.data.get("modes") or {}
        if not isinstance(modes, dict) or mode_name not in modes:
            raise ConfigError(f"mode not found: {mode_name}")
        overlay = modes[mode_name]
        if not isinstance(overlay, dict):
            raise ConfigError(f"mode must be a mapping: {mode_name}")
        return AppConfig(deep_update(self.data, overlay), self.source_path)

    def with_channel_profile(self, metadata: dict[str, Any]) -> "AppConfig":
        channels_dir = self.source_path.parent / "channels"
        if not channels_dir.exists():
            return self
        channel_id = metadata.get("channel_id")
        channel_name = metadata.get("channel") or metadata.get("channel_name") or ""
        merged = self.data
        for path in sorted(channels_dir.glob("*.yaml")):
            candidate = load_yaml(path)
            match = candidate.get("match") or {}
            if match.get("channel_id") and match.get("channel_id") != channel_id:
                continue
            name_pattern = match.get("channel_name_regex") or match.get("channel_name_glob")
            if name_pattern and not fnmatch.fnmatch(channel_name, name_pattern):
                continue
            body = {k: v for k, v in candidate.items() if k not in {"match", "inherit"}}
            merged = deep_update(merged, body)
        return AppConfig(merged, self.source_path)

    def with_overrides(self, overrides: dict[str, Any]) -> "AppConfig":
        return AppConfig(deep_update(self.data, overrides), self.source_path)

    def get(self, dotted: str, default: Any = None) -> Any:
        current: Any = self.data
        for part in dotted.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def path(self, dotted: str) -> Path:
        value = self.get(dotted)
        if value is None:
            raise ConfigError(f"path missing: {dotted}")
        return Path(str(value)).expanduser()

    @property
    def archive_dir(self) -> Path:
        return self.path("paths.archive_dir")

    @property
    def work_dir(self) -> Path:
        return self.path("paths.work_dir")

    @property
    def version(self) -> str:
        return str(self.get("app.version", "0.1.0"))


def bool_from_cli(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"expected on/off boolean, got: {value}")


ALLOWED_MODES = {"quick", "standard", "full"}
ALLOWED_ASR_WHEN = {"always", "no_subtitle", "low_coverage", "no_good_subtitle", "never"}
ALLOWED_AUTO_SWITCH = {"auto", "on", "off"}
ALLOWED_RERUN = {"reuse_done", "overwrite", "new_run"}
ALLOWED_IF_RUNNING = {"fail", "skip", "wait"}


def normalize_enum(value: str, allowed: set[str], label: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in allowed:
        choices = ", ".join(sorted(item.replace("_", "-") for item in allowed))
        raise ConfigError(f"expected {label} to be one of: {choices}")
    return normalized


def auto_switch_from_cli(value: str | bool | None) -> str | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"on", "true", "1", "yes"}:
        return True
    if normalized in {"off", "false", "0", "no"}:
        return False
    raise ConfigError(f"expected auto/on/off, got: {value}")


def transcribe_overrides(
    *,
    lang: str | None = None,
    asr_model: str | None = None,
    asr_when: str | None = None,
    asr_device: str | None = None,
    strict_device: str | bool | None = None,
    ocr: str | bool | None = None,
    diarization: str | bool | None = None,
    max_height: int | None = None,
    subtitle_policy: str | None = None,
    keep_temp_media: str | bool | None = None,
    resume: str | bool | None = None,
    rerun: str | None = None,
    if_running: str | None = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if lang and lang != "auto":
        overrides.setdefault("asr", {})["language"] = lang
    elif lang == "auto":
        overrides.setdefault("asr", {})["language"] = "auto"
    if asr_model:
        overrides.setdefault("asr", {})["model"] = asr_model
    if asr_when:
        overrides.setdefault("asr", {})["when"] = normalize_enum(asr_when, ALLOWED_ASR_WHEN, "asr.when")
    if asr_device:
        overrides.setdefault("asr", {})["device"] = asr_device
    if strict_device is not None:
        overrides.setdefault("asr", {})["strict_device"] = bool_from_cli(strict_device)
    if ocr is not None:
        overrides.setdefault("ocr", {})["enabled"] = auto_switch_from_cli(ocr)
    if diarization is not None:
        overrides.setdefault("diarization", {})["enabled"] = auto_switch_from_cli(diarization)
    if max_height:
        overrides.setdefault("ytdlp", {})["max_height"] = int(max_height)
        overrides.setdefault("media", {})["max_video_height_for_ocr"] = int(max_height)
    if subtitle_policy:
        overrides.setdefault("subtitles", {})["policy"] = subtitle_policy
    if keep_temp_media is not None:
        overrides.setdefault("storage", {})["keep_temp_media"] = bool_from_cli(keep_temp_media)
    if resume is not None:
        overrides.setdefault("runtime", {})["resume"] = bool_from_cli(resume)
    if rerun:
        overrides.setdefault("runtime", {})["rerun_policy"] = normalize_enum(rerun, ALLOWED_RERUN, "runtime.rerun_policy")
    if if_running:
        overrides.setdefault("runtime", {})["if_running"] = normalize_enum(if_running, ALLOWED_IF_RUNNING, "runtime.if_running")
    return overrides
