from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig
from .utils.commands import ExternalCommandRunner


@dataclass(slots=True)
class CheckResult:
    name: str
    status: str
    detail: str | None = None
    required: bool = False
    impact: str | None = None
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def command_version(
    command: str,
    args: list[str] | None = None,
    *,
    runner: ExternalCommandRunner | None = None,
) -> str | None:
    binary = shutil.which(command)
    if not binary:
        return None
    result = (runner or ExternalCommandRunner(timeout=10)).run([binary, *(args or ["--version"])])
    output = result.stdout or result.stderr
    if result.error:
        return f"{binary}: {result.error}"
    first_line = output.splitlines()[0] if output else binary
    return first_line.strip()


def tesseract_languages(*, runner: ExternalCommandRunner | None = None) -> list[str]:
    binary = shutil.which("tesseract")
    if not binary:
        return []
    result = (runner or ExternalCommandRunner(timeout=10)).run([binary, "--list-langs"])
    if result.error:
        return []
    output = result.stdout or result.stderr
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return [line for line in lines if not line.lower().startswith("list of")]


def torch_cuda_detail() -> str:
    if not module_available("torch"):
        return "torch not installed"
    try:
        import torch  # type: ignore

        available = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if available else "none"
        return f"available={available}, device={name}"
    except Exception as exc:
        return f"torch import failed: {exc}"


def run_doctor(config: AppConfig | None = None) -> list[CheckResult]:
    config = config or AppConfig.load()
    runner = ExternalCommandRunner(timeout=10)
    checks: list[CheckResult] = []

    command_guidance = {
        "yt-dlp": ("download and metadata fetch cannot run", "Install with the download extra or add yt-dlp to PATH."),
        "ffmpeg": ("audio extraction and media conversion cannot run", "Install ffmpeg and make it available on PATH."),
        "ffprobe": ("media inspection cannot run", "Install ffmpeg/ffprobe and make it available on PATH."),
        "tesseract": ("Tesseract OCR fallback is unavailable", "Install tesseract if OCR fallback is needed."),
    }
    for command, required in [("yt-dlp", True), ("ffmpeg", True), ("ffprobe", True), ("tesseract", False)]:
        detail = command_version(command, runner=runner)
        impact, suggestion = command_guidance[command]
        checks.append(
            CheckResult(
                name=command,
                status="ok" if detail else "missing",
                detail=detail,
                required=required,
                impact=None if detail else impact,
                suggestion=None if detail else suggestion,
            )
        )

    module_guidance = {
        "faster_whisper": ("ASR transcription cannot run", "Install the asr extra."),
        "cv2": ("frame preprocessing and OCR image handling may be limited", "Install the ocr extra."),
        "paddleocr": ("PaddleOCR engine is unavailable", "Install the paddleocr extra if PaddleOCR is needed."),
        "pytesseract": ("Tesseract Python OCR adapter is unavailable", "Install the ocr extra if Tesseract OCR is needed."),
        "yaml": ("configuration loading cannot run", "Install the core dependencies."),
        "typer": ("CLI cannot run", "Install the core dependencies."),
        "rich": ("CLI table output may fail", "Install the core dependencies."),
    }
    for module, required in [
        ("faster_whisper", bool(config.get("asr.enabled", True))),
        ("cv2", bool(config.get("ocr.enabled", True))),
        ("paddleocr", False),
        ("pytesseract", False),
        ("yaml", True),
        ("typer", True),
        ("rich", True),
    ]:
        available = module_available(module)
        impact, suggestion = module_guidance[module]
        checks.append(
            CheckResult(
                name=f"python:{module}",
                status="ok" if available else "missing",
                required=required,
                impact=None if available else impact,
                suggestion=None if available else suggestion,
            )
        )

    langs = tesseract_languages(runner=runner)
    has_tesseract = bool(shutil.which("tesseract"))
    lang_ok = bool({"jpn", "eng"}.intersection(langs))
    checks.append(
        CheckResult(
            name="tesseract:langs",
            status="ok" if lang_ok else ("missing" if has_tesseract else "skipped"),
            detail=", ".join(langs[:20]) if langs else None,
            required=False,
            impact=None if lang_ok or not has_tesseract else "Tesseract OCR language coverage is limited",
            suggestion=None if lang_ok or not has_tesseract else "Install jpn or eng tessdata as needed.",
        )
    )

    checks.append(
        CheckResult(
            name="torch:cuda",
            status="ok",
            detail=torch_cuda_detail(),
            required=False,
            impact="GPU ASR may fall back to CPU if CUDA is unavailable",
            suggestion="Install CUDA-enabled torch if GPU ASR is required.",
        )
    )

    for label, path in [("archive_dir", config.archive_dir), ("work_dir", config.work_dir)]:
        try:
            Path(path).expanduser().mkdir(parents=True, exist_ok=True)
            status = "ok"
            detail = str(Path(path).expanduser())
        except Exception as exc:
            status = "error"
            detail = str(exc)
        checks.append(
            CheckResult(
                name=f"path:{label}",
                status=status,
                detail=detail,
                required=True,
                impact=None if status == "ok" else "archive or work files cannot be written",
                suggestion=None if status == "ok" else "Check path permissions and parent directories.",
            )
        )

    return checks


def doctor_exit_code(checks: list[CheckResult]) -> int:
    return 1 if any(check.required and check.status not in {"ok", "skipped"} for check in checks) else 0


def doctor_as_json(checks: list[CheckResult]) -> str:
    missing_required = sum(1 for check in checks if check.required and check.status not in {"ok", "skipped"})
    payload = {
        "summary": {
            "total": len(checks),
            "ok": sum(1 for check in checks if check.status == "ok"),
            "warning": sum(1 for check in checks if check.status not in {"ok", "skipped"} and not check.required),
            "error": sum(1 for check in checks if check.required and check.status not in {"ok", "skipped"}),
            "missing_required": missing_required,
        },
        "checks": [check.to_dict() for check in checks],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
