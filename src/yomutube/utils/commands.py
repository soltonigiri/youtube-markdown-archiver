from __future__ import annotations

import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_REDACTION_PATTERNS = [
    re.compile(r"(?i)\b(authorization\s*[:=]\s*)(bearer\s+)?[^\s;]+"),
    re.compile(r"(?i)\b(cookie\s*[:=]\s*)[^\r\n]+"),
    re.compile(r"(?i)\b((?:(?:access_)?token|refresh_token|id_token|api_?key|secret)\s*[:=]\s*)[^\s&;]+"),
]


def redact_sensitive_text(value: str | None) -> str:
    if not value:
        return ""
    redacted = value
    for pattern in _REDACTION_PATTERNS:
        redacted = pattern.sub(lambda match: f"{match.group(1)}[REDACTED]", redacted)
    return redacted


@dataclass(slots=True)
class CommandResult:
    args: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExternalCommandRunner:
    def __init__(self, *, timeout: float = 30) -> None:
        self.timeout = timeout

    def run(
        self,
        args: Sequence[str | Path],
        *,
        timeout: float | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        command = [str(arg) for arg in args]
        started = time.monotonic()
        try:
            proc = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout if timeout is None else timeout,
                cwd=str(cwd) if cwd is not None else None,
                env=dict(env) if env is not None else None,
            )
            duration = time.monotonic() - started
            return CommandResult(
                args=command,
                returncode=proc.returncode,
                stdout=redact_sensitive_text(proc.stdout),
                stderr=redact_sensitive_text(proc.stderr),
                duration_sec=duration,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
            return CommandResult(
                args=command,
                returncode=None,
                stdout=redact_sensitive_text(stdout),
                stderr=redact_sensitive_text(stderr),
                duration_sec=duration,
                timed_out=True,
                error=f"timed out after {exc.timeout} seconds",
            )
        except Exception as exc:
            duration = time.monotonic() - started
            return CommandResult(
                args=command,
                returncode=None,
                stdout="",
                stderr="",
                duration_sec=duration,
                error=redact_sensitive_text(str(exc)),
            )
