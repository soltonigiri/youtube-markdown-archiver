from __future__ import annotations

import json
import sys

from yomutube.doctor import CheckResult, command_version, doctor_as_json
from yomutube.utils.commands import ExternalCommandRunner


def test_external_command_runner_captures_duration_and_redacts() -> None:
    runner = ExternalCommandRunner(timeout=5)

    result = runner.run(
        [
            sys.executable,
            "-c",
            "print('authorization: Bearer secret-token'); print('cookie=session=abc123')",
        ]
    )

    assert result.ok is True
    assert result.duration_sec >= 0
    assert "secret-token" not in result.stdout
    assert "abc123" not in result.stdout
    assert "[REDACTED]" in result.stdout


def test_command_version_uses_external_command_runner() -> None:
    runner = ExternalCommandRunner(timeout=5)

    version = command_version(
        sys.executable,
        ["-c", "print('tool 1.2.3')"],
        runner=runner,
    )

    assert version == "tool 1.2.3"


def test_doctor_json_report_has_summary_and_check_guidance() -> None:
    payload = json.loads(
        doctor_as_json(
            [
                CheckResult(
                    name="yt-dlp",
                    status="missing",
                    detail=None,
                    required=True,
                    impact="download disabled",
                    suggestion="Install download extra.",
                ),
                CheckResult(name="torch:cuda", status="ok", detail="available=False"),
            ]
        )
    )

    assert payload["summary"]["total"] == 2
    assert payload["summary"]["missing_required"] == 1
    assert payload["checks"][0]["impact"] == "download disabled"
    assert payload["checks"][0]["suggestion"] == "Install download extra."
