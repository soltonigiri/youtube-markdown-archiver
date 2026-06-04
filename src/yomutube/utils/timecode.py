from __future__ import annotations

import re


TIMECODE_RE = re.compile(
    r"(?:(?P<hours>\d+):)?(?P<minutes>\d{1,2}):(?P<seconds>\d{2})(?:[.,](?P<millis>\d{1,3}))?"
)


def ms_to_timecode(ms: int, *, include_millis: bool = True) -> str:
    ms = max(0, int(ms))
    hours, remainder = divmod(ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    if include_millis:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def seconds_to_timecode(seconds: float, *, include_millis: bool = True) -> str:
    return ms_to_timecode(round(float(seconds) * 1000), include_millis=include_millis)


def parse_timecode(value: str) -> int:
    match = TIMECODE_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"invalid timecode: {value}")
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    millis_raw = match.group("millis") or "0"
    millis = int(millis_raw.ljust(3, "0")[:3])
    return hours * 3_600_000 + minutes * 60_000 + seconds * 1000 + millis


def youtube_seconds(ms: int) -> int:
    return max(0, int(ms) // 1000)
