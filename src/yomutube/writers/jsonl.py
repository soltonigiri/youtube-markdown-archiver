from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from yomutube.state import atomic_write_text


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(target, content)


def read_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict] = []
    with source.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows
