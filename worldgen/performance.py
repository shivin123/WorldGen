"""Lightweight runtime performance recording for WorldGen.

This module intentionally uses a tiny process-local recorder rather than a
logging framework. It lets the CLI and generation stages write one portable
performance report that can be bundled with outputs and uploaded for review.
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

_START = time.perf_counter()
_EVENTS: list[tuple[float, str]] = []
_STAGES: list[tuple[str, float]] = []
_METADATA: dict[str, str] = {}


def reset() -> None:
    global _START
    _START = time.perf_counter()
    _EVENTS.clear()
    _STAGES.clear()
    _METADATA.clear()


def add_metadata(key: str, value: object) -> None:
    _METADATA[str(key)] = str(value)


def mark(message: str) -> None:
    _EVENTS.append((time.perf_counter() - _START, message))


def record_stage(name: str, seconds: float) -> None:
    _STAGES.append((name, float(seconds)))


def write_report(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = time.perf_counter() - _START
    lines: list[str] = [
        "WorldGen Performance Report",
        "===========================",
        "",
        f"Total wall time captured: {total:.2f} s",
        f"Python: {sys.version.split()[0]}",
        f"Platform: {platform.platform()}",
    ]
    if _METADATA:
        lines.append("")
        lines.append("Run metadata")
        lines.append("------------")
        for key in sorted(_METADATA):
            lines.append(f"{key}: {_METADATA[key]}")

    lines.append("")
    lines.append("Timed stages")
    lines.append("------------")
    if _STAGES:
        for name, seconds in sorted(_STAGES, key=lambda item: item[1], reverse=True):
            lines.append(f"{seconds:9.2f} s  {name}")
    else:
        lines.append("No timed stages were recorded.")

    lines.append("")
    lines.append("Progress events")
    lines.append("---------------")
    if _EVENTS:
        for seconds, message in _EVENTS:
            lines.append(f"{seconds:9.2f} s  {message}")
    else:
        lines.append("No progress events were recorded.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
