"""
modules/utils.py
==============================================================================
Small shared helpers used across the pipeline: filesystem helpers, timecode
formatting, JSON-safe LLM response parsing, and a subprocess/ffmpeg runner.
==============================================================================
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger("viral_shorts.utils")


def sanitize_filename(name: str, max_len: int = 80) -> str:
    """Strip characters that are unsafe for filenames and cap the length."""
    name = re.sub(r"[^\w\s\-]", "", name).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:max_len] or "clip"


def seconds_to_timecode(seconds: float) -> str:
    """Convert seconds -> SRT-style HH:MM:SS,mmm timecode."""
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def extract_json_block(raw_text: str) -> Any:
    """
    LLMs sometimes wrap JSON in markdown fences or add preamble text.
    This pulls out the first valid JSON object/array found in the string.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to locating the outermost { } or [ ]
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from LLM response:\n{raw_text[:500]}")


def run_ffmpeg(cmd: list[str], description: str = "ffmpeg operation") -> None:
    """
    Run an ffmpeg (or any) subprocess command with unified logging and error
    handling. Raises RuntimeError with captured stderr on failure.
    """
    logger.debug("Running command: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if result.stderr:
            logger.debug("%s stderr tail: %s", description, result.stderr[-500:])
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg (or the requested binary) was not found on PATH. "
            "Install ffmpeg and ensure it is accessible from the command line."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{description} failed (exit code {exc.returncode}).\n"
            f"--- stderr ---\n{exc.stderr[-2000:] if exc.stderr else '(empty)'}"
        ) from exc


def ensure_parent_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
