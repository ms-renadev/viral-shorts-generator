"""
modules/downloader.py
==============================================================================
Handles acquisition of the source video, either by downloading it from
YouTube via `yt-dlp`, or by validating a local MP4 path the user supplied.

Public API:
    resolve_input_source(source: str, temp_dir: str) -> DownloadResult
==============================================================================
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from modules.utils import sanitize_filename

logger = logging.getLogger("viral_shorts.downloader")

YOUTUBE_URL_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/"
)


@dataclass
class DownloadResult:
    video_path: str          # local path to the downloaded/validated MP4
    title: str                # video title (or filename stem for local files)
    duration_seconds: float   # total duration
    source_url: str | None    # original URL, or None for local files


def is_youtube_url(source: str) -> bool:
    return bool(YOUTUBE_URL_PATTERN.match(source.strip()))


def download_youtube_video(url: str, temp_dir: str) -> DownloadResult:
    """
    Download the best-quality progressive (video+audio combined) MP4 stream
    for a given YouTube URL using yt-dlp's Python API.
    """
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    out_template = str(Path(temp_dir) / "%(id)s_source.%(ext)s")

    ydl_opts = {
        # Prefer a single muxed mp4 stream <=1080p to keep processing fast;
        # falls back to best video+audio merged into mp4 otherwise.
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": out_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
    }

    logger.info("Downloading video from YouTube: %s", url)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)
            # If merged, extension will be mp4 per merge_output_format
            filepath = str(Path(filepath).with_suffix(".mp4"))
    except Exception as exc:  # yt_dlp raises its own DownloadError subclasses
        raise RuntimeError(f"Failed to download video from '{url}': {exc}") from exc

    if not Path(filepath).exists():
        raise RuntimeError(
            f"yt-dlp reported success but output file was not found at: {filepath}"
        )

    title = info.get("title", "youtube_video")
    duration = float(info.get("duration") or 0.0)

    logger.info("Download complete: '%s' (%.1fs) -> %s", title, duration, filepath)
    return DownloadResult(
        video_path=filepath,
        title=title,
        duration_seconds=duration,
        source_url=url,
    )


def validate_local_file(path: str) -> DownloadResult:
    """Validate a user-supplied local video file and probe its duration."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Local video file not found: {p}")
    if p.suffix.lower() not in (".mp4", ".mov", ".mkv", ".webm", ".avi"):
        logger.warning("File extension '%s' is unusual for a video; attempting anyway.", p.suffix)

    duration = _probe_duration(str(p))
    logger.info("Using local video file: %s (%.1fs)", p, duration)
    return DownloadResult(
        video_path=str(p),
        title=sanitize_filename(p.stem),
        duration_seconds=duration,
        source_url=None,
    )


def _probe_duration(path: str) -> float:
    """Use ffprobe to get duration in seconds; returns 0.0 if it fails."""
    import subprocess
    import json

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0.0))
    except Exception as exc:  # noqa: BLE001 - best-effort probing
        logger.warning("Could not probe duration via ffprobe: %s", exc)
        return 0.0


def resolve_input_source(source: str, temp_dir: str) -> DownloadResult:
    """
    Entry point used by main.py. Accepts either a YouTube URL or a local
    file path and returns a normalized DownloadResult.
    """
    source = source.strip()
    if is_youtube_url(source):
        return download_youtube_video(source, temp_dir)
    return validate_local_file(source)
