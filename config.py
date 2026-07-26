"""
config.py
==============================================================================
Central configuration loader for the Viral Shorts Generator.

All modules import settings from here instead of reading os.environ directly.
This keeps configuration in a single, typed, validated place and makes the
rest of the codebase easy to test (you can monkeypatch `config.settings`).
==============================================================================
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root (no-op if it doesn't exist yet)
load_dotenv()


def _get_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    try:
        return float(val) if val is not None else default
    except ValueError:
        return default


@dataclass
class Settings:
    # --- LLM ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # --- Transcription ---
    transcriber_backend: str = os.getenv("TRANSCRIBER_BACKEND", "whisper").lower()
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")

    # --- Highlight extraction ---
    min_clip_duration: int = _get_int("MIN_CLIP_DURATION_SECONDS", 30)
    max_clip_duration: int = _get_int("MAX_CLIP_DURATION_SECONDS", 60)
    num_highlights: int = _get_int("NUM_HIGHLIGHTS_TO_EXTRACT", 5)
    min_viral_score: int = _get_int("MIN_VIRAL_SCORE", 60)

    # --- Output / crop ---
    output_width: int = _get_int("OUTPUT_WIDTH", 1080)
    output_height: int = _get_int("OUTPUT_HEIGHT", 1920)
    output_fps: int = _get_int("OUTPUT_FPS", 30)
    crop_smoothing_window: int = _get_int("CROP_SMOOTHING_WINDOW", 15)
    enable_split_screen: bool = _get_bool("ENABLE_SPLIT_SCREEN", True)

    # --- Captions ---
    caption_font_path: str = os.getenv(
        "CAPTION_FONT_PATH", "assets/fonts/Montserrat-ExtraBold.ttf"
    )
    caption_font_size: int = _get_int("CAPTION_FONT_SIZE", 90)
    caption_color: str = os.getenv("CAPTION_COLOR", "#FFFFFF")
    caption_highlight_color: str = os.getenv("CAPTION_HIGHLIGHT_COLOR", "#FFFC00")
    caption_stroke_color: str = os.getenv("CAPTION_STROKE_COLOR", "#000000")
    caption_stroke_width: int = _get_int("CAPTION_STROKE_WIDTH", 4)
    caption_position: str = os.getenv("CAPTION_POSITION", "center_bottom")
    caption_animation: str = os.getenv("CAPTION_ANIMATION", "pop")

    # --- Viral enhancements ---
    enable_sfx: bool = _get_bool("ENABLE_SFX", True)
    sfx_dir: str = os.getenv("SFX_DIR", "assets/sfx")
    enable_zoom_effect: bool = _get_bool("ENABLE_ZOOM_EFFECT", True)
    zoom_interval_min: float = _get_float("ZOOM_INTERVAL_MIN_SECONDS", 3)
    zoom_interval_max: float = _get_float("ZOOM_INTERVAL_MAX_SECONDS", 5)
    zoom_intensity: float = _get_float("ZOOM_INTENSITY", 1.08)

    enable_background_music: bool = _get_bool("ENABLE_BACKGROUND_MUSIC", True)
    music_dir: str = os.getenv("MUSIC_DIR", "assets/music")
    music_volume_db: float = _get_float("MUSIC_VOLUME_DB", -22)
    speech_duck_threshold_db: float = _get_float("SPEECH_DUCK_THRESHOLD_DB", -35)

    # --- Paths ---
    temp_dir: str = os.getenv("TEMP_DIR", "temp")
    output_dir: str = os.getenv("OUTPUT_DIR", "output")

    # --- Logging ---
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems (empty = OK)."""
        problems = []

        if self.llm_provider not in ("openai", "anthropic"):
            problems.append(f"LLM_PROVIDER must be 'openai' or 'anthropic', got '{self.llm_provider}'")
        if self.llm_provider == "openai" and not self.openai_api_key:
            problems.append("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            problems.append("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")

        if self.transcriber_backend not in ("whisper", "deepgram"):
            problems.append(
                f"TRANSCRIBER_BACKEND must be 'whisper' or 'deepgram', got '{self.transcriber_backend}'"
            )
        if self.transcriber_backend == "deepgram" and not self.deepgram_api_key:
            problems.append("DEEPGRAM_API_KEY is required when TRANSCRIBER_BACKEND=deepgram")

        if self.min_clip_duration >= self.max_clip_duration:
            problems.append("MIN_CLIP_DURATION_SECONDS must be less than MAX_CLIP_DURATION_SECONDS")

        return problems

    def ensure_dirs(self) -> None:
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.sfx_dir).mkdir(parents=True, exist_ok=True)
        Path(self.music_dir).mkdir(parents=True, exist_ok=True)
        Path(self.caption_font_path).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()


def setup_logging() -> logging.Logger:
    """Configure root logging once and return the app logger."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("viral_shorts")
    return logger


# A set of well-known SFX "cues" the LLM is allowed to request in highlight_finder.py.
# Map cue name -> expected filename inside SFX_DIR. Users drop matching audio files there.
SFX_CUE_MAP = {
    "swoosh": "swoosh.mp3",
    "pop": "pop.mp3",
    "vine_boom": "vine_boom.mp3",
    "ding": "ding.mp3",
    "whoosh_transition": "whoosh_transition.mp3",
    "impact": "impact.mp3",
    "notification": "notification.mp3",
}
