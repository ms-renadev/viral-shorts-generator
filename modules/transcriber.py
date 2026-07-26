"""
modules/transcriber.py
==============================================================================
Produces a word-level timestamped transcript from the source video's audio
track. Supports two backends selected via config.settings.transcriber_backend:

    - "whisper"  : local openai-whisper (or faster-whisper if installed)
    - "deepgram" : Deepgram's cloud transcription API (nova-2 model)

Public API:
    transcribe(video_path: str) -> Transcript
==============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from config import settings
from modules.utils import run_ffmpeg

logger = logging.getLogger("viral_shorts.transcriber")


@dataclass
class Word:
    text: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class Transcript:
    words: list[Word] = field(default_factory=list)
    full_text: str = ""
    language: str = "en"

    def words_in_range(self, start: float, end: float) -> list[Word]:
        return [w for w in self.words if w.start >= start and w.end <= end]

    def text_in_range(self, start: float, end: float) -> str:
        return " ".join(w.text for w in self.words_in_range(start, end))


def _extract_audio(video_path: str, temp_dir: str) -> str:
    """Extract a 16kHz mono WAV track for transcription (Whisper's preferred format)."""
    out_path = str(Path(temp_dir) / "audio_for_transcription.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        out_path,
    ]
    run_ffmpeg(cmd, description="audio extraction for transcription")
    return out_path


def _transcribe_with_whisper(audio_path: str) -> Transcript:
    """
    Local transcription using openai-whisper. Falls back gracefully if
    faster-whisper is installed and preferred (kept simple here using
    openai-whisper's `word_timestamps` feature).
    """
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "openai-whisper is not installed. Run `pip install -r requirements.txt`, "
            "or set TRANSCRIBER_BACKEND=deepgram in your .env."
        ) from exc

    logger.info("Loading Whisper model '%s' (first run downloads weights)...", settings.whisper_model_size)
    model = whisper.load_model(settings.whisper_model_size)

    logger.info("Transcribing audio with word-level timestamps... this can take a while.")
    result = model.transcribe(audio_path, word_timestamps=True, verbose=False)

    words: list[Word] = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            token = (w.get("word") or "").strip()
            if not token:
                continue
            words.append(
                Word(
                    text=token,
                    start=float(w.get("start", segment.get("start", 0.0))),
                    end=float(w.get("end", segment.get("end", 0.0))),
                    confidence=float(w.get("probability", 1.0)),
                )
            )

    full_text = result.get("text", "").strip()
    language = result.get("language", "en")

    logger.info("Whisper transcription complete: %d words detected.", len(words))
    return Transcript(words=words, full_text=full_text, language=language)


def _transcribe_with_deepgram(audio_path: str) -> Transcript:
    """Cloud transcription using Deepgram's nova-2 model with word timestamps."""
    try:
        from deepgram import DeepgramClient, PrerecordedOptions, FileSource
    except ImportError as exc:
        raise RuntimeError(
            "deepgram-sdk is not installed. Run `pip install -r requirements.txt`, "
            "or set TRANSCRIBER_BACKEND=whisper in your .env."
        ) from exc

    if not settings.deepgram_api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set in your .env file.")

    logger.info("Sending audio to Deepgram for transcription (model: nova-2)...")
    client = DeepgramClient(settings.deepgram_api_key)

    with open(audio_path, "rb") as f:
        buffer_data = f.read()

    payload: FileSource = {"buffer": buffer_data}
    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        punctuate=True,
        words=True,
        language="en",
    )

    response = client.listen.prerecorded.v("1").transcribe_file(payload, options)

    words: list[Word] = []
    channel = response.results.channels[0]
    alt = channel.alternatives[0]
    for w in alt.words:
        words.append(
            Word(
                text=w.word,
                start=float(w.start),
                end=float(w.end),
                confidence=float(getattr(w, "confidence", 1.0)),
            )
        )

    full_text = alt.transcript
    logger.info("Deepgram transcription complete: %d words detected.", len(words))
    return Transcript(words=words, full_text=full_text, language="en")


def transcribe(video_path: str) -> Transcript:
    """
    Main entry point: extracts audio from `video_path` and transcribes it
    using the backend configured in settings.transcriber_backend.
    """
    audio_path = _extract_audio(video_path, settings.temp_dir)

    if settings.transcriber_backend == "deepgram":
        return _transcribe_with_deepgram(audio_path)
    return _transcribe_with_whisper(audio_path)
