"""
modules/editor.py
==============================================================================
The rendering engine. Takes a source video + a Highlight + a Transcript +
a CropPlan and produces a finished, vertical (9:16) short-form video with:

    - Speaker-tracked / split-screen cropping
    - Word-by-word animated captions ("Hormozi style") with emphasis colors
    - Context-aware SFX stings at LLM-suggested cue points
    - Periodic subtle zoom pulses for retention
    - Background music with speech-aware ducking

Built on MoviePy (which wraps ffmpeg) for composition, with a raw ffmpeg
fallback path available via modules/utils.run_ffmpeg for anything MoviePy
handles awkwardly (e.g. final encode settings).

Public API:
    render_short(video_path, highlight, transcript, crop_plan, output_path)
==============================================================================
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
from moviepy.editor import (
    VideoFileClip,
    CompositeVideoClip,
    ImageClip,
    AudioFileClip,
    CompositeAudioClip,
    concatenate_videoclips,
)
from moviepy.video.fx.resize import resize as mp_resize
from moviepy.audio.fx.volumex import volumex
from PIL import Image, ImageDraw, ImageFont

from config import settings, SFX_CUE_MAP
from modules.cropper import CropPlan
from modules.highlight_finder import Highlight
from modules.transcriber import Transcript, Word

logger = logging.getLogger("viral_shorts.editor")


# ==============================================================================
# CROPPING
# ==============================================================================

def _crop_center_at(t: float, crop_plan: CropPlan, target_w: int, target_h: int) -> tuple[int, int]:
    """
    Given a timestamp, interpolate the tracked speaker position and return
    the top-left (x, y) pixel offset to crop a target_w x target_h window
    from the source frame, keeping the speaker centered.
    """
    src_w, src_h = crop_plan.source_width, crop_plan.source_height
    frames = crop_plan.frames
    if not frames:
        return max(0, (src_w - target_w) // 2), max(0, (src_h - target_h) // 2)

    # Find surrounding keyframes and linearly interpolate
    if t <= frames[0].time:
        cx, cy = frames[0].center_x, frames[0].center_y
    elif t >= frames[-1].time:
        cx, cy = frames[-1].center_x, frames[-1].center_y
    else:
        for i in range(len(frames) - 1):
            f0, f1 = frames[i], frames[i + 1]
            if f0.time <= t <= f1.time:
                span = max(f1.time - f0.time, 1e-6)
                ratio = (t - f0.time) / span
                cx = f0.center_x + (f1.center_x - f0.center_x) * ratio
                cy = f0.center_y + (f1.center_y - f0.center_y) * ratio
                break
        else:
            cx, cy = 0.5, 0.4

    px, py = cx * src_w, cy * src_h
    x = int(np.clip(px - target_w / 2, 0, max(0, src_w - target_w)))
    y = int(np.clip(py - target_h / 2, 0, max(0, src_h - target_h)))
    return x, y


def _compute_crop_dimensions(src_w: int, src_h: int, out_w: int, out_h: int) -> tuple[int, int]:
    """
    Compute the crop window size (in source pixels) that matches the target
    aspect ratio while fitting inside the source frame, maximizing coverage.
    """
    target_aspect = out_w / out_h
    # Try cropping full height, deriving width from target aspect
    crop_h = src_h
    crop_w = int(round(crop_h * target_aspect))
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(round(crop_w / target_aspect))
    return crop_w, crop_h


def apply_speaker_tracked_crop(clip: VideoFileClip, crop_plan: CropPlan) -> VideoFileClip:
    """
    Apply a per-frame dynamic crop that follows the tracked speaker, then
    resize to the final output resolution. Handles both single-speaker and
    split-screen modes.
    """
    out_w, out_h = settings.output_width, settings.output_height

    if crop_plan.mode == "split_screen":
        return _apply_split_screen_crop(clip, crop_plan)

    crop_w, crop_h = _compute_crop_dimensions(clip.w, clip.h, out_w, out_h)

    def make_frame(get_frame, t):
        frame = get_frame(t)
        x, y = _crop_center_at(t, crop_plan, crop_w, crop_h)
        x = min(x, frame.shape[1] - crop_w)
        y = min(y, frame.shape[0] - crop_h)
        return frame[y : y + crop_h, x : x + crop_w]

    cropped = clip.fl(make_frame, apply_to=["mask"])
    cropped = cropped.resize((out_w, out_h))
    return cropped


def _apply_split_screen_crop(clip: VideoFileClip, crop_plan: CropPlan) -> VideoFileClip:
    """
    Build a vertical split-screen: top half follows speaker A, bottom half
    follows speaker B, each cropped to a square-ish window and stacked.
    """
    out_w, out_h = settings.output_width, settings.output_height
    half_h = out_h // 2
    src_w, src_h = clip.w, clip.h

    # Each half uses a crop window sized to fill out_w x half_h
    crop_w, crop_h = _compute_crop_dimensions(src_w, src_h, out_w, half_h)

    def make_half(frames: list, get_frame, t):
        frame = get_frame(t)
        cx, cy = 0.5, 0.4
        if frames:
            if t <= frames[0].time:
                cx, cy = frames[0].center_x, frames[0].center_y
            elif t >= frames[-1].time:
                cx, cy = frames[-1].center_x, frames[-1].center_y
            else:
                for i in range(len(frames) - 1):
                    f0, f1 = frames[i], frames[i + 1]
                    if f0.time <= t <= f1.time:
                        span = max(f1.time - f0.time, 1e-6)
                        ratio = (t - f0.time) / span
                        cx = f0.center_x + (f1.center_x - f0.center_x) * ratio
                        cy = f0.center_y + (f1.center_y - f0.center_y) * ratio
                        break
        px, py = cx * src_w, cy * src_h
        x = int(np.clip(px - crop_w / 2, 0, max(0, src_w - crop_w)))
        y = int(np.clip(py - crop_h / 2, 0, max(0, src_h - crop_h)))
        return frame[y : y + crop_h, x : x + crop_w]

    top = clip.fl(lambda gf, t: make_half(crop_plan.speaker_a_frames, gf, t))
    top = top.resize((out_w, half_h))
    bottom = clip.fl(lambda gf, t: make_half(crop_plan.speaker_b_frames, gf, t))
    bottom = bottom.resize((out_w, half_h))

    top = top.set_position(("center", 0))
    bottom = bottom.set_position(("center", half_h))

    composite = CompositeVideoClip([top, bottom], size=(out_w, out_h))
    composite = composite.set_duration(clip.duration).set_audio(clip.audio)
    return composite


# ==============================================================================
# ZOOM PULSES (retention effect)
# ==============================================================================

def apply_zoom_pulses(clip: VideoFileClip) -> VideoFileClip:
    """
    Applies a subtle rhythmic zoom in/out every ZOOM_INTERVAL seconds to keep
    the frame visually "alive" without being distracting.
    """
    if not settings.enable_zoom_effect:
        return clip

    interval = random.uniform(settings.zoom_interval_min, settings.zoom_interval_max)
    intensity = settings.zoom_intensity

    def zoom_factor(t: float) -> float:
        # Smooth triangular pulse between 1.0 and `intensity`, repeating every `interval`.
        phase = (t % interval) / interval  # 0..1
        # Ease in/out using a sine curve for a smoother pulse than a linear triangle
        pulse = (np.sin(phase * np.pi)) ** 2
        return 1.0 + (intensity - 1.0) * pulse

    return clip.fl_time(lambda t: t).resize(lambda t: zoom_factor(t))


# ==============================================================================
# CAPTIONS (word-by-word, Hormozi-style)
# ==============================================================================

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    font_path = Path(settings.caption_font_path)
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    logger.warning(
        "Caption font not found at '%s' — falling back to default font. "
        "Add a .ttf file at that path for proper styling (see README).",
        font_path,
    )
    return ImageFont.load_default()


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (r, g, b, alpha)


def _render_caption_frame(
    word: str, is_emphasis: bool, canvas_size: tuple[int, int], scale: float
) -> Image.Image:
    """Render a single word as an RGBA image with stroke outline, used as a moving ImageClip frame."""
    font = _load_font(int(settings.caption_font_size * scale))
    color = _hex_to_rgba(
        settings.caption_highlight_color if is_emphasis else settings.caption_color
    )
    stroke_color = _hex_to_rgba(settings.caption_stroke_color)

    # Measure text
    dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), word, font=font, stroke_width=settings.caption_stroke_width)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding = 20
    img = Image.new("RGBA", (text_w + padding * 2, text_h + padding * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text(
        (padding - bbox[0], padding - bbox[1]),
        word,
        font=font,
        fill=color,
        stroke_width=settings.caption_stroke_width,
        stroke_fill=stroke_color,
    )
    return img


def _caption_vertical_position(out_h: int) -> str | tuple:
    mapping = {
        "top": ("center", int(out_h * 0.12)),
        "center": "center",
        "center_bottom": ("center", int(out_h * 0.62)),
        "bottom": ("center", int(out_h * 0.85)),
    }
    return mapping.get(settings.caption_position, mapping["center_bottom"])


def build_caption_clips(
    words: list[Word], emphasis_words: list[str], clip_duration: float
) -> list:
    """
    Build a list of MoviePy ImageClips, one per word, timed to appear/disappear
    in sync with the transcript, with a pop-in/bounce/fade animation and
    highlight coloring for emphasis words.
    """
    out_w, out_h = settings.output_width, settings.output_height
    position = _caption_vertical_position(out_h)
    emphasis_lower = {w.lower().strip(".,!?") for w in emphasis_words}

    clips = []
    for word in words:
        start = max(0.0, word.start)
        end = min(clip_duration, word.end + 0.05)  # tiny linger for readability
        if end <= start:
            continue

        is_emphasis = word.text.lower().strip(".,!?") in emphasis_lower
        word_img = _render_caption_frame(word.text, is_emphasis, (out_w, out_h), scale=1.0)

        base_clip = ImageClip(np.array(word_img)).set_start(start).set_duration(end - start)
        base_clip = base_clip.set_position(position)

        animation = settings.caption_animation
        word_duration = end - start
        pop_time = min(0.12, word_duration * 0.4)

        if animation == "pop":
            def scaler(t, _pop=pop_time):
                if t < _pop:
                    # ease-out pop from 60% -> 105% -> 100%
                    progress = t / _pop
                    return 0.6 + 0.5 * progress
                return 1.0
            base_clip = base_clip.resize(scaler)
        elif animation == "bounce":
            def scaler(t, _pop=pop_time):
                if t < _pop:
                    progress = t / _pop
                    return 0.7 + 0.4 * np.sin(progress * np.pi / 2)
                return 1.0
            base_clip = base_clip.resize(scaler)
        elif animation == "fade":
            base_clip = base_clip.crossfadein(min(0.15, word_duration * 0.5))

        clips.append(base_clip)

    return clips


# ==============================================================================
# SFX
# ==============================================================================

def build_sfx_audio_clips(highlight: Highlight, clip_duration: float, transcript_words: list[Word]) -> list:
    """
    Build AudioFileClips for each SFX cue suggested by the LLM, positioned at
    the timestamp of the referenced transcript word (time relative to the
    start of the extracted clip).
    """
    if not settings.enable_sfx or not highlight.sfx_cues:
        return []

    sfx_clips = []
    word_lookup = {w.text.lower().strip(".,!?"): w for w in transcript_words}

    for cue in highlight.sfx_cues:
        cue_name = cue.get("cue")
        filename = SFX_CUE_MAP.get(cue_name)
        if not filename:
            continue
        sfx_path = Path(settings.sfx_dir) / filename
        if not sfx_path.exists():
            logger.warning(
                "SFX asset missing for cue '%s': expected file at %s — skipping. "
                "See README 'Sound Asset Setup' section.",
                cue_name, sfx_path,
            )
            continue

        target_word = str(cue.get("word", "")).lower().strip(".,!?")
        matched = word_lookup.get(target_word)
        cue_time = (matched.start - highlight.start_time) if matched else 0.0
        cue_time = max(0.0, min(cue_time, clip_duration - 0.05))

        try:
            sfx_clip = AudioFileClip(str(sfx_path)).set_start(cue_time)
            sfx_clips.append(sfx_clip)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load SFX asset '%s': %s", sfx_path, exc)

    return sfx_clips


# ==============================================================================
# BACKGROUND MUSIC WITH DUCKING
# ==============================================================================

def build_background_music_track(clip_duration: float, speech_audio: AudioFileClip | None):
    """
    Select a random track from MUSIC_DIR, loop/trim it to clip_duration, and
    apply simple volume ducking: lower music volume, with an extra dip
    whenever speech is present (approximated via overall reduced level under
    dialogue — a full sidechain compressor is out of scope for MoviePy, so we
    apply a flat, conservative duck level defined by MUSIC_VOLUME_DB).
    """
    if not settings.enable_background_music:
        return None

    music_dir = Path(settings.music_dir)
    tracks = list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav"))
    if not tracks:
        logger.warning(
            "ENABLE_BACKGROUND_MUSIC is true but no tracks found in %s — skipping music. "
            "Add royalty-free .mp3/.wav files there (see README).",
            music_dir,
        )
        return None

    track_path = random.choice(tracks)
    try:
        music = AudioFileClip(str(track_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load background music '%s': %s", track_path, exc)
        return None

    # Loop the track if shorter than the clip, trim if longer
    if music.duration < clip_duration:
        loops_needed = int(np.ceil(clip_duration / music.duration))
        from moviepy.audio.AudioClip import concatenate_audioclips
        music = concatenate_audioclips([music] * loops_needed)
    music = music.subclip(0, clip_duration)

    # Convert target dB to a linear gain multiplier and apply
    gain = 10 ** (settings.music_volume_db / 20)
    music = volumex(music, gain)

    logger.info("Using background music track: %s (ducked to %.1f dB)", track_path.name, settings.music_volume_db)
    return music


# ==============================================================================
# MAIN RENDER PIPELINE
# ==============================================================================

def render_short(
    video_path: str,
    highlight: Highlight,
    transcript: Transcript,
    crop_plan: CropPlan,
    output_path: str,
) -> str:
    """
    Renders a single finished vertical short from the source video + analysis
    artifacts, writing the final MP4 to `output_path`.
    """
    logger.info("Rendering short: '%s' [%.1fs - %.1fs]", highlight.title, highlight.start_time, highlight.end_time)

    source = VideoFileClip(video_path).subclip(highlight.start_time, highlight.end_time)
    clip_duration = source.duration

    # 1. Speaker-tracked crop to 9:16
    cropped = apply_speaker_tracked_crop(source, crop_plan)

    # 2. Retention zoom pulses
    cropped = apply_zoom_pulses(cropped)
    cropped = cropped.set_duration(clip_duration)

    # 3. Word-by-word captions
    clip_words = [
        Word(text=w.text, start=w.start - highlight.start_time, end=w.end - highlight.start_time, confidence=w.confidence)
        for w in transcript.words_in_range(highlight.start_time, highlight.end_time)
    ]
    caption_clips = build_caption_clips(clip_words, highlight.emphasis_words, clip_duration)

    composite = CompositeVideoClip([cropped, *caption_clips], size=(settings.output_width, settings.output_height))
    composite = composite.set_duration(clip_duration)

    # 4. Audio: original speech + SFX + ducked background music
    speech_audio = source.audio
    sfx_clips = build_sfx_audio_clips(highlight, clip_duration, clip_words)
    music_clip = build_background_music_track(clip_duration, speech_audio)

    audio_tracks = [t for t in [speech_audio, *sfx_clips, music_clip] if t is not None]
    final_audio = CompositeAudioClip(audio_tracks) if audio_tracks else None
    if final_audio is not None:
        composite = composite.set_audio(final_audio)

    # 5. Export
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    composite.write_videofile(
        output_path,
        fps=settings.output_fps,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=4,
        logger=None,  # suppress moviepy's own progress bar; we log ourselves
    )

    # Explicitly close clips to release file handles / memory
    for c in (source, cropped, composite, speech_audio, music_clip, *sfx_clips):
        try:
            if c is not None:
                c.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass

    logger.info("Short rendered successfully -> %s", output_path)
    return output_path
