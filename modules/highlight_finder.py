"""
modules/highlight_finder.py
==============================================================================
Sends the full timestamped transcript to an LLM (OpenAI GPT-4o or Anthropic
Claude) and asks it to identify the top N most viral-worthy segments.

The LLM must return, for each candidate segment:
    - start_time / end_time  (seconds, aligned to actual spoken words)
    - viral_score (1-100)
    - hook_summary (why this moment grabs attention)
    - title (short punchy caption/title suggestion for the Short)
    - styling_cues (dict): emphasis_words, sfx_cues, suggested_zoom, mood

Public API:
    find_highlights(transcript: Transcript) -> list[Highlight]
==============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings, SFX_CUE_MAP
from modules.transcriber import Transcript
from modules.utils import extract_json_block, clamp

logger = logging.getLogger("viral_shorts.highlight_finder")


@dataclass
class Highlight:
    start_time: float
    end_time: float
    viral_score: int
    hook_summary: str
    title: str
    emphasis_words: list[str] = field(default_factory=list)
    sfx_cues: list[dict[str, Any]] = field(default_factory=list)   # [{word/time, cue}]
    suggested_zoom: str = "subtle"          # "subtle" | "punchy" | "none"
    mood: str = "energetic"

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


SYSTEM_PROMPT = """You are a viral short-form video strategist who has produced hundreds of \
YouTube Shorts / TikToks / Reels that generated millions of views. You analyze long-form \
video transcripts and identify the exact moments with the highest potential to go viral as \
a standalone vertical short clip.

Selection criteria, in priority order:
1. Strong HOOK potential in the first 1-2 seconds of the clip (question, bold claim, shocking
   statement, pattern interrupt, controversy, emotional peak).
2. Self-contained: the clip must make sense WITHOUT the surrounding context.
3. Clear payoff, punchline, insight, or emotional resolution before the clip ends.
4. Natural spoken-word boundaries only — never cut a clip mid-sentence or mid-thought.
5. High information density / emotional intensity / humor / controversy / relatability.

You must respond with ONLY a valid JSON array (no markdown fences, no commentary) matching \
exactly this schema:

[
  {
    "start_time": <float, seconds>,
    "end_time": <float, seconds>,
    "viral_score": <int, 1-100>,
    "title": "<short punchy 3-8 word title for the Short>",
    "hook_summary": "<1-2 sentences: why this moment is engaging and what hooks the viewer>",
    "emphasis_words": ["<word1>", "<word2>", ...],
    "sfx_cues": [{"word": "<word from transcript>", "cue": "<one of: swoosh, pop, vine_boom, ding, whoosh_transition, impact, notification>"}],
    "suggested_zoom": "<one of: subtle, punchy, none>",
    "mood": "<one of: energetic, dramatic, funny, inspirational, controversial, calm>"
  }
]

Rules:
- Only choose start_time/end_time values that land on actual word boundaries present in the transcript.
- Each clip duration MUST be between {min_dur} and {max_dur} seconds.
- Return between 1 and {num_highlights} candidates, ranked by viral_score descending.
- Do not overlap clips.
- emphasis_words must be words that literally appear in the selected transcript range — these \
will be highlighted in bright color in the caption overlay.
- sfx_cues should be sparse and purposeful (typically 1-3 per clip) — only at genuine emphasis \
or transition points, using ONLY the allowed cue names listed above.
"""

USER_PROMPT_TEMPLATE = """Here is the full timestamped transcript (format: [start-end] word):

{transcript_block}

Identify the best {num_highlights} viral short-form clips from this transcript. \
Respond with ONLY the JSON array described in your instructions."""


def _build_transcript_block(transcript: Transcript, max_words: int = 6000) -> str:
    """
    Render the transcript as compact timestamped lines for the LLM prompt.
    Truncates very long transcripts to control token usage — for videos
    longer than ~45-60 min, consider chunking (see README for guidance).
    """
    words = transcript.words[:max_words]
    lines = []
    # Group into ~12-word chunks with a timestamp range prefix for readability
    chunk: list[str] = []
    chunk_start = None
    for w in words:
        if chunk_start is None:
            chunk_start = w.start
        chunk.append(w.text)
        if len(chunk) >= 12:
            lines.append(f"[{chunk_start:.2f}-{w.end:.2f}] {' '.join(chunk)}")
            chunk = []
            chunk_start = None
    if chunk:
        lines.append(f"[{chunk_start:.2f}-{words[-1].end:.2f}] {' '.join(chunk)}")
    return "\n".join(lines)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _call_openai(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in your .env file.")

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        response_format={"type": "json_object"} if "gpt-4" in settings.openai_model else None,
    )
    return response.choices[0].message.content or ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _call_anthropic(system_prompt: str, user_prompt: str) -> str:
    import anthropic

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in your .env file.")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.4,
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    if settings.llm_provider == "anthropic":
        logger.info("Requesting highlight analysis from Anthropic (%s)...", settings.anthropic_model)
        return _call_anthropic(system_prompt, user_prompt)
    logger.info("Requesting highlight analysis from OpenAI (%s)...", settings.openai_model)
    return _call_openai(system_prompt, user_prompt)


def _snap_to_word_boundary(transcript: Transcript, t: float, prefer_start: bool) -> float:
    """Snap an arbitrary LLM-provided timestamp to the nearest real word boundary."""
    if not transcript.words:
        return t
    best = min(transcript.words, key=lambda w: abs((w.start if prefer_start else w.end) - t))
    return best.start if prefer_start else best.end


def _parse_llm_response(raw: str, transcript: Transcript) -> list[Highlight]:
    data = extract_json_block(raw)
    # Some models may wrap the array in {"highlights": [...]}
    if isinstance(data, dict):
        data = data.get("highlights") or data.get("clips") or next(iter(data.values()))

    if not isinstance(data, list):
        raise ValueError("LLM response did not contain a JSON array of highlights.")

    highlights: list[Highlight] = []
    for item in data:
        try:
            start = _snap_to_word_boundary(transcript, float(item["start_time"]), prefer_start=True)
            end = _snap_to_word_boundary(transcript, float(item["end_time"]), prefer_start=False)
            if end <= start:
                continue

            score = int(clamp(float(item.get("viral_score", 50)), 1, 100))

            sfx_cues = []
            for cue in item.get("sfx_cues", []) or []:
                cue_name = cue.get("cue")
                if cue_name in SFX_CUE_MAP:
                    sfx_cues.append(cue)
                else:
                    logger.debug("Dropping unknown SFX cue '%s' suggested by LLM.", cue_name)

            highlights.append(
                Highlight(
                    start_time=start,
                    end_time=end,
                    viral_score=score,
                    hook_summary=str(item.get("hook_summary", "")).strip(),
                    title=str(item.get("title", "Untitled Clip")).strip(),
                    emphasis_words=[str(w) for w in item.get("emphasis_words", [])],
                    sfx_cues=sfx_cues,
                    suggested_zoom=item.get("suggested_zoom", "subtle"),
                    mood=item.get("mood", "energetic"),
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed highlight candidate from LLM: %s (%s)", item, exc)

    return highlights


def _dedupe_overlaps(highlights: list[Highlight]) -> list[Highlight]:
    """Greedily remove overlapping clips, keeping the higher-scoring one."""
    ordered = sorted(highlights, key=lambda h: h.viral_score, reverse=True)
    accepted: list[Highlight] = []
    for h in ordered:
        overlaps = any(not (h.end_time <= a.start_time or h.start_time >= a.end_time) for a in accepted)
        if not overlaps:
            accepted.append(h)
    return sorted(accepted, key=lambda h: h.start_time)


def find_highlights(transcript: Transcript) -> list[Highlight]:
    """
    Main entry point: analyzes the transcript with an LLM and returns a
    filtered, deduplicated, score-sorted list of Highlight candidates.
    """
    if not transcript.words:
        raise ValueError("Transcript is empty — cannot find highlights.")

    system_prompt = SYSTEM_PROMPT.format(
        min_dur=settings.min_clip_duration,
        max_dur=settings.max_clip_duration,
        num_highlights=settings.num_highlights,
    )
    transcript_block = _build_transcript_block(transcript)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        transcript_block=transcript_block,
        num_highlights=settings.num_highlights,
    )

    raw_response = _call_llm(system_prompt, user_prompt)
    highlights = _parse_llm_response(raw_response, transcript)

    # Filter by min score, duration bounds, and remove overlaps
    filtered = [
        h for h in highlights
        if h.viral_score >= settings.min_viral_score
        and settings.min_clip_duration - 3 <= h.duration <= settings.max_clip_duration + 3
    ]
    deduped = _dedupe_overlaps(filtered)[: settings.num_highlights]

    if not deduped:
        logger.warning(
            "No highlights met the MIN_VIRAL_SCORE=%d threshold. "
            "Consider lowering it in .env, or the source video may lack strong hooks.",
            settings.min_viral_score,
        )

    for h in deduped:
        logger.info(
            "Highlight found: [%.1fs - %.1fs] score=%d title='%s'",
            h.start_time, h.end_time, h.viral_score, h.title,
        )

    return deduped
