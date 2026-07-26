#!/usr/bin/env python3
"""
main.py
==============================================================================
Viral Shorts Generator — Pipeline Orchestrator

Usage:
    python main.py --source "https://www.youtube.com/watch?v=XXXXXXXXXXX"
    python main.py --source "/path/to/local_video.mp4"
    python main.py --source "<url>" --num-highlights 3 --keep-temp

Run `python main.py --help` for all options.
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from config import settings, setup_logging
from modules import downloader, transcriber, highlight_finder, cropper, editor
from modules.utils import sanitize_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn a long-form YouTube video (or local MP4) into viral vertical Shorts."
    )
    parser.add_argument(
        "--source", required=True,
        help="YouTube URL or path to a local video file.",
    )
    parser.add_argument(
        "--num-highlights", type=int, default=None,
        help="Override NUM_HIGHLIGHTS_TO_EXTRACT from .env for this run.",
    )
    parser.add_argument(
        "--min-score", type=int, default=None,
        help="Override MIN_VIRAL_SCORE from .env for this run.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override OUTPUT_DIR from .env for this run.",
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Do not delete the temp/ working directory contents after the run.",
    )
    parser.add_argument(
        "--skip-render", action="store_true",
        help="Run transcription + highlight detection only; save highlights.json and exit "
             "without rendering video (useful for reviewing picks before spending render time).",
    )
    return parser.parse_args()


def save_highlights_manifest(highlights, transcript, output_dir: str) -> str:
    """Persist the LLM's highlight picks + metadata to a JSON file for review/debugging."""
    manifest_path = str(Path(output_dir) / "highlights_manifest.json")
    data = [
        {
            "title": h.title,
            "start_time": h.start_time,
            "end_time": h.end_time,
            "duration": round(h.duration, 2),
            "viral_score": h.viral_score,
            "hook_summary": h.hook_summary,
            "emphasis_words": h.emphasis_words,
            "sfx_cues": h.sfx_cues,
            "suggested_zoom": h.suggested_zoom,
            "mood": h.mood,
            "transcript_excerpt": transcript.text_in_range(h.start_time, h.end_time),
        }
        for h in highlights
    ]
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return manifest_path


def main() -> int:
    args = parse_args()
    logger = setup_logging()

    # Apply CLI overrides on top of .env settings
    if args.num_highlights is not None:
        settings.num_highlights = args.num_highlights
    if args.min_score is not None:
        settings.min_viral_score = args.min_score
    if args.output_dir is not None:
        settings.output_dir = args.output_dir

    problems = settings.validate()
    if problems:
        logger.error("Configuration problems detected:")
        for p in problems:
            logger.error("  - %s", p)
        logger.error("Fix your .env file (see .env.example) and try again.")
        return 1

    settings.ensure_dirs()

    start_time = time.time()
    logger.info("=" * 70)
    logger.info("VIRAL SHORTS GENERATOR — pipeline starting")
    logger.info("Source: %s", args.source)
    logger.info("LLM provider: %s | Transcriber: %s", settings.llm_provider, settings.transcriber_backend)
    logger.info("=" * 70)

    try:
        # ----------------------------------------------------------------
        # STAGE 1: Acquire source video
        # ----------------------------------------------------------------
        logger.info("[1/5] Resolving input source...")
        source_result = downloader.resolve_input_source(args.source, settings.temp_dir)

        # ----------------------------------------------------------------
        # STAGE 2: Transcribe with word-level timestamps
        # ----------------------------------------------------------------
        logger.info("[2/5] Transcribing audio...")
        transcript = transcriber.transcribe(source_result.video_path)
        if not transcript.words:
            logger.error("Transcription returned no words. Aborting — cannot proceed without a transcript.")
            return 1

        # ----------------------------------------------------------------
        # STAGE 3: LLM highlight detection
        # ----------------------------------------------------------------
        logger.info("[3/5] Identifying viral highlights via LLM...")
        highlights = highlight_finder.find_highlights(transcript)
        if not highlights:
            logger.error(
                "No highlights met the quality bar (MIN_VIRAL_SCORE=%d). "
                "Try lowering --min-score or check the source video's content.",
                settings.min_viral_score,
            )
            return 1

        video_output_dir = str(Path(settings.output_dir) / sanitize_filename(source_result.title))
        manifest_path = save_highlights_manifest(highlights, transcript, video_output_dir)
        logger.info("Saved highlight manifest -> %s", manifest_path)

        if args.skip_render:
            logger.info("--skip-render set: stopping after highlight detection.")
            _print_summary(highlights, video_output_dir, elapsed=time.time() - start_time)
            return 0

        # ----------------------------------------------------------------
        # STAGE 4 + 5: Per-highlight crop analysis + render
        # ----------------------------------------------------------------
        rendered_paths = []
        for idx, highlight in enumerate(highlights, start=1):
            logger.info(
                "[4-5/5] Processing highlight %d/%d: '%s' (score %d)",
                idx, len(highlights), highlight.title, highlight.viral_score,
            )
            try:
                crop_plan = cropper.analyze_crop_path(
                    source_result.video_path, highlight.start_time, highlight.end_time
                )

                filename = f"{idx:02d}_{sanitize_filename(highlight.title)}_score{highlight.viral_score}.mp4"
                output_path = str(Path(video_output_dir) / filename)

                editor.render_short(
                    video_path=source_result.video_path,
                    highlight=highlight,
                    transcript=transcript,
                    crop_plan=crop_plan,
                    output_path=output_path,
                )
                rendered_paths.append(output_path)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to render highlight %d ('%s'): %s", idx, highlight.title, exc)
                logger.debug(traceback.format_exc())
                continue

        if not args.keep_temp:
            _cleanup_temp(settings.temp_dir, logger)

        elapsed = time.time() - start_time
        _print_summary(highlights, video_output_dir, elapsed, rendered_paths)
        return 0 if rendered_paths else 1

    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline failed: %s", exc)
        logger.debug(traceback.format_exc())
        return 1


def _cleanup_temp(temp_dir: str, logger) -> None:
    import shutil

    try:
        for item in Path(temp_dir).iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        logger.debug("Cleaned up temp directory: %s", temp_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fully clean temp directory: %s", exc)


def _print_summary(highlights, output_dir: str, elapsed: float, rendered_paths: list[str] | None = None) -> None:
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Highlights identified : {len(highlights)}")
    if rendered_paths is not None:
        print(f"Shorts rendered       : {len(rendered_paths)}")
    print(f"Output directory       : {output_dir}")
    print(f"Total time elapsed     : {elapsed:.1f}s")
    print("-" * 70)
    for h in highlights:
        print(f"  [{h.viral_score:3d}] {h.title}  ({h.duration:.1f}s)  — {h.hook_summary}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    sys.exit(main())
