"""
modules/cropper.py
==============================================================================
Analyzes a video clip's frames with MediaPipe Face Detection to determine a
smooth, speaker-centered crop path for converting the source aspect ratio
into vertical 9:16. Supports an automatic split-screen layout when exactly
two distinct faces are consistently detected (e.g. podcast / interview
format).

Public API:
    analyze_crop_path(video_path, start_time, end_time) -> CropPlan
==============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import settings
from modules.utils import clamp

logger = logging.getLogger("viral_shorts.cropper")


@dataclass
class CropFrame:
    time: float
    center_x: float   # normalized 0-1, horizontal center of crop box
    center_y: float    # normalized 0-1, vertical center of crop box


@dataclass
class CropPlan:
    mode: str                                  # "single" | "split_screen"
    frames: list[CropFrame] = field(default_factory=list)   # for "single" mode
    speaker_a_frames: list[CropFrame] = field(default_factory=list)  # split mode
    speaker_b_frames: list[CropFrame] = field(default_factory=list)  # split mode
    source_width: int = 0
    source_height: int = 0


def _init_face_detector():
    """Initialize a MediaPipe face detector, raising a clear error if missing."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "mediapipe is not installed. Run `pip install -r requirements.txt`."
        ) from exc
    return mp.solutions.face_detection.FaceDetection(
        model_selection=1,  # 1 = full-range model, better for varied framing
        min_detection_confidence=0.5,
    )


def _sample_face_positions(
    video_path: str, start_time: float, end_time: float, sample_fps: float = 5.0
) -> tuple[list[tuple[float, list[tuple[float, float, float]]]], int, int]:
    """
    Sample frames between start_time/end_time at `sample_fps` and run face
    detection on each. Returns a list of (timestamp, [(cx, cy, area), ...])
    where cx/cy are normalized face-center coordinates and area is the
    normalized bounding-box area (used to pick the "main" speaker).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for face analysis: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_interval = max(1, int(round(src_fps / sample_fps)))
    start_frame = int(start_time * src_fps)
    end_frame = int(end_time * src_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    detector = _init_face_detector()
    samples: list[tuple[float, list[tuple[float, float, float]]]] = []

    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)
            faces = []
            if result.detections:
                for det in result.detections:
                    box = det.location_data.relative_bounding_box
                    cx = clamp(box.xmin + box.width / 2, 0.0, 1.0)
                    cy = clamp(box.ymin + box.height / 2, 0.0, 1.0)
                    area = max(0.0, box.width) * max(0.0, box.height)
                    faces.append((cx, cy, area))
            timestamp = frame_idx / src_fps
            samples.append((timestamp, faces))
        frame_idx += 1

    cap.release()
    detector.close()
    return samples, width, height


def _smooth_positions(positions: list[float], window: int) -> list[float]:
    """Simple moving-average smoothing to remove crop-box jitter."""
    if window <= 1 or len(positions) <= 1:
        return positions
    arr = np.array(positions, dtype=np.float32)
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(positions)].tolist()


def _detect_two_speaker_layout(
    samples: list[tuple[float, list[tuple[float, float, float]]]],
) -> bool:
    """
    Heuristic: if the majority of sampled frames consistently show exactly
    two faces of comparable size, treat this as a two-speaker layout eligible
    for split-screen framing.
    """
    if not settings.enable_split_screen:
        return False

    two_face_frames = 0
    valid_frames = 0
    for _, faces in samples:
        if not faces:
            continue
        valid_frames += 1
        if len(faces) == 2:
            areas = sorted([f[2] for f in faces])
            # Require both faces to be reasonably similar in size (avoid
            # background/incidental faces being mistaken for a co-host)
            if areas[0] > 0 and areas[1] / max(areas[0], 1e-6) < 3.0:
                two_face_frames += 1

    if valid_frames == 0:
        return False
    return (two_face_frames / valid_frames) > 0.6


def _build_single_speaker_path(
    samples: list[tuple[float, list[tuple[float, float, float]]]],
) -> list[CropFrame]:
    """Track the largest (most prominent) face per frame -> crop center path."""
    raw_x: list[float] = []
    raw_y: list[float] = []
    times: list[float] = []
    last_known = (0.5, 0.4)  # default center-upper framing if no face found

    for t, faces in samples:
        if faces:
            main_face = max(faces, key=lambda f: f[2])
            last_known = (main_face[0], main_face[1])
        times.append(t)
        raw_x.append(last_known[0])
        raw_y.append(last_known[1])

    smooth_x = _smooth_positions(raw_x, settings.crop_smoothing_window)
    smooth_y = _smooth_positions(raw_y, settings.crop_smoothing_window)

    return [
        CropFrame(time=t, center_x=x, center_y=y)
        for t, x, y in zip(times, smooth_x, smooth_y)
    ]


def _build_split_screen_paths(
    samples: list[tuple[float, list[tuple[float, float, float]]]],
) -> tuple[list[CropFrame], list[CropFrame]]:
    """
    Track two speakers separately by consistently assigning detections to
    "left" and "right" identity slots based on horizontal position, so each
    speaker gets a stable half of the split screen.
    """
    a_x, a_y, b_x, b_y, times = [], [], [], [], []
    last_a, last_b = (0.25, 0.4), (0.75, 0.4)

    for t, faces in samples:
        if len(faces) >= 2:
            sorted_faces = sorted(faces, key=lambda f: f[0])
            left, right = sorted_faces[0], sorted_faces[-1]
            last_a, last_b = (left[0], left[1]), (right[0], right[1])
        elif len(faces) == 1:
            # Assign the lone detected face to whichever slot it's closer to
            f = faces[0]
            if abs(f[0] - last_a[0]) <= abs(f[0] - last_b[0]):
                last_a = (f[0], f[1])
            else:
                last_b = (f[0], f[1])

        times.append(t)
        a_x.append(last_a[0]); a_y.append(last_a[1])
        b_x.append(last_b[0]); b_y.append(last_b[1])

    window = settings.crop_smoothing_window
    a_x, a_y = _smooth_positions(a_x, window), _smooth_positions(a_y, window)
    b_x, b_y = _smooth_positions(b_x, window), _smooth_positions(b_y, window)

    a_frames = [CropFrame(t, x, y) for t, x, y in zip(times, a_x, a_y)]
    b_frames = [CropFrame(t, x, y) for t, x, y in zip(times, b_x, b_y)]
    return a_frames, b_frames


def analyze_crop_path(video_path: str, start_time: float, end_time: float) -> CropPlan:
    """
    Main entry point: samples faces across the given time range and builds
    a smoothed crop plan (single-speaker centered, or split-screen for two).
    """
    logger.info("Analyzing speaker positions for crop [%.1fs - %.1fs]...", start_time, end_time)
    samples, width, height = _sample_face_positions(video_path, start_time, end_time)

    if _detect_two_speaker_layout(samples):
        logger.info("Two-speaker layout detected -> using split-screen framing.")
        a_frames, b_frames = _build_split_screen_paths(samples)
        return CropPlan(
            mode="split_screen",
            speaker_a_frames=a_frames,
            speaker_b_frames=b_frames,
            source_width=width,
            source_height=height,
        )

    frames = _build_single_speaker_path(samples)
    return CropPlan(mode="single", frames=frames, source_width=width, source_height=height)
