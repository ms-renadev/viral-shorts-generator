# Viral Shorts Generator

Turn long-form YouTube videos (or local MP4s) into engaging, vertical (9:16)
YouTube Shorts — automatically. The pipeline downloads the source, transcribes
it word-by-word, asks an LLM to pick the highest-potential viral moments,
tracks the speaker's face to build a smart vertical crop, overlays animated
word-by-word captions, layers in SFX + zoom pulses + ducked background music,
and renders finished MP4s ready to upload.

```
YouTube URL / MP4
      │
      ▼
 [1] downloader.py     → source MP4 on disk
      │
      ▼
 [2] transcriber.py    → word-level timestamped transcript (Whisper/Deepgram)
      │
      ▼
 [3] highlight_finder.py → LLM picks top N clips (start/end/score/hooks/cues)
      │
      ▼
 [4] cropper.py        → MediaPipe face tracking → smoothed 9:16 crop path
      │
      ▼
 [5] editor.py          → crop + captions + SFX + zoom + music → final MP4
```

---

## 1. Project Structure

```
viral-shorts-generator/
├── main.py                    # pipeline orchestrator (entry point)
├── config.py                  # typed settings loader (.env-driven)
├── .env.example                # copy to .env and fill in
├── requirements.txt
├── README.md                   # this file
├── modules/
│   ├── downloader.py           # yt-dlp video acquisition
│   ├── transcriber.py          # Whisper / Deepgram word-level transcription
│   ├── highlight_finder.py     # GPT-4o / Claude highlight detection
│   ├── cropper.py               # MediaPipe face tracking + crop path
│   ├── editor.py                 # MoviePy/ffmpeg rendering engine
│   └── utils.py                  # shared helpers
├── assets/
│   ├── fonts/                    # caption font(s) — see assets/fonts/README.md
│   ├── sfx/                       # SFX stings — see assets/sfx/README.md
│   └── music/                     # background music — see assets/music/README.md
├── temp/                          # scratch space (audio extracts, downloads)
└── output/                        # final rendered Shorts land here
```

---

## 2. Prerequisites

### 2.1 Python

Python **3.10 or 3.11** is recommended (MediaPipe wheels lag behind the very
latest Python releases).

```bash
python3 --version
```

### 2.2 FFmpeg (required — used by MoviePy, yt-dlp, and transcription)

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**
1. Download a build from https://www.gyan.dev/ffmpeg/builds/
2. Extract it and add the `bin/` folder to your system `PATH`.
3. Verify with:
```bash
ffmpeg -version
```

### 2.3 (Optional) GPU for faster Whisper transcription

If you have an NVIDIA GPU + CUDA installed, install a CUDA-enabled PyTorch
build before `openai-whisper` for a large speed boost. CPU-only works fine
for shorter videos, just slower. See https://pytorch.org/get-started/locally/
for the correct install command for your system.

---

## 3. Installation

```bash
# 1. Clone / unzip the project, then cd into it
cd viral-shorts-generator

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `openai-whisper` will download model weights on first run
> (e.g. the `base` model is ~140MB). Ensure you have an internet connection
> the first time you execute the pipeline with `TRANSCRIBER_BACKEND=whisper`.

---

## 4. Environment Setup

```bash
cp .env.example .env
```

Open `.env` in your editor and fill in, at minimum:

| Variable | Required when | Notes |
|---|---|---|
| `LLM_PROVIDER` | always | `openai` or `anthropic` |
| `OPENAI_API_KEY` | `LLM_PROVIDER=openai` | from platform.openai.com |
| `ANTHROPIC_API_KEY` | `LLM_PROVIDER=anthropic` | from console.anthropic.com |
| `TRANSCRIBER_BACKEND` | always | `whisper` (local/free) or `deepgram` (cloud/paid) |
| `DEEPGRAM_API_KEY` | `TRANSCRIBER_BACKEND=deepgram` | from console.deepgram.com |

Everything else in `.env.example` has a sensible default — tune crop
smoothing, caption styling, zoom intensity, and music ducking to taste.

---

## 5. Sound & Font Asset Setup

The pipeline references named SFX cues and font/music files that **you must
supply** (licensing reasons — no bundled copyrighted assets):

1. **Captions font** → drop a bold `.ttf` at `assets/fonts/` and point
   `CAPTION_FONT_PATH` in `.env` at it. See `assets/fonts/README.md`.
2. **SFX stings** → drop files named exactly as listed in
   `assets/sfx/README.md` (e.g. `swoosh.mp3`, `pop.mp3`, `vine_boom.mp3`...).
3. **Background music** → drop one or more `.mp3`/`.wav` tracks in
   `assets/music/`. See `assets/music/README.md`.

If any of these are missing, the pipeline **does not crash** — it logs a
warning and skips that enhancement (default font is used, SFX cue is
skipped, no music track is added).

---

## 6. Running the Pipeline

### Basic usage — YouTube URL

```bash
python main.py --source "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Local MP4 file

```bash
python main.py --source "/path/to/podcast_episode.mp4"
```

### Useful flags

```bash
# Only pull the top 3 highlights instead of the .env default
python main.py --source "<url>" --num-highlights 3

# Raise the quality bar — only accept clips scored 75+
python main.py --source "<url>" --min-score 75

# Preview highlight picks WITHOUT rendering video (fast, cheap — good for
# reviewing the LLM's picks before spending time on rendering)
python main.py --source "<url>" --skip-render

# Keep temp/ working files around after the run (debugging)
python main.py --source "<url>" --keep-temp

# Send finished Shorts to a custom folder
python main.py --source "<url>" --output-dir "./my_shorts"
```

### What you get

```
output/<video_title>/
├── highlights_manifest.json          # every picked clip's metadata
├── 01_<title>_score87.mp4
├── 02_<title>_score79.mp4
└── 03_<title>_score74.mp4
```

`highlights_manifest.json` contains the LLM's reasoning (hook summary,
emphasis words, SFX cues, mood, viral score) for every selected clip — handy
for manually reviewing before you post, or for building an approval UI on
top of this pipeline later.

---

## 7. How Each Stage Works

1. **`downloader.py`** — Detects whether `--source` is a YouTube URL or local
   path. YouTube videos are pulled via `yt-dlp` at up to 1080p, merged to a
   single MP4. Local files are validated and duration-probed via `ffprobe`.

2. **`transcriber.py`** — Extracts a 16kHz mono WAV track via ffmpeg, then
   transcribes it with word-level timestamps using either local Whisper
   (`openai-whisper`) or the Deepgram cloud API — selected via
   `TRANSCRIBER_BACKEND`.

3. **`highlight_finder.py`** — Feeds the full timestamped transcript to
   GPT-4o or Claude with a strict system prompt describing viral-short
   selection criteria (hook strength, self-containment, payoff, natural
   sentence boundaries). The LLM returns strict JSON; the module snaps
   fuzzy timestamps to real word boundaries, filters by `MIN_VIRAL_SCORE`,
   and removes overlapping picks.

4. **`cropper.py`** — Samples frames across each highlight's time range at
   ~5fps and runs MediaPipe Face Detection. If a single dominant face is
   present, it builds a smoothed (moving-average) horizontal/vertical crop
   path that keeps the speaker centered in a 9:16 window. If two
   similarly-sized faces are consistently detected, it switches to an
   automatic top/bottom split-screen layout, tracking each speaker
   independently.

5. **`editor.py`** — The rendering engine:
   - Applies the per-frame dynamic crop (or split-screen composite) and
     resizes to your configured output resolution (default 1080×1920).
   - Applies a subtle sine-eased zoom pulse every 3–5s for retention.
   - Renders word-by-word animated captions (pop / bounce / fade) with
     emphasis-word highlighting in a configurable accent color.
   - Places SFX stings at the transcript-word timestamps the LLM flagged.
   - Loads/loops a background music track and applies volume ducking.
   - Composites everything and exports via MoviePy → ffmpeg (H.264/AAC).

---

## 8. Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `ffmpeg (or the requested binary) was not found on PATH` | Install ffmpeg (§2.2) and confirm `ffmpeg -version` works in your terminal. |
| `OPENAI_API_KEY is not set` / `ANTHROPIC_API_KEY is not set` | Fill in `.env` per §4; make sure `LLM_PROVIDER` matches the key you supplied. |
| No highlights returned / "No highlights met the quality bar" | Lower `MIN_VIRAL_SCORE` (or `--min-score`), or the source video may genuinely lack strong isolated hooks — try a different segment/video. |
| Captions render with a plain/ugly font | You haven't dropped a `.ttf` into `assets/fonts/` — see §5. |
| SFX/music silently missing from output | Check the console warnings — asset files are named/pathed incorrectly, see `assets/sfx/README.md` / `assets/music/README.md`. |
| Whisper transcription is very slow | Use a smaller `WHISPER_MODEL_SIZE` (e.g. `tiny` or `base`), or install a CUDA-enabled PyTorch build for GPU acceleration, or switch to `TRANSCRIBER_BACKEND=deepgram`. |
| `mediapipe` install fails | MediaPipe wheels support a limited set of Python versions/platforms — use Python 3.10/3.11 on a supported OS (Windows/macOS/Linux x86_64; Apple Silicon supported on recent mediapipe releases). |
| Split-screen triggers when it shouldn't (or doesn't trigger when it should) | Tune `ENABLE_SPLIT_SCREEN` and the detection heuristic in `cropper._detect_two_speaker_layout` (face-count/size-ratio thresholds) to match your source footage. |

---

## 9. Extending the Project

- **Chunked transcripts for very long videos**: `highlight_finder._build_transcript_block`
  currently truncates at ~6000 words to control LLM token usage. For 1h+
  videos, split the transcript into overlapping windows, run highlight
  detection per-window, then merge/dedupe results.
- **Auto-upload**: add a `modules/publisher.py` using the YouTube Data API
  (`videos.insert`) to push rendered Shorts directly to a channel.
- **Real sidechain audio ducking**: the current ducking is a flat gain
  reduction; swapping in `pydub`'s RMS-based dynamic processing (already a
  listed dependency) would let you duck music only *while* speech is active.
- **Web UI**: wrap `main.py`'s stages behind a small FastAPI/Flask service
  and a review UI so a human can approve/reject `highlights_manifest.json`
  entries before the render stage runs.

---

## 10. License & Asset Disclaimer

This codebase is provided as-is for you to adapt. It does **not** bundle any
copyrighted fonts, sound effects, or music — you are responsible for sourcing
properly licensed assets (see §5) and for complying with YouTube's Terms of
Service and applicable copyright law when downloading and repurposing video
content.
