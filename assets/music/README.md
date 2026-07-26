# Background Music Assets

Drop royalty-free `.mp3` or `.wav` background tracks in this folder. At
render time, `modules/editor.py` picks one at random per short, loops or
trims it to the clip length, and applies gain reduction per `MUSIC_VOLUME_DB`
in your `.env` so it sits under the speaker's voice.

Recommended sources for royalty-free background music:
- YouTube Audio Library (studio.youtube.com -> Audio Library)
- Pixabay Music (pixabay.com/music)
- Free Music Archive (freemusicarchive.org, check individual licenses)

Tips:
- Prefer instrumental tracks with minimal dynamic range (lyrics compete with
  captions/speech for attention).
- Normalize tracks to a similar loudness before dropping them here so the
  ducking level in `.env` behaves predictably across different tracks.
