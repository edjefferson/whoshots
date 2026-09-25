# subshots

Take a screenshot of every subtitle in a video: the frame at the midpoint of
each cue's on-screen time, with the subtitle burned in.

Requires `ffmpeg`/`ffprobe` on your PATH.

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python subshots.py episode.mkv --list-tracks     # see embedded subtitle tracks
.venv/bin/python subshots.py episode.mkv                   # track 0 -> episode_shots/
.venv/bin/python subshots.py episode.mkv -t 1 -o shots -f jpg
.venv/bin/python subshots.py episode.mp4 -s episode.srt    # external subtitle file
.venv/bin/python subshots.py dvdrip.mkv --deinterlace
```

Screenshots are named `NNNN_HH-MM-SS.mmm.png` (cue number and timestamp).

- **Text subs** (SRT, ASS, WebVTT, mov_text…) are rendered with ffmpeg's libass
  `subtitles` filter if available (keeps ASS styling and embedded fonts),
  otherwise drawn with Pillow (`--font`, `--font-scale`).
- **Bitmap subs** (DVD, PGS/Blu-ray, DVB) are composited with ffmpeg's `overlay`.
- Anamorphic sources are scaled to their display aspect ratio.
