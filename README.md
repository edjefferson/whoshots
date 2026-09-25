# subshots

Take a screenshot of every subtitle in a video: the frame at the midpoint of
each cue's on-screen time, with the subtitle burned in.

Requires `ffmpeg`/`ffprobe` on your PATH.

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python subshots.py episode.mkv --list-tracks     # see embedded subtitle tracks
.venv/bin/python subshots.py episode.mkv                   # track 0 -> episode_shots/
.venv/bin/python subshots.py episode.mkv -t 1 -o shots -f png --max-height 0
.venv/bin/python subshots.py episode.mp4 -s episode.srt    # external subtitle file
.venv/bin/python subshots.py dvdrip.mkv --deinterlace
```

Screenshots are JPEGs named `NNNN_HH-MM-SS.mmm.jpg` (cue number and frame time),
scaled down to at most 576px tall (DVD height) (`--max-height`, `-f png` for lossless).

Rather than the exact midpoint of each cue, the sharpest frame within ±0.4s of
it is used (`-w/--window`, `-w 0` for the exact midpoint), which avoids a lot of
motion blur.

- **Text subs** (SRT, ASS, WebVTT, mov_text…) are rendered with ffmpeg's libass
  `subtitles` filter if available (keeps ASS styling and embedded fonts),
  otherwise drawn with Pillow (`--font`, `--font-scale`).
- **Bitmap subs** (DVD, PGS/Blu-ray, DVB) are composited with ffmpeg's `overlay`.
- Anamorphic sources are scaled to their display aspect ratio.

## Episode catalogue

`catalogue.py` builds `episodes.csv`: one row per classic episode (from TVmaze),
matched to the best file in a library laid out as `Season N/<serial>/…`.

```sh
.venv/bin/python catalogue.py [location]"
```

Preference order: original versions with original effects / TV editions, then
updated effects, special editions, extended cuts and so on; for missing
episodes, reconstructions over animations. Extras folders are ignored. Each
row records whether the file has dialogue subtitles and which track to pass
to `subshots.py -t`. Files that couldn't be matched are listed on stderr.
ffprobe results are cached in `.cache/`, so re-runs are quick.
