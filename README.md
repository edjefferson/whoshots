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

## Batch screenshots of the classic series

`batch_shots.py` runs `subshots.py --save-subs` over the episodes in
`episodes.csv`, into `output/Doctor Who (1963–1996)/Season NN/NN - Serial, Part N/`
alongside `subtitles.srt` and `subtitles.csv` (bitmap DVD/Blu-ray subtitles
are read with tesseract OCR). Each file is copied locally first, with the next
copying while the current one is screenshotted. Re-run to resume.

```sh
.venv/bin/python batch_shots.py "/Volumes/blobby/Doctor Who" -s 7-26 --dry-run
.venv/bin/python batch_shots.py "/Volumes/blobby/Doctor Who" -s 7-26
```

## iPlayer episodes

```sh
python3 iplayer_urls.py                         # episode list -> iplayer_episodes.csv
caffeinate -i .venv/bin/python iplayer_shots.py # download + screenshot everything -> output/
```

Each episode folder gets its screenshots plus `subtitles.srt` and `subtitles.csv`
(shot, start, end, text). Videos are deleted once screenshotted; re-running skips
finished episodes.

## Bluesky bot

`whobot.py` posts a random screenshot, least-posted first, so everything gets posted
once before anything repeats. On Christmas Day it only posts Christmas episodes, and
on New Year's Day New Year's ones. State is kept in a SQLite database (`whobot.db`).

```sh
cp whobot.env.example whobot.env && chmod 600 whobot.env   # then fill it in
.venv/bin/python whobot.py build-db output   # re-run after adding episodes; keeps post counts
.venv/bin/python whobot.py post --dry-run
.venv/bin/python whobot.py stats
```

The images can be in a local folder (`IMAGES_DIR`) or on a web server (`IMAGES_URL`).
In the URL case, `build-db` only needs the `subtitles.csv` files:

```sh
rsync -av --include '*/' --include 'subtitles.csv' --exclude '*' output/ box:whobot/output/
```

To post every hour, surviving reboots, install the systemd timer (run it as the
user the bot should run as; it fills in this folder's path and your username):

```sh
deploy/install.sh
systemctl list-timers whobot.timer   # next run
journalctl -u whobot                 # what it posted
```
