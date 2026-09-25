#!/usr/bin/env python3
"""Download iPlayer episodes with subtitles and screenshot every subtitle cue.

Reads the CSV written by iplayer_urls.py, downloads each episode with yt-dlp
(subtitles converted to SRT), runs subshots.py on it, then deletes the video.
The next episode downloads while the current one is being screenshotted.

Screenshots are filed by programme and series, e.g.

    output/Doctor Who (2005–2022)/Series 01/01 - Rose/
    output/Doctor Who (2005–2022)/Series 01/14 - The Christmas Invasion/
    output/Doctor Who (2023–)/Season 01/01 - Space Babies/

Each episode folder also gets subtitles.srt and subtitles.csv (shot, start,
end, text: which line each screenshot shows, times in seconds).

Specials go in the series iPlayer lists them under, numbered on from its last
episode. Finished episodes are skipped, so it can be stopped and re-run;
finished ones without subtitle files get just their subtitles downloaded.
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
SHOT_EXTS = {".jpg", ".png"}
DONE_MARKER = ".done"  # written once an episode's screenshots are complete
PROGRAMMES = {"Doctor Who": "Doctor Who (2023–)"}
NUMBERED = re.compile(r"(?:.*: )?(\d+)\. (.+)")


def safe(name):
    return re.sub(r'[/\\:*?"<>|]', "-", name).strip()


def episode_dirs(rows):
    """Map each pid to its screenshot folder, relative to the output folder.

    Episodes go in the series iPlayer lists them under. Specials, which iPlayer
    lists after a series' episodes, are numbered on from its last episode.
    """
    dirs = {}
    last = {}
    for row in rows:
        programme = PROGRAMMES.get(row["programme"], row["programme"])
        series = re.sub(r"\d+", lambda m: f"{int(m.group()):02d}", row["series"])
        key = (programme, series)
        m = NUMBERED.fullmatch(row["episode"])
        if m:
            number, title = int(m.group(1)), m.group(2)
        else:
            # "The Christmas Invasion", "Christmas Special: The Church on Ruby Road"
            number, title = last.get(key, 0) + 1, re.sub(r"^.*Special: ", "", row["episode"])
        last[key] = number
        dirs[row["pid"]] = Path(programme, series, f"{number:02d} - {safe(title)}")
    return dirs


def has_shots(shots_dir):
    return (shots_dir / DONE_MARKER).exists()


def short(name):
    """Episode folder without the programme, e.g. "Series 01/01 - Rose"."""
    return str(Path(*name.parts[1:]))


def duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s" if m else f"{s}s"


class Status:
    """One live line showing overall progress, the download and the screenshotting.

    Finished episodes and errors are printed above it. When output isn't a
    terminal, only those are printed.
    """

    def __init__(self, total):
        self.total = total
        self.finished = 0
        self.start = time.monotonic()
        self.jobs = {}
        self.lock = threading.Lock()
        self.tty = sys.stdout.isatty()

    def set(self, job, text=None):
        with self.lock:
            if text:
                self.jobs[job] = text
            else:
                self.jobs.pop(job, None)
            self._draw()

    def log(self, message, done=False):
        with self.lock:
            if done:
                self.finished += 1
            if self.tty:
                sys.stdout.write("\r\033[K")
            print(message, flush=True)
            self._draw()

    def clear(self):
        with self.lock:
            self.jobs.clear()
            if self.tty:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
            self.tty = False  # stop drawing

    def _draw(self):
        if not self.tty:
            return
        overall = f"[{self.finished}/{self.total}"
        if self.finished:
            left = (time.monotonic() - self.start) / self.finished * (self.total - self.finished)
            overall += f", ~{duration(left)} left"
        line = " │ ".join([overall + "]"] + [self.jobs[j] for j in ("download", "shoot") if j in self.jobs])
        width = shutil.get_terminal_size().columns
        if len(line) >= width:
            line = line[: width - 2] + "…"
        sys.stdout.write("\r\033[K" + line)
        sys.stdout.flush()


def stream(cmd, on_line):
    """Run cmd, passing each line of its output (split on \\r or \\n) to on_line.

    Raises RuntimeError with the last lines that on_line didn't claim if it fails.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    other = []
    buf = b""
    while chunk := proc.stdout.read1(4096):
        *lines, buf = re.split(rb"[\r\n]", buf + chunk)
        for line in lines:
            line = line.decode(errors="replace").strip()
            if line and not on_line(line):
                other.append(line)
    if proc.wait():
        raise RuntimeError("\n".join(other[-5:]) or f"{cmd[0]} exited with {proc.returncode}")
    return other


def download(row, dest, max_height, status, label):
    """Download the episode and English subs; return (video, subs) paths."""
    return fetch(row, dest, status, label, ["-S", f"res:{max_height}"] if max_height else [])


def download_subs(row, dest, status, label):
    """Download just the English subs; return their path."""
    return fetch(row, dest, status, label, ["--skip-download"])[1]


def fetch(row, dest, status, label, extra_args):

    def on_line(line):
        if not line.startswith("PROGRESS "):
            return False
        percent, speed, eta = (x.strip() for x in line[9:].split("|"))
        status.set("download", f"↓ {label} {percent} {speed} ETA {eta}")
        return True

    status.set("download", f"↓ {label}")
    stream([
        "yt-dlp", "--quiet", "--no-warnings", "--progress", "--newline", "--no-playlist", *extra_args,
        "--progress-template", "download:PROGRESS %(progress._percent_str)s|%(progress._speed_str)s|%(progress._eta_str)s",
        "--write-subs", "--sub-langs", "en.*", "--convert-subs", "srt",
        "-o", str(dest / f"{row['pid']}.%(ext)s"),
        row["url"],
    ], on_line)
    status.set("download")
    videos = [p for p in dest.glob(f"{row['pid']}.*") if p.suffix in {".mp4", ".mkv", ".webm"}]
    subs = sorted(dest.glob(f"{row['pid']}.*.srt"))
    if not subs:
        raise RuntimeError("no English subtitles available")
    if not videos and "--skip-download" not in extra_args:
        raise RuntimeError("yt-dlp produced no video file")
    return videos[0] if videos else None, subs[0]


def trim_subtitles(video, subs):
    """Drop subtitles that start after the video ends, and cut any that run past it.

    iPlayer's subtitles sometimes run on past the end of the video (e.g. the
    subtitler's credit after a longer broadcast ending).
    """
    import pysubs2

    end = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout) * 1000 - 100  # ms, with a little margin for the last frame
    srt = pysubs2.load(str(subs), encoding="utf-8")
    srt.events = [ev for ev in srt.events if ev.start < end]
    for ev in srt.events:
        ev.end = min(ev.end, end)
    srt.save(str(subs))


def save_subtitles(subs, shots_dir):
    """Keep the subtitles with the shots: the SRT, and a CSV of the cue each shot shows."""
    from subshots import load_text_events, text_cues

    shutil.copyfile(subs, shots_dir / "subtitles.srt")
    _, events = load_text_events(subs)
    # subshots.py names shots NNNN_<time>, NNNN being the cue's number.
    shots = {p.name[:4]: p.name for p in shots_dir.iterdir() if p.suffix in SHOT_EXTS}
    with open(shots_dir / "subtitles.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["shot", "start", "end", "text"])
        for i, cue in enumerate(text_cues(events), 1):
            writer.writerow([shots.get(f"{i:04d}", ""), f"{cue.start:.3f}", f"{cue.end:.3f}", cue.text])


def shoot(video, subs, shots_dir, max_height, extra_args, keep_video, status, label):
    """Screenshot every cue; return how many screenshots were taken."""

    def on_line(line):
        if m := re.fullmatch(r"(\d+)/(\d+)", line):
            status.set("shoot", f"▣ {label} {m[1]}/{m[2]}")
            return True
        return line.startswith(tuple("0123456789"))  # "414 cues, renderer: ..."

    status.set("shoot", f"▣ {label}")
    trim_subtitles(video, subs)
    try:
        failed = stream([
            sys.executable, "-u", str(HERE / "subshots.py"), str(video),
            "-s", str(subs), "-o", str(shots_dir), "--max-height", str(max_height), *extra_args,
        ], on_line)
    finally:
        status.set("shoot")
    for line in failed:  # individual cues that failed, which subshots.py reports but carries on past
        status.log(f"    {line}")
    save_subtitles(subs, shots_dir)
    (shots_dir / DONE_MARKER).touch()
    if not keep_video:
        video.unlink()
        subs.unlink()
    return sum(p.suffix in SHOT_EXTS for p in shots_dir.iterdir())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Any unrecognised options are passed through to subshots.py (e.g. -f png, -j 4).",
    )
    ap.add_argument("csv", type=Path, nargs="?", default=HERE / "iplayer_episodes.csv",
                    help="episode list from iplayer_urls.py (default: iplayer_episodes.csv)")
    ap.add_argument("-o", "--output", type=Path, default=HERE / "output",
                    help="where screenshots go (default: ./output); downloads go in its .downloads folder")
    ap.add_argument("-m", "--match", help="only episodes whose programme/episode matches this regex")
    ap.add_argument("--limit", type=int, help="only do the first N matching episodes")
    ap.add_argument("--keep-video", action="store_true",
                    help="keep videos and subtitles instead of deleting them after screenshotting")
    ap.add_argument("--max-height", type=int, default=576,
                    help="download at about this height and scale shots down to it "
                         "(default: 576, as subshots.py; 0 = best available)")
    args, subshots_args = ap.parse_known_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    dirs = episode_dirs(rows)  # before filtering, as specials are numbered from the episodes before them
    if args.match:
        pattern = re.compile(args.match, re.I)
        rows = [r for r in rows if pattern.search(f"{r['programme']} {r['episode']}")]
    if args.limit:
        rows = rows[: args.limit]
    todo = [r for r in rows if not has_shots(args.output / dirs[r["pid"]])]
    backfill = [r for r in rows if r not in todo and not (args.output / dirs[r["pid"]] / "subtitles.csv").exists()]
    if len(todo) < len(rows):
        print(f"Skipping {len(rows) - len(todo)} episode(s) already done.")
    downloads = args.output / ".downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    if backfill:
        print(f"Fetching subtitles for {len(backfill)} of them.")
        status = Status(len(backfill))
        for row in backfill:
            name = dirs[row["pid"]]
            try:
                subs = download_subs(row, downloads, status, label=short(name))
                save_subtitles(subs, args.output / name)
                subs.unlink()
                status.log(f"✓ {name}  subtitles", done=True)
            except RuntimeError as e:
                status.log(f"✗ {name}: subtitles failed\n    {e}", done=True)
        status.clear()
    if not todo:
        return
    print(f"{len(todo)} episode(s) to do, screenshots in {args.output}")

    status = Status(len(todo))
    failures = []

    def fail(name, stage, error):
        failures.append(name)
        detail = "\n".join(f"    {line}" for line in str(error).splitlines())
        status.log(f"✗ {name}: {stage} failed\n{detail}", done=True)

    def shoot_and_report(name, started, video, subs):
        shots_start = time.monotonic()
        count = shoot(video, subs, args.output / name, args.max_height, subshots_args,
                      args.keep_video, status, label=short(name))
        status.log(f"✓ {name}  {count} shots  (download {duration(shots_start - started)}, "
                   f"shots {duration(time.monotonic() - shots_start)})", done=True)

    pending = None  # screenshotting of the previous episode, running in the background
    with ThreadPoolExecutor(max_workers=1) as shooter:
        for row in todo:
            name = dirs[row["pid"]]
            started = time.monotonic()
            try:
                video, subs = download(row, downloads, args.max_height, status, label=short(name))
            except RuntimeError as e:
                status.set("download")
                fail(name, "download", e)
                continue
            if pending:
                wait(*pending, fail)
            pending = shooter.submit(shoot_and_report, name, started, video, subs), name
        if pending:
            wait(*pending, fail)

    status.clear()
    print(f"\nDone {len(todo) - len(failures)}/{len(todo)} episode(s) in {duration(time.monotonic() - status.start)}.")
    if failures:
        print("Failed:\n  " + "\n  ".join(map(str, failures)), file=sys.stderr)
        sys.exit(1)


def wait(future, name, fail):
    try:
        future.result()
    except RuntimeError as e:
        fail(name, "screenshots", e)


if __name__ == "__main__":
    main()
