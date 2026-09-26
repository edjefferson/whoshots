#!/usr/bin/env python3
"""Download iPlayer episodes and take a clean screenshot for every subtitle.

Reads the CSV written by iplayer_urls.py and downloads each episode with yt-dlp,
along with its original subtitles (TTML). For each subtitle it saves the
sharpest nearby frame *without* the subtitle burned in; the subtitle text, with
iPlayer's speaker colours, goes in subtitles.csv so whobot.py can draw it on
when posting. The video is then deleted. The next episode downloads while the
current one is being screenshotted.

Screenshots are filed by programme and series, e.g.

    output/Doctor Who (2005–2022)/Series 01/01 - Rose/
    output/Doctor Who (2005–2022)/Series 01/14 - The Christmas Invasion/
    output/Doctor Who (2023–)/Season 01/01 - Space Babies/

Each episode folder gets:
    cNNNN_HH-MM-SS.mmm.jpg   one clean frame per subtitle (NNNN = its number)
    subtitles.ttml           iPlayer's original subtitles
    subtitles.csv            shot, start, end, text, segments: the subtitle each
                             frame belongs to; segments is JSON, a list of lines,
                             each a list of [text, "#rrggbb"] runs

Specials go in the series iPlayer lists them under, numbered on from its last
episode. Finished episodes are skipped, so it can be stopped and re-run.
Episodes screenshotted the old way (subtitles burned in, no .clean marker) are
redone: their old files are removed first.
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
SHOT_EXTS = {".jpg", ".png"}
DONE_MARKER = ".done"    # written once an episode's screenshots are complete
CLEAN_MARKER = ".clean"  # ...and this one if they're clean frames (subtitles not burned in)
JPEG_QUALITY = 85
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
    return (shots_dir / CLEAN_MARKER).exists()


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
    """Download the episode and its English TTML subtitles; return (video, subs) paths."""

    def on_line(line):
        if not line.startswith("PROGRESS "):
            return False
        percent, speed, eta = (x.strip() for x in line[9:].split("|"))
        status.set("download", f"↓ {label} {percent} {speed} ETA {eta}")
        return True

    status.set("download", f"↓ {label}")
    # No point downloading more pixels than the screenshots keep.
    sort = ["-S", f"res:{max_height}"] if max_height else []
    stream([
        "yt-dlp", "--quiet", "--no-warnings", "--progress", "--newline", "--no-playlist", *sort,
        "--progress-template", "download:PROGRESS %(progress._percent_str)s|%(progress._speed_str)s|%(progress._eta_str)s",
        "--write-subs", "--sub-langs", "en.*", "--sub-format", "ttml",
        "-o", str(dest / f"{row['pid']}.%(ext)s"),
        row["url"],
    ], on_line)
    status.set("download")
    videos = [p for p in dest.glob(f"{row['pid']}.*") if p.suffix in {".mp4", ".mkv", ".webm"}]
    subs = sorted(dest.glob(f"{row['pid']}.*.ttml"))
    if not subs:
        raise RuntimeError("no English subtitles available")
    if not videos:
        raise RuntimeError("yt-dlp produced no video file")
    return videos[0], subs[0]


# --- Subtitles ------------------------------------------------------------------

TT = "{http://www.w3.org/ns/ttml}"
TTS = "{http://www.w3.org/ns/ttml#styling}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
WHITE = "#ffffff"


def ttml_time(value):
    """Seconds from a TTML clock time ("00:27:21.440", "00:27:48") or offset ("12.5s")."""
    if m := re.fullmatch(r"(\d+):(\d+):(\d+(?:\.\d+)?)", value):
        return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    if m := re.fullmatch(r"(\d+(?:\.\d+)?)(h|m|s|ms)", value):
        return float(m[1]) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[m[2]]
    raise ValueError(f"unsupported TTML time: {value}")


def parse_ttml(path, video_length=None):
    """The cues in a TTML file, as dicts with start, end and lines.

    Each line is a list of [text, colour] runs, the colour as "#rrggbb", so a
    line can change colour mid-way when a second speaker starts. Paragraphs
    shown at the same time (e.g. a sound label above dialogue) become one cue,
    in document order. With video_length, cues starting after the video ends
    are dropped and ones running past it cut short: iPlayer's subtitles
    sometimes run on past the end (the subtitler's credit, say).
    """
    root = ET.parse(path).getroot()
    styles = {s.get(XML_ID): s for s in root.iter(f"{TT}style")}

    def colour_of(element, inherited):
        for style_id in (element.get("style") or "").split():
            if style_id in styles:
                inherited = colour_of(styles[style_id], inherited)
        if colour := element.get(f"{TTS}color"):
            inherited = "#" + colour.lstrip("#")[:6].lower()
        return inherited

    def runs(element, colour, lines):
        # Text inside an element takes its colour; text after it (tail) its parent's.
        colour = colour_of(element, colour)
        if element.text:
            lines[-1].append([element.text, colour])
        for child in element:
            if child.tag == f"{TT}br":
                lines.append([])
            else:
                runs(child, colour, lines)
            if child.tail:
                lines[-1].append([child.tail, colour])

    paragraphs = []
    for p in root.iter(f"{TT}p"):
        start, end = ttml_time(p.get("begin")), ttml_time(p.get("end"))
        lines = [[]]
        runs(p, WHITE, lines)
        lines = [line for line in map(tidy_line, lines) if line]
        if lines and end > start:
            paragraphs.append((start, end, lines))

    if video_length is not None:
        limit = video_length - 0.1  # a little margin for the last frame
        paragraphs = [(s, min(e, limit), l) for s, e, l in paragraphs if s < limit]

    # One cue per distinct (start, end), with everything visible at its midpoint.
    cues = []
    for start, end in sorted({(s, e) for s, e, _ in paragraphs}):
        mid = (start + end) / 2
        lines = [line for s, e, ls in paragraphs if s <= mid < e for line in ls]
        cues.append({"start": start, "end": end, "lines": lines})
    return cues


def tidy_line(line):
    """Collapse whitespace (xml:space="default") and merge neighbouring runs of one colour."""
    out = []
    for text, colour in line:
        text = re.sub(r"\s+", " ", text)
        if out and out[-1][1] == colour:
            out[-1][0] += text
        elif text:
            out.append([text, colour])
    for run in out:
        run[0] = re.sub(r"\s+", " ", run[0])
    if out:
        out[0][0] = out[0][0].lstrip()
        out[-1][0] = out[-1][0].rstrip()
    return [run for run in out if run[0]]


def plain_text(lines):
    return "\n".join("".join(text for text, _ in line) for line in lines)


# --- Screenshots ----------------------------------------------------------------

def clear_episode(shots_dir):
    """Remove an episode's old screenshots and subtitles so it can be redone."""
    if not shots_dir.exists():
        return
    # The markers go first, so nothing (e.g. copying finished episodes to the Pi)
    # treats a half-redone folder as finished.
    for marker in (DONE_MARKER, CLEAN_MARKER):
        (shots_dir / marker).unlink(missing_ok=True)
    for p in shots_dir.iterdir():
        if p.suffix in SHOT_EXTS or p.name.startswith("subtitles."):
            p.unlink()


def shoot(video, subs, shots_dir, max_height, jobs, keep_video, status, label):
    """Save a clean frame for every cue plus the subtitle files; return how many frames."""
    import subshots

    subshots.MAX_HEIGHT = max_height or None
    probe = subshots.ffprobe_json("-show_format", str(video))["format"]
    video_length = float(probe["duration"]) - float(probe.get("start_time") or 0)
    cues = parse_ttml(subs, video_length)
    if not cues:
        raise RuntimeError("no subtitles in the TTML")

    clear_episode(shots_dir)
    shots_dir.mkdir(parents=True, exist_ok=True)
    status.set("shoot", f"▣ {label}")

    def grab(i, cue, tmp):
        c = subshots.Cue(cue["start"], cue["end"])
        t, img = subshots.grab_sharpest(video, c, 0.4, False, Path(tmp) / f"{i:04d}")
        name = f"c{i:04d}_{subshots.timestamp(t)}.jpg"
        img.save(shots_dir / name, quality=JPEG_QUALITY)
        return name

    names = {}
    failed = []
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(grab, i, cue, tmp): i for i, cue in enumerate(cues, 1)}
        for done, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            try:
                names[i] = fut.result()
            except Exception as e:
                failed.append(f"cue {i} @ {cues[i - 1]['start']:.3f}s failed: {e}")
            status.set("shoot", f"▣ {label} {done}/{len(cues)}")
    status.set("shoot")
    for line in failed:
        status.log(f"    {line}")
    if len(failed) > len(cues) // 20:
        raise RuntimeError(f"{len(failed)} of {len(cues)} frames failed")

    shutil.copyfile(subs, shots_dir / "subtitles.ttml")
    with open(shots_dir / "subtitles.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["shot", "start", "end", "text", "segments"])
        for i, cue in enumerate(cues, 1):
            writer.writerow([names.get(i, ""), f"{cue['start']:.3f}", f"{cue['end']:.3f}",
                             plain_text(cue["lines"]), json.dumps(cue["lines"], ensure_ascii=False)])
    (shots_dir / DONE_MARKER).touch()
    (shots_dir / CLEAN_MARKER).touch()
    if not keep_video:
        video.unlink()
        subs.unlink()
    return len(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, nargs="?", default=HERE / "iplayer_episodes.csv",
                    help="episode list from iplayer_urls.py (default: iplayer_episodes.csv)")
    ap.add_argument("-o", "--output", type=Path, default=HERE / "output",
                    help="where screenshots go (default: ./output); downloads go in its .downloads folder")
    ap.add_argument("-m", "--match", help="only episodes whose programme/episode matches this regex")
    ap.add_argument("--limit", type=int, help="only do the first N episodes still to do")
    ap.add_argument("--reverse", action="store_true", help="newest episodes first")
    ap.add_argument("--keep-video", action="store_true",
                    help="keep videos and subtitles instead of deleting them after screenshotting")
    ap.add_argument("--max-height", type=int, default=720,
                    help="download at about this height and keep frames at most this tall "
                         "(default: 720, iPlayer's best; 0 = no limit)")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4,
                    help="frames to grab in parallel (default: number of CPUs)")
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    dirs = episode_dirs(rows)  # before filtering, as specials are numbered from the episodes before them
    if args.match:
        pattern = re.compile(args.match, re.I)
        rows = [r for r in rows if pattern.search(f"{r['programme']} {r['episode']}")]
    if args.reverse:
        rows.reverse()
    todo = [r for r in rows if not has_shots(args.output / dirs[r["pid"]])]
    if args.limit:
        todo = todo[: args.limit]
    if len(todo) < len(rows):
        print(f"Skipping {len(rows) - len(todo)} episode(s) already done (or over --limit).")
    if not todo:
        return
    redo = sum((args.output / dirs[r["pid"]]).exists() for r in todo)
    downloads = args.output / ".downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    print(f"{len(todo)} episode(s) to do ({redo} replacing old burned-in screenshots), "
          f"screenshots in {args.output}")

    status = Status(len(todo))
    failures = []

    def fail(name, stage, error):
        failures.append(name)
        detail = "\n".join(f"    {line}" for line in str(error).splitlines())
        status.log(f"✗ {name}: {stage} failed\n{detail}", done=True)

    def shoot_and_report(name, started, video, subs):
        shots_start = time.monotonic()
        count = shoot(video, subs, args.output / name, args.max_height, args.jobs,
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
    except Exception as e:
        fail(name, "screenshots", e)


if __name__ == "__main__":
    main()
