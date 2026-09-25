#!/usr/bin/env python3
"""Screenshot every subtitle cue of the classic episodes listed in episodes.csv.

Each episode's file is copied from the library to a local folder first (the
next one copies while the current one is being screenshotted, as reading
straight off the network share is slow), then subshots.py runs on it with
--save-subs and the copy is deleted. Screenshots are filed like
iplayer_shots.py's, e.g.

    output/Doctor Who (1963–1996)/Season 07/01 - Spearhead from Space, Part 1/

with subtitles.srt and subtitles.csv (shot, start, end, text) alongside.
Episodes are done in random order (--in-order for broadcast order). Finished
episodes are skipped, so it can be stopped and re-run.
"""

import argparse
import csv
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from iplayer_shots import DONE_MARKER, SHOT_EXTS, Status, duration, safe, stream

HERE = Path(__file__).parent
PROGRAMME = "Doctor Who (1963–1996)"
COPY_CHUNK = 8 * 1024 * 1024


def parse_seasons(spec):
    seasons = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        seasons.update(range(int(lo), int(hi or lo) + 1))
    return seasons


def episode_dirs(rows):
    """Folder for each row, numbered by its position in its season.

    A row covering several parts (the condensed Marco Polo recon) takes the
    numbers of those parts, e.g. "14-20 - Marco Polo (condensed recon)".
    """
    dirs, count, numbers = [], {}, {}
    for r in rows:
        if r["part"].isdigit():
            season = int(r["season"])
            count[season] = count.get(season, 0) + 1
            numbers[(season, r["serial"], int(r["part"]))] = count[season]
    for r in rows:
        season = Path(PROGRAMME, f"Season {int(r['season']):02d}")
        if r["part"].isdigit():
            n = numbers[(int(r["season"]), r["serial"], int(r["part"]))]
            dirs.append(season / f"{n:02d} - {safe(r['serial'])}, Part {r['part']}")
        else:
            lo, hi = (int(x) for x in r["part"].split("-"))
            first, last = (numbers[(int(r["season"]), r["serial"], p)] for p in (lo, hi))
            label = "condensed recon" if "condensed" in r["version"] else f"Parts {r['part']}"
            dirs.append(season / f"{first:02d}-{last:02d} - {safe(r['serial'])} ({label})")
    return dirs


def is_animation(row):
    """Animated stand-in for a missing episode, including unlabelled ones presumed to be."""
    return row["version"].startswith("animation")


def animated_only(rows):
    """Stories whose only available versions are animations."""
    stories = {}
    for r in rows:
        if r["file"]:
            stories.setdefault((r["season"], r["serial"]), []).append(r["version"].startswith("animation"))
    return {story for story, animated in stories.items() if all(animated)}


def short(name):
    return str(Path(*name.parts[1:]))


def copy(src, dest, status, label):
    """Copy src to dest, showing progress; return dest."""
    size = src.stat().st_size
    start = time.monotonic()
    part = dest.with_name(dest.name + ".part")
    with open(src, "rb") as fin, open(part, "wb") as fout:
        done = 0
        while chunk := fin.read(COPY_CHUNK):
            fout.write(chunk)
            done += len(chunk)
            speed = done / max(time.monotonic() - start, 1e-6)
            left = (size - done) / speed if speed else 0
            status.set("download", f"↓ {label} {done / size:4.0%} {speed / 1e6:.1f}MB/s ETA {duration(left)}")
    part.rename(dest)
    status.set("download")
    return dest


def shoot(video, row, shots_dir, extra_args, status, label):
    """Screenshot every cue; return how many screenshots were taken."""
    import re

    def on_line(line):
        if m := re.fullmatch(r"(\d+)/(\d+)", line):
            status.set("shoot", f"▣ {label} {m[1]}/{m[2]}")
            return True
        return line.startswith(tuple("0123456789"))  # "414 cues, renderer: ..."

    status.set("shoot", f"▣ {label}")
    if shots_dir.exists():
        shutil.rmtree(shots_dir)  # partial output from an interrupted run
    try:
        failed = stream([
            sys.executable, "-u", str(HERE / "subshots.py"), str(video), "-o", str(shots_dir),
            "-t", row["subs_track"], "--save-subs", *extra_args,
        ], on_line)
    finally:
        status.set("shoot")
        video.unlink(missing_ok=True)
    for line in failed:  # individual cues that failed, which subshots.py reports but carries on past
        status.log(f"    {line}")
    (shots_dir / DONE_MARKER).touch()
    return sum(p.suffix in SHOT_EXTS for p in shots_dir.iterdir())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Any unrecognised options are passed through to subshots.py (e.g. --max-height 720, -j 4).",
    )
    ap.add_argument("root", type=Path, help="library folder the CSV's paths are relative to")
    ap.add_argument("--csv", type=Path, default=HERE / "episodes.csv", help="from catalogue.py")
    ap.add_argument("-s", "--seasons", default="1-26", help="e.g. 7-26 or 1,3,7-9 (default: all)")
    ap.add_argument("--include-animations", action="store_true",
                    help="also do animated stand-ins for missing episodes (skipped by default)")
    ap.add_argument("-q", "--quiet", action="store_true", help="don't list the episodes being skipped")
    ap.add_argument("-o", "--output", type=Path, default=HERE / "output",
                    help="where screenshots go (default: ./output); copies go in its .downloads folder")
    ap.add_argument("--limit", type=int, help="only do the first N episodes")
    ap.add_argument("--in-order", action="store_true",
                    help="go through episodes in broadcast order (default: random order)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be done")
    args, subshots_args = ap.parse_known_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    dirs = dict(zip(map(id, rows), episode_dirs(rows)))  # before filtering, so numbering is stable
    seasons = parse_seasons(args.seasons)
    animated = animated_only(rows)
    rows = [r for r in rows if int(r["season"]) in seasons]
    for story in sorted(animated, key=lambda s: (int(s[0]), s[1])):
        if int(story[0]) in seasons:
            print(f"Skipping {story[1]}: only animated versions available")
    rows = [r for r in rows if (r["season"], r["serial"]) not in animated]
    if not args.include_animations:
        for r in rows:
            if is_animation(r):
                print(f"Skipping {short(dirs[id(r)])}: animation")
        rows = [r for r in rows if not is_animation(r)]
    for r in rows:
        if not r["file"] or r["has_subs"] != "yes":
            why = "no file" if not r["file"] else "no subtitles"
            if not args.quiet:
                print(f"Skipping {short(dirs[id(r)])}: {why}")
    rows = [r for r in rows if r["file"] and r["has_subs"] == "yes"]
    todo = [r for r in rows if not (args.output / dirs[id(r)] / DONE_MARKER).exists()]
    if not args.in_order:
        random.shuffle(todo)
    if args.limit:
        todo = todo[: args.limit]
    if len(todo) < len(rows):
        print(f"Skipping {len(rows) - len(todo)} episode(s) already done or over the limit.")
    if args.dry_run:
        for r in todo:
            print(f"  {short(dirs[id(r)])}  <-  {r['file']} (track {r['subs_track']})")
        return
    if not todo:
        return
    print(f"{len(todo)} episode(s) to do, screenshots in {args.output / PROGRAMME}")

    downloads = args.output / ".downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    status = Status(len(todo))
    failures = []

    def fail(name, stage, error):
        failures.append(name)
        detail = "\n".join(f"    {line}" for line in str(error).splitlines())
        status.log(f"✗ {name}: {stage} failed\n{detail}", done=True)

    def shoot_and_report(name, started, video, row):
        shots_start = time.monotonic()
        count = shoot(video, row, args.output / name, subshots_args, status, label=short(name))
        status.log(f"✓ {name}  {count} shots  (copy {duration(shots_start - started)}, "
                   f"shots {duration(time.monotonic() - shots_start)})", done=True)

    def wait(future, name):
        try:
            future.result()
        except (RuntimeError, OSError) as e:
            fail(name, "screenshots", e)

    pending = None  # screenshotting of the previous episode, running in the background
    try:
        with ThreadPoolExecutor(max_workers=1) as shooter:
            for n, row in enumerate(todo):
                name = dirs[id(row)]
                src = args.root / row["file"]
                started = time.monotonic()
                try:
                    video = copy(src, downloads / f"classic-{n}{src.suffix}", status, label=short(name))
                except OSError as e:
                    status.set("download")
                    fail(name, "copy", e)
                    continue
                if pending:
                    wait(*pending)
                pending = shooter.submit(shoot_and_report, name, started, video, row), name
            if pending:
                wait(*pending)
    except KeyboardInterrupt:
        status.clear()
        print("\nStopped. Run the same command again to carry on.")
        sys.exit(130)
    finally:
        for p in downloads.glob("classic-*"):
            p.unlink(missing_ok=True)

    status.clear()
    print(f"\nDone {len(todo) - len(failures)}/{len(todo)} episode(s) in {duration(time.monotonic() - status.start)}.")
    if failures:
        print("Failed:\n  " + "\n  ".join(map(str, failures)), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
