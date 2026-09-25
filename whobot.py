#!/usr/bin/env python3
"""Post a random Doctor Who screenshot to Bluesky.

    whobot.py build-db [SHOTS_DIR]   load new/changed shots from SHOTS_DIR/**/subtitles.csv
    whobot.py post [--dry-run]       post the next shot (run hourly by a systemd timer)
    whobot.py stats                  how far through the shots it's got

Shots are posted least-posted first, at random, so every shot is posted once
before any is posted twice. On Christmas Day only Christmas episodes are used,
and on New Year's Day only New Year's ones.

Settings come from the environment, or from whobot.env next to this script:

    BSKY_HANDLE, BSKY_APP_PASSWORD   the account, with an app password
    IMAGES_DIR or IMAGES_URL         where the images are: a folder, or a web
                                     address they're served from (default: ./output)
    DB_PATH                          default: ./whobot.db
    POST_TEXT, ALT_TEXT              templates, using {programme} {series}
                                     {episode} {title} {text} {timestamp} {path};
                                     \\n for a new line
"""

import argparse
import csv
import datetime
import io
import os
import re
import shutil
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

DEFAULTS = {
    "IMAGES_DIR": str(HERE / "output"),
    "DB_PATH": str(HERE / "whobot.db"),
    "POST_TEXT": "",
    "ALT_TEXT": "{text}\\n\\nDoctor Who, {title} ({series})",
}

# Episode titles (as in the folder names) posted on these days.
TAGS = {
    "christmas": {
        "The Christmas Invasion", "The Runaway Bride", "Voyage of the Damned", "The Next Doctor",
        "The End of Time - Part One", "A Christmas Carol", "The Doctor, the Widow and the Wardrobe",
        "The Snowmen", "The Time of the Doctor", "Last Christmas", "The Husbands of River Song",
        "The Return of Doctor Mysterio", "Twice Upon a Time", "The Church on Ruby Road",
        "Joy to the World",
    },
    "new_year": {
        "The End of Time - Part Two", "Resolution", "Revolution of the Daleks", "Eve of the Daleks",
    },
}
TAG_DAYS = {(12, 25): "christmas", (1, 1): "new_year"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS shots (
    path TEXT PRIMARY KEY,        -- relative to the images folder/URL
    dir TEXT NOT NULL,            -- episode folder, e.g. "Doctor Who (2005–2022)/Series 01/01 - Rose"
    programme TEXT NOT NULL,
    series TEXT NOT NULL,         -- e.g. "Series 1", "Specials"
    episode INTEGER,
    title TEXT NOT NULL,
    start REAL,
    end REAL,
    text TEXT,
    tag TEXT,                     -- christmas, new_year or NULL
    skip INTEGER NOT NULL DEFAULT 0,
    post_count INTEGER NOT NULL DEFAULT 0,
    last_posted_at TEXT,
    post_uri TEXT
);
CREATE INDEX IF NOT EXISTS shots_pick ON shots (skip, tag, post_count);
CREATE INDEX IF NOT EXISTS shots_dir ON shots (dir);
-- Each episode's subtitles.csv as last loaded, so unchanged ones can be skipped.
CREATE TABLE IF NOT EXISTS episodes (
    dir TEXT PRIMARY KEY,
    csv_mtime REAL NOT NULL,
    csv_size INTEGER NOT NULL
);
"""


def load_env():
    """Fill in settings from whobot.env (KEY=value lines), without overriding the environment."""
    env_file = HERE / "whobot.env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                os.environ.setdefault(key.strip(), value)
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)


def connect():
    db = sqlite3.connect(os.environ["DB_PATH"])
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


# --- build-db -----------------------------------------------------------------

def episode_info(rel_dir):
    """("Doctor Who (2005–2022)", "Series 1", 14, "The Christmas Invasion") from its folder."""
    programme, series, folder = rel_dir.parts[-3:]
    number, title = folder.split(" - ", 1)
    series = re.sub(r"\b0+(\d)", r"\1", series)  # "Series 01" -> "Series 1"
    # A recon covering several missing episodes is numbered with a range, e.g. "14-20".
    return programme, series, int(number.split("-")[0]), title


def subdirs(path):
    return sorted(e.path for e in os.scandir(path) if e.is_dir() and not e.name.startswith("."))


def find_episodes(shots_dir):
    """(csv path, stat) for each <programme>/<series>/<episode>/subtitles.csv.

    Walks only the folder levels, so the (many) images in episode folders are
    never listed, which matters on a slow disk.
    """
    found = []
    for programme in subdirs(shots_dir):
        for series in subdirs(programme):
            for episode in subdirs(series):
                path = Path(episode, "subtitles.csv")
                try:
                    found.append((path, path.stat()))
                except FileNotFoundError:
                    pass  # not finished yet
    return found


def status_line(text):
    width = shutil.get_terminal_size().columns
    sys.stdout.write("\r\033[K" + (text if len(text) < width else text[: width - 2] + "…"))
    sys.stdout.flush()


def build_db(db, shots_dir, full=False):
    started = time.monotonic()
    print(f"Looking for episodes in {shots_dir}...", flush=True)
    found = find_episodes(shots_dir)
    if not found:
        sys.exit(f"No subtitles.csv files under {shots_dir}")
    loaded = {d: (m, s) for d, m, s in db.execute("SELECT dir, csv_mtime, csv_size FROM episodes")}
    todo = [(p, st) for p, st in found
            if full or loaded.get(p.parent.relative_to(shots_dir).as_posix()) != (st.st_mtime, st.st_size)]
    print(f"Found {len(found)} episodes, {len(todo)} new or changed.", flush=True)

    before = db.execute("SELECT count(*) FROM shots").fetchone()[0]
    known_dirs = {d for (d,) in db.execute("SELECT DISTINCT dir FROM shots")}
    new_episodes = removed = 0
    tty = sys.stdout.isatty()
    with db:
        for i, (path, st) in enumerate(todo, 1):
            rel_dir = path.parent.relative_to(shots_dir)
            if tty:
                status_line(f"[{i}/{len(todo)}] {Path(*rel_dir.parts[1:])}")
            new_episodes += rel_dir.as_posix() not in known_dirs
            programme, series, number, title = episode_info(rel_dir)
            with open(path, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r["shot"]]
            shot_paths = [(rel_dir / r["shot"]).as_posix() for r in rows]
            db.executemany("""
                INSERT INTO shots (path, dir, programme, series, episode, title, start, end, text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    dir = excluded.dir, programme = excluded.programme, series = excluded.series,
                    episode = excluded.episode, title = excluded.title, start = excluded.start,
                    end = excluded.end, text = excluded.text
            """, [
                (p, rel_dir.as_posix(), programme, series, number, title,
                 float(r["start"]), float(r["end"]), r["text"])
                for p, r in zip(shot_paths, rows)
            ])
            # Shots this episode's CSV no longer lists (e.g. it was redone).
            removed += db.execute(
                f"DELETE FROM shots WHERE dir = ? AND path NOT IN ({','.join('?' * len(shot_paths))})",
                [rel_dir.as_posix(), *shot_paths],
            ).rowcount
            db.execute("INSERT OR REPLACE INTO episodes (dir, csv_mtime, csv_size) VALUES (?, ?, ?)",
                       (rel_dir.as_posix(), st.st_mtime, st.st_size))
        # Tags are applied to every episode each time, so editing TAGS takes effect without --full.
        db.execute("UPDATE shots SET tag = NULL WHERE tag IS NOT NULL")
        for tag, titles in TAGS.items():
            db.execute(f"UPDATE shots SET tag = ? WHERE title IN ({','.join('?' * len(titles))})",
                       [tag, *titles])
    if tty:
        sys.stdout.write("\r\033[K")
    total = db.execute("SELECT count(*) FROM shots").fetchone()[0]
    added = total - before + removed
    print(f"Done in {time.monotonic() - started:.1f}s: {new_episodes} new episode(s), "
          f"{added} new shot(s), {removed} removed, {total} in total")
    for programme, episodes, shots in db.execute(
        "SELECT programme, count(DISTINCT dir), count(*) FROM shots GROUP BY programme ORDER BY programme"
    ):
        print(f"  {programme}: {episodes} episodes, {shots} shots")
    tagged = {t for (t,) in db.execute("SELECT DISTINCT title FROM shots WHERE tag IS NOT NULL")}
    missing = [t for titles in TAGS.values() for t in titles if t not in tagged]
    if missing:
        print(f"Tagged episodes not in the database yet: {', '.join(sorted(missing))}")


# --- post ---------------------------------------------------------------------

def pick(db, day):
    tag = TAG_DAYS.get((day.month, day.day))
    query = "SELECT * FROM shots WHERE skip = 0 {} ORDER BY post_count, random() LIMIT 1"
    shot = db.execute(query.format("AND tag = ?"), (tag,)).fetchone() if tag else None
    return shot or db.execute(query.format("")).fetchone()


def timestamp(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def render(template, shot):
    return template.replace("\\n", "\n").format(
        programme=shot["programme"], series=shot["series"], episode=shot["episode"],
        title=shot["title"], text=shot["text"], timestamp=timestamp(shot["start"]), path=shot["path"],
    ).strip()


def load_image(path):
    if url := os.environ.get("IMAGES_URL"):
        full = url.rstrip("/") + "/" + urllib.parse.quote(path)
        with urllib.request.urlopen(full, timeout=30) as resp:
            return resp.read()
    return (Path(os.environ["IMAGES_DIR"]) / path).read_bytes()


def post(db, dry_run, day):
    from PIL import Image

    shot = pick(db, day)
    if not shot:
        sys.exit("No shots in the database; run build-db first.")
    text = render(os.environ["POST_TEXT"], shot)
    alt = render(os.environ["ALT_TEXT"], shot)
    image = load_image(shot["path"])
    width, height = Image.open(io.BytesIO(image)).size
    print(f"{shot['path']} ({width}x{height}, posted {shot['post_count']}x before)")
    print(f"text: {text!r}\nalt:  {alt!r}")
    if dry_run:
        return

    from atproto import Client, models

    client = Client()
    client.login(os.environ["BSKY_HANDLE"], os.environ["BSKY_APP_PASSWORD"])
    resp = client.send_image(
        text=text, image=image, image_alt=alt,
        image_aspect_ratio=models.AppBskyEmbedDefs.AspectRatio(width=width, height=height),
    )
    with db:
        db.execute(
            "UPDATE shots SET post_count = post_count + 1, last_posted_at = ?, post_uri = ? WHERE path = ?",
            (datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), resp.uri, shot["path"]),
        )
    print(f"posted {resp.uri}")


# --- stats --------------------------------------------------------------------

def stats(db):
    total, posted, lowest = db.execute(
        "SELECT count(*), sum(post_count > 0), min(post_count) FROM shots WHERE skip = 0"
    ).fetchone()
    episodes = db.execute("SELECT count(DISTINCT dir) FROM shots").fetchone()[0]
    print(f"{total} shots from {episodes} episodes, {posted or 0} posted at least once, "
          f"everything posted at least {lowest or 0}x")
    for tag, count in db.execute("SELECT tag, count(*) FROM shots WHERE tag IS NOT NULL GROUP BY tag"):
        print(f"  {tag}: {count} shots")
    skipped = db.execute("SELECT count(*) FROM shots WHERE skip = 1").fetchone()[0]
    if skipped:
        print(f"  skipped: {skipped} shots")
    last = db.execute("SELECT path, last_posted_at FROM shots ORDER BY last_posted_at DESC LIMIT 1").fetchone()
    if last and last["last_posted_at"]:
        print(f"last post {last['last_posted_at']}: {last['path']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-db", help="load/refresh shots from subtitles.csv files")
    b.add_argument("shots_dir", type=Path, nargs="?", help="folder of episode folders (default: IMAGES_DIR)")
    b.add_argument("--full", action="store_true", help="reload every episode, not just new or changed ones")
    p = sub.add_parser("post", help="post the next shot")
    p.add_argument("--dry-run", action="store_true", help="show what would be posted, without posting")
    p.add_argument("--date", type=datetime.date.fromisoformat, default=None, help=argparse.SUPPRESS)
    sub.add_parser("stats", help="how far through the shots it's got")
    args = ap.parse_args()

    load_env()
    db = connect()
    if args.command == "build-db":
        build_db(db, args.shots_dir or Path(os.environ["IMAGES_DIR"]), args.full)
    elif args.command == "post":
        post(db, args.dry_run, args.date or datetime.date.today())
    else:
        stats(db)


if __name__ == "__main__":
    main()
