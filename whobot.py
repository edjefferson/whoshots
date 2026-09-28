#!/usr/bin/env python3
"""Post a random Doctor Who screenshot to Bluesky.

    whobot.py build-db [SHOTS_DIR]   load new/changed shots from SHOTS_DIR/**/subtitles.csv
    whobot.py post [--dry-run]       post the next shot (run hourly by a systemd timer)
    whobot.py stats                  how far through the shots it's got
    whobot.py prune-images           list (--delete: remove) images no shot uses any more

Shots are posted least-posted first, at random, so every shot is posted once
before any is posted twice. Christmas and New Year episodes are tagged in the
database (the tag column), but that doesn't affect what's posted.

Shots come in two kinds: with the subtitle burned into the image (classic
episodes), or clean frames whose subtitle is stored as text with its speaker
colours (iPlayer episodes, see iplayer_shots.py). For those the subtitle is
drawn on when posting. Every image is scaled up before uploading, as Bluesky
re-compresses what it's given and a bigger image comes through cleaner.

Settings come from the environment, or from whobot.env next to this script:

    BSKY_HANDLE, BSKY_APP_PASSWORD   the account, with an app password
    IMAGES_DIR or IMAGES_URL         where the images are: a folder, or a web
                                     address they're served from (default: ./output)
    DB_PATH                          default: ./whobot.db
    POST_TEXT, ALT_TEXT              templates, using {programme} {series}
                                     {episode} {title} {text} {timestamp} {path};
                                     \\n for a new line
    UPLOAD_WIDTH                     scale images up to this width before posting
                                     (default: 1920; 0 = as stored)
    SUBTITLE_FONT                    font file for drawn subtitles (default: DejaVu
                                     Sans on Linux, Arial on a Mac)
"""

import argparse
import csv
import datetime
import io
import json
import os
import random
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
    "UPLOAD_WIDTH": "1920",
    "SUBTITLE_FONT": "",
}

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]
BLUESKY_MAX_BYTES = 950_000  # Bluesky's limit is about 1 MB

# Episode titles (as in the folder names) tagged in the database. The tags are only
# recorded for now; picking shots ignores them.
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
    segments TEXT,                -- JSON lines of [text, colour] runs if the subtitle isn't burned in
    overlay TEXT,                 -- JSON {image, x, y}: a subtitle image to paste on, if kept separately
    placement TEXT,               -- JSON [[lines, anchor, position], ...]: where drawn subtitles go
    tag TEXT,                     -- christmas, new_year or NULL
    skip INTEGER NOT NULL DEFAULT 0,
    post_count INTEGER NOT NULL DEFAULT 0,
    last_posted_at TEXT,
    post_uri TEXT
);
CREATE INDEX IF NOT EXISTS shots_pick ON shots (skip, tag, post_count);
CREATE INDEX IF NOT EXISTS shots_dir ON shots (dir);
-- Small things to remember between runs, e.g. the tag list last applied.
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
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
    # A generous timeout: if build-db is writing, a post waits rather than failing.
    db = sqlite3.connect(os.environ["DB_PATH"], timeout=120)
    db.row_factory = sqlite3.Row
    # WAL lets a post read while build-db writes, and syncs to disk far less often,
    # which matters on a slow USB disk. It's remembered in the database file.
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(SCHEMA)
    # Databases made before these existed lack the columns.
    columns = {row[1] for row in db.execute("PRAGMA table_info(shots)")}
    for column in ("segments", "overlay", "placement"):
        if column not in columns:
            db.execute(f"ALTER TABLE shots ADD COLUMN {column} TEXT")
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


def overlay_json(rel_dir, row):
    """The overlay column for a CSV row with a separate subtitle image, else None."""
    if not row.get("sub_image"):
        return None
    x, y = (int(v) for v in row["sub_pos"].split(","))
    return json.dumps({"image": (rel_dir / row["sub_image"]).as_posix(), "x": x, "y": y})


def remove_old_shots(db, dir, keep):
    """Delete an episode's shots its CSV no longer lists (e.g. it was redone); return how many.

    A redone episode's new shots have new names, so each old shot's post count
    is carried over to the new shot of the same subtitle (same start time),
    to keep already-posted lines from being posted again first.
    """
    others = f"dir = ? AND path NOT IN ({','.join('?' * len(keep))})"
    for old in db.execute(f"SELECT start, post_count, last_posted_at, post_uri FROM shots "
                          f"WHERE {others} AND post_count > 0", [dir, *keep]).fetchall():
        db.execute(f"""
            UPDATE shots SET post_count = max(post_count, ?), last_posted_at = ?, post_uri = ?
            WHERE dir = ? AND abs(start - ?) < 0.05 AND path IN ({','.join('?' * len(keep))})
        """, [old["post_count"], old["last_posted_at"], old["post_uri"], dir, old["start"], *keep])
    return db.execute(f"DELETE FROM shots WHERE {others}", [dir, *keep]).rowcount


def build_db(db, shots_dir, full=False):
    started = time.monotonic()
    print(f"Looking for episodes in {shots_dir}...", flush=True)
    found = find_episodes(shots_dir)
    finding = time.monotonic() - started
    if not found:
        sys.exit(f"No subtitles.csv files under {shots_dir}")
    loaded = {d: (m, s) for d, m, s in db.execute("SELECT dir, csv_mtime, csv_size FROM episodes")}
    todo = [(p, st) for p, st in found
            if full or loaded.get(p.parent.relative_to(shots_dir).as_posix()) != (st.st_mtime, st.st_size)]
    print(f"Found {len(found)} episodes, {len(todo)} new or changed.", flush=True)

    before = db.execute("SELECT count(*) FROM shots").fetchone()[0]
    known_dirs = {d for (d,) in db.execute("SELECT DISTINCT dir FROM shots")}
    new_episodes = removed = 0
    load_start = time.monotonic()
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
                INSERT INTO shots (path, dir, programme, series, episode, title, start, end, text, segments,
                                   overlay, placement)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    dir = excluded.dir, programme = excluded.programme, series = excluded.series,
                    episode = excluded.episode, title = excluded.title, start = excluded.start,
                    end = excluded.end, text = excluded.text, segments = excluded.segments,
                    overlay = excluded.overlay, placement = excluded.placement
                -- Only rewrite rows that actually changed.
                WHERE (shots.dir, shots.programme, shots.series, shots.episode, shots.title,
                       shots.start, shots.end, shots.text, shots.segments, shots.overlay, shots.placement)
                    IS NOT (excluded.dir, excluded.programme, excluded.series, excluded.episode,
                            excluded.title, excluded.start, excluded.end, excluded.text, excluded.segments,
                            excluded.overlay, excluded.placement)
            """, [
                (p, rel_dir.as_posix(), programme, series, number, title,
                 float(r["start"]), float(r["end"]), r["text"], r.get("segments") or None,
                 overlay_json(rel_dir, r), r.get("placement") or None)
                for p, r in zip(shot_paths, rows)
            ])
            removed += remove_old_shots(db, rel_dir.as_posix(), shot_paths)
            db.execute("INSERT OR REPLACE INTO episodes (dir, csv_mtime, csv_size) VALUES (?, ?, ?)",
                       (rel_dir.as_posix(), st.st_mtime, st.st_size))
        loading = time.monotonic() - load_start
        # Tags are (re)applied when episodes were loaded or TAGS has been edited, so
        # editing TAGS takes effect without --full; only rows whose tag changes are written.
        tag_start = time.monotonic()
        tags_now = json.dumps({tag: sorted(titles) for tag, titles in TAGS.items()}, sort_keys=True)
        tags_before = db.execute("SELECT value FROM meta WHERE key = 'tags'").fetchone()
        if todo or not tags_before or tags_before[0] != tags_now:
            cases = " ".join(f"WHEN title IN ({','.join('?' * len(titles))}) THEN ?" for titles in TAGS.values())
            params = [x for tag, titles in TAGS.items() for x in (*titles, tag)]
            tag_of = f"CASE {cases} END"
            db.execute(f"UPDATE shots SET tag = {tag_of} WHERE tag IS NOT {tag_of}", params * 2)
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('tags', ?)", (tags_now,))
        tagging = time.monotonic() - tag_start
    if tty:
        sys.stdout.write("\r\033[K")
    total = db.execute("SELECT count(*) FROM shots").fetchone()[0]
    added = total - before + removed
    print(f"Done in {time.monotonic() - started:.1f}s (finding {finding:.1f}s, loading {len(todo)} "
          f"episode(s) {loading:.1f}s, tags {tagging:.1f}s): {new_episodes} new episode(s), "
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

def pick(db):
    return db.execute("SELECT * FROM shots WHERE skip = 0 ORDER BY post_count, random() LIMIT 1").fetchone()


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


def subtitle_font(size):
    from PIL import ImageFont

    path = os.environ["SUBTITLE_FONT"] or next((f for f in FONT_CANDIDATES if Path(f).exists()), None)
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size)


def wrap_runs(draw, line, font, max_width):
    """Split one subtitle line (a list of [text, colour] runs) into lines that fit max_width."""
    words = [(word, colour) for text, colour in line for word in re.findall(r"\S+\s*|\s+", text)]
    lines, current = [], []
    for word in words:
        trial = current + [word]
        if current and draw.textlength("".join(w for w, _ in trial).rstrip(), font=font) > max_width:
            lines.append(current)
            current = [word] if word[0].strip() else []
        else:
            current = trial
    lines.append(current)
    return [line for line in lines if "".join(w for w, _ in line).strip()]


def draw_subtitles(img, lines, placement=None):
    """Draw subtitle lines onto a frame, centred, each run in its speaker's colour,
    with a black outline (the same size and look as subshots.py's).

    placement, from the broadcast subtitles, says where each group of lines goes:
    [number of lines, "bottom"/"top"/"center", fraction of the height]. Without it
    they all go at the bottom.
    """
    from PIL import ImageDraw

    w, h = img.size
    size = max(12, round(h * 0.055))
    font = subtitle_font(size)
    draw = ImageDraw.Draw(img)
    ascent, descent = font.getmetrics()
    line_height = round(size * 1.2)
    placement = placement or [[len(lines), "bottom", 0.95]]
    start = 0
    for count, anchor, position in placement:
        rows = [row for line in lines[start:start + count] for row in wrap_runs(draw, line, font, w * 0.9)]
        start += count
        if not rows:
            continue
        height = line_height * (len(rows) - 1) + ascent + descent
        top = {"top": position * h, "center": position * h - height / 2}.get(anchor, position * h - height)
        top = min(max(top, h * 0.02), h * 0.98 - height)  # keep it on the picture
        baseline = top + ascent
        for row in rows:
            row[-1] = (row[-1][0].rstrip(), row[-1][1])
            x = (w - draw.textlength("".join(t for t, _ in row), font=font)) / 2
            for text, colour in row:
                draw.text((x, baseline), text, font=font, fill=colour, anchor="ls",
                          stroke_width=max(2, size // 14), stroke_fill="black")
                x += draw.textlength(text, font=font)
            baseline += line_height


def prepare_image(shot, data):
    """The image to upload: scaled up, with the subtitle drawn on if it isn't burned in."""
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    upload_width = int(os.environ["UPLOAD_WIDTH"] or 0)
    scale = upload_width / img.width if upload_width and img.width < upload_width else 1
    if scale != 1:
        img = img.resize((upload_width, round(img.height * scale)), Image.LANCZOS)
    if shot["overlay"]:
        # A disc subtitle kept as an image: scaled like the frame, and kept inside
        # it (it may have run into pillarbox bars that were cropped off).
        overlay = json.loads(shot["overlay"])
        sub = Image.open(io.BytesIO(load_image(overlay["image"]))).convert("RGBA")
        if scale != 1:
            sub = sub.resize((round(sub.width * scale), round(sub.height * scale)), Image.LANCZOS)
        x = min(max(round(overlay["x"] * scale), 0), max(img.width - sub.width, 0))
        y = min(max(round(overlay["y"] * scale), 0), max(img.height - sub.height, 0))
        img.paste(sub, (x, y), sub)
    if shot["segments"]:
        # Drawn after scaling up, so the text is sharp at full size.
        draw_subtitles(img, json.loads(shot["segments"]),
                       json.loads(shot["placement"]) if shot["placement"] else None)
    for quality in (90, 85, 80, 75, 70):
        out = io.BytesIO()
        img.save(out, "JPEG", quality=quality)
        if out.tell() <= BLUESKY_MAX_BYTES:
            break
    return out.getvalue(), img.size


POST_TRIES = 3
RETRY_WAIT = 30  # seconds
BLUESKY_TIMEOUT = 30  # seconds per request; the library's default of 5 was sometimes too short
TRY_AGAIN_LATER = 75  # exit status (EX_TEMPFAIL) that makes systemd retry the run in 10 minutes
TID_CHARS = "234567abcdefghijklmnopqrstuvwxyz"


def new_tid():
    """A record key in Bluesky's TID format: microseconds since 1970 and a random
    clock id, in its sortable base 32."""
    n = (time.time_ns() // 1000) << 10 | random.getrandbits(10)
    return "".join(TID_CHARS[(n >> (5 * i)) & 31] for i in reversed(range(13)))


def status_of(e):
    return getattr(getattr(e, "response", None), "status_code", None)


def is_transient(e):
    """Whether an error is worth trying again: Bluesky not responding, a dropped
    connection, rate limiting or a server error, rather than being refused."""
    from atproto_client.exceptions import InvokeTimeoutError, RequestErrorBase

    if isinstance(e, InvokeTimeoutError):
        return True
    if isinstance(e, RequestErrorBase):
        status = status_of(e)
        return status is None or status == 429 or status >= 500
    return False


def with_retries(what, fn):
    """Run fn, retrying if the error is transient (Bluesky is occasionally slow)."""
    for attempt in range(1, POST_TRIES + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == POST_TRIES or not is_transient(e):
                raise
            print(f"{what}: Bluesky didn't respond ({type(e).__name__} {status_of(e) or ''}); "
                  f"trying again in {RETRY_WAIT}s")
            time.sleep(RETRY_WAIT)


def bluesky_client():
    """A logged-in Bluesky client, reusing the session saved last time if it's still
    good (Bluesky rate-limits logins, and logging in is the call that was timing out)."""
    from atproto import Client
    from atproto_client.request import Request

    session_file = Path(os.environ["DB_PATH"]).with_name("whobot.session")

    def save_session(*_):
        session_file.write_text(client.export_session_string(), encoding="utf-8")
        session_file.chmod(0o600)

    client = Client(request=Request(timeout=BLUESKY_TIMEOUT))
    client.on_session_change(save_session)
    if session_file.exists():
        try:
            client.login(session_string=session_file.read_text(encoding="utf-8"), fetch_bsky_profile=False)
            return client
        except Exception as e:  # expired or revoked: log in afresh
            print(f"Saved session didn't work ({type(e).__name__}); logging in")
    client.login(os.environ["BSKY_HANDLE"], os.environ["BSKY_APP_PASSWORD"], fetch_bsky_profile=False)
    save_session()
    return client


def account_did(client):
    # With fetch_bsky_profile=False (one request fewer), client.me isn't filled in;
    # the session, which is private to the library, has the account's DID.
    return client._session.did


def post_exists(client, rkey):
    """Whether this account has a post with this record key."""
    from atproto_client.exceptions import BadRequestError

    try:
        client.com.atproto.repo.get_record(
            {"repo": account_did(client), "collection": "app.bsky.feed.post", "rkey": rkey})
        return True
    except BadRequestError as e:
        if getattr(getattr(e.response, "content", None), "error", None) == "RecordNotFound":
            return False
        raise


class Rejected(Exception):
    """Bluesky refused this post itself (e.g. too long or too big), so retrying won't help."""


def send_post(client, image, size, text, alt, rkey, created_at):
    """Post the image with the given record key; return the post's URI.

    Creating the post is the one step that isn't safe to simply retry: it may have
    gone through even though the reply didn't arrive. With a fixed key, a retry
    first checks whether that post already exists.
    """
    from atproto import models
    from atproto_client.exceptions import RequestErrorBase

    did = account_did(client)
    uri = f"at://{did}/app.bsky.feed.post/{rkey}"
    try:
        blob = with_retries("uploading the image", lambda: client.upload_blob(image).blob)
        record = models.AppBskyFeedPost.Record(
            created_at=created_at, text=text, langs=["en"],
            embed=models.AppBskyEmbedImages.Main(images=[models.AppBskyEmbedImages.Image(
                alt=alt, image=blob,
                aspect_ratio=models.AppBskyEmbedDefs.AspectRatio(width=size[0], height=size[1]))]),
        )
        tried = False

        def create():
            nonlocal tried
            if tried and post_exists(client, rkey):  # did the attempt that timed out work?
                return uri
            tried = True
            return client.app.bsky.feed.post.create(did, record, rkey=rkey).uri

        return with_retries("posting", create)
    except RequestErrorBase as e:
        if status_of(e) in (400, 413):
            raise Rejected(f"{type(e).__name__} {status_of(e)}: {getattr(e.response, 'content', '')}") from e
        raise


def set_pending(db, pending):
    with db:
        if pending:
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('pending_post', ?)", (json.dumps(pending),))
        else:
            db.execute("DELETE FROM meta WHERE key = 'pending_post'")


def record_post(db, path, uri, when):
    with db:
        db.execute("UPDATE shots SET post_count = post_count + 1, last_posted_at = ?, post_uri = ? WHERE path = ?",
                   (when, uri, path))
        db.execute("DELETE FROM meta WHERE key = 'pending_post'")


def post(db, dry_run, save=None):
    """Post the next shot.

    The shot and the post's record key are saved before posting. If a run fails,
    the next one first checks whether that post went through anyway (and records
    it), and otherwise posts the same shot with the same key, so nothing is posted
    twice. A post Bluesky refuses outright is dropped and its shot skipped.
    """
    pending = db.execute("SELECT value FROM meta WHERE key = 'pending_post'").fetchone()
    pending = json.loads(pending[0]) if pending else None
    client = None
    shot = None
    if pending and not dry_run:
        client = with_retries("logging in", bluesky_client)
        shot = db.execute("SELECT * FROM shots WHERE path = ?", (pending["path"],)).fetchone()
        if with_retries("checking the unfinished post", lambda: post_exists(client, pending["rkey"])):
            uri = f"at://{account_did(client)}/app.bsky.feed.post/{pending['rkey']}"
            now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            record_post(db, pending["path"], uri, pending.get("tried_at", now))
            print(f"The unfinished post went through after all: {uri}")
            return
        if not shot or shot["skip"]:
            print("Dropping the unfinished post: its shot is no longer in the database, or is skipped.")
            set_pending(db, None)
            pending, shot = None, None
        else:
            print(f"Finishing the post that didn't complete last time ({pending['rkey']}).")
    if not shot:
        pending = None
        shot = pick(db)
    if not shot:
        sys.exit("No shots in the database; run build-db first.")
    text = render(os.environ["POST_TEXT"], shot)
    alt = render(os.environ["ALT_TEXT"], shot)
    image, (width, height) = prepare_image(shot, load_image(shot["path"]))
    kind = ("subtitle drawn on" if shot["segments"] else "subtitle image pasted on" if shot["overlay"]
            else "subtitle burned in")
    print(f"{shot['path']} ({kind}, uploading {width}x{height} {len(image) // 1024} KB, "
          f"posted {shot['post_count']}x before)")
    print(f"text: {text!r}\nalt:  {alt!r}")
    if save:
        Path(save).write_bytes(image)
        print(f"saved {save}")
    if dry_run:
        return

    # Posted now, whenever the shot was first tried, so it isn't backdated in feeds.
    now = datetime.datetime.now(datetime.timezone.utc)
    created_at = now.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    pending = {"path": shot["path"], "rkey": pending["rkey"] if pending else new_tid(), "tried_at": created_at}
    set_pending(db, pending)
    try:
        client = client or with_retries("logging in", bluesky_client)
        uri = send_post(client, image, (width, height), text, alt, pending["rkey"], created_at)
    except Rejected as e:
        with db:
            db.execute("UPDATE shots SET skip = 1 WHERE path = ?", (shot["path"],))
        set_pending(db, None)
        sys.exit(f"Bluesky refused this post, so its shot is now skipped: {e}")
    except Exception as e:
        if is_transient(e):
            print(f"Bluesky still isn't responding ({type(e).__name__}); the next run will finish this post.")
            sys.exit(TRY_AGAIN_LATER)
        raise
    record_post(db, shot["path"], uri, created_at)
    print(f"posted {uri}")


# --- stats --------------------------------------------------------------------

def prune_images(db, delete):
    """Images in the episode folders the database knows about that no shot uses,
    e.g. old burned-in frames left behind after an episode was redone."""
    if os.environ.get("IMAGES_URL"):
        sys.exit("prune-images only works on a local IMAGES_DIR.")
    root = Path(os.environ["IMAGES_DIR"])
    # A database that's behind the subtitles.csv files doesn't know about newer
    # screenshots, and would list them as unused.
    loaded = {d: (m, s) for d, m, s in db.execute("SELECT dir, csv_mtime, csv_size FROM episodes")}
    stale = [p.parent.relative_to(root).as_posix() for p, st in find_episodes(root)
             if loaded.get(p.parent.relative_to(root).as_posix()) != (st.st_mtime, st.st_size)]
    if stale:
        sys.exit(f"The database is out of date for {len(stale)} episode(s) (e.g. {stale[0]}); "
                 f"run build-db first, so current screenshots aren't counted as unused.")
    used = {p for (p,) in db.execute("SELECT path FROM shots")}
    used |= {json.loads(o)["image"] for (o,) in db.execute("SELECT overlay FROM shots WHERE overlay IS NOT NULL")}
    unused = []
    for (d,) in db.execute("SELECT DISTINCT dir FROM shots ORDER BY dir"):
        folder = root / d
        if folder.is_dir():
            unused += [p for p in folder.iterdir()
                       if p.suffix in {".jpg", ".png"} and p.relative_to(root).as_posix() not in used]
    size = sum(p.stat().st_size for p in unused)
    for p in unused[:10]:
        print(f"  {p.relative_to(root)}")
    if len(unused) > 10:
        print(f"  ... and {len(unused) - 10} more")
    print(f"{len(unused)} unused image(s), {size / 1e6:.0f} MB, in {len({p.parent for p in unused})} episode(s)")
    if delete:
        for p in unused:
            p.unlink()
        print("Deleted.")
    elif unused:
        print("Run again with --delete to remove them.")


def stats(db):
    total, posted, lowest = db.execute(
        "SELECT count(*), sum(post_count > 0), min(post_count) FROM shots WHERE skip = 0"
    ).fetchone()
    episodes = db.execute("SELECT count(DISTINCT dir) FROM shots").fetchone()[0]
    print(f"{total} shots from {episodes} episodes, {posted or 0} posted at least once, "
          f"everything posted at least {lowest or 0}x")
    drawn, pasted = db.execute("SELECT count(segments), count(overlay) FROM shots WHERE skip = 0").fetchone()
    print(f"  {drawn} with the subtitle drawn on when posting, {pasted} with a subtitle image pasted on, "
          f"{total - drawn - pasted} burned in")
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
    p.add_argument("--save", metavar="FILE", help="also save the image that would be uploaded")
    sub.add_parser("stats", help="how far through the shots it's got")
    pr = sub.add_parser("prune-images", help="list images no shot uses any more")
    pr.add_argument("--delete", action="store_true", help="delete them")
    args = ap.parse_args()

    load_env()
    db = connect()
    if args.command == "build-db":
        build_db(db, args.shots_dir or Path(os.environ["IMAGES_DIR"]), args.full)
    elif args.command == "post":
        post(db, args.dry_run, args.save)
    elif args.command == "prune-images":
        prune_images(db, args.delete)
    else:
        stats(db)


if __name__ == "__main__":
    main()
