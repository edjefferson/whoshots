#!/usr/bin/env python3
"""Work out what's in each screenshot: frame quality, faces (and who they are),
and CLIP tags and search. Runs on the Mac, in its own environment:

    uv venv .venv-analysis --python 3.12
    uv pip install --python .venv-analysis/bin/python -r requirements-analysis.txt

    .venv-analysis/bin/python analyse.py run [--steps quality,faces,clip]
    .venv-analysis/bin/python analyse.py cluster      # writes review/clusters.html
    #   ...name the groups you recognise in people.csv...
    .venv-analysis/bin/python analyse.py label
    .venv-analysis/bin/python analyse.py search "a dalek in a corridor"
    .venv-analysis/bin/python analyse.py stats

Shots are found through the episodes' subtitles.csv files (the same paths as
whobot.db). Every step records what it has done, so it can be stopped and
re-run, and new episodes are picked up. Results:

    analysis.db     per shot: brightness, contrast, sharpness, face count, largest
                    face, people, tags; per face: box, score, cluster, person.
                    Small enough to copy to the Pi.
    embeddings.db   face and CLIP embeddings (float16), for clustering and search.
"""

import argparse
import csv
import html
import io
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
ANALYSIS_DB = HERE / "analysis.db"
EMBEDDINGS_DB = HERE / "embeddings.db"
REVIEW = HERE / "review"
PEOPLE_CSV = HERE / "people.csv"

FACE_MIN_SCORE = 0.6    # detection confidence to count as a face
FACE_MIN_SIZE = 0.03    # a face at least this fraction of the frame height counts
MATCH_THRESHOLD = 0.45  # cosine similarity to a person's centre to be labelled as them
CLIP_MODEL = ("ViT-B-32", "laion2b_s34b_b79k")

# Zero-shot tags: prompt -> tag. Scores for all are stored and tags worked out from
# them (see tag_shots), so the rules can be retuned without re-running CLIP.
TAG_PROMPTS = {
    "a Dalek": "dalek", "a Cyberman": "cyberman", "a Weeping Angel statue": "weeping angel",
    "a blue police box": "tardis", "the inside of the TARDIS control room": "tardis interior",
    "an Ood with tentacles on its face": "ood", "a Sontaran with a domed head": "sontaran",
    "a monster": "monster", "a robot": "robot", "a spaceship in space": "spaceship",
    "an explosion": "explosion", "a planet seen from space": "space", "a corridor": "corridor",
    "a quarry": "quarry", "a forest": "forest", "a beach": "beach", "a city street": "street",
    "a laboratory": "laboratory", "a control panel with buttons": "control panel",
    "a close-up of a face": "close-up", "a crowd of people": "crowd", "a soldier with a gun": "soldier",
    "a cartoon": "animation",
    "a person screaming": "scream", "a person laughing": "laughing", "a kiss": "kiss", "a dog": "dog",
    "a car": "car", "a horse": "horse", "a fire": "fire", "snow": "snow", "night time": "night",
}
TAG_TOP = 0.03      # a shot gets a tag if it's in the top 3% of all shots for that tag...
TAG_MARGIN = 0.02   # ...and that tag's score beats the shot's average by this much


# --- Shots and databases ----------------------------------------------------------

def all_shots():
    """Every shot's image path relative to OUTPUT, from the episodes' subtitles.csv."""
    shots = []
    for path in sorted(OUTPUT.glob("*/*/*/subtitles.csv")):
        rel = path.parent.relative_to(OUTPUT)
        with open(path, newline="", encoding="utf-8") as f:
            shots += [(rel / r["shot"]).as_posix() for r in csv.DictReader(f) if r["shot"]]
    return shots


def connect():
    db = sqlite3.connect(ANALYSIS_DB, timeout=60)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS shots (
            path TEXT PRIMARY KEY,
            brightness REAL, contrast REAL, sharpness REAL,
            face_count INTEGER, largest_face REAL,
            people TEXT,      -- JSON list of names
            tags TEXT,        -- JSON list
            tag_scores TEXT,  -- JSON {tag: score}
            done_quality INTEGER DEFAULT 0, done_faces INTEGER DEFAULT 0, done_clip INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS faces (
            -- AUTOINCREMENT: ids are never reused, so an embedding left behind by an
            -- interrupted run (embeddings.db is saved separately) can't be mistaken
            -- for a new face's.
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            x REAL, y REAL, w REAL, h REAL,  -- fractions of the frame
            score REAL,
            cluster INTEGER,
            person TEXT
        );
        CREATE INDEX IF NOT EXISTS faces_path ON faces (path);
        CREATE INDEX IF NOT EXISTS faces_cluster ON faces (cluster);
    """)
    db.execute(f"ATTACH DATABASE ? AS emb", (str(EMBEDDINGS_DB),))
    db.executescript("""
        CREATE TABLE IF NOT EXISTS emb.face_embeddings (id INTEGER PRIMARY KEY, vec BLOB);
        CREATE TABLE IF NOT EXISTS emb.clip_embeddings (path TEXT PRIMARY KEY, vec BLOB);
    """)
    return db


def todo(db, step, shots):
    db.executemany("INSERT OR IGNORE INTO shots (path) VALUES (?)", [(s,) for s in shots])
    db.commit()
    done = {p for (p,) in db.execute(f"SELECT path FROM shots WHERE done_{step} = 1")}
    return [s for s in shots if s not in done]


def vec(blob):
    import numpy as np
    return np.frombuffer(blob, dtype=np.float16).astype(np.float32)


def blob(v):
    import numpy as np
    return np.asarray(v, dtype=np.float16).tobytes()


class Progress:
    """A live progress line in a terminal (count, bar, speed, time left); in a log,
    a line every 30 seconds instead. finish() prints a summary."""

    def __init__(self, what, total):
        self.what, self.total, self.done = what, total, 0
        self.start = self.last_log = time.monotonic()
        self.tty = sys.stdout.isatty()

    def add(self, n=1):
        self.done += n
        now = time.monotonic()
        if self.tty or now - self.last_log >= 30 or self.done >= self.total:
            self.last_log = now
            elapsed = now - self.start
            rate = self.done / elapsed if elapsed else 0
            left = (self.total - self.done) / rate if rate else 0
            pct = self.done / self.total if self.total else 1
            bar = "█" * round(pct * 20) + "·" * (20 - round(pct * 20))
            line = (f"  {self.what:8} {bar} {self.done:,}/{self.total:,} ({pct:.0%})  "
                    f"{rate:.1f}/s  {human(left)} left")
            print(("\r\033[K" if self.tty else "") + line, end="" if self.tty else "\n", flush=True)

    def finish(self):
        elapsed = time.monotonic() - self.start
        print(("\r\033[K" if self.tty else "") +
              f"  {self.what:8} done: {self.done:,} shots in {human(elapsed)}", flush=True)


def human(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s" if m else f"{s}s"


def prefetch(paths, load, workers=6):
    """Yield (path, load(path)) in order, loading ahead in threads."""
    with ThreadPoolExecutor(workers) as pool:
        window = []
        it = iter(paths)
        for p in it:
            window.append((p, pool.submit(load, p)))
            if len(window) >= workers * 4:
                p0, f0 = window.pop(0)
                yield p0, f0.result()
        for p0, f0 in window:
            yield p0, f0.result()


# --- Steps -------------------------------------------------------------------------

def step_quality(db, shots):
    """Mean brightness and contrast (0-255), and sharpness (edge variance), for
    spotting black, washed-out or blurry frames."""
    from PIL import Image, ImageFilter, ImageStat

    def measure(path):
        img = Image.open(OUTPUT / path).convert("L")
        img.thumbnail((320, 320))
        stat = ImageStat.Stat(img)
        edges = ImageStat.Stat(img.filter(ImageFilter.FIND_EDGES))
        return stat.mean[0], stat.stddev[0], edges.var[0]

    shots = todo(db, "quality", shots)
    if not shots:
        return
    progress = Progress("quality", len(shots))
    batch = []
    for path, (b, c, s) in prefetch(shots, measure, workers=os.cpu_count() or 4):
        batch.append((b, c, s, path))
        if len(batch) >= 1000:
            db.executemany("UPDATE shots SET brightness=?, contrast=?, sharpness=?, done_quality=1 WHERE path=?", batch)
            db.commit()
            progress.add(len(batch))
            batch = []
    db.executemany("UPDATE shots SET brightness=?, contrast=?, sharpness=?, done_quality=1 WHERE path=?", batch)
    db.commit()
    progress.add(len(batch))
    progress.finish()


def face_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                       providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def step_faces(db, shots):
    """Detect faces (InsightFace RetinaFace) and store a box, score and 512-d
    embedding for each; face_count counts clear, reasonably sized ones."""
    import cv2

    shots = todo(db, "faces", shots)
    if not shots:
        return
    print("  faces    loading the face model (the first time, it downloads about 300 MB)...", flush=True)
    app = face_app()
    progress = Progress("faces", len(shots))
    pending = 0
    for path, img in prefetch(shots, lambda p: cv2.imread(str(OUTPUT / p))):
        count, largest = 0, 0.0
        # In case this shot was partly done before: clear its faces and their embeddings.
        old = [i for (i,) in db.execute("SELECT id FROM faces WHERE path = ?", (path,))]
        if old:
            db.execute(f"DELETE FROM emb.face_embeddings WHERE id IN ({','.join('?' * len(old))})", old)
            db.execute("DELETE FROM faces WHERE path = ?", (path,))
        if img is not None:
            h, w = img.shape[:2]
            for face in app.get(img):
                x1, y1, x2, y2 = (float(v) for v in face.bbox)
                fh = (y2 - y1) / h
                cur = db.execute("INSERT INTO faces (path, x, y, w, h, score) VALUES (?, ?, ?, ?, ?, ?)",
                                 (path, x1 / w, y1 / h, (x2 - x1) / w, fh, float(face.det_score)))
                db.execute("INSERT OR REPLACE INTO emb.face_embeddings (id, vec) VALUES (?, ?)",
                           (cur.lastrowid, blob(face.normed_embedding)))
                if face.det_score >= FACE_MIN_SCORE and fh >= FACE_MIN_SIZE:
                    count += 1
                    largest = max(largest, fh * (x2 - x1) / w)
        db.execute("UPDATE shots SET face_count=?, largest_face=?, done_faces=1 WHERE path=?", (count, largest, path))
        pending += 1
        if pending >= 200:
            db.commit()
            progress.add(pending)
            pending = 0
    db.commit()
    progress.add(pending)
    progress.finish()


def clip_model():
    import open_clip
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL[0], pretrained=CLIP_MODEL[1])
    model = model.to(device).eval()
    return model, preprocess, open_clip.get_tokenizer(CLIP_MODEL[0]), device


def text_embeddings(texts, model, tokenizer, device):
    import torch

    with torch.no_grad():
        t = model.encode_text(tokenizer(texts).to(device))
        return (t / t.norm(dim=-1, keepdim=True)).float().cpu().numpy()


def step_clip(db, shots):
    """CLIP image embeddings, and zero-shot tag scores against TAG_PROMPTS."""
    import numpy as np
    import torch
    from PIL import Image

    todo_shots = todo(db, "clip", shots)
    if not todo_shots:
        tag_shots(db)
        return
    print("  clip     loading the CLIP model (the first time, it downloads about 600 MB)...", flush=True)
    model, preprocess, tokenizer, device = clip_model()
    prompts = list(TAG_PROMPTS)
    prompt_vecs = text_embeddings(prompts, model, tokenizer, device)
    shots = todo_shots
    progress = Progress("clip", len(shots))
    batch_paths, batch_imgs = [], []

    def flush():
        with torch.no_grad():
            x = torch.stack(batch_imgs).to(device)
            e = model.encode_image(x)
            e = (e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        scores = e @ prompt_vecs.T
        for path, v, sc in zip(batch_paths, e, scores):
            tag_scores = {TAG_PROMPTS[p]: round(float(s), 4) for p, s in zip(prompts, sc)}
            db.execute("INSERT OR REPLACE INTO emb.clip_embeddings (path, vec) VALUES (?, ?)", (path, blob(v)))
            db.execute("UPDATE shots SET tag_scores=?, done_clip=1 WHERE path=?", (json.dumps(tag_scores), path))
        db.commit()
        progress.add(len(batch_paths))
        batch_paths.clear()
        batch_imgs.clear()

    for path, img in prefetch(shots, lambda p: preprocess(Image.open(OUTPUT / p).convert("RGB"))):
        batch_paths.append(path)
        batch_imgs.append(img)
        if len(batch_paths) >= 64:
            flush()
    if batch_paths:
        flush()
    progress.finish()
    tag_shots(db)


def tag_shots(db):
    """Tags: for each tag, the shots in the top TAG_TOP of all shots for it, if that
    score also stands out from the shot's other scores by TAG_MARGIN. (CLIP's raw
    scores sit close together, so a fixed threshold would tag nearly everything.)"""
    import numpy as np

    rows = db.execute("SELECT path, tag_scores FROM shots WHERE tag_scores IS NOT NULL").fetchall()
    if not rows:
        return
    names = [t for t in json.loads(rows[0]["tag_scores"]) if t in TAG_PROMPTS.values()]
    scores = np.array([[json.loads(r["tag_scores"]).get(t, 0) for t in names] for r in rows])
    cutoffs = np.quantile(scores, 1 - TAG_TOP, axis=0)
    means = scores.mean(axis=1, keepdims=True)
    keep = (scores >= cutoffs) & (scores - means >= TAG_MARGIN)
    updates = []
    for r, row_scores, row_keep in zip(rows, scores, keep):
        tags = [names[i] for i in np.argsort(-row_scores) if row_keep[i]]
        updates.append((json.dumps(tags), r["path"]))
    db.executemany("UPDATE shots SET tags=? WHERE path=?", updates)
    db.commit()


# --- Who's on screen ------------------------------------------------------------------

def cluster(db, min_cluster=25, crops_per_cluster=24):
    """Group faces into likely people: HDBSCAN within each episode, then the
    episode groups' centres across everything. Writes review/clusters.html."""
    import numpy as np
    from sklearn.cluster import HDBSCAN

    faces = db.execute("SELECT f.id, f.path FROM faces f WHERE f.score >= ? AND f.h >= ?",
                       (FACE_MIN_SCORE, FACE_MIN_SIZE)).fetchall()
    vecs = {i: vec(b) for i, b in db.execute("SELECT id, vec FROM emb.face_embeddings")}
    by_episode = {}
    for f in faces:
        if f["id"] in vecs:  # (a run interrupted at the wrong moment can leave one without)
            by_episode.setdefault(f["path"].rsplit("/", 1)[0], []).append(f["id"])
    print(f"{len(faces)} faces in {len(by_episode)} episodes")

    # 1. within each episode
    groups = []  # (face ids, centre)
    for ids in by_episode.values():
        if len(ids) < 5:
            continue
        x = np.stack([vecs[i] for i in ids])
        labels = HDBSCAN(min_cluster_size=5, metric="euclidean").fit_predict(x)
        for lab in set(labels) - {-1}:
            members = [i for i, l in zip(ids, labels) if l == lab]
            c = x[labels == lab].mean(axis=0)
            groups.append((members, c / np.linalg.norm(c)))
    print(f"{len(groups)} groups within episodes")

    # 2. across episodes
    centres = np.stack([c for _, c in groups])
    labels = HDBSCAN(min_cluster_size=3, metric="euclidean").fit_predict(centres)
    db.execute("UPDATE faces SET cluster = NULL")
    clusters = {}
    for (members, _), lab in zip(groups, labels):
        if lab == -1:
            continue
        clusters.setdefault(int(lab), []).extend(members)
    clusters = {k: v for k, v in clusters.items() if len(v) >= min_cluster}
    # number clusters by size, biggest first
    order = sorted(clusters, key=lambda k: -len(clusters[k]))
    renumbered = {new: clusters[old] for new, old in enumerate(order, 1)}
    for n, members in renumbered.items():
        db.executemany("UPDATE faces SET cluster = ? WHERE id = ?", [(n, i) for i in members])
    db.commit()
    print(f"{len(renumbered)} people-like clusters with at least {min_cluster} faces")
    write_cluster_page(db, renumbered, crops_per_cluster)


def face_crop(path, x, y, w, h, size=96):
    from PIL import Image

    img = Image.open(OUTPUT / path)
    W, H = img.size
    pad = 0.25
    box = (max(0, (x - w * pad) * W), max(0, (y - h * pad) * H),
           min(W, (x + w * (1 + pad)) * W), min(H, (y + h * (1 + pad)) * H))
    crop = img.crop(tuple(round(v) for v in box)).convert("RGB")
    crop.thumbnail((size, size))
    return crop


def write_cluster_page(db, clusters, per):
    import random

    crops_dir = REVIEW / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    names = read_people()
    parts = ["<!doctype html><meta charset=utf-8><title>Face clusters</title>"
             "<style>body{font:14px system-ui;margin:20px} h2{margin:24px 0 6px}"
             "img{width:72px;height:72px;object-fit:cover;margin:1px;border-radius:4px}"
             ".muted{color:#888}</style>",
             "<h1>Face clusters</h1><p>Name the ones you recognise in <code>people.csv</code> "
             "(<code>cluster,name</code>), then run <code>analyse.py label</code>.</p>"]
    for n, members in clusters.items():
        rows = db.execute(f"SELECT id, path, x, y, w, h FROM faces WHERE id IN ({','.join('?' * len(members))})",
                          members).fetchall()
        sample = random.Random(n).sample(rows, min(per, len(rows)))
        episodes = len({r["path"].rsplit("/", 1)[0] for r in rows})
        label = f" — <b>{html.escape(names[n])}</b>" if n in names else ""
        parts.append(f"<h2>Cluster {n}{label}</h2><div class=muted>{len(rows)} faces in {episodes} episodes</div><div>")
        for r in sample:
            f = crops_dir / f"{r['id']}.jpg"
            if not f.exists():
                face_crop(r["path"], r["x"], r["y"], r["w"], r["h"]).save(f, quality=85)
            parts.append(f'<img src="crops/{r["id"]}.jpg" title="{html.escape(r["path"])}">')
        parts.append("</div>")
    (REVIEW / "clusters.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {REVIEW / 'clusters.html'}")


def read_people():
    """{cluster number: name} from people.csv."""
    if not PEOPLE_CSV.exists():
        return {}
    with open(PEOPLE_CSV, newline="", encoding="utf-8") as f:
        return {int(r["cluster"]): r["name"].strip() for r in csv.DictReader(f)
                if r.get("cluster", "").strip().isdigit() and r.get("name", "").strip()}


def label(db):
    """Name every face: the nearest named person's centre, if similar enough; the
    centre of a name is the average of the clusters given that name."""
    import numpy as np

    names = read_people()
    if not names:
        sys.exit(f"No names in {PEOPLE_CSV} (columns: cluster,name).")
    vecs = {i: vec(b) for i, b in db.execute("SELECT id, vec FROM emb.face_embeddings")}
    centres = {}
    for n, name in names.items():
        ids = [i for (i,) in db.execute("SELECT id FROM faces WHERE cluster = ?", (n,))]
        centres.setdefault(name, []).extend(vecs[i] for i in ids if i in vecs)
    people = sorted(centres)
    matrix = np.stack([np.mean(centres[p], axis=0) / np.linalg.norm(np.mean(centres[p], axis=0)) for p in people])
    rows = [r for r in db.execute("SELECT id, path, score, h FROM faces") if r["id"] in vecs]
    ids = [r["id"] for r in rows]
    x = np.stack([vecs[i] for i in ids])
    sims = x @ matrix.T
    best = sims.argmax(axis=1)
    updates, per_shot = [], {}
    for r, b, s in zip(rows, best, sims[np.arange(len(ids)), best]):
        name = people[b] if s >= MATCH_THRESHOLD and r["score"] >= FACE_MIN_SCORE else None
        updates.append((name, r["id"]))
        if name:
            per_shot.setdefault(r["path"], set()).add(name)
    db.executemany("UPDATE faces SET person = ? WHERE id = ?", updates)
    db.execute("UPDATE shots SET people = NULL")
    db.executemany("UPDATE shots SET people = ? WHERE path = ?",
                   [(json.dumps(sorted(v)), p) for p, v in per_shot.items()])
    db.commit()
    counts = {}
    for name, _ in updates:
        if name:
            counts[name] = counts.get(name, 0) + 1
    print(f"Labelled {sum(counts.values())} faces in {len(per_shot)} shots:")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {n}")


# --- Search and stats ---------------------------------------------------------------

def search(db, query, n):
    import numpy as np

    model, _, tokenizer, device = clip_model()
    q = text_embeddings([query], model, tokenizer, device)[0]
    rows = db.execute("SELECT path, vec FROM emb.clip_embeddings").fetchall()
    x = np.stack([vec(r["vec"]) for r in rows])
    sims = x @ q
    top = np.argsort(-sims)[:n]
    REVIEW.mkdir(exist_ok=True)
    rel = os.path.relpath(OUTPUT, REVIEW)
    parts = [f"<!doctype html><meta charset=utf-8><title>{html.escape(query)}</title>"
             "<style>body{font:13px system-ui;margin:20px}figure{display:inline-block;width:320px;margin:6px;"
             "vertical-align:top}img{width:320px;border-radius:4px}figcaption{color:#555}</style>",
             f"<h1>{html.escape(query)}</h1>"]
    for i in top:
        p = rows[i]["path"]
        tags = db.execute("SELECT tags, people FROM shots WHERE path = ?", (p,)).fetchone()
        extra = ", ".join(json.loads(tags["people"] or "[]") + json.loads(tags["tags"] or "[]")) if tags else ""
        parts.append(f'<figure><img src="{html.escape(rel + "/" + p)}" loading=lazy>'
                     f"<figcaption>{sims[i]:.3f} · {html.escape(p.split('/', 2)[-1])}<br>{html.escape(extra)}"
                     "</figcaption></figure>")
    out = REVIEW / "search.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out}")
    return out


def stats(db):
    total = db.execute("SELECT count(*) FROM shots").fetchone()[0]
    for step in ("quality", "faces", "clip"):
        done = db.execute(f"SELECT count(*) FROM shots WHERE done_{step} = 1").fetchone()[0]
        print(f"{step}: {done}/{total}")
    faces = db.execute("SELECT count(*), count(person) FROM faces").fetchone()
    with_faces = db.execute("SELECT count(*) FROM shots WHERE face_count > 0").fetchone()[0]
    print(f"faces: {faces[0]} found, {faces[1]} named; {with_faces} shots with a clear face")
    tags = {}
    for (t,) in db.execute("SELECT tags FROM shots WHERE tags IS NOT NULL"):
        for tag in json.loads(t):
            tags[tag] = tags.get(tag, 0) + 1
    if tags:
        print("top tags:", ", ".join(f"{t} {n}" for t, n in sorted(tags.items(), key=lambda kv: -kv[1])[:15]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="analyse new shots")
    r.add_argument("--steps", default="quality,faces,clip")
    r.add_argument("--limit", type=int, help="only the first N shots (for testing)")
    r.add_argument("--match", help="only shots whose path contains this")
    sub.add_parser("cluster", help="group faces into people; writes review/clusters.html")
    sub.add_parser("label", help="name faces from people.csv")
    s = sub.add_parser("search", help="find shots by description; writes review/search.html")
    s.add_argument("query")
    s.add_argument("-n", type=int, default=60)
    sub.add_parser("stats")
    sub.add_parser("retag", help="recompute tags from stored scores (after changing TAG_MARGIN)")
    args = ap.parse_args()

    db = connect()
    if args.command == "run":
        shots = all_shots()
        if args.match:
            shots = [s for s in shots if args.match in s]
        if args.limit:
            shots = shots[: args.limit]
        steps = {"quality": step_quality, "faces": step_faces, "clip": step_clip}
        chosen = [x.strip() for x in args.steps.split(",")]
        print(f"{len(shots):,} shots. Still to do: " + ", ".join(
            f"{step} {len(todo(db, step, shots)):,}" for step in chosen))
        print("(Safe to stop with Ctrl-C and run again later: it carries on where it left off.)")
        started = time.monotonic()
        for n, step in enumerate(chosen, 1):
            print(f"\n[{n}/{len(chosen)}] {step}", flush=True)
            steps[step](db, shots)
        print(f"\nAll done in {human(time.monotonic() - started)}.")
        stats(db)
    elif args.command == "cluster":
        cluster(db)
    elif args.command == "label":
        label(db)
    elif args.command == "search":
        search(db, args.query, args.n)
    elif args.command == "retag":
        tag_shots(db)
    else:
        stats(db)


if __name__ == "__main__":
    main()
