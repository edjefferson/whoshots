#!/usr/bin/env python3
"""Work out what's in each screenshot: frame quality, faces (and who they are),
and CLIP tags and search. Runs on the Mac, in its own environment:

    uv venv .venv-analysis --python 3.12
    uv pip install --python .venv-analysis/bin/python -r requirements-analysis.txt

    .venv-analysis/bin/python analyse.py run [--steps quality,faces,clip]
    .venv-analysis/bin/python analyse.py cluster      # group faces into likely people
    .venv-analysis/bin/python analyse.py serve        # name the groups at /people
    .venv-analysis/bin/python analyse.py label        # (or "Apply names" on that page)
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

FACE_MIN_SCORE = 0.6    # detection confidence to count as a face
FACE_MIN_SIZE = 0.03    # a face at least this fraction of the frame height counts
MATCH_THRESHOLD = 0.45  # cosine similarity to a person's centre to be labelled as them
CLIP_MODEL = ("ViT-B-32", "laion2b_s34b_b79k")

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
        -- Tags trained from marked examples (see the web page's /tags).
        CREATE TABLE IF NOT EXISTS tag_defs (
            tag TEXT PRIMARY KEY, prompt TEXT,
            coef BLOB, intercept REAL,  -- the trained classifier (logistic regression on CLIP embeddings)
            accuracy REAL, trained_at TEXT
        );
        CREATE TABLE IF NOT EXISTS tag_labels (tag TEXT, path TEXT, label INTEGER, PRIMARY KEY (tag, path));
        CREATE TABLE IF NOT EXISTS tag_members (tag TEXT, path TEXT, prob REAL, PRIMARY KEY (tag, path));
        CREATE INDEX IF NOT EXISTS tag_members_path ON tag_members (path);
        CREATE INDEX IF NOT EXISTS faces_cluster ON faces (cluster);
        -- Names given on the web page's /people, kept per face so they survive re-clustering:
        -- person = who this face is; not_person = marked as wrong in a group of that name
        -- ('' if the group was unnamed), so it's never named from its group.
        CREATE TABLE IF NOT EXISTS face_marks (face_id INTEGER PRIMARY KEY, person TEXT, not_person TEXT);
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
    first = not (Path.home() / ".insightface/models/buffalo_l").is_dir()
    print("  faces    " + ("downloading the face model (about 300 MB, this time only)..." if first
                         else "loading the face model..."), flush=True)
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)  # an insightface/scikit-image deprecation, every face
    app = face_app()
    progress = Progress("faces", len(shots))
    batch = []  # (path, faces, count, largest), written together so the database isn't locked during detection

    def save():
        for path, faces, count, largest in batch:
            # In case this shot was partly done before: clear its faces and their embeddings.
            old = [i for (i,) in db.execute("SELECT id FROM faces WHERE path = ?", (path,))]
            if old:
                db.execute(f"DELETE FROM emb.face_embeddings WHERE id IN ({','.join('?' * len(old))})", old)
                db.execute("DELETE FROM faces WHERE path = ?", (path,))
            for box, vec in faces:
                cur = db.execute("INSERT INTO faces (path, x, y, w, h, score) VALUES (?, ?, ?, ?, ?, ?)", (path, *box))
                db.execute("INSERT OR REPLACE INTO emb.face_embeddings (id, vec) VALUES (?, ?)", (cur.lastrowid, vec))
            db.execute("UPDATE shots SET face_count=?, largest_face=?, done_faces=1 WHERE path=?", (count, largest, path))
        db.commit()
        progress.add(len(batch))
        batch.clear()

    for path, img in prefetch(shots, lambda p: cv2.imread(str(OUTPUT / p))):
        faces, count, largest = [], 0, 0.0
        if img is not None:
            h, w = img.shape[:2]
            for face in app.get(img):
                x1, y1, x2, y2 = (float(v) for v in face.bbox)
                fh = (y2 - y1) / h
                faces.append(((x1 / w, y1 / h, (x2 - x1) / w, fh, float(face.det_score)), blob(face.normed_embedding)))
                if face.det_score >= FACE_MIN_SCORE and fh >= FACE_MIN_SIZE:
                    count += 1
                    largest = max(largest, fh * (x2 - x1) / w)
        batch.append((path, faces, count, largest))
        if len(batch) >= 200:
            save()
    save()
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
    """CLIP image embeddings, for search and for tags trained on the web page."""
    import torch
    from PIL import Image

    shots = todo(db, "clip", shots)
    if not shots:
        return
    first = not any((Path.home() / ".cache/huggingface/hub").glob("models--*" + CLIP_MODEL[0].replace("-", "*") + "*"))
    print("  clip     " + ("downloading the CLIP model (about 600 MB, this time only)..." if first
                         else "loading the CLIP model..."), flush=True)
    model, preprocess, _, device = clip_model()
    progress = Progress("clip", len(shots))
    batch_paths, batch_imgs = [], []

    def flush():
        with torch.no_grad():
            e = model.encode_image(torch.stack(batch_imgs).to(device))
            e = (e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        db.executemany("INSERT OR REPLACE INTO emb.clip_embeddings (path, vec) VALUES (?, ?)",
                       [(p, blob(v)) for p, v in zip(batch_paths, e)])
        db.executemany("UPDATE shots SET done_clip=1 WHERE path=?", [(p,) for p in batch_paths])
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
    if db.execute("SELECT count(*) FROM tag_defs WHERE coef IS NOT NULL").fetchone()[0]:
        retag(db)  # put the new shots through the trained tags


# --- Tags trained from examples --------------------------------------------------

def clip_matrix(db):
    """(paths, {path: row}, float16 matrix) of every CLIP embedding."""
    import numpy as np

    rows = db.execute("SELECT path, vec FROM emb.clip_embeddings").fetchall()
    paths = [r[0] for r in rows]
    matrix = np.stack([np.frombuffer(r[1], dtype=np.float16) for r in rows]) if rows else None
    return paths, {p: i for i, p in enumerate(paths)}, matrix


def matvec(matrix, v, chunk=50000):
    """matrix @ v, in float32 chunks (the matrix is kept in float16 to save memory)."""
    import numpy as np

    return np.concatenate([matrix[i:i + chunk].astype(np.float32) @ v for i in range(0, len(matrix), chunk)])


def tag_probabilities(tag_def, matrix):
    import numpy as np

    coef = np.frombuffer(tag_def["coef"], dtype=np.float32)
    return 1 / (1 + np.exp(-(matvec(matrix, coef) + tag_def["intercept"])))


def train_tag(db, tag, paths, index, matrix, weak_negatives=3000):
    """Fit a tag's classifier to its marked examples, then tag every shot.

    Marked ✓ and ✗ shots are the training data; some random unmarked shots are
    added as weak negatives, since most shots aren't whatever the tag is, so ✗
    marks are optional (they help with near misses). Marks always win: ✓ shots
    are in the tag and ✗ shots out, whatever it predicts.
    """
    import datetime
    import random

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    labels = {p: l for p, l in db.execute("SELECT path, label FROM tag_labels WHERE tag = ?", (tag,)) if p in index}
    pos = [p for p, l in labels.items() if l == 1]
    neg = [p for p, l in labels.items() if l == 0]
    if len(pos) < 5:
        return {"error": f"Mark at least 5 ✓ first (so far {len(pos)})."}
    rng = random.Random(0)
    weak = [p for p in rng.sample(paths, min(weak_negatives, len(paths))) if p not in labels]
    marked = pos + neg
    x_marked = matrix[[index[p] for p in marked]].astype(np.float32)
    y_marked = np.array([1] * len(pos) + [0] * len(neg))
    x = np.concatenate([x_marked, matrix[[index[p] for p in weak]].astype(np.float32)])
    y = np.concatenate([y_marked, np.zeros(len(weak), dtype=int)])
    w = np.concatenate([np.ones(len(marked)), np.full(len(weak), 0.1)])
    model = LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000)
    model.fit(x, y, sample_weight=w)
    # Accuracy on the marked shots alone, by cross-validation (each shot predicted by a
    # model that didn't see it); it needs some of each, so none until there are 5 ✗.
    accuracy = None
    if len(neg) >= 5:
        folds = min(5, len(pos), len(neg))
        accuracy = float(cross_val_score(LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000),
                                         x_marked, y_marked, cv=StratifiedKFold(folds, shuffle=True, random_state=0)).mean())
    coef = model.coef_[0].astype(np.float32)
    db.execute("UPDATE tag_defs SET coef = ?, intercept = ?, accuracy = ?, trained_at = ? WHERE tag = ?",
               (coef.tobytes(), float(model.intercept_[0]), accuracy,
                datetime.datetime.now().isoformat(timespec="seconds"), tag))
    db.commit()
    members = apply_tag(db, tag, paths, matrix)
    return {"positives": len(pos), "negatives": len(neg), "accuracy": accuracy, "members": members}


def apply_tag(db, tag, paths, matrix):
    """Put every shot through a trained tag; returns how many are in it."""
    tag_def = db.execute("SELECT * FROM tag_defs WHERE tag = ?", (tag,)).fetchone()
    probs = tag_probabilities(tag_def, matrix)
    labels = dict(db.execute("SELECT path, label FROM tag_labels WHERE tag = ?", (tag,)).fetchall())
    members = [(tag, p, float(pr)) for p, pr in zip(paths, probs)
               if labels.get(p, 1 if pr >= 0.5 else 0) == 1]
    db.execute("DELETE FROM tag_members WHERE tag = ?", (tag,))
    db.executemany("INSERT INTO tag_members (tag, path, prob) VALUES (?, ?, ?)", members)
    db.commit()
    rebuild_shot_tags(db)
    return len(members)


def rebuild_shot_tags(db):
    """shots.tags (for copying elsewhere) from the trained tags' members."""
    db.execute("UPDATE shots SET tags = NULL")
    per_shot = {}
    for tag, path in db.execute("SELECT tag, path FROM tag_members ORDER BY tag"):
        per_shot.setdefault(path, []).append(tag)
    db.executemany("UPDATE shots SET tags = ? WHERE path = ?", [(json.dumps(t), p) for p, t in per_shot.items()])
    db.commit()


def retag(db):
    """Re-apply every trained tag to all shots (e.g. after new shots were analysed)."""
    paths, _, matrix = clip_matrix(db)
    if matrix is None:
        return
    rebuild_shot_tags(db)  # clears anything left by older versions
    for (tag,) in db.execute("SELECT tag FROM tag_defs WHERE coef IS NOT NULL").fetchall():
        print(f"  {tag}: {apply_tag(db, tag, paths, matrix):,} shots")


# --- Who's on screen ------------------------------------------------------------------

def cluster(db, min_cluster=25):
    """Group faces into likely people: HDBSCAN within each episode, then the
    episode groups' centres across everything. Name them on the web page's /people."""
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
    print("Name them at http://localhost:8765/people (analyse.py serve).")


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


def face_crop_file(db, face_id):
    """A face's crop for the web page, made the first time it's asked for."""
    f = REVIEW / "crops" / f"{face_id}.jpg"
    if not f.exists():
        r = db.execute("SELECT path, x, y, w, h FROM faces WHERE id = ?", (face_id,)).fetchone()
        if not r:
            return None
        f.parent.mkdir(parents=True, exist_ok=True)
        face_crop(r["path"], r["x"], r["y"], r["w"], r["h"]).save(f, quality=85)
    return f


def name_cluster(db, n, name):
    """Give every face in group n a name (or none), apart from ones marked wrong."""
    name = " ".join((name or "").split())
    ids = "SELECT id FROM faces WHERE cluster = ?"
    if name:
        db.execute(f"""INSERT INTO face_marks (face_id, person) SELECT id, ? FROM faces WHERE cluster = ?
                       AND id NOT IN (SELECT face_id FROM face_marks WHERE not_person IS NOT NULL)
                       ON CONFLICT (face_id) DO UPDATE SET person = excluded.person""", (name, n))
    else:
        db.execute(f"DELETE FROM face_marks WHERE face_id IN ({ids}) AND not_person IS NULL", (n,))
        db.execute(f"UPDATE face_marks SET person = NULL WHERE face_id IN ({ids})", (n,))
    db.commit()


def cluster_names(db):
    """{group: name}: the name most of a group's faces have, if at least half do."""
    sizes = dict(db.execute("SELECT cluster, count(*) FROM faces WHERE cluster IS NOT NULL GROUP BY cluster").fetchall())
    names = {}
    for n, person, count in db.execute("""SELECT f.cluster, m.person, count(*) FROM faces f JOIN face_marks m ON m.face_id = f.id
                                          WHERE f.cluster IS NOT NULL AND m.person IS NOT NULL
                                          GROUP BY f.cluster, m.person ORDER BY count(*)"""):
        if count * 2 >= sizes[n]:
            names[n] = person
    return names


def label(db, chunk=20000):
    """Name every face: faces in named groups get that name; the rest get the
    nearest person's centre (the average of their named faces), if similar enough."""
    import numpy as np

    marked = db.execute("""SELECT m.face_id, m.person, e.vec FROM face_marks m
                           JOIN emb.face_embeddings e ON e.id = m.face_id WHERE m.person IS NOT NULL""").fetchall()
    if not marked:
        return {"error": "No faces named yet: name some groups at /people first."}
    centres = {}
    for r in marked:
        centres.setdefault(r["person"], []).append(vec(r["vec"]))
    people = sorted(centres)
    matrix = np.stack([np.mean(centres[p], axis=0) / np.linalg.norm(np.mean(centres[p], axis=0)) for p in people])
    marks = {r["face_id"]: (r["person"], r["not_person"]) for r in db.execute("SELECT * FROM face_marks")}
    updates, per_shot = [], {}
    rows = db.execute("SELECT f.id, f.path, f.score, e.vec FROM faces f JOIN emb.face_embeddings e ON e.id = f.id")
    while batch := rows.fetchmany(chunk):  # in chunks: there are a lot of faces
        sims = np.stack([vec(r["vec"]) for r in batch]) @ matrix.T
        for r, row_sims in zip(batch, sims):
            person, not_person = marks.get(r["id"], (None, None))
            if not person and r["score"] >= FACE_MIN_SCORE:
                if not_person in people:
                    row_sims[people.index(not_person)] = -1
                best = int(row_sims.argmax())
                person = people[best] if row_sims[best] >= MATCH_THRESHOLD else None
            updates.append((person, r["id"]))
            if person:
                per_shot.setdefault(r["path"], set()).add(person)
    db.executemany("UPDATE faces SET person = ? WHERE id = ?", updates)
    db.execute("UPDATE shots SET people = NULL")
    db.executemany("UPDATE shots SET people = ? WHERE path = ?", [(json.dumps(sorted(v)), p) for p, v in per_shot.items()])
    db.commit()
    counts = {}
    for name, _ in updates:
        if name:
            counts[name] = counts.get(name, 0) + 1
    return {"faces": sum(counts.values()), "shots": len(per_shot),
            "people": sorted(counts.items(), key=lambda kv: -kv[1])}


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
    tags = db.execute("SELECT tag, count(*) FROM tag_members GROUP BY tag ORDER BY count(*) DESC").fetchall()
    if tags:
        print("tags:", ", ".join(f"{t} {n:,}" for t, n in tags))


# --- Web page ------------------------------------------------------------------

def subtitle_texts():
    """{shot path: subtitle text} from the episodes' subtitles.csv."""
    texts = {}
    for path in OUTPUT.glob("*/*/*/subtitles.csv"):
        rel = path.parent.relative_to(OUTPUT)
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["shot"]:
                    texts[(rel / r["shot"]).as_posix()] = r["text"]
    return texts


def serve(_, port):
    """A local web page for browsing and searching the analysed shots (/), for
    training tags from marked examples (/tags) and for naming faces (/people)."""
    import random
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, unquote, urlparse

    import numpy as np

    print("Reading subtitles...", flush=True)
    texts = subtitle_texts()
    lock = threading.Lock()
    cache = {"count": None, "paths": [], "index": {}, "matrix": None, "model": None}

    def clip_vectors(db):
        """All CLIP embeddings (float16, to save memory), reloaded when more have been added."""
        count = db.execute("SELECT max(rowid) FROM emb.clip_embeddings").fetchone()[0]
        with lock:
            if count != cache["count"]:
                cache["paths"], cache["index"], cache["matrix"] = clip_matrix(db)
                cache["count"] = count
            return cache["paths"], cache["index"], cache["matrix"]

    def query_vector(text):
        with lock:
            if cache["model"] is None:
                print("Loading CLIP for search...", flush=True)
                cache["model"] = clip_model()
            model, _, tokenizer, device = cache["model"]
            return text_embeddings([text], model, tokenizer, device)[0]

    def shot_json(db, rows, scores=None, tag=None):
        """The page's view of some shots: subtitle, measurements, faces, tags."""
        paths = [r["path"] for r in rows]
        marks = ",".join("?" * len(paths))
        faces_by, tags_by, labels = {}, {}, {}
        if paths:
            for f in db.execute(f"SELECT * FROM faces WHERE path IN ({marks})", paths):
                faces_by.setdefault(f["path"], []).append({k: f[k] for k in ("x", "y", "w", "h", "score", "cluster", "person")})
            for m in db.execute(f"SELECT tag, path, prob FROM tag_members WHERE path IN ({marks})", paths):
                tags_by.setdefault(m["path"], {})[m["tag"]] = m["prob"]
            if tag:
                labels = dict(db.execute(f"SELECT path, label FROM tag_labels WHERE tag = ? AND path IN ({marks})",
                                         [tag, *paths]).fetchall())
        return [{
            "path": r["path"], "text": texts.get(r["path"], ""), "score": (scores or {}).get(r["path"]),
            "brightness": r["brightness"], "contrast": r["contrast"], "sharpness": r["sharpness"],
            "face_count": r["face_count"], "largest_face": r["largest_face"],
            "tags": tags_by.get(r["path"], {}), "people": json.loads(r["people"] or "[]"),
            "faces": faces_by.get(r["path"], []), "label": labels.get(r["path"]),
        } for r in rows]

    def options(db):
        rows = db.execute("SELECT path, people FROM shots WHERE done_quality OR done_faces OR done_clip").fetchall()
        people, programmes = {}, {}
        for r in rows:
            programmes[r["path"].split("/")[0]] = programmes.get(r["path"].split("/")[0], 0) + 1
            for p in json.loads(r["people"] or "[]"):
                people[p] = people.get(p, 0) + 1
        tags = db.execute("SELECT tag, count(*) FROM tag_members GROUP BY tag ORDER BY count(*) DESC").fetchall()
        clusters = db.execute("SELECT cluster, count(*) FROM faces WHERE cluster IS NOT NULL "
                              "GROUP BY cluster ORDER BY cluster").fetchall()
        return {"analysed": len(rows), "programmes": programmes, "clusters": [list(c) for c in clusters],
                "tags": [list(t) for t in tags], "people": sorted(people.items(), key=lambda kv: -kv[1])}

    def shots(db, q):
        get = lambda k, d="": q.get(k, [d])[0]
        where, params = ["(done_quality OR done_faces OR done_clip)"], []
        if get("programme"):
            where.append("path LIKE ?"); params.append(get("programme") + "/%")
        faces = get("faces")
        if faces == "none":
            where.append("face_count = 0")
        elif faces == "any":
            where.append("face_count > 0")
        elif faces == "2":
            where.append("face_count >= 2")
        if get("tag"):
            where.append("path IN (SELECT path FROM tag_members WHERE tag = ?)"); params.append(get("tag"))
        if get("person"):
            where.append("people LIKE ?"); params.append(f'%"{get("person")}"%')
        if get("cluster"):
            where.append("path IN (SELECT path FROM faces WHERE cluster = ?)"); params.append(int(get("cluster")))
        rows = db.execute(f"SELECT * FROM shots WHERE {' AND '.join(where)}", params).fetchall()
        search = get("q").strip()
        scores = {}
        if search:
            _, index, matrix = clip_vectors(db)
            if matrix is not None:
                sims = matvec(matrix, query_vector(search))
                scores = {r["path"]: float(sims[index[r["path"]]]) for r in rows if r["path"] in index}
                rows = sorted((r for r in rows if r["path"] in scores), key=lambda r: -scores[r["path"]])
        else:
            sort = get("sort", "random")
            keys = {"sharp": lambda r: -(r["sharpness"] or 0), "blurry": lambda r: r["sharpness"] or 0,
                    "bright": lambda r: -(r["brightness"] or 0), "dark": lambda r: r["brightness"] or 0,
                    "faces": lambda r: -(r["largest_face"] or 0)}
            if sort in keys:
                rows = sorted(rows, key=keys[sort])
            else:
                rows = list(rows)
                random.Random(get("seed", "1")).shuffle(rows)
        offset, limit = int(get("offset", "0")), int(get("limit", "60"))
        return {"total": len(rows), "shots": shot_json(db, rows[offset:offset + limit], scores)}

    # --- tag training ---

    def tags(db):
        out = []
        for t in db.execute("SELECT * FROM tag_defs ORDER BY tag"):
            counts = dict(db.execute("SELECT label, count(*) FROM tag_labels WHERE tag = ? GROUP BY label",
                                     (t["tag"],)).fetchall())
            members = db.execute("SELECT count(*) FROM tag_members WHERE tag = ?", (t["tag"],)).fetchone()[0]
            out.append({"tag": t["tag"], "prompt": t["prompt"], "trained": t["coef"] is not None,
                        "accuracy": t["accuracy"], "yes": counts.get(1, 0), "no": counts.get(0, 0),
                        "members": members})
        return out

    def candidates(db, q):
        """Shots to mark for a tag: suggestions from its description, the ones its
        classifier is least sure about, what it currently tags, or what's marked."""
        get = lambda k, d="": q.get(k, [d])[0]
        tag, mode = get("tag"), get("mode", "suggest")
        offset, limit = int(get("offset", "0")), int(get("limit", "40"))
        tag_def = db.execute("SELECT * FROM tag_defs WHERE tag = ?", (tag,)).fetchone()
        if not tag_def:
            return {"error": f"No tag called {tag!r}"}
        paths, index, matrix = clip_vectors(db)
        if matrix is None:
            return {"total": 0, "shots": []}
        labels = dict(db.execute("SELECT path, label FROM tag_labels WHERE tag = ?", (tag,)).fetchall())
        if mode == "marked":
            order = sorted(labels, key=lambda p: -labels[p])
            probs = tag_probabilities(tag_def, matrix) if tag_def["coef"] else None
        else:
            if mode != "suggest" and not tag_def["coef"]:
                return {"total": 0, "shots": []}
            if mode == "suggest":
                probs = None
                score = matvec(matrix, query_vector(tag_def["prompt"] or tag))
                key = -score
            else:
                probs = tag_probabilities(tag_def, matrix)
                key = np.abs(probs - 0.5) if mode == "unsure" else -probs
                if mode == "tagged":
                    key = np.where(probs >= 0.5, key, np.inf)
            order = [paths[i] for i in np.argsort(key) if paths[i] not in labels and np.isfinite(key[i])]
        page = order[offset:offset + limit]
        rows = {r["path"]: r for r in db.execute(
            f"SELECT * FROM shots WHERE path IN ({','.join('?' * len(page))})", page)} if page else {}
        shots_out = shot_json(db, [rows[p] for p in page if p in rows], tag=tag)
        if probs is not None:
            for s_ in shots_out:
                s_["prob"] = float(probs[index[s_["path"]]])
        return {"total": len(order), "shots": shots_out}

    # --- naming faces ---

    def clusters(db, q):
        """Face groups, biggest first, each with some sample faces to recognise them by."""
        get = lambda k, d="": q.get(k, [d])[0]
        show, offset, limit = get("show", "unnamed"), int(get("offset", "0")), int(get("limit", "20"))
        names = cluster_names(db)
        sizes = db.execute("SELECT cluster, count(*) FROM faces WHERE cluster IS NOT NULL GROUP BY cluster ORDER BY cluster").fetchall()
        chosen = [(n, size) for n, size in sizes if show == "all" or (show == "named") == (n in names)]
        out = []
        for n, size in chosen[offset:offset + limit]:
            out.append({"cluster": n, "faces": size, "name": names.get(n), **cluster_faces(db, n, 0, 24)})
        return {"total": len(chosen), "groups": len(sizes), "named": len(names), "clusters": out}

    def cluster_faces(db, n, offset, limit):
        rows = db.execute("""SELECT f.id, f.path, m.not_person IS NOT NULL AS wrong FROM faces f
                             LEFT JOIN face_marks m ON m.face_id = f.id WHERE f.cluster = ?""", (n,)).fetchall()
        episodes = {}
        for r in rows:
            episodes.setdefault(r["path"].rsplit("/", 1)[0], []).append(r)
        # A spread across episodes: one from each (in a random order), then a second from each, ...
        rng = random.Random(n)
        lists = list(episodes.values())
        rng.shuffle(lists)
        for faces in lists:
            rng.shuffle(faces)
        spread = [faces[i] for i in range(max(map(len, lists), default=0)) for faces in lists if i < len(faces)]
        seasons = {}
        for ep, faces in episodes.items():
            season = " · ".join(ep.split("/")[:2])
            seasons[season] = seasons.get(season, 0) + len(faces)
        top = sorted(seasons.items(), key=lambda kv: -kv[1])[:3]
        return {"episodes": len(episodes), "seasons": [[k, v] for k, v in top],
                "sample": [{"id": r["id"], "path": r["path"], "wrong": bool(r["wrong"])} for r in spread[offset:offset + limit]]}

    def people_list(db):
        named = db.execute("SELECT person, count(*) FROM face_marks WHERE person IS NOT NULL GROUP BY person").fetchall()
        labelled = dict(db.execute("SELECT person, count(*) FROM faces WHERE person IS NOT NULL GROUP BY person").fetchall())
        return sorted(({"name": p, "marked": n, "labelled": labelled.get(p, 0)} for p, n in named), key=lambda x: -x["marked"])

    def post(db, path, body):
        if path == "/api/name-cluster":
            name_cluster(db, int(body["cluster"]), body.get("name"))
            return {"ok": True, "name": cluster_names(db).get(int(body["cluster"]))}
        if path == "/api/face-wrong":
            face = int(body["face"])
            n = db.execute("SELECT cluster FROM faces WHERE id = ?", (face,)).fetchone()[0]
            name = cluster_names(db).get(n)
            if body.get("wrong"):
                db.execute("""INSERT INTO face_marks (face_id, person, not_person) VALUES (?, NULL, ?)
                              ON CONFLICT (face_id) DO UPDATE SET person = NULL, not_person = excluded.not_person""",
                           (face, name or ""))
            else:
                db.execute("DELETE FROM face_marks WHERE face_id = ?", (face,))
                if name:
                    db.execute("INSERT INTO face_marks (face_id, person) VALUES (?, ?)", (face, name))
            db.commit()
            return {"ok": True}
        if path == "/api/apply-names":
            with lock:  # one at a time: it's a lot of faces
                return label(db)
        if path == "/api/tags":
            tag = body["tag"].strip().lower()
            if not tag:
                return {"error": "Give the tag a name."}
            db.execute("INSERT OR IGNORE INTO tag_defs (tag, prompt) VALUES (?, ?)", (tag, body.get("prompt") or tag))
            db.execute("UPDATE tag_defs SET prompt = ? WHERE tag = ?", (body.get("prompt") or tag, tag))
            db.commit()
            return {"ok": True, "tag": tag}
        if path == "/api/label":
            if body.get("label") is None:
                db.execute("DELETE FROM tag_labels WHERE tag = ? AND path = ?", (body["tag"], body["path"]))
            else:
                db.execute("INSERT OR REPLACE INTO tag_labels (tag, path, label) VALUES (?, ?, ?)",
                           (body["tag"], body["path"], int(body["label"])))
            db.commit()
            return {"ok": True}
        if path == "/api/train":
            paths, index, matrix = clip_vectors(db)
            return train_tag(db, body["tag"], paths, index, matrix)
        if path == "/api/delete-tag":
            for table in ("tag_defs", "tag_labels", "tag_members"):
                db.execute(f"DELETE FROM {table} WHERE tag = ?", (body["tag"],))
            db.commit()
            rebuild_shot_tags(db)
            return {"ok": True}
        return None

    class Handler(BaseHTTPRequestHandler):
        def send(self, body, kind, status=200):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def reply(self, data):
            if data is None:
                return self.send(b"not found", "text/plain", 404)
            self.send(json.dumps(data).encode(), "application/json", 400 if "error" in data else 200)

        def do_GET(self):
            url = urlparse(self.path)
            try:
                pages = {"/": PAGE, "/tags": TAGS_PAGE, "/people": PEOPLE_PAGE}
                if url.path in pages:
                    return self.send(pages[url.path].encode(), "text/html; charset=utf-8")
                if url.path.startswith("/crop/") and url.path[6:].isdigit():
                    file = face_crop_file(connect(), int(url.path[6:]))
                    return self.send(file.read_bytes(), "image/jpeg") if file else self.send(b"not found", "text/plain", 404)
                if url.path.startswith("/img/"):
                    file = (OUTPUT / unquote(url.path[5:])).resolve()
                    if OUTPUT.resolve() not in file.parents or not file.is_file():
                        return self.send(b"not found", "text/plain", 404)
                    return self.send(file.read_bytes(), "image/png" if file.suffix == ".png" else "image/jpeg")
                db = connect()
                q = parse_qs(url.query)
                handlers = {"/api/options": lambda: options(db), "/api/shots": lambda: shots(db, q),
                            "/api/tags": lambda: tags(db), "/api/candidates": lambda: candidates(db, q),
                            "/api/clusters": lambda: clusters(db, q), "/api/people": lambda: people_list(db),
                            "/api/cluster-faces": lambda: cluster_faces(db, int(q["cluster"][0]), int(q.get("offset", ["0"])[0]), 48)}
                self.reply(handlers[url.path]() if url.path in handlers else None)
            except Exception as e:  # show errors in the page rather than hanging
                self.send(json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json", 500)

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                self.reply(post(connect(), urlparse(self.path).path, body))
            except Exception as e:
                self.send(json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json", 500)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Open http://localhost:{port}  (tags: /tags, naming faces: /people; Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shot inspector</title>
<style>
:root { --bg:#f6f5f2; --panel:#fff; --text:#1d1d1f; --muted:#6e6e73; --line:#e3e1dc; --accent:#0a66c2; --face:#1db954; --weak:#c7c7c7; }
@media (prefers-color-scheme: dark) { :root { --bg:#141414; --panel:#1f1f1f; --text:#ececec; --muted:#9a9a9a; --line:#333; --accent:#5aa2ff; --face:#35d07f; --weak:#666; } }
* { box-sizing:border-box }
body { margin:0; font:14px/1.4 -apple-system, system-ui, sans-serif; background:var(--bg); color:var(--text) }
header { position:sticky; top:0; z-index:5; background:var(--panel); border-bottom:1px solid var(--line); padding:12px 16px; display:flex; flex-wrap:wrap; gap:8px; align-items:center }
header h1 { font-size:16px; margin:0 12px 0 0 }
input, select, button { font:inherit; padding:6px 8px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--text) }
#q { flex:1 1 260px }
button { cursor:pointer } button.primary { background:var(--accent); color:#fff; border-color:var(--accent) }
#status { color:var(--muted); padding:8px 16px }
#grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:12px; padding:0 16px 16px }
.card { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; cursor:pointer }
.pic { position:relative; line-height:0 } .pic img { width:100%; display:block }
.box { position:absolute; border:2px solid var(--face); border-radius:3px } .box.weak { border:1px dashed var(--weak) }
.box span { position:absolute; left:-2px; top:-18px; font-size:11px; line-height:16px; background:var(--face); color:#000; padding:0 4px; border-radius:3px; white-space:nowrap }
.meta { padding:8px 10px } .ep { color:var(--muted); font-size:12px } .sub { margin:4px 0; white-space:pre-line }
.chips { display:flex; flex-wrap:wrap; gap:4px } .chip { font-size:11px; padding:1px 6px; border-radius:10px; background:var(--bg); border:1px solid var(--line) }
.chip.person { border-color:var(--face) } .score { float:right; color:var(--muted); font-size:12px }
#more { display:block; margin:0 auto 24px }
#detail { position:fixed; inset:0; background:rgba(0,0,0,.6); display:none; z-index:10; padding:24px; overflow:auto }
#detail .inner { background:var(--panel); max-width:1100px; margin:0 auto; border-radius:10px; padding:16px; display:grid; grid-template-columns:minmax(0,2fr) minmax(0,1fr); gap:16px }
#detail table { border-collapse:collapse; width:100%; font-size:13px } #detail td { padding:2px 6px; border-bottom:1px solid var(--line) }
#detail .bar { height:6px; background:var(--accent); border-radius:3px }
@media (max-width:760px) { #detail .inner { grid-template-columns:1fr } }
</style></head><body>
<header>
  <h1>Shot inspector</h1><a href="/tags">Train tags →</a> <a href="/people">Name faces →</a>
  <input id="q" placeholder="Describe a shot, e.g. a Dalek in a corridor" autocomplete="off">
  <select id="programme"><option value="">All programmes</option></select>
  <select id="faces"><option value="">Any faces</option><option value="any">With faces</option><option value="2">2+ faces</option><option value="none">No faces</option></select>
  <select id="tag"><option value="">Any tag</option></select>
  <select id="person"><option value="">Anyone</option></select>
  <select id="cluster"><option value="">Any face group</option></select>
  <select id="sort"><option value="random">Random</option><option value="faces">Biggest face</option><option value="sharp">Sharpest</option><option value="blurry">Blurriest</option><option value="bright">Brightest</option><option value="dark">Darkest</option></select>
  <button class="primary" id="go">Show</button>
</header>
<div id="status">Loading…</div>
<div id="grid"></div>
<button id="more" hidden>Show more</button>
<div id="detail"><div class="inner"></div></div>
<script>
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let offset = 0, seed = Math.floor(Math.random() * 1e6), last = [];

function params() {
  const p = new URLSearchParams({ seed, offset, limit: 60 });
  for (const k of ["q", "programme", "faces", "tag", "person", "cluster", "sort"]) if ($("#" + k).value) p.set(k, $("#" + k).value);
  return p;
}
function episode(path) { const parts = path.split("/"); return parts.slice(0, 3).join(" · "); }
function boxes(s) {
  return s.faces.map(f => `<div class="box ${f.score < 0.6 ? "weak" : ""}" style="left:${f.x*100}%;top:${f.y*100}%;width:${f.w*100}%;height:${f.h*100}%">` +
    (f.person || f.cluster ? `<span>${esc(f.person || "group " + f.cluster)}</span>` : "") + `</div>`).join("");
}
function card(s, i) {
  const chips = s.people.map(p => `<span class="chip person">${esc(p)}</span>`).join("") + Object.keys(s.tags).map(t => `<span class="chip">${esc(t)}</span>`).join("");
  return `<div class="card" data-i="${i}"><div class="pic"><img loading="lazy" src="/img/${encodeURI(s.path)}">${boxes(s)}</div>
    <div class="meta">${s.score != null ? `<span class="score">${s.score.toFixed(3)}</span>` : ""}<div class="ep">${esc(episode(s.path))}</div>
    <div class="sub">${esc(s.text)}</div><div class="chips">${chips}</div></div></div>`;
}
async function load(reset) {
  if (reset) { offset = 0; last = []; $("#grid").innerHTML = ""; }
  $("#status").textContent = $("#q").value ? "Searching…" : "Loading…";
  const r = await fetch("/api/shots?" + params()); const data = await r.json();
  if (data.error) { $("#status").textContent = data.error; return; }
  $("#grid").insertAdjacentHTML("beforeend", data.shots.map((s, k) => card(s, last.length + k)).join(""));
  last = last.concat(data.shots); offset += data.shots.length;
  $("#status").textContent = `${data.total.toLocaleString()} shots` + (data.total > offset ? `, showing ${offset}` : "");
  $("#more").hidden = offset >= data.total;
}
function detail(s) {
  const tags = Object.entries(s.tags).sort((a, b) => b[1] - a[1]);
  const n = v => v == null ? "–" : (+v).toFixed(v > 10 ? 0 : 3);
  $("#detail .inner").innerHTML = `<div><div class="pic"><img src="/img/${encodeURI(s.path)}">${boxes(s)}</div>
      <p class="sub">${esc(s.text)}</p><p class="ep">${esc(s.path)}</p></div>
    <div><h3>Frame</h3><table><tr><td>Brightness</td><td>${n(s.brightness)}</td></tr><tr><td>Contrast</td><td>${n(s.contrast)}</td></tr>
      <tr><td>Sharpness</td><td>${n(s.sharpness)}</td></tr><tr><td>Clear faces</td><td>${s.face_count ?? "–"}</td></tr><tr><td>Largest face</td><td>${n(s.largest_face)}</td></tr></table>
    <h3>Faces</h3><table>${s.faces.map(f => `<tr><td>${esc(f.person || (f.cluster ? "group " + f.cluster : "unknown"))}</td><td>score ${f.score.toFixed(2)}</td><td>height ${(f.h*100).toFixed(0)}%</td></tr>`).join("") || "<tr><td>none</td></tr>"}</table>
    <h3>Tags</h3><table>${tags.map(([t, v]) => `<tr><td>${esc(t)}</td><td style="width:45%"><div class="bar" style="width:${Math.max(4, v * 100)}%"></div></td><td>${(v * 100).toFixed(0)}%</td></tr>`).join("") || "<tr><td>none yet (<a href='/tags'>train some</a>)</td></tr>"}</table></div>`;
  $("#detail").style.display = "block";
}
$("#grid").addEventListener("click", e => { const c = e.target.closest(".card"); if (c) detail(last[+c.dataset.i]); });
$("#detail").addEventListener("click", e => { if (e.target.id === "detail") e.currentTarget.style.display = "none"; });
document.addEventListener("keydown", e => { if (e.key === "Escape") $("#detail").style.display = "none"; });
$("#go").onclick = () => { seed = Math.floor(Math.random() * 1e6); load(true); };
$("#q").addEventListener("keydown", e => { if (e.key === "Enter") load(true); });
for (const id of ["programme", "faces", "tag", "person", "cluster", "sort"]) $("#" + id).onchange = () => load(true);
$("#more").onclick = () => load(false);
(async () => {
  const o = await (await fetch("/api/options")).json();
  const fill = (id, items, label) => items.forEach(([v, n]) => $("#" + id).insertAdjacentHTML("beforeend", `<option value="${esc(v)}">${esc(label ? label(v) : v)} (${n})</option>`));
  fill("programme", Object.entries(o.programmes)); fill("tag", o.tags); fill("person", o.people); fill("cluster", o.clusters, v => "Group " + v);
  load(true);
})();
</script></body></html>
"""


TAGS_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tag training</title>
<style>
:root { --bg:#f6f5f2; --panel:#fff; --text:#1d1d1f; --muted:#6e6e73; --line:#e3e1dc; --accent:#0a66c2; --yes:#1f9d55; --no:#d64545; }
@media (prefers-color-scheme: dark) { :root { --bg:#141414; --panel:#1f1f1f; --text:#ececec; --muted:#9a9a9a; --line:#333; --accent:#5aa2ff; --yes:#35c46f; --no:#ff6b6b; } }
* { box-sizing:border-box }
body { margin:0; font:14px/1.4 -apple-system, system-ui, sans-serif; background:var(--bg); color:var(--text); display:grid; grid-template-columns:260px 1fr; min-height:100vh }
aside { background:var(--panel); border-right:1px solid var(--line); padding:14px; position:sticky; top:0; height:100vh; overflow:auto }
aside h1 { font-size:16px; margin:0 0 4px } aside a.back { font-size:13px }
.tag { padding:8px; border-radius:6px; cursor:pointer; margin:2px 0 } .tag:hover { background:var(--bg) } .tag.on { background:var(--bg); outline:1px solid var(--line) }
.tag b { display:block } .tag small { color:var(--muted) }
form { margin-top:14px; display:grid; gap:6px } input, button { font:inherit; padding:6px 8px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--text) }
button { cursor:pointer } .primary { background:var(--accent); color:#fff; border-color:var(--accent) }
main { padding:14px 18px }
.bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:10px }
.modes button.on { background:var(--accent); color:#fff; border-color:var(--accent) }
#msg { color:var(--muted); margin:6px 0 12px }
#grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(230px, 1fr)); gap:10px }
.card { background:var(--panel); border:3px solid transparent; border-radius:8px; overflow:hidden }
.card.yes { border-color:var(--yes) } .card.no { border-color:var(--no); opacity:.6 }
.card img { width:100%; display:block; cursor:zoom-in }
.card .row { display:flex; gap:6px; padding:6px } .card .row button { flex:1; font-size:16px }
.card .sub { padding:0 8px 8px; font-size:12px; color:var(--muted); white-space:pre-line }
.prob { float:right; font-size:12px; color:var(--muted); padding:6px 8px 0 }
.empty { color:var(--muted); padding:30px 0 }
#zoom { position:fixed; inset:0; background:rgba(0,0,0,.8); display:none; place-items:center; z-index:9 } #zoom img { max-width:95vw; max-height:95vh }
@media (max-width:700px) { body { grid-template-columns:1fr } aside { position:static; height:auto } }
</style></head><body>
<aside>
  <h1>Tag training</h1><a class="back" href="/">← Shot inspector</a>
  <div id="tags"></div>
  <form id="new"><b>New tag</b>
    <input id="name" placeholder="name, e.g. dalek" required>
    <input id="prompt" placeholder="description, e.g. a Dalek">
    <button class="primary">Add</button></form>
  <p style="color:var(--muted);font-size:12px">Mark shots ✓ (has it) or ✗ (doesn't); click again to unmark. Start with
  <b>Suggestions</b>; after training, <b>Unsure</b> shows the shots the tag is least certain about, the most useful ones to
  mark. Aim for 20+ of each, then train again. Your marks always override the prediction.</p>
</aside>
<main>
  <div class="bar"><h2 id="title" style="margin:0 12px 0 0">Pick or add a tag</h2>
    <span class="modes"><button data-m="suggest" class="on">Suggestions</button> <button data-m="unsure">Unsure</button>
    <button data-m="tagged">Tagged</button> <button data-m="marked">Marked</button></span>
    <button class="primary" id="train">Train &amp; apply</button> <button id="more">Next page →</button> <button id="del">Delete tag</button></div>
  <div id="msg"></div>
  <div id="grid"></div>
</main>
<div id="zoom"><img></div>
<script>
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let current = null, mode = "suggest", offset = 0;
const api = async (url, body) => { const r = await fetch(url, body ? { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body) } : {}); return r.json(); };

async function loadTags() {
  const tags = await api("/api/tags");
  $("#tags").innerHTML = tags.map(t => `<div class="tag ${t.tag === current ? "on" : ""}" data-t="${esc(t.tag)}"><b>${esc(t.tag)}</b>
    <small>${t.yes} ✓ · ${t.no} ✗ · ${t.trained ? `${t.members.toLocaleString()} tagged` + (t.accuracy != null ? ` · ~${(t.accuracy * 100).toFixed(0)}% accurate` : "") : "not trained yet"}</small></div>`).join("")
    || "<p style='color:var(--muted)'>No tags yet.</p>";
}
async function loadShots() {
  if (!current) return;
  $("#msg").textContent = "Loading…";
  const d = await api(`/api/candidates?tag=${encodeURIComponent(current)}&mode=${mode}&offset=${offset}&limit=40`);
  if (d.error) { $("#msg").textContent = d.error; return; }
  const hint = { suggest: "best matches for the description, not yet marked", unsure: "the shots the tag is least sure about, not yet marked",
                 tagged: "shots the tag currently includes, not yet marked", marked: "everything you've marked" }[mode];
  $("#msg").textContent = `${d.total.toLocaleString()} shots: ${hint}` + (offset ? ` (from #${offset + 1})` : "");
  $("#grid").innerHTML = d.shots.map(s => `<div class="card ${s.label === 1 ? "yes" : s.label === 0 ? "no" : ""}" data-p="${esc(s.path)}">
      ${s.prob != null ? `<span class="prob">${(s.prob * 100).toFixed(1)}% · ${s.prob >= 0.5 ? "in" : "out"}</span>` : ""}
      <div class="row"><button data-l="1" title="has it">✓</button><button data-l="0" title="doesn't">✗</button></div>
      <img loading="lazy" src="/img/${encodeURI(s.path)}"><div class="sub">${esc(s.text)}</div></div>`).join("")
    || `<div class="empty">Nothing here${mode === "unsure" || mode === "tagged" ? " yet: train the tag first" : ""}.</div>`;
}
$("#tags").addEventListener("click", e => { const t = e.target.closest(".tag"); if (!t) return; current = t.dataset.t; offset = 0;
  $("#title").textContent = current; loadTags(); loadShots(); });
$("#grid").addEventListener("click", async e => {
  const card = e.target.closest(".card"); if (!card) return;
  if (e.target.tagName === "IMG") { $("#zoom img").src = e.target.src; $("#zoom").style.display = "grid"; return; }
  const b = e.target.closest("button"); if (!b) return;
  const label = +b.dataset.l, same = card.classList.contains(label ? "yes" : "no");
  card.classList.toggle("yes", !same && label === 1); card.classList.toggle("no", !same && label === 0);
  await api("/api/label", { tag: current, path: card.dataset.p, label: same ? null : label });
  loadTags();
});
$("#zoom").onclick = () => $("#zoom").style.display = "none";
document.querySelectorAll(".modes button").forEach(b => b.onclick = () => {
  mode = b.dataset.m; offset = 0; document.querySelectorAll(".modes button").forEach(x => x.classList.toggle("on", x === b)); loadShots(); });
$("#more").onclick = () => { offset += 40; loadShots(); window.scrollTo(0, 0); };
$("#train").onclick = async () => {
  if (!current) return;
  $("#msg").textContent = "Training…";
  const r = await api("/api/train", { tag: current });
  if (r.error) { $("#msg").textContent = r.error; return; }
  loadTags();
  // New predictions mean a new set of shots to mark: show them from the top.
  if (mode === "suggest") { mode = "unsure"; document.querySelectorAll(".modes button").forEach(x => x.classList.toggle("on", x.dataset.m === mode)); }
  offset = 0; await loadShots(); window.scrollTo(0, 0);
  $("#msg").textContent = `Trained on ${r.positives} ✓ and ${r.negatives} ✗` + (r.accuracy != null ? `: about ${(r.accuracy * 100).toFixed(0)}% accurate on your marks` : "") + `; ${r.members.toLocaleString()} shots tagged. ` + $("#msg").textContent;
};
$("#del").onclick = async () => { if (!current || !confirm(`Delete the tag "${current}" and its marks?`)) return;
  await api("/api/delete-tag", { tag: current }); current = null;
  $("#title").textContent = "Pick or add a tag"; $("#grid").innerHTML = ""; $("#msg").textContent = ""; loadTags(); };
$("#new").onsubmit = async e => { e.preventDefault();
  const r = await api("/api/tags", { tag: $("#name").value, prompt: $("#prompt").value });
  if (r.error) { $("#msg").textContent = r.error; return; }
  current = r.tag; offset = 0; mode = "suggest"; $("#title").textContent = current; $("#name").value = $("#prompt").value = "";
  document.querySelectorAll(".modes button").forEach(x => x.classList.toggle("on", x.dataset.m === "suggest"));
  await loadTags(); loadShots(); };
loadTags();
</script></body></html>
"""


PEOPLE_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Naming faces</title>
<style>
:root { --bg:#f6f5f2; --panel:#fff; --text:#1d1d1f; --muted:#6e6e73; --line:#e3e1dc; --accent:#0a66c2; --yes:#1f9d55; --no:#d64545; }
@media (prefers-color-scheme: dark) { :root { --bg:#141414; --panel:#1f1f1f; --text:#ececec; --muted:#9a9a9a; --line:#333; --accent:#5aa2ff; --yes:#35c46f; --no:#ff6b6b; } }
* { box-sizing:border-box }
body { margin:0; font:14px/1.4 -apple-system, system-ui, sans-serif; background:var(--bg); color:var(--text); display:grid; grid-template-columns:260px 1fr; min-height:100vh }
aside { background:var(--panel); border-right:1px solid var(--line); padding:14px; position:sticky; top:0; height:100vh; overflow:auto }
aside h1 { font-size:16px; margin:0 0 4px } aside a { font-size:13px } .muted { color:var(--muted) }
input, button { font:inherit; padding:6px 8px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--text) }
button { cursor:pointer } .primary { background:var(--accent); color:#fff; border-color:var(--accent) }
.modes { display:flex; gap:4px; margin:12px 0 } .modes button.on { background:var(--accent); color:#fff; border-color:var(--accent) }
.person { display:flex; justify-content:space-between; padding:3px 0; border-bottom:1px solid var(--line); font-size:13px }
main { padding:14px 18px; min-width:0 }
#msg { color:var(--muted); margin:0 0 12px }
.group { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin-bottom:12px }
.group.done { border-color:var(--yes) }
.head { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:8px }
.head .info { flex:1 1 260px; font-size:13px } .head input { width:220px }
.saved { color:var(--yes); font-size:13px }
.faces { display:flex; flex-wrap:wrap; gap:3px }
.face { position:relative; width:72px; height:72px; cursor:pointer } .face img { width:72px; height:72px; object-fit:cover; border-radius:4px; display:block }
.face.wrong img { opacity:.35 } .face.wrong::after { content:"✗"; position:absolute; inset:0; display:grid; place-items:center; color:var(--no); font-size:32px; font-weight:bold }
.more { margin-top:6px; font-size:12px }
#next { display:block; margin:8px auto 30px }
#zoom { position:fixed; inset:0; background:rgba(0,0,0,.8); display:none; place-items:center; z-index:9 } #zoom img { max-width:95vw; max-height:95vh }
@media (max-width:700px) { body { grid-template-columns:1fr } aside { position:static; height:auto } .head input { width:100% } }
</style></head><body>
<aside>
  <h1>Naming faces</h1><a href="/">← Shot inspector</a> · <a href="/tags">Tags</a>
  <div class="modes"><button data-s="unnamed" class="on">Unnamed</button><button data-s="named">Named</button><button data-s="all">All</button></div>
  <p id="counts" class="muted"></p>
  <button class="primary" id="apply" style="width:100%">Apply names to all faces</button>
  <p id="applied" class="muted" style="font-size:12px"></p>
  <p class="muted" style="font-size:12px">Each group is faces that look like the same person, biggest first. Type a name and press
  Enter; give two groups the same name if they're the same person (e.g. in black and white and in colour). Click a face that
  doesn't belong to mark it wrong (click again to undo); shift-click to see the whole shot. Skip anyone you don't know.
  Then <b>Apply names</b>: every face, grouped or not, gets the name of the person it looks most like, if it's close enough.</p>
  <h3 style="font-size:14px">People</h3><div id="people"></div>
</aside>
<main>
  <div id="msg">Loading…</div>
  <div id="groups"></div>
  <button id="next" hidden>Next groups →</button>
</main>
<datalist id="names"></datalist>
<div id="zoom"><img></div>
<script>
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const api = async (url, body) => { const r = await fetch(url, body ? { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body) } : {}); return r.json(); };
let show = "unnamed", offset = 0, shown = 0, total = 0;

const faceHtml = f => `<div class="face ${f.wrong ? "wrong" : ""}" data-id="${f.id}" data-p="${esc(f.path)}" title="${esc(f.path)}"><img loading="lazy" src="/crop/${f.id}"></div>`;
function groupHtml(g) {
  const where = g.seasons.map(([k, n]) => `${esc(k.split(" · ").pop())} (${Math.round(n * 100 / g.faces)}%)`).join(", ");
  return `<section class="group ${g.name ? "done" : ""}" data-n="${g.cluster}" data-named="${g.name ? 1 : 0}" data-loaded="${g.sample.length}">
    <div class="head"><b>Group ${g.cluster}</b><span class="info muted">${g.faces.toLocaleString()} faces in ${g.episodes} episodes · mostly ${where}</span>
    <input list="names" placeholder="Who is this?" value="${esc(g.name || "")}"><button class="save">Save</button><span class="saved"></span></div>
    <div class="faces">${g.sample.map(faceHtml).join("")}</div>
    ${g.faces > g.sample.length ? `<button class="more">More faces</button>` : ""}</section>`;
}
async function load() {
  $("#msg").textContent = "Loading…"; $("#groups").innerHTML = "";
  const d = await api(`/api/clusters?show=${show}&offset=${offset}&limit=20`);
  if (d.error) { $("#msg").textContent = d.error; return; }
  total = d.total; shown = d.clusters.length;
  $("#counts").textContent = `${d.named} of ${d.groups} groups named`;
  $("#msg").textContent = d.groups ? `${d.total} ${show === "all" ? "" : show + " "}groups` + (d.total ? `, showing ${offset + 1}–${offset + shown}` : "")
    : "No face groups yet: run analyse.py cluster once the faces step has finished.";
  $("#groups").innerHTML = d.clusters.map(groupHtml).join("");
  $("#next").hidden = offset + shown >= total;
  window.scrollTo(0, 0);
}
async function loadPeople() {
  const people = await api("/api/people");
  $("#people").innerHTML = people.map(p => `<div class="person"><span>${esc(p.name)}</span><span class="muted">${p.labelled ? p.labelled.toLocaleString() + " faces" : "not applied yet"}</span></div>`).join("")
    || "<p class='muted' style='font-size:13px'>Nobody named yet.</p>";
  $("#names").innerHTML = people.map(p => `<option value="${esc(p.name)}">`).join("");
}
async function save(group) {
  const name = group.querySelector("input").value.trim();
  const r = await api("/api/name-cluster", { cluster: +group.dataset.n, name });
  if (r.error) { group.querySelector(".saved").textContent = r.error; return; }
  group.classList.toggle("done", !!r.name); group.dataset.named = r.name ? 1 : 0;
  group.querySelector(".saved").textContent = r.name ? "✓ saved" : "cleared";
  loadPeople();
}
$("#groups").addEventListener("keydown", e => { if (e.key === "Enter" && e.target.matches("input")) { save(e.target.closest(".group")); e.target.blur(); } });
$("#groups").addEventListener("click", async e => {
  const group = e.target.closest(".group"); if (!group) return;
  if (e.target.matches(".save")) return save(group);
  if (e.target.matches(".more")) {
    const d = await api(`/api/cluster-faces?cluster=${group.dataset.n}&offset=${group.dataset.loaded}`);
    group.querySelector(".faces").insertAdjacentHTML("beforeend", d.sample.map(faceHtml).join(""));
    group.dataset.loaded = +group.dataset.loaded + d.sample.length;
    if (!d.sample.length || d.sample.length < 48) e.target.remove();
    return;
  }
  const face = e.target.closest(".face"); if (!face) return;
  if (e.shiftKey) { $("#zoom img").src = "/img/" + encodeURI(face.dataset.p); $("#zoom").style.display = "grid"; return; }
  const wrong = !face.classList.contains("wrong");
  face.classList.toggle("wrong", wrong);
  await api("/api/face-wrong", { face: +face.dataset.id, wrong });
});
$("#zoom").onclick = () => $("#zoom").style.display = "none";
document.querySelectorAll(".modes button").forEach(b => b.onclick = () => {
  show = b.dataset.s; offset = 0; document.querySelectorAll(".modes button").forEach(x => x.classList.toggle("on", x === b)); load(); });
$("#next").onclick = () => {
  // groups named (or cleared) on this page have left the list, so don't skip past the ones after them
  const left = [...document.querySelectorAll(".group")].filter(g => show !== "all" && (g.dataset.named === "1") !== (show === "named")).length;
  offset += shown - left; load();
};
$("#apply").onclick = async () => {
  $("#applied").textContent = "Applying… (a minute or two)";
  const r = await api("/api/apply-names", {});
  $("#applied").textContent = r.error || `Named ${r.faces.toLocaleString()} faces in ${r.shots.toLocaleString()} shots.`;
  loadPeople();
};
load(); loadPeople();
</script></body></html>
"""


def is_classic(path):
    return path.startswith(("Doctor Who (1963", "Doctor Who (1993", "Doctor Who (1996"))


def sample_shots(shots, n, seed):
    """n random shots from the classic series (incl. the 1990s specials) and n from the new series."""
    import random

    rng = random.Random(seed)
    classic = [s for s in shots if is_classic(s)]
    new = [s for s in shots if not is_classic(s)]
    return sorted(rng.sample(classic, min(n, len(classic))) + rng.sample(new, min(n, len(new))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="analyse new shots")
    r.add_argument("--steps", default="quality,faces,clip")
    r.add_argument("--limit", type=int, help="only the first N shots (for testing)")
    r.add_argument("--match", help="only shots whose path contains this")
    r.add_argument("--sample", type=int, metavar="N",
                   help="only N random shots from the classic series and N from the new series")
    r.add_argument("--seed", type=int, default=1, help="for --sample (default: 1); change it for different shots")
    sub.add_parser("cluster", help="group faces into likely people, to name at /people")
    sub.add_parser("label", help="name every face from the groups named at /people")
    s = sub.add_parser("search", help="find shots by description; writes review/search.html")
    s.add_argument("query")
    s.add_argument("-n", type=int, default=60)
    sub.add_parser("stats")
    sub.add_parser("retag", help="re-apply the trained tags to every shot (e.g. after analysing new ones)")
    v = sub.add_parser("serve", help="a web page to browse and search the analysed shots")
    v.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    db = connect()
    if args.command == "run":
        shots = all_shots()
        if args.match:
            shots = [s for s in shots if args.match in s]
        if args.sample:
            shots = sample_shots(shots, args.sample, args.seed)
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
        result = label(db)
        if "error" in result:
            sys.exit(result["error"])
        print(f"Labelled {result['faces']:,} faces in {result['shots']:,} shots:")
        for name, n in result["people"]:
            print(f"  {name}: {n:,}")
    elif args.command == "search":
        search(db, args.query, args.n)
    elif args.command == "serve":
        serve(db, args.port)
    elif args.command == "retag":
        retag(db)
    else:
        stats(db)


if __name__ == "__main__":
    main()
