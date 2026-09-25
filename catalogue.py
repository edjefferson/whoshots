#!/usr/bin/env python3
"""Build a CSV of every classic Doctor Who episode, matched to a file in a library.

For each episode, picks the best file (original effects / TV editions over
updated or extended versions; for missing episodes, reconstructions over
animations) and records whether it has subtitle tracks.

Episode list comes from TVmaze (cached in .cache/). Extras folders are ignored.
"""

import argparse
import csv
import difflib
import json
import re
import subprocess
import sys
import threading
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

TVMAZE_URL = "https://api.tvmaze.com/shows/766/episodes?specials=1"
CACHE = Path(__file__).parent / ".cache" / "tvmaze_episodes.json"
PROBE_CACHE = CACHE.with_name("probe.json")

VIDEO_EXTS = {".mkv", ".mp4", ".m4v", ".avi"}
EXTRAS_DIR = re.compile(r"extras|special features|bonus|behind the sofa|audio items|pdfs|documentaries", re.I)

# Episodes missing from the BBC archive: (serial, parts).
# The Daleks' Master Plan 1 and 3 have since been recovered.
MISSING = {
    "Marco Polo": range(1, 8),
    "The Reign of Terror": [4, 5],
    "The Crusade": [2, 4],
    "Galaxy 4": [1, 2, 4],
    "Mission to the Unknown": [1],
    "The Myth Makers": range(1, 5),
    "The Daleks' Master Plan": [4, 6, 7, 8, 9, 11, 12],
    "The Massacre of St Bartholomew's Eve": range(1, 5),
    "The Celestial Toymaker": [1, 2, 3],
    "The Savages": range(1, 5),
    "The Smugglers": range(1, 5),
    "The Tenth Planet": [4],
    "The Power of the Daleks": range(1, 7),
    "The Highlanders": range(1, 5),
    "The Underwater Menace": [1, 4],
    "The Moonbase": [1, 3],
    "The Macra Terror": range(1, 5),
    "The Faceless Ones": [2, 4, 5, 6],
    "The Evil of the Daleks": [1, 3, 4, 5, 6, 7],
    "The Abominable Snowmen": [1, 3, 4, 5, 6],
    "The Ice Warriors": [2, 3],
    "The Web of Fear": [3],
    "Fury from the Deep": range(1, 7),
    "The Wheel in Space": [1, 2, 4, 5],
    "The Invasion": [1, 4],
    "The Space Pirates": [1, 3, 4, 5, 6],
}

NOTES = {
    "Shada": "never broadcast; 2017 version completing the surviving footage with animation (1992 VHS version is in extras)",
    "Mission to the Unknown": "2019 fan recreation, not the original",
}
# Notes for individual episodes: (serial, part).
EPISODE_NOTES = {
    ("The Daleks' Master Plan", 1): "recently recovered; file name has the wrong episode title (Devil's Planet)",
    ("The Daleks' Master Plan", 3): "recently recovered; file name has the wrong episode title (The Nightmare Begins)",
}

NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen".split())}

RECON = re.compile(r"recon|telesnap|loose cannon", re.I)
ANIMATION = re.compile(r"animat", re.I)
# (pattern, score, label). Positive = preferred.
VERSION_RULES = [
    (r"original effects", 20, "original effects"),
    (r"(updated|new|cgi) effects", -50, "updated effects"),
    (r"tv version", 20, "TV version"),
    (r"original (broadcast|edition|version)", 20, "original version"),
    (r"special edition", -30, "special edition"),
    (r"extended", -30, "extended"),
    (r"alternat(e|ive) (edit|edition)", -30, "alternate edit"),
    (r"remix", -30, "remix"),
    (r"vhs", -30, "VHS version"),
    (r"workprint", -30, "workprint"),
    (r"omnibus", -30, "omnibus"),
    (r"anniversary edition", -30, "anniversary edition"),
    (r"blu-?ray", 2, "Blu-ray"),
    (r"dvd", 0, "DVD"),
]


@dataclass
class Episode:
    season: int
    serial: str
    part: int
    title: str
    airdate: str
    missing: bool = False
    candidates: list = field(default_factory=list)


@dataclass
class Candidate:
    path: Path
    kind: str  # original / recon / animation
    labels: list
    score: int
    subs: list = None  # [(index, codec, language, title)]

    @property
    def dialogue_track(self):
        """Index of the first subtitle track that isn't an info-text track."""
        for i, _, _, title in self.subs or []:
            if "info" not in (title or "").lower():
                return i
        return None


def norm(s):
    s = s.lower().replace("æ", "ae").replace("&", "and").replace("_", " ")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"^doctor who and ", "", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\b(the|a|an|of|and)\b", " ", s).split()


def part_number(word):
    return int(word) if word.isdigit() else NUMBER_WORDS.get(word.lower())


# --- Episode list -----------------------------------------------------------

def load_tvmaze(refresh):
    if refresh or not CACHE.exists():
        CACHE.parent.mkdir(exist_ok=True)
        with urllib.request.urlopen(TVMAZE_URL) as r:
            CACHE.write_bytes(r.read())
    return json.loads(CACHE.read_text())


def episode_list(raw):
    episodes = []
    for e in raw:
        name, season = e["name"], e["season"]
        if e["type"] == "regular":
            if m := re.match(r"(.+), Part (\w+) \((.+)\)$", name):  # Trial of a Time Lord
                serial, part, title = m[1], part_number(m[2]), m[3]
            elif m := re.match(r"(.+) \((.+), Part (\w+)\)$", name):
                title, serial, part = m[1], m[2], part_number(m[3])
            elif m := re.match(r"(.+), Part (\w+)$", name):
                serial, part, title = m[1], part_number(m[2]), ""
            else:
                serial, part, title = name, 1, ""
        elif name in ("The Five Doctors", "Doctor Who: The Movie"):
            serial, part, title = name, 1, ""
        else:
            continue
        missing = part in MISSING.get(serial, ())
        episodes.append(Episode(season, serial, part, title, e["airdate"] or "", missing))
    return episodes


# --- Files ------------------------------------------------------------------

def episode_files(root):
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if (p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("._")
                and not any(EXTRAS_DIR.search(d) for d in rel.parts[:-1])):
            yield p


def classify(path, missing):
    """Return (kind, labels, score) for a file."""
    text = " ".join(path.relative_to(path.parents[1]).parts)
    labels = []
    if RECON.search(text) or path.parent.name.lower() == "telesnap":
        kind, score = "recon", 200
    elif re.search(r"recreation", text, re.I):
        kind, score = "recreation", 150
    elif ANIMATION.search(text) or path.parent.name.lower() in ("colour", "b&w") or (
            missing and re.search(r"colou?r|b&w", text, re.I)):
        kind, score = "animation", 100
        # Prefer the black-and-white animation, as the episodes were broadcast.
        if re.search(r"b[&_]w", text, re.I):
            score += 5
            labels.append("B&W")
        elif re.search(r"colou?r", text, re.I):
            labels.append("colour")
    elif missing:
        # Unlabelled file for a missing episode: DVD releases filled these gaps with animations.
        kind, score = "animation", 100
        labels.append("presumed, unlabelled")
    else:
        kind, score = "original", 300
        if re.search(r"colou?r", text, re.I):
            score -= 10
            labels.append("colourised")
    for pattern, points, label in VERSION_RULES:
        if re.search(pattern, text, re.I):
            score += points
            labels.append(label)
    return kind, labels, score


def match_serial(folder, season_serials):
    words = norm(re.sub(r"^(\d+ - |S\d+ \d+ )", "", folder))
    words = [w for w in words if not re.fullmatch(r"\d+", w)] or words
    best = max(season_serials, key=lambda s: difflib.SequenceMatcher(None, norm(s), words).ratio())
    ratio = difflib.SequenceMatcher(None, norm(best), words).ratio()
    return best if ratio >= 0.6 else None


def assign(root, episodes):
    by_key = {(e.season, e.serial, e.part): e for e in episodes}
    by_number = {}
    for season in {e.season for e in episodes}:
        regular = [e for e in episodes if e.season == season and e.serial not in
                   ("The Five Doctors", "Doctor Who: The Movie")]
        for i, e in enumerate(regular, 1):
            by_number[(season, i)] = e
    unmatched = []

    for path in episode_files(root):
        rel = path.relative_to(root)
        season_m = re.match(r"Season (\d+)$", rel.parts[0])
        if not season_m or len(rel.parts) < 3:
            unmatched.append((rel, "not in a season/serial folder"))
            continue
        season, folder = int(season_m[1]), rel.parts[1]
        name = path.stem

        ep = None
        if (m := re.search(r"S(\d+)E(\d+)", name)) and int(m[1]) > 0:
            ep = by_number.get((int(m[1]), int(m[2])))
        else:
            serials = sorted({e.serial for e in episodes if e.season == season})
            serial = match_serial(folder, serials)
            if serial is None:
                unmatched.append((rel, "unknown serial"))
                continue
            part = None
            for pat in (r"\bPart (\w+)", r"\bEpisode (\d+)", r" - (\d+) - "):
                if (m := re.search(pat, name, re.I)) and (part := part_number(m[1])):
                    break
            if part is None and sum(e.serial == serial for e in episodes) == 1:
                part = 1
            # e.g. "The Trial Of A Time Lord 9-12" folder holding a "Part 1" of Vervoids.
            if part and (r := re.search(r"(\d+)-(\d+)$", folder)) and part < int(r[1]):
                part += int(r[1]) - 1
            ep = by_key.get((season, serial, part)) if part else None
            if ep is None:
                unmatched.append((rel, "no part number" if not part else f"no {serial} part {part}"))
                continue
        if ep is None:
            unmatched.append((rel, "episode number not in list"))
            continue
        kind, labels, score = classify(path, ep.missing)
        ep.candidates.append(Candidate(path, kind, labels, score))
    return unmatched


_probe_cache = {}
_probe_lock = threading.Lock()


def probe_subs(path):
    st = path.stat()
    key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    with _probe_lock:
        if key in _probe_cache:
            return [tuple(t) for t in _probe_cache[key]]
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True)
    try:
        streams = json.loads(out.stdout).get("streams", [])
    except json.JSONDecodeError:
        return []
    tracks = [(i, s.get("codec_name"), s.get("tags", {}).get("language", ""), s.get("tags", {}).get("title", ""))
              for i, s in enumerate(streams)]
    with _probe_lock:
        _probe_cache[key] = tracks
    return tracks


# --- Main -------------------------------------------------------------------

def describe(c):
    return ", ".join([c.kind] + c.labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="library folder containing 'Season N' folders")
    ap.add_argument("-o", "--output", type=Path, default=Path("episodes.csv"))
    ap.add_argument("--refresh", action="store_true", help="re-download the TVmaze episode list")
    ap.add_argument("-j", "--jobs", type=int, default=16, help="parallel ffprobe processes")
    args = ap.parse_args()

    if PROBE_CACHE.exists():
        _probe_cache.update(json.loads(PROBE_CACHE.read_text()))
    episodes = episode_list(load_tvmaze(args.refresh))
    unmatched = assign(args.root, episodes)

    # Special case: condensed Marco Polo recon filed with The Edge of Destruction extras.
    marco = next(args.root.glob("Season 1/*Edge of Destruction*/*Special Features*/Marco Polo Recon.*"), None)

    candidates = [c for e in episodes for c in e.candidates]
    with ThreadPoolExecutor(args.jobs) as pool:
        for c, subs in zip(candidates, pool.map(lambda c: probe_subs(c.path), candidates)):
            c.subs = subs
    marco_subs = probe_subs(marco) if marco else []
    PROBE_CACHE.write_text(json.dumps(_probe_cache))

    rows = []
    for e in episodes:
        # Best version first; among equals, prefer one with dialogue subtitles.
        ranked = sorted(e.candidates, key=lambda c: (c.score, c.dialogue_track is not None), reverse=True)
        best = ranked[0] if ranked else None
        rows.append({
            "season": e.season,
            "serial": e.serial,
            "part": e.part,
            "title": e.title,
            "airdate": e.airdate,
            "missing_from_archive": "yes" if e.missing else "",
            "file": str(best.path.relative_to(args.root)) if best else "",
            "version": describe(best) if best else "no file",
            "has_subs": ("yes" if best.dialogue_track is not None else "no") if best else "",
            "subs_track": best.dialogue_track if best and best.dialogue_track is not None else "",
            "subtitle_tracks": "; ".join(f"{i}: {codec} {lang} {title}".strip() for i, codec, lang, title in best.subs)
                               if best else "",
            "other_versions": "; ".join(
                describe(c) + (" [subs]" if c.dialogue_track is not None else "") for c in ranked[1:]),
            "notes": EPISODE_NOTES.get((e.serial, e.part)) or NOTES.get(e.serial, ""),
        })
    if marco:
        dialogue = next((i for i, _, _, t in marco_subs if "info" not in (t or "").lower()), None)
        rows.append({
            "season": 1, "serial": "Marco Polo", "part": "1-7", "title": "(condensed reconstruction)",
            "airdate": "1964-02-22", "missing_from_archive": "yes",
            "file": str(marco.relative_to(args.root)), "version": "recon, condensed, special features",
            "has_subs": "yes" if dialogue is not None else "no",
            "subs_track": dialogue if dialogue is not None else "",
            "subtitle_tracks": "; ".join(f"{i}: {c} {l} {t}".strip() for i, c, l, t in marco_subs),
            "other_versions": "",
            "notes": "condensed ~30 min recon of the whole serial, filed with The Edge of Destruction extras",
        })

    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    matched = sum(bool(r["file"]) for r in rows)
    subs = sum(r["has_subs"] == "yes" for r in rows)
    print(f"{len(rows)} rows, {matched} with a file, {subs} with subtitles -> {args.output}")
    if unmatched:
        print(f"\n{len(unmatched)} files not matched to an episode:", file=sys.stderr)
        for rel, why in unmatched:
            print(f"  {rel}  ({why})", file=sys.stderr)


if __name__ == "__main__":
    main()
