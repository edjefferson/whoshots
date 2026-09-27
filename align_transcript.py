#!/usr/bin/env python3
"""Time an untimed transcript using a video's automatic (YouTube) captions.

For specials with no proper subtitles, e.g. Dimensions in Time: the transcript
has the right words but no timings, YouTube's automatic captions have timings
but unreliable words. The transcript's words are matched against the caption
words (like a diff), so each transcript line takes its times from the words it
matched; lines with no match are fitted in between their neighbours.

    align_transcript.py transcript.txt captions.json3 -o review.csv
        writes a CSV to check and edit: n, start, end, speaker, text, match, check
        (match: share of the line's words found in the captions; check: CHECK
        where that's low, so look at those first)
    align_transcript.py --srt review.csv -o subtitles.srt
        turns the (edited) CSV into subtitles for subshots.py -s

Captions come from yt-dlp --write-auto-subs --sub-format json3. The transcript
is a script-style one: "SPEAKER: line", with scene headings, (stage
directions) and title blocks, which are left out; other lines continue the
previous speaker. Times in the CSV are H:MM:SS.mmm; rows with no text are
dropped from the subtitles.
"""

import argparse
import csv
import difflib
import json
import re
import sys
from pathlib import Path

# Words the transcript's source censored.
UNCENSOR = {"t*rd": "TARDIS", "k*ller": "killer", "g*n": "gun"}
SPEAKER = re.compile(r"^((?:\d+(?:st|nd|rd|th) )?[A-Z][A-Z0-9 ']*):\s*(.*)$")
SKIP = re.compile(r"^(\d+\.\s|DIMENSIONS IN TIME$|PART (ONE|TWO|THREE|FOUR)$|by [A-Z]|first broadcast|"
                  r"running time|TO BE CONTINUED)")
MAX_CHARS = 80  # about two lines of subtitle
CHECK_BELOW = 0.6


def read_transcript(path):
    """[(speaker, text)] for each spoken line, in order."""
    lines, speaker, in_direction = [], None, False
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        for censored, word in UNCENSOR.items():
            line = line.replace(censored, word)
        if in_direction:  # a (stage direction) running over several lines
            in_direction = ")" not in line
            continue
        if not line or SKIP.match(line):
            continue
        if line.startswith("(") and ")" not in line:
            in_direction = True
            continue
        text = re.sub(r"\([^)]*\)\.?", "", line)  # stage directions, whole or inline
        if m := SPEAKER.match(text.strip()):
            speaker, text = m[1], m[2]
        text = re.sub(r"\s+", " ", text).strip()
        if text and speaker:
            lines.append((speaker, text))
    return lines


def chunks(text):
    """Split a speech into subtitle-sized pieces: whole sentences where they fit."""
    pieces = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if len(sentence) > MAX_CHARS:  # split long sentences at a comma or semicolon, else by words
            parts = re.split(r"(?<=[,;])\s+", sentence)
            if max(map(len, parts)) > MAX_CHARS:
                words, parts, cur = sentence.split(), [], ""
                for w in words:
                    if cur and len(cur) + len(w) + 1 > MAX_CHARS * 0.6:
                        parts.append(cur)
                        cur = w
                    else:
                        cur = f"{cur} {w}".strip()
                parts.append(cur)
        else:
            parts = [sentence]
        for part in parts:
            if pieces and len(pieces[-1]) + len(part) + 1 <= MAX_CHARS and not pieces[-1].endswith(("?", "!")):
                pieces[-1] += " " + part
            else:
                pieces.append(part)
    return pieces


def norm(word):
    return re.sub(r"[^a-z0-9]", "", word.lower())


def caption_words(path):
    """[(start seconds, word)] from a json3 caption file."""
    words = []
    for ev in json.loads(Path(path).read_text(encoding="utf-8")).get("events", []):
        for seg in ev.get("segs") or []:
            for w in seg.get("utf8", "").split():
                if norm(w):
                    words.append(((ev.get("tStartMs", 0) + seg.get("tOffsetMs", 0)) / 1000, w))
    return words


def clock(seconds):
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h}:{m:02d}:{s:02d}.{ms:03d}"


def seconds(value):
    parts = [float(p) for p in value.strip().split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def align(transcript, captions):
    subs = [(speaker, piece) for speaker, text in transcript for piece in chunks(text)]
    t_words = [(i, w) for i, (_, piece) in enumerate(subs) for w in piece.split() if norm(w)]
    matcher = difflib.SequenceMatcher(None, [norm(w) for _, w in t_words],
                                      [norm(w) for _, w in captions], autojunk=False)
    matched = {}  # transcript word index -> caption word index
    for a, b, size in matcher.get_matching_blocks():
        for k in range(size):
            matched[a + k] = b + k

    rows = []
    for i, (speaker, piece) in enumerate(subs):
        idx = [k for k, (j, _) in enumerate(t_words) if j == i]
        hits = [matched[k] for k in idx if k in matched]
        start = end = None
        if hits:
            start = captions[hits[0]][0]
            # a word ends when the next caption word starts (within reason)
            last = hits[-1]
            nxt = captions[last + 1][0] if last + 1 < len(captions) else captions[last][0] + 0.6
            end = min(nxt, captions[last][0] + 1.0)
        rows.append({"speaker": speaker, "text": piece, "start": start, "end": end,
                     "match": len(hits) / len(idx) if idx else 0})

    # Fit unmatched lines between their neighbours, in proportion to their length.
    i = 0
    while i < len(rows):
        if rows[i]["start"] is not None:
            i += 1
            continue
        j = i
        while j < len(rows) and rows[j]["start"] is None:
            j += 1
        lo = rows[i - 1]["end"] if i else 0.0
        hi = rows[j]["start"] if j < len(rows) else lo + 3 * (j - i)
        total = sum(len(r["text"]) for r in rows[i:j]) or 1
        t = lo
        for r in rows[i:j]:
            r["start"] = t
            t += (hi - lo) * len(r["text"]) / total
            r["end"] = t
        i = j

    # Tidy: no overlaps, and at least about a second on screen where there's room.
    for k, r in enumerate(rows):
        nxt = rows[k + 1]["start"] if k + 1 < len(rows) else None
        r["end"] = max(r["end"], r["start"] + 1.0)
        if nxt is not None:
            r["end"] = min(r["end"], max(nxt - 0.04, r["start"] + 0.3))
    return rows


def write_review(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "start", "end", "speaker", "text", "match", "check"])
        for n, r in enumerate(rows, 1):
            w.writerow([n, clock(r["start"]), clock(r["end"]), r["speaker"], r["text"],
                        f"{r['match']:.0%}", "CHECK" if r["match"] < CHECK_BELOW else ""])
    checks = sum(r["match"] < CHECK_BELOW for r in rows)
    print(f"{len(rows)} subtitles -> {path} ({checks} marked CHECK)")


def review_to_srt(review, path):
    with open(review, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["text"].strip()]
    rows.sort(key=lambda r: seconds(r["start"]))
    with open(path, "w", encoding="utf-8") as f:
        for n, r in enumerate(rows, 1):
            srt_time = lambda v: clock(seconds(v)).replace(".", ",").rjust(12, "0")
            f.write(f"{n}\n{srt_time(r['start'])} --> {srt_time(r['end'])}\n{r['text'].strip()}\n\n")
    print(f"{len(rows)} subtitles -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="transcript and captions, or with --srt the review CSV")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--srt", action="store_true", help="turn an edited review CSV into an SRT")
    args = ap.parse_args()
    if args.srt:
        review_to_srt(args.inputs[0], args.output)
    elif len(args.inputs) == 2:
        write_review(align(read_transcript(args.inputs[0]), caption_words(args.inputs[1])), args.output)
    else:
        sys.exit("Give a transcript and a captions file (or --srt and a review CSV).")


if __name__ == "__main__":
    main()
