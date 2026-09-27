#!/usr/bin/env python3
"""List the iPlayer episode URLs for Doctor Who (2005-2022) and Doctor Who (2023-),
plus any single episodes listed in iplayer_extra.txt (e.g. classic episodes).

Reads the Redux state embedded in each iPlayer "episodes" page, walking every
series slice and page. Writes a CSV of url, pid, programme, series (the iPlayer
series tab it is listed under) and episode subtitle.

iplayer_extra.txt has one iPlayer episode URL per line (# starts a comment).
"""

import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

BRANDS = [
    "https://www.bbc.co.uk/iplayer/episodes/b006q2x0/doctor-who-20052022",
    "https://www.bbc.co.uk/iplayer/episodes/p0gglvqn/doctor-who",
]
STATE = re.compile(r"window\.__IPLAYER_REDUX_STATE__ = (\{.*?\});</script>")
SKIP_SLICES = {"More Like This"}
EXTRA = Path(__file__).parent / "iplayer_extra.txt"
EPISODE_PID = re.compile(r"/iplayer/episode/([a-z0-9]+)")


def fetch_state(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")
    return json.loads(STATE.search(html).group(1))


def brand_episodes(brand_url):
    state = fetch_state(brand_url)
    slices = [s for s in state["header"].get("availableSlices") or [] if s["title"] not in SKIP_SLICES]
    if not slices:
        slices = [{"id": None, "title": ""}]
    for sl in slices:
        page = 1
        while True:
            params = [f"seriesId={sl['id']}"] if sl["id"] else []
            params.append(f"page={page}")
            state = fetch_state(f"{brand_url}?{'&'.join(params)}")
            for item in state["entities"]["results"]:
                ep = item.get("episode")
                if ep:
                    yield ep["id"], ep["title"]["default"], sl["title"], ep["subtitle"]["default"]
            if page >= state["pagination"]["totalPages"]:
                break
            page += 1
            time.sleep(0.5)


def extra_episodes():
    """The episodes listed by URL in iplayer_extra.txt."""
    if not EXTRA.exists():
        return
    for line in EXTRA.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if not (m := EPISODE_PID.search(line)):
            print(f"Not an iPlayer episode URL, skipping: {line}", file=sys.stderr)
            continue
        ep = fetch_state(f"https://www.bbc.co.uk/iplayer/episode/{m[1]}")["episode"]
        # e.g. "Season 3: The Daleks' Master Plan: The Nightmare Begins"
        series = ep["subtitle"].split(":", 1)[0] if ":" in ep["subtitle"] else ""
        yield ep["id"], ep["title"], series, ep["subtitle"]
        time.sleep(0.5)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "iplayer_episodes.csv"
    seen = set()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "pid", "programme", "series", "episode"])
        episodes = [e for brand in BRANDS for e in brand_episodes(brand)] + list(extra_episodes())
        for pid, title, series, subtitle in episodes:
            if pid in seen:
                continue
            seen.add(pid)
            writer.writerow([f"https://www.bbc.co.uk/iplayer/episode/{pid}", pid, title, series, subtitle])
    print(f"{len(seen)} episodes -> {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
