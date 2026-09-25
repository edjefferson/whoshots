#!/usr/bin/env python3
"""List the iPlayer episode URLs for Doctor Who (2005-2022) and Doctor Who (2023-).

Reads the Redux state embedded in each iPlayer "episodes" page, walking every
series slice and page. Writes a CSV of url, pid, programme and episode subtitle.
"""

import csv
import json
import re
import sys
import time
import urllib.request

BRANDS = [
    "https://www.bbc.co.uk/iplayer/episodes/b006q2x0/doctor-who-20052022",
    "https://www.bbc.co.uk/iplayer/episodes/p0gglvqn/doctor-who",
]
STATE = re.compile(r"window\.__IPLAYER_REDUX_STATE__ = (\{.*?\});</script>")
SKIP_SLICES = {"More Like This"}


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
                    yield ep["id"], ep["title"]["default"], ep["subtitle"]["default"]
            if page >= state["pagination"]["totalPages"]:
                break
            page += 1
            time.sleep(0.5)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "iplayer_episodes.csv"
    seen = set()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "pid", "programme", "episode"])
        for brand in BRANDS:
            for pid, title, subtitle in brand_episodes(brand):
                if pid in seen:
                    continue
                seen.add(pid)
                writer.writerow([f"https://www.bbc.co.uk/iplayer/episode/{pid}", pid, title, subtitle])
    print(f"{len(seen)} episodes -> {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
