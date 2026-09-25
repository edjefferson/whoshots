#!/usr/bin/env python3
"""Take a screenshot of every subtitle in a video.

For each subtitle cue, grabs the frame at the midpoint of the cue's on-screen
time with the subtitle burned in.

Subtitles can come from a track embedded in the video (text or bitmap, e.g.
SRT/ASS/mov_text or DVD/PGS) or from an external subtitle file.

Text subtitles are rendered with ffmpeg's libass `subtitles` filter when your
ffmpeg has it (keeps ASS styling), otherwise they're drawn with Pillow.
Bitmap subtitles are composited with ffmpeg's `overlay` filter.
"""

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

BITMAP_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"}

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


@dataclass
class Cue:
    start: float
    end: float
    text: str = ""

    @property
    def mid(self):
        return (self.start + self.end) / 2


def run(cmd, **kwargs):
    try:
        return subprocess.run(cmd, check=True, capture_output=True, **kwargs)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace").strip() if e.stderr else ""
        raise RuntimeError(f"{cmd[0]} failed: {stderr}") from None


def ffprobe_json(*args):
    return json.loads(run(["ffprobe", "-v", "error", "-of", "json", *args]).stdout)


def ffmpeg_has_filter(name):
    out = run(["ffmpeg", "-hide_banner", "-filters"]).stdout.decode()
    return any(line.split()[1:2] == [name] for line in out.splitlines())


def escape_filter_path(path):
    # Escaping for a value inside an ffmpeg filtergraph option.
    p = str(path).replace("\\", "/")
    for ch in "\\':,[];":
        p = p.replace(ch, "\\" + ch)
    return p


def timestamp(seconds):
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}-{m:02d}-{s:02d}.{ms:03d}"


# --- Loading cues -----------------------------------------------------------

def load_text_events(path):
    import pysubs2

    try:
        subs = pysubs2.load(str(path), encoding="utf-8")
    except UnicodeDecodeError:
        subs = pysubs2.load(str(path), encoding="latin-1")
    events = []
    for ev in subs.events:
        if ev.is_comment or getattr(ev, "is_drawing", False):
            continue
        text = ev.plaintext.strip()
        if text and ev.end > ev.start:
            events.append(Cue(ev.start / 1000, ev.end / 1000, text))
    return subs, events


def text_cues(events):
    """One cue per distinct (start, end); text is everything visible at its midpoint."""
    spans = sorted({(e.start, e.end) for e in events})
    cues = []
    for start, end in spans:
        cue = Cue(start, end)
        visible = [e.text for e in events if e.start <= cue.mid < e.end]
        cue.text = "\n".join(dict.fromkeys(visible))
        cues.append(cue)
    return cues


def bitmap_cues(video, track, start_time):
    data = ffprobe_json(
        "-select_streams", f"s:{track}", "-show_frames",
        "-show_entries", "subtitle=pts_time,start_display_time,end_display_time,num_rects",
        str(video),
    )
    frames = sorted(
        (f for f in data.get("frames", []) if f.get("pts_time") not in (None, "N/A")),
        key=lambda f: float(f["pts_time"]),
    )
    cues = []
    for i, f in enumerate(frames):
        if not int(f.get("num_rects", 0)):
            continue  # "clear screen" event
        pts = float(f["pts_time"])
        start = pts + int(f.get("start_display_time", 0)) / 1000
        next_pts = float(frames[i + 1]["pts_time"]) if i + 1 < len(frames) else None
        edt = int(f.get("end_display_time", 0))
        # PGS often leaves the end time unset/huge and relies on a following clear event.
        if 0 < edt < 60_000:
            end = pts + edt / 1000
            if next_pts is not None:
                end = min(end, next_pts)
        elif next_pts is not None:
            end = next_pts
        else:
            end = start + 2
        if end > start:
            cues.append(Cue(start - start_time, end - start_time))
    return cues


# --- Rendering --------------------------------------------------------------

def base_filters(deinterlace):
    filters = ["bwdif"] if deinterlace else []
    # Square the pixels so anamorphic (e.g. DVD) sources come out at display aspect.
    filters.append("scale=trunc(iw*sar/2)*2:ih,setsar=1")
    return filters


def jpeg_args(out):
    return ["-q:v", "2"] if out.suffix.lower() in (".jpg", ".jpeg") else []


def render_libass(video, cue, out, sub_file, fonts_dir, start_time, deinterlace):
    sub_filter = f"subtitles=filename={escape_filter_path(sub_file)}"
    if fonts_dir:
        sub_filter += f":fontsdir={escape_filter_path(fonts_dir)}"
    # -copyts keeps original timestamps so libass knows what time it is after seeking;
    # subtract the container start time so they line up with the extracted subs.
    filters = base_filters(deinterlace) + [f"setpts=PTS-{start_time}/TB", sub_filter]
    run([
        "ffmpeg", "-v", "error", "-y", "-ss", f"{cue.mid:.3f}", "-copyts", "-i", str(video),
        "-map", "0:v:0", "-vf", ",".join(filters), "-frames:v", "1", *jpeg_args(out), str(out),
    ])


def render_bitmap(video, cue, out, track, deinterlace):
    # Seek to a little before the cue starts so its subtitle packet is decoded,
    # then drop frames up to the midpoint.
    seek = max(0.0, cue.start - 1)
    pre = "bwdif," if deinterlace else ""
    post = ",".join(base_filters(False))
    # Rips are often cropped while the subtitle canvas keeps the original frame
    # size (e.g. 1920x1080 PGS over a 1432x1070 pillarbox crop), so centre it.
    graph = f"[0:v:0]{pre}null[v];[v][0:s:{track}]overlay=x=(W-w)/2:y=(H-h)/2,{post}[out]"
    run([
        "ffmpeg", "-v", "error", "-y", "-ss", f"{seek:.3f}", "-i", str(video),
        "-filter_complex", graph, "-map", "[out]",
        "-ss", f"{cue.mid - seek:.3f}", "-frames:v", "1", *jpeg_args(out), str(out),
    ])


def find_font(explicit):
    if explicit:
        return explicit
    return next((f for f in FONT_CANDIDATES if Path(f).exists()), None)


def wrap_lines(draw, text, font, max_width):
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            trial = f"{line} {word}".strip()
            if line and draw.textlength(trial, font=font) > max_width:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return lines


def render_pillow(video, cue, out, font_path, font_scale, deinterlace):
    from PIL import Image, ImageDraw, ImageFont

    png = run([
        "ffmpeg", "-v", "error", "-ss", f"{cue.mid:.3f}", "-i", str(video),
        "-map", "0:v:0", "-vf", ",".join(base_filters(deinterlace)),
        "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-",
    ]).stdout
    if not png:
        raise RuntimeError(f"no frame at {cue.mid:.3f}s")
    img = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = img.size

    size = max(12, round(h * font_scale))
    font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default(size)
    draw = ImageDraw.Draw(img)
    text = "\n".join(wrap_lines(draw, cue.text, font, w * 0.9))
    draw.multiline_text(
        (w / 2, h - h * 0.05), text, font=font, anchor="md", align="center",
        fill="white", stroke_width=max(2, size // 14), stroke_fill="black",
        spacing=round(size * 0.2),
    )
    save_kwargs = {"quality": 92} if out.suffix.lower() in (".jpg", ".jpeg") else {}
    img.save(out, **save_kwargs)


# --- Main -------------------------------------------------------------------

def list_tracks(sub_streams):
    if not sub_streams:
        print("No subtitle tracks.")
    for i, s in enumerate(sub_streams):
        tags = s.get("tags", {})
        kind = "bitmap" if s.get("codec_name") in BITMAP_CODECS else "text"
        desc = " ".join(x for x in (tags.get("language"), tags.get("title")) if x)
        print(f"  {i}: {s.get('codec_name')} ({kind}) {desc}".rstrip())


def dump_attachments(video, dest):
    """Extract fonts embedded in the container (common in MKVs with ASS subs)."""
    dest.mkdir()
    # ffmpeg complains about having no output but still dumps the attachments.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-dump_attachment:t", "", "-i", str(video)],
        cwd=dest, capture_output=True,
    )
    return dest if any(dest.iterdir()) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="output directory (default: <video name>_shots)")
    ap.add_argument("-s", "--subs", type=Path, help="external subtitle file (.srt, .ass, .vtt, ...)")
    ap.add_argument("-t", "--track", type=int, default=0, help="embedded subtitle track index (default: 0)")
    ap.add_argument("--list-tracks", action="store_true", help="list embedded subtitle tracks and exit")
    ap.add_argument("--renderer", choices=["auto", "libass", "pillow"], default="auto",
                    help="how to draw text subs (auto: libass if ffmpeg supports it)")
    ap.add_argument("-f", "--format", choices=["png", "jpg"], default="png")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4, help="parallel ffmpeg processes")
    ap.add_argument("--deinterlace", action="store_true", help="deinterlace frames (useful for DVD rips)")
    ap.add_argument("--font", help="font file for the Pillow renderer")
    ap.add_argument("--font-scale", type=float, default=0.055,
                    help="Pillow font size as a fraction of frame height (default: 0.055)")
    ap.add_argument("--limit", type=int, help="only do the first N cues")
    args = ap.parse_args()

    if not args.video.exists():
        sys.exit(f"No such file: {args.video}")

    probe = ffprobe_json("-show_streams", "-show_format", str(args.video))
    sub_streams = [s for s in probe["streams"] if s.get("codec_type") == "subtitle"]
    if args.list_tracks:
        list_tracks(sub_streams)
        return
    if not any(s.get("codec_type") == "video" for s in probe["streams"]):
        sys.exit("No video stream found.")
    start_time = float(probe["format"].get("start_time", 0) or 0)

    out_dir = args.output or args.video.with_name(f"{args.video.stem}_shots")
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        renderer = None
        fonts_dir = None

        if args.subs:
            if not args.subs.exists():
                sys.exit(f"No such file: {args.subs}")
            subs, events = load_text_events(args.subs)
            sub_file = tmp / "subs.ass"
            subs.save(str(sub_file))  # normalised copy with a filter-safe path
        else:
            if not sub_streams:
                sys.exit("Video has no subtitle tracks; pass one with --subs.")
            if not 0 <= args.track < len(sub_streams):
                print(f"Track {args.track} doesn't exist. Tracks:", file=sys.stderr)
                list_tracks(sub_streams)
                sys.exit(1)
            codec = sub_streams[args.track].get("codec_name")
            if codec in BITMAP_CODECS:
                renderer = "bitmap"
                cues = bitmap_cues(args.video, args.track, start_time)
            else:
                sub_file = tmp / "subs.ass"
                run(["ffmpeg", "-v", "error", "-y", "-i", str(args.video),
                     "-map", f"0:s:{args.track}", "-c:s", "ass", str(sub_file)])
                _, events = load_text_events(sub_file)
                fonts_dir = dump_attachments(args.video, tmp / "fonts")

        if renderer is None:
            cues = text_cues(events)
            renderer = args.renderer
            if renderer == "auto":
                renderer = "libass" if ffmpeg_has_filter("subtitles") else "pillow"
            elif renderer == "libass" and not ffmpeg_has_filter("subtitles"):
                sys.exit("This ffmpeg has no 'subtitles' filter (built without libass); use --renderer pillow.")

        font_path = find_font(args.font) if renderer == "pillow" else None
        if args.limit:
            cues = cues[: args.limit]
        if not cues:
            sys.exit("No subtitle cues found.")

        def shoot(i, cue):
            out = out_dir / f"{i:04d}_{timestamp(cue.mid)}.{args.format}"
            if renderer == "libass":
                render_libass(args.video, cue, out, sub_file, fonts_dir, start_time, args.deinterlace)
            elif renderer == "bitmap":
                render_bitmap(args.video, cue, out, args.track, args.deinterlace)
            else:
                render_pillow(args.video, cue, out, font_path, args.font_scale, args.deinterlace)
            return out

        print(f"{len(cues)} cues, renderer: {renderer}, output: {out_dir}")
        failures = 0
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = {pool.submit(shoot, i, c): (i, c) for i, c in enumerate(cues, 1)}
            for done, fut in enumerate(as_completed(futures), 1):
                i, cue = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    failures += 1
                    print(f"\n  cue {i} @ {timestamp(cue.mid)} failed: {e}", file=sys.stderr)
                print(f"\r  {done}/{len(cues)}", end="", flush=True)
        print()
        if failures:
            sys.exit(f"{failures} screenshot(s) failed.")


if __name__ == "__main__":
    main()
