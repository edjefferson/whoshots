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
import csv
import io
import json
import os
import re
import shutil
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


# --- Frame choice -----------------------------------------------------------

SHARPNESS_SIZE = (640, 480)


def sharpness(gray_bytes):
    from PIL import Image, ImageFilter, ImageStat

    img = Image.frombytes("L", SHARPNESS_SIZE, gray_bytes)
    return ImageStat.Stat(img.filter(ImageFilter.FIND_EDGES)).var[0]


def window_bounds(cue, window):
    """(start, end) of the frames to choose from around the cue's midpoint, or
    None to just use the frame at the midpoint. Stays inside the cue so the
    subtitle is still showing. A very short cue (a subtitle flashed up for a
    frame or two) leaves a window too small to hold a frame, so that uses the
    midpoint too."""
    lo = max(cue.start + 0.05, cue.mid - window)
    hi = min(cue.end - 0.05, cue.mid + window)
    return (lo, hi) if window > 0 and hi - lo >= 0.1 else None


def grab_sharpest(video, cue, window, deinterlace, workdir, track=None, sub_image=None):
    """Return (time, image) of the sharpest frame near the cue's midpoint.

    The exact midpoint often lands on a motion-blurred frame; nearby frames can
    be much crisper. One ffmpeg run decodes the window once, writing every
    frame at output size plus a small greyscale copy of each to score. With
    track, the bitmap subtitle is overlaid on the frames and, with sub_image,
    also saved on its own for OCR.
    """
    from PIL import Image

    bounds = window_bounds(cue, window)
    lo, hi = bounds or (cue.mid, cue.mid + 0.5)
    single = ["-frames:v", "1"] if bounds is None else []
    w, h = SHARPNESS_SIZE
    pre = "bwdif," if deinterlace else ""
    post = ",".join(base_filters(False))
    gray = f"scale={w}:{h},format=gray,showinfo"
    if track is None:
        seek = lo
        inputs = ["-ss", f"{lo:.3f}", "-t", f"{hi - lo:.3f}", *HWACCEL, "-i", str(video)]
        graph = f"[0:v:0]{pre}{post},split[full][g];[g]{gray}[gray]"
    else:
        # Start a little before the cue so its subtitle packet is decoded, then
        # trim to the window after overlaying it.
        seek = max(0.0, cue.start - 1)
        inputs = ["-ss", f"{seek:.3f}", *HWACCEL, "-i", str(video)]
        trim = f"trim=start={lo - seek:.3f}:end={hi - seek:.3f}"
        subs = f"[0:s:{track}]"
        graph = f"[0:v:0]{pre}split[va][vb];"
        if sub_image:
            graph += f"{subs}split[s1][s2];[s2]format=rgba[subimg];"
            subs = "[s1]"
        # Rips are often cropped while the subtitle canvas keeps the original frame
        # size (e.g. 1920x1080 PGS over a 1432x1070 pillarbox crop), so centre it.
        graph += (f"[va]{subs}overlay=x=(W-w)/2:y=(H-h)/2,{trim},{post}[full];"
                  f"[vb]{trim},{gray}[gray]")
    workdir.mkdir()
    cmd = [
        "ffmpeg", "-hide_banner", "-y", *inputs, "-filter_complex", graph,
        "-map", "[full]", "-fps_mode", "passthrough", *single, str(workdir / "f%04d.bmp"),
        "-map", "[gray]", "-fps_mode", "passthrough", *single, "-f", "rawvideo", str(workdir / "gray.raw"),
    ]
    if sub_image:
        cmd += ["-map", "[subimg]", "-ss", f"{cue.mid - seek:.3f}", "-frames:v", "1", str(sub_image)]
    try:
        proc = run(cmd)
        times = [float(x) for x in re.findall(r"pts_time:\s*(-?[\d.]+)", proc.stderr.decode())]
        raw = (workdir / "gray.raw").read_bytes()
        size = w * h
        grays = [raw[i:i + size] for i in range(0, len(raw) - size + 1, size)]
        fulls = sorted(workdir.glob("f*.bmp"))
        n = min(len(grays), len(fulls), len(times))
        if not n:
            raise RuntimeError(f"no frames around {cue.mid:.3f}s")
        best = max(range(n), key=lambda i: sharpness(grays[i]))
        img = Image.open(fulls[best]).convert("RGB")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return seek + times[best], img


def save_image(img, out):
    img.save(out, **({"quality": 80} if out.suffix.lower() in (".jpg", ".jpeg") else {}))


def sharpest_frame_time(video, cue, window, start_time, deinterlace):
    """Time of the sharpest frame within `window` seconds of the cue's midpoint.

    The exact midpoint often lands on a motion-blurred frame; nearby frames can
    be much crisper. Stays inside the cue so the subtitle is still showing.
    """
    bounds = window_bounds(cue, window)
    if bounds is None:
        return cue.mid
    lo, hi = bounds
    w, h = SHARPNESS_SIZE
    filters = (["bwdif"] if deinterlace else []) + [f"scale={w}:{h}", "format=gray", "showinfo"]
    proc = run([
        "ffmpeg", "-hide_banner", "-ss", f"{lo:.3f}", "-t", f"{hi - lo:.3f}", "-copyts",
        *HWACCEL, "-i", str(video), "-map", "0:v:0", "-vf", ",".join(filters),
        "-f", "rawvideo", "-",
    ])
    times = [float(x) - start_time for x in re.findall(r"pts_time:\s*(-?[\d.]+)", proc.stderr.decode())]
    size = w * h
    frames = [proc.stdout[i:i + size] for i in range(0, len(proc.stdout) - size + 1, size)]
    if not frames or len(times) != len(frames):
        return cue.mid
    best = max(range(len(frames)), key=lambda i: sharpness(frames[i]))
    # Nudge just before the frame so seeking lands on it, not the one after.
    return max(cue.start, times[best] - 0.005)


# --- Rendering --------------------------------------------------------------

# Output frames taller than this are scaled down (set from --max-height).
MAX_HEIGHT = None
# ffmpeg input options for hardware video decoding (set from --hwaccel). ffmpeg
# falls back to software decoding for codecs the hardware can't handle.
HWACCEL = []


def base_filters(deinterlace):
    filters = ["bwdif"] if deinterlace else []
    # Square the pixels so anamorphic (e.g. DVD) sources come out at display aspect.
    filters.append("scale=trunc(iw*sar/2)*2:ih:flags=lanczos,setsar=1")
    if MAX_HEIGHT:
        filters.append(f"scale=-2:min(ih\\,{MAX_HEIGHT}):flags=lanczos")
    return filters


def jpeg_args(out):
    return ["-q:v", "3"] if out.suffix.lower() in (".jpg", ".jpeg") else []


def render_libass(video, t, out, sub_file, fonts_dir, start_time, deinterlace):
    sub_filter = f"subtitles=filename={escape_filter_path(sub_file)}"
    if fonts_dir:
        sub_filter += f":fontsdir={escape_filter_path(fonts_dir)}"
    # -copyts keeps original timestamps so libass knows what time it is after seeking;
    # subtract the container start time so they line up with the extracted subs.
    filters = base_filters(deinterlace) + [f"setpts=PTS-{start_time}/TB", sub_filter]
    run([
        "ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-copyts", *HWACCEL, "-i", str(video),
        "-map", "0:v:0", "-vf", ",".join(filters), "-frames:v", "1", *jpeg_args(out), str(out),
    ])


def prepare_for_ocr(image_path, out_path):
    """Turn a bitmap subtitle into an image tesseract reads well, saved to
    out_path. Returns False if the subtitle is blank."""
    from PIL import Image, ImageOps

    img = Image.open(image_path).convert("RGBA")
    bbox = img.getchannel("A").getbbox()
    if not bbox:
        return False
    # Scale to the size of a 1620px-tall subtitle canvas (about 2.8x for DVD,
    # 1.5x for Blu-ray): small DVD text needs enlarging, but enlarging Blu-ray
    # text much more than this makes tesseract misread "i" as "I".
    scale = 1620 / img.height
    # White text on black, inverted to black on white, reads best.
    img = img.crop(bbox)
    flat = Image.new("RGB", img.size, (0, 0, 0))
    flat.paste(img, mask=img.getchannel("A"))
    flat = ImageOps.invert(flat).resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    ImageOps.expand(flat, 30, fill=(255, 255, 255)).save(out_path)
    return True


def tesseract(images):
    """OCR images in one tesseract run (it takes about a second just to start);
    returns one text per image."""
    listing = images[0].with_suffix(".list")
    listing.write_text("".join(f"{p}\n" for p in images))
    # One thread per tesseract: several run at once, and their threads would compete.
    out = run(["tesseract", str(listing), "stdout", "--psm", "6", "-l", "eng"],
              env={**os.environ, "OMP_THREAD_LIMIT": "1"}).stdout.decode(errors="replace")
    pages = out.split("\f")  # tesseract ends each image's text with a form feed
    if len(pages) < len(images):
        raise RuntimeError(f"tesseract returned {len(pages)} pages for {len(images)} images")
    return ["\n".join(fix_ocr(line.strip()) for line in page.splitlines() if line.strip())
            for page in pages[:len(images)]]


def ocr_images(images, jobs):
    """OCR prepared images, split across `jobs` tesseract processes."""
    if not images:
        return []
    per_job = -(-len(images) // max(1, jobs))
    chunks = [images[i:i + per_job] for i in range(0, len(images), per_job)]
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        return [text for texts in pool.map(tesseract, chunks) for text in texts]


# Words tesseract capitalises by misreading "i" as "I"; never capitalised mid-sentence.
I_WORDS = re.compile(r"(?<=[a-z,] )(Is|It|It's|Its|In|If|Into)\b")


def fix_ocr(line):
    """Correct tesseract's usual mistakes on subtitle fonts."""
    line = re.sub(r"(?<![\w|])\|(?![\w|])", "I", line)  # a lone "|" is a misread "I"
    line = I_WORDS.sub(lambda m: m[1].lower(), line)
    # Slashed zeros read as Q or @, e.g. "6-Q" for "6-0".
    line = re.sub(r"(?<=\d-)(Q@|Q|@|Ø)|(Q@|Q|@|Ø)(?=-\d)", "0", line)
    # Dialogue dashes run into the next letter: "-Wwhat"/"-\What" for "-What",
    # and "“Ves" for "-Yes".
    line = re.sub(r"(?<=-)\\(?=[A-Z])", "", line)
    line = re.sub(r"\bWw(?=[a-z])", "W", line)
    line = re.sub(r"^[“”\"]V(?=es\b)", "-Y", line)
    return line


def save_subtitles(out_dir, results):
    """Write subtitles.srt, and subtitles.csv saying which line each shot shows.

    results: (cue, shot filename or None, text) in cue order.
    """
    import pysubs2

    srt = pysubs2.SSAFile()
    for cue, _, text in results:
        if text:
            srt.events.append(pysubs2.SSAEvent(
                start=round(cue.start * 1000), end=round(cue.end * 1000), text=text.replace("\n", "\\N")))
    srt.save(str(out_dir / "subtitles.srt"))
    with open(out_dir / "subtitles.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["shot", "start", "end", "text"])
        for cue, shot, text in results:
            writer.writerow([shot or "", f"{cue.start:.3f}", f"{cue.end:.3f}", text])


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


def draw_subtitle(img, text, font_path, font_scale):
    """Draw subtitle text onto a frame, white with a black outline, bottom centre."""
    from PIL import ImageDraw, ImageFont

    w, h = img.size

    size = max(12, round(h * font_scale))
    font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default(size)
    draw = ImageDraw.Draw(img)
    text = "\n".join(wrap_lines(draw, text, font, w * 0.9))
    draw.multiline_text(
        (w / 2, h - h * 0.05), text, font=font, anchor="md", align="center",
        fill="white", stroke_width=max(2, size // 14), stroke_fill="black",
        spacing=round(size * 0.2),
    )


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


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="output directory (default: <video name>_shots)")
    ap.add_argument("-s", "--subs", type=Path, help="external subtitle file (.srt, .ass, .vtt, ...)")
    ap.add_argument("-t", "--track", type=int, default=0, help="embedded subtitle track index (default: 0)")
    ap.add_argument("--list-tracks", action="store_true", help="list embedded subtitle tracks and exit")
    ap.add_argument("--renderer", choices=["auto", "libass", "pillow"], default="auto",
                    help="how to draw text subs (auto: libass if ffmpeg supports it)")
    ap.add_argument("-f", "--format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--max-height", type=int, default=576,
                    help="scale down frames taller than this (default: 576, DVD height; 0 = never)")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4, help="parallel ffmpeg processes")
    ap.add_argument("--deinterlace", action="store_true", help="deinterlace frames (useful for DVD rips)")
    ap.add_argument("-w", "--window", type=float, default=0.4,
                    help="use the sharpest frame within this many seconds of the midpoint "
                         "(default: 0.4; 0 = exact midpoint)")
    ap.add_argument("--font", help="font file for the Pillow renderer")
    ap.add_argument("--font-scale", type=float, default=0.055,
                    help="Pillow font size as a fraction of frame height (default: 0.055)")
    ap.add_argument("--limit", type=int, help="only do the first N cues")
    ap.add_argument("--hwaccel", default="auto",
                    help="ffmpeg hardware decoder, e.g. videotoolbox, or 'none' "
                         "(default: auto = videotoolbox for HEVC video on macOS)")
    ap.add_argument("--save-subs", action="store_true",
                    help="also write subtitles.srt and subtitles.csv (shot, start, end, text) to the "
                         "output folder; bitmap subtitles are read with tesseract OCR")
    return ap


def main():
    args = build_parser().parse_args()

    if not args.video.exists():
        sys.exit(f"No such file: {args.video}")
    global MAX_HEIGHT
    MAX_HEIGHT = args.max_height

    probe = ffprobe_json("-show_streams", "-show_format", str(args.video))
    sub_streams = [s for s in probe["streams"] if s.get("codec_type") == "subtitle"]
    if args.list_tracks:
        list_tracks(sub_streams)
        return
    video_stream = next((s for s in probe["streams"] if s.get("codec_type") == "video"), None)
    if not video_stream:
        sys.exit("No video stream found.")
    global HWACCEL
    hwaccel = args.hwaccel
    if hwaccel == "auto":
        # Hardware decoding pays off for HEVC, which is slow to decode in software;
        # for H.264 (e.g. iPlayer) starting the decoder costs more than it saves.
        hevc = video_stream.get("codec_name") == "hevc"
        hwaccel = "videotoolbox" if sys.platform == "darwin" and hevc else "none"
    HWACCEL = [] if hwaccel == "none" else ["-hwaccel", hwaccel]
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

        if args.save_subs and renderer == "bitmap" and not shutil.which("tesseract"):
            sys.exit("--save-subs needs tesseract to read bitmap subtitles (brew install tesseract).")

        def shoot(i, cue):
            """Take the screenshot; return its name and the subtitle's text (for
            bitmap subtitles, the image prepared for OCR, or "" if blank)."""
            if renderer == "libass":
                t = sharpest_frame_time(args.video, cue, args.window, start_time, args.deinterlace)
                out = out_dir / f"{i:04d}_{timestamp(t)}.{args.format}"
                render_libass(args.video, t, out, sub_file, fonts_dir, start_time, args.deinterlace)
                return out.name, cue.text
            bitmap = renderer == "bitmap"
            sub_image = tmp / f"sub{i:04d}.png" if bitmap and args.save_subs else None
            t, img = grab_sharpest(args.video, cue, args.window, args.deinterlace, tmp / f"frames{i:04d}",
                                   track=args.track if bitmap else None, sub_image=sub_image)
            if not bitmap:
                draw_subtitle(img, cue.text, font_path, args.font_scale)
            out = out_dir / f"{i:04d}_{timestamp(t)}.{args.format}"
            save_image(img, out)
            if sub_image:
                prepared = tmp / f"ocr{i:04d}.png"
                has_text = prepare_for_ocr(sub_image, prepared)
                sub_image.unlink()
                return out.name, prepared if has_text else ""
            return out.name, cue.text

        print(f"{len(cues)} cues, renderer: {renderer}, output: {out_dir}")
        failures = 0
        results = {}
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = {pool.submit(shoot, i, c): (i, c) for i, c in enumerate(cues, 1)}
            for done, fut in enumerate(as_completed(futures), 1):
                i, cue = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    failures += 1
                    print(f"\n  cue {i} @ {timestamp(cue.mid)} failed: {e}", file=sys.stderr)
                print(f"\r  {done}/{len(cues)}", end="", flush=True)
        print()
        if args.save_subs:
            if renderer == "bitmap":
                prepared = {i: text for i, (_, text) in results.items() if isinstance(text, Path)}
                for i, text in zip(prepared, ocr_images(list(prepared.values()), args.jobs)):
                    results[i] = (results[i][0], text)
            blank = "" if renderer == "bitmap" else None
            save_subtitles(out_dir, [(c, *results.get(i, (None, c.text if blank is None else blank)))
                                     for i, c in enumerate(cues, 1)])
        if failures:
            sys.exit(f"{failures} screenshot(s) failed.")


if __name__ == "__main__":
    main()
