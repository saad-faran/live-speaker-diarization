#!/usr/bin/env python3
"""
overlay.py — burn speaker labels onto a video for audiovisual verification.

Renders a small colored "SPEAKER N" chip (PIL) and composites it onto the video
during each speaker's turns using ffmpeg's `overlay` filter. Uses no libass /
drawtext, so it works with any ffmpeg build (Windows / Linux / macOS).
"""
import os
import shutil
import subprocess
from PIL import Image, ImageDraw, ImageFont

PALETTE = ["#3498DB", "#2ECC71", "#E74C3C", "#F39C12", "#9B59B6",
           "#1ABC9C", "#E91E63", "#FF9800", "#34495E", "#16A085"]

_FONTS = [
    "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _font(size):
    for p in _FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def make_chip(text, hexcolor, out_png, scale=1.0):
    fs = int(34 * scale)
    font = _font(fs)
    pad = int(16 * scale)
    dot = int(10 * scale)
    d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    b = d0.textbbox((0, 0), text, font=font)
    tw, th = b[2] - b[0], b[3] - b[1]
    w = tw + pad * 3 + dot * 2
    h = th + pad * 2
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=int(12 * scale), fill=(0, 0, 0, 160))
    cy = h // 2
    cx = pad + dot
    d.ellipse([cx - dot, cy - dot, cx + dot, cy + dot], fill=_hex_rgb(hexcolor) + (255,))
    d.text((pad * 2 + dot * 2, (h - th) // 2 - b[1]), text, font=font, fill=(255, 255, 255, 255))
    img.save(out_png)


def burn(video, timeline, out_video, workdir=None, scale=1.0):
    """timeline: list of (start, end, stable_id). Overlays a SPEAKER chip during each turn.
    Returns out_video path."""
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    workdir = workdir or (os.path.dirname(os.path.abspath(out_video)) or ".")
    # normalize labels (offline "SPEAKER_00" or live int) -> 1..N for display + color
    distinct = sorted(set(t[2] for t in timeline), key=lambda x: str(x))
    remap = {lab: i + 1 for i, lab in enumerate(distinct)}
    timeline = [(s, e, remap[lab]) for s, e, lab in timeline]
    sids = sorted(set(t[2] for t in timeline))
    color = {sid: PALETTE[(sid - 1) % len(PALETTE)] for sid in sids}
    chips = {}
    for sid in sids:
        png = os.path.join(workdir, f"_chip_{sid}.png")
        make_chip(f"SPEAKER {sid}", color[sid], png, scale=scale)
        chips[sid] = png
    inputs = ["-i", video]
    for sid in sids:
        inputs += ["-i", chips[sid]]
    fc, last = [], "0:v"
    for idx, sid in enumerate(sids, start=1):
        spans = [f"between(t,{s:.2f},{e:.2f})" for s, e, ss in timeline if ss == sid]
        enable = "+".join(spans) if spans else "0"
        fc.append(f"[{last}][{idx}:v]overlay=x=(W-w)/2:y=24:enable='{enable}'[v{idx}]")
        last = f"v{idx}"
    cmd = ([ffmpeg, "-y"] + inputs + ["-filter_complex", ";".join(fc),
            "-map", f"[{last}]", "-map", "0:a?",
            "-c:a", "copy", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            out_video])
    subprocess.run(cmd, check=True, capture_output=True)
    for p in chips.values():
        try:
            os.remove(p)
        except OSError:
            pass
    return out_video


if __name__ == "__main__":
    import sys
    vid, out = sys.argv[1], sys.argv[2]
    demo = [(0.2, 4.0, 1), (4.0, 8.0, 2), (8.0, 11.8, 1)]
    print("burning ->", burn(vid, demo, out))
