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


def _ass_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _ass_color(hexc):
    r, g, b = hexc[1:3], hexc[3:5], hexc[5:7]
    return f"&H00{b}{g}{r}".upper()          # ASS is &HAABBGGRR


def burn(video, timeline, out_video, workdir=None, scale=1.0):
    """Burn a colored 'SPEAKER N' label onto the video during each turn, via an ASS
    subtitle track (libass). Scales to thousands of turns (a 1.5hr video is fine)."""
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    workdir = workdir or (os.path.dirname(os.path.abspath(out_video)) or ".")
    distinct = sorted(set(t[2] for t in timeline), key=lambda x: str(x))
    remap = {lab: i + 1 for i, lab in enumerate(distinct)}
    fs = int(30 * scale)
    styles = []
    for lab, idx in remap.items():
        c = _ass_color(PALETTE[(idx - 1) % len(PALETTE)])
        # Alignment 8 = top-center; opaque box (BorderStyle 3) for readability
        styles.append(f"Style: S{idx},Arial,{fs},{c},&H00000000,&H00000000,&HA0000000,"
                      f"-1,0,0,0,100,100,0,0,3,2,0,8,10,10,24,1")
    events = []
    for s, e, lab in timeline:
        if e <= s:
            continue
        events.append(f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},S{remap[lab]},,0,0,0,,"
                      f"SPEAKER {remap[lab]}")
    ass = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\n"
           "WrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
           "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
           "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
           "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
           "MarginV, Encoding\n" + "\n".join(styles) + "\n\n"
           "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
           "Effect, Text\n" + "\n".join(events) + "\n")
    ass_name = "_subs.ass"
    with open(os.path.join(workdir, ass_name), "w", encoding="utf-8") as f:
        f.write(ass)
    tmp_out = "_labeled_tmp.mp4"
    # run from workdir with a relative .ass name to avoid Windows filter-path escaping
    cmd = [ffmpeg, "-y", "-i", os.path.abspath(video), "-vf", f"ass={ass_name}",
           "-c:a", "copy", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", tmp_out]
    subprocess.run(cmd, check=True, capture_output=True, cwd=workdir)
    os.replace(os.path.join(workdir, tmp_out), out_video)
    try:
        os.remove(os.path.join(workdir, ass_name))
    except OSError:
        pass
    return out_video


if __name__ == "__main__":
    import sys
    vid, out = sys.argv[1], sys.argv[2]
    demo = [(0.2, 4.0, 1), (4.0, 8.0, 2), (8.0, 11.8, 1)]
    print("burning ->", burn(vid, demo, out))
