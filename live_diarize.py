#!/usr/bin/env python3
"""
live_diarize.py — REAL-TIME speaker diarization of a live stream.

    # replay a local file as a "live" stream (no network — best first test):
    python live_diarize.py sample.wav --simulate

    # a direct live stream (HLS / RTMP / SRT / icecast) — no yt-dlp needed:
    python live_diarize.py "https://.../stream.m3u8"

    # a live YouTube URL (needs yt-dlp; see README for the anti-bot flags):
    python live_diarize.py "https://www.youtube.com/watch?v=<LIVE_ID>" \
        --cookies-from-browser chrome --js-runtime deno --remote-components ejs:npm

    # bounded run (auto-stop after N seconds):
    python live_diarize.py <input> --max-seconds 60

Design (bounded-latency real-time)
----------------------------------
    source -> ffmpeg (16 kHz mono s16le PCM)
       -> [reader thread]  fills a rolling buffer at real-time rate
       -> [main loop]      every `stride`s, diarizes the LATEST `window`s, maps clusters
                           to a persistent speaker registry, and emits who is talking now.
                           If compute falls behind it simply diarizes less often
                           (skips ahead) — latency never accumulates.
Runs on NVIDIA CUDA, Apple MPS, or CPU (auto-selected).
"""
import os
import sys
import time
import shutil
import subprocess
import json
import pathlib
import warnings
import argparse
import threading
import webbrowser
import numpy as np

warnings.filterwarnings("ignore")            # silence pyannote/torch/numpy noise
np.seterr(invalid="ignore", divide="ignore")

# --- make output robust on Windows terminals (UTF-8 + ANSI colors) ---
if os.name == "nt":
    os.system("")                                  # enable ANSI escape processing
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
import torch
from core import load_pipeline, SpeakerRegistry, pick_device

SR = 16000
COL = ["\033[94m", "\033[92m", "\033[91m", "\033[93m", "\033[95m", "\033[96m"]
RESET = "\033[0m"
PALETTE_HEX = ["#3498DB", "#2ECC71", "#E74C3C", "#F39C12", "#9B59B6",
               "#1ABC9C", "#E91E63", "#FF9800", "#34495E", "#16A085"]


def _fmt(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def build_html(events, total, out_path, title="Live Speaker Timeline"):
    """Render a self-contained colored-timeline HTML from speaker-change events.
    events: list of (stream_time_sec, stable_id, is_new)."""
    blocks = []
    for i, (t, sid, _) in enumerate(events):
        end = events[i + 1][0] if i + 1 < len(events) else total
        if end > t:
            blocks.append((t, end, sid))
    color = {sid: PALETTE_HEX[(sid - 1) % len(PALETTE_HEX)]
             for sid in sorted(set(b[2] for b in blocks))}
    bars = "".join(
        f'<div title="SPEAKER {sid}: {_fmt(s)}-{_fmt(e)}" style="display:inline-block;'
        f'width:{max((e - s) / total * 100, 0.05):.3f}%;height:60px;background:{color[sid]};'
        f'vertical-align:top;border-right:1px solid rgba(0,0,0,.15)"></div>'
        for s, e, sid in blocks)
    chips = "".join(
        f'<span style="background:{c};color:#fff;padding:4px 12px;border-radius:16px;'
        f'font:600 13px sans-serif;margin:3px">SPEAKER {sid}</span>' for sid, c in color.items())
    axis = "".join(f"<span>{_fmt(total * i / 8)}</span>" for i in range(9))
    rows = "".join(
        f'<tr><td>{i + 1}</td><td>{_fmt(s)}</td><td>{_fmt(e)}</td><td>{e - s:.1f}s</td>'
        f'<td style=color:{color[sid]}>SPEAKER {sid}</td></tr>'
        for i, (s, e, sid) in enumerate(blocks))
    html = f"""<!doctype html><meta charset=utf-8>
<body style="background:#0d0d1f;color:#eee;font-family:sans-serif;padding:26px">
<h2>{title} — {len(color)} speaker(s), {len(blocks)} changes over {_fmt(total)}</h2>
<div style="width:100%;border-radius:10px;overflow:hidden;border:1px solid #333">{bars}</div>
<div style="display:flex;justify-content:space-between;color:#888;font:11px monospace;margin-top:4px">{axis}</div>
<div style="margin-top:16px">{chips}</div>
<table style="margin-top:20px;border-collapse:collapse;font:13px monospace">
<tr style="color:#888"><td>#</td><td>start</td><td>end</td><td>dur</td><td>speaker</td></tr>
{rows}</table></body>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def _tool(name):
    exe = shutil.which(name)
    if not exe:
        sys.exit(f"ERROR: '{name}' not found on PATH. Please install it (see README).")
    return exe


def is_direct_stream(url):
    """A URL ffmpeg can read directly (no yt-dlp needed)."""
    u = url.lower()
    return (u.startswith(("rtmp://", "rtmps://", "srt://", "udp://", "rtp://", "http://", "https://"))
            and (".m3u8" in u or u.endswith((".ts", ".flv", ".aac", ".mp3", ".wav")))) \
        or u.startswith(("rtmp://", "rtmps://", "srt://", "udp://", "rtp://"))


def resolve_stream(url, cookies_from_browser=None, js_runtime=None, remote_components=None):
    """Resolve a URL to a direct media URL ffmpeg can read.
    Direct stream URLs (HLS/RTMP/SRT/…) bypass yt-dlp entirely."""
    if is_direct_stream(url):
        return url
    ytdlp = _tool("yt-dlp")
    base = [ytdlp, "--no-warnings"]
    if cookies_from_browser:
        base += ["--cookies-from-browser", cookies_from_browser]
    if js_runtime:
        base += ["--js-runtimes", js_runtime]
    if remote_components:
        base += ["--remote-components", remote_components]

    last_err = ""
    for fmt in ["91", "bestaudio/best"]:           # 91 = YouTube live HLS (audio-cheap)
        try:
            out = subprocess.run(base + ["-f", fmt, "-g", url],
                                 capture_output=True, text=True, timeout=90)
            links = [l for l in out.stdout.strip().splitlines() if l.startswith("http")]
            if links:
                return links[0]
            last_err = out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "no URL"
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(
        "Could not resolve a playable stream URL.\n"
        f"  yt-dlp said: {last_err}\n"
        "  For a live YouTube URL, YouTube's anti-bot challenge is likely blocking it — see\n"
        "  the README 'Live YouTube' section (cookies + deno + --remote-components).\n"
        "  Direct HLS/RTMP/SRT URLs and --simulate need none of that.")


def to_16k_mono_wav(path):
    """Convert any local audio/video file to a 16 kHz mono wav (via ffmpeg)."""
    out = os.path.splitext(path)[0] + "_16k.wav"
    subprocess.run([_tool("ffmpeg"), "-y", "-i", path, "-ac", "1", "-ar", str(SR), out],
                   check=True, capture_output=True)
    return out


class LiveDiarizer:
    """Rolling-buffer + speaker-registry engine with bounded-latency scheduling."""

    def __init__(self, window=15.0, stride=2.0, threshold=0.85, num_speakers=None,
                 warmup=8.0, match_sim=0.30, recent=3.0, confirm=2):
        self.device = pick_device()
        print(f"-> loading diarizer (pyannote community-1 on {self.device.upper()})...", flush=True)
        self.pipe = load_pipeline(device=self.device,
                                  clustering_threshold=None if num_speakers else threshold)
        self.reg = SpeakerRegistry(match_sim=match_sim)
        self.win_n = int(window * SR)
        self.stride = stride
        self.warmup_n = int(warmup * SR)
        self.num_speakers = num_speakers
        self.recent = recent            # judge "current speaker" from the last `recent`s
        self.confirm = confirm          # a change must persist this many strides before emitting
        self.buf = np.zeros(0, dtype=np.float32)
        self.total = 0                  # total samples ever received (= stream position)
        self.lock = threading.Lock()
        self.stop = False

    # ── ingestion ────────────────────────────────────────────────────────────
    def feed(self, samples):
        with self.lock:
            self.buf = np.concatenate([self.buf, samples])
            if len(self.buf) > self.win_n:
                self.buf = self.buf[-self.win_n:]   # keep only the rolling window
            self.total += len(samples)

    def _snapshot(self):
        with self.lock:
            return self.buf.copy(), self.total

    # ── one diarization pass on the latest window -> current speaker ──────────
    def _current_speaker(self, buf):
        kw = {"num_speakers": self.num_speakers} if self.num_speakers else {}
        out = self.pipe({"waveform": torch.from_numpy(buf).unsqueeze(0), "sample_rate": SR}, **kw)
        if self.device == "mps":
            torch.mps.empty_cache()     # prevent MPS memory accrual across strides
        ann = out.exclusive_speaker_diarization
        embs = out.speaker_embeddings
        labels = ann.labels()
        if not labels:
            return None, False
        lab2emb = {l: (embs[i] if embs is not None and i < len(embs) else None)
                   for i, l in enumerate(labels)}
        mapping = self.reg.match(list(lab2emb.items()))
        # current speaker = whoever speaks MOST in the last `recent`s (robust to backchannels)
        win_end = len(buf) / SR
        lo = win_end - self.recent
        talk = {}
        for turn, _, label in ann.itertracks(yield_label=True):
            ov = min(turn.end, win_end) - max(turn.start, lo)
            if ov > 0:
                talk[label] = talk.get(label, 0.0) + ov
        if not talk:
            return None, False
        label = max(talk, key=talk.get)
        sid = mapping.get(label)
        return sid, (self.reg.counts.get(sid, 0) <= 1)

    # ── main real-time loop ──────────────────────────────────────────────────
    def run(self, max_seconds=None, on_change=None):
        cur = None
        cand, cand_hits = None, 0
        t0 = time.time()
        next_tick = t0 + self.stride
        while not self.stop:
            now = time.time()
            if max_seconds and now - t0 >= max_seconds:
                break
            if now < next_tick:
                time.sleep(min(0.05, next_tick - now))
                continue
            next_tick = now + self.stride           # schedule next; skip-ahead if we were slow
            buf, total = self._snapshot()
            if len(buf) < self.warmup_n:
                continue
            sid, is_new = self._current_speaker(buf)
            if sid is None:
                continue
            # debounce: require the same new speaker for `confirm` consecutive strides
            cand_hits = cand_hits + 1 if sid == cand else 1
            cand = sid
            need = 1 if cur is None else self.confirm
            if sid != cur and cand_hits >= need:
                if on_change:
                    on_change(sid, is_new, total / SR, now - t0)
                cur = sid


def main():
    ap = argparse.ArgumentParser(description="Real-time speaker diarization of a live stream.")
    ap.add_argument("input", help="local file (with --simulate), or a live stream URL")
    ap.add_argument("--simulate", action="store_true", help="replay a local file at 1x real-time")
    ap.add_argument("--window", type=float, default=15.0, help="rolling buffer length (s)")
    ap.add_argument("--stride", type=float, default=2.0, help="re-diarize every N seconds")
    ap.add_argument("--threshold", type=float, default=0.85, help="clustering merge threshold")
    ap.add_argument("--speakers", type=int, default=None, help="force a known speaker count")
    ap.add_argument("--max-seconds", type=float, default=None, help="auto-stop after N seconds")
    ap.add_argument("--cookies-from-browser", default=None,
                    help="chrome|edge|firefox — auth to bypass YouTube's bot check")
    ap.add_argument("--js-runtime", default=None, help="JS runtime for yt-dlp (node|deno)")
    ap.add_argument("--remote-components", default=None,
                    help="e.g. ejs:npm — let yt-dlp fetch YouTube's JS challenge solver")
    args = ap.parse_args()

    ld = LiveDiarizer(window=args.window, stride=args.stride,
                      threshold=args.threshold, num_speakers=args.speakers)

    # ── ingestion thread ─────────────────────────────────────────────────────
    if args.simulate:
        import soundfile as sf
        wav = args.input
        info = sf.info(wav)
        if info.samplerate != SR or info.channels != 1:
            wav = to_16k_mono_wav(args.input)
        audio, _ = sf.read(wav, dtype="float32")
        print(f"-> SIMULATING live stream from {args.input} ({len(audio)/SR:.0f}s) at 1x\n", flush=True)

        def reader():
            step = int(0.5 * SR)
            for i in range(0, len(audio), step):
                if ld.stop:
                    return
                time.sleep(0.5)
                ld.feed(audio[i:i + step])
            ld.stop = True
    else:
        print(f"-> resolving stream: {args.input}", flush=True)
        src = resolve_stream(args.input,
                             cookies_from_browser=args.cookies_from_browser,
                             js_runtime=args.js_runtime,
                             remote_components=args.remote_components)
        proc = subprocess.Popen([_tool("ffmpeg"), "-loglevel", "quiet", "-i", src,
                                 "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                                stdout=subprocess.PIPE, bufsize=10 ** 7)
        print("-> LIVE. Listening... (Ctrl+C to stop)\n", flush=True)

        def reader():
            step_bytes = int(0.5 * SR) * 2
            while not ld.stop:
                raw = proc.stdout.read(step_bytes)
                if not raw:
                    ld.stop = True
                    return
                ld.feed(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)

    threading.Thread(target=reader, daemon=True).start()

    events = []

    def on_change(sid, is_new, stream_t, wall):
        events.append((stream_t, sid, is_new))
        c = COL[(sid - 1) % len(COL)]
        tag = "  [NEW VOICE]" if is_new else ""
        print(f"[wall {wall:6.1f}s | stream {stream_t:6.1f}s]  {c}* SPEAKER {sid}{RESET}{tag}", flush=True)

    try:
        ld.run(max_seconds=args.max_seconds, on_change=on_change)
    except KeyboardInterrupt:
        pass
    finally:
        ld.stop = True
        total = ld.total / SR
        spk = sorted(ld.reg.centroids.keys())
        print(f"\n{'=' * 50}\n  speakers seen: {len(spk)} {spk}   changes: {len(events)}\n{'=' * 50}", flush=True)
        if events:
            out_html = os.path.join(os.getcwd(), "live_timeline.html")
            out_json = os.path.join(os.getcwd(), "live_result.json")
            build_html(events, total, out_html)
            blocks = [[round(events[i][0], 1),
                       round(events[i + 1][0] if i + 1 < len(events) else total, 1),
                       events[i][1]] for i in range(len(events))]
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({"n_speakers": len(spk), "n_changes": len(events),
                           "duration_sec": round(total, 1), "timeline": blocks}, f, indent=2)
            print(f"  HTML timeline: {out_html}\n  JSON: {out_json}", flush=True)
            try:
                webbrowser.open(pathlib.Path(out_html).resolve().as_uri())
            except Exception:
                pass


if __name__ == "__main__":
    main()
