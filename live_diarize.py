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

Design (bounded-latency real-time, boundary-accurate)
-----------------------------------------------------
    source -> ffmpeg (16 kHz mono s16le PCM)
       -> [reader thread]  fills a rolling buffer at real-time rate
       -> [main loop]      every `stride`s, diarizes the LATEST `window`s. It COMMITS the
                           fine-grained speaker turns in the region that is now settled
                           (older than `commit_lag`), mapping each to a persistent speaker
                           registry. The committed turns are post-processed exactly like the
                           offline tool (merge + min-duration), so speaker-CHANGE timestamps
                           track the offline boundaries. If compute falls behind, it commits
                           less often (skips ahead) — latency never accumulates.
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


def _tool(name):
    exe = shutil.which(name)
    if not exe:
        sys.exit(f"ERROR: '{name}' not found on PATH. Please install it (see README).")
    return exe


def is_direct_stream(url):
    """A URL ffmpeg can read directly (no yt-dlp needed)."""
    u = url.lower()
    return (u.startswith(("http://", "https://"))
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


def download_video(url, out, cookies=None, js=None, remote=None):
    """Download a YouTube (or other) video with audio, as an mp4."""
    ytdlp = _tool("yt-dlp")
    cmd = [ytdlp, "--no-warnings", "-f", "bv*+ba/b", "--merge-output-format", "mp4", "-o", out]
    if cookies:
        cmd += ["--cookies-from-browser", cookies]
    if js:
        cmd += ["--js-runtimes", js]
    if remote:
        cmd += ["--remote-components", remote]
    subprocess.run(cmd + [url], check=True)
    return out


def _fmt(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def build_html(blocks, total, out_path, title="Live Speaker Timeline"):
    """Self-contained colored-timeline HTML (same layout as the offline tool).
    blocks: list of (start_sec, end_sec, stable_id)."""
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
    n_changes = sum(1 for i in range(1, len(blocks)) if blocks[i][2] != blocks[i - 1][2])
    html = f"""<!doctype html><meta charset=utf-8>
<body style="background:#0d0d1f;color:#eee;font-family:sans-serif;padding:26px">
<h2>{title} — {len(color)} speaker(s), {n_changes} changes over {_fmt(total)}</h2>
<div style="width:100%;border-radius:10px;overflow:hidden;border:1px solid #333">{bars}</div>
<div style="display:flex;justify-content:space-between;color:#888;font:11px monospace;margin-top:4px">{axis}</div>
<div style="margin-top:16px">{chips}</div>
<table style="margin-top:20px;border-collapse:collapse;font:13px monospace">
<tr style="color:#888"><td>#</td><td>start</td><td>end</td><td>dur</td><td>speaker</td></tr>
{rows}</table></body>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


class LiveDiarizer:
    """Rolling-buffer + speaker-registry engine that commits fine-grained turn boundaries."""

    def __init__(self, window=15.0, stride=1.5, threshold=0.85,
                 warmup=4.0, match_sim=0.30, commit_lag=1.0,
                 min_change_dur=0.4, merge_gap=0.4):
        self.device = pick_device()
        print(f"-> loading diarizer (pyannote community-1 on {self.device.upper()})...", flush=True)
        # windows are ALWAYS diarized unconstrained (each 15s window has a variable, usually
        # small speaker count) — a known cast count is applied later, to the GLOBAL clustering.
        self.pipe = load_pipeline(device=self.device, clustering_threshold=threshold)
        self.reg = SpeakerRegistry(match_sim=match_sim)
        self.win_n = int(window * SR)
        self.stride = stride
        self.warmup_n = int(warmup * SR)
        self.commit_lag = commit_lag        # only commit turns older than this (gives boundaries context)
        self.min_change_dur = min_change_dur
        self.merge_gap = merge_gap
        self.buf = np.zeros(0, dtype=np.float32)
        self.total = 0                      # total samples ever received (= stream position)
        self.lock = threading.Lock()
        self.stop = False
        self.raw = []                       # committed fine-grained (start, end, stable_id) turns
        self.committed_until = 0.0          # absolute stream time already committed

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

    # ── diarize one window -> absolute-time turns mapped to stable IDs ────────
    def _diarize_window(self, buf, buf_start):
        out = self.pipe({"waveform": torch.from_numpy(buf).unsqueeze(0), "sample_rate": SR})
        if self.device == "mps":
            torch.mps.empty_cache()
        ann = out.exclusive_speaker_diarization
        embs = out.speaker_embeddings
        labels = ann.labels()
        lab2emb = {l: (embs[i] if embs is not None and i < len(embs) else None)
                   for i, l in enumerate(labels)}
        mapping = self.reg.match(list(lab2emb.items()))
        turns = []
        for turn, _, label in ann.itertracks(yield_label=True):
            sid = mapping.get(label)
            if sid is not None:
                turns.append((buf_start + turn.start, buf_start + turn.end, sid, lab2emb.get(label)))
        return turns

    def process(self, buf, stream_end, flush=False):
        """Commit the fine-grained turns that are now settled. Returns the newly
        committed (start, end, stable_id) pieces. `flush` commits right up to stream_end."""
        buf_start = stream_end - len(buf) / SR
        commit_edge = stream_end if flush else stream_end - self.commit_lag
        if commit_edge <= self.committed_until + 1e-3:
            return []
        turns = self._diarize_window(buf, buf_start)
        new = []
        lo, hi = self.committed_until, commit_edge
        for s, e, sid, emb in sorted(turns, key=lambda t: t[0]):
            s2, e2 = max(s, lo), min(e, hi)
            if e2 - s2 > 0:
                self.raw.append((s2, e2, sid, emb))
                new.append((s2, e2, sid))
        self.committed_until = commit_edge
        return new

    @staticmethod
    def _merge_relabel(turns, merge_gap, min_dur):
        """turns: sorted list of (start, end, label) -> merged blocks with stable 1..N ids."""
        remap, nxt = {}, 1
        out = []
        for s, e, lab in sorted(turns):
            if lab not in remap:
                remap[lab] = nxt
                nxt += 1
            out.append((s, e, remap[lab]))
        merged = []
        for s, e, sid in out:
            if merged and merged[-1][2] == sid and s - merged[-1][1] <= merge_gap:
                merged[-1] = [merged[-1][0], max(merged[-1][1], e), sid]
            else:
                merged.append([s, e, sid])
        return [(s, e, sid) for s, e, sid in merged if e - s >= min_dur]

    def finalize(self):
        """Registry labels (provisional, per-window). Fast but weaker on short turns."""
        return self._merge_relabel([(s, e, sid) for s, e, sid, _ in self.raw],
                                   self.merge_gap, self.min_change_dur)

    def finalize_global(self, num_speakers=None, dist_threshold=0.55, min_speaker_dur=20.0):
        """PRODUCTION labels: re-cluster ALL committed turn embeddings globally
        (like the offline tool), so short turns are labeled using each speaker's full
        evidence — not just one 15s window. Keeps the live-detected boundaries."""
        from scipy.cluster.hierarchy import linkage, fcluster
        items = [(s, e, emb) for (s, e, sid, emb) in self.raw
                 if emb is not None and not np.isnan(emb).any()]
        if len(items) < 3:
            return self.finalize()
        X = np.array([it[2] for it in items], dtype=float)
        Z = linkage(X, method="average", metric="cosine")
        if num_speakers:
            lab = fcluster(Z, t=num_speakers, criterion="maxclust")
        else:
            lab = fcluster(Z, t=dist_threshold, criterion="distance")
        # consolidate tiny clusters (< min_speaker_dur total) into nearest by centroid.
        # SKIP when a count is explicitly requested — respect the given cast size.
        dur = {}
        for (s, e, _), c in zip(items, lab):
            dur[c] = dur.get(c, 0.0) + (e - s)
        centroid = {c: X[[i for i in range(len(lab)) if lab[i] == c]].mean(0) for c in set(lab)}
        big = [c for c in dur if dur[c] >= min_speaker_dur]
        if big and not num_speakers:
            def _n(v):
                return v / (np.linalg.norm(v) + 1e-9)
            remap = {}
            for c in dur:
                if c in big:
                    remap[c] = c
                else:
                    remap[c] = max(big, key=lambda b: float(_n(centroid[c]) @ _n(centroid[b])))
            lab = [remap[c] for c in lab]
        turns = [(items[i][0], items[i][1], int(lab[i])) for i in range(len(items))]
        return self._merge_relabel(turns, self.merge_gap, self.min_change_dur)

    # ── main real-time loop ──────────────────────────────────────────────────
    def run(self, max_seconds=None, on_change=None):
        last_printed = None
        t0 = time.time()
        next_tick = t0 + self.stride
        while not self.stop:
            now = time.time()
            if max_seconds and now - t0 >= max_seconds:
                break
            if now < next_tick:
                time.sleep(min(0.05, next_tick - now))
                continue
            next_tick = now + self.stride
            buf, total = self._snapshot()
            if len(buf) < self.warmup_n:
                continue
            for s, e, sid in self.process(buf, total / SR):
                if on_change and (e - s) >= self.min_change_dur and sid != last_printed:
                    on_change(sid, self.reg.counts.get(sid, 0) <= 1, s, now - t0)
                    last_printed = sid
        # flush the tail that was held back by commit_lag
        buf, total = self._snapshot()
        if len(buf) >= self.warmup_n:
            self.process(buf, total / SR, flush=True)


def _run_over_audio(ld, audio, stride):
    """Drive the live engine over an in-memory audio array (file/overlay mode)."""
    win_n, stride_n = ld.win_n, int(stride * SR)
    end = stride_n
    while end <= len(audio):
        buf = audio[max(0, end - win_n):end]
        if len(buf) >= ld.warmup_n:
            ld.process(buf, end / SR)
        end += stride_n
    ld.process(audio[max(0, len(audio) - win_n):len(audio)], len(audio) / SR, flush=True)


def _write_outputs(blocks, total, tag):
    spk = sorted(set(b[2] for b in blocks))
    n_changes = sum(1 for i in range(1, len(blocks)) if blocks[i][2] != blocks[i - 1][2])
    print(f"\n  {tag}: {len(spk)} speakers, {len(blocks)} turns, {n_changes} changes", flush=True)
    out_html = os.path.join(os.getcwd(), "live_timeline.html")
    out_json = os.path.join(os.getcwd(), "live_result.json")
    build_html(blocks, total, out_html)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"n_speakers": len(spk), "n_changes": n_changes, "duration_sec": round(total, 1),
                   "timeline": [[round(s, 2), round(e, 2), sid] for s, e, sid in blocks]}, f, indent=2)
    return out_html, out_json


def overlay_mode(args):
    """URL/local video -> diarize its audio with the LIVE engine -> burn speaker labels."""
    import soundfile as sf
    import overlay
    cwd = os.getcwd()
    is_url = args.input.lower().startswith(("http://", "https://", "rtmp", "rtmps", "srt", "ytsearch"))
    if os.path.exists(args.input):
        video = args.input
    elif is_url:
        print("-> downloading video...", flush=True)
        video = download_video(args.input, os.path.join(cwd, "video.mp4"),
                               args.cookies_from_browser, args.js_runtime, args.remote_components)
    else:
        sys.exit(f"ERROR: file not found: {args.input}\n"
                 f"  Pass the full path (it's not in this folder), e.g.\n"
                 f'  python live_diarize.py "$env:USERPROFILE\\Downloads\\video.mp4" --overlay ...')
    wav = os.path.join(cwd, "_ov_audio.wav")
    subprocess.run([_tool("ffmpeg"), "-y", "-i", video, "-ac", "1", "-ar", str(SR), wav],
                   check=True, capture_output=True)
    audio, _ = sf.read(wav, dtype="float32")
    device = pick_device()
    if args.separate_vocals:
        import separate
        if not separate.available():
            sys.exit("Demucs not installed. Run: pip install demucs")
        print("-> separating vocals (Demucs)... this can take a while", flush=True)
        audio = separate.separate_vocals(audio, device=device)

    print(f"-> diarizing (live engine) on {device.upper()}...", flush=True)
    ld = LiveDiarizer(window=args.window, stride=args.stride,
                      commit_lag=args.commit_lag, threshold=args.threshold,
                      min_change_dur=args.min_change_dur, match_sim=args.match_sim)
    _run_over_audio(ld, audio, args.stride)
    blocks = (ld.finalize_global(num_speakers=args.speakers) if args.speakers
              else ld.finalize_global() if args.global_labels else ld.finalize())
    total = len(audio) / SR
    _write_outputs(blocks, total, "LIVE (overlay)")

    print("-> burning speaker labels onto the video...", flush=True)
    out_mp4 = os.path.join(cwd, "live_labeled.mp4")
    overlay.burn(video, blocks, out_mp4)
    print(f"  labeled video: {out_mp4}\n  HTML: {os.path.join(cwd, 'live_timeline.html')}", flush=True)
    try:
        webbrowser.open(pathlib.Path(out_mp4).resolve().as_uri())
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Real-time speaker diarization of a live stream.")
    ap.add_argument("input", help="local file (with --simulate), or a live stream URL")
    ap.add_argument("--simulate", action="store_true", help="replay a local file at 1x real-time")
    ap.add_argument("--window", type=float, default=15.0, help="rolling buffer length (s)")
    ap.add_argument("--stride", type=float, default=1.5, help="commit cadence (s) — lower = faster transitions")
    ap.add_argument("--commit-lag", type=float, default=1.0,
                    help="hold back the newest N s before committing (lower = faster, less boundary context)")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="per-window clustering threshold. LOWER (e.g. 0.5) to over-segment so "
                         "short/quiet speakers become distinct; pair with --speakers N.")
    ap.add_argument("--min-change-dur", type=float, default=0.4,
                    help="drop turns shorter than this (s) — lower keeps very short interjections")
    ap.add_argument("--match-sim", type=float, default=0.30,
                    help="registry same-speaker cosine cutoff")
    ap.add_argument("--speakers", type=int, default=None,
                    help="known cast size — applied to GLOBAL re-clustering (recommended when known)")
    ap.add_argument("--global-labels", action="store_true",
                    help="label turns by global re-clustering of all embeddings (auto count) "
                         "instead of the online registry")
    ap.add_argument("--max-seconds", type=float, default=None, help="auto-stop after N seconds")
    ap.add_argument("--overlay", action="store_true",
                    help="download the video (or use a local video), diarize its audio, and "
                         "burn speaker labels onto the video for audiovisual verification")
    ap.add_argument("--separate-vocals", action="store_true",
                    help="strip background music with Demucs before diarizing (music-heavy audio)")
    ap.add_argument("--cookies-from-browser", default=None,
                    help="chrome|edge|firefox — auth to bypass YouTube's bot check")
    ap.add_argument("--js-runtime", default=None, help="JS runtime for yt-dlp (node|deno)")
    ap.add_argument("--remote-components", default=None,
                    help="e.g. ejs:npm — let yt-dlp fetch YouTube's JS challenge solver")
    args = ap.parse_args()

    if args.overlay:
        overlay_mode(args)
        return

    ld = LiveDiarizer(window=args.window, stride=args.stride, commit_lag=args.commit_lag,
                      threshold=args.threshold, min_change_dur=args.min_change_dur,
                      match_sim=args.match_sim)

    # ── ingestion thread ─────────────────────────────────────────────────────
    if args.simulate:
        import soundfile as sf
        wav = args.input
        info = sf.info(wav)
        if info.samplerate != SR or info.channels != 1:
            wav = to_16k_mono_wav(args.input)
        audio, _ = sf.read(wav, dtype="float32")
        if args.separate_vocals:
            import separate
            if not separate.available():
                sys.exit("Demucs not installed. Run: pip install demucs")
            print("-> separating vocals (Demucs)...", flush=True)
            audio = separate.separate_vocals(audio, device=pick_device())
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

    def on_change(sid, is_new, stream_t, wall):
        c = COL[(sid - 1) % len(COL)]
        tag = "  [NEW VOICE]" if is_new else ""
        print(f"[wall {wall:6.1f}s | stream {stream_t:6.1f}s]  {c}* SPEAKER {sid}{RESET}{tag}", flush=True)

    try:
        ld.run(max_seconds=args.max_seconds, on_change=on_change)
    except KeyboardInterrupt:
        pass
    finally:
        ld.stop = True
        if args.speakers:                                    # known cast -> global clustering to N
            blocks = ld.finalize_global(num_speakers=args.speakers)
        elif args.global_labels:                             # opt-in auto global re-clustering
            blocks = ld.finalize_global()
        else:                                                # default: online registry (proven)
            blocks = ld.finalize()
        total = ld.total / SR if ld.total else 1.0
        spk = sorted(set(b[2] for b in blocks))
        n_changes = sum(1 for i in range(1, len(blocks)) if blocks[i][2] != blocks[i - 1][2])
        print(f"\n{'=' * 50}\n  speakers seen: {len(spk)} {spk}   turns: {len(blocks)}   changes: {n_changes}\n{'=' * 50}", flush=True)
        if blocks:
            out_html = os.path.join(os.getcwd(), "live_timeline.html")
            out_json = os.path.join(os.getcwd(), "live_result.json")
            build_html(blocks, total, out_html)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({"n_speakers": len(spk), "n_changes": n_changes,
                           "duration_sec": round(total, 1),
                           "timeline": [[round(s, 2), round(e, 2), sid] for s, e, sid in blocks]},
                          f, indent=2)
            print(f"  HTML timeline: {out_html}\n  JSON: {out_json}", flush=True)
            try:
                webbrowser.open(pathlib.Path(out_html).resolve().as_uri())
            except Exception:
                pass


if __name__ == "__main__":
    main()
