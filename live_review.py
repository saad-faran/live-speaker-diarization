#!/usr/bin/env python3
"""
live_review.py — LOCAL stress-test harness for the live diarizer.

Runs the ONLINE live pipeline (LiveDiarizer registry + faster-whisper ASR) over a
LOCAL video/audio file (or a direct HLS/RTMP stream), and gives you everything you
need to REVIEW it by eye and catch where it hallucinates:

    * a browser dashboard that plays the VIDEO with real-time captions, the current
      speaker label + color, a live speaker timeline, speaker count + change count,
      and a running clock — all kept in sync with the video's own clock.
    * captions.srt         — speaker-labeled subtitles ([SPEAKER N] text) you can
                             load into any player / burn into the video.
    * review_log.jsonl     — every speaker-change and caption, with media time,
                             wall-clock decision time, and detection LATENCY, plus
                             heuristic hallucination flags (repeats, empty regions).
    * live_timeline.html   — the finalized colored speaker timeline (same as the
                             offline tool) written at the end.
    * live_result.json     — machine-readable finalized timeline + summary metrics.

No YouTube / yt-dlp cookies needed: point it at a file you already have.

    # honest real-time replay (1x) of a local recording, with the review dashboard:
    python live_review.py "/path/to/stream1.mp4"

    # process as fast as the GPU allows (quick iteration; sync is still exact):
    python live_review.py "/path/to/stream1.mp4" --fast

    # known cast size -> globally-consistent IDs in the FINAL artifacts:
    python live_review.py "/path/to/stream1.mp4" --speakers 4 --threshold 0.5 --separate-vocals

    # a direct live stream that needs no cookies (icecast/HLS/RTMP):
    python live_review.py "https://.../stream.m3u8"

Then open the URL it prints (auto-opens). Press play. To stop a live/real-time run,
Ctrl+C — the SRT, timeline and JSON are still written from what was processed.
"""
import os
import sys
import json
import time
import queue
import pathlib
import argparse
import threading
import subprocess
import http.server
import socketserver
import webbrowser
import warnings
import numpy as np

warnings.filterwarnings("ignore")
np.seterr(invalid="ignore", divide="ignore")
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import asyncio
import websockets
from core import pick_device
from live_diarize import (resolve_stream, to_16k_mono_wav, _tool, SR, LiveDiarizer,
                          build_html, _fmt, is_direct_stream)

WS_PORT = 8765
HTTP_PORT = 8770

# ── WebSocket broadcast plumbing ─────────────────────────────────────────────
_clients = set()
_loop = None
_backlog = []            # events buffered so a client that connects late gets the history


async def _send_all(msg):
    for ws in list(_clients):
        try:
            await ws.send(msg)
        except Exception:
            _clients.discard(ws)


def emit(evt):
    """Thread-safe broadcast + keep a replay backlog for late-joining browsers."""
    _backlog.append(evt)
    if _loop is not None:
        asyncio.run_coroutine_threadsafe(_send_all(json.dumps(evt)), _loop)


async def _ws_handler(ws, *_):
    _clients.add(ws)
    try:
        for evt in list(_backlog):            # replay history to this client
            await ws.send(json.dumps(evt))
        async for _m in ws:
            pass
    finally:
        _clients.discard(ws)


# ── tiny static file server WITH HTTP range support (so <video> can seek) ────
class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _serve_range(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return False
        rng = self.headers.get("Range")
        size = os.path.getsize(path)
        ctype = self.guess_type(path)
        if not rng:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(path, "rb") as f:
                self.copyfile(f, self.wfile)
            return True
        try:
            units, rng = rng.split("=")
            start_s, end_s = rng.split("-")
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        except Exception:
            return False
        end = min(end, size - 1)
        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    def do_GET(self):
        try:
            if self._serve_range():
                return
            super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            return


def _start_http(directory):
    handler = lambda *a, **k: _RangeHandler(*a, directory=directory, **k)
    httpd = socketserver.ThreadingTCPServer(("localhost", HTTP_PORT), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


# ── SRT writer ───────────────────────────────────────────────────────────────
def _srt_ts(t):
    t = max(0.0, t)
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class SrtWriter:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.n = 0

    def add(self, start, end, speaker, text):
        self.n += 1
        self.f.write(f"{self.n}\n{_srt_ts(start)} --> {_srt_ts(max(end, start + 0.4))}\n"
                     f"[SPEAKER {speaker}] {text}\n\n")
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


# ── hallucination heuristics (whisper loves to loop on silence / music) ──────
def _hallucination_flags(text, dur, speaker, recent_texts):
    flags = []
    words = text.split()
    if speaker is None:
        flags.append("no_speaker")            # caption landed in a gap with no diarized turn
    if len(words) >= 4 and len(set(words)) <= max(2, len(words) // 4):
        flags.append("repeated_words")        # e.g. "thank you thank you thank you ..."
    if recent_texts and text and recent_texts[-1].strip().lower() == text.strip().lower():
        flags.append("dup_prev")              # identical to the previous caption
    if dur > 0 and len(words) / dur > 6.0:
        flags.append("too_fast")              # more words than humanly possible for the audio
    return flags


def _speaker_at(blocks, t):
    best = None
    for s, e, sid in blocks:
        if s <= t < e:
            return sid
        if s <= t:
            best = sid
    return best


# ── the pipeline (runs in a worker thread) ───────────────────────────────────
def pipeline(args, audio, video_url, log_f, srt, media_dur):
    """audio: float32 16k mono numpy array of the whole input (we already decoded it)."""
    device = pick_device()
    ld = LiveDiarizer(window=args.window, stride=args.stride, commit_lag=args.commit_lag,
                      threshold=args.threshold, min_change_dur=args.min_change_dur,
                      match_sim=args.match_sim)
    from faster_whisper import WhisperModel
    asr = WhisperModel(args.asr_model, device="cuda" if device == "cuda" else "cpu",
                       compute_type="float16" if device == "cuda" else "int8")
    emit({"type": "init", "video": video_url, "duration": round(media_dur, 2),
          "realtime": not args.fast, "asr": args.asr_model,
          "device": device, "language": args.language or "auto"})
    emit({"type": "status", "msg": f"live on {device.upper()} — diarizer + faster-whisper '{args.asr_model}'"})
    print(f"-> processing on {device.upper()} ({'real-time 1x' if not args.fast else 'fast'})", flush=True)

    t0 = time.time()
    step = int(0.5 * SR)                      # 0.5s ingest granularity
    fed = 0
    asr_buf = np.zeros(0, dtype=np.float32)
    asr_base = 0.0
    next_diar = ld.stride
    next_asr = args.asr_interval
    cur = None
    changes = 0
    seen = set()
    recent_texts = []
    n_caps = 0
    n_flagged = 0

    for i in range(0, len(audio), step):
        if _STOP.is_set():
            break
        pkt = audio[i:i + step]
        ld.feed(pkt)
        fed += len(pkt)
        media_t = fed / SR
        asr_buf = np.concatenate([asr_buf, pkt])
        if not args.fast:
            # pace ingestion to 1x real-time so the dashboard's video stays in step
            target = t0 + media_t
            dt = target - time.time()
            if dt > 0:
                time.sleep(min(dt, 0.5))

        # ---- diarization commit ----
        if media_t >= next_diar:
            next_diar = media_t + ld.stride
            buf, tot = ld._snapshot()
            if len(buf) >= ld.warmup_n:
                ld.process(buf, tot / SR)
                blocks = ld.finalize()
                if blocks:
                    for _, _, s in blocks:
                        seen.add(s)
                    sid = blocks[-1][2]
                    if sid != cur:
                        cur = sid
                        changes += 1
                        wall = time.time() - t0
                        rec = {"type": "speaker", "speaker": sid, "changes": changes,
                               "total_speakers": len(seen), "media_t": round(media_t, 2),
                               "wall_t": round(wall, 2),
                               "latency": None if args.fast else round(wall - media_t, 2)}
                        emit(rec)
                        log_f.write(json.dumps(rec) + "\n"); log_f.flush()

        # ---- ASR transcription ----
        if media_t >= next_asr and len(asr_buf) >= args.asr_interval * SR:
            next_asr = media_t + args.asr_interval
            seg_audio, base = asr_buf, asr_base
            asr_base += len(asr_buf) / SR
            asr_buf = np.zeros(0, dtype=np.float32)
            try:
                segs, _ = asr.transcribe(seg_audio, vad_filter=True, language=args.language)
                blocks = ld.finalize()
                for sg in segs:
                    txt = sg.text.strip()
                    if not txt:
                        continue
                    a_start, a_end = base + sg.start, base + sg.end
                    mid = (a_start + a_end) / 2.0
                    spk = _speaker_at(blocks, mid)
                    dur = max(a_end - a_start, 0.1)
                    flags = _hallucination_flags(txt, dur, spk, recent_texts)
                    recent_texts.append(txt); recent_texts[:] = recent_texts[-5:]
                    disp_spk = spk if spk is not None else (cur or 1)
                    n_caps += 1
                    if flags:
                        n_flagged += 1
                    wall = time.time() - t0
                    cap = {"type": "caption", "speaker": disp_spk, "text": txt,
                           "media_t": round(a_start, 2), "end_t": round(a_end, 2),
                           "wall_t": round(wall, 2),
                           "latency": None if args.fast else round(wall - a_end, 2),
                           "flags": flags}
                    emit(cap)
                    srt.add(a_start, a_end, disp_spk, txt)
                    log_f.write(json.dumps(cap) + "\n"); log_f.flush()
            except Exception as e:
                emit({"type": "status", "msg": f"asr error: {e}"})

    emit({"type": "done", "captions": n_caps, "flagged": n_flagged, "changes": changes,
          "speakers": len(seen)})
    print(f"-> finished: {len(seen)} speakers, {changes} changes, "
          f"{n_caps} captions ({n_flagged} flagged for review)", flush=True)
    return ld, seen


_STOP = threading.Event()


def _finalize_artifacts(ld, args, media_dur, cwd):
    if args.speakers:
        blocks = ld.finalize_global(num_speakers=args.speakers)
    elif args.global_labels:
        blocks = ld.finalize_global()
    else:
        blocks = ld.finalize()
    total = media_dur or (ld.total / SR if ld.total else 1.0)
    spk = sorted(set(b[2] for b in blocks))
    n_changes = sum(1 for i in range(1, len(blocks)) if blocks[i][2] != blocks[i - 1][2])
    out_html = os.path.join(cwd, "live_timeline.html")
    out_json = os.path.join(cwd, "live_result.json")
    build_html(blocks, total, out_html, title="Live Review — finalized timeline")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"n_speakers": len(spk), "n_changes": n_changes,
                   "duration_sec": round(total, 1),
                   "final_labeling": ("global:%d" % args.speakers if args.speakers
                                      else "global:auto" if args.global_labels else "registry"),
                   "timeline": [[round(s, 2), round(e, 2), sid] for s, e, sid in blocks]},
                  f, indent=2)
    print(f"\n{'=' * 56}\n  FINAL: {len(spk)} speakers {spk}, {n_changes} changes, {_fmt(total)}"
          f"\n  timeline: {out_html}\n  result:   {out_json}\n{'=' * 56}", flush=True)


async def main_async(args):
    global _loop
    cwd = os.getcwd()
    src = args.input

    # ---- decode the whole input to 16k mono once (we drive the engine over it) ----
    video_url = None
    if os.path.exists(src):
        print(f"-> decoding audio from {src}", flush=True)
        wav = to_16k_mono_wav(src)
        import soundfile as sf
        audio, _ = sf.read(wav, dtype="float32")
        # the browser plays the ORIGINAL file (served with range support)
        video_url = f"http://localhost:{HTTP_PORT}/{os.path.basename(src)}"
        serve_dir = os.path.dirname(os.path.abspath(src)) or cwd
    elif is_direct_stream(src):
        print("-> direct stream: capturing audio to a temp wav (Ctrl+C to stop)", flush=True)
        # for a live stream we can't scrub a video element; capture audio, no video panel
        tmp = os.path.join(cwd, "_review_capture.wav")
        subprocess.run([_tool("ffmpeg"), "-y", "-loglevel", "quiet", "-i", src,
                        "-ac", "1", "-ar", str(SR), tmp], check=True)
        import soundfile as sf
        audio, _ = sf.read(tmp, dtype="float32")
        serve_dir = cwd
    else:
        sys.exit(f"ERROR: not a local file and not a direct stream: {src}\n"
                 "  Pass a local video/audio file path, or a direct HLS/RTMP/icecast URL.\n"
                 "  (Live YouTube needs cookies/yt-dlp — out of scope for local review.)")

    if args.separate_vocals:
        import separate
        if not separate.available():
            sys.exit("Demucs not installed. Run: pip install demucs")
        print("-> separating vocals (Demucs)...", flush=True)
        audio = separate.separate_vocals(audio, device=pick_device())

    media_dur = len(audio) / SR

    # copy the dashboard next to the served files so it loads over http
    dash_src = os.path.join(os.path.dirname(__file__), "review.html")
    dash_dst = os.path.join(serve_dir, "review.html")
    if os.path.abspath(dash_src) != os.path.abspath(dash_dst):
        import shutil
        shutil.copyfile(dash_src, dash_dst)

    _start_http(serve_dir)
    log_f = open(os.path.join(cwd, "review_log.jsonl"), "w", encoding="utf-8")
    srt = SrtWriter(os.path.join(cwd, "captions.srt"))

    _loop = asyncio.get_running_loop()
    async with websockets.serve(_ws_handler, "localhost", WS_PORT):
        dash_url = f"http://localhost:{HTTP_PORT}/review.html"
        print(f"-> dashboard: {dash_url}\n-> websocket: ws://localhost:{WS_PORT}", flush=True)
        try:
            webbrowser.open(dash_url)
        except Exception:
            pass
        holder = {}

        def _run():
            try:
                ld, _ = pipeline(args, audio, video_url, log_f, srt, media_dur)
                holder["ld"] = ld
            except Exception as e:
                emit({"type": "status", "msg": f"pipeline error: {e}"})
                print(f"pipeline error: {e}", flush=True)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        try:
            while t.is_alive():
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            _STOP.set()
            t.join(timeout=10)

    srt.close(); log_f.close()
    if "ld" in holder:
        _finalize_artifacts(holder["ld"], args, media_dur, cwd)
    print(f"  captions: {os.path.join(cwd, 'captions.srt')}\n"
          f"  log:      {os.path.join(cwd, 'review_log.jsonl')}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Local video+captions review harness for the live diarizer.")
    ap.add_argument("input", help="local video/audio file, or a direct HLS/RTMP/icecast URL")
    ap.add_argument("--fast", action="store_true",
                    help="process as fast as possible (default replays at 1x real-time)")
    ap.add_argument("--asr-model", default="small", help="faster-whisper: tiny|base|small|medium")
    ap.add_argument("--asr-interval", type=float, default=6.0, help="transcribe every N seconds")
    ap.add_argument("--language", default=None, help="force ASR language (e.g. ur, en); default auto-detect")
    ap.add_argument("--window", type=float, default=15.0)
    ap.add_argument("--stride", type=float, default=1.5)
    ap.add_argument("--commit-lag", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="per-window clustering threshold (lower e.g. 0.5 to split short/quiet speakers)")
    ap.add_argument("--min-change-dur", type=float, default=0.4)
    ap.add_argument("--match-sim", type=float, default=0.30)
    ap.add_argument("--speakers", type=int, default=None,
                    help="known cast size -> global re-clustering for consistent FINAL IDs")
    ap.add_argument("--global-labels", action="store_true")
    ap.add_argument("--separate-vocals", action="store_true", help="Demucs vocal separation (music-heavy)")
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        _STOP.set()
        print("\nstopped.")


if __name__ == "__main__":
    main()
