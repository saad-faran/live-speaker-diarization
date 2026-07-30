#!/usr/bin/env python3
"""
live_app.py — paste-a-URL live diarization console.

A small server + browser dashboard. Paste any YouTube URL (live OR recorded) — or a
direct HLS/RTMP/icecast URL — and it diarizes the audio in real time, streaming
speaker-labeled subtitles to the page with a per-speaker roster (unique speakers,
turn counts, talk time, who just (re)appeared). Swap the URL any time.

    python live_app.py                       # then open the printed URL, paste a link
    # for live YouTube (anti-bot), start it once with your working flags:
    python live_app.py --cookies-from-browser chrome --js-runtime deno --remote-components ejs:npm

Design
------
    browser  --(WebSocket: start/stop/mark)-->  server
    server:  yt-dlp/ffmpeg (-re = real-time) -> PCM
             -> LiveDiarizer (ONLINE registry, honest live; NO global re-clustering)
             -> faster-whisper (overlapping windows, boundary-safe)
             -> events --(WebSocket)--> browser  (subtitles + roster, live)
    Every event is also appended to session.jsonl (+ session.srt). A "Mark issue"
    button timestamps a marker in the log at the exact stream position — so when you
    SEE a mislabel, click it, then send me session.jsonl and I can jump straight there.

The models load ONCE at startup and are reused across URL swaps (fast switching).
"""
import os
import sys
import json
import time
import shutil
import asyncio
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

import websockets
from core import pick_device, load_pipeline, SpeakerRegistry, ensure_av
from live_diarize import resolve_stream, _tool, SR, LiveDiarizer

WS_PORT = 8766
HTTP_PORT = 8771

_clients = set()
_loop = None


async def _send_all(msg):
    for ws in list(_clients):
        try:
            await ws.send(msg)
        except Exception:
            _clients.discard(ws)


def broadcast(evt):
    if _loop is not None:
        asyncio.run_coroutine_threadsafe(_send_all(json.dumps(evt)), _loop)


# ── static + session-file server (with no-cache for the live log) ────────────
class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        if self.path.startswith("/session."):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        try:
            super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            pass


def _start_http(directory):
    handler = lambda *a, **k: _Handler(*a, directory=directory, **k)
    httpd = socketserver.ThreadingTCPServer(("localhost", HTTP_PORT), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _srt_ts(t):
    t = max(0.0, t); h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _hallucination_flags(text, dur, speaker, recent):
    flags = []
    w = text.split()
    if speaker is None:
        flags.append("no_speaker")
    if len(w) >= 4 and len(set(w)) <= max(2, len(w) // 4):
        flags.append("repeated_words")
    if recent and text.strip().lower() == recent[-1].strip().lower():
        flags.append("dup_prev")
    if dur > 0 and len(w) / dur > 6.0:
        flags.append("too_fast")
    return flags


def _speaker_at(blocks, t):
    best = None
    for s, e, sid in blocks:
        if s <= t < e:
            return sid
        if s <= t:
            best = sid
    return best


class Manager:
    """Owns the single reusable engine and the current streaming session."""

    def __init__(self, args):
        self.args = args
        self.device = pick_device()
        print(f"-> loading models on {self.device.upper()} (one time)...", flush=True)
        # load the heavy models ONCE; reuse across URL swaps
        self.pipe = load_pipeline(device=self.device, clustering_threshold=args.threshold)
        ensure_av()
        from faster_whisper import WhisperModel
        self.asr = WhisperModel(args.asr_model,
                                device="cuda" if self.device == "cuda" else "cpu",
                                compute_type="float16" if self.device == "cuda" else "int8")
        self.ld = LiveDiarizer.__new__(LiveDiarizer)      # shell; we inject the shared pipe
        self._init_engine()
        self.thread = None
        self.stop_evt = threading.Event()
        self.proc = None
        self.media_t = 0.0
        self.log_f = None
        self.srt_f = None
        self.srt_n = 0
        self.session_dir = os.getcwd()
        print("-> models ready.", flush=True)

    def _init_engine(self):
        ld = self.ld
        a = self.args
        ld.device = self.device
        ld.pipe = self.pipe
        ld.reg = SpeakerRegistry(match_sim=a.match_sim)
        ld.win_n = int(15.0 * SR)
        ld.stride = 1.5
        ld.warmup_n = int(4.0 * SR)
        ld.commit_lag = 1.0
        ld.min_change_dur = 0.4
        ld.merge_gap = 0.4
        ld.buf = np.zeros(0, dtype=np.float32)
        ld.total = 0
        ld.lock = threading.Lock()
        ld.stop = False
        ld.raw = []
        ld.committed_until = 0.0

    # ── session control (called from the WS handler / async loop) ────────────
    def start(self, url):
        self.stop()
        self._init_engine()
        self.stop_evt = threading.Event()
        self.media_t = 0.0
        ts = time.strftime("%Y%m%d-%H%M%S")
        # roll previous logs to an archive, start fresh "session.*" for easy download
        for base in ("session.jsonl", "session.srt"):
            p = os.path.join(self.session_dir, base)
            if os.path.exists(p):
                try:
                    os.replace(p, os.path.join(self.session_dir, f"_{ts}_{base}"))
                except Exception:
                    pass
        self.log_f = open(os.path.join(self.session_dir, "session.jsonl"), "w", encoding="utf-8")
        self.srt_f = open(os.path.join(self.session_dir, "session.srt"), "w", encoding="utf-8")
        self.srt_n = 0
        self.log({"type": "session", "url": url, "started": ts,
                  "device": self.device, "asr": self.args.asr_model})
        self.thread = threading.Thread(target=self._run, args=(url,), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_evt.set()
        if self.proc:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        for f in (self.log_f, self.srt_f):
            try:
                if f:
                    f.flush()
            except Exception:
                pass

    def mark(self, note):
        self.log({"type": "mark", "t": round(self.media_t, 2), "note": note or ""})
        broadcast({"type": "marked", "t": round(self.media_t, 2), "note": note or ""})

    def log(self, evt):
        broadcast(evt)
        if self.log_f and evt.get("type") in ("session", "speaker", "caption", "mark", "status"):
            try:
                self.log_f.write(json.dumps(evt) + "\n")
                self.log_f.flush()
            except Exception:
                pass

    def _srt(self, s, e, sid, text):
        self.srt_n += 1
        self.srt_f.write(f"{self.srt_n}\n{_srt_ts(s)} --> {_srt_ts(max(e, s + 0.4))}\n"
                         f"[SPEAKER {sid}] {text}\n\n")
        self.srt_f.flush()

    # ── the streaming pipeline (worker thread) ───────────────────────────────
    def _run(self, url):
        a = self.args
        ld = self.ld
        try:
            self.log({"type": "status", "msg": "resolving stream…"})
            src = resolve_stream(url, cookies_from_browser=a.cookies_from_browser,
                                 js_runtime=a.js_runtime, remote_components=a.remote_components,
                                 cookies=a.cookies)
        except Exception as e:
            self.log({"type": "status", "msg": f"could not open URL: {e}"})
            return
        # -re paces both live streams AND recorded VODs to real-time (1x)
        self.proc = subprocess.Popen(
            [_tool("ffmpeg"), "-loglevel", "quiet", "-re", "-i", src,
             "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
            stdout=subprocess.PIPE, bufsize=10 ** 7)
        self.log({"type": "status", "msg": f"● live on {self.device.upper()} — diarizing"})

        t0 = time.time()
        step = int(0.5 * SR) * 2
        fed = 0
        asr_audio = np.zeros(0, dtype=np.float32)
        asr_start = 0.0
        emitted_until = 0.0
        overlap = a.asr_overlap
        next_diar = ld.stride
        next_asr = a.asr_interval
        cur = None
        changes = 0
        seen = set()
        stats = {}                      # sid -> {turns, talk, last_t}
        recent = []
        last_stats = 0.0

        while not self.stop_evt.is_set():
            raw = self.proc.stdout.read(step)
            if not raw:
                break
            pkt = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            ld.feed(pkt)
            fed += len(pkt)
            media_t = fed / SR
            self.media_t = media_t
            asr_audio = np.concatenate([asr_audio, pkt])

            # ---- diarization ----
            if media_t >= next_diar:
                next_diar = media_t + ld.stride
                buf, tot = ld._snapshot()
                if len(buf) >= ld.warmup_n:
                    ld.process(buf, tot / SR)
                    blocks = ld.finalize()
                    if blocks:
                        # per-speaker talk time + roster
                        talk = {}
                        for s, e, sid in blocks:
                            talk[sid] = talk.get(sid, 0.0) + (e - s)
                        sid = blocks[-1][2]
                        is_new = sid not in seen
                        seen.add(sid)
                        for k in talk:
                            st = stats.setdefault(k, {"turns": 0, "talk": 0.0, "last_t": 0.0})
                            st["talk"] = round(talk[k], 1)
                        if sid != cur:
                            cur = sid
                            changes += 1
                            st = stats.setdefault(sid, {"turns": 0, "talk": 0.0, "last_t": 0.0})
                            st["turns"] += 1
                            st["last_t"] = round(media_t, 1)
                            wall = time.time() - t0
                            self.log({"type": "speaker", "speaker": sid, "changes": changes,
                                      "total_speakers": len(seen), "t": round(media_t, 1),
                                      "is_new": is_new, "latency": round(wall - media_t, 2)})
                        if media_t - last_stats >= 1.0:
                            last_stats = media_t
                            broadcast({"type": "stats", "t": round(media_t, 1),
                                       "total_speakers": len(seen), "changes": changes,
                                       "current": cur,
                                       "roster": [{"speaker": k, "turns": v["turns"],
                                                   "talk": v["talk"], "last_t": v["last_t"]}
                                                  for k, v in sorted(stats.items())]})

            # ---- ASR (overlapping, boundary-safe) ----
            if media_t >= next_asr:
                next_asr = media_t + a.asr_interval
                seg_from = max(emitted_until - overlap, asr_start)
                i0 = int(round((seg_from - asr_start) * SR))
                chunk = asr_audio[max(i0, 0):]
                if len(chunk) >= SR:
                    try:
                        segs, _ = self.asr.transcribe(
                            chunk, language=a.language, beam_size=5, vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=400),
                            condition_on_previous_text=False, word_timestamps=True)
                        blocks = ld.finalize()
                        for sg in segs:
                            words = [w for w in (sg.words or [])
                                     if seg_from + w.end > emitted_until + 0.05]
                            if sg.words:
                                if not words:
                                    continue
                                txt = "".join(w.word for w in words).strip()
                                a_start = seg_from + words[0].start
                                a_end = seg_from + words[-1].end
                            else:
                                txt = sg.text.strip()
                                a_start, a_end = seg_from + sg.start, seg_from + sg.end
                                if a_end <= emitted_until + 0.05:
                                    continue
                            if not txt:
                                continue
                            emitted_until = max(emitted_until, a_end)
                            mid = (a_start + a_end) / 2.0
                            spk = _speaker_at(blocks, mid)
                            dur = max(a_end - a_start, 0.1)
                            flags = _hallucination_flags(txt, dur, spk, recent)
                            recent.append(txt); recent[:] = recent[-5:]
                            disp = spk if spk is not None else (cur or 1)
                            wall = time.time() - t0
                            self.log({"type": "caption", "speaker": disp, "text": txt,
                                      "t": round(a_start, 1), "end_t": round(a_end, 1),
                                      "latency": round(wall - a_end, 2), "flags": flags})
                            self._srt(a_start, a_end, disp, txt)
                    except Exception as e:
                        self.log({"type": "status", "msg": f"asr error: {e}"})
                    keep_from = max(emitted_until - overlap - 1.0, asr_start)
                    drop = int(round((keep_from - asr_start) * SR))
                    if drop > 0:
                        asr_audio = asr_audio[drop:]
                        asr_start = keep_from

        self.log({"type": "status", "msg": "stream ended / stopped"})


_MGR = None


async def _ws_handler(ws, *_):
    _clients.add(ws)
    try:
        broadcast({"type": "status", "msg": "connected — paste a URL and press Start"})
        async for raw in ws:
            try:
                m = json.loads(raw)
            except Exception:
                continue
            cmd = m.get("cmd")
            if cmd == "start" and m.get("url"):
                _MGR.start(m["url"].strip())
            elif cmd == "stop":
                _MGR.stop()
                broadcast({"type": "status", "msg": "stopped"})
            elif cmd == "mark":
                _MGR.mark(m.get("note", ""))
    finally:
        _clients.discard(ws)


async def main_async(args):
    global _loop, _MGR
    serve_dir = os.getcwd()
    # copy the dashboard next to the session files so it loads over http
    src = os.path.join(os.path.dirname(__file__), "app.html")
    dst = os.path.join(serve_dir, "app.html")
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)
    _start_http(serve_dir)
    _MGR = Manager(args)
    _MGR.session_dir = serve_dir

    _loop = asyncio.get_running_loop()
    async with websockets.serve(_ws_handler, "localhost", WS_PORT):
        url = f"http://localhost:{HTTP_PORT}/app.html"
        print(f"\n-> open the console:  {url}\n-> paste a YouTube (live or recorded) or direct stream URL, press Start\n", flush=True)
        try:
            webbrowser.open(url)
        except Exception:
            pass
        while True:
            await asyncio.sleep(3600)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Paste-a-URL live diarization console.")
    ap.add_argument("--asr-model", default="small", help="faster-whisper: tiny|base|small|medium")
    ap.add_argument("--asr-interval", type=float, default=6.0)
    ap.add_argument("--asr-overlap", type=float, default=2.0)
    ap.add_argument("--language", default=None, help="force ASR language (e.g. en, ur); default auto")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--match-sim", type=float, default=0.30)
    ap.add_argument("--cookies-from-browser", default=None)
    ap.add_argument("--cookies", default=None,
                    help="path to an exported cookies.txt (use when --cookies-from-browser "
                         "fails, e.g. Chrome's locked cookie DB on Windows)")
    ap.add_argument("--js-runtime", default=None)
    ap.add_argument("--remote-components", default=None)
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        if _MGR:
            _MGR.stop()
        print("\nstopped.")


if __name__ == "__main__":
    main()
