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
import torch
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


def _reg_at(raw, t):
    """Registry stable-id of the committed turn active at time t (raw = ld.raw)."""
    best = None
    for s, e, sid, _emb in raw:
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
        ld.stride = a.stride
        ld.warmup_n = int(3.0 * SR)
        ld.commit_lag = a.commit_lag
        ld.min_change_dur = 0.4
        ld.merge_gap = 0.4
        ld.buf = np.zeros(0, dtype=np.float32)
        ld.total = 0
        ld.lock = threading.Lock()
        ld.stop = False
        ld.raw = []
        ld.committed_until = 0.0
        # incremental global re-clustering state (drift-free, consistent IDs)
        self.gcentroids = {}       # stable_id -> unit centroid (recomputed each recluster)
        self.gnext = 1
        self.last_turns = []       # merged global timeline (authoritative IDs)
        self.cur_global = None
        self.reg2global = {}       # registry stable_id -> global id (for FRESH-tail tagging)
        self.fast_sid = None       # fast detector's current speaker ('NEW' = provisional)

    # ── incremental global re-clustering ─────────────────────────────────────
    @staticmethod
    def _merge_keep(turns, merge_gap, min_dur):
        """Merge adjacent same-ID turns, PRESERVING the ids (unlike LiveDiarizer's
        _merge_relabel which renumbers)."""
        merged = []
        for s, e, sid in sorted(turns):
            if merged and merged[-1][2] == sid and s - merged[-1][1] <= merge_gap:
                merged[-1] = [merged[-1][0], max(merged[-1][1], e), sid]
            else:
                merged.append([s, e, sid])
        return [(s, e, sid) for s, e, sid in merged if e - s >= min_dur]

    def _recluster(self):
        """Re-cluster ALL committed turn embeddings from scratch (drift-free), then
        RECONCILE the clusters to persistent stable IDs by centroid similarity, so a
        speaker keeps the same ID across the whole stream no matter how long it runs,
        and short turns that were provisionally merged get corrected using every
        speaker's full evidence. Causal (only past turns) → honest online."""
        from scipy.cluster.hierarchy import linkage, fcluster
        raw = list(self.ld.raw)
        items = [(s, e, emb, sid) for (s, e, sid, emb) in raw
                 if emb is not None and not np.isnan(emb).any()]
        if len(items) < 3:
            return None
        MAX = 2500                                   # bound O(n^2); persistent centroids keep old IDs
        if len(items) > MAX:
            items = items[-MAX:]
        X = np.array([it[2] for it in items], dtype=float)
        Z = linkage(X, method="average", metric="cosine")
        lab = fcluster(Z, t=self.args.recluster_dist, criterion="distance")
        clu = {}
        for i, c in enumerate(lab):
            clu.setdefault(int(c), []).append(i)

        def unit(v):
            return v / (np.linalg.norm(v) + 1e-9)
        cen = {c: unit(X[idxs].mean(0)) for c, idxs in clu.items()}
        # reconcile new clusters to existing global IDs (greedy by cosine)
        pairs = sorted(((float(cen[c] @ gc), c, sid)
                        for c in cen for sid, gc in self.gcentroids.items()),
                       reverse=True, key=lambda x: x[0])
        cmap, used = {}, set()
        for sim, c, sid in pairs:
            if c in cmap or sid in used or sim < 0.5:
                continue
            cmap[c] = sid; used.add(sid)
        for c in clu:
            if c not in cmap:
                cmap[c] = self.gnext; self.gnext += 1
        self.gcentroids = {cmap[c]: cen[c] for c in clu}     # recomputed → no EMA drift
        # registry stable_id -> global id, by majority, so FRESH-tail captions (not yet
        # reclustered) can be tagged from the online registry instead of the previous ID.
        votes = {}
        for i in range(len(items)):
            reg_sid, gid = items[i][3], cmap[int(lab[i])]
            votes.setdefault(reg_sid, {}).setdefault(gid, 0)
            votes[reg_sid][gid] += 1
        self.reg2global = {r: max(v, key=v.get) for r, v in votes.items()}
        turns = [(items[i][0], items[i][1], cmap[int(lab[i])]) for i in range(len(items))]
        return self._merge_keep(turns, self.ld.merge_gap, self.ld.min_change_dur)

    @staticmethod
    def _roster(merged):
        info = {}
        last = None
        for s, e, sid in merged:
            d = info.setdefault(sid, {"turns": 0, "talk": 0.0, "last_t": 0.0})
            d["talk"] += e - s
            d["last_t"] = max(d["last_t"], e)
            if sid != last:
                d["turns"] += 1; last = sid
        return [{"speaker": sid, "turns": d["turns"], "talk": round(d["talk"], 1),
                 "last_t": round(d["last_t"], 1)} for sid, d in sorted(info.items())]

    # ── fast change detection (Tier 1) ───────────────────────────────────────
    def _fast_current(self, audio, decide_n):
        """Diarize a SHORT recent window (not dominated by the previous speaker) and
        return who is talking in its last ~1.5 s, matched against the drift-free
        global centroids. Returns (global_id, sim) for a confident known speaker,
        ('NEW', sim) for an as-yet-unknown voice (→ show provisional, never the old
        ID), or (None, 0) when there isn't a usable signal yet."""
        if decide_n <= 0 or len(audio) < self.ld.warmup_n:
            return None, 0.0
        win = audio[-decide_n:]
        out = self.ld.pipe({"waveform": torch.from_numpy(win).unsqueeze(0), "sample_rate": SR})
        if self.device == "mps":
            torch.mps.empty_cache()
        ann = out.exclusive_speaker_diarization
        embs = out.speaker_embeddings
        labels = ann.labels()
        end = len(win) / SR
        dom = {}                                       # who dominates the last 1.5 s
        for turn, _, label in ann.itertracks(yield_label=True):
            ov = max(0.0, min(turn.end, end) - max(turn.start, end - 1.5))
            if ov > 0:
                dom[label] = dom.get(label, 0.0) + ov
        if not dom:
            return None, 0.0
        label = max(dom, key=dom.get)
        i = labels.index(label) if label in labels else -1
        emb = embs[i] if (embs is not None and 0 <= i < len(embs)) else None
        if emb is None or np.isnan(emb).any():
            return None, 0.0
        u = emb / (np.linalg.norm(emb) + 1e-9)
        best_sid, best = None, -1.0
        for sid, gc in self.gcentroids.items():
            sim = float(u @ gc)
            if sim > best:
                best, best_sid = sim, sid
        if best >= self.args.fast_match:
            return best_sid, best
        return "NEW", best

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
                                 cookies=a.cookies, player_client=a.yt_player_client,
                                 extra_args=a.ytdlp_arg)
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
        next_clock = 1.0
        next_recluster = a.recluster_sec if a.recluster_sec > 0 else float("inf")
        decide_n = int(a.decide_window * SR)
        last_fast = None
        recent = []
        last_word = ""

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

            # ---- diarization: commit fine-grained turns + FAST change detection ----
            if media_t >= next_diar:
                next_diar = media_t + ld.stride
                buf, tot = ld._snapshot()
                if len(buf) >= ld.warmup_n:
                    ld.process(buf, tot / SR)
                    # fast detector: who's talking NOW, from a short window (Tier 1)
                    if decide_n > 0 and self.gcentroids:
                        try:
                            fsid, fsim = self._fast_current(buf, decide_n)
                        except Exception:
                            fsid, fsim = None, 0.0
                        if fsid is not None:
                            self.fast_sid = fsid
                            prov = (fsid == "NEW")
                            cur = None if prov else fsid
                            if (cur, prov) != last_fast:
                                last_fast = (cur, prov)
                                broadcast({"type": "current", "t": round(media_t, 1),
                                           "speaker": cur, "provisional": prov,
                                           "sim": round(fsim, 2)})

            # ---- lightweight clock for the dashboard's video-sync frontier ----
            if media_t >= next_clock:
                next_clock = media_t + 1.0
                broadcast({"type": "clock", "t": round(media_t, 1)})

            # ---- incremental global re-clustering → consistent, drift-free IDs ----
            if media_t >= next_recluster:
                next_recluster = media_t + a.recluster_sec
                try:
                    merged = self._recluster()
                except Exception as e:
                    merged = None
                    broadcast({"type": "status", "msg": f"recluster error: {e}"})
                if merged:
                    self.last_turns = merged
                    self.cur_global = merged[-1][2]
                    roster = self._roster(merged)
                    n_changes = sum(1 for i in range(1, len(merged)) if merged[i][2] != merged[i - 1][2])
                    broadcast({"type": "relabel", "t": round(media_t, 1),
                               "current": self.cur_global, "total_speakers": len(self.gcentroids),
                               "changes": n_changes, "roster": roster,
                               "timeline": [[round(s, 1), round(e, 1), sid] for s, e, sid in merged[-200:]]})
                    if self.log_f:
                        self.log_f.write(json.dumps({"type": "relabel", "t": round(media_t, 1),
                                                     "current": self.cur_global,
                                                     "total_speakers": len(self.gcentroids),
                                                     "changes": n_changes, "roster": roster}) + "\n")
                        self.log_f.flush()

            # ---- ASR (overlapping, boundary-safe) ----
            if media_t >= next_asr:
                next_asr = media_t + a.asr_interval
                seg_from = max(emitted_until - overlap, asr_start)
                i0 = int(round((seg_from - asr_start) * SR))
                chunk = asr_audio[max(i0, 0):]
                if len(chunk) >= SR:
                    try:
                        segs, _ = self.asr.transcribe(
                            chunk, language=a.language, beam_size=5,
                            vad_filter=not a.no_vad,
                            vad_parameters=dict(threshold=0.3, min_silence_duration_ms=300,
                                                min_speech_duration_ms=0, speech_pad_ms=200),
                            condition_on_previous_text=False, word_timestamps=True)
                        blocks = ld.finalize()
                        norm = lambda s: s.strip().lower().strip(".,!?\"'-")
                        for sg in segs:
                            # START-based dedup: keep only words that begin at/after what we've
                            # already emitted, so a boundary word the overlap re-transcribes
                            # (with a slightly shifted end time) is not shown twice.
                            words = [w for w in (sg.words or [])
                                     if seg_from + w.start >= emitted_until - 0.10]
                            # belt-and-suspenders: strip leading word(s) that repeat the last emitted
                            while words and last_word and norm(words[0].word) == last_word:
                                words = words[1:]
                            if sg.words:
                                if not words:
                                    continue
                                txt = "".join(w.word for w in words).strip()
                                a_start = seg_from + words[0].start
                                a_end = seg_from + words[-1].end
                                last_word = norm(words[-1].word)
                            else:
                                txt = sg.text.strip()
                                a_start, a_end = seg_from + sg.start, seg_from + sg.end
                                if a_end <= emitted_until + 0.05:
                                    continue
                            if not txt:
                                continue
                            emitted_until = max(emitted_until, a_end)
                            mid = (a_start + a_end) / 2.0
                            spk_o = _speaker_at(blocks, mid)            # online presence (for flags)
                            # identity, freshest-first: reclustered → fast detector →
                            # registry→global map. Never fall back to the previous ID:
                            # if we can't tell yet, mark the caption PROVISIONAL.
                            spk_g = _speaker_at(self.last_turns, mid)   # settled (reclustered)
                            provisional = False
                            if spk_g is None:                            # fresh tail
                                fs = self.fast_sid
                                if fs == "NEW":
                                    provisional = True
                                elif fs is not None:
                                    spk_g = fs
                                else:
                                    rsid = _reg_at(ld.raw, mid)
                                    spk_g = self.reg2global.get(rsid) if rsid is not None else None
                            if spk_g is None and not provisional:
                                provisional = True
                            disp = spk_g if spk_g is not None else (self.cur_global or 1)
                            dur = max(a_end - a_start, 0.1)
                            flags = _hallucination_flags(txt, dur, spk_o, recent)
                            recent.append(txt); recent[:] = recent[-5:]
                            wall = time.time() - t0
                            self.log({"type": "caption", "speaker": disp, "text": txt,
                                      "provisional": provisional,
                                      "t": round(a_start, 1), "end_t": round(a_end, 1),
                                      "latency": round(wall - a_end, 2), "flags": flags})
                            self._srt(a_start, a_end, ("?" if provisional else disp), txt)
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
    ap.add_argument("--asr-model", default="medium",
                    help="faster-whisper: tiny|base|small|medium|large-v3 (heavier = more accurate)")
    ap.add_argument("--asr-interval", type=float, default=2.5,
                    help="transcribe every N s — LOWER = lower caption latency (needs a fast GPU)")
    ap.add_argument("--asr-overlap", type=float, default=2.0)
    ap.add_argument("--no-vad", action="store_true",
                    help="disable VAD filtering (captures every word; may hallucinate on silence/music)")
    ap.add_argument("--recluster-sec", type=float, default=2.0,
                    help="re-cluster all turns every N s for consistent, drift-free IDs (0 disables)")
    ap.add_argument("--recluster-dist", type=float, default=0.55,
                    help="agglomerative distance cutoff for re-clustering (lower = more speakers)")
    ap.add_argument("--stride", type=float, default=1.0,
                    help="diarization commit cadence (s) — lower = faster speaker switches")
    ap.add_argument("--commit-lag", type=float, default=0.5,
                    help="hold back the newest N s before committing (lower = faster, less context)")
    ap.add_argument("--decide-window", type=float, default=6.0,
                    help="short window (s) diarized each stride for FAST speaker-change detection; "
                         "0 disables. Not dominated by the previous speaker, so a new voice is "
                         "spotted in ~1s instead of waiting for the 15s window to split it.")
    ap.add_argument("--fast-match", type=float, default=0.5,
                    help="cosine cutoff for the fast detector to trust a known speaker; below this "
                         "the new voice is shown as provisional ('identifying…') — never the old ID")
    ap.add_argument("--language", default=None, help="force ASR language (e.g. en, ur); default auto")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--match-sim", type=float, default=0.30)
    ap.add_argument("--cookies-from-browser", default=None)
    ap.add_argument("--cookies", default=None,
                    help="path to an exported cookies.txt (use when --cookies-from-browser "
                         "fails, e.g. Chrome's locked cookie DB on Windows)")
    ap.add_argument("--js-runtime", default=None)
    ap.add_argument("--remote-components", default=None)
    ap.add_argument("--yt-player-client", default=None,
                    help="YouTube client(s) to avoid the JS challenge (no deno needed), "
                         "e.g. tv,web_safari,android — use when Application Control blocks deno")
    ap.add_argument("--ytdlp-arg", action="append", default=None,
                    help="extra raw yt-dlp argument (repeatable)")
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        if _MGR:
            _MGR.stop()
        print("\nstopped.")


if __name__ == "__main__":
    main()
