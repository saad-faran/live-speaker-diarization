#!/usr/bin/env python3
"""
live_captions.py — real-time speaker diarization + speaker-labeled captions, streamed
to a browser over WebSocket. The end goal of live diarization.

    python live_captions.py "https://www.youtube.com/watch?v=<LIVE_ID>" \
        --cookies-from-browser chrome --js-runtime deno --remote-components ejs:npm
    # then open captions.html in your browser

Pipeline (one stream, two consumers, honest online behavior):
    yt-dlp/ffmpeg -> PCM
       -> LiveDiarizer (ONLINE registry, no global re-clustering)  -> speaker turns
       -> faster-whisper (ASR on rolling chunks)                    -> text segments
       -> merge (each caption tagged with the speaker talking then) -> WebSocket events
A browser dashboard renders: current speaker, accumulating speaker + change counts,
and a rolling per-speaker caption feed. Runs indefinitely while the stream is live.
"""
import os
import sys
import json
import time
import asyncio
import threading
import subprocess
import warnings
import numpy as np

warnings.filterwarnings("ignore")
np.seterr(invalid="ignore", divide="ignore")
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import websockets
from core import pick_device
from live_diarize import resolve_stream, _tool, SR, LiveDiarizer

PORT = 8765
_clients = set()
_loop = None


async def _send_all(msg):
    for ws in list(_clients):
        try:
            await ws.send(msg)
        except Exception:
            _clients.discard(ws)


def emit(evt):
    if _loop is not None:
        asyncio.run_coroutine_threadsafe(_send_all(json.dumps(evt)), _loop)


async def _ws_handler(ws, *_):
    _clients.add(ws)
    emit({"type": "status", "msg": "connected"})
    try:
        async for _m in ws:
            pass
    finally:
        _clients.discard(ws)


def _speaker_at(blocks, t):
    """Stable speaker id active at time t (nearest preceding turn if in a gap)."""
    best = None
    for s, e, sid in blocks:
        if s <= t < e:
            return sid
        if s <= t:
            best = sid
    return best


def pipeline(args):
    print(f"-> resolving stream: {args.input}", flush=True)
    src = resolve_stream(args.input, cookies_from_browser=args.cookies_from_browser,
                         js_runtime=args.js_runtime, remote_components=args.remote_components)
    proc = subprocess.Popen([_tool("ffmpeg"), "-loglevel", "quiet", "-i", src,
                             "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                            stdout=subprocess.PIPE, bufsize=10 ** 7)
    device = pick_device()
    ld = LiveDiarizer(threshold=args.threshold)          # ONLINE registry (true-live)
    from faster_whisper import WhisperModel
    asr = WhisperModel(args.asr_model,
                       device="cuda" if device == "cuda" else "cpu",
                       compute_type="float16" if device == "cuda" else "int8")
    emit({"type": "status", "msg": f"live on {device.upper()} — diarizer + {args.asr_model} ASR"})

    total = 0
    asr_buf = np.zeros(0, dtype=np.float32)
    asr_base = 0.0
    cur = None
    changes = 0
    seen = set()
    next_diar = time.time() + ld.stride
    next_asr = time.time() + args.asr_interval
    step = int(0.5 * SR) * 2
    while True:
        raw = proc.stdout.read(step)
        if not raw:
            break
        pkt = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        ld.feed(pkt)
        total += len(pkt)
        asr_buf = np.concatenate([asr_buf, pkt])
        now = time.time()

        if now >= next_diar:
            next_diar = now + ld.stride
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
                        emit({"type": "speaker", "speaker": sid, "changes": changes,
                              "total_speakers": len(seen), "t": round(tot / SR, 1)})

        if now >= next_asr and len(asr_buf) >= args.asr_interval * SR:
            next_asr = now + args.asr_interval
            seg_audio = asr_buf
            base = asr_base
            asr_base += len(asr_buf) / SR
            asr_buf = np.zeros(0, dtype=np.float32)
            try:
                segs, _ = asr.transcribe(seg_audio, vad_filter=True)
                blocks = ld.finalize()
                for sg in segs:
                    txt = sg.text.strip()
                    if not txt:
                        continue
                    mid = base + (sg.start + sg.end) / 2.0
                    spk = _speaker_at(blocks, mid) or cur or 1
                    emit({"type": "caption", "speaker": spk, "text": txt,
                          "t": round(base + sg.start, 1)})
            except Exception as e:
                emit({"type": "status", "msg": f"asr error: {e}"})
    emit({"type": "status", "msg": "stream ended"})


async def main_async(args):
    global _loop
    _loop = asyncio.get_running_loop()
    async with websockets.serve(_ws_handler, "localhost", PORT):
        print(f"WebSocket up on ws://localhost:{PORT}", flush=True)
        print("-> open captions.html in your browser now", flush=True)
        t = threading.Thread(target=pipeline, args=(args,), daemon=True)
        t.start()
        while t.is_alive():
            await asyncio.sleep(1)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Real-time diarization + captions over WebSocket.")
    ap.add_argument("input", help="live stream URL (or direct HLS/RTMP)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--asr-model", default="small", help="faster-whisper model: tiny|base|small|medium")
    ap.add_argument("--asr-interval", type=float, default=6.0, help="transcribe every N seconds")
    ap.add_argument("--cookies-from-browser", default=None)
    ap.add_argument("--js-runtime", default=None)
    ap.add_argument("--remote-components", default=None)
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
