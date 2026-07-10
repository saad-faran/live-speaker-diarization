#!/usr/bin/env python3
"""
compare_labelers.py — measure LIVE labeling accuracy against the OFFLINE gold, on YOUR audio.

    python compare_labelers.py path/to/audio.wav
    python compare_labelers.py path/to/audio.wav --speakers 3     # known cast size

Runs, on the same file:
  * OFFLINE (whole-file) diarization  -> the gold reference
  * LIVE, registry labels             -> current default
  * LIVE, global re-clustering (auto)  -> --global-labels
  * LIVE, global re-clustering to N     -> --speakers N   (if you pass --speakers)
and prints each live variant's frame-level agreement with the gold (labels best-mapped).

Use this to pick the best labeler *for your kind of audio* and to tune --window / --threshold,
instead of guessing. Needs the offline repo checked out next to this one, OR pass --offline-dir.
"""
import os, sys, argparse, numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
from live_diarize import LiveDiarizer, SR

ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("--speakers", type=int, default=None)
ap.add_argument("--window", type=float, default=15.0)
ap.add_argument("--threshold", type=float, default=0.85)
ap.add_argument("--offline-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "..", "offline-speaker-diarization"))
args = ap.parse_args()

sys.path.insert(0, os.path.abspath(args.offline_dir))
try:
    import diarizer as off
except Exception as e:
    sys.exit(f"Could not import the offline diarizer from {args.offline_dir}\n"
             f"Clone https://github.com/saad-faran/offline-speaker-diarization next to this repo, "
             f"or pass --offline-dir. ({e})")

audio, sr = sf.read(args.audio, dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
dur = len(audio) / SR
print(f"audio: {args.audio}  ({dur:.0f}s)\n")


def frame_acc(gold, pred, hop=0.1):
    from collections import Counter, defaultdict
    def grid(bl):
        g = [None] * int(dur / hop)
        for s, e, sid in bl:
            for i in range(int(s / hop), min(int(e / hop), len(g))):
                g[i] = sid
        return g
    g, p = grid(gold), grid(pred)
    ov = defaultdict(Counter)
    for gi, pi in zip(g, p):
        if gi is not None and pi is not None:
            ov[pi][gi] += 1
    mp = {pi: c.most_common(1)[0][0] for pi, c in ov.items()}
    both = corr = 0
    for gi, pi in zip(g, p):
        if gi is not None and pi is not None:
            both += 1
            corr += (mp.get(pi) == gi)
    return 100.0 * corr / max(both, 1)


gpipe = off.load_pipeline(clustering_threshold=args.threshold)
gold, gn = off.diarize_file(args.audio, pipe=gpipe,
                            num_speakers=args.speakers,
                            min_speaker_dur=0.0 if args.speakers else 20.0)
print(f"[OFFLINE gold] {gn} speakers, {len(gold)} turns\n")

ld = LiveDiarizer(window=args.window, stride=2.0, threshold=args.threshold, commit_lag=2.0)
win_n, stride_n = ld.win_n, int(2 * SR)
end = stride_n
while end <= len(audio):
    buf = audio[max(0, end - win_n):end]
    if len(buf) >= ld.warmup_n:
        ld.process(buf, end / SR)
    end += stride_n
ld.process(audio[max(0, len(audio) - win_n):len(audio)], len(audio) / SR, flush=True)

reg = ld.finalize()
glob = ld.finalize_global()
print("LIVE labeling vs offline gold (higher = better):")
print(f"  registry (default)          : {frame_acc(gold, reg):5.1f}%   ({len(set(b[2] for b in reg))} spk)")
print(f"  global re-cluster (auto)     : {frame_acc(gold, glob):5.1f}%   ({len(set(b[2] for b in glob))} spk)")
if args.speakers:
    gN = ld.finalize_global(num_speakers=args.speakers)
    print(f"  global re-cluster (N={args.speakers})        : {frame_acc(gold, gN):5.1f}%   ({len(set(b[2] for b in gN))} spk)")
print("\nTip: try --speakers <cast size> and --window 20/25 to see which lifts accuracy on your audio.")
