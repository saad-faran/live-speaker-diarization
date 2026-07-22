#!/usr/bin/env python3
"""
review_report.py — turn a run's logs into a short "where to look" report, so you
don't have to watch the whole thing. Reads review_log.jsonl (+ live_result.json)
from one or more run folders and writes review_report.md in each, flagging the
timestamps worth eyeballing:

    * ASR hallucinations        (captions the pipeline already flagged)
    * very short speaker turns   (< SHORT_SEG s — the classic mislabel case)
    * rapid speaker flip-flops   (many changes in a short span)
    * long caption gaps          (silence / music / ASR dropout)
    * per-speaker talk-time       (spot phantom speakers / who dominates)
    * detection latency stats     (real-time runs only)

Usage:
    python review_report.py runs/stream1                 # one run
    python review_report.py runs/stream1 runs/stream2 …  # many -> also an index

Jump to any listed timestamp in the dashboard (click the timeline) or in a player
with captions.srt loaded (VLC: Subtitle > Add Subtitle File).
"""
import os
import sys
import json

SHORT_SEG = 2.0        # speaker turns shorter than this are mislabel-prone
FLIP_WIN = 10.0        # window to count rapid speaker flips
FLIP_N = 4             # >= this many changes within FLIP_WIN = flip-flop hotspot
GAP_S = 30.0           # caption gap longer than this is worth a look


def _fmt(t):
    t = max(0.0, float(t)); h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    return (f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}")


def _load(dir_):
    log = os.path.join(dir_, "review_log.jsonl")
    if not os.path.isfile(log):
        return None
    speakers, captions = [], []
    with open(log, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            (speakers if e.get("type") == "speaker" else
             captions if e.get("type") == "caption" else []).append(e)
    result = None
    rp = os.path.join(dir_, "live_result.json")
    if os.path.isfile(rp):
        with open(rp, encoding="utf-8") as f:
            result = json.load(f)
    return {"speakers": speakers, "captions": captions, "result": result}


def analyze(dir_):
    d = _load(dir_)
    if not d:
        return None
    caps, spks, res = d["captions"], d["speakers"], d["result"]
    name = os.path.basename(os.path.abspath(dir_))

    # duration + final timeline (prefer the finalized result)
    timeline = res["timeline"] if res else []
    duration = (res["duration_sec"] if res else
                max([c.get("end_t", c.get("media_t", 0)) for c in caps] + [0]))

    # per-speaker talk time from the finalized timeline
    talk = {}
    for s, e, sid in timeline:
        talk[sid] = talk.get(sid, 0.0) + (e - s)

    # very short turns (mislabel-prone)
    short = [(s, e, sid) for s, e, sid in timeline if (e - s) < SHORT_SEG]

    # flagged captions (hallucinations)
    flagged = [c for c in caps if c.get("flags")]

    # rapid flip-flops: slide over speaker-change events
    flips = []
    ch = [(s["media_t"], s["speaker"]) for s in spks]
    for i in range(len(ch)):
        j = i
        while j < len(ch) and ch[j][0] - ch[i][0] <= FLIP_WIN:
            j += 1
        if j - i >= FLIP_N:
            flips.append((ch[i][0], ch[j - 1][0], j - i))
    # de-overlap flip hotspots
    merged_flips = []
    for a, b, n in flips:
        if merged_flips and a <= merged_flips[-1][1]:
            merged_flips[-1] = (merged_flips[-1][0], max(merged_flips[-1][1], b),
                                max(merged_flips[-1][2], n))
        else:
            merged_flips.append((a, b, n))

    # long caption gaps
    gaps = []
    last = 0.0
    for c in sorted(caps, key=lambda x: x.get("media_t", 0)):
        st = c.get("media_t", 0)
        if st - last > GAP_S:
            gaps.append((last, st))
        last = max(last, c.get("end_t", st))
    if duration - last > GAP_S:
        gaps.append((last, duration))

    # latency (real-time runs only)
    lats = [c["latency"] for c in caps if c.get("latency") is not None]

    return {"name": name, "dir": dir_, "duration": duration, "n_caps": len(caps),
            "n_flagged": len(flagged), "talk": talk, "n_speakers": len(talk),
            "short": short, "flagged": flagged, "flips": merged_flips, "gaps": gaps,
            "lats": lats, "timeline": timeline}


def write_report(a):
    L = []
    L.append(f"# Review report — {a['name']}\n")
    L.append(f"- Duration: **{_fmt(a['duration'])}**  ·  speakers: **{a['n_speakers']}**  ·  "
             f"captions: **{a['n_caps']}**  ·  flagged: **{a['n_flagged']}**\n")
    if a["lats"]:
        s = sorted(a["lats"]); md = s[len(s) // 2]
        L.append(f"- Detection latency (real-time run): median **{md:.1f}s**, "
                 f"max **{max(s):.1f}s** over {len(s)} captions\n")

    L.append("\n## Speaker talk-time (finalized)\n")
    tot = sum(a["talk"].values()) or 1
    for sid, sec in sorted(a["talk"].items(), key=lambda x: -x[1]):
        L.append(f"- SPEAKER {sid}: {_fmt(sec)}  ({sec / tot * 100:.0f}%)")
    tiny = [sid for sid, sec in a["talk"].items() if sec < 5.0]
    if tiny:
        L.append(f"\n> ⚠ phantom-speaker suspects (<5s total): {tiny} — check if these are real.")

    L.append(f"\n\n## ⚠ Likely ASR hallucinations ({len(a['flagged'])})\n")
    if a["flagged"]:
        for c in a["flagged"][:60]:
            L.append(f"- `{_fmt(c['media_t'])}` SPK{c['speaker']} [{', '.join(c['flags'])}] — "
                     f"{c['text'][:90]}")
        if len(a["flagged"]) > 60:
            L.append(f"- … and {len(a['flagged']) - 60} more (see review_log.jsonl)")
    else:
        L.append("- none")

    L.append(f"\n\n## Very short speaker turns (< {SHORT_SEG:.0f}s) — mislabel-prone ({len(a['short'])})\n")
    if a["short"]:
        for s, e, sid in a["short"][:60]:
            L.append(f"- `{_fmt(s)}`–`{_fmt(e)}` SPK{sid} ({e - s:.1f}s)")
        if len(a["short"]) > 60:
            L.append(f"- … and {len(a['short']) - 60} more")
    else:
        L.append("- none")

    L.append(f"\n\n## Rapid speaker flip-flops ({len(a['flips'])})\n")
    if a["flips"]:
        for st, en, n in a["flips"]:
            L.append(f"- `{_fmt(st)}`–`{_fmt(en)}`: {n} changes in {en - st:.0f}s")
    else:
        L.append("- none")

    L.append(f"\n\n## Long caption gaps (> {GAP_S:.0f}s) — silence/music/ASR dropout ({len(a['gaps'])})\n")
    if a["gaps"]:
        for st, en in a["gaps"]:
            L.append(f"- `{_fmt(st)}`–`{_fmt(en)}` ({en - st:.0f}s)")
    else:
        L.append("- none")

    out = os.path.join(a["dir"], "review_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return out


def main():
    dirs = sys.argv[1:]
    if not dirs:
        sys.exit("usage: python review_report.py <run_dir> [run_dir ...]")
    analyses = []
    for d in dirs:
        a = analyze(d)
        if not a:
            print(f"  skip {d}: no review_log.jsonl", flush=True)
            continue
        out = write_report(a)
        analyses.append(a)
        print(f"-> {out}  ({a['n_speakers']} spk, {a['n_flagged']} flagged, "
              f"{len(a['short'])} short turns, {len(a['flips'])} flip spots)", flush=True)

    if len(analyses) > 1:
        L = ["# Review index — all runs\n",
             "| run | dur | speakers | captions | flagged | short turns | flip spots |",
             "|---|---|---|---|---|---|---|"]
        for a in sorted(analyses, key=lambda x: -(x["n_flagged"] + len(x["short"]))):
            L.append(f"| {a['name']} | {_fmt(a['duration'])} | {a['n_speakers']} | "
                     f"{a['n_caps']} | {a['n_flagged']} | {len(a['short'])} | {len(a['flips'])} |")
        L.append("\nSorted worst-first (flagged + short turns). Open each folder's "
                 "review_report.md for the timestamps.")
        idx = os.path.join(os.getcwd(), "review_index.md")
        with open(idx, "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
        print(f"-> {idx}", flush=True)


if __name__ == "__main__":
    main()
