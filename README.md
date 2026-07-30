# Live Speaker Diarization (real-time)

Answer **“who is speaking, right now?”** on a *live, never-ending* audio stream — assign
each moment a **stable speaker label** incrementally, with low latency. Built on the
open-source [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) neural diarizer
(`speaker-diarization-community-1`).

- **Input:** a live stream URL (YouTube live *or* recorded, HLS/RTMP/SRT, internet radio) — or any
  local file replayed at real-time speed for testing.
- **Output:** live speaker labels — in the terminal, **or** in a browser dashboard with the video,
  real-time **speaker-labeled subtitles** (ASR), a per-speaker roster, and downloadable logs.
- **Runs on:** Windows / Linux / macOS — **NVIDIA CUDA**, Apple **MPS**, or CPU (auto-selected).
- 100% open-source, no paid APIs, no cloud.

> A detailed, self-contained explanation of the *whole* project (offline foundation +
> this real-time system) is in **[`docs/project_overview.pdf`](docs/project_overview.pdf)** —
> readable from scratch, no prior knowledge needed.

> Looking for whole-file (offline) diarization of recordings instead? See the companion repo
> **[offline-speaker-diarization](https://github.com/saad-faran/offline-speaker-diarization)**.

---

## What's in this repo

| Tool | What it does | Best for |
|---|---|---|
| **`live_app.py`** + `app.html` | Paste-a-URL browser **console**: type any YouTube (live/recorded) or direct URL → real-time diarization + subtitles overlaid on the video, a speaker roster (unique speakers, turns, talk-time), a **⚑ Mark issue** button, and downloadable session logs. Swap the URL any time. | interactive live testing & sharing results |
| **`live_review.py`** + `review.html` | **Local review harness**: run the pipeline over a downloaded video → a dashboard that plays the video with **frame-synced** captions, plus `captions.srt`, `review_log.jsonl`, and an HTML timeline. | precise, offline review of recordings |
| **`review_report.py`** | Turns a run's `review_log.jsonl` into a short **"where to look"** report (hallucinations, short/mislabel-prone turns, flip-flops, gaps, talk-time). | fast triage without watching everything |
| **`live_diarize.py`** | The core engine + CLI: terminal speaker events, `--overlay` (burn labels on video), `--simulate`. | scripting / batch / burned-in labels |
| **`live_captions.py`** | Minimal WebSocket captions demo (superseded by `live_app.py`). | reference |

All share `core.py` (device pick, pyannote pipeline, speaker registry). ASR uses
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper).

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | 3.10–3.12 recommended |
| **ffmpeg** | on `PATH` — decodes the live stream to audio |
| **NVIDIA GPU + CUDA** | strongly recommended for real-time (auto-detected) |
| **Hugging Face token** | free; one-time model download |
| **yt-dlp** | only for web-stream URLs (installed via requirements); not needed for `--simulate` or direct HLS/RTMP |

**Install ffmpeg**
- Windows: `winget install Gyan.FFmpeg` (then reopen the terminal) · Linux: `sudo apt install ffmpeg` · macOS: `brew install ffmpeg`

---

## 2. Setup

```bash
git clone https://github.com/saad-faran/live-speaker-diarization.git
cd live-speaker-diarization

python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate

# --- install PyTorch + torchaudio FIRST, together, from ONE CUDA index ---
# torch and torchaudio MUST be the same version. A widely-available matched pair
# that runs on any recent NVIDIA GPU (a CUDA 12.8 build works on newer drivers):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# --- then the rest ---
pip install -r requirements.txt
```

> **Version-match matters.** `torch` and `torchaudio` are compiled together; mixing e.g.
> `torch 2.13` with `torchaudio 2.11` crashes with an `undefined symbol` error, and the newest
> CUDA indexes (e.g. `cu132`) may ship torch with **no** matching torchaudio. If your first
> choice fails, use an index that has **both at the same version** (`cu128` currently gives
> `torch 2.11.0 + torchaudio 2.11.0`). Verify the GPU:
> `python -c "import torch; print(torch.cuda.is_available())"` → should print `True`.

---

## 3. Hugging Face token (one-time)

1. Create a free **read** token: <https://huggingface.co/settings/tokens>
2. Accept the model terms (logged in): <https://huggingface.co/pyannote/speaker-diarization-community-1>
3. Provide it — copy `.env.example` to `.env` and paste the token, **or** set an env var:
   - Windows (PowerShell): `$env:HF_TOKEN="hf_xxx"` (this session) / `setx HF_TOKEN "hf_xxx"` (permanent)
   - Linux/macOS: `export HF_TOKEN="hf_xxx"`

Weights (~1–2 GB) download once and cache locally.

---

## 4. Usage

### The browser console (recommended) — paste a URL, watch it diarize live
```bash
python live_app.py --asr-model small --language en
# for live YouTube, add your anti-bot flags once (see §4C):
#   --cookies cookies.txt --js-runtime deno --remote-components ejs:npm
```
Open the printed URL (`http://localhost:8771/app.html`), **paste any YouTube URL (live or
recorded) or a direct HLS/RTMP URL, press Start.** You get the video with subtitles + speaker
IDs overlaid, a live speaker roster, and a session log. Notes:
- **"Sync video to captions"** holds the video back so subtitles/speakers match what you hear
  (the pipeline runs a couple seconds behind live); use the offset slider to fine-tune. Toggle
  it off to show captions the instant they're produced.
- **⚑ Mark issue** timestamps the exact moment in the log — then **⤓ log** downloads
  `session.jsonl` (+ **⤓ srt**) so a wrong label is easy to report and fix.
- Run it from a **writable folder** (session files are written to the current directory).

### Local review of a downloaded file (frame-accurate captions + SRT)
```bash
python live_review.py "video.mp4" --asr-model small --language en    # 1x real-time dashboard
python live_review.py "video.mp4" --fast                             # process fast, then review artifacts
python review_report.py .                                            # "where to look" report
```
Writes `captions.srt`, `review_log.jsonl`, `live_timeline.html`, `live_result.json`. Because
the harness owns the video element, captions here are **exactly** frame-synced (unlike the
live console, which can't drive YouTube's player).

### A) Simulated live (terminal, no network)
Replays a local file at 1× real-time through the exact live pipeline:
```bash
python live_diarize.py sample.wav --simulate
```

### B) A direct live stream (HLS / RTMP / SRT / internet radio) — no yt-dlp needed
```bash
python live_diarize.py "https://npr-ice.streamguys1.com/live.mp3" --max-seconds 60
python live_diarize.py "https://example.com/live/stream.m3u8"
python live_diarize.py "rtmp://your-encoder/live/streamkey"
```

### C) A YouTube URL (live or recorded — needs anti-bot flags)
YouTube requires solving a JS challenge. You need cookies (to prove you're signed in) plus a
JS runtime + yt-dlp's challenge solver. The same flags work on `live_app.py`, `live_diarize.py`,
and `live_review.py`:
```bash
python live_diarize.py "https://www.youtube.com/watch?v=<ID>" \
  --cookies cookies.txt --js-runtime deno --remote-components ejs:npm
```
- Install a JS runtime first (`deno` recommended, or `node`).
- `--remote-components ejs:npm` lets yt-dlp fetch its challenge solver (runs external code — opt-in).
- **Cookies:** prefer **`--cookies cookies.txt`** — an exported cookies file. On Windows,
  `--cookies-from-browser chrome` often fails ("Could not copy Chrome cookie database" — Chrome's
  cookie DB is locked/app-bound-encrypted). Export `cookies.txt` with the **"Get cookies.txt
  LOCALLY"** browser extension (keep it private; re-export when it expires).
- If it still fails, that's a YouTube-scraping limitation, not the diarizer — use A or B.

**What you'll see** (any mode):
```
[wall   10.0s | stream    9.5s]  * SPEAKER 1  [NEW VOICE]
[wall   40.1s | stream   39.5s]  * SPEAKER 2
==================================================
  speakers seen: 2 [1, 2]   changes: 2
==================================================
```
Add `--max-seconds N` for a bounded test; press `Ctrl+C` to stop a live run.

---

## 5. Tuning

| Flag | Default | Effect |
|---|---|---|
| `--window` | `15` | rolling buffer length (s). Larger = more context, heavier. |
| `--stride` | `1.5` (`live_diarize`) · `1.0` (`live_app`) | commit cadence (s). Smaller = finer/lower latency, more compute. |
| `--commit-lag` | `1.0` (`live_diarize`) · `0.75` (`live_app`) | hold back the newest N s before committing a boundary (more future context = sharper boundaries, slightly more latency). |
| `--speakers` | auto | force a known speaker count (most reliable when you know it). |
| `--threshold` | `0.85` (`live_diarize`) · `0.5` (`live_app`/review) | clustering merge threshold. Lower over-segments (splits short/quiet speakers). |
| `--asr-model` | `small` | faster-whisper size: `tiny\|base\|small\|medium` (console/review). |
| `--asr-interval` | `2.5` (`live_app`) · `6` (`review`) | transcribe every N s. **Lower = lower caption latency**, needs a faster GPU. |

> **Latency note:** in the live console, caption lag is dominated by `--asr-interval`; the
> speaker-ID itself has a small (~1 s) floor because online diarization must *hear* ~1 s of a new
> voice before it can distinguish it. Very short (<~2 s) interjections can still briefly inherit the
> previous speaker's ID — an inherent online-diarization limit (see below).

The engine commits the **fine-grained speaker turns** from each window's settled region
(mapped to stable IDs) and post-processes them exactly like the offline tool, so
speaker-**change timestamps track the offline boundaries closely** (validated: matched
changes land within ~0.1 s of the offline reference). Detection is emitted a few seconds
after the fact (buffer fill + `commit_lag`), but the *recorded* timestamp is the true
boundary time.

**Honest limit:** a speaker who appears only *briefly* with no other speech nearby in the
rolling window (a 2–4 s interjection) can be mislabeled live — offline catches it only
because that voice has minutes of evidence across the whole file. So expect the *matched*
change-times to be near-exact, with a few short no-context interjections missed. Increasing
`--window` gives more context and can help.
Lower `--stride`/`--window` reduce it at the cost of compute.

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `undefined symbol` / `torchaudio 2.x+cpu` with `torch 2.y+cuXXX` | torch/torchaudio **version or build mismatch** — reinstall both from the same CUDA index at the same version (step 2). |
| `torchcodec is not available` (warning) | harmless — this project decodes audio itself and never uses torchcodec. |
| `torch.cuda.is_available()` is `False` | CPU wheel installed — reinstall the CUDA build (step 2). |
| `'ffmpeg' not found` | install ffmpeg and reopen the terminal. |
| YouTube: "Sign in to confirm you're not a bot" / challenge error | see §4C (cookies + JS runtime + `--remote-components`), or use a direct stream / a local file. |
| `Could not copy Chrome cookie database` | Chrome's cookie DB is locked/encrypted — use **`--cookies cookies.txt`** (exported file) instead of `--cookies-from-browser` (§4C). |
| `An Application Control policy has blocked this file` importing `av` | Windows blocked PyAV's DLL. Handled automatically — we decode via ffmpeg and stub `av` (you'll see a harmless "PyAV unavailable; stubbing it" note). |
| faster-whisper: `libcudnn` / cuDNN load error | `pip install nvidia-cudnn-cu12`, or run ASR on CPU (it still works, just slower). |
| Console: captions bunch up / fall behind on live | raise `--asr-interval` (e.g. 3–4) or use a smaller `--asr-model`. |
| Console: subtitles slightly off the video | drag the **offset slider**; or the source has no embeddable video (sync needs the YouTube player). |
| Falls behind / high latency | you're likely on CPU — use an NVIDIA GPU; or raise `--stride`/`--asr-interval`. |

---

## 7. How it works (short version)

A background thread fills a **rolling buffer** (last ~15 s) with live audio. Every ~1–1.5 s the
diarizer runs pyannote on the **latest** window, extracts 256-dim voice embeddings, and matches
them to a **persistent speaker registry** (cosine similarity) so labels stay stable across
runs — then emits whoever is talking now. If compute falls behind it just diarizes less often,
so latency never accumulates. The console/review tools add a parallel `faster-whisper` pass
(overlapping, boundary-safe windows) for the subtitles, and stream everything to the browser
over WebSocket. This is **honest online** diarization — the live labels use the registry only,
never a global re-cluster over the whole (future) recording. Full details, diagrams, and all
parameters: **[`docs/project_overview.pdf`](docs/project_overview.pdf)**.

## Getting every speaker (incl. short / quiet ones)

For hard content (many speakers, short 1–3 s turns, background music) the defaults can miss or
mislabel brief speakers. The recipe that fixes this:

```bash
python live_diarize.py <video-or-url> --overlay \
    --speakers 4 --threshold 0.5 --separate-vocals
```
- **`--speakers N`** (real cast size) → globally re-clusters every turn to N consistent IDs and
  skips small-cluster consolidation, so a speaker who only talks briefly still gets their own ID.
- **`--threshold 0.5`** over-segments each window so short/quiet speakers become distinct.
- **`--separate-vocals`** removes music that masks quiet speakers.
- Lower **`--stride`/`--commit-lag`** (defaults 1.5 / 1.0) for faster, more real-time colour changes.

## Video overlay & background-music removal

**See the labels on the video** (`--overlay`) — instead of only reading logs, get a video
with the speaker label burned in, synced to who's talking, for easy audiovisual checking:
```bash
python live_diarize.py "https://www.youtube.com/watch?v=<ID>" --overlay          # downloads the video
python live_diarize.py local_video.mp4 --overlay --speakers 3
```
Produces `live_labeled.mp4` (labels burned on) **plus** the usual `live_timeline.html` + JSON.
Uses PIL + ffmpeg's `overlay` filter, so it needs no special ffmpeg build.

**Strip background music** before diarizing (`--separate-vocals`) — isolates vocals with
[Demucs](https://github.com/facebookresearch/demucs) so music doesn't contaminate speaker
embeddings (dramas, broadcasts, songs):
```bash
pip install demucs
python live_diarize.py video.mp4 --overlay --separate-vocals --speakers 3
```
Demucs is heavy — practical on an NVIDIA GPU; slow on CPU/MPS. Best combined with `--speakers N`.

## License

MIT — see [LICENSE](LICENSE). Model weights are governed by their own licenses on Hugging Face.
