# Live Speaker Diarization (real-time)

Answer **“who is speaking, right now?”** on a *live, never-ending* audio stream — assign
each moment a **stable speaker label** incrementally, with low latency. Built on the
open-source [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) neural diarizer
(`speaker-diarization-community-1`).

- **Input:** a live stream URL (YouTube live, HLS/RTMP/SRT, internet radio) — or any local
  file replayed at real-time speed for testing (`--simulate`).
- **Output:** live `* SPEAKER N` events in the terminal as the speaker changes, with stream
  + wall-clock timestamps and a `[NEW VOICE]` tag when a new speaker first appears.
- **Runs on:** Windows / Linux / macOS — **NVIDIA CUDA**, Apple **MPS**, or CPU (auto-selected).
- 100% open-source, no paid APIs, no cloud.

> A detailed, self-contained explanation of the *whole* project (offline foundation +
> this real-time system) is in **[`docs/project_overview.pdf`](docs/project_overview.pdf)** —
> readable from scratch, no prior knowledge needed.

> Looking for whole-file (offline) diarization of recordings instead? See the companion repo
> **[offline-speaker-diarization](https://github.com/saad-faran/offline-speaker-diarization)**.

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

### A) Simulated live (best first test — no network)
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

### C) A live YouTube URL (advanced — needs anti-bot flags)
YouTube requires solving a JS challenge to pull a live stream. You need a browser you're
logged into, plus a JS runtime + yt-dlp's challenge solver:
```bash
python live_diarize.py "https://www.youtube.com/watch?v=<LIVE_ID>" \
  --cookies-from-browser chrome --js-runtime deno --remote-components ejs:npm
```
- Install a JS runtime first (`deno` recommended, or `node`).
- `--remote-components ejs:npm` lets yt-dlp fetch its challenge solver (runs external code — opt-in).
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
| `--stride` | `2` | commit cadence (s). Smaller = finer/lower latency, more compute. |
| `--commit-lag` | `2` | hold back the newest N s before committing a boundary (more future context = sharper boundaries, slightly more latency). |
| `--speakers` | auto | force a known speaker count (most reliable when you know it). |
| `--threshold` | `0.85` | clustering merge threshold. |

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
| YouTube live: "Sign in to confirm you're not a bot" / challenge error | see §4C (cookies + JS runtime + `--remote-components`), or use a direct stream / `--simulate`. |
| Falls behind / high latency | you're likely on CPU — use an NVIDIA GPU; or raise `--stride`. |

---

## 7. How it works (short version)

A background thread fills a **rolling buffer** (last ~15 s) with live audio. Every ~2 s the
diarizer runs pyannote on the **latest** window, extracts 256-dim voice embeddings, and matches
them to a **persistent speaker registry** (cosine similarity) so labels stay stable across
runs — then emits whoever is talking now. If compute falls behind it just diarizes less often,
so latency never accumulates. Full details, diagrams, and all parameters:
**[`docs/project_overview.pdf`](docs/project_overview.pdf)**.

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
