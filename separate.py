#!/usr/bin/env python3
"""
separate.py — optional vocal separation (Demucs) to strip background music before
diarization. Music under speech contaminates speaker embeddings; isolating vocals
makes diarization far more robust on music-heavy content (dramas, broadcasts, songs).

Loads the Demucs model once and separates an in-memory numpy buffer. Demucs runs at
44.1 kHz, so we resample 16k -> 44.1k -> separate -> vocals -> back to 16k.
"""
import numpy as np
import torch

_MODEL = None
_DEMUCS_SR = 44100


def _get_model(device):
    global _MODEL
    if _MODEL is None:
        from demucs.pretrained import get_model
        _MODEL = get_model("htdemucs")
        _MODEL.to(device)
        _MODEL.eval()
    return _MODEL


def available():
    try:
        import demucs  # noqa: F401
        return True
    except Exception:
        return False


def separate_vocals(audio_16k_mono, device="cpu", sr=16000):
    """audio_16k_mono: 1-D float32 numpy @ 16 kHz. Returns the vocals-only track,
    1-D float32 @ 16 kHz (same length)."""
    import torchaudio
    model = _get_model(device)
    x = torch.from_numpy(np.asarray(audio_16k_mono, dtype=np.float32))
    up = torchaudio.functional.resample(x, sr, _DEMUCS_SR)
    stereo = up.unsqueeze(0).repeat(2, 1).unsqueeze(0).to(device)   # (1, 2, T)
    with torch.no_grad():
        from demucs.apply import apply_model
        est = apply_model(model, stereo, device=device, progress=False)[0]  # (stems, 2, T)
    vocals = est[model.sources.index("vocals")].mean(0).cpu()             # mono
    down = torchaudio.functional.resample(vocals, _DEMUCS_SR, sr)
    out = down.numpy().astype(np.float32)
    n = len(audio_16k_mono)
    if len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    return out[:n]
