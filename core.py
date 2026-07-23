#!/usr/bin/env python3
"""
core.py — shared engine for real-time speaker diarization.

Provides:
  * pick_device()   — choose CUDA > Apple MPS > CPU
  * load_pipeline() — load pyannote `speaker-diarization-community-1` on the best device
  * SpeakerRegistry — persistent stable speaker IDs across buffers (solves label permutation)

The streaming path never reads audio files through pyannote (it passes in-memory
{waveform, sample_rate} dicts), so the fragile `torchcodec` dependency is never used.
"""
import os
import numpy as np
import torch
from pyannote.audio import Pipeline

MODEL = "pyannote/speaker-diarization-community-1"
REGISTRY_MATCH_SIM = 0.30   # cosine sim >= this => same speaker (same≈0.83 vs different≈0.10)


def _cos(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(a @ b)


def pick_device():
    """CUDA (NVIDIA) > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(device=None, token=None, clustering_threshold=None):
    """Load the pyannote pipeline once.

    device:  'cuda' | 'mps' | 'cpu' (auto-selected when None).
    token:   Hugging Face token; falls back to the HF_TOKEN env var.
    clustering_threshold:
        VBx agglomerative threshold. None keeps pyannote's default (0.6). Raising it
        (e.g. 0.85) merges more aggressively — useful when one expressive speaker is
        over-split. Ignored if you pass a known speaker count downstream.
    """
    device = device or pick_device()
    token = token or os.environ.get("HF_TOKEN")
    pipe = Pipeline.from_pretrained(MODEL, token=token)
    pipe.to(torch.device(device))
    if clustering_threshold is not None:
        pipe.instantiate({"clustering": {"Fa": 0.07, "Fb": 0.8,
                                         "threshold": clustering_threshold},
                          "segmentation": {"min_duration_off": 0.0}})
    return pipe


class SpeakerRegistry:
    """Persistent stable speaker IDs, matched across buffers by embedding cosine similarity.

    Every time we re-diarize the rolling buffer, pyannote invents fresh local labels
    (SPEAKER_00, SPEAKER_01, ...) that mean nothing across runs — "label permutation".
    The registry fixes identity to the *voice fingerprint*: each new local cluster is
    matched to the most-similar stored centroid (>= match_sim) and keeps that stable ID;
    otherwise it becomes a new speaker. Centroids update with an EMA so a voice's natural
    variation is absorbed instead of spawning phantom speakers.
    """
    def __init__(self, match_sim=REGISTRY_MATCH_SIM, ema=0.2, ema_gate=0.45):
        # ema_gate: only pull a speaker's centroid toward a new match when the match
        # is CONFIDENT (cosine >= ema_gate). Borderline matches (match_sim..ema_gate)
        # still get the stable ID, but must NOT drag the fingerprint — otherwise a
        # well-established voice slowly drifts, then fails to match itself later
        # (spawns a phantom new ID) or gets stolen by a different speaker. Clean,
        # high-similarity matches (~0.83) are unaffected.
        self.match_sim, self.ema, self.ema_gate = match_sim, ema, ema_gate
        self.centroids, self.counts, self._next = {}, {}, 1

    def match(self, local_embs):
        """local_embs: list[(local_label, embedding)] -> {local_label: stable_id}."""
        cand = [e for e in local_embs if e[1] is not None and not np.isnan(e[1]).any()]
        pairs = sorted(
            ((_cos(e, c), ll, sid) for ll, e in cand for sid, c in self.centroids.items()),
            reverse=True, key=lambda x: x[0])
        assigned, used_sid, used_ll = {}, set(), set()
        for sim, ll, sid in pairs:
            if ll in used_ll or sid in used_sid or sim < self.match_sim:
                continue
            emb = dict(cand)[ll]
            if sim >= self.ema_gate:                # only confident matches move the fingerprint
                self.centroids[sid] = (1 - self.ema) * self.centroids[sid] + self.ema * emb
            self.counts[sid] += 1
            assigned[ll] = sid
            used_sid.add(sid)
            used_ll.add(ll)
        for ll, emb in cand:                       # unmatched local clusters -> new speakers
            if ll not in assigned:
                sid = self._next
                self._next += 1
                self.centroids[sid] = emb.copy()
                self.counts[sid] = 1
                assigned[ll] = sid
        return assigned
