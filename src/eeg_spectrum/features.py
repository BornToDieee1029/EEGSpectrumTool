"""Feature extraction (pipeline stage 5).

Static features describe each map; DYNAMIC features describe how the brain moves
between maps -- the part NEBA never measured and where the monk<->ADHD
hypothesis lives. See docs/ARCHITECTURE.md section 6.

Core hypothesis, in feature terms:
  monk  -> long duration, low switch_rate, low transition_entropy
  ADHD  -> short duration, high switch_rate, high transition_entropy
"""

from __future__ import annotations

import mne
import numpy as np

from .microstates import Segmentation

# Standard EEG bands. The theta/beta ratio is the classic ADHD spectral marker
# (and what NEBA relied on) -- here it is ONE feature among many, not the whole
# claim. Frontal channels stand in for the fronto-central sites where the ADHD
# theta/beta effect is strongest (our montage has no midline Fz/Cz).
_BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12),
          "beta": (12, 30), "gamma": (30, 40)}
_FRONTAL = ("Fp1", "Fp2", "F3", "F4", "F7", "F8")


def spectral_features(raw: mne.io.BaseRaw) -> dict[str, float]:
    """Relative band powers + theta/beta ratios from a cleaned recording."""
    psd = raw.compute_psd(fmin=1, fmax=40, verbose="ERROR")
    psds, freqs = psd.get_data(return_freqs=True)
    feats: dict[str, float] = {}

    mean_psd = psds.mean(axis=0)
    total = mean_psd.sum() + 1e-30
    for band, (lo, hi) in _BANDS.items():
        idx = (freqs >= lo) & (freqs < hi)
        feats[f"rel_{band}"] = float(mean_psd[idx].sum() / total)
    feats["theta_beta_ratio"] = feats["rel_theta"] / (feats["rel_beta"] + 1e-9)

    frontal = [c for c in _FRONTAL if c in raw.info["ch_names"]]
    if frontal:
        fp = psds[[raw.info["ch_names"].index(c) for c in frontal]].mean(axis=0)
        ft = fp[(freqs >= 4) & (freqs < 8)].sum()
        fb = fp[(freqs >= 12) & (freqs < 30)].sum()
        feats["frontal_theta_beta"] = float(ft / (fb + 1e-30))
    return feats


def _runs(labels: np.ndarray):
    """Yield (state, run_length) for each maximal run of equal labels."""
    if len(labels) == 0:
        return
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)]))
    for s, e in zip(starts, ends):
        yield int(labels[s]), int(e - s)


def transition_matrix(labels: np.ndarray, n_states: int) -> np.ndarray:
    """Row-stochastic m->n transition probabilities, ignoring self-loops and
    unassigned (-1) samples. Counts transitions between consecutive *distinct*
    state segments."""
    seq = [s for s, _ in _runs(labels) if s >= 0]
    counts = np.zeros((n_states, n_states))
    for a, b in zip(seq[:-1], seq[1:]):
        if a != b:
            counts[a, b] += 1
    row = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.where(row > 0, counts / row, 0.0)
    return probs


def extract(seg: Segmentation) -> dict[str, float]:
    """Return a flat feature dict for one segmented recording.

    Static (per map m):
      duration_m    mean ms dwelling in map m   (stability signal)
      coverage_m    fraction of assigned time in map m
      occurrence_m  appearances per second
    Dynamic (the part NEBA never measured):
      switch_rate          transitions per second
      transition_entropy   Shannon entropy (bits) of the off-diagonal transition
                           distribution (low = ordered; high = near-random)
      trans_m_n            individual m->n transition probabilities
    """
    labels = seg.labels
    k = seg.n_states
    sf = seg.sfreq
    feats: dict[str, float] = {}

    runs = [(s, n) for s, n in _runs(labels) if s >= 0]
    total_assigned = sum(n for _, n in runs)
    total_s = len(labels) / sf

    # Static per-map features.
    for m in range(k):
        m_runs = [n for s, n in runs if s == m]
        n_samples = sum(m_runs)
        feats[f"duration_{m}"] = (np.mean(m_runs) / sf * 1000) if m_runs else 0.0
        feats[f"coverage_{m}"] = (n_samples / total_assigned) if total_assigned else 0.0
        feats[f"occurrence_{m}"] = (len(m_runs) / total_s) if total_s else 0.0

    # Dynamic features.
    seq = [s for s, _ in runs]
    src_counts = np.zeros(k)
    n_trans = 0
    for a, b in zip(seq[:-1], seq[1:]):
        if a != b:
            src_counts[a] += 1
            n_trans += 1
    feats["switch_rate"] = n_trans / total_s if total_s else 0.0

    tm = transition_matrix(labels, k)
    # Conditional entropy H(next | current): predictability of switching.
    # Deterministic sequencing -> ~0 (ordered, monk); near-random -> ~log2(k-1).
    pi = src_counts / src_counts.sum() if src_counts.sum() > 0 else src_counts
    H = 0.0
    for m in range(k):
        rp = tm[m][tm[m] > 0]
        if len(rp):
            H += pi[m] * float(-(rp * np.log2(rp)).sum())
    feats["transition_entropy"] = H
    for i in range(k):
        for j in range(k):
            if i != j:
                feats[f"trans_{i}_{j}"] = float(tm[i, j])

    return feats
