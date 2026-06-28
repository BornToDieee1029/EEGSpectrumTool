#!/usr/bin/env python3
"""Generate SYNTHETIC demo EEG for the live demo.

This is SIMULATED data — NOT a real recording, not a real person. It exists only
so the app's maps/PSD render cleanly in a demo. Never present it as real data,
and never count it in any result, statistic, or cohort.

Two profiles, both 16-channel / 128 Hz / ~4 min:
  - "stable": four topographies cycling in a mostly-ordered sequence + strong
    posterior alpha -> lands toward the calm end of the spectrum.
  - "adhd": the SAME topographies switching rapidly and near-randomly + weaker
    alpha -> lands toward the ADHD end. (A simulated *pattern*, not a real child.)

    python scripts/make_synthetic_demo.py --profile stable
    python scripts/make_synthetic_demo.py --profile adhd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mne
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CH = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "C3", "C4",
      "T7", "T8", "P3", "P4", "P7", "P8", "O1", "O2"]
PROFILES = {
    # cycle_p: chance the next state follows an orderly cycle (high = ordered).
    # dwell: (min,max) samples held per state (short = faster switching).
    "stable": dict(seed=7, cycle_p=0.8, dwell=(11, 20), alpha=0.6,
                   out="synthetic_demo_eeg.fif"),
    "adhd": dict(seed=11, cycle_p=0.0, dwell=(6, 11), alpha=0.35,
                 out="synthetic_demo_adhd_eeg.fif"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="stable")
    args = ap.parse_args()
    pf = PROFILES[args.profile]
    OUT = ROOT / "samples" / pf["out"]

    rng = np.random.default_rng(pf["seed"])
    sfreq, secs = 128.0, 240
    n = int(sfreq * secs)

    info = mne.create_info(CH, sfreq, "eeg")
    info.set_montage("standard_1020", verbose="ERROR")
    pos = info.get_montage().get_positions()["ch_pos"]
    P = np.array([pos[c] for c in CH])
    xn = P[:, 0] / np.abs(P[:, 0]).max()      # -1 left .. +1 right
    yn = P[:, 1] / np.abs(P[:, 1]).max()      # -1 back .. +1 front

    def g(cx, cy, s=0.6):
        return np.exp(-((xn - cx) ** 2 + (yn - cy) ** 2) / (2 * s ** 2))

    # Four distinct topographies (normalised to unit norm).
    T = np.stack([
        xn,                       # left–right gradient
        yn,                       # front–back gradient
        g(0, 0.9) - g(0, -0.9),   # frontal vs occipital focal
        xn * yn,                  # diagonal
    ])
    T = T / np.linalg.norm(T, axis=1, keepdims=True)

    # Microstate sequence: ordered cycle vs near-random switching by profile.
    lo, hi = pf["dwell"]
    labels = np.zeros(n, dtype=int)
    t, state = 0, 0
    while t < n:
        d = int(rng.integers(lo, hi))
        labels[t:t + d] = state
        t += d
        if rng.random() < pf["cycle_p"]:
            state = (state + 1) % 4              # orderly
        else:
            state = int(rng.choice([s for s in range(4) if s != state]))  # random

    env = 1.0 + 0.5 * np.sin(2 * np.pi * 0.1 * np.arange(n) / sfreq)
    sig = T[labels].T * env                   # (16, n) topography time series

    # Posterior alpha (~10 Hz, weighted to the back of the head).
    post = np.clip(-yn, 0, None)
    post = post / post.max()
    alpha = np.sin(2 * np.pi * 10 * np.arange(n) / sfreq + rng.random() * 6)
    sig = sig + pf["alpha"] * np.outer(post, alpha)

    sig = sig + 0.4 * rng.standard_normal((len(CH), n))   # background noise
    sig = sig / sig.std() * 15e-6                          # ~15 µV scale

    raw = mne.io.RawArray(sig, info, verbose="ERROR")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw.save(OUT, overwrite=True, verbose="ERROR")
    print(f"saved SYNTHETIC '{args.profile}' demo: {OUT}  "
          f"({len(CH)} ch, {sfreq:.0f} Hz, {secs}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
