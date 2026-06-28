#!/usr/bin/env python3
"""Generate a SYNTHETIC Muse-headband demo recording (Mind Monitor CSV format).

SIMULATED data — NOT a real recording, not a real person. For demoing the Muse
band-power screen only. Never present as real data.

Builds a 4-channel (TP9, AF7, AF8, TP10), 256 Hz, ~3-minute eyes-closed-like
signal with a posterior-weighted ~10 Hz alpha rhythm + 1/f-ish noise, written as
RAW_* columns like Mind Monitor's export.

    python scripts/make_synthetic_muse.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CH = ["TP9", "AF7", "AF8", "TP10"]            # temporoparietal + frontal
PROFILES = {
    # alpha: 10 Hz amplitude; theta: 6 Hz amplitude; noise: broadband (beta-ish).
    "calm": dict(seed=3, alpha=7, theta=2, noise=9, out="synthetic_muse_demo.csv"),
    # ADHD-pattern: strong frontal theta, weaker alpha, less beta -> high theta/beta.
    "adhd": dict(seed=5, alpha=3, theta=10, noise=5,
                 out="synthetic_muse_adhd_demo.csv"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="calm")
    args = ap.parse_args()
    pf = PROFILES[args.profile]
    OUT = ROOT / "samples" / pf["out"]

    rng = np.random.default_rng(pf["seed"])
    sfreq, secs = 256.0, 180
    n = int(sfreq * secs)
    t = np.arange(n) / sfreq

    alpha_w = {"TP9": 1.0, "TP10": 1.0, "AF7": 0.3, "AF8": 0.3}   # alpha posterior
    theta_w = {"TP9": 0.4, "TP10": 0.4, "AF7": 1.0, "AF8": 1.0}   # theta frontal
    alpha = np.sin(2 * np.pi * 10 * t + rng.random() * 6)
    theta = np.sin(2 * np.pi * 6 * t + rng.random() * 6)

    data = {"TimeStamp": np.round(t, 4)}
    for ch in CH:
        noise = np.cumsum(rng.standard_normal(n))
        noise = noise - np.convolve(noise, np.ones(64) / 64, mode="same")
        sig = (alpha_w[ch] * pf["alpha"] * alpha
               + theta_w[ch] * pf["theta"] * theta
               + pf["noise"] * rng.standard_normal(n) + 1.2 * noise)
        data[f"RAW_{ch}"] = np.round(sig, 3)   # microvolts

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(OUT, index=False)
    print(f"saved SYNTHETIC Muse '{args.profile}' demo: {OUT}  "
          f"({len(CH)} ch, {sfreq:.0f} Hz, {secs}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
