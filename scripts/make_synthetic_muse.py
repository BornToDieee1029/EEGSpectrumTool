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

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples" / "synthetic_muse_demo.csv"
CH = ["TP9", "AF7", "AF8", "TP10"]            # temporoparietal + frontal


def main() -> int:
    rng = np.random.default_rng(3)
    sfreq, secs = 256.0, 180
    n = int(sfreq * secs)
    t = np.arange(n) / sfreq

    # Alpha stronger at the temporoparietal (posterior-ish) sensors TP9/TP10.
    alpha_w = {"TP9": 1.0, "TP10": 1.0, "AF7": 0.3, "AF8": 0.3}
    alpha = np.sin(2 * np.pi * 10 * t + rng.random() * 6)

    data = {"TimeStamp": np.round(t, 4)}
    for ch in CH:
        # 1/f-ish background: cumulative noise high-passed lightly.
        noise = np.cumsum(rng.standard_normal(n))
        noise = noise - np.convolve(noise, np.ones(64) / 64, mode="same")
        sig = alpha_w[ch] * 7 * alpha + 9 * rng.standard_normal(n) + 1.2 * noise
        data[f"RAW_{ch}"] = np.round(sig, 3)   # microvolts

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(OUT, index=False)
    print(f"saved SYNTHETIC Muse demo: {OUT}  ({len(CH)} ch, {sfreq:.0f} Hz, {secs}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
