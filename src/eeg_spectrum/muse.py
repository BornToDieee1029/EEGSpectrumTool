"""Muse headband mode — an EXPERIMENTAL low-density band-power screen.

A 4-channel headband (TP9, AF7, AF8, TP10) cannot produce scalp microstate
topographies, so it CANNOT be placed on the state-stability spectrum (that needs
16+ electrodes). This module computes only what 4 channels honestly support:
relative band powers and the theta/beta ratio — a historically-studied but weak
attention-related marker. It is not a diagnosis and is not comparable to the
microstate spectrum or the monk anchor.
"""

from __future__ import annotations

import mne
import numpy as np

_BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12),
          "beta": (12, 30), "gamma": (30, 40)}


def muse_screen(raw: mne.io.BaseRaw) -> dict:
    """Band-power screen from a 4-channel Muse recording."""
    raw = raw.copy().filter(1, 40, verbose="ERROR")
    psd = raw.compute_psd(fmin=1, fmax=40, verbose="ERROR")
    psds, freqs = psd.get_data(return_freqs=True)
    mean = psds.mean(axis=0)
    total = mean.sum() + 1e-30

    bands = {b: float(mean[(freqs >= lo) & (freqs < hi)].sum() / total)
             for b, (lo, hi) in _BANDS.items()}
    theta_beta = bands["theta"] / (bands["beta"] + 1e-9)

    frontal = [c for c in ("AF7", "AF8") if c in raw.info["ch_names"]]
    frontal_tb = None
    if frontal:
        fp = psds[[raw.info["ch_names"].index(c) for c in frontal]].mean(axis=0)
        ft = fp[(freqs >= 4) & (freqs < 8)].sum()
        fb = fp[(freqs >= 12) & (freqs < 30)].sum()
        frontal_tb = float(ft / (fb + 1e-30))

    # EXPERIMENTAL ADHD screening indicator from the theta/beta ratio — the
    # classic (and weak) marker. NOT validated, NOT a diagnosis. Maps theta/beta
    # through a logistic curve centred near a typical resting value.
    tb = theta_beta if frontal_tb is None else 0.5 * (theta_beta + frontal_tb)
    indicator = 100.0 / (1.0 + np.exp(-(tb - 2.0) / 0.8))
    band = ("low" if indicator < 35 else
            "elevated" if indicator > 65 else "borderline")

    return {"bands": bands, "theta_beta": theta_beta,
            "frontal_theta_beta": frontal_tb, "psd": psd,
            "adhd_indicator": float(indicator), "adhd_band": band,
            "n_channels": raw.info["nchan"],
            "channels": list(raw.info["ch_names"])}
