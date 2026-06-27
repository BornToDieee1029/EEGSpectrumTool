"""Preprocessing (pipeline stage 3): bandpass, notch, bad channels, artifacts.

Every random-seeded step (ICA) must use cfg.random_seed so re-runs are
deterministic -- a clinical-support tool that changes its answer on re-run is
worthless. See docs/ARCHITECTURE.md section 4.
"""

from __future__ import annotations

import mne
import numpy as np

from .config import CleanConfig


def detect_bad_channels(raw: mne.io.BaseRaw, z_thresh: float = 5.0) -> list[str]:
    """Flag flat or extreme-variance channels by robust z-score of log-variance.

    Robust (median/MAD) so a few bad channels don't mask each other. Returns the
    list of channel names judged bad.
    """
    data = raw.get_data()
    var = data.var(axis=1)
    logv = np.log(var + 1e-30)
    med = np.median(logv)
    mad = np.median(np.abs(logv - med)) + 1e-30
    z = 0.6745 * (logv - med) / mad        # robust z-score
    flat = var < 1e-20
    bad = flat | (np.abs(z) > z_thresh)
    return [raw.info["ch_names"][i] for i in np.where(bad)[0]]


def preprocess(raw: mne.io.BaseRaw, cfg: CleanConfig) -> mne.io.BaseRaw:
    """Bandpass -> notch -> bad-channel interpolation -> (optional) artifacts.

    First-pass-to-Gate-1 policy (ARCHITECTURE.md section 4): minimal defensible
    cleaning. The 1 Hz high-pass alone removes the large DC drift in raw OpenBCI
    data. Heavier artifact removal (ICA/ASR) is gated behind cfg.artifact_method
    and only worth adding once Gate #1 shows the signal is real.

    Returns a cleaned copy; does not mutate the input.
    """
    raw = raw.copy()
    raw.filter(cfg.l_freq, cfg.h_freq, verbose="ERROR")

    if cfg.notch_freq:
        raw.notch_filter(cfg.notch_freq, verbose="ERROR")

    bads = detect_bad_channels(raw)
    raw.info["bads"] = bads
    if bads and raw.get_montage() is not None:
        raw.interpolate_bads(verbose="ERROR")

    if cfg.artifact_method == "ica":
        _run_ica(raw, cfg)
    elif cfg.artifact_method not in ("none", None):
        raise ValueError(f"unknown artifact_method {cfg.artifact_method!r}")

    return raw


def _run_ica(raw: mne.io.BaseRaw, cfg: CleanConfig) -> None:
    """Fit ICA and drop components correlated with frontal (ocular) activity.

    In place. Only sensible on longer recordings; callers use cfg.artifact_method
    == 'none' for short clips. Seeded for determinism.
    """
    ica = mne.preprocessing.ICA(
        n_components=cfg.ica_n_components,
        random_state=cfg.random_seed,
        max_iter="auto",
        verbose="ERROR",
    )
    ica.fit(raw)
    # Use frontal channels as an EOG proxy (no dedicated EOG in either dataset).
    frontal = [c for c in ("Fp1", "Fp2") if c in raw.info["ch_names"]]
    bad_ic: list[int] = []
    for ch in frontal:
        idx, _ = ica.find_bads_eog(raw, ch_name=ch, verbose="ERROR")
        bad_ic.extend(idx)
    ica.exclude = sorted(set(bad_ic))
    ica.apply(raw, verbose="ERROR")
