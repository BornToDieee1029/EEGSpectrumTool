"""Cross-dataset harmonization (pipeline stage 2) -- the central problem.

Microstates cluster scalp voltage *topographies*, which only exist relative to a
montage. Any montage/sampling mismatch leaks into results as fake "brain
differences". This module removes that confound BEFORE analysis.
See docs/ARCHITECTURE.md section 2.
"""

from __future__ import annotations

import mne

from .config import HarmonizeConfig


def shared_channels(*ch_name_lists: list[str]) -> list[str]:
    """Intersection of channel names across datasets (strategy 1, preferred).

    Case-insensitive on the standard 10-20 labels. Order follows the first list.
    """
    if not ch_name_lists:
        return []
    lowered = [{c.lower() for c in names} for names in ch_name_lists]
    common = set.intersection(*lowered)
    return [c for c in ch_name_lists[0] if c.lower() in common]


def to_common_space(raw: mne.io.BaseRaw, cfg: HarmonizeConfig) -> mne.io.BaseRaw:
    """Bring one recording into the shared analysis space.

    Steps (order matters):
      1. pick the common channel subset, in a fixed shared order so topography
         vectors align across datasets (reorder_channels, not just pick)
      2. resample to cfg.target_sfreq  (durations comparable across datasets;
         MNE's FFT resampling band-limits to the new Nyquist, so no aliasing)
      3. average reference             (required for microstate topographies)

    Returns a cleaned copy; does not mutate the input.
    """
    raw = raw.copy()

    if cfg.common_channels:
        order = [c for c in cfg.common_channels if c in raw.info["ch_names"]]
        missing = [c for c in cfg.common_channels if c not in raw.info["ch_names"]]
        if missing:
            raise ValueError(f"recording is missing common channels: {missing}")
        raw.pick(order)
        raw.reorder_channels(order)   # identical channel order across datasets

    if cfg.target_sfreq and raw.info["sfreq"] != cfg.target_sfreq:
        raw.resample(cfg.target_sfreq, verbose="ERROR")

    if cfg.set_average_reference:
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    return raw
