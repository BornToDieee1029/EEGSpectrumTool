"""Microstate analysis (pipeline stage 4).

Method (see docs/ARCHITECTURE.md section 5):
  1. Global Field Power (GFP); topographies are most stable at GFP peaks.
  2. Extract maps at GFP peaks only.
  3. Cluster with modified k-means -- POLARITY-INVARIANT. Ordinary k-means is
     wrong for EEG topographies (dipole sign flips must be ignored).
  4. Fixed k (default 4) so subjects are comparable.
  5. Build a GROUP template, then backfit every recording onto it. Individually
     fit maps are NOT comparable across people.

Backed by `pycrostates`.
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
from pycrostates.cluster import ModKMeans
from pycrostates.io import ChData
from pycrostates.preprocessing import extract_gfp_peaks

from .config import MicrostateConfig


@dataclass
class MicrostateMaps:
    """A fitted set of template topographies plus the cluster model that made
    them (kept so new recordings can be backfit onto the same template)."""
    cluster: ModKMeans          # fitted model, used by backfit()
    maps: np.ndarray            # (n_states, n_channels)
    ch_names: list[str]
    gev: float                  # global explained variance of the fit


@dataclass
class Segmentation:
    """A recording backfit onto template maps: per-sample state labels."""
    labels: np.ndarray          # (n_times,), values in -1..n_states-1
    sfreq: float
    n_states: int


def fit_group_template(
    recordings: list[mne.io.BaseRaw], cfg: MicrostateConfig
) -> MicrostateMaps:
    """Fit shared template maps across a cohort.

    GFP peaks are pooled across all recordings (topographies are most stable and
    informative at GFP peaks), then clustered once with polarity-invariant
    modified k-means. A single shared template is the only way per-subject
    segmentations are comparable. See ARCHITECTURE.md section 5.
    """
    peak_data = []
    info = None
    for raw in recordings:
        peaks = extract_gfp_peaks(raw, verbose="ERROR")
        peak_data.append(peaks.get_data())
        info = peaks.info
    pooled = ChData(np.hstack(peak_data), info)

    cluster = ModKMeans(
        n_clusters=cfg.n_states,
        random_state=cfg.random_seed,
    )
    cluster.fit(pooled, verbose="ERROR")
    return MicrostateMaps(
        cluster=cluster,
        maps=cluster.cluster_centers_,
        ch_names=list(info["ch_names"]),
        gev=float(cluster.GEV_),
    )


def backfit(
    raw: mne.io.BaseRaw, template: MicrostateMaps, cfg: MicrostateConfig
) -> Segmentation:
    """Label every sample of one recording with the nearest template map.

    Competitive backfitting onto the shared template (polarity-invariant). Edge
    and very short segments are rejected (left as -1) per pycrostates defaults.
    """
    min_seg = int(round(cfg.min_segment_ms / 1000 * raw.info["sfreq"]))
    seg = template.cluster.predict(
        raw,
        factor=cfg.smoothing_factor,
        half_window_size=cfg.smoothing_half_window,
        min_segment_length=min_seg,
        reject_by_annotation=False,
        verbose="ERROR",
    )
    return Segmentation(
        labels=np.asarray(seg.labels),
        sfreq=raw.info["sfreq"],
        n_states=cfg.n_states,
    )
