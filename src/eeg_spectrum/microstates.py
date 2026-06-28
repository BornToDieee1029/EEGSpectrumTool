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


_REGION = {
    "Fp1": "left prefrontal", "Fp2": "right prefrontal",
    "F3": "left frontal", "F4": "right frontal",
    "F7": "left frontotemporal", "F8": "right frontotemporal",
    "C3": "left central", "C4": "right central",
    "T7": "left temporal", "T8": "right temporal",
    "P3": "left parietal", "P4": "right parietal",
    "P7": "left posterior-temporal", "P8": "right posterior-temporal",
    "O1": "left occipital", "O2": "right occipital",
}


def describe_maps(maps: np.ndarray, ch_names: list[str]) -> list[dict]:
    """Describe each template map's dominant voltage axis from electrode geometry.

    Returns, per map: dominant axis (anterior-posterior vs left-right), the
    strongest electrode and its scalp region, and a tentative canonical-microstate
    association (heuristic, from topography only).
    """
    info = mne.create_info(list(ch_names), 100, "eeg")
    info.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")
    pos = info.get_montage().get_positions()["ch_pos"]
    P = np.array([pos[c] for c in ch_names])
    x, y = P[:, 0], P[:, 1]   # +x = right, +y = anterior
    out = []
    for m in range(maps.shape[0]):
        v = maps[m]
        cx = abs(np.corrcoef(v, x)[0, 1])
        cy = abs(np.corrcoef(v, y)[0, 1])
        peak = ch_names[int(np.argmax(np.abs(v)))]
        region = _REGION.get(peak, peak)
        posterior = any(w in region for w in ("occipital", "parietal", "posterior"))
        if cy >= cx and posterior:
            axis = "front-to-back, anchored at the back of the head"
            assoc = "visual / posterior network (canonical map B/C)"
            meaning = ("activity sweeps from the front of the head to the back and "
                       "is strongest over the visual area at the back. It reflects "
                       "the brain's visual and resting sensory network.")
        elif cy >= cx:
            axis = "front-to-back, anchored at the front of the head"
            assoc = "attention / executive network (canonical map D)"
            meaning = ("activity sweeps from front to back and is strongest over "
                       "the forehead. It reflects the attention and "
                       "executive-control network — the pattern most often altered "
                       "in ADHD.")
        else:
            axis = "side-to-side, between the two hemispheres"
            assoc = "lateralized language / visual network (canonical map A/B)"
            meaning = ("one side of the head is positive while the other is "
                       "negative. It reflects left-versus-right hemisphere balance, "
                       "linked to the language and visual networks.")
        out.append({"index": m, "axis": axis, "peak": peak,
                    "region": region, "assoc": assoc, "meaning": meaning})
    return out


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
