"""End-to-end single-recording pipeline + the trained model artifact.

An Artifact bundles everything needed to score a new recording: the shared
channel space, the group microstate template, and the fitted stability axis.
Training it once (build_artifact) and saving it (save/load) means the app scores
an upload in seconds instead of re-processing the 121-subject cohort.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd

from . import io
from .clean import preprocess
from .config import CleanConfig, HarmonizeConfig, MicrostateConfig, ScoreConfig
from .features import extract
from .harmonize import shared_channels, to_common_space
from .microstates import MicrostateMaps, backfit, fit_group_template
from .score import SpectrumAxis, fit_axis

# The robust core-dynamics axis is what places out-of-distribution anchors (like
# the monk) honestly; the full feature vector extrapolates unstably. See M5.
CORE_FEATURES = ["transition_entropy", "switch_rate"]


# The monk's two conditions are quiescence vs task (see the Zen Brain research):
#   "Without 念经" = 無念 / Empty Mind  -> deep quiescence, THE healthy anchor
#   "With 念经"    = 念经 / Sutra Recitation -> internally active (Beta/Gamma task)
ANCHOR_CONDITION = "empty mind (無念)"


@dataclass
class Artifact:
    template: MicrostateMaps
    axis: SpectrumAxis
    harmonize: HarmonizeConfig
    clean: CleanConfig
    microstates: MicrostateConfig
    monk_positions: dict[str, float]      # labeled landmarks for display
    anchor: str = ANCHOR_CONDITION        # which monk condition is the anchor


def process_raw(raw: mne.io.BaseRaw, art: Artifact) -> dict:
    """Harmonize -> clean -> backfit -> features -> place, for one recording."""
    raw = to_common_space(raw, art.harmonize)
    raw = preprocess(raw, art.clean)
    seg = backfit(raw, art.template, art.microstates)
    feats = extract(seg)
    from .score import place
    placement = place(art.axis, feats)
    placement["features_entropy"] = feats["transition_entropy"]
    placement["features_switch"] = feats["switch_rate"]
    return {"features": feats, "placement": placement}


def process_file(path: str | Path, art: Artifact) -> dict:
    return process_raw(io.load_any(path), art)


def build_artifact(adhd_csv: str | Path, monk_dir: str | Path) -> Artifact:
    """Train the template + core-dynamics axis on the reference cohort."""
    adhd_csv, monk_dir = Path(adhd_csv), Path(monk_dir)
    monk_files = sorted(monk_dir.glob("*.txt"))

    common = shared_channels(
        io.load_openbci_txt(monk_files[0]).info["ch_names"],
        next(io.iter_adhd_subjects(adhd_csv))[2].info["ch_names"],
    )
    harm = HarmonizeConfig(target_sfreq=100.0, common_channels=tuple(common))
    clean = CleanConfig(artifact_method="none")
    ms = MicrostateConfig(n_states=4)

    cohort, labels, durs = [], [], []
    for sid, label, raw in io.iter_adhd_subjects(adhd_csv):
        cohort.append(preprocess(to_common_space(raw, harm), clean))
        labels.append(1 if label.upper().startswith("ADHD") else 0)
        durs.append(raw.n_times / raw.info["sfreq"])

    template = fit_group_template(cohort, ms)
    rows = [extract(backfit(r, template, ms)) for r in cohort]
    df = pd.DataFrame(rows)
    axis = fit_axis(df[CORE_FEATURES].to_numpy(), np.array(labels),
                    CORE_FEATURES, ScoreConfig())

    art = Artifact(template, axis, harm, clean, ms, monk_positions={})

    # Place the monk conditions as landmarks (cropped to cohort mean length).
    # "Without 念经" = Empty Mind (the anchor); "With 念经" = sutra recitation.
    crop_s = float(np.mean(durs))
    for f in monk_files:
        cond = ANCHOR_CONDITION if "Without" in f.name else "sutra recitation (念经)"
        raw = preprocess(to_common_space(io.load_openbci_txt(f), harm), clean)
        raw.crop(tmax=min(crop_s, raw.n_times / raw.info["sfreq"]))
        res = process_raw_features_only(raw, template, ms, axis)
        art.monk_positions[cond] = res
    return art


def process_raw_features_only(raw, template, ms, axis) -> float:
    from .score import _project
    return _project(axis, extract(backfit(raw, template, ms)))


def save_artifact(art: Artifact, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, path)


def load_artifact(path: str | Path) -> Artifact:
    return joblib.load(path)
