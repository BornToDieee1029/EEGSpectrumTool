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
    reference: pd.DataFrame | None = None  # per-subject features for all graphs


def process_raw(raw: mne.io.BaseRaw, art: Artifact) -> dict:
    """Harmonize -> clean -> backfit -> features (+spectral) -> place + subtype."""
    from .features import spectral_features
    from .score import place
    from .subtypes import estimate

    raw = to_common_space(raw, art.harmonize)
    raw = preprocess(raw, art.clean)
    seg = backfit(raw, art.template, art.microstates)

    feats = extract(seg)
    feats.update(spectral_features(raw))

    placement = place(art.axis, feats)
    placement["features_entropy"] = feats["transition_entropy"]
    placement["features_switch"] = feats["switch_rate"]

    subtype = None
    if art.reference is not None:
        subtype = estimate(feats, art.reference)
    return {"features": feats, "placement": placement, "subtype": subtype,
            "psd": raw.compute_psd(fmin=1, fmax=40, verbose="ERROR")}


def process_file(path: str | Path, art: Artifact) -> dict:
    return process_raw(io.load_any(path), art)


def _full_features(raw, template, ms):
    """Microstate + spectral features for one prepared recording."""
    from .features import spectral_features
    f = extract(backfit(raw, template, ms))
    f.update(spectral_features(raw))
    return f


def _physionet_adults(harm, clean, template, ms, n: int, cache_dir: Path):
    """Download n healthy-adult eyes-closed baselines (PhysioNet) and featurize.

    No auth required. Returns a list of feature dicts (silently skips failures).
    """
    import subprocess
    base = "https://physionet.org/files/eegmmidb/1.0.0"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for sid in range(1, n + 1):
        s = f"S{sid:03d}"
        dest = cache_dir / f"{s}R02.edf"
        if not dest.exists():
            url = f"{base}/{s}/{s}R02.edf?download"
            if subprocess.run(["curl", "-sf", "-o", str(dest), url]).returncode != 0:
                continue
        try:
            raw = preprocess(to_common_space(io.load_eeg(dest), harm), clean)
            out.append(_full_features(raw, template, ms))
        except Exception:  # noqa: BLE001
            continue
    return out


def _ds004504_older_adults(harm, clean, template, ms, cache_dir: Path):
    """Download OpenNeuro ds004504 healthy controls (older adults, 57-78 yrs,
    eyes-closed, 19ch 10-20) and featurize. No auth. Skips failures silently."""
    import csv
    import io as _io
    import subprocess
    import urllib.request

    base = "https://s3.amazonaws.com/openneuro.org/ds004504"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        txt = urllib.request.urlopen(f"{base}/participants.tsv", timeout=30).read().decode()
    except Exception:  # noqa: BLE001
        return []
    healthy = [r["participant_id"] for r in csv.DictReader(_io.StringIO(txt), delimiter="\t")
               if r.get("Group") == "C"]

    out = []
    for pid in healthy:
        dest = cache_dir / f"{pid}.set"
        if not dest.exists():
            url = f"{base}/{pid}/eeg/{pid}_task-eyesclosed_eeg.set"
            if subprocess.run(["curl", "-sf", "-o", str(dest), url]).returncode != 0:
                continue
        try:
            raw = preprocess(to_common_space(io.load_eeg(dest), harm), clean)
            out.append(_full_features(raw, template, ms))
        except Exception:  # noqa: BLE001
            continue
    return out


def _hbn_adolescents(harm, clean, template, ms, n: int, cache_dir: Path):
    """Download n HBN resting-state recordings (children/adolescents, 129-ch EGI,
    OpenNeuro ds005505) and featurize. No auth. Skips failures silently."""
    import csv
    import io as _io
    import subprocess
    import urllib.request

    base = "https://s3.amazonaws.com/openneuro.org/ds005505"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        txt = urllib.request.urlopen(f"{base}/participants.tsv", timeout=30).read().decode()
    except Exception:  # noqa: BLE001
        return []
    subs = [r["participant_id"] for r in csv.DictReader(_io.StringIO(txt), delimiter="\t")
            if r.get("RestingState") == "available"]

    out = []
    for pid in subs:
        if len(out) >= n:
            break
        dest = cache_dir / f"{pid}.set"
        if not dest.exists():
            url = f"{base}/{pid}/eeg/{pid}_task-RestingState_eeg.set"
            if subprocess.run(["curl", "-sf", "-o", str(dest), url]).returncode != 0:
                continue
        try:
            raw = preprocess(to_common_space(io.load_hbn_set(dest), harm), clean)
            out.append(_full_features(raw, template, ms))
        except Exception:  # noqa: BLE001
            continue
    return out


def build_artifact(adhd_csv: str | Path, monk_dir: str | Path,
                   n_adults: int = 80, n_adolescents: int = 15) -> Artifact:
    """Train the template + core-dynamics axis, and build the reference table.

    The reference table holds microstate + spectral features for every cohort
    subject plus many PhysioNet healthy adults -- it powers all the app graphs
    and the exploratory subtype estimate.
    """
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
    rows = [_full_features(r, template, ms) for r in cohort]
    df = pd.DataFrame(rows)
    df["ADHD"] = labels
    df["group"] = ["ADHD" if a else "control" for a in labels]
    axis = fit_axis(df[CORE_FEATURES].to_numpy(), np.array(labels),
                    CORE_FEATURES, ScoreConfig())

    # Fold in healthy references across age ranges so it isn't pediatric-only.
    ext = Path(adhd_csv).parent.parent / "external"
    extra = []
    adult_rows = _physionet_adults(harm, clean, template, ms, n_adults, ext / "physionet")
    if adult_rows:
        a = pd.DataFrame(adult_rows); a["ADHD"] = 0; a["group"] = "adult_control"
        extra.append(a)
    older_rows = _ds004504_older_adults(harm, clean, template, ms, ext / "ds004504")
    if older_rows:
        o = pd.DataFrame(older_rows); o["ADHD"] = 0; o["group"] = "older_adult"
        extra.append(o)
    teen_rows = _hbn_adolescents(harm, clean, template, ms, n_adolescents, ext / "hbn")
    if teen_rows:
        t = pd.DataFrame(teen_rows); t["ADHD"] = 0; t["group"] = "adolescent"
        extra.append(t)
    if extra:
        df = pd.concat([df, *extra], ignore_index=True)

    art = Artifact(template, axis, harm, clean, ms, monk_positions={},
                   reference=df)

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
