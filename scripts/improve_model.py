#!/usr/bin/env python3
"""Improve accuracy with spectral features, and explore ADHD subtypes honestly.

Part A -- accuracy: compare leave-one-subject-out CV AUC for
  (1) microstate dynamics only, (2) spectral only, (3) combined.
Part B -- subtypes: the cohort has NO subtype labels, so we cannot classify
  inattentive / hyperactive-impulsive / combined. Instead we cluster the ADHD
  subjects unsupervised and report whether distinct EEG subgroups emerge.

    python scripts/improve_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, silhouette_score  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from eeg_spectrum import io  # noqa: E402
from eeg_spectrum.clean import preprocess  # noqa: E402
from eeg_spectrum.config import CleanConfig, HarmonizeConfig, MicrostateConfig  # noqa: E402
from eeg_spectrum.features import extract, spectral_features  # noqa: E402
from eeg_spectrum.harmonize import shared_channels, to_common_space  # noqa: E402
from eeg_spectrum.microstates import backfit, fit_group_template  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ADHD_CSV = ROOT / "data" / "raw" / "adhdata.csv"
MONK = next((ROOT / "data" / "monk").glob("*With*"))
CLEAN = CleanConfig(artifact_method="none")
MS = MicrostateConfig(n_states=4)
MICRO = ["transition_entropy", "switch_rate"] + \
        [f"{m}_{i}" for m in ("duration", "coverage", "occurrence") for i in range(4)]
SPEC = ["rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma",
        "theta_beta_ratio", "frontal_theta_beta"]


def loo_auc(X, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, preds)


def main() -> int:
    common = shared_channels(
        io.load_openbci_txt(MONK).info["ch_names"],
        next(io.iter_adhd_subjects(ADHD_CSV))[2].info["ch_names"],
    )
    harm = HarmonizeConfig(target_sfreq=100.0, common_channels=tuple(common))

    print("cleaning cohort + extracting microstate + spectral features...")
    cohort, labels = [], []
    for sid, label, raw in io.iter_adhd_subjects(ADHD_CSV):
        cohort.append(preprocess(to_common_space(raw, harm), CLEAN))
        labels.append(1 if label.upper().startswith("ADHD") else 0)
    template = fit_group_template(cohort, MS)

    rows = []
    for raw in cohort:
        f = extract(backfit(raw, template, MS))
        f.update(spectral_features(raw))
        rows.append(f)
    df = pd.DataFrame(rows)
    df["ADHD"] = labels
    df.to_csv(ROOT / "data" / "interim" / "features_combined.csv", index=False)
    y = np.array(labels)

    print("\n=== Part A: accuracy (leave-one-subject-out CV AUC) ===")
    print(f"  microstate dynamics only : {loo_auc(df[MICRO].to_numpy(), y):.3f}")
    print(f"  spectral only            : {loo_auc(df[SPEC].to_numpy(), y):.3f}")
    print(f"  COMBINED                 : {loo_auc(df[MICRO + SPEC].to_numpy(), y):.3f}")
    print("  (prior microstate-core baseline was 0.70)")

    print("\n=== Part B: unsupervised ADHD subtype exploration ===")
    print("  NOTE: cohort has NO subtype labels -> this is exploratory, not")
    print("  detection of inattentive/hyperactive/combined.")
    adhd = df[df["ADHD"] == 1][MICRO + SPEC]
    Xa = StandardScaler().fit_transform(adhd.to_numpy())
    for k in (2, 3):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xa)
        sil = silhouette_score(Xa, km.labels_)
        sizes = np.bincount(km.labels_)
        print(f"  k={k}: silhouette {sil:.3f}  cluster sizes {sizes.tolist()}")
    print("  (silhouette < ~0.25 => no convincing subgroup structure in EEG)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
