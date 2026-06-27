#!/usr/bin/env python3
"""M3 -- GATE #1: do ADHD and controls separate on microstate dynamics?

The decisive test (ARCHITECTURE.md section 6). Harmonize+clean all 121 subjects,
fit one group template, backfit, extract features, then:
  - univariate group stats (Mann-Whitney + AUC) for the key dynamic features
  - multivariate leave-one-subject-out CV AUC of a logistic model
Saves a per-subject feature table and a boxplot figure.

    python scripts/m3_gate1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from eeg_spectrum import io  # noqa: E402
from eeg_spectrum.clean import preprocess  # noqa: E402
from eeg_spectrum.config import CleanConfig, HarmonizeConfig, MicrostateConfig  # noqa: E402
from eeg_spectrum.features import extract  # noqa: E402
from eeg_spectrum.harmonize import shared_channels, to_common_space  # noqa: E402
from eeg_spectrum.microstates import backfit, fit_group_template  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MONK = next((ROOT / "data" / "monk").glob("*With*"))
ADHD_CSV = ROOT / "data" / "raw" / "adhdata.csv"
MS = MicrostateConfig(n_states=4)
KEY = ["transition_entropy", "switch_rate"]
ARTIFACT_METHOD = "ica" if "--ica" in sys.argv else "none"
CLEAN = CleanConfig(artifact_method=ARTIFACT_METHOD)


def prep(raw, common):
    cfg = HarmonizeConfig(target_sfreq=100.0, common_channels=tuple(common))
    return preprocess(to_common_space(raw, cfg), CLEAN)


def main() -> int:
    out_csv = ROOT / "data" / "interim" / f"features_{ARTIFACT_METHOD}.csv"
    if "--use-cache" in sys.argv and out_csv.exists():
        df = pd.read_csv(out_csv)
        print(f"loaded cached features from {out_csv}")
    else:
        monk_raw = io.load_openbci_txt(MONK)
        common = shared_channels(
            monk_raw.info["ch_names"],
            next(io.iter_adhd_subjects(ADHD_CSV))[2].info["ch_names"],
        )

        print("cleaning all 121 subjects...")
        cohort, sids, labels = [], [], []
        for sid, label, raw in io.iter_adhd_subjects(ADHD_CSV):
            cohort.append(prep(raw, common))
            sids.append(sid)
            labels.append(1 if label.upper().startswith("ADHD") else 0)

        print("fitting group template + backfitting + extracting features...")
        template = fit_group_template(cohort, MS)
        rows = [extract(backfit(raw, template, MS)) for raw in cohort]
        df = pd.DataFrame(rows)
        df.insert(0, "ID", sids)
        df.insert(1, "ADHD", labels)
        df.to_csv(out_csv, index=False)
        print(f"  saved {out_csv}  (template GEV={template.gev:.3f})")

    y = df["ADHD"].to_numpy()
    print(f"\ncohort: {len(y)}  (ADHD={y.sum()}, Control={(y == 0).sum()})")

    # Univariate: group means, Mann-Whitney, single-feature AUC.
    print("\n--- univariate separation ---")
    print(f"{'feature':<20}{'ADHD':>9}{'Control':>9}{'p':>9}{'AUC':>7}")
    for f in KEY:
        a, c = df.loc[y == 1, f], df.loc[y == 0, f]
        p = mannwhitneyu(a, c).pvalue
        auc = roc_auc_score(y, df[f])
        auc = max(auc, 1 - auc)            # direction-agnostic
        print(f"{f:<20}{a.mean():>9.3f}{c.mean():>9.3f}{p:>9.3g}{auc:>7.3f}")

    # Multivariate: leave-one-subject-out CV AUC on ALL features.
    feat_cols = [c for c in df.columns if c not in ("ID", "ADHD")]
    X = df[feat_cols].to_numpy()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict_proba(X[te])[:, 1]
    loo_auc = roc_auc_score(y, preds)
    print(f"\nmultivariate leave-one-subject-out CV AUC = {loo_auc:.3f}")
    print("  (0.5 = chance; >0.7 = the premise holds and is worth building on)")

    # Boxplots for the key features.
    fig, axes = plt.subplots(1, len(KEY), figsize=(5 * len(KEY), 4))
    for ax, f in zip(np.atleast_1d(axes), KEY):
        ax.boxplot([df.loc[y == 0, f], df.loc[y == 1, f]], tick_labels=["Control", "ADHD"])
        ax.set_title(f)
    fig.suptitle(f"Gate #1: microstate dynamics by group (LOO-CV AUC={loo_auc:.2f})")
    fig.tight_layout()
    out_png = ROOT / "data" / "interim" / "gate1.png"
    fig.savefig(out_png, dpi=110)
    print(f"saved {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
