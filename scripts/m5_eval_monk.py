#!/usr/bin/env python3
"""M5 -- evaluate the tool against the monk dataset.

Builds the stability axis from the 121-subject cohort, then projects BOTH monk
conditions (with / without chanting) onto it. The hypothesis: the monk lands at
the stable extreme -- left of the control distribution.

For a fair comparison the monk (10 min) is cropped to the cohort's mean duration
so its dynamic features are estimated from a comparable amount of data.

    python scripts/m5_eval_monk.py
    open data/interim/monk_on_spectrum.png
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

from eeg_spectrum import io  # noqa: E402
from eeg_spectrum.clean import preprocess  # noqa: E402
from eeg_spectrum.config import (  # noqa: E402
    CleanConfig, HarmonizeConfig, MicrostateConfig, ScoreConfig,
)
from eeg_spectrum.features import extract  # noqa: E402
from eeg_spectrum.harmonize import shared_channels, to_common_space  # noqa: E402
from eeg_spectrum.microstates import backfit, fit_group_template  # noqa: E402
from eeg_spectrum.score import fit_axis, place  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ADHD_CSV = ROOT / "data" / "raw" / "adhdata.csv"
MONK_DIR = ROOT / "data" / "monk"
CLEAN = CleanConfig(artifact_method="none")
MS = MicrostateConfig(n_states=4)


def prep(raw, common):
    cfg = HarmonizeConfig(target_sfreq=100.0, common_channels=tuple(common))
    return preprocess(to_common_space(raw, cfg), CLEAN)


def main() -> int:
    # Reference channel space from a sample ADHD subject.
    common = shared_channels(
        io.load_openbci_txt(next(MONK_DIR.glob("*With*"))).info["ch_names"],
        next(io.iter_adhd_subjects(ADHD_CSV))[2].info["ch_names"],
    )

    print("cleaning cohort + fitting template...")
    cohort, labels, durs = [], [], []
    for sid, label, raw in io.iter_adhd_subjects(ADHD_CSV):
        cohort.append(prep(raw, common))
        labels.append(1 if label.upper().startswith("ADHD") else 0)
        durs.append(raw.n_times / raw.info["sfreq"])
    template = fit_group_template(cohort, MS)
    crop_s = float(np.mean(durs))
    print(f"  template GEV={template.gev:.3f}; cohort mean dur={crop_s:.0f}s")

    # Cohort feature matrix + two axes: full vector vs robust core dynamics.
    rows = [extract(backfit(r, template, MS)) for r in cohort]
    df = pd.DataFrame(rows)
    y = np.array(labels)
    core_cols = ["transition_entropy", "switch_rate"]

    full_cols = list(df.columns)
    axis_full = fit_axis(df[full_cols].to_numpy(), y, full_cols, ScoreConfig())
    axis_core = fit_axis(df[core_cols].to_numpy(), y, core_cols, ScoreConfig())
    print(f"  full-vector axis  LOO-CV AUC = {axis_full.cv_auc:.3f}")
    print(f"  core-dynamics axis LOO-CV AUC = {axis_core.cv_auc:.3f}")

    # Process both monk conditions, cropped to the cohort mean duration.
    print("\n--- monk on the spectrum (cropped to cohort mean length) ---")
    print(f"{'condition':<18}{'entropy':>9}{'switch':>8}"
          f"{'  stable>controls (core)':>24}{'  (full)':>10}")
    results = {}
    for f in sorted(MONK_DIR.glob("*.txt")):
        cond = "without chanting" if "Without" in f.name else "with chanting"
        raw = prep(io.load_openbci_txt(f), common).crop(tmax=crop_s)
        feats = extract(backfit(raw, template, MS))
        rc = place(axis_core, feats)
        rf = place(axis_full, feats)
        results[cond] = (rc, feats)
        print(f"{cond:<18}{feats['transition_entropy']:>9.3f}"
              f"{feats['switch_rate']:>8.2f}"
              f"{rc['stability_vs_controls']:>21.1f}%"
              f"{rf['stability_vs_controls']:>9.1f}%")

    print("\n(core-dynamics axis is the robust one for out-of-distribution"
          " anchors like the monk; the full vector extrapolates unstably.)")
    _plot(axis_core, results, ROOT / "data" / "interim" / "monk_on_spectrum.png")
    return 0


def _plot(axis, results, out):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.scatter(axis.hc_positions, np.full_like(axis.hc_positions, 0.0),
               c="#888780", s=40, alpha=0.6, label="controls")
    ax.scatter(axis.adhd_positions, np.full_like(axis.adhd_positions, 0.0),
               c="#E24B4A", s=40, alpha=0.6, label="ADHD")
    colors = {"with chanting": "#0F6E56", "without chanting": "#1D9E75"}
    for cond, (res, _) in results.items():
        ax.axvline(res["projection"], color=colors[cond], lw=2,
                   label=f"monk ({cond})")
    ax.set_yticks([])
    ax.set_xlabel("instability axis  (left = stable,  right = ADHD-like)")
    ax.set_title("Monk placed on the cohort stability axis")
    ax.legend(loc="upper center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    raise SystemExit(main())
