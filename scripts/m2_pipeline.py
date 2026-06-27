#!/usr/bin/env python3
"""M2 end-to-end: harmonize -> clean -> group microstate template -> backfit ->
features, on real data. Renders the template maps and prints features for the
monk vs an ADHD subject vs a control.

    python scripts/m2_pipeline.py --limit 40
    open data/interim/microstate_maps.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eeg_spectrum import io  # noqa: E402
from eeg_spectrum.clean import preprocess  # noqa: E402
from eeg_spectrum.config import CleanConfig, HarmonizeConfig, MicrostateConfig  # noqa: E402
from eeg_spectrum.features import extract  # noqa: E402
from eeg_spectrum.harmonize import shared_channels, to_common_space  # noqa: E402
from eeg_spectrum.microstates import backfit, fit_group_template  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MONK = next((ROOT / "data" / "monk").glob("*With*"))
ADHD_CSV = ROOT / "data" / "raw" / "adhdata.csv"

HARM = HarmonizeConfig(target_sfreq=100.0)
CLEAN = CleanConfig(artifact_method="none")     # filter-only first pass
MS = MicrostateConfig(n_states=4)


def prep(raw, common):
    cfg = HarmonizeConfig(target_sfreq=100.0, common_channels=tuple(common))
    return preprocess(to_common_space(raw, cfg), CLEAN)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40, help="ADHD subjects to use")
    args = ap.parse_args()

    monk_raw = io.load_openbci_txt(MONK)
    common = shared_channels(
        monk_raw.info["ch_names"],
        next(io.iter_adhd_subjects(ADHD_CSV))[2].info["ch_names"],
    )

    # Clean a sample of subjects; keep labels for the feature comparison.
    print(f"cleaning {args.limit} ADHD subjects...")
    cohort, meta = [], []
    for i, (sid, label, raw) in enumerate(io.iter_adhd_subjects(ADHD_CSV)):
        if i >= args.limit:
            break
        cohort.append(prep(raw, common))
        meta.append((sid, label))

    print("fitting group microstate template (k=4) on pooled GFP peaks...")
    template = fit_group_template(cohort, MS)
    print(f"  GEV = {template.gev:.3f}")

    # Render the 4 template maps.
    template.cluster.plot(show=False)
    out = ROOT / "data" / "interim" / "microstate_maps.png"
    plt.savefig(out, dpi=110)
    print(f"  saved {out}")

    # Feature comparison: monk vs first ADHD vs first control.
    monk = prep(monk_raw, common)
    examples = [("monk", monk)]
    for want in ("ADHD", "Control"):
        for (sid, label), raw in zip(meta, cohort):
            if label.upper().startswith(want.upper()):
                examples.append((f"{want} {sid}", raw))
                break

    print("\n--- key features (filter-only pass) ---")
    print(f"{'subject':<16}{'entropy':>9}{'switch/s':>10}{'mean dur ms':>13}")
    for name, raw in examples:
        f = extract(backfit(raw, template, MS))
        mean_dur = sum(f[f"duration_{m}"] for m in range(4)) / 4
        print(f"{name:<16}{f['transition_entropy']:>9.3f}"
              f"{f['switch_rate']:>10.2f}{mean_dur:>13.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
