#!/usr/bin/env python3
"""Place the monk against a HEALTHY ADULT reference (addresses the pediatric
confound, LIMITATIONS.md sec 2).

Downloads N eyes-closed resting baselines from PhysioNet EEGMMIDB (healthy
adults, run R02), runs them through the SAME pipeline + group template as the
trained artifact, and shows where the monk's Empty Mind state sits relative to
healthy adults on the core dynamics. No auth required.

    python scripts/adult_reference.py --limit 40
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eeg_spectrum import io  # noqa: E402
from eeg_spectrum.clean import preprocess  # noqa: E402
from eeg_spectrum.features import extract  # noqa: E402
from eeg_spectrum.harmonize import to_common_space  # noqa: E402
from eeg_spectrum.microstates import backfit  # noqa: E402
from eeg_spectrum.pipeline import load_artifact  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external" / "physionet"
BASE = "https://physionet.org/files/eegmmidb/1.0.0"
KEY = ["transition_entropy", "switch_rate"]


def fetch(sid: int) -> Path | None:
    s = f"S{sid:03d}"
    dest = EXT / f"{s}R02.edf"
    if dest.exists():
        return dest
    EXT.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{s}/{s}R02.edf?download"
    r = subprocess.run(["curl", "-sf", "-o", str(dest), url])
    return dest if r.returncode == 0 and dest.stat().st_size > 0 else None


def feats_for(raw, art) -> dict:
    raw = preprocess(to_common_space(raw, art.harmonize), art.clean)
    return extract(backfit(raw, art.template, art.microstates))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    art = load_artifact(ROOT / "models" / "spectrum_model.joblib")

    print(f"downloading + processing up to {args.limit} healthy adults...")
    ent, sw = [], []
    for sid in range(1, args.limit + 1):
        f = fetch(sid)
        if f is None:
            continue
        try:
            fe = feats_for(io.load_eeg(f), art)
            ent.append(fe["transition_entropy"])
            sw.append(fe["switch_rate"])
        except Exception as exc:  # noqa: BLE001
            print(f"  skip S{sid:03d}: {exc}")
    ent, sw = np.array(ent), np.array(sw)
    print(f"  adults processed: {len(ent)}")

    # Monk Empty Mind features (the anchor), same pipeline.
    monk_file = next((ROOT / "data" / "monk").glob("*Without*"))
    mf = feats_for(io.load_openbci_txt(monk_file), art)
    me, ms_ = mf["transition_entropy"], mf["switch_rate"]

    pct_e = 100 * float(np.mean(ent > me))
    print("\n--- monk Empty Mind vs HEALTHY ADULTS ---")
    print(f"adult entropy : mean {ent.mean():.3f}  sd {ent.std():.3f}  "
          f"range [{ent.min():.3f}, {ent.max():.3f}]")
    print(f"monk  entropy : {me:.3f}  -> more stable than {pct_e:.0f}% of adults")
    print(f"adult switch  : mean {sw.mean():.2f}/s ; monk {ms_:.2f}/s")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, vals, mv, name in [(axes[0], ent, me, "transition entropy"),
                               (axes[1], sw, ms_, "switch rate /s")]:
        ax.hist(vals, bins=12, color="#85B7EB", edgecolor="#185FA5")
        ax.axvline(mv, color="#0F6E56", lw=2.5, label="monk (empty mind)")
        ax.set_title(name)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Monk vs {len(ent)} healthy adults (PhysioNet, eyes closed)")
    fig.tight_layout()
    out = ROOT / "data" / "interim" / "monk_vs_adults.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
