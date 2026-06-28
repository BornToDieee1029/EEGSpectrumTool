#!/usr/bin/env python3
"""Train and persist the model artifact the Streamlit app loads.

Builds the group microstate template + core-dynamics stability axis from the
reference cohort, places the monk landmarks, and saves everything to
models/spectrum_model.joblib.

    python scripts/train_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eeg_spectrum.pipeline import build_artifact, save_artifact  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "spectrum_model.joblib"


def main() -> int:
    print("training artifact on the reference cohort (this takes a few min)...")
    art = build_artifact(ROOT / "data" / "raw" / "adhdata.csv",
                         ROOT / "data" / "monk", n_adolescents=8)
    save_artifact(art, OUT)
    print(f"  template GEV = {art.template.gev:.3f}")
    print(f"  axis LOO-CV AUC = {art.axis.cv_auc:.3f}")
    print(f"  monk landmarks = {art.monk_positions}")
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
