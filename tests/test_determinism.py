"""Reproducibility guard (M7): the pipeline must be deterministic.

A clinical-support tool that changes its answer on re-run is worthless. This
runs the full single-recording pipeline twice on the trained artifact and the
monk file, and asserts identical placement. Skipped if the artifact/data aren't
present (so CI without the dataset still passes).
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "spectrum_model.joblib"
MONK = ROOT / "data" / "monk"


@pytest.mark.skipif(
    not MODEL.exists() or not MONK.exists(),
    reason="trained artifact / monk data not present",
)
def test_pipeline_is_deterministic():
    from eeg_spectrum.pipeline import load_artifact, process_file

    art = load_artifact(MODEL)
    monk_file = next(MONK.glob("*.txt"))
    a = process_file(monk_file, art)["placement"]
    b = process_file(monk_file, art)["placement"]
    assert a["projection"] == b["projection"]
    assert a["position_0_100"] == b["position_0_100"]
