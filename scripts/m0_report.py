#!/usr/bin/env python3
"""M0 reconnaissance on the ACTUAL downloaded data (monk .txt + ADHD .csv).

Prints true headers, durations, cohort composition, and the channel sets so we
can settle the harmonization design. Run from the repo root:

    python scripts/m0_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eeg_spectrum import io  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MONK_DIR = ROOT / "data" / "monk"
ADHD_CSV = ROOT / "data" / "raw" / "adhdata.csv"


def report_monk() -> list[str]:
    print("=" * 60)
    print("MONK (OpenBCI Cyton+Daisy)")
    print("=" * 60)
    ch_names: list[str] = []
    for f in sorted(MONK_DIR.glob("*.txt")):
        raw = io.load_openbci_txt(f)
        info = io.describe(raw, f)
        ch_names = info.ch_names
        print(info.summary())
        print()
    return ch_names


def report_adhd() -> list[str]:
    print("=" * 60)
    print("ADHD COHORT (Kaggle danizo CSV)")
    print("=" * 60)
    n_adhd = n_ctrl = 0
    durations = []
    ch_names: list[str] = []
    for sid, label, raw in io.iter_adhd_subjects(ADHD_CSV):
        ch_names = list(raw.info["ch_names"])
        durations.append(raw.n_times / raw.info["sfreq"])
        if label.upper().startswith("ADHD"):
            n_adhd += 1
        else:
            n_ctrl += 1
    print(f"subjects   : {n_adhd + n_ctrl}  (ADHD={n_adhd}, Control={n_ctrl})")
    print(f"sfreq      : 128 Hz")
    print(f"channels   : {len(ch_names)} -> {', '.join(ch_names)}")
    if durations:
        print(f"duration   : min {min(durations):.1f}s  "
              f"mean {sum(durations)/len(durations):.1f}s  max {max(durations):.1f}s")
    print()
    return ch_names


def main() -> int:
    monk_ch = report_monk()
    adhd_ch = report_adhd()

    print("=" * 60)
    print("HARMONIZATION READINESS")
    print("=" * 60)
    print(f"monk channels : {', '.join(monk_ch)}")
    print(f"adhd channels : {', '.join(adhd_ch)}")
    if monk_ch and monk_ch[0].startswith("EXG"):
        print("\n>> BLOCKER: monk channels are unlabeled (EXG0..15).")
        print(">> Provide the EXG -> 10-20 mapping to compute the shared subset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
