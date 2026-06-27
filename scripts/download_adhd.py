#!/usr/bin/env python3
"""Download the Kaggle ADHD/control children EEG dataset into data/raw/.

Requires Kaggle credentials. Two options:
  1. kagglehub (recommended) -- pip install kagglehub, then `kagglehub login`
     or place ~/.kaggle/kaggle.json.
  2. Manual -- download the zip from the dataset page and unzip into data/raw/.

Edit DATASET below to the exact Kaggle slug you are using (verify it on the
dataset page -- there are several ADHD EEG datasets and they differ in montage).
"""

from __future__ import annotations

from pathlib import Path

# IEEE 19-channel EEG, 61 ADHD / 60 control children, 128 Hz, 10-20 montage.
# NOTE: this dataset ships as MATLAB .mat files, NOT EDF -- io.load_eeg() needs a
# .mat reader before M1 (see flag below).
DATASET = "danizo/eeg-dataset-for-adhd"

DEST = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> int:
    try:
        import kagglehub
    except ImportError:
        print("kagglehub not installed. Run: pip install kagglehub")
        print("Or download manually and unzip into data/raw/.")
        return 1

    if "<" in DATASET:
        print("Set DATASET to the real Kaggle slug first (see M0).")
        return 1

    path = kagglehub.dataset_download(DATASET)
    print(f"Downloaded to: {path}")
    print(f"Move/symlink the EEG files into {DEST} to keep the repo layout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
