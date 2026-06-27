#!/usr/bin/env python3
"""M0 -- Data reconnaissance.

Point this at your EEG files and it prints their TRUE headers: channel names,
sampling rate, duration, line frequency. The instant we see real channel names,
half the open design decisions in docs/ARCHITECTURE.md resolve -- especially the
shared-electrode-subset that the whole harmonization strategy hinges on.

Usage:
    python scripts/m0_reconnaissance.py path/to/monk.edf path/to/adhd_subject.edf
    python scripts/m0_reconnaissance.py data/raw/*.edf

If two or more files are given, it also prints the SHARED channel subset --
the candidate common analysis space.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eeg_spectrum import io                       # noqa: E402
from eeg_spectrum.harmonize import shared_channels  # noqa: E402


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv]
    if not paths:
        print(__doc__)
        return 1

    infos = []
    for p in paths:
        if not p.exists():
            print(f"!! missing: {p}")
            continue
        try:
            raw = io.load_eeg(p, preload=False)
            info = io.describe(raw, p)
            infos.append(info)
            print("\n" + info.summary())
        except Exception as exc:  # noqa: BLE001 -- recon should never crash
            print(f"!! failed to read {p}: {exc}")

    if len(infos) >= 2:
        common = shared_channels(*[i.ch_names for i in infos])
        print("\n=== SHARED CHANNEL SUBSET (candidate analysis space) ===")
        print(f"  {len(common)} channels: {', '.join(common) or '(none!)'}")
        if len(common) < 12:
            print("  WARNING: <12 shared channels -- consider interpolation")
            print("  (ARCHITECTURE.md section 2, strategy 2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
