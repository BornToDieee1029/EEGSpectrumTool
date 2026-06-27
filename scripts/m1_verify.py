#!/usr/bin/env python3
"""M1 -- verify harmonization brings monk + ADHD into one identical space.

Loads a monk recording and one ADHD subject, computes the shared channel
subset, harmonizes both, and asserts they end up with the SAME channels in the
SAME order at the SAME sampling rate. Run from repo root:

    python scripts/m1_verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from eeg_spectrum import io  # noqa: E402
from eeg_spectrum.config import HarmonizeConfig  # noqa: E402
from eeg_spectrum.harmonize import shared_channels, to_common_space  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MONK = next((ROOT / "data" / "monk").glob("*.txt"))
ADHD_CSV = ROOT / "data" / "raw" / "adhdata.csv"


def main() -> int:
    monk = io.load_openbci_txt(MONK)
    sid, label, adhd = next(io.iter_adhd_subjects(ADHD_CSV))

    common = shared_channels(monk.info["ch_names"], adhd.info["ch_names"])
    print(f"shared subset ({len(common)}): {', '.join(common)}")

    cfg = HarmonizeConfig(target_sfreq=100.0, common_channels=tuple(common))
    monk_h = to_common_space(monk, cfg)
    adhd_h = to_common_space(adhd, cfg)

    print(f"\nmonk : {monk.info['sfreq']:.0f}Hz/{monk.info['nchan']}ch "
          f"-> {monk_h.info['sfreq']:.0f}Hz/{monk_h.info['nchan']}ch")
    print(f"adhd : {adhd.info['sfreq']:.0f}Hz/{adhd.info['nchan']}ch "
          f"-> {adhd_h.info['sfreq']:.0f}Hz/{adhd_h.info['nchan']}ch ({sid}/{label})")

    assert monk_h.info["ch_names"] == adhd_h.info["ch_names"], "channel order mismatch!"
    assert monk_h.info["sfreq"] == adhd_h.info["sfreq"] == 100.0
    # Average reference => every sample sums to ~0 across channels.
    assert np.allclose(monk_h.get_data().sum(axis=0), 0, atol=1e-12)
    assert np.allclose(adhd_h.get_data().sum(axis=0), 0, atol=1e-12)

    print("\nOK: identical 16-ch / 100Hz / average-referenced space on both.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
