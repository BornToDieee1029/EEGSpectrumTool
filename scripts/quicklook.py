#!/usr/bin/env python3
"""Visual sanity check of the harmonized data (run after M1).

Saves a PNG comparing a monk recording to one ADHD subject in the shared space:
  - power spectral density (should show 1/f falloff; maybe an alpha bump ~10Hz)
  - a few seconds of raw traces (should look like EEG, amplitudes in microvolts)
  - sensor layout of the 16 shared channels

    python scripts/quicklook.py
    open data/interim/quicklook.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eeg_spectrum import io  # noqa: E402
from eeg_spectrum.clean import preprocess  # noqa: E402
from eeg_spectrum.config import CleanConfig, HarmonizeConfig  # noqa: E402
from eeg_spectrum.harmonize import shared_channels, to_common_space  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MONK = next((ROOT / "data" / "monk").glob("*With*"))   # the chanting condition
ADHD_CSV = ROOT / "data" / "raw" / "adhdata.csv"
OUT = ROOT / "data" / "interim" / "quicklook.png"


def psd_db(raw, fmax=45):
    psd = raw.compute_psd(fmax=fmax, verbose="ERROR")
    freqs = psd.freqs
    power = psd.get_data().mean(axis=0)      # average over channels
    return freqs, 10 * np.log10(power)


def main() -> int:
    monk = io.load_openbci_txt(MONK)
    sid, label, adhd = next(io.iter_adhd_subjects(ADHD_CSV))

    common = shared_channels(monk.info["ch_names"], adhd.info["ch_names"])
    cfg = HarmonizeConfig(target_sfreq=100.0, common_channels=tuple(common))
    # Harmonize, then clean (filter only for this preview -> no ICA).
    clean_cfg = CleanConfig(artifact_method="none")
    monk_h = preprocess(to_common_space(monk, cfg), clean_cfg)
    adhd_h = preprocess(to_common_space(adhd, cfg), clean_cfg)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (0,0) PSD comparison
    for raw, name, c in [(monk_h, "monk (chanting)", "C0"),
                         (adhd_h, f"ADHD {sid}", "C3")]:
        f, p = psd_db(raw)
        axes[0, 0].plot(f, p, label=name, color=c)
    axes[0, 0].axvspan(8, 12, color="gray", alpha=0.15, label="alpha 8-12Hz")
    axes[0, 0].set(title="Power spectral density (16-ch avg)",
                   xlabel="Hz", ylabel="dB")
    axes[0, 0].legend(fontsize=8)

    # (0,1) monk traces, first 5 s, a few channels
    _plot_traces(axes[0, 1], monk_h, "monk traces (5s)")
    # (1,0) adhd traces
    _plot_traces(axes[1, 0], adhd_h, f"ADHD {sid} traces (5s)")

    # (1,1) sensor positions
    monk_h.plot_sensors(axes=axes[1, 1], show_names=True, show=False)
    axes[1, 1].set_title("Shared 16-channel layout")

    fig.suptitle("Harmonized data quick-look (M1 output)")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"saved {OUT}")
    return 0


def _plot_traces(ax, raw, title, n_ch=4, secs=5):
    sf = raw.info["sfreq"]
    n = int(secs * sf)
    data = raw.get_data()[:n_ch, :n] * 1e6      # V -> uV for display
    t = np.arange(n) / sf
    for i in range(n_ch):
        ax.plot(t, data[i] + i * 80, lw=0.5)    # 80 uV vertical offset
    ax.set(title=title, xlabel="s", yticks=[])


if __name__ == "__main__":
    raise SystemExit(main())
