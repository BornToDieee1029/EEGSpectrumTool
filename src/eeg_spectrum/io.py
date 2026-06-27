"""EEG file loading and header introspection (pipeline stage 1).

The loader detects format from the extension and returns an MNE Raw object plus
a small RecordingInfo summary used by the M0 reconnaissance step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

# Monk OpenBCI EXG -> 10-20 mapping. Confirmed standard Ultracortex Mark IV
# 16-channel wiring order (user-confirmed M0, 2026-06-27). Earlobes were wired to
# BIAS/SRB2 (reference/ground) and are not signal channels. No midline (Fz/Cz/Pz).
# All 16 fall inside the ADHD 19 -> 16-channel shared space, no interpolation.
MONK_MONTAGE_MAP: dict[str, str] | None = {
    "EXG0": "Fp1", "EXG1": "Fp2", "EXG2": "C3", "EXG3": "C4",
    "EXG4": "P7", "EXG5": "P8", "EXG6": "O1", "EXG7": "O2",
    "EXG8": "F7", "EXG9": "F8", "EXG10": "F3", "EXG11": "F4",
    "EXG12": "T7", "EXG13": "T8", "EXG14": "P3", "EXG15": "P4",
}


@dataclass
class RecordingInfo:
    path: str
    n_channels: int
    ch_names: list[str]
    sfreq: float
    duration_s: float
    line_freq: float | None

    def summary(self) -> str:
        mins = self.duration_s / 60.0
        return (
            f"{Path(self.path).name}\n"
            f"  channels  : {self.n_channels}\n"
            f"  names     : {', '.join(self.ch_names)}\n"
            f"  sfreq     : {self.sfreq} Hz\n"
            f"  duration  : {self.duration_s:.1f} s ({mins:.1f} min)\n"
            f"  line_freq : {self.line_freq}"
        )


# Map file extensions to the MNE reader that handles them.
_READERS = {
    ".edf": mne.io.read_raw_edf,
    ".bdf": mne.io.read_raw_bdf,
    ".set": mne.io.read_raw_eeglab,   # EEGLAB
    ".vhdr": mne.io.read_raw_brainvision,
    ".fif": mne.io.read_raw_fif,
    ".cnt": mne.io.read_raw_cnt,
}


# Old 10-20 labels -> modern equivalents used by the ADHD montage.
_CH_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def normalize_ch_names(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Clean channel labels so external datasets match our 10-20 montage.

    Strips the trailing dots/whitespace many EDF exports carry (PhysioNet writes
    'Fp1.', 'T7..'), normalizes case, and maps old T3/T4/T5/T6 names to the
    modern T7/T8/P7/P8. In place; returns raw for chaining.
    """
    mapping = {}
    for name in raw.info["ch_names"]:
        clean = name.strip().strip(".").strip()
        canon = clean.capitalize() if len(clean) > 2 else clean.upper()
        # Preserve standard 10-20 casing (Fp1, Fz, Cz, Pz, Oz, etc.).
        for std in ("Fp1", "Fp2", "Fz", "Cz", "Pz", "Oz", "Fpz"):
            if clean.lower() == std.lower():
                canon = std
        canon = _CH_ALIASES.get(canon.upper(), canon)
        if canon != name:
            mapping[name] = canon
    if mapping:
        raw.rename_channels(mapping)
    return raw


def load_eeg(path: str | Path, preload: bool = True) -> mne.io.BaseRaw:
    """Load an EEG file, dispatching on extension. Raises on unknown format.

    Channel labels are normalized so external 10-20 datasets line up with our
    montage without manual relabeling.
    """
    path = Path(path)
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(
            f"Unsupported EEG format {path.suffix!r}. "
            f"Supported: {sorted(_READERS)}"
        )
    raw = reader(path, preload=preload, verbose="ERROR")
    return normalize_ch_names(raw)


def load_openbci_txt(
    path: str | Path, montage_map: dict[str, str] | None = MONK_MONTAGE_MAP
) -> mne.io.BaseRaw:
    """Load an OpenBCI Cyton+Daisy raw .txt export (the monk recordings).

    Parses the %-prefixed header for sample rate and channel count, then reads
    the 16 EXG columns. OpenBCI exports microvolts; MNE wants volts, so we scale
    by 1e-6. If `montage_map` is given (EXG name -> 10-20 name) channels are
    renamed and a standard_1020 montage attached.
    """
    path = Path(path)
    sfreq = 125.0
    with path.open() as fh:
        for line in fh:
            if not line.startswith("%"):
                break
            if "Sample Rate" in line:
                sfreq = float(line.split("=")[1].strip().split()[0])

    df = pd.read_csv(path, skiprows=4, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    exg_cols = [c for c in df.columns if c.startswith("EXG Channel")]
    data = df[exg_cols].to_numpy(dtype=float).T * 1e-6   # uV -> V

    ch_names = [c.replace("EXG Channel ", "EXG") for c in exg_cols]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    if montage_map:
        raw.rename_channels({f"EXG{i}": montage_map[f"EXG{i}"] for i in range(len(ch_names))})
        raw.set_montage("standard_1020", on_missing="warn", verbose="ERROR")
    return raw


def iter_adhd_subjects(path: str | Path):
    """Yield (subject_id, label, Raw) for each subject in the ADHD CSV.

    The CSV stacks every subject's samples; columns are 19 10-20 channels plus
    Class (ADHD/Control) and ID. 128 Hz. Values are microvolts -> scaled to V.
    """
    path = Path(path)
    df = pd.read_csv(path)
    ch_names = [c for c in df.columns if c not in ("Class", "ID")]
    for sid, sub in df.groupby("ID", sort=False):
        label = sub["Class"].iloc[0]
        data = sub[ch_names].to_numpy(dtype=float).T * 1e-6   # uV -> V
        info = mne.create_info(ch_names, sfreq=128.0, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose="ERROR")
        raw.set_montage("standard_1020", on_missing="warn", verbose="ERROR")
        yield str(sid), str(label), raw


def load_any(path: str | Path) -> mne.io.BaseRaw:
    """Load any supported recording, dispatching OpenBCI .txt vs standard formats.

    Used by the app's file uploader, which doesn't know the format in advance.
    """
    path = Path(path)
    if path.suffix.lower() == ".txt":
        return load_openbci_txt(path)
    return load_eeg(path)


def describe(raw: mne.io.BaseRaw, path: str | Path) -> RecordingInfo:
    """Build a RecordingInfo summary from a loaded Raw object."""
    info = raw.info
    return RecordingInfo(
        path=str(path),
        n_channels=info["nchan"],
        ch_names=list(info["ch_names"]),
        sfreq=info["sfreq"],
        duration_s=raw.n_times / info["sfreq"],
        line_freq=info.get("line_freq"),
    )
