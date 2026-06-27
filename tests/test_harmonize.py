"""Unit tests for harmonization, on synthetic data (no real EEG needed)."""

from __future__ import annotations

import mne
import numpy as np

from eeg_spectrum.config import HarmonizeConfig
from eeg_spectrum.harmonize import shared_channels, to_common_space


def _fake_raw(ch_names, sfreq, n_seconds=2):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((len(ch_names), int(sfreq * n_seconds))) * 1e-5
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


def test_to_common_space_subset_resample_reref():
    # Monk-like 125 Hz with extra channels, ADHD-like 128 Hz with midline.
    monk = _fake_raw(["Fp1", "Fp2", "C3", "C4", "Oz"], sfreq=125)
    adhd = _fake_raw(["Fp1", "Fp2", "C3", "C4", "Fz", "Cz"], sfreq=128)

    common = shared_channels(monk.info["ch_names"], adhd.info["ch_names"])
    assert common == ["Fp1", "Fp2", "C3", "C4"]

    cfg = HarmonizeConfig(target_sfreq=100.0, common_channels=tuple(common))
    m, a = to_common_space(monk, cfg), to_common_space(adhd, cfg)

    # Same channels, same order, same rate.
    assert m.info["ch_names"] == a.info["ch_names"] == common
    assert m.info["sfreq"] == a.info["sfreq"] == 100.0
    # Average reference: channels sum to ~0 at every sample.
    assert np.allclose(m.get_data().sum(axis=0), 0, atol=1e-12)


def test_missing_common_channel_raises():
    raw = _fake_raw(["Fp1", "Fp2"], sfreq=128)
    cfg = HarmonizeConfig(common_channels=("Fp1", "Fp2", "Cz"))
    try:
        to_common_space(raw, cfg)
    except ValueError as e:
        assert "Cz" in str(e)
    else:
        raise AssertionError("expected ValueError for missing channel")
