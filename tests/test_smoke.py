"""Smoke tests that run without any real EEG data.

These verify the scaffold is wired correctly; real-data tests come per milestone.
"""

from __future__ import annotations

from eeg_spectrum import DEFAULT
from eeg_spectrum.harmonize import shared_channels


def test_config_defaults():
    assert DEFAULT.harmonize.target_sfreq == 100.0
    assert DEFAULT.microstates.n_states == 4
    # Determinism: every seeded stage shares one seed.
    assert DEFAULT.clean.random_seed == DEFAULT.microstates.random_seed


def test_shared_channels_case_insensitive_intersection():
    a = ["Fp1", "Fp2", "Cz", "O1", "O2"]
    b = ["FP1", "fp2", "Cz", "Pz"]
    # Order follows the first list; case-insensitive match.
    assert shared_channels(a, b) == ["Fp1", "Fp2", "Cz"]


def test_shared_channels_empty():
    assert shared_channels() == []
    assert shared_channels(["Cz"], ["Pz"]) == []
