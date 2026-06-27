"""Unit tests for feature extraction on synthetic label sequences."""

from __future__ import annotations

import numpy as np

from eeg_spectrum.features import extract, transition_matrix
from eeg_spectrum.microstates import Segmentation


def _seg(labels, sfreq=100.0, k=4):
    return Segmentation(labels=np.array(labels), sfreq=sfreq, n_states=k)


def test_static_durations_and_coverage():
    # 0 held 4 samples, 1 held 2, 2 held 2; at 100 Hz -> 40ms / 20ms / 20ms.
    f = extract(_seg([0, 0, 0, 0, 1, 1, 2, 2]))
    assert f["duration_0"] == 40.0
    assert f["duration_1"] == 20.0
    assert abs(f["coverage_0"] - 0.5) < 1e-9
    assert f["occurrence_0"] == 1 / (8 / 100.0)   # one appearance / total secs


def test_ordered_cycle_has_low_entropy():
    # Perfect 0->1->2->3->0 cycle: next state fully determined => entropy ~0.
    labels = np.repeat(np.tile([0, 1, 2, 3], 50), 2)
    f = extract(_seg(labels))
    assert f["transition_entropy"] < 1e-9


def test_random_switching_has_higher_entropy():
    rng = np.random.default_rng(0)
    # Distinct consecutive states drawn ~uniformly => high conditional entropy.
    seq = []
    prev = -1
    for _ in range(4000):
        s = rng.integers(0, 4)
        while s == prev:
            s = rng.integers(0, 4)
        seq.append(s)
        prev = s
    labels = np.repeat(seq, 2)
    f = extract(_seg(labels))
    ordered = extract(_seg(np.repeat(np.tile([0, 1, 2, 3], 50), 2)))
    assert f["transition_entropy"] > ordered["transition_entropy"]
    # Max conditional entropy for 3 reachable next-states is log2(3) ~ 1.585.
    assert f["transition_entropy"] <= np.log2(3) + 1e-6


def test_transition_matrix_ignores_unassigned_and_self():
    # -1 samples and self-loops must not count as transitions.
    tm = transition_matrix(np.array([0, 0, -1, 1, 1, 0]), n_states=2)
    # Distinct-segment sequence is 0,1,0 -> rows sum to 1 where transitions exist.
    assert tm[0, 1] == 1.0
    assert tm[1, 0] == 1.0
    assert tm[0, 0] == 0.0


def test_switch_rate_higher_for_unstable():
    stable = extract(_seg(np.repeat([0, 1], 500)))          # 1 switch / 10s
    unstable = extract(_seg(np.tile([0, 1], 500)))          # ~999 switches / 10s
    assert unstable["switch_rate"] > stable["switch_rate"]
