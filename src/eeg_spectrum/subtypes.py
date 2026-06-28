"""EXPLORATORY ADHD-presentation likeness — NOT a validated classifier.

There is no subtype-labeled training data (the cohort is binary ADHD/control),
so this CANNOT detect clinical subtype. It is a transparent, literature-grounded
heuristic that scores a recording against the documented EEG signatures of the
three DSM/CDC presentations, for hypothesis generation only:

  - inattentive            : slow-wave / theta excess, "hypoarousal", low
                             behavioural switching (elevated theta/beta ratio)
  - hyperactive-impulsive  : fast, unstable state switching; less theta excess
  - combined               : both slow-wave excess AND unstable switching

Clinical diagnosis depends on symptom history across settings, not EEG. Always
present these numbers with that caveat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SUBTYPES = ("inattentive", "hyperactive_impulsive", "combined")

# Sign of each feature's contribution to each presentation's EEG signature.
# Rows = subtype, cols = (theta_beta, switch_rate, transition_entropy).
_PATTERN = {
    "inattentive":           (+1.0, -1.0, -0.5),
    "hyperactive_impulsive": (-1.0, +1.0, +1.0),
    "combined":              (+1.0, +1.0, +1.0),
}
_FEATS = ("theta_beta_ratio", "switch_rate", "transition_entropy")


def _z(value, ref):
    mu, sd = float(np.mean(ref)), float(np.std(ref) + 1e-9)
    return (value - mu) / sd


def estimate(features: dict[str, float], reference: pd.DataFrame) -> dict:
    """Return exploratory subtype likeness (percentages) + per-subtype reasons.

    `reference` is the ADHD subset of the reference cohort, used only to z-score
    the subject's features so "high/low" is relative to the ADHD group.
    """
    adhd = reference[reference["ADHD"] == 1]
    z = {f: _z(features[f], adhd[f]) for f in _FEATS}

    scores = {}
    for st, weights in _PATTERN.items():
        scores[st] = sum(w * z[f] for w, f in zip(weights, _FEATS))

    # Tempered softmax to percentages. Temperature > 1 avoids saturated
    # 100/0/0 outputs -- the heuristic is too weak to justify that confidence.
    temperature = 2.5
    vals = np.array([scores[s] for s in SUBTYPES])
    ex = np.exp((vals - vals.max()) / temperature)
    pct = 100 * ex / ex.sum()

    def _why(st):
        bits = []
        label = {"theta_beta_ratio": "theta/beta ratio",
                 "switch_rate": "switch rate",
                 "transition_entropy": "transition entropy"}
        for w, f in zip(_PATTERN[st], _FEATS):
            zf = z[f]
            if abs(zf) < 0.25:
                continue
            direction = "high" if zf > 0 else "low"
            agrees = (w > 0) == (zf > 0)
            verb = "supports" if agrees else "argues against"
            bits.append(f"{direction} {label[f]} (z={zf:+.1f}) {verb} this")
        return "; ".join(bits) or "near the ADHD-group average on all markers"

    return {
        st: {"pct": float(round(pct[i], 1)), "why": _why(st)}
        for i, st in enumerate(SUBTYPES)
    }
