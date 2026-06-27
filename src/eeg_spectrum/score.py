"""Spectrum scoring (pipeline stage 6).

Place a recording on the state-stability axis defined by the HC->ADHD contrast
learned on the reference cohort. See docs/ARCHITECTURE.md sections 7-8.

STATISTICAL HONESTY (non-negotiable):
  * The monk is n=1: a labeled LANDMARK on the axis, never a trained class.
  * The axis position is NOT a probability of having ADHD.
  * Report leave-one-subject-out CV, never train-set performance.
  * Kaggle is pediatric; the monk is an adult -> a real developmental confound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ScoreConfig


@dataclass
class SpectrumAxis:
    """A fitted HC->ADHD direction plus reference landmarks for display."""
    model: object                 # fitted logreg/lda
    feature_names: list[str]
    hc_positions: np.ndarray      # projected reference healthy controls
    adhd_positions: np.ndarray    # projected reference ADHD
    monk_position: float | None   # single labeled landmark, if available
    cv_auc: float | None          # leave-one-subject-out performance


def fit_axis(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], cfg: ScoreConfig
) -> SpectrumAxis:
    """Learn the HC->ADHD stability axis from the reference cohort.

    The signed decision function is the axis: higher = more ADHD-like (less
    stable). We also report leave-one-subject-out CV AUC as the honest accuracy,
    and store the projected reference positions as display landmarks.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    # Honest out-of-sample accuracy.
    preds = np.zeros(len(y), dtype=float)
    for tr, te in LeaveOneOut().split(X):
        model.fit(X[tr], y[tr])
        preds[te] = model.predict_proba(X[te])[:, 1]
    cv_auc = float(roc_auc_score(y, preds)) if len(np.unique(y)) > 1 else None

    model.fit(X, y)
    proj = model.decision_function(X)
    return SpectrumAxis(
        model=model,
        feature_names=feature_names,
        hc_positions=proj[y == 0],
        adhd_positions=proj[y == 1],
        monk_position=None,
        cv_auc=cv_auc,
    )


def _project(axis: SpectrumAxis, features: dict[str, float]) -> float:
    x = np.array([[features[f] for f in axis.feature_names]])
    return float(axis.model.decision_function(x)[0])


def place(axis: SpectrumAxis, features: dict[str, float]) -> dict:
    """Project one new recording onto the axis.

    Returns the raw instability projection, a 0-100 position scaled across the
    reference span, the stability percentile (fraction of the reference cohort
    LESS stable than this recording), and an honest confidence note.
    """
    ref = np.concatenate([axis.hc_positions, axis.adhd_positions])
    proj = _project(axis, features)

    lo, hi = ref.min(), ref.max()
    position = 100.0 * (proj - lo) / (hi - lo) if hi > lo else 50.0
    # Higher projection = less stable; stability percentile counts how much of
    # the cohort is MORE unstable (i.e. has a higher projection) than this one.
    stability_pct = 100.0 * float(np.mean(ref > proj))
    hc_stability_pct = 100.0 * float(np.mean(axis.hc_positions > proj))

    return {
        "projection": proj,
        "position_0_100": float(np.clip(position, 0, 100)),
        "stability_percentile": stability_pct,
        "stability_vs_controls": hc_stability_pct,
        "cv_auc": axis.cv_auc,
        "note": ("Position is decision support, not a diagnosis. Reference "
                 "cohort is pediatric (n=121); interpret adult recordings with "
                 "the developmental-confound caveat."),
    }
