"""Plain-language reporting (pipeline stage 7).

Translates a spectrum placement into language a clinician can act on, while
surfacing -- never hiding -- the caveats from docs/ARCHITECTURE.md section 8
(n=1 monk, pediatric->adult transfer, montage harmonization).
"""

from __future__ import annotations


def _band(stability_vs_controls: float) -> str:
    """Plain-language band from the stability-vs-controls percentile."""
    p = stability_vs_controls
    if p >= 90:
        return "exceptional stability (contemplative-master range)"
    if p >= 65:
        return "high stability"
    if p >= 35:
        return "typical range"
    if p >= 10:
        return "low stability (toward the clinical pole)"
    return "very low stability (clinical-disorder range)"


def render_text(placement: dict, seed: int = 42) -> str:
    """Build the human-readable result string.

    Includes: where on the axis, percentile vs reference, the explicit
    decision-SUPPORT (not diagnosis) framing, and the active caveats. `seed` is
    logged so the run is reproducible.
    """
    pos = placement["position_0_100"]
    sp = placement["stability_vs_controls"]
    auc = placement.get("cv_auc")
    auc_str = f"{auc:.2f}" if auc is not None else "n/a"

    lines = [
        f"State-stability position: {pos:.0f} / 100 "
        f"(0 = extraordinary stability, 100 = clinical-disorder pole).",
        f"This recording is more stable than {sp:.0f}% of the healthy-control "
        f"reference group — {_band(sp)}.",
        "",
        "Switching dynamics:",
        f"  transition entropy {placement['features_entropy']:.3f}, "
        f"switch rate {placement['features_switch']:.2f}/s"
        if "features_entropy" in placement else "",
        "",
        "Interpretation: this is decision SUPPORT, not a diagnosis. It measures "
        "how stably the brain holds its momentary states, placed against a "
        f"reference cohort (separation AUC {auc_str}).",
        "",
        "Caveats:",
        "  - The healthy extreme is anchored to a single adult recording (n=1) — "
        "a labeled landmark, not a statistical class.",
        "  - The reference cohort is pediatric (n=121); adult recordings carry a "
        "developmental confound.",
        "  - Montage harmonized to 16 shared 10-20 channels (no midline).",
        f"  - Reproducible run, fixed seed {seed}.",
    ]
    return "\n".join(line for line in lines if line is not None)
