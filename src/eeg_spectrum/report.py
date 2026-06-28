"""Plain-language reporting (pipeline stage 7).

Builds a comprehensive, human-readable result: the spectrum placement, the
state-switching dynamics, a per-map explanation of the microstate templates, the
spectral profile, the exploratory ADHD-presentation estimate, and the methods,
confidence, and limitations — always framed as decision support, never a
diagnosis (ARCHITECTURE.md section 8).
"""

from __future__ import annotations

_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
_BAND_DESC = {
    "delta": "slowest; prominent in deep sleep and, abnormally, in some task/child EEG",
    "theta": "drowsiness, memory, and internal focus",
    "alpha": "the relaxed eyes-closed rhythm; should peak near 10 Hz at rest",
    "beta": "active, alert processing",
    "gamma": "fastest; fine-grained processing (and muscle artifact if elevated)",
}


def _entropy_phrase(e: float) -> str:
    return ("highly ordered and predictable" if e < 1.15
            else "moderately ordered" if e < 1.45 else "close to random")


def _switch_phrase(s: float) -> str:
    return "slowly" if s < 7 else "at a typical rate" if s <= 9 else "rapidly"


def _alpha_line(a: float) -> str:
    p = f"{a * 100:.0f}%"
    if a >= 0.15:
        return f"Alpha is strong ({p}), consistent with genuine eyes-closed rest."
    if a >= 0.07:
        return f"Alpha is modest ({p})."
    return (f"Alpha is low ({p}) — confirm the recording was eyes-closed and at "
            "rest, as this can otherwise distort the result.")


def _band(stability_vs_controls: float) -> str:
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


def _rule(title: str) -> str:
    return f"\n\n{title}"


def render_text(
    placement: dict,
    features: dict | None = None,
    subtype: dict | None = None,
    maps: list[dict] | None = None,
    age_label: str | None = None,
    control_name: str | None = None,
    control_n: int | None = None,
    adhd_n: int | None = None,
    seed: int = 42,
) -> str:
    """Build the full report. `features`/`maps`/`subtype` are optional; when
    omitted the report degrades gracefully to the core summary."""
    f = features or {}
    pos = placement["position_0_100"]
    sp = placement["stability_vs_controls"]
    auc = placement.get("cv_auc")
    auc_str = f"{auc:.2f}" if auc is not None else "n/a"
    L: list[str] = []

    # ---- Header -----------------------------------------------------------
    L.append("EEG MICROSTATE STATE-STABILITY SPECTRUM — ANALYSIS REPORT")
    if age_label:
        L.append(f"Patient age range : {age_label}")
    if control_name:
        n = f" (n={control_n})" if control_n is not None else ""
        L.append(f"Reference group   : {control_name}{n}")
    L.append(f"Reproducibility   : fixed random seed {seed} (deterministic)")

    # ---- Summary ----------------------------------------------------------
    L.append(_rule("1. SUMMARY"))
    L.append(f"State-stability position : {pos:.0f} / 100   "
             "(0 = extraordinary stability, 100 = clinical-disorder pole)")
    L.append(f"Relative standing        : more stable than {sp:.0f}% of the "
             f"{control_name or 'reference'} group")
    L.append(f"Interpretation           : {_band(sp)}")
    L.append("")
    L.append("This measures how steadily the brain holds and sequences its "
             "momentary states at rest. A low position means ordered, predictable "
             "state-switching (the stable extreme, anchored by the monk); a high "
             "position means restless, less predictable switching (the ADHD pole).")
    if f:
        L.append("")
        L.append(f"For this recording specifically: the brain changes state "
                 f"{_switch_phrase(f.get('switch_rate', 0))} "
                 f"({f.get('switch_rate', 0):.1f}/s), and its switching is "
                 f"{_entropy_phrase(f.get('transition_entropy', 0))} "
                 f"(transition entropy {f.get('transition_entropy', 0):.2f}). "
                 f"That places it in the {_band(sp)} band relative to the "
                 f"{control_name or 'reference'} group.")

    # ---- Dynamics ---------------------------------------------------------
    if f:
        L.append(_rule("2. STATE-SWITCHING DYNAMICS"))
        L.append(f"Transition entropy : {f.get('transition_entropy', float('nan')):.3f}   "
                 "(predictability of switching; LOW = ordered, the stable signature)")
        L.append(f"Switch rate        : {f.get('switch_rate', float('nan')):.2f} /s   "
                 "(how often the brain changes state per second)")
        durs = [f.get(f"duration_{m}", 0.0) for m in range(4)]
        if any(durs):
            L.append(f"Mean dwell time    : {sum(durs)/len(durs):.0f} ms per state "
                     "(physiological range ~60–120 ms)")
        L.append("")
        L.append("Transition entropy is the core feature: an orderly, repeating "
                 "trajectory through states scores low, while near-random jumping "
                 "scores high. Note the monk switches states at a normal-to-high "
                 "RATE but with exceptionally LOW entropy — his stability is about "
                 "order, not stillness.")

    # ---- Microstate maps --------------------------------------------------
    if maps:
        L.append(_rule("3. MICROSTATE TEMPLATE MAPS"))
        L.append("The brain cycles through a few recurring scalp-voltage patterns "
                 "(microstates). The four below are computed from THIS recording; "
                 "what is scored is how the brain transitions among them, not which "
                 "it contains. Colour polarity is arbitrary.")
        if any(d.get("coverage") is not None for d in maps):
            dom = max(maps, key=lambda d: d.get("coverage") or 0)
            L.append("")
            L.append(f"This recording dwelt most in Map {dom['index']} "
                     f"(over the {dom['region']}, {(dom.get('coverage') or 0)*100:.0f}% "
                     "of the time).")
        L.append("")
        for d in maps:
            m = d["index"]
            cov = d.get("coverage")
            dur = d.get("dwell")
            L.append(f"Map {m} — strongest over the {d['region']} ({d['axis']}).")
            L.append(f"   What it is: {d['meaning']}")
            L.append(f"   Network: {d['assoc']}.")
            if cov is not None and dur is not None:
                L.append(f"   This recording spends {cov*100:.0f}% of the time in "
                         f"Map {m}, about {dur:.0f} ms per visit.")
            L.append("")
        L.append("The network labels are tentative associations from the EEG "
                 "literature, not proven for these specific maps.")

    # ---- Spectral profile -------------------------------------------------
    if f and all(f"rel_{b}" in f for b in _BANDS):
        L.append(_rule("4. SPECTRAL PROFILE (relative band power)"))
        for b in _BANDS:
            L.append(f"  {b:<6} {f['rel_'+b]*100:5.1f}%   — {_BAND_DESC[b]}")
        top = max(_BANDS, key=lambda b: f["rel_" + b])
        L.append("")
        L.append(f"This recording is dominated by {top} activity "
                 f"({f['rel_' + top] * 100:.0f}%). " + _alpha_line(f["rel_alpha"]))
        if "theta_beta_ratio" in f:
            L.append("")
            L.append(f"Theta/beta ratio        : {f['theta_beta_ratio']:.2f}")
        if "frontal_theta_beta" in f:
            L.append(f"Frontal theta/beta ratio: {f['frontal_theta_beta']:.2f}")
        L.append("")
        L.append("The theta/beta ratio is the classic ADHD spectral marker (the "
                 "basis of the discontinued NEBA system). It is reported for "
                 "transparency only: in this model's validation, spectral power — "
                 "including theta/beta — did NOT separate ADHD from controls, which "
                 "is why scoring is based on switching dynamics instead.")

    # ---- Exploratory subtype ---------------------------------------------
    if subtype:
        L.append(_rule("5. ADHD-PRESENTATION LIKENESS (EXPLORATORY)"))
        L.append("NOT a diagnosis or a validated classifier. No subtype-labelled "
                 "training data exists; this is a literature-based heuristic for "
                 "hypothesis generation only.")
        L.append("")
        pretty = {"inattentive": "Inattentive",
                  "hyperactive_impulsive": "Hyperactive-impulsive",
                  "combined": "Combined"}
        for key, d in subtype.items():
            L.append(f"  {pretty.get(key, key):<22} {d['pct']:5.1f}%   ({d['why']})")
        if sp >= 40:
            L.append("")
            L.append("Note: this recording is in the stable/healthy range, so the "
                     "above breakdown is not clinically applicable.")

    # ---- Methods & confidence --------------------------------------------
    L.append(_rule("6. METHODS AND CONFIDENCE"))
    if adhd_n is not None:
        L.append(f"- Reference: {adhd_n} ADHD vs healthy references across age groups.")
    L.append(f"- Separation: leave-one-subject-out cross-validated AUC {auc_str} "
             "(0.5 = chance). Group-level signal — NOT diagnostic-grade.")
    L.append("- Pipeline: 16 shared 10-20 channels, resampled to 100 Hz, "
             "average-referenced; 4 microstates via polarity-invariant k-means on "
             "global-field-power peaks; logistic stability axis.")
    L.append("- Percentiles are empirical (fraction of the reference group "
             "exceeded), not modelled probabilities.")

    # ---- Limitations ------------------------------------------------------
    L.append(_rule("7. LIMITATIONS"))
    L.append("- The healthy extreme is anchored to a single adult recording (n=1).")
    L.append("- The ADHD reference cohort is pediatric and task-based; adult "
             "recordings carry a developmental/condition confound.")
    L.append("- 16-channel montage with no midline (Fz/Cz/Pz inferred).")
    L.append("- ADHD subtype cannot be detected, only explored (no labelled data).")

    # ---- Disclaimer -------------------------------------------------------
    L.append(_rule("8. DISCLAIMER"))
    L.append("This report is DECISION SUPPORT, not a medical diagnosis. It "
             "describes brain-state dynamics relative to reference populations and "
             "must be interpreted by a qualified clinician alongside full clinical "
             "history. It does not, by itself, diagnose ADHD or any condition.")
    return "\n".join(L)
