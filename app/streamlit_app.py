"""EEG Spectrum Tool -- Streamlit UI.

Upload one resting-state EEG; get its position on the state-stability spectrum,
per-band spectral data, microstate maps, and an exploratory ADHD-presentation
estimate. Decision support, not a diagnosis.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import streamlit as st

from eeg_spectrum import io as eeg_io
from eeg_spectrum.microstates import describe_maps
from eeg_spectrum.muse import muse_screen
from eeg_spectrum.pipeline import CORE_FEATURES, load_artifact, process_file
from eeg_spectrum.report import render_text

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "spectrum_model.joblib"
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

# Muted palette (matches the warm paper theme).
PAPER, INK, CLAY = "#FAF9F5", "#1F1E1D", "#CC785C"
NEUTRAL, RED, GREEN, AXIS = "#B8B5AC", "#BE5A4E", "#6E8B7B", "#44423D"

AGE_RANGES = {
    "Child / adolescent (7–18)": "adolescent",
    "Adult (18–59)":             "adult_control",
    "Older adult (60+)":         "older_adult",
}
GROUP_NAMES = {"adolescent": "adolescent reference (community sample)",
               "adult_control": "adult controls", "older_adult": "older-adult controls"}

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": "#D9D6CD", "axes.linewidth": 0.8, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK,
    "text.color": AXIS, "axes.labelcolor": AXIS,
    "xtick.color": "#6B6862", "ytick.color": "#6B6862", "legend.frameon": False,
})

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, button, input, select, textarea {
  font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}
[class*="material-symbols"], [class*="material-icons"],
span[data-testid="stIconMaterial"] {
  font-family: 'Material Symbols Outlined', 'Material Symbols Rounded',
               'Material Icons' !important;
}
#MainMenu, header, footer {visibility: hidden;}
[data-testid="stToolbar"], [data-testid="stDecoration"] {display: none;}
.block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem;
       padding-left: 3rem; padding-right: 3rem;}
h1, h2, h3 {font-family: 'Inter', -apple-system, sans-serif;
            font-weight: 600; letter-spacing: -0.018em; color: #1F1E1D;}
h2 {font-size: 19px; margin-top: 2.4rem; text-wrap: balance;}
.apptitle {font-size: 27px; margin: 0 0 0.55rem; line-height: 1.2; text-wrap: balance;}
.lead {color: #57544E; font-size: 15px; margin-bottom: 1.4rem; line-height: 1.6;
       max-width: 68ch; text-wrap: pretty;}
.headrule {border: none; border-top: 1px solid #E4E1D8; margin: 0.2rem 0 2rem;}
.footer {color: #5E5B54; font-size: 12px; line-height: 1.65; margin-top: 2.5rem;
         padding-top: 1rem; border-top: 1px solid #E4E1D8; max-width: 72ch;}
.step {color: #44423D; font-size: 13px; font-weight: 500; margin: 2.2rem 0 0.5rem;}
[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #E4E1D8;
       border-radius: 12px; padding: 14px 16px;}
[data-testid="stMetricValue"] {font-size: 23px; font-weight: 500; color: #1F1E1D;}
[data-testid="stMetricLabel"] p {font-size: 13px; color: #57544E;}
.note {border: 1px solid #E7DBD2; background: #F7F1EC; border-radius: 8px;
       padding: 0.7rem 0.95rem; font-size: 13px; color: #54504A;
       margin: 0.3rem 0 1.2rem; line-height: 1.55;}
.reqs {background: #FFFFFF; border: 1px solid #E4E1D8; border-radius: 12px;
       padding: 0.95rem 1.15rem; margin-bottom: 1rem; font-size: 13.5px;
       color: #3D3B36;}
.reqs b {color: #1F1E1D; font-weight: 500;}
.reqs ul {margin: 0.5rem 0 0; padding-left: 1.15rem;}
.reqs li {margin: 0.3rem 0; line-height: 1.5;}
.reqs .no {color: #5E5B54; font-size: 12.5px; margin-top: 0.6rem;
       border-top: 1px solid #EFEDE6; padding-top: 0.55rem;}
.explain {color: #57544E; font-size: 13px; line-height: 1.6;
       margin: 0.2rem 0 1.5rem; max-width: 68ch; text-wrap: pretty;}
.explain b {color: #3D3B36; font-weight: 500;}
.explain .muted {color: #6E6B63;}
.caption {color: #57544E; font-size: 12.5px; line-height: 1.55;
       margin: 0.55rem 0 1.7rem; max-width: 82ch; text-wrap: pretty;
       border-top: 1px solid #ECE9E1; padding-top: 0.55rem;}
.caption b {color: #1F1E1D; font-weight: 600;}
.caption .lbl {color: #1F1E1D; font-weight: 600;}
.caption .muted {color: #6E6B63;}
hr {border-color: #E4E1D8;}
[data-testid="stFileUploader"] {background: #FFFFFF; border: 1px solid #E4E1D8;
       border-radius: 12px; padding: 0.5rem;}
.stDataFrame {border-radius: 10px;}
:focus-visible {outline: 2px solid #CC785C; outline-offset: 2px;}
@media (prefers-reduced-motion: reduce) {
  * {animation-duration: 0.01ms !important; transition-duration: 0.01ms !important;}
}
</style>
"""


@st.cache_resource
def _artifact():
    return load_artifact(MODEL)


def _note(text):
    st.markdown(f"<div class='note'>{text}</div>", unsafe_allow_html=True)


def _step(text):
    st.markdown(f"<div class='step'>{text}</div>", unsafe_allow_html=True)


def _explain(text):
    st.markdown(f"<div class='explain'>{text}</div>", unsafe_allow_html=True)


def _caption(label, text):
    """Research-paper style caption placed directly under a figure/output."""
    st.markdown(f"<div class='caption'><span class='lbl'>{label}.</span> {text}</div>",
                unsafe_allow_html=True)


def _project_group(art, sub_df):
    return art.axis.model.decision_function(sub_df[CORE_FEATURES].to_numpy())


def _spectrum_fig(art, projection):
    ref = np.concatenate([art.axis.hc_positions, art.axis.adhd_positions])
    fig, ax = plt.subplots(figsize=(8.4, 1.9))
    ax.scatter(art.axis.hc_positions, np.zeros_like(art.axis.hc_positions),
               c=NEUTRAL, s=24, alpha=0.7, label="controls")
    ax.scatter(art.axis.adhd_positions, np.zeros_like(art.axis.adhd_positions),
               c=RED, s=24, alpha=0.7, label="ADHD")
    monk_style = {
        "empty mind (無念)": ("monk · empty mind", "#0F6E56", "-"),
        "sutra recitation (念经)": ("monk · meditation", "#5DCAA5", "--"),
    }
    for cond, pos in art.monk_positions.items():
        label, color, ls = monk_style.get(cond, (cond, GREEN, ":"))
        ax.axvline(pos, color=color, lw=2.0, ls=ls, alpha=0.95, label=label)
    ax.axvline(projection, color=CLAY, lw=2.6, label="this recording")
    pad = 0.3 * (ref.max() - ref.min())
    ax.set_xlim(min(ref.min(), projection) - pad, ref.max() + pad)
    ax.set_yticks([])
    ax.set_xlabel("more stable  (left)                              (right)  more ADHD-like")
    ax.legend(loc="upper center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, 1.34),
              columnspacing=1.0, handletextpad=0.5)
    fig.tight_layout()
    return fig


def _band_table(feats, ref):
    rows = []
    for b in BANDS:
        col = f"rel_{b}"
        rows.append({"band": b, "this %": round(100 * feats[col], 1),
                     "child %": round(100 * ref[ref.group == "control"][col].mean(), 1),
                     "ADHD %": round(100 * ref[ref.group == "ADHD"][col].mean(), 1),
                     "adult %": round(100 * ref[ref.group == "adult_control"][col].mean(), 1)})
    return pd.DataFrame(rows)


def _band_fig(table):
    fig, ax = plt.subplots(figsize=(8.4, 3))
    x = np.arange(len(BANDS))
    w = 0.27
    ax.bar(x - w, table["this %"], w, label="this recording", color=CLAY)
    ax.bar(x, table["child %"], w, label="child controls", color=NEUTRAL)
    ax.bar(x + w, table["ADHD %"], w, label="ADHD", color=RED)
    ax.set_xticks(x); ax.set_xticklabels(BANDS)
    ax.set_ylabel("relative power (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _psd_fig(psd):
    psds, freqs = psd.get_data(return_freqs=True)
    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    ax.plot(freqs, 10 * np.log10(psds.mean(axis=0)), color=CLAY, lw=1.6)
    for lo, hi, name in [(1, 4, "δ"), (4, 8, "θ"), (8, 12, "α"), (12, 30, "β"), (30, 40, "γ")]:
        ax.axvspan(lo, hi, alpha=0.05, color=AXIS)
        ax.text((lo + hi) / 2, ax.get_ylim()[1], name, ha="center", va="top", fontsize=9)
    ax.set(xlabel="frequency (Hz)", ylabel="power (dB)")
    fig.tight_layout()
    return fig


def _maps_fig(maps, ch_names):
    info = mne.create_info(list(ch_names), sfreq=100, ch_types="eeg")
    info.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")
    k = maps.shape[0]
    fig, axes = plt.subplots(1, k, figsize=(2.6 * k, 2.8))
    for i, ax in enumerate(np.atleast_1d(axes)):
        mne.viz.plot_topomap(maps[i], info, axes=ax, show=False, contours=4)
        ax.set_title(f"Map {i}", fontsize=11)
    fig.tight_layout()
    return fig


def _dist_fig(feats, control_df, adhd_df, feature, label, control_name):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.hist(control_df[feature], bins=14, alpha=0.7, color=NEUTRAL, label=control_name)
    ax.hist(adhd_df[feature], bins=14, alpha=0.6, color=RED, label="ADHD")
    ax.axvline(feats[feature], color=CLAY, lw=2.4, label="this recording")
    ax.set(title=label, yticks=[])
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def _muse_bands_fig(bands):
    fig, ax = plt.subplots(figsize=(8.4, 3))
    names = BANDS
    ax.bar(names, [bands[b] * 100 for b in names], color=CLAY, width=0.6)
    ax.set_ylabel("relative power (%)")
    fig.tight_layout()
    return fig


def _muse_flow():
    st.markdown("## Muse headband — experimental band-power screen")
    _note("This is a <b>4-channel</b> headband (TP9, AF7, AF8, TP10). It "
          "<b>cannot</b> compute microstate dynamics or place a recording on the "
          "state-stability spectrum — that needs 16+ scalp electrodes. This screen "
          "reports band powers and the theta/beta ratio only: a historically-"
          "studied but <b>weak</b> attention-related marker. It is not a diagnosis, "
          "and not comparable to the monk or ADHD spectrum.")
    st.markdown(
        "<div class='reqs'><b>Muse recording requirements</b>"
        "<ul>"
        "<li><b>Resting, eyes closed, still</b>, 2–5 minutes</li>"
        "<li>Good headband fit — all four sensors making contact</li>"
        "<li>Export a <b>Mind Monitor CSV</b> (raw EEG), or an EDF/FIF containing "
        "TP9, AF7, AF8, TP10</li>"
        "</ul></div>", unsafe_allow_html=True)

    if not hasattr(eeg_io, "load_muse"):
        _note("Muse support is in the code but not loaded in this running "
              "process (a stale-module cache). <b>Fully reboot the app</b> "
              "(Manage app → Reboot, not just rerun) to pick it up.")
        return

    up = st.file_uploader("Muse recording (.csv, .edf, or .fif)",
                          type=["csv", "edf", "fif"], label_visibility="collapsed")
    if up is None:
        _note("Upload a Muse recording to run the band-power screen.")
        return

    with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as tmp:
        tmp.write(up.getbuffer())
        tmp_path = tmp.name
    try:
        with st.spinner("Reading Muse data and computing band powers…"):
            res = muse_screen(eeg_io.load_muse(tmp_path))
    except Exception as exc:  # noqa: BLE001
        _note(f"Could not process this file: {exc}")
        return

    st.markdown("## Result — band-power screen")
    c1, c2, c3 = st.columns(3)
    c1.metric("Theta/beta ratio", f"{res['theta_beta']:.2f}")
    c2.metric("Frontal theta/beta",
              f"{res['frontal_theta_beta']:.2f}" if res['frontal_theta_beta'] else "—")
    c3.metric("Channels", f"{res['n_channels']} (Muse)")

    st.pyplot(_muse_bands_fig(res["bands"]), use_container_width=True)
    _caption(
        "Figure M1",
        "Relative power in each frequency band from the four Muse sensors "
        "(bands sum to ~100%). The <b>theta/beta ratio</b> above is the classic "
        "attention-related EEG marker — elevated values have historically been "
        "associated with inattention, but the marker is weak (it performed at "
        "chance for ADHD in our 16-channel data), so read it as a rough screen, "
        "not evidence.")
    st.pyplot(_psd_fig(res["psd"]), use_container_width=True)
    _caption(
        "Figure M2",
        "Power spectral density from the Muse sensors. A bump near 10 Hz (alpha) "
        "confirms a genuine relaxed, eyes-closed recording.")
    st.markdown(
        "<div class='footer'>Experimental low-density screen. A 4-channel headband "
        "cannot measure microstate dynamics, so this mode does not produce a "
        "state-stability position and is not comparable to the clinical-cap "
        "results. For the full spectrum, use a 16+ channel 10-20 system.</div>",
        unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="EEG Microstate State-Stability Spectrum Model",
                       layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("<h1 class='apptitle'>EEG Microstate State-Stability Spectrum Model</h1>",
                unsafe_allow_html=True)
    st.markdown("<div class='lead'>An objective, reproducible measure of how steadily "
                "the brain holds its momentary states — anchored from the ordered calm "
                "of a Zen monk to the restless switching of ADHD. Decision support for "
                "clinicians and researchers, not a diagnosis.</div>", unsafe_allow_html=True)
    st.markdown("<hr class='headrule'>", unsafe_allow_html=True)

    with st.expander("About this tool — what it is and how it works"):
        st.markdown(
            "**What it is.** You give this website an **EEG** (electroencephalogram — "
            "a recording of the brain's electrical activity, picked up by sensors "
            "resting on the scalp), and it measures how *steady* or *restless* that "
            "brain is compared with other people.\n\n"
            "**The core idea.** Even when you sit still doing nothing, your brain "
            "keeps flickering between a few basic electrical patterns called "
            "**microstates** (brief whole-scalp “snapshots” of activity, each "
            "lasting a fraction of a second). What matters isn't *which* snapshots "
            "appear — everyone has the same few — but *how the brain moves between "
            "them*: in a calm, predictable rhythm, or restlessly jumping around.\n\n"
            "**The two ends of the scale.**\n"
            "- A **Zen monk** in deep meditation has one of the most ordered, "
            "predictable switching patterns ever recorded — the calm anchor at one "
            "end.\n"
            "- People with **ADHD** (attention-deficit/hyperactivity disorder — a "
            "condition where the brain struggles to hold a steady focus) tend to "
            "switch more often and less predictably — the other end.\n"
            "- The tool places any new recording on a **spectrum** (a sliding scale) "
            "between these two.\n\n"
            "**How it works, step by step.**\n"
            "1. **Clean** the signal — filter out noise like muscle twitches and "
            "electrical interference.\n"
            "2. **Harmonize** the data — different machines use different numbers of "
            "sensors and speeds, so every recording is trimmed to the same 16 "
            "standard scalp locations and the same sampling rate, making "
            "comparisons fair.\n"
            "3. **Find the microstates** by **clustering** (grouping similar "
            "snapshots together), then measure how long each is held and how "
            "predictably the brain switches — using **transition entropy** (how "
            "orderly vs. random the switching is; low = orderly) and **switch "
            "rate** (switches per second).\n"
            "4. **Compare** those numbers with hundreds of reference recordings "
            "(children with and without ADHD, plus healthy adolescents, adults, and "
            "older adults) and with the monk, then place the recording on the "
            "spectrum.\n\n"
            "**What you get.** A position on the spectrum, how the person compares "
            "with others their age, charts of their brain-wave bands (delta, theta, "
            "alpha, beta, gamma — rhythms at different speeds), the microstate maps, "
            "and a plain-language report.\n\n"
            "**What it's used for.** It is a **decision-support** tool — it gives "
            "doctors and researchers one extra objective measurement to consider. "
            "It does **not** diagnose anyone: a real ADHD diagnosis requires a "
            "clinician reviewing a person's history and behavior across settings, "
            "not an EEG alone.")

    if not MODEL.exists():
        _note("Model not found. Run <code>python scripts/train_model.py</code> first.")
        return
    art = _artifact()

    _step("Recording device")
    device = st.selectbox(
        "Device",
        ["Clinical / research cap (16+ channels) — full spectrum",
         "Muse headband (4 channels) — experimental band-power screen"],
        label_visibility="collapsed")
    if device.startswith("Muse"):
        _muse_flow()
        return

    _step("Step 1 · Patient age range")
    age_label = st.selectbox("Age range", ["— select —", *AGE_RANGES.keys()],
                             label_visibility="collapsed")
    if age_label == "— select —":
        _note("Select an age range to begin. The recording is compared against an "
              "age-matched healthy reference population.")
        return
    ref_group = AGE_RANGES[age_label]
    control_df = art.reference[art.reference.group == ref_group]
    control_name = GROUP_NAMES[ref_group]
    if len(control_df) == 0:
        _note(f"No reference data is available for {control_name} yet. Retrain the "
              f"model to populate this group, or choose another age range.")
        return

    _step("Step 2 · Upload EEG")
    st.markdown(
        "<div class='reqs'><b>Recording requirements</b> — the recording must match "
        "the resting-state references, or the result is not valid:"
        "<ul>"
        "<li><b>Resting and awake</b> — relaxed, doing nothing (no task, talking, "
        "reading, or phone)</li>"
        "<li><b>Seated and still</b> — minimal movement, no jaw clenching</li>"
        "<li><b>Eyes closed</b>, alert (not drowsy or asleep)</li>"
        "<li><b>2–5 minutes</b> in length</li>"
        "<li>Standard <b>10-20 montage</b> including the 16 shared channels, "
        "sampled at ≥100 Hz</li>"
        "</ul>"
        "<div class='no'>Do not upload task, movement, sleep, or stimulus-locked "
        "recordings.</div></div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div class='reqs'><b>Equipment requirements</b> — the device must capture "
        "scalp topography, not a single signal:"
        "<ul>"
        "<li><b>≥16-channel scalp EEG</b> at standard 10-20 positions, including "
        "the 16 used here (Fp1/2, F3/4/7/8, C3/4, T7/8, P3/4, P7/8, O1/2)</li>"
        "<li>Research- or clinical-grade system — e.g. a 19-channel clinical rig "
        "or OpenBCI Cyton+Daisy. <b>Not</b> single-channel consumer headbands</li>"
        "<li><b>Reference and ground</b> electrodes (e.g. earlobes/mastoids); "
        "wet/gel, saline, or validated dry electrodes at low impedance</li>"
        "<li>Sampling rate <b>≥100 Hz</b> (≥250 Hz recommended)</li>"
        "</ul></div>",
        unsafe_allow_html=True)
    up = st.file_uploader(
        "Resting-state EEG (.edf, .set, .vhdr, .fif, .bdf, .cnt, or OpenBCI .txt)",
        type=["edf", "set", "vhdr", "fif", "bdf", "cnt", "txt"],
        label_visibility="collapsed")
    if up is None:
        _note("Upload a resting-state recording to place it on the spectrum.")
        return

    with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as tmp:
        tmp.write(up.getbuffer())
        tmp_path = tmp.name
    try:
        with st.spinner("Harmonizing, cleaning, and analyzing…"):
            out = process_file(tmp_path, art)
    except Exception as exc:  # noqa: BLE001
        _note(f"Could not process this file: {exc}")
        return

    feats, placement = out["features"], out["placement"]
    placement["cv_auc"] = art.axis.cv_auc
    ref, adhd_df = art.reference, art.reference[art.reference.ADHD == 1]
    proj = placement["projection"]
    stable_vs = 100.0 * float((_project_group(art, control_df) > proj).mean())
    placement["stability_vs_controls"] = stable_vs

    st.markdown("## Result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Position", f"{placement['position_0_100']:.0f} / 100")
    c2.metric("More stable than", f"{stable_vs:.0f}%")
    c3.metric("Switch rate", f"{feats['switch_rate']:.1f}/s")
    c4.metric("Theta/beta", f"{feats['theta_beta_ratio']:.2f}")
    st.markdown(f"<div class='lead'>{age_label} · compared against {control_name} "
                f"(n={len(control_df)}).</div>", unsafe_allow_html=True)
    _caption(
        "Table 1",
        "Summary metrics for this recording. <b>Position</b> (0–100) locates it on "
        "the stability scale: 0 is the calm, highly-ordered end (the monk), 100 the "
        "restless, ADHD-like end. <b>More stable than</b> is the percentage of "
        "same-age healthy people this recording is calmer than. <b>Switch rate</b> "
        "is state changes per second. <b>Theta/beta</b> is a traditional EEG marker, "
        "shown for transparency though it carries little weight here.")

    st.markdown("## State-stability spectrum")
    st.pyplot(_spectrum_fig(art, proj), use_container_width=True)
    _caption(
        "Figure 1",
        "This is the core result — a single ruler of mental "
        "stability. Every dot is one person from the reference data, placed on the "
        "ruler by <b>how their brain switches between states</b>: calm, ordered "
        "brains sit toward the <b>left</b>, restless ADHD-pattern brains toward the "
        "<b>right</b>. Grey dots are healthy controls; red dots are children "
        "diagnosed with ADHD. The two green lines are the Zen monk — the solid line "
        "is his <b>empty-mind</b> state and the dashed line his <b>active "
        "meditation</b> (sutra recitation); both sit far to the stable left and "
        "anchor the healthy extreme. The <b>orange line is your uploaded "
        "recording</b>. The ruler itself was drawn by the model as the single line "
        "that best separates ADHD from controls, so a position is a relative "
        f"placement — not a diagnosis (cross-validated accuracy AUC "
        f"{art.axis.cv_auc:.2f}, where 0.5 would be a coin-flip).")

    if out.get("subtype"):
        st.markdown("## ADHD-presentation likeness")
        cols = st.columns(3)
        pretty = {"inattentive": "Inattentive",
                  "hyperactive_impulsive": "Hyperactive-impulsive",
                  "combined": "Combined"}
        for col, (key, d) in zip(cols, out["subtype"].items()):
            col.metric(pretty[key], f"{d['pct']:.0f}%")
            col.caption(d["why"])
        _caption(
            "Table 2",
            "Exploratory likeness to the three ADHD presentations (inattentive, "
            "hyperactive-impulsive, combined), estimated by comparing three features "
            "(theta/beta ratio, switch rate, transition entropy) against each "
            "presentation's reported EEG tendencies; the line under each percentage "
            "names the driving features. <b>Not a diagnosis or a validated "
            "classifier</b> — there is no subtype-labelled training data, so this is "
            "a literature-based heuristic for hypothesis generation only. Clinical "
            "subtype depends on symptom history across settings, not EEG.")
        if stable_vs >= 40:
            _note("This recording is in the stable / healthy range, so the "
                  "breakdown above is not clinically applicable.")

    st.markdown("## Spectral power by wavelength")
    table = _band_table(feats, ref)
    st.dataframe(table.set_index("band"), use_container_width=True)
    st.pyplot(_band_fig(table), use_container_width=True)
    _caption(
        "Figure 2",
        "Relative band power — each frequency band's share of the total "
        "(bands sum to ~100%), for this recording versus group averages. "
        "<b>Delta</b> (1–4 Hz) slow waves; <b>theta</b> (4–8 Hz) drowsiness and "
        "memory; <b>alpha</b> (8–12 Hz) the eyes-closed rhythm; <b>beta</b> "
        "(12–30 Hz) active processing; <b>gamma</b> (30–40 Hz) fastest. Note: "
        "band power alone did <b>not</b> separate ADHD from controls here — which "
        "is why the tool scores switching dynamics, not static power.")
    st.pyplot(_psd_fig(out["psd"]), use_container_width=True)
    _caption(
        "Figure 3",
        "Power spectral density — power versus frequency (slow on the left, fast "
        "on the right), averaged across all 16 electrodes; shaded stripes mark "
        "the five bands. A clear bump near 10 Hz (alpha) is a quick visual "
        "confirmation that this is genuine relaxed, eyes-closed resting EEG.")

    st.markdown("## Microstate maps — this recording")
    pmaps, pch = out.get("patient_maps",
                         (art.template.maps, list(art.template.ch_names)))
    pstats = out.get("patient_stats", {
        m: (feats.get(f"coverage_{m}", 0.0), feats.get(f"duration_{m}", 0.0))
        for m in range(pmaps.shape[0])})
    st.pyplot(_maps_fig(pmaps, pch), use_container_width=True)
    maps_desc = describe_maps(pmaps, list(pch))
    for d in maps_desc:
        cov, dwell = pstats.get(d["index"], (0.0, 0.0))
        d["coverage"], d["dwell"] = cov, dwell
    rows = "".join(
        f"<p style='margin:0.45rem 0'><b>Map {d['index']}</b> — strongest over the "
        f"{d['region']}; held {d['coverage'] * 100:.0f}% of the time. "
        f"{d['meaning'][0].upper() + d['meaning'][1:]}</p>"
        for d in maps_desc)
    st.markdown(
        "<div class='caption'><span class='lbl'>Figure 4.</span> The four recurring "
        "scalp-voltage patterns (<b>microstates</b>) <b>computed from this uploaded "
        "recording</b> — they change with every upload. Colour polarity (red/blue) "
        "is arbitrary. The tool scores not which maps appear but how the brain moves "
        "between them (dwell time and switching order)." + rows +
        "<span class='muted'>Network labels are tentative literature associations.</span>"
        "</div>", unsafe_allow_html=True)

    st.markdown(f"## Distribution vs {control_name} and ADHD")
    g1, g2, g3 = st.columns(3)
    g1.pyplot(_dist_fig(feats, control_df, adhd_df, "transition_entropy", "transition entropy", control_name))
    g2.pyplot(_dist_fig(feats, control_df, adhd_df, "switch_rate", "switch rate /s", control_name))
    g3.pyplot(_dist_fig(feats, control_df, adhd_df, "theta_beta_ratio", "theta/beta ratio", control_name))
    _caption(
        "Figure 5",
        "Each panel places one measurement across two reference groups — the "
        f"age-matched {control_name} (blue, n={len(control_df)}) and children with "
        f"ADHD (red, n={len(adhd_df)}) — with the orange line marking where this "
        "recording falls. (a) <b>Transition entropy</b>: predictability of switching "
        "(low = orderly, the monk-like end). (b) <b>Switch rate</b>: state changes "
        "per second. (c) <b>Theta/beta ratio</b>: the traditional ADHD marker, shown "
        "so you can see it barely separates the two groups here.")

    with st.expander("Methods and confidence"):
        st.markdown(
            f"- **Reference cohort:** ADHD n={int((ref.ADHD==1).sum())}, "
            f"controls/healthy n={int((ref.ADHD==0).sum())} across age groups.\n"
            f"- **Separation:** leave-one-subject-out cross-validated AUC "
            f"{art.axis.cv_auc:.2f} (0.5 = chance). Group-level signal — **not** "
            "diagnostic-grade.\n"
            "- **Harmonisation:** all recordings reduced to the 16 shared 10-20 "
            "channels and resampled to 100 Hz, average-referenced.\n"
            "- **Reproducible:** fixed random seed; the same input always yields "
            "the same output.\n"
            "- **Percentiles are empirical** — the fraction of the reference group "
            "the recording exceeds, not a modelled probability.")

    st.markdown("## Report")
    report = render_text(
        placement,
        features=feats,
        subtype=out.get("subtype"),
        maps=maps_desc,
        age_label=age_label,
        control_name=control_name,
        control_n=len(control_df),
        adhd_n=len(adhd_df),
        seed=art.microstates.random_seed,
    )
    st.text(report)
    st.download_button("Download report", report, file_name="eeg_spectrum_report.txt")

    st.markdown(
        "<div class='footer'>This tool provides decision support, not a medical "
        "diagnosis. Results describe brain-state dynamics relative to reference "
        "populations and must be interpreted by a qualified clinician alongside "
        "clinical history. Reference data: pediatric ADHD/control cohort, plus "
        "healthy adult, older-adult, and adolescent resting-state references. "
        "Reproducible analysis with a fixed random seed.</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
