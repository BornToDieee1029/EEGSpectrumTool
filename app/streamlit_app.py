"""EEG Spectrum Tool -- Streamlit UI (M6).

Upload one resting-state EEG; get its position on the state-stability spectrum,
anchored by the monk (stable extreme) and the ADHD reference cohort. Decision
support, not a diagnosis.

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
import numpy as np
import streamlit as st

from eeg_spectrum.pipeline import load_artifact, process_file
from eeg_spectrum.report import render_text

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "spectrum_model.joblib"


@st.cache_resource
def _artifact():
    return load_artifact(MODEL)


def _spectrum_fig(art, projection):
    ref = np.concatenate([art.axis.hc_positions, art.axis.adhd_positions])
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.scatter(art.axis.hc_positions, np.zeros_like(art.axis.hc_positions),
               c="#888780", s=30, alpha=0.5, label="controls")
    ax.scatter(art.axis.adhd_positions, np.zeros_like(art.axis.adhd_positions),
               c="#E24B4A", s=30, alpha=0.5, label="ADHD")
    anchor = getattr(art, "anchor", None)
    for cond, pos in art.monk_positions.items():
        is_anchor = cond == anchor
        ax.axvline(pos, color="#0F6E56", lw=2.5 if is_anchor else 1.2,
                   ls="-" if is_anchor else "--", alpha=0.9 if is_anchor else 0.5,
                   label="monk: empty mind (anchor)" if is_anchor else None)
    ax.axvline(projection, color="#042C53", lw=3, label="this recording")
    pad = 0.3 * (ref.max() - ref.min())
    ax.set_xlim(min(ref.min(), projection) - pad, ref.max() + pad)
    ax.set_yticks([])
    ax.set_xlabel("instability axis   (left = stable,   right = ADHD-like)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="EEG Spectrum Tool", layout="centered")
    st.title("EEG state-stability spectrum")
    st.caption("Decision support for researchers and clinicians — not a diagnosis.")

    if not MODEL.exists():
        st.error("Model artifact not found. Run `python scripts/train_model.py` first.")
        return
    art = _artifact()

    up = st.file_uploader(
        "Upload a resting-state EEG (.edf, .set, .vhdr, .fif, or OpenBCI .txt)",
        type=["edf", "set", "vhdr", "fif", "bdf", "cnt", "txt"],
    )
    if up is None:
        st.info("Upload a recording to place it on the spectrum.")
        return

    with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as tmp:
        tmp.write(up.getbuffer())
        tmp_path = tmp.name

    try:
        with st.spinner("Harmonizing, cleaning, and analyzing microstates..."):
            out = process_file(tmp_path, art)
    except Exception as exc:  # noqa: BLE001 -- surface errors to the clinician
        st.error(f"Could not process this file: {exc}")
        return

    placement = out["placement"]
    placement["cv_auc"] = art.axis.cv_auc

    c1, c2, c3 = st.columns(3)
    c1.metric("Position", f"{placement['position_0_100']:.0f} / 100")
    c2.metric("More stable than", f"{placement['stability_vs_controls']:.0f}%",
              help="of healthy controls")
    c3.metric("Switch rate", f"{placement['features_switch']:.1f}/s")

    st.pyplot(_spectrum_fig(art, placement["projection"]))
    st.text(render_text(placement, seed=art.microstates.random_seed))


if __name__ == "__main__":
    main()
