# EEG Spectrum Tool

Places a resting-state EEG on a **state-stability axis** derived from microstate
dynamics — from extraordinary stability (anchored by a Zen monk's empty-mind
recording) to the constant state-switching characteristic of ADHD. A decision-
**support** tool for researchers and clinicians, not a diagnosis.

Full design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data]"
```

## Build status (milestones)

| #  | Milestone                                   | State |
|----|---------------------------------------------|-------|
| M0 | Data reconnaissance (real headers)          | ✅ done |
| M1 | Harmonization + tests                       | ✅ done |
| M2 | Clean + microstate pipeline                 | ✅ done |
| M3 | Features + **Gate #1** (ADHD vs HC separate?) | ✅ **PASSED** (LOO-CV AUC 0.70) |
| M4 | Scoring axis (leave-one-subject-out CV)     | ✅ done |
| M5 | Place the monk on the axis                  | ✅ **stable extreme** (>96.7% of controls) |
| M6 | Streamlit UI + plain-language report        | ✅ done |
| M7 | Reproducibility + limitations writeup       | ✅ done |

**Rule:** don't build the UI (M6) before Gate #1 (M3) passes. Everything up to
the gate is hypothesis-testing; everything after is engineering.

## Run the app

```bash
python scripts/train_model.py            # one-time: builds models/spectrum_model.joblib
streamlit run app/streamlit_app.py       # upload an EEG -> spectrum placement
```

## M0 — run this once you have data

```bash
# Download the ADHD dataset (set the Kaggle slug in the script first):
python scripts/download_adhd.py
# Inspect the true headers of whatever you have:
python scripts/m0_reconnaissance.py data/raw/*.edf path/to/monk.edf
```
