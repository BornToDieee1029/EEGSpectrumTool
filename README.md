# EEG Spectrum Tool

Places a resting-state EEG on a **state-stability axis** derived from microstate
dynamics — from extraordinary stability (anchored by a Zen monk's empty-mind
recording) to the constant state-switching characteristic of ADHD. A decision-
**support** tool for researchers and clinicians, not a diagnosis.

Full design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · limitations:
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Scope and claims (honest)

- **Binary signal is real:** ADHD vs control separate on microstate dynamics
  (leave-one-subject-out CV AUC ~0.70).
- **Spectral features don't help:** theta/beta ratio and band powers score
  *below chance* here and degrade the classifier — directly reproducing why the
  theta/beta-based NEBA tool failed. Microstate dynamics carry the signal.
- **Subtypes (inattentive / hyperactive-impulsive / combined): NOT detectable.**
  The cohort has no subtype labels, and unsupervised clustering finds no
  convincing EEG subgroups (silhouette ~0.21). The honest claim is that the tool
  *explores whether* microstate dynamics differ across ADHD presentations — it
  cannot classify them without subtype-labeled training data.

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
