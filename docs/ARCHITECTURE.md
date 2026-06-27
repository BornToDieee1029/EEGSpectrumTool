# Technical Architecture — EEG Microstate Spectrum Tool

> Working name: **neuro-spectrum** (not finalized)
> Status: Design draft v0.1 — 2026-06-27

---

## 0. What this document decides

This is a design-before-code plan. It commits to: the data harmonization
strategy, the microstate method, the feature set, the spectrum-scoring model,
and the honest statistical limits of the v1 claim. Every section ends with an
**Open decision** where a real choice still has to be made.

---

## 1. The scientific claim, stated precisely

We are NOT claiming "this tool diagnoses ADHD." We are claiming:

> Given a resting-state EEG, the tool places the recording on a one-dimensional
> **state-stability axis** derived from microstate dynamics, with the monk's
> empty-mind recording near the high-stability pole and the ADHD group's
> distribution near the low-stability pole.

This framing matters because it is *defensible* and *falsifiable*. NEBA failed
clinically because it reduced the brain to one ratio (theta/beta) and claimed
diagnostic power it didn't have. We avoid both traps: we use a richer feature
set AND we make a weaker, more honest claim (placement on an axis, decision
*support*, not a diagnosis).

**Open decision:** Is the output one axis (stability) or two (e.g. stability ×
syntax-regularity)? One axis is cleaner to communicate; two captures more of the
biology. Recommend: compute several axes internally, surface one primary axis in
v1, expose the rest as a "detail" panel.

---

## 2. The central problem: cross-dataset harmonization

Microstate analysis clusters scalp **voltage topographies** — the spatial
*pattern* of voltage across electrodes at each instant. A topography is only
defined relative to a montage. Therefore any difference in electrode layout
becomes a confound that masquerades as a difference in brain dynamics.

| Source              | Channels | Sampling | n   | Role                  |
|---------------------|----------|----------|-----|-----------------------|
| Monk (yours)        | 16       | 125 Hz   | 1   | High-stability anchor |
| ADHD (Kaggle)       | 19       | 128 Hz   | 121 | Clinical reference    |
| Microstate ref      | (varies) | (varies) | —   | Template validation   |

### Strategy (in order of preference)

1. **Common electrode subset.** Find the intersection of the monk's 16 and the
   Kaggle 19 (likely the standard 10-20 set: Fp1/2, F3/4/7/8, Fz, C3/4, Cz,
   P3/4, T-row, O1/2). Run *all* analysis on the shared subset. This is the
   most defensible because no data is invented. The cost is throwing away
   channels — acceptable, since microstate maps are low spatial frequency and
   robust to modest channel reduction.
2. **Spherical-spline interpolation to a common montage** (MNE
   `interpolate_bads` / `set_montage` + interpolation). Reconstructs the missing
   electrodes onto a shared layout. More channels retained, but interpolated
   channels are not independent measurements — flag them and never let them
   dominate the GFP peak selection.
3. **Resample time to a common rate.** 125 → 128 Hz (or both → 100 Hz) via MNE
   `raw.resample`. Microstate durations are reported in ms, so the rate must be
   identical across datasets or durations are not comparable. This is cheap and
   non-negotiable.

**Recommendation for v1:** Strategy 1 (shared subset) + Strategy 3 (resample to
100 Hz). Simplest, most honest, fewest confounds. Revisit interpolation only if
the shared subset proves too small (<12 channels) for stable clustering.

**Open decision:** confirm the actual electrode names in each file before
committing. This must be verified against the real headers, not assumed.

---

## 3. Pipeline architecture

```
                      ┌─────────────────────────────────────────┐
  raw EEG file  ──►   │  io.load_eeg()                          │
  (.edf/.set/...)     │  format detection, montage attach        │
                      └───────────────┬─────────────────────────┘
                                      ▼
                      ┌─────────────────────────────────────────┐
                      │  harmonize.to_common_space()             │
                      │  channel subset → resample → reref(avg)  │
                      └───────────────┬─────────────────────────┘
                                      ▼
                      ┌─────────────────────────────────────────┐
                      │  clean.preprocess()                      │
                      │  bandpass 1–40Hz, notch, bad-ch detect,  │
                      │  ICA/ASR artifact removal, epoch          │
                      └───────────────┬─────────────────────────┘
                                      ▼
                      ┌─────────────────────────────────────────┐
                      │  microstates.fit() / .backfit()          │
                      │  GFP-peak extraction → modified k-means   │
                      │  → 4–7 maps → backfit full recording      │
                      └───────────────┬─────────────────────────┘
                                      ▼
                      ┌─────────────────────────────────────────┐
                      │  features.extract()                      │
                      │  duration, coverage, occurrence,         │
                      │  transition matrix, entropy, GEV          │
                      └───────────────┬─────────────────────────┘
                                      ▼
                      ┌─────────────────────────────────────────┐
                      │  score.place_on_spectrum()               │
                      │  vs monk anchor + HC + ADHD reference     │
                      └───────────────┬─────────────────────────┘
                                      ▼
                      ┌─────────────────────────────────────────┐
                      │  report / Streamlit UI                   │
                      │  plain-language placement + detail panel  │
                      └─────────────────────────────────────────┘
```

Each box is one module with a typed function boundary, so each can be unit-
tested on synthetic data independently.

---

## 4. Preprocessing (`clean`)

Standard resting-state microstate prep, in this order:
- Bandpass **1–40 Hz** (microstate literature standard; some use 2–20 Hz —
  this is a tunable, default 1–40).
- Notch at line frequency (50/60 Hz — detect from data or config).
- Average reference — **required** for microstates (topographies are reference-
  dependent; the field standardizes on average reference).
- Bad-channel detection (flat / extreme-variance / correlation-based).
- Artifact removal: **ICA** for ocular/muscle, or **ASR** for shorter
  recordings. ICA needs enough data; for short clips ASR is safer.

**Determinism requirement:** every random-seeded step (ICA, k-means) must accept
a fixed seed. A clinical-support tool that gives different answers on re-run is
worthless. Seed everything, log the seed in the report.

**Open decision:** ICA vs ASR as the v1 default. Recommend ICA when recording
> 2 min, ASR fallback otherwise. Need to know the monk recording's length.

---

## 5. Microstate analysis (`microstates`)

The method, concretely:
1. Compute **Global Field Power (GFP)** = spatial standard deviation across
   channels at each time point. Topography is most stable and informative at GFP
   peaks.
2. Extract topographies **at GFP peaks only** (reduces data, noise, polarity
   issues).
3. Cluster with **modified k-means** (polarity-invariant — EEG topography sign
   flips with the underlying dipole and must be ignored). This is THE microstate
   algorithm; ordinary k-means is wrong here. Alternative: AAHC (atomize-and-
   agglomerate hierarchical clustering).
4. Choose **k**. Classic literature uses **4** canonical maps (A/B/C/D). Recent
   work uses 5–7. Select k by **Global Explained Variance (GEV)** + cross-
   validation criterion, but for cross-subject comparison k must be FIXED across
   all subjects. Recommend k=4 for v1 (comparability + literature anchor), test
   k=5..7 in validation.
5. **Group-level template maps.** For comparison to work, every subject must be
   labeled with the *same* set of maps. So: fit maps per subject, then compute a
   group template (e.g. mean maps across the ADHD reference cohort), then
   **backfit** every recording — including the monk and any new patient — onto
   that shared template. Individual-fit maps are not comparable across people;
   shared-template backfitting is the standard solution.

Library: `mne-microstates` or `pycrostates` (the latter is MNE-endorsed, better
maintained — recommend **pycrostates**).

**Open decision:** k=4 vs data-driven k. Recommend fixed k=4 for v1.

---

## 6. Feature set (`features`)

Per recording, after backfitting to shared templates:

**Static (per-map):**
- `duration[m]` — mean ms the brain dwells in map m (the stability signal).
- `coverage[m]` — fraction of total time in map m.
- `occurrence[m]` — times per second map m appears.
- `gev` — global explained variance (fit quality / map sharpness).

**Dynamic (the part NEBA never measured):**
- `transition_matrix[m→n]` — probability of switching from map m to n.
- `transition_entropy` — Shannon entropy of the transition distribution. LOW
  entropy = predictable, ordered switching (monk hypothesis). HIGH entropy =
  disordered, near-random switching (ADHD hypothesis).
- `switch_rate` — total transitions per second.
- `markov_test` — does the sequence violate Markov order-0/1/2? Non-randomness
  IS the signal; quantify it.

**The core hypothesis in feature terms:**
- Monk → long `duration`, low `switch_rate`, low `transition_entropy`.
- ADHD → short `duration`, high `switch_rate`, high `transition_entropy`.

This is testable on the Kaggle cohort *before* the monk is even involved: if
ADHD vs HC don't separate on these features, the whole premise needs revisiting.
**Build this validation as gate #1.**

---

## 7. Spectrum scoring (`score`)

### The honest version (v1)
A single recording is placed by comparing its feature vector to the reference
distributions:
- Fit a **z-scored composite** along the axis defined by the HC→ADHD contrast
  (e.g. LDA or logistic-regression direction trained on the 121-subject cohort),
  then project the new recording onto that axis.
- The monk sits as a **labeled landmark** on the same axis, NOT as a trained
  class. With n=1 you cannot fit a "monk distribution." He is one annotated dot
  at the healthy-extreme end. Say this explicitly in the report.
- Output: percentile/position on the axis + confidence interval that honestly
  reflects the reference cohort size.

### What you must NOT do
- Do not train a classifier with the monk as a class (n=1).
- Do not present the axis position as a probability of having ADHD.
- Do not hide the montage-harmonization caveat from the clinical user.

**Open decision:** axis from LDA vs logistic regression vs a simple
distance-to-centroid. Recommend logistic regression (gives a calibrated-ish
score + interpretable coefficients) with leave-one-subject-out CV reported.

---

## 8. Statistical honesty (read this twice)

- **n=1 monk.** A single extraordinary recording is an existence proof and a
  landmark, not a reference distribution. It anchors the *narrative* and the
  *display*, and motivates the stability axis. It cannot carry statistical
  weight. This is the project's biggest scientific exposure — own it in the
  writeup rather than letting a reviewer find it.
- **n=121 ADHD/HC.** Enough for group-level feature contrasts with proper CV,
  not enough to claim individual diagnostic accuracy. Report leave-one-subject-
  out performance, never train-set performance.
- **Generalization.** Models trained on children's EEG (Kaggle is pediatric) do
  not transfer cleanly to adults. The monk is presumably an adult. Flag the
  developmental confound explicitly.
- **Reproducibility.** Fixed seeds, logged versions, deterministic output.

---

## 9. Module / repo layout

```
neuro-spectrum/
├── docs/ARCHITECTURE.md          # this file
├── pyproject.toml                # deps, pinned
├── src/neuro_spectrum/
│   ├── io.py                     # load_eeg, format detection
│   ├── harmonize.py              # channel subset, resample, reref
│   ├── clean.py                  # preprocessing
│   ├── microstates.py            # GFP, clustering, backfit
│   ├── features.py               # static + dynamic features
│   ├── score.py                  # spectrum placement
│   ├── report.py                 # plain-language output
│   └── config.py                 # all tunables in one place
├── app/streamlit_app.py          # UI
├── tests/                        # synthetic-data unit tests per module
└── notebooks/                    # exploratory validation (gate #1 etc.)
```

---

## 10. Build order (milestones)

1. **M0 — Data reconnaissance.** Load each real file, print channels, rate,
   duration, units. Resolve every "Open decision" marked *verify against the
   real header*. No analysis yet. (Cheap, unblocks everything.)
2. **M1 — Harmonization module + tests** on synthetic data.
3. **M2 — Clean + microstate pipeline** producing 4 maps from one ADHD subject.
4. **M3 — Feature extraction** + **Gate #1**: do ADHD vs HC separate on dynamic
   features across the cohort? If no → stop and rethink before building UI.
5. **M4 — Scoring axis** with leave-one-subject-out CV.
6. **M5 — Place the monk** on the axis; sanity-check he lands at the stable end.
7. **M6 — Streamlit UI + plain-language report.**
8. **M7 — Reproducibility pass**, write up limitations section.

Gate #1 (M3) is the moment of truth. Everything after it is engineering;
everything up to it is hypothesis testing. Don't build the UI before the gate.

---

## 11. Immediate next action

Run M0 — point the loader at the three real datasets and print their true
headers. Half the "Open decisions" above resolve the instant we see real channel
names and recording lengths.
