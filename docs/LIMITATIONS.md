# Limitations and honest scope

> Read this before citing, deploying, or trusting any number this tool produces.
> Status: 2026-06-27, covering the M0–M6 build.

This tool places a resting-state EEG on a state-stability axis built from
microstate dynamics. It is **decision support, not a diagnosis.** The following
constraints are not caveats to bury — they define what the tool can and cannot
honestly claim.

## 1. The healthy anchor is n=1
The "extraordinary stability" pole is anchored to a single adult monk recording
(two conditions). One recording is an **existence proof and a display landmark**,
not a statistical reference class. We never train a classifier with the monk as
a class, and we never report a monk-derived probability. Validated finding: on
the robust core-dynamics features the monk sits more stable than ~96.7% of the
control cohort, in both conditions — encouraging, but it is one person. The Zen
Brain research's own caveats apply: ICA was done in EEGLAB but frontal Delta may
still include ocular/muscle artifact; session order was not counterbalanced; and
there was no simultaneous control subject to separate monk-specific from
meditation-general effects.

## 1b. Which monk condition is the anchor
The monk recorded two states: **無念 / Empty Mind** (deep quiescence, frontal
Delta 87-89%) and **念经 / Silent Sutra Recitation** (internally active —
Beta/Gamma, working-memory and left-lateralized language network). The healthy-
stability anchor is **Empty Mind** (the "Without 念经" file). Sutra recitation is
a cognitively *active* task state and is shown only as a secondary landmark.

## 1c. This tool measures switching dynamics, not spectral power
The Zen Brain research characterized the monk via band power (Delta/Theta/Alpha/
Beta/Gamma). This tool instead measures microstate *switching* dynamics
(transition entropy, switch rate). These are partly orthogonal: our eval found
Empty Mind and Sutra Recitation roughly equally stable in switching dynamics
(entropy ~1.05 vs ~1.06) despite their large spectral differences. So a "stable"
placement here means ordered state-switching — not necessarily the Delta-
dominant fingerprint the spectral study emphasized.

## 2. Pediatric reference, adult anchor (developmental confound)
The 121-subject ADHD/control cohort is children (ages 7–12). The monk is an
adult. EEG microstate dynamics change with age, so projecting an adult onto a
pediatric-trained axis is extrapolation. Any adult result inherits this
confound and must be read with it in mind.

## 3. Group-level signal, not individual diagnostic accuracy
Leave-one-subject-out CV AUC ≈ 0.70 (full features) / 0.63 (core dynamics). This
is real separation but **far below diagnostic-grade**. It supports a clinician's
judgment; it does not replace it. This is deliberate — NEBA (the one FDA-cleared
EEG ADHD tool) failed precisely by overclaiming from a single thin measure.

## 4. Out-of-distribution placement is fragile on the full feature vector
The full 14-feature logistic axis extrapolates unstably on out-of-distribution
inputs: it threw the two near-identical monk recordings to opposite ends of the
axis. Spectrum placement therefore uses the small, robust core-dynamics axis
(transition entropy + switch rate). New recordings that are themselves far from
the cohort distribution may still be placed unreliably — flag low-confidence
cases rather than trusting the dot.

## 5. Montage harmonization discards information
To compare across datasets we restrict to the 16 electrodes shared by the monk
cap and the ADHD montage. This drops the midline (Fz/Cz/Pz), so microstate map C
(midline-centred) is inferred from surrounding electrodes rather than measured
directly. 16 channels is adequate for microstates but below the 19–30 of richer
studies.

## 6. Minimal artifact cleaning — tested, kept deliberately
Results use bandpass + bad-channel interpolation only, no ICA/ASR. This is not
laziness: ICA was tested on the full cohort and **degraded** separation
(multivariate LOO-CV AUC 0.70 -> 0.64; transition-entropy AUC 0.66 -> 0.53,
losing significance). The cause is instructive — removing frontal components
correlated with Fp1/Fp2 as "ocular" also strips frontal signal, and ADHD is a
frontal disorder; ICA is also unreliable on the 60-140 s clips. So filter-only is
the validated default. The monk recordings still contain visible transient
artifacts (e.g. an electrode-pop step) that are not removed; treat absolute
durations/entropies as provisional.

## 7. Recording-length asymmetry
The monk recordings are ~10 min; ADHD clips average ~2 min (some ~1 min).
Features are rate-normalized, and for the monk eval we crop to the cohort mean
length, but entropy estimates from short clips are noisier than from long ones.

## 8. Fixed methodological choices not yet validated as optimal
k=4 microstates (literature default, not data-selected here); temporal smoothing
parameters chosen to yield physiological ~110 ms durations; logistic axis rather
than alternatives. These are reasonable defaults, not tuned optima.

## What is solid
- Reproducible: fixed seeds throughout, determinism guarded by a test.
- The harmonization is correct (verified: monk + ADHD land in one identical
  16-ch/100 Hz/avg-ref space; post-filter monk shows a clean ~10 Hz alpha peak).
- Gate #1 passed honestly with leave-one-subject-out CV.
- The monk anchor lands where the hypothesis predicted, robustly across both
  conditions.

## Responsible-use line
Do not present any output as a diagnosis, a probability of ADHD, or a substitute
for clinical assessment. Present it as one objective, reproducible measurement of
brain-state stability, with these limitations attached.
