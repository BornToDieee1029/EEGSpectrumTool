# EEG Microstate State-Stability Spectrum Model — Judges' Document

*A complete walkthrough of what we built, why it matters, how we validated it,
and where it's honestly limited.*

---

## The hook

A Zen monk sat perfectly still in the Anhui mountains, and his brain produced one
of the most ordered, stable electrical patterns ever recorded in a conscious
adult. A child with ADHD also sits still — and their brain does the opposite,
flickering restlessly between states, never settling.

We had that one extraordinary monk recording, and we built a tool to answer a
simple question: **where does any other brain sit, relative to it?**

The result is a software tool that reads a resting-state EEG, measures *how
steadily the brain holds and sequences its momentary states*, and places it on a
single spectrum — from the monk's extraordinary stability at one end to the
restless switching characteristic of ADHD at the other. No questionnaires, no
opinions. Just the brain data, analyzed automatically and reproducibly.

It is **decision support, not a diagnosis.** It gives a clinician one more
objective anchor — the way a blood test informs but doesn't replace a doctor's
judgment.

---

## 1. What it is

A doctor or researcher uploads a resting-state EEG file. The tool cleans it,
extracts the recurring brain patterns (microstates), measures the *dynamics* of
how the brain moves between them, compares those numbers to hundreds of reference
recordings and to the monk, and returns a plain-language report with charts
showing where that brain falls on the stability spectrum — compared against an
**age-matched** reference population.

### The two anchors

| | Monk (stable extreme) | ADHD (clinical pole) |
|---|---|---|
| **Brain behavior** | Ordered, predictable state-switching | Restless, less predictable switching |
| **Data** | Own recording, Anhui mountains, 16ch/125Hz, 2 conditions | Kaggle/IEEE, 121 children, 19ch/128Hz |
| **Role** | Healthy-extreme landmark | Clinical reference + axis training |

Healthy controls sit in between and define the normal range. We then added
**healthy adolescent, adult, and older-adult reference populations** so a patient
is always compared to people their own age.

---

## 2. Why it's novel — and why it's *more honest* than NEBA

NEBA was the only FDA-cleared EEG tool for ADHD (cleared 2013, pulled from
clinical guidance ~2016 after the American Academy of Pediatrics found
insufficient evidence). It reduced the brain to **one number** — the theta/beta
power ratio — and claimed diagnostic power it didn't have.

We do two things differently:
1. **We measure the full dynamic pattern**, not one ratio — how long the brain
   holds each state, how often it switches, and *how predictably* (transition
   entropy).
2. **We make a weaker, falsifiable claim** — placement on a spectrum, decision
   support, not a diagnosis.

**And we proved the difference empirically.** We tested the theta/beta ratio and
all spectral band-power features on our cohort directly: they scored **below
chance** (AUC 0.45) and actually *degraded* the classifier. Meanwhile the
microstate dynamics carried real signal. We reproduced exactly why NEBA failed —
in our own data — and stepped past it. That negative result is one of our
strongest pieces of evidence.

---

## 3. How it works (the pipeline)

```
upload → harmonize → clean → microstates → features → score → report
```

1. **Harmonize** — different headsets have different channel counts and sampling
   rates. We reduce every recording to the **16 electrodes shared** across all
   datasets and resample to **100 Hz**. No data invented, no interpolation.
2. **Clean** — bandpass filter (1–40 Hz), bad-channel handling, average
   reference. (We tested ICA artifact removal and it *hurt* separation — see §5 —
   so filter-only is the validated default.)
3. **Microstates** — extract the 4 recurring scalp-voltage maps using
   polarity-invariant modified k-means on global-field-power peaks (via
   `pycrostates`), then backfit every recording onto the shared templates so
   subjects are comparable.
4. **Features** — per recording: dwell time, coverage, switch rate, and
   **transition entropy** (how ordered vs. random the switching is), plus
   spectral band powers for transparency.
5. **Score** — project onto a stability axis (logistic regression trained on the
   121 ADHD/control subjects), report the age-matched percentile.
6. **Report** — an 8-section plain-language report + interactive charts.

---

## 4. The data (5 reference populations, ~238 subjects)

| Population | Source | n | Role |
|---|---|---|---|
| Monk (anchor) | own data | 1 | stable extreme |
| ADHD | Kaggle / IEEE DataPort | 61 | clinical pole + axis |
| Child controls | Kaggle / IEEE DataPort | 60 | axis + pediatric reference |
| Adolescents | OpenNeuro ds005505 (HBN) | ~8 | 7–18 reference |
| Adults | PhysioNet EEGMMIDB | ~80 | adult reference |
| Older adults | OpenNeuro ds004504 | 29 | 60+ reference |

All open datasets, all harmonized to the same 16 channels / 100 Hz / average
reference. Full citations in [`REFERENCES.md`](REFERENCES.md).

---

## 5. What we validated (the honest scoreboard)

- **The premise holds.** ADHD vs. control separate on microstate dynamics with a
  **leave-one-subject-out cross-validated AUC of ~0.70** (0.5 = chance; most
  clinical EEG tools sit 0.70–0.80). Significant on transition entropy
  (p = 0.0025).
- **The monk anchors the stable extreme** — more stable than **~95% of healthy
  adults** (not just children), and the result survives length-matching.
- **Stability means *order*, not *stillness*.** The monk switches states as often
  as anyone, but his *sequence* is far more predictable (low entropy). This
  refined the whole project's definition of stability.
- **Spectral features don't help** (AUC 0.45, below chance) — reproducing NEBA's
  failure and vindicating the dynamics-based approach.
- **ICA cleaning was tested and rejected** — it removed frontal signal (and ADHD
  is a frontal disorder), dropping AUC 0.70 → 0.64. We kept the simpler, better
  method.

Every one of these is a result we can defend — including the negative ones.

---

## 6. The three hard problems (and how we solved them)

1. **Harmonizing the data.** The monk (16ch/125Hz), ADHD (19ch/128Hz), and every
   reference set use different montages. Microstates are *topographic*, so a
   montage mismatch masquerades as a brain difference. We solved it by analyzing
   only the shared 16-channel subset, resampled to a common rate — verified by
   confirming a clean ~10 Hz alpha peak appears after harmonization.
2. **Designing the score.** We collapse 15+ microstate metrics into one readable
   spectrum position via a cross-validated logistic axis — and learned that for
   placing *out-of-distribution* anchors like the monk, a small robust feature
   set (entropy + switch rate) is far more reliable than the full vector, which
   extrapolates unstably.
3. **Handling ambiguity.** Most patients land in the middle. The output reports an
   *empirical percentile* against an age-matched group, states the
   cross-validated AUC, and never converts the position into a probability of
   disorder.

---

## 7. Honest limitations

- **The monk is n=1** — a labeled landmark, never a trained class.
- **The ADHD cohort is pediatric and task-based**; adult recordings carry a
  developmental/condition confound (mitigated by age-matched references, not
  eliminated).
- **AUC ~0.70 is group-level signal, not diagnostic-grade.**
- **ADHD subtypes cannot be detected**, only explored — there is no
  subtype-labeled training data, so the in-app subtype breakdown is a clearly
  flagged literature-based heuristic.
- **16-channel montage with no midline** (Fz/Cz/Pz are inferred).

We surface these in the tool's UI and report, not just in a doc — owning them is
part of the design. Full detail in [`LIMITATIONS.md`](LIMITATIONS.md).

---

## 8. Technology

| Tool | Job |
|---|---|
| MNE-Python | EEG I/O, cleaning, montages, topographies |
| pycrostates | Microstate clustering and backfitting |
| scikit-learn | Stability axis, cross-validation |
| Matplotlib | Charts and microstate maps |
| Streamlit | The web interface clinicians use |

Fully reproducible: every random-seeded step is fixed and logged in the report —
the same input always yields the same output.

---

## 9. Using the tool (the demo)

1. Select the patient's **age range** (the reference group adapts to it).
2. Read the **recording and equipment requirements** (resting, eyes-closed,
   2–5 min, ≥16-channel 10-20 system).
3. **Upload** an EEG file (EDF, EEGLAB `.set`, BrainVision, FIF, or OpenBCI `.txt`
   — EGI 128-channel nets are auto-mapped).
4. Get the **spectrum position**, age-matched percentile, per-wavelength spectral
   data, the four microstate maps (each explained), feature distributions, and a
   downloadable 8-section report.

A live demo of the Streamlit UI beats any slide — lead with it.

---

## 10. Future directions

- **Adult ADHD + subtypes** via the TDBRAIN archive (1,274 patients incl. 271
  ADHD across the lifespan) once its data-use agreement is in place — the path to
  a real, validated subtype classifier and an adult-valid ADHD pole.
- **Closed-loop neurofeedback** — the deeper vision: using the monk's empty-mind
  signature as a real-time target so people with insomnia, anxiety, or chronic
  stress could *train toward* that ordered, calm state (e.g., via generative
  audio feedback). This tool is the foundation that proves the signature is
  measurable. *(Vision, not current capability.)*

---

## 11. Questions we can answer cold

**Why microstates instead of theta/beta ratio?** Microstates capture *temporal
dynamics* — how the brain switches — not a static ratio. We use 8+ features
including transition entropy. And we showed theta/beta scores below chance here.

**What does AUC 0.70 mean?** ADHD vs. healthy controls; chance is 0.50; most
clinical EEG tools sit 0.70–0.80. Meaningful signal, not a diagnostic.

**Why only 4 microstate maps?** The literature standard (canonical A/B/C/D),
which ensures cross-subject comparability and physiological dwell times.

**Different headsets?** We take the common 16-electrode subset and resample to
100 Hz — no data invented, no interpolation.

**Isn't n=1 for the monk a problem?** Yes, and we say so explicitly. He's a
landmark at the stable extreme, not a training class. He anchors the narrative,
not the statistics.

**Pediatric training data on adults?** Doesn't transfer cleanly — so we added
healthy adult and older-adult references and compare age-to-age, and we flag the
residual confound in the UI.

**Could it be gamed?** You'd need to consciously control your resting brain
dynamics for minutes — essentially what years of meditation training produce.
Not realistic clinically.

**Is it reproducible?** Yes — all seeds fixed and logged; same input, same output.

---

## 12. The one-line takeaway

Two people sit still with their eyes closed. You can't tell them apart by looking
— but their brains move through their internal states in profoundly different
ways. We measure that difference, anchor it to a human extreme of mental
stability, and turn it into a single, honest, reproducible number a clinician can
act on.
