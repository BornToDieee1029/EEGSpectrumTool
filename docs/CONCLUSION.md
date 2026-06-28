# The EEG State-Stability Spectrum — Project Conclusion

*A plain-language overview for anyone getting to know this project.*

---

## The one-sentence version

This is a software tool that reads a resting-state EEG recording and places the
brain on a **stability spectrum** — from the extraordinary, ordered calm of a
Zen monk's empty mind at one end, to the restless state-switching characteristic
of ADHD at the other — to give clinicians and researchers an objective,
reproducible measurement that *supports* (never replaces) their judgment.

## The idea

Two people can look identical from the outside — eyes closed, sitting still —
while their brains do completely different things. The brain constantly jumps
between a handful of recurring electrical "microstates." **How** it jumps —
how long it holds each state, how often it switches, and how *predictably* it
moves between them — turns out to be a meaningful signal of mental stability.

We anchor the two ends of the spectrum to real data:

- **The healthy extreme — a Zen monk.** A 16-channel recording of a trained monk
  (一音禅师) in deep "empty mind" meditation, captured in the Anhui mountains.
  His brain shows one of the most ordered state-switching patterns measurable in
  a conscious adult.
- **The clinical pole — ADHD.** A public dataset of 121 children (61 ADHD, 60
  control). The ADHD brain does the opposite of the monk: it switches states
  more erratically and less predictably.

## Why it's different from what came before

The only FDA-cleared EEG tool for ADHD (NEBA) relied on a **single spectral
ratio** (theta/beta) and ultimately failed. This project's bet was that the
*dynamics* of brain-state switching carry more signal than any single power
ratio. **We tested that directly and it held:** in our own data the theta/beta
ratio and other spectral features performed *below chance* for separating ADHD
from controls, while the microstate-dynamics features carried real signal. We
reproduced NEBA's failure and stepped past it.

## What the tool actually does

```
raw EEG  →  harmonize  →  clean  →  microstates  →  features  →  score  →  report
            (shared 16   (filter)   (4 maps,        (entropy,    (place on
             channels,              group template,  switching,   the axis)
             100 Hz)               backfit)          band power)
```

Upload one recording, pick an age range, and get back: a position on the
stability spectrum, how the recording compares to an age-matched reference
population, per-wavelength spectral data, the microstate maps, and a
plain-language report.

## What we found (the honest scoreboard)

- **The premise is real.** ADHD and controls separate on microstate dynamics
  with a leave-one-subject-out cross-validated AUC of ~0.70 — group-level signal,
  not diagnostic-grade.
- **The monk anchors the stable extreme** — more stable than ~95% of healthy
  *adults* (not just children), and the result survives length-matching.
- **Stability means *order*, not *stillness*.** The monk doesn't switch states
  less often — he switches as much as anyone, but his *sequence* of switching is
  dramatically more predictable (low transition entropy). That is the real
  signal the tool measures.
- **Spectral features don't help** — confirming the project's founding argument
  against single-ratio approaches.

## What it deliberately does NOT claim

- **It is not a diagnosis.** It is decision support. Clinical ADHD diagnosis
  depends on symptom history across settings, not EEG alone.
- **It cannot detect ADHD subtypes.** There is no subtype-labeled training data,
  so the subtype breakdown shown in the app is an *exploratory*, literature-based
  heuristic for hypothesis generation — clearly flagged as such.
- **The monk is n=1** — a powerful landmark and existence proof, not a
  statistical class.
- **Reference cohorts are imperfect** — pediatric for the ADHD contrast; the tool
  compares against age-matched healthy references (children, adults, older
  adults) but cannot finely age- or gender-adjust beyond that.

Full detail in [`LIMITATIONS.md`](LIMITATIONS.md).

## Where it's going

The deeper vision behind the monk data is a **closed-loop neurofeedback system**
— using the monk's empty-mind signature as a real-time target, so people with
insomnia, anxiety, or chronic stress could *train toward* that ordered, calm
state without decades in a monastery. This diagnostic spectrum tool is the
foundation: it proves the signature is measurable and where ordinary brains sit
relative to it.

## How to run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data]"
python scripts/train_model.py          # builds the model + reference table
streamlit run app/streamlit_app.py     # upload an EEG, see the spectrum
```

## The bottom line

Two people sit still with their eyes closed. You cannot tell them apart by
looking. But their brains move through their internal states in profoundly
different ways — one ordered and calm, one restless and unpredictable. This tool
measures that difference, anchors it to a human extreme of mental stability, and
turns it into a single, honest, reproducible number a clinician can act on.
