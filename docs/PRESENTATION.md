# Hackathon Presentation — EEG Microstate State-Stability Spectrum

Speaker notes + run-of-show + how the algorithm works + judge Q&A.
Target: ~5 minutes + Q&A. Lead with the monk. End with the live demo.

---

## Run of show (chronological)

**1. Open cold — the monk story (~45s)**
- "I traveled to a monastery in the Anhui mountains and recorded the EEG of a
  Zen monk in deep meditation."
- His empty-mind state produced one of the most ordered, stable brain patterns
  ever recorded in a conscious adult.
- "We had this one extraordinary recording — and built a tool to answer: *where
  does any other brain sit relative to it?*"

**2. The problem (~30s)**
- Two people sit still, eyes closed — you can't tell them apart by looking. Their
  brains behave completely differently.
- ADHD is the monk's opposite: the brain can't hold a state, it switches
  constantly.
- Diagnosis today leans on questionnaires and judgment. We give an objective,
  reproducible anchor.

**3. Why we beat NEBA (~30s)**
- The only FDA-cleared EEG-ADHD tool (NEBA) used **one number** — the theta/beta
  ratio — and was pulled after the evidence didn't hold.
- We measure the **full dynamics** of state-switching, not one ratio.
- Punchline: *we tested theta/beta in our own data — it scored below chance.* We
  reproduced NEBA's failure and stepped past it.

**4. The big idea — spectrum & microstates (~40s)**
- The brain jumps between a few recurring electrical patterns ("microstates").
- What matters isn't *which* patterns — everyone has them — but *how the brain
  moves between them*: calm/predictable vs restless/random.
- We place any recording on a spectrum: **monk = stable extreme → ADHD = clinical
  pole.**

**5. How it works — pipeline (~40s; see the deep-dive below)**
- Upload → harmonize → clean → extract microstates → measure switching dynamics →
  score against references → plain-language report.

**6. The data (~20s)**
- Monk (our own recording, the anchor) · 121 ADHD/control children (Kaggle/IEEE)
  · healthy **adolescent, adult, and older-adult** references so we compare
  age-to-age.

**7. Results — the proof (~40s)**
- ADHD vs control separate on microstate dynamics, **cross-validated AUC ~0.70**
  (clinical EEG tools sit 0.70–0.80).
- The **monk lands more stable than ~95% of healthy adults** — survives the hard
  test.
- Key insight: his stability is about **order, not stillness** — he switches as
  much as anyone, but predictably.

**8. LIVE DEMO (~60s — this wins it)**
- Upload a file → spectrum placement, the per-recording microstate maps, band
  data, plain-language report.
- Have a backup screen recording in case wifi dies.

**9. Honest limitations (~20s — judges reward this)**
- "Decision **support**, not a diagnosis."
- Monk is n=1 (a landmark, not a class); ADHD cohort is pediatric — both flagged
  in the tool itself.

**10. The vision (~20s)**
- Long-term: a **closed-loop neurofeedback system** using the monk's signature as
  a real-time target — helping people with insomnia/anxiety train toward that
  calm state. *(Frame as vision, not built.)*

**11. Close (~15s)**
- "We turned one extraordinary brain into a ruler that tells a clinician where any
  brain falls — objective, reproducible, honest."

---

## How the algorithm works (the technical core)

Explain it in this order; each step is one sentence you can expand on.

### Step 1 — Harmonization (make recordings comparable)
Different headsets have different electrode counts and sampling rates, and
microstates are *topographic* (they're patterns of voltage across the scalp), so
a montage mismatch would masquerade as a brain difference. We:
- keep only the **16 electrodes shared** across every dataset (standard 10-20),
- **resample** everything to 100 Hz (so durations in milliseconds are
  comparable),
- apply an **average reference** (the standard for microstate analysis).
No data is invented and no channels are interpolated.

### Step 2 — Cleaning
Band-pass filter 1–40 Hz (removes slow drift and high-frequency noise),
bad-channel handling, average reference. We deliberately do **not** run ICA —
we tested it and it *hurt* separation (it strips frontal signal, and ADHD is a
frontal disorder), so the simpler filter-only pipeline is the validated default.

### Step 3 — Microstate extraction (the heart of it)
1. **Global Field Power (GFP)** = how strong the overall scalp field is at each
   instant. The topography is most stable and meaningful at GFP *peaks*, so we
   sample the maps there.
2. **Modified k-means clustering** groups those topographies into **4 canonical
   maps** (the literature standard, labeled A/B/C/D). "Modified" =
   **polarity-invariant**: a map and its color-flipped twin are treated as the
   same, because EEG topography sign flips with the underlying brain dipole and
   carries no extra information.
3. We build **one shared group template** from the reference cohort, then
   **backfit** every recording onto it — assigning each moment to its nearest of
   the 4 maps. Using a shared template is what makes subjects comparable.
   (In the app we *also* fit each upload's **own** 4 maps for display.)

### Step 4 — Features (turn the map sequence into numbers)
From the sequence of map labels over time we compute, per recording:
- **dwell time** — how long the brain holds each map,
- **coverage** — fraction of time in each map,
- **occurrence** — how often each map appears per second,
- **switch rate** — total state changes per second,
- **transition entropy** — the key one: the **Shannon conditional entropy** of
  the transition matrix, i.e. *how predictable the next state is given the
  current one*. Low entropy = an orderly, repeating sequence (the monk). High
  entropy = near-random jumping (ADHD).

We also compute spectral band powers (delta/theta/alpha/beta/gamma) for
transparency — and showed they don't separate the groups, which is the whole
point of using dynamics instead.

### Step 5 — Scoring (place it on the spectrum)
- We train a **logistic-regression axis** on the 121-subject cohort using the
  robust core features (transition entropy + switch rate). The model's signed
  decision value is the **stability axis** — a single number per recording.
- A new recording is **projected** onto that axis and reported as an **empirical
  percentile** against an **age-matched** reference group (children, adults, or
  older adults), never as a probability of disorder.
- The **monk is placed as a labeled landmark** at the stable extreme — he is
  n=1, an annotated dot, *not* a training class. We use the small 2-feature axis
  on purpose: it places out-of-distribution recordings (like the monk) reliably,
  whereas the full feature vector extrapolates unstably.

### Step 6 — Reproducibility
Every random step (k-means, any seeded estimator) uses a **fixed seed** that's
logged in the report. Same input → same output, every time.

### One-paragraph version (if you only get 20 seconds)
"We clean the EEG, reduce every headset to the same 16 electrodes, and find the
four recurring whole-scalp voltage patterns the brain cycles through. Then we
measure *how the brain moves between them* — especially transition entropy, how
predictable the switching is. A logistic model turns those dynamics into a single
position on a stability spectrum, anchored by the monk at the calm end and the
ADHD group at the restless end. Everything is reproducible and age-matched."

---

## Judge Q&A — answer cold

**Science**
- *"Why microstates instead of theta/beta?"* — Microstates capture temporal
  dynamics (how the brain switches), not a static ratio. We use 8+ features
  including transition entropy. And theta/beta scored below chance in our data.
- *"What does AUC ~0.70 mean?"* — ADHD vs controls; 0.5 is a coin flip; clinical
  EEG tools sit 0.70–0.80. Real signal, not a diagnosis. *(If a judge sees 0.63
  in the app: that's the small 2-feature placement axis we deploy on purpose for
  out-of-distribution robustness; the ~0.70 is the full microstate-dynamics
  separation. Honesty point, not a gotcha.)*
- *"Why only 4 microstate maps?"* — Literature standard (canonical A/B/C/D);
  ensures cross-subject comparability and physiological dwell times.
- *"How do you handle different headsets?"* — Shared 16-electrode subset,
  resampled to 100 Hz. No data invented, no interpolation.
- *"What is transition entropy, exactly?"* — The conditional Shannon entropy of
  the state-transition matrix: how much uncertainty there is about the next
  microstate given the current one. Order → low, randomness → high.

**Statistics / limitations**
- *"Isn't n=1 for the monk a problem?"* — Yes, and we say so. He's a labeled
  landmark at the stable extreme, not a training class. He anchors the narrative,
  not the statistics.
- *"Pediatric training data on adults?"* — Doesn't transfer cleanly — so we added
  adult and older-adult references and compare age-to-age, and flag the residual
  confound in the UI.
- *"Could it be gamed?"* — You'd have to consciously control your resting brain
  dynamics for minutes — essentially what years of meditation produce. Not
  realistic clinically.

**Engineering**
- *"Is it reproducible?"* — Yes — fixed seeds, logged in the report. Same input,
  same output.
- *"What formats does it accept?"* — EDF, EEGLAB .set, BrainVision, FIF, OpenBCI
  .txt; even 128-channel EGI nets auto-map to our 16. Auto-detected on upload.
- *"How long does analysis take?"* — Under a minute on a laptop for a 2-min clip.

**Curveball**
- *"Why ADHD?"* — It's the clearest clinical opposite of the monk (can't hold a
  state) and has open labeled data. The *method* generalizes to other
  state-instability conditions later.

**Tactics**
- Lead with the monk, not the tech — more memorable than "we trained a
  classifier."
- If pushed on clinical validity, pivot to "decision support — like a blood test
  informs a doctor, it doesn't replace them."
