# References and data sources

Every dataset, tool, and method the EEG Microstate State-Stability Spectrum Model
relies on. Repository accessions / URLs are the authoritative pointers; verify
author lists and DOIs against each source page before formal citation.

---

## 1. Reference data used in the model

### 1.1 Monk recording — the healthy-stability anchor (n=1)
- **Role:** anchors the stable (ordered-switching) extreme of the spectrum.
- **Source:** original data collected by YiRen Lin — "Zen Brain" study, Anhui
  Mountains monastery, 2025. Subject: 一音禅师 (a trained Zen monk).
- **Specs:** OpenBCI Cyton+Daisy, 16 channels (standard 10-20), 125 Hz, two
  10-minute conditions — 無念 / Empty Mind (deep quiescence, the anchor) and 念经
  / Silent Sutra Recitation (active meditation).
- **Access:** private (subject's biometric data — not redistributed).

### 1.2 ADHD / control cohort — the clinical pole and axis training
- **Role:** defines the ADHD (low-stability) pole; trains the stability axis.
- **Source:** "EEG Dataset for ADHD," Kaggle slug `danizo/eeg-dataset-for-adhd`,
  redistributing the IEEE-DataPort ADHD/control children dataset.
- **Original citation:** Nasrabadi, A. M., Allahverdy, A., Samavati, M.,
  Mohammadi, M. R. "EEG data for ADHD / Control children." IEEE DataPort, 2020.
  DOI: 10.21227/rzfh-zn36.
- **Specs:** 121 children (61 ADHD, 60 control), ages 7–12, 19 channels (10-20),
  128 Hz. Recorded during a visual-attention task (counting cartoon characters)
  — note: task-based, not pure rest (a documented limitation).
- **Access:** open (Kaggle / IEEE DataPort).
- **URL:** https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd

### 1.3 Healthy adults — adult reference (n≈80)
- **Role:** age-matched healthy reference for adult uploads.
- **Source:** EEG Motor Movement/Imagery Dataset (EEGMMIDB), PhysioNet.
- **Citation:** Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N.,
  Wolpaw, J. R. "BCI2000: A general-purpose brain-computer interface system."
  IEEE Trans. Biomed. Eng., 2004. Hosted via Goldberger, A. L. et al.
  "PhysioBank, PhysioToolkit, and PhysioNet." Circulation, 2000.
- **Specs:** 109 healthy adults, 64 channels (10-10), 160 Hz. Used the
  eyes-closed baseline run (R02). ~80 subjects processed.
- **Access:** open, no authentication.
- **URL:** https://physionet.org/content/eegmmidb/1.0.0/

### 1.4 Healthy older adults — older-adult reference (n=29)
- **Role:** age-matched healthy reference for older-adult uploads (57–78 yrs).
- **Source:** OpenNeuro **ds004504** — "A dataset of EEG recordings from:
  Alzheimer's disease, Frontotemporal dementia and Healthy subjects" (the
  healthy-control group, Group "C").
- **Citation:** Miltiadous, A. et al., 2023 (OpenNeuro ds004504).
- **Specs:** 29 healthy controls, ages 57–78, eyes-closed resting, 19 channels
  (10-20), EEGLAB `.set`. All 16 model channels present.
- **Access:** open.
- **URL:** https://openneuro.org/datasets/ds004504

### 1.5 Adolescent reference — community sample (n≈8)
- **Role:** age reference for the 7–18 range.
- **Source:** OpenNeuro **ds005505** — Healthy Brain Network (HBN) EEG, Release 1,
  RestingState task.
- **Citation:** Alexander, L. M. et al. "An open resource for transdiagnostic
  research in pediatric mental health and learning disorders: the Healthy Brain
  Network." Scientific Data, 2017. (EEG release: OpenNeuro ds005505.)
- **Specs:** children/adolescents 5–21, eyes-open/closed resting, 128-channel
  EGI HydroCel net, reduced to our 16 channels via the documented EGI→10-20
  correspondence (functionally validated: eyes-closed posterior alpha dominance).
- **Caveat:** HBN is a **community / transdiagnostic** sample — not screened
  healthy — so it is a reference population, not a clean control group. Small n.
- **Access:** open.
- **URL:** https://github.com/OpenNeuroDatasets/ds005505

---

## 2. Datasets evaluated but NOT integrated

- **TDBRAIN** (Two Decades Brainclinics Research Archive) — resting EEG, 1274
  psychiatric patients incl. ADHD (N=271), lifespan 5–89. The best candidate for
  *adult ADHD* and possible subtypes. **Not used:** access is gated behind a
  data-use agreement (no anonymous download). van Dijk, H. et al., Scientific
  Data, 2022. https://www.nature.com/articles/s41597-022-01409-z
- **Mendeley adult ADHD** (`6k4g25fhzg`) — adult ADHD/control resting EEG.
  **Not used:** only 5 channels (O1, F3, F4, Cz, Fz); incompatible with the
  16-channel model. https://data.mendeley.com/datasets/6k4g25fhzg/1

---

## 3. Software

- **MNE-Python** — EEG I/O, preprocessing, montages, topographies.
  Gramfort, A. et al. "MEG and EEG data analysis with MNE-Python." Frontiers in
  Neuroscience, 2013.
- **pycrostates** — microstate clustering and backfitting (modified k-means).
  Férat, V. et al. "Pycrostates: a Python library to study EEG microstates."
  Journal of Open Source Software, 2022.
- **scikit-learn** — logistic-regression axis, cross-validation, clustering.
  Pedregosa, F. et al. JMLR, 2011.
- **NumPy / SciPy / pandas / Matplotlib / Streamlit** — computation, plotting, UI.

---

## 4. Methods and conceptual prior art

- **Microstate segmentation (modified k-means):** Pascual-Marqui, R. D.,
  Michel, C. M., Lehmann, D. "Segmentation of brain electrical activity into
  microstates." IEEE Trans. Biomed. Eng., 1995.
- **Microstate review / canonical maps A–D:** Michel, C. M., Koenig, T. "EEG
  microstates as a tool for studying the temporal dynamics of whole-brain
  neuronal networks: A review." NeuroImage, 2018.
- **Theta/beta ratio in ADHD (and its limits):** Arns, M., Conners, C. K.,
  Kraemer, H. C. "A decade of EEG theta/beta ratio research in ADHD: a
  meta-analysis." Journal of Attention Disorders, 2013. Context for the
  FDA-cleared NEBA system that this project deliberately moves beyond.

---

## 5. How the references combine

| Population | Source | n | Role |
|---|---|---|---|
| Monk (anchor) | own data | 1 | stable extreme |
| ADHD | Kaggle/IEEE | 61 | clinical pole + axis |
| Child controls | Kaggle/IEEE | 60 | axis + pediatric reference |
| Adolescents | HBN ds005505 | ~8 | 7–18 reference |
| Adults | PhysioNet EEGMMIDB | ~80 | adult reference |
| Older adults | OpenNeuro ds004504 | 29 | 60+ reference |

All recordings are harmonized to the same 16 shared 10-20 channels, resampled to
100 Hz, and average-referenced before analysis. See `ARCHITECTURE.md` for the
pipeline and `LIMITATIONS.md` for the honest scope of every source above.
