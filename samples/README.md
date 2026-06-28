# Sample files — SYNTHETIC, not real recordings

**These are simulated signals. They are NOT real EEGs, not real people, and must
never be presented as real data or counted in any result, statistic, or cohort.**

They exist for one purpose: a clean live demo that shows the spectrum working
across its full range. Both are generated 16-channel, 128 Hz, ~4-minute
resting-state-like signals from four distinct microstate topographies plus a
posterior ~10 Hz alpha rhythm and background noise.

| File | Pattern | Lands around |
|---|---|---|
| `synthetic_demo_eeg.fif` | ordered, mostly-cyclic switching, strong alpha | ~40/100 (calm/typical) |
| `synthetic_demo_adhd_eeg.fif` | rapid, near-random switching, weaker alpha | ~76/100 (ADHD/restless) |

The ADHD file simulates the *dynamics pattern* associated with ADHD — it is a
generated signal, **not a real child and not a diagnosis**.

## `synthetic_muse_demo.csv` — SYNTHETIC Muse headband recording

A simulated 4-channel Muse recording (TP9, AF7, AF8, TP10; 256 Hz; ~3 min, Mind
Monitor CSV format) for the app's **experimental Muse band-power screen**. NOT a
real person. A 4-channel headband cannot do microstate dynamics, so this only
demos the band-power view — never the state-stability spectrum.

Regenerate with:

```bash
python scripts/make_synthetic_demo.py --profile stable
python scripts/make_synthetic_demo.py --profile adhd
python scripts/make_synthetic_muse.py
```

When demoing, the honest phrasing is: *"these are synthetic recordings we
generated to show the pipeline across the spectrum."* Never imply either is a
patient or a study subject.
