"""Single source of truth for every tunable in the pipeline.

Keeping all knobs here (rather than scattered as magic numbers) is what makes
the tool reproducible and auditable -- a hard requirement for a clinical-support
tool. See docs/ARCHITECTURE.md sections 4-7.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HarmonizeConfig:
    # Resample every recording to a common rate so microstate durations (ms) are
    # comparable across datasets. See ARCHITECTURE.md section 2, strategy 3.
    target_sfreq: float = 100.0
    # Strategy 1 (preferred): restrict to the electrodes shared by all datasets.
    # Filled in at M0 once we see the real headers; None means "use intersection".
    common_channels: tuple[str, ...] | None = None
    set_average_reference: bool = True


@dataclass(frozen=True)
class CleanConfig:
    l_freq: float = 1.0
    h_freq: float = 40.0
    notch_freq: float | None = None       # None -> auto-detect 50/60 Hz at M0
    artifact_method: str = "ica"          # "ica" (>2 min) or "asr" (short clips)
    ica_n_components: float = 0.99        # variance explained
    random_seed: int = 42                 # seed EVERYTHING for determinism


@dataclass(frozen=True)
class MicrostateConfig:
    n_states: int = 4                     # fixed k for cross-subject comparability
    gfp_peaks_only: bool = True
    cluster_method: str = "modkmeans"     # polarity-invariant modified k-means
    random_seed: int = 42
    # Temporal smoothing at backfit so labels don't flip every few samples.
    # Physiological microstates last ~60-120 ms; reject sub-30 ms segments.
    smoothing_half_window: int = 3        # samples (~30 ms at 100 Hz)
    smoothing_factor: int = 10            # Pascual-Marqui smoothing weight
    min_segment_ms: float = 30.0


@dataclass(frozen=True)
class ScoreConfig:
    # The HC->ADHD contrast direction is learned on the reference cohort.
    axis_model: str = "logreg"            # "logreg" | "lda" | "centroid"
    cv: str = "leave_one_subject_out"


@dataclass(frozen=True)
class Config:
    harmonize: HarmonizeConfig = field(default_factory=HarmonizeConfig)
    clean: CleanConfig = field(default_factory=CleanConfig)
    microstates: MicrostateConfig = field(default_factory=MicrostateConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)


DEFAULT = Config()
