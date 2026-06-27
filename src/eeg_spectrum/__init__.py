"""eeg_spectrum: place a resting-state EEG on a microstate state-stability axis.

Pipeline (see docs/ARCHITECTURE.md):
    io -> harmonize -> clean -> microstates -> features -> score -> report
"""

from __future__ import annotations

from .config import DEFAULT, Config

__all__ = ["DEFAULT", "Config"]
__version__ = "0.1.0"
