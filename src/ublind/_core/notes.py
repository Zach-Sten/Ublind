"""Note events, scales, and pitch mapping utilities."""

from __future__ import annotations

import dataclasses

import numpy as np


# ── Scales (semitone offsets from root) ──────────────────────────

SCALES: dict[str, list[int]] = {
    "chromatic": list(range(12)),
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
    "whole_tone": [0, 2, 4, 6, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
}

LOWEST_NOTE = 21   # A0
HIGHEST_NOTE = 108  # C8


@dataclasses.dataclass(frozen=True, slots=True)
class NoteEvent:
    """A single note."""
    time: float        # seconds
    duration: float    # seconds
    pitch: int         # MIDI 21–108
    velocity: int      # 0–127
    instrument: int    # GM program number
    channel: int = 0


def map_to_pitches(
    values: np.ndarray,
    *,
    lo: int = LOWEST_NOTE,
    hi: int = HIGHEST_NOTE,
    scale: str | None = None,
    root: int = 0,
) -> np.ndarray:
    """
    Linearly map a 1-D array of values to MIDI pitches [lo, hi],
    optionally quantising to a musical scale.
    """
    v = np.asarray(values, dtype=float)
    vmin, vmax = v.min(), v.max()
    span = vmax - vmin
    if span == 0:
        span = 1.0
    normed = (v - vmin) / span
    pitches = np.round(normed * (hi - lo) + lo).astype(int)
    pitches = np.clip(pitches, lo, hi)

    if scale is not None:
        offsets = SCALES.get(scale, SCALES["major"])
        allowed = sorted({
            oct * 12 + root + s
            for oct in range(11)
            for s in offsets
            if lo <= oct * 12 + root + s <= hi
        })
        allowed_arr = np.array(allowed)
        idx = np.searchsorted(allowed_arr, pitches, side="left")
        idx = np.clip(idx, 0, len(allowed_arr) - 1)
        left = allowed_arr[np.clip(idx - 1, 0, len(allowed_arr) - 1)]
        right = allowed_arr[idx]
        pitches = np.where(
            np.abs(pitches - left) < np.abs(pitches - right), left, right
        )

    return pitches
