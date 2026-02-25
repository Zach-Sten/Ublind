"""Inspect preprocessed sonification data."""

from __future__ import annotations

from ublind._core.notes import NoteEvent


def summary(adata) -> dict:
    """
    Return a summary dict of the preprocessed sonification.

    Keys: embedding, n_points, n_voices, n_events, duration_sec,
    instruments, pitch_range, scale, tempo_bpm.
    """
    ub = _get_uns(adata)
    pitches = ub["pitches"]

    return {
        "embedding": ub["embedding"],
        "n_points": len(ub["coords"]),
        "n_voices": pitches.shape[1],
        "n_events": len(ub["events"]),
        "duration_sec": ub["time_sec"],
        "instruments": [(v, name) for v, _, name in ub["instruments"]],
        "pitch_range": (int(pitches.min()), int(pitches.max())),
        "scale": ub.get("scale"),
        "tempo_bpm": ub["tempo_bpm"],
    }


def get_events(adata) -> list[NoteEvent]:
    """Return the list of NoteEvent objects."""
    return _get_uns(adata)["events"]


def _get_uns(adata) -> dict:
    if "ublind" not in adata.uns:
        raise RuntimeError("No ublind data found. Run ub.pp.preprocess(adata) first.")
    return adata.uns["ublind"]
