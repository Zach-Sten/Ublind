"""
Preprocessing: embedding → NoteEvents stored in adata.uns['ublind'].
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ublind._core.notes import NoteEvent, map_to_pitches, LOWEST_NOTE, HIGHEST_NOTE
from ublind._core.instruments import (
    DEFAULT_DIM_INSTRUMENTS,
    GENERAL_MIDI_INSTRUMENTS,
    resolve_program,
)


def preprocess(
    adata,
    *,
    embedding: str = "X_umap",
    time: float = 10.0,
    dim_inst_map: Optional[dict[int, str | int]] = None,
    time_dim: int = 0,
    scale: Optional[str] = None,
    root: int = 0,
    velocity: int = 90,
    note_duration: Optional[float] = None,
    subsample: Optional[int] = None,
    tempo_bpm: float = 120.0,
    seed: int = 0,
) -> None:
    """
    Preprocess an embedding for sonification.

    Extracts an embedding from ``adata.obsm[embedding]``, maps one
    dimension to time and the remaining dimensions to instrument
    voices, then stores the result in ``adata.uns['ublind']``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    embedding : str
        Key in ``adata.obsm`` (e.g. ``"X_umap"``, ``"X_pca"``).
    time : float
        Total duration of the piece in seconds.
    dim_inst_map : dict, optional
        Map dimension index (0-based, *after* removing the time dim)
        to instrument name or GM program number.
        Example: ``{0: "piano", 1: "cello", 2: "flute"}``
        Unmapped dimensions get pleasant defaults.
    time_dim : int
        Which column of the embedding to use as the time axis.
    scale : str, optional
        Quantise pitches to a scale (``"pentatonic"``, ``"major"``, etc.).
    root : int
        Root note offset from C (0–11).
    velocity : int
        MIDI velocity (0–127).
    note_duration : float, optional
        Duration of each note in seconds. If ``None``, auto-computed.
    subsample : int, optional
        Randomly subsample to this many points.
    tempo_bpm : float
        Tempo (used by renderers).
    seed : int
        Random seed for subsampling.
    """
    _validate_inputs(adata, embedding, time_dim)

    coords = np.array(adata.obsm[embedding], dtype=float)
    n_points, n_dims = coords.shape

    # Subsample
    subsample_idx = None
    if subsample is not None and subsample < n_points:
        rng = np.random.default_rng(seed)
        subsample_idx = np.sort(rng.choice(n_points, subsample, replace=False))
        coords = coords[subsample_idx]
        n_points = len(coords)

    # Split time vs value dimensions
    raw_time = coords[:, time_dim]
    value_cols = np.delete(coords, time_dim, axis=1)
    n_voices = value_cols.shape[1]

    # Sort by time
    order = np.argsort(raw_time)
    raw_time = raw_time[order]
    value_cols = value_cols[order]
    coords = coords[order]
    if subsample_idx is not None:
        subsample_idx = subsample_idx[order]

    # Normalise time to [0, total_duration]
    time_values = _normalize_time_axis(raw_time, time)

    # Note duration
    if note_duration is None:
        note_duration = max(0.05, time / n_points * 2)

    # Resolve instruments per voice
    instruments_info = _resolve_instruments(n_voices, dim_inst_map)

    # Map values to pitches
    pitches = _compute_pitches(value_cols, n_voices, scale, root)

    # Build NoteEvents
    events = _build_events(
        time_values, pitches, instruments_info, note_duration, velocity,
    )

    # Store in adata.uns
    adata.uns["ublind"] = {
        "events": events,
        "embedding": embedding,
        "time_dim": time_dim,
        "time_sec": time,
        "tempo_bpm": tempo_bpm,
        "coords": coords,
        "time_values": time_values,
        "pitches": pitches,
        "instruments": instruments_info,
        "subsample_idx": subsample_idx,
        "scale": scale,
        "root": root,
    }

    inst_names = [info[2] for info in instruments_info]
    print(
        f"ublind: preprocessed {embedding} → {len(events)} notes, "
        f"{n_voices} voice(s) [{', '.join(inst_names)}], "
        f"{time:.1f}s"
    )


# ── Private helpers ──────────────────────────────────────────────


def _validate_inputs(adata, embedding: str, time_dim: int) -> None:
    """Check that the embedding exists and has enough dimensions."""
    if embedding not in adata.obsm:
        raise KeyError(
            f"'{embedding}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )
    n_dims = adata.obsm[embedding].shape[1]
    if n_dims < 2:
        raise ValueError(f"Embedding must have ≥2 dims, got {n_dims}")
    if time_dim < 0 or time_dim >= n_dims:
        raise ValueError(f"time_dim={time_dim} out of range for {n_dims} dims")


def _normalize_time_axis(raw_time: np.ndarray, total: float) -> np.ndarray:
    """Rescale raw time values to [0, total]."""
    t_min, t_max = raw_time.min(), raw_time.max()
    t_span = t_max - t_min
    if t_span == 0:
        t_span = 1.0
    return (raw_time - t_min) / t_span * total


def _resolve_instruments(
    n_voices: int,
    dim_inst_map: dict[int, str | int] | None,
) -> list[tuple[int, int, str]]:
    """Return [(voice_idx, program, name), ...] for each voice."""
    dim_inst_map = dim_inst_map or {}
    info: list[tuple[int, int, str]] = []
    for v in range(n_voices):
        if v in dim_inst_map:
            prog = resolve_program(dim_inst_map[v])
        else:
            prog = DEFAULT_DIM_INSTRUMENTS[v % len(DEFAULT_DIM_INSTRUMENTS)]
        name = GENERAL_MIDI_INSTRUMENTS.get(prog, f"Program {prog}")
        info.append((v, prog, name))
    return info


def _compute_pitches(
    value_cols: np.ndarray,
    n_voices: int,
    scale: str | None,
    root: int,
) -> np.ndarray:
    """Map each voice column to MIDI pitches."""
    pitches = np.zeros_like(value_cols, dtype=int)
    for v in range(n_voices):
        pitches[:, v] = map_to_pitches(value_cols[:, v], scale=scale, root=root)
    return pitches


def _build_events(
    time_values: np.ndarray,
    pitches: np.ndarray,
    instruments_info: list[tuple[int, int, str]],
    note_duration: float,
    velocity: int,
) -> list[NoteEvent]:
    """Convert arrays of times + pitches into a flat list of NoteEvents."""
    n_points = len(time_values)
    n_voices = pitches.shape[1]
    events: list[NoteEvent] = []

    for i in range(n_points):
        t = float(time_values[i])
        for v in range(n_voices):
            p = int(pitches[i, v])
            if LOWEST_NOTE <= p <= HIGHEST_NOTE:
                _, prog, _ = instruments_info[v]
                # Channel 9 is drums in GM — skip it
                ch = v if v < 9 else v + 1
                ch = ch % 16
                if ch == 9:
                    ch = 10
                events.append(NoteEvent(
                    time=t,
                    duration=note_duration,
                    pitch=p,
                    velocity=velocity,
                    instrument=prog,
                    channel=ch,
                ))

    return events
