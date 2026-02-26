"""
Preprocessing: embedding → NoteEvents stored in adata.uns['ublind'].

Every dimension of the embedding becomes its own voice/instrument.
Each dimension independently sweeps through time (sorted by that
dimension's values) and maps its values to pitch.
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


def compose(
    adata,
    *,
    embedding: str = "X_umap",
    time: float = 10.0,
    dim_inst_map: Optional[dict[int, str | int]] = None,
    counterpoint: bool = True,
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

    Every dimension of the embedding becomes a separate voice/instrument.
    Each dimension sweeps through time independently — sorted by that
    dimension's values — so no axis of variation is lost.

    For a 2D UMAP with piano + cello:
    - Piano sweeps left → right (sorted by dim 0)
    - Cello sweeps bottom → top (sorted by dim 1)
    - You hear both axes simultaneously

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    embedding : str
        Key in ``adata.obsm`` (e.g. ``"X_umap"``, ``"X_pca"``).
    time : float
        Total duration of the piece in seconds.
    dim_inst_map : dict, optional
        Map dimension index to instrument name or GM program number.
        Example: ``{0: "piano", 1: "cello"}``
        Unmapped dimensions get pleasant defaults.
    counterpoint : bool
        If True (default), odd-numbered dimensions have their pitch
        direction flipped — so while dim 0 sweeps low→high, dim 1
        sweeps high→low. Creates musical counterpoint instead of
        all voices rising together.
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
    _validate_inputs(adata, embedding)

    coords = np.array(adata.obsm[embedding], dtype=float)
    n_points, n_dims = coords.shape

    # Subsample
    subsample_idx = None
    if subsample is not None and subsample < n_points:
        rng = np.random.default_rng(seed)
        subsample_idx = np.sort(rng.choice(n_points, subsample, replace=False))
        coords = coords[subsample_idx]
        n_points = len(coords)

    n_voices = n_dims

    # Note duration
    if note_duration is None:
        note_duration = max(0.05, time / n_points * 2)

    # Resolve instruments per voice
    instruments_info = _resolve_instruments(n_voices, dim_inst_map)

    # Per-dimension: independent time + pitch
    # Even dims: low → high pitch, Odd dims: high → low pitch
    # This creates counterpoint instead of all dims rising together
    time_per_dim = {}
    pitches_per_dim = {}
    order_per_dim = {}

    for d in range(n_dims):
        col = coords[:, d]
        order = np.argsort(col)
        order_per_dim[d] = order
        time_per_dim[d] = _normalize_axis(col[order], time)

        p = map_to_pitches(col[order], scale=scale, root=root)
        # Flip pitch direction on odd dimensions if counterpoint is on
        if counterpoint and d % 2 == 1:
            p = (LOWEST_NOTE + HIGHEST_NOTE) - p
            # Re-snap to scale after flipping
            if scale is not None:
                p = map_to_pitches(
                    p.astype(float), lo=LOWEST_NOTE, hi=HIGHEST_NOTE,
                    scale=scale, root=root,
                )
        pitches_per_dim[d] = p

    # Build NoteEvents — each dim is its own voice
    events = _build_events(
        time_per_dim, pitches_per_dim, order_per_dim,
        instruments_info, note_duration, velocity, n_dims,
    )

    # Store in adata.uns
    adata.uns["ublind"] = {
        "events": events,
        "embedding": embedding,
        "time_sec": time,
        "tempo_bpm": tempo_bpm,
        "coords": coords,
        "time_per_dim": time_per_dim,
        "pitches_per_dim": pitches_per_dim,
        "order_per_dim": order_per_dim,
        "instruments": instruments_info,
        "subsample_idx": subsample_idx,
        "scale": scale,
        "root": root,
        "n_dims": n_dims,
    }

    inst_names = [info[2] for info in instruments_info]
    print(
        f"ublind: preprocessed {embedding} → {len(events)} notes, "
        f"{n_voices} voice(s) [{', '.join(inst_names)}], "
        f"{time:.1f}s"
    )


# ── Private helpers ──────────────────────────────────────────────


def _validate_inputs(adata, embedding: str) -> None:
    """Check that the embedding exists and has enough dimensions."""
    if embedding not in adata.obsm:
        raise KeyError(
            f"'{embedding}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )
    n_dims = adata.obsm[embedding].shape[1]
    if n_dims < 1:
        raise ValueError(f"Embedding must have ≥1 dims, got {n_dims}")


def _normalize_axis(sorted_values: np.ndarray, total: float) -> np.ndarray:
    """Rescale sorted values to [0, total]."""
    v_min, v_max = sorted_values.min(), sorted_values.max()
    span = v_max - v_min
    if span == 0:
        span = 1.0
    return (sorted_values - v_min) / span * total


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


def _build_events(
    time_per_dim: dict,
    pitches_per_dim: dict,
    order_per_dim: dict,
    instruments_info: list[tuple[int, int, str]],
    note_duration: float,
    velocity: int,
    n_dims: int,
) -> list[NoteEvent]:
    """Build NoteEvents — each dimension sweeps independently."""
    events: list[NoteEvent] = []

    for d in range(n_dims):
        times = time_per_dim[d]
        pitches = pitches_per_dim[d]
        _, prog, _ = instruments_info[d]

        # Channel assignment (skip channel 9 = drums)
        ch = d if d < 9 else d + 1
        ch = ch % 16
        if ch == 9:
            ch = 10

        for i in range(len(times)):
            p = int(pitches[i])
            if LOWEST_NOTE <= p <= HIGHEST_NOTE:
                events.append(NoteEvent(
                    time=float(times[i]),
                    duration=note_duration,
                    pitch=p,
                    velocity=velocity,
                    instrument=prog,
                    channel=ch,
                ))

    return events
