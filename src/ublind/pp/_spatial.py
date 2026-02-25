"""
Spatial preprocessing: sonify cells by their physical tissue coordinates.

Sweeps across the spatial field — each cell plays a note when the
sweep line passes its position. Pitch is mapped from the perpendicular
axis, so you hear the spatial structure of the tissue.
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


def preprocess_spatial(
    adata,
    *,
    spatial_key: str = "spatial",
    time: float = 10.0,
    sweep_axis: str = "x",
    instrument: str = "piano",
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
    Preprocess spatial coordinates for sonification.

    Sweeps across the tissue — time maps to position along
    ``sweep_axis``, pitch maps to the perpendicular axis.
    You hear the spatial landscape of the tissue as the sweep
    line moves across it.

    Parameters
    ----------
    adata : AnnData
        Must have spatial coordinates in ``adata.obsm[spatial_key]``.
    spatial_key : str
        Key in ``adata.obsm`` for spatial coordinates.
    time : float
        Total duration in seconds.
    sweep_axis : str
        ``"x"`` sweeps left→right (time=X, pitch=Y), or
        ``"y"`` sweeps bottom→top (time=Y, pitch=X), or
        ``"both"`` sweeps both axes with two instruments.
    instrument : str or int
        Instrument for single-axis sweep. For ``"both"``, use
        the first instrument; the second gets a default partner.
    counterpoint : bool
        If ``sweep_axis="both"``, flip pitch on the second axis.
    scale : str, optional
        Musical scale.
    root : int
        Root note (0=C).
    velocity : int
        MIDI velocity.
    note_duration : float, optional
        Note length. Auto-computed if ``None``.
    subsample : int, optional
        Subsample to this many cells.
    """
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"'{spatial_key}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )

    coords = np.array(adata.obsm[spatial_key], dtype=float)
    if coords.ndim == 2 and coords.shape[1] > 2:
        coords = coords[:, :2]  # take first 2 cols (some have z)
    n_points = coords.shape[0]

    # Subsample
    subsample_idx = None
    if subsample is not None and subsample < n_points:
        rng = np.random.default_rng(seed)
        subsample_idx = np.sort(rng.choice(n_points, subsample, replace=False))
        coords = coords[subsample_idx]
        n_points = len(coords)

    sx, sy = coords[:, 0], coords[:, 1]

    # Note duration
    if note_duration is None:
        note_duration = max(0.05, time / n_points * 2)

    events: list[NoteEvent] = []
    instruments_info = []

    if sweep_axis in ("x", "both"):
        # Sweep left → right: time=X, pitch=Y
        prog = resolve_program(instrument)
        prog_name = GENERAL_MIDI_INSTRUMENTS.get(prog, f"Program {prog}")
        instruments_info.append((0, prog, prog_name))

        order_x = np.argsort(sx)
        times_x = _normalize(sx[order_x], time)
        pitches_x = map_to_pitches(sy[order_x], scale=scale, root=root)

        ch = 0
        for i in range(n_points):
            p = int(pitches_x[i])
            if LOWEST_NOTE <= p <= HIGHEST_NOTE:
                events.append(NoteEvent(
                    time=float(times_x[i]), duration=note_duration,
                    pitch=p, velocity=velocity, instrument=prog, channel=ch,
                ))

    if sweep_axis in ("y", "both"):
        # Sweep bottom → top: time=Y, pitch=X
        if sweep_axis == "both":
            # Second instrument
            prog2 = DEFAULT_DIM_INSTRUMENTS[1]
            prog2_name = GENERAL_MIDI_INSTRUMENTS.get(prog2, f"Program {prog2}")
            instruments_info.append((1, prog2, prog2_name))
            ch = 1
        else:
            prog2 = resolve_program(instrument)
            prog2_name = GENERAL_MIDI_INSTRUMENTS.get(prog2, f"Program {prog2}")
            instruments_info.append((0, prog2, prog2_name))
            ch = 0

        order_y = np.argsort(sy)
        times_y = _normalize(sy[order_y], time)
        pitches_y = map_to_pitches(sx[order_y], scale=scale, root=root)

        # Counterpoint: flip pitch on second axis
        if counterpoint and sweep_axis == "both":
            pitches_y = (LOWEST_NOTE + HIGHEST_NOTE) - pitches_y
            if scale is not None:
                pitches_y = map_to_pitches(
                    pitches_y.astype(float), lo=LOWEST_NOTE, hi=HIGHEST_NOTE,
                    scale=scale, root=root,
                )

        for i in range(n_points):
            p = int(pitches_y[i])
            if LOWEST_NOTE <= p <= HIGHEST_NOTE:
                events.append(NoteEvent(
                    time=float(times_y[i]), duration=note_duration,
                    pitch=p, velocity=velocity, instrument=prog2, channel=ch,
                ))

    # Build order/pitch dicts for compatibility with sweep plots
    n_dims = 2
    time_per_dim = {}
    pitches_per_dim = {}
    order_per_dim = {}

    if sweep_axis in ("x", "both"):
        order_per_dim[0] = order_x
        time_per_dim[0] = times_x
        pitches_per_dim[0] = pitches_x
    if sweep_axis in ("y", "both"):
        d = 1 if sweep_axis == "both" else 0
        order_per_dim[d] = order_y
        time_per_dim[d] = times_y
        pitches_per_dim[d] = pitches_y

    adata.uns["ublind"] = {
        "events": events,
        "embedding": spatial_key,
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
        "mode": "spatial",
        "sweep_axis": sweep_axis,
    }

    n_events = len(events)
    inst_names = [info[2] for info in instruments_info]
    print(
        f"ublind: preprocessed {spatial_key} → {n_events} notes, "
        f"sweep={sweep_axis}, [{', '.join(inst_names)}], {time:.1f}s"
    )


def _normalize(sorted_values: np.ndarray, total: float) -> np.ndarray:
    v_min, v_max = sorted_values.min(), sorted_values.max()
    span = v_max - v_min
    if span == 0:
        span = 1.0
    return (sorted_values - v_min) / span * total
