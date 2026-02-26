"""
Cluster-order preprocessing: play clusters one at a time.

Each cluster gets a time slice and plays its notes as a chord.
Order can be by size, alphabetical, or custom.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ublind._core.notes import NoteEvent, map_to_pitches, LOWEST_NOTE, HIGHEST_NOTE
from ublind._core.instruments import (
    DEFAULT_DIM_INSTRUMENTS,
    GENERAL_MIDI_INSTRUMENTS,
    resolve_program,
)


def compose_clusters(
    adata,
    *,
    embedding: str = "X_umap",
    cluster_key: str = "leiden",
    time: float = 10.0,
    order: str = "largest",
    custom_order: Optional[Sequence[str]] = None,
    dim_inst_map: Optional[dict[int, str | int]] = None,
    counterpoint: bool = True,
    scale: Optional[str] = None,
    root: int = 0,
    velocity: int = 90,
    max_notes_per_cluster: int = 20,
    subsample: Optional[int] = None,
    tempo_bpm: float = 120.0,
    seed: int = 0,
) -> None:
    """
    Preprocess an embedding for cluster-by-cluster sonification.

    Each cluster gets an equal time slice and plays its notes as a
    chord. You hear clusters one at a time, like a roll call.

    Parameters
    ----------
    adata : AnnData
    embedding : str
        Key in ``adata.obsm``.
    cluster_key : str
        Column in ``adata.obs`` with cluster labels.
    time : float
        Total duration in seconds.
    order : str
        How to order clusters:
        ``"largest"``, ``"smallest"``, ``"alphabetical"``,
        ``"reverse_alpha"``, or ``"custom"``.
    custom_order : list of str, optional
        Explicit cluster order (requires ``order="custom"``).
    dim_inst_map : dict, optional
        Map dimension index to instrument.
    counterpoint : bool
        Flip pitch direction on odd dims.
    scale : str, optional
        Musical scale.
    root : int
        Root note (0=C).
    velocity : int
        MIDI velocity.
    max_notes_per_cluster : int
        Max notes sampled per cluster per dimension.
    subsample : int, optional
        Subsample points before processing.
    """
    if embedding not in adata.obsm:
        raise KeyError(f"'{embedding}' not in adata.obsm. Available: {list(adata.obsm.keys())}")
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"'{cluster_key}' not in adata.obs. Available: {list(adata.obs.columns)}")

    coords = np.array(adata.obsm[embedding], dtype=float)
    n_points, n_dims = coords.shape
    labels = adata.obs[cluster_key].astype("category")

    # Subsample
    subsample_idx = None
    rng = np.random.default_rng(seed)
    if subsample is not None and subsample < n_points:
        subsample_idx = np.sort(rng.choice(n_points, subsample, replace=False))
        coords = coords[subsample_idx]
        labels = labels.iloc[subsample_idx]
        n_points = len(coords)

    cat = labels.cat
    cluster_names = list(cat.categories)
    codes = cat.codes.values

    ordered_names = _resolve_order(cluster_names, codes, order, custom_order)
    n_clusters = len(ordered_names)

    instruments_info = _resolve_instruments(n_dims, dim_inst_map)
    time_per_cluster = time / n_clusters

    events = []
    cluster_info = []

    for ci, cname in enumerate(ordered_names):
        t_start = ci * time_per_cluster
        mask = np.array([str(labels.iloc[i]) == cname for i in range(n_points)])
        n_in_cluster = mask.sum()

        if n_in_cluster == 0:
            cluster_info.append({
                "name": cname, "t_start": t_start,
                "t_end": t_start + time_per_cluster,
                "n_cells": 0, "notes": [],
            })
            continue

        cluster_coords = coords[mask]

        if n_in_cluster > max_notes_per_cluster:
            sampled = rng.choice(n_in_cluster, max_notes_per_cluster, replace=False)
        else:
            sampled = np.arange(n_in_cluster)

        cluster_notes = []

        for d in range(n_dims):
            col = cluster_coords[sampled, d]
            pitches = map_to_pitches(col, scale=scale, root=root)

            if counterpoint and d % 2 == 1:
                pitches = (LOWEST_NOTE + HIGHEST_NOTE) - pitches
                if scale is not None:
                    pitches = map_to_pitches(
                        pitches.astype(float), lo=LOWEST_NOTE, hi=HIGHEST_NOTE,
                        scale=scale, root=root,
                    )

            _, prog, _ = instruments_info[d]
            ch = d if d < 9 else d + 1
            ch = ch % 16
            if ch == 9:
                ch = 10

            sorted_pitches = np.sort(pitches)
            n_notes = len(sorted_pitches)
            note_spacing = time_per_cluster * 0.8 / max(n_notes, 1)
            note_dur = min(time_per_cluster * 0.9, max(0.1, note_spacing * 2))

            for j, p in enumerate(sorted_pitches):
                p = int(p)
                if LOWEST_NOTE <= p <= HIGHEST_NOTE:
                    t = t_start + j * note_spacing
                    events.append(NoteEvent(
                        time=t, duration=note_dur, pitch=p,
                        velocity=velocity, instrument=prog, channel=ch,
                    ))
                    cluster_notes.append(p)

        cluster_info.append({
            "name": cname, "t_start": t_start,
            "t_end": t_start + time_per_cluster,
            "n_cells": int(n_in_cluster),
            "notes": sorted(set(cluster_notes)),
        })

    adata.uns["ublind"] = {
        "events": events,
        "embedding": embedding,
        "time_sec": time,
        "tempo_bpm": tempo_bpm,
        "coords": coords,
        "instruments": instruments_info,
        "subsample_idx": subsample_idx,
        "scale": scale,
        "root": root,
        "n_dims": n_dims,
        "mode": "clusters",
        "cluster_key": cluster_key,
        "cluster_order": ordered_names,
        "cluster_info": cluster_info,
        "time_per_cluster": time_per_cluster,
        "time_per_dim": {},
        "pitches_per_dim": {},
        "order_per_dim": {},
    }

    inst_names = [info[2] for info in instruments_info]
    print(
        f"ublind: preprocessed {embedding} → {len(events)} notes, "
        f"{n_clusters} clusters [{order}], "
        f"{n_dims} voice(s) [{', '.join(inst_names)}], "
        f"{time:.1f}s ({time_per_cluster:.2f}s/cluster)"
    )


def _resolve_order(cluster_names, codes, order, custom_order):
    from collections import Counter
    counts = Counter(codes)
    name_counts = {name: counts.get(i, 0) for i, name in enumerate(cluster_names)}

    if order == "custom":
        if custom_order is None:
            raise ValueError("order='custom' requires custom_order=[...]")
        return list(custom_order)
    elif order == "largest":
        return sorted(cluster_names, key=lambda n: -name_counts[n])
    elif order == "smallest":
        return sorted(cluster_names, key=lambda n: name_counts[n])
    elif order == "alphabetical":
        return sorted(cluster_names)
    elif order == "reverse_alpha":
        return sorted(cluster_names, reverse=True)
    else:
        raise ValueError(f"Unknown order '{order}'. Use 'largest', 'smallest', 'alphabetical', 'reverse_alpha', or 'custom'.")


def _resolve_instruments(n_voices, dim_inst_map):
    dim_inst_map = dim_inst_map or {}
    info = []
    for v in range(n_voices):
        if v in dim_inst_map:
            prog = resolve_program(dim_inst_map[v])
        else:
            prog = DEFAULT_DIM_INSTRUMENTS[v % len(DEFAULT_DIM_INSTRUMENTS)]
        name = GENERAL_MIDI_INSTRUMENTS.get(prog, f"Program {prog}")
        info.append((v, prog, name))
    return info
