"""Interactive scatter plot with cluster sonification."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ublind.pl._utils import get_ublind_uns

_HTML_DIR = Path(__file__).parent / "html"


def interactive(
    adata,
    *,
    color_by: str = "leiden",
    embedding: Optional[str] = None,
    dims: tuple[int, int] = (0, 1),
    invert_y: bool = True,
    play_axis: str = "both",
    counterpoint: bool = True,
    scale: str = "pentatonic",
    root: int = 0,
    n_chord_notes: int = 12,
    point_size: int = 4,
    width: int = 700,
    height: int = 700,
):
    """
    Interactive scatter plot — click a cluster to hear its chord.

    Renders an HTML widget in Jupyter with Web Audio API.
    Each cluster's points are mapped to pitches; clicking a cluster
    plays them as an arpeggiated chord.

    Parameters
    ----------
    adata : AnnData
        Must have been preprocessed with ``ub.pp.compose()``.
    color_by : str
        Categorical column in ``adata.obs`` for cluster identity.
    embedding : str, optional
        Override embedding key. Defaults to whatever was preprocessed.
        Use ``"spatial"`` to show spatial coordinates.
    dims : tuple
        Which 2 dims to plot.
    invert_y : bool
        Flip Y axis. Default True (correct for spatial/imaging data).
        Set False for UMAP/PCA where low-Y should be at bottom.
    play_axis : str
        Which axis notes to play on hover: ``"x"``, ``"y"``, or
        ``"both"`` (default). ``"both"`` plays the note from each
        axis simultaneously for a richer sound.
    scale : str
        Scale for pitch mapping.
    root : int
        Root note (0=C).
    n_chord_notes : int
        Max notes to play per cluster chord.
    point_size : int
        Scatter point radius in pixels.
    width, height : int
        Widget size.

    Returns
    -------
    IPython.display.HTML
    """
    from IPython.display import HTML
    from ublind._core.notes import map_to_pitches, SCALES

    ub = get_ublind_uns(adata)
    subsample_idx = ub.get("subsample_idx")

    emb_key = embedding or ub["embedding"]

    # If a different embedding is requested, pull directly from adata.obsm
    if emb_key != ub["embedding"] and emb_key in adata.obsm:
        raw_coords = np.array(adata.obsm[emb_key], dtype=float)
        if raw_coords.ndim == 2 and raw_coords.shape[1] > 2:
            raw_coords = raw_coords[:, :2]
        # For spatial: use all points (no subsample)
        if emb_key == "spatial":
            coords = raw_coords
            subsample_idx = None
        elif subsample_idx is not None:
            coords = raw_coords[subsample_idx]
        else:
            coords = raw_coords
    else:
        coords = ub["coords"]

    d0, d1 = dims
    x = coords[:, d0]
    y = coords[:, d1]

    # Get cluster labels — match to coords length
    col = adata.obs[color_by]
    if subsample_idx is not None:
        col = col.iloc[subsample_idx]
    else:
        col = col.iloc[:len(coords)]

    cat = col.astype("category")
    codes = cat.cat.codes.values
    names = list(cat.cat.categories)

    # Colors from scanpy or defaults
    color_key = f"{color_by}_colors"
    if color_key in adata.uns:
        palette = list(adata.uns[color_key])
    else:
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
            "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
        ]

    # Compute pitches for both dims
    from ublind._core.notes import LOWEST_NOTE, HIGHEST_NOTE

    pitches_d0 = map_to_pitches(x, scale=scale, root=root)
    pitches_d1 = map_to_pitches(y, scale=scale, root=root)

    # Counterpoint: flip dim 1 pitch direction
    if counterpoint:
        pitches_d1 = (LOWEST_NOTE + HIGHEST_NOTE) - pitches_d1
        if scale is not None:
            pitches_d1 = map_to_pitches(
                pitches_d1.astype(float), lo=LOWEST_NOTE, hi=HIGHEST_NOTE,
                scale=scale, root=root,
            )

    # Build per-cluster chord data
    cluster_data = []
    for i, name in enumerate(names):
        mask = codes == i
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        rng = np.random.default_rng(i)
        if len(idx) > n_chord_notes:
            sampled = rng.choice(idx, n_chord_notes, replace=False)
        else:
            sampled = idx
        notes = sorted(set(
            [int(v) for v in pitches_d0[sampled]] +
            [int(v) for v in pitches_d1[sampled]]
        ))
        if len(notes) > n_chord_notes:
            notes = sorted([int(v) for v in rng.choice(notes, n_chord_notes, replace=False)])
        notes = sorted(notes)

        note_names = _midi_to_note_names(notes)

        cluster_data.append({
            "name": name,
            "color": palette[i % len(palette)],
            "cx": float(x[mask].mean()),
            "cy": float(y[mask].mean()),
            "notes": notes,
            "note_names": note_names,
            "n_cells": int(mask.sum()),
        })

    # Build JSON for JS
    import json
    points_js = json.dumps([
        {"x": float(x[i]), "y": float(y[i]),
         "c": int(codes[i]), "p0": int(pitches_d0[i]), "p1": int(pitches_d1[i])}
        for i in range(len(x))
    ])
    clusters_js = json.dumps(cluster_data)
    palette_js = json.dumps(palette)
    names_js = json.dumps(names)

    # Load and fill HTML template
    template = (_HTML_DIR / "interactive.html").read_text()
    html = template.replace("{{POINTS}}", points_js)
    html = html.replace("{{CLUSTERS}}", clusters_js)
    html = html.replace("{{PALETTE}}", palette_js)
    html = html.replace("{{NAMES}}", names_js)
    html = html.replace("{{INVERT_Y}}", "true" if invert_y else "false")
    html = html.replace("{{POINT_SIZE}}", str(point_size))
    html = html.replace("{{WIDTH}}", str(width))
    html = html.replace("{{HEIGHT}}", str(height))
    html = html.replace("{{EMB_KEY}}", emb_key)
    html = html.replace("{{D0}}", str(d0))
    html = html.replace("{{D1}}", str(d1))
    html = html.replace("{{PLAY_AXIS}}", f'"{play_axis}"')

    return HTML(html)


def _midi_to_note_names(midi_notes):
    """Convert list of MIDI notes to compact note name summary."""
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    names = []
    for m in midi_notes:
        note = NOTE_NAMES[m % 12]
        octave = (m // 12) - 1
        names.append(f"{note}{octave}")
    return names
