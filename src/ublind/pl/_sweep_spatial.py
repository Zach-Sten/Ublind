"""
Spatial sweep — replay cluster sounds triggered by tissue position.

Uses the cluster chords defined by compose_clusters(), but
triggers them based on when the sweep line passes each cell's
physical location in the tissue.

Requires:
1. ub.pp.compose_clusters() to define the sounds
2. adata.obsm['spatial'] (or other spatial key) for coordinates
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ublind._core.notes import NoteEvent, map_to_pitches, LOWEST_NOTE, HIGHEST_NOTE
from ublind.pl._utils import get_ublind_uns


def sweep_spatial(
    adata,
    *,
    spatial_key: str = "spatial",
    sweep_axis: str = "x",
    color_by: Optional[str] = None,
    legend_loc: str = "on data",
    subsample: Optional[int] = 3000,
    figsize: tuple[float, float] = (8, 8),
    point_size: float = 4.0,
    alpha: float = 0.6,
    n_frames: int = 100,
    line_color: str = "red",
    line_width: float = 1.5,
    invert_y: bool = True,
    save: Optional[str] = None,
    dpi: int = 80,
    seed: int = 0,
):
    """
    Sweep across tissue space, playing cluster sounds as the line
    passes each cell.

    Must call ``ub.pp.compose_clusters()`` first to define what
    each cluster sounds like. This function re-renders the audio
    with spatial timing and creates a video of the sweep across
    real tissue coordinates.

    Parameters
    ----------
    adata : AnnData
        Must have spatial coordinates and cluster preprocessing.
    spatial_key : str
        Key in ``adata.obsm`` for spatial coordinates.
    sweep_axis : str
        ``"x"`` (left→right), ``"y"`` (bottom→top), or ``"both"``.
    color_by : str, optional
        Column in ``adata.obs`` to color by. Defaults to the
        cluster_key used in compose_clusters.
    legend_loc : str
        ``"on data"``, ``"right margin"``, or ``"none"``.
    subsample : int, optional
        Subsample to this many points for speed. Default 3000.
        Set to ``None`` to use all points.
    invert_y : bool
        Invert Y axis (standard for imaging coordinates).
    save : str, optional
        Save MP4 to this path.

    Returns
    -------
    IPython.display.Video
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    ub = get_ublind_uns(adata)

    if ub.get("mode") != "clusters":
        raise RuntimeError(
            "Run ub.pp.compose_clusters() first to define cluster sounds."
        )

    if spatial_key not in adata.obsm:
        raise KeyError(
            f"'{spatial_key}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )

    cluster_key = ub["cluster_key"]
    cluster_info = ub["cluster_info"]
    cluster_order = ub["cluster_order"]
    instruments_info = ub["instruments"]
    total_time = ub["time_sec"]
    scale = ub.get("scale")
    root = ub.get("root", 0)

    if color_by is None:
        color_by = cluster_key

    # Get spatial coords
    spatial_coords = np.array(adata.obsm[spatial_key], dtype=float)
    if spatial_coords.ndim == 2 and spatial_coords.shape[1] > 2:
        spatial_coords = spatial_coords[:, :2]

    # Match to compose_clusters subsample if it exists
    subsample_idx = ub.get("subsample_idx")
    if subsample_idx is not None:
        spatial_coords = spatial_coords[subsample_idx]

    # Further subsample for speed
    n_total = len(spatial_coords)
    if subsample is not None and subsample < n_total:
        rng = np.random.default_rng(seed)
        spatial_sub_idx = np.sort(rng.choice(n_total, subsample, replace=False))
        spatial_coords = spatial_coords[spatial_sub_idx]
    else:
        spatial_sub_idx = None

    n_points = len(spatial_coords)
    sx, sy = spatial_coords[:, 0], spatial_coords[:, 1]

    # Get cluster labels — apply both subsample layers
    labels = adata.obs[cluster_key].astype("category")
    if subsample_idx is not None:
        labels = labels.iloc[subsample_idx]
    else:
        labels = labels.iloc[:n_total]
    if spatial_sub_idx is not None:
        labels = labels.iloc[spatial_sub_idx]
    cat_names = list(labels.cat.categories)
    codes = labels.cat.codes.values

    print(f"ublind: spatial sweep with {n_points} points")

    # Build cluster name → notes mapping from cluster_info
    cluster_notes = {}
    for ci in cluster_info:
        cluster_notes[ci["name"]] = ci["notes"]

    # Re-render audio with spatial timing
    events = _build_spatial_events(
        sx, sy, codes, cat_names, cluster_order, cluster_notes,
        instruments_info, total_time, sweep_axis, scale, root,
    )

    # Re-render audio
    from ublind._core.renderers import SynthRenderer
    renderer = SynthRenderer()
    wav_bytes = renderer.render_bytes(events, total_time, ub.get("tempo_bpm", 120))

    # Store spatial wav
    adata.uns["ublind"]["spatial_wav"] = wav_bytes

    fps = max(1, round(n_frames / total_time))

    # Resolve colors
    colors, categorical, color_names, color_palette = _resolve_colors(
        adata, ub, color_by, subsample_idx, n_points
    )

    # Build figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="datalim")

    if categorical:
        for i, name in enumerate(color_names):
            mask = colors == i
            c = color_palette[i] if color_palette else f"C{i % 10}"
            ax.scatter(sx[mask], sy[mask], c=c, s=point_size,
                       alpha=alpha * 0.3, edgecolors="none", label=name)
        if legend_loc == "on data":
            _add_on_data_labels(ax, sx, sy, colors, color_names, color_palette)
        elif legend_loc != "none":
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                      fontsize=7, markerscale=2, frameon=False)
    else:
        ax.scatter(sx, sy, c=colors, cmap="viridis", s=point_size,
                   alpha=alpha * 0.3, edgecolors="none")

    if invert_y:
        ax.invert_yaxis()

    ax.set_xlabel(f"{spatial_key} X")
    ax.set_ylabel(f"{spatial_key} Y")
    title_text = ax.set_title("ublind spatial sweep", fontsize=13)

    # Sweep lines
    sweep_lines = []
    if sweep_axis in ("x", "both"):
        vline = ax.axvline(sx.min(), color=line_color, linewidth=line_width, alpha=0.8)
        sweep_lines.append(("x", vline))
    if sweep_axis in ("y", "both"):
        lc = "blue" if sweep_axis == "both" else line_color
        hline = ax.axhline(sy.min(), color=lc, linewidth=line_width, alpha=0.8)
        sweep_lines.append(("y", hline))

    # Highlight layer
    hl = ax.scatter([], [], s=point_size * 3, edgecolors="k",
                    linewidths=0.3, zorder=10)

    x_min, x_max = sx.min(), sx.max()
    y_min, y_max = sy.min(), sy.max()
    x_positions = np.linspace(x_min, x_max, n_frames)
    y_positions = np.linspace(y_min, y_max, n_frames)
    x_window = (x_max - x_min) / n_frames * 3
    y_window = (y_max - y_min) / n_frames * 3

    def update(frame):
        mask = np.zeros(n_points, dtype=bool)

        for axis, line in sweep_lines:
            if axis == "x":
                xpos = x_positions[frame]
                line.set_xdata([xpos, xpos])
                mask |= np.abs(sx - xpos) < x_window
            else:
                ypos = y_positions[frame]
                line.set_ydata([ypos, ypos])
                mask |= np.abs(sy - ypos) < y_window

        if mask.any():
            hl.set_offsets(np.column_stack([sx[mask], sy[mask]]))
            if categorical:
                hl_colors = [color_palette[int(colors[j]) % len(color_palette)]
                             for j in np.where(mask)[0]]
                hl.set_facecolors(hl_colors)
            else:
                hl.set_facecolors("gold")

            # Show active cluster names in title
            active_codes = set(codes[mask])
            active_names = [cat_names[c] for c in active_codes if c < len(cat_names)]
            if active_names:
                title_text.set_text(f"♪ {', '.join(active_names[:3])}")
        else:
            hl.set_offsets(np.empty((0, 2)))
            title_text.set_text("ublind spatial sweep")

        return [l for _, l in sweep_lines] + [hl, title_text]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=int(total_time / n_frames * 1000), blit=False,
    )
    plt.close(fig)

    # Render MP4
    tmp_dir = tempfile.mkdtemp(prefix="ublind_")
    silent_mp4 = Path(tmp_dir) / "silent.mp4"
    final_mp4 = Path(tmp_dir) / "ublind_spatial.mp4"

    print("ublind: rendering spatial animation...")
    anim.save(str(silent_mp4), writer="ffmpeg", fps=fps, dpi=dpi,
              extra_args=["-pix_fmt", "yuv420p"])

    # Mux with spatial audio
    wav_path = Path(tmp_dir) / "audio.wav"
    wav_path.write_bytes(wav_bytes)
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(silent_mp4), "-i", str(wav_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest", "-movflags", "+faststart",
        str(final_mp4),
    ], capture_output=True)
    print("ublind: muxed audio into video")

    if save:
        import shutil
        shutil.copy2(str(final_mp4), save)
        print(f"ublind: saved → {save}")

    from IPython.display import Video
    return Video(str(final_mp4), embed=True, mimetype="video/mp4")


def _build_spatial_events(
    sx, sy, codes, cat_names, cluster_order, cluster_notes,
    instruments_info, total_time, sweep_axis, scale, root,
):
    """
    Build NoteEvents triggered by spatial position.

    As the sweep line moves across tissue, each cell triggers
    a note from its cluster's chord. The specific note is chosen
    based on the cell's position on the perpendicular axis,
    so spatial structure maps to pitch within each chord.
    """
    events = []
    n_points = len(sx)

    def _emit_events_for_axis(positions, perp_positions, ch_offset=0):
        axis_events = []
        order = np.argsort(positions)
        p_min, p_max = positions.min(), positions.max()
        span = p_max - p_min
        if span == 0:
            span = 1.0

        perp_min = perp_positions.min()
        perp_range = perp_positions.max() - perp_min
        if perp_range == 0:
            perp_range = 1.0

        # Instrument for this axis
        inst_idx = min(ch_offset, len(instruments_info) - 1)
        _, prog, _ = instruments_info[inst_idx]
        ch = inst_idx if inst_idx < 9 else inst_idx + 1
        ch = ch % 16
        if ch == 9:
            ch = 10

        note_dur = max(0.05, total_time / n_points * 2)

        for idx in order:
            t = (positions[idx] - p_min) / span * total_time
            cluster_name = cat_names[codes[idx]]

            notes = cluster_notes.get(cluster_name, [])
            if not notes:
                continue

            # Pick note from chord based on perpendicular position
            perp_norm = (perp_positions[idx] - perp_min) / perp_range
            note_idx = int(perp_norm * (len(notes) - 1))
            note_idx = max(0, min(note_idx, len(notes) - 1))
            pitch = notes[note_idx]

            if LOWEST_NOTE <= pitch <= HIGHEST_NOTE:
                axis_events.append(NoteEvent(
                    time=float(t), duration=note_dur, pitch=pitch,
                    velocity=90, instrument=prog, channel=ch,
                ))

        return axis_events

    if sweep_axis in ("x", "both"):
        events.extend(_emit_events_for_axis(sx, sy, ch_offset=0))

    if sweep_axis in ("y", "both"):
        events.extend(_emit_events_for_axis(sy, sx, ch_offset=1 if sweep_axis == "both" else 0))

    return events


def _resolve_colors(adata, ub, color_by, subsample_idx, n_points):
    """Get colors from adata.obs."""
    if color_by is not None and color_by in adata.obs.columns:
        col = adata.obs[color_by]
        if subsample_idx is not None:
            col = col.iloc[subsample_idx]
        else:
            col = col.iloc[:n_points]

        if hasattr(col, "cat") or col.dtype == object:
            cat = col.astype("category")
            codes = cat.cat.codes.values
            names = list(cat.cat.categories)
            color_key = f"{color_by}_colors"
            if color_key in adata.uns:
                cat_colors = list(adata.uns[color_key])
            else:
                cat_colors = [f"C{i % 10}" for i in range(len(names))]
            return codes, True, names, cat_colors
        else:
            return col.values.astype(float), False, None, None

    return np.zeros(n_points), False, None, None


def _add_on_data_labels(ax, x, y, codes, names, colors):
    """Place category labels at centroids."""
    for i, name in enumerate(names):
        mask = codes == i
        if mask.any():
            cx, cy = x[mask].mean(), y[mask].mean()
            c = colors[i] if colors is not None else "black"
            ax.annotate(
                name, (cx, cy),
                fontsize=7, fontweight="bold", ha="center", va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor="white",
                    edgecolor=c, alpha=0.8, linewidth=0.5,
                ),
            )
