"""Animated sweep-line visualisation with audio baked into MP4."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ublind.pl._utils import get_ublind_uns


def sweep(
    adata,
    *,
    dims: tuple[int, int] = (0, 1),
    color_by: Optional[str] = None,
    legend_loc: str = "right margin",
    cmap: str = "viridis",
    figsize: tuple[float, float] = (8, 8),
    point_size: float = 4.0,
    alpha: float = 0.6,
    n_frames: int = 100,
    line_color_x: str = "red",
    line_color_y: str = "blue",
    line_width: float = 1.5,
    save: Optional[str] = None,
    dpi: int = 80,
):
    """
    Animated sweep with two perpendicular lines (one per dimension).

    The vertical line sweeps left→right (dim 0) and the horizontal
    line sweeps bottom→top (dim 1), each representing its instrument's
    time axis. Points highlight as each sweep passes them.

    Parameters
    ----------
    color_by : str, optional
        Column in ``adata.obs`` to color by (e.g. ``"cell_type"``).
        If ``None``, colors by mean pitch.
    legend_loc : str
        ``"right margin"``, ``"on data"``, or ``"none"``.
    save : str, optional
        Save the final MP4 to this path.
    dpi : int
        Resolution of animation frames.

    Returns
    -------
    IPython.display.Video
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    ub = get_ublind_uns(adata)
    coords = ub["coords"]
    total_time = ub["time_sec"]
    instruments = ub["instruments"]
    wav_bytes = ub.get("wav")

    d0, d1 = dims
    x = coords[:, d0]
    y = coords[:, d1]
    fps = max(1, round(n_frames / total_time))

    # Resolve colors
    colors, categorical, cat_names, cat_colors = _resolve_colors(
        adata, ub, color_by, cmap
    )

    # Build figure — square aspect
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="datalim")

    if categorical:
        # Plot each category separately for legend
        for i, name in enumerate(cat_names):
            mask = colors == i
            c = cat_colors[i] if cat_colors is not None else f"C{i % 10}"
            ax.scatter(
                x[mask], y[mask], c=c, s=point_size,
                alpha=alpha * 0.4, edgecolors="none", label=name,
            )
        if legend_loc == "on data":
            _add_on_data_labels(ax, x, y, colors, cat_names, cat_colors)
        elif legend_loc != "none":
            ax.legend(
                loc="center left", bbox_to_anchor=(1.02, 0.5),
                fontsize=7, markerscale=2, frameon=False,
            )
    else:
        ax.scatter(
            x, y, c=colors, cmap=cmap, s=point_size,
            alpha=alpha * 0.4, edgecolors="none",
        )

    emb = ub["embedding"]
    inst_x = instruments[d0][2] if d0 < len(instruments) else f"dim {d0}"
    inst_y = instruments[d1][2] if d1 < len(instruments) else f"dim {d1}"
    ax.set_xlabel(f"{emb} {d0+1} — {inst_x}")
    ax.set_ylabel(f"{emb} {d1+1} — {inst_y}")
    ax.set_title(f"ublind sweep — {emb}")

    # Sweep lines
    vline = ax.axvline(x.min(), color=line_color_x, linewidth=line_width, alpha=0.8)
    hline = ax.axhline(y.min(), color=line_color_y, linewidth=line_width, alpha=0.8)

    # Highlight scatter (on top)
    hl = ax.scatter(
        [], [], s=point_size * 3, c="gold",
        edgecolors="k", linewidths=0.3, zorder=10,
    )

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_positions = np.linspace(x_min, x_max, n_frames)
    y_positions = np.linspace(y_min, y_max, n_frames)
    x_window = (x_max - x_min) / n_frames * 3
    y_window = (y_max - y_min) / n_frames * 3

    def update(frame):
        xpos = x_positions[frame]
        ypos = y_positions[frame]
        vline.set_xdata([xpos, xpos])
        hline.set_ydata([ypos, ypos])
        # Highlight points near either sweep line
        mask_x = np.abs(x - xpos) < x_window
        mask_y = np.abs(y - ypos) < y_window
        mask = mask_x | mask_y
        if mask.any():
            hl.set_offsets(np.column_stack([x[mask], y[mask]]))
        else:
            hl.set_offsets(np.empty((0, 2)))
        return vline, hline, hl

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=int(total_time / n_frames * 1000), blit=True,
    )
    plt.close(fig)

    # Render MP4
    tmp_dir = tempfile.mkdtemp(prefix="ublind_")
    silent_mp4 = Path(tmp_dir) / "silent.mp4"
    final_mp4 = Path(tmp_dir) / "ublind_sweep.mp4"

    print("ublind: rendering animation frames...")
    anim.save(
        str(silent_mp4), writer="ffmpeg", fps=fps, dpi=dpi,
        extra_args=["-pix_fmt", "yuv420p"],
    )

    # Mux with audio
    if wav_bytes is not None:
        wav_path = Path(tmp_dir) / "audio.wav"
        wav_path.write_bytes(wav_bytes)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(silent_mp4),
                "-i", str(wav_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "128k",
                "-shortest",
                "-movflags", "+faststart",
                str(final_mp4),
            ],
            capture_output=True,
        )
        print("ublind: muxed audio into video")
    else:
        final_mp4 = silent_mp4

    if save:
        import shutil
        shutil.copy2(str(final_mp4), save)
        print(f"ublind: saved → {save}")

    from IPython.display import Video
    return Video(str(final_mp4), embed=True, mimetype="video/mp4")


def _resolve_colors(adata, ub, color_by, cmap):
    """
    Return (color_array, is_categorical, category_names, category_colors).
    """
    subsample_idx = ub.get("subsample_idx")

    if color_by is not None and color_by in adata.obs.columns:
        col = adata.obs[color_by]
        if subsample_idx is not None:
            col = col.iloc[subsample_idx]
        else:
            col = col.iloc[:len(ub["coords"])]

        if hasattr(col, "cat") or col.dtype == object:
            # Categorical
            cat = col.astype("category")
            codes = cat.cat.codes.values
            names = list(cat.cat.categories)
            # Try to get scanpy colors
            color_key = f"{color_by}_colors"
            if color_key in adata.uns:
                cat_colors = list(adata.uns[color_key])
            else:
                cat_colors = [f"C{i % 10}" for i in range(len(names))]
            return codes, True, names, cat_colors
        else:
            return col.values.astype(float), False, None, None

    # Default: mean pitch across dims
    pitches = []
    for d in range(ub["n_dims"]):
        p = np.zeros(len(ub["coords"]))
        p[ub["order_per_dim"][d]] = ub["pitches_per_dim"][d]
        pitches.append(p)
    mean_pitch = np.mean(pitches, axis=0)
    return mean_pitch, False, None, None


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
