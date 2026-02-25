"""Animated cluster-by-cluster visualisation with audio."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ublind.pl._utils import get_ublind_uns


def sweep_clusters(
    adata,
    *,
    dims: tuple[int, int] = (0, 1),
    figsize: tuple[float, float] = (8, 8),
    point_size: float = 4.0,
    alpha_bg: float = 0.1,
    alpha_active: float = 0.8,
    n_frames: int = 100,
    legend_loc: str = "on data",
    save: Optional[str] = None,
    dpi: int = 80,
):
    """
    Animated cluster-by-cluster playback.

    Each cluster lights up in sequence as its audio plays.
    Requires ``ub.pp.preprocess_clusters()`` to have been run.

    Parameters
    ----------
    save : str, optional
        Save the MP4 to this path.

    Returns
    -------
    IPython.display.Video
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    ub = get_ublind_uns(adata)

    if ub.get("mode") != "clusters":
        raise RuntimeError(
            "Run ub.pp.preprocess_clusters() first (not ub.pp.preprocess)."
        )

    coords = ub["coords"]
    total_time = ub["time_sec"]
    cluster_order = ub["cluster_order"]
    cluster_info = ub["cluster_info"]
    cluster_key = ub["cluster_key"]
    instruments = ub["instruments"]
    wav_bytes = ub.get("wav")

    # Get codes from adata.obs
    labels = adata.obs[cluster_key].astype("category")
    subsample_idx = ub.get("subsample_idx")
    if subsample_idx is not None:
        labels = labels.iloc[subsample_idx]
    else:
        labels = labels.iloc[:len(coords)]
    cat_names = list(labels.cat.categories)
    codes = labels.cat.codes.values

    d0, d1 = dims
    x = coords[:, d0]
    y = coords[:, d1]
    fps = max(1, round(n_frames / total_time))

    # Colors
    color_key = f"{cluster_key}_colors"
    if color_key in adata.uns:
        palette = list(adata.uns[color_key])
    else:
        palette = [f"C{i % 10}" for i in range(len(cat_names))]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="datalim")
    emb = ub["embedding"]
    inst_x = instruments[d0][2] if d0 < len(instruments) else f"dim {d0}"
    inst_y = instruments[d1][2] if d1 < len(instruments) else f"dim {d1}"
    ax.set_xlabel(f"{emb} {d0+1} — {inst_x}")
    ax.set_ylabel(f"{emb} {d1+1} — {inst_y}")

    # Pre-draw all points very faint
    for i, name in enumerate(cat_names):
        mask = codes == i
        c = palette[i % len(palette)]
        ax.scatter(x[mask], y[mask], c=c, s=point_size,
                   alpha=alpha_bg, edgecolors="none")

    # Cluster label text object
    title_text = ax.set_title("", fontsize=14, fontweight="bold")

    # Active highlight layer
    hl = ax.scatter([], [], s=point_size * 2, c="gold",
                    edgecolors="k", linewidths=0.3, zorder=10)

    # On-data labels (always visible but dim)
    if legend_loc == "on data":
        for i, name in enumerate(cat_names):
            mask = codes == i
            if mask.any():
                cx, cy = x[mask].mean(), y[mask].mean()
                c = palette[i % len(palette)]
                ax.annotate(
                    name, (cx, cy),
                    fontsize=7, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=c, alpha=0.7, linewidth=0.5),
                    zorder=5,
                )

    # Time → cluster mapping
    frame_times = np.linspace(0, total_time, n_frames, endpoint=False)

    def get_active_cluster(t):
        for ci in cluster_info:
            if ci["t_start"] <= t < ci["t_end"]:
                return ci["name"]
        return None

    def update(frame):
        t = frame_times[frame]
        active = get_active_cluster(t)

        if active is not None:
            cat_idx = cat_names.index(active)
            mask = codes == cat_idx
            hl.set_offsets(np.column_stack([x[mask], y[mask]]))
            c = palette[cat_idx % len(palette)]
            hl.set_facecolors(c)
            title_text.set_text(f"♪ {active}")
            title_text.set_color(c)
        else:
            hl.set_offsets(np.empty((0, 2)))
            title_text.set_text("")

        return hl, title_text

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=int(total_time / n_frames * 1000), blit=False,
    )
    plt.close(fig)

    # Render MP4
    tmp_dir = tempfile.mkdtemp(prefix="ublind_")
    silent_mp4 = Path(tmp_dir) / "silent.mp4"
    final_mp4 = Path(tmp_dir) / "ublind_clusters.mp4"

    print("ublind: rendering cluster animation...")
    anim.save(
        str(silent_mp4), writer="ffmpeg", fps=fps, dpi=dpi,
        extra_args=["-pix_fmt", "yuv420p"],
    )

    if wav_bytes is not None:
        wav_path = Path(tmp_dir) / "audio.wav"
        wav_path.write_bytes(wav_bytes)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(silent_mp4),
                "-i", str(wav_path),
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                "-shortest", "-movflags", "+faststart",
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
