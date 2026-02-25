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
    cmap: str = "viridis",
    figsize: tuple[float, float] = (10, 7),
    point_size: float = 4.0,
    alpha: float = 0.6,
    n_frames: int = 100,
    line_color: str = "red",
    line_width: float = 1.5,
    save: Optional[str] = None,
    dpi: int = 80,
):
    """
    Animated sweep-line with audio, displayed as an inline MP4.

    Renders the animation, muxes it with stored WAV audio via ffmpeg,
    and displays the result using ``IPython.display.Video``.

    Requires ffmpeg: ``conda install -c conda-forge ffmpeg``

    Parameters
    ----------
    save : str, optional
        Also save the final MP4 to this path.
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
    pitches = ub["pitches"]
    total_time = ub["time_sec"]
    wav_bytes = ub.get("wav")

    x = coords[:, dims[0]]
    y = coords[:, dims[1]]
    c = pitches.mean(axis=1)
    fps = max(1, round(n_frames / total_time))

    # Build animation
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        x, y, c=c, cmap=cmap, s=point_size, alpha=alpha * 0.4, edgecolors="none",
    )
    ax.set_xlabel(f"{ub['embedding']} {dims[0]+1}")
    ax.set_ylabel(f"{ub['embedding']} {dims[1]+1}")
    ax.set_title(f"ublind sweep — {ub['embedding']}")

    vline = ax.axvline(x.min(), color=line_color, linewidth=line_width, alpha=0.8)
    highlight = ax.scatter(
        [], [], c=[], cmap=cmap, s=point_size * 3, edgecolors="k", linewidths=0.3,
    )

    x_min, x_max = x.min(), x.max()
    x_positions = np.linspace(x_min, x_max, n_frames)
    x_window = (x_max - x_min) / n_frames * 3

    def update(frame):
        xpos = x_positions[frame]
        vline.set_xdata([xpos, xpos])
        mask = np.abs(x - xpos) < x_window
        if mask.any():
            highlight.set_offsets(np.column_stack([x[mask], y[mask]]))
            highlight.set_array(c[mask])
        else:
            highlight.set_offsets(np.empty((0, 2)))
        return vline, highlight

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=int(total_time / n_frames * 1000), blit=True,
    )
    plt.close(fig)

    # Render to files
    tmp_dir = tempfile.mkdtemp(prefix="ublind_")
    silent_mp4 = Path(tmp_dir) / "silent.mp4"
    final_mp4 = Path(tmp_dir) / "ublind_sweep.mp4"

    print("ublind: rendering animation frames...")
    anim.save(
        str(silent_mp4), writer="ffmpeg", fps=fps, dpi=dpi,
        extra_args=["-pix_fmt", "yuv420p"],
    )

    # Mux with audio if available
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

    # Save copy if requested
    if save:
        import shutil
        shutil.copy2(str(final_mp4), save)
        print(f"ublind: saved → {save}")

    # Display inline
    from IPython.display import Video
    return Video(str(final_mp4), embed=True, mimetype="video/mp4")
