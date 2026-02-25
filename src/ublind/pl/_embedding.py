"""Static embedding scatter plot coloured by pitch / time / voice."""

from __future__ import annotations

from typing import Optional

from ublind.pl._utils import get_ublind_uns


def embedding(
    adata,
    *,
    color_by: str = "pitch",
    dims: tuple[int, int] = (0, 1),
    cmap: str = "viridis",
    figsize: tuple[float, float] = (10, 7),
    point_size: float = 4.0,
    alpha: float = 0.6,
    show_instruments: bool = True,
    title: Optional[str] = None,
    save: Optional[str] = None,
    ax=None,
):
    """
    Scatter plot of the embedding coloured by MIDI pitch, time, or voice.

    Parameters
    ----------
    adata : AnnData
        Must have been preprocessed with ``ub.pp.preprocess()``.
    color_by : str
        ``"pitch"`` (mean across voices), ``"time"``, or ``"voice_N"``.
    dims : tuple
        Which 2 columns of the original embedding to plot as x, y.
    cmap : str
        Matplotlib colourmap.
    show_instruments : bool
        Annotate with instrument names.
    save : str, optional
        Save figure to this path.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    ub = get_ublind_uns(adata)
    coords = ub["coords"]
    pitches = ub["pitches"]
    time_values = ub["time_values"]
    instruments = ub["instruments"]

    x = coords[:, dims[0]]
    y = coords[:, dims[1]]

    # Determine colour values
    c, clabel = _resolve_color(color_by, pitches, time_values, instruments)

    # Plot
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sc = ax.scatter(
        x, y, c=c, cmap=cmap, s=point_size, alpha=alpha, edgecolors="none",
    )
    fig.colorbar(sc, ax=ax, label=clabel, shrink=0.8)

    emb_name = ub["embedding"]
    ax.set_xlabel(f"{emb_name} {dims[0] + 1}")
    ax.set_ylabel(f"{emb_name} {dims[1] + 1}")
    ax.set_title(title or f"ublind — {emb_name}")

    if show_instruments and instruments:
        inst_text = "  |  ".join(f"Dim {v}: {name}" for v, _, name in instruments)
        ax.annotate(
            inst_text,
            xy=(0.5, -0.08),
            xycoords="axes fraction",
            ha="center",
            fontsize=8,
            color="grey",
        )

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    if own_fig:
        plt.tight_layout()

    return fig


def _resolve_color(color_by, pitches, time_values, instruments):
    """Return (color_array, label) based on color_by string."""
    if color_by == "time":
        return time_values, "Time (s)"
    elif color_by.startswith("voice_"):
        v = int(color_by.split("_")[1])
        if v >= pitches.shape[1]:
            raise ValueError(
                f"voice_{v} out of range (have {pitches.shape[1]} voices)"
            )
        _, _, name = instruments[v]
        return pitches[:, v], f"Pitch — {name}"
    else:
        # Default: mean pitch
        return pitches.mean(axis=1), "Mean MIDI pitch"
