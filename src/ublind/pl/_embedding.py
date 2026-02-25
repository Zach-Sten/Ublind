"""Static embedding scatter plot."""

from __future__ import annotations

from typing import Optional

from ublind.pl._utils import get_ublind_uns
import numpy as np


def embedding(
    adata,
    *,
    color_by: str = "pitch",
    dims: tuple[int, int] = (0, 1),
    legend_loc: str = "right margin",
    cmap: str = "viridis",
    figsize: tuple[float, float] = (8, 8),
    point_size: float = 4.0,
    alpha: float = 0.6,
    show_instruments: bool = True,
    title: Optional[str] = None,
    save: Optional[str] = None,
    ax=None,
):
    """
    Scatter plot of the embedding.

    Parameters
    ----------
    adata : AnnData
        Must have been preprocessed with ``ub.pp.preprocess()``.
    color_by : str
        ``"pitch"`` (mean across voices), or a column in ``adata.obs``
        (e.g. ``"cell_type"``).
    dims : tuple
        Which 2 columns of the embedding to plot.
    legend_loc : str
        ``"right margin"``, ``"on data"``, or ``"none"``.
    save : str, optional
        Save figure to this path.

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    ub = get_ublind_uns(adata)
    coords = ub["coords"]
    instruments = ub["instruments"]

    d0, d1 = dims
    x = coords[:, d0]
    y = coords[:, d1]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.set_aspect("equal", adjustable="datalim")

    # Check if color_by is an obs column
    if color_by != "pitch" and color_by in adata.obs.columns:
        colors, categorical, cat_names, cat_colors = _resolve_obs_colors(
            adata, ub, color_by
        )
        if categorical:
            for i, name in enumerate(cat_names):
                mask = colors == i
                c = cat_colors[i] if cat_colors is not None else f"C{i % 10}"
                ax.scatter(
                    x[mask], y[mask], c=c, s=point_size,
                    alpha=alpha, edgecolors="none", label=name,
                )
            if legend_loc == "on data":
                _add_on_data_labels(ax, x, y, colors, cat_names, cat_colors)
            elif legend_loc != "none":
                ax.legend(
                    loc="center left", bbox_to_anchor=(1.02, 0.5),
                    fontsize=7, markerscale=2, frameon=False,
                )
        else:
            sc = ax.scatter(
                x, y, c=colors, cmap=cmap, s=point_size,
                alpha=alpha, edgecolors="none",
            )
            fig.colorbar(sc, ax=ax, label=color_by, shrink=0.8)
    else:
        # Default: mean pitch
        pitches = []
        for d in range(ub["n_dims"]):
            p = np.zeros(len(coords))
            p[ub["order_per_dim"][d]] = ub["pitches_per_dim"][d]
            pitches.append(p)
        c = np.mean(pitches, axis=0)
        sc = ax.scatter(
            x, y, c=c, cmap=cmap, s=point_size,
            alpha=alpha, edgecolors="none",
        )
        fig.colorbar(sc, ax=ax, label="Mean MIDI pitch", shrink=0.8)

    emb_name = ub["embedding"]
    inst_x = instruments[d0][2] if d0 < len(instruments) else f"dim {d0}"
    inst_y = instruments[d1][2] if d1 < len(instruments) else f"dim {d1}"
    ax.set_xlabel(f"{emb_name} {d0+1} — {inst_x}")
    ax.set_ylabel(f"{emb_name} {d1+1} — {inst_y}")
    ax.set_title(title or f"ublind — {emb_name}")

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    if own_fig:
        plt.tight_layout()

    return fig


def _resolve_obs_colors(adata, ub, color_by):
    """Get colors from adata.obs column."""
    subsample_idx = ub.get("subsample_idx")
    col = adata.obs[color_by]
    if subsample_idx is not None:
        col = col.iloc[subsample_idx]
    else:
        col = col.iloc[:len(ub["coords"])]

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
