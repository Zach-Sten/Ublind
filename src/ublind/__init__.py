"""
ublind — Exploring high-dimensional data with sound.

Turn embeddings (UMAP, PCA, t-SNE, …) stored in AnnData objects into
multi-instrument music.

    import ublind as ub

    ub.pp.preprocess(adata, embedding="X_umap", time=10)
    ub.pl.embedding(adata)
    ub.tl.render(adata, "output.wav")
"""

from __future__ import annotations

from ublind import pp, pl, tl
from ublind._core.instruments import GENERAL_MIDI_INSTRUMENTS

__version__ = "0.1.0"
__all__ = ["pp", "pl", "tl", "GENERAL_MIDI_INSTRUMENTS"]
