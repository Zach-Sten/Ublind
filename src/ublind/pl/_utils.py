"""Shared utilities for the pl submodule."""

from __future__ import annotations


def get_ublind_uns(adata) -> dict:
    """Retrieve ublind data from adata.uns, with a clear error if missing."""
    if "ublind" not in adata.uns:
        raise RuntimeError(
            "No ublind data found. Run ub.pp.preprocess(adata) first."
        )
    return adata.uns["ublind"]
