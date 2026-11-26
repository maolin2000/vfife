"""Assembling element vectors onto nodes, reproducibly (1.22: from
`core.mesh`, which still exports it)."""
from __future__ import annotations

import numpy as np


def scatter_add(values: np.ndarray, conn: np.ndarray, n_nodes: int) -> np.ndarray:
    """Assemble (n_elem, 8, 3) element vectors into (n_nodes, 3).

    Uses bincount, whose accumulation order is fixed by the index array, so the
    result is bit-for-bit reproducible run to run -- which is what makes the
    later CPU-vs-GPU comparison meaningful.
    """
    flat = conn.ravel()
    out = np.empty((n_nodes, 3), dtype=values.dtype)
    v = values.reshape(-1, 3)
    for c in range(3):
        out[:, c] = np.bincount(flat, weights=v[:, c], minlength=n_nodes)
    return out
