"""Deterministic gather-based assembly.

Scattering element forces to nodes has two shapes.  `np.bincount` (the NumPy
path) walks the element list and accumulates; it is deterministic because the
index order is fixed, but it is a single-threaded pass over 8x the mesh.

This is the other shape: invert the connectivity once, then let each NODE sum
its own contributions.  Nothing is written twice, so it parallelises without
atomics and stays bit-for-bit reproducible -- which is what makes the eventual
CPU-vs-GPU comparison a real check rather than a tolerance argument.

It also stopped being optional.  Once the fused kernel landed, bincount was
44% of a reduced-integration time step: the assembly had become the bottleneck
by standing still while everything around it got 8x faster.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


def inverse_connectivity(conn: np.ndarray, n_nodes: int):
    """Return (order, offsets) mapping each node to its element-corner slots.

    `order[offsets[n]:offsets[n+1]]` lists the flat indices e*8+a whose node is n.
    """
    flat = np.ascontiguousarray(conn, dtype=np.int64).ravel()
    order = np.argsort(flat, kind="stable").astype(np.int64)
    offsets = np.searchsorted(flat[order], np.arange(n_nodes + 1)).astype(np.int64)
    return order, offsets


@njit(cache=True, fastmath=True, parallel=True, nogil=True)
def gather_add(values, order, offsets, out):
    """out[n] = sum of values[i] over the slots i belonging to node n."""
    v = values.reshape(-1, 3)
    for n in prange(out.shape[0]):
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for p in range(offsets[n], offsets[n + 1]):
            i = order[p]
            s0 += v[i, 0]
            s1 += v[i, 1]
            s2 += v[i, 2]
        out[n, 0] = s0
        out[n, 1] = s1
        out[n, 2] = s2


class GatherAssembler:
    def __init__(self, conn: np.ndarray, n_nodes: int):
        self.order, self.offsets = inverse_connectivity(conn, n_nodes)
        self.out = np.zeros((n_nodes, 3))

    def __call__(self, f_e: np.ndarray) -> np.ndarray:
        gather_add(f_e, self.order, self.offsets, self.out)
        return self.out
