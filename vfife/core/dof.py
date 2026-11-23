"""Unified nodal-DOF layout and deterministic mixed-element assembly.

The layout is the single mapping between physical component names and global
equation numbers.  Elements ask it for maps; they never infer ``stride*node``.
This keeps three-DOF solids/trusses and six-DOF frames compatible without
inventing rotational inertia for nodes that do not own rotations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse


TRANSLATION = ("ux", "uy", "uz")
ROTATION = ("rx", "ry", "rz")


def repeated_rotation(rotation: np.ndarray, blocks: int = 4) -> np.ndarray:
    """Dense block transform for two-node translations/rotations.

    This replaces SciPy's changing sparse ``block_diag`` array/matrix API on a
    fixed 12x12 hot-path object.
    """
    rotation = np.asarray(rotation)
    if rotation.shape != (3, 3) or blocks < 1:
        raise ValueError("repeated_rotation needs a 3x3 matrix and positive block count")
    return np.kron(np.eye(blocks, dtype=rotation.dtype), rotation)


@dataclass(frozen=True)
class DofLayout:
    n_nodes: int
    components: tuple[str, ...]

    def __post_init__(self):
        if self.n_nodes < 1:
            raise ValueError("a DOF layout needs at least one node")
        if not self.components or len(set(self.components)) != len(self.components):
            raise ValueError("DOF component names must be non-empty and unique")

    @classmethod
    def translations(cls, n_nodes: int):
        return cls(n_nodes, TRANSLATION)

    @classmethod
    def frame(cls, n_nodes: int):
        return cls(n_nodes, TRANSLATION + ROTATION)

    @property
    def stride(self) -> int:
        return len(self.components)

    @property
    def size(self) -> int:
        return self.n_nodes * self.stride

    def component_indices(self, names: Sequence[str]) -> np.ndarray:
        try:
            return np.asarray([self.components.index(name) for name in names], dtype=int)
        except ValueError as exc:
            raise ValueError(f"unknown DOF component; layout has {self.components}") from exc

    def node_dofs(self, nodes, components: Sequence[str] | None = None) -> np.ndarray:
        nodes = np.atleast_1d(np.asarray(nodes, dtype=int))
        if np.any(nodes < 0) or np.any(nodes >= self.n_nodes):
            raise IndexError("node index is outside the DOF layout")
        local = (np.arange(self.stride) if components is None
                 else self.component_indices(tuple(components)))
        return nodes[:, None] * self.stride + local[None, :]

    def element_dofs(self, connectivity, components: Sequence[str] | None = None) -> np.ndarray:
        """Flattened node-major equation map for one element."""
        return self.node_dofs(connectivity, components).reshape(-1)

    def view(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector)
        if vector.size != self.size:
            raise ValueError(f"vector has {vector.size} entries; layout requires {self.size}")
        return vector.reshape(self.n_nodes, self.stride)

    def field(self, vector: np.ndarray, components: Sequence[str]) -> np.ndarray:
        return self.view(vector)[:, self.component_indices(tuple(components))]

    def add_nodal(self, vector: np.ndarray, nodes, values,
                  components: Sequence[str]) -> None:
        dofs = self.node_dofs(nodes, components)
        values = np.asarray(values)
        if values.ndim == 1:
            values = np.broadcast_to(values, dofs.shape)
        np.add.at(vector, dofs, values)


class SparseAssembler:
    """Accumulate element matrices through explicit equation maps."""

    def __init__(self, size: int):
        self.size = int(size)
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.data: list[float] = []

    def add_matrix(self, dofs, matrix) -> None:
        dofs = np.asarray(dofs, dtype=int)
        matrix = np.asarray(matrix)
        if matrix.shape != (len(dofs), len(dofs)):
            raise ValueError("element matrix shape does not match its DOF map")
        ii, jj = np.nonzero(matrix)
        self.rows.extend(dofs[ii]); self.cols.extend(dofs[jj]); self.data.extend(matrix[ii, jj])

    def csr(self):
        return sparse.coo_matrix((self.data, (self.rows, self.cols)),
                                 shape=(self.size, self.size)).tocsr()
