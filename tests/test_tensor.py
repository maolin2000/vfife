"""Closed-form tensor kernels.

These have to be exact: a GPU kernel has no eigensolver to fall back on, so the
Cardano routine here is the only path in production.
"""

import numpy as np

from vfife.tensor import ddot, deviator, principal_values


def _to_matrix(t):
    T = np.zeros(t.shape[:-1] + (3, 3))
    T[..., 0, 0], T[..., 1, 1], T[..., 2, 2] = t[..., 0], t[..., 1], t[..., 2]
    T[..., 0, 1] = T[..., 1, 0] = t[..., 3]
    T[..., 1, 2] = T[..., 2, 1] = t[..., 4]
    T[..., 2, 0] = T[..., 0, 2] = t[..., 5]
    return T


def test_principal_values_match_lapack():
    rng = np.random.default_rng(11)
    t = rng.normal(size=(50000, 6))
    t[:20, 3:] = 0.0                       # already diagonal
    t[20:30] = 0.0
    t[20:30, :3] = 1.0                     # isotropic
    t[30:40, :3] = 1.0                     # isotropic plus shear
    t[40:50] *= 1e-14                      # tiny
    t[50:60] *= 1e14                       # huge
    with np.errstate(all="raise"):
        got = principal_values(t)
    ref = np.linalg.eigvalsh(_to_matrix(t))
    scale = np.maximum(np.abs(ref).max(axis=1, keepdims=True), 1e-300)
    assert (np.abs(got - ref) / scale).max() < 1e-12


def test_principal_values_are_ascending():
    rng = np.random.default_rng(3)
    pv = principal_values(rng.normal(size=(5000, 6)))
    assert np.all(np.diff(pv, axis=1) >= -1e-12)


def test_ddot_matches_full_contraction():
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(500, 6)), rng.normal(size=(500, 6))
    A, B = _to_matrix(a), _to_matrix(b)
    assert np.allclose(ddot(a, b), np.einsum("nij,nij->n", A, B))


def test_deviator_is_traceless_and_preserves_shear():
    rng = np.random.default_rng(9)
    s = deviator(rng.normal(size=(500, 6)))
    assert np.abs(s[:, :3].sum(axis=1)).max() < 1e-12
