"""Finite rotations for the frame members: exp, log, and the co-rotational
element frame with its local nodal rotations.

WHY THIS EXISTS
    Until v0.1 a frame member removed its rigid rotation by subtracting
    rotation vectors: the element frame followed the chord alone (the
    minimal rotation taking the reference axis to the current chord, so the
    cross-section never twisted with the nodes), and the local nodal rotation
    was `R (theta_node - phi)`. Both are exact in a plane, where every
    rotation is about the same axis and rotation vectors add. In space they
    are not: rotation vectors do not add, and a member whose nodes have
    twisted has a frame the chord cannot see. Lee's frame (case_16) is
    planar and never noticed; the 45-degree bend of Bathe & Bolourchi
    (case_18) bends about two axes and twists at once, and came out 2% stiff
    with the in-plane displacement 10% off in the direction across the chord.

THE EXACT VERSION (Crisfield 1990; Crisfield & Jelenic 1999)
    Each node carries a triad T_i = exp(theta_i) T_0, T_0 the reference
    element frame (columns = reference axes). The element frame T_e has e1
    along the current chord and e2, e3 from the MEAN nodal triad
    T_m = T_1 exp(log(T_1^T T_2) / 2) projected orthogonal to e1: t2 = T_m
    e2_0, e3 = e1 x t2 normalised, e2 = e3 x e1. The local rotation of node
    i is the relative rotation of its triad with respect to the element
    frame, theta_loc,i = log(T_e^T T_i), a genuinely small rotation to which
    the linear beam law applies. In a plane this reduces exactly to the old
    theta_i - phi, so the planar cases keep their numbers to round-off.

    Rows-are-axes is the convention the rest of the frame code uses for its
    R (R = T_e^T), and it is what these functions return.

ABOUT THE ROTATION DOFS
    Each structural point carries its ORIENTATION as a rotation matrix,
    advanced multiplicatively by the spatial spin of the step,
    R <- exp(omega dt) R, and driven by the spatial moment directly -- the
    particle equation of Ting & Wu (2020) as written. The rotation vector
    log R is kept alongside for output, monitors, prescribed values and the
    convergence norm; it is derived, never integrated. Until v0.1 the
    vector itself was the dof and was advanced additively, which needed the
    tangent map of the exponential to make the moment conjugate to it; a
    correct equilibrium, an approximate dynamics.
"""
from __future__ import annotations

import numpy as np


def _skew(v):
    """(n,3) -> (n,3,3) skew matrices."""
    A = np.zeros(v.shape[:-1] + (3, 3))
    A[..., 0, 1] = -v[..., 2]; A[..., 0, 2] = v[..., 1]
    A[..., 1, 0] = v[..., 2]; A[..., 1, 2] = -v[..., 0]
    A[..., 2, 0] = -v[..., 1]; A[..., 2, 1] = v[..., 0]
    return A


def exp_map(theta):
    """Rodrigues: (..., 3) rotation vectors -> (..., 3, 3) rotation matrices.
    Series for the two coefficients below 1e-6 so the small-rotation limit
    is exact rather than 0/0."""
    theta = np.asarray(theta, float)
    angle = np.linalg.norm(theta, axis=-1)
    small = angle < 1e-6
    a = np.where(small, 1.0, angle)
    s = np.where(small, 1.0 - angle**2 / 6.0, np.sin(a) / a)
    c = np.where(small, 0.5 - angle**2 / 24.0, (1.0 - np.cos(a)) / a**2)
    A = _skew(theta)
    I = np.broadcast_to(np.eye(3), A.shape)
    return I + s[..., None, None] * A + c[..., None, None] * (A @ A)


def log_map(R):
    """(..., 3, 3) rotation matrices -> (..., 3) rotation vectors.

    Angle from the trace; axis from the skew part, with a series below 1e-6
    and, within 1e-7 of pi where the skew part vanishes, from the dominant
    column of R + I (the eigenvector of eigenvalue 1)."""
    R = np.asarray(R, float)
    tr = np.einsum("...ii->...", R)
    angle = np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0))
    axial = 0.5 * np.stack((R[..., 2, 1] - R[..., 1, 2],
                            R[..., 0, 2] - R[..., 2, 0],
                            R[..., 1, 0] - R[..., 0, 1]), axis=-1)
    small = angle < 1e-6
    flip = (np.pi - angle) < 1e-7
    a = np.where(small | flip, 1.0, angle)
    f = np.where(small, 1.0 + angle**2 / 6.0, a / np.sin(a))
    phi = f[..., None] * axial
    if np.any(flip):
        for idx in zip(*np.nonzero(flip)):
            M = R[idx] + np.eye(3)
            col = M[:, int(np.argmax(np.linalg.norm(M, axis=0)))]
            phi[idx] = angle[idx] * col / np.linalg.norm(col)
    return phi


def beam_frames(x, theta, R0):
    """The co-rotational frame and local rotations of every member.

    x (ne,2,3) current end positions; theta (ne,2,3) the nodes' rotation
    vectors; R0 (ne,3,3) the reference frames, rows = axes. Returns R
    (ne,3,3) with rows = current axes, theta_loc (ne,2,3) the local nodal
    rotations in that frame, and L (ne,) the current length.
    """
    return beam_frames_from_triads(x, exp_map(theta), R0)


def beam_frames_from_triads(x, T, R0):
    """`beam_frames` with the nodal triads given: T (ne,2,3,3) the two
    nodes' orientation matrices (columns = rotated axes, i.e. exp(theta))."""
    axis = x[:, 1] - x[:, 0]
    L = np.linalg.norm(axis, axis=1)
    e1 = axis / np.maximum(L, 1e-300)[:, None]
    T1 = T[:, 0]; T2 = T[:, 1]
    # the mean nodal rotation, halfway from node 1 to node 2 along the geodesic
    Tm = T1 @ exp_map(0.5 * log_map(np.einsum("eji,ejk->eik", T1, T2)))
    t2 = np.einsum("eij,ej->ei", Tm, R0[:, 1])                 # reference e2, rotated by the mean
    e3 = np.cross(e1, t2)
    n3 = np.linalg.norm(e3, axis=1)
    if np.any(n3 < 1e-12):
        # the mean e2 has fallen along the chord: the reference e3 instead
        bad = n3 < 1e-12
        t3 = np.einsum("eij,ej->ei", Tm[bad], R0[bad, 2])
        e2b = np.cross(t3, e1[bad]); e2b /= np.linalg.norm(e2b, axis=1)[:, None]
        e3[bad] = np.cross(e1[bad], e2b); n3 = np.linalg.norm(e3, axis=1)
    e3 /= n3[:, None]
    e2 = np.cross(e3, e1)
    R = np.stack((e1, e2, e3), axis=1)                            # rows = axes = T_e^T
    # local rotation of each node: T_e^T T_i T_0 = R exp(theta_i) R0^T
    loc1 = log_map(np.einsum("eij,ejk,elk->eil", R, T1, R0))
    loc2 = log_map(np.einsum("eij,ejk,elk->eil", R, T2, R0))
    return R, np.stack((loc1, loc2), axis=1), L


def orthonormalize(R):
    """The nearest rotation to each (..., 3, 3) matrix (polar decomposition
    by SVD). A product of a million exp(omega dt) factors is orthogonal to
    about 1e-10; this brings it back to round-off at each load level."""
    U, _s, Vt = np.linalg.svd(R)
    Q = U @ Vt
    neg = np.linalg.det(Q) < 0
    if np.any(neg):
        U[neg, :, 2] *= -1.0
        Q = U @ Vt
    return Q
