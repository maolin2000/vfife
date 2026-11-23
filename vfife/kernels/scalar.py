"""Scalar device functions shared by the Numba kernels.

Everything here operates on ONE integration point or ONE element, with fixed-size
local arrays and no allocation that escapes.  That is not a Numba idiom for its
own sake: it is the shape a GPU kernel has to be, so these functions transcribe
to Warp almost line for line.  The NumPy backend stays the readable reference;
this is the same algebra written the way hardware wants it.

`inline="always"` throughout, so the element kernel ends up as one fused loop
body rather than a chain of calls.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Mueller's polar-decomposition iteration normalises by tr(S); the exact Newton
# step needs (tr(S) I - S)^-1, which for near-spherical stretch is 3/2 times
# larger.  Same constant as the NumPy path -- see elements/hexa8.py.
OMEGA_SCALE = 1.5

_JIT = dict(cache=True, fastmath=True, inline="always")


@njit(**_JIT)
def quat_to_mat(qw, qx, qy, qz, R):
    """Rotation matrix of a unit quaternion, written into R (3,3)."""
    R[0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    R[0, 1] = 2.0 * (qx * qy - qw * qz)
    R[0, 2] = 2.0 * (qx * qz + qw * qy)
    R[1, 0] = 2.0 * (qx * qy + qw * qz)
    R[1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    R[1, 2] = 2.0 * (qy * qz - qw * qx)
    R[2, 0] = 2.0 * (qx * qz - qw * qy)
    R[2, 1] = 2.0 * (qy * qz + qw * qx)
    R[2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)


@njit(**_JIT)
def extract_rotation(A, q, max_iter, tol):
    """Polar rotation of A (3,3), warm-started from the quaternion q (4,).

    q is updated in place.  Always a proper rotation by construction, so there
    is no reflection case to branch on.
    """
    R = np.empty((3, 3))
    for _ in range(max_iter):
        quat_to_mat(q[0], q[1], q[2], q[3], R)

        # omega = sum_c (R_col_c x A_col_c) / |tr(R^T A)|
        wx = 0.0
        wy = 0.0
        wz = 0.0
        den = 0.0
        for c in range(3):
            r0, r1, r2 = R[0, c], R[1, c], R[2, c]
            a0, a1, a2 = A[0, c], A[1, c], A[2, c]
            wx += r1 * a2 - r2 * a1
            wy += r2 * a0 - r0 * a2
            wz += r0 * a1 - r1 * a0
            den += r0 * a0 + r1 * a1 + r2 * a2
        s = OMEGA_SCALE / (abs(den) + 1e-12)
        wx *= s
        wy *= s
        wz *= s

        w = np.sqrt(wx * wx + wy * wy + wz * wz)
        if w < tol:
            break
        half = 0.5 * w
        sn = np.sin(half) / w
        dw = np.cos(half)
        dx = wx * sn
        dy = wy * sn
        dz = wz * sn

        qw = dw * q[0] - dx * q[1] - dy * q[2] - dz * q[3]
        qx = dw * q[1] + dx * q[0] + dy * q[3] - dz * q[2]
        qy = dw * q[2] - dx * q[3] + dy * q[0] + dz * q[1]
        qz = dw * q[3] + dx * q[2] - dy * q[1] + dz * q[0]
        n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        q[0] = qw / n
        q[1] = qx / n
        q[2] = qy / n
        q[3] = qz / n


@njit(**_JIT)
def principal(t0, t1, t2, t3, t4, t5, out):
    """Eigenvalues of a symmetric 3x3 in TENSOR-Voigt form, ascending, into out(3).

    Closed-form Cardano.  A GPU kernel has no eigensolver to call, and this is
    also 7x faster than LAPACK on the batched path, so it is the only version.
    """
    q = (t0 + t1 + t2) / 3.0
    a = t0 - q
    b = t1 - q
    c = t2 - q
    p1 = t3 * t3 + t4 * t4 + t5 * t5
    p2 = a * a + b * b + c * c + 2.0 * p1
    if p2 <= 0.0:
        out[0] = q
        out[1] = q
        out[2] = q
        return
    p = np.sqrt(p2 / 6.0)
    safe = p if p > 1e-30 else 1e-30
    det = (a * (b * c - t4 * t4)
           - t3 * (t3 * c - t4 * t5)
           + t5 * (t3 * t4 - b * t5))
    r = det / (2.0 * safe * safe * safe)
    if r > 1.0:
        r = 1.0
    elif r < -1.0:
        r = -1.0
    phi = np.arccos(r) / 3.0
    e1 = q + 2.0 * p * np.cos(phi)
    e3 = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    out[0] = e3
    out[1] = 3.0 * q - e1 - e3
    out[2] = e1


@njit(**_JIT)
def curve_eval(ep_tab, sig_tab, dmg_tab, slope_tab, x, res):
    """Piecewise-linear (ep -> effective stress, slope, damage) lookup into res(3).

    Matches materials/curves.py::Curve.evaluate exactly, including holding the
    last stress flat past the end of the table -- extrapolating a softening
    branch into negative stress is the fastest way to wreck a run.
    """
    n = ep_tab.shape[0]
    if x >= ep_tab[n - 1]:
        res[0] = sig_tab[n - 1]
        res[1] = 0.0
        res[2] = dmg_tab[n - 1]
        return
    if x <= ep_tab[0]:
        res[0] = sig_tab[0]
        res[1] = slope_tab[0]
        res[2] = dmg_tab[0]
        return
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ep_tab[mid] <= x:
            lo = mid
        else:
            hi = mid
    dx = ep_tab[lo + 1] - ep_tab[lo]
    t = x - ep_tab[lo]
    res[0] = sig_tab[lo] + slope_tab[lo] * t
    res[1] = slope_tab[lo]
    res[2] = dmg_tab[lo] + (dmg_tab[lo + 1] - dmg_tab[lo]) / dx * t
