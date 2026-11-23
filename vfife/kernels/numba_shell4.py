"""Numba element kernel for shell4.

The batched NumPy path in `core/shell_solver.py` is the readable reference
and stays the fallback. It is already vectorised over elements, so what this
kernel removes is not Python overhead but temporaries: the NumPy version
allocates (ne, 24, 24)-sized intermediates for the rotation blocks and
einsum contractions every iteration; this one does one element in scalar
locals and a 24x24 matvec against that element's own local stiffness.

The element is geometrically nonlinear and materially linear, so per element
the work is: current frame (two cross products and three normalisations), the
rigid rotation Q = R^T R0 and its rotation vector, the local deformation
(current local coordinates minus the reference, and the nodal rotations with
the rigid part removed), K_local @ q_local, and the end actions rotated back
to global. Everything else about the element -- shape functions, selective
shear integration, hourglass and drilling stabilisation -- is frozen into
K_local at precompute, which is why the kernel needs no material at all.

SERIAL, LIKE THE FIBRE-BEAM KERNEL, AND FOR THE SAME REASON
A prange over 256 elements on two cores ran 20x SLOWER than the serial loop
(5.6 ms against 0.24): the per-element work is a few microseconds and the
thread scheduling and per-iteration allocation swamp it. The win is removing
temporaries, not adding cores -- and a serial loop can assemble straight into
the global vector, which saves the scatter pass as well.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
except ImportError:                                   # pragma: no cover
    njit = None
    prange = range


def available() -> bool:
    return njit is not None


if njit is not None:

    @njit(cache=True, fastmath=False, nogil=True, inline="always")
    def _cross(ax, ay, az, bx, by, bz):
        return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx

    @njit(cache=True, fastmath=False, nogil=True)
    def shell4_forces(nodes, conn, q, orient, frame0, local0, K, Bres, warp_tol,
                      out, result, warped):
        """One element at a time, assembled straight into `out` (ndof,), which
        the caller has zeroed. Writes result (ne,8) and warped[e] = 1 where the
        current face exceeds the warpage tolerance, 2 where it is degenerate
        (the caller raises; a kernel cannot).

        q is (nn, 6): translations then rotations per node; orient (nn,3,3)
        the nodes' orientation matrices. frame0 (ne,3,3)
        has the reference local axes as ROWS; local0 (ne,4,3) the reference
        nodal coordinates in that frame with z = 0; K (ne,24,24) the local
        stiffness; Bres (ne,8,24) the centroid resultant operator.
        """
        ne = conn.shape[0]
        for e in range(ne):
            # --- current nodal positions ------------------------------------
            x = np.empty((4, 3))
            for a in range(4):
                n = conn[e, a]
                for c in range(3):
                    x[a, c] = nodes[n, c] + q[n, c]
            cx = 0.25 * (x[0, 0] + x[1, 0] + x[2, 0] + x[3, 0])
            cy = 0.25 * (x[0, 1] + x[1, 1] + x[2, 1] + x[3, 1])
            cz = 0.25 * (x[0, 2] + x[1, 2] + x[2, 2] + x[3, 2])
            # --- current frame: ex along (x1-x0)+(x2-x3), ez the diagonal normal
            exx = (x[1, 0] - x[0, 0]) + (x[2, 0] - x[3, 0])
            exy = (x[1, 1] - x[0, 1]) + (x[2, 1] - x[3, 1])
            exz = (x[1, 2] - x[0, 2]) + (x[2, 2] - x[3, 2])
            d1x, d1y, d1z = x[2, 0] - x[0, 0], x[2, 1] - x[0, 1], x[2, 2] - x[0, 2]
            d2x, d2y, d2z = x[3, 0] - x[1, 0], x[3, 1] - x[1, 1], x[3, 2] - x[1, 2]
            nx_, ny_, nz_ = _cross(d1x, d1y, d1z, d2x, d2y, d2z)
            nex = np.sqrt(exx * exx + exy * exy + exz * exz)
            nn_ = np.sqrt(nx_ * nx_ + ny_ * ny_ + nz_ * nz_)
            if nex <= 1e-14 or nn_ <= 1e-14:
                warped[e] = 2
                continue
            exx /= nex; exy /= nex; exz /= nex
            ezx, ezy, ezz = nx_ / nn_, ny_ / nn_, nz_ / nn_
            eyx, eyy, eyz = _cross(ezx, ezy, ezz, exx, exy, exz)
            ney = np.sqrt(eyx * eyx + eyy * eyy + eyz * eyz)
            eyx /= ney; eyy /= ney; eyz /= ney
            R = np.empty((3, 3))
            R[0, 0], R[0, 1], R[0, 2] = exx, exy, exz
            R[1, 0], R[1, 1], R[1, 2] = eyx, eyy, eyz
            R[2, 0], R[2, 1], R[2, 2] = ezx, ezy, ezz
            # --- current local coordinates, warpage check ---------------------
            loc = np.empty((4, 3))
            for a in range(4):
                dx, dy, dz = x[a, 0] - cx, x[a, 1] - cy, x[a, 2] - cz
                for i in range(3):
                    loc[a, i] = R[i, 0] * dx + R[i, 1] * dy + R[i, 2] * dz
            span = max(loc[:, 0].max() - loc[:, 0].min(), loc[:, 1].max() - loc[:, 1].min(), 1e-30)
            if loc[:, 2].max() - loc[:, 2].min() > warp_tol * span:
                warped[e] = 1
                continue
            # the flat element: whatever out-of-plane deviation survived the
            # warpage check is projected away, exactly as `shell.frame` does
            for a in range(4):
                loc[a, 2] = 0.0
            # --- local deformation vector -------------------------------------
            # translations: current local minus reference local coordinates;
            # rotations: the node's triad relative to the face frame,
            # M = R T_n frame0^T, as a rotation vector
            lq = np.empty(24)
            for a in range(4):
                n = conn[e, a]
                for i in range(3):
                    lq[6 * a + i] = loc[a, i] - local0[e, a, i]
                M = np.empty((3, 3))
                for i in range(3):
                    for j in range(3):
                        s = 0.0
                        for k in range(3):
                            for l in range(3):
                                s += R[i, k] * orient[n, k, l] * frame0[e, j, l]
                        M[i, j] = s
                tr = M[0, 0] + M[1, 1] + M[2, 2]
                cang = 0.5 * (tr - 1.0)
                if cang > 1.0: cang = 1.0
                if cang < -1.0: cang = -1.0
                angle = np.arccos(cang)
                ax0 = 0.5 * (M[2, 1] - M[1, 2]); ax1 = 0.5 * (M[0, 2] - M[2, 0]); ax2 = 0.5 * (M[1, 0] - M[0, 1])
                if angle < 1e-6:
                    f = 1.0 + angle * angle / 6.0
                    p0, p1, p2 = f * ax0, f * ax1, f * ax2
                elif np.pi - angle < 1e-7:
                    best = 0; bn = -1.0
                    for j in range(3):
                        cn = 0.0
                        for i in range(3):
                            v = M[i, j] + (1.0 if i == j else 0.0)
                            cn += v * v
                        if cn > bn:
                            bn = cn; best = j
                    v0 = M[0, best] + (1.0 if best == 0 else 0.0)
                    v1 = M[1, best] + (1.0 if best == 1 else 0.0)
                    v2 = M[2, best] + (1.0 if best == 2 else 0.0)
                    nv = np.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
                    p0, p1, p2 = angle * v0 / nv, angle * v1 / nv, angle * v2 / nv
                else:
                    f = angle / np.sin(angle)
                    p0, p1, p2 = f * ax0, f * ax1, f * ax2
                lq[6 * a + 3] = p0; lq[6 * a + 4] = p1; lq[6 * a + 5] = p2
            # --- local forces and the centroid resultants ---------------------
            fl = np.empty(24)
            for i in range(24):
                s = 0.0
                for j in range(24):
                    s += K[e, i, j] * lq[j]
                fl[i] = s
            for i in range(8):
                s = 0.0
                for j in range(24):
                    s += Bres[e, i, j] * lq[j]
                result[e, i] = s
            # --- back to global: each 3-block is R^T @ block, straight into out
            for b in range(8):
                base = 6 * conn[e, b // 2] + 3 * (b % 2)
                for i in range(3):
                    out[base + i] += (R[0, i] * fl[3 * b] + R[1, i] * fl[3 * b + 1]
                                      + R[2, i] * fl[3 * b + 2])
            warped[e] = 0

else:                                                  # pragma: no cover
    shell4_forces = None
