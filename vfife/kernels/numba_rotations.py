"""The orientation update of the structural points, one point at a time.

`BeamSolver._advance` does R <- exp(omega dt) R and q_rot = log R for every
point every iteration. In numpy that is a dozen array operations on
(nn, 3) and (nn, 3, 3) arrays, each a few microseconds of overhead: on the
256-element roof (289 points) it cost as much as the numba element kernel
itself and halved the iteration rate. This is the same arithmetic in one
scalar pass; numpy (`core/rotations`) stays the reference and the fallback.
"""
import numpy as np

try:
    from numba import njit
except ImportError:                                   # pragma: no cover
    njit = None


def available() -> bool:
    return njit is not None


if njit is not None:

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def advance_orientations(w, dt, orient, out_rot):
        """w (nn,3) spins; orient (nn,3,3) updated in place; out_rot (nn,3)
        receives log(orient). Same branches as core/rotations.exp_map and
        log_map: series below 1e-6, the R + I column near pi."""
        nn = w.shape[0]
        E = np.empty((3, 3)); M = np.empty((3, 3))
        for n in range(nn):
            advance_node(n, w, dt, orient, out_rot, E, M)

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def advance_node(n, w, dt, orient, out_rot, E, M):
        """`advance_orientations` for point n (the compiled step runs it over
        blocks of points on several threads). Its 3x3 products
        on local variables, each entry from 0.0 and += over k as the loop was:
        the same numbers, a fifth faster. E, M (3,3) are scratch for the rare
        branch near a half turn."""
        tx, ty, tz = w[n, 0] * dt, w[n, 1] * dt, w[n, 2] * dt
        a2 = tx * tx + ty * ty + tz * tz
        a = np.sqrt(a2)
        if a < 1e-6:
            s = 1.0 - a2 / 6.0; c = 0.5 - a2 / 24.0
        else:
            s = np.sin(a) / a; c = (1.0 - np.cos(a)) / a2
        e00 = 1.0 + c * (-ty * ty - tz * tz); e01 = -s * tz + c * tx * ty; e02 = s * ty + c * tx * tz
        e10 = s * tz + c * tx * ty; e11 = 1.0 + c * (-tx * tx - tz * tz); e12 = -s * tx + c * ty * tz
        e20 = -s * ty + c * tx * tz; e21 = s * tx + c * ty * tz; e22 = 1.0 + c * (-tx * tx - ty * ty)
        o = orient[n]
        o00 = o[0, 0]; o01 = o[0, 1]; o02 = o[0, 2]
        o10 = o[1, 0]; o11 = o[1, 1]; o12 = o[1, 2]
        o20 = o[2, 0]; o21 = o[2, 1]; o22 = o[2, 2]
        m00 = 0.0; m00 += e00 * o00; m00 += e01 * o10; m00 += e02 * o20
        m01 = 0.0; m01 += e00 * o01; m01 += e01 * o11; m01 += e02 * o21
        m02 = 0.0; m02 += e00 * o02; m02 += e01 * o12; m02 += e02 * o22
        m10 = 0.0; m10 += e10 * o00; m10 += e11 * o10; m10 += e12 * o20
        m11 = 0.0; m11 += e10 * o01; m11 += e11 * o11; m11 += e12 * o21
        m12 = 0.0; m12 += e10 * o02; m12 += e11 * o12; m12 += e12 * o22
        m20 = 0.0; m20 += e20 * o00; m20 += e21 * o10; m20 += e22 * o20
        m21 = 0.0; m21 += e20 * o01; m21 += e21 * o11; m21 += e22 * o21
        m22 = 0.0; m22 += e20 * o02; m22 += e21 * o12; m22 += e22 * o22
        o[0, 0] = m00; o[0, 1] = m01; o[0, 2] = m02
        o[1, 0] = m10; o[1, 1] = m11; o[1, 2] = m12
        o[2, 0] = m20; o[2, 1] = m21; o[2, 2] = m22
        cang = 0.5 * (m00 + m11 + m22 - 1.0)
        if cang > 1.0: cang = 1.0
        if cang < -1.0: cang = -1.0
        angle = np.arccos(cang)
        ax0 = 0.5 * (m21 - m12); ax1 = 0.5 * (m02 - m20); ax2 = 0.5 * (m10 - m01)
        if angle < 1e-6:
            f = 1.0 + angle * angle / 6.0
            out_rot[n, 0] = f * ax0; out_rot[n, 1] = f * ax1; out_rot[n, 2] = f * ax2
        elif np.pi - angle < 1e-7:
            for i in range(3):
                for j in range(3):
                    M[i, j] = o[i, j]
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
            out_rot[n, 0] = angle * v0 / nv; out_rot[n, 1] = angle * v1 / nv; out_rot[n, 2] = angle * v2 / nv
        else:
            f = angle / np.sin(angle)
            out_rot[n, 0] = f * ax0; out_rot[n, 1] = f * ax1; out_rot[n, 2] = f * ax2

else:                                                  # pragma: no cover
    advance_orientations = advance_node = None


if njit is not None:

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy", inline="always")
    def exp3(tx, ty, tz, E):
        """E <- exp([t x]) for the rotation vector (tx, ty, tz)."""
        a2 = tx * tx + ty * ty + tz * tz
        a = np.sqrt(a2)
        if a < 1e-6:
            s = 1.0 - a2 / 6.0; c = 0.5 - a2 / 24.0
        else:
            s = np.sin(a) / a; c = (1.0 - np.cos(a)) / a2
        E[0, 0] = 1.0 + c * (-ty * ty - tz * tz); E[0, 1] = -s * tz + c * tx * ty; E[0, 2] = s * ty + c * tx * tz
        E[1, 0] = s * tz + c * tx * ty; E[1, 1] = 1.0 + c * (-tx * tx - tz * tz); E[1, 2] = -s * tx + c * ty * tz
        E[2, 0] = -s * ty + c * tx * tz; E[2, 1] = s * tx + c * ty * tz; E[2, 2] = 1.0 + c * (-tx * tx - ty * ty)

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy", inline="always")
    def log3(M, out):
        """out <- the rotation vector of the rotation matrix M."""
        cang = 0.5 * (M[0, 0] + M[1, 1] + M[2, 2] - 1.0)
        if cang > 1.0: cang = 1.0
        if cang < -1.0: cang = -1.0
        angle = np.arccos(cang)
        ax0 = 0.5 * (M[2, 1] - M[1, 2]); ax1 = 0.5 * (M[0, 2] - M[2, 0]); ax2 = 0.5 * (M[1, 0] - M[0, 1])
        if angle < 1e-6:
            f = 1.0 + angle * angle / 6.0
            out[0] = f * ax0; out[1] = f * ax1; out[2] = f * ax2
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
            out[0] = angle * v0 / nv; out[1] = angle * v1 / nv; out[2] = angle * v2 / nv
        else:
            f = angle / np.sin(angle)
            out[0] = f * ax0; out[1] = f * ax1; out[2] = f * ax2

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy", inline="always")
    def _log3s(m00, m01, m02, m10, m11, m12, m20, m21, m22):
        """`log3` of the matrix given by its entries, returned as a tuple."""
        cang = 0.5 * (m00 + m11 + m22 - 1.0)
        if cang > 1.0: cang = 1.0
        if cang < -1.0: cang = -1.0
        angle = np.arccos(cang)
        ax0 = 0.5 * (m21 - m12); ax1 = 0.5 * (m02 - m20); ax2 = 0.5 * (m10 - m01)
        if angle < 1e-6:
            f = 1.0 + angle * angle / 6.0
            return f * ax0, f * ax1, f * ax2
        elif np.pi - angle < 1e-7:
            M = np.empty((3, 3))
            M[0, 0] = m00; M[0, 1] = m01; M[0, 2] = m02
            M[1, 0] = m10; M[1, 1] = m11; M[1, 2] = m12
            M[2, 0] = m20; M[2, 1] = m21; M[2, 2] = m22
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
            return angle * v0 / nv, angle * v1 / nv, angle * v2 / nv
        else:
            f = angle / np.sin(angle)
            return f * ax0, f * ax1, f * ax2

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy", inline="always")
    def _rtr(R, T, R0, i, j):
        """(R T R0^T)[i, j], summed as `member_frame` always has."""
        v = 0.0
        for k in range(3):
            for l in range(3):
                v += R[i, k] * T[k, l] * R0[j, l]
        return v

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy", inline="always")
    def _tt(T1, T2, i, j):
        """(T1^T T2)[i, j]."""
        v = 0.0
        for k in range(3):
            v += T1[k, i] * T2[k, j]
        return v

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def member_frame(x1, x2, T1, T2, R0, R, loc):
        """`core.rotations.beam_frames_from_triads` for ONE member, scalar:
        x1, x2 (3,) the end positions; T1, T2 (3,3) the end orientations;
        R0 (3,3) the reference frame, rows = axes. Writes R (3,3), rows =
        current axes, and loc (2,3), the local nodal rotations; returns L.

        THE SMALL MATRICES ARE SCALARS: T1^T T2, its
        logarithm, the half rotation, T1 E and the relative rotations were
        3x3 arrays, four of them allocated per member and every entry stored
        and read back through memory. The same operations in the same order on
        local variables are the same numbers to the bit, and the frame kernel
        on the Xihoumen model went from 0.47 to 0.35 ms."""
        ax = x2[0] - x1[0]; ay = x2[1] - x1[1]; az = x2[2] - x1[2]
        L = np.sqrt(ax * ax + ay * ay + az * az)
        Ls = L if L > 1e-300 else 1e-300
        e1x, e1y, e1z = ax / Ls, ay / Ls, az / Ls
        # the mean nodal rotation: T1 exp(log(T1^T T2) / 2)
        p0, p1, p2 = _log3s(_tt(T1, T2, 0, 0), _tt(T1, T2, 0, 1), _tt(T1, T2, 0, 2),
                            _tt(T1, T2, 1, 0), _tt(T1, T2, 1, 1), _tt(T1, T2, 1, 2),
                            _tt(T1, T2, 2, 0), _tt(T1, T2, 2, 1), _tt(T1, T2, 2, 2))
        tx = 0.5 * p0; ty = 0.5 * p1; tz = 0.5 * p2
        a2 = tx * tx + ty * ty + tz * tz
        a = np.sqrt(a2)
        if a < 1e-6:
            s = 1.0 - a2 / 6.0; c = 0.5 - a2 / 24.0
        else:
            s = np.sin(a) / a; c = (1.0 - np.cos(a)) / a2
        E00 = 1.0 + c * (-ty * ty - tz * tz); E01 = -s * tz + c * tx * ty; E02 = s * ty + c * tx * tz
        E10 = s * tz + c * tx * ty; E11 = 1.0 + c * (-tx * tx - tz * tz); E12 = -s * tx + c * ty * tz
        E20 = -s * ty + c * tx * tz; E21 = s * tx + c * ty * tz; E22 = 1.0 + c * (-tx * tx - ty * ty)
        # the mean rotation Tm = T1 E: each entry from 0.0, then += over k, as the loop was
        Tm00 = 0.0; Tm00 += T1[0, 0] * E00; Tm00 += T1[0, 1] * E10; Tm00 += T1[0, 2] * E20
        Tm01 = 0.0; Tm01 += T1[0, 0] * E01; Tm01 += T1[0, 1] * E11; Tm01 += T1[0, 2] * E21
        Tm02 = 0.0; Tm02 += T1[0, 0] * E02; Tm02 += T1[0, 1] * E12; Tm02 += T1[0, 2] * E22
        Tm10 = 0.0; Tm10 += T1[1, 0] * E00; Tm10 += T1[1, 1] * E10; Tm10 += T1[1, 2] * E20
        Tm11 = 0.0; Tm11 += T1[1, 0] * E01; Tm11 += T1[1, 1] * E11; Tm11 += T1[1, 2] * E21
        Tm12 = 0.0; Tm12 += T1[1, 0] * E02; Tm12 += T1[1, 1] * E12; Tm12 += T1[1, 2] * E22
        Tm20 = 0.0; Tm20 += T1[2, 0] * E00; Tm20 += T1[2, 1] * E10; Tm20 += T1[2, 2] * E20
        Tm21 = 0.0; Tm21 += T1[2, 0] * E01; Tm21 += T1[2, 1] * E11; Tm21 += T1[2, 2] * E21
        Tm22 = 0.0; Tm22 += T1[2, 0] * E02; Tm22 += T1[2, 1] * E12; Tm22 += T1[2, 2] * E22
        t2x = Tm00 * R0[1, 0] + Tm01 * R0[1, 1] + Tm02 * R0[1, 2]
        t2y = Tm10 * R0[1, 0] + Tm11 * R0[1, 1] + Tm12 * R0[1, 2]
        t2z = Tm20 * R0[1, 0] + Tm21 * R0[1, 1] + Tm22 * R0[1, 2]
        e3x = e1y * t2z - e1z * t2y; e3y = e1z * t2x - e1x * t2z; e3z = e1x * t2y - e1y * t2x
        n3 = np.sqrt(e3x * e3x + e3y * e3y + e3z * e3z)
        if n3 < 1e-12:
            t3x = Tm00 * R0[2, 0] + Tm01 * R0[2, 1] + Tm02 * R0[2, 2]
            t3y = Tm10 * R0[2, 0] + Tm11 * R0[2, 1] + Tm12 * R0[2, 2]
            t3z = Tm20 * R0[2, 0] + Tm21 * R0[2, 1] + Tm22 * R0[2, 2]
            bx = t3y * e1z - t3z * e1y; by = t3z * e1x - t3x * e1z; bz = t3x * e1y - t3y * e1x
            nb = np.sqrt(bx * bx + by * by + bz * bz); bx /= nb; by /= nb; bz /= nb
            e3x = e1y * bz - e1z * by; e3y = e1z * bx - e1x * bz; e3z = e1x * by - e1y * bx
            n3 = np.sqrt(e3x * e3x + e3y * e3y + e3z * e3z)
        e3x /= n3; e3y /= n3; e3z /= n3
        e2x = e3y * e1z - e3z * e1y; e2y = e3z * e1x - e3x * e1z; e2z = e3x * e1y - e3y * e1x
        R[0, 0], R[0, 1], R[0, 2] = e1x, e1y, e1z
        R[1, 0], R[1, 1], R[1, 2] = e2x, e2y, e2z
        R[2, 0], R[2, 1], R[2, 2] = e3x, e3y, e3z
        # local rotations: log(R T_i R0^T)
        for node in range(2):
            T = T1 if node == 0 else T2
            l0, l1, l2 = _log3s(_rtr(R, T, R0, 0, 0), _rtr(R, T, R0, 0, 1), _rtr(R, T, R0, 0, 2),
                                _rtr(R, T, R0, 1, 0), _rtr(R, T, R0, 1, 1), _rtr(R, T, R0, 1, 2),
                                _rtr(R, T, R0, 2, 0), _rtr(R, T, R0, 2, 1), _rtr(R, T, R0, 2, 2))
            loc[node, 0] = l0; loc[node, 1] = l1; loc[node, 2] = l2
        return L

else:                                                  # pragma: no cover
    exp3 = log3 = member_frame = None
