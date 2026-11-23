"""The six-component connector, compiled.

`elements.connector.response` is the reference: the deformation of the
link after its rigid motion is removed, its 6 x 12 Jacobian by central
differences over 25 stacked configurations, the stiffness, the curves, the
path-dependent laws (gap, bilinear, friction, and the two in the plane), the
dashpot and the force conjugate to the solver's spins. Stacked in NumPy it
was still 0.4 ms a connector: on the Hangzhou Bay seismic model two pot
bearings cost 0.8 ms of a 2.3 ms step, as much as the fibre kernel with
two threads.

This is the same arithmetic, one configuration at a time, with the
rotation maps of `core.rotations` written out for one rotation. It is not
the NumPy path to the bit -- NumPy's vectorised sin, cos and arccos, and
its 3 x 3 products, may round differently -- but to round-off, as the
frame and fibre kernels are: the deformation to 1e-12 of its scale, the
forces to about 1e-8 (the central differences amplify a last-place
difference in a deformation; the tests hold 1e-7). Not near a relative
rotation of pi between the two ends: there the logarithm's axis comes from
a skew part that has lost its digits, and a one-ulp difference in T0^T T1
moves the forces by 1e-3 at pi - 1e-3 -- in the reference as much as here.
No bearing turns that far.

`encode(props_list)` turns the connectors' settings into arrays once;
`respond(...)` evaluates every connector of a model into the global force.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:                                   # pragma: no cover
    njit = None

from ..elements import connector as _ref

#: the spin increment of the Jacobian's rotational columns, as the reference
SPIN_STEP = _ref._SPIN_STEP
#: the law codes
GAP, BILINEAR, FRICTION, DAMPER, PENDULUM = 1, 2, 3, 4, 5
_LAW_CODE = {"gap": GAP, "bilinear": BILINEAR, "friction": FRICTION, "damper": DAMPER, "pendulum": PENDULUM}


def available() -> bool:
    return njit is not None


def encode(props_list):
    """The settings of every connector, as arrays (index = connector kind)."""
    n = len(props_list)
    K = np.zeros((n, 6, 6)); C = np.zeros((n, 6, 6))
    local = np.ones(n, np.bool_); has_c = np.zeros(n, np.bool_)
    ncur = np.zeros((n, 6), np.int64)
    width = 2
    for p in props_list:
        for t in (p.get("curves") or {}).values():
            width = max(width, len(t))
    cx = np.zeros((n, 6, width)); cy = np.zeros((n, 6, width))
    law = np.zeros((n, 6), np.int64)
    # per component: k (or k1), k2, fy, mu, lo, hi, has_lo, has_hi, normal_index (-1 = value), normal_value, H
    # a damper (1.10): k beside it, c, alpha, -, linear_below, ..., [10] = its linear coefficient
    # a pendulum (1.10): as friction, with [1] = 1 / radius
    lp = np.zeros((n, 6, 11))
    planar = np.zeros(n, np.bool_)
    for c, p in enumerate(props_list):
        K[c] = _ref.matrix(p.get("stiffness"))
        cd = p.get("connector_damping")
        if cd is not None:
            C[c] = _ref.matrix(cd); has_c[c] = True
        local[c] = p.get("coordinate_system", "local") == "local"
        for name, table in (p.get("curves") or {}).items():
            i = _ref.COMPONENTS.index(name)
            a = np.asarray(table, dtype=float)
            ncur[c, i] = len(a); cx[c, i, :len(a)] = a[:, 0]; cy[c, i, :len(a)] = a[:, 1]
        laws = p.get("laws") or {}
        for name, lw in laws.items():
            i = _ref.COMPONENTS.index(name)
            t = lw["type"]
            law[c, i] = _LAW_CODE[t]
            row = lp[c, i]
            if t == "gap":
                row[0] = float(lw["stiffness"])
                if lw.get("lo") is not None:
                    row[4] = float(lw["lo"]); row[6] = 1.0
                if lw.get("hi") is not None:
                    row[5] = float(lw["hi"]); row[7] = 1.0
            elif t == "bilinear":
                k1, k2, fy = float(lw["k1"]), float(lw["k2"]), float(lw["fy"])
                row[0], row[1], row[2] = k1, k2, fy
                row[10] = k1 * k2 / (k1 - k2) if k1 > k2 else 0.0
            elif t == "damper":
                row[0] = float(lw.get("stiffness") or 0.0)
                row[1] = float(lw["c"]); row[2] = float(lw["exponent"])
                row[4] = float(lw.get("linear_below") or 0.0)
                row[10] = _ref.damper_linear(lw)
                has_c[c] = True                    # the rate is needed
            else:
                row[0] = float(lw["stiffness"]); row[3] = float(lw["mu"])
                if t == "pendulum":
                    row[1] = 1.0 / float(lw["radius"])
                nm = lw.get("normal")
                if isinstance(nm, str):
                    row[8] = float(_ref.COMPONENTS.index(nm))
                else:
                    row[8] = -1.0; row[9] = float(nm)
        planar[c] = ("uy" in laws and "uz" in laws and laws["uy"]["type"] in ("friction", "bilinear", "pendulum")
                     and _ref._same_law(laws["uy"], laws["uz"]))
    return K, C, local, has_c, ncur, cx, cy, law, lp, planar


if njit is not None:

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def _exp(tx, ty, tz, E):
        """core.rotations.exp_map for one vector: I + s A + c A A."""
        angle = np.sqrt(tx * tx + ty * ty + tz * tz)
        if angle < 1e-6:
            s = 1.0 - angle ** 2 / 6.0; c = 0.5 - angle ** 2 / 24.0
        else:
            s = np.sin(angle) / angle; c = (1.0 - np.cos(angle)) / angle ** 2
        A = np.zeros((3, 3))
        A[0, 1] = -tz; A[0, 2] = ty; A[1, 0] = tz; A[1, 2] = -tx; A[2, 0] = -ty; A[2, 1] = tx
        for i in range(3):
            for j in range(3):
                aa = 0.0
                for k in range(3):
                    aa += A[i, k] * A[k, j]
                E[i, j] = (1.0 if i == j else 0.0) + s * A[i, j] + c * aa

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def _log(R, out):
        """core.rotations.log_map for one matrix."""
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        x = (tr - 1.0) * 0.5
        if x > 1.0:
            x = 1.0
        if x < -1.0:
            x = -1.0
        angle = np.arccos(x)
        ax0 = 0.5 * (R[2, 1] - R[1, 2]); ax1 = 0.5 * (R[0, 2] - R[2, 0]); ax2 = 0.5 * (R[1, 0] - R[0, 1])
        if (np.pi - angle) < 1e-7:
            best = 0; bn = -1.0
            for j in range(3):
                cn = 0.0
                for i in range(3):
                    v = R[i, j] + (1.0 if i == j else 0.0)
                    cn += v * v
                cn = np.sqrt(cn)
                if cn > bn:
                    bn = cn; best = j
            v0 = R[0, best] + (1.0 if best == 0 else 0.0)
            v1 = R[1, best] + (1.0 if best == 1 else 0.0)
            v2 = R[2, best] + (1.0 if best == 2 else 0.0)
            nv = np.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
            out[0] = angle * v0 / nv; out[1] = angle * v1 / nv; out[2] = angle * v2 / nv
            return
        if angle < 1e-6:
            f = 1.0 + angle ** 2 / 6.0
        else:
            f = angle / np.sin(angle)
        out[0] = f * ax0; out[1] = f * ax1; out[2] = f * ax2

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def _deform(u, T0, T1, x0, R0, local, d, M, N, P, Q, v):
        """connector._deformations for one configuration: d (6,)."""
        # the mean triad Q = T0 exp(0.5 log(T0^T T1))
        for i in range(3):
            for j in range(3):
                a = 0.0
                for k in range(3):
                    a += T0[k, i] * T1[k, j]
                M[i, j] = a
        _log(M, v)
        _exp(0.5 * v[0], 0.5 * v[1], 0.5 * v[2], N)
        for i in range(3):
            for j in range(3):
                a = 0.0
                for k in range(3):
                    a += T0[i, k] * N[k, j]
                Q[i, j] = a
        c0 = x0[1, 0] - x0[0, 0]; c1 = x0[1, 1] - x0[0, 1]; c2 = x0[1, 2] - x0[0, 2]
        pt0 = ((x0[1, 0] + u[1, 0]) - (x0[0, 0] + u[0, 0])) - (Q[0, 0] * c0 + Q[0, 1] * c1 + Q[0, 2] * c2)
        pt1 = ((x0[1, 1] + u[1, 1]) - (x0[0, 1] + u[0, 1])) - (Q[1, 0] * c0 + Q[1, 1] * c1 + Q[1, 2] * c2)
        pt2 = ((x0[1, 2] + u[1, 2]) - (x0[0, 2] + u[0, 2])) - (Q[2, 0] * c0 + Q[2, 1] * c1 + Q[2, 2] * c2)
        # the relative rotation log(T1 T0^T)
        for i in range(3):
            for j in range(3):
                a = 0.0
                for k in range(3):
                    a += T1[i, k] * T0[j, k]
                M[i, j] = a
        _log(M, v)
        if local:
            # axes = R0 Q^T
            for i in range(3):
                for j in range(3):
                    a = 0.0
                    for k in range(3):
                        a += R0[i, k] * Q[j, k]
                    P[i, j] = a
            for i in range(3):
                d[i] = P[i, 0] * pt0 + P[i, 1] * pt1 + P[i, 2] * pt2
                d[3 + i] = P[i, 0] * v[0] + P[i, 1] * v[1] + P[i, 2] * v[2]
        else:
            d[0] = pt0; d[1] = pt1; d[2] = pt2
            d[3] = v[0]; d[4] = v[1]; d[5] = v[2]

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def _plastic(d, slip, k, fy, H):
        alpha = H * slip
        trial = k * (d - slip)
        xi = trial - alpha
        over = abs(xi) - fy
        if over <= 0.0:
            return trial, slip
        dg = over / (k + H)
        sgn = 1.0 if xi > 0.0 else -1.0
        slip_new = slip + dg * sgn
        return k * (d - slip_new), slip_new

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def response_one(u, T, ve, x0, R0, spins, K, C, local, has_c, ncur, cx, cy, law, lp, planar,
                     sc, fe, act, d, slip_out, include_damping, d0):
        """connector.response for one connector: fe (12), act (6), d (6),
        slip_out (6). `spins` (6,3,3) are exp(+-h e_k). `d0` (6,) is the
        deformation it was born with (zeros: built from the start)."""
        un = 0.0
        for n in range(2):
            for k in range(3):
                un += u[n, k] * u[n, k]
        h_u = 2.0e-7 * max(np.sqrt(un), 1.0)
        M = np.empty((3, 3)); N = np.empty((3, 3)); P = np.empty((3, 3)); Q = np.empty((3, 3))
        v = np.empty(3)
        Up = np.empty((2, 3)); Tp = np.empty((2, 3, 3))
        dp = np.empty(6); dm = np.empty(6)
        J = np.empty((6, 12))
        _deform(u, T[0], T[1], x0, R0, local, d, M, N, P, Q, v)
        for n in range(2):
            for k in range(3):
                for s in range(2):
                    Up[:, :] = u
                    Up[n, k] += h_u if s == 0 else -h_u
                    _deform(Up, T[0], T[1], x0, R0, local, dp if s == 0 else dm, M, N, P, Q, v)
                for i in range(6):
                    J[i, 6 * n + k] = (dp[i] - dm[i]) / (2.0 * h_u)
                for s in range(2):
                    Tp[:, :, :] = T
                    sp = spins[k] if s == 0 else spins[3 + k]
                    for i in range(3):
                        for j in range(3):
                            a = 0.0
                            for m in range(3):
                                a += sp[i, m] * T[n, m, j]
                            Tp[n, i, j] = a
                    _deform(u, Tp[0], Tp[1], x0, R0, local, dp if s == 0 else dm, M, N, P, Q, v)
                for i in range(6):
                    J[i, 6 * n + 3 + k] = (dp[i] - dm[i]) / (2.0 * SPIN_STEP)
        for i in range(6):
            d[i] -= d0[i]                     # x - 0.0 is x to the bit
        for i in range(6):
            a = 0.0
            for j in range(6):
                a += K[i, j] * d[j]
            act[i] = a
        for i in range(6):
            m = ncur[i]
            if m:
                xs = cx[i]; ys = cy[i]
                val = d[i]
                if val <= xs[0]:
                    f = ys[0] + (val - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
                elif val >= xs[m - 1]:
                    f = ys[m - 1] + (val - xs[m - 1]) * (ys[m - 1] - ys[m - 2]) / (xs[m - 1] - xs[m - 2])
                else:
                    f = np.interp(val, xs[:m], ys[:m])
                act[i] += f - K[i, i] * d[i]
        gap_open = np.zeros(6, np.bool_)
        for i in range(6):
            t = law[i]
            if t == 0 or (planar and (i == 1 or i == 2)):
                continue
            row = lp[i]
            slip = sc[i]
            if t == GAP:
                k = row[0]
                if row[7] > 0.0 and d[i] > row[5]:
                    f = k * (d[i] - row[5])
                elif row[6] > 0.0 and d[i] < row[4]:
                    f = k * (d[i] - row[4])
                else:
                    f = 0.0
                s_new = slip
            elif t == BILINEAR:
                f, s_new = _plastic(d[i], slip, row[0], row[2], row[10])
            elif t == DAMPER:
                f = row[0] * d[i]
                s_new = slip
            else:
                if row[8] >= 0.0:
                    ii = int(row[8])
                    Nn = -K[ii, ii] * d[ii]
                else:
                    Nn = row[9]
                Nn = max(Nn, 0.0)
                fy = row[3] * Nn
                if fy <= 0.0:
                    f, s_new = 0.0, d[i]
                else:
                    f, s_new = _plastic(d[i], slip, row[0], fy, 0.0)
                    if t == PENDULUM:
                        f += Nn * row[1] * d[i]
            act[i] = f
            slip_out[i] = s_new
            if t == GAP and f == 0.0:
                gap_open[i] = True
        if planar:
            row = lp[1]
            t = law[1]
            d1 = d[1]; d2 = d[2]; s1 = sc[1]; s2 = sc[2]
            if t == BILINEAR:
                k = row[0]; fy = row[2]; H = row[10]
                go = True
            else:
                if row[8] >= 0.0:
                    ii = int(row[8])
                    Nn = -K[ii, ii] * d[ii]
                else:
                    Nn = row[9]
                Nn = max(Nn, 0.0)
                fy = row[3] * Nn
                k = row[0]; H = 0.0
                go = fy > 0.0
            if not go:
                act[1] = 0.0; act[2] = 0.0
                slip_out[1] = d1; slip_out[2] = d2
            else:
                a1 = H * s1; a2 = H * s2
                tr1 = k * (d1 - s1); tr2 = k * (d2 - s2)
                x1 = tr1 - a1; x2 = tr2 - a2
                nx = np.sqrt(x1 * x1 + x2 * x2)
                over = nx - fy
                if over <= 0.0:
                    act[1] = tr1; act[2] = tr2
                    slip_out[1] = s1; slip_out[2] = s2
                else:
                    dg = over / (k + H)
                    n1 = s1 + dg * x1 / nx; n2 = s2 + dg * x2 / nx
                    act[1] = k * (d1 - n1); act[2] = k * (d2 - n2)
                    slip_out[1] = n1; slip_out[2] = n2
                if t == PENDULUM:
                    act[1] += Nn * row[1] * d1; act[2] += Nn * row[1] * d2
        if include_damping and has_c:
            jv = np.empty(6)
            for i in range(6):
                a = 0.0
                for j in range(12):
                    a += J[i, j] * ve[j]
                jv[i] = a
            for i in range(6):
                a = 0.0
                for j in range(6):
                    a += C[i, j] * jv[j]
                if not gap_open[i]:
                    act[i] += a
            for i in range(6):
                if law[i] == DAMPER:
                    row = lp[i]
                    r = jv[i]
                    if row[2] >= 1.0:
                        act[i] += row[1] * r
                    else:
                        av = r if r >= 0.0 else -r
                        if av < row[4]:
                            act[i] += row[10] * r
                        elif r > 0.0:
                            act[i] += row[1] * av ** row[2]
                        else:
                            act[i] -= row[1] * av ** row[2]
        for j in range(12):
            a = 0.0
            for i in range(6):
                a += J[i, j] * act[i]
            fe[j] = a
        if not local:
            p0 = x0[0, 0] + u[0, 0]; p1 = x0[0, 1] + u[0, 1]; p2 = x0[0, 2] + u[0, 2]
            q0 = x0[1, 0] + u[1, 0]; q1 = x0[1, 1] + u[1, 1]; q2 = x0[1, 2] + u[1, 2]
            n0 = (p1 * fe[2] - p2 * fe[1]) + (q1 * fe[8] - q2 * fe[7]) + fe[3] + fe[9]
            n1_ = (p2 * fe[0] - p0 * fe[2]) + (q2 * fe[6] - q0 * fe[8]) + fe[4] + fe[10]
            n2_ = (p0 * fe[1] - p1 * fe[0]) + (q0 * fe[7] - q1 * fe[6]) + fe[5] + fe[11]
            fe[3] -= 0.5 * n0; fe[4] -= 0.5 * n1_; fe[5] -= 0.5 * n2_
            fe[9] -= 0.5 * n0; fe[10] -= 0.5 * n1_; fe[11] -= 0.5 * n2_

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def respond(ids, kind, conn, nodes, q2, orient, v, R0, spins,
                K, C, local, has_c, ncur, cx, cy, law, lp, planar,
                committed, slip, include_damping, out, actions, end_actions, deformation, nact,
                rows_out, to_rows, check, d0):
        """Every connector `ids` (elements) into the global force `out`
        (6 dofs a node) -- or, with `to_rows`, each one's twelve forces into
        its row of `rows_out`, for a caller that adds them in its own order --
        its actions (the first `nact` = 6 columns of `actions` take
        |action|), end actions and deformation; its law history into `slip`
        from `committed`. In element order, as the Python loops add them.
        Returns -1, or with `check` the first connector whose ends have met
        (the mixed solver refuses it), before it is evaluated. `d0` is
        (nel, 6), each connector's deformation at its birth, or (0, 6):
        none was born late."""
        none = np.zeros(6)
        late = d0.shape[0] > 0
        u = np.empty((2, 3)); T = np.empty((2, 3, 3)); ve = np.empty(12)
        fe = np.empty(12); act = np.empty(6); d = np.empty(6); so = np.empty(6)
        for r in range(ids.shape[0]):
            e = ids[r]; c = kind[r]
            a = conn[e, 0]; b = conn[e, 1]
            if check:
                dx = (nodes[b, 0] + q2[b, 0]) - (nodes[a, 0] + q2[a, 0])
                dy = (nodes[b, 1] + q2[b, 1]) - (nodes[a, 1] + q2[a, 1])
                dz = (nodes[b, 2] + q2[b, 2]) - (nodes[a, 2] + q2[a, 2])
                if np.sqrt(dx * dx + dy * dy + dz * dz) < 1e-14:
                    return r
            for k in range(3):
                u[0, k] = q2[a, k]; u[1, k] = q2[b, k]
            for i in range(3):
                for j in range(3):
                    T[0, i, j] = orient[a, i, j]; T[1, i, j] = orient[b, i, j]
            for k in range(6):
                ve[k] = v[6 * a + k]; ve[6 + k] = v[6 * b + k]
            x0 = np.empty((2, 3))
            for k in range(3):
                x0[0, k] = nodes[a, k]; x0[1, k] = nodes[b, k]
            for k in range(6):
                so[k] = slip[e, k]
            response_one(u, T, ve, x0, R0[e], spins, K[c], C[c], local[c], has_c[c], ncur[c],
                         cx[c], cy[c], law[c], lp[c], planar[c], committed[e], fe, act, d, so,
                         include_damping, d0[e] if late else none)
            for k in range(6):
                slip[e, k] = so[k]
            if to_rows:
                for k in range(12):
                    rows_out[r, k] = fe[k]
            else:
                for k in range(6):
                    out[6 * a + k] += fe[k]
                for k in range(6):
                    out[6 * b + k] += fe[6 + k]
            for k in range(nact):
                actions[e, k] = act[k] if act[k] >= 0.0 else -act[k]
            for k in range(6):
                end_actions[e, k] = -act[k]; end_actions[e, 6 + k] = act[k]
                deformation[e, k] = d[k]
        return -1

else:                                                  # pragma: no cover
    response_one = respond = None


#: `respond`'s `d0` for a model none of whose connectors was born late
NO_BIRTH = np.zeros((0, 6))


def spins():
    """exp(+h e_k), then exp(-h e_k), k = 0..2: the reference's `_SPINS`."""
    return np.ascontiguousarray(_ref._SPINS)
