"""The axle loads' consistent end actions, compiled (1.6).

`core.moving_load.LoadPath.shape` is the reference: for each axle on the
path, the member under it (path index k) and its position xi from the
member's own node i, the force rotated into the member's axes, the cubic
shape functions for the transverse components and the lever rule for the
axial one, the end releases' condensation, and the rotation back to global.
The arithmetic here is the same, one axle at a time, and `LoadPath.shape`
calls it whenever numba is present, so the step-by-step loop and the
compiled step (kernels/numba_step) are one to the bit by construction.

`vehicle_step` is `Vehicle.__call__`'s step, one vehicle at a time; the
compiled step (kernels/numba_step.moving_step) runs the same functions per
step, every axle into its group's nodal vector in the order `np.add.at`
adds them.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:                                   # pragma: no cover
    njit = None


def available() -> bool:
    return njit is not None


if njit is not None:

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def axle_actions(a, L, R, p, rel, has_rel, fl, fg):
        """One axle: `fl` (12) the member-local end actions of the force `p`
        (global, 3) at xi = `a` on a member of length L and axes R (rows),
        condensed through `rel` (12, 12) when `has_rel`; `fg` (12) the same in
        global axes (T^T fl)."""
        pl0 = 0.0; pl1 = 0.0; pl2 = 0.0
        for j in range(3):
            pl0 += R[0, j] * p[j]
        for j in range(3):
            pl1 += R[1, j] * p[j]
        for j in range(3):
            pl2 += R[2, j] * p[j]
        b = 1.0 - a
        n1 = 1.0 - 3.0 * a * a + 2.0 * a ** 3
        n2 = 3.0 * a * a - 2.0 * a ** 3
        m1 = L * a * b * b
        m2 = -L * a * a * b
        for i in range(12):
            fl[i] = 0.0
        fl[0] = pl0 * b; fl[6] = pl0 * a
        fl[1] = pl1 * n1; fl[7] = pl1 * n2
        fl[5] = pl1 * m1; fl[11] = pl1 * m2
        fl[2] = pl2 * n1; fl[8] = pl2 * n2
        fl[4] = -pl2 * m1; fl[10] = -pl2 * m2
        if has_rel:
            for i in range(12):
                acc = 0.0
                for j in range(12):
                    acc += rel[i, j] * fl[j]
                fg[i] = acc
            for i in range(12):
                fl[i] = fg[i]
        # T^T fl: each 3-block is R^T times the block
        for blk in range(4):
            for i in range(3):
                acc = 0.0
                for j in range(3):
                    acc += R[j, i] * fl[3 * blk + j]
                fg[3 * blk + i] = acc

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def path_actions(k, a, R, L, p, rel, has_rel, fl, fg):
        """`LoadPath.shape` for the axles given: path indices `k` (n,), xi
        `a` (n,), the path members' axes R (npath, 3, 3) and lengths L
        (npath,), forces p (n, 3), releases `rel` (npath, 12, 12) used when
        `has_rel`; fills fl, fg (n, 12)."""
        for r in range(k.shape[0]):
            kk = k[r]
            axle_actions(a[r], L[kk], R[kk], p[r], rel[kk] if has_rel else rel[0], has_rel, fl[r], fg[r])

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def psd_road(x, kn, phi, amp, out):
        """`vehicle.roughness_profile`'s harmonic sum, one station at a time:
        out[i] = sum_k amp[k] cos(kn[k] x[i] + phi[k]), the terms added in
        order -- the same number whether a step asks for its few axles or a
        chunk for thousands of steps at once."""
        for i in range(x.shape[0]):
            acc = 0.0
            for k in range(kn.shape[0]):
                acc += np.cos(x[i] * kn[k] + phi[k]) * amp[k]
            out[i] = acc

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def vehicle_step(q, P_R, P_L, P_rel, P_has, P_edofs, mk, ma, mon, rough, up,
                     nb, fi, flex, m, B, rigid, ks, cs, kt, ct, weight, dt,
                     x, v, acc, contact, surf_prev, has_prev, fl, fg, out, sa, sf):
        """`Vehicle.__call__` for one vehicle and one step, the loop's and the
        compiled step's one arithmetic.

        Its axles: `mk`, `ma`, `mon` (na,) the path member under each, xi from
        the member's node i, and whether it is on the path; `rough` (na,) the
        road's elevation there (zeros without a profile); `up` (3) the unit
        vector against gravity. Its parameters as `Vehicle` holds them: nb
        body dofs, `fi` the tyred axles, `flex` (na,) their mask, `m`,
        `weight` (n,), `B` (na, n), `rigid`, `ks`, `cs`, `kt`, `ct` (na,).
        Its state, advanced here: `x`, `v`, `acc` (n,), `contact`,
        `surf_prev` (na,), `has_prev` (1,) uint8. Scratch: `fl`, `fg`
        (na, 12) left holding the unit-force actions of the axles on the
        path, `sa` (6, na), `sf` (n,). The tyre forces are ADDED to `out`,
        axle by axle, dof by dof. Returns (tyres off the road, rigid axles
        pulling)."""
        na = mk.shape[0]
        n = x.shape[0]
        nf = fi.shape[0]
        surf = sa[0]; rate = sa[1]; S = sa[2]; P = sa[3]
        # the deck under each axle (the consistent actions of a unit force
        # up, dotted with the particles' displacements) plus the road
        for ax in range(na):
            w = 0.0
            if mon[ax]:
                kk = mk[ax]
                axle_actions(ma[ax], P_L[kk], P_R[kk], up, P_rel[kk] if P_has[kk] else P_rel[0],
                             P_has[kk], fl[ax], fg[ax])
                for i in range(12):
                    w += fg[ax, i] * q[P_edofs[kk, i]]
            surf[ax] = w + rough[ax]
            rate[ax] = (surf[ax] - surf_prev[ax]) / dt if has_prev[0] else 0.0
            surf_prev[ax] = surf[ax]
        has_prev[0] = 1
        # Vehicle._forces: suspension extensions e = B x - R s, their rates
        for ax in range(na):
            e = 0.0
            de = 0.0
            for j in range(n):
                e += B[ax, j] * x[j]
            for j in range(n):
                de += B[ax, j] * v[j]
            e = e - rigid[ax] * surf[ax]
            de = de - rigid[ax] * rate[ax]
            S[ax] = -ks[ax] * e - cs[ax] * de
            P[ax] = S[ax]
        for j in range(n):
            a = 0.0
            for ax in range(na):
                a += B[ax, j] * S[ax]
            sf[j] = a - weight[j]
        for r in range(nf):
            ax = fi[r]
            u = nb + r
            d = surf[ax] - x[u]
            dd = rate[ax] - v[u]
            # a tyre pushes only while compressed, and never pulls
            Pt = 0.0
            if d > 0.0:
                val = kt[ax] * d + ct[ax] * dd
                Pt = val if val > 0.0 or val != val else 0.0
            P[ax] = Pt
            sf[u] += Pt
        tyres = 0
        rig = 0
        for ax in range(na):
            if flex[ax]:
                if P[ax] <= 0.0:
                    tyres += 1
            elif P[ax] < 0.0:
                rig += 1
        for j in range(n):
            acc[j] = sf[j] / m[j]
            v[j] = v[j] + acc[j] * dt
            x[j] = x[j] + v[j] * dt
        for ax in range(na):
            contact[ax] = P[ax]
        # each tyre pushes the deck down by P: minus P along "up"
        for ax in range(na):
            if mon[ax]:
                kk = mk[ax]
                for i in range(12):
                    out[P_edofs[kk, i]] += (-P[ax]) * fg[ax, i]
        return tyres, rig

else:                                                  # pragma: no cover
    axle_actions = path_actions = psd_road = vehicle_step = None
