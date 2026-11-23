"""The compiled kernels of the contact between members (core/member_contact,
v1.11): the closest points of two segments, the sweep that finds the
pairs, the neighbour list, the forces -- and `contact_step`, one evaluation
whole, which the step-by-step loop (`MemberContact.force`) and the compiled
step (`numba_step`) both call.

They live here, with the other kernels, because the compiled step calls
them: numba's cache of a function does not see another file change, and
`kernels/_cache` watches this folder.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception:                                      # pragma: no cover
    njit = None


def _closest(p1, q1, p2, q2):
    """Closest points of the segments p1-q1 and p2-q2: (s, t), the
    parameters along each (Ericson, Real-Time Collision Detection, 5.1.9)."""
    d1x = q1[0] - p1[0]; d1y = q1[1] - p1[1]; d1z = q1[2] - p1[2]
    d2x = q2[0] - p2[0]; d2y = q2[1] - p2[1]; d2z = q2[2] - p2[2]
    rx = p1[0] - p2[0]; ry = p1[1] - p2[1]; rz = p1[2] - p2[2]
    a = d1x * d1x + d1y * d1y + d1z * d1z
    e = d2x * d2x + d2y * d2y + d2z * d2z
    f = d2x * rx + d2y * ry + d2z * rz
    tiny = 1e-300
    if a <= tiny and e <= tiny:
        return 0.0, 0.0
    if a <= tiny:
        s = 0.0
        t = min(max(f / e, 0.0), 1.0)
    else:
        c = d1x * rx + d1y * ry + d1z * rz
        if e <= tiny:
            t = 0.0
            s = min(max(-c / a, 0.0), 1.0)
        else:
            b = d1x * d2x + d1y * d2y + d1z * d2z
            den = a * e - b * b
            # parallel axes: any point of the overlap is a closest point; the
            # MIDDLE of it, so that two parallel members rest on each other
            # without a couple (the textbook's s = 0 put the whole force at
            # one end)
            if den > 1e-12 * a * e:
                s = min(max((b * f - c * e) / den, 0.0), 1.0)
            else:
                # the range of s whose foot on segment 2 lies inside it
                s0 = -c / a
                s1 = (b - c) / a
                lo = min(s0, s1); hi = max(s0, s1)
                lo = min(max(lo, 0.0), 1.0); hi = min(max(hi, 0.0), 1.0)
                s = 0.5 * (lo + hi)
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t = 1.0
                s = min(max((b - c) / a, 0.0), 1.0)
    return s, t


def _find(x, conn, rad, cap):
    """The overlapping pairs of capsules: sort the members' boxes along the
    longest axis of the model and sweep. Returns (n, i, j, s, t, overlap)
    with i < j indices into `conn`; n > cap means the arrays were too short
    (the caller asks again)."""
    m = conn.shape[0]
    pi = np.empty(cap, np.int64); pj = np.empty(cap, np.int64)
    ps = np.empty(cap); pt = np.empty(cap); po = np.empty(cap)
    if m < 2:
        return 0, pi, pj, ps, pt, po
    lo = np.empty((m, 3)); hi = np.empty((m, 3))
    for e in range(m):
        a = conn[e, 0]; b = conn[e, 1]; r = rad[e]
        for k in range(3):
            u = x[a, k]; w = x[b, k]
            if u < w:
                lo[e, k] = u - r; hi[e, k] = w + r
            else:
                lo[e, k] = w - r; hi[e, k] = u + r
    ext0 = 0.0; ax = 0
    for k in range(3):
        mn = lo[0, k]; mx = hi[0, k]
        for e in range(1, m):
            if lo[e, k] < mn:
                mn = lo[e, k]
            if hi[e, k] > mx:
                mx = hi[e, k]
        if mx - mn > ext0:
            ext0 = mx - mn; ax = k
    order = np.argsort(lo[:, ax], kind="mergesort")
    a1 = (ax + 1) % 3; a2 = (ax + 2) % 3
    n = 0
    for ii in range(m):
        i = order[ii]
        top = hi[i, ax]
        for jj in range(ii + 1, m):
            j = order[jj]
            if lo[j, ax] > top:
                break
            if lo[j, a1] > hi[i, a1] or lo[i, a1] > hi[j, a1] or lo[j, a2] > hi[i, a2] or lo[i, a2] > hi[j, a2]:
                continue
            ia = conn[i, 0]; ib = conn[i, 1]; ja = conn[j, 0]; jb = conn[j, 1]
            if ia == ja or ia == jb or ib == ja or ib == jb:
                continue
            e1 = i if i < j else j
            e2 = j if i < j else i
            s, t = _closest(x[conn[e1, 0]], x[conn[e1, 1]], x[conn[e2, 0]], x[conn[e2, 1]])
            dx = 0.0
            for k in range(3):
                c1 = (1.0 - s) * x[conn[e1, 0], k] + s * x[conn[e1, 1], k]
                c2 = (1.0 - t) * x[conn[e2, 0], k] + t * x[conn[e2, 1], k]
                dx += (c2 - c1) * (c2 - c1)
            over = rad[e1] + rad[e2] - np.sqrt(dx)
            if over > 0.0:
                if n < cap:
                    pi[n] = e1; pj[n] = e2; ps[n] = s; pt[n] = t; po[n] = over
                n += 1
    return n, pi, pj, ps, pt, po


def _forces(n, pi, pj, ps, pt, x, v, w, conn, rad, gid, nel, mass, inertia, dt, k_user, mu, zeta, dynamic,
            keys, vals, slack, f, mom, pen_out):
    """The contact forces of the pairs `_find` returned (some of which may
    not overlap: the caller searches a little wide), into f and mom
    (nn, 3): on the points of both members. `keys`/`vals` are the pairs'
    allowances (key = global element ids e1 * nel + e2, sorted); they shrink
    here and the ones no overlapping pair claims any more go to zero.
    Returns (active pairs, deepest overlap past the allowance, a user
    stiffness was capped)."""
    nn = x.shape[0]
    count = np.zeros(nn, np.int64)
    seen = np.zeros(keys.shape[0], np.bool_)
    allow = np.zeros(n)
    for c in range(n):
        i = pi[c]; j = pj[c]
        a = 0.0
        if keys.shape[0]:
            g1 = gid[i]; g2 = gid[j]
            key = (g1 * nel + g2) if g1 < g2 else (g2 * nel + g1)
            pos = np.searchsorted(keys, key)
            if pos < keys.shape[0] and keys[pos] == key:
                seen[pos] = True
                a = vals[pos]
        allow[c] = a
    # the geometric overlap of each pair again, for the allowance and the force
    over = np.empty(n)
    nrm = np.empty((n, 3))
    for c in range(n):
        i = pi[c]; j = pj[c]; s = ps[c]; t = pt[c]
        d2 = 0.0
        for k in range(3):
            c1 = (1.0 - s) * x[conn[i, 0], k] + s * x[conn[i, 1], k]
            c2 = (1.0 - t) * x[conn[j, 0], k] + t * x[conn[j, 1], k]
            nrm[c, k] = c2 - c1
            d2 += (c2 - c1) * (c2 - c1)
        d = np.sqrt(d2)
        over[c] = rad[i] + rad[j] - d
        if d > 1e-9 * (rad[i] + rad[j]):
            for k in range(3):
                nrm[c, k] /= d
        else:
            # the axes cross: any direction normal to both
            u0 = x[conn[i, 1], 0] - x[conn[i, 0], 0]; u1 = x[conn[i, 1], 1] - x[conn[i, 0], 1]
            u2 = x[conn[i, 1], 2] - x[conn[i, 0], 2]
            w0 = x[conn[j, 1], 0] - x[conn[j, 0], 0]; w1 = x[conn[j, 1], 1] - x[conn[j, 0], 1]
            w2 = x[conn[j, 1], 2] - x[conn[j, 0], 2]
            c0 = u1 * w2 - u2 * w1; c1_ = u2 * w0 - u0 * w2; c2_ = u0 * w1 - u1 * w0
            cl = np.sqrt(c0 * c0 + c1_ * c1_ + c2_ * c2_)
            if cl > 1e-12 * (np.sqrt(u0 * u0 + u1 * u1 + u2 * u2) * np.sqrt(w0 * w0 + w1 * w1 + w2 * w2) + 1e-300):
                nrm[c, 0] = c0 / cl; nrm[c, 1] = c1_ / cl; nrm[c, 2] = c2_ / cl
            else:
                # collinear: normal to the axis, towards whichever global axis it is least along
                ul = np.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
                if ul < 1e-150:                         # no axis at all: member j's, or up
                    u0 = w0; u1 = w1; u2 = w2
                    ul = np.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
                    if ul < 1e-150:
                        u0 = 1.0; u1 = 0.0; u2 = 0.0; ul = 1.0
                a0 = abs(u0) / ul; a1 = abs(u1) / ul; a2 = abs(u2) / ul
                if a0 <= a1 and a0 <= a2:
                    e0 = 1.0; e1 = 0.0; e2 = 0.0
                elif a1 <= a2:
                    e0 = 0.0; e1 = 1.0; e2 = 0.0
                else:
                    e0 = 0.0; e1 = 0.0; e2 = 1.0
                dd = (e0 * u0 + e1 * u1 + e2 * u2) / (ul * ul)
                e0 -= dd * u0; e1 -= dd * u1; e2 -= dd * u2
                el = np.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
                nrm[c, 0] = e0 / el; nrm[c, 1] = e1 / el; nrm[c, 2] = e2 / el
        # an allowance only shrinks: to the overlap the pair has now, and the slack
        if allow[c] > 0.0:
            top = over[c] + slack * (rad[i] if rad[i] < rad[j] else rad[j])
            if allow[c] > top:
                allow[c] = top if top > 0.0 else 0.0
        if over[c] - allow[c] > 0.0:
            count[conn[i, 0]] += 1; count[conn[i, 1]] += 1
            count[conn[j, 0]] += 1; count[conn[j, 1]] += 1
    active = 0
    deepest = 0.0
    capped = False
    for c in range(n):
        i = pi[c]; j = pj[c]
        pen = over[c] - allow[c]
        if keys.shape[0]:
            g1 = gid[i]; g2 = gid[j]
            key = (g1 * nel + g2) if g1 < g2 else (g2 * nel + g1)
            pos = np.searchsorted(keys, key)
            if pos < keys.shape[0] and keys[pos] == key:
                vals[pos] = allow[c]
        if pen <= 0.0:
            continue
        s = ps[c]; t = pt[c]
        ia = conn[i, 0]; ib = conn[i, 1]; ja = conn[j, 0]; jb = conn[j, 1]
        # the lighter of each member's two points; a HELD point's mass is
        # infinite (the caller says so with a value above 1e290): a beam
        # landing on a fixed one is stopped by its own mass, as a node at a
        # rigid surface is, whatever density the fixed one was given
        m1 = mass[ia] if mass[ia] < mass[ib] else mass[ib]
        m2 = mass[ja] if mass[ja] < mass[jb] else mass[jb]
        if m1 < 1e-30:
            m1 = 1e-30
        if m2 < 1e-30:
            m2 = 1e-30
        if m1 > 1e290 and m2 > 1e290:
            continue                                    # both held: nothing can move
        if m1 > 1e290:
            me = m2
        elif m2 > 1e290:
            me = m1
        else:
            me = m1 * m2 / (m1 + m2)
        cnt = count[ia]
        if count[ib] > cnt:
            cnt = count[ib]
        if count[ja] > cnt:
            cnt = count[ja]
        if count[jb] > cnt:
            cnt = count[jb]
        kmax = me / (dt * dt) / cnt
        if not dynamic:
            # a relaxation's kinetic damping and a spring at omega dt = 1 lock
            # into a three-step cycle (found in review): omega dt = 1/2 there
            kmax = 0.25 * kmax
        k = kmax
        if k_user > 0.0:
            if k_user < kmax:
                k = k_user
            else:
                capped = True
        fn = k * pen
        n0 = nrm[c, 0]; n1 = nrm[c, 1]; n2 = nrm[c, 2]
        # THE CONTACT POINT is one point, the middle of the overlap: ri and rj
        # below are its distances from the two axes, and they add up to the
        # axes' distance. With the capsules' own radii the two "surface
        # points" are an overlap apart and friction leaves a couple
        # overlap x F_t that belongs to nobody (the moments of the test
        # tests/test_member_contact.py did not balance)
        ri = rad[i] - 0.5 * over[c]; rj = rad[j] - 0.5 * over[c]
        # A MEMBER WHOSE POINTS HAVE NO ROTARY INERTIA -- a bar, a cable --
        # cannot take a couple: its points' rotational mass is a floor of
        # 1e-20, and friction's couple there span them to 1e19 rad/s, which
        # came back through the lever as a velocity (found in review). For
        # such a member the contact point is ON its axis (the other member's
        # lever is then the whole distance, so the system is still null)
        Ii = inertia[ia] if inertia[ia] < inertia[ib] else inertia[ib]
        Ij = inertia[ja] if inertia[ja] < inertia[jb] else inertia[jb]
        rot_i = Ii > 0.0
        rot_j = Ij > 0.0
        dist = ri + rj
        if not rot_i and not rot_j:
            ri = 0.0; rj = 0.0
        elif not rot_i:
            ri = 0.0; rj = dist
        elif not rot_j:
            rj = 0.0; ri = dist
        ft0 = 0.0; ft1 = 0.0; ft2 = 0.0
        if dynamic:
            # the contact point's velocity on each member: the axis point's,
            # plus the member's spin about it times the lever
            r0 = 0.0; r1 = 0.0; r2 = 0.0
            wi = np.zeros(3); wj = np.zeros(3); vi = np.zeros(3); vj = np.zeros(3)
            for kk in range(3):
                vi[kk] = (1.0 - s) * v[ia, kk] + s * v[ib, kk]
                vj[kk] = (1.0 - t) * v[ja, kk] + t * v[jb, kk]
                wi[kk] = (1.0 - s) * w[ia, kk] + s * w[ib, kk]
                wj[kk] = (1.0 - t) * w[ja, kk] + t * w[jb, kk]
            # v_i + w_i x (ri n),  v_j + w_j x (-rj n)
            r0 = (vj[0] - rj * (wj[1] * n2 - wj[2] * n1)) - (vi[0] + ri * (wi[1] * n2 - wi[2] * n1))
            r1 = (vj[1] - rj * (wj[2] * n0 - wj[0] * n2)) - (vi[1] + ri * (wi[2] * n0 - wi[0] * n2))
            r2 = (vj[2] - rj * (wj[0] * n1 - wj[1] * n0)) - (vi[2] + ri * (wi[0] * n1 - wi[1] * n0))
            vn = r0 * n0 + r1 * n1 + r2 * n2            # < 0: closing
            if zeta > 0.0:
                cd = 2.0 * zeta * np.sqrt(k * me)
                cmax = me / dt / cnt
                if cd > cmax:
                    cd = cmax
                fn = fn - cd * vn
                if fn < 0.0:
                    fn = 0.0                            # it never pulls
            if mu > 0.0 and fn > 0.0:
                t0 = r0 - vn * n0; t1 = r1 - vn * n1; t2 = r2 - vn * n2
                speed = np.sqrt(t0 * t0 + t1 * t1 + t2 * t2)
                # the regularisation velocity keeps the equivalent dashpot
                # mu F_n / v_reg inside the explicit step -- through the
                # LEVERS too: at a radius r the same force turns the member
                # with r^2 / I, and with the translation alone a square bar
                # rolling on two rails had its couple flip sign every step
                # (ledger error 30%; found in review)
                inv = 1.0 / me
                if rot_i and Ii < 1e290:
                    inv += ri * ri / Ii
                if rot_j and Ij < 1e290:
                    inv += rj * rj / Ij
                vreg = 2.0 * mu * fn * dt * cnt * inv
                if vreg < 1e-6:
                    vreg = 1e-6
                sc = mu * fn / (speed if speed > vreg else vreg)
                # on member j, against its velocity relative to i
                ft0 = -sc * t0; ft1 = -sc * t1; ft2 = -sc * t2
        # on member j: +fn n + ft ; on member i: the opposite
        F0 = fn * n0 + ft0; F1 = fn * n1 + ft1; F2 = fn * n2 + ft2
        f[ja, 0] += (1.0 - t) * F0; f[ja, 1] += (1.0 - t) * F1; f[ja, 2] += (1.0 - t) * F2
        f[jb, 0] += t * F0; f[jb, 1] += t * F1; f[jb, 2] += t * F2
        f[ia, 0] -= (1.0 - s) * F0; f[ia, 1] -= (1.0 - s) * F1; f[ia, 2] -= (1.0 - s) * F2
        f[ib, 0] -= s * F0; f[ib, 1] -= s * F1; f[ib, 2] -= s * F2
        if ft0 != 0.0 or ft1 != 0.0 or ft2 != 0.0:
            # friction acts at the contact point, off each axis: (-rj n) x ft on j, (ri n) x (-ft) on i
            q0 = n1 * ft2 - n2 * ft1; q1 = n2 * ft0 - n0 * ft2; q2 = n0 * ft1 - n1 * ft0
            mom[ja, 0] -= (1.0 - t) * rj * q0; mom[ja, 1] -= (1.0 - t) * rj * q1; mom[ja, 2] -= (1.0 - t) * rj * q2
            mom[jb, 0] -= t * rj * q0; mom[jb, 1] -= t * rj * q1; mom[jb, 2] -= t * rj * q2
            mom[ia, 0] -= (1.0 - s) * ri * q0; mom[ia, 1] -= (1.0 - s) * ri * q1; mom[ia, 2] -= (1.0 - s) * ri * q2
            mom[ib, 0] -= s * ri * q0; mom[ib, 1] -= s * ri * q1; mom[ib, 2] -= s * ri * q2
        active += 1
        if pen > deepest:
            deepest = pen
        for node in (ia, ib, ja, jb):
            if pen > pen_out[node]:
                pen_out[node] = pen
    for p in range(keys.shape[0]):
        if not seen[p]:
            vals[p] = 0.0                               # more than the slack apart: an ordinary pair now
    return active, deepest, capped


def _narrow(nc, ci, cj, x, conn, rad, cap):
    """Of the candidate pairs (ci, cj), the ones that overlap now: the
    arrays `_find` returns."""
    pi = np.empty(cap, np.int64); pj = np.empty(cap, np.int64)
    ps = np.empty(cap); pt = np.empty(cap); po = np.empty(cap)
    n = 0
    for c in range(nc):
        e1 = ci[c]; e2 = cj[c]
        s, t = _closest(x[conn[e1, 0]], x[conn[e1, 1]], x[conn[e2, 0]], x[conn[e2, 1]])
        dx = 0.0
        for k in range(3):
            c1 = (1.0 - s) * x[conn[e1, 0], k] + s * x[conn[e1, 1], k]
            c2 = (1.0 - t) * x[conn[e2, 0], k] + t * x[conn[e2, 1], k]
            dx += (c2 - c1) * (c2 - c1)
        over = rad[e1] + rad[e2] - np.sqrt(dx)
        if over > 0.0:
            if n < cap:
                pi[n] = e1; pj[n] = e2; ps[n] = s; pt[n] = t; po[n] = over
            n += 1
    return n, pi, pj, ps, pt, po


def _moved2(x, ref):
    """The largest squared displacement of a point since `ref`."""
    worst = 0.0
    for i in range(x.shape[0]):
        d = 0.0
        for k in range(3):
            d += (x[i, k] - ref[i, k]) * (x[i, k] - ref[i, k])
        if d > worst:
            worst = d
    return worst


def _keep_list(x, conn, rad_s, ci, cj, lstate, ref, stats):
    """The neighbour list of `contact_step`, kept or made again: the number
    of pairs in ci, cj, or -(pairs found) when they are too short (nothing
    changed then)."""
    m = conn.shape[0]
    cap = ci.shape[0]
    skin = stats[3]
    if lstate[2] == 0 and (m == 0 or 4.0 * _moved2(x, ref) < skin * skin):
        # (no member at all: the empty list is kept. Its skin is zero, it
        # was made again at every call, and the fibre step, which asks for
        # the list and then for the forces, counted two sweeps a step where
        # the loop counted one -- review)
        nc = lstate[0]
    else:
        skin = 0.0
        if m:
            skin = rad_s[0]
            for e in range(1, m):
                if rad_s[e] < skin:
                    skin = rad_s[e]
            skin = 0.5 * skin
        wide = np.empty(m)
        for e in range(m):
            wide[e] = rad_s[e] + 0.5 * skin
        n, pi, pj, _s, _t, _o = _find(x, conn, wide, cap)
        if n > cap:
            return -n
        key = np.empty(n, np.int64)
        for k in range(n):
            key[k] = pi[k] * m + pj[k]
        order = np.argsort(key, kind="mergesort")
        for k in range(n):
            ci[k] = pi[order[k]]; cj[k] = pj[order[k]]
        lstate[0] = n; lstate[1] += 1; lstate[2] = 0
        stats[3] = skin
        for i in range(x.shape[0]):
            for k in range(3):
                ref[i, k] = x[i, k]
        nc = n
    return nc


def contact_step(x, v, w, conn, rad, rad_s, gid, nel, mass, inertia, dt, k_user, mu, zeta, dynamic,
                 keys, vals, slack, ci, cj, lstate, ref, f, mom, pen, stats):
    """ONE EVALUATION of the contact, the whole of it: the neighbour list
    (kept in ci, cj, lstate = [pairs listed, sweeps made, make-it-again
    flag], ref and stats[3], the skin), the pairs that overlap now, their
    forces into f and mom and the deepest overlaps into pen (all three
    zeroed first), the allowances `vals` shrunk. stats[0:3] = active pairs,
    deepest overlap past the allowance, a user stiffness was capped.

    Written once and compiled: `MemberContact.force` calls it from the
    step-by-step loop and the compiled step calls it from inside its chunk,
    so the two give the same forces to the bit by
    construction.

    THE NEIGHBOUR LIST: the sweep over every member's box cost more than a
    step of the Xihoumen model (1.05 ms for 1207 members against 0.9), so
    it is made with the search radii `rad_s` a skin larger and kept until
    some point has moved half the skin -- no pair outside the list can have
    closed the skin before then. It is stored in the order of the pairs'
    numbers, not of the sweep: the forces are then added in an order that
    does not depend on WHEN the list was made, and a restart (which makes a
    new one) gives the same bits.

    Returns 0, or -(pairs found) when ci and cj are too short for the list:
    nothing has been changed then, and the caller comes back with longer
    ones."""
    nc = _keep_list(x, conn, rad_s, ci, cj, lstate, ref, stats)
    if nc < 0:
        return nc
    n2, pi, pj, ps, pt, _po = _narrow(nc, ci, cj, x, conn, rad_s, nc if nc > 0 else 1)
    for i in range(x.shape[0]):
        pen[i] = 0.0
        for k in range(3):
            f[i, k] = 0.0; mom[i, k] = 0.0
    active, deepest, capped = _forces(n2, pi, pj, ps, pt, x, v, w, conn, rad, gid, nel, mass, inertia, dt,
                                      k_user, mu, zeta, dynamic, keys, vals, slack, f, mom, pen)
    stats[0] = active; stats[1] = deepest
    if capped:
        stats[2] = 1.0
    return 0


if njit is not None:
    _closest = njit(cache=True, fastmath=False, nogil=True, error_model="numpy")(_closest)
    _find = njit(cache=True, fastmath=False, nogil=True, error_model="numpy")(_find)
    _forces = njit(cache=True, fastmath=False, nogil=True, error_model="numpy")(_forces)
    _narrow = njit(cache=True, fastmath=False, nogil=True, error_model="numpy")(_narrow)
    _moved2 = njit(cache=True, fastmath=False, nogil=True, error_model="numpy")(_moved2)
    _keep_list = njit(cache=True, fastmath=False, nogil=True, error_model="numpy")(_keep_list)
    contact_step = njit(cache=True, fastmath=False, nogil=True, error_model="numpy")(contact_step)
