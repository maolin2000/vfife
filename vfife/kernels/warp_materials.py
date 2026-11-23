"""Warp device functions for the material models.

These are the third implementation of each model, after the batched NumPy
reference and the Numba scalar twin.  They are not in the material modules
themselves for one mechanical reason: Warp evaluates a `wp.func`'s argument
annotations against the defining module's globals, and the material files use
`from __future__ import annotations`, which turns those annotations into
strings that then fail to resolve.  So they live here, where `Any` and `wp` are
plain module-level names.

`test_backends.py` runs all three implementations through the same load
histories and requires them to agree, so the separation cannot let them drift.

Every material presents the same signature, so the element kernel factory in
`warp_backend.py` never needs to know which material it is compiling:

    point(prm, tab, strain, state, p, dt, k) -> vec6

`prm` holds the scalars and `tab` the packed curve tables (see Curve.pack).
Two flat buffers, whatever the model.
"""

from typing import Any

import numpy as np
import warp as wp

from .warp_scalar import curve_eval, like, principal


@wp.func
def elastic_point(prm: wp.array(dtype=Any), tab: wp.array(dtype=Any),
                  e: Any, state: wp.array2d(dtype=Any), p: int,
                  dt: Any, k: Any):
    lam = prm[0]
    G = prm[1]
    two = type(lam)(2.0)
    tr = e[0] + e[1] + e[2]
    return wp.vector(lam * tr + two * G * e[0],
                     lam * tr + two * G * e[1],
                     lam * tr + two * G * e[2],
                     G * e[3], G * e[4], G * e[5])


# --------------------------------------------------------------------------- #
#  Chaboche
#
#  prm layout:
#   0 lam  1 G  2 E  3 sy0  4 Q  5 b_iso  6 H_iso
#   7 C1   8 g1 9 C2 10 g2 11 eps_D 12 dD_rate 13 D_max 14 eta
#  15 n_newton 16 has_damage 17 sqrt(3/2) 18 sqrt(2/3) 19 sqrt(6)
# --------------------------------------------------------------------------- #


@wp.func
def chaboche_point(prm: wp.array(dtype=Any), tab: wp.array(dtype=Any),
                   e: Any, state: wp.array2d(dtype=Any), p: int,
                   dt: Any, k: Any):
    lam = prm[0]
    G = prm[1]
    E = prm[2]
    # irrational constants come from prm, packed by NumPy at the target
    # precision: a literal would be rounded to fp32 (see warp_scalar.like)
    SQ32 = prm[17]
    SQ23 = prm[18]
    SQ6 = prm[19]
    two = like(dt, 2.0)

    alpha = state[p, 6]
    D_prev = state[p, 19]

    e0 = e[0] - state[p, 0]
    e1 = e[1] - state[p, 1]
    e2 = e[2] - state[p, 2]
    e3 = e[3] - state[p, 3]
    e4 = e[4] - state[p, 4]
    e5 = e[5] - state[p, 5]
    tr = e0 + e1 + e2
    t0 = lam * tr + two * G * e0
    t1 = lam * tr + two * G * e1
    t2 = lam * tr + two * G * e2
    t3 = G * e3
    t4 = G * e4
    t5 = G * e5

    pm = (t0 + t1 + t2) / like(dt, 3.0)
    xi0 = t0 - pm - state[p, 7] - state[p, 13]
    xi1 = t1 - pm - state[p, 8] - state[p, 14]
    xi2 = t2 - pm - state[p, 9] - state[p, 15]
    xi3 = t3 - state[p, 10] - state[p, 16]
    xi4 = t4 - state[p, 11] - state[p, 17]
    xi5 = t5 - state[p, 12] - state[p, 18]

    nn = (xi0 * xi0 + xi1 * xi1 + xi2 * xi2
          + two * (xi3 * xi3 + xi4 * xi4 + xi5 * xi5))
    xi_norm = wp.sqrt(wp.max(nn, like(dt, 0.0)))
    sy = prm[3] + prm[4] * (like(dt, 1.0) - wp.exp(-prm[5] * alpha)) + prm[6] * alpha
    f_tr = SQ32 * xi_norm - sy

    if f_tr <= like(dt, 1.0e-9) * wp.max(sy, like(dt, 1.0)):
        if prm[14] > like(dt, 0.0):
            D_prev = (prm[14] * D_prev + dt * state[p, 20]) / (prm[14] + dt)
            state[p, 19] = D_prev
        one_d = like(dt, 1.0) - D_prev
        return wp.vector(t0 * one_d, t1 * one_d, t2 * one_d,
                         t3 * one_d, t4 * one_d, t5 * one_d)

    inv = like(dt, 1.0) / wp.max(xi_norm, like(dt, 1.0e-30))
    N0 = xi0 * inv
    N1 = xi1 * inv
    N2 = xi2 * inv
    N3 = xi3 * inv
    N4 = xi4 * inv
    N5 = xi5 * inv

    x1 = (state[p, 7] * N0 + state[p, 8] * N1 + state[p, 9] * N2
          + two * (state[p, 10] * N3 + state[p, 11] * N4 + state[p, 12] * N5))
    x2 = (state[p, 13] * N0 + state[p, 14] * N1 + state[p, 15] * N2
          + two * (state[p, 16] * N3 + state[p, 17] * N4 + state[p, 18] * N5))
    C1 = prm[7]
    g1 = prm[8]
    C2 = prm[9]
    g2 = prm[10]
    k1 = SQ23 * C1
    k2 = SQ23 * C2
    base = xi_norm + x1 + x2

    syp = prm[4] * prm[5] * wp.exp(-prm[5] * alpha) + prm[6]
    den0 = wp.max(like(dt, 3.0) * G + C1 + C2 - g1 * SQ32 * x1 - g2 * SQ32 * x2 + syp,
                  like(dt, 1.0e-12) * E)
    dp = wp.max(f_tr / den0, like(dt, 0.0))

    for _ in range(int(prm[15])):
        d1 = like(dt, 1.0) + g1 * dp
        d2 = like(dt, 1.0) + g2 * dp
        a_new = alpha + dp
        R = SQ32 * (base - SQ6 * G * dp
                    - (x1 + k1 * dp) / d1
                    - (x2 + k2 * dp) / d2) - (
            prm[3] + prm[4] * (like(dt, 1.0) - wp.exp(-prm[5] * a_new)) + prm[6] * a_new)
        dR = SQ32 * (-SQ6 * G
                     - (k1 - g1 * x1) / (d1 * d1)
                     - (k2 - g2 * x2) / (d2 * d2)) - (
            prm[4] * prm[5] * wp.exp(-prm[5] * a_new) + prm[6])
        dR = wp.min(dR, like(dt, -1.0e-12) * E)
        dp = wp.max(dp - R / dR, like(dt, 0.0))

    # the flow rule gives a TENSOR increment; shear doubles into engineering
    c = SQ32 * dp
    state[p, 0] = state[p, 0] + c * N0
    state[p, 1] = state[p, 1] + c * N1
    state[p, 2] = state[p, 2] + c * N2
    state[p, 3] = state[p, 3] + two * c * N3
    state[p, 4] = state[p, 4] + two * c * N4
    state[p, 5] = state[p, 5] + two * c * N5

    alpha_new = alpha + dp
    state[p, 6] = alpha_new
    i1 = like(dt, 1.0) / (like(dt, 1.0) + g1 * dp)
    i2 = like(dt, 1.0) / (like(dt, 1.0) + g2 * dp)
    kd1 = k1 * dp
    kd2 = k2 * dp
    state[p, 7] = (state[p, 7] + kd1 * N0) * i1
    state[p, 8] = (state[p, 8] + kd1 * N1) * i1
    state[p, 9] = (state[p, 9] + kd1 * N2) * i1
    state[p, 10] = (state[p, 10] + kd1 * N3) * i1
    state[p, 11] = (state[p, 11] + kd1 * N4) * i1
    state[p, 12] = (state[p, 12] + kd1 * N5) * i1
    state[p, 13] = (state[p, 13] + kd2 * N0) * i2
    state[p, 14] = (state[p, 14] + kd2 * N1) * i2
    state[p, 15] = (state[p, 15] + kd2 * N2) * i2
    state[p, 16] = (state[p, 16] + kd2 * N3) * i2
    state[p, 17] = (state[p, 17] + kd2 * N4) * i2
    state[p, 18] = (state[p, 18] + kd2 * N5) * i2

    D_new = D_prev
    if prm[16] > like(dt, 0.5):
        a0 = wp.max(alpha, prm[11])
        D_bar = wp.min(state[p, 20] + prm[12] * wp.max(alpha_new - a0, like(dt, 0.0)),
                       prm[13])
        if prm[14] > like(dt, 0.0):
            D_new = (prm[14] * D_prev + dt * D_bar) / (prm[14] + dt)
        else:
            D_new = D_bar
        state[p, 20] = D_bar
    state[p, 19] = D_new

    e0 = e[0] - state[p, 0]
    e1 = e[1] - state[p, 1]
    e2 = e[2] - state[p, 2]
    tr = e0 + e1 + e2
    one_d = like(dt, 1.0) - D_new
    return wp.vector((lam * tr + two * G * e0) * one_d,
                     (lam * tr + two * G * e1) * one_d,
                     (lam * tr + two * G * e2) * one_d,
                     G * (e[3] - state[p, 3]) * one_d,
                     G * (e[4] - state[p, 4]) * one_d,
                     G * (e[5] - state[p, 5]) * one_d)


# --------------------------------------------------------------------------- #
#  Concrete damaged plasticity (Lee-Fenves)
#
#  prm layout:
#   0 lam  1 G  2 K  3 E  4 alpha  5 gamma  6 (1-alpha)  7 tan(psi)  8 a_ecc
#   9 wt  10 wc  11 mu  12 d_max  13 n_return  14 use_kt  15 use_kc
#  16 base_c 17 n_c 18 base_t 19 n_t
#  20 yield tol  21 q floor  22 denominator floor  23 r floor
#
#  The denominator floor is a tenth of the elastic deviatoric term, not a
#  token epsilon: too small a floor lets a strongly softening point take a
#  plastic step of F/epsilon, which reached plastic strains of 1e10.
#
#  No irrational literals appear below, and neither do the epsilons: alpha,
#  gamma, tan(psi), the eccentricity term and every tolerance arrive through prm
#  at the target precision.  The tolerances matter more than they look -- 1e-9
#  written as a literal rounds to 1.0000000134e-09 in fp32, which shifts where
#  the return map decides it has converged and cost 5e-9 against the reference
#  instead of 1e-13.
# --------------------------------------------------------------------------- #


@wp.func
def cdp_point(prm: wp.array(dtype=Any), tab: wp.array(dtype=Any),
              e: Any, state: wp.array2d(dtype=Any), p: int,
              dt: Any, k: Any):
    lam = prm[0]
    G = prm[1]
    K = prm[2]
    E = prm[3]
    a_p = prm[4]
    g_p = prm[5]
    one_m_a = prm[6]
    tanpsi = prm[7]
    a_ecc = prm[8]
    two = like(dt, 2.0)
    three = like(dt, 3.0)
    zero = like(dt, 0.0)
    one = like(dt, 1.0)

    base_c = int(prm[16])
    n_c = int(prm[17])
    base_t = int(prm[18])
    n_t = int(prm[19])
    kt = k
    kc = k
    if prm[14] < like(dt, 0.5):
        kt = one
    if prm[15] < like(dt, 0.5):
        kc = one

    ept = state[p, 6]
    epc = state[p, 7]
    ept0 = ept
    epc0 = epc

    e0 = e[0] - state[p, 0]
    e1 = e[1] - state[p, 1]
    e2 = e[2] - state[p, 2]
    tr = e0 + e1 + e2
    sg0 = lam * tr + two * G * e0
    sg1 = lam * tr + two * G * e1
    sg2 = lam * tr + two * G * e2
    sg3 = G * (e[3] - state[p, 3])
    sg4 = G * (e[4] - state[p, 4])
    sg5 = G * (e[5] - state[p, 5])

    dep0 = zero
    dep1 = zero
    dep2 = zero
    dep3 = zero
    dep4 = zero
    dep5 = zero
    yielded = int(0)
    t3 = tanpsi / three

    for _ in range(int(prm[13])):
        pv = principal(sg0, sg1, sg2, sg3, sg4, sg5)
        smin = pv[0]
        smax = pv[2]
        tr3 = (sg0 + sg1 + sg2) / three
        p_bar = -tr3
        s0 = sg0 - tr3
        s1 = sg1 - tr3
        s2 = sg2 - tr3
        q2 = wp.max(like(dt, 1.5) * (s0 * s0 + s1 * s1 + s2 * s2
                                     + two * (sg3 * sg3 + sg4 * sg4 + sg5 * sg5)),
                    zero)
        q = wp.sqrt(q2)
        q_safe = wp.max(q, prm[21])

        rc = curve_eval(tab, base_c, n_c, epc * kc)
        rt = curve_eval(tab, base_t, n_t, ept * kt)
        sc = rc[0]
        dsc = rc[1] * kc
        st = rt[0]
        dst = rt[1] * kt

        beta = (sc / st) * one_m_a - (one + a_p)
        pos_max = wp.max(smax, zero)
        neg_max = wp.max(-smax, zero)
        F = (q - three * a_p * p_bar + beta * pos_max - g_p * neg_max) / one_m_a - sc
        if F <= prm[20] * sc:
            break
        yielded = 1

        dGdq = q / wp.sqrt(a_ecc * a_ecc + q2)
        coef = dGdq * like(dt, 1.5) / q_safe
        m0 = s0 * coef + t3
        m1 = s1 * coef + t3
        m2 = s2 * coef + t3
        m3 = sg3 * coef
        m4 = sg4 * coef
        m5 = sg5 * coef
        m_max = coef * (smax + p_bar) + t3
        m_min = coef * (smin + p_bar) + t3

        c_max = zero
        if smax > zero:
            c_max = beta
        if smax < zero:
            c_max = g_p
        A = (three * G * dGdq + three * K * a_p * tanpsi
             + c_max * (two * G * (m_max - t3) + K * tanpsi)) / one_m_a

        apv = wp.abs(pv[0]) + wp.abs(pv[1]) + wp.abs(pv[2])
        rp = wp.max(pv[0], zero) + wp.max(pv[1], zero) + wp.max(pv[2], zero)
        r = zero
        if apv > prm[23]:
            r = rp / apv

        h_t = wp.max(r * m_max, zero)
        h_c = wp.max(-(one - r) * m_min, zero)
        dFt = -sc * dst * pos_max / (st * st)
        dFc = dsc * (pos_max / st - one)
        A = wp.max(A - (dFt * h_t + dFc * h_c), prm[22])
        dl = wp.max(F / A, zero)

        # tensor flow direction -> ENGINEERING plastic increment
        q0 = dl * m0
        q1 = dl * m1
        q2p = dl * m2
        q3 = two * dl * m3
        q4 = two * dl * m4
        q5 = two * dl * m5
        trp = q0 + q1 + q2p
        sg0 -= lam * trp + two * G * q0
        sg1 -= lam * trp + two * G * q1
        sg2 -= lam * trp + two * G * q2p
        sg3 -= G * q3
        sg4 -= G * q4
        sg5 -= G * q5

        state[p, 0] = state[p, 0] + q0
        state[p, 1] = state[p, 1] + q1
        state[p, 2] = state[p, 2] + q2p
        state[p, 3] = state[p, 3] + q3
        state[p, 4] = state[p, 4] + q4
        state[p, 5] = state[p, 5] + q5
        dep0 += q0
        dep1 += q1
        dep2 += q2p
        dep3 += q3
        dep4 += q4
        dep5 += q5
        ept += dl * h_t
        epc += dl * h_c

    # --- hardening split from the CONVERGED stress ------------------------
    pv = principal(sg0, sg1, sg2, sg3, sg4, sg5)
    apv = wp.abs(pv[0]) + wp.abs(pv[1]) + wp.abs(pv[2])
    rp = wp.max(pv[0], zero) + wp.max(pv[1], zero) + wp.max(pv[2], zero)
    r = zero
    if apv > prm[23]:
        r = rp / apv

    if yielded == 1:
        half = like(dt, 0.5)
        dpv = principal(dep0, dep1, dep2, half * dep3, half * dep4, half * dep5)
        ept = ept0 + r * wp.max(dpv[2], zero)
        epc = epc0 - (one - r) * wp.min(dpv[0], zero)

    rc = curve_eval(tab, base_c, n_c, epc * kc)
    rt = curve_eval(tab, base_t, n_t, ept * kt)
    s_t = one - prm[9] * r
    s_c = one - prm[10] * (one - r)
    d = wp.clamp(one - (one - s_t * rc[2]) * (one - s_c * rt[2]), zero, prm[12])

    mu = prm[11]
    if mu > zero:
        f = dt / (mu + dt)
        for i in range(6):
            state[p, 8 + i] = state[p, 8 + i] + f * (state[p, i] - state[p, 8 + i])
        state[p, 14] = state[p, 14] + f * (d - state[p, 14])
    else:
        for i in range(6):
            state[p, 8 + i] = state[p, i]
        state[p, 14] = d
    state[p, 6] = ept
    state[p, 7] = epc

    e0 = e[0] - state[p, 8]
    e1 = e[1] - state[p, 9]
    e2 = e[2] - state[p, 10]
    tr = e0 + e1 + e2
    one_d = one - state[p, 14]
    return wp.vector((lam * tr + two * G * e0) * one_d,
                     (lam * tr + two * G * e1) * one_d,
                     (lam * tr + two * G * e2) * one_d,
                     G * (e[3] - state[p, 11]) * one_d,
                     G * (e[4] - state[p, 12]) * one_d,
                     G * (e[5] - state[p, 13]) * one_d)


# --------------------------------------------------------------------------- #
#  J2 plasticity
#
#  prm layout:
#   0 lam  1 G  2 E  3 H_kin  4 n_newton  5 n_table  6 sqrt(3/2)
#   7 yield tolerance  8 denominator floor
#
#  sqrt(3/2) and the tolerances come through prm rather than as literals: a bare
#  literal in Warp device code is fp32 whatever the surrounding precision.
# --------------------------------------------------------------------------- #


@wp.func
def j2_point(prm: wp.array(dtype=Any), tab: wp.array(dtype=Any),
             e: Any, state: wp.array2d(dtype=Any), p: int,
             dt: Any, k: Any):
    lam = prm[0]
    G = prm[1]
    H_kin = prm[3]
    SQ32 = prm[6]
    n_tab = int(prm[5])
    two = like(dt, 2.0)
    three = like(dt, 3.0)

    e0 = e[0] - state[p, 0]
    e1 = e[1] - state[p, 1]
    e2 = e[2] - state[p, 2]
    e3 = e[3] - state[p, 3]
    e4 = e[4] - state[p, 4]
    e5 = e[5] - state[p, 5]
    tr = e0 + e1 + e2
    t0 = lam * tr + two * G * e0
    t1 = lam * tr + two * G * e1
    t2 = lam * tr + two * G * e2
    t3 = G * e3
    t4 = G * e4
    t5 = G * e5

    pm = (t0 + t1 + t2) / three
    x0 = t0 - pm - state[p, 7]
    x1 = t1 - pm - state[p, 8]
    x2 = t2 - pm - state[p, 9]
    x3 = t3 - state[p, 10]
    x4 = t4 - state[p, 11]
    x5 = t5 - state[p, 12]
    nn = (x0 * x0 + x1 * x1 + x2 * x2
          + two * (x3 * x3 + x4 * x4 + x5 * x5))
    nrm = wp.sqrt(wp.max(nn, like(dt, 0.0)))
    q_tr = SQ32 * nrm

    alpha = state[p, 6]
    r = curve_eval(tab, 0, n_tab, alpha)
    sy = r[0]
    if q_tr - sy <= prm[7] * wp.max(sy, like(dt, 1.0)):
        return wp.vector(t0, t1, t2, t3, t4, t5)

    inv = like(dt, 1.0) / wp.max(nrm, like(dt, 1.0e-30))
    n0 = x0 * inv
    n1 = x1 * inv
    n2 = x2 * inv
    n3 = x3 * inv
    n4 = x4 * inv
    n5 = x5 * inv

    base = three * G + H_kin
    dp = wp.max((q_tr - sy) / wp.max(base + r[1], prm[8]), like(dt, 0.0))
    for _ in range(int(prm[4])):
        r = curve_eval(tab, 0, n_tab, alpha + dp)
        R = q_tr - base * dp - r[0]
        dR = wp.min(-base - r[1], -prm[8])
        dp = wp.max(dp - R / dR, like(dt, 0.0))

    # the flow rule gives a TENSOR increment; shear doubles into engineering
    c = SQ32 * dp
    state[p, 0] = state[p, 0] + c * n0
    state[p, 1] = state[p, 1] + c * n1
    state[p, 2] = state[p, 2] + c * n2
    state[p, 3] = state[p, 3] + two * c * n3
    state[p, 4] = state[p, 4] + two * c * n4
    state[p, 5] = state[p, 5] + two * c * n5
    state[p, 6] = alpha + dp
    hk = (two / three) * H_kin * c
    state[p, 7] = state[p, 7] + hk * n0
    state[p, 8] = state[p, 8] + hk * n1
    state[p, 9] = state[p, 9] + hk * n2
    state[p, 10] = state[p, 10] + hk * n3
    state[p, 11] = state[p, 11] + hk * n4
    state[p, 12] = state[p, 12] + hk * n5

    e0 = e[0] - state[p, 0]
    e1 = e[1] - state[p, 1]
    e2 = e[2] - state[p, 2]
    tr = e0 + e1 + e2
    return wp.vector(lam * tr + two * G * e0,
                     lam * tr + two * G * e1,
                     lam * tr + two * G * e2,
                     G * (e[3] - state[p, 3]),
                     G * (e[4] - state[p, 4]),
                     G * (e[5] - state[p, 5]))


# --------------------------------------------------------------------------- #
#  Rebar: uniaxial bilinear steel
#
#  prm layout:  0 E  1 sigma_y  2 H_iso  3 H_kin  4 yield tolerance
#
#  Uniaxial: reads e[0], writes stress[0], leaves the rest at zero.  Only the
#  truss element may use it.
# --------------------------------------------------------------------------- #



@wp.func
def bond_ceb_envelope(prm: wp.array(dtype=Any), a: Any):
    """MC2010 monotonic bond envelope for |slip|. See materials/bond.py.

    prm[7] is the elastic cut-off and prm[8] its secant: below it the branch is
    linear, because alpha < 1 leaves the power law with an infinite tangent at
    the origin and an explicit scheme cannot size a time step against that.
    """
    if a < prm[7]:
        return prm[8] * a
    if a <= prm[1]:
        return prm[0] * wp.pow(a / prm[1], prm[4])
    if a <= prm[2]:
        return prm[0]
    if a <= prm[3]:
        return prm[0] - (prm[0] - prm[5]) * (a - prm[2]) / (prm[3] - prm[2])
    return prm[5]


@wp.func
def bond_ceb_point(prm: wp.array(dtype=Any), tab: wp.array(dtype=Any),
                   e: Any, state: wp.array2d(dtype=Any), p: int,
                   dt: Any, k: Any):
    """Bond stress from slip. `e[0]` is a SLIP in metres, not a strain.

    prm = [tau_max, s1, s2, s3, alpha, tau_f, k_unload]; state = [s_max,
    s_prev, tau_prev]. Mirrors BondCEB.update line for line -- the line of
    slope k_unload from the last point, held inside the virgin envelope at
    the CURRENT slip. (Not at s_max: see the readable version for the force
    jump that rule had on a sign reversal.)
    """
    zero = like(dt, 0.0)
    s = e[0]
    a = wp.abs(s)
    bound = bond_ceb_envelope(prm, a)
    reach = bond_ceb_envelope(prm, wp.max(state[p, 0], a))
    if reach < bound:
        bound = reach
    tau = state[p, 2] + prm[6] * (s - state[p, 1])
    if tau > bound:
        tau = bound
    if tau < -bound:
        tau = -bound
    if a > state[p, 0]:
        state[p, 0] = a
    state[p, 1] = s
    state[p, 2] = tau
    return wp.vector(tau, zero, zero, zero, zero, zero)


@wp.func
def rebar_point(prm: wp.array(dtype=Any), tab: wp.array(dtype=Any),
                e: Any, state: wp.array2d(dtype=Any), p: int,
                dt: Any, k: Any):
    zero = like(dt, 0.0)
    E = prm[0]
    # prm[5] is the accumulated-plastic-strain rupture limit, 0 meaning off.
    # Checked before the trial stress as well as after the update: `alpha` is
    # monotone, so a bar that has ruptured stays ruptured, and reading the
    # limit only after the plastic branch would let an already-dead bar carry
    # load on any step that happened to stay elastic.
    if prm[5] > zero and state[p, 1] >= prm[5]:
        return wp.vector(zero, zero, zero, zero, zero, zero)
    sig_tr = E * (e[0] - state[p, 0])
    xi = sig_tr - state[p, 2]
    f = wp.abs(xi) - (prm[1] + prm[2] * state[p, 1])
    if f <= prm[4]:
        return wp.vector(sig_tr, zero, zero, zero, zero, zero)
    dp = f / (E + prm[2] + prm[3])
    sgn = like(dt, 1.0)
    if xi < zero:
        sgn = like(dt, -1.0)
    state[p, 0] = state[p, 0] + sgn * dp
    state[p, 1] = state[p, 1] + dp
    state[p, 2] = state[p, 2] + prm[3] * sgn * dp
    if prm[5] > zero and state[p, 1] >= prm[5]:
        return wp.vector(zero, zero, zero, zero, zero, zero)
    return wp.vector(E * (e[0] - state[p, 0]), zero, zero, zero, zero, zero)


@wp.func
def bond_cyclic_point(prm: wp.array(dtype=Any), tab: wp.array(dtype=Any),
                      e: Any, state: wp.array2d(dtype=Any), p: int,
                      dt: Any, k: Any):
    """Cyclic bond: MC2010 envelope, energy-driven damage, pinched reversal.

    prm = the nine of `bond_ceb_point` plus [9] E_ref, [10] damage_coeff,
    [11] damage_exponent, [12] tau_fric0, [13] s_reload.
    state = [s_max, s_prev, tau_prev, energy].

    Mirrors BondCyclic.update line for line. The envelope helper is shared
    with the monotonic law, which is the point: the two must not drift apart,
    and a copy of `bond_ceb_envelope` here would eventually do so.
    """
    zero = like(dt, 0.0)
    one = like(dt, 1.0)
    s = e[0]
    a = wp.abs(s)
    s_max = state[p, 0]
    tau_prev = state[p, 2]

    e_n = state[p, 3] / prm[9]
    if e_n < zero:
        e_n = zero
    g = wp.exp(-prm[10] * wp.pow(e_n, prm[11]))

    tau = zero
    if a < s_max:
        reach = g * bond_ceb_envelope(prm, s_max)
        fric = g * prm[12]
        tau = tau_prev + prm[6] * (s - state[p, 1])
        r_p = one - (s_max - s) / prm[13]
        if r_p < zero:
            r_p = zero
        if r_p > one:
            r_p = one
        r_n = one - (s + s_max) / prm[13]
        if r_n < zero:
            r_n = zero
        if r_n > one:
            r_n = one
        cap_p = fric + (reach - fric) * r_p
        cap_n = fric + (reach - fric) * r_n
        if tau > cap_p:
            tau = cap_p
        if tau < -cap_n:
            tau = -cap_n
        # nothing is banked here: damage is driven by the converged load path,
        # not the iteration path (see materials/bond_cyclic.py, commit)
    else:
        tau = g * bond_ceb_envelope(prm, a)
        if s < zero:
            tau = -tau
        state[p, 0] = a

    state[p, 1] = s
    state[p, 2] = tau
    return wp.vector(tau, zero, zero, zero, zero, zero)
