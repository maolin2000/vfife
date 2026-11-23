"""Warp device functions -- the GPU twins of `kernels/scalar.py`.

Written once and generic over precision: Warp resolves `Any` at the call site,
so the same source compiles for fp32 (the element kernel on a consumer GPU,
where fp64 runs at 1/64 rate) and for fp64 (the mode used to check the port
against the NumPy reference to 1e-12).

Quaternion convention: the state array stores (w, x, y, z) to match the NumPy
and Numba backends, while `wp.quat` is (x, y, z, w).  The swap happens at the
kernel boundary, once, so a run can switch backends mid-checkpoint.
"""

from typing import Any

import warp as wp

# Mueller's iteration normalises by tr(S); the exact Newton step needs
# (tr(S) I - S)^-1, which is 3/2 larger for near-spherical stretch.  Same
# constant as the other two backends -- see elements/hexa8.py.
OMEGA_SCALE = 1.5


@wp.func
def like(x: Any, v: float):
    """A literal in x's precision.

    Warp accepts `type(x)(v)` only in call position -- binding the type to a
    name (`T = type(x)`) fails codegen -- so this wraps the one legal form and
    keeps the generic device code readable.

    CAUTION: Warp treats a bare Python literal as fp32 whatever this parameter
    is annotated, so `like` is only safe for values that are EXACT in single
    precision -- small integers, powers of two, epsilon guards.  An irrational
    constant written this way silently rounds to fp32 and drags an fp64 kernel
    down to ~1e-7 against the reference (measured, on sqrt(3/2) in the Chaboche
    return map).  Those come from the material's `prm` buffer instead, packed by
    NumPy at the target precision, or are built from exact operations -- see
    `principal`, which gets pi from wp.acos(-1) rather than a literal.
    """
    return type(x)(v)


@wp.func
def polar_quat(A: Any, q: Any, max_iter: int, tol: Any):
    """Polar rotation of A (3x3), warm-started from quaternion q. Returns the quat."""
    zero = like(tol, 0.0)
    for _ in range(max_iter):
        R = wp.quat_to_matrix(q)
        wx = zero
        wy = zero
        wz = zero
        den = zero
        for c in range(3):
            r0 = R[0, c]
            r1 = R[1, c]
            r2 = R[2, c]
            a0 = A[0, c]
            a1 = A[1, c]
            a2 = A[2, c]
            wx += r1 * a2 - r2 * a1
            wy += r2 * a0 - r0 * a2
            wz += r0 * a1 - r1 * a0
            den += r0 * a0 + r1 * a1 + r2 * a2
        s = like(tol, OMEGA_SCALE) / (wp.abs(den) + like(tol, 1.0e-12))
        wx *= s
        wy *= s
        wz *= s
        w = wp.sqrt(wx * wx + wy * wy + wz * wz)
        if w < tol:
            break
        axis = wp.vector(wx / w, wy / w, wz / w)
        q = wp.normalize(wp.quat_from_axis_angle(axis, w) * q)
    return q


@wp.func
def principal(t0: Any, t1: Any, t2: Any, t3: Any, t4: Any, t5: Any):
    """Eigenvalues of a symmetric 3x3 in TENSOR-Voigt form, ascending, as vec3.

    Closed-form Cardano.  There is no eigensolver to call in a GPU kernel, which
    is exactly why the CDP denominator was rearranged to need only principal
    VALUES (see materials/cdp.py).
    """
    third = like(t0, 1.0) / like(t0, 3.0)
    q = (t0 + t1 + t2) * third
    a = t0 - q
    b = t1 - q
    c = t2 - q
    p1 = t3 * t3 + t4 * t4 + t5 * t5
    p2 = a * a + b * b + c * c + like(t0, 2.0) * p1
    if p2 <= like(t0, 0.0):
        return wp.vector(q, q, q)
    p = wp.sqrt(p2 / like(t0, 6.0))
    safe = wp.max(p, like(t0, 1.0e-30))
    det = (a * (b * c - t4 * t4)
           - t3 * (t3 * c - t4 * t5)
           + t5 * (t3 * t4 - b * t5))
    r = wp.clamp(det / (like(t0, 2.0) * safe * safe * safe),
                 like(t0, -1.0), like(t0, 1.0))
    phi = wp.acos(r) * third
    two_p = like(t0, 2.0) * p
    # 2*pi/3 built from acos(-1): a literal here would be rounded to fp32
    tau3 = like(t0, 2.0) * wp.acos(like(t0, -1.0)) / like(t0, 3.0)
    e1 = q + two_p * wp.cos(phi)
    e3 = q + two_p * wp.cos(phi + tau3)
    return wp.vector(e3, like(t0, 3.0) * q - e1 - e3, e1)


@wp.func
def curve_eval(tab: wp.array(dtype=Any), base: int, n: int, x: Any):
    """(effective stress, slope, damage) from a packed piecewise-linear table.

    The four columns live contiguously in one array so a material needs exactly
    two buffers on the device whatever its shape:
        tab[base + 0*n .. ] ep
        tab[base + 1*n .. ] sigma_bar
        tab[base + 2*n .. ] damage
        tab[base + 3*n .. ] slope        (n-1 entries used)

    Holds the last stress flat past the end of the table -- extrapolating a
    softening branch into negative stress wrecks a run faster than anything else.
    """
    e = base
    s = base + n
    d = base + 2 * n
    sl = base + 3 * n
    if x >= tab[e + n - 1]:
        return wp.vector(tab[s + n - 1], like(x, 0.0), tab[d + n - 1])
    if x <= tab[e]:
        return wp.vector(tab[s], tab[sl], tab[d])
    lo = int(0)
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) / 2
        if tab[e + mid] <= x:
            lo = mid
        else:
            hi = mid
    dx = tab[e + lo + 1] - tab[e + lo]
    t = x - tab[e + lo]
    return wp.vector(tab[s + lo] + tab[sl + lo] * t,
                     tab[sl + lo],
                     tab[d + lo] + (tab[d + lo + 1] - tab[d + lo]) / dx * t)
