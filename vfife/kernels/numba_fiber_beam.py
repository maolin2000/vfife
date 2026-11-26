"""Fused Numba element kernel for beam2_fiber.

The NumPy path in `core/fiber_beam_solver.py` is the readable reference. It is
also, measured on `cases/case_11`, almost entirely Python overhead: 300
internal-force evaluations spend a third of their time in the loop body itself,
another ~13% in the co-rotational frame (`np.cross`, `np.kron` and friends on
3x3 and 12x12 objects), and allocate 282 small arrays per evaluation for
B-matrices whose entries depend only on the Gauss station and the element
length -- both constant for the life of the analysis.

This kernel does one element in one pass with scalar locals: gather, rotate,
fibre strains, constitutive update, section resultants, Hermite virtual work,
rotate back. The two uniaxial laws enter as scalar device functions
(`Material.numba_uniaxial_fn()`) and are inlined, exactly as
`numba_hexa8.make_kernel` inlines a solid law.

WHY range AND NOT prange
------------------------
The solid kernel parallelises over elements and needs a gather assembler to
avoid races on shared nodes. A fibre beam model is tens of elements, not
hundreds of thousands; threading would buy nothing and the race is real, so
this loop is serial and writes straight into `out`. The win here is removing
Python, not adding cores.

WHAT IT DOES NOT COVER
----------------------
Two constitutive law TYPES: one for the section fibres, one for the rebars.
Their PARAMETERS are free per element and per fibre -- `cpar[e, f]` and
`spar[e, f]` -- so two steel grades in one section, or a hinge element whose
bars carry a different slip length from the rest, are ordinary cases and stay
on the fast path. So is the GEOMETRY, since v0.1: `cy[e, f]`,
`cz[e, f]`, `ca[e, f]` and `G0J0[e]`, so a model whose members carry different
fibre sections (padded to one fibre count with zero-area fibres) is one
kernel, not one per section. Only a second law type (steel02 alongside a bilinear rebar,
say) cannot be expressed as two inlined point functions, and the solver keeps
the NumPy path for that.

The per-fibre parameter arrays cost nothing measurable: 22.99 us against
23.09 us for the global-parameter version on case_11, with the internal force
identical to the last bit,
because they are a few hundred doubles that never leave cache. Drawing the
limit at the law TYPE rather than at the parameters is what lets cover and
confined core be different concretes, which is the same shape of change.
"""

from __future__ import annotations

import numpy as np

try:                                            # numba is optional
    from numba import njit, prange
    from .numba_rotations import member_frame
except Exception:                               # pragma: no cover
    njit = None


def precompute_shape_derivatives(length, gauss_s, phi_y=None, phi_z=None):
    """Bky/Bkz per (element, station). Constant, so build them once.

    These are the curvature-displacement rows of the Hermite basis. They depend
    on the station s and the element length only, and the reference path
    rebuilt all four of them, plus four 12-vectors of zeros, inside the Gauss
    loop of every internal-force evaluation.

    With `phi_y` / `phi_z` (per element, 12 E I / (G A_s L^2)) they are the
    curvature rows of Reddy's interdependent Timoshenko interpolation:
    mu = 1/(1 + phi) scales every entry and phi enters the
    rotation entries; at phi = 0 every expression reduces to the Hermite one
    exactly (the same floating-point operations on 0.0), so a member without
    shear is bit-identical to what it was.
    """
    n_elem = len(length)
    n_gp = len(gauss_s)
    Bky = np.zeros((n_elem, n_gp, 4))
    Bkz = np.zeros((n_elem, n_gp, 4))
    for e in range(n_elem):
        L0 = float(length[e])
        py = 0.0 if phi_y is None else float(phi_y[e])
        pz = 0.0 if phi_z is None else float(phi_z[e])
        for g, s in enumerate(gauss_s):
            if pz == 0.0:
                Bkz[e, g, 0] = (-6.0 + 12.0 * s) / L0 ** 2
                Bkz[e, g, 1] = (-4.0 + 6.0 * s) / L0
                Bkz[e, g, 2] = (6.0 - 12.0 * s) / L0 ** 2
                Bkz[e, g, 3] = (-2.0 + 6.0 * s) / L0
            else:
                mu = 1.0 / (1.0 + pz)
                Bkz[e, g, 0] = mu * (-6.0 + 12.0 * s) / L0 ** 2
                Bkz[e, g, 1] = mu * (-4.0 + 6.0 * s - pz) / L0
                Bkz[e, g, 2] = mu * (6.0 - 12.0 * s) / L0 ** 2
                Bkz[e, g, 3] = mu * (-2.0 + 6.0 * s + pz) / L0
            if py == 0.0:
                Bky[e, g, 0] = (6.0 - 12.0 * s) / L0 ** 2
                Bky[e, g, 1] = (-4.0 + 6.0 * s) / L0
                Bky[e, g, 2] = (-6.0 + 12.0 * s) / L0 ** 2
                Bky[e, g, 3] = (-2.0 + 6.0 * s) / L0
            else:
                mu = 1.0 / (1.0 + py)
                Bky[e, g, 0] = mu * (6.0 - 12.0 * s) / L0 ** 2
                Bky[e, g, 1] = mu * (-4.0 + 6.0 * s - py) / L0
                Bky[e, g, 2] = mu * (-6.0 + 12.0 * s) / L0 ** 2
                Bky[e, g, 3] = mu * (-2.0 + 6.0 * s + py) / L0
    return Bky, Bkz


def precompute_shear_rows(length, phi_y, phi_z):
    """The constant shear-strain rows of the interdependent interpolation,
    per element: gamma = Bg . [v_i, theta_i, v_j, theta_j] with
    Bg = mu phi [-1/L, -1/2, 1/L, -1/2] in the z-bending plane (local y
    displacements, rotations about z) and its mirror in the y-bending plane
    (local z displacements w with theta_y = -w'). Zero rows where phi = 0.
    """
    n_elem = len(length)
    Bgy = np.zeros((n_elem, 4))
    Bgz = np.zeros((n_elem, 4))
    for e in range(n_elem):
        L0 = float(length[e])
        pz = float(phi_z[e]); py = float(phi_y[e])
        if pz != 0.0:
            c = pz / (1.0 + pz)
            Bgz[e, 0] = -c / L0; Bgz[e, 1] = -0.5 * c
            Bgz[e, 2] = c / L0;  Bgz[e, 3] = -0.5 * c
        if py != 0.0:
            c = py / (1.0 + py)
            Bgy[e, 0] = c / L0;  Bgy[e, 1] = -0.5 * c
            Bgy[e, 2] = -c / L0; Bgy[e, 3] = -0.5 * c
    return Bgy, Bgz


_KERNEL_CACHE = {}


if njit is not None:

    @njit(cache=True, fastmath=False, nogil=True)
    def shear_law_point(g, gcr, gmax, GA0, GAcr, Vy, h):
        """(V, tangent) of the cracking shear law at gamma `g` with its
        history (the crack strain `gcr`, inf before cracking; the largest
        |gamma| `gmax`): `FiberBeamSolver._shear_law`,
        the same arithmetic in the same order (1.7)."""
        a = abs(g)
        sgn = 1.0 if g >= 0.0 else -1.0
        if not np.isfinite(gcr) or max(a, gmax) <= gcr:
            return GA0 * g, GA0
        Vc = GA0 * gcr
        gy = gcr + (Vy - Vc) / GAcr if np.isfinite(Vy) else np.inf
        if a >= gmax:
            # the envelope at a
            if a <= gcr:
                v = GA0 * a; k = GA0
            elif a <= gy:
                v = Vc + GAcr * (a - gcr); k = GAcr
            else:
                hh = h * GAcr
                v = Vy + hh * (a - gy); k = hh
            return sgn * v, k
        # inside the envelope once cracked: towards the origin on the secant
        # through the largest excursion
        if gmax <= gcr:
            v = GA0 * gmax
        elif gmax <= gy:
            v = Vc + GAcr * (gmax - gcr)
        else:
            v = Vy + (h * GAcr) * (gmax - gy)
        sec = v / gmax
        return sec * g, sec

#: a later station must beat the reported one by this factor to replace it:
#: the signed station value of a tie is the first one's on
#: every backend. The NumPy reference path uses the same number.
TIE = 1.0 + 1e-9


def make_kernel(concrete_point, steel_point, n_gauss, n_conc, n_steel, parallel=False):
    """Build the fused kernel for one pair of laws and one section size.

    `n_gauss`, `n_conc` and `n_steel` are baked in as compile-time constants so
    the fibre loops carry no bounds bookkeeping and the section resultants
    accumulate in registers.

    Compiled kernels are kept, keyed by exactly the things baked into them. The
    material point functions are module-level singletons, so two solvers built
    from the same laws and the same section size share one compilation instead
    of paying ~10 s again. `cache=False` on the kernel means numba's on-disk
    cache never helps here -- it closes over freshly built device functions --
    so without this every solver construction recompiled. It showed up as six
    regression tests costing 90 s of which about 60 was the compiler.
    """
    if njit is None:                            # pragma: no cover
        return None
    key = (concrete_point, steel_point, int(n_gauss), int(n_conc), int(n_steel), bool(parallel))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    NGP = int(n_gauss)
    NC = int(n_conc)
    NS = int(n_steel)

    # ONE SOURCE, TWO COMPILATIONS. `prange` is `range`
    # under a serial njit, so the serial kernel and the threaded one are the
    # same text. Each element writes its global-frame forces into its own
    # row of `ge` and the rows are added into `out` afterwards, in element
    # order: no two threads touch one node, and the serial sum is the one it
    # always was, to the bit.
    @njit(cache=False, fastmath=False, nogil=True, parallel=bool(parallel))
    def kernel(edofs, x0, q, orient, R0, length, gauss_w, Bky, Bkz,
               cy, cz, ca_, cpre, sy, sz, sa_, spre, state_c, state_s,
               cpar, spar, G0J0, GA, Bgy, Bgz, out, actions, end_actions, active, fstress, birth,
               sl_on, sl_ri, sl_rj, sl_GA0, sl_GAcr, sl_Vy, sl_h, sl_kb, sl_flex, sl_epscr,
               sl_gamma, sl_gcr_c, sl_gmax_c, sl_gcr_t, sl_gmax_t, sl_V, sl_update, offs, has_off):
        n_elem = edofs.shape[0]
        ge = np.zeros((n_elem, 12))
        collapsed = np.zeros(n_elem, dtype=np.uint8)

        for e in prange(n_elem):
            R = np.empty((3, 3))
            loc = np.empty((2, 3))
            x1 = np.empty(3); x2 = np.empty(3)
            local = np.empty(12)
            fl = np.empty(12)
            # A fractured member (core/fracture.py) contributes
            # nothing and its fibres are not driven any further, which is the
            # same rule the NumPy reference applies -- and one branch on a
            # uint8, so a run with no fracture pays for the array and nothing
            # more. The mass is not here to be removed: it lives on the nodes.
            if active[e] == 0:
                for i in range(actions.shape[1]):
                    actions[e, i] = 0.0
                for i in range(12):
                    end_actions[e, i] = 0.0
                continue
            # --- current configuration: the co-rotational frame and the
            # local nodal rotations from the points' orientations
            # (kernels/numba_rotations.member_frame) ---------
            for i in range(3):
                x1[i] = x0[e, 0, i] + q[edofs[e, i]]
                x2[i] = x0[e, 1, i] + q[edofs[e, 6 + i]]
            n1 = edofs[e, 0] // 6; n2 = edofs[e, 6] // 6
            # end offsets (1.21, core/member_geometry): the elastic ends x + Q e
            r1 = np.zeros(3); r2 = np.zeros(3)
            if has_off[e]:
                for i in range(3):
                    r1[i] = (orient[n1, i, 0] * offs[e, 0, 0] + orient[n1, i, 1] * offs[e, 0, 1]
                             + orient[n1, i, 2] * offs[e, 0, 2])
                    r2[i] = (orient[n2, i, 0] * offs[e, 1, 0] + orient[n2, i, 1] * offs[e, 1, 1]
                             + orient[n2, i, 2] * offs[e, 1, 2])
                    x1[i] += r1[i]
                    x2[i] += r2[i]
            L = member_frame(x1, x2, orient[n1], orient[n2], R0[e], R, loc)
            if L < 1e-14:
                collapsed[e] = 1              # raised below: not from a thread
                continue
            L0 = length[e]
            for i in range(12):
                local[i] = 0.0
            local[6] = L - L0
            for i in range(3):
                local[3 + i] = loc[0, i]; local[9 + i] = loc[1, i]
            # a member born late is stress-free where it was born (1.2): its
            # local deformation then is subtracted; zeros otherwise, and
            # x - 0.0 is x to the bit
            for i in range(12):
                local[i] -= birth[e, i]

            eps0 = local[6] / L0
            # a cracking shear law (in the kernel since 1.7):
            # the bending sees the end rotations less the shear rotation
            shear_law = sl_on[e] != 0
            g_e = 0.0
            ri = 0; rj = 0
            cracked_here = False
            if shear_law:
                ri = sl_ri[e]; rj = sl_rj[e]
                g_e = sl_gamma[e]
                local[ri] -= g_e; local[rj] -= g_e
            twist = (local[9] - local[3]) / L0
            Mt = G0J0[e] * twist

            for i in range(12):
                fl[i] = 0.0
            peak_N = 0.0
            peak_Mt = 0.0
            peak_My = 0.0
            peak_Mz = 0.0
            # the plain magnitudes, by the old strictly-larger rule, so that
            # `*_max_abs` is the pre-v1.0 field bit for bit (1.0 re-review)
            abs_N = 0.0
            abs_Mt = 0.0
            abs_My = 0.0
            abs_Mz = 0.0

            for gi in range(NGP):
                kz = (Bkz[e, gi, 0] * local[1] + Bkz[e, gi, 1] * local[5]
                      + Bkz[e, gi, 2] * local[7] + Bkz[e, gi, 3] * local[11])
                ky = (Bky[e, gi, 0] * local[2] + Bky[e, gi, 1] * local[4]
                      + Bky[e, gi, 2] * local[8] + Bky[e, gi, 3] * local[10])

                N = 0.0
                My = 0.0
                Mz = 0.0
                for f in range(NC):
                    # cpre: an imposed strain on a CONCRETE fibre -- a
                    # temperature field. Zero on every model
                    # that has none, so those are unchanged to the bit.
                    eps = eps0 - cy[e, f] * kz + cz[e, f] * ky + cpre[e, f]
                    sig = concrete_point(eps, state_c[e, gi, f], cpar[e, f])
                    fstress[e, gi, f] = sig
                    a = ca_[e, f]
                    # flexural cracking: a concrete fibre of THIS section (a
                    # padded one has no area) past the cracking strain
                    if shear_law and not cracked_here and sl_flex[e] and a > 0.0 and eps > sl_epscr[e]:
                        cracked_here = True
                    N += sig * a
                    My += sig * cz[e, f] * a
                    Mz -= sig * cy[e, f] * a
                for f in range(NS):
                    # spre: a tendon's initial strain, zero on a rebar
                    eps = eps0 - sy[e, f] * kz + sz[e, f] * ky + spre[e, f]
                    sig = steel_point(eps, state_s[e, gi, f], spar[e, f])
                    fstress[e, gi, NC + f] = sig
                    a = sa_[e, f]
                    N += sig * a
                    My += sig * sz[e, f] * a
                    Mz -= sig * sy[e, f] * a

                # the station with the largest magnitude, WITH its sign.
                # A tie (a moment through zero mid-member
                # is +M and -M at two gauss stations) goes to the first
                # station unless the later one is larger beyond round-off, so
                # the sign does not depend on the backend's last bit
                if abs(N) > abs(peak_N) * TIE:
                    peak_N = N
                if abs(Mt) > abs(peak_Mt) * TIE:
                    peak_Mt = Mt
                if abs(My) > abs(peak_My) * TIE:
                    peak_My = My
                if abs(Mz) > abs(peak_Mz) * TIE:
                    peak_Mz = Mz
                if abs(N) > abs_N:
                    abs_N = abs(N)
                if abs(Mt) > abs_Mt:
                    abs_Mt = abs(Mt)
                if abs(My) > abs_My:
                    abs_My = abs(My)
                if abs(Mz) > abs_Mz:
                    abs_Mz = abs(Mz)

                # fl += w*L0*(N*Ba + Mt*Bt + My*By + Mz*Bz), written out because
                # Ba and Bt have two non-zero entries and By/Bz have four.
                c = gauss_w[gi] * L0
                cn = c * N / L0
                fl[0] -= cn
                fl[6] += cn
                ct = c * Mt / L0
                fl[3] -= ct
                fl[9] += ct
                cy_ = c * My
                fl[2] += cy_ * Bky[e, gi, 0]
                fl[4] += cy_ * Bky[e, gi, 1]
                fl[8] += cy_ * Bky[e, gi, 2]
                fl[10] += cy_ * Bky[e, gi, 3]
                cz_ = c * Mz
                fl[1] += cz_ * Bkz[e, gi, 0]
                fl[5] += cz_ * Bkz[e, gi, 1]
                fl[7] += cz_ * Bkz[e, gi, 2]
                fl[11] += cz_ * Bkz[e, gi, 3]

            # elastic member shear: a constant shear strain
            # per plane from the interdependent interpolation, V = GA gamma,
            # entering the local force through the same rows; nothing here
            # when the section states no shear stiffness (GA = 0, rows 0)
            Vy = 0.0; Vz = 0.0
            if shear_law:
                # the member's shear from its law, and one Newton step on
                # M_i + M_j = L0 V(gamma) for the next evaluation
                gcr = sl_gcr_c[e]
                if not np.isfinite(gcr) and cracked_here:
                    gcr = max(abs(g_e), 1e-30)      # cracks at the shear it carries now
                gmax = max(sl_gmax_c[e], abs(g_e))
                V, kv = shear_law_point(g_e, gcr, gmax, sl_GA0[e], sl_GAcr[e], sl_Vy[e], sl_h[e])
                sl_gcr_t[e] = gcr; sl_gmax_t[e] = gmax
                sl_V[e] = V
                if sl_update:
                    r = fl[ri] + fl[rj] - L0 * V
                    sl_gamma[e] = g_e + r / (sl_kb[e] + L0 * kv)
                # the law's V is (M_i + M_j)/L0: +S_z about y, -S_y about z
                if ri == 4:
                    Vz = -V
                else:
                    Vy = -V
            if GA[e] > 0.0:
                gz = (Bgz[e, 0] * local[1] + Bgz[e, 1] * local[5]
                      + Bgz[e, 2] * local[7] + Bgz[e, 3] * local[11])
                gy = (Bgy[e, 0] * local[2] + Bgy[e, 1] * local[4]
                      + Bgy[e, 2] * local[8] + Bgy[e, 3] * local[10])
                Vy = GA[e] * gz          # shear along local y (the z-bending plane)
                Vz = GA[e] * gy          # shear along local z
                cv = L0 * Vy
                fl[1] += cv * Bgz[e, 0]; fl[5] += cv * Bgz[e, 1]
                fl[7] += cv * Bgz[e, 2]; fl[11] += cv * Bgz[e, 3]
                cw = L0 * Vz
                fl[2] += cw * Bgy[e, 0]; fl[4] += cw * Bgy[e, 1]
                fl[8] += cw * Bgy[e, 2]; fl[10] += cw * Bgy[e, 3]

            actions[e, 0] = peak_N
            # signed in the member's internal convention (core/member_actions):
            # GA gy is minus the z shear on the +x face, GA gz is the y shear
            actions[e, 1] = Vy
            actions[e, 2] = -Vz
            actions[e, 3] = peak_Mt
            actions[e, 4] = peak_My
            actions[e, 5] = peak_Mz
            if actions.shape[1] >= 10:
                # columns 6-9: |N|, |Mt|, |My|, |Mz| at their largest station
                actions[e, 6] = abs_N
                actions[e, 7] = abs_Mt
                actions[e, 8] = abs_My
                actions[e, 9] = abs_Mz
            for i in range(12):
                end_actions[e, i] = fl[i]

            # ge[e] = blockdiag(R.T,...) @ fl
            for blk in range(4):
                for i in range(3):
                    acc = 0.0
                    for k in range(3):
                        acc += R[k, i] * fl[3 * blk + k]
                    ge[e, 3 * blk + i] = acc
            if has_off[e]:
                # the nodes get m + r x f
                ge[e, 3] += r1[1] * ge[e, 2] - r1[2] * ge[e, 1]
                ge[e, 4] += r1[2] * ge[e, 0] - r1[0] * ge[e, 2]
                ge[e, 5] += r1[0] * ge[e, 1] - r1[1] * ge[e, 0]
                ge[e, 9] += r2[1] * ge[e, 8] - r2[2] * ge[e, 7]
                ge[e, 10] += r2[2] * ge[e, 6] - r2[0] * ge[e, 8]
                ge[e, 11] += r2[0] * ge[e, 7] - r2[1] * ge[e, 6]

        for e in range(n_elem):
            if collapsed[e]:
                raise ValueError("beam2_fiber member collapsed to zero length")
        for e in range(n_elem):
            if active[e] == 0:
                continue
            for i in range(12):
                out[edofs[e, i]] += ge[e, i]

        return out, actions

    _KERNEL_CACHE[key] = kernel
    return kernel
