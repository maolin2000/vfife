"""Concrete Damaged Plasticity (Lubliner 1989 / Lee & Fenves 1998).

The formulation Abaqus calls CDP, implemented for an explicit solver: batched
over integration points, no eigenvectors, no matrix inverses, no LAPACK calls,
so the same algebra ports directly into a GPU kernel.

MODEL
-----
Effective stress          sig_bar = D0 : (eps - eps_p),  sigma = (1-d) sig_bar

Yield (Lee-Fenves form of the Lubliner surface), all stresses positive:

    F = 1/(1-a) [ q_bar - 3 a p_bar + b(ep) <sig_max> - g <-sig_max> ] - sc(epc)

    p_bar = -tr(sig_bar)/3          q_bar = sqrt(3/2 s:s)
    a = (fb0/fc0 - 1) / (2 fb0/fc0 - 1)          default fb0/fc0 = 1.16
    b = sc(epc)/st(ept) (1-a) - (1+a)
    g = 3(1-Kc) / (2Kc - 1)                      default Kc = 2/3

Flow (non-associated, Drucker-Prager hyperbolic potential):

    G = sqrt((e_ecc st0 tan psi)^2 + q_bar^2) - p_bar tan psi
    m = dG/dsig_bar = (dG/dq) 3s/(2q) + (tan psi / 3) I

Hardening: two variables, driven by the extreme principal plastic strain rates
weighted by the stress-state factor r = sum<sig_i> / sum|sig_i|:

    d(ept) = r m_max dlam            d(epc) = -(1-r) m_min dlam

Damage with stiffness recovery:

    d = 1 - (1 - st_f dc)(1 - sc_f dt)
    st_f = 1 - wt r          sc_f = 1 - wc (1-r)      Abaqus default wt=0, wc=1

Viscoplastic (Duvaut-Lions) regularisation, exactly as Abaqus defines it: the
INVISCID backbone (eps_p, ept, epc) evolves by the return map, and separate
viscous copies lag it,

    eps_p_v <- eps_p_v + dt/(mu+dt) (eps_p - eps_p_v)
    d_v     <- d_v     + dt/(mu+dt) (d     - d_v)
    sigma = (1 - d_v) D0 : (eps - eps_p_v)

mu = 0 collapses this to the rate-independent model exactly.

WHY THERE ARE NO EIGENVECTORS
-----------------------------
dF/dsig_bar contains d<sig_max>/dsig_bar = n1 (x) n1, which normally forces a
closed-form eigenvector.  But that derivative only ever appears inside the
scalar denominator of the cutting-plane step

    A = n : D0 : m + H

and D0 is isotropic, so with C:m = 2G m_dev + 3K m_vol,

    (n1 (x) n1) : C : m = 2G (m_max - tan psi / 3) + K tan psi

because n1 (x) n1 : m_dev is just the principal VALUE of m_dev along the same
direction, and n1 (x) n1 : I = 1.  The same argument collapses the deviatoric
and volumetric parts to 3G dG/dq and 3K a tan psi.  So the whole denominator is
built from principal VALUES, which Cardano gives in closed form -- the flow
direction m itself never needed eigenvectors to begin with.  This is exact, not
an approximation.

The return map is a cutting-plane algorithm (`n_return` iterations, default 5).
It converges to F = 0 along the flow direction; the denominator only sets how
fast, which is why an explicit solver with small strain increments does not
need the consistent tangent an implicit code would.

State layout (n_state = 15), one row per integration point:

    [0:6]   eps_p     inviscid plastic strain, ENGINEERING Voigt
    [6]     ept       inviscid equivalent plastic strain, tension
    [7]     epc       inviscid equivalent plastic strain, compression
    [8:14]  eps_p_v   viscoplastic strain, ENGINEERING Voigt
    [14]    d_v       viscous damage
"""

from __future__ import annotations

import numpy as np

from ..base.tensor import ddot, deviator, principal_values
from ..base.voigt import (ENG_TO_TENSOR, TENSOR_TO_ENG,
                     dilatational_wave_speed, elastic_matrix)
from . import curves as cv
from .base import Material, register


def _build_curve(spec: dict, E0: float, l_ref: float, kind: str) -> cv.Curve:
    spec = dict(spec)
    source = spec.pop("source", "table")
    name = f"CDP {kind} ({source})"
    if source == "gb50010":
        if kind == "compression":
            if "Gc" in spec:
                spec = dict(spec, l_ref=l_ref)
            return cv.gb50010_compression(E0=E0, name=name, **spec)
        return cv.gb50010_tension(E0=E0, name=name, **spec)
    if source == "fracture_energy":
        if kind != "tension":
            raise ValueError("fracture_energy input is only for the tension curve")
        return cv.from_fracture_energy(E0=E0, l_ref=l_ref, name=name, **spec)
    if source == "stress_displacement":
        return cv.from_stress_displacement(E0=E0, l_ref=l_ref, name=name, **spec)
    if source == "table":
        return cv.from_abaqus_tables(E0=E0, name=name, **spec)
    raise ValueError(
        f"unknown {kind} curve source {source!r}; expected one of "
        "gb50010, fracture_energy, stress_displacement, table"
    )


#: Under `crack_band: both`, l_e/l_ref on the compression side must lie here.
COMPRESSION_BAND_MIN = 0.5
COMPRESSION_BAND_MAX = 2.0


@register("cdp", regularised=True, verification="constitutive_path",
          output_fields=("stress", "strain", "von_mises", "eps_p", "ept", "epc", "d"))
class ConcreteDamagedPlasticity(Material):
    n_state = 15
    state_names = {
        "eps_p": slice(0, 6),
        "ept": slice(6, 7),
        "epc": slice(7, 8),
        "eps_p_visc": slice(8, 14),
        "d": slice(14, 15),
    }

    def __init__(
        self,
        E: float,
        nu: float,
        rho: float,
        compression: dict,
        tension: dict,
        dilation_angle: float = 36.0,
        eccentricity: float = 0.1,
        fb0_fc0: float = 1.16,
        Kc: float = 2.0 / 3.0,
        wt: float = 0.0,
        wc: float = 1.0,
        viscosity: float = 0.0,
        damage_max: float = 0.99,
        n_return: int = 5,
        crack_band: str = "tension",
        l_ref: float = 1.0,
        **_ignored,
    ):
        self.E, self.nu, self.rho = float(E), float(nu), float(rho)
        self.D0 = elastic_matrix(self.E, self.nu)
        self.wave_speed = dilatational_wave_speed(self.E, self.nu, self.rho)
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))

        if not 1.0 < fb0_fc0 < 2.0:
            raise ValueError("fb0/fc0 must lie strictly between 1 and 2")
        if not 0.5 < Kc <= 1.0:
            raise ValueError("Kc must lie in (0.5, 1]")
        self.alpha = (fb0_fc0 - 1.0) / (2.0 * fb0_fc0 - 1.0)
        self.gamma = 3.0 * (1.0 - Kc) / (2.0 * Kc - 1.0)
        self.fb0_fc0, self.Kc = float(fb0_fc0), float(Kc)

        self.tanpsi = float(np.tan(np.deg2rad(dilation_angle)))
        self.ecc = float(eccentricity)
        self.wt, self.wc = float(wt), float(wc)
        for w, n in ((self.wt, "wt"), (self.wc, "wc")):
            if not 0.0 <= w <= 1.0:
                raise ValueError(f"{n} must lie in [0,1]")
        self.mu = float(viscosity)
        self.d_max = float(damage_max)
        self.n_return = int(n_return)

        if crack_band not in ("tension", "both", "none"):
            raise ValueError("crack_band must be tension, both or none")
        self.crack_band = crack_band
        self.l_ref = float(l_ref)
        # the compression band factor an element may carry under `both`;
        # see the guard in bind_points for what the numbers rest on
        self.compression_band_limits = (COMPRESSION_BAND_MIN, COMPRESSION_BAND_MAX)

        # lower bound on the cutting-plane denominator: a tenth of the elastic
        # deviatoric term, so one iteration can never take more than ~10x the
        # elastic-predictor step whatever the softening slope
        self.A_floor = 0.1 * 3.0 * self.G / (1.0 - self.alpha)

        if isinstance(compression, dict) and compression.get("Gc") is not None \
                and crack_band != "both":
            raise ValueError(
                "the compression curve has a crushing energy Gc, which puts it on "
                f"the reference length l_ref, but crack_band is {crack_band!r}: "
                "nothing would rescale it per element and the regularisation would "
                "hold only where the element size happens to equal l_ref. Use "
                "crack_band: both, or drop Gc.")
        self.curve_c = _build_curve(compression, self.E, self.l_ref, "compression")
        self.curve_t = _build_curve(tension, self.E, self.l_ref, "tension")
        self.sigma_t0 = self.curve_t.sigma0
        self.a_ecc = self.ecc * self.sigma_t0 * self.tanpsi

        # crack-band factors, replaced by bind_points() once the mesh is known
        self.k_band = 1.0
        self.k_t = 1.0
        self.k_c = 1.0

    # ------------------------------------------------------------------ #

    def bind_points(self, char_length: np.ndarray) -> None:
        """Receive the per-integration-point element characteristic length.

        The tension curve is stored against `l_ref`; querying it at ep * l_e/l_ref
        is the crack-band scaling, so the dissipated energy per unit crack AREA
        is independent of the element size.  One shared table, one scalar per
        point -- nothing that has to be duplicated per element.
        """
        k = np.ascontiguousarray(char_length, dtype=np.float64) / self.l_ref
        self.k_band = k                       # raw l_e/l_ref, for the Numba path
        self.k_t = k if self.crack_band in ("tension", "both") else 1.0
        self.k_c = k if self.crack_band == "both" else 1.0

        # Compression-side guard.  The tension curve is built FROM l_ref
        # (fracture energy -> crack opening -> strain over l_ref), so
        # querying it at ep * l_e/l_ref is a change of variable and l_ref
        # cancels; any value of it gives the same sigma-w curve.  The
        # compression curve is not: a code curve (GB 50010) is a fixed
        # strain-axis curve with no reference length the solver knows, and
        # under `crack_band: both` the same k = l_e/l_ref is multiplied onto
        # its plastic strain with no calibration behind it.  On the Alberta
        # beam `l_ref: 0.15` on a 15 mm mesh (k = 0.10) stretched the crushing
        # branch tenfold, the compression zone never softened, and the
        # curve went 3% OVER the measured peak with no descent -- a wrong
        # answer that looked better than the right one.
        # `l_ref: auto` (k = 1 on a uniform mesh) is what the case files use
        # and what the verification contracts were written against.
        #
        # So k_c is held near one.  Outside [1/2, 2] the run is refused with
        # the ratio, because either the case wrote a number for l_ref that
        # is not the element size, or the mesh is graded more than 2:1 and
        # the compression band would silently steepen with element size --
        # in which case `crack_band: tension` (compression unscaled) is the
        # supported choice.  A compressive fracture energy with its own
        # reference length is the regularisation that would lift this, and
        # it needs its own evidence before it is written.
        if self.crack_band == "both":
            kc = np.atleast_1d(np.asarray(self.k_c, dtype=np.float64))
            lo, hi = self.compression_band_limits
            k_min, k_max_c = float(kc.min()), float(kc.max())
            if k_min < lo or k_max_c > hi:
                from ..base.errors import UserError
                l_min, l_max = k_min * self.l_ref, k_max_c * self.l_ref
                # `auto` puts l_ref at the smallest element, so a mesh whose
                # sizes span more than `hi` cannot be brought inside by it
                graded = l_max / max(l_min, 1e-300) > hi
                raise UserError(
                    "COMPRESSION_BAND_RATIO",
                    f"crack_band: both scales the compression curve by l_e/l_ref = "
                    f"{k_min:.3g}..{k_max_c:.3g} (elements {l_min:.4g}..{l_max:.4g} m "
                    f"against l_ref = {self.l_ref:.4g} m), outside the supported "
                    f"[{lo:g}, {hi:g}]",
                    ("the mesh is graded more than "
                     f"{hi:g}:1, so even `l_ref: auto` (the smallest element) leaves "
                     "the largest outside; the compression softening would steepen with "
                     "element size with no calibration behind it; use `crack_band: "
                     "tension` (the crack band on the tension curve only, compression "
                     "as the code curve gives it)"
                     if graded else
                     "the compression curve is a strain curve with no reference length "
                     "the solver knows, and a k far from one stretches or shortens its "
                     "crushing branch by that factor (on the Alberta beam k = 0.1 removed "
                     "the descent and overshot the measured peak). Write `l_ref: auto`, "
                     "which is the element size, or `crack_band: tension`."),
                )

        # Snap-back check.  Along the softening branch the total uniaxial strain
        # is eps = sig_bar/E + ep/k, so it stops increasing once
        #     -d(sig_bar)/d(ep) * k > E
        # and the sigma-eps response turns back on itself.  No strain-driven
        # solver can follow that; the element unloads elastically instead and
        # the fracture energy is never dissipated.  It is a limit on how COARSE
        # the mesh may be for a given Gf, and it is much better to say so here
        # than to let the run produce a plausible-looking wrong answer.
        steepest = -float(np.min(self.curve_t.slope))
        if steepest > 0 and self.crack_band in ("tension", "both"):
            k_max = self.E / steepest
            worst = float(np.max(k)) if np.ndim(k) else float(k)
            if worst > k_max:
                raise ValueError(
                    f"element characteristic length {worst * self.l_ref:.4g} m "
                    f"exceeds the snap-back limit {k_max * self.l_ref:.4g} m for "
                    "this tension softening curve. Refine the mesh, raise the "
                    "fracture energy, or soften the curve."
                )

    def describe(self) -> str:
        return (
            f"    alpha = {self.alpha:.4f} (fb0/fc0 = {self.fb0_fc0}), "
            f"gamma = {self.gamma:.3f} (Kc = {self.Kc:.4f})\n"
            f"    psi = {np.rad2deg(np.arctan(self.tanpsi)):.1f} deg, "
            f"ecc = {self.ecc}, wt = {self.wt}, wc = {self.wc}, mu = {self.mu:g}\n"
            f"    sigma_c0 = {self.curve_c.sigma0 / 1e6:.2f} MPa, "
            f"sigma_t0 = {self.curve_t.sigma0 / 1e6:.2f} MPa, "
            f"crack band = {self.crack_band} (l_ref = {self.l_ref:.4g} m"
            + (f", k_c = {float(np.min(self.k_c)):.3g}..{float(np.max(self.k_c)):.3g}"
               if self.crack_band == "both" and np.ndim(self.k_c) else "") + ")\n"
            f"    plastic fraction beta: compression "
            f"{self.curve_c.beta_requested:.2f} -> {self.curve_c.beta_effective:.2f}, "
            f"tension {self.curve_t.beta_requested:.2f} -> "
            f"{self.curve_t.beta_effective:.2f} "
            f"(effective values differ where the no-expansion cap binds)"
        )

    # ------------------------------------------------------------------ #

    def yield_function(self, sig: np.ndarray, ept=0.0, epc=0.0) -> np.ndarray:
        """F(sig_bar, ept, epc) for STRESS-like Voigt arrays. Exposed for testing.

        F < 0 inside the surface, 0 on it.  Used by the tests that check the
        biaxial strength ratio (fb0/fc0) and the meridian ratio (Kc), which are
        statements about the SHAPE of the surface and are therefore best checked
        on the surface itself rather than through a loading path.
        """
        sig = np.atleast_2d(sig)
        ept = np.broadcast_to(np.asarray(ept, float), sig.shape[:1])
        epc = np.broadcast_to(np.asarray(epc, float), sig.shape[:1])
        pv = principal_values(sig)
        smax = pv[:, 2]
        p_bar = -(sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
        q = np.sqrt(np.maximum(1.5 * ddot(deviator(sig), deviator(sig)), 0.0))
        sc, _, _ = self.curve_c.evaluate(epc, 1.0)
        st, _, _ = self.curve_t.evaluate(ept, 1.0)
        beta = (sc / st) * (1.0 - self.alpha) - (1.0 + self.alpha)
        return (q - 3.0 * self.alpha * p_bar
                + beta * np.maximum(smax, 0.0)
                - self.gamma * np.maximum(-smax, 0.0)) / (1.0 - self.alpha) - sc

    def update(self, strain: np.ndarray, state: np.ndarray, dt: float) -> np.ndarray:
        D0T = self.D0.T
        eps_p = state[:, 0:6]
        ept = state[:, 6]
        epc = state[:, 7]

        sig = (strain - eps_p) @ D0T          # effective stress, inviscid backbone
        one_m_a = 1.0 - self.alpha
        tanpsi = self.tanpsi
        ept0, epc0 = ept.copy(), epc.copy()
        dep_total = np.zeros_like(eps_p)
        yielded = False

        for _ in range(self.n_return):
            pv = principal_values(sig)
            smin, smax = pv[:, 0], pv[:, 2]
            tr3 = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
            p_bar = -tr3
            s = deviator(sig)
            q2 = 1.5 * ddot(s, s)
            q = np.sqrt(np.maximum(q2, 0.0))
            q_safe = np.maximum(q, 1e-12 * self.E)

            sc, dsc, _ = self.curve_c.evaluate(epc, self.k_c)
            st, dst, _ = self.curve_t.evaluate(ept, self.k_t)
            beta = (sc / st) * one_m_a - (1.0 + self.alpha)

            pos_max = np.maximum(smax, 0.0)
            neg_max = np.maximum(-smax, 0.0)
            F = (q - 3.0 * self.alpha * p_bar
                 + beta * pos_max - self.gamma * neg_max) / one_m_a - sc

            active = F > 1e-9 * sc
            if not active.any():
                break

            # --- flow direction -------------------------------------------
            dGdq = q / np.sqrt(self.a_ecc ** 2 + q2)
            coef = dGdq * 1.5 / q_safe
            m = s * coef[:, None]
            m[:, 0] += tanpsi / 3.0
            m[:, 1] += tanpsi / 3.0
            m[:, 2] += tanpsi / 3.0
            # principal values of m come free: m is a linear combination of s
            # and I, so it is coaxial with sig
            m_max = coef * (smax + p_bar) + tanpsi / 3.0
            m_min = coef * (smin + p_bar) + tanpsi / 3.0

            # --- cutting-plane denominator, from principal VALUES only -----
            c_max = beta * (smax > 0.0) + self.gamma * (smax < 0.0)
            A_stress = (
                3.0 * self.G * dGdq
                + 3.0 * self.K * self.alpha * tanpsi
                + c_max * (2.0 * self.G * (m_max - tanpsi / 3.0)
                           + self.K * tanpsi)
            ) / one_m_a

            apv = np.abs(pv).sum(axis=1)
            r = np.where(apv > 1e-30, np.maximum(pv, 0.0).sum(axis=1) / np.maximum(apv, 1e-30), 0.0)
            h_t = np.maximum(r * m_max, 0.0)
            h_c = np.maximum(-(1.0 - r) * m_min, 0.0)

            dF_dept = -sc * dst * pos_max / (st * st)
            dF_depc = dsc * (pos_max / st - 1.0)
            A = A_stress - (dF_dept * h_t + dF_depc * h_c)
            # the floor matters more than it looks: at 1e-6*E it is small enough
        # that a strongly softening point can drive the plastic multiplier to
        # F/3e4, which produced equivalent plastic strains of 1e10 and stresses
        # of 1e20 when an element was pushed far outside the curve range.  A
        # floor set to a fraction of the ELASTIC denominator bounds one
        # iteration to a physically scaled step instead, which matters at 10^6
        # elements where some element is always in a bad state.
            A = np.maximum(A, self.A_floor)

            dlam = np.where(active, F / A, 0.0)
            np.maximum(dlam, 0.0, out=dlam)

            deps_p = (dlam[:, None] * m) * TENSOR_TO_ENG
            sig -= deps_p @ D0T
            eps_p += deps_p
            dep_total += deps_p
            yielded = True
            # provisional, so the surface sees the right hardening on the next
            # iteration; corrected below from the converged stress
            ept += dlam * h_t
            epc += dlam * h_c

        # --- hardening split, from the CONVERGED stress --------------------
        # Lee-Fenves define the split by r evaluated at the updated stress.  A
        # cutting-plane algorithm instead accumulates it over iterations, and
        # the early iterations see intermediate stress states that a pure
        # uniaxial compression path should never visit: the dilatant flow
        # direction plus the lambda coupling in D0 briefly pushes the lateral
        # effective stress positive, giving r > 0 and leaking tensile hardening
        # into a compression path (measured: ept reaching 1.5% of epc).  That is
        # invisible in the stress -- in compression r = 0 kills the dt term
        # anyway -- but it pre-damages a subsequent tension leg, which is
        # exactly what a cyclic model must not do.
        pv = principal_values(sig)
        apv = np.abs(pv).sum(axis=1)
        r = np.where(apv > 1e-30,
                     np.maximum(pv, 0.0).sum(axis=1) / np.maximum(apv, 1e-30), 0.0)
        if yielded:
            dpv = principal_values(dep_total * ENG_TO_TENSOR)
            ept = ept0 + r * np.maximum(dpv[:, 2], 0.0)
            epc = epc0 - (1.0 - r) * np.minimum(dpv[:, 0], 0.0)

        # --- damage with stiffness recovery -------------------------------
        _, _, dc = self.curve_c.evaluate(epc, self.k_c)
        _, _, dtn = self.curve_t.evaluate(ept, self.k_t)
        s_t = 1.0 - self.wt * r
        s_c = 1.0 - self.wc * (1.0 - r)
        d = 1.0 - (1.0 - s_t * dc) * (1.0 - s_c * dtn)
        np.clip(d, 0.0, self.d_max, out=d)

        # --- Duvaut-Lions viscous regularisation --------------------------
        if self.mu > 0.0:
            f = dt / (self.mu + dt)
            state[:, 8:14] += f * (eps_p - state[:, 8:14])
            state[:, 14] += f * (d - state[:, 14])
        else:
            state[:, 8:14] = eps_p
            state[:, 14] = d

        state[:, 6] = ept
        state[:, 7] = epc
        eps_p_v = state[:, 8:14]
        d_v = state[:, 14]
        return ((strain - eps_p_v) @ D0T) * (1.0 - d_v)[:, None]

    def damage_measure(self, state: np.ndarray):
        """The scalar stiffness degradation d, capped at `damage_max`."""
        return state[:, 14]

    def failure(self, state: np.ndarray):
        """max(dt, dc): the monotone halves, NOT `d`.

        `damage_measure` returns `d`, which carries the stiffness recovery and
        therefore FALLS when a crack closes under compression -- the erosion
        criterion in `core/solver.py` has to pair it with a strain that only
        grows for exactly that reason. A failure fraction may not fall, so it
        is built from `dt` and `dc`, which are read off the curves at the
        equivalent plastic strains and are monotone because those are.
        """
        f = self.output_fields(state)
        return np.maximum(f["dt"], f["dc"])

    def output_fields(self, state: np.ndarray) -> dict:
        """`d` is the stiffness degradation the stress actually carries --
        with recovery, so it FALLS where a crack closes under compression.
        `dt` and `dc` are the tensile and compressive damage variables on
        their own, read off the curves at the equivalent plastic strains:
        Abaqus's DAMAGET and DAMAGEC, monotone, and what a crack map wants.
        The band factor is per point; a block that does not match the bound
        points (a probe, a partial state) gets the reference curve."""
        ept, epc = state[:, 6], state[:, 7]
        k_t, k_c = self.k_t, self.k_c
        if np.ndim(k_t) and np.shape(k_t)[0] != ept.shape[0]:
            k_t = 1.0
        if np.ndim(k_c) and np.shape(k_c)[0] != epc.shape[0]:
            k_c = 1.0
        _, _, dt = self.curve_t.evaluate(ept, k_t)
        _, _, dc = self.curve_c.evaluate(epc, k_c)
        return {
            "d": state[:, 14].copy(),
            "dt": np.asarray(dt, dtype=float),
            "dc": np.asarray(dc, dtype=float),
            "ept": ept.copy(),
            "epc": epc.copy(),
        }

    # ------------------------------------------------------------------ #

    def numba_point_fn(self):
        """Scalar twin of update(), for the Numba/GPU element kernel.

        Line-for-line the same algebra as the batched version -- which is only
        possible because the denominator collapses to principal VALUES (see the
        module docstring).  There is no eigensolver, no matrix inverse and no
        allocation that escapes, so this is also the Warp kernel in all but
        syntax.  `test_backends.py` asserts it matches the NumPy path to 1e-12.
        """
        from numba import njit

        from ..kernels.scalar import curve_eval, principal

        E = self.E
        lam = E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        G, K = self.G, self.K
        a_p, g_p = self.alpha, self.gamma
        one_m_a = 1.0 - self.alpha
        tanpsi, a_ecc = self.tanpsi, self.a_ecc
        wt, wc, mu, d_max = self.wt, self.wc, self.mu, self.d_max
        n_return = self.n_return
        A_floor = self.A_floor
        use_kt = self.crack_band in ("tension", "both")
        use_kc = self.crack_band == "both"

        cep, csig, cdmg, cslope = (self.curve_c.ep, self.curve_c.sig,
                                   self.curve_c.dmg, self.curve_c.slope)
        tep, tsig, tdmg, tslope = (self.curve_t.ep, self.curve_t.sig,
                                   self.curve_t.dmg, self.curve_t.slope)

        @njit(cache=True, fastmath=True)
        def point(strain, state, dt, kband, stress):
            kt = kband if use_kt else 1.0
            kc = kband if use_kc else 1.0
            ept = state[6]
            epc = state[7]
            ept0 = ept
            epc0 = epc

            sg = np.empty(6)
            e0 = strain[0] - state[0]
            e1 = strain[1] - state[1]
            e2 = strain[2] - state[2]
            tr = e0 + e1 + e2
            sg[0] = lam * tr + 2.0 * G * e0
            sg[1] = lam * tr + 2.0 * G * e1
            sg[2] = lam * tr + 2.0 * G * e2
            sg[3] = G * (strain[3] - state[3])
            sg[4] = G * (strain[4] - state[4])
            sg[5] = G * (strain[5] - state[5])

            dep = np.zeros(6)
            pv = np.empty(3)
            resc = np.empty(3)
            rest = np.empty(3)
            yielded = False

            for _ in range(n_return):
                principal(sg[0], sg[1], sg[2], sg[3], sg[4], sg[5], pv)
                smin = pv[0]
                smax = pv[2]
                tr3 = (sg[0] + sg[1] + sg[2]) / 3.0
                p_bar = -tr3
                s0 = sg[0] - tr3
                s1 = sg[1] - tr3
                s2 = sg[2] - tr3
                s3 = sg[3]
                s4 = sg[4]
                s5 = sg[5]
                q2 = 1.5 * (s0 * s0 + s1 * s1 + s2 * s2
                            + 2.0 * (s3 * s3 + s4 * s4 + s5 * s5))
                if q2 < 0.0:
                    q2 = 0.0
                q = np.sqrt(q2)
                q_safe = q if q > 1e-12 * E else 1e-12 * E

                curve_eval(cep, csig, cdmg, cslope, epc * kc, resc)
                sc = resc[0]
                dsc = resc[1] * kc
                curve_eval(tep, tsig, tdmg, tslope, ept * kt, rest)
                st = rest[0]
                dst = rest[1] * kt

                beta = (sc / st) * one_m_a - (1.0 + a_p)
                pos_max = smax if smax > 0.0 else 0.0
                neg_max = -smax if smax < 0.0 else 0.0
                F = (q - 3.0 * a_p * p_bar
                     + beta * pos_max - g_p * neg_max) / one_m_a - sc
                if F <= 1e-9 * sc:
                    break
                yielded = True

                dGdq = q / np.sqrt(a_ecc * a_ecc + q2)
                coef = dGdq * 1.5 / q_safe
                t3 = tanpsi / 3.0
                m0 = s0 * coef + t3
                m1 = s1 * coef + t3
                m2 = s2 * coef + t3
                m3 = s3 * coef
                m4 = s4 * coef
                m5 = s5 * coef
                m_max = coef * (smax + p_bar) + t3
                m_min = coef * (smin + p_bar) + t3

                if smax > 0.0:
                    c_max = beta
                elif smax < 0.0:
                    c_max = g_p
                else:
                    c_max = 0.0
                A = (3.0 * G * dGdq + 3.0 * K * a_p * tanpsi
                     + c_max * (2.0 * G * (m_max - t3) + K * tanpsi)) / one_m_a

                apv = abs(pv[0]) + abs(pv[1]) + abs(pv[2])
                rp = 0.0
                for i in range(3):
                    if pv[i] > 0.0:
                        rp += pv[i]
                r = rp / apv if apv > 1e-30 else 0.0

                h_t = r * m_max
                if h_t < 0.0:
                    h_t = 0.0
                h_c = -(1.0 - r) * m_min
                if h_c < 0.0:
                    h_c = 0.0

                dFt = -sc * dst * pos_max / (st * st)
                dFc = dsc * (pos_max / st - 1.0)
                A -= dFt * h_t + dFc * h_c
                if A < A_floor:
                    A = A_floor
                dl = F / A
                if dl < 0.0:
                    dl = 0.0

                # tensor flow direction -> ENGINEERING plastic increment
                p0 = dl * m0
                p1 = dl * m1
                p2 = dl * m2
                p3 = 2.0 * dl * m3
                p4 = 2.0 * dl * m4
                p5 = 2.0 * dl * m5
                trp = p0 + p1 + p2
                sg[0] -= lam * trp + 2.0 * G * p0
                sg[1] -= lam * trp + 2.0 * G * p1
                sg[2] -= lam * trp + 2.0 * G * p2
                sg[3] -= G * p3
                sg[4] -= G * p4
                sg[5] -= G * p5

                state[0] += p0
                state[1] += p1
                state[2] += p2
                state[3] += p3
                state[4] += p4
                state[5] += p5
                dep[0] += p0
                dep[1] += p1
                dep[2] += p2
                dep[3] += p3
                dep[4] += p4
                dep[5] += p5
                ept += dl * h_t
                epc += dl * h_c

            # --- hardening split from the CONVERGED stress ------------------
            principal(sg[0], sg[1], sg[2], sg[3], sg[4], sg[5], pv)
            apv = abs(pv[0]) + abs(pv[1]) + abs(pv[2])
            rp = 0.0
            for i in range(3):
                if pv[i] > 0.0:
                    rp += pv[i]
            r = rp / apv if apv > 1e-30 else 0.0

            if yielded:
                principal(dep[0], dep[1], dep[2],
                          0.5 * dep[3], 0.5 * dep[4], 0.5 * dep[5], pv)
                dmax = pv[2] if pv[2] > 0.0 else 0.0
                dmin = pv[0] if pv[0] < 0.0 else 0.0
                ept = ept0 + r * dmax
                epc = epc0 - (1.0 - r) * dmin

            curve_eval(cep, csig, cdmg, cslope, epc * kc, resc)
            curve_eval(tep, tsig, tdmg, tslope, ept * kt, rest)
            s_t = 1.0 - wt * r
            s_c = 1.0 - wc * (1.0 - r)
            d = 1.0 - (1.0 - s_t * resc[2]) * (1.0 - s_c * rest[2])
            if d < 0.0:
                d = 0.0
            elif d > d_max:
                d = d_max

            if mu > 0.0:
                f = dt / (mu + dt)
                for i in range(6):
                    state[8 + i] += f * (state[i] - state[8 + i])
                state[14] += f * (d - state[14])
            else:
                for i in range(6):
                    state[8 + i] = state[i]
                state[14] = d
            state[6] = ept
            state[7] = epc

            e0 = strain[0] - state[8]
            e1 = strain[1] - state[9]
            e2 = strain[2] - state[10]
            tr = e0 + e1 + e2
            one_d = 1.0 - state[14]
            stress[0] = (lam * tr + 2.0 * G * e0) * one_d
            stress[1] = (lam * tr + 2.0 * G * e1) * one_d
            stress[2] = (lam * tr + 2.0 * G * e2) * one_d
            stress[3] = G * (strain[3] - state[11]) * one_d
            stress[4] = G * (strain[4] - state[12]) * one_d
            stress[5] = G * (strain[5] - state[13]) * one_d

        return point

    # ------------------------------------------------------------------ #

    def warp_params(self, dtype=np.float64):
        """(scalars, packed tables) for the GPU backend.

        Two flat buffers whatever the model: the layout is documented in
        kernels/warp_materials.py.  Both curves go in one `tab` array, so the
        element kernel signature is the same for CDP as for linear elasticity.
        """
        c = self.curve_c.pack(dtype)
        t = self.curve_t.pack(dtype)
        prm = np.array([
            self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu)),
            self.G, self.K, self.E,
            self.alpha, self.gamma, 1.0 - self.alpha,
            self.tanpsi, self.a_ecc,
            self.wt, self.wc, self.mu, self.d_max, float(self.n_return),
            1.0 if self.crack_band in ("tension", "both") else 0.0,
            1.0 if self.crack_band == "both" else 0.0,
            0.0, float(self.curve_c.ep.size),
            float(c.size), float(self.curve_t.ep.size),
            # tolerances travel with the parameters: written as literals in the
            # device code they would round to fp32 and move the convergence
            # decision (see kernels/warp_materials.py)
            1.0e-9, 1.0e-12 * self.E, self.A_floor, 1.0e-30,
        ], dtype=dtype)
        return prm, np.concatenate([c, t]).astype(dtype)

    @staticmethod
    def warp_point_fn():
        """Device twin, in kernels/warp_materials.py."""
        from ..kernels.warp_materials import cdp_point

        return cdp_point
