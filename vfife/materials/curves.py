"""Hardening / softening curves for the concrete damaged plasticity model.

A CDP curve is three columns:

    ep    equivalent plastic strain  (monotone increasing, starts at 0)
    sig   EFFECTIVE stress           sigma_bar = sigma / (1 - d)
    dmg   damage variable d

Two ways in, one way out.  Whatever the source, the runtime sees a table and
interpolates it, so the hot path has exactly one code path -- and on a GPU it
is one array to upload rather than a branch between analytic formulas.

  from_abaqus_tables()   the four tables Abaqus CDP takes:
                         (sigma, inelastic strain) and (d, inelastic strain).
                         Uses Abaqus's own conversion
                             ep = e_in - d/(1-d) * sigma/E0
                         and rejects data that makes ep decrease -- the same
                         input error Abaqus reports as "plastic strains are
                         negative", caught here at setup rather than mid-run.

  gb50010_*()            GB 50010-2010 appendix C uniaxial curves, converted
                         with the plastic-fraction split described below.

CRACK BAND
----------
Tension softening is stored against a REFERENCE element length `l_ref`.  At
runtime each integration point carries `k = l_e / l_ref` and the curve is
queried at `ep * k`, with the slope multiplied by `k`.  That is exactly the
crack-band scaling: the crack opening w = ep * l_e is what the curve really
depends on, so the dissipated energy per unit crack AREA stays Gf whatever the
element size.  One shared table plus one per-point scalar -- no per-element
curves, nothing that would have to be duplicated on a GPU.

WHY GB 50010 NEEDS A SPLIT PARAMETER
------------------------------------
GB 50010 appendix C is a pure DAMAGE model: sigma = (1-d) E ep, which implies
zero plastic strain.  Feeding it to CDP unchanged gives a model with no
plasticity at all.  The standard resolution is to declare what fraction of the
inelastic strain is plastic,

    ep = beta * e_in,        e_in = e - sigma/E0

and then recover the damage that Abaqus's own conversion implies:

    d = (1-beta) e_in E0 / (sigma + (1-beta) e_in E0)

beta -> 0 approaches GB's own d (pure damage); beta = 1 gives pure plasticity
with no stiffness degradation.  Literature values are beta_c ~ 0.35-0.7 in
compression and beta_t ~ 0.5-0.95 in tension.  This is a modelling choice, not
a derivation -- it is exposed in the case file and the defaults are stated, not
hidden.  beta = 0 exactly is rejected: with no plastic strain there is no
hardening variable for CDP to index its curves by.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Curve:
    """Piecewise-linear (ep -> effective stress, damage) table."""

    ep: np.ndarray        # (m,) equivalent plastic strain, ep[0] == 0
    sig: np.ndarray       # (m,) effective stress
    dmg: np.ndarray       # (m,) damage
    name: str = ""
    beta_requested: float = 1.0
    beta_effective: float = 1.0

    def __post_init__(self):
        self.ep = np.ascontiguousarray(self.ep, dtype=np.float64)
        self.sig = np.ascontiguousarray(self.sig, dtype=np.float64)
        self.dmg = np.ascontiguousarray(self.dmg, dtype=np.float64)
        if self.ep.ndim != 1 or self.ep.size < 2:
            raise ValueError(f"{self.name}: need at least two curve points")
        if abs(self.ep[0]) > 1e-14:
            raise ValueError(f"{self.name}: curve must start at ep = 0")
        d = np.diff(self.ep)
        if np.any(d <= 0):
            bad = int(np.argmin(d))
            raise ValueError(
                f"{self.name}: equivalent plastic strain is not increasing at "
                f"point {bad + 1} (ep = {self.ep[bad]:.6e} -> "
                f"{self.ep[bad + 1]:.6e}). With damage data this usually means "
                "the damage grows faster than the stress falls; reduce d or "
                "soften the stress curve."
            )
        if np.any(self.dmg < 0) or np.any(self.dmg >= 1.0):
            raise ValueError(f"{self.name}: damage must lie in [0, 1)")
        if np.any(self.sig <= 0):
            raise ValueError(f"{self.name}: effective stress must be positive")
        # per-segment slopes, so evaluate() never differentiates at run time
        self.slope = np.diff(self.sig) / d

    # ------------------------------------------------------------------ #

    @property
    def sigma0(self) -> float:
        """Effective stress at zero plastic strain: the elastic limit."""
        return float(self.sig[0])

    def pack(self, dtype=np.float64) -> np.ndarray:
        """Flatten to [ep | sigma | damage | slope] for the GPU backend.

        Four equal-length columns in one buffer, so however many curves a
        material has it still hands the device exactly two arrays: scalars and
        this.  `slope` has one fewer real entry; the last slot is padding.
        """
        n = self.ep.size
        out = np.empty(4 * n, dtype=dtype)
        out[0 * n:1 * n] = self.ep
        out[1 * n:2 * n] = self.sig
        out[2 * n:3 * n] = self.dmg
        out[3 * n:4 * n - 1] = self.slope
        out[4 * n - 1] = 0.0
        return out

    def evaluate(self, ep: np.ndarray, k: np.ndarray | float = 1.0):
        """Return (effective stress, d(stress)/d(ep), damage) at `ep`.

        `k = l_e / l_ref` applies the crack-band scaling: the curve is queried
        at ep*k and the slope comes back multiplied by k.
        """
        x = ep * k
        i = np.searchsorted(self.ep, x, side="right") - 1
        np.clip(i, 0, self.ep.size - 2, out=i)
        past = x >= self.ep[-1]

        t = x - self.ep[i]
        sig = self.sig[i] + self.slope[i] * t
        slope = self.slope[i]
        dmg = np.interp(x, self.ep, self.dmg)

        # beyond the table: hold the last stress and damage, zero slope.
        # Extrapolating a softening branch to negative stress is the fastest
        # way to turn a converging run into garbage.
        sig = np.where(past, self.sig[-1], sig)
        slope = np.where(past, 0.0, slope)
        return sig, slope * k, dmg


# --------------------------------------------------------------------------- #
#  Abaqus-style tabular input
# --------------------------------------------------------------------------- #


def from_abaqus_tables(stress, inelastic_strain, E0, damage=None,
                       damage_strain=None, name="") -> Curve:
    """Build a Curve from the tables Abaqus CDP takes.

    stress / inelastic_strain : the hardening table (yield stress vs inelastic
        or cracking strain).  The first point may be at zero inelastic strain.
    damage / damage_strain : the damage table, against the same inelastic
        strain measure.  Omit for an undamaged (plasticity-only) curve.
    """
    e_in = np.asarray(inelastic_strain, dtype=float)
    sig = np.asarray(stress, dtype=float)
    if e_in.shape != sig.shape:
        raise ValueError(f"{name}: stress and inelastic strain lengths differ")
    order = np.argsort(e_in)
    e_in, sig = e_in[order], sig[order]
    if e_in[0] > 1e-14:
        e_in = np.concatenate([[0.0], e_in])
        sig = np.concatenate([[sig[0]], sig])

    if damage is None:
        d = np.zeros_like(e_in)
    else:
        ds = np.asarray(damage_strain if damage_strain is not None
                        else inelastic_strain, dtype=float)
        d = np.interp(e_in, np.sort(ds), np.asarray(damage, float)[np.argsort(ds)])
    d = np.clip(d, 0.0, 0.999999)

    # Abaqus's own conversion from inelastic to plastic strain
    ep = e_in - d / (1.0 - d) * sig / E0
    ep = ep - ep[0]
    return Curve(ep, sig / (1.0 - d), d, name=name)


# --------------------------------------------------------------------------- #
#  GB 50010-2010 appendix C
# --------------------------------------------------------------------------- #
#
# The formulas below are transcribed from GB 50010-2010 appendix C.2 with
# stresses in MPa.  They are reproduced here in full so they can be checked
# against your copy of the code in half a minute:
#
#   compression (C.2.4)
#       eps_cr = (700 + 172*sqrt(fc)) * 1e-6
#       alpha_c = 0.157 * fc**0.785 - 0.905
#       rho_c   = fc / (E0 * eps_cr)
#       n       = E0*eps_cr / (E0*eps_cr - fc)
#       x = eps/eps_cr
#       x <= 1 :  d_c = 1 - rho_c*n / (n - 1 + x**n)
#       x >  1 :  d_c = 1 - rho_c / (alpha_c*(x-1)**2 + x)
#       sigma = (1 - d_c) * E0 * eps
#
#   tension (C.2.3)
#       eps_tr = ft**0.54 * 65e-6
#       alpha_t = 0.312 * ft**2
#       rho_t   = ft / (E0 * eps_tr)
#       x = eps/eps_tr
#       x <= 1 :  d_t = 1 - rho_t*(1.2 - 0.2*x**5)
#       x >  1 :  d_t = 1 - rho_t / (alpha_t*(x-1)**1.7 + x)
#       sigma = (1 - d_t) * E0 * eps
#
# Two properties are worth knowing before using them as CDP input:
#
#  * The COMPRESSION curve has initial tangent exactly E0 (d_c(0) = 0), so it
#    can be used from any stress level.
#  * The TENSION curve does NOT: d_t(0) = 1 - 1.2*rho_t, i.e. its initial
#    secant modulus is 1.2*ft/eps_tr, typically about 0.85 E0.  That is a real
#    feature of the empirical fit, not a transcription error, but it means the
#    ascending branch cannot be fed to CDP as-is without implying damage at
#    zero strain.  `gb50010_tension` therefore uses the DESCENDING branch only
#    and treats the material as linear elastic up to ft -- which is what CDP
#    assumes anyway.
#
# test_curves.py checks both properties, so a transcription error in the shape
# of either curve fails the suite.


def gb50010_compression(fc, E0, beta_c=0.7, sigma_c0_ratio=0.4,
                        alpha_c=None, eps_cr=None, x_max=6.0, n_points=120,
                        Gc=None, l_ref=None,
                        name="GB50010 compression") -> Curve:
    """GB 50010 appendix C.2.4 compressive curve, converted to CDP input.

    fc, E0 in Pa.  `sigma_c0_ratio` is the elastic limit as a fraction of fc
    (0.4 is the usual choice); below it the material is treated as elastic.

    `Gc` (J/m2) REGULARISES THE DESCENDING BRANCH, and without it the branch
    is not regularised at all.  The code curve is a STRAIN curve: crushing
    localises into one element row whatever its size, so the energy that row
    dissipates per unit crushed AREA is the curve's energy per volume times
    the element size -- 9.6 N/mm at 75 mm and 2.4 N/mm at 18.75 mm on the
    Alberta beam's concrete.  Measured there: the peak load fell from 334 to
    300 kN over that refinement while the pre-peak branch did not move.
    That is the compression twin of what a crack band does
    for tension, and the answer is the same one: state the energy per area,
    put the curve on a REFERENCE length, and let the crack-band factor
    k = l_e/l_ref rescale it per element.

    Only the post-peak branch is stretched; the ascending branch is the
    code's (Feenstra & de Borst 1995, and how Abaqus' "crushing energy" is
    meant).  Literature for the value: 8.8 sqrt(f_c[MPa]) N/mm (Nakamura &
    Higai 2001) or ~250 Gf (Feenstra) -- 58 and 35 N/mm for f_c = 44 MPa.
    It is a MODELLING CHOICE and belongs in the case file, so there is no
    default: without `Gc` the curve is what the code gives, unregularised,
    as every version of this project before v0.1.
    """
    fc_MPa, E_MPa = fc / 1e6, E0 / 1e6
    if eps_cr is None:
        eps_cr = (700.0 + 172.0 * np.sqrt(fc_MPa)) * 1e-6
    if alpha_c is None:
        alpha_c = 0.157 * fc_MPa ** 0.785 - 0.905
    rho_c = fc_MPa / (E_MPa * eps_cr)
    if not 0.0 < rho_c < 1.0:
        raise ValueError(
            f"GB50010 compression: rho_c = {rho_c:.3f} is outside (0,1); "
            f"E0 = {E_MPa:.0f} MPa and fc = {fc_MPa:.1f} MPa are inconsistent "
            "with the code's peak strain."
        )
    n_exp = 1.0 / (1.0 - rho_c)

    x = np.linspace(1e-6, x_max, 4000)
    d = np.where(
        x <= 1.0,
        1.0 - rho_c * n_exp / (n_exp - 1.0 + np.power(x, n_exp)),
        1.0 - rho_c / (alpha_c * np.maximum(x - 1.0, 0.0) ** 2 + x),
    )
    eps = x * eps_cr
    sig = (1.0 - d) * E0 * eps

    # start the table at the elastic limit
    peak = int(np.argmax(sig))
    lo = np.argmax(sig[:peak + 1] >= sigma_c0_ratio * fc)
    eps, sig = eps[lo:], sig[lo:]
    idx = np.unique(np.linspace(0, eps.size - 1, n_points).astype(int))
    eps, sig = eps[idx], sig[idx]
    if Gc is not None:
        eps = _crushing_energy(eps, sig, E0, float(Gc), l_ref, name)
    return _split(eps, sig, E0, beta_c, name)


def _crushing_energy(eps, sig, E0, Gc, l_ref, name):
    """Stretch the post-peak INELASTIC strain so a band `l_ref` thick
    dissipates `Gc` per unit area, keeping the code curve's stress values."""
    if l_ref is None:
        raise ValueError(f"{name}: Gc needs a reference length; set the "
                         "material's `l_ref` (auto is the element size)")
    e_in = eps - sig / E0
    i0 = int(np.argmax(sig))
    g = float(np.trapezoid(sig[i0:], e_in[i0:]) if hasattr(np, "trapezoid")
              else np.trapz(sig[i0:], e_in[i0:]))
    if g <= 0.0:
        raise ValueError(f"{name}: the curve has no descending branch to "
                         "calibrate Gc against")
    scale = Gc / (g * float(l_ref))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{name}: Gc = {Gc:g} J/m2 gives a non-positive scale")
    e_in = e_in.copy()
    e_in[i0:] = e_in[i0] + scale * (e_in[i0:] - e_in[i0])
    return sig / E0 + e_in


def gb50010_tension(ft, E0, beta_t=0.9, alpha_t=None, eps_tr=None,
                    x_max=25.0, n_points=120,
                    name="GB50010 tension") -> Curve:
    """GB 50010 appendix C.2.3 tensile curve (descending branch), as CDP input.

    Elastic to ft, then GB's softening branch.  See the note above for why the
    ascending branch is not used.
    """
    ft_MPa, E_MPa = ft / 1e6, E0 / 1e6
    if eps_tr is None:
        eps_tr = ft_MPa ** 0.54 * 65e-6
    if alpha_t is None:
        alpha_t = 0.312 * ft_MPa ** 2
    rho_t = ft_MPa / (E_MPa * eps_tr)

    x = np.concatenate([[1.0], np.linspace(1.0, x_max, 4000)[1:]])
    d = 1.0 - rho_t / (alpha_t * (x - 1.0) ** 1.7 + x)
    eps_gb = x * eps_tr
    sig = (1.0 - d) * E0 * eps_gb

    # re-anchor: elastic to ft, so the strain at the peak is ft/E0, not eps_tr
    eps = ft / E0 + (eps_gb - eps_tr)
    sig[0] = ft
    idx = np.unique(np.linspace(0, eps.size - 1, n_points).astype(int))
    return _split(eps[idx], sig[idx], E0, beta_t, name)


def _split(eps, sig, E0, beta, name) -> Curve:
    """Split inelastic strain into plastic strain + damage (see module docstring).

    The plastic fraction alone is not enough.  Substituting the derived damage
    back gives

        sigma_bar = sigma / (1-d) = sigma + (1-beta) e_in E0

    so for any beta < 1 the EFFECTIVE stress eventually grows without bound on a
    softening branch: with the beam case's own numbers the tensile yield stress
    reaches 4.8 MPa against ft = 3 MPa.  That is a yield surface EXPANDING while
    the material cracks -- the opposite of softening, and it makes the return map
    chase a target that runs away from it.

    So the effective stress is additionally capped to be non-increasing after the
    peak.  Lowering sigma_bar lowers d = 1 - sigma/sigma_bar, so where the cap
    binds MORE of the softening is carried by plasticity and less by damage than
    `beta` asked for -- the surface shrinks instead of the stiffness degrading.
    `Curve.beta_effective` reports the fraction actually achieved and it is
    printed in the run summary rather than applied silently.
    """
    if not 0.0 < beta <= 1.0:
        raise ValueError(
            f"{name}: plastic fraction beta must be in (0,1]. beta = 0 means no "
            "plastic strain at all, and CDP indexes its curves by the equivalent "
            "PLASTIC strain -- there would be nothing to index by. Use a pure "
            "damage model if that is what you want."
        )
    e_in = np.maximum(eps - sig / E0, 0.0)
    e_in = e_in - e_in[0]

    num = (1.0 - beta) * e_in * E0
    d = np.clip(num / (sig + num), 0.0, 0.999)
    sig_bar = sig / (1.0 - d)

    # cap: no expansion of the surface once the Cauchy stress has peaked
    i0 = int(np.argmax(sig))
    sig_bar[i0:] = np.minimum.accumulate(sig_bar[i0:])
    d = np.clip(1.0 - sig / sig_bar, 0.0, 0.999)
    # keep the Abaqus relation exactly, so the curve is still valid CDP input
    ep = e_in - d / (1.0 - d) * sig / E0
    ep = ep - ep[0]

    keep = np.concatenate([[True], np.diff(ep) > 1e-16])
    c = Curve(ep[keep], sig_bar[keep], d[keep], name=name)
    m = e_in[keep] > 1e-12
    c.beta_effective = (float(np.median(c.ep[m] / e_in[keep][m])) if m.any()
                        else float(beta))
    c.beta_requested = float(beta)
    return c


# --------------------------------------------------------------------------- #
#  Stress-displacement (crack opening) input
# --------------------------------------------------------------------------- #


def from_stress_displacement(stress, displacement, E0, l_ref,
                             damage=None, name="") -> Curve:
    """Tension softening given as stress vs crack opening w.

    Converted to a strain-based curve at the reference length `l_ref`; the
    per-element crack-band factor then rescales it at run time.
    """
    w = np.asarray(displacement, dtype=float)
    return from_abaqus_tables(stress, w / l_ref, E0, damage=damage,
                              damage_strain=None if damage is None else w / l_ref,
                              name=name)


def from_fracture_energy(ft, Gf, E0, l_ref, beta_t=0.9, shape="exponential",
                         n_points=100, name="Gf tension") -> Curve:
    """Exponential (or linear) tension softening calibrated to Gf.

    Exponential: sigma = ft exp(-ft w / Gf), which dissipates exactly Gf per
    unit crack area.  Linear: sigma = ft (1 - w/w_u) with w_u = 2 Gf / ft.
    """
    if shape == "linear":
        w_u = 2.0 * Gf / ft
        w = np.linspace(0.0, w_u * 0.999, n_points)
        sig = ft * (1.0 - w / w_u)
    elif shape == "exponential":
        w_u = 8.0 * Gf / ft                    # exp(-8) ~ 3e-4 of ft
        w = np.linspace(0.0, w_u, n_points)
        sig = ft * np.exp(-ft * w / Gf)
    else:
        raise ValueError(f"unknown softening shape {shape!r}")

    # w is the INELASTIC crack opening, so the total strain is sigma/E0 + w/l_ref,
    # not ft/E0 + w/l_ref.  Anchoring at ft instead folds the elastic recovery
    # (ft - sigma)/E into the opening, which inflates the dissipated energy
    # (measured: 153 J/m2 against a requested Gf of 120) and, worse, does it by
    # an amount that does not scale with the element size -- so the crack band
    # stops being mesh objective at the extremes.
    eps = sig / E0 + w / l_ref
    return _split(eps, sig, E0, beta_t, name)


# --------------------------------------------------------------------------- #
#  Isotropic hardening curves for metals
#
#  A hardening curve is a Curve with zero damage: sigma_y against equivalent
#  plastic strain.  Reusing the same type means J2 gets the table interpolation,
#  the packed GPU layout and the flat-past-the-end guard for free -- and the
#  same `evaluate` returns (sigma_y, H', 0), which is exactly what a return map
#  needs.
# --------------------------------------------------------------------------- #


def hardening_linear(yield_strength, H=0.0, ep_max=10.0, name="linear hardening"):
    """sigma_y = sy + H ep.  H = 0 is perfect plasticity."""
    sy = float(yield_strength)
    H = float(H)
    if H < 0:
        raise ValueError("linear hardening modulus must be >= 0")
    ep = np.array([0.0, float(ep_max)])
    # a strictly flat curve would make the return map's denominator depend only
    # on 3G, which is correct; the tiny slope floor avoids a zero-length table
    return Curve(ep, sy + H * ep, np.zeros(2), name=name)


def hardening_power(yield_strength, K, n, ep_max=1.0, n_points=80,
                    name="power hardening"):
    """sigma_y = sy + K ep^n  (Ludwik).  n in (0, 1]."""
    sy, K, n = float(yield_strength), float(K), float(n)
    if not 0.0 < n <= 1.0:
        raise ValueError("power-law exponent must lie in (0, 1]")
    # the tabulated range reaches well past necking: a curve that stopped at
    # a plastic strain of 1 held sigma_y flat from there (1300 MPa where the
    # law gives 1894 at 1.59)
    ep = np.unique(np.r_[np.linspace(0.0, float(ep_max), n_points), [2.0, 3.0, 5.0, 10.0]])
    return Curve(ep, sy + K * ep ** n, np.zeros(len(ep)), name=name)


def hardening_table(stress, plastic_strain, name="tabulated hardening"):
    """sigma_y against equivalent plastic strain, as a table."""
    ep = np.asarray(plastic_strain, dtype=float)
    sy = np.asarray(stress, dtype=float)
    if ep.shape != sy.shape:
        raise ValueError(f"{name}: stress and plastic strain lengths differ")
    order = np.argsort(ep)
    ep, sy = ep[order], sy[order]
    if ep[0] > 1e-14:
        raise ValueError(
            f"{name}: the table must start at zero plastic strain -- its first "
            "point IS the initial yield stress"
        )
    return Curve(ep - ep[0], sy, np.zeros(ep.size), name=name)


def build_hardening(spec, name="hardening") -> Curve:
    """Dispatch a hardening spec from a case file."""
    spec = dict(spec)
    source = spec.pop("source", "linear")
    if source == "linear":
        return hardening_linear(name=name, **spec)
    if source == "power":
        return hardening_power(name=name, **spec)
    if source == "table":
        return hardening_table(name=name, **spec)
    raise ValueError(
        f"unknown hardening source {source!r}; expected linear, power or table"
    )
