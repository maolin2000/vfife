"""Uniaxial laws made of other uniaxial laws, and a gap (1.21).

    parallel   every part sees the same strain; the stresses add,
               sigma = sum f_i sigma_i(eps), each scaled by its factor
    series     every part carries the same stress; the strains add,
               eps = sum eps_i, the split found at each point by iteration
    gap        free travel between two strains, linear elastic past them:
               sigma = E (eps - hi) above hi, E (eps - lo) below lo, else 0

With these a device is put together from laws the library already has --
a buckling-restrained brace's core beside an elastic spring, a stop with
its slack in series with a yielding element -- instead of each being a law
of its own. The names and the arithmetic are those of the usual wrappers
(OpenSees' Parallel and Series uniaxial materials, its ElasticPPGap without
the plasticity); the code is written from the definitions above.

A PART IS READ AS A UNIAXIAL STRESS. `linear_elastic` handed
[eps, 0, 0, 0, 0, 0] answers with the constrained modulus C11 eps, 35% stiff
at nu = 0.3, which is why the fibre beam and the bar read it as E eps
(fiber_beam_solver, elements/truss); a part here is read the same way.

THE STATE IS THE PARTS' STATES, END TO END, and a series law's own split
after them (the strain each part carries). `update` advances the state it
is given in place, as every law does: the caller hands it the committed
state (or a copy of it, for a trial) and gets the new one back in the same
array. A series law's iteration evaluates its parts on COPIES of what it
was given and writes the parts' states once, at the converged split, so an
iteration that did not end where it began leaves nothing behind.

A SERIES SPLIT THAT DOES NOT CONVERGE IS AN ERROR, never a quiet answer:
the stresses of the parts would differ and the law would carry a force no
part carries.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .base import Material, register


def _part(spec):
    """A part's Material, from its spec (a mapping with its `type`)."""
    from .factory import create_material
    if isinstance(spec, Material):
        return spec
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError(f"a part of a composite material is a material spec with a `type`, not {spec!r}")
    m = create_material(dict(spec))
    if not getattr(m, "uniaxial", False) and m.name != "linear_elastic":
        raise ValueError(f"a part of a composite material must be a one-dimensional law; "
                         f"{m.name!r} is not (use linear_elastic, steel02, concrete01/02, "
                         "piecewise_uniaxial, steel_cyclic, rebar, gap, bouc_wen, parallel or series)")
    return m


def uniaxial_stress(mat, eps: np.ndarray, state: np.ndarray, dt: float) -> np.ndarray:
    """sigma(eps) of a one-dimensional law at n points, advancing `state`
    (n, n_state) in place; `linear_elastic` is E eps (see the module note)."""
    if mat.name == "linear_elastic":
        return mat.E * np.asarray(eps, dtype=float)
    strain6 = np.zeros((len(eps), 6))
    strain6[:, 0] = eps
    return np.asarray(mat.update(strain6, state, dt))[:, 0].copy()


def _drive(mat, eps, state, sl, dt, keep: bool):
    """A part's stress, from its columns `sl` of `state`, on a contiguous copy
    (a compiled law takes contiguous rows); written back when `keep`."""
    sub = np.array(state[:, sl], dtype=float, order="C", copy=True)     # never a view
    sig = uniaxial_stress(mat, eps, sub, dt)
    if keep and mat.n_state:
        state[:, sl] = sub
    return sig


class _Composite(Material):
    uniaxial = True

    def _build(self, of, what):
        if not isinstance(of, (list, tuple)) or len(of) < 2:
            raise ValueError(f"{what}: `of` lists two or more materials, not {of!r}")
        self.parts: List[Material] = [_part(s) for s in of]
        self.slices = []
        at = 0
        for p in self.parts:
            self.slices.append(slice(at, at + p.n_state))
            at += p.n_state
        self._name_parts()
        return at

    def _name_parts(self):
        """Each part's state is one named block: `part1_steel02`, ..."""
        self.state_names = {f"part{i + 1}_{p.name}": sl
                            for i, (p, sl) in enumerate(zip(self.parts, self.slices)) if p.n_state}

    def _no_slow(self, what):
        for p in self.parts:
            for k in ("creep", "shrinkage", "ageing", "relaxation"):
                if getattr(p, k, None) is not None:
                    raise ValueError(f"{what}: a part carries `{k}`, which a composite does not pass on; "
                                     "a composite material has no time-dependent strain")
            if float(getattr(p, "thermal_expansion", 0.0) or 0.0) != 0.0:
                raise ValueError(f"{what}: a part carries `thermal_expansion`, which a composite does not "
                                 "pass on; give the member its temperature on a single law")

    #: how the parts' failures make the composite's: side by side it is
    #: gone when every part is, end to end when any part is
    _gone = staticmethod(np.minimum)

    def failure(self, state):
        """The parts' failures, combined (`_gone`); a part with no notion of
        failure counts as intact. None when no part has one."""
        got = [p.failure(np.ascontiguousarray(state[:, sl])) for p, sl in zip(self.parts, self.slices)]
        if all(g is None for g in got):
            return None
        n = len(state)
        got = [np.zeros(n) if g is None else np.asarray(g, float) for g in got]
        out = got[0]
        for g in got[1:]:
            out = self._gone(out, g)
        return out

    def damage_class(self, state):
        """The worst class of any part (a picture of what has happened in it)."""
        got = [p.damage_class(np.ascontiguousarray(state[:, sl])) for p, sl in zip(self.parts, self.slices)]
        got = [g for g in got if g is not None]
        if not got:
            return None
        return np.max(np.stack(got), axis=0).astype(np.int8)

    def init_state(self, n_points: int, dtype=np.float64) -> np.ndarray:
        state = np.zeros((n_points, self.n_state), dtype=dtype)
        for p, sl in zip(self.parts, self.slices):
            if p.n_state:
                state[:, sl] = p.init_state(n_points, dtype)
        return state


@register("parallel", kinematics=("uniaxial",), backends=("numpy",),
          verification="analytical", output_fields=("stress",),
          caveat="no compiled form: a fibre section with it runs on the NumPy reference")
class Parallel(_Composite):
    """sigma = sum f_i sigma_i(eps): the parts side by side."""

    state_names: dict = {}

    def __init__(self, of: Sequence, factors: Optional[Sequence[float]] = None,
                 rho: Optional[float] = None, **_ignored):
        self.n_state = self._build(of, "parallel")
        n = len(self.parts)
        f = [1.0] * n if factors is None else [float(x) for x in factors]
        if len(f) != n:
            raise ValueError(f"parallel: {len(f)} factors for {n} materials")
        if any(not np.isfinite(x) or x < 0.0 for x in f) or not any(x > 0.0 for x in f):
            raise ValueError(f"parallel: factors are finite, non-negative and not all zero, not {f}")
        self.factors = f
        if rho is not None and not float(rho) > 0.0:
            raise ValueError("parallel: rho is positive")
        self._no_slow("parallel")
        self.E = float(sum(fi * p.E for fi, p in zip(f, self.parts)))
        # the parts side by side, each its factor of the section: their
        # densities add the same way, unless the case says
        self.rho = float(sum(fi * p.rho for fi, p in zip(f, self.parts))) if rho is None else float(rho)
        self.nu = float(getattr(self.parts[0], "nu", 0.0) or 0.0)
        self.wave_speed = float(np.sqrt(self.E / self.rho))

    def update(self, strain, state, dt):
        eps = np.asarray(strain, dtype=float)[:, 0]
        sig = np.zeros(len(eps))
        for fi, p, sl in zip(self.factors, self.parts, self.slices):
            sig += fi * _drive(p, eps, state, sl, dt, keep=True)
        out = np.zeros_like(strain, dtype=float)
        out[:, 0] = sig
        return out

    def describe(self) -> str:
        return ("    parallel of " + ", ".join(p.name for p in self.parts)
                + f" (factors {self.factors}); E = {self.E:.4g}")


#: the relative strain step of a part's numerical tangent
_H = 1.0e-7


@register("series", kinematics=("uniaxial",), backends=("numpy",),
          verification="analytical", output_fields=("stress",),
          caveat="no compiled form: a fibre section with it runs on the NumPy reference; "
                 "the split is iterated at each point and a split that does not converge is an error")
class Series(_Composite):
    """eps = sum eps_i at one stress: the parts end to end.

    (A part's failure is the series': `_gone` is the maximum.)

    Two parts at a time: three or more are the first in series with the
    series of the rest. For two, the split is the root of

        f(e1) = sigma_1(e1) - sigma_2(eps - e1),

    the first part's strain e1 at which the two carry the same stress. It
    is bracketed between the committed e1 and the committed e1 plus the
    whole increment (a part whose stress does not fall as its strain grows
    takes a share of the increment between nothing and all of it), the
    bracket widened by doublings when a softening part puts the root
    outside, and found by Newton steps on the parts' tangents that fall
    back to bisection whenever a step would leave the bracket. Converged
    when |f| is within `tolerance` of the stress plus E x 1e-15 (the stress
    of a strain at round-off, so a point at rest is not asked for an exact
    zero). Every evaluation starts
    from the committed state of the parts; the parts' states are written
    once, at the root."""

    state_names: dict = {}
    _gone = staticmethod(np.maximum)

    def __init__(self, of: Sequence, max_iterations: int = 100, tolerance: float = 1.0e-10,
                 rho: Optional[float] = None, **_ignored):
        if not isinstance(of, (list, tuple)) or len(of) < 2:
            raise ValueError(f"series: `of` lists two or more materials, not {of!r}")
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        if self.max_iterations < 1 or not (0.0 < self.tolerance < 1.0e-2):
            raise ValueError("series: max_iterations is at least 1 and tolerance in (0, 1e-2)")
        first = _part(of[0])
        second = (_part(of[1]) if len(of) == 2
                  else Series(of=list(of[1:]), max_iterations=max_iterations, tolerance=tolerance))
        self.parts = [first, second]
        self.slices = [slice(0, first.n_state), slice(first.n_state, first.n_state + second.n_state)]
        # after the parts' states: the first part's strain and the total, committed
        self.e1 = first.n_state + second.n_state
        self.tot = self.e1 + 1
        self.n_state = self.e1 + 2
        self._name_parts()
        self.state_names.update({"first_part_strain": slice(self.e1, self.e1 + 1),
                                 "total_strain": slice(self.tot, self.tot + 1)})
        self._no_slow("series")
        if not (first.E > 0.0 and second.E > 0.0):
            raise ValueError("series: every part needs a positive initial modulus `E`")
        self.E = 1.0 / (1.0 / first.E + 1.0 / second.E)
        # the parts end to end: the mean of their densities (each part an
        # equal share of the length), unless the case says
        self._leaf_rho = [first.rho] + (second._leaf_rho if isinstance(second, Series) else [second.rho])
        self.rho = float(np.mean(self._leaf_rho)) if rho is None else float(rho)
        if not self.rho > 0.0:
            raise ValueError("series: rho is positive")
        self.nu = float(getattr(first, "nu", 0.0) or 0.0)
        self.wave_speed = float(np.sqrt(self.E / self.rho))
        self._names = [p.name for p in (first,)] + (
            [second.name] if len(of) == 2 else second._names)

    def _f(self, x, eps, state, dt):
        """f, its slope and sigma_1 at e1 = x, from the committed state."""
        p1, p2 = self.parts
        s1 = _drive(p1, x, state, self.slices[0], dt, keep=False)
        s2 = _drive(p2, eps - x, state, self.slices[1], dt, keep=False)
        h = _H * np.maximum(np.maximum(np.abs(x), np.abs(eps - x)), 1.0e-4)
        k1 = (_drive(p1, x + h, state, self.slices[0], dt, keep=False) - s1) / h
        k2 = (s2 - _drive(p2, eps - x - h, state, self.slices[1], dt, keep=False)) / h
        return s1 - s2, k1 + k2, s1

    def update(self, strain, state, dt):
        eps = np.asarray(strain, dtype=float)[:, 0]
        floor = self.E * 1.0e-15            # the stress of a strain at round-off: a point at rest
        e1c = state[:, self.e1].copy()
        d = eps - state[:, self.tot]                 # the increment from the committed total
        a = e1c.copy(); b = e1c + d
        fa = self._f(a, eps, state, dt)[0]; fb = self._f(b, eps, state, dt)[0]
        width = np.maximum(np.abs(d), 1.0e-12)
        for _ in range(8):                                       # a softening part: widen
            same = np.sign(fa) * np.sign(fb) > 0.0
            if not np.any(same):
                break
            lo = np.minimum(a, b) - width; hi = np.maximum(a, b) + width
            a = np.where(same, lo, a); b = np.where(same, hi, b)
            fa = np.where(same, self._f(a, eps, state, dt)[0], fa)
            fb = np.where(same, self._f(b, eps, state, dt)[0], fb)
            width = width * 2.0
        same = np.sign(fa) * np.sign(fb) > 0.0
        if np.any(same):
            # still no change of sign: a part whose stress fell away (a
            # rupture, a crushed concrete) puts the root where the other part
            # carries what the broken one does -- near nothing, its strain
            # near zero. The bracket then spans zero, the whole strain and the
            # committed split (found in review: a series with a rupturing
            # steel02 refused the step where it broke)
            span = np.abs(eps) + np.abs(e1c) + 1.0e-12
            lo = np.minimum.reduce([a, b, np.zeros_like(a), eps]) - span
            hi = np.maximum.reduce([a, b, np.zeros_like(a), eps]) + span
            a = np.where(same, lo, a); b = np.where(same, hi, b)
            fa = np.where(same, self._f(a, eps, state, dt)[0], fa)
            fb = np.where(same, self._f(b, eps, state, dt)[0], fb)
        x = e1c + d * (1.0 / self.parts[0].E) * self.E              # shared by the initial compliances
        x = np.clip(x, np.minimum(a, b), np.maximum(a, b))
        done = np.zeros(len(eps), dtype=bool)
        for it in range(self.max_iterations + 1):
            f, g, s1 = self._f(x, eps, state, dt)
            ok = np.abs(f) <= self.tolerance * np.abs(s1) + floor
            done = done | ok
            if np.all(done):
                break
            if it == self.max_iterations:
                bad = int(np.flatnonzero(~done)[0])
                from ..base.errors import UserError
                raise UserError(
                    "SERIES_NOT_CONVERGED",
                    f"material 'series' ({' + '.join(self._names)}): the parts' stresses at point {bad} "
                    f"differ by {abs(f[bad]):.4g} after {self.max_iterations} iterations (strain {eps[bad]:.6g})",
                    "raise `max_iterations` or loosen `tolerance`; or a part's stress cannot reach the "
                    "other's (a part that has broken carries nothing)")
            # keep the bracket on the sign of f
            left = np.sign(f) == np.sign(fa)
            a = np.where(~done & left, x, a); fa = np.where(~done & left, f, fa)
            b = np.where(~done & ~left, x, b); fb = np.where(~done & ~left, f, fb)
            with np.errstate(divide="ignore", invalid="ignore"):
                newton = x - f / g
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            inside = np.isfinite(newton) & (newton > lo) & (newton < hi)
            step = np.where(inside, newton, 0.5 * (a + b))
            x = np.where(done, x, step)
        # the parts' states at the root, written once
        _drive(self.parts[0], x, state, self.slices[0], dt, keep=True)
        _drive(self.parts[1], eps - x, state, self.slices[1], dt, keep=True)
        state[:, self.e1] = x
        state[:, self.tot] = eps
        out = np.zeros_like(strain, dtype=float)
        out[:, 0] = s1
        return out

    def describe(self) -> str:
        return ("    series of " + " + ".join(self._names)
                + f"; E = {self.E:.4g}, at most {self.max_iterations} iterations to {self.tolerance:g}")


@register("gap", kinematics=("uniaxial",), backends=("numpy",), nonlinear=True,
          verification="analytical", output_fields=("stress",))
class Gap(Material):
    """Free travel between `lo` and `hi` (strains), linear elastic past them.

    sigma = E (eps - hi) for eps > hi, E (eps - lo) for eps < lo, else 0.
    Either bound may be left out (no stop that side); at least one is given.
    No history. Read through a connector's `area` and `length` (or a bar's
    length), the bounds are the device's free travel."""

    n_state = 0
    state_names: dict = {}
    uniaxial = True

    def __init__(self, E: float, rho: float, lo: Optional[float] = None, hi: Optional[float] = None,
                 **_ignored):
        self.E = float(E)
        self.rho = float(rho)
        self.nu = 0.0
        self.lo = None if lo is None else float(lo)
        self.hi = None if hi is None else float(hi)
        if not (np.isfinite(self.E) and self.E > 0.0):
            raise ValueError("gap: E is finite and positive")
        if not self.rho > 0.0:
            raise ValueError("gap: rho is positive")
        if self.lo is None and self.hi is None:
            raise ValueError("gap: give `lo`, `hi` or both (with neither it carries nothing)")
        if self.lo is not None and self.hi is not None and not self.lo <= self.hi:
            raise ValueError(f"gap: lo ({self.lo}) is above hi ({self.hi})")
        self.wave_speed = float(np.sqrt(self.E / self.rho))

    def update(self, strain, state, dt):
        eps = np.asarray(strain, dtype=float)[:, 0]
        sig = np.zeros(len(eps))
        if self.hi is not None:
            sig = np.where(eps > self.hi, self.E * (eps - self.hi), sig)
        if self.lo is not None:
            sig = np.where(eps < self.lo, self.E * (eps - self.lo), sig)
        out = np.zeros_like(strain, dtype=float)
        out[:, 0] = sig
        return out

    def describe(self) -> str:
        return f"    gap E={self.E:.4g}, free between {self.lo} and {self.hi}"
