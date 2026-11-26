"""Rayleigh damping by region (1.22): `solver.damping.regions`.

WHAT. The model's Rayleigh damping is alpha_0 M v + beta_0 K v, on the mass
in force and the reference K (`BeamFamily.step`). An entry of
`damping.regions` gives the members of some regions their own alpha_e and
beta_e, and the damping force becomes

    f = alpha_0 M v + beta_0 K v
        + sum over the region members in force of
          (alpha_e - alpha_0) M_e v_e + (beta_e - beta_0) K_e v_e

with M_e the member's lumped mass at its two ends (`member_mass_share`, and
its rigid zones' mass, `member_zone_mass`: the zones are the member's),
K_e its reference block (`member_reference_block`) and v_e the velocities
of the two nodes it is on NOW (a split moves a member's end to a spare
particle). The concentrated masses and every member outside the regions keep
alpha_0 and beta_0, so a model without `regions` is damped to the bit as
before: this term is not there.

Only the correction is summed here, on the region members alone: the K the
global term multiplies is kept up by the solvers (a member removed or broken
leaves it only when beta_0 is in use), and nothing here depends on that.
"In force" is `_live_members()` and, with analysis steps, built.

Where two entries overlap the later one holds (`end_offsets`' rule).

The explicit step is sized with the largest alpha and beta of the model
(`bounds`): on the safe side, by the factor of the stiffest region.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def bounds(solver) -> Tuple[float, float]:
    """(alpha, beta) to size the explicit step with: the model's, or the
    largest of any region."""
    a, b = float(getattr(solver, "alpha", 0.0)), float(getattr(solver, "beta", 0.0))
    regions = _spec_regions(solver)
    for r in regions or ():
        a, b = max(a, float(r.alpha)), max(b, float(r.beta))
    return a, b


def _spec(solver):
    p = getattr(solver, "p", None)
    sol = getattr(p, "solver", None) if p is not None else None
    return getattr(sol, "damping", None) if sol is not None else None


def _spec_regions(solver):
    damp = _spec(solver)
    return getattr(damp, "regions", None) if damp is not None else None


def members(solver):
    """(ids, d_alpha, d_beta) of the members whose damping differs from the
    model's, on the connectivity as read (called at setup, as the element
    monitors are: a member that later splits stays in its region, and a
    region that is not there is refused before the run); None without
    `damping.regions`."""
    from .member_actions import region_members
    regions = _spec_regions(solver)
    if not regions:
        return None
    # the model's coefficients as written: a relaxation with dynamic steps
    # has them on the solver only while a dynamic step runs
    spec = _spec(solver)
    a0, b0 = float(spec.alpha), float(spec.beta)
    s = solver
    e0 = np.asarray(s.mesh.elements)[:, :2].copy()
    da = np.zeros(s.nel); db = np.zeros(s.nel); mine = np.zeros(s.nel, bool)
    for i, r in enumerate(regions):
        ids = region_members(s.mesh, list(r.regions), e0, what=f"damping.regions[{i}]")
        da[ids] = float(r.alpha) - a0
        db[ids] = float(r.beta) - b0
        mine[ids] = True
    ids = np.flatnonzero(mine & ((da != 0.0) | (db != 0.0)))
    return ids, da[ids], db[ids]


class RegionalRayleigh:
    """The region members' correction to the model's Rayleigh damping."""

    def __init__(self, solver, members):
        s = self.s = solver
        self.ids, self.da, self.db = members
        self._dofs_key = None
        n = len(self.ids)
        #: (n, 12) the members' lumped mass at their two ends, and their
        #: (n, 12, 12) reference blocks -- both fixed at setup
        self.m = np.zeros((n, 12))
        self.k = np.zeros((n, 12, 12))
        for i, e in enumerate(self.ids):
            e = int(e)
            share = np.asarray(s.member_mass_share(e), float)
            zone = s.member_zone_mass(e) if s.geometry.any else (0.0, 0.0)
            for k in range(2):
                self.m[i, 6 * k:6 * k + 6] = share
                self.m[i, 6 * k:6 * k + 3] += float(zone[k])
            if self.db[i] != 0.0:
                self.k[i] = np.asarray(s.member_reference_block(e), float)

    def _in_force(self):
        s = self.s
        live = np.asarray(s.live_members(), bool)[self.ids]
        built = getattr(s, "member_built", None) if getattr(s, "plan", None) is not None else None
        if built is not None:
            live &= np.asarray(built, bool)[self.ids]
        return live

    def force(self, v) -> np.ndarray:
        """(ndof,) the correction at the velocities `v`."""
        s = self.s
        out = np.zeros_like(v)
        if not len(self.ids):
            return out
        on = self._in_force()
        if not on.any():
            return out
        ids = self.ids[on]
        conn = np.asarray(s.mesh.elements)[ids, :2]
        key = (ids.tobytes(), conn.tobytes())
        if key != self._dofs_key:                 # members built, removed or split since
            self._dofs = np.asarray([s.dofs.element_dofs(c) for c in conn])
            self._dofs_key = key
        dofs = self._dofs
        ve = v[dofs]                                          # (n, 12)
        f = self.da[on][:, None] * self.m[on] * ve
        if np.any(self.db[on]):
            f += self.db[on][:, None] * np.einsum("nij,nj->ni", self.k[on], ve)
        np.add.at(out, dofs.ravel(), f.ravel())
        return out


def regional(solver, members) -> Optional[RegionalRayleigh]:
    """A `RegionalRayleigh` of the solver over `members` (`members(solver)`);
    None when no member's damping differs from the model's
    (`BeamFamily.regional_damping` keeps the one it uses)."""
    return RegionalRayleigh(solver, members) if members is not None and len(members[0]) else None
