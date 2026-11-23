"""Lumped masses that are not the members' own.

Two things live here, both read by every solver that lumps its mass:

* `nodal_masses` -- a mass at the nodes of a region. Until this stage it was
  isotropic and it weighed: the same kg in x, y and z, and m g under
  `gravity`. It now takes `components` (which translations carry it),
  `inertia` (a rotary inertia about the global axes, for the six-dof
  families) and `weight` (whether gravity sees it). A directional mass is
  added mass -- a fluid's inertia, a heavy deck's inertia along one axis --
  and it does not weigh: `weight` defaults to True only when all three
  translations carry the mass, and is refused when it is asked for on fewer.

* `hydrodynamic_mass` -- the Morison added mass of the submerged part of the
  members of a region, C_a rho_w A per unit length, acting across the member
  and not along it, weighing nothing. It is what a pier or a pile in water
  adds to a seismic model, and what the Xihoumen survey found it could only
  write as 72 000 t of isotropic mass that weighed 706 MN.

* a `line_load` with `with_mass: true` (1.1) -- the mass |q|/g of what the
  load is the weight of: superimposed dead load, surfacing, parapets. Written
  as a load alone it weighed and had no inertia, so a seismic run, the modes
  and a response spectrum missed it (v1.0 found the bridge deck's
  second-phase dead load missing that way). Half of each member's |q| L / g
  goes to each end node in all three translations; it weighs nothing, since
  the load itself is its weight.

`table()` turns all three into three (nn, 3) arrays -- translational mass, rotary
inertia, and the part of the translational mass that does not weigh -- that
a solver adds to its own lumped mass. Gravity is then the translational mass
minus the weightless part, times g.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np

#: a member counts as along a global axis when it is within this angle of it
AXIS_TOLERANCE_DEG = 5.0


def _flags(spec) -> np.ndarray:
    return np.asarray(spec.get("components") or [1, 1, 1], bool)


def weighs(spec) -> bool:
    """A nodal mass weighs unless it says it does not; a directional one
    does not unless... it cannot (the schema refuses it)."""
    w = spec.get("weight")
    if w is None:
        return bool(_flags(spec).all())
    return bool(w)


def nodal(specs: Iterable[dict], region: Callable, nn: int):
    """(mass, inertia, weightless), each (nn, 3), from `nodal_masses`."""
    mass = np.zeros((nn, 3)); rot = np.zeros((nn, 3)); light = np.zeros((nn, 3))
    for spec in specs or []:
        idx = np.asarray(region(spec.get("region", "all")))
        share = 1.0 if spec.get("per_node") else 1.0 / max(len(idx), 1)
        m = float(spec.get("mass") or 0.0) * share
        add = m * _flags(spec)
        np.add.at(mass, idx, add)
        if not weighs(spec):
            np.add.at(light, idx, add)
        if spec.get("inertia") is not None:
            np.add.at(rot, idx, np.asarray(spec["inertia"], float) * share)
    return mass, rot, light


def _submerged(x0, x1, level, up):
    """The submerged fraction [s0, s1] of the segment x0 -> x1 (s along it)
    below `level` on axis `up`; None when nothing is under water."""
    z0, z1 = float(x0[up]), float(x1[up])
    if z0 <= level and z1 <= level:
        return 0.0, 1.0
    if z0 > level and z1 > level:
        return None
    s = (level - z0) / (z1 - z0)
    return (0.0, s) if z0 <= level else (s, 1.0)


def hydrodynamic(specs: Iterable[dict], region: Callable, nodes: np.ndarray,
                 members: np.ndarray, member_ok: Optional[np.ndarray] = None):
    """(mass, weightless), each (nn, 3), from `hydrodynamic_mass`.

    `members` is the (n, 2) connectivity of the line members; `member_ok`
    masks the ones a fluid can act on (connectors carry no section)."""
    nn = len(nodes)
    mass = np.zeros((nn, 3))
    for spec in specs or []:
        idx = np.asarray(region(spec["region"]))
        inside = np.zeros(nn, bool); inside[idx] = True
        sel = inside[members].all(axis=1)
        if member_ok is not None:
            sel &= member_ok
        chosen = np.flatnonzero(sel)
        if not len(chosen):
            raise ValueError(f"hydrodynamic_mass region {spec['region']!r} contains no complete "
                             "members: it is a set of members (a pile, a pier), not of nodes")
        area = spec.get("area")
        if area is None:
            area = np.pi * float(spec["diameter"]) ** 2 / 4.0
        per_length = float(spec.get("coefficient", 1.0)) * float(spec.get("density", 1025.0)) * float(area)
        up = "xyz".index(spec.get("vertical", "z"))
        level = float(spec.get("water_level", 0.0))
        wet = 0
        for e in chosen:
            a, b = members[e]
            x0, x1 = nodes[a], nodes[b]
            d = x1 - x0
            L = float(np.linalg.norm(d))
            if L <= 0.0:
                continue
            e_ = np.abs(d) / L
            axis = int(np.argmax(e_))
            # the TRUE angle to the nearest axis; tested per component, a
            # member 4.9 degrees off in two planes (6.9 in all) passed
            if e_[axis] < np.cos(np.radians(AXIS_TOLERANCE_DEG)):
                raise ValueError(
                    f"hydrodynamic_mass on {spec['region']!r}: member {int(e)} is "
                    f"{np.degrees(np.arccos(e_[axis])):.1f} degrees off the nearest global axis. The "
                    "added mass acts across a member and not along it, and a lumped (diagonal) mass "
                    f"can hold that only for members within {AXIS_TOLERANCE_DEG:g} degrees of x, y or "
                    "z -- piers and piles. An inclined member needs its own nodal_masses")
            seg = _submerged(x0, x1, level, up)
            if seg is None:
                continue
            s0, s1 = seg
            m = per_length * L * (s1 - s0)
            if m <= 0.0:
                continue
            wet += 1
            sc = 0.5 * (s0 + s1)                    # the wet part's centroid along the member
            across = np.ones(3); across[axis] = 0.0
            mass[a] += m * (1.0 - sc) * across
            mass[b] += m * sc * across
        if not wet:
            raise ValueError(f"hydrodynamic_mass region {spec['region']!r}: no member is below "
                             f"water_level {level:g} on the {spec.get('vertical', 'z')} axis")
    return mass, mass.copy()


#: standard gravity, m/s^2: the g that turns a `with_mass` line load into mass
STANDARD_GRAVITY = 9.80665


def line_load_g(units) -> float:
    """g in the case's acceleration unit (9.80665 when it declares none, or
    one that does not fix the unit of acceleration)."""
    from .units import to_case
    try:
        g = to_case(STANDARD_GRAVITY, "acceleration", units or {})
    except Exception:
        g = None
    return STANDARD_GRAVITY if not g else float(g)


def member_line_mass(loads: Iterable[dict], region: Callable, nodes: np.ndarray, members: np.ndarray,
                     member_ok: Optional[np.ndarray] = None, units=None) -> np.ndarray:
    """(n_members,): the mass each END of each member carries from the
    model-level `line_load`s with `with_mass`, |q| L / (2 g). A member that
    splits off takes it with it (`move_line_mass`), as it takes the load."""
    per_end = np.zeros(len(members))
    chosen_loads = [ld for ld in loads or [] if ld.get("with_mass") and ld.get("line_load") is not None]
    if not chosen_loads:
        return per_end
    nn = len(nodes)
    g = line_load_g(units)
    for ld in chosen_loads:
        idx = np.asarray(region(ld.get("region", "all")))
        inside = np.zeros(nn, bool); inside[idx] = True
        sel = inside[members].all(axis=1)
        if member_ok is not None:
            sel &= member_ok
        q = float(np.linalg.norm(np.asarray(ld["line_load"], float)))
        e = np.flatnonzero(sel)
        L = np.linalg.norm(nodes[members[e, 1]] - nodes[members[e, 0]], axis=1)
        per_end[e] += 0.5 * q * L / g
    return per_end


def line_loads(loads: Iterable[dict], region: Callable, nodes: np.ndarray, members: np.ndarray,
               member_ok: Optional[np.ndarray] = None, units=None) -> np.ndarray:
    """(nn, 3): the mass of every model-level `line_load` with `with_mass`,
    |q| L / g per member, half to each end node, every translation."""
    mass = np.zeros((len(nodes), 3))
    per_end = member_line_mass(loads, region, nodes, members, member_ok, units)
    np.add.at(mass, members[:, 0], per_end[:, None])
    np.add.at(mass, members[:, 1], per_end[:, None])
    return mass


def move_line_mass(solver, e: int, parent: int, new: int) -> float:
    """The end of member `e` that left particle `parent` for `new` takes its
    share of the `with_mass` line mass, weightless as it was (the load it is
    the mass of goes with the member). Returns the share; the caller moves it
    in `physical_mass` with the member's own."""
    per_end = getattr(solver, "line_mass_per_end", None)
    if per_end is None or e >= len(per_end) or per_end[e] <= 0.0:
        return 0.0
    m = float(per_end[e])
    wl = getattr(solver, "weightless_mass", None)
    if wl is not None:
        if new >= len(wl):
            wl = np.vstack([wl, np.zeros((new + 1 - len(wl), 3))])
        wl[parent] = np.maximum(wl[parent] - m, 0.0)
        wl[new] += m
        solver.weightless_mass = wl
    return m


def table(p: dict, region: Callable, nodes: np.ndarray, members: Optional[np.ndarray] = None,
          member_ok: Optional[np.ndarray] = None, rotations: bool = True):
    """(mass, inertia, weightless), each (nn, 3), everything added."""
    nn = len(nodes)
    mass, rot, light = nodal(p.get("nodal_masses") or [], region, nn)
    if not rotations and np.any(rot):
        raise ValueError("nodal_masses `inertia` is a rotary inertia: this model's nodes have no "
                         "rotations (a truss or a solid)")
    hyd = p.get("hydrodynamic_mass") or []
    if hyd:
        if members is None:
            raise ValueError("hydrodynamic_mass acts on line members (piers, piles); this model has none")
        hm, hl = hydrodynamic(hyd, region, nodes, members, member_ok)
        mass += hm; light += hl
    if members is not None:
        lm = line_loads(p.get("loads") or [], region, nodes, members, member_ok, p.get("units"))
        mass += lm; light += lm
    return mass, rot, light


def weighed(translational: np.ndarray, weightless: Optional[np.ndarray]) -> np.ndarray:
    """The translational mass gravity sees, (n, 3): all of it less the
    weightless part. Particles added by a split carry no added mass."""
    if weightless is None or not np.any(weightless):
        return translational
    out = np.array(translational, float, copy=True)
    k = min(len(out), len(weightless))
    out[:k] -= weightless[:k]
    return np.maximum(out, 0.0)
