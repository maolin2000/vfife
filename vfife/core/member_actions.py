"""Signed member forces from member end actions.

ONE SIGN CONVENTION for every line family (beam2, beam2_timoshenko,
mixed_line and the members of shell_line and coupled models, beam2_fiber,
truss2): the element fields `axial_force`, `shear_y`, `shear_z`, `torsion`,
`bending_y`, `bending_z` are the INTERNAL forces at a cross-section, in the
member's local axes (x from node i to node j, y the assignment's
`orientation`, z = x cross y), acting on the face whose outward normal is
+x -- the face of the part of the member on the node-i side:

* `axial_force` positive in TENSION;
* `shear_y`, `shear_z` positive along +y, +z on that face;
* `torsion`, `bending_y`, `bending_z` positive by the right-hand rule about
  +x, +y, +z on that face, so `bending_y` = integral of sigma z dA and
  `bending_z` = -integral of sigma y dA (for a girder with local z up, a
  SAGGING moment is a NEGATIVE `bending_y`).

The end actions `*_i`, `*_j` are something else: the forces the nodes exert
ON THE MEMBER, in the same local axes (f = k d). By equilibrium of a short
piece next to each end the internal force there is -f_i at end i and +f_j at
end j. A member reports the end with the larger magnitude, with its sign;
`*_max_abs` keeps the envelope max(|f_i|, |f_j|) the plain names held until
v1.0. A fibre member reports its section resultants at the integration
station with the largest magnitude (the same convention), not its end
actions.
"""
from __future__ import annotations

import numpy as np

ACTIONS = ("axial_force", "shear_y", "shear_z", "torsion", "bending_y", "bending_z")


def from_end_actions(ends: np.ndarray):
    """(signed, max_abs) member forces, each (nel, 6), from (nel, 12) end actions."""
    ends = np.asarray(ends, dtype=float)
    s_i = -ends[:, :6]
    s_j = ends[:, 6:]
    # a tie (an antisymmetric moment, +M at one end and -M at the other) goes
    # to end i unless end j is larger beyond round-off: the sign does not
    # depend on the backend's last bit
    signed = np.where(np.abs(s_j) > np.abs(s_i) * (1.0 + 1e-9), s_j, s_i)
    return signed, np.maximum(np.abs(s_i), np.abs(s_j))


def write(out: dict, signed: np.ndarray, max_abs: np.ndarray, ends: np.ndarray,
          failed=None) -> dict:
    """The six signed fields, their `*_max_abs` envelopes and the end actions
    into `out`. A member that has failed (`failed` true) reports zero: what a
    broken-off fragment carries is not a force of the structure."""
    signed = np.array(signed, dtype=float, copy=True)
    max_abs = np.array(max_abs, dtype=float, copy=True)
    ends = np.array(ends, dtype=float, copy=True)
    if failed is not None:
        dead = np.asarray(failed, dtype=bool)
        if dead.shape == (len(signed),) and dead.any():
            signed[dead] = 0.0; max_abs[dead] = 0.0; ends[dead] = 0.0
    for a, name in enumerate(ACTIONS):
        out[name] = signed[:, a]
        out[f"{name}_max_abs"] = max_abs[:, a]
        out[f"{name}_i"] = ends[:, a]
        out[f"{name}_j"] = ends[:, a + 6]
    return out


# ------------------------------------------------ element monitors on line models
#: how the members' values are reduced: the solids' rules (core/solver.py),
#: `absmax` being the signed value of largest magnitude
REDUCE = {
    "max": lambda v: float(np.max(v)),
    "min": lambda v: float(np.min(v)),
    "mean": lambda v: float(np.mean(v)),
    "absmax": lambda v: float(v[np.argmax(np.abs(v))]),
}


def region_members(mesh, regions, elements0) -> np.ndarray:
    """The members a monitor reads: those of its regions' physical groups
    where the mesh records them, else every member whose nodes are all in
    the region -- taken on the connectivity as read (`elements0`), so a
    member that later splits onto spare particles stays in its monitor."""
    if not regions:                       # no region: every member
        return np.arange(len(elements0), dtype=np.int64)
    ids = []
    blocks = list(getattr(mesh, "cell_blocks", None) or {}) or ["line2"]
    for r in regions:
        got = None
        for b in blocks:
            got = (getattr(mesh, "element_regions", {}) or {}).get(r, {}).get(b)
            if got is not None:
                break
        if got is None:
            nodes = mesh.regions.get(r)
            if nodes is None:
                raise ValueError(f"element monitor: no region {r!r}")
            inside = np.zeros(int(max(mesh.n_nodes, np.max(elements0) + 1)), dtype=bool)
            inside[np.asarray(nodes, dtype=np.int64)] = True
            got = np.flatnonzero(inside[elements0].all(axis=1))
        ids.append(np.asarray(got, dtype=np.int64))
    out = np.unique(np.concatenate(ids)) if ids else np.zeros(0, np.int64)
    out = out[out < len(elements0)]
    if not len(out):
        raise ValueError(f"element monitor on {', '.join(regions)}: the region owns no member "
                         "(every node of a member must be in it)")
    return out


def monitor_value(fields: dict, field: str, ids: np.ndarray, reduce: str | None) -> float:
    """One element monitor's value from a frame's `nodal_fields`."""
    arr = fields.get(field)
    if arr is None:
        return float("nan")
    v = np.asarray(arr, dtype=float)
    if v.ndim != 1:
        return float("nan")
    return REDUCE[reduce or "max"](v[ids])
