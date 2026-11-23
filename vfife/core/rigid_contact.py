"""Rigid surfaces a node cannot pass through.

    contact:
      rigid:
        - {plane: {point: [0, 0, -40], normal: [0, 0, 1]}, friction: 0.3}

WHAT IT IS. A rigid, immovable surface -- the ground, the sea floor, a
pile cap, a wall -- and the force it puts on a node that has crossed it:
a penalty spring on the penetration, `k g` along the normal, and under
`friction` a Coulomb force against the node's tangential velocity. No
pairs, no master and slave, no search: every node of `region` (default:
all of them) is checked against every surface at every evaluation, which
for the few planes a bridge model needs is a handful of dot products.

WHY THE SPRING IS SIZED FROM THE NODE'S MASS. A penalty stiffness is a
numerical device with one job, to be stiff enough, and one cost, the
explicit step it demands. `stiffness: auto` takes k = m (2 / dt)^2 / 4 per
node: the contact spring's own frequency then has omega dt = 1, which with
the structure's own omega dt <= 1.2 (dt_safety 0.6) stays inside the
central-difference limit of 2 -- so the step never has to shrink for the
contact, and the penetration of a node resting on the surface under its own
weight is g dt^2 (k = m / dt^2), microns. A free particle (debris) has no
stiffness of its own to borrow a scale from, and this is what makes `auto`
work for exactly the nodes contact exists for. A number instead of `auto`
is taken as given UP TO that same value and capped there (v1.0: a larger
one was unstable and went unreported).

DAMPING. `damping: zeta` puts a dashpot 2 zeta sqrt(k m) beside the spring
while the node is inside the surface, with the total never allowed to pull:
a bounce then loses energy at each impact and debris settles instead of
bouncing for ever on a perfectly elastic floor. HOW MUCH is not
controllable: an impact is resolved in a step or two at omega dt = 1, so the
restitution depends on where in a step the node arrives -- measured 0.00 to
0.87 at zeta = 1.0 for drops a few millimetres apart, and the cap c <= m/dt
makes zeta = 1.0 act like 0.5 (this said exp(-pi zeta /
sqrt(1 - zeta^2)) before). Off by default, because a floor that dissipates
is a modelling choice.

FRICTION IS A VELOCITY LAW, so it is dynamic only: relaxation velocities
are pseudo-time and a friction force built on them would be noise. The
Coulomb force is regularised, F_t = -mu F_n v_t / max(|v_t|, v_reg), with
v_reg chosen per node and step so the regularisation never acts as a
dashpot stiffer than the explicit step allows (c dt / m <= 1). A node
truly at rest on a rough surface therefore creeps at v_reg under a
tangential load below mu F_n -- of the order of 0.1 mm/s (87 um/s measured
at half of mu W) -- rather than sticking
exactly; a bearing that must stick uses a connector `friction` law
(elements/connector.py), which carries the slip as history.

WHAT IT IS NOT. Not contact between members (that is `contact.members`,
core/member_contact.py, since 1.11), not a deformable surface, not a
moving one. The surface knows nothing of the member's section: a beam
node stops at the surface, the box around it does not. Initial
penetration is refused by name rather than pushed out on the first step.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class RigidContact:
    def __init__(self, spec: Dict[str, Any], mesh, regions: Dict[str, np.ndarray], mode: str,
                 tol: float | None = None):
        self.surfaces: List[Dict[str, Any]] = []
        self.mode = mode
        nn = len(mesh.nodes)
        diag = float(np.linalg.norm(np.ptp(mesh.nodes, axis=0))) or 1.0
        tol = 1e-9 * diag if tol is None else tol
        for k, s in enumerate(spec.get("rigid") or []):
            p = np.asarray(s["plane"]["point"], float)
            n = np.asarray(s["plane"]["normal"], float)
            n = n / np.linalg.norm(n)
            name = s.get("name") or f"rigid[{k}]"
            region = s.get("region", "all")
            if region == "all":
                ids = np.arange(nn, dtype=np.int64)
            elif region in regions:
                ids = np.asarray(regions[region], dtype=np.int64)
            else:
                raise ValueError(f"contact {name}: unknown region {region!r}. Available: "
                                 + ", ".join(sorted(regions)))
            mu = float(s.get("friction") or 0.0)
            if mu > 0.0 and mode != "dynamic":
                raise ValueError(f"contact {name}: friction is a velocity law and needs "
                                 "solver.mode = dynamic; drop `friction` for a relaxation run")
            g0 = (mesh.nodes[ids] - p) @ n
            inside = ids[g0 < -tol]
            if len(inside):
                worst = ids[np.argmin(g0)]
                raise ValueError(
                    f"contact {name}: {len(inside)} node(s) of {region!r} start behind the "
                    f"surface (node {int(worst)} by {-float(g0.min()):.3g} m); move the surface "
                    "or the mesh -- a contact does not push a model into shape")
            zeta = float(s.get("damping") or 0.0)
            if zeta > 0.0 and mode != "dynamic":
                raise ValueError(f"contact {name}: damping is a velocity law and needs solver.mode = dynamic")
            self.surfaces.append({"name": name, "point": p, "normal": n, "ids": ids,
                                  "all": region == "all",
                                  "stiffness": s.get("stiffness", "auto"), "mu": mu, "zeta": zeta})
        self._bonds = None          # the solver's BondGraph, set by `attach`
        self._seen_splits = 0
        self.stiffness_capped = False
        self.nodal_force = np.zeros((nn, 3))
        self.penetration = np.zeros(nn)
        self.n_active = 0
        self.max_penetration = 0.0

    # ----------------------------------------------------------------- force
    def force(self, x: np.ndarray, v: np.ndarray | None, node_mass: np.ndarray, dt: float) -> np.ndarray:
        """The contact force on every node, (nn, 3), for current positions
        `x`, velocities `v` (None outside dynamics) and translational node
        masses `node_mass` (nn,)."""
        self._follow_splits()
        f = np.zeros_like(x)
        pen = np.zeros(len(x))
        active = 0
        for s in self.surfaces:
            ids = s["ids"]
            g = (x[ids] - s["point"]) @ s["normal"]
            hit = g < 0.0
            if not hit.any():
                continue
            i = ids[hit]; gi = g[hit]
            m = np.maximum(node_mass[i], 1e-30)
            k_max = m * (2.0 / dt) ** 2 / 4.0
            if s["stiffness"] == "auto":
                k = k_max
            else:
                # A NUMBER IS CAPPED at what the explicit step can carry,
                # omega dt = 1 as `auto` takes: past omega dt = 2 the spring is
                # unstable, and a beam resting on a 5e10 N/m floor was thrown
                # 1.4 mm into the air while the run reported success.
                k_user = float(s["stiffness"])
                k = np.minimum(k_user, k_max)
                if np.any(k < k_user):
                    self.stiffness_capped = True
            fn = -k * gi                                   # > 0, pushes the node out
            if s["zeta"] > 0.0 and v is not None:
                # a dashpot beside the spring while the node is inside, 2 zeta
                # sqrt(k m), capped at m / dt so it can never outrun the explicit
                # step; the sum is never allowed to pull (no adhesion), which is
                # what lets a bounce die out instead of the node being held
                c = np.minimum(2.0 * s["zeta"] * np.sqrt(k * m), m / dt)
                fn = np.maximum(fn - c * (v[i] @ s["normal"]), 0.0)
            f[i] += fn[:, None] * s["normal"]
            pen[i] = np.maximum(pen[i], -gi)
            active += int(hit.sum())
            if s["mu"] > 0.0 and v is not None:
                vt = v[i] - np.outer(v[i] @ s["normal"], s["normal"])
                speed = np.linalg.norm(vt, axis=1)
                # the regularisation velocity: the analyst's floor, or what
                # keeps the equivalent dashpot mu F_n / v_reg inside c dt/m <= 1
                v_reg = np.maximum(1e-6, s["mu"] * fn * dt / m)
                scale = s["mu"] * fn / np.maximum(speed, v_reg)
                f[i] -= scale[:, None] * vt
        self.nodal_force = f
        self.penetration = pen
        self.n_active = active
        self.max_penetration = float(pen.max()) if len(pen) else 0.0
        return f

    def _follow_splits(self) -> None:
        """A particle split off a node of a surface's region is in that
        region too: debris of a member cut from `deck`
        fell through a ground restricted to `deck` to z = -2.05 m, because
        the ids were fixed at setup and the spares belong to no region."""
        b = self._bonds
        if b is None or len(b.parents) == self._seen_splits:
            return
        new = range(self._seen_splits, len(b.parents))
        self._seen_splits = len(b.parents)
        for s in self.surfaces:
            if s.get("all"):
                continue
            ids = set(int(i) for i in s["ids"])
            add = [int(b.bonds[b.made_for[j]]) for j in new if int(b.parents[j]) in ids]
            if add:
                s["ids"] = np.unique(np.r_[s["ids"], np.asarray(add, dtype=np.int64)])

    # ---------------------------------------------------------------- attach
    def attach(self, solver) -> None:
        """Attach to the solver as a component (core/components): the contact
        force after the members' (solvers on dof vectors) or the bars' (the
        truss solver), and the contact fields; the continuum solver calls
        `contact.force(...)` from its loop itself (wrappers
        put on the solver's methods until then)."""
        if getattr(solver, "resident", False):
            raise ValueError("contact is not on the device-resident (warp) path; use backend numpy or numba")
        solver.contact = self
        self._bonds = getattr(solver, "bonds", None)
        solver.attach_component(RigidContactComponent(self, solver))

    def summary(self) -> str:
        return "; ".join(f"{s['name']}: {len(s['ids'])} nodes, stiffness {s['stiffness']}, "
                        f"friction {s['mu']}, damping {s['zeta']}" for s in self.surfaces)



class RigidContactComponent:
    """Contact with rigid surfaces as a solver component (core/components):
    what the wrappers of v0.5 did, in the same order."""

    def __init__(self, contact: RigidContact, solver):
        self.contact = contact
        self._nodes = solver.mesh.nodes
        on_dofs = hasattr(solver, "_member_forces") and hasattr(solver, "dofs")
        on_bars = (not on_dofs and hasattr(solver, "_bar_forces") and getattr(solver, "u", None) is not None
                   and np.ndim(solver.u) == 2 and not hasattr(solver, "blocks"))
        # the continuum solver applies the force in its loop: this component
        # only adds the fields there
        if not on_dofs:
            self.line_force = None
        if not on_bars:
            self.bar_force = None

    @staticmethod
    def _node_mass(solver):
        # the SMALLEST translational mass of the node: a relaxation run's
        # fictitious masses differ by direction, and the spring must be
        # stable along the normal whichever direction that is
        m = np.asarray(solver.mass, float)
        if m.ndim == 2:                       # (nn, 3): truss, solid
            return m.min(axis=1)
        stride = solver.dofs.stride
        return m.reshape(-1, stride)[:, :3].min(axis=1)

    def line_force(self, solver, out):
        stride = solver.dofs.stride
        q = solver.q.reshape(-1, stride)
        x = self._nodes + q[:, :3]
        v = solver.v.reshape(-1, stride)[:, :3] if solver.mode == "dynamic" else None
        f = self.contact.force(x, v, self._node_mass(solver), solver.dt)
        out = out.reshape(-1, stride)
        out[:, :3] -= f
        return out.reshape(-1)

    def bar_force(self, solver, f_int):
        v = solver.v if solver.mode == "dynamic" else None
        return f_int - self.contact.force(self._nodes + solver.u, v, self._node_mass(solver), solver.dt)

    def fields(self, out):
        out["contact_force"] = self.contact.nodal_force.copy()
        out["contact_penetration"] = self.contact.penetration.copy()

def attach_contact(solver, problem, mesh) -> RigidContact | None:
    """Build the contact from the case and attach it; None when the case has
    none. Called by the solver factories after the solver exists, because the
    regions it names are resolved by the solver's own setup."""
    spec = problem.get("contact")
    if not spec or not spec.get("rigid"):
        return None
    contact = RigidContact(spec, mesh, solver.mesh.regions, solver.mode)
    contact.attach(solver)
    return contact
