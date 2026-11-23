"""Contact between line members (1.11, route item 26).

    contact:
      members:
        bodies:
          - {region: upper_deck, radius: 1.2}
          - {region: lower_deck, radius: 1.0}
          - {region: hangers}                  # bars: radius from the area
        friction: 0.4
        damping: 0.3

WHAT IT IS. A line member has no volume; in contact it is a CAPSULE -- the
set of points within `radius` of its axis, the straight segment between its
two points. Two members touch when their axes come closer than the sum of
their radii, and then push each other apart along the common normal of the
two axes with a penalty force on the overlap, plus, in a dynamic step, a
dashpot (`damping`) and Coulomb friction against the relative velocity of
the two surface points. Every member of every body is tested against every
other, its own body's included (a deck folding onto itself), except the
members it shares a point with.

A falling span landing on the deck below, a girder end striking its
neighbour, a broken hanger whipping across the others, the pile a collapse
leaves: what `contact.rigid` (a fixed plane) could not do, because there the
other side does not move.

THE FORCE GOES TO THE MEMBERS' POINTS with the weights of the closest
point, (1 - s, s): statically equivalent to the force at the contact point,
force and moment. Friction acts at the CONTACT POINT, the middle of the
overlap, off each axis, so it also twists the members: that couple goes to the points'
rotations (a rolling log slows down; without it friction would act through
the axis and nothing would ever roll).

THE SPRING IS SIZED FROM THE MASSES, as the rigid surface's is: k = m / dt^2
with m the reduced mass of the two members' lighter points, so its own
frequency has omega dt = 1 and the explicit step never shrinks for it --
divided by the number of contacts the busiest of the four points is in, so
that a pile of members on one point stays inside the limit too (a quarter
of that outside a dynamic step: a relaxation's kinetic damping and a spring
at omega dt = 1 lock into a cycle). A point held in every step counts as an
infinite mass: what lands on a fixed member is stopped by its own mass. A
number is capped there. The overlap a member resting on another under its
weight settles at is W / k, of the order of g dt^2 times the ratio of the
mass carried to the contact points' own: tens of microns.

WHAT OVERLAPS AT THE START IS LET BE. A girder sits 30 mm above its pier
cap on a bearing; with honest radii the two capsules overlap by a metre
before anything has happened. Refusing that, as the rigid surface refuses a
node behind it, would refuse every real bridge; pushing it out would launch
the deck. Each pair that overlaps in the reference configuration gets an
ALLOWANCE equal to its overlap plus a SLACK, `slack` times the smaller of
the two radii (5% by default): the contact acts on what exceeds it. The
slack is what the bearing between them needs -- without it the contact
spring, far stiffer than any bearing, took the dead load off the bearing
at the first tenth of a millimetre of its compression, and a friction
bearing lost the normal force it slides on. The allowance only ever
shrinks -- when the pair moves apart it follows, never more than the slack
ahead, and once they are more than the slack apart it is gone and the pair is an ordinary
one -- so a deck whose bearing fails drops by the slack and lands on the
cap from where it stood. A member
that breaks off its neighbour is the same case (the two were one point a
step ago): the pair gets its allowance at the step it comes into being.
The allowances are state: they are in the checkpoint.

HOW FAST IS TOO FAST. The overlap is what the force is built from, so two
members must not pass more than about half the sum of their radii in one
step: closing speed x dt < 0.5 (r1 + r2). A girder and a deck are nowhere
near it; a thin cable under mass scaling can be (r = 0.07 m and dt = 1e-3 s
give 70 m/s). Past it the pair tunnels, and comes out faster than it went in.

WHAT IT IS NOT. Not for `mode: static` (refused: the Newton tangent's
colouring does not know the pairs, and a penalty has a kink where it
engages) -- a relaxation does take it. Not a section's shape: a box girder twelve metres wide and
two deep is a capsule of the radius the case writes, and where its edge
really is is not modelled (a rectangular box is the natural next shape; the
kernel's pair list does not care). Not contact with a solid mesh. Not a
debris particle's (a parked particle has no member; `contact.rigid` still
stops it at the ground). And friction is a velocity law, regularised as the
rigid surface's: a member at rest on a slope creeps at the regularisation
velocity rather than sticking.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..kernels.numba_member_contact import njit                                           # noqa: F401
from ..kernels.numba_member_contact import (_closest, _find, _forces, _narrow, _moved2,   # noqa: F401
                                            _keep_list, contact_step)


class MemberContact:
    def __init__(self, spec: Dict[str, Any], solver):
        self.solver = solver
        self.mu = float(spec.get("friction") or 0.0)
        self.zeta = float(spec.get("damping") or 0.0)
        st = spec.get("stiffness", "auto")
        self.k_user = 0.0 if st in (None, "auto") else float(st)
        nel = int(solver.nel)
        radius = np.zeros(nel)
        body = np.full(nel, -1, np.int64)
        kinds = getattr(solver, "member_kind", None)
        connectors = set(int(e) for e in getattr(solver, "_connector_ids", ()))
        self.names = []
        for b, entry in enumerate(spec.get("bodies") or []):
            region = entry.get("region", "all")
            members = self._region_members(region)
            members = np.asarray([e for e in members if int(e) not in connectors
                                  and not (kinds is not None and str(kinds[e]) == "connector2")], np.int64)
            if not len(members):
                raise ValueError(f"contact.members.bodies[{b}]: region {region!r} holds no complete member "
                                 "(a member is in a region when both its points are; connectors take no part)")
            again = members[body[members] >= 0]
            if len(again):
                raise ValueError(f"contact.members.bodies[{b}]: {len(again)} member(s) of {region!r} are in "
                                 f"body {int(body[again[0]])} already (member {int(again[0])}); a member has "
                                 "one radius")
            r = entry.get("radius", "auto")
            if r in (None, "auto"):
                radius[members] = [self._bar_radius(int(e), region, b) for e in members]
            else:
                radius[members] = float(r)
            body[members] = b
            self.names.append(str(region))
        self.radius = radius
        self._rad_of = radius
        self.slack = float(spec.get("slack", 0.05) if spec.get("slack") is not None else 0.05)
        self.body = body
        self.nel = nel
        self.keys = np.zeros(0, np.int64)
        self.vals = np.zeros(0)
        self._conn_seen = None
        self._pending = None
        self._restored = False
        self._cap = 64
        self._list_ids = None
        # the neighbour list, in the arrays `contact_step` keeps it in
        self._ci = np.zeros(256, np.int64); self._cj = np.zeros(256, np.int64)
        self._lstate = np.array([0, 0, 1], np.int64)          # pairs, sweeps, make-it-again
        self._ref = np.zeros((solver.nn, 3))
        self._stats = np.zeros(4)
        self.n_active = 0
        self.max_penetration = 0.0
        self.stiffness_capped = False
        self.initial_pairs = 0
        self._f = np.zeros((solver.nn, 3)); self._mom = np.zeros((solver.nn, 3)); self._pen = np.zeros(solver.nn)
        self.n_sweeps = 0
        self.nodal_force = np.zeros((solver.nn, 3))
        self.nodal_moment = np.zeros((solver.nn, 3))
        self.penetration = np.zeros(solver.nn)

    # ----------------------------------------------------------------- setup
    def _region_members(self, region):
        """The members of a region: the groups the mesh RECORDED for it (a
        physical group of lines, a `lines` group -- what `assign` means by
        the name), else the members with both points in it. By the points
        alone `hangers` also held the main cable's segments between two
        hanger tops (found in review: 62 members for 32 hangers)."""
        s = self.solver
        mesh = s.mesh
        if region == "all" and "all" not in getattr(mesh, "regions", {}):
            return np.arange(self.nel if hasattr(self, "nel") else int(s.nel), dtype=np.int64)
        rec = getattr(mesh, "element_regions", {}).get(region) if hasattr(mesh, "element_regions") else None
        if rec:
            ids = np.unique(np.concatenate([np.asarray(v, np.int64).ravel() for v in rec.values()]))
            return ids[ids < int(s.nel)]
        return self._members_in(np.asarray(s._region(region), np.int64))

    def _members_in(self, idx):
        s = self.solver
        if hasattr(s, "_members_in"):
            return np.asarray(s._members_in(idx), np.int64)
        sel = np.zeros(s.nn, bool); sel[idx] = True
        return np.flatnonzero(sel[np.asarray(s.mesh.elements)[:, :2]].all(axis=1))

    def _bar_radius(self, e, region, b):
        s = self.solver
        area = None
        props = getattr(s, "member_props", None)
        kinds = getattr(s, "member_kind", None)
        if props is not None and kinds is not None and str(kinds[e]) == "truss2":
            area = props[e].get("area")
        elif props is None and hasattr(getattr(s, "pre", None), "area"):       # the bar solver
            a = np.asarray(s.pre.area, float)
            area = float(a[e]) if a.ndim else float(a)
        if area is None or not float(area) > 0.0:
            raise ValueError(f"contact.members.bodies[{b}] ({region!r}) needs `radius`: `auto` is for bars and "
                             "cables alone (the circle of their area); a beam's section does not say "
                             "how far its surface is from its axis in every direction")
        return float(np.sqrt(float(area) / np.pi))

    def _active(self):
        s = self.solver
        on = self.body >= 0
        for name in ("member_active", "_active"):
            a = getattr(s, name, None)
            if a is not None and np.shape(a) == (self.nel,):
                on = on & np.asarray(a, bool)
                break
        return np.flatnonzero(on)

    # ----------------------------------------------------------------- force
    def _pairs(self, x, conn, rad):
        while True:
            n, pi, pj, ps, pt, po = _find(x, conn, rad, self._cap)
            if n <= self._cap:
                return n, pi, pj, ps, pt, po
            self._cap = int(2 * n)

    def _grant(self, x, conn, rad, gids, only=None, shared=None):
        """Allowances for the pairs that overlap now or are within the slack
        of touching, where they have none: the present overlap plus the
        slack. All of them (the first evaluation); those with a member in
        `only` (members born in a later step); or, with `shared` (the
        connectivity before a break), those that SHARED A POINT then and do
        not now -- the two sides of the break, and only they: the broken
        member's other contacts are contacts (found in review: every pair of
        a member that split got one, and a beam resting on another sank a
        slack into it when the member under it broke off).

        WITHIN THE SLACK, not only overlapping: two members of one girder two
        elements apart, each as long as the girder is deep, touch exactly,
        and the girder's own strain under its dead load, 1e-12 m, was a
        contact."""
        grow = 1.0 + 0.5 * self.slack + 1e-9
        n, pi, pj, _ps, _pt, po = self._pairs(x, conn, rad * grow)
        if not n:
            return 0
        g1 = gids[pi[:n]]; g2 = gids[pj[:n]]
        r1 = self._rad_of[g1]; r2 = self._rad_of[g2]
        rmin = np.minimum(r1, r2)
        allowance = (po[:n] - (grow - 1.0) * (r1 + r2)) + self.slack * rmin
        keys = np.minimum(g1, g2) * self.nel + np.maximum(g1, g2)
        take = allowance > 0.0
        if only is not None:
            take &= np.isin(g1, only) | np.isin(g2, only)
        if shared is not None:
            ok = (g1 < len(shared)) & (g2 < len(shared))
            a = shared[np.where(ok, g1, 0)]; b = shared[np.where(ok, g2, 0)]
            was = ok & ((a[:, 0] == b[:, 0]) | (a[:, 0] == b[:, 1]) | (a[:, 1] == b[:, 0]) | (a[:, 1] == b[:, 1]))
            take &= was
        take &= ~np.isin(keys, self.keys[self.vals > 0.0])
        if not take.any():
            return 0
        live = self.vals > 0.0
        keys_all = np.r_[self.keys[live], keys[take]]
        vals_all = np.r_[self.vals[live], allowance[take]]
        order = np.argsort(keys_all, kind="mergesort")
        keys_all = keys_all[order]; vals_all = vals_all[order]
        first = np.r_[True, keys_all[1:] != keys_all[:-1]]
        self.keys = np.ascontiguousarray(keys_all[first]); self.vals = np.ascontiguousarray(vals_all[first])
        self._pending = None
        return int(take.sum())

    def force(self, x, v, w, node_mass, node_inertia, dt, dynamic, commit=True):
        """(f, mom), each (nn, 3): the contact forces and the friction's
        couples on every point, for positions `x`, velocities `v`, spins `w`
        (zeros outside a dynamic step), translational point masses (above
        1e290: held) and rotary inertias (0: none).

        `commit` False is a QUESTION, not a step (a probe, a trial state, a
        Newton iterate, an iteration of a relaxation): the allowances are
        read and the force is the one they give, but they are not written.
        One probe with the girder lifted 0.3 m cut its allowance by 0.3 m,
        and the force at the state it came back to was 4e11 N (found in
        review). A relaxation's are written when its level is committed."""
        ids, conn, rad, gids = self.prepare(x)
        x = np.ascontiguousarray(x, dtype=float)
        if commit:
            # a step's evaluation writes the allowances itself: what a
            # question before it arrived at is not to be written after it
            self._pending = None
        vals = self.vals if commit else self.vals.copy()
        rad_s = rad * (1.0 + 0.5 * self.slack + 1e-9)
        args = (x, np.ascontiguousarray(v, dtype=float), np.ascontiguousarray(w, dtype=float), conn, rad, rad_s,
                gids, self.nel, np.ascontiguousarray(node_mass, dtype=float),
                np.ascontiguousarray(node_inertia, dtype=float), float(dt), self.k_user, self.mu, self.zeta,
                bool(dynamic), self.keys, vals, self.slack)
        while True:
            status = contact_step(*args, self._ci, self._cj, self._lstate, self._ref, self._f, self._mom,
                                  self._pen, self._stats)
            if status == 0:
                break
            self._ci = np.zeros(2 * int(-status), np.int64); self._cj = np.zeros(2 * int(-status), np.int64)
        if not commit:
            self._pending = (self.keys, vals)
        self.book()
        return self.nodal_force, self.nodal_moment

    def prepare(self, x):
        """What an evaluation needs before the kernel, and the events only
        Python sees: (active member ids, their connectivity, radii, ids
        again as the kernel wants them), after giving allowances to WHO IS
        NEW -- at the first evaluation every pair; later the pairs of a
        member that has just come into being (an analysis step's
        `activate`), and the two sides of a break -- and asking for a new
        neighbour list when the members have changed."""
        s = self.solver
        ids = self._active()
        elements = np.asarray(s.mesh.elements)
        conn = np.ascontiguousarray(elements[ids][:, :2], dtype=np.int64)
        rad = np.ascontiguousarray(self.radius[ids])
        gids = np.ascontiguousarray(ids, dtype=np.int64)
        x = np.ascontiguousarray(x, dtype=float)
        seen = self._conn_seen
        changed_conn = seen is None or seen.shape != elements[:, :2].shape \
            or not np.array_equal(seen, elements[:, :2])
        new_ids = (self._list_ids is None or len(self._list_ids) != len(ids)
                   or not np.array_equal(self._list_ids, ids))
        if seen is None:
            self.initial_pairs = self._grant(x, conn, rad, gids)
            if self.initial_pairs and hasattr(s, "log"):
                s.log(f"  contact       {self.initial_pairs} pair(s) of members overlap or touch at the start "
                      "(bearings, joints, neighbours closer than their radii): each keeps its overlap, and "
                      f"a slack of {self.slack:g} of the smaller radius, as an allowance until it has separated")
        else:
            if new_ids and self._list_ids is not None:
                born = np.setdiff1d(ids, self._list_ids)
                if len(born):
                    self._grant(x, conn, rad, gids, only=born)
            if changed_conn:
                self._grant(x, conn, rad, gids, shared=seen)
        if changed_conn:
            self._conn_seen = np.array(elements[:, :2], copy=True)
        if new_ids:
            self._list_ids = ids.copy()
        if self._f.shape != x.shape:
            nn = x.shape[0]
            self._f = np.zeros((nn, 3)); self._mom = np.zeros((nn, 3)); self._pen = np.zeros(nn)
        if new_ids or changed_conn or self._ref.shape != x.shape:
            if self._ref.shape != x.shape:
                self._ref = np.zeros(x.shape)
            self._lstate[2] = 1
        return ids, conn, rad, gids

    def chunk_args(self):
        """What the compiled step (kernels/numba_step) hands
        `contact_step` in a run of steps, at the solver's state now: the
        events of `prepare` seen, the masses as an evaluation reads them.
        The arrays are this object's own -- the allowances, the neighbour
        list, the forces of the last evaluation -- and the kernel writes
        them as `force` does; `booked` says so afterwards."""
        s = self.solver
        x = s.mesh.nodes + s.q.reshape(-1, s.dofs.stride)[:, :3]
        _ids, conn, rad, gids = self.prepare(x)
        mt, rot = self.masses()
        return (True, conn, rad, rad * (1.0 + 0.5 * self.slack + 1e-9), gids, self.nel,
                np.ascontiguousarray(mt, dtype=float), np.ascontiguousarray(rot, dtype=float),
                self.k_user, self.mu, self.zeta, self.keys, self.vals, self.slack,
                self._ci, self._cj, self._lstate, self._ref, self._f, self._mom, self._pen, self._stats)

    def booked(self, evaluations: int) -> None:
        """After a compiled run that evaluated the contact `evaluations`
        times: what `force(commit=True)` leaves."""
        if evaluations > 0:
            self._pending = None
            self.book()

    def book(self) -> None:
        """What an evaluation leaves for the output and the summary, from
        the arrays the kernel wrote."""
        self.n_active = int(self._stats[0]); self.max_penetration = float(self._stats[1])
        self.stiffness_capped = self.stiffness_capped or bool(self._stats[2])
        self.n_sweeps = int(self._lstate[1])
        self.nodal_force = self._f.copy(); self.nodal_moment = self._mom.copy(); self.penetration = self._pen.copy()

    def commit(self) -> None:
        """Write the allowances the last uncommitted evaluation arrived at:
        called when a relaxation's level is committed."""
        p = self._pending
        if p is not None and p[0] is self.keys:
            self.vals = p[1]
        self._pending = None

    def masses(self):
        """(translational mass, rotary inertia) of every point: the
        smallest of each three; a point held in all three translations in
        every step has an infinite mass, one with no rotary inertia of
        its own (a bar's) has none."""
        m = np.asarray(self.solver.mass, float)
        if m.ndim == 2:
            mt = m.min(axis=1).copy(); rot = np.zeros(len(mt))
            fixed = getattr(self.solver, "fixed", None)
            if fixed is not None and np.shape(fixed) == m.shape:
                mt[np.asarray(fixed, bool).all(axis=1)] = np.inf
            return mt, rot
        stride = self.solver.dofs.stride
        mm = m.reshape(-1, stride)
        mt = mm[:, :3].min(axis=1).copy()
        rot = mm[:, 3:6].min(axis=1).copy() if stride >= 6 else np.zeros(len(mt))
        # the floors the solvers put under a point that has no rotary
        # inertia of its own are of the order of 1e-20
        scale = float(rot.max()) if len(rot) else 0.0
        rot[rot <= 1e-12 * scale] = 0.0
        if hasattr(self.solver, "_held_dofs"):
            held = np.asarray(self.solver._held_dofs(), bool).reshape(-1, stride)
            mt[held[:, :3].all(axis=1)] = np.inf
            if stride >= 6:
                rot[held[:, 3:6].all(axis=1)] = np.inf
        return mt, rot

    # ---------------------------------------------------------------- attach
    def attach(self) -> None:
        """Attach to the solver as a component (core/components): the contact
        force after the members', the allowances committed after the
        solver's history, the contact fields, and the allowances in the
        checkpoint (they were wrappers put on the solver's
        methods until then)."""
        solver = self.solver
        if hasattr(solver, "_member_forces") and hasattr(solver, "dofs"):
            pass
        elif hasattr(solver, "_bar_forces") and getattr(solver, "u", None) is not None \
                and np.ndim(solver.u) == 2 and not hasattr(solver, "blocks"):
            pass
        else:
            raise ValueError("contact.members acts between line members (beams, fibre beams, bars, cables); "
                             f"a {type(solver).__name__} model has none")
        solver.member_contact = self
        solver.attach_component(MemberContactComponent(self))

    def summary(self) -> str:
        n = int((self.body >= 0).sum())
        return (f"{n} members in {len(self.names)} bod{'y' if len(self.names) == 1 else 'ies'}, "
                f"friction {self.mu}, damping {self.zeta}, "
                f"stiffness {'auto' if self.k_user == 0.0 else self.k_user}")




class MemberContactComponent:
    """The contact between members as a solver component (core/components).
    Each hook is what the wrapper of the same place did before v1.18, in
    the same order, so a run gives the same answer to the bit."""

    def __init__(self, contact: MemberContact):
        self.contact = contact
        self._nodes = contact.solver.mesh.nodes
        self._zeros = {}

    def _zero(self, nn):
        z = self._zeros.get(nn)
        if z is None:
            z = self._zeros[nn] = np.zeros((nn, 3))
        return z

    def _asking(self, solver):
        # a probe of the force (a tangent, a trial) commits nothing
        return bool(getattr(solver, "_probing", False) or getattr(solver, "_trial_mode", False))

    def line_force(self, solver, out):
        contact = self.contact
        stride = solver.dofs.stride
        q = solver.q.reshape(-1, stride)
        x = self._nodes + q[:, :3]
        dyn = solver.mode == "dynamic"
        vv = solver.v.reshape(-1, stride)
        v = vv[:, :3] if dyn else self._zero(len(x))
        w = vv[:, 3:6] if dyn and stride >= 6 else self._zero(len(x))
        mt, rot = contact.masses()
        f, mom = contact.force(x, v, w, mt, rot, solver.dt, dyn, commit=dyn and not self._asking(solver))
        out = out.reshape(-1, stride)
        out[:, :3] -= f
        if stride >= 6:
            out[:, 3:6] -= mom
        return out.reshape(-1)

    def bar_force(self, solver, f_int):
        contact = self.contact
        dyn = solver.mode == "dynamic"
        x = self._nodes + solver.u
        mt, rot = contact.masses()
        f, _mom = contact.force(x, solver.v if dyn else self._zero(len(x)), self._zero(len(x)), mt, rot,
                                solver.dt, dyn, commit=dyn and not self._asking(solver))
        return f_int - f

    def commit(self):
        self.contact.commit()

    def fields(self, out):
        contact = self.contact
        prev = out.get("contact_force")
        out["contact_force"] = contact.nodal_force.copy() if prev is None else prev + contact.nodal_force
        prevp = out.get("contact_penetration")
        out["contact_penetration"] = (contact.penetration.copy() if prevp is None
                                      else np.maximum(prevp, contact.penetration))

    # the allowances are state: a restart that looked at the pairs again
    # would let be whatever is in contact at that moment
    def checkpoint(self, ck):
        contact = self.contact
        live = contact.vals > 0.0
        ck["member_contact_keys"] = contact.keys[live].copy()
        ck["member_contact_allowance"] = contact.vals[live].copy()
        ck["member_contact_conn"] = (np.zeros((0, 2), np.int64) if contact._conn_seen is None
                                     else np.asarray(contact._conn_seen, np.int64))
        ck["member_contact_active"] = (np.zeros(0, np.int64) if contact._list_ids is None
                                       else np.asarray(contact._list_ids, np.int64))

    def restore(self, ck):
        contact = self.contact
        if "member_contact_keys" in ck:
            keys = np.asarray(ck["member_contact_keys"], np.int64).ravel()
            if len(keys) and int(keys.max()) >= contact.nel * contact.nel:
                raise ValueError("the checkpoint's contact allowances are another model's "
                                 "(a pair of members this model does not have)")
            contact.keys = np.ascontiguousarray(keys)
            contact.vals = np.ascontiguousarray(np.asarray(ck["member_contact_allowance"], float).ravel())
            seen = np.asarray(ck["member_contact_conn"], np.int64)
            contact._conn_seen = seen.reshape(-1, 2) if seen.size else None
            act = ck.get("member_contact_active")
            # the neighbour list is made again; who was active is what
            # says who is born later
            contact._lstate[2] = 1
            contact._list_ids = None if act is None or contact._conn_seen is None \
                else np.asarray(act, np.int64).ravel()
            contact._restored = True
            contact._pending = None

def no_chunk_args():
    """`MemberContact.chunk_args` of a model without contact."""
    zi = np.zeros(1, np.int64); z3 = np.zeros((1, 3))
    return (False, np.zeros((0, 2), np.int64), np.zeros(0), np.zeros(0), np.zeros(0, np.int64), 0,
            np.zeros(1), np.zeros(1), 0.0, 0.0, 0.0, np.zeros(0, np.int64), np.zeros(0), 0.0,
            zi, zi.copy(), np.zeros(3, np.int64), z3, z3.copy(), z3.copy(), np.zeros(1), np.zeros(4))


def attach_member_contact(solver, problem):
    """Build the member contact of the case and attach it; None without one."""
    spec = (problem.get("contact") or {}).get("members")
    if not spec:
        return None
    if njit is None:
        raise ValueError("contact.members needs numba (pip install numba): the pair search is compiled")
    if getattr(solver, "resident", False):
        raise ValueError("contact is not on the device-resident (warp) path; use backend numpy or numba")
    contact = MemberContact(spec, solver)
    contact.attach()
    return contact
