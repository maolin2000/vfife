"""Elastic-brittle breaking and scheduled removal of line members, one table
for every line solver (v0.4 wrote it into mixed_line; v0.5 lifted
it out so beam2, beam2_timoshenko, the standalone truss and the fibre beam
share it).

    assign:                                   # mixed_line
      - {region: stays, section: stay, failure: {tensile_strength: 1.77e9}}
    element:                                  # beam2 / beam2_timoshenko / truss2
      type: beam2
      failure: {yield_strength: 3.7e8, extreme_fibre: [0.3, 0.6], shape_factor: 1.12}
    removals:                                 # any of them, and beam2_fiber
      - {region: hangers_mid, time: 3.0}

WHAT IS SHARED is the bookkeeping, which is the same whatever carries the
force: which members can break and at what capacity, which are due to be
removed and when, which have gone, the ratio demand/capacity of every member
at the last evaluation, the event list, and how all of that goes through a
checkpoint and into the output fields. WHAT IS NOT is how a solver stops a
member from carrying load, because each family keeps its members differently:
mixed_line rebuilds its batches from the active mask, beam2 zeroes the
member's local stiffness, the truss zeroes its area, the fibre beam already
has a latch (`_active`, core/fracture.py) and only needs it set. Each solver
hands `check` its end actions after a force evaluation and deactivates the
ids it gets back.

THE CRITERION is what v0.4 chose and why: a bar breaks at N >= f_t A; a
beam at either end when |N|/(A f_y) + |M_y|/M_y,cap + |M_z|/M_z,cap >= 1 with
M_cap = shape_factor f_y I / c -- a capacity, because an elastic member never
reaches a real steel's tearing strain first. A fibre beam gets NO capacity
criterion here: its section already knows its own strength, fibre by fibre,
and `element.fracture` is where it breaks (the schema refuses `failure` on
it). Judged in dynamic steps only, after the forces of the step, never
inside relaxation iterations.

WHAT A BREAK DOES. Two things can happen to a member that has
broken. It can be DEACTIVATED: it carries nothing from then on and its mass
stays on the nodes it was lumped on, which is v0.4's rule and is what a
snapped cable is. Or it can be SPLIT OFF: both its ends move onto spare
particles reserved for them at setup (core/topology.BondGraph, the same
reservation the fibre beam makes for `fracture.split`), its share of the
mass goes with it, and it goes on carrying force between its own two
particles -- a girder segment that broke at both ends is an elastic beam
flying free, with its weight, until the ground stops it. The frame it left
loses exactly that member's weight and is no longer tugged by a piece
lying on the sea floor. `split` on a `failure` or a `removal` chooses,
each for its own cause (v1.0: a removal's used to be overwritten by the
member's failure spec); left out, the solver's default applies (the frame solvers split, the
standalone truss deactivates).

A BEAM BREAKS AT AN END, not as a whole. Its criterion is judged
at each end, and the section that reached its capacity is where the frame
parts: a split member leaves the particle at THAT end and stays welded at
the other, and may later break there too (then it is a free piece). The
first version took the whole member off at both ends at its first break --
each fracture cut a whole element out of the girder, and the Xihoumen k = 3
collapse unzipped into 267 single-element pieces, 190 of the 268 girder
breaks happening in the air. A bar (a cable: the same force at both ends)
and a scheduled removal still go at both ends, as does a member that is
deactivated rather than split.

THE END-ACTION LAYOUT every caller must supply is the frame member's:
(nel, 12), local, node i then node j, [N, V_y, V_z, T, M_y, M_z] each, with
N tension-positive at column 6 (the j end). A bar fills columns 0 and 6.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


#: the twist, end to end, at which a breakable beam is taken as torn (rad;
#: v1.17): a quarter turn, short of the half turn the co-rotational
#: frame cannot follow
TWIST_LIMIT = 0.5 * np.pi
#: the multiple of its capacity at which a piece broken at both ends is
#: removed: it carries its capacity at the instant it parts
FRAGMENT_LIMIT = 2.0


class Breaking:
    def __init__(self, nel: int, mesh, removals: Optional[list], log=None, split_default: bool = False):
        self.nel = nel
        self.log = log
        #: carries force (a deactivated member does not; a SPLIT one still does,
        #: between the particles of its own it was moved onto)
        self.member_active = np.ones(nel, dtype=bool)
        #: has broken, either way: no second break, grey in the viewer
        self.member_failed = np.zeros(nel, dtype=bool)
        #: which ends have parted (a split beam may break at one end first)
        self.end_broken = np.zeros((nel, 2), dtype=bool)
        #: ends broken since the solver last took them (`take_ends`)
        self._ends_due = {}
        #: what a break does to the member: split it off onto its own
        #: particles (where the solver reserved them) or deactivate it.
        #: This is the `failure` spec's; a removal carries its own (below),
        #: and a member's failure spec used to overwrite its removal's -- a
        #: `removal: {split: true}` on members with `failure: {split: false}`
        #: never split
        self.split = np.zeros(nel, dtype=bool)
        self.split_default = bool(split_default)
        #: what the member's break DID: split (True) or deactivate, per member
        self.split_on_break = np.zeros(nel, dtype=bool)
        # NaN for a member with no criterion: "it cannot break" is not the
        # same colour as "it carries nothing" (the viewer draws NaN neutral)
        self.capacity_ratio = np.full(nel, np.nan)
        self.failure_events: List[list] = []             # [t, element, ratio, cause]
        self._bar_cap = np.full(nel, np.inf)              # N at which a bar breaks
        #: N_p, M_y,cap, M_z,cap, and the torque of a quarter turn's twist
        #: (`twist_caps`, v1.17; inf until set)
        self._beam_cap = np.zeros((nel, 4))
        self._beam_cap[:, 3] = np.inf
        self._beam_on = np.zeros(nel, dtype=bool)
        #: plastic-hinge members: their moments are capped by the
        #: solver before `check`, and an end breaks on its accumulated plastic
        #: rotation (`hinge_damage` = rotation / capacity, set by the solver)
        #: or on axial force alone, not on the interaction reaching 1
        self.hinged = np.zeros(nel, dtype=bool)
        self.rotation_capacity = np.full(nel, np.inf)
        self.hinge_damage = np.zeros((nel, 2))
        self._removals = []
        for r in removals or []:                # schema.RemovalSpec
            ids = mesh.region_elements(r.region, "truss2", mesh.elements)
            if ids is None or not len(ids):
                raise ValueError(f"removal region {r.region!r} contains no complete members")
            # [time, members, fired, split]
            self._removals.append([float(r.time), np.asarray(ids, int), False, self._wants_split(r)])

    def _wants_split(self, spec) -> bool:
        """A failure spec (a dict) or a removal (`schema.RemovalSpec`)."""
        v = spec.get("split") if isinstance(spec, dict) else getattr(spec, "split", None)
        return self.split_default if v is None else bool(v)

    # ------------------------------------------------------------ capacities
    def bars(self, ids, spec: dict, area: float) -> None:
        """`spec` is a FailureSpec of the bar kind for the members `ids`."""
        ids = np.asarray(ids, int)
        self._bar_cap[ids] = float(spec["tensile_strength"]) * float(area)
        self.capacity_ratio[ids] = 0.0
        self.split[ids] = self._wants_split(spec)

    def beams(self, ids, spec: dict, area: float, iy: float, iz: float) -> None:
        """`spec` is a FailureSpec of the beam kind: `extreme_fibre` is
        [c_y, c_z] along the section's local y and z, so M_y (bending about
        local y) is capped by the fibre at c_z and M_z by the one at c_y."""
        ids = np.asarray(ids, int)
        fy = float(spec["yield_strength"]); cy, cz = map(float, spec["extreme_fibre"])
        sf = float(spec.get("shape_factor", 1.0))
        self._beam_cap[ids, :3] = (float(area) * fy, sf * fy * float(iy) / cz, sf * fy * float(iz) / cy)
        self._beam_on[ids] = True
        self.capacity_ratio[ids] = 0.0
        self.split[ids] = self._wants_split(spec)
        if spec.get("rotation_capacity") is not None:
            self.hinged[ids] = True
            self.rotation_capacity[ids] = float(spec["rotation_capacity"])

    def twist_caps(self, k_torsion) -> None:
        """The torque at which each breakable beam is taken as torn by
        twisting, from its torsional stiffness G J / L ((nel,), N m/rad;
        v1.17). Torsion is not checked against a strength (see the
        criterion above); what is checked is that no member twists a
        quarter turn end to end. The co-rotational frame reads a member's
        twist as the rotation between its two ends' triads, which cannot
        pass half a turn: past it the torque the frame computes reverses,
        and a fragment spinning about its own axis gained energy at every
        turn -- a free 6 m stringer at 1e9 J, flying 200 m. A steel member
        twisted that far has torn long before (a 20 mm wall over 6 m yields
        in shear at about 0.75 rad), so it is removed, whole, at TWIST_LIMIT."""
        k = np.asarray(k_torsion, float)
        on = self._beam_on & (k > 0.0)
        self._beam_cap[on, 3] = TWIST_LIMIT * k[on]

    def add(self, ids, spec: Optional[dict], props: dict, kind: str) -> None:
        """The failure `spec` of a member group of element `kind` whose
        section properties are in `props` (area, iy, iz)."""
        if not spec:
            return
        if kind == "truss2":
            self.bars(ids, spec, props["area"])
        else:
            self.beams(ids, spec, props["area"], props["iy"], props["iz"])

    @property
    def can_break(self) -> bool:
        return bool(np.isfinite(self._bar_cap).any() or self._beam_on.any() or self._removals)

    # ----------------------------------------------------------------- check
    @property
    def breakable(self) -> np.ndarray:
        """Elements a break can happen to (a criterion or a removal), for the
        solver that reserves split particles before anything is sized."""
        on = np.isfinite(self._bar_cap) | self._beam_on
        for rem in self._removals:
            on[rem[1]] = True
        return on

    def mark(self, ids, t: float, ratio, cause: int, ends=None, split=None) -> np.ndarray:
        """Break the members `ids` at `ends` ((n, 2) bool; both when None)
        where they have not broken yet; returns the members that broke
        somewhere new. A member marked to split stays active (it goes on
        carrying force between its own particles) and parts only at the
        ends given; one that is not is deactivated, whole, once. `split`
        is the cause's own choice (a removal's); None is the failure
        spec's (`self.split`)."""
        ids = np.asarray(ids, int)
        sp = self.split[ids] if split is None else np.broadcast_to(np.asarray(split, bool), ids.shape).copy()
        want = np.ones((len(ids), 2), bool) if ends is None else np.asarray(ends, bool).reshape(-1, 2)
        want = want & ~self.end_broken[ids]
        gone = self.member_failed[ids] & ~self.member_active[ids]
        whole = ~sp | gone
        want[whole & gone] = False                                # deactivated already
        want[whole & ~gone] = True                                # deactivation takes both
        keep = want.any(axis=1)
        ids, want, sp = ids[keep], want[keep], sp[keep]
        ratio = np.broadcast_to(ratio, keep.shape)[keep] if np.ndim(ratio) else np.full(len(ids), float(ratio))
        if len(ids):
            self.member_failed[ids] = True
            self.member_active[ids] = sp
            self.split_on_break[ids] = sp
            self.end_broken[ids] |= want
            for e, w in zip(ids, want):
                self._ends_due[int(e)] = self._ends_due.get(int(e), np.zeros(2, bool)) | w
            for e, r in zip(ids, ratio):
                self.failure_events.append([float(t), int(e), float(r), float(cause)])
            if self.log is not None and cause == 3:
                self.log(f"  member(s) {', '.join(map(str, ids[:8]))}{' ...' if len(ids) > 8 else ''}"
                         f" twisted a quarter turn at t = {t:.6g} s: removed")
            if self.log is not None and cause == 0:
                self.log(f"  member(s) {', '.join(map(str, ids[:8]))}{' ...' if len(ids) > 8 else ''}"
                         f" broke at t = {t:.6g} s ({int(self.member_failed.sum())}/{self.nel} broken)")
        return ids

    def take_ends(self, e: int) -> tuple:
        """The ends of member e that broke since the last call: (0,), (1,)
        or (0, 1) -- both when nothing was recorded (a caller that marks
        by hand)."""
        w = self._ends_due.pop(int(e), None)
        return (0, 1) if w is None or not w.any() else tuple(int(k) for k in np.flatnonzero(w))

    def check(self, t: float, end_actions: np.ndarray, removals: bool = True, whole: bool = False,
              observe: bool = False) -> np.ndarray:
        """After a force evaluation of a DYNAMIC step: the removals now due
        and every member whose demand has reached its capacity. Returns the
        ids newly gone (possibly empty). A dynamic step is a real instant, so
        a break here is a break in a real state -- the caller never asks
        under relaxation. The implicit static solver asks at an
        ACCEPTED equilibrium, with `removals=False` (it has no clock) and
        `whole=True`: a broken member is taken out whole, never split off
        (a fragment has no static equilibrium; 1.7).

        `observe`: a READING of the state (an output frame, a progress
        check) -- the capacity ratios are written, so that a
        frame's `capacity_ratio` belongs to the forces in the same frame,
        and nothing is removed or broken: the step does that."""
        if not self.can_break:
            return np.zeros(0, int)
        # the two rules of v1.17 judge the members as they were on ENTRY: a
        # member that parts here, carrying its capacity, is split off as its
        # failure says, not removed in the same breath (found in review)
        entry_whole = self._beam_on & self.member_active & self.end_broken.all(axis=1)
        entry_on = self._beam_on & self.member_active & np.isfinite(self._beam_cap[:, 3])
        n_events = len(self.failure_events)
        gone = []
        for rem in (self._removals if removals and not observe else ()):
            if not rem[2] and t >= rem[0]:
                rem[2] = True
                gone.append(self.mark(rem[1], t, 0.0, 1, split=rem[3]))
        on = np.isfinite(self._bar_cap) & ~self.member_failed
        if on.any():
            ids = np.flatnonzero(on)
            ratio = end_actions[ids, 6] / self._bar_cap[ids]            # tension positive
            self.capacity_ratio[ids] = np.maximum(ratio, 0.0)
            hit = (ratio >= 1.0) & (not observe)
            if hit.any():
                gone.append(self.mark(ids[hit], t, ratio[hit], 0, split=False if whole else None))
        # a beam is judged at every end it is still welded at: a split one
        # that broke at one end is a cantilever off the other, and can break
        # there as well
        on = self._beam_on & self.member_active & ~self.end_broken.all(axis=1)
        if on.any():
            ids = np.flatnonzero(on)
            fl = end_actions[ids]; cap = self._beam_cap[ids]
            r_i = np.abs(fl[:, 0]) / cap[:, 0] + np.abs(fl[:, 4]) / cap[:, 1] + np.abs(fl[:, 5]) / cap[:, 2]
            r_j = np.abs(fl[:, 6]) / cap[:, 0] + np.abs(fl[:, 10]) / cap[:, 1] + np.abs(fl[:, 11]) / cap[:, 2]
            r = np.where(self.end_broken[ids], 0.0, np.column_stack([r_i, r_j]))
            ratio = r.max(axis=1)
            self.capacity_ratio[ids] = ratio
            # a hinge end breaks on its rotation, or on axial force alone
            h = self.hinged[ids]
            if h.any():
                axial = np.column_stack([np.abs(fl[:, 0]), np.abs(fl[:, 6])]) / cap[:, :1]
                crit = np.where(self.end_broken[ids], 0.0, np.maximum(self.hinge_damage[ids], axial))
                r = np.where(h[:, None], crit, r)
            hit = (r.max(axis=1) >= 1.0) & (not observe)
            if hit.any():
                gone.append(self.mark(ids[hit], t, r[hit].max(axis=1), 0, ends=r[hit] >= 1.0,
                                      split=False if whole else None))
        # A PIECE BROKEN AT BOTH ENDS IS STILL STEEL. It was judged
        # nowhere once both its ends had parted, and a free 10 m diagonal of
        # the generic truss rang on the ground with six times its plastic
        # moment end to end, 35 degrees between its two ends, gaining energy
        # at every bounce. Such a piece is removed, whole, when its demand
        # reaches FRAGMENT_LIMIT times its capacity -- not at the capacity
        # itself, which it carries at the instant it parts, and a piece that
        # leaves the frame is meant to fall as a piece -- and its two
        # particles fall on as masses.
        on = entry_whole & self.member_active
        if on.any() and not observe:
            ids = np.flatnonzero(on)
            fl = end_actions[ids]; cap = self._beam_cap[ids]
            r = np.maximum(np.abs(fl[:, 0]) / cap[:, 0] + np.abs(fl[:, 4]) / cap[:, 1] + np.abs(fl[:, 5]) / cap[:, 2],
                           np.abs(fl[:, 6]) / cap[:, 0] + np.abs(fl[:, 10]) / cap[:, 1] + np.abs(fl[:, 11]) / cap[:, 2])
            hit = r >= FRAGMENT_LIMIT
            if hit.any():
                gone.append(self.mark(ids[hit], t, r[hit], 0, split=False))
        # twisted a quarter turn: removed whole, whatever split says
        marked = np.zeros(self.nel, bool)
        for ev in self.failure_events[n_events:]:
            marked[int(ev[1])] = True
        on = entry_on & self.member_active & ~marked
        if on.any() and not observe:
            ids = np.flatnonzero(on)
            tw = np.maximum(np.abs(end_actions[ids, 3]), np.abs(end_actions[ids, 9])) / self._beam_cap[ids, 3]
            hit = tw >= 1.0
            if hit.any():
                gone.append(self.mark(ids[hit], t, tw[hit], 3, split=False))
        return np.concatenate(gone) if gone else np.zeros(0, int)

    # ------------------------------------------------------ persistence, io
    def checkpoint(self, ck: dict) -> dict:
        ck["member_active"] = self.member_active.astype(np.int8)
        ck["member_failed"] = self.member_failed.astype(np.int8)
        ck["end_broken"] = self.end_broken.astype(np.int8)
        # the events themselves: a restarted result had no member_failures
        # table though its members were broken
        if self.failure_events:
            ck["member_failure_events"] = np.asarray(self.failure_events, float).reshape(-1, 4)
        return ck

    def restore(self, ck: dict) -> np.ndarray:
        """Returns the ids that are inactive after the restore, for the
        solver to deactivate in its own representation."""
        if "member_active" in ck:                       # a broken member stays broken
            self.member_active[...] = np.asarray(ck["member_active"]).astype(bool)
            self.member_failed[...] = (np.asarray(ck["member_failed"]).astype(bool) if "member_failed" in ck
                                       else ~self.member_active)
            self.end_broken[...] = (np.asarray(ck["end_broken"]).astype(bool) if "end_broken" in ck
                                    else self.member_failed[:, None])
        if "member_failure_events" in ck:
            # in place: the solvers hold this very list as `failure_events`
            self.failure_events[:] = [[float(t), int(e), float(r), float(c)] for t, e, r, c in
                                      np.asarray(ck["member_failure_events"], float).reshape(-1, 4)]
        return np.flatnonzero(~self.member_active)

    def fields(self, out: dict) -> dict:
        out["member_failed"] = self.member_failed.astype(float)
        out["capacity_ratio"] = self.capacity_ratio.copy()
        return out

    def diagnostics(self, diag) -> None:
        if isinstance(diag, dict) and self.failure_events:
            diag["member_failures"] = self.failure_events
