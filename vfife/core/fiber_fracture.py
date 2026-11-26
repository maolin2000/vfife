"""The fibre beam's fracture (1.22, out of `fiber_beam_solver`): the spare
particles, the failure criteria, the members that break and the splits
they make. A mixin of `FiberBeamSolver`, which is where its state lives."""
from __future__ import annotations
from typing import Dict, List
import numpy as np

from . import fracture
from .member_failure import Breaking
from . import topology
from ..base.dof import DofLayout


class FiberFracture:
    def _reserve_split_particles(self):
        """Park a spare particle on every element end, before anything is sized.

        THE WHOLE OF THE PRE-ALLOCATION is here, and it happens between the
        regions and everything else for two reasons that are both about not
        being clever later:

        REGIONS FIRST, so that a spare is in no region. A spare sits exactly on
        top of the particle it may one day replace, so a geometric selector --
        `where: {x: 0}` -- would pick it up, and the fragment that finally uses
        it would be born clamped to the ground it was supposed to fall away
        from. Resolving the regions before the spares exist is the only way to
        get that right that does not involve special cases in the selector.

        EVERYTHING ELSE AFTER, so that no array is ever resized at run time:
        `q`, `v`, the mass, the orientations, the reference stiffness and the
        output mesh are all built at the final size. That is not a performance
        argument -- a handful of fractures could afford a reallocation -- it is
        that `self.orient` and `self.v` are REBOUND in the time loop (a
        multiplicative orientation update, a Rayleigh velocity update), so a
        design where the live arrays are windows onto a bigger buffer would
        lose the window silently on the first step after a split.

        The reservation is exactly two per element, indexed `base + 2e + k`,
        because an element end can separate once. There is no capacity to
        choose and none to run out of. The cost is 2 x n_elements particles'
        worth of state -- for a frame, about three times the original -- and
        it buys a run whose memory is known at setup.
        """
        self._spare_base = None
        spec = (self.p.get("element") or {}).get("fracture")
        if not spec or spec is True or not dict(spec).get("split"):
            return
        if self.p.solver.mode != "dynamic":
            from .diagnostics import UserError
            raise UserError(
                "SPLIT_NEEDS_DYNAMIC",
                "element.fracture.split needs solver.mode = dynamic",
                "a fragment that separates has nothing to relax to: it "
                "accelerates. Use the dynamic solver, or turn the split off "
                "and keep the deactivation-only fracture of v0.1")
        self._spare_base = self.nn
        # (n_elements, 2), element-major: the spare for end k of element e is
        # parked ON that end's particle, so its reference coordinates, and
        # therefore the element's undeformed geometry, do not change when the
        # bond moves onto it.
        self._spare_of = (self._spare_base
                          + 2 * np.arange(self.nel, dtype=np.int64)[:, None]
                          + np.arange(2, dtype=np.int64)[None, :])
        parked = self.mesh.nodes[self.mesh.elements].reshape(-1, 3)
        self._elements0 = self.mesh.elements[:, :2].copy()      # the connectivity as read
        self.mesh.nodes = np.vstack([self.mesh.nodes, parked])
        self.nn = self.mesh.n_nodes          # a property: the rows just added
        self.dofs = DofLayout.frame(self.nn)
        self.ndof = self.dofs.size

    def _setup_fracture(self, spec_root):
        """`element.fracture: {threshold: 1.0}` -- off unless it is asked for.

        THE ELEMENT'S ONLY CONTRIBUTION to the criterion is the layout: this
        section has `n_fibers` fibres, each sampled at every Gauss station,
        and a fibre is gone when it is gone at ANY of them (Cai et al. eq. 20).
        What makes a fibre fail is the fibre material's business and is not
        mentioned here or anywhere else in this file (core/fracture.py).

        Which is why the check below is worth its lines: a section whose laws
        declare no failure at all can never reach the threshold, so asking for
        fracture and getting a run that never fractures would look exactly
        like a structure that is strong enough. Refuse it instead.
        """
        self.fracture = None
        self._active = np.ones(self.nel, dtype=np.uint8)
        self._refresh_kernel_mask()
        spec = (spec_root or {}).get("fracture")
        if not spec:
            return
        spec = {} if spec is True else dict(spec)
        silent = [type(m).__name__ for g, m in enumerate(self.materials)
                  if m.failure(m.init_state(1)) is None]
        if len(silent) == len(self.materials):
            raise ValueError(
                "element.fracture is on but no fibre material declares a "
                f"failure criterion ({', '.join(sorted(set(silent)))}); give "
                "the bars a `rupture_strain`, or use a law with damage")
        self._fracture_silent = sorted(set(silent))
        # WHICH FIBRES ARE WORTH WATCHING, decided once. A law that declares no
        # criterion returns None forever, so its fibres are structurally zero
        # in the gather -- on an RC section that is 64 of 68 columns whose
        # contents can never change. Only the rest are gathered and reduced,
        # and `fiber_layout(..., tracked=)` keeps the denominator at n_fibers
        # so FAC still means the fraction of the SECTION that is gone.
        self._failing_groups, n_tracked = [], 0
        for g, m in enumerate(self.materials):
            n = len(self.group_indices[g])
            if self.fiber_states[g].shape[3] == 0 or m.failure(m.init_state(1)) is None:
                continue
            self._failing_groups.append((g, n_tracked, n))
            n_tracked += n
        if not n_tracked:                  # a criterion on a law with no state
            raise ValueError(
                "element.fracture is on but no fibre carries a law that can "
                "report failure; give the bars a `rupture_strain`, or use a "
                "law with damage")
        self._tracked_failure = np.zeros(
            (self.nel, n_tracked, len(self.gauss_s)))
        # A PADDED FIBRE CANNOT BREAK. With several sections
        # the fibre arrays are padded to the widest one; a padded fibre has
        # zero area, sees the axial strain, and its law may well report
        # failure -- so its column is masked out of the gather, and the
        # fraction is taken over the member's OWN fibre count.
        self._tracked_valid = None
        over = None
        if len(self._layouts) > 1 or not bool(self.fiber_valid.all()):
            # a tracked column belongs to one section's group: it is live on
            # that section's rows and on real (unpadded) fibres only
            valid = np.zeros((self.nel, n_tracked), dtype=float)
            for g, start, n in self._failing_groups:
                rows = self._rows[int(self.group_section[g])]
                valid[rows, start:start + n] = self.fiber_valid[rows][:, self.group_indices[g]]
            self._tracked_valid = valid[:, :, None]
            over = self.fiber_valid.sum(axis=1).astype(float)
            # a member with no fibres at all (a connector, a bearing) has
            # nothing that can break and a fraction of 0 out of anything:
            # its 0 here was refused by the latch ("every element is out of
            # at least one thing"), and a fibre model with `fracture`, two
            # fibre sections and a bearing stopped at its first step (found
            # in the review of v1.12; there since the padded layouts)
            over[over < 1.0] = 1.0
        self._pending_splits: List[int] = []
        #: (time, element, parent particle, new particle) of every separation
        self.splits: List[tuple] = []
        #: spare particle -> the particle it was split from (the frame
        #: solvers' `_with_fragments`)
        self._fragment_parent: Dict[int, int] = {}
        self.bonds = None
        if self._spare_base is not None:
            self.bonds = topology.BondGraph(self.mesh.elements, self.nn,
                                            reserved=self._spare_of)
            self._components0 = self.bonds.n_components()
        self.fracture = fracture.Fracture(
            self.nel,
            # with several sections the tracked columns are the sections'
            # groups side by side, more than any one row holds; the fraction
            # is then taken over each member's own count (`over`), and the
            # layout's denominator is only its shape
            fracture.fiber_layout(max(self.n_fibers, n_tracked), len(self.gauss_s),
                                  tracked=n_tracked),
            threshold=float(spec.get("threshold", 1.0)),
            on_fracture=(self._queue_split if self.bonds is not None else None),
            over=over)

    # ------------------------------------------------------------------ #
    #  Scheduled removal, and the shared event list (core/member_failure)
    # ------------------------------------------------------------------ #
    def _setup_breaking(self):
        """A fibre member has no capacity criterion here -- its section knows
        its own strength, fibre by fibre, and `fracture` is where it breaks --
        but it can be REMOVED at a time (`removals`), and both go to
        the same `member_failures` list and `member_failed` field the other
        line solvers write."""
        self.breaking = Breaking(self.nel, self.mesh, self.p.case.removals, self.log)
        self.member_active = self.breaking.member_active
        self.failure_events = self.breaking.failure_events

    def _check_failure(self, observe=False):
        """After a dynamic force evaluation: the removals now due. `observe`:
        a reading -- the capacity ratios alone."""
        b = self.breaking
        if self.mode != "dynamic" or not b.can_break or getattr(self, "_trial_mode", False):
            return
        if observe:
            b.check(float(self.time), self.end_actions, observe=True)
            return
        gone = b.check(float(self.time), self.end_actions)
        if len(gone):
            self._deactivate(gone)
            b.diagnostics(getattr(self, "diagnostics", None))
            # what the removal set loose is looked at between two steps, as
            # a split is (`apply_pending_splits`): only where members can
            # split, the frame solvers' scope
            if self.bonds is not None:
                self._pending_debris = True

    def _deactivate(self, ids):
        ids = np.asarray(ids, int)
        if len(ids):
            self._active[ids] = 0
            self._refresh_kernel_mask()

    def _live_members(self):
        """(nel,) bool: neither removed nor fractured (a fracture without a
        `breaking` table is in `_active` alone)."""
        return self.breaking.member_active & self._active.astype(bool)

    def _queue_split(self, newly, t):
        """The `on_fracture` seam of v0.1, and all it does is write a note.

        NOT the split itself, because of WHERE this is called from: inside the
        internal-force evaluation, where `F`, `fint` and the boundary masks of
        the step in progress are already vectors of the current length. Moving
        a bond changes which particle carries a member's mass, and doing that
        halfway through a step would leave the step's own force vector
        describing a structure that no longer exists. The loop applies the
        queue at the end of the step instead (`apply_pending_splits`), which
        is the only instant at which the model is between two consistent
        states.
        """
        self._pending_splits.extend(int(e) for e in newly)

    def apply_pending_splits(self, t):
        """Move the fractured members' bonds onto their own particles.

        WHAT A SPLIT IS, in one line: the member's share of the mass moves
        from the frame's particle to the fragment's, and the fragment's
        particle starts with the frame particle's position and velocity.

        MASS IS CONSERVED because the same row leaves one particle and lands
        on another (`_element_mass_share`, one definition). MOMENTUM IS
        CONSERVED for the same reason and no other: the mass that moves takes
        the velocity it already had, so sum(m v) is unchanged at the instant of
        the split. Giving the fragment anything else -- a separation velocity,
        a zeroed velocity -- would be inventing an impulse, and the invariant
        test would be right to fail.

        WHAT THE FRAGMENT IS NOW: its two particles, and the dead member
        between them. The member carries no force, so under gravity
        alone both particles see the same acceleration and the fragment falls
        as a rigid segment; nothing holds it together against anything else,
        which is honest about what this stage does and does not have -- there
        is no contact yet, so it falls THROUGH the floor below it.
        """
        debris, self._pending_debris = self._pending_debris, False
        if not self._pending_splits:
            # a removal alone can set a piece of the frame loose too
            return bool(debris) and self._debris_isotropic(self._grounded(self._split_held()),
                                                           self._energy_ledger())
        pending, self._pending_splits = self._pending_splits, []
        pm = self.physical_mass.reshape(self.nn, self.dofs.stride)
        moved = []
        # a particle held in this step is the world's: an end there leaves it
        # even when it is the particle's only bond, or the fragment stays
        # clamped where it broke off (the frame solvers' rule)
        held = self._split_held()
        ledger = self._energy_ledger()
        for e in sorted(set(pending)):
            share = self._element_mass_share(e)
            for k, parent, new in self.bonds.detach(e, held=held):
                # the mass, the state and the ledger, as the frame solvers
                # move them (core/beam_family)
                self._move_end(e, parent, new, share, pm, ledger)
                moved.append((e, k, parent, new))
            self.mesh.elements[e] = self.bonds.bonds[e]
            if getattr(self, "_edofs", None) is not None:
                self._edofs[e] = self.dofs.element_dofs(self.mesh.elements[e])
                self._x0[e] = self.mesh.nodes[self.mesh.elements[e]]
        if not moved:
            return False
        # A CONNECTOR THAT HOLDS NOTHING UP GOES, and a particle left with
        # nothing that is a body stops where it is: a girder member that
        # fractured over its bearing left the bearing's particle on the
        # connector alone, with no mass, and the step bound at 5e-16 s (the
        # review of v1.18; the frame solvers' rules, v0.5)
        device = ~np.asarray(self._is_fiber, bool)
        if device.any():
            touched = {int(p) for _e, _k, p, _n in moved}
            hang = self._hanging_devices(device, held, device)
            if hang:
                self._deactivate(hang)
                b = self.breaking
                b.member_active[hang] = False
                b.member_failed[hang] = True
                b.failure_events.extend([float(t), e, 0.0, 2.0] for e in hang)
                touched.update(int(n) for n in self.mesh.elements[hang, :2].ravel())
                self.log(f"  {len(hang)} connector(s) left holding nothing up: removed")
            self._park_bodiless(touched, t, device)
        self.mass = np.maximum(self.physical_mass, 1e-20)
        # A fragment's particles are in no region -- a fragment that
        # inherited its parent's CONSTRAINT regions would be born clamped to
        # the support it just broke away from. Gravity reaches them through
        # their parent (`_with_fragments`, as the frame solvers do); a nodal
        # force stays on the nodes it was written on. They used to join
        # "all": gravity on a named region missed them (the debris hung in
        # the air) and a force on "all" was spread onto them.
        self._rebuild_reference_stiffness()
        self._floor_split_rotations(moved)
        # ON DEBRIS ONE ROTARY INERTIA FOR ALL THREE AXES: a piece the
        # fracture set loose tumbles out of the frame its per-axis inertia
        # was set in (the frame solvers' rule of v1.17, here from v1.18)
        self._debris_isotropic(self._grounded(held), ledger)
        before, self.dt_bound = self.dt_bound, self._time_step_bound()
        self._bc_cache = None
        self.splits.extend((float(t), e, p, n) for e, _k, p, n in moved)
        self.log(f"  split: {len(moved)} particle(s) at t = {t:.6g} s "
                 f"({self.bonds.n_components()} component(s), "
                 f"dt bound {before:.3e} -> {self.dt_bound:.3e} s)")
        # the run's dt is usually the bound itself, and a split that leaves
        # it where it was recomputes it to within a rounding: only a bound
        # that really fell is said (the warning came at every split)
        if self.dt_bound < self.dt * (1.0 - 1e-9):
            self.log(f"  WARNING: dt = {self.dt:.3e} s now exceeds the stability "
                     f"bound {self.dt_bound:.3e} s; the step is NOT adapted "
                     f"mid-run")
        return True

    def _floor_split_rotations(self, moved):
        """The particles a split touched keep a rotary inertia stable at the
        run's step.

        The share that moves is the member's TRUE rotary inertia,
        and the setup raised a rotation below what the step needs to the least
        stable value; the fragment's particle got the unfloored share and the
        bound fell under the step (0.90 dt, a bar along z). As at the setup,
        and as the frame solvers' split: m_rot >= row (dt / dt_safety / f)^2
        on a free rotation, added with its angular momentum kept (v scaled by
        m_old / m_new). A rotation does not weigh, so nothing is booked
        against gravity."""
        if getattr(self, "mode", None) != "dynamic" or not moved:
            return
        stride = self.dofs.stride
        pm = self.physical_mass.reshape(self.nn, stride)
        row = np.asarray(abs(self.K).sum(axis=1)).ravel().reshape(self.nn, stride)
        free = (~self._held_dofs()).reshape(self.nn, stride)
        f = float(getattr(self, "damping_step_factor", 1.0) or 1.0)
        need = (self.dt / (float(self.p.solver.dt_safety) * f)) ** 2
        touched = sorted({int(p) for _e, _k, p, _n in moved} | {int(n) for _e, _k, _p, n in moved})
        for n in touched:
            on = free[n, 3:] & (row[n, 3:] > 1e-20)
            old = pm[n, 3:].copy()
            new = np.where(on, np.maximum(old, row[n, 3:] * need), old)
            if not (new > old).any():
                continue
            rd = self.dofs.node_dofs(np.array([n])).ravel()[3:]
            self.v[rd] = self.v[rd] * np.where(new > 0.0, old / np.maximum(new, 1e-300), 1.0)
            pm[n, 3:] = new
        self.mass = np.maximum(self.physical_mass, 1e-20)

    def _committed_failures(self):
        """How many fibre points of the ACCEPTED state have failed, over every
        law that declares a criterion (fracture on or not). The staged
        relaxation compares it before and after a commit,
        the static solver too (`implicit._settle_ruptures`).

        THE COMMITTED STATE, not the live one. The two are the same wherever
        a commit is compared, except after a restart or an `initial_state`
        taken inside a level: the live state there is the level's TRIAL,
        with the points past their limit already failed, and the commit
        found nothing new -- the level was not solved again and its frame
        paired the intact positions with the cut forces, 372 kN against the
        uninterrupted run's 294 kN, in staged relaxation and in the static
        solver's fallback alike."""
        n = 0
        for g, m in enumerate(self.materials):
            st = self.fiber_committed[g]
            if st.shape[-1] == 0:
                continue
            f = m.failure(st.reshape(-1, st.shape[-1]))
            if f is not None:
                n += int(np.count_nonzero(f > 0.5))
        return n

    def _point_failure(self):
        """(nel, tracked_fibres * n_stations), fibre-major, straight from the laws.

        The gather is the same on both force paths because the kernel's
        consolidated state arrays are VIEWS: `fiber_states[g]` still addresses
        live data after `_relink_kernel_states` (see there), so this reads the
        fibres the kernel just wrote without knowing the kernel exists.

        ONLY THE FIBRES THAT CAN FAIL are gathered (`_failing_groups`, decided
        at setup), and into a buffer that is allocated once: every tracked
        column is written on every call, so there is nothing to zero. This runs
        at every step of every fracture run, and on an RC section it used to
        build a 68-column array of which 64 columns were a law answering None.
        """
        f = self._tracked_failure
        for g, start, n in self._failing_groups:
            st = self.fiber_states[g]
            fg = self.materials[g].failure(st.reshape(-1, st.shape[3]))
            f[:, start:start + n, :] = fg.reshape(st.shape[:3]).transpose(0, 2, 1)
        if self._tracked_valid is not None:
            f *= self._tracked_valid
        return f.reshape(len(f), -1)

    def _latch_fracture(self, committed: bool = False, t: float = None):
        """Ask the laws what has failed, AFTER the forces are assembled.

        After, and not before, for a reason that decides what the first
        fractured step looks like: a fibre that ruptures during this step has
        already had its stress zeroed by its own law, so the section force
        this step is the one the broken section carries. Latching first would
        deactivate the element a step EARLIER than its own material did, and
        the two halves would then disagree about which step the member died in.

        Nothing happens during a staged relaxation. `_trial_mode` means the
        state being read is a trial that is about to be thrown away and
        re-driven from the last converged one (see `fiber_committed`), and a
        latch set from a trial path cannot be un-set.
        """
        if self.fracture is None:
            return
        if getattr(self, "_trial_mode", False) and not committed:
            return
        t = float(self.time if t is None else t)
        newly = self.fracture.update(self._point_failure(), t=t)
        if len(newly):
            self._active[newly] = 0
            self._refresh_kernel_mask()
            self.log(f"  fracture: element(s) {', '.join(map(str, newly[:8]))}"
                     f"{' ...' if len(newly) > 8 else ''} at t = {t:.6g} s "
                     f"({self.fracture.n_fractured}/{self.nel} fractured)")
            if getattr(self, "breaking", None) is not None:
                # the same event list a mixed_line break goes to: cause 0,
                # and the ratio is the fraction of the section that is gone
                b = self.breaking
                fresh = newly[~b.member_failed[newly]]
                b.failure_events.extend(
                    [float(t), int(e), float(self.fracture.fac[e]), 0.0] for e in fresh)
                # FAILED as well as inactive: a fractured member reached by
                # its removal later was reported failed a second time
                b.member_active[newly] = False
                b.member_failed[newly] = True
                b.end_broken[newly] = True
                b.diagnostics(getattr(self, "diagnostics", None))
