"""What the frame solver and the fibre-beam solver share.

`BeamSolver` (beam2, beam2_timoshenko, and through it the shell, mixed and
coupled solvers) and `FiberBeamSolver` (beam2_fiber) grew as two copies of
one solver: the architecture review of 1.17 counted 38
methods of the same name in the two, several of them identical, and fixes
that reached one and not the other (1.17's debris rule is the latest). P1 of
the review moves the common parts here, one group at a time; each move keeps
both solvers' answers to the bit, or is announced where it does not.

What a host class provides: `nn`, `mesh`, `p`, `dofs`, `q`, `mode`, `length`,
`R0`, `orient`, `K`, `ndof`, `total_time`, `dt`, `plan`, `staging`, `_turned_k`
(the relaxation mass's blocks), and the class attributes below.
"""
from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Dict

import numpy as np

from . import added_mass
from . import amplitude as amp
from . import ground as ground_motion
from . import monitors as monitor_records
from . import resume
from .resume import resume_position, check_resumed
from . import selectors
from .diagnostics import NonFiniteFailure, StopAnalysis, first_nonfinite
from .diagnostics import finish as finish_diagnostics
from .diagnostics import start as start_diagnostics, ConvergenceFailure, worst_residual
from .dof import ROTATION, TRANSLATION, SparseAssembler, repeated_rotation
from .orientation import OrientedPoints
from .rotations import beam_frames_from_triads


class BeamFamily(OrientedPoints):
    #: the element type the solver's messages name
    ELEMENT_NAME = "beam2"
    #: the monitor kinds the solver records; None: it does not check them here
    MONITOR_KINDS = None
    #: massless members carried rigidly by one particle (the frame solvers'
    #: `_make_appendage`); none until one is made
    _appendages = ()
    #: particles that are no body, held where they stopped: {node: q (6,)}
    #: (`_park_bodiless`)
    _parked = None
    #: whether `step_data` holds the parked particles where they stopped:
    #: the fibre solver's way (the frame solvers carry theirs)
    HOLD_PARKED = False
    #: the graph the splits move bonds on, and the failure table (none
    #: until the solver builds them)
    bonds = None
    breaking = None
    #: mass gravity does not weigh (a stability floor, `added_mass`)
    weightless_mass = None
    #: the fibre solver's element dof table (`_restore_split`)
    _edofs = None
    #: what the dynamic loop's failure message calls the integration
    DYNAMIC_WHAT = "beam dynamic integration"
    #: inside a probe (`_reading`): a force evaluation that reads the state
    _probing = False
    #: the convergence measures' floors, the largest force and displacement
    #: the analysis has had (`_convergence_measures`), and the load scales
    #: that join the force scale: a thermal field's restraint force, the
    #: tendons' force (the fibre solver's)
    _force_scale = 0.0
    _motion_scale = 0.0
    _thermal_scale = 0.0
    _tendon_force_scale = 0.0

    @property
    def u(self):
        return self.dofs.field(self.q, TRANSLATION)

    # ------------------------------------------------------------------ #
    #  Regions and monitors
    # ------------------------------------------------------------------ #

    def _region(self, name):
        try:
            return self.mesh.regions[name]
        except KeyError:
            raise ValueError(f"unknown region {name!r}. Available: "
                             + ", ".join(sorted(self.mesh.regions))) from None

    def _setup_regions(self):
        self.amplitudes = amp.build(self.p.case.amplitudes)
        predefined = dict(self.mesh.regions)
        predefined.setdefault("all", np.arange(self.nn, dtype=np.int64))
        predefined.setdefault("surface", np.arange(self.nn, dtype=np.int64))
        self.mesh.regions = selectors.resolve_all(self.mesh.nodes,
                                                  self.p.get("regions") or {}, predefined)
        self.steps = self.p.get("steps") or [{"name": "step1"}]
        self.monitors = {name: monitor_records.from_spec(spec)
                         for name, spec in self.p.get("output", {}).get("monitors", {}).items()}
        if self.MONITOR_KINDS is not None:
            for name, mon in self.monitors.items():
                if mon.kind not in self.MONITOR_KINDS:
                    raise ValueError(f"monitor {name!r}: {self.ELEMENT_NAME} records "
                                     + ", ".join(self.MONITOR_KINDS[:-1]) + f" and {self.MONITOR_KINDS[-1]} "
                                     f"monitors, not {mon.kind!r}")
        # the members an element monitor reads, on the connectivity as read:
        # a member that later splits stays in its monitor
        self._monitor_members = {}
        if any(m.kind == "element" for m in self.monitors.values()):
            from . import member_actions
            e0 = np.asarray(self.mesh.elements[:, :2], dtype=np.int64).copy()
            self._monitor_members = {name: member_actions.region_members(self.mesh, m.regions, e0)
                                     for name, m in self.monitors.items() if m.kind == "element"}

    # ------------------------------------------------------------------ #
    #  Member frames
    # ------------------------------------------------------------------ #

    @classmethod
    def _frame(cls, xi, xj, reference):
        """(R, L): the member's local axes as rows -- x along it, y the
        orientation vector made normal to it -- and its length."""
        ex = xj - xi
        L = np.linalg.norm(ex)
        if L <= 0:
            raise ValueError(f"{cls.ELEMENT_NAME} has a zero-length member")
        ex /= L
        ref = np.asarray(reference if reference is not None else [0., 0., 1.])
        ey = ref - ex * np.dot(ref, ex)
        if np.linalg.norm(ey) < 1e-10:
            raise ValueError(f"{cls.ELEMENT_NAME} element.orientation is parallel to a member; "
                             "choose another vector")
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        return np.vstack([ex, ey, ez]), L

    # ------------------------------------------------------------------ #
    #  Vehicles
    # ------------------------------------------------------------------ #

    def _vehicle(self, idx, ld, k):
        """One `vehicle` load: built once per step and kept."""
        from .moving_load import LoadPath, path_releases
        from .vehicle import Vehicle
        if self.mode != "dynamic":
            raise ValueError("a vehicle load needs solver.mode = dynamic: it has mass and a clock")
        spec = ld["vehicle"]
        cache = self.__dict__.setdefault("_vehicle_cache", {})
        key = (k, spec.get("name") or "vehicle")
        if key not in cache:
            selected = np.zeros(self.nn, bool); selected[idx] = True
            conn = getattr(self, "_elements0", None)
            conn = self.mesh.elements if conn is None else conn
            elements = np.flatnonzero(selected[conn[:, :2]].all(axis=1))
            kinds = getattr(self, "member_kind", None)
            if kinds is not None:
                elements = np.array([e for e in elements if kinds[e] in self.BEAM_TYPES], int)
            path = LoadPath(SimpleNamespace(nodes=self.mesh.nodes, elements=conn), elements, "vehicle")
            cache[key] = Vehicle(spec, path, self, self.dofs, self.mesh, self.length, self.R0,
                                 self.total_time, self.dt)
            # its events on the run's clock
            cache[key].clock_offset = k * float(self.total_time)
            # a member's end releases condense the load it takes;
            # None where no path member has any -- every fibre member
            cache[key].releases = path_releases(self, path.elements)
        veh = cache[key]
        # the step's data is rebuilt when a member splits: the vehicle keeps
        # its state and takes the members' CURRENT particles (v1.0:
        # it went on pushing the particles the fragment had left)
        veh.dofs = self.dofs; veh._prepared = False
        self.__dict__.setdefault("_vehicles", {})[veh.name] = veh
        return veh

    # ------------------------------------------------------------------ #
    #  Output
    # ------------------------------------------------------------------ #

    def _frame_time(self, k, progress):
        """The time a quasi-static frame is recorded at: k + progress (the
        load factor) -- or, in a run whose steps include dynamic ones (1.2),
        the clock, which a quasi-static step does not move."""
        if self.plan is not None and self.staging.time_axis == "time":
            return float(self.staging.clock)
        return k + progress

    # ------------------------------------------------------------------ #
    #  Convergence measures, probes, and the dynamic loop's step end
    # ------------------------------------------------------------------ #

    @contextmanager
    def _reading(self):
        """A force evaluation that is not the step's own -- the frame's
        reactions and actions, a convergence measure: path-dependent member
        state (plastic hinges, the fibres, the shear laws) is read, not
        advanced; nor does it latch a fracture, remove a member or write the
        contact's allowances (member_contact: v1.12 and v1.18). `_record`
        accumulated hinge rotation at every output frame, so the answer
        depended on `output.every`. What a solver holds
        besides `_probing` is its `_probe_hold`. The frame solvers call it
        `_probe` too (`BeamSolver._probe`); the fibre solver has no `_probe`,
        which `analysis_steps._probe`, the buckling and modal tangents read
        as "put the fibres' states back yourself"."""
        was = self._probing
        undo = self._probe_hold()
        self._probing = True
        try:
            yield
        finally:
            self._probing = was
            if undo is not None:
                undo()

    def _probe_hold(self):
        """What a probe (`_reading`) holds besides `_probing`, and returns what puts it
        back: nothing, for the frame solvers (their hinges read `_probing`)."""
        return None

    def _norm(self, x):
        """The 2-norm the convergence measures take."""
        return np.linalg.norm(x)

    def _thermal_force_scale(self):
        """A thermal field's restraint force, ||k d_th||: the load's magnitude
        whether or not any of it survives to the answer."""
        return float(self._thermal_scale)

    def _residual_norm(self, r, free):
        """|r| over the free dofs."""
        return float(np.linalg.norm(r[free]))

    def _convergence_measures(self, F, free, ke, f_int=None):
        """(relative residual, kinetic ratio) on one force scale.

        The scale is max(|F_ext on the free dofs|, |F_int over ALL dofs|).
        Until v0.1 it was max(|F_ext[free]|, 1.0), which under
        displacement control -- F_ext identically zero -- made the residual an
        ABSOLUTE number of newtons and `tolerance: 1e-6` a demand for 1e-6 N
        of imbalance on a frame carrying kilonewtons. Lee's frame (case_16)
        sat at 100 N for 60000 iterations, converging to nothing. Including
        the internal force over the constrained dofs puts the reactions in
        the denominator, which is what the solid solver has done since its
        own version of this bug (core/solver.py, "Normalise by the internal
        force over ALL dofs"). One body for both solvers since v1.19.
        """
        if f_int is None:
            with self._reading():
                f_int = self.internal_force()[0]
        nrm = self._norm
        now = max(float(nrm(F[free])), float(nrm(f_int)), self._thermal_force_scale())
        # THE SCALE MAY NOT COLLAPSE WITH THE LOAD. A displacement-controlled
        # CYCLIC run passes through states where nothing is loaded at all --
        # the protocol returns to zero displacement between peaks -- and there
        # F_ext is zero, the internal force is round-off, and the relative
        # residual is that round-off divided by itself: 0.998 at a state whose
        # worst unbalanced force is 5e-11 N and whose displacements are zero
        # to four decimals in millimetres. The run then fails at the one state
        # it had exactly right (Cai et al.'s Tube A).
        #
        # The floor is a millionth of the largest force this analysis has had
        # to carry, so it is inert wherever anything IS loaded -- every number
        # in the library is unchanged -- and it is a scale rather than a number
        # of newtons, which matters in a library that refuses to convert units.
        peak = self._force_scale = max(self._force_scale, now)
        # A PRESTRESSED MEMBER UNDER NO EXTERNAL LOAD is self-equilibrated:
        # F_ext is zero and, once the concrete has shortened under the
        # tendons, so are the reactions and the section resultants -- the
        # only force left in f_int is the residual itself, and the relative
        # residual would be that divided by itself. The tendons' force is the
        # load this analysis carries, so it is the scale;
        # zero without tendons, which leaves every other model untouched.
        scale = max(now, 1e-6 * peak, self._tendon_force_scale, 1e-30)
        resid = self._residual_norm(F - f_int, free) / scale
        # AND NEITHER MAY THE MOTION SCALE. The kinetic ratio divides the
        # kinetic energy by the work scale `force x displacement`, and the
        # displacement half of that collapses in exactly the way the force
        # half does above: a stage whose structure is genuinely UNDEFORMED --
        # a moving load standing on a support, so the whole of it goes
        # straight into the reaction and nothing bends -- has ||q|| at the
        # 1e-12 floor, and a model at rest to 4e-20 J reports a kinetic ratio
        # of 8e-9 against a 1e-10 tolerance and fails at the one state it had
        # exactly right (the last station of case_26b). The
        # floor is a millionth of the largest displacement this analysis has
        # reached, so it is inert wherever anything IS deflected.
        motion = float(nrm(self.q[free]))
        self._motion_scale = max(self._motion_scale, motion)
        kr = ke / max(scale * max(motion, 1e-6 * self._motion_scale, 1e-12), 1e-30)
        return resid, kr

    def _removal_clock(self, k, steps, plan):
        """The time `_check_failure` reads at the forces of each of `steps`
        of analysis step k: the run's clock at the START of the step,
        `self.time` of `one_dynamic_step`."""
        p0 = (steps - 1) / self.n_steps
        return (k + p0) * self.total_time if plan is None else self.staging.clock + p0 * self.total_time

    def _event_stop(self, k, local, last, plan):
        """The first step in `local` .. `last` whose forces a scheduled
        removal falls due at (`_check_failure` reads `_removal_clock` >=
        time), or None: the compiled step ends with the step before it, and
        the loop takes that one (1.6)."""
        b = getattr(self, "breaking", None)
        rems = [r for r in (getattr(b, "_removals", None) or []) if not r[2]]
        if not rems:
            return None
        steps = np.arange(local, last + 1)
        t = self._removal_clock(k, steps, plan)
        stop = None
        for rem in rems:
            due = np.flatnonzero(t >= rem[0])
            if len(due):
                stop = int(steps[due[0]]) if stop is None else min(stop, int(steps[due[0]]))
        return stop

    def compiled_event_stop(self, k, local, last, plan):
        return self._event_stop(k, local, last, plan)

    def _reload_after_split(self, k, data):
        """What the step reloads before it starts: nothing, for the fibre
        solver (it reloads where it splits, `_between_steps`)."""

    def _project_diaphragms(self, acc):
        """The acceleration with rigid floors imposed: none here."""
        return acc

    def _between_steps(self, k, t, data):
        """BETWEEN two steps is the only instant at which the model is in one
        consistent state: the step's force vector has been used and the next
        one has not been built. Moving a bond here and rebuilding the step's
        data means the new particles get this step's gravity, and nothing
        that was computed for the old topology is used against the new one."""
        if (self._pending_splits or self._pending_debris) and self.apply_pending_splits(k*self.total_time + t):
            data[:] = self.step_data(k)
            self.current_fixed = data[0]

    def one_dynamic_step(self, k, local, data):
        """Step `local` of analysis step k as the loop takes it; `data` is
        [fixed, moving, prescribed, groups, ground], reloaded when a split
        moved mass. Returns the step's external force. One body for the loop
        and for the steps the compiled step hands back (1.6), and for both
        solvers."""
        plan = self.plan
        self._reload_after_split(k, data)
        fixed, moving, prescribed, groups, ground = data
        # `self.time` so that anything evaluated inside the internal
        # force can stamp itself -- today the fibre solver's fracture log,
        # which is useless without the instant each member went
        progress = local/self.n_steps; t = progress*self.total_time
        # THE LOAD AT THE START OF THE STEP, where the state is: the force of
        # step n is F(t_n). Read at t_(n+1) it made a ramp or a ground motion
        # first-order in dt -- convergence ratio 2.00 on halving the step,
        # 4.00 with this. The boundary conditions below still
        # take t_(n+1): they place the state the update arrives at.
        p0 = (local - 1)/self.n_steps
        # THE RUN'S CLOCK, not the step's: a removal is written in
        # seconds from the start of the run, and with the step's
        # clock one due in step 2 never fired and every fracture of
        # step 2 was logged at its time into the step.
        # The loads still take the step's own time.
        self.time = (k + p0)*self.total_time if plan is None else self.staging.clock + p0*self.total_time
        F = self._force(groups, ground, p0, p0*self.total_time); fint, _actions = self.internal_force()
        if self.damp_kind == "rayleigh":
            damp = self.alpha*self.mass*self.v + self.beta*(self.K @ self.v)
        elif self.damp_kind == "viscous":
            damp = self.zeta*self.mass*self.v
        else:
            damp = 0.
        # the ledger takes the state and the forces of ONE instant,
        # before the update below moves q
        self.diagnostics["energy"].update(self.q, self._ledger_force(F, fint, damp), fint)
        # Kinematic DOFs are advanced exclusively in _apply_bc below;
        # otherwise a prescribed velocity would receive both the
        # force integrator increment and its imposed increment.
        acc = self._project_diaphragms((F-fint-damp)/self.mass)
        acc[fixed] = 0.; self.v[fixed] = 0.
        # half the acceleration on a restart: v_(1/2) = a dt / 2.
        # Using a dt injects energy at every restart, and under
        # kinetic damping that is every few steps. Found by
        # tests/test_dimensional_smoke.py, where the centre of
        # mass came out (1 + 1/N) times Newton's answer -- exactly
        # the difference between sum(n) and sum(n + 1/2).
        self.v += (0.5 if self._half else 1.0) * self.dt*acc
        self._half = False
        self.commit_state()
        self._advance(self.dt); self._apply_bc(fixed, moving, prescribed, progress); self.step += 1
        if self.damp_kind in ("rayleigh", "viscous"):
            # the damping force's work over this step, which the
            # ledger booked as error until v0.4 -- on the step's
            # own velocity, before a split below rescales it (a stability
            # floor, debris' one rotary inertia), as the fibre kernel books
            # it inside the step it took. The frame solvers
            # booked it after their splits until v1.19; their kernel
            # still books it after its appendage carry, which differs from
            # this only by the work of massless tips (1e-20 kg, below
            # rounding: review of v1.19)
            self.diagnostics["energy"].damping(damp, self.dt*self.v)
        self._between_steps(k, t, data)
        if not np.isfinite(self.q).all() or not np.isfinite(self.v).all():
            location = first_nonfinite(displacement=self.q, velocity=self.v)
            finish_diagnostics(self.diagnostics, "NONFINITE", step=self.step, load_step=k+1,
                               message=f"{self.DYNAMIC_WHAT} became non-finite", location=location)
            raise NonFiniteFailure(f"{self.DYNAMIC_WHAT} became non-finite", location=location)
        return F

    def after_dynamic_step(self, k, local, F, fixed, plan, on_output, output_every,
                           on_progress, progress_every):
        """What the dynamic loop does after step `local` of analysis step k:
        the progress check and the output frame, when they fall there. One
        place for the step-by-step loop and the compiled one,
        and for both solvers."""
        t = (local/self.n_steps)*self.total_time
        if local % progress_every == 0 or local == self.n_steps:
            free = ~fixed; resid, _ = self._convergence_measures(F, free, float(.5*np.sum(self.mass*self.v*self.v)))
            self.residuals.append(float(resid)); self.kinetic.append(float(.5*np.sum(self.mass*self.v*self.v)))
            self.diagnostics["energy"].snapshot(self.step, self._ledger_kinetic())
            stop = bool(on_progress and on_progress(local, self.n_steps, resid, self.kinetic[-1], 0) is False)
        else:
            stop = False
        if local % output_every == 0 or local == self.n_steps:
            # the frame's state is at the END of the step: what its force
            # evaluation finds (a removal due now, a fracture) is stamped
            # there
            self.time = k*self.total_time + t if plan is None else self.staging.clock + t
            self.frame_equilibrium = None
            # THE LAST FRAME OF THE RUN CLOSES IT: a removal due at the very
            # end, a member at its capacity in the final state, are found
            # there -- no step follows to find them, and none whose forces
            # they could change (every other frame only reads)
            self._closing = self._closes_the_run(k, local)
            try:
                self._record(self.time, F, on_output)
            finally:
                self._closing = False
        if stop:
            # after this step's frame, not before it: the checkpoint
            # is taken after the step, so a frame skipped here was in
            # neither the stopped result nor the resumed one
            raise StopAnalysis()

    # ------------------------------------------------------------------ #
    #  Checkpoint and restore: one body for both solvers.
    #  What only one member family carries is its `_checkpoint_members` /
    #  `_restore_members`; how a solver queues and moves its splits is its
    #  `_checkpoint_split` / `_restore_split` / `_restore_queue`.
    # ------------------------------------------------------------------ #

    def _checkpoint_members(self, out):
        """The members' own path-dependent state: none here."""
        return out

    def _restore_members(self, ck):
        """The members' own path-dependent state: none here."""

    def _checkpoint_split(self, out):
        """The splits made and queued (the fibre solver's: the list of
        splits and the members whose split is queued)."""
        if self.bonds is not None:
            out["splits"] = np.array(self.splits, float).reshape(-1, 4)
            # a fracture latched in the frame's own force evaluation
            # (`_record`) splits at the end of the next step: a checkpoint in
            # between lost it
            if self._pending_splits:
                out["pending_splits"] = np.array(sorted(set(self._pending_splits)), dtype=np.int64)
        return out

    def _restore_split(self, ck):
        """The bonds, masses and particles a split moved: the
        fibre solver's, which rebuilds its reference stiffness whole."""
        el = np.asarray(ck["elements"], dtype=np.int64)
        self.mesh.elements[:, :2] = el
        self.bonds.bonds[...] = el
        self.physical_mass = np.asarray(ck["physical_mass"], float).copy()
        self.mass = np.maximum(self.physical_mass, 1e-20)
        self.splits = [(float(t), int(e), int(p), int(n))
                       for t, e, p, n in np.asarray(ck.get("splits", np.zeros((0, 4)))).reshape(-1, 4)]
        self._fragment_parent = {int(n): int(p) for _t, _e, p, n in self.splits}
        self._pending_splits = [int(e) for e in np.asarray(ck.get("pending_splits", np.zeros(0))).ravel()]
        if self._edofs is not None:
            for e in range(self.nel):
                self._edofs[e] = self.dofs.element_dofs(self.mesh.elements[e])
                self._x0[e] = self.mesh.nodes[self.mesh.elements[e]]
        self._rebuild_reference_stiffness()
        self.dt_bound = self._time_step_bound()
        self._bc_cache = None

    def _restore_queue(self, ck):
        """The splits queued for the end of the next step, once the broken
        members are restored: the fibre solver's are in `_restore_split`."""

    def _checkpoint(self):
        """COPIES of the state and the loop position, one
        body for both solvers since v1.19."""
        out = {"q": self.q.copy(), "v": self.v.copy(), "orient": self.orient.copy(),
               "step": np.array([self.step])}
        out = self._checkpoint_members(out)
        if self.bonds is not None:
            # a split moved bonds and mass: the state is not just q and v
            out["elements"] = self.mesh.elements[:, :2].astype(np.int64)
            out["physical_mass"] = self.physical_mass.copy()
        # THE SPLIT'S STABILITY FLOOR does not weigh, and a
        # parked particle's floor went with it (`_park_bodiless`): what a
        # split or a park changes, saved where either can have happened (the
        # fibre solver did not save it, the frame solvers only with spares)
        w = self.weightless_mass
        if (self.bonds is not None or self._parked) and w is not None and np.shape(w) == (self.nn, 3):
            out["weightless_mass"] = np.asarray(w, float).copy()
        # THE PARTICLES HELD WHERE THEY STOPPED and the massless members
        # carried rigidly (`_park_bodiless`, `_make_appendage`): the frame
        # solvers' checkpoint had neither, and a restart after a hanger was
        # cut let its joint and the crossbeam's tip go free -- 3 cm on a
        # 10 cm swing of the girder (the fibre solver saved
        # its parked particles since v1.18)
        if self._parked:
            out["parked"] = np.array([[n, *Q] for n, Q in sorted(self._parked.items())], float)
        if self._appendages:
            aps = self._appendages
            out["appendage_head"] = np.array([[ap["anchor"], ap["ref"], len(ap["members"]), len(ap["tips"])]
                                              for ap in aps], dtype=np.int64)
            out["appendage_members"] = np.array([e for ap in aps for e in ap["members"]], dtype=np.int64)
            out["appendage_tips"] = np.array([n for ap in aps for n in ap["tips"]], dtype=np.int64)
            out["appendage_rel"] = np.concatenate([np.asarray(ap["rel"], float).reshape(-1, 3) for ap in aps])
            out["appendage_orient0"] = np.concatenate([np.asarray(ap["orient0"], float).reshape(-1, 3, 3)
                                                       for ap in aps])
            out["appendage_d0"] = np.array([np.asarray(ap["d0"], float) for ap in aps]).reshape(-1, 3)
        out = self._checkpoint_split(out)
        if self.breaking is not None:
            out = self.breaking.checkpoint(out)
        if self.plan is not None:
            from . import analysis_steps
            out = analysis_steps.checkpoint(self, out)
        out.update(resume.loop_checkpoint(self))
        return {k: np.array(v, copy=True) for k, v in out.items()}

    def _restore(self, ck):
        q, v = np.asarray(ck["q"], float), np.asarray(ck["v"], float)
        if len(q) < self.ndof:
            # a state from a run without spare particles (a relaxed dead
            # state, where the fibre beam's `fracture.split` is refused) laid
            # onto one with them: the spares are parked and at rest. It
            # failed with "boolean index did not match" (found in review,
            # v1.0)
            q = np.concatenate([q, np.zeros(self.ndof - len(q))])
            v = np.concatenate([v, np.zeros(self.ndof - len(v))])
        self.q, self.v, self.step = np.array(q, float), np.array(v, float), int(ck["step"][0])
        self._restore_orientation(ck)
        if len(self.orient) < self.nn:
            self.orient = np.concatenate([self.orient, np.tile(np.eye(3), (self.nn - len(self.orient), 1, 1))])
        if self.bonds is not None and "elements" in ck:
            self._restore_split(ck)
        if "weightless_mass" in ck:
            w = np.asarray(ck["weightless_mass"], float).reshape(-1, 3)
            if len(w) < self.nn:
                # an initial state from a model without the spare particles
                # (`fingerprint._PADDED`): theirs are nothing
                w = np.concatenate([w, np.zeros((self.nn - len(w), 3))])
            self.weightless_mass = w.copy()
        if "parked" in ck or self._parked:
            rows = np.asarray(ck.get("parked", np.zeros((0, 7))), float).reshape(-1, 7)
            self._parked = {int(r[0]): r[1:].copy() for r in rows} or None
            self._park_cache = None
        if "appendage_head" in ck:
            head = np.asarray(ck["appendage_head"], np.int64).reshape(-1, 4)
            mem = np.asarray(ck["appendage_members"], np.int64).ravel()
            tip = np.asarray(ck["appendage_tips"], np.int64).ravel()
            rel = np.asarray(ck["appendage_rel"], float).reshape(-1, 3)
            o0 = np.asarray(ck["appendage_orient0"], float).reshape(-1, 3, 3)
            d0 = np.asarray(ck["appendage_d0"], float).reshape(-1, 3)
            aps, im, it = [], 0, 0
            for i, (a, ref, nm, nt) in enumerate(head):
                aps.append({"anchor": int(a), "ref": int(ref),
                            "members": [int(e) for e in mem[im:im + nm]],
                            "tips": [int(n) for n in tip[it:it + nt]],
                            "rel": rel[it:it + nt].copy(), "orient0": o0[it:it + nt].copy(),
                            "d0": d0[i].copy()})
                im += nm; it += nt
            self._appendages = aps
            self._app_cache = None
        if self.breaking is not None:
            self._deactivate(self.breaking.restore(ck))     # a broken member stays broken
        self._restore_queue(ck)
        self._restore_members(ck)
        if self.plan is not None:
            from . import analysis_steps
            analysis_steps.restore(self, ck)
        resume.loop_restore(self, ck)

    # ------------------------------------------------------------------ #
    #  The run: one loop for both solvers. A staged level
    #  of the fibre solver is a trial, commits its fibres, may be solved
    #  again when a strand broke at the commit, and latches fractures on
    #  the committed state (`_staged_trials` ... `_level_latch`).
    # ------------------------------------------------------------------ #

    #: the name the solver's messages give its loop
    SOLVER_WHAT = "beam"
    #: what a level that did not converge suggests (`_relax_one_level`)
    RELAX_HINT = ""
    #: a staged level is a trial thrown away and driven again from the last
    #: converged state (the fibre solver's `_trial_mode`)
    STAGED_TRIALS = False

    def _running(self):
        """What holds while the solver runs: nothing here."""
        return nullcontext()

    def _default_load_steps(self):
        """The levels a relaxation step takes when neither it nor the solver
        block says: none, one continuous relaxation."""
        return 0

    def _default_ramp(self, budget):
        """The iterations a level blends its prescribed displacements in
        over: a fifth of the budget, measured on Lee's frame."""
        return max(budget // 5, 1)

    def _level_failures(self):
        """The points failed in the accepted state: none here."""
        return 0

    def _level_commit(self):
        """A converged level becomes the accepted state."""
        self.commit_state()

    def _level_latch(self, t):
        """What a converged level latches on the committed state: nothing."""

    def run(self, on_output=None, output_every=200, on_progress=None, progress_every=500):
        with self._running():
            return self._run(on_output, output_every, on_progress, progress_every)

    def _run(self, on_output=None, output_every=200, on_progress=None, progress_every=500):
        started = time.perf_counter(); self.times, self.residuals, self.kinetic = [], [], []
        self.diagnostics = start_diagnostics(self.mode, "numpy")
        from .diagnostics import resume_ledger
        resume_ledger(self)      # a restart carries the ledger on
        if self.staging.mixed_procedures:
            led = self.diagnostics.get("energy")
            if led is not None:
                # kept by the time-stepping loops alone; a run that mixes
                # procedures would audit part of its history (review, 1.2)
                led.available = False
                led.reason = "the run mixes procedures (analysis steps); its energy is not audited"
        self.frame_equilibrium = None
        # A RESTART RESUMES. It used to begin again at step 0, t = 0 and
        # load factor 0 on top of the restored state -- the ground motion,
        # the vehicles and the staged loads all replayed.
        # `resume` is the loop position the checkpoint was taken at.
        resume = resume_position(self, len(self.steps))
        plan = self.plan
        if plan is not None:
            from . import analysis_steps
        prev = None
        for k in range(resume.k if resume else 0, len(self.steps)):
            r = resume if (resume is not None and k == resume.k) else None
            if plan is not None:
                # analysis steps (1.2): the clock, the procedure, the model change
                if prev is not None:
                    analysis_steps.end_step(self, prev)
                analysis_steps.begin_step(self, k, resumed=r is not None)
                if plan.procedure[k] == "static":
                    from .. import implicit
                    implicit.static_step(self, k, r, on_output, output_every, on_progress)
                    prev = k
                    continue
            prev = k
            self.step_index = k; fixed, moving, prescribed, groups, ground = self.step_data(k)
            self.current_fixed = fixed
            if r is None:
                self.step_start = self.step
            else:
                r.after_step_data(self)
            self._loop = [k, self.step_start, 0, self.step_start, 0]
            if self.mode == "relaxation":
                # VFIFE quasi-static path: current geometry -> pure-deformation
                # member forces -> particle update.  A fixed K solve would be
                # inconsistent once the member has rotated.
                if r is None:
                    self.v.fill(0.); self._half = True; self.ke_prev = 0.0
                # Per-step first, the solver block as the fallback. Until
                # v0.1 this read ONLY the solver block, so a step that
                # declared `load_steps: 50` ran as one continuous relaxation to
                # its full target -- fifty levels asked for, one delivered, and
                # nothing said so.
                step_cfg = self.steps[k]
                nstage = int(step_cfg.get("load_steps") or self.p.solver.load_steps or
                             self._default_load_steps())
                # the step's own tolerance first, as the solid solver reads it
                # (found in review)
                tol = float(step_cfg.get("tolerance") or self.tolerance)
                if nstage:
                    budget = int(step_cfg.get("max_steps_per_load") or
                                 self.p.solver.max_steps_per_load)
                    # The prescribed displacements are blended in from where
                    # the level finds them over the level's first `ramp`
                    # iterations, and the level may not declare itself
                    # converged inside that ramp -- the residual of a moving
                    # target is the ramp's, not equilibrium's (`_default_ramp`)
                    ramp = int(step_cfg.get("stage_ramp_iterations") or
                               self.p.solver.stage_ramp_iterations or
                               self._default_ramp(budget))
                    if self.STAGED_TRIALS:
                        self._trial_mode = True
                    cv_s = (self.zeta if self.damp_kind == "viscous" else
                            (self.alpha if self.damp_kind in ("kinetic", "rayleigh") else 0.0))
                    for stage in range(r.first_stage() if r else 1, nstage + 1):
                        progress = stage / nstage
                        rs = r.in_stage(stage) if r else None
                        # the count the commit is compared with: the ACCEPTED
                        # state's, also after a restart inside the level
                        # (`_committed_failures`)
                        redo = 0; n_failed_start = self._level_failures()
                        while True:
                            F = self._force(groups, ground, progress)
                            restarted = rs is not None
                            if rs is None:
                                self.v.fill(0.0); self._half = True; self.ke_prev = 0.0
                                start = [self.q[dofs].copy() for dofs, _v, _a, _m in prescribed]
                                # the fictitious mass follows the geometry (see
                                # BeamFamily._refresh_fictitious_mass)
                                self._reorthonormalize()
                                self._refresh_fictitious_mass()
                                it0, converged = 0, False
                            else:
                                start, it0, converged = rs.start, rs.iterations, rs.converged
                                resid, kr, ke = rs.resid, rs.kr, rs.ke
                                rs = None
                            self._loop = [k, self.step_start, stage, self.step - it0, 0]
                            self._stage_start_q = start
                            if not converged:
                                report = ((lambda resid_, ke_: on_progress(stage, nstage, resid_, ke_, self.n_resets))
                                          if on_progress else None)
                                converged, resid, kr, ke = self._relax_stage(
                                    F, fixed, moving, prescribed, progress, start, tol, budget, ramp, it0,
                                    progress_every, report, cv_s, stage,
                                    measures=(resid, kr, ke) if restarted else None)
                            self._loop_measures = (resid, kr, ke)
                            self._loop[4] = 1 if converged else 0     # converged, not yet recorded
                            self.residuals.append(float(resid)); self.kinetic.append(float(ke))
                            self.diagnostics["energy"].snapshot(self.step, ke)
                            if on_progress and on_progress(stage, nstage, resid, ke, self.n_resets) is False: raise StopAnalysis()
                            if not converged:
                                # both measures: printing only the residual made a
                                # level that failed on kinetic energy read as a
                                # contradiction
                                message = (f"{self.SOLVER_WHAT} load stage {stage}/{nstage} did not converge in "
                                           f"{budget} iterations: residual={resid:.3e} (tol {tol:.1e}), "
                                           f"kinetic_ratio={kr:.3e} (tol {self.kinetic_tolerance:.1e})")
                                location = worst_residual(F-self.internal_force()[0], ~fixed, 6)
                                finish_diagnostics(self.diagnostics, "MAX_STEPS", step=self.step, residual=resid,
                                                   kinetic=ke, load_step=stage, message=message, location=location)
                                raise ConvergenceFailure(message, location=location)
                            self._level_commit()
                            # A STRAND THAT BROKE AT THIS LEVEL IS CUT AT THE COMMIT, NOT INSIDE
                            # THE ITERATIONS (steel02 reads its rupture against the ACCEPTED
                            # state). The level just converged with it intact
                            # and its converged strain past the limit; the commit has now cut
                            # it. Recording here would pair the intact positions with the cut
                            # forces -- an unbalanced reaction (-169 kN on Li's A7). So the SAME
                            # level is solved again with the cut in the accepted state, and
                            # again while that cascades into more failures; only an equilibrium
                            # is recorded.
                            n_failed = self._level_failures()
                            if n_failed > n_failed_start and redo < 50:
                                redo += 1; n_failed_start = n_failed
                                self.log(f"  rupture: {n_failed} point(s) failed at load stage {stage}/{nstage}; "
                                         f"re-solving the stage without them (pass {redo})")
                                continue
                            # ON THE COMMITTED STATE, and only here, in a staged
                            # relaxation: every evaluation inside the stage above
                            # is a trial, so a latch set from one would record a
                            # fracture on a path the model does not believe in
                            self._level_latch(k + progress)
                            # A TIME STATION HAS CONVERGED -- after the commit,
                            # because the history is fed the committed state
                            if self.time_dependent:
                                self._record_time_station(self._current_age)
                            self.frame_equilibrium = True
                            self._loop[4] = 2                     # this level is done
                            self._record(self._frame_time(k, progress), F, on_output)
                            break
                    if self.STAGED_TRIALS:
                        self._trial_mode = False
                    continue
                # ONE LEVEL, to the full load (OrientedPoints._relax_one_level,
                # shared by the two solvers since v1.12)
                F = self._relax_one_level(
                    k, step_cfg, r, fixed, moving, prescribed, groups, ground, on_progress, progress_every,
                    self.SOLVER_WHAT, self.RELAX_HINT)
                if F is None:
                    continue                                      # it had converged
                self.commit_state()
                self.frame_equilibrium = True
                if self.time_dependent:
                    # the step's one time station: without it
                    # a continuous step's load never entered the history
                    self._record_time_station(self._current_age)
                self._loop[4] = 2
                self._record(self._frame_time(k, 1.0), F, on_output); continue
            # THE COMPILED STEP: when the model and the
            # step are ones it reproduces to the bit, the runs of steps between
            # two points where this loop has work of its own -- a progress
            # check, an output frame -- go through one compiled call each
            from . import compiled_step
            compiled_step.drive(self, k, [fixed, moving, prescribed, groups, ground], plan,
                                on_output, output_every, on_progress, progress_every, self.SOLVER_WHAT)
        if plan is not None and prev is not None:
            analysis_steps.end_step(self, prev)
        check_resumed(self, resume)
        status = "CONVERGED" if self.mode == "relaxation" else "COMPLETED_DYNAMIC"
        finish_diagnostics(self.diagnostics, status, step=self.step,
                           residual=self.residuals[-1] if self.residuals else None,
                           kinetic=self.kinetic[-1] if self.kinetic else None,
                           load_step=len(self.steps))
        self.wall_time = time.perf_counter()-started; return self

    # ------------------------------------------------------------------ #
    #  The output frame: one body for both solvers
    # ------------------------------------------------------------------ #

    def _record_scope(self):
        """What the whole frame is evaluated inside: nothing here (the
        fibre solver reads each force evaluation in its own `_reading`, so
        its fibres are put back before the next)."""
        return nullcontext()

    def _monitor_fields(self, actions):
        """The element fields an `element` monitor reads."""
        return self.nodal_fields(actions)

    def _record(self, t, F, callback):
        with self._record_scope():
            with self._reading():
                fint, actions_now = self.internal_force()
            residual = fint - F
            # THE SOIL IS A SUPPORT: its force on the pile is a
            # reaction, as a held dof's is -- the structure's own internal force
            # less the load, at the held dofs and at the springs' nodes
            from . import soil
            residual, soil_mask = soil.reaction_residual(self, residual)
            self.last_reaction_force = np.zeros((self.nn, 3))
            self.last_reaction_moment = np.zeros((self.nn, 3))
            held = self.current_fixed
            if self.plan is not None:
                from . import analysis_steps as _as
                dd = _as.dormant_dofs(self)
                if dd is not None:
                    held = held.copy(); held[dd] = False    # a dormant node is no support
            mask = self.dofs.view(held | soil_mask)
            rr = self.dofs.view(residual)
            self.last_reaction_force[mask[:, :3]] = rr[:, :3][mask[:, :3]]
            self.last_reaction_moment[mask[:, 3:]] = rr[:, 3:][mask[:, 3:]]
            strains = None
            frame_fields = None
            for name, mon in self.monitors.items():
                if mon.kind == "vehicle":
                    veh = getattr(self, "_vehicles", {}).get(mon.of)
                    mon.values.append(veh.value(mon.field, mon.axle) if veh is not None else float("nan"))
                    continue
                if mon.kind == "element":
                    # a member force history, over the members
                    # the region owns
                    from . import member_actions
                    if frame_fields is None:
                        frame_fields = self._monitor_fields(actions_now)
                    mon.values.append(member_actions.monitor_value(
                        frame_fields, mon.field, self._monitor_members[name], mon.reduce))
                    continue
                idx = np.unique(np.concatenate([self._region(r) for r in mon.regions]))   # a node once, however many regions name it
                if mon.kind == "fiber" and hasattr(self, "section_strains"):
                    if strains is None:
                        strains = self.section_strains()
                    mon.values.append(self._fiber_monitor_value(mon, idx, strains))
                    continue
                dof = self.dofs.node_dofs(idx, (self.dofs.components[mon.dof],)).ravel()
                if mon.kind == "reaction":
                    mon.values.append(float(residual[dof].sum()))
                else:
                    v = float(self.q[dof].mean())
                    if mon.relative_to:
                        ref = self.dofs.node_dofs(self._region(mon.relative_to),
                                                  (self.dofs.components[mon.dof],)).ravel()
                        v -= float(self.q[ref].mean())
                    mon.values.append(v)
            self.times.append(float(t))
            if self.time_dependent:
                self.ages.append(float(self._current_age))
            if callback:
                with self._reading():
                    actions = self.internal_force()[1]
                callback(self, self.step, t, actions)

    def _cell_actions(self):
        """One value per member (`member_actions`), read, not advanced."""
        with self._reading():
            return self.internal_force()[1]

    def _commit_members(self):
        """Hook for path-dependent member state (connector laws):
        the frame solver's members have none."""

    # ------------------------------------------------------------------ #
    #  Separation: a member's end moving onto its spare particle, and the
    #  debris it leaves
    # ------------------------------------------------------------------ #

    def carried_nodes(self):
        """The particles that are carried rather than integrated: those
        parked where they stopped (`_park_bodiless`) and the tips of the
        appendages (`_make_appendage`). They have no mass of their own."""
        nodes = set((self._parked or {}).keys())
        for ap in self._appendages:
            nodes.update(ap["tips"])
        return np.array(sorted(nodes), np.int64)

    def _energy_ledger(self):
        """The energy ledger of the run, or None before the run has one."""
        d = getattr(self, "diagnostics", None)
        return d.get("energy") if isinstance(d, dict) else None

    def _move_end(self, e, parent, new, share, pm, ledger):
        """End `parent` of member e moves onto the spare particle `new`.

        The member's share of the mass (`_element_mass_share`, and the line
        mass of a `with_mass` load on it, 1.1) leaves the parent and lands on
        the spare; the spare starts with the parent's position, velocity and
        orientation. Mass and momentum are conserved at the instant, nothing
        is invented: the mass that moves takes the velocity it already had.

        * NEVER BELOW ZERO: the parent's mass is a sum of shares, and the
          difference of two sums is -1e-10 kg as often as +0; a negative mass
          under gravity is an upward force, and over the 1e-20 kg floor it
          sent two Xihoumen particles to 190 km (the frame solvers' since
          v0.5; the fibre beam's from v1.18).
        * THE LEDGER IS RELABELLED, not charged: the spare's displacement
          jumps from 0 to its parent's in this assignment, which is no
          motion, and the energy ledger must not book work over it --
          Xihoumen's k = 3 run read +-1.6e11 J at every split before this
          (frames since v0.5; the fibre beam booked it until v1.18).
        * The spare descends from the parent: gravity written on a region the
          parent is in reaches it (`_with_fragments`); a nodal force does not.
        """
        end_share = share.copy()
        end_share[:3] += added_mass.move_line_mass(self, int(e), int(parent), int(new))
        pm[parent] = np.maximum(pm[parent] - end_share, 0.0)
        pm[new] += end_share
        pd = self.dofs.node_dofs(parent).ravel(); nd = self.dofs.node_dofs(new).ravel()
        self.q[nd] = self.q[pd]; self.v[nd] = self.v[pd]
        self.orient[new] = self.orient[parent]
        if ledger is not None and getattr(ledger, "_u", None) is not None:
            ledger._u[nd] = self.q[nd]
        self._fragment_parent[int(new)] = int(parent)
        # a staged run holds the particles no built member reaches, and the
        # spares are among them until a member moves onto one: the fragment's
        # end stayed clamped where it broke while its parent fell 300 m (the
        # review of v1.18; both beam solvers since staging, 1.2)
        if self.plan is not None and self.staging.dormant is not None:
            self.staging.dormant[int(new)] = False

    def _frames(self):
        """(R, theta_loc, L) of every member: `rotations.beam_frames_from_triads`
        on the points' orientations (the fibre solver's, shared since v1.19)."""
        conn = self.mesh.elements
        x = self.mesh.nodes[conn] + self.dofs.view(self.q)[conn][:, :, :3]
        return beam_frames_from_triads(x, self.orient[conn], self.R0)

    def step_data(self, k):
        """(fixed, moving, prescribed, load groups, ground motions) of analysis
        step k: both beam solvers' (one body since v1.19; the hosts
        give `_thermal_load` and `_line_load_local`)."""
        # `or`, not a .get default: a validated step always HAS the key (None
        # when unset), so the solver's load_steps was never the fallback and a
        # named step under it took `ramp_hold` -- full load at half the levels
        # (found in review)
        step = self.steps[k]; staged = bool(step.get("load_steps") or self.p.solver.load_steps)
        def inherited(items):
            return [{**x, "amplitude": "hold"} if k and x.get("amplitude") is None else x for x in items]
        plan = self.plan
        if plan is None:
            constraints = inherited(self.p.get("constraints") or []) + list(step.get("constraints") or [])
            loads = inherited(self.p.get("loads") or []) + list(step.get("loads") or [])
        else:
            # analysis steps (1.2): the kept items of earlier steps are
            # inherited too, at the value they ended their own step with
            from . import analysis_steps as _as
            constraints = [{**x, "_inherited": True} for x in
                           _as.inherited(self, k, inherited(self.p.get("constraints") or []), "constraints")] \
                + list(step.get("constraints") or [])
            loads = [{**x, "_inherited": True} for x in
                     _as.inherited(self, k, inherited(self.p.get("loads") or []), "loads")] \
                + list(step.get("loads") or [])
        constraints, loads = amp.timed_hold(step, constraints, loads)
        fixed = np.zeros(self.ndof, bool); moving = np.zeros(self.ndof, bool)
        prescribed = []; groups: Dict[str, np.ndarray] = {}; ground = []; movers = []
        valued = np.zeros(self.ndof, bool)           # dofs a constraint gives a `value`
        # the member-local line loads per amplitude, for the end actions
        # (v1.0: the fixed-end moments were missing from them)
        self._line_local = {}
        # the vehicles of THIS step only: a monitor on one that is not in it
        # records NaN, not the frozen state of an earlier step's
        self._vehicles = {}
        driven = self._driven = np.zeros(self.ndof, bool)
        self._thermal = []
        # The step's own time window. `age` defaults to where
        # the previous step left off, so a run reads as one clock and not as a
        # list of independent intervals.
        if self.time_dependent and getattr(self, "_ages_step", None) != k:
            # once per step: `step_data` is called again mid-step when a
            # member splits off, and the clock must not be re-read there
            from .creep import read_step_ages, refuse_moving_clock
            self._ages_step = k
            was = getattr(self, "_current_age", None)
            self._step_ages = read_step_ages(
                step, was, self._age0,
                latest=max(self.ages) if self.ages else getattr(self, "_clock_floor", None))
            if self.mode == "dynamic" and getattr(self, "_creep_who", None) is not None:
                refuse_moving_clock(step, self._step_ages, was, self._creep_who)   # v1.0
            if self.mode == "relaxation" and not staged and self._step_ages[1] is not None:
                # A CONTINUOUS relaxation step is ONE time station, at its end.
                # The age followed the iteration count and stopped wherever
                # the residual converged -- 301 days of a step to 1028.
                self._step_ages = (self._step_ages[1], None, self._step_ages[2])
            self._current_age = self._step_ages[0]
        for c in constraints:
            if len(c["dofs"]) != 6:
                raise ValueError(f"{self.ELEMENT_NAME} constraints require six DOF flags [ux,uy,uz,rx,ry,rz]")
            if c.get("record") is not None:
                # a support's own ground motion
                if self.mode != "dynamic":
                    raise ValueError("a constraint `record` (a support's own ground motion) requires "
                                     "solver.mode = dynamic")
                idx = self._region(c["region"])
                entries = ground_motion.support_prescriptions(
                    c, idx, lambda i, d: self.dofs.node_dofs(i, (self.dofs.components[d],)).ravel(),
                    self.mesh.nodes, self.total_time, self.__dict__.setdefault("_ground_cache", {}))
                for dofs, value, a, motion in entries:
                    fixed[dofs] = True; driven[dofs] = True
                    if motion != "displacement":
                        moving[dofs] = True
                    prescribed.append((dofs, value, a, motion))
                continue
            idx = self._region(c["region"]); a = amp.get(self.amplitudes, c.get("amplitude"), "beam constraint", staged)
            if plan is not None and c.get("_inherited"):
                # a released support frees its nodes of what they inherited --
                # from before the release: one kept after it holds (review, 1.2)
                freed = _as.freed_since(self, k, c.get("_from", -1))
                if freed is not None:
                    idx = np.setdiff1d(idx, freed)
                    if not len(idx):
                        continue
            val, motion = c.get("value"), c.get("motion", "displacement")
            if motion != "displacement" and self.mode != "dynamic":
                raise ValueError(f"{motion} constraints require solver.mode = dynamic")
            for d, locked in enumerate(c["dofs"]):
                if locked:
                    dofs = self.dofs.node_dofs(idx, (self.dofs.components[d],)).ravel()
                    fixed[dofs] = True
                    if motion != "displacement":
                        moving[dofs] = True
                    if val is not None:
                        valued[dofs] = True
                        prescribed.append((dofs, float(val[d] if hasattr(val, "__len__") else val), a, motion))
        for ld in loads:
            idx = self._region(ld.get("region", "all")); a = amp.get(self.amplitudes, ld.get("amplitude"), "beam load", staged)
            F = groups.setdefault(a.name, np.zeros(self.ndof))
            if ld.get("vector") is not None:
                v = np.asarray(ld["vector"], float); value = v if ld.get("per_node") else v/len(idx)
                self.dofs.add_nodal(F, idx, value, TRANSLATION)
            elif ld.get("moment") is not None:
                v = np.asarray(ld["moment"], float); value = v if ld.get("per_node") else v/len(idx)
                self.dofs.add_nodal(F, idx, value, ROTATION)
            elif ld.get("line_load") is not None:
                elements = self._members_in(idx)
                if not len(elements):
                    raise ValueError(f"beam line_load region {ld.get('region', 'all')!r} contains no complete members")
                q = np.asarray(ld["line_load"], float)
                built = None if plan is None else self.member_built
                F_held = F
                for e in elements:
                    if hasattr(self, "member_kind") and self.member_kind[e] not in self.BEAM_TYPES:
                        raise ValueError("line_load can only be applied to beam members")
                    if built is not None:
                        if not built[e]:
                            continue                    # a member not built carries nothing
                        # an inherited load on a member born now comes on over the step
                        if ld.get("_inherited") and e in self.staging.born_now:
                            a_r = amp.get(self.amplitudes, None, "beam load", staged)
                            F = groups.setdefault(a_r.name, np.zeros(self.ndof))
                        else:
                            F = F_held
                    L, R = self.length[e], self.R0[e]
                    ql = q if ld.get("coordinate_system", "global") == "local" else R @ q
                    fl = np.zeros(12)
                    fl[[0, 6]] = ql[0]*L/2
                    fl[[1, 7]] = ql[1]*L/2
                    fl[5], fl[11] = ql[1]*L**2/12, -ql[1]*L**2/12
                    fl[[2, 8]] = ql[2]*L/2
                    fl[4], fl[10] = -ql[2]*L**2/12, ql[2]*L**2/12
                    fl = self._line_load_local(fl, L, e)
                    T = repeated_rotation(R)
                    conn = self.mesh.elements[e]
                    dof = self.dofs.element_dofs(conn)
                    F[dof] += T.T @ fl
                    self._line_local.setdefault(
                        a.name if F is F_held or built is None else a_r.name,
                        np.zeros((self.nel, 12)))[e] += fl
            elif ld.get("temperature") is not None:
                # the step's own default amplitude, unlike a moving load's:
                # a temperature IS applied gradually, and under staged
                # relaxation each stage is then a warmer structure.
                self._thermal.append((amp.get(self.amplitudes, ld.get("amplitude"),
                                              "beam temperature", staged),
                                      self._thermal_load(idx, ld["temperature"])))
            elif ld.get("vehicle") is not None:
                # a vehicle with mass and suspension; it keeps
                # its state when the step's data is rebuilt (a split)
                movers.append((amp.get(self.amplitudes, "hold", "vehicle", staged),
                               self._vehicle(idx, ld, k)))
            elif ld.get("moving_load") is not None:
                # A moving load holds its magnitude by default: what varies is
                # where it acts, and the continuous step's `ramp_hold` would
                # have the first half of the crossing carry a fraction of the
                # axle -- a vehicle that fades in, which nobody means.
                movers.append((amp.get(self.amplitudes, ld.get("amplitude") or "hold",
                                       "beam moving_load", staged),
                               self._moving_load(idx, ld)))
            elif ld.get("pressure") is not None or ld.get("traction") is not None:
                if not hasattr(self, "_surface_load"):
                    raise ValueError("surface pressure/traction requires shell4 or solid elements")
                F += self._surface_load(idx, ld)
            elif ld.get("gravity") is not None:
                g = np.asarray(ld["gravity"], float)
                idx = self._with_fragments(idx)
                # added mass does not weigh
                mass = added_mass.weighed(self.dofs.field(self.physical_mass, TRANSLATION),
                                          getattr(self, "weightless_mass", None))[idx]
                self.dofs.add_nodal(F, idx, mass*g, TRANSLATION)
                if plan is not None and ld.get("_inherited"):
                    # the members born this step weigh from nothing to all of
                    # it over the step (Abaqus' reactivation), not at once
                    from . import analysis_steps as _as
                    W = _as.newborn_weight(self, idx, g)
                    if W is not None:
                        F -= W
                        a_r = amp.get(self.amplitudes, None, "beam gravity", staged)
                        groups.setdefault(a_r.name, np.zeros(self.ndof))[:] += W
            elif ld.get("ground_motion") is not None:
                if self.mode != "dynamic":
                    raise ValueError("ground_motion requires solver.mode = dynamic")
                ground.append(ground_motion.read_record(
                    ld["ground_motion"], self.__dict__.setdefault("_ground_cache", {})))
            else:
                raise ValueError(f"{self.ELEMENT_NAME} accepts vector, moment, line_load, moving_load, "
                                 "temperature, gravity and ground_motion loads")
        # the dofs a constraint fixes and gives no value: taken before the
        # nodes not yet built are added to the held set
        plain = fixed & ~valued & ~driven & ~moving
        out_groups = [(next(a for a in self.amplitudes.values() if a.name == n), f)
                      for n, f in groups.items()] + movers
        if plan is not None:
            from . import analysis_steps as _as
            dd = _as.dormant_dofs(self)
            if dd is not None:
                fixed[dd] = True                     # a node nothing built reaches is held
            out_groups += _as.extra_groups(self, k)
        self._hold_in_place(k, fixed, plain, prescribed)
        # A PARKED PARTICLE IS HELD WHERE IT STOPPED: it has no member and no
        # mass, but a load written on its node would still push it -- 1e22 m/s
        # under a nodal force (the review of v1.18). Held in place
        # as a support put under a node that has moved is, so every loop
        # places it the same way. The fibre solver's only (`HOLD_PARKED`):
        # the frame solvers park particles too, and carry them
        # (`_carry_appendages`), which a hold here would undo
        for n, Q in ((self._parked or {}) if self.HOLD_PARKED else {}).items():
            d = self.dofs.node_dofs(int(n)).ravel()
            fixed[d] = True
            for j in range(3):
                if Q[j] != 0.0:
                    prescribed.append((np.array([d[j]], np.int64), float(Q[j]), amp.IN_PLACE, "displacement"))
        return fixed, moving, prescribed, out_groups, ground

    def _moving_load(self, idx, ld):
        """A load group whose DISTRIBUTION is recomputed every step, rather
        than a fixed vector the amplitude scales; both beam
        solvers' since v1.19 (the fibre solver has no releases:
        `path_releases` gives it None)."""
        from types import SimpleNamespace
        from .moving_load import LoadPath, MovingLoad
        elements = self._members_in(idx)
        if hasattr(self, "member_kind"):
            elements = np.array([e for e in elements if self.member_kind[e] in self.BEAM_TYPES], int)
        # the carriageway is the chain of members as the road was laid: a
        # member that split off is still on it, and the axles that are over
        # it ride whatever particles it is on now (the fallen segment)
        conn0 = getattr(self, "_elements0", None)
        road = self.mesh if conn0 is None else SimpleNamespace(nodes=self.mesh.nodes, elements=conn0)
        path = LoadPath(road, elements)
        ml = MovingLoad(ld["moving_load"], path, self.dofs, self.mesh,
                        self.length, self.R0, self.total_time)
        # a member's end releases shape its share, as a line_load's
        from .moving_load import path_releases
        ml.releases = path_releases(self, path.elements)
        return ml

    def _age_groups(self, ids):
        """[(rows, shift, floor)]: the members `ids` by their own clock (1.2).
        A member built from the start has shift 0 and no floor; one born in a
        step is `shift` days older or younger than the model's clock, and
        nothing about it is dated before its birth age."""
        shift = self.staging.age_shift
        ids = np.asarray(ids, dtype=np.int64)
        if shift is None or (not np.any(shift[ids]) and not np.isfinite(self.staging.age_floor[ids]).any()):
            return [(ids, 0.0, -np.inf)]
        floor = self.staging.age_floor
        out = []
        keys = np.stack([shift[ids], floor[ids]], axis=1)
        for key in np.unique(keys, axis=0):
            rows = ids[(keys == key).all(axis=1)]
            out.append((rows, float(key[0]), float(key[1])))
        return out

    def _line_load_local(self, fl, L, e):
        """The fixed-end loads `fl` of member `e` as its ends take them: as
        they are (`BeamSolver` condenses them through its releases)."""
        return fl

    def _refresh_fictitious_mass(self):
        """The relaxation mass from the CURRENT geometry, for every line
        solver: the frame, the fibre-beam and the mixed one.

        The mass is the Gershgorin row sum of |K| over the relaxation safety
        squared, and until v0.1 it was computed once, from the
        reference configuration. That bound is only a bound for the
        reference configuration: a member that has rotated puts its axial
        stiffness into several global rows at once, and the row sums grow by
        up to a factor of about two. When they outgrow the margin the
        stiffest mode crosses the stability limit and oscillates with a
        period of two to four iterations; the kinetic damping then resets on
        every one of those peaks -- 16882 resets in 60000 iterations on
        Lee's frame -- and the global mode the level is about never moves.
        Measured: at the default safety the frame stalled at v/L = 0.13; at
        0.4 it reached 0.5; the stall moved with the safety, which is the
        signature. Recomputing the rows from the turned members at each load
        level keeps the bound a bound.

        Each member's block in `K` is replaced by the same block turned into
        its current frame (`_turned_blocks`: the host says which members
        turn and how), and the rows are taken from the sum, assembled first
        and |.| second as the setup mass is (a per-element sum would be up
        to 1.94x larger on the rotations: adjacent members' 6EI/L^2
        couplings cancel in the assembled matrix and add up in a
        per-element one). Everything else `K` holds stays: a soil spring's
        diagonal (core/soil), a member that is dormant or removed and was
        not taken out of it (`_stiffness_off` takes it out under stiffness
        damping). Until v1.19 the mixed solver rebuilt the rows from its
        members alone, which dropped the springs' -- the mass of a node on a
        stiff spring fell to its members' after the first level -- and the
        frame and fibre solvers did not refresh at all until v1.19. At the
        reference configuration this is the setup mass to round-off (the
        current frame is `beam_frames_from_triads`', the reference one
        `_frame`'s: they agree to 1e-15, to the bit only for a member along
        an axis). Relaxation only: a dynamic run's mass is physical.
        """
        if self.mode != "relaxation":
            return
        rows = np.asarray(abs(self.K + self._turned_delta()).sum(axis=1)).ravel()
        self.mass = np.maximum(rows / float(self.p.solver.relaxation_safety)**2, 1e-20)

    def _turned_delta(self):
        """What turning every member into its current frame adds to `K`
        (`_refresh_fictitious_mass`; the shell_line solver adds its line
        half's)."""
        delta = SparseAssembler(self.ndof)
        for dof, now, kdof, ref in self._turned_blocks():
            if kdof is not None and np.array_equal(kdof, dof):
                d = now - ref
                if d.any():
                    delta.add_matrix(dof, d)
                continue
            # K holds the block elsewhere (a split moved the member's ends
            # and K did not follow) or not at all (a member built while
            # stiffness damping was off): the turned block goes where the
            # member is, the reference one comes off where K has it
            delta.add_matrix(dof, now)
            if kdof is not None:
                delta.add_matrix(kdof, -ref)
        return delta.csr()

    #: where K holds each member's reference block (nel, 12), -1 where it
    #: does not: the frame solvers' (`BeamSolver.__init__`). None: every
    #: member at its element's dofs (the fibre solver, which rebuilds K over
    #: its live members)
    _k_dofs = None

    def _k_block(self, e):
        """The dofs at which K holds member `e`'s reference block, or None."""
        if self._k_dofs is None:
            return self.dofs.element_dofs(self.mesh.elements[e])
        d = self._k_dofs[e]
        return None if d[0] < 0 else d

    def _turned_blocks(self):
        """(dofs, block turned into the current frame, dofs where `K` holds
        the reference block or None, that block) of each live member that
        turns: the frame and fibre solvers' through their `_turned_k`
        (None: the member keeps its reference)."""
        R = self._frames()[0]
        for e in np.flatnonzero(np.asarray(self._live_members(), bool)):
            e = int(e)
            k = self._turned_k(e, repeated_rotation(R[e]))
            if k is not None:
                yield self.dofs.element_dofs(self.mesh.elements[e]), k[0], self._k_block(e), k[1]

    def _live_members(self):
        """(nel,) bool: the members that carry force."""
        return self.breaking.member_active

    def _grounded(self, held):
        """(nn,) bool: the particle is joined to a support through members
        that carry force -- the standing structure, not a piece of debris."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        el = self.mesh.elements[np.asarray(self._live_members(), bool), :2]
        g = coo_matrix((np.ones(len(el)), (el[:, 0], el[:, 1])), shape=(self.nn, self.nn))
        _, lab = connected_components(g, directed=False)
        return np.isin(lab, np.unique(lab[held]))


    def _debris_isotropic(self, grounded, ledger=None):
        """ON A PIECE OF DEBRIS EVERY PARTICLE'S ROTARY INERTIA IS ONE NUMBER.
        The rotary inertia is a diagonal fixed in GLOBAL
        axes -- each member's [rho Ip L/2, rho A L^3/24, ..] turned by its
        reference frame, with the stability floor of each axis on
        top -- and the step has no gyroscopic term. Both are right while the
        member keeps its frame, which the standing structure does. A
        fragment tumbles: its stiff bending turned onto the axis whose small
        torsional inertia (or floor) was set for it -- omega dt 5.5 and
        1e13 J on a 70-degree-over stringer, two ends of a free 10 m
        stringer ringing at 16 Hz up to 70 m/s. An inertia that is a scalar
        times the identity is the same in every frame and has no gyroscopic
        term, so on debris (particles joined to no support through members
        that carry force) the three are made the largest of them. Raised
        with the angular momentum kept (omega scaled by I_old / I_new); the
        spin that goes is booked as dissipated. The standing structure
        keeps its per-axis inertia: a modal analysis or an intact dynamic
        run is unchanged (review: an isotropic floor everywhere halved a
        beam's torsional frequency). True when an inertia was raised.
        Both beam solvers since v1.18 (the fibre beam's debris went
        without it in 1.17)."""
        if grounded is None or bool(np.all(grounded)):
            return False
        pm = self.physical_mass.reshape(self.nn, self.dofs.stride)
        if pm.shape[1] < 6:
            return False
        rot = pm[:, 3:6]
        top = rot.max(axis=1)
        tips = {n for ap in self._appendages for n in ap["tips"]}
        parked = getattr(self, "_parked", None) or {}
        loose = ~np.asarray(grounded, bool)
        # a node a staged run has not built yet is held by the staging, not by
        # a constraint, and is no debris: what it carries now is its floor,
        # and the member born on it adds its share on top (the review of
        # v1.18)
        dormant = getattr(getattr(self, "staging", None), "dormant", None)
        if dormant is not None:
            loose &= ~np.asarray(dormant, bool)
        lift = np.flatnonzero(loose & (top > 0.0) & (rot.min(axis=1) < top))
        lift = [int(n) for n in lift if int(n) not in tips and int(n) not in parked]
        if not lift:
            return False
        for n in lift:
            old = pm[n, 3:6].copy()
            nd = self.dofs.node_dofs(n).ravel()[3:6]
            w0 = self.v[nd].copy()
            pm[n, 3:6] = top[n]
            self.v[nd] = w0 * old / top[n]
            if ledger is not None:
                ledger.dissipate(float(0.5 * np.sum(old * w0 ** 2 - top[n] * self.v[nd] ** 2)))
        self.mass = np.maximum(self.physical_mass, 1e-20)
        self._reload_step = True
        return True

    def _hanging_devices(self, device, held, candidates, bonds=None):
        """The live members of `candidates` (massless: `device`) that hold
        nothing up any more: one of their ends is no support and has no
        massive live member left on it. A MASSLESS BAR IS A SUPPORT BETWEEN
        TWO BODIES: a bearing pressing a girder onto a tower, a
        link. At a Xihoumen tower whose girders had fallen, the bearings and
        crossbeam left between the corbels were a pinned mechanism of
        stability masses, swinging 13 m in a twentieth of a second; in the
        fibre beam a girder member fracturing over its bearing left the
        bearing's particle with the connector alone and no mass, and the
        step bound fell to 5e-16 s. Taking one such member
        out changes no other's answer (the test reads massive members only),
        so one pass in member order finds them all."""
        el = self.mesh.elements[:, :2] if bonds is None else bonds
        active = np.asarray(self._live_members(), bool)
        # a nodal mass is a body the member holds up (the review of v1.18:
        # a lumped mass at a girder's end fell through its bearing)
        extra = getattr(self, "additional_mass", None)
        out = []
        for e in np.flatnonzero(np.asarray(candidates, bool) & active):
            for n in el[e]:
                n = int(n)
                if held[n] or (extra is not None and n < len(extra) and np.any(extra[n])):
                    continue
                on = ((el[:, 0] == n) | (el[:, 1] == n)) & active
                if not (~device[on]).any():
                    out.append(int(e))
                    break
        return out

    def _park_bodiless(self, nodes, t, device=None):
        """PARTICLES THAT ARE NO BODY STOP WHERE THEY ARE. A particle with no
        member left that carries force, no support, no mass of its own (a
        nodal mass) and nothing on it but massless members (devices) is the
        leftover joint of connectors that broke or were removed: its mass
        was only ever the stability floor, and whatever velocity it had when
        its last member went it would keep for ever -- a crossbeam joint in
        the k = 3 run slid 1300 m along the sea floor at 70 m/s. It is parked:
        zero mass, zero velocity, held at its position every step; the
        broken connectors on it are marked removed (cause 2), so the viewer
        stops drawing them. A particle a MASSIVE member left (a weight on a
        cable that snapped with split: false) is a body and keeps falling.
        `device`: the massless members (the frame solvers' `_device_members`;
        the fibre beam's connectors). In the fibre beam
        `step_data` holds the parked particle where it stopped (since the
        review of v1.18: a load on its node pushed it); the frame solvers
        carry theirs (`_carry_appendages`)."""
        if not nodes:
            return
        if device is None:
            if getattr(self, "_kg_ref", None) is None:
                return
            device = self._device_members()
        parked = getattr(self, "_parked", None)
        if parked is None:
            parked = self._parked = {}
        el = self.mesh.elements[:, :2]; active = np.asarray(self._live_members(), bool)
        held = self._split_held()
        tips = {n for ap in self._appendages for n in ap["tips"]}
        extra = getattr(self, "additional_mass", None)
        pm = self.physical_mass.reshape(self.nn, self.dofs.stride)
        qv = self.dofs.view(self.q); vv = self.dofs.view(self.v)
        dropped = {int(ev[1]) for ev in self.breaking.failure_events if ev[3] >= 2}
        new, drop = [], []
        for n in sorted(int(x) for x in nodes):
            if n in parked or n in tips or held[n]:
                continue
            on = (el[:, 0] == n) | (el[:, 1] == n)
            if (on & active).any() or not device[on].all():
                continue
            if extra is not None and n < len(extra) and np.any(extra[n]):
                continue
            new.append(n)
            drop.extend(int(e) for e in np.flatnonzero(on) if int(e) not in dropped and self.breaking.member_failed[e])
        if not new:
            return
        w = getattr(self, "weightless_mass", None)
        for n in new:
            pm[n] = 0.0; vv[n] = 0.0
            parked[n] = qv[n].copy()
            if w is not None and np.shape(w) == (self.nn, 3):
                w[n] = 0.0                      # its floor went with it
        drop = sorted(set(drop))
        self.breaking.failure_events.extend([float(t), e, 0.0, 2.0] for e in drop)
        self.mass = np.maximum(self.physical_mass, 1e-20)
        self._park_cache = None

