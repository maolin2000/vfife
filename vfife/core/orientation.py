"""The orientation of the structural points: a mixin for
every solver with six dofs per point.

Each point carries a rotation matrix, advanced multiplicatively by the spin
of the step, R <- exp(omega dt) R, driven by the spatial moment directly --
the particle equation of Ting & Wu (2020) as written. The rotation part of
the flat state vector q is the logarithm of that matrix: kept for output,
monitors, prescribed values and the convergence norm, derived after every
change, never integrated. The frame solvers (BeamSolver and its subclasses)
and the fibre-beam solver share this through inheritance, so there is one
copy of the update, the boundary-condition semantics and the checkpoint
convention.
"""
from __future__ import annotations

import numpy as np

from ..base.rotations import exp_map, log_map, orthonormalize
from . import amplitude as amp
from .components import HasComponents

try:
    from ..kernels.numba_rotations import advance_orientations as _advance_kernel
except ImportError:                                   # pragma: no cover
    _advance_kernel = None


class OrientedPoints(HasComponents):
    """Requires self.nn, self.dofs (a six-dof DofLayout), self.q, self.v,
    self.dt on the host class."""

    #: the multi-point constraints (`core.constraints.ConstraintLayer`, 1.21),
    #: made by `constraint_layer()` the first time they are asked for; None until then
    ties = None

    def internal_force(self):
        """The members' internal force (`_member_forces`, each solver's own)
        and what the attached components add to it (contact; core/components)."""
        out, actions = self._member_forces()
        for fn in self._each("line_force"):
            out = fn(self, out)
        return out, actions

    def commit_state(self):
        """Commit the step's history: the members' (`_commit_members`), then
        the components' (core/components)."""
        out = self._commit_members()
        for fn in self._each("commit"):
            fn()
        return out

    def _mode_text(self) -> str:
        """The `mode` line of the summary. A static run is built as a
        relaxation one and driven by `implicit.run`; it was summarised as
        "relaxation; dt=1.0000e+00, 1 steps" (found in review)."""
        if (getattr(self, "p", None) or {}).get("solver", {}).get("static_run"):
            return "static (Newton on a finite-difference tangent, vfife/implicit.py)"
        return f"{self.mode}; dt={self.dt:.4e}, {self.n_steps} steps"

    def _shrink_step_for_damping(self, omega_max):
        """Cut `self.dt` by what the EXPLICIT damping terms cost in stability.

        Called after the damping coefficients are parsed, because the step was
        being set one line before them. The dynamic branch of every solver on
        this base applies `alpha M v + beta K v` and the viscous `coeff M v`
        from the lagged velocity, so all three are explicit here and all three
        go in. `elements.stability.damped_step_factor` has the derivation; it
        returns exactly 1.0 when nothing is damped, so an undamped run's step
        does not move by a bit.
        """
        if self.mode == "relaxation":
            self.damping_step_factor = 1.0
            return
        f = self.damping_step_factor = self._damping_step_factor(omega_max)
        if f < 1.0:
            self.dt = float(self.dt * f)
            if hasattr(self, "omega_max_sharp") and callable(getattr(self, "log", None)):
                self.log(f"  damping       step x{f:.3f} for beta at omega_max {self.omega_max_sharp:.4g} rad/s "
                         f"(power iteration; Gershgorin {self.omega_max_gershgorin:.4g})")
        # n_steps * dt is the run's duration exactly
        from ..elements.stability import aligned_step
        self.stable_dt = self.dt
        self.dt, self.n_steps = aligned_step(float(self.p.solver.total_time), self.dt)

    def _damping_step_factor(self, omega_max):
        """The factor alone, from this solver's parsed damping; 1.0 if none.

        With a stiffness-proportional term the Gershgorin `omega_max` the
        caller hands in is replaced by the top frequency itself (power
        iteration on the reference K and the mass in force),
        never above the bound it replaces. Only then: an undamped or
        mass-damped run's step does not move by a bit."""
        from ..elements.stability import damped_step_factor, power_omega_max
        kind = getattr(self, "damp_kind", None)
        rayleigh = kind == "rayleigh"
        from .damping import bounds as damping_bounds
        if rayleigh and damping_bounds(self)[1] > 0.0 and omega_max > 0.0 \
                and getattr(self, "K", None) is not None:
            sharp = power_omega_max(self.K, self.mass, ~self._held_dofs())
            if sharp > 0.0:
                self.omega_max_sharp, self.omega_max_gershgorin = sharp, float(omega_max)
                omega_max = min(float(omega_max), 1.05 * sharp)
        # the largest alpha and beta of the model's regions (core/damping)
        a, b = damping_bounds(self) if rayleigh else (0.0, 0.0)
        return damped_step_factor(
            omega_max,
            alpha=a,
            beta=b,
            coeff=getattr(self, "zeta", 0.0) if kind == "viscous" else 0.0)

    def _held_dofs(self):
        """Degrees of freedom that are constrained in EVERY step.

        They never move, so they cannot destabilise the explicit step -- and
        they are exactly where the tightest stiffness-to-mass ratio tends to
        sit, because a support node carries a share of the mass of one element
        and the stiffness of the ground. On the building of v0.1, ten
        of the twenty tightest degrees of freedom were fixed ones, and letting
        them set the step made it 47% smaller than the model needs.

        Only the always-fixed ones count: a dof released in a later step has to
        be stable when it is released.
        """
        held = None
        for k in range(len(self.steps)):
            fixed = np.zeros(self.ndof, bool)
            constraints = list(self.p.get("constraints") or []) \
                + list(self.steps[k].get("constraints") or [])
            plan = self.plan
            freed = plan.released_by[k] if plan is not None and k < len(plan.released_by) else None
            n_model = len(self.p.get("constraints") or [])
            for i, c in enumerate(constraints):
                if len(c.get("dofs", ())) != 6 or c.get("value") is not None:
                    continue                      # prescribed motion still moves
                idx = self._region(c["region"])
                if freed is not None and len(freed) and i < n_model:
                    idx = np.setdiff1d(idx, freed)   # released in a step (1.2): it moves then
                for d, locked in enumerate(c["dofs"]):
                    if locked:
                        fixed[self.dofs.node_dofs(idx, (self.dofs.components[d],)).ravel()] = True
            held = fixed if held is None else (held & fixed)
        return held if held is not None else np.zeros(self.ndof, bool)

    def _split_held(self):
        """(nn,) bool: particles a split must treat as the world's -- any dof
        constrained in the CURRENT step, prescribed motion and ground records
        included. Not `_held_dofs`, which is the step bound's question
        (never moves) and skips a constraint with a `value`: a member that
        broke at a driven support kept its particle and went on carrying,
        its reaction 5.5x its capacity. The current step's,
        not any step's: a shore released in an earlier step is free now, and
        counted as a support it was split off and left as a massless
        particle drifting at its last velocity, never parked."""
        out = np.zeros(self.nn, bool)
        cons = list(self.p.get("constraints") or [])
        steps = getattr(self, "steps", None) or []
        k = int(getattr(self, "step_index", 0) or 0)
        if 0 <= k < len(steps):
            cons += list(steps[k].get("constraints") or [])
        for c in cons:
            if any(c.get("dofs") or ()):
                out[np.asarray(self._region(c["region"]), np.int64)] = True
        return out

    def _hold_in_place(self, k, fixed, plain, prescribed):
        """A SUPPORT PUT UNDER A NODE THAT HAS MOVED HOLDS IT WHERE IT STANDS
        (1.14). `plain` are the dofs a constraint of step k
        fixes without a `value`; `fixed` is the step's whole held set, the
        nodes not yet built included.

        A plain constraint that takes a dof which was FREE at the end of the
        step before holds it at the displacement it has then, and keeps
        holding it there for as long as plain constraints hold it. The
        constraint used to hold zero -- the design position -- so an arch
        foot hinged for the erection and then cast in was turned back
        through everything it had turned, 85 mm at the rib's control points
        of a real arch, with nothing in the case asking for it. To bring a
        support to a position is a `value`.

        What does not change: a dof held from the start, or from the step
        that builds its node (a support is where it is: `_build` puts the
        node there), stays at zero; and a dof that a `value` moved and a
        plain constraint then takes returns to zero as before -- it was
        never free.

        A rotation is held in place when all three of the node's are (the
        clamp, which is the identity otherwise); one or two held components
        hold the spin, which never did turn anything back.

        The held values are written as prescribed displacements of the
        built-in `hold` amplitude, one a dof, so every loop places them the
        way it places any other. They are taken ONCE a step (the step's data
        is rebuilt when a member splits off) and are in the checkpoint."""
        st = self.__dict__.get("_hold")
        if st is None or len(st["at"]) != self.ndof:
            old = st
            st = self._hold = {"at": np.full(self.ndof, np.nan), "step": -1, "seen": {}}
            if old is not None:                     # the model grew spare particles: carried over
                n = min(len(old["at"]), self.ndof)
                st["at"][:n] = old["at"][:n]; st["step"] = old["step"]
                for j, m in old["seen"].items():
                    g = np.zeros(self.ndof, bool); g[:n] = m[:n]; st["seen"][j] = g
        at = st["at"]
        stride = self.dofs.stride
        if st["step"] != k:
            prev = st["seen"].get(k - 1)
            if prev is not None and k > 0:
                newly = plain & ~prev
                pl = plain.reshape(self.nn, stride); nw = newly.reshape(self.nn, stride)
                take = np.zeros((self.nn, stride), bool)
                take[:, :3] = nw[:, :3]
                clamp = pl[:, 3:6].all(axis=1) & nw[:, 3:6].any(axis=1)
                take[clamp, 3:6] = True
                take = take.ravel()
                at[take] = self.q[take]
            at[~plain] = np.nan
            # a rotation that is no longer all three held is the spin's again
            av = at.reshape(self.nn, stride)
            loose = ~plain.reshape(self.nn, stride)[:, 3:6].all(axis=1)
            av[loose, 3:6] = np.nan
            st["step"] = k
        st["seen"] = {j: m for j, m in st["seen"].items() if j == k - 1}
        st["seen"][k] = np.array(fixed, bool, copy=True)
        held = np.flatnonzero(np.isfinite(at) & (at != 0.0))
        if len(held):
            for d in held:
                prescribed.append((np.array([d], np.int64), float(at[d]), amp.IN_PLACE, "displacement"))

    def _hold_checkpoint(self, ck):
        st = self.__dict__.get("_hold")
        # a run of one step has none to write -- unless it began from a state
        # in which a support was holding (`initial_state`; found in review)
        if st is not None and (len(getattr(self, "steps", None) or ()) > 1 or np.isfinite(st["at"]).any()):
            on = np.isfinite(st["at"])
            ck["hold_at"] = np.where(on, st["at"], 0.0)
            ck["hold_on"] = on
            ck["hold_step"] = np.array([st["step"]], np.int64)
            for j, m in st["seen"].items():
                ck["hold_seen_now" if j == st["step"] else "hold_seen_before"] = np.array(m, bool, copy=True)
        return ck

    def _hold_restore(self, ck):
        self.__dict__.pop("_hold", None)
        if "hold_at" not in ck:
            return
        at = np.asarray(ck["hold_at"], float)
        if len(at) != self.ndof:
            return
        at = np.where(np.asarray(ck["hold_on"], bool), at, np.nan)
        k = int(np.asarray(ck["hold_step"]).ravel()[0])
        seen = {}
        if "hold_seen_now" in ck:
            seen[k] = np.asarray(ck["hold_seen_now"], bool).copy()
        if "hold_seen_before" in ck:
            seen[k - 1] = np.asarray(ck["hold_seen_before"], bool).copy()
        self._hold = {"at": at.copy(), "step": k, "seen": seen}

    def _members_in(self, idx):
        """The members whose two nodes are both in `idx`, judged on the
        connectivity the mesh was READ with: a member that split off sits on
        spare particles that are in no region, and it is still the member of
        the carriageway, the loaded span, the heated deck it was.
        Its forces go to whatever particles it is on now."""
        conn = getattr(self, "_elements0", None)
        conn = self.mesh.elements if conn is None else conn
        selected = np.zeros(self.nn, bool); selected[idx] = True
        return np.flatnonzero(selected[conn[:, :2]].all(axis=1))

    def _members_of(self, region):
        """The members of the region NAMED `region`: the ones the mesh
        recorded for it (a physical group of lines, a `lines` group -- what
        `assign` means by the name), else `_members_in` of its points. By
        the points alone a group of chords also holds every brace between
        two of its nodes, whenever that brace is built."""
        rec = (getattr(self.mesh, "element_regions", None) or {}).get(region)
        if rec is not None and rec.get("truss2") is not None:
            # the LINE block's rows alone: in a model with shells or solids
            # the name may also hold rows of theirs, numbered on their own
            ids = np.unique(np.asarray(rec["truss2"], np.int64).ravel())
            return ids[ids < int(self.nel)]
        return self._members_in(self._region(region))

    def _with_fragments(self, idx):
        """`idx` plus the split-off particles that descend from a particle in
        it: what a mass-proportional load on a region reaches."""
        fp = getattr(self, "_fragment_parent", None)
        if not fp:
            return idx
        inside = set(np.asarray(idx).tolist())
        extra = []
        for new, parent in fp.items():
            p = parent
            while p in fp:
                p = fp[p]
            if p in inside:
                extra.append(new)
        return np.concatenate([np.asarray(idx, np.int64), np.asarray(extra, np.int64)]) if extra else idx

    def _init_orientation(self):
        #: every structural point's orientation, advanced multiplicatively
        #: by the step's spin (`_advance`); the rotation part of `q` is its
        #: logarithm, derived from it after every change
        self.orient = np.tile(np.eye(3), (self.nn, 1, 1))
        self._rot_scratch = np.zeros((self.nn, 3))

    def _restore_orientation(self, ck):
        # a checkpoint from before v0.1 has no orientation: it is
        # the exponential of the vector it did save
        self.orient = (np.asarray(ck["orient"]).copy() if "orient" in ck
                       else exp_map(self.dofs.view(self.q)[:, 3:]))
        self._rot_scratch = np.zeros((self.nn, 3))

    def _apply_bc(self, fixed, moving, prescribed, progress, start=None, approach=1.0):
        """Install the boundary conditions at `progress`.

        `start` and `approach` ramp a prescribed DISPLACEMENT from where the
        dof was when the load level began (`start`) to this level's target
        over the level's first iterations, instead of jumping there in one
        step. A jump is a discontinuity at the driven node, and on a slender
        frame it re-excites the stiffest local modes every level; the kinetic
        damping then resets on those instead of on the global mode the level
        is actually about. The solid solver has had this since v0.1
        (`approach` in core/solver.py); v0.1 gave it to the frames.
        """
        fixed_zero = fixed & ~moving
        self.v[fixed_zero] = 0.
        # translations: held at zero. Rotations: the orientation is a matrix
        # and a component of its logarithm cannot be zeroed on its own; what
        # a fixed rotational dof holds at zero is the SPIN component
        # (`_advance` masks it), which keeps a clamped node at the identity
        # and a symmetry-plane node (two components held) exactly in the
        # one-parameter subgroup. A single held component is held in the
        # spin, and its logarithm component stays at second order.
        # the masks depend only on the step's constraint set; built once per step
        cache = getattr(self, "_bc_cache", None)
        if cache is None or cache[0] is not fixed or cache[1] is not moving or cache[-1] is not prescribed:
            fz = self.dofs.view(fixed_zero)
            trans_zero = np.flatnonzero(fixed_zero.reshape(self.nn, self.dofs.stride)[:, :3].ravel())
            trans_zero = (trans_zero // 3) * self.dofs.stride + trans_zero % 3
            clamped = np.flatnonzero(fz[:, 3:].all(axis=1))
            touched = np.zeros(self.nn, bool)
            for dofs, _value, _a, _motion in prescribed:
                d = np.asarray(dofs); rot = (d % self.dofs.stride) >= 3
                touched[d[rot] // self.dofs.stride] = True
            # the supports holding a node where it found it (`_hold_in_place`):
            # constant, one dof each, placed in one assignment -- a later
            # constraint on a whole region is hundreds of them
            ip = [i for i, e in enumerate(prescribed) if e[2] is amp.IN_PLACE]
            ip_d = np.array([int(prescribed[i][0][0]) for i in ip], np.int64)
            ip_v = np.array([float(prescribed[i][1]) for i in ip], float)
            inplace = set(ip)
            rest = [i for i in range(len(prescribed)) if i not in inplace]
            cache = self._bc_cache = (fixed, moving, trans_zero, clamped, np.flatnonzero(touched), ip_d, ip_v, rest,
                                           prescribed)
        _f, _m, trans_zero, clamped, touched, ip_d, ip_v, rest, _p = cache
        qv = self.dofs.view(self.q)
        self.q[trans_zero] = 0.0
        if len(clamped):
            self.orient[clamped] = np.eye(3); qv[clamped, 3:] = 0.0
        # the velocities of the kinematic dofs, seeded from v (which a restart
        # restores: `_apply_bc` leaves them in v at the end of every step)
        kv = getattr(self, "_kinematic_v", None)
        if kv is None:
            kv = self._kinematic_v = np.array(self.v, float, copy=True)
        elif kv.shape != self.v.shape:
            grown = np.array(self.v, float, copy=True); grown[:len(kv)] = kv[:len(grown)]
            kv = self._kinematic_v = grown
        if len(ip_d):
            self.q[ip_d] = ip_v; self.v[ip_d] = 0.
        for i in rest:
            dofs, value, a, motion = prescribed[i]
            target = value*a(progress)
            if motion == "displacement":
                if start is not None and approach < 1.0:
                    target = start[i] + (target - start[i]) * approach
                self.q[dofs] = target
                # a support following a displacement record moves with the
                # record's velocity; a constant target is at rest
                rate = getattr(a, "rate", None)
                self.v[dofs] = value*rate(progress) if rate is not None else 0.
            elif hasattr(a, "motion"):
                # a support's velocity or acceleration RECORD: integrated
                # exactly, second order and step-independent like a
                # displacement record
                u, du = a.motion(progress)
                self.q[dofs] = value*u; self.v[dofs] = value*du; kv[dofs] = value*du
            elif motion == "velocity":
                self.q[dofs] += self.dt*target; self.v[dofs] = target
                kv[dofs] = target
            else:
                # the dof's own velocity is kept here: the dynamic loop
                # zeroes v on every held dof before `_advance`, so reading
                # it back from `v` integrated from rest at every step and an
                # acceleration constraint on a frame never moved its node
                # (found by v1.0; the truss solver had it right)
                self.q[dofs] += self.dt*kv[dofs] + .5*self.dt**2*target
                kv[dofs] += self.dt*target; self.v[dofs] = kv[dofs]
        if len(touched):
            # a prescribed rotation component: the orientation is rebuilt
            # from the vector it was written into
            self.orient[touched] = exp_map(qv[touched, 3:])
        if self.ties is not None:
            # a rigid link's slaves' tied components set from their masters,
            # which a support may just have moved (core/constraints)
            self.ties.correct(fixed)

    def _ledger_force(self, F, fint, damp):
        """The external force the energy ledger books. At a
        support that follows its own record, the ground does work on the
        structure through the reaction; the ledger books what the structure
        receives there, fint + damp, and leaves the support particle's own
        kinetic energy out (`_ledger_kinetic`) -- the ground carries it."""
        driven = getattr(self, "_driven", None)
        if driven is None or not driven.any():
            return F
        out = np.array(F, float, copy=True)
        out[driven] = (fint + damp)[driven] if np.ndim(damp) else fint[driven]
        return out

    def _ledger_kinetic(self):
        driven = getattr(self, "_driven", None)
        v2 = self.mass * self.v * self.v
        if driven is not None and driven.any():
            v2 = np.where(driven, 0.0, v2)
        return float(.5 * np.sum(v2))

    def _advance(self, dt):
        """One position update from the velocities: translations additively,
        orientations multiplicatively by the spin of the step,
        R <- exp(omega dt) R, with the logarithm written back into q. The
        spin's held components were zeroed by the caller (v[fixed] = 0)."""
        qv = self.dofs.view(self.q); vv = self.dofs.view(self.v)
        qv[:, :3] += dt * vv[:, :3]
        if _advance_kernel is not None:
            # one scalar pass (kernels/numba_rotations): the numpy version
            # below is a dozen array operations and, on a few hundred points,
            # cost as much as the element kernel
            _advance_kernel(np.ascontiguousarray(vv[:, 3:]), float(dt), self.orient, self._rot_scratch)
            qv[:, 3:] = self._rot_scratch
        else:
            self.orient = exp_map(dt * vv[:, 3:]) @ self.orient
            qv[:, 3:] = log_map(self.orient)

    def sync_orientation(self):
        """Rebuild every point's orientation from the rotation part of q.
        For code that writes q directly (tests, restarts from a vector
        state): the orientation is the state, q's rotation part its
        logarithm, and this makes the two agree again."""
        self.orient = exp_map(self.dofs.view(self.q)[:, 3:])

    def _reorthonormalize(self):
        """Called once per load level: the product of a level's exp factors
        is orthogonal to round-off times the step count; this resets it."""
        self.orient = orthonormalize(self.orient)
        self.dofs.view(self.q)[:, 3:] = log_map(self.orient)

    def _closes_the_run(self, k, local):
        """True for the last dynamic step of the run: step `local` is the
        last of analysis step k and no dynamic step comes after it. The
        frame written there CLOSES the dynamic history -- a removal due at
        its very end, a member at its capacity in its final state, are
        found by that frame's force evaluation, because no step follows to
        find them (every other frame only reads). A
        relaxation or a static step after it starts from what the frame
        left: a removal at the end of the dynamic step of [dynamic,
        relaxation] was dropped while this asked for the last step of all
        (found in review)."""
        if local != self.n_steps:
            return False
        mode = self.p.solver.mode
        return not any((st.get("procedure") or mode) == "dynamic" for st in self.steps[k + 1:])

    # ------------------------------------------------- a level of a staged one
    _relax_what = "staged relaxation"
    _relax_hint = ""

    def _relax_kinetic(self):
        return float(0.5*np.sum(self.mass*self.v*self.v))

    def _relax_compiled(self, *args):
        """(converged, residual, kinetic ratio, kinetic energy) of the level
        from the compiled relaxation, or None where there is none."""
        return None

    def _relax_stage(self, F, fixed, moving, prescribed, progress, start, tol, budget, ramp, it0=0,
                     progress_every=500, report=None, cv_s=None, stage=0,
                     what=None, measures=None):
        """One level of a staged relaxation, from the current state to the
        loads `F` and the constraints at `progress`: iterations it0 + 1 ..
        budget of the explicit scheme with kinetic damping, the prescribed
        displacements blended in from `start` over the first `ramp`. Returns
        (converged, residual, kinetic ratio, kinetic energy). `report(resid,
        ke)` is asked every `progress_every` iterations and stops the run
        when it answers False. The body of the staged loop of `run` since
        v0.1; a method since 1.7, so that the implicit static solver
        can fall back to it for an increment Newton cannot take.

        One body for the frame and the fibre solvers (the
        fibre states are evaluated in trial mode throughout and committed
        by the caller). What differs is the solver's: `_relax_what` and
        `_relax_hint` in the messages, `_relax_acceleration`,
        `_relax_kinetic`, and `_relax_compiled` -- the compiled relaxation
        (1.9), which the frames have."""
        from .diagnostics import finish as finish_diagnostics, NonFiniteFailure, StopAnalysis, first_nonfinite
        if what is None:
            what = self._relax_what
        if cv_s is None:
            cv_s = (self.zeta if self.damp_kind == "viscous" else
                    (self.alpha if self.damp_kind in ("kinetic", "rayleigh") else 0.0))
        if measures is not None and it0 > ramp and measures[0] <= tol and measures[1] <= self.kinetic_tolerance:
            # restarted from a checkpoint written at the progress check of
            # the iteration that converged the level (1.9)
            return True, measures[0], measures[1], measures[2]
        done = self._relax_compiled(F, fixed, moving, prescribed, progress, start, tol, budget, ramp, it0,
                                    progress_every, report, cv_s, stage, what, measures)
        if done is not None:
            return done
        c1_s = 1.0/(1.0 + .5*cv_s*self.dt); c2_s = c1_s*(1.0 - .5*cv_s*self.dt)
        converged = False
        resid, kr, ke = measures if measures is not None else (np.inf, np.inf, 0.0)
        free = ~fixed
        for it in range(it0 + 1, budget + 1):
            approach = min(1.0, it / ramp)
            fint, _ = self.internal_force()
            self.diagnostics["energy"].update(self.q, F, fint)  # same instant as the forces
            acc = self._relax_acceleration(F, fint)
            acc[fixed] = 0.0; self.v[fixed] = 0.0
            # the relaxation's mass-proportional term, as in the continuous
            # branch of `run`
            if cv_s > 0.0:
                v_o = self.v.copy()
                self.v = (c2_s*self.v
                          + (0.5 if self._half else 1.0)*c1_s*self.dt*acc)
                self.diagnostics["energy"].damping(
                    0.5*cv_s*self.mass*(v_o + self.v), self.dt*self.v)
            else:
                self.v += (0.5 if self._half else 1.0) * self.dt*acc
            self._half = False
            self._advance(self.dt)
            self._apply_bc(fixed, moving, prescribed, progress, start, approach); self.step += 1
            ke = self._relax_kinetic()
            if not np.isfinite(ke) or not np.isfinite(self.q).all():
                location = first_nonfinite(displacement=self.q, velocity=self.v)
                finish_diagnostics(self.diagnostics, "NONFINITE", step=self.step, load_step=stage,
                                   message=f"{what} became non-finite", location=location)
                raise NonFiniteFailure(f"{what} became non-finite{self._relax_hint}", location=location)
            if ke < self.ke_prev and self.ke_prev > 0.0:
                self.diagnostics["energy"].dissipate(self.ke_prev)
                self.v.fill(0.0); self._half = True; self.n_resets += 1; ke = 0.0
            self.ke_prev = ke
            # `fint` is from before this iteration's update: one iteration
            # stale, which at a 1e-5 tolerance is nothing, and it halves the
            # cost of every staged iteration -- the check used to evaluate the
            # internal force a second time
            resid, kr = self._convergence_measures(F, free, ke, fint)
            # before the report: a checkpoint written there carries THIS
            # iteration's measures, and a restart from it sees the level
            # converged if it was (1.9; it ran on past it)
            self._loop_measures = (resid, kr, ke)
            if it % progress_every == 0:
                # sampled at the progress checks, as the result format says
                self.residuals.append(float(resid)); self.kinetic.append(float(ke))
                if report is not None and report(resid, ke) is False:
                    raise StopAnalysis()
            # NEVER INSIDE THE RAMP: the residual of a moving target is the
            # ramp's, not equilibrium's. And `fint` above is from before this
            # iteration's boundary conditions were applied, so on iteration 1
            # of a stage it is the PREVIOUS stage's equilibrium: with no
            # external load -- pure displacement control -- that is zero
            # against zero, the residual is 0/0, and the stage declared
            # itself converged before the prescribed displacement had done
            # anything; the reaction recorded for the first stage of every
            # staged fibre-beam run was 0 N. A fibre level's
            # `ramp` is 1 unless `stage_ramp_iterations` says more.
            if it > ramp and resid <= tol and kr <= self.kinetic_tolerance:
                converged = True; break
        return converged, resid, kr, ke

    # ---------------------------------------------------- one-level relaxation
    def _relax_acceleration(self, F, fint):
        """(F - fint) / m of a relaxation iteration; a solver with rigid
        diaphragms projects it."""
        return (F - fint) / self.mass

    def _relax_after_bc(self):
        """What a solver does after the supports of a relaxation iteration
        (the frames: pending splits, parked particles and appendages)."""

    def _relax_one_level(self, k, step_cfg, r, fixed, moving, prescribed, groups, ground,
                         on_progress, progress_every, what, hint=""):
        """Analysis step k as ONE relaxation to its full load (no
        `load_steps`): iterate until the residual and the kinetic ratio are
        inside their tolerances, the loads' curves having settled. Returns
        the external force of the converged state, or None when a resumed
        run had already converged this step.

        One body for the frame and the fibre solvers: each
        had its own copy, 60 lines that differed in the two hooks above and
        in the name in the messages (`what`; `hint` is what the frames add
        to a non-finite one)."""
        from .diagnostics import (finish as finish_diagnostics, ConvergenceFailure, NonFiniteFailure,
                                  StopAnalysis, first_nonfinite, worst_residual)
        # THIS STEP'S OWN BUDGET, the solver's as the fallback: the
        # step's `n_steps` was read by the solid solver alone, and a
        # 150 000-iteration step ran out at the solver's 300.
        # Without a solver-level one the schema has
        # written the SUM of the steps' budgets there.
        self.n_steps = int(step_cfg.get("n_steps") or self.__dict__.setdefault("_n_steps_solver", self.n_steps))
        # the mass-proportional term of a relaxation: `coeff` under
        # viscous, and `alpha` under kinetic or rayleigh -- the
        # combination v0.1 made for softening problems, which the
        # frame, fibre and truss relaxations never applied.
        # Implicit midpoint, as the viscous one was.
        cv = (self.zeta if self.damp_kind == "viscous" else (self.alpha if self.damp_kind in ("kinetic", "rayleigh") else 0.0))
        c1 = 1.0/(1.0 + .5*cv*self.dt)
        c2 = c1*(1.0 - .5*cv*self.dt)
        converged = False
        # the step's own tolerance, as the staged branch and the solid
        # solver read it (found in review)
        tol = float(step_cfg.get("tolerance") or self.tolerance)
        if r is not None and r.phase == 2:
            return None                                       # it had converged
        for local in range(self.step - self.step_start + 1, self.n_steps + 1):
            progress = local / self.n_steps
            F = self._force(groups, ground, progress)
            fint, _ = self.internal_force()
            self.diagnostics["energy"].update(self.q, F, fint)  # same instant as the forces
            acc = self._relax_acceleration(F, fint)
            acc[fixed] = 0.0; self.v[fixed] = 0.0
            v_old = self.v.copy() if cv > 0.0 else None
            # HALF the acceleration on a restart, not the whole of it.
            # A central-difference start is v_(1/2) = a dt / 2; using
            # a dt injects energy every time the integrator restarts,
            # and under kinetic damping that is every few steps. The
            # solid and truss paths have always done this; the frame
            # paths did not, and tests/test_dimensional_smoke.py found
            # it -- the centre of mass came out (1 + 1/N) times
            # Newton's answer, which is exactly the difference between
            # sum(n) and sum(n + 1/2).
            self.v = (c2*self.v
                      + (0.5 if self._half else 1.0) * c1*self.dt*acc)
            self._half = False
            self._advance(self.dt)
            self._apply_bc(fixed, moving, prescribed, progress); self.step += 1
            self._relax_after_bc()
            ke = float(0.5 * np.sum(self.mass * self.v * self.v))
            if not np.isfinite(ke) or not np.isfinite(self.q).all():
                location = first_nonfinite(displacement=self.q, velocity=self.v)
                finish_diagnostics(self.diagnostics, "NONFINITE", step=self.step,
                                   message=f"{what} dynamic relaxation became non-finite", location=location)
                raise NonFiniteFailure(f"{what} dynamic relaxation became non-finite{hint}", location=location)
            if v_old is not None:
                # c1/c2 is the implicit-midpoint viscous update: the
                # force it applied is zeta m (v_old + v_new) / 2
                self.diagnostics["energy"].damping(
                    0.5*cv*self.mass*(v_old + self.v), self.dt*self.v)
            if self.damp_kind == "kinetic" and ke < self.ke_prev and self.ke_prev > 0.0:
                self.diagnostics["energy"].dissipate(self.ke_prev)
                # and the energy it had is gone: `ke_prev` held the
                # pre-reset value, the next half-step was always below
                # it, and the run reset on every step -- 73 941 resets
                # in 74 043 iterations
                self.v.fill(0.0); self._half = True; self.n_resets += 1; ke = 0.0
            self.ke_prev = ke
            free = ~fixed; resid, _kr = self._convergence_measures(F, free, ke)
            # A residual is meaningful only after every applied curve has settled.
            settled = all(a.settled(progress) for a, _ in groups)
            kinetic_ratio = _kr
            diagnostic_written = local % progress_every == 0 or local == self.n_steps
            if diagnostic_written:
                self.residuals.append(float(resid)); self.kinetic.append(ke)
                self.diagnostics["energy"].snapshot(self.step, ke)
                if on_progress and on_progress(local, self.n_steps, resid, ke, self.n_resets) is False: raise StopAnalysis()
            if settled and resid <= tol and kinetic_ratio <= self.kinetic_tolerance:
                # A converged run may stop before the first progress interval.
                # Persist its terminal diagnostics for verification/audit.
                if not diagnostic_written:
                    self.residuals.append(float(resid)); self.kinetic.append(ke)
                    self.diagnostics["energy"].snapshot(self.step, ke)
                converged = True; break
        if not converged:
            message = f"{what} relaxation reached {self.n_steps} iterations with residual={resid:.3e} > tolerance={tol:.3e}"
            location = worst_residual(F-self.internal_force()[0], free, 6)
            finish_diagnostics(self.diagnostics, "MAX_STEPS", step=self.step, residual=resid,
                               kinetic=ke, load_step=k+1, message=message, location=location)
            raise ConvergenceFailure(message, location=location)
        return F
