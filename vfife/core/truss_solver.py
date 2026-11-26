"""Independent two-node truss solver for the first standalone bar workflow.

This is intentionally separate from :mod:`solver`: a truss has no host solid,
no rotational degrees of freedom and no continuum stress recovery.  Keeping the
small first implementation explicit prevents the embedded-reinforcement
interpolation from being accidentally reused as structural connectivity.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np

from .problem import Problem
from .diagnostics import StopAnalysis
from .member_failure import Breaking
from .components import HasComponents

from . import amplitude as amp
from . import selectors
from . import member_axes
from .mesh import Mesh, scatter_add
from .diagnostics import (UserError, start as start_diagnostics, finish as finish_diagnostics,
                          ConvergenceFailure, NonFiniteFailure, first_nonfinite,
                          worst_residual)
from ..base.dof import DofLayout
from ..elements import truss
from ..materials.points import initialise_points, validate_restored_state


# one monitor record for every solver (core/monitors)
from .monitors import Monitor  # noqa: F401  (the name this module exported)
from . import monitors as monitor_records


class TrussSolver(HasComponents):
    """CPU explicit/dynamic-relaxation solver for independent ``truss2`` bars."""

    def __init__(self, problem, mesh: Mesh, material, log: Callable = print):
        problem = Problem.of(problem)        # typed, and its dict
        if problem.case.element.type != "truss2":
            raise ValueError("TrussSolver requires element.type = truss2")
        if not (material.name == "linear_elastic" or getattr(material, "uniaxial", False)):
            raise ValueError("standalone truss2 needs linear_elastic or a uniaxial material")
        if problem.solver.backend != "numpy":
            raise ValueError("standalone truss2 currently supports solver.backend = numpy")
        self.p, self.mesh, self.material, self.log = problem, mesh, material, log
        self.pre = truss.precompute(mesh.nodes, mesh.elements,
                                    problem.case.element.area)
        # a standalone truss as cables: the effective prestress
        # is the strain offset the embedded-tendon path already carries, and
        # tension_only clips compression at the element
        if problem.case.element.prestress is not None:
            self.pre.prestrain = float(problem.case.element.prestress) / float(material.E)
        self.pre.tension_only = bool(problem.case.element.tension_only)
        self.nn, self.nel, self.n_gp, self.n_nodes_e = (mesh.n_nodes, mesh.n_elem,
                                                         1, 2)
        self.dofs = DofLayout.translations(self.nn)
        self.x0_e = np.ascontiguousarray(mesh.nodes[mesh.elements])
        self.state, self.material_points = initialise_points(
            material, self.nel, kinematics="uniaxial", backend="numpy",
            char_length=self.pre.char_len,
        )
        self.u = np.zeros((self.nn, 3))
        self.v = np.zeros((self.nn, 3))
        self.rebar = []
        self.step = self.step_start = self.step_index = 0
        self._resuming = False
        self._half = True
        self.ke_prev = 0.0
        self.n_resets = 0
        self.frozen = np.zeros(self.nn, dtype=bool)
        self.erosion = None
        self.ground_groups = []
        self._setup_regions()
        self._setup_mass_and_dt()
        self._setup_breaking()

    # ------------------------------------------------------------------ #
    #  Elastic-brittle breaking and scheduled removal (core/member_failure)
    # ------------------------------------------------------------------ #
    def _setup_breaking(self):
        """`element.failure.tensile_strength` for every bar."""
        e = self.p["element"]
        self.breaking = Breaking(self.nel, self.mesh, self.p.case.removals, self.log)
        self.member_active = self.breaking.member_active
        self.capacity_ratio = self.breaking.capacity_ratio
        self.failure_events = self.breaking.failure_events
        if e.get("failure"):
            self.breaking._bar_cap[:] = float(e["failure"]["tensile_strength"]) * self.pre.area
        self.pre.area = self.pre.area.copy()          # writable: a broken bar's area goes to zero

    def _check_failure(self, sig):
        """After a dynamic force evaluation: the bar forces against their
        capacities. A broken bar has zero area from then on -- the force it
        carries is zero and its stress is still reported, so the strain it
        broke at stays visible."""
        b = self.breaking
        if self.mode != "dynamic" or not b.can_break or sig is None:
            return
        end_actions = np.zeros((self.nel, 12))
        end_actions[:, 6] = sig[:, 0, 0] * self.pre.area
        end_actions[:, 0] = -end_actions[:, 6]
        if getattr(self, "_probing", False):
            # a frame reads; the step breaks (as the frame
            # and the fibre solvers: a bar broken by the frame after step n
            # carried nothing in step n + 1, and the end state depended on
            # `output.every` -- 0.1 mm on the hanging weight; found in review)
            b.check(self.step * self.dt, end_actions, observe=True)
            return
        gone = b.check(self.step * self.dt, end_actions)
        if len(gone):
            self._deactivate(gone)
            b.diagnostics(self.diagnostics)

    def _deactivate(self, ids):
        self.pre.area[np.asarray(ids, int)] = 0.0

    def _region(self, name):
        try:
            return self.mesh.regions[name]
        except KeyError:
            raise ValueError(f"unknown region {name!r}. Available: "
                             f"{', '.join(sorted(self.mesh.regions))}") from None

    def _additional_nodal_mass(self):
        """Nodal and hydrodynamic masses: directional and
        weightless ones too; no rotary inertia on a truss."""
        from . import added_mass
        out, _rot, self.weightless_mass = added_mass.table(
            self.p, self._region, self.mesh.nodes, self.mesh.elements[:, :2], rotations=False)
        return out

    def _setup_regions(self):
        self.amplitudes = amp.build(self.p.case.amplitudes)
        predefined = dict(self.mesh.regions)
        predefined.setdefault("all", np.arange(self.nn, dtype=np.int64))
        predefined.setdefault("surface", np.arange(self.nn, dtype=np.int64))
        self.mesh.regions = selectors.resolve_all(self.mesh.nodes,
                                                  self.p.get("regions") or {},
                                                  predefined)
        self.monitors = {}
        for name, spec in self.p.get("output", {}).get("monitors", {}).items():
            self.monitors[name] = monitor_records.from_spec(spec)
        # element monitors read the bars the region owns
        self._monitor_members = {}
        if any(m.kind == "element" for m in self.monitors.values()):
            from .member_actions import region_members
            e0 = np.asarray(self.mesh.elements[:, :2], dtype=np.int64).copy()
            self._monitor_members = {n: region_members(self.mesh, m.regions, e0)
                                     for n, m in self.monitors.items() if m.kind == "element"}
        self.steps = self.p.get("steps") or [{"name": "step1"}]
        self.apply_step(0)

    def apply_step(self, k):
        self.step_index = k
        if not self._resuming:
            self.step_start = self.step
        self._resuming = False
        step = self.steps[k]
        self.step_name = step.get("name") or f"step{k + 1}"
        staged = bool(step.get("load_steps") or self.p.solver.load_steps)
        if getattr(self, "mode", None) == "relaxation" and hasattr(self, "n_steps"):
            # this step's own budget, the solver's as the fallback
            self.n_steps = int(step.get("n_steps") or self.__dict__.setdefault("_n_steps_solver", self.n_steps))
        def base(items):
            # Model-level dead load ramps in the first step, then remains on.
            return [{**item, "amplitude": "hold"}
                    if item.get("amplitude") is None and k > 0 else item
                    for item in items]
        constraints = base(self.p.get("constraints") or []) + list(step.get("constraints") or [])
        loads = base(self.p.get("loads") or []) + list(step.get("loads") or [])
        self.fixed = np.zeros((self.nn, 3), dtype=bool)
        self.moving = np.zeros((self.nn, 3), dtype=bool)
        self.motion_velocity = np.zeros((self.nn, 3))
        self.ground_groups = []
        self.prescribed = []
        for c in constraints:
            if len(c["dofs"]) != 3:
                raise ValueError("truss2 constraints require three translational DOF flags [ux,uy,uz]")
            if c.get("record") is not None:
                raise ValueError("a constraint `record` (a support's own ground motion) is "
                                 "implemented for frame, fibre-beam and shell models; this is a truss model")
            idx = self._region(c["region"])
            a = amp.get(self.amplitudes, c.get("amplitude"), "truss constraint", staged)
            val = c.get("value")
            motion = c.get("motion", "displacement")
            if motion != "displacement" and self.p.solver.mode != "dynamic":
                raise ValueError(f"{motion} constraints require solver.mode = dynamic")
            for d, locked in enumerate(c["dofs"]):
                if not locked:
                    continue
                self.fixed[idx, d] = True
                if motion != "displacement":
                    self.moving[idx, d] = True
                if val is not None:
                    v = float(val[d] if hasattr(val, "__len__") else val)
                    self.prescribed.append((idx, d, v, a, motion))
        grouped: Dict[str, np.ndarray] = {}
        for ld in loads:
            if ld.get("moment") is not None:
                raise ValueError("nodal moment loads require beam2 elements")
            idx = self._region(ld.get("region", "all"))
            a = amp.get(self.amplitudes, ld.get("amplitude"), "truss load", staged)
            F = grouped.setdefault(a.name, np.zeros((self.nn, 3)))
            if ld.get("vector") is not None:
                vec = np.asarray(ld["vector"], dtype=float)
                F[idx] += vec if ld.get("per_node") else vec / len(idx)
            elif ld.get("body_force") is not None or ld.get("gravity") is not None:
                density = (np.asarray(ld["body_force"], dtype=float)
                           if ld.get("body_force") is not None
                           else self.material.rho * np.asarray(ld["gravity"], dtype=float))
                F += self._body_force(idx, density)
                if ld.get("gravity") is not None:
                    # the nodal masses in the region weigh too, as they do on
                    # every frame solver; until v1.0 a truss's did not
                    from . import added_mass
                    extra = added_mass.weighed(self._additional_nodal_mass(),
                                               self.weightless_mass)
                    F[idx] += extra[idx] * np.asarray(ld["gravity"], dtype=float)
            elif ld.get("ground_motion") is not None:
                if self.p.solver.mode != "dynamic":
                    raise ValueError("ground_motion requires solver.mode = dynamic")
                gm = ld["ground_motion"]
                direction = np.asarray(gm.get("direction", [1, 0, 0]), dtype=float)
                if direction.shape != (3,) or np.linalg.norm(direction) == 0:
                    raise ValueError("ground_motion.direction must be a non-zero 3-vector")
                data = np.loadtxt(gm["path"], dtype=float, encoding="utf-8")
                if data.ndim != 2 or data.shape[1] < 2 or len(data) < 2:
                    raise ValueError("ground_motion path needs at least two [seconds, acceleration] rows")
                if np.any(np.diff(data[:, 0]) <= 0):
                    raise ValueError("ground_motion time values must strictly increase")
                scale = float(gm.get("scale", 1.0))
                self.ground_groups.append((data[:, 0], data[:, 1] * scale, direction))
            else:
                # name the load that was given: a temperature (or a moving
                # load) was told "traction and pressure require solid
                # exterior faces" (found in review)
                kind = next((k for k in ("temperature", "moving_load", "vehicle", "line_load", "moment",
                                         "traction", "pressure") if ld.get(k) is not None), "this load")
                hint = (" A bar's temperature is carried when the bars sit beside a beam "
                        "(a `mixed_line` model); a model of bars alone does not "
                        "take one yet." if kind == "temperature" else "")
                raise ValueError(f"truss2 does not take a `{kind}` load; use vector, body_force, "
                                 f"gravity or ground_motion.{hint}")
        self.load_groups = [(self.amplitudes[name], F) for name, F in grouped.items()]

    def _body_force(self, nodes, density):
        """Consistent equivalent load for a constant 3-D body force on bars.

        A line bar represents volume ``A L``.  Its linear two-node shape
        functions give one half of ``b A L`` to each end, independent of how
        the member was subdivided.  The selected region must contain whole
        bars; this avoids silently loading a fraction of a physical member.
        """
        member = np.zeros(self.nn, dtype=bool)
        member[np.asarray(nodes, dtype=np.int64)] = True
        active = member[self.mesh.elements].all(axis=1)
        if not active.any():
            raise ValueError("truss body_force/gravity region contains no complete bars")
        local = np.repeat((0.5 * self.pre.volume * active)[:, None, None], 2, axis=1)
        return scatter_add(local * np.asarray(density, dtype=float), self.mesh.elements, self.nn)

    def _setup_mass_and_dt(self):
        sol = self.p.solver
        self.mode = sol.mode
        self.mass_kind = sol.mass or ("physical" if self.mode == "dynamic" else "underwood")
        self.additional_mass = self._additional_nodal_mass()
        x = self.x0_e
        n = (x[:, 1] - x[:, 0]) / self.pre.L0[:, None]
        k = self.material.E * self.pre.area / self.pre.L0
        # sum_j |K_ij| for each translational row of one end of a bar
        row = 2.0 * k[:, None] * np.abs(n) * np.abs(n).sum(axis=1)[:, None]
        if self.mass_kind == "underwood":
            safety = float(sol.relaxation_safety)
            self.mass = scatter_add(np.repeat((0.25 * row / safety**2)[:, None, :], 2, axis=1),
                                    self.mesh.elements, self.nn)
            self.mass += self.additional_mass
            self.dt = 1.0
        else:
            m = self.material.rho * self.pre.volume / 2.0
            self.mass = scatter_add(np.repeat(m[:, None, None], 2, axis=1) * np.ones((1, 1, 3)),
                                    self.mesh.elements, self.nn)
            self.mass += self.additional_mass
            # THE BOUND FROM THE MASS THE NODES CARRY, 2 sqrt(m / sum|K|) per
            # dof (Gershgorin): for a bare bar exactly the old L / c, and it
            # sees lumped `nodal_masses` too. The wave-speed form did not --
            # rho = 0 with 300 kg on every node divided by zero, and rho = 1
            # took a step 500 times too small.
            node_row = scatter_add(np.repeat(row[:, None, :], 2, axis=1), self.mesh.elements, self.nn)
            on = node_row > 1e-30
            if not np.any(on):
                raise ValueError("truss2 dynamic run: no bar has stiffness")
            if getattr(self, "fixed", None) is not None and self.fixed.shape == node_row.shape \
                    and np.any(on & ~self.fixed):
                on &= ~self.fixed          # a held dof cannot go unstable
            ratio = np.min(self.mass[on] / node_row[on])
            if not ratio > 0.0:
                raise ValueError("truss2 dynamic run: a node with stiffness has no mass "
                                 "(give the material a density or the node a nodal mass)")
            self.dt = float(sol.dt_safety * 2.0 * np.sqrt(ratio))
        # Earthquake inertia always uses physical bar mass, never the
        # artificial Underwood mass used only by quasi-static relaxation.
        m = self.material.rho * self.pre.volume / 2.0
        self.physical_mass = scatter_add(
            np.repeat(m[:, None, None], 2, axis=1) * np.ones((1, 1, 3)),
            self.mesh.elements, self.nn)
        self.physical_mass += self.additional_mass
        self.mass = np.maximum(self.mass, 1e-30)
        self.added_mass_frac = 0.0
        self.n_steps = int(sol.n_steps) if self.mode == "relaxation" else max(
            1, int(round(float(sol.total_time) / self.dt)))
        self.total_time = float(self.n_steps) if self.mode == "relaxation" else float(sol.total_time)
        damp = sol.damping
        self.damp_kind = damp.type or ("kinetic" if self.mode == "relaxation" else "viscous")
        self.zeta = float(damp.coeff)
        self.rayleigh_alpha = float(damp.alpha)
        self.rayleigh_beta = float(damp.beta)
        # Rayleigh is applied explicitly (`_rayleigh_force`, from the lagged
        # velocity) and so cuts the stable step; the viscous coefficient is
        # NOT passed, because `_run_step` treats it by the averaged form,
        # which is unconditionally stable in the damping term. For a lumped
        # bar the top frequency is 2c/L_min.
        from ..elements.stability import damped_step_factor
        self.damping_step_factor = 1.0
        if self.mode != "relaxation" and self.damp_kind == "rayleigh":
            omega_max = 2.0 * np.sqrt(self.material.E / self.material.rho) / float(self.pre.L0.min())
            f = damped_step_factor(omega_max, alpha=self.rayleigh_alpha, beta=self.rayleigh_beta)
            if f < 1.0:
                self.damping_step_factor = f
                self.dt = float(self.dt * f)
        if self.mode != "relaxation":
            # n_steps * dt is the run's duration exactly
            from ..elements.stability import aligned_step
            self.stable_dt = self.dt
            self.dt, self.n_steps = aligned_step(float(sol.total_time), self.dt)
        self.tolerance = float(sol.tolerance)
        self.kinetic_tolerance = float(sol.kinetic_tolerance)

    def external_force(self, progress, physical_time=None):
        F = sum((unit * a(progress) for a, unit in self.load_groups), np.zeros((self.nn, 3)))
        if physical_time is not None:
            for t, acc, direction in self.ground_groups:
                # Relative-coordinate equation: M u¨ + f_int = -M r a_g(t).
                ag = np.interp(physical_time, t, acc, left=0.0, right=0.0)
                F -= self.physical_mass * (ag * direction)[None, :]
        return F

    def apply_bc(self, progress):
        fixed_zero = self.fixed & ~self.moving
        self.u[fixed_zero] = 0.0
        self.v[fixed_zero] = 0.0
        for idx, d, value, a, motion in self.prescribed:
            target = value * a(progress)
            if motion == "displacement":
                self.u[idx, d] = target
                self.v[idx, d] = 0.0
            elif motion == "velocity":
                self.u[idx, d] += self.dt * target
                self.v[idx, d] = target
                self.motion_velocity[idx, d] = target
            else:
                old = self.motion_velocity[idx, d].copy()
                self.motion_velocity[idx, d] = old + self.dt * target
                self.u[idx, d] += self.dt * old + 0.5 * self.dt**2 * target
                self.v[idx, d] = self.motion_velocity[idx, d]

    def _force(self, want_stress=True):
        """The bars' forces (`_bar_forces`) and what the attached components
        add to them (contact; core/components)."""
        f_int, sig = self._bar_forces(want_stress)
        for fn in self._each("bar_force"):
            f_int = fn(self, f_int)
        return f_int, sig

    def _bar_forces(self, want_stress=True):
        f, sig = truss.internal_force(self.pre, self.x0_e, self.u[self.mesh.elements],
                                      self.material, self.state, self.dt, want_stress)
        self._check_failure(sig)
        return scatter_add(f, self.mesh.elements, self.nn), sig

    def _rayleigh_force(self):
        """Initial-stiffness Rayleigh damping force ``alpha*M*v + beta*K0*v``."""
        out = self.rayleigh_alpha * self.mass * self.v
        if self.rayleigh_beta:
            x = self.x0_e
            n = (x[:, 1] - x[:, 0]) / self.pre.L0[:, None]
            k = self.material.E * self.pre.area / self.pre.L0
            dv = self.v[self.mesh.elements[:, 1]] - self.v[self.mesh.elements[:, 0]]
            local = np.empty((self.nel, 2, 3))
            local[:, 1] = k[:, None] * (dv * n).sum(axis=1)[:, None] * n
            local[:, 0] = -local[:, 1]
            out += self.rayleigh_beta * scatter_add(local, self.mesh.elements, self.nn)
        return out

    def _record(self, t, f_int, f_ext, sig, callback):
        self.last_reaction = np.zeros((self.nn, 3))
        residual = f_int-f_ext
        self.last_reaction[self.fixed] = residual[self.fixed]
        frame_fields = None
        for name, mon in self.monitors.items():
            if mon.kind == "element":
                # a bar force history
                from .member_actions import monitor_value
                if frame_fields is None:
                    frame_fields = self.nodal_fields(sig)
                mon.values.append(monitor_value(frame_fields, mon.field,
                                                self._monitor_members[name], mon.reduce))
                continue
            idx = np.unique(np.concatenate([self._region(r) for r in mon.regions]))   # a node once, however many regions name it
            if mon.kind == "reaction":
                mon.values.append(float((f_int - f_ext)[idx, mon.dof].sum()))
            else:
                v = float(self.u[idx, mon.dof].mean())
                if mon.relative_to:
                    v -= float(self.u[self._region(mon.relative_to), mon.dof].mean())
                mon.values.append(v)
        self.times.append(t)
        if callback:
            callback(self, self.step, t, sig)

    def run(self, on_output=None, output_every=200, on_progress=None, progress_every=500):
        started = time.perf_counter()
        self.times, self.residuals, self.kinetic = [], [], []
        self.diagnostics = start_diagnostics(self.mode, "numpy")
        from .diagnostics import resume_ledger
        resume_ledger(self)      # a restart carries the ledger on
        self.frame_equilibrium = None
        for k in range(self.step_index, len(self.steps)):
            mid_step = bool(getattr(self, "_resuming", False))
            self.apply_step(k)
            if k > 0 and not mid_step and self.mode == "relaxation":
                # A quasi-static step is a new equilibrium problem, not an
                # invitation to retain kinetic energy from its predecessor.
                # (A dynamic step is not: it is the same time history,
                # v1.0.)
                self.v.fill(0.0)
                self._half = True
                self.ke_prev = 0.0
            self._run_step(on_output, output_every, on_progress, progress_every)
        status = "CONVERGED" if self.mode == "relaxation" else "COMPLETED_DYNAMIC"
        finish_diagnostics(self.diagnostics, status, step=self.step,
                           residual=self.residuals[-1] if self.residuals else None,
                           kinetic=self.kinetic[-1] if self.kinetic else None,
                           load_step=len(self.steps))
        self.wall_time = time.perf_counter() - started
        return self

    def _run_step(self, on_output, output_every, on_progress, progress_every):
        # coeff under viscous; alpha alongside kinetic damping in a
        # relaxation (v1.0: it was read and never applied here).
        # Rayleigh keeps its explicit force below.
        cv = (self.zeta if self.damp_kind == "viscous" else
              (self.rayleigh_alpha if self.damp_kind == "kinetic" else 0.0))
        c1 = 1.0 / (1.0 + 0.5 * cv * self.dt)
        c2 = c1 * (1.0 - 0.5 * cv * self.dt)
        # counted from where the step BEGAN: a restart used to start the
        # step's clock again at its own step, re-running a whole step with
        # the ground motion at t = 0
        first, last = self.step_start, self.step_start + self.n_steps
        if self.step >= last:
            raise UserError(
                "RESTART_BUDGET_EXHAUSTED",
                f"this restart has no steps left to run: the checkpoint is at step {self.step}, "
                f"and this step ends at step {last}",
                "raise solver.n_steps (or total_time) and restart again")
        # the step's own tolerance first (found in review)
        tol = float((self.steps[self.step_index] if self.steps else {}).get("tolerance") or self.tolerance)
        for step in range(self.step + 1, last + 1):
            progress = (step - first) / self.n_steps
            f_int, sig = self._force(True)
            # a dynamic step's load at t_n, where the state is
            if self.mode == "dynamic":
                p0 = (step - 1 - first) / self.n_steps
                f_ext = self.external_force(p0, p0 * self.total_time)
            else:
                f_ext = self.external_force(progress, None)
            f_damp = self._rayleigh_force() if self.damp_kind == "rayleigh" else 0.0
            # one instant: the forces AT u, before the update moves it
            self.diagnostics["energy"].update(self.u, f_ext, f_int)
            v_old = self.v.copy() if cv > 0.0 else None
            acc = (f_ext - f_int - f_damp) / self.mass
            # Kinematic DOFs are driven below; never let the force update add
            # a second, unconstrained increment before that prescription.
            acc[self.fixed] = 0.0
            self.v[self.fixed] = 0.0
            self.v = c2 * self.v + (0.5 if self._half else 1.0) * c1 * self.dt * acc
            self._half = False
            self.u += self.dt * self.v
            self.apply_bc(progress)
            self.step = step
            free = ~self.fixed
            resid = float(np.linalg.norm((f_ext - f_int)[free]) / max(np.linalg.norm(f_ext[free]), 1.0))
            ke = float(0.5 * np.sum(self.mass * self.v * self.v))
            if not np.isfinite(ke) or not np.isfinite(self.u).all():
                location = first_nonfinite(displacement=self.u, velocity=self.v, state=self.state)
                finish_diagnostics(self.diagnostics, "NONFINITE", step=self.step,
                                   message="truss integration became non-finite", location=location)
                raise NonFiniteFailure("truss integration became non-finite; check the time step or a structural mechanism", location=location)
            if self.damp_kind == "rayleigh":
                self.diagnostics["energy"].damping(f_damp, self.dt * self.v)
            elif v_old is not None:
                # c1/c2 is the implicit-midpoint viscous update: the force it
                # applied is zeta m (v_old + v_new) / 2
                self.diagnostics["energy"].damping(
                    0.5 * cv * self.mass * (v_old + self.v), self.dt * self.v)
            if self.damp_kind == "kinetic" and ke < self.ke_prev and self.ke_prev > 0:
                self.diagnostics["energy"].dissipate(self.ke_prev)
                # the reset energy is gone (see beam_solver)
                self.v.fill(0.0); self._half = True; self.n_resets += 1; ke = 0.0
            self.ke_prev = ke
            if step % progress_every == 0 or step == last:
                self.residuals.append(resid); self.kinetic.append(ke)
                self.diagnostics["energy"].snapshot(self.step, ke)
                stop = bool(on_progress and on_progress(step - first, self.n_steps, resid, ke,
                                                        self.n_resets) is False)
            else:
                stop = False
            if step % output_every == 0 or step == last:
                # the frame reads the state; the last frame of the dynamic
                # history closes it (a removal due at its very end)
                closing = step == last and not any(
                    (st.get("procedure") or self.p.solver.mode) == "dynamic"
                    for st in self.steps[self.step_index + 1:])
                self._probing = self.mode == "dynamic" and not closing
                # ... AND IT LEAVES THE BARS' STATE AS IT FOUND IT:
                # a law accepts its last evaluation when it is evaluated
                # again -- steel02 cuts a ruptured bar in the evaluation
                # AFTER the one that passed the limit -- and the frame was
                # that evaluation: with a frame after every step a bar
                # ruptured a step early (the fibre beam's fault of v1.12,
                # here in the truss; found by the sweep of every solver and
                # law over `output.every`). The frame's fields are of its
                # own evaluation; the state goes back after they are read.
                kept = self.state.copy()
                try:
                    f_int, sig = self._force(True)
                    time_value = (self.step_index + progress if self.mode == "relaxation"
                                  else self.step_index * self.total_time + progress * self.total_time)
                    self.frame_equilibrium = (None if self.mode == "dynamic" else
                                              bool(step == last and resid <= tol))
                    self._record(time_value,
                                 f_int, f_ext, sig, on_output)
                finally:
                    self._probing = False
                    self.state[...] = kept
            if stop:
                raise StopAnalysis()          # after this step's frame
        if self.mode == "relaxation":
            scale = max(float(np.linalg.norm(f_ext[free]))*max(float(np.linalg.norm(self.u[free])), 1e-12), 1.0)
            kinetic_ratio = ke/scale
            if resid > tol or kinetic_ratio > self.kinetic_tolerance:
                message = (f"truss relaxation reached {self.n_steps} iterations with residual={resid:.3e}, "
                           f"kinetic_ratio={kinetic_ratio:.3e}; tolerances are {tol:.3e} and {self.kinetic_tolerance:.3e}")
                location = worst_residual(f_ext-f_int, free, 3)
                finish_diagnostics(self.diagnostics, "MAX_STEPS", step=self.step, residual=resid,
                                   kinetic=ke, load_step=self.step_index+1, message=message,
                                   location=location)
                raise ConvergenceFailure(message, location=location)

    def _nodal_fields(self, sig: Optional[np.ndarray], recover_stress=False):
        out = {"displacement": self.u.copy()}
        if hasattr(self, "last_reaction"): out["reaction_force"] = self.last_reaction
        if sig is not None:
            stress = sig[:, 0, 0]
            force = np.asarray(stress * self.pre.area, dtype=float)
            # tension positive, as every line family; a
            # failed bar carries nothing, whatever strain it is left with
            failed = np.asarray(getattr(self.breaking, "member_failed", np.zeros(self.nel, bool)), bool)
            if failed.shape == force.shape and failed.any():
                force = np.where(failed, 0.0, force)
            out.update(axial_stress=stress, axial_force=force, axial_force_max_abs=np.abs(force),
                       von_mises=np.abs(stress))
        self.breaking.fields(out)
        for name, values in self.material.output_fields(self.state).items():
            values = np.asarray(values)
            if values.shape == (self.nel,):
                out.setdefault(name, values.copy())
        return out

    def _checkpoint(self):
        ck = self.breaking.checkpoint({"u": self.u, "v": self.v, "state": self.state,
                "material_signature": np.asarray(self.material_points.signature.encode()),
                "step": np.array([self.step]), "ke_prev": np.array([self.ke_prev]),
                "half": np.array([1 if self._half else 0]),
                "n_resets": np.array([getattr(self, "n_resets", 0)]),
                "step_index": np.array([self.step_index]),
                "step_start": np.array([self.step_start])})
        if hasattr(self, "diagnostics"):
            ck["energy"] = self.diagnostics["energy"].checkpoint()
        return {k: np.array(v, copy=True) for k, v in ck.items()}

    def _restore(self, ck):
        self.u, self.v = np.array(ck["u"], dtype=float), np.array(ck["v"], dtype=float)
        if "material_signature" in ck:
            saved = bytes(np.asarray(ck["material_signature"]).item()).decode()
            if saved != self.material_points.signature:
                raise ValueError("checkpoint material-point contract does not match this model")
        self.state = validate_restored_state(ck["state"], self.material_points, self.nel)
        self.step, self.ke_prev = int(ck["step"][0]), float(ck["ke_prev"][0])
        if "half" in ck:
            self._half = bool(ck["half"][0])
        if "n_resets" in ck:
            self.n_resets = int(ck["n_resets"][0])
        self._restored_energy = ck.get("energy")
        if "step_index" in ck:
            self.step_index, self.step_start = int(ck["step_index"][0]), int(ck["step_start"][0])
            self._resuming = True
        self._deactivate(self.breaking.restore(ck))          # a broken bar stays broken

    def summary(self):
        return (f"  mesh          {self.nn} nodes, {self.nel} elements\n"
                f"  element       truss2 / full (1 integration point)\n"
                f"  section       A = {self.pre.area[0]:.6e} m^2\n"
                f"  material      {self.material.name}\n"
                f"  nodal mass    {self.additional_mass[:, 0].sum():.6g} kg\n"
                f"  mode          {self.mode}, mass = {self.mass_kind}, damping = {self.damp_kind}\n"
                f"  backend       numpy\n"
                f"  time step     dt = {self.dt:.4e}, {self.n_steps} steps"
                + member_axes.summary_lines(self))        # v1.0
