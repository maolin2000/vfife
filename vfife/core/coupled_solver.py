"""Conforming solid--beam--truss assembly with shared translational DOFs."""
from __future__ import annotations

import numpy as np

from .problem import Problem
from .mesh import Mesh
from .mixed_structural_solver import MixedStructuralSolver
from .solver import Solver as SolidSolver
from ..base.dof import ROTATION, TRANSLATION


class CoupledSolver(MixedStructuralSolver):
    """Six-DOF line system plus a three-DOF solid block on common nodes.

    Translation entries q[6*n:6*n+3] are the single source of truth. Solid
    forces assemble into those entries; rotations exist only for frame members.
    """
    #: which half integrates each load kind. Every kind the case schema allows
    #: must appear in exactly one of these, and `_split_loads` refuses the
    #: remainder rather than dropping it.
    SOLID_LOADS = frozenset(("traction", "pressure", "body_force"))
    LINE_LOADS = frozenset(("vector", "moment", "line_load", "gravity", "ground_motion"))

    def _split_loads(self, sp, lp):
        """Hand each load to the half that integrates it, and refuse the rest.

        The solid half has no steps of its own -- it is evaluated at the
        parent's state -- so a SURFACE load written inside a step reaches
        neither half. That is refused here too, by name, instead of vanishing.
        """
        def claim(loads, where):
            solid, line = [], []
            for ld in loads or []:
                kinds = [k for k in (self.SOLID_LOADS | self.LINE_LOADS) if ld.get(k) is not None]
                if not kinds:
                    named = next((k for k in ("temperature", "moving_load")
                                  if ld.get(k) is not None), None)
                    raise ValueError(
                        f"a coupled model does not carry a {named!r} load yet: the solid "
                        "and the line half integrate different load kinds and neither owns "
                        "this one" if named else
                        f"{where}: a load names no kind this model can integrate")
                (solid if kinds[0] in self.SOLID_LOADS else line).append(ld)
            return solid, line

        solid, line = claim(sp.get("loads"), "loads")
        sp["loads"], lp["loads"] = solid, line
        sp["steps"] = []
        for k, step in enumerate(lp.get("steps") or []):
            s_step, l_step = claim(step.get("loads"), f"steps[{k}].loads")
            if s_step:
                raise ValueError(
                    f"step {step.get('name') or k!r} applies a surface or body load to the "
                    "solid half; a coupled model evaluates its solid at the parent's state "
                    "and has no step of its own for it -- state traction/pressure/body_force "
                    "at model level")
            step["loads"] = l_step
    def __init__(self, problem, mesh, material, log=print):
        problem = Problem.of(problem)        # typed, and its dict
        if problem.solver.backend not in ("numpy", "numba"):
            raise ValueError("coupled solid-line models support solver.backend = numpy or numba")
        if material.name != "linear_elastic":
            raise ValueError("coupled solid-line models currently require linear_elastic material")
        if problem.case.reinforcement:
            raise ValueError("coupled models cannot also use embedded reinforcement; model those bars as shared-node members")
        damping = problem.solver.damping
        if damping.type == "rayleigh" and float(damping.beta):
            raise ValueError("coupled models currently support Rayleigh alpha only; beta needs the combined solid-line tangent")
        blocks = mesh.cell_blocks
        solids = [k for k in blocks if k != "truss2"]
        if len(solids) != 1 or "truss2" not in blocks:
            raise ValueError("coupled model requires one solid block and one Line2 block")
        self.solid_type = solids[0]

        # ONE PARTITION, NOT TWO FILTERS. Until then this was a
        # solid whitelist and a line whitelist, applied independently, and a
        # load kind in NEITHER -- `temperature`, `moving_load` -- was quietly
        # removed from both halves. The model ran, converged and reported
        # success with the load simply absent: 50 K on case_07 at
        # alpha = 1.2e-5 gave |f_int| = 0 where a restrained bar should carry
        # 1.2e5 N. Two whitelists cannot be audited against each other; one
        # partition with a remainder can, and the remainder is refused here.
        sp = problem.plain(); sp["element"] = dict(sp["element"]); sp["element"]["type"] = self.solid_type
        sp["constraints"] = []
        lp = problem.plain()
        self._split_loads(sp, lp)
        sp["output"] = {"monitors": {}}
        # A LUMPED MASS IS ADDED ONCE, BY THE PARENT. This copy
        # of the problem used to keep `nodal_masses`, so the solid half added
        # them and the parent added them again and line 74 below summed both:
        # on case_07 a 1e6 kg lump made the model 2e6 kg heavier, exactly twice.
        # The parent resolves every region over the SAME node array, so it
        # already covers the solid-only nodes; the solid half needs none of it.
        sp["nodal_masses"] = []
        sp["hydrodynamic_mass"] = []          # the frame half's members carry it
        solid_mesh = Mesh(mesh.nodes, blocks[self.solid_type], dict(mesh.regions))
        self.solid = SolidSolver(problem.derived(sp), solid_mesh, material, log=lambda *_: None)

        # Parent owns Line2 mechanics, boundary conditions, monitors and time loop.
        line_mesh = Mesh(mesh.nodes, blocks["truss2"], dict(mesh.regions),
                         element_regions=dict(mesh.element_regions))
        super().__init__(problem.derived(lp), line_mesh, material, log)
        self.mesh.cell_blocks = blocks
        beam_e = np.flatnonzero(np.isin(self.member_kind, tuple(self.BEAM_TYPES)))
        self.beam_nodes = (np.unique(self.mesh.elements[beam_e]) if len(beam_e)
                           else np.empty(0, dtype=int))
        rot_dofs = self.dofs.node_dofs(np.arange(self.nn), ROTATION).ravel()
        rot_rows = np.asarray(abs(self.K[rot_dofs]).sum(axis=1)).ravel()
        scale = max(float(rot_rows.max(initial=0.0)), 1.0)
        self.active_rot = (rot_rows > scale*1e-13).reshape(self.nn, 3)
        # With solid translations regarded as supported by the continuum, the
        # remaining rotational block must still be full rank after user BCs.
        fixed0 = super().step_data(0)[0][rot_dofs]
        free = self.active_rot.reshape(-1) & ~fixed0
        if np.any(free):
            Krr = self.K[rot_dofs][:, rot_dofs][free][:, free].toarray()
            rank = np.linalg.matrix_rank(Krr, tol=max(np.abs(Krr).max(), 1.0)*1e-11)
            if rank < Krr.shape[0]:
                raise ValueError(f"coupled beam rotations contain {Krr.shape[0]-rank} mechanism DOF(s); "
                                 "add physical rotational constraints/releases or use truss members for a pinned connection")
        self.solid_mass = self.solid.physical_mass.copy()
        line_mass = self.dofs.view(self.physical_mass).copy()
        line_mass[:, self.dofs.component_indices(TRANSLATION)] += self.solid_mass
        self.physical_mass = line_mass.reshape(-1)
        if self.mode == "relaxation":
            mm = self.dofs.view(self.mass)
            mm[:, self.dofs.component_indices(TRANSLATION)] += self.solid.mass
            self.mass = mm.reshape(-1)
        else:
            self.mass = np.maximum(self.physical_mass, 1e-20)
            from ..elements.stability import aligned_step
            self.stable_dt = min(getattr(self, "stable_dt", self.dt), getattr(self.solid, "stable_dt", self.solid.dt))
            self.dt, self.n_steps = aligned_step(self.total_time, self.stable_dt)

    def _refresh_fictitious_mass(self):
        """The members' relaxation mass from the current geometry, and the
        solid's on top, as `__init__` puts it: `K` here is the members'
        alone, and the rows taken from it carry nothing of the solid (until
        v1.19 a refreshed coupled model lost its solid's relaxation
        mass at every staged level)."""
        super()._refresh_fictitious_mass()
        if self.mode == "relaxation":
            mm = self.dofs.view(self.mass)
            mm[:, self.dofs.component_indices(TRANSLATION)] += self.solid.mass
            self.mass = mm.reshape(-1)

    def step_data(self, k):
        fixed, moving, prescribed, groups, ground = super().step_data(k)
        # Solid and truss-only nodes have no rotational DOFs. Keep these
        # algebraic slots inactive instead of assigning artificial tiny masses.
        inactive = np.argwhere(~self.active_rot)
        inactive_rot = self.dofs.node_dofs(inactive[:, 0], ROTATION)[
            np.arange(len(inactive)), inactive[:, 1]
        ]
        fixed[inactive_rot] = True
        return fixed, moving, prescribed, groups, ground

    def _force(self, groups, ground, progress, t=None):
        force = super()._force(groups, ground, progress, t)
        fs = self.solid.external_force(progress, t)
        self.dofs.view(force)[:, self.dofs.component_indices(TRANSLATION)] += fs
        return force

    def _member_forces(self):
        out, actions = super()._member_forces()
        self.solid.u[:] = self.u
        f_e, sig = self.solid._force(self.solid.u, self.solid.quat, self.solid.state,
                                     self.dt, rot_iters=self.solid.rot_iters,
                                     want_stress=True)
        fs = self.solid._force.assemble(f_e)
        self.dofs.view(out)[:, self.dofs.component_indices(TRANSLATION)] += fs
        self.solid_stress = sig
        return out, actions

    def _nodal_fields(self, actions, recover_stress=False):
        out = super()._nodal_fields(actions, recover_stress)
        if getattr(self, "solid_stress", None) is not None:
            out.update(self.solid.nodal_fields(self.solid_stress, recover_stress))
            out["rotation"] = self.dofs.field(self.q, ROTATION).copy()
        return out

    def summary(self):
        return (super().summary() + f"\n  coupled       {self.solid.nel} {self.solid_type} solids; "
                "shared translations")

    # the solid half's rotations and material state: a
    # coupled model saved only the shared q and v, so a restart began its
    # continuum from a virgin material -- every plastic strain and crack gone
    def _checkpoint(self):
        ck = super()._checkpoint()
        for i, b in enumerate(self.solid.blocks):
            ck[f"solid_quat{i}"] = np.array(b.quat, copy=True)
            ck[f"solid_state{i}"] = np.array(b.state, copy=True)
        return ck

    def _restore(self, ck):
        super()._restore(ck)
        for i, b in enumerate(self.solid.blocks):
            if f"solid_state{i}" in ck:
                b.state = np.array(ck[f"solid_state{i}"], dtype=b.state.dtype, copy=True)
                b.quat = np.array(ck[f"solid_quat{i}"], dtype=b.quat.dtype, copy=True)
