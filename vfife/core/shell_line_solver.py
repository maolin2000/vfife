"""Conforming Shell4--frame assembly with shared six-DOF particles."""
import numpy as np

from .problem import Problem
from . import mass_scaling
from . import member_axes
from .mesh import Mesh
from .mixed_structural_solver import MixedStructuralSolver
from .shell_solver import ShellSolver


class ShellLineSolver(ShellSolver):
    """Shell midsurface and selected Line2 members sharing all six nodal DOFs."""
    def __init__(self,problem,mesh,material,log=print):
        problem = Problem.of(problem)        # typed, and its dict
        blocks=mesh.cell_blocks
        if set(blocks)!={"shell4","truss2"}:
            raise ValueError("shell_line requires one Shell4 block and selected Line2 members")
        shell_mesh=Mesh(mesh.nodes,blocks["shell4"],dict(mesh.regions),blocks)
        super().__init__(problem,shell_mesh,material,log)
        lp=problem.plain(); lp["element"]={"type":"mixed_line","formulation":"full"}
        lp["loads"]=[]; lp["steps"]=[]; lp["constraints"]=[]; lp["output"]={"monitors":{}}
        # the multi-point constraints are the whole model's, held by this
        # (the shell) solver's layer; the line half carries none of its own
        lp["rigid_links"]=[]; lp["diaphragms"]=[]
        # AND `nodal_masses`, which this line forgot. The shell
        # half above already added them over the same node array, and line 36
        # sums the two physical masses -- so a 1e6 kg lump arrived as 2e6 kg,
        # exactly twice, and every natural period came out sqrt(2) too long
        # while nothing in the summary disclosed it.
        lp["nodal_masses"]=[]
        # EACH HALF TAKES THE BEST BACKEND IT HAS. In v0.1 the line half was
        # pinned to numpy because the frame members had no compiled kernel;
        # once they were the only cost left (3.70 ms of a 4.4 ms step against
        # 0.08 ms for 55 shells) v0.1 wrote one, and pinning them here would
        # have quietly thrown it away. `warp` is a shell-only backend, so the
        # line half takes numba under it when numba is installed.
        from ..kernels import numba_frame
        want=problem.solver.backend
        lp["solver"]=dict(lp["solver"])
        lp["solver"]["backend"]=("numba" if want in ("numba", "warp")
                                 and numba_frame.available() else "numpy")
        line_mesh=Mesh(mesh.nodes,blocks["truss2"],dict(mesh.regions),
                       element_regions=dict(mesh.element_regions))
        self.line=MixedStructuralSolver(problem.derived(lp),line_mesh,material,log=lambda *_:None)
        self.K=self.K+self.line.K
        # the step from the old rotary rule, the motion from the true one
        step_mass=self.physical_mass+self.line._step_mass
        true_rot=self.dofs.view(self.physical_mass)[:,3:]+self.line._rotary_true
        self.physical_mass=self.physical_mass+self.line.physical_mass
        # the line half's hydrodynamic mass is weightless here too
        self.weightless_mass=self.weightless_mass+self.line.weightless_mass
        self.additional_mass=self.additional_mass+self.line.additional_mass
        row=np.asarray(abs(self.K).sum(axis=1)).ravel()
        if self.mode=="relaxation":
            safety=float(problem.solver.relaxation_safety)
            self.mass=np.maximum(row/safety**2,1e-20)
        else:
            self.mass=np.maximum(step_mass,1e-20)
            active=(row>1e-20)&(~self._held_dofs())
            root=float(np.sqrt(np.min(self.mass[active]/row[active])))
            self.dt=float(problem.solver.dt_safety*root)
            self.n_steps=max(1,int(round(self.total_time/self.dt)))
            # the base class parsed the damping already; the combined K has a
            # new top frequency, so the factor is taken again here
            self._shrink_step_for_damping(1.0/root)
        self.physical_mass,_=mass_scaling.true_rotary(step_mass,true_rot,row,~self._held_dofs())
        if self.mode!="relaxation":
            self.mass=np.maximum(self.physical_mass,1e-20)
        self.mesh.cell_blocks=blocks

    def step_data(self,k):
        loads=list(self.p.get("loads") or [])+list((self.p.get("steps") or [{}])[k].get("loads") or [])
        if any(x.get("line_load") is not None for x in loads):
            raise ValueError("shell_line line_load integration is not yet available; use equivalent nodal loads")
        return super().step_data(k)

    def _member_forces(self):
        shell_force,shell_actions=super()._member_forces()
        # THE ORIENTATION IS PART OF THE STATE, and this line is where that was
        # forgotten. Since v0.1 a structural point carries its
        # triad as a matrix, advanced multiplicatively, and a member's frame
        # comes from the triads and not from the rotation part of `q`. Handing
        # the line solver `q` alone left it computing every member's local
        # rotation against the identity triad it was constructed with: a
        # shell_line model rotated rigidly through a finite angle put a
        # spurious force of 37% of its own stiffness scale into its line
        # members, and nothing in the library asked. The two solvers share the
        # node set, so the array is shared, not copied.
        self.line.q=self.q; self.line.v=self.v; self.line.orient=self.orient
        # a probe of the shell is a probe of its members
        self.line._probing=getattr(self,"_probing",False)
        self.line._closing=getattr(self,"_closing",False)
        line_force,self.line_actions=self.line._member_forces()
        return shell_force+line_force,shell_actions

    def _refresh_fictitious_mass(self):
        """The relaxation mass from the current geometry, both halves.

        Until v1.20 a shell_line model did not refresh at all: the
        shell's refresh is for `shell4` only and returned, so a staged
        relaxation kept the reference-configuration rows that v0.1 found
        stall a turned member (the frame and fibre solvers until v1.19).
        The shells' blocks are turned as the shell solver turns them, the
        line members' as the mixed solver does (`_turned_delta`, which keeps
        whatever else the line half's `K` holds -- a soil spring's
        diagonal); assembled first and |.| second, so at the reference it is
        the setup mass to round-off."""
        if self.mode != "relaxation":
            return
        self.line.q = self.q; self.line.v = self.v; self.line.orient = self.orient
        K = self._turned_shell_K() + self.line.K + self.line._turned_delta()
        rows = np.asarray(abs(K).sum(axis=1)).ravel()
        safety = float(self.p.solver.relaxation_safety)
        self.mass = np.maximum(rows / safety**2, 1e-20)

    def _nodal_fields(self,actions,recover_stress=False):
        out=super()._nodal_fields(actions,recover_stress)
        if hasattr(self,"line_actions"):
            line=self.line.nodal_fields(self.line_actions,recover_stress)
            for name,value in line.items():
                if name not in {"displacement","rotation","reaction_force","reaction_moment"}:
                    out[name]=value
        return out

    def summary(self):
        return (super().summary()+f"\n  coupled       {self.line.nel} selected Line2 members; shared translations and rotations"
                + member_axes.summary_lines(self.line, self.p))        # v1.0

    # the line members' own state -- slipped bearings, hinges, broken
    # members -- which a restart of a shell_line model dropped;
    # q, v and the triads are shared with the shell.
    # Nothing parked and no appendage to save: those come
    # from a member that breaks or is removed, and the schema refuses
    # failure and removals for shell_line (`_breaking_is_line_dynamics`) --
    # the line half's breaking has no clock of its own here (its `step`
    # stays 0) and its queued splits would never be applied.
    _SHARED = ("q", "v", "orient", "step")

    def _checkpoint(self):
        ck = super()._checkpoint()
        for k, v in self.line._checkpoint_member_state().items():
            ck["line_" + k] = v
        return ck

    def _restore(self, ck):
        super()._restore(ck)
        sub = {k[5:]: v for k, v in ck.items() if k.startswith("line_")}
        if sub:
            self.line._restore_member_state(sub)
        self.line.q = self.q; self.line.v = self.v; self.line.orient = self.orient
