"""Six-DOF co-rotational Mindlin--Reissner Shell solver (Quad4 + Tri3)."""
from __future__ import annotations

import time

import numpy as np

from .problem import Problem
from .beam_solver import BeamSolver
from .diagnostics import (StopAnalysis, start as start_diagnostics, finish as finish_diagnostics,
                          ConvergenceFailure, NonFiniteFailure, UserError, worst_residual)
from .dof import DofLayout, ROTATION, SparseAssembler, TRANSLATION, repeated_rotation
from .rotations import log_map
from ..elements import shell


def _rotation_vector(Q):
    c=np.clip((np.trace(Q)-1)*.5,-1.,1.); angle=float(np.arccos(c))
    if angle<1e-12: return .5*np.array([Q[2,1]-Q[1,2],Q[0,2]-Q[2,0],Q[1,0]-Q[0,1]])
    if np.pi-angle<1e-7:
        vals,vecs=np.linalg.eigh(.5*(Q+np.eye(3))); axis=vecs[:,np.argmax(vals)]
        return angle*axis
    return angle/(2*np.sin(angle))*np.array([Q[2,1]-Q[1,2],Q[0,2]-Q[2,0],Q[1,0]-Q[0,1]])


class ShellSolver(BeamSolver):
    def __init__(self,problem,mesh,material,log=print):
        problem = Problem.of(problem)        # typed, and its dict
        if material.name!="linear_elastic": raise ValueError("shell currently supports linear_elastic only")
        self.backend = problem.solver.backend
        if self.backend not in ("numpy", "numba", "warp"):
            raise ValueError(f"shell supports backend numpy, numba or warp, not {self.backend!r}")
        if self.backend == "numba":
            from ..kernels import numba_shell4
            if not numba_shell4.available():
                raise ValueError("solver.backend = numba but numba is not installed")
        if self.backend == "warp":
            if problem.case.element.type != "shell4":
                raise ValueError("the warp backend covers shell4 only")
            from ..kernels import warp_shell4
            if not warp_shell4.available():
                raise ValueError("solver.backend = warp but warp is not installed")
        kind=problem.case.element.type
        if kind=="shell3" and self.backend!="numpy":
            # it ran the per-element Python loop under any backend: 435 s for
            # 128 elements against 6 s for the quad mesh
            raise ValueError("shell3 (triangles) runs on the numpy backend only; quads (shell4) "
                             "have the compiled kernels")
        if kind=="shell3" and problem.case.element.hourglass_coeff != 0.1:
            raise ValueError("hourglass_coeff is the quad's (shell4) stabilisation; a triangle has "
                             "none to set")
        if kind=="shell3" and mesh.elements.shape[1]!=3: raise ValueError("shell3 requires Gmsh Tri3 elements")
        if kind in ("shell4","shell_line") and mesh.elements.shape[1]!=4: raise ValueError("shell4 requires Gmsh Quad4 elements")
        if kind not in ("shell3","shell4","shell_line"): raise ValueError(f"ShellSolver kind {kind!r} not supported, use shell3 or shell4")
        self.p,self.mesh,self.material,self.log=problem,mesh,material,log
        self.nn,self.nel=mesh.n_nodes,mesh.n_elem; self.dofs=DofLayout.frame(self.nn); self.ndof=self.dofs.size
        self._setup_regions(); self._assemble_reference(); self._setup_mass_and_time()
        self.q=np.zeros(self.ndof); self.v=np.zeros(self.ndof); self.ke_prev=0.; self.n_resets=0
        self.orient=np.tile(np.eye(3),(self.nn,1,1)); self._rot_scratch=np.zeros((self.nn,3))
        self.step=self.step_index=self.step_start=0; self.rebar=[]

    def _assemble_reference(self):
        e=self.p.case.element; assembly=SparseAssembler(self.ndof); self.data=[]
        kind=e.type
        for conn in self.mesh.elements:
            if kind=="shell3":
                data=shell.precompute_tri(self.mesh.nodes[conn],self.material.E,self.material.nu,e.thickness,
                    e.shear_correction,e.drilling_factor,e.warpage_tolerance)
                T=repeated_rotation(data.frame,6)
            else:
                data=shell.precompute(self.mesh.nodes[conn],self.material.E,self.material.nu,e.thickness,
                    e.shear_correction,e.drilling_factor,e.hourglass_coeff,
                    e.warpage_tolerance)
                T=repeated_rotation(data.frame,8)
            dof=self.dofs.element_dofs(conn)
            assembly.add_matrix(dof,T.T@data.stiffness@T); self.data.append(data)
        self.K=assembly.csr(); self.area=np.asarray([d.area for d in self.data])

    def _local_deformation(self,e):
        conn=self.mesh.elements[e]; nA=len(conn); qe=self.q[self.dofs.element_dofs(conn)].reshape(nA,6)
        kind=self.p.case.element.type
        if kind=="shell3":
            x=self.mesh.nodes[conn]+qe[:,:3]; R,c,current=shell.frame_tri(x)
        else:
            x=self.mesh.nodes[conn]+qe[:,:3]; R,c,current=shell.frame(x,self.p.case.element.warpage_tolerance*4)
        ref=self.data[e]
        local=np.zeros((nA,6)); local[:,:3]=current-ref.local_nodes
        for a,n in enumerate(conn):
            local[a,3:]=_rotation_vector(R@self.orient[n]@ref.frame.T)      # the node's triad relative to the face
        return R,local.ravel()

    # ------------------------------------------------------------------ #
    #  Internal force
    # ------------------------------------------------------------------ #
    #  shell4 is evaluated for ALL elements at once -- batched numpy, a numba
    #  kernel, or a warp kernel on a device (`solver.backend`); shell3 keeps
    #  the loop below (it has no benchmark that needs the speed yet). The loop
    #  version ran the 256-element Sze hemisphere at a few iterations a
    #  second, for the same reason the frame solver did before v0.1:
    #  a dozen small numpy calls per element per iteration.

    def _shell4_batch(self):
        if getattr(self, "_sh_dofs", None) is None:
            ne = self.nel
            self._sh_dofs = np.array([self.dofs.element_dofs(c) for c in self.mesh.elements]).reshape(ne, 24)
            self._sh_frame0 = np.array([d.frame for d in self.data])            # (ne,3,3)
            self._sh_local0 = np.array([d.local_nodes for d in self.data])      # (ne,4,3)
            self._sh_K = np.array([d.stiffness for d in self.data])             # (ne,24,24)
            # the resultant operator at the centroid, [Dm Bm; Db Bb; Ds Bs]
            Bres = np.zeros((ne, 8, 24))
            for e, d in enumerate(self.data):
                N, dn = shell.shape(0, 0); J = dn @ d.local_nodes[:, :2]; dxy = np.linalg.solve(J, dn)
                Bm = np.zeros((3, 24)); Bb = np.zeros((3, 24)); Bs = np.zeros((2, 24))
                for a in range(4):
                    i = 6*a; dx, dy = dxy[:, a]
                    Bm[:, i:i+2] = [[dx, 0], [0, dy], [dy, dx]]
                    Bb[:, i+3:i+5] = [[0, dx], [-dy, 0], [-dx, dy]]
                    Bs[:, i:i+6] = [[0, 0, dx, 0, N[a], 0], [0, 0, dy, -N[a], 0, 0]]
                Bres[e] = np.vstack((d.membrane @ Bm, d.bending @ Bb, d.shear @ Bs))
            self._sh_Bres = Bres
            self._sh_warp = float(self.p.case.element.warpage_tolerance) * 4
        return self._sh_dofs

    def _current_frames(self):
        """(R, current local coordinates) of every shell4 in the current
        configuration -- `shell.frame` over an element axis."""
        dofs = self._shell4_batch()
        q = self.q[dofs].reshape(-1, 4, 6)
        x = self.mesh.nodes[self.mesh.elements] + q[:, :, :3]
        c = x.mean(axis=1)
        ex = (x[:, 1] - x[:, 0]) + (x[:, 2] - x[:, 3])
        normal = np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 1])
        nx = np.linalg.norm(ex, axis=1); nn = np.linalg.norm(normal, axis=1)
        if np.any(nx <= 1e-14) or np.any(nn <= 1e-14):
            raise ValueError("shell4 has a degenerate midsurface")
        ex = ex / nx[:, None]; ez = normal / nn[:, None]
        ey = np.cross(ez, ex); ey /= np.linalg.norm(ey, axis=1)[:, None]
        R = np.stack((ex, ey, ez), axis=1)                                   # rows = local axes
        local = np.einsum("eij,enj->eni", R, x - c[:, None, :])
        span = np.maximum(np.maximum(np.ptp(local[:, :, 0], axis=1), np.ptp(local[:, :, 1], axis=1)), 1e-30)
        if np.any(np.ptp(local[:, :, 2], axis=1) > self._sh_warp * span):
            raise ValueError("shell4 reference face is too warped; refine the surface mesh")
        local[:, :, 2] = 0.0
        return R, local, q

    def step_data(self, k):
        """A shell has its own force path (`_member_forces` below), so the
        frame's `f = k (d - d_th)` never runs here: a temperature load would be
        accepted and silently do nothing. Refused by name instead.
        A shell temperature field is a real thing -- a
        membrane/bending split of alpha T through the thickness -- and it is
        simply not built yet.
        """
        loads = (list(self.p.get("loads") or [])
                 + list((self.p.get("steps") or [{}])[k].get("loads") or []))
        if any(x.get("temperature") is not None for x in loads):
            raise ValueError("shell elements do not carry a temperature load yet "
                             "(it is implemented on beam2, beam2_timoshenko and "
                             "beam2_fiber); see docs/temperature.md")
        return super().step_data(k)

    def _member_forces(self):
        # shell3 has no batched kernel; `shell_line` is quad4 like `shell4` and
        # was reaching the reference path only because it is spelled
        # differently. On the building of v0.1 that cost 35.8 ms a
        # step for 55 elements -- ten times the 759 frame members beside them.
        if self.p.case.element.type == "shell3":
            return self._internal_force_reference()
        if self.backend == "numba":
            return self._internal_force_numba()
        return self._internal_force_numpy()

    def _internal_force_numba(self):
        """One element at a time in `kernels/numba_shell4`, assembled there."""
        from ..kernels.numba_shell4 import shell4_forces
        self._shell4_batch()
        if getattr(self, "_nb_result", None) is None:
            self._nb_result = np.zeros((self.nel, 8))
            self._nb_warped = np.zeros(self.nel, dtype=np.int8)
            self._nb_conn = np.ascontiguousarray(self.mesh.elements, dtype=np.int64)
        out = np.zeros(self.ndof)
        shell4_forces(self.mesh.nodes, self._nb_conn, self.q.reshape(self.nn, 6), self.orient,
                      self._sh_frame0, self._sh_local0, self._sh_K, self._sh_Bres, self._sh_warp,
                      out, self._nb_result, self._nb_warped)
        if self._nb_warped.any():
            if (self._nb_warped == 2).any():
                raise ValueError("shell4 has a degenerate midsurface")
            raise ValueError("shell4 reference face is too warped; refine the surface mesh")
        return out, self._nb_result.copy()

    def _internal_force_numpy(self):
        """Batched NumPy over all shell4 elements: the readable reference the
        numba and warp kernels are pinned to."""
        dofs = self._shell4_batch()
        R, local, q = self._current_frames()
        # each node's local rotation is the relative rotation of its triad
        # with respect to the face frame, R T_n frame0^T, as a vector
        # (until then the difference of two rotation vectors,
        # exact only when they were coaxial)
        Tn = self.orient[self.mesh.elements]                                  # (ne,4,3,3)
        rel = np.einsum("eij,enjk,elk->enil", R, Tn, self._sh_frame0)
        lq = np.zeros((self.nel, 4, 6))
        lq[:, :, :3] = local - self._sh_local0
        lq[:, :, 3:] = log_map(rel)
        lq = lq.reshape(self.nel, 24)
        fl = np.einsum("eij,ej->ei", self._sh_K, lq)
        contrib = np.einsum("eji,ebj->ebi", R, fl.reshape(self.nel, 8, 3)).reshape(self.nel, 24)
        out = np.zeros(self.ndof); np.add.at(out, dofs, contrib)
        result = np.einsum("eij,ej->ei", self._sh_Bres, lq)
        return out, result

    def _refresh_fictitious_mass(self):
        """The relaxation mass from the CURRENT geometry: the shell twin of
        BeamFamily._refresh_fictitious_mass, for the same reason.
        Assembled first, |.| second, so at the reference it
        is the setup mass exactly."""
        if self.p.case.element.type != "shell4":
            return
        rows = np.asarray(abs(self._turned_shell_K()).sum(axis=1)).ravel()
        safety = float(self.p.solver.relaxation_safety)
        self.mass = np.maximum(rows / safety**2, 1e-20)

    def _turned_shell_K(self):
        """The shells' blocks turned into their current frames, assembled."""
        dofs = self._shell4_batch()
        R, _local, _q = self._current_frames()
        T = np.zeros((self.nel, 24, 24))
        for blk in range(8):
            T[:, 3*blk:3*blk+3, 3*blk:3*blk+3] = R
        kg = np.einsum("eji,ejk,ekl->eil", T, self._sh_K, T)
        assembly = SparseAssembler(self.ndof)
        for e in range(self.nel):
            assembly.add_matrix(dofs[e], kg[e])
        return assembly.csr()

    # ------------------------------------------------------------------ #
    # the warp backend: a device-resident relaxation                      #
    # ------------------------------------------------------------------ #

    def run(self, on_output=None, output_every=200, on_progress=None, progress_every=500):
        if self.backend != "warp":
            return super().run(on_output, output_every, on_progress, progress_every)
        return self._run_warp(on_output, on_progress, progress_every)

    def _make_warp_shell(self):
        """The device twin of the batched arrays, or the same environment
        contract the solid backend gives (kernels.make_force): a missing
        driver is the user's to act on, so it is a UserError sentence, not
        Warp's stack trace."""
        from ..kernels.warp_shell4 import WarpShell
        self._shell4_batch()
        sol = self.p.solver; device = sol.device
        try:
            return WarpShell(self.mesh.nodes, self.mesh.elements, self._sh_frame0, self._sh_local0,
                             self._sh_K, self._sh_Bres, self._sh_warp,
                             device=device, precision=sol.precision)
        except UserError:
            raise
        except Exception as exc:
            raise UserError(
                "BACKEND_UNAVAILABLE",
                f"this case asks for the warp backend on device {device!r}, "
                f"which this machine cannot provide: {exc}",
                "set solver.backend = numba (or numpy) to run it here -- the answer is "
                "the same, the run is slower. A GPU run needs an NVIDIA card "
                "with a working CUDA driver.") from None

    def _run_warp(self, on_output, on_progress, progress_every):
        """BeamSolver.run's relaxation branch with the state on the device.
        Per iteration nothing crosses the bus but five
        scalars: the kinetic energy, for the damping reset, and the four
        norms of the convergence check, evaluated every iteration exactly as
        the host loop does so the two stop at the same iteration. u, F_int
        and the resultants come back once per converged level, for the
        record; the fictitious mass is refreshed on the host from that u.

        Not here: the dynamic mode, ground motion, velocity- and
        acceleration-prescribed dofs, and the per-iteration energy ledger.
        Those name the numba backend."""
        started = time.perf_counter(); self.times, self.residuals, self.kinetic = [], [], []
        self.diagnostics = start_diagnostics(self.mode, "warp", True)
        from .diagnostics import resume_ledger
        resume_ledger(self)      # a restart carries the ledger on
        self.frame_equilibrium = None
        if self.mode != "relaxation":
            raise ValueError("the warp shell backend is the relaxation path; use backend = numba for solver.mode = dynamic")
        if getattr(self, "_resume", None) is not None:
            # the device loop keeps its level and iteration on the GPU and
            # cannot be picked up mid-way; refusing is better than replaying
            # the run on the restored state
            raise UserError("RESTART_UNSUPPORTED",
                            "a warp shell run cannot be resumed from a checkpoint",
                            "restart it with solver.backend = numba (the same answer), or use the "
                            "result as solver.initial_state for a new run")
        ws = self._make_warp_shell(); self._ws = ws
        sol = self.p.solver
        for k in range(len(self.steps)):
            self.step_index = k; fixed, moving, prescribed, groups, ground = self.step_data(k)
            self.current_fixed = fixed
            if ground or moving.any() or any(m != "displacement" for _d, _v, _a, m in prescribed):
                raise ValueError("the warp shell backend takes displacement constraints and loads only; "
                                 "use backend = numba for ground motion or velocity/acceleration constraints")
            step_cfg = self.steps[k]
            nstage = int(step_cfg.get("load_steps") or sol.load_steps or 0)
            if nstage:
                budget = int(step_cfg.get("max_steps_per_load") or sol.max_steps_per_load)
                tol = float(step_cfg.get("tolerance") or self.tolerance)
                ramp = int(step_cfg.get("stage_ramp_iterations") or
                               self.p.solver.stage_ramp_iterations or
                               max(budget // 5, 1))
                levels = [(stage, stage / nstage) for stage in range(1, nstage + 1)]
            else:
                budget, tol, ramp, levels = self.n_steps, self.tolerance, 0, [(1, None)]
            for stage, progress in levels:
                staged = progress is not None
                self.v.fill(0.0); self._half = True; self.ke_prev = 0.0
                starts = [self.q[dofs].copy() for dofs, _v, _a, _m in prescribed]
                if staged:
                    self._reorthonormalize()
                    self._refresh_fictitious_mass()
                ws.set_mass(self.mass)
                ws.set_nodal(fixed, groups, [(d, v, a) for d, v, a, _m in prescribed], starts)
                ws.set_u(self.q, self.v, self.orient)
                converged = False; resid = kr = float("nan"); ke = 0.0
                for it in range(1, budget + 1):
                    p = progress if staged else it / budget
                    approach = min(1.0, it / ramp) if staged else 1.0
                    ws.force_resident()
                    ws.integrate(p, 1 if self._half else 0, approach)
                    self._half = False; self.step += 1
                    ke, r_free, n_int, n_ext, q_free, refused = ws.measures(p)
                    if refused:
                        ws.check_flags()
                    if not np.isfinite(ke) or not np.isfinite(r_free):
                        location = {"array": "device_scalar", "index": []}
                        finish_diagnostics(self.diagnostics, "NONFINITE", step=self.step, load_step=stage,
                                           message="shell relaxation became non-finite on the device", location=location)
                        raise NonFiniteFailure("shell relaxation became non-finite; reduce relaxation_safety or check for a structural mechanism", location=location)
                    if ke < self.ke_prev and self.ke_prev > 0.0:
                        self.diagnostics["energy"].dissipate(self.ke_prev)
                        ws.zero_velocity(); self._half = True; self.n_resets += 1; ke = 0.0
                    self.ke_prev = ke
                    scale = max(n_ext, n_int, 1e-30)
                    resid = r_free / scale
                    kr = ke / max(scale * max(q_free, 1e-12), 1e-30)
                    if on_progress and it % progress_every == 0:
                        if not staged:
                            self.residuals.append(float(resid)); self.kinetic.append(float(ke))
                        if on_progress(stage if staged else it, nstage if staged else budget, resid, ke, self.n_resets) is False:
                            raise StopAnalysis()
                    settled = it > ramp if staged else all(a.settled(p) for a, _ in groups)
                    if settled and resid <= tol and kr <= self.kinetic_tolerance:
                        converged = True; break
                self.q = ws.get_u(); self.v = ws.get_v(); self.orient = ws.get_orient()
                self.residuals.append(float(resid)); self.kinetic.append(float(ke))
                self.diagnostics["energy"].snapshot(self.step, ke)
                F = self._force(groups, ground, progress if staged else 1.0)
                if staged and on_progress and on_progress(stage, nstage, resid, ke, self.n_resets) is False:
                    raise StopAnalysis()
                if not converged:
                    message = (f"shell load stage {stage}/{nstage} did not converge in {budget} iterations: "
                               f"residual={resid:.3e} (tol {tol:.1e}), kinetic_ratio={kr:.3e} (tol {self.kinetic_tolerance:.1e})"
                               if staged else
                               f"shell relaxation reached {budget} iterations with residual={resid:.3e} > tolerance={tol:.3e}")
                    location = worst_residual(F - self.internal_force()[0], ~fixed, 6)
                    finish_diagnostics(self.diagnostics, "MAX_STEPS", step=self.step, residual=resid,
                                       kinetic=ke, load_step=stage, message=message, location=location)
                    raise ConvergenceFailure(message, location=location)
                self.frame_equilibrium = True
                self._record(k + (progress if staged else 1.0), F, on_output)
        finish_diagnostics(self.diagnostics, "CONVERGED", step=self.step,
                           residual=self.residuals[-1] if self.residuals else None,
                           kinetic=self.kinetic[-1] if self.kinetic else None,
                           load_step=len(self.steps))
        self.wall_time = time.perf_counter() - started; return self

    def _internal_force_reference(self):
        """The element-at-a-time version: the shell3 path, and the reference
        the batched shell4 path is pinned to."""
        out=np.zeros(self.ndof); result=np.zeros((self.nel,8))
        for e,conn in enumerate(self.mesh.elements):
            R,local=self._local_deformation(e); fl=self.data[e].stiffness@local
            nA=len(conn); out[self.dofs.element_dofs(conn)]+=repeated_rotation(R, nA*2).T@fl
            kind=self.p.case.element.type
            if kind=="shell3":
                result[e]=shell.resultants_tri(self.data[e],local)
            else:
                result[e]=shell.resultants(self.data[e],local)
        return out,result

    def _setup_mass_and_time(self):
        e,sol=self.p.case.element,self.p.solver; rho=self.material.rho; t=float(e.thickness)
        m=np.zeros((self.nn,6))
        kind=e.type
        div = 3 if kind=="shell3" else 4
        for conn,A in zip(self.mesh.elements,self.area):
            m[conn,:3]+=rho*t*A/div; m[conn,3:]+=rho*t**3*A/12/div
        # nodal masses: directional, weightless, rotary
        from . import added_mass
        if self.p.case.hydrodynamic_mass and type(self).__name__ == "ShellSolver":
            raise ValueError("hydrodynamic_mass acts on line members (piers, piles); a shell model has "
                             "none -- write the added mass as directional nodal_masses")
        # a shell-line model's line half adds the hydrodynamic mass on its members
        add, rot, self.weightless_mass = added_mass.table(
            {"nodal_masses": self.p.get("nodal_masses")}, self._region, self.mesh.nodes[:self.nn])
        m[:, :3] += add; m[:, 3:] += rot; self.additional_mass = add
        self.physical_mass=m.ravel(); row=np.asarray(abs(self.K).sum(axis=1)).ravel(); self.mode=sol.mode
        if self.mode=="relaxation":
            self.mass=np.maximum(row/float(sol.relaxation_safety)**2,1e-20); self.dt=1.; self.n_steps=int(sol.n_steps); self.total_time=float(self.n_steps)
        else:
            self.mass=np.maximum(self.physical_mass,1e-20); active=row>1e-20
            active &= ~self._held_dofs()
            root=float(np.sqrt(np.min(self.mass[active]/row[active])))
            self.dt=float(sol.dt_safety*root)
            self.n_steps=max(1,int(round(sol.total_time/self.dt))); self.total_time=float(sol.total_time)
        damp=sol.damping; self.damp_kind=damp.type or ("kinetic" if self.mode=="relaxation" else "viscous")
        self.zeta,self.alpha,self.beta=float(damp.coeff),float(damp.alpha),float(damp.beta)
        # the undamped bound was set before the damping it must survive was read
        self._shrink_step_for_damping(1.0/root if self.mode!="relaxation" else 0.0)
        self.tolerance=float(sol.tolerance); self.kinetic_tolerance=float(sol.kinetic_tolerance)

    def _surface_load(self,nodes,ld):
        kind=self.p.case.element.type
        name="shell3" if kind=="shell3" else "shell4"
        # THE REGION'S OWN ELEMENTS where Gmsh recorded them: "every node in
        # the set" also took the strip between two wheel patches, whose
        # nodes all belong to one patch or the other -- 50% more load, and
        # the run converged. A geometric region keeps the
        # node rule, the only thing it can mean.
        rec=None
        if hasattr(self.mesh,"region_elements") and ld.get("region") is not None:
            rec=self.mesh.element_regions.get(ld["region"],{}).get(name)
        if rec is not None:
            ids=np.asarray(rec,dtype=np.int64)
        else:
            selected=np.zeros(self.nn,bool); selected[np.asarray(nodes,int)]=True
            ids=np.flatnonzero(selected[self.mesh.elements].all(axis=1))
        if not len(ids): raise ValueError(f"shell surface load region contains no complete {name} elements")
        F=np.zeros(self.ndof)
        div=3 if kind=="shell3" else 4
        for e in ids:
            if ld.get("pressure") is not None: value=-float(ld["pressure"])*self.data[e].frame[2]
            else: value=np.asarray(ld["traction"],float)
            self.dofs.add_nodal(F,self.mesh.elements[e],value*self.area[e]/div,TRANSLATION)
        return F

    def _nodal_fields(self,actions,recover_stress=False):
        out={"displacement":self.u.copy(),"rotation":self.dofs.field(self.q,ROTATION).copy()}
        if hasattr(self,"last_reaction_force"):
            out["reaction_force"]=self.last_reaction_force; out["reaction_moment"]=self.last_reaction_moment
        if actions is not None:
            names=("membrane_x","membrane_y","membrane_xy","moment_x","moment_y","moment_xy","shear_x","shear_y")
            out.update({name:actions[:,i] for i,name in enumerate(names)})
            t=float(self.p.case.element.thickness); s=np.zeros((self.nel,6)); s2=np.zeros_like(s)
            top=actions[:,:3]/t+6*actions[:,3:6]/t**2
            bottom=actions[:,:3]/t-6*actions[:,3:6]/t**2
            s[:,0],s[:,1],s[:,3]=top[:,0],top[:,1],top[:,2]
            s2[:,0],s2[:,1],s2[:,3]=bottom[:,0],bottom[:,1],bottom[:,2]
            # transverse shear vanishes at the traction-free faces; its peak,
            # 1.5 Q / t, is at mid-surface (`shear_x`, `shear_y` are Q). It
            # was written here as Q / t
            s[:,5]=s2[:,5]=0.0; s[:,4]=s2[:,4]=0.0
            out["shell_stress_top"]=s; out["shell_stress_bottom"]=s2
            out["von_mises_top"]=np.sqrt(s[:,0]**2-s[:,0]*s[:,1]+s[:,1]**2+3*s[:,3]**2)
            out["von_mises_bottom"]=np.sqrt(s2[:,0]**2-s2[:,0]*s2[:,1]+s2[:,1]**2+3*s2[:,3]**2)
        return out

    def summary(self):
        e=self.p.case.element; kind=e.type
        name="Tri3" if kind=="shell3" else "Quad4"
        return (f"  mesh          {self.nn} nodes, {self.nel} {name} elements\n"
                f"  element       {kind} / Mindlin-Reissner co-rotational, 6 DOF/node\n"
                f"  section       thickness={e.thickness:.6e} m, kappa={e.shear_correction:.6g}\n"
                f"  stabilization selective shear + energy-scaled hourglass + continuum-spin drilling\n"
                f"  material      linear_elastic; backend {self.backend}\n"
                f"  mode          {self.mode}; dt={self.dt:.4e}, {self.n_steps} steps")
