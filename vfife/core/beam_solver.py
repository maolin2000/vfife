"""Co-rotational 3-D Euler--Bernoulli and Timoshenko frame solver.

It deliberately uses the same user-facing contract as :mod:`truss_solver`:
named node regions, amplitudes, steps, physical mass and XDMF fields.  The
section law is linear elastic with six DOFs per node.  Rigid translation and
rotation are removed before either section law is evaluated.
"""
from __future__ import annotations


import numpy as np

from .problem import Problem

from . import mass_scaling
from . import member_axes
from . import added_mass
from . import member_geometry
from .dof import (DofLayout, ROTATION, SparseAssembler,
                  repeated_rotation)
from .beam_family import BeamFamily
from .member_failure import Breaking
from .rotations import beam_frames_from_triads


# one monitor record for every solver (core/monitors)
from .monitors import Monitor  # noqa: F401  (the name this module exported)


def _norm_fn(solver):
    """The 2-norm the convergence measures take: under `backend: numba` the
    compiled one the compiled relaxation takes too (1.9) --
    sqrt(x . x), what np.linalg.norm computes -- so that loop and kernel
    agree whichever BLAS NumPy is linked to (the ledger's dot products
    are still NumPy's in the loop, numba's in the kernel)."""
    if getattr(solver, "backend", "numpy") == "numba":
        from ..kernels.numba_step import norm2
        if norm2 is not None:
            return lambda x: norm2(np.ascontiguousarray(x, dtype=float))
    return np.linalg.norm


def _kinetic(solver):
    """0.5 * sum(m v^2) of a relaxation iteration; compiled under `backend:
    numba` (NumPy's pairwise sum, the compiled relaxation's own)."""
    if getattr(solver, "backend", "numpy") == "numba":
        from ..kernels.numba_step import kinetic_energy
        if kinetic_energy is not None:
            buf = solver.__dict__.get("_ke_buf")
            if buf is None or buf.shape != solver.v.shape:
                buf = solver._ke_buf = np.empty(solver.v.shape)
            return float(kinetic_energy(solver.mass, np.ascontiguousarray(solver.v, dtype=float), buf))
    return float(0.5*np.sum(solver.mass*solver.v*solver.v))

class BeamSolver(BeamFamily):
    """Six-DOF co-rotational Line2 frame solver."""
    #: A CLASS attribute so every subclass has it without having to
    #: remember: `ShellSolver` overrides __init__ and does not run the
    #: line below that sets it, which is exactly how it went missing.
    #: True means the next velocity update is a central-difference
    #: restart and takes HALF the acceleration.
    _half = True
    RELAX_HINT = "; reduce relaxation_safety or check for a structural mechanism"

    BEAM_TYPES = frozenset(("beam2", "beam2_timoshenko"))

    def __init__(self, problem, mesh, material, log=print):
        problem = Problem.of(problem)        # typed, and its dict
        if material.name != "linear_elastic":
            raise ValueError("beam elements currently support linear_elastic only")
        self.backend = problem.solver.backend
        if self.backend not in ("numpy", "numba"):
            raise ValueError("beam elements support solver.backend = numpy or numba, "
                             f"not {self.backend!r}")
        if self.backend == "numba":
            from ..kernels import numba_frame
            if not numba_frame.available():
                raise ValueError("solver.backend = numba but numba is not installed")
        self.p, self.mesh, self.material, self.log = problem, mesh, material, log
        self.nn, self.nel = mesh.n_nodes, mesh.n_elem
        self.dofs = DofLayout.frame(self.nn)
        self.ndof = self.dofs.size
        if mesh.elements.shape[1] != 2:
            raise ValueError("beam elements require Gmsh Line2 elements")
        self._setup_regions()
        self._reserve_split_particles()
        self._assemble_reference()
        # analysis steps (1.2): what each step builds, removes and frees;
        # None for a case whose steps change nothing (core/analysis_steps)
        from . import analysis_steps
        self.plan = analysis_steps.plan(self)
        # soil springs on piles: their stiffness is in the
        # reference matrix the step and the relaxation masses come from
        from . import soil
        soil.attach(self)
        soil.add_stiffness(self)
        # where each member's reference block sits in K: its element's dofs,
        # until a stiffness-damping edit takes it out or moves it
        # (`_drop_from_reference_stiffness`, analysis_steps._reference_K);
        # -1 where K does not hold it. The relaxation mass reads it
        # (`BeamFamily._turned_blocks`)
        self._k_dofs = np.array([self.dofs.element_dofs(c) for c in self.mesh.elements],
                                dtype=np.int64).reshape(self.nel, -1)
        self._setup_time_dependence()
        self._setup_mass_and_time()
        self._setup_breaking()
        self.q = np.zeros(self.ndof)
        self.v = np.zeros(self.ndof)
        self._init_orientation()
        self._carried_step()
        #: the next velocity update is a central-difference RESTART and
        #: takes half the acceleration; set again wherever v is zeroed
        self._half = True
        self.ke_prev = 0.0
        self.n_resets = 0
        self.step = self.step_index = self.step_start = 0
        self.rebar = []
        if self.p.case.rigid_links or any(st.rigid_links for st in self.p.case.steps):
            # the links are made where the nodes stand now, before the steps
            # are installed (core/constraints)
            self.constraint_layer()
        if self.plan is not None:
            analysis_steps.install(self)

    @staticmethod
    def _release_indices(props):
        release_index = {"i_rx": 3, "i_ry": 4, "i_rz": 5,
                         "j_rx": 9, "j_ry": 10, "j_rz": 11}
        return [release_index[r] for r in props.get("releases", [])]

    def _local_stiffness(self, L, props=None, condense=True, material=None):
        """Return the 12x12 local section stiffness.

        ``beam2`` uses the Euler--Bernoulli limit.  For
        ``beam2_timoshenko`` each bending plane uses the exact two-node
        Timoshenko stability parameter ``phi = 12 EI/(kappa G As L^2)``.
        This avoids shear locking and tends continuously to the existing
        Euler--Bernoulli matrix as shear rigidity grows.

        `material` is THIS MEMBER'S material where it has one of its own;
        left out it is the model's, which is what a
        single-material model passes and what makes it identical to the bit.
        """
        e = props or self.p["element"]
        m = self.material if material is None else material
        E, nu = m.E, m.nu
        A, Iy, Iz, J = (float(e[k]) for k in ("area", "iy", "iz", "j"))
        G = E / (2.0 * (1.0 + nu))
        k = np.zeros((12, 12))
        def pair(a, b, value):
            k[a, a] += value; k[b, b] += value; k[a, b] -= value; k[b, a] -= value
        pair(0, 6, E * A / L)
        pair(3, 9, G * J / L)
        # (a dict read still: preflight's probes hold `p = {"element": {}}`;
        # this goes with the member properties, P2's member families)
        timoshenko = e.get("type", self.p["element"].get("type")) == "beam2_timoshenko"
        if timoshenko:
            phi_y = 12.0 * E * Iz / (
                float(e.get("kappa_y", 1.0)) * G * float(e["ay"]) * L**2
            )
            phi_z = 12.0 * E * Iy / (
                float(e.get("kappa_z", 1.0)) * G * float(e["az"]) * L**2
            )
        else:
            phi_y = phi_z = 0.0
        # local v / rz: y-shear and bending about z use Ay, Iz
        c = E * Iz / (L**3 * (1.0 + phi_y))
        ids = [1, 5, 7, 11]
        block = c * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, (4+phi_y)*L*L, -6*L, (2-phi_y)*L*L],
            [-12, -6*L, 12, -6*L],
            [6*L, (2-phi_y)*L*L, -6*L, (4+phi_y)*L*L],
        ])
        k[np.ix_(ids, ids)] += block
        # local w / ry: z-shear and bending about y use Az, Iy
        c = E * Iy / (L**3 * (1.0 + phi_z))
        ids = [2, 4, 8, 10]
        block = c * np.array([
            [12, -6*L, -12, -6*L],
            [-6*L, (4+phi_z)*L*L, 6*L, (2-phi_z)*L*L],
            [-12, 6*L, 12, 6*L],
            [-6*L, (2-phi_z)*L*L, 6*L, (4+phi_z)*L*L],
        ])
        k[np.ix_(ids, ids)] += block
        rel = self._release_indices(e)
        if rel and condense:
            keep = np.array([i for i in range(12) if i not in rel])
            # Static condensation makes the released internal rotation moment-free.
            kaa, kab, kbb = k[np.ix_(keep, keep)], k[np.ix_(keep, rel)], k[np.ix_(rel, rel)]
            if np.linalg.matrix_rank(kbb) < len(rel):
                raise ValueError("beam2 releases create a singular internal member mechanism")
            out = np.zeros_like(k)
            out[np.ix_(keep, keep)] = kaa - kab @ np.linalg.solve(kbb, kab.T)
            k = out
        return k

    def _material_of(self, e):
        """The material of element `e` -- the model's unless the member's own
        group named one."""
        mats = getattr(self, "member_material", None)
        return self.material if mats is None else mats[e]

    def _apply_load_releases(self, fl, L, props, material=None):
        """Condense fixed-end loads through the same released internal DOFs."""
        rel = self._release_indices(props)
        if not rel:
            return fl
        keep = np.array([i for i in range(12) if i not in rel])
        raw = self._local_stiffness(L, props, condense=False, material=material)
        out = np.zeros(12)
        out[keep] = fl[keep] - raw[np.ix_(keep, rel)] @ np.linalg.solve(
            raw[np.ix_(rel, rel)], fl[rel])
        return out

    def _assemble_reference(self):
        assembly = SparseAssembler(self.ndof)
        self.T, self.R0, self.k_local, self.length, self._kg_ref = [], [], [], [], []
        orient = self.p.case.element.orientation
        self.geometry = geometry = member_geometry.MemberGeometry.build(
            self.mesh, self.nel, [(np.arange(self.nel), self.p.case.element.end_offsets, orient, "all")])
        for e, conn in enumerate(self.mesh.elements):
            R, L = self._frame(*geometry.reference_ends(e), orient)
            T = repeated_rotation(R)
            kl = geometry.full_axial(e, self._local_stiffness(L), L)
            kg = geometry.stiffness_to_nodes(e, T.T @ kl @ T)
            dof = self.dofs.element_dofs(conn)
            assembly.add_matrix(dof, kg)
            self.T.append(T); self.k_local.append(kl); self.length.append(L)
            self.R0.append(R); self._kg_ref.append(kg)
        self.K = assembly.csr()
        self.T, self.R0, self.k_local, self.length = (np.asarray(self.T), np.asarray(self.R0),
                                                        np.asarray(self.k_local), np.asarray(self.length))

    def _kernel_dth(self):
        """The imposed deformation the frame law subtracts in the compiled
        step: the plastic hinge rotations, per ELEMENT (1.6), zeros when there
        are none."""
        pl = getattr(self, "_plastic_d", None)
        if pl is not None and pl.shape == (self.nel, 12):
            return pl
        dth = getattr(self, "_zero_dth_el", None)
        if dth is None or dth.shape != (self.nel, 12):
            dth = self._zero_dth_el = np.zeros((self.nel, 12))
        return dth

    def kernel_members(self):
        """The members' arrays for the compiled step (core/protocol.KernelMembers,
        v1.18)."""
        from .protocol import KernelMembers
        from ..kernels.numba_frame import configure_threads
        dth = self._kernel_dth()
        self._frame_batch()
        if getattr(self, "_threads", None) is None:
            self._threads = configure_threads(self.p.solver.threads)
        return KernelMembers(
            f=(np.arange(self.nel, dtype=np.int64), np.ascontiguousarray(self._fr_conn, dtype=np.int64),
               self._fr_R0, self._fr_k, self._fr_L0, self._fr_dofs, dth),
            collapse="beam2 member collapsed to zero length",
            t_ids=np.zeros(0, np.int64), t_conn=np.zeros((0, 2), np.int64), t_EA=np.zeros(0),
            threads=int(getattr(self, "_threads", 1) or 1), connectors=None)

    def _turned_k(self, e, T):
        """(member `e`'s block turned into the frame `T`, at the modulus
        `k_local` holds now -- an aged one too -- and the block as `K` holds
        it) for the relaxation mass (`BeamFamily._refresh_fictitious_mass`)."""
        return T.T @ self.k_local[e] @ T, self._kg_ref[e]

    def rebatch(self, frames=True, axial=True):
        """The members changed (a step built or removed some, a split moved
        an end, an age changed their stiffness): the batched member arrays
        are gathered again at their next use -- this solver's `_frame_batch`,
        the mixed solver's member families (`MixedStructuralSolver.rebatch`).
        `axial`: the bars and connectors, which only the mixed solver has."""
        if frames:
            self._fr_dofs = None; self.whole_frames = None

    #: the frame members as a family (core/families): every
    #: member, over `_frame_batch`. Not `frame_family`, which is the mixed
    #: solver's own (its frame members among its bars and connectors): the
    #: mixed solver inherits `_member_forces` from this class, and a call of
    #: this class's on a mixed solver (a test pins the two paths to each
    #: other) must not replace that solver's family
    whole_frames = None

    def _frame_members(self):
        """This solver's members as a family, gathered again whenever the
        batch under it was (`_frame_batch`)."""
        dofs = self._frame_batch()
        f = self.whole_frames
        if f is None or f.dofs is not dofs:
            from .families import FrameMembers
            f = self.whole_frames = FrameMembers.whole(self.nel, dofs, self._fr_conn, self._fr_R0,
                                                       self._fr_k, self._fr_L0, self._fr_zero_dth)
        return f

    def _frame_eigen_d(self):
        """The deformation this solver's frame law subtracts before the
        stiffness sees it: the thermal (and creep, shrinkage) one. (The
        mixed solver's `_eigen_d` adds its plastic hinges' to it, for its
        own frame family; this one is this class's, on a mixed solver too.)"""
        return getattr(self, "_thermal_d", None)

    def _frame_batch(self):
        """The dof map and the reference arrays, gathered once.

        Rebuilt on every internal force until v0.1, which meant a
        Python loop over the elements inside the time loop -- the one thing
        the batched path was written to remove.
        """
        if getattr(self, "_fr_dofs", None) is None:
            conn = np.ascontiguousarray(self.mesh.elements[:, :2], dtype=np.int64)
            self._fr_conn = conn
            self._fr_dofs = np.ascontiguousarray(
                np.array([self.dofs.element_dofs(c) for c in conn]).reshape(self.nel, 12),
                dtype=np.int64)
            self._fr_k = np.ascontiguousarray(np.asarray(self.k_local, dtype=float))
            self._fr_R0 = np.ascontiguousarray(np.asarray(self.R0, dtype=float))
            self._fr_L0 = np.ascontiguousarray(np.asarray(self.length, dtype=float))
            self._fr_zero_dth = np.zeros((self.nel, 12))
        return self._fr_dofs

    def _internal_force_numba(self):
        from ..kernels.numba_frame import configure_threads
        f = self._frame_members()
        if getattr(self, "_threads", None) is None:
            self._threads = configure_threads(self.p.solver.threads)
        out = np.zeros(self.ndof)
        end_actions = np.zeros((self.nel, 12))
        actions = np.zeros((self.nel, 6))
        f.forces(self, out, actions, end_actions)
        self.end_actions = end_actions
        self._check_failure()
        return out, actions

    def _member_forces(self):
        """VFIFE-style member forces: reverse rigid motion, classical local beam,
        then rotate equivalent forces/moments back to the physical configuration.
        The frame and the local rotations are `rotations.beam_frames_from_triads`,
        exact in space, from the points' orientations: the
        frame members' family (`families.FrameMembers`) does it."""
        if getattr(self, "backend", "numpy") == "numba":
            out, actions = self._internal_force_numba()
            from . import soil
            return soil.add_force(self, out), actions
        f = self._frame_members()
        out = np.zeros(self.ndof)
        end_actions = np.zeros((self.nel, 12))
        actions = np.zeros((self.nel, 6))
        f.forces(self, out, actions, end_actions)
        self.end_actions = end_actions
        self._check_failure()
        from . import soil
        return soil.add_force(self, out), actions

    # ------------------------------------------------------------------ #
    #  Elastic-brittle breaking and scheduled removal (core/member_failure)
    # ------------------------------------------------------------------ #
    #: what a break does when the case does not say: the frame solvers split
    #: (core/member_failure.py has the two alternatives)
    SPLIT_DEFAULT = True

    def _breakable_specs(self):
        """(failure specs of the element / member groups, removal specs)
        as the case wrote them -- read before anything is sized, to know
        which members may ever break and whether they split."""
        e = self.p["element"]
        groups = [(np.arange(self.nel), e["failure"])] if e.get("failure") else []
        return groups, list(self.p.case.removals)

    def _reserve_split_particles(self):
        """Park a spare particle on every end of every member that can break
        and is to split, BEFORE the dof layout, the masses and the reference
        stiffness are built at the final size (the fibre beam's
        `_reserve_split_particles` has the reasons: regions first so a spare is
        in none, everything else after so no live array is ever resized).
        `_spare_of[e]` is the pair of particles element e's ends may move onto,
        -1 where it may not; `bonds` is the graph that moves them."""
        self._spare_of = None
        self.bonds = None
        if self.p.solver.mode != "dynamic":
            return
        groups, removals = self._breakable_specs()
        def wants(spec):
            # a failure spec (a dict) or a removal (schema.RemovalSpec)
            v = spec.get("split") if isinstance(spec, dict) else getattr(spec, "split", None)
            return self.SPLIT_DEFAULT if v is None else bool(v)
        can = np.zeros(self.nel, bool)
        for ids, spec in groups:
            if wants(spec):
                can[np.asarray(ids, int)] = True
        for r in removals:
            if wants(r):
                ids = self.mesh.region_elements(r.region, "truss2", self.mesh.elements)
                if ids is not None:
                    can[np.asarray(ids, int)] = True
        if not can.any():
            return
        which = np.flatnonzero(can)
        base = self.nn
        spare = np.full((self.nel, 2), -1, dtype=np.int64)
        spare[which] = base + 2 * np.arange(len(which))[:, None] + np.arange(2)[None, :]
        parked = self.mesh.nodes[self.mesh.elements[which][:, :2]].reshape(-1, 3)
        self.mesh.nodes = np.vstack([self.mesh.nodes, parked])
        self.nn = self.mesh.n_nodes
        self.dofs = DofLayout.frame(self.nn)
        self.ndof = self.dofs.size
        self._spare_of = spare
        self._elements0 = self.mesh.elements[:, :2].copy()      # the connectivity as read
        from . import topology
        reserved = np.where(spare >= 0, spare, self.mesh.elements[:, :2])
        self.bonds = topology.BondGraph(self.mesh.elements[:, :2], self.nn, reserved=reserved)
        self.log(f"  split         {len(which)} member(s) may split off: {2 * len(which)} spare particles reserved")

    def _setup_breaking(self):
        """A single-section frame model: `element.failure` is the capacity
        of every member, from the element's own A, I_y, I_z."""
        e = self.p["element"]
        self.breaking = Breaking(self.nel, self.mesh, self.p.case.removals, self.log,
                                 split_default=self.SPLIT_DEFAULT)
        self.member_active = self.breaking.member_active
        self.capacity_ratio = self.breaking.capacity_ratio
        self.failure_events = self.breaking.failure_events
        if e.get("failure"):
            self.breaking.beams(np.arange(self.nel), e["failure"], e["area"], e["iy"], e["iz"])
            self.breaking.twist_caps(np.asarray(self.k_local)[:, 3, 3])     # G J / L
        self._reload_step = False
        self.splits = []                       # (t, element, parent, new) per moved end
        self._fragment_parent = {}             # spare particle -> the particle it was split from
        self._appendages = []                  # massless members carried rigidly by one particle

    def _element_mass_share(self, e):
        """The (6,) mass one of member e's two particles carries for it."""
        el = self.p["element"]
        L = float(self.length[e]); A = float(el["area"]); rho = self.material.rho
        out = np.full(6, rho * A * L / 2.0)
        # the true rotary inertia, rotated to global axes:
        # rho [Ip, Iy, Iz] L/2 for Timoshenko; the Euler beam keeps rho A L^3/24
        # about its two transverse axes and takes rho Ip L/2 about its own
        Iy, Iz = float(el["iy"]), float(el["iz"])
        if el.get("type") == "beam2_timoshenko":
            local = rho * L * np.array([Iy + Iz, Iy, Iz]) / 2.0
        else:
            local = np.array([rho * (Iy + Iz) * L / 2.0] + [rho * A * L ** 3 / 24.0] * 2)
        out[3:] = mass_scaling.rotary_diagonal(self.R0[e], local)
        return out

    def _weightless_add(self, n, dm):
        """Book `dm` kg on particle n's x, y, z as mass gravity does not see
        (the split's stability floor, v1.0; v1.0's device)."""
        w = getattr(self, "weightless_mass", None)
        if w is None or np.shape(w) != (self.nn, 3):
            w0 = np.zeros((self.nn, 3))
            if w is not None and np.size(w):
                w = np.asarray(w, float)
                w0[:len(w)] = w[:self.nn]
            w = w0
        else:
            w = np.array(w, float, copy=True)
        w[int(n)] += float(dm)
        self.weightless_mass = w

    def _check_failure(self):
        """After a force evaluation of a dynamic run: hand the end actions to
        the table and do to the members it says have broken what their spec
        asks -- split them off, or take them out."""
        b = getattr(self, "breaking", None)
        if b is None or self.mode != "dynamic" or not b.can_break:
            return
        if getattr(self, "_probing", False) and not getattr(self, "_closing", False):
            # A QUESTION DOES NOT BREAK A MEMBER. A progress
            # check or an output frame evaluates the forces of the state the
            # NEXT step starts from; a member at its capacity there was
            # broken by that evaluation, one step before its own step would
            # have -- it carried nothing in the step it should have carried
            # its last force in, and the answer depended on `output.every`
            # and on `progress_every` (the fibre beam's fracture: 1.3 mm at
            # the end of a run, found in the review of v1.12; the hinges
            # had the same fault). The step finds it; the run's last
            # frame, after which there is no step, still does (`_closing`).
            # The ratios are written all the same: a frame's capacity_ratio
            # is of the forces in that frame.
            b.check(0.0, self.end_actions, observe=True)
            return
        t = self.step * self.dt
        if self.plan is not None:
            # the clock of the dynamic steps (1.2), not the iteration count
            t = self.staging.clock + (self.step - self.step_start) * self.dt
        gone = b.check(t, self.end_actions)
        if len(gone):
            self._break(gone)
            b.diagnostics(getattr(self, "diagnostics", None))

    def _break(self, ids):
        ids = np.asarray(ids, int)
        if not len(ids):
            return
        # what THIS break asked for: a removal's `split`, or the failure's
        split = ids[self.breaking.split_on_break[ids]] if self.bonds is not None else np.zeros(0, int)
        gone = ids[~np.isin(ids, split)]
        self._deactivate(gone)
        if len(gone):
            self._park_check = set(getattr(self, "_park_check", set())) | {int(n) for n in self.mesh.elements[gone, :2].ravel()}
        # A SPLIT WAITS FOR THE END OF THE STEP. The break is found inside the
        # force evaluation, and the step goes on to divide THOSE forces by the
        # mass: moving the mass now would pair the old connectivity's forces
        # with the new masses -- a pin's rotation whose beam had just left
        # got the beam's end moment over 1e-20 kg and reached 1e18 rad/s.
        # The fibre beam queues its splits for the same reason.
        self._pending_split = list(getattr(self, "_pending_split", [])) + [int(e) for e in split]

    def apply_pending_splits(self, t=None):
        """The splits and removals queued during the step (`t` is the fibre
        solver's: this one stamps them with its own step's time)."""
        pending = getattr(self, "_pending_split", None)
        if pending:
            self._pending_split = []
            self._split_off(np.asarray(sorted(set(pending)), int))
        check = getattr(self, "_park_check", None)
        if check:
            self._park_check = set()
            self._park_bodiless(check, self.step * self.dt)
            # a member removed whole can set a piece of the frame loose too
            if self.bonds is not None and getattr(self, "_kg_ref", None) is not None:
                self._debris_isotropic(self._grounded(self._split_held()), self._energy_ledger())

    def _split_off(self, ids):
        """Move both ends of the members `ids` onto their spare particles.

        The member's share of the mass leaves the frame's particle and lands
        on the fragment's; the fragment's particle starts with the frame
        particle's position, velocity and orientation (mass and momentum are
        conserved at the instant, nothing is invented -- the fibre beam's
        `apply_pending_splits` has the argument). The member keeps its
        stiffness: it is a free elastic segment now, with its own weight.
        Its new particles join the regions the LOADS name -- gravity above
        all -- and no constraint region: a fragment is not clamped to the
        support it broke away from. The step's load vectors are rebuilt on
        the next step (`_reload_step`), because gravity is mass times g and
        the mass has moved."""
        ids = np.asarray(ids, int)
        if not len(ids):
            return
        t = self.step * self.dt
        pm = self.physical_mass.reshape(self.nn, self.dofs.stride)
        # a particle with a constraint on any of its dofs is the world's, not
        # the fragment's: an end there is duplicated even when it is the
        # particle's only bond -- a driven support too
        held = self._split_held()
        # every particle a split touches -- the fragment's and the one it
        # left -- must be stable at the step the run is on: the same rule the
        # setup applies (dt = dt_safety sqrt(m / row)), from the reference
        # stiffness rows of the members still on it, so a light member (a
        # crossbeam whose mass the model smeared into the deck) and a node
        # whose heavy member just left do not blow up -- selective mass
        # scaling for the fragment, v0.4's device at v0.5's moment
        need = (self.dt / float(self.p.solver.dt_safety)) ** 2
        stride = self.dofs.stride
        added = 0.0
        moved = []
        touched = set()
        # A SPLIT NEEDS A BODY. A member whose own mass is nothing next to
        # what its stiffness demands at this step -- a crossbeam modelled at
        # rho = 1 with its mass smeared into the deck, a rigid arm -- cannot
        # be a fragment: it would be a massless spring let go with its strain
        # energy, and Xihoumen's went off at 14 km/s. It is deactivated
        # instead, which is what a modelling device that broke deserves.
        device = self._device_members()
        massless = [int(e) for e in ids if device[e]]
        if massless:
            self.breaking.member_active[massless] = False
            self._deactivate(massless)
            self.log(f"  {len(massless)} broken member(s) had no mass of their own to split off with "
                     f"(rho ~ 0): deactivated instead")
            ids = ids[~np.isin(ids, massless)]
            for e in massless:
                self.breaking.take_ends(e)                    # whole, whichever end broke
                touched.update(int(x) for x in self.mesh.elements[e, :2])
        ledger = self._energy_ledger()
        for e in ids:
            share = self._element_mass_share(e)
            # only the end(s) that broke: the member stays welded at the
            # other
            for k, parent, new in self.bonds.detach(int(e), held=held, ends=self.breaking.take_ends(e)):
                # the mass, the state and the ledger (core/beam_family)
                self._move_end(e, parent, new, share, pm, ledger)
                touched.update((int(parent), int(new)))
                moved.append((e, k, parent, new))
            self._drop_from_reference_stiffness([e], add_back=True)
            self.mesh.elements[e, :2] = self.bonds.bonds[e]
        if not moved and not touched:
            return
        # THE PARENT NEEDS THE FLOOR AS MUCH AS THE FRAGMENT. The particle a
        # girder left keeps its crossbeam and its rigid arm, whose stiffness
        # the girder's mass was carrying at the setup bound: Xihoumen's first
        # girder split left a deck node with 8.8 kg on a 2e12 N/m arm, and
        # the run was at 1e40 within twenty steps
        bonds = self.bonds.bonds
        # A DEVICE NEEDS SOMETHING TO HANG FROM. A massless member (a crossbeam
        # whose weight the model put into the girders' density, a rigid arm)
        # is part of the massive members it is attached to; a cluster of
        # devices that touches no massive member and no support anywhere is
        # a spring with artificial masses on its ends -- in the k = 3 run a
        # deck node whipped round the sea floor at 350 m/s on one. Such a
        # CLUSTER is removed (cause 2 in member_failures, "connects nothing").
        # A cluster still attached to a girder at one end -- an outrigger
        # whose hanger was cut -- stays: it is a cantilever off that girder
        # (the first version removed the devices at any particle left with
        # only devices on it, and took the outriggers of the cut hangers in
        # the k = 2 act with them).
        orphaned = []
        active = self.breaking.member_active
        on_n = lambda n: np.flatnonzero(((bonds[:, 0] == n) | (bonds[:, 1] == n)) & active)
        # a massless bar that holds nothing up any more (core/beam_family)
        kind = getattr(self, "member_kind", None)
        if kind is not None:
            for e2 in self._hanging_devices(device, held, device & (np.asarray(kind) == "truss2"), bonds):
                active[e2] = False
                orphaned.append(int(e2))
                touched.update(int(x) for x in bonds[e2])
        grounded = self._grounded(held)
        seen = set()
        # the clusters to look at: those at the particles this split touched,
        # and every one on a piece of debris (a girder segment that fell away
        # takes its outriggers with it without any split touching them)
        starts = set(touched)
        loose = ~grounded
        if loose.any():
            dev_on = bonds[active & device]
            starts.update(int(n) for n in dev_on.ravel() if loose[n])
        for start_n in sorted(starts):
            for e0 in on_n(start_n):
                if not device[e0] or int(e0) in seen:
                    continue
                cluster, nodes_c, stack, anchors = [], set(), [int(e0)], set()
                while stack:
                    e2 = stack.pop()
                    if e2 in seen:
                        continue
                    seen.add(e2); cluster.append(e2)
                    for n in bonds[e2]:
                        n = int(n)
                        if n in nodes_c:
                            continue
                        nodes_c.add(n)
                        if held[n]:
                            anchors.add(n)
                        for e3 in on_n(n):
                            if not device[e3]:
                                anchors.add(n)
                            elif int(e3) not in seen:
                                stack.append(int(e3))
                if not anchors:
                    for e2 in cluster:
                        active[e2] = False
                    orphaned.extend(cluster)
                    touched.update(nodes_c)
                elif len(anchors) == 1 and not grounded[next(iter(anchors))]:
                    # ...but only on the standing structure. On a piece of
                    # debris the anchor is the end of a free girder segment,
                    # and ITS rotation -- a single element rolling about its
                    # own axis, with none of the crossbeam's rotary inertia
                    # (rho = 1: that mass sits on the girder's axis) -- swung
                    # 9 m arms round at 150 m/s in the k = 3 run. A massless
                    # connector on a piece of debris is no part of the debris'
                    # mechanics: removed, like the unanchored ones.
                    for e2 in cluster:
                        active[e2] = False
                    orphaned.extend(cluster)
                    touched.update(nodes_c)
                elif len(anchors) == 1:
                    # AN APPENDAGE: massless members hanging from ONE massive
                    # particle -- an outrigger whose hanger was cut, a crossbeam
                    # still on one girder segment. Mechanically it is inert (no
                    # mass, nothing on its free end); left elastic, its free
                    # end is a spring with an artificial mass that every
                    # impact of the segment whips about (461 m/s in the k = 3
                    # run). It rides rigidly with its anchor instead: drawn
                    # where it is, carrying nothing.
                    self._make_appendage(cluster, next(iter(anchors)), nodes_c)
        released = self._release_appendages(t, device, held, bonds, grounded)
        for e2 in released:
            touched.update(int(n) for n in bonds[e2])
        orphaned.extend(released)
        if orphaned:
            self._deactivate(orphaned)
            self.breaking.member_failed[orphaned] = True
            self.breaking.failure_events.extend([float(t), e2, 0.0, 2.0] for e2 in orphaned)
            self.log(f"  {len(orphaned)} massless member(s) left attached to nothing massive, or to debris: removed")
        self._park_bodiless(touched, t)
        if getattr(self, "_kg_ref", None) is not None:
            for n in touched:
                row = np.zeros(stride)
                for e2 in np.flatnonzero((bonds[:, 0] == n) | (bonds[:, 1] == n)):
                    if not self.member_active[e2]:
                        continue
                    k2 = 0 if bonds[e2, 0] == n else 1
                    row += np.abs(self._kg_ref[e2]).sum(axis=1)[k2 * stride:(k2 + 1) * stride]
                floor = row * need
                old_m = pm[n].copy()
                new_m = np.maximum(old_m, floor)
                # ONE TRANSLATIONAL MASS PER PARTICLE: the same dm on x, y
                # and z, the largest any of them needs. Floored per dof, a
                # fragment had 36.6 kg in x and 7.9 kg in y and z, and its
                # velocity was cut 89% in x only
                dm = float(np.max(new_m[:3] - old_m[:3]))
                new_m[:3] = old_m[:3] + max(dm, 0.0)
                new_m[3:] = np.maximum(old_m[3:], floor[3:])
                # ON DEBRIS, ONE ROTARY INERTIA FOR ALL THREE AXES:
                # the floor rows and the diagonal are the reference frame's,
                # and a fragment tumbles out of it -- see _debris_isotropic
                if not grounded[n]:
                    new_m[3:] = float(np.max(new_m[3:]))
                if (new_m > old_m).any():
                    # the mass that makes the particle stable is added with
                    # MOMENTUM conserved (v scaled by m_old / m_new): an
                    # addition that kept v would inject 0.5 dm v^2. The
                    # kinetic energy that goes is booked as dissipated.
                    nd = self.dofs.node_dofs(n).ravel()
                    ratio = np.where(new_m > 0, old_m / np.maximum(new_m, 1e-300), 1.0)
                    v0 = self.v[nd].copy()
                    self.v[nd] = v0 * ratio
                    if ledger is not None:
                        ledger.dissipate(float(0.5 * np.sum(old_m * v0 ** 2 - new_m * self.v[nd] ** 2)))
                    # and it does not weigh: a stability device, not steel --
                    # gravity on it doubled a fragment's weight
                    if dm > 0.0:
                        self._weightless_add(n, dm)
                    added += max(dm, 0.0)
                    pm[n] = new_m
        self.mass = np.maximum(self.physical_mass, 1e-20)
        self.rebatch()
        self._reload_step = True
        self._debris_isotropic(grounded, ledger)
        self.splits.extend((float(t), int(e), int(p), int(n)) for e, _k, p, n in moved)
        if isinstance(getattr(self, "diagnostics", None), dict):
            self.diagnostics["member_splits"] = self.splits
        self.log(f"  split: {len(set(int(e) for e, *_ in moved))} member(s) off at t = {t:.6g} s "
                 f"({self.bonds.n_components()} component(s) now"
                 + (f", {added:.3g} kg added to keep the fragments stable at this step)" if added > 0 else ")"))

    def _make_appendage(self, members, anchor, nodes_c):
        """Take `members` out of the force evaluation and have their other
        particles follow `anchor` rigidly from now on (`_carry_appendages`)."""
        members = [int(e) for e in members]
        self.breaking.member_active[members] = False
        self.capacity_ratio[members] = np.nan
        self._deactivate(members)
        tips = sorted(int(n) for n in nodes_c if n != anchor)
        ap = {"anchor": int(anchor), "members": members, "tips": tips}
        self._frame_appendage(ap)
        self._app_cache = None
        self._appendages.append(ap)
        pm = self.physical_mass.reshape(self.nn, self.dofs.stride)
        for n in tips:
            pm[n] = 0.0                           # carried, not integrated
        self.mass = np.maximum(self.physical_mass, 1e-20)

    def _frame_appendage(self, ap):
        """(Re)take an appendage's frame from where things are now: its tips'
        offsets and orientations, and the CHORD of a massive member at the
        anchor that it turns with. Not the anchor's own rotation: a girder
        element lying on the sea floor rolls about its axis without end (the
        floor has no friction, a line has no section to stop on), and with
        none of the crossbeam's rotary inertia in it (rho = 1, the mass is on
        the girder's axis) -- a 23 m crossbeam carried by that roll swept
        round at 170 m/s while its girder lay still. A roll a line member
        cannot resist is not one to hang a crossbeam on."""
        a = ap["anchor"]; tips = ap["tips"]
        qa = self.dofs.view(self.q)
        x = lambda n: self.mesh.nodes[n] + qa[n, :3]
        ap["rel"] = np.array([x(n) - x(a) for n in tips]).reshape(-1, 3)
        ap["orient0"] = self.orient[tips].copy() if tips else np.zeros((0, 3, 3))
        el = self.mesh.elements[:, :2]
        cand = np.flatnonzero(((el[:, 0] == a) | (el[:, 1] == a)) & self.breaking.member_active
                              & ~self._device_members())
        ap["ref"] = int(cand[0]) if len(cand) else -1
        if ap["ref"] >= 0:
            b = el[ap["ref"], 1] if el[ap["ref"], 0] == a else el[ap["ref"], 0]
            d = x(b) - x(a)
            ap["d0"] = d / max(np.linalg.norm(d), 1e-30)
        else:
            ap["d0"] = np.array([1.0, 0.0, 0.0])

    def _carry_arrays(self):
        """The parked particles and the appendages as arrays, for
        `_carry_appendages` and the compiled step (1.6): (tips, anc, rel, o0,
        far, d0, parked_idx, parked_Q), or None when there is nothing to
        carry. Cached per change of the lists."""
        parked = getattr(self, "_parked", None)
        if parked:
            if getattr(self, "_park_cache", None) is None:
                self._park_cache = (np.ascontiguousarray(np.fromiter(parked.keys(), np.int64)),
                                    np.ascontiguousarray(np.array(list(parked.values()), float).reshape(-1, 6)))
            idx, Q = self._park_cache
        else:
            idx, Q = np.zeros(0, np.int64), np.zeros((0, 6))
        apps = getattr(self, "_appendages", None) or []
        if not apps and not len(idx):
            return None
        if getattr(self, "_app_cache", None) is None or self._app_cache[0] != len(apps):
            # flattened once per change of the appendage list: every tip with
            # its anchor, offset, orientation and reference chord
            tips, anc, rel, o0, far, d0 = [], [], [], [], [], []
            el = self.mesh.elements[:, :2]
            for ap in apps:
                a, e = ap["anchor"], ap["ref"]
                b = -1 if e < 0 else int(el[e, 1] if el[e, 0] == a else el[e, 0])
                for k, n in enumerate(ap["tips"]):
                    tips.append(n); anc.append(a); rel.append(ap["rel"][k])
                    o0.append(ap["orient0"][k]); far.append(b); d0.append(ap["d0"])
            self._app_cache = (len(apps), np.ascontiguousarray(np.asarray(tips, np.int64)),
                               np.ascontiguousarray(np.asarray(anc, np.int64)),
                               np.ascontiguousarray(np.asarray(rel, float).reshape(-1, 3)),
                               np.ascontiguousarray(np.asarray(o0, float).reshape(-1, 3, 3)),
                               np.ascontiguousarray(np.asarray(far, np.int64)),
                               np.ascontiguousarray(np.asarray(d0, float).reshape(-1, 3)))
        _, tips, anc, rel, o0, far, d0 = self._app_cache
        return tips, anc, rel, o0, far, d0, idx, Q

    def _carry_appendages(self):
        """Place every appendage's particles where its anchor carries them:
        x = x_anchor + R (x0 - x0_anchor) with R the least rotation that
        takes the reference chord from where it was when the appendage was
        framed to where it is (`_frame_appendage`); velocity the anchor's.
        Parked particles (`_park_bodiless`) are held where they stopped. An appendage whose
        anchor has lost its last massive member (and is no support) is
        released and removed -- cause 2."""
        arrays = self._carry_arrays()
        if arrays is None:
            return
        tips, anc, rel, o0, far, d0, idx, Q = arrays
        try:
            from ..kernels.numba_step import carry_appendages as _carry
        except ImportError:                                   # pragma: no cover
            _carry = None
        if _carry is not None:
            # the compiled version (1.6), which the compiled step runs too
            _carry(self.dofs.view(self.q), self.dofs.view(self.v), self.orient, self.mesh.nodes,
                   tips, anc, rel, o0, far, d0, idx, Q, np.empty((3, 3)))
            return
        qv = self.dofs.view(self.q); vv = self.dofs.view(self.v)
        if len(idx):
            qv[idx] = Q; vv[idx] = 0.0
        if not len(tips):
            return
        from .rotations import log_map
        xa = self.mesh.nodes[anc] + qv[anc, :3]
        # the least rotation taking the reference chord from d0 to d (no roll)
        d = np.where((far >= 0)[:, None], self.mesh.nodes[far] + qv[far, :3] - xa, d0)
        d /= np.maximum(np.linalg.norm(d, axis=1), 1e-30)[:, None]
        w = np.cross(d0, d); c = np.einsum("ni,ni->n", d0, d)
        W = np.zeros((len(d), 3, 3))
        W[:, 0, 1], W[:, 0, 2], W[:, 1, 2] = -w[:, 2], w[:, 1], -w[:, 0]
        W[:, 1, 0], W[:, 2, 0], W[:, 2, 1] = w[:, 2], -w[:, 1], w[:, 0]
        R = np.eye(3)[None] + W + np.einsum("nij,njk->nik", W, W) / np.maximum(1.0 + c, 1e-9)[:, None, None]
        qv[tips, :3] = xa + np.einsum("nij,nj->ni", R, rel) - self.mesh.nodes[tips]
        vv[tips] = vv[anc]
        self.orient[tips] = np.einsum("nij,njk->nik", R, o0)
        qv[tips, 3:] = log_map(self.orient[tips])

    def _release_appendages(self, t, device, held, bonds, grounded=None):
        """Appendages whose anchor no longer holds a massive member, or is on
        a piece of debris now."""
        keep, gone = [], []
        for ap in self._appendages:
            a = ap["anchor"]
            on = np.flatnonzero(((bonds[:, 0] == a) | (bonds[:, 1] == a)) & self.breaking.member_active)
            if (held[a] or (~device[on]).any()) and (grounded is None or grounded[a]):
                e = ap["ref"]
                if e < 0 or not self.breaking.member_active[e] or a not in bonds[e]:
                    self._frame_appendage(ap)        # its girder split off: turn with another
                keep.append(ap)
            else:
                gone.extend(ap["members"])
        self._appendages = keep
        self._app_cache = None
        return gone

    def _device_members(self):
        """Members with no mass of their own next to what their stiffness
        demands at the run's step (rho ~ 0: a crossbeam whose weight is in
        the girders, a rigid arm) -- modelling devices, which neither split
        off as bodies nor survive once they connect nothing."""
        dev = getattr(self, "_device", None)
        if dev is not None:
            return dev
        dev = np.zeros(self.nel, bool)
        if getattr(self, "_kg_ref", None) is not None and self.mode == "dynamic":
            need = (self.dt / float(self.p.solver.dt_safety)) ** 2
            stride = self.dofs.stride
            line_mass = getattr(self, "line_mass_per_end", None)
            for e in range(self.nel):
                # a member carrying a `with_mass` load has that body (1.1)
                share = self._element_mass_share(e)[0]
                if line_mass is not None and e < len(line_mass):
                    share += float(line_mass[e])
                rows = np.abs(self._kg_ref[e]).sum(axis=1).reshape(2, stride)[:, :3]
                dev[e] = rows.max() * need > 10.0 * max(share, 1e-30)
        self._device = dev
        return dev

    def _deactivate(self, ids):
        """A broken frame member is one whose local stiffness is zero: it
        carries nothing, its two nodes keep their mass, and both force paths
        (the batched NumPy law and the numba kernel, which copies k_local
        into `_fr_k`) see the same zero. The creep reference `_k_ref` is
        zeroed too, so an ageing rescale cannot bring the member back."""
        ids = np.asarray(ids, int)
        if not len(ids):
            return
        self.k_local[ids] = 0.0
        if getattr(self, "_fr_dofs", None) is not None:
            self._fr_k[ids] = 0.0
        if getattr(self, "_k_ref", None) is not None:
            self._k_ref[ids] = 0.0
        self._drop_from_reference_stiffness(ids)

    def record_k_block(self, e, dofs):
        """Member `e`'s reference block is now in K at `dofs` (None: K no
        longer holds it) -- for an edit of K made outside this class
        (analysis_steps._reference_K)."""
        self._k_dofs[e] = -1 if dofs is None else dofs

    def _drop_from_reference_stiffness(self, ids, add_back=False):
        """Stiffness-proportional Rayleigh damping is beta K v on the
        REFERENCE K, and a member that has gone must not go on damping the
        relative motion of its two nodes -- a falling fragment tied to the
        deck by a dashpot. Only when beta is in use: the row sums that size
        the relaxation masses were taken at setup and are not touched.
        `add_back` puts a split member's block onto its NEW particles
        (called before the connectivity is rewritten, from the bond graph)."""
        if not getattr(self, "beta", 0.0) or getattr(self, "_kg_ref", None) is None:
            return
        import scipy.sparse as sp
        K = sp.lil_matrix(self.K)
        for e in ids:
            dof = self.dofs.element_dofs(self.mesh.elements[e])
            K[np.ix_(dof, dof)] = K[np.ix_(dof, dof)].toarray() - self._kg_ref[e]
            self._k_dofs[e] = -1
            if add_back:
                dof = self.dofs.element_dofs(self.bonds.bonds[e])
                K[np.ix_(dof, dof)] = K[np.ix_(dof, dof)].toarray() + self._kg_ref[e]
                self._k_dofs[e] = dof
        self.K = K.tocsr()

    def _setup_mass_and_time(self):
        e, sol = self.p.case.element, self.p.solver
        self.mode = sol.mode
        A = float(e.area)
        m = np.zeros((self.nn, 6))
        true_rot = np.zeros((self.nn, 3))
        for k, (conn, L) in enumerate(zip(self.mesh.elements, self.length)):
            mt = self.material.rho * A * L / 2.0
            m[conn, :3] += mt
            # the step is taken from the OLD rule below, unrotated; modes and
            # motion get the true inertia
            if e.type == "beam2_timoshenko":
                m[conn, 3:] += self.material.rho * L * np.array([
                    float(e.j), float(e.iy), float(e.iz)]) / 2.0
            else:
                m[conn, 3:] += self.material.rho * A * L**3 / 24.0
            true_rot[conn] += self._element_mass_share(k)[3:]
        # nodal and hydrodynamic masses: directional, weightless, rotary
        add, rot, self.weightless_mass = added_mass.table(
            self.p, self._region, self.mesh.nodes[:self.nn], self.mesh.elements[:, :2])
        self.line_mass_per_end = added_mass.member_line_mass(
            self.p.get("loads"), self._region, self.mesh.nodes[:self.nn], self.mesh.elements[:, :2],
            None, self.p.case.units)
        m[:, :3] += add; m[:, 3:] += rot; self.additional_mass = add
        self._add_zone_mass(m)                      # the rigid zones' own (1.21)
        true_rot += rot
        self.physical_mass = m.reshape(-1)
        row = np.asarray(abs(self.K).sum(axis=1)).ravel()
        # selective mass scaling, before anything reads the mass
        self.mass_scaling = mass_scaling.scale(self.physical_mass, row, ~self._held_dofs(),
                                               sol.mass_scaling,
                                               sol.dt_safety)
        step_mass = np.maximum(self.physical_mass, 1e-20)
        self._rotary_true = true_rot
        self.physical_mass, self.rotary_floored = mass_scaling.true_rotary(
            self.physical_mass, true_rot, row, ~self._held_dofs())
        if self.mode == "relaxation":
            self.mass = np.maximum(row / float(sol.relaxation_safety)**2, 1e-20)
            self.dt, self.n_steps, self.total_time = 1.0, int(sol.n_steps), float(sol.n_steps)
            self._n_steps_solver = self.n_steps
        else:
            self.mass = step_mass           # the step's mass, until the step is set
            free = ~self._held_dofs()
            ratio = np.where(free, self.mass / np.maximum(row, 1e-20), np.inf)
            self.dt = float(sol.dt_safety * np.sqrt(np.min(ratio)))
            self.n_steps, self.total_time = max(1, int(round(sol.total_time / self.dt))), float(sol.total_time)
            omega_max = 1.0 / float(np.sqrt(np.min(ratio)))
        damp = sol.damping
        self.damp_kind = damp.type or ("kinetic" if self.mode == "relaxation" else "viscous")
        self.zeta, self.alpha, self.beta = float(damp.coeff), float(damp.alpha), float(damp.beta)
        # the step above is the UNDAMPED bound, and the damping it has to
        # survive was parsed on the line after it
        self._shrink_step_for_damping(omega_max if self.mode != "relaxation" else 0.0)
        if self.mode != "relaxation":
            self.mass = np.maximum(self.physical_mass, 1e-20)
        self.tolerance = float(sol.tolerance)
        self.kinetic_tolerance = float(sol.kinetic_tolerance)

    _relax_what = "beam staged relaxation"
    _relax_hint = "; reduce relaxation_safety or check for a structural mechanism"

    def _relax_kinetic(self):
        return _kinetic(self)

    def _relax_compiled(self, F, fixed, moving, prescribed, progress, start, tol, budget, ramp, it0,
                        progress_every, report, cv_s, stage, what, measures):
        """The compiled relaxation (1.9): the same numbers,
        to the bit, under backend numba. The level's loop itself is
        OrientedPoints._relax_stage."""
        if type(self).__name__ in ("BeamSolver", "MixedStructuralSolver"):
            from . import compiled_step
            rs = compiled_step.relax_stepper(self, fixed, moving, prescribed, start)
            if rs is not None:
                return rs.relax(F, fixed, moving, prescribed, progress, start, tol, budget, ramp, it0,
                                progress_every, report, cv_s, stage, what, measures)
        return None

    # `_members_in` and `_with_fragments` are OrientedPoints' (core/orientation):
    # the fibre beam's fragments need them too

    def _line_load_local(self, fl, L, e):
        """The fixed-end loads `fl` of member `e`, through its released ends
        (the frame solvers have releases, the fibre solver none)."""
        props = self.member_props[e] if hasattr(self, "member_props") else self.p["element"]
        return self._apply_load_releases(fl, L, props, self._material_of(e))

    def _thermal_load(self, idx, spec):
        """(nel, 12) of the free thermal deformation of the members in a
        region, per unit amplitude (core/temperature.py)."""
        from .temperature import local_deformation
        elements = self._members_in(idx)
        axial = np.zeros(0, int)
        if hasattr(self, "member_kind"):
            # A CABLE EXPANDS TOO. This filter used to keep
            # the beams and drop everything else without a word, so a
            # temperature on a region of girder and stays heated the girder
            # and ran, converged and reported success with the stays at their
            # erection temperature -- v0.4's scan found it and it stayed
            # open until the bridge line was closed out. An axial member's
            # free thermal deformation is its elongation at the centroid,
            # alpha dT Lu; a gradient bends nothing it could carry.
            kinds = np.asarray(self.member_kind, dtype=object)
            axial = np.array([e for e in elements if kinds[e] == "truss2"], int)
            elements = np.array([e for e in elements if kinds[e] in self.BEAM_TYPES], int)
        if not len(elements) and not len(axial):
            raise ValueError("a temperature load needs complete members; region "
                             "contains none")
        # ALPHA IS THE MEMBER'S, NOT THE MODEL'S. A steel deck
        # on concrete towers expands at 1.2e-5 and they at 1.0e-5, and before
        # per-group materials existed there was only one number to have. The
        # members are grouped by their alpha so a model with one of them calls
        # `local_deformation` exactly once, with exactly the argument it always
        # got -- the single-material answer is identical to the bit.
        alphas = np.array([float(getattr(self._material_of(e), "thermal_expansion", 0.0) or 0.0)
                           for e in elements])
        out = np.zeros((self.nel, 12))
        for a in np.unique(alphas):
            rows = elements[alphas == a]
            d = local_deformation(float(a), spec, self.length[rows])
            if d is None:
                raise ValueError(
                    "a temperature `profile` needs a section with fibres to integrate it "
                    "over -- a `general` beam section states area and second moments and "
                    "has no shape, so the mean and the slope of the profile are not "
                    "recoverable. Use `uniform`/`gradient` here, or beam2_fiber, which "
                    "carries the profile itself (and its self-equilibrated stresses).")
            out[rows] = d
        if len(axial):
            a_ax = np.array([float(getattr(self._material_of(e), "thermal_expansion", 0.0) or 0.0)
                             for e in axial])
            # from its UNSTRESSED length: a cable's free expansion is of the
            # cable it would be with no force in it
            Lu = np.asarray(getattr(self, "member_Lu", self.length))[axial]
            d = local_deformation(1.0, spec, Lu)
            if d is None:
                raise ValueError(
                    "a temperature `profile` on a region with axial members: a bar section "
                    "has no shape to integrate a profile over -- give those members "
                    "`uniform` (a gradient is accepted and bends nothing a bar can carry)")
            out[axial, 6] = a_ax * d[:, 6]
        return out

    # ----------------------------------------------------------------- #
    #  Time-dependent concrete: creep and shrinkage
    # ----------------------------------------------------------------- #
    def _setup_time_dependence(self):
        """Every member's OWN material's creep / shrinkage / ageing, read once.

        The model's first material used to stand for every member: a steel
        bar beside a creeping concrete one crept with it (10 -> 30 mm), or,
        with the steel assigned first, the concrete did not creep at all.
        Members are grouped by the material object they
        carry; one whose material declares none of the three -- and every
        connector -- takes no part. A model with none of them is untouched:
        `_time_deformation` returns None and nothing downstream runs.
        """
        from .creep import CreepHistory, bind_reference
        self._age0 = float(self.p.solver.age or 28.0)
        self._current_age = self._age0
        self._step_ages = (self._age0, None, "log")
        self._history = None
        #: [{"ids", "creep", "shrink", "ageing"}, ...], one per material
        self._tgroups = []
        #: tendon relaxation on prestressed bars: [(ids, spec, t0, eps_pre0)]
        self._relax_members = []
        #: the first material that creeps, for the dynamic-clock refusal
        self._creep_who = None
        kinds = getattr(self, "member_kind", None)
        owners = {}
        for e in range(self.nel):
            if kinds is not None and kinds[e] not in self.BEAM_TYPES and kinds[e] != "truss2":
                continue                        # a connector's material is not its law
            m = self._material_of(e)
            owners.setdefault(id(m), (m, []))[1].append(e)
        for m, ids in owners.values():
            ids = np.asarray(ids, dtype=np.int64)
            # the MATERIAL is what declares the laws, and a group's first
            # region is only one of its members: every region it is on
            # (the case's name for it is not carried this far)
            regions = sorted({str(self.member_props[e].get("region")) for e in ids
                              if self.member_props[e].get("region")}) if kinds is not None else []
            who = (f"the {m.name} material of member group(s) " + ", ".join(map(repr, regions))
                   if regions else f"material {m.name!r}")
            c, s_, ag = (getattr(m, k, None) for k in ("creep", "shrinkage", "ageing"))
            if getattr(m, "relaxation", None) is not None:
                self._read_member_relaxation(ids, m, who)
            if c is None and s_ is None and ag is None:
                continue
            if c is not None and self._creep_who is None:
                # a dynamic run may carry creep only on a clock that does not
                # move (`step_data` refuses a step that moves it): the
                # history restored from `initial_state` is then a frozen
                # imposed strain
                self._creep_who = who
                if self.p.solver.mode == "dynamic":
                    # a `to_age` is known now; an `age` that moves the clock
                    # only once a restored state says where it is
                    from .creep import refuse_moving_clock
                    for st in self.steps or []:
                        if st.get("to_age") is not None:
                            refuse_moving_clock(st, (float(st.get("age") or self._age0),
                                                     float(st["to_age"]), "log"), None, who)
            self._tgroups.append({"ids": ids,
                                  "creep": bind_reference(c, m.E, ag, who),
                                  "shrink": s_, "ageing": ag})
        creeping = [g for g in self._tgroups if g["creep"] is not None]
        if creeping:
            if len(creeping) == 1 and len(creeping[0]["ids"]) == self.nel:
                self._history = CreepHistory(creeping[0]["creep"])
            else:
                self._history = CreepHistory(creeping[0]["creep"],
                                             [(g["ids"], g["creep"]) for g in creeping])
        self._k_ref = (np.array(self.k_local, dtype=float)
                       if self._tgroups else None)
        self._k_factor = 1.0
        self._prev_station = None        # the age of the last recorded station
        #: this station's own creep coefficient, member by member
        self._phi_cur = np.zeros(self.nel)
        # the concrete age of every recorded frame, written beside the
        # monitors as `age`. A creep run's own axis is days, not the step
        # counter: without this nothing downstream can plot against it, and
        # a contract cannot say `at, against: age`.
        self.ages = []

    def _read_member_relaxation(self, ids, m, who):
        """`relaxation` on a prestressed bar: its prestrain falls by r(t),
        exactly as a fibre tendon's does (core/relaxation.py). It was read by
        nothing on this path. Anything that carries no
        prestrain -- a beam, an ordinary bar -- has nothing to relax, and the
        declaration is refused rather than left to do nothing.

        EACH BAR ON ITS OWN CLOCK: a bar prestressed from
        the start relaxes from `tensioned_at`; one born in a later step from
        the age it is born at; a bar JACKED in a step from the age its step
        locks it off, from the strain the jack left it at. Until then it has
        no prestress to lose. Entries are (ids, spec, t0, eps0, base) with
        t0 NaN while a bar's clock has not started, and the unstressed
        length at r is base / (1 + eps0 (1 - r))."""
        from . import relaxation
        spec = dict(m.relaxation)
        relaxation.validate(spec, f"{who} `relaxation`")
        kinds = getattr(self, "member_kind", None)
        Lu = getattr(self, "member_Lu", None)
        jt = getattr(self, "_jack_T", None)
        # the PRESTRESSED bars of the material's members: one steel may make
        # the stays and the girder, and only the stays relax
        # (v1.0 refused the pair and said none was prestressed)
        if kinds is not None and Lu is not None:
            bar = kinds[ids] == "truss2"
            jacked = bar & (np.asarray(jt)[ids] > 0.0) if jt is not None else np.zeros(len(ids), bool)
            ids = ids[(bar & (Lu[ids] != self.length[ids])) | jacked]
        if kinds is None or Lu is None or not len(ids):
            raise ValueError(
                f"{who} declares `relaxation`, but none of its members is a PRESTRESSED bar "
                "(a bar section with `prestress`, `unstressed_length` or `jack`), so there is "
                "nothing to relax and the declaration would do nothing. Relaxation is "
                "the loss of a tendon's initial strain; a beam's or an ordinary bar's "
                "stress comes from the structure and is not held")
        from .relaxation import stressed_at
        t_start = stressed_at(spec, self._age0, self.steps, who)
        jk = (np.asarray(jt)[ids] > 0.0) if jt is not None else np.zeros(len(ids), bool)
        t0 = np.where(jk, np.nan, t_start)
        eps0 = np.where(jk, 0.0, self.length[ids] / Lu[ids] - 1.0)
        base = np.asarray(self.length, float)[ids].copy()
        self._relax_members.append((ids, spec, t0, eps0, base))

    def _relax_start(self, ids, age, jacked=False, restart=False):
        """Start the relaxation clock of the bars `ids` at `age`: born then,
        or (`jacked`) locked off then at the strain the jack left -- the
        length it is held at is `member_Lu (1 + T / EA)`, `_jack_set`'s.
        `restart`: a bar born now starts again whatever its clock said (it
        was given the model's stressing age before it existed)."""
        for ids_g, _spec, t0, eps0, base in getattr(self, "_relax_members", ()):
            rows = np.flatnonzero(np.isin(ids_g, ids) & (np.isnan(t0) | restart))
            if not len(rows):
                continue
            e = ids_g[rows]
            t0[rows] = float(age)
            if jacked:
                EA = np.array([float(self.member_E[i]) * float(self.member_props[i]["area"]) for i in e])
                kf = getattr(self, "_k_factor", 1.0)
                if np.ndim(kf):
                    EA = EA * np.asarray(kf)[e]
                eps0[rows] = self._jack_T[e] / EA
                base[rows] = self.member_Lu[e] * (1.0 + eps0[rows])

    @property
    def time_dependent(self):
        return bool(getattr(self, "_tgroups", None)) or \
            bool(getattr(self, "_relax_members", None))

    def _age_at(self, progress):
        """The concrete age at this point of the current step -- `core.creep`
        owns the rule, because the fibre solver runs on the same one."""
        from .creep import age_at
        return age_at(self._step_ages, progress)

    def _station_phi(self, age):
        """The creep coefficient of the increment that arrives AT this
        station; `core.creep.station_phi` says why it is dated where it is."""
        from .creep import station_phi
        phi = np.zeros(self.nel)
        if self._history is None:
            return phi
        for g in self._tgroups:
            if g["creep"] is not None:
                # each segment on its own clock (1.2): age + shift, and its
                # first increment dated from its birth, not before it
                for rows, sh, floor in self._age_groups(g["ids"]):
                    prev = None if self._prev_station is None else max(self._prev_station + sh, floor)
                    phi[rows] = station_phi(g["creep"], age + sh, prev)
        return phi

    def _time_deformation(self, age):
        """The free local deformation creep and shrinkage have imposed by
        `age`: shrinkage from the clock alone, creep from the member's own
        elastic history.

        Only the increments ALREADY RECORDED appear here. The one this station
        is about to produce is carried by `_station_phi` and the reduced
        stiffness instead, because it is not known until the station is solved
        -- see `_record_time_station`.
        """
        d = None
        for g in self._tgroups:
            if g["shrink"] is None:
                continue
            from .creep import shrinkage_strain
            for rows, sh, floor in self._age_groups(g["ids"]):
                eps = shrinkage_strain(g["shrink"], max(age + sh, floor))
                if eps:
                    if d is None:
                        d = np.zeros((self.nel, 12))
                    d[rows, 6] = eps * np.asarray(self.length, dtype=float)[rows]
        if self._history is not None:
            shift = self.staging.age_shift
            if shift is not None and (np.any(shift) or np.isfinite(self.staging.age_floor).any()):
                from .creep import shifted_strain
                c = shifted_strain(self._history, age, self._age_groups)
            else:
                c = self._history.strain(age)
            if c is not None:
                d = c.copy() if d is None else d + c
            # the current increment creeps from where the last station left
            # off, so its own creep is `phi_cur * (e - e_prev)`: the `e` half
            # is the stiffness reduction below and the `e_prev` half is a
            # deformation imposed the other way.
            phi = self._phi_array()
            if phi.any() and self._history._last is not None:
                e_prev = phi[:, None] * self._history._last
                d = -e_prev if d is None else d - e_prev
        return d

    def _age_stiffness(self, age):
        """The local stiffness this station solves with.

        Two factors, both of them one multiplier on the whole local stiffness:

        AGEING -- Ec(t) / Ec(28), if the material declares it.

        THE CURRENT INCREMENT'S OWN CREEP -- `1 / (1 + phi_cur)`. With the
        imposed deformation of `_time_deformation`, a member then satisfies

            e (1 + phi_cur) = d - d_imposed_past + phi_cur e_prev

        which is the increment's own creep solved for TOGETHER with it, i.e.
        the same implicit step `core.creep.relaxation` takes. Leaving it out
        makes the scheme explicit, and on a held strain that is not merely
        first order but marginally UNSTABLE: case_29b's twelve stations then
        wobbled 0.332, 0.336, 0.329, 0.341 and landed on 0.317 of the initial
        force where the answer is 1/3, and twenty-four stations were worse,
        not better (0.360). Dating the increments earlier without this term is
        worse still -- that is pure over-relaxation, and it diverged.
        """
        if self._k_ref is None:
            return
        # one factor per member now: each carries its own material's laws,
        # and a truss bar reads it too (`_truss_force`) -- it did not, and a
        # bar under a held load crept and then came back
        f = 1.0 / (1.0 + self._phi_array())
        for g in self._tgroups:
            if g["ageing"] is not None:
                from .creep import modulus_factor
                spec = g["ageing"]
                s_ = float(spec.get("s", 0.25))
                of = str(spec.get("of", "strength"))
                ref = float(spec.get("reference_age", 28.0))
                for rows, sh, floor in self._age_groups(g["ids"]):
                    f[rows] *= modulus_factor(max(age + sh, floor), s_, of) / modulus_factor(ref, s_, of)
        if np.all(np.abs(f - self._k_factor) < 1e-14):
            return
        self.k_local = self._k_ref * f[:, None, None]
        self._k_factor = f
        # rebuild the batched references, which carry their own copy of
        # k_local (the frame batch here, the frame family in
        # MixedStructuralSolver). Clearing one field of a cache and not its
        # guard (`_beam_dofs`, until v0.1) left the map None, and the next
        # element-action pass reshaped `q[None]`: `rebatch` clears the whole.
        self.rebatch(axial=False)
        if hasattr(self, "_refresh_fictitious_mass"):
            self._refresh_fictitious_mass()

    def _local_elastic(self):
        """The MECHANICAL local deformation of every member right now: what
        the creep history has to be fed, because creep follows the elastic
        strain and not the total one."""
        conn = self.mesh.elements
        dofs = self._frame_batch()
        q = self.q[dofs].reshape(self.nel, 2, 6)
        x = self.mesh.nodes[conn] + q[:, :, :3]
        x = self.geometry.ends(np.arange(self.nel), x, self.orient)
        R, theta_loc, L = beam_frames_from_triads(x, self.orient[conn], self.R0)
        local = np.zeros((self.nel, 12))
        # a bar's stretch is from its unstressed length (a prestressed one's
        # is not its mesh length); equal to `length` for every other member
        local[:, 6] = L - getattr(self, "member_Lu", self.length)
        local[:, 3:6] = theta_loc[:, 0]
        local[:, 9:12] = theta_loc[:, 1]
        if getattr(self, "_thermal_d", None) is not None:
            local = local - self._thermal_d
        # the station solved with `k / (1 + phi_cur)`, so what it settled at is
        # `(1 + phi_cur)` times the elastic deformation the reference stiffness
        # would give for the same force. The history is fed the elastic one.
        phi = self._phi_array()
        if phi.any():
            local = local / (1.0 + phi)[:, None]
        built = getattr(self, "member_built", None) if self.plan is not None else None
        if built is not None:
            local[~built] = 0.0       # a member not built has no history (1.2)
        return local

    def _phi_array(self):
        """`_phi_cur` member by member (a checkpoint from before v1.0
        restores one number for the whole model)."""
        return np.broadcast_to(np.asarray(self._phi_cur, dtype=float), (self.nel,)).copy()

    def _relax_members_at(self, age):
        """A prestressed bar's unstressed length at `age`: its prestrain
        eps0 (1 - r) on the length it is held at, with r the relaxation
        since its own clock started (a bar whose clock has not started is
        left as it is)."""
        if not getattr(self, "_relax_members", None):
            return
        from .relaxation import relaxation_loss
        for ids, spec, t0, eps0, base in self._relax_members:
            live = np.isfinite(t0)
            if not live.any():
                continue
            for t in np.unique(t0[live]):
                rows = np.flatnonzero(t0 == t)
                r = relaxation_loss(spec, max(0.0, age - float(t)) * 24.0)
                self.member_Lu[ids[rows]] = base[rows] / (1.0 + eps0[rows] * (1.0 - r))

    def _record_time_station(self, age):
        """Called once a time station has reached equilibrium: the elastic
        deformation it settled at goes into the history, and everything after
        it creeps from there.

        The increment is dated to the geometric MIDDLE of the interval it
        arrived in, which is the same date `_station_phi` assumed while the
        station was being solved. The two go together and neither works
        alone: mid-dating without the station's own creep in the equation is
        pure over-relaxation and diverges, and the station's own creep with
        end-dating is the explicit scheme, which on a held strain wobbles and
        gets worse as the axis is refined. Together they are the classical
        step-by-step method, exact to 2e-5 against the Zener relaxation
        modulus on six stations.
        """
        if self._history is None:
            return
        from .creep import station_date
        self._history.add(station_date(age, self._prev_station),
                          self._local_elastic())
        if self.staging.age_shift is not None:
            # the interval itself, so that a member on its own clock files the
            # increment where `_station_phi` dated it (review, 1.2)
            if not hasattr(self._history, "windows") or len(self._history.windows) != len(self._history.ages) - 1:
                self._history.windows = [(None, a) for a in self._history.ages[:-1]]
            self._history.windows.append((self._prev_station, float(age)))
        self._prev_station = age

    def hold_time_strain(self) -> bool:
        """The creep reached held as an imposed deformation at the
        instantaneous stiffness, for a tangent asked for vibration (modes, a
        response spectrum): the last station's `1 + phi_cur`
        on the member stiffness is the implicit creep step of the interval
        just solved, not a dynamic stiffness. With the station recorded the
        equilibrium does not move. False when there is nothing to hold or
        the state lies between stations."""
        if not self.time_dependent:
            return False
        age = float(self._current_age)
        prev = self._prev_station
        if self._history is not None and prev is not None and age > prev * (1 + 1e-12) + 1e-12:
            return False
        self._phi_cur = np.zeros(self.nel)
        d = None
        for a, dd in getattr(self, "_thermal", ()):
            d = dd * a(1.0) if d is None else d + dd * a(1.0)
        extra = self._time_deformation(age)
        if extra is not None:
            d = extra if d is None else d + extra
        birth = self.staging.birth_d if self.plan is not None else None
        if birth is not None:
            d = birth if d is None else d + birth
        self._thermal_d = d
        return True

    def _force(self, groups, ground, progress, t=None):
        # The thermal deformation rides with the external force because it is
        # refreshed exactly as often: once per stage under staged relaxation,
        # once per step in a dynamic run.
        if self.plan is not None and getattr(self, "_birth_raw", None) is not None:
            # the members born this step: their offsets and their clocks, before
            # the clock's deformations below are read with them (1.2)
            from . import analysis_steps as _as
            _as.birth_offsets(self)
        self._jack_p = float(progress)                # how far the step's jacks have got (1.13)
        self._thermal_d = None
        for a, d in getattr(self, "_thermal", ()):
            factor = a(progress)
            self._thermal_d = d*factor if self._thermal_d is None else self._thermal_d + d*factor
        if self.time_dependent:
            # Creep and shrinkage ride in the same slot a temperature does:
            # they are the same kind of thing, a strain the member takes on
            # for reasons that are not force.
            age = self._age_at(progress)
            self._current_age = age
            self._relax_members_at(age)
            self._phi_cur = self._station_phi(age)
            self._age_stiffness(age)
            extra = self._time_deformation(age)
            if extra is not None:
                self._thermal_d = (extra if self._thermal_d is None
                                   else self._thermal_d + extra)
        if self._thermal_d is not None:
            # THE SIZE OF A LOAD THAT HAS NO FORCE. A temperature field is the
            # first load in this library whose converged state has neither an
            # external force nor an internal one: a beam free to expand ends up
            # longer and carries nothing. The relative residual then divides a
            # small number by a smaller one -- 3 N of imbalance against a scale
            # floor of 12.6 N read as 0.238, on a member whose thermal action
            # was 12.6 MN and which was converged to 2.4e-7 of it. So the
            # restraint force, ||k d_th|| -- what it would take to suppress the
            # field entirely -- joins the scale. It is the load's magnitude
            # whether or not any of it survives to the answer.
            self._thermal_scale = float(np.linalg.norm(
                np.einsum("eij,ej->ei", np.asarray(self.k_local), self._thermal_d)))
        birth = self.staging.birth_d if self.plan is not None else None
        if birth is not None:
            # a member born late is stress-free where it was born (1.2): its
            # local deformation then is its reference, in the slot a
            # temperature takes, and it is no load (not in the scale)
            self._thermal_d = birth if self._thermal_d is None else self._thermal_d + birth
        f = np.zeros(self.ndof)
        stacked = self._stacked_groups(groups)
        if stacked is not None:
            # MANY FIXED-SHAPE LOADS, EACH WITH ITS OWN TIME HISTORY -- a wind
            # field is one per deck node and component. Summed
            # one by one they cost more than the structure: two hundred groups
            # were 2 ms a step against the Xihoumen model's 0.84. As one sparse
            # matrix times the vector of their factors they cost 50 us.
            G, P, A, rest = stacked
            if P is not None:
                i = min(max(int(np.searchsorted(P, progress, side="right")) - 1, 0), len(P) - 2)
                w = min(max((progress - P[i]) / (P[i + 1] - P[i]), 0.0), 1.0)
                vals = A[:, i] * (1.0 - w) + A[:, i + 1] * w
            else:
                vals = np.array([a(progress) for a in A])
            f += G @ vals
            groups = rest
        for a, F in groups:
            # a moving load is callable: it puts its axles where they are at
            # this instant and returns the nodal forces for that position
            f += F(progress, t, np.zeros(self.ndof))*a(progress) if callable(F) else F*a(progress)
        # what acts inside the members now, for their end actions
        self._member_load_at = (groups, progress)
        if t is not None:
            self._ground_force(f, ground, t)
        self._carried_body_force(f, ground, progress, t)
        return f

    def _stacked_groups(self, groups, threshold=8):
        """The fixed-shape load groups as one sparse matrix, when there are
        more than `threshold` of them; cached per step's group list. The
        factors are interpolated all at once when every curve is sampled at
        the same progress points (a generated time history always is)."""
        cache = getattr(self, "_stack_cache", None)
        if cache is not None and cache[0] is groups:
            return cache[1]
        fixed = [(a, F) for a, F in groups if not callable(F)]
        out = None
        if len(fixed) > threshold:
            from scipy.sparse import csc_matrix
            cols = [np.asarray(F, float) for _a, F in fixed]
            G = csc_matrix(np.stack(cols, axis=1))
            amps = [a for a, _F in fixed]
            P0 = amps[0].p
            if all(len(a.p) == len(P0) and np.array_equal(a.p, P0) for a in amps) and len(P0) >= 2:
                P, A = P0, np.stack([a.a for a in amps])
            else:
                P, A = None, amps
            rest = [(a, F) for a, F in groups if callable(F)]
            out = (G, P, A, rest)
        self._stack_cache = (groups, out)
        return out

    def _reload_after_split(self, k, data):
        """A member split off at the end of the last step: its weight now
        sits on other particles, and gravity is a load vector built from the
        masses when the step began (`_reload_step`)."""
        if getattr(self, "_reload_step", False):
            data[:] = self.step_data(k)
            self.current_fixed = data[0]; self._reload_step = False

    def _between_steps(self, k, t, data):
        """The frame solvers' splits and appendages between two steps; the
        step data is reloaded at the start of the next (`_reload_after_split`)."""
        self.apply_pending_splits()
        self._carry_appendages()

    def _relax_after_bc(self):
        self.apply_pending_splits()
        self._carry_appendages()

    def compiled_pending(self):
        """A break found by a probe (the progress check's force evaluation)
        left work for the end of the next step (compiled_step.drive)."""
        return bool(getattr(self, "_pending_split", None) or getattr(self, "_park_check", None)
                    or getattr(self, "_reload_step", False))

    #: a force evaluation that reads the state (`BeamFamily._reading`) under
    #: the name the frame solvers' callers use: `analysis_steps._probe`, the
    #: buckling and modal tangents take a solver with a `_probe` as one that
    #: holds its own member state
    _probe = BeamFamily._reading

    def _removal_clock(self, k, steps, plan):
        """The time `_check_failure` reads at the forces of each of `steps`:
        the step count before the step is counted, times dt (Breaking.check
        reads t >= time) -- in a run of steps (1.2), the clock plus the
        steps of this one."""
        count = self.step_start + steps - 1
        if self.plan is not None:
            return self.staging.clock + (count - self.step_start) * self.dt
        return count * self.dt

    def _norm(self, x):
        """The convergence measures' 2-norm: the compiled one under
        `backend: numba` (`_norm_fn`)."""
        return _norm_fn(self)(x)

    def _thermal_force_scale(self):
        """`_thermal_scale` is current only while a thermal field is on."""
        return float(getattr(self, "_thermal_scale", 0.0)) if \
            getattr(self, "_thermal_d", None) is not None else 0.0

    def _record_scope(self):
        """The frame solvers read the whole frame in one probe: the hinges
        are read, not advanced."""
        return self._reading()

    def _monitor_fields(self, actions):
        """The frame's own element field (its `_nodal_fields` reads the
        members from the state)."""
        return self.nodal_fields(np.zeros((self.nel, 6)))

    def _nodal_fields(self, actions, recover_stress=False):
        out = {"displacement": self.u.copy(),
               "rotation": self.dofs.field(self.q, ROTATION).copy()}
        if hasattr(self, "last_reaction_force"):
            out["reaction_force"] = self.last_reaction_force
            out["reaction_moment"] = self.last_reaction_moment
        if actions is not None:
            ends = self.end_actions
            # k d less the loads inside the members: the fixed-end moments of
            # an axle or a line load between the nodes
            from .moving_load import member_loads
            ml = member_loads(self, *self._member_load_at) if hasattr(self, "_member_load_at") else None
            if ml is not None:
                ends = ends - ml
            # SIGNED internal forces, tension positive, the larger end with
            # its sign; `*_max_abs` is what the plain names held before
            # (core/member_actions). A failed member reports 0
            from . import member_actions
            signed, max_abs = member_actions.from_end_actions(ends)
            brk = getattr(self, "breaking", None)
            member_actions.write(out, signed, max_abs, ends,
                                 failed=None if brk is None else brk.member_failed)
        if getattr(self, "breaking", None) is not None:
            self.breaking.fields(out)
        if self.plan is not None:
            # which members stand in this frame (1.2): 1 built, 0 not yet or removed
            out["member_built"] = self.member_built.astype(float)
        if getattr(self, "bonds", None) is not None:
            # once members can move onto other particles, the frame has to
            # carry the connectivity the mesh group cannot
            out["connectivity"] = self.mesh.elements[:, :2].astype(float)
        return out

    def _checkpoint_split(self, out):
        """The frame solvers' splits: which particle every fragment's came
        from, and the queue the next step's end splits (the list of splits
        itself is `resume.split_checkpoint`'s)."""
        if self.bonds is not None:
            fp = getattr(self, "_fragment_parent", {})
            out["fragment_parent"] = np.array([[n, p] for n, p in fp.items()], dtype=np.int64).reshape(-1, 2)
        # A BREAK FOUND IN THE FRAME'S OWN FORCE EVALUATION (`_record`) is
        # queued and split at the end of the next step; a checkpoint taken in
        # between lost it, and the restart never split the member.
        # The queue and the ends it parts at go with it.
        pending = sorted(set(getattr(self, "_pending_split", None) or []))
        if pending and self.breaking is not None:
            due = self.breaking._ends_due
            out["pending_split"] = np.array(
                [[e] + [int(x) for x in due.get(e, np.zeros(2, bool))] for e in pending], dtype=np.int64)
        park = sorted(getattr(self, "_park_check", None) or [])
        if park:
            out["park_check"] = np.array(park, dtype=np.int64)
        return out

    def _restore_split(self, ck):
        """The bonds and masses the splits moved."""
        new = np.asarray(ck["elements"], dtype=np.int64)
        # beta K v is on the REFERENCE K, whose blocks the run moved with
        # every split: a restart kept the setup's and damped the fragment
        # against the frame. Moved here the same way.
        moved = np.flatnonzero((new != self.mesh.elements[:, :2]).any(axis=1))
        self.bonds.bonds[...] = new
        self._drop_from_reference_stiffness(moved, add_back=True)
        self.mesh.elements[:, :2] = new
        self.physical_mass = np.asarray(ck["physical_mass"], float).copy()
        self.mass = np.maximum(self.physical_mass, 1e-20)
        if "fragment_parent" in ck:
            self._fragment_parent = {int(n): int(p) for n, p in np.asarray(ck["fragment_parent"]).reshape(-1, 2)}
        self.rebatch()
        self._reload_step = True

    def _restore_queue(self, ck):
        if self.breaking is not None and "pending_split" in ck:      # v1.0
            rows = np.asarray(ck["pending_split"], dtype=np.int64).reshape(-1, 3)
            self._pending_split = [int(r[0]) for r in rows]
            for r in rows:
                self.breaking._ends_due[int(r[0])] = r[1:].astype(bool)
        if "park_check" in ck:
            self._park_check = {int(n) for n in np.asarray(ck["park_check"]).ravel()}
    def summary(self):
        kind = self.p["element"].get("type", "beam2")
        theory = "Timoshenko" if kind == "beam2_timoshenko" else "Euler-Bernoulli"
        shear = (f"\n  shear         Ay={self.p.case.element.ay:.6e}, Az={self.p.case.element.az:.6e}, "
                 f"kappa_y={self.p.case.element.kappa_y:.6g}, "
                 f"kappa_z={self.p.case.element.kappa_z:.6g}"
                 if kind == "beam2_timoshenko" else "")
        return (f"  mesh          {self.nn} nodes, {self.nel} Line2 elements\n"
                f"  element       {kind} / {theory}, 6 DOF/node\n"
                f"  section       A={self.p.case.element.area:.6e}, Iy={self.p.case.element.iy:.6e}, "
                f"Iz={self.p.case.element.iz:.6e}, J={self.p.case.element.j:.6e} m^4{shear}\n"
                f"  material      {self.material.name}; backend {getattr(self, 'backend', 'numpy')}\n"
                f"  mode          {self._mode_text()}"
                + mass_scaling.summary_line(getattr(self, "mass_scaling", None))
                + member_axes.summary_lines(self))        # v1.0
