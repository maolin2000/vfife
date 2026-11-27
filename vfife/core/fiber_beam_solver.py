"""Co-rotational fiber beam solver.

Single beam type ``beam2_fiber`` (Euler-Bernoulli, 6 DOF/node) with a
rectangular fiber section.  Each fiber carries a uniaxial material
(piecewise_uniaxial / rebar / steel02 / steel_cyclic / concrete01 / linear_elastic
mapped to axial).  Section
forces are integrated at Gauss points along the member and assembled by the
Euler--Bernoulli Hermite virtual-work matrix B.T @ [N, My, Mz].
"""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np

#: the compiled connectors' row buffer when they add into the force directly
_NO_ROWS = np.zeros((1, 12))
_NO_BIRTH = np.zeros((0, 6))          # no connector here is born late (kernels/numba_connector.respond)

from .problem import Problem
from . import mass_scaling
from . import member_axes

#: kernels/numba_fiber_beam.TIE, here so the NumPy path needs no numba: a
#: later integration station replaces the reported one only beyond round-off
_TIE = 1.0 + 1e-9
from . import added_mass
from . import member_geometry
from . import creep
from . import relaxation
from ..elements import connector
from ..base.dof import (DofLayout, ROTATION, SparseAssembler, TRANSLATION,
                  repeated_rotation)
from .beam_family import BeamFamily
from .capabilities import refuse

# one monitor record for every solver (core/monitors)
from .monitors import Monitor  # noqa: F401  (the name this module exported)

# the sections, the fracture and the time-dependent strains are mixins of
# their own modules (1.22); the names other modules import from here stay
from .fiber_sections import (FiberSections, INTEGRATION_RULES, _GAUSS_S, _GAUSS_W,  # noqa: F401
                             _SectionLayout, _piecewise_parabola)
from .fiber_fracture import FiberFracture
from .fiber_time import FiberTimeDependence


class FiberBeamSolver(FiberSections, FiberFracture, FiberTimeDependence, BeamFamily):
    BEAM_TYPES = frozenset(("beam2_fiber",))
    #: the bond graph of a model whose members split (`_setup_fracture`)
    bonds = None
    #: a removal in the step that ended set something loose, to be looked
    #: at between two steps (`apply_pending_splits`)
    _pending_debris = False
    #: the history of the connectors' `material` laws (`connector.MaterialStates`, 1.21)
    connector_materials = None
    ELEMENT_NAME = "beam2_fiber"
    HOLD_PARKED = True
    SOLVER_WHAT = "fiber beam"
    TRIAL_STATE = True
    DYNAMIC_WHAT = "fiber beam dynamic integration"
    MONITOR_KINDS = ("reaction", "displacement", "fiber", "vehicle", "element")

    def __init__(self, problem, mesh, material, log=print):
        problem = Problem.of(problem)        # typed, and its dict
        # material is the *fiber* material (uniaxial or linear_elastic)
        self.backend = problem.solver.backend
        if self.backend not in ("numpy", "numba"):
            raise ValueError(
                f"beam2_fiber supports solver.backend = numpy (the readable path, promoted "
                f"to the fused numba kernel when the model qualifies) or numba (the kernel "
                f"is required), not {self.backend!r}")
        self.p, self.mesh, self.material, self.log = problem, mesh, material, log
        self.nn, self.nel = mesh.n_nodes, mesh.n_elem
        self.dofs = DofLayout.frame(self.nn)
        self.ndof = self.dofs.size
        if mesh.elements.shape[1] != 2:
            raise ValueError("beam2_fiber requires Gmsh Line2 elements")
        self._setup_regions()
        self._reserve_split_particles()
        self._setup_section()
        self._setup_time_dependence()
        self._assemble_reference()
        # analysis steps (1.2; core/analysis_steps)
        from . import analysis_steps
        self.plan = analysis_steps.plan(self)
        # soil springs on piles
        from . import soil
        soil.attach(self)
        soil.add_stiffness(self)
        self._setup_mass_and_time()
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
        self.time = 0.0
        self.rebar = []
        self._pending_splits = getattr(self, "_pending_splits", [])
        if self.p.case.rigid_links or any(st.rigid_links for st in self.p.case.steps):
            # the links are made where the nodes stand now, before the steps
            # are installed (core/constraints), as the frame solver's are
            self.constraint_layer()
        if self.plan is not None:
            analysis_steps.install(self)

    def _setup_connectors(self, spec):
        """`element.fiber_connectors` -- the members of this mesh that are NOT
        fibre beams.

        A bearing is two nodes and six stiffnesses: no section to integrate, no
        material to drive, no state to keep. That is why it is the one element
        type that can share a mesh with fibre sections, and why the whole of
        this feature is bookkeeping rather than mechanics -- `elements.connector`
        answers with the fibre solver's own `q`, `orient`, `v` and `R0`
        unchanged, and was verified to before a line of this was written.

        `_is_fiber` is the mask everything downstream reads. It is kept SEPARATE
        from `_active`, which the fracture latch owns and writes: a connector is
        not a broken member, and sharing one flag would make a fracture silently
        revive it.
        """
        self._connector_props = []
        self._connector_of = np.full(self.nel, -1, dtype=np.int64)
        self._is_fiber = np.ones(self.nel, dtype=bool)
        for k, m in enumerate(spec.get("fiber_connectors") or []):
            block = next(iter(getattr(self.mesh, "cell_blocks", None) or {"line2": None}))
            ids = self.mesh.region_elements(m["region"], block, self.mesh.elements)
            if ids is None:
                ids = self.mesh.region_elements(m["region"], "beam2_fiber", self.mesh.elements)
            if ids is None or len(ids) == 0:
                raise ValueError(
                    f"connector region {m['region']!r} covers no complete Line2 element")
            taken = ids[self._connector_of[ids] >= 0]
            if len(taken):
                raise ValueError(
                    f"connector regions overlap at element(s) {taken[:8].tolist()}")
            self._connector_of[ids] = k
            self._is_fiber[ids] = False
            self._connector_props.append(dict(m))
        self._connector_ids = np.flatnonzero(~self._is_fiber)
        region = spec.get("fiber_region")
        if region:
            # A SINGLE SECTION NAMES ITS REGION, AND THE REGION MUST BE THE
            # MODEL. The section is the whole fibre model; a region that owns
            # only some members (or none) was ignored and the section put on
            # every member (found in review)
            block = next(iter(getattr(self.mesh, "cell_blocks", None) or {"line2": None}))
            self._region(region)                          # an unknown name, by name
            ids = self.mesh.region_elements(region, block, self.mesh.elements)
            if ids is None:
                ids = self.mesh.region_elements(region, "beam2_fiber", self.mesh.elements)
            owned = np.zeros(self.nel, dtype=bool)
            if ids is not None and len(ids):
                owned[ids] = True
            missing = np.flatnonzero(self._is_fiber & ~owned)
            if len(missing):
                raise ValueError(
                    f"the fibre section is assigned to region {region!r}, which owns "
                    f"{int((self._is_fiber & owned).sum())} of the {int(self._is_fiber.sum())} fibre "
                    f"members (not e.g. element(s) {missing[:6].tolist()}); a single fibre section is "
                    "the whole model -- assign it to `all` or to a region covering every member, or "
                    "give the other members their own sections")
        self.connector_deformation = np.zeros((self.nel, 6))
        # connector laws' slipped deformation, committed / working,
        # under the same trial discipline as the fibre states
        self._conn_committed = np.zeros((self.nel, 6))
        self._conn_slip = np.zeros((self.nel, 6))
        #: the history of the connectors' `material` laws (1.21)
        self.connector_materials = connector.MaterialStates(
            self._connector_ids, lambda e: self._connector_props[int(self._connector_of[e])])

    @property
    def has_connectors(self):
        return bool(getattr(self, "_connector_props", ()))

    def member_reference_block(self, e):
        """(`BeamFamily.member_reference_block`) the fibre solver's
        `_reference_k`, the block it re-assembles `K` from."""
        return self._reference_k(int(e))

    def _reference_k(self, e, T=None):
        """The 12x12 elastic reference stiffness of element `e`, in global axes
        (in the frame `T`, if given: the relaxation mass turns it).

        Reference only: it sets the initial K, the relaxation mass and the
        stable step estimate, and it never carries load. Factored out of
        `_assemble_reference` so that the same matrix can be re-assembled over
        the SURVIVING members after a fracture (`_rebuild_reference_stiffness`).
        """
        if not self._is_fiber[e]:
            # `connector.reference_stiffness` returns it in GLOBAL axes already
            return connector.reference_stiffness(
                self.mesh.nodes[self.mesh.elements[e]], self.R0[e],
                self._connector_props[int(self._connector_of[e])])
        # THIS ELEMENT'S SECTION'S LAW, not the first section's.
        # It never carries load, but it does set the initial K, the relaxation
        # mass and the step estimate, and a steel member sized on concrete's
        # modulus gets a mass six times too small.
        E = float(self.member_E[e]); G = float(self.member_G[e])
        L = float(self.length[e])
        k = np.zeros((12, 12))
        def pair(a, b, val):
            k[a, a] += val; k[b, b] += val; k[a, b] -= val; k[b, a] -= val
        A, Iy, Iz = float(self.total_area[e]), float(self.Iy[e]), float(self.Iz[e])
        J = float(self.J[e])          # the section's, or Iy + Iz
        pair(0, 6, E * A / L)
        pair(3, 9, G * J / L)
        c = E * Iz / (L**3)
        ids = [1, 5, 7, 11]
        block = c * np.array([[12, 6*L, -12, 6*L],
                              [6*L, 4*L*L, -6*L, 2*L*L],
                              [-12, -6*L, 12, -6*L],
                              [6*L, 2*L*L, -6*L, 4*L*L]])
        k[np.ix_(ids, ids)] += block
        c = E * Iy / (L**3)
        ids = [2, 4, 8, 10]
        block = c * np.array([[12, -6*L, -12, -6*L],
                              [-6*L, 4*L*L, 6*L, 2*L*L],
                              [-12, 6*L, 12, 6*L],
                              [-6*L, 2*L*L, 6*L, 4*L*L]])
        k[np.ix_(ids, ids)] += block
        if T is None:
            # the reference block K holds: with the rigid stretch of any end
            # offsets in it (a turned one is turned by `_turned_blocks`, at
            # the offsets as they are then)
            return self.geometry.stiffness_to_nodes(e, self.T[e].T @ k @ self.T[e])
        return T.T @ k @ T

    def _connector_force(self, e, conn, dof):
        """One connector's generalized force, actions and deformation.

        `elements.connector` takes exactly what this solver already holds --
        the two nodes' translations and triads, their velocities, the
        reference coordinates and `R0` -- so nothing is converted and nothing
        is copied. This is the whole of the mechanics.
        """
        qe = self.q[dof].reshape(2, 6)
        return connector.response(
            qe[:, :3], self.orient[conn], self.v[dof], self.mesh.nodes[conn],
            self.R0[e], self._connector_props[int(self._connector_of[e])],
            include_damping=self.mode == "dynamic",
            slip_committed=self._conn_committed[e], slip_out=self._conn_slip[e],
            offset=None if self._conn_birth is None else self._conn_birth[e],
            materials=self.connector_materials.of(e) if self.connector_materials else None)

    def connector_props(self, e):
        """Member `e`'s connector properties, or None when it is a fibre
        member."""
        return None if self._is_fiber[e] else self._connector_props[int(self._connector_of[e])]

    def _connector_kernel(self):
        """The compiled connectors' arrays (kernels/numba_connector), built
        once, or None: no connectors, no numba, or not six dofs a node. Used
        wherever the fibre kernel is."""
        ck = getattr(self, "_conn_kernel", False)
        if ck is not False:
            return ck
        ck = None
        if len(getattr(self, "_connector_ids", ())) and self.dofs.stride == 6 \
                and not any(connector.has_material_law(p) for p in self._connector_props):
            try:
                from ..kernels import numba_connector as nc
                if nc.available():
                    ck = {"enc": nc.encode(self._connector_props), "spins": nc.spins()}
            except ImportError:                                   # pragma: no cover
                ck = None
        self._conn_kernel = ck
        return ck

    def _commit_members(self):
        """The connector laws' history the next evaluation starts from."""
        self._conn_committed[...] = self._conn_slip
        if self.connector_materials:
            self.connector_materials.commit()
        if getattr(self, "_sl_on", None) is not None and self._sl_on.any():
            # the shear laws' history
            self._sl_gcr_c[...] = self._sl_gcr_t
            self._sl_gmax_c[...] = self._sl_gmax_t

    def _refresh_kernel_mask(self):
        """`_active & _is_fiber`, which is what the fused kernel is given.

        Called wherever `_active` changes: at setup and after a fracture. Two
        masks rather than one because they mean different things and one of
        them is written by the solver.
        """
        self._kernel_active = (np.asarray(self._active, dtype=np.uint8)
                               * np.asarray(self._is_fiber, dtype=np.uint8))

    def _rebuild_reference_stiffness(self):
        """Re-assemble K over the members that still carry load.

        WHY IT HAS TO BE REBUILT AFTER A SPLIT, and this is the hazard the
        graphics survey flagged (X-FEM
        literature): the explicit step is bounded by min(m/k) over the free
        degrees of freedom, and a split takes mass OFF a particle. Leave the
        dead member's stiffness in K and that particle's ratio falls by the
        share that just left -- the step bound collapses for a member that is
        not there any more. Dropping the same member from both sides is not a
        correction, it is the accounting: a fractured element contributes
        neither mass to the fragment's old particle nor stiffness to it.
        """
        assembly = SparseAssembler(self.ndof)
        for e in range(self.nel):
            if not self._active[e]:
                continue
            assembly.add_matrix(self.dofs.element_dofs(self.mesh.elements[e]),
                                self._reference_k(e))
        self.K = assembly.csr()

    def _turned_k(self, e, T):
        """(member `e`'s reference block turned into the frame `T`, the block
        as `K` holds it) for the relaxation mass (`BeamFamily._refresh_fictitious_mass`);
        None for a connector, which keeps its reference rows as the mixed
        solver's do."""
        if not self._is_fiber[e]:
            return None
        return self._reference_k(e, T), self._reference_k(e)

    def _assemble_reference(self):
        self.T, self.R0, self.length = [], [], []
        orient = self.p.case.element.orientation
        el = self._elem_spec
        many = el.get("fiber_sections")
        if many:
            groups = [(self._rows[k], m.get("end_offsets"), m.get("orientation") or orient, m["region"])
                      for k, m in enumerate(many)]
        else:
            groups = [(self._rows[0], el.get("end_offsets"), orient, el.get("fiber_region") or "all")]
        self.geometry = geometry = member_geometry.MemberGeometry.build(self.mesh, self.nel, groups)
        if geometry.full is not None:
            raise ValueError("end_offsets.axial: full is for frame beams (beam2, beam2_timoshenko, the beams "
                             "of mixed_line): a fibre section's axial response is its fibres' over the "
                             "elastic part")
        for e, conn in enumerate(self.mesh.elements):
            # a section's own orientation wins over the model's: a
            # column and a beam in one frame do not share a local y
            o = orient
            if self._section_orient is not None and self._is_fiber[e]:
                own = self._section_orient[int(self._section_of[e])]
                if own is not None:
                    o = own
            if not self._is_fiber[e]:
                # the connector's own reference, or -- as `mixed_line` does --
                # the global axis least aligned with its chord
                o = self._connector_props[int(self._connector_of[e])].get("orientation")
                if o is None:
                    axis = self.mesh.nodes[conn[1]] - self.mesh.nodes[conn[0]]
                    basis = np.eye(3)
                    o = basis[np.argmin(np.abs(basis @ (axis / np.linalg.norm(axis))))]
            R, L = self._frame(*geometry.reference_ends(e), o)
            self.T.append(repeated_rotation(R)); self.R0.append(R); self.length.append(L)
        self.T, self.R0, self.length = (np.asarray(self.T), np.asarray(self.R0),
                                         np.asarray(self.length))
        self._active = np.ones(self.nel, dtype=np.uint8)
        self._refresh_kernel_mask()
        self._setup_shear_interpolation()
        self._rebuild_reference_stiffness()
        self._setup_strain_penetration(self._elem_spec)
        self._setup_kernel()
        if self.backend == "numba" and self._kernel is None:
            # asked for the kernel by name and it cannot be built: say why
            # in the kernel's own words rather than quietly running the
            # readable path 100x slower
            raise ValueError(
                f"solver.backend = numba but the beam2_fiber fused kernel is not "
                f"available for this model: {self.kernel_status}. Use backend numpy "
                f"to accept the NumPy reference path.")
        self._setup_fracture(self._elem_spec)
        self._setup_breaking()

    def _setup_kernel(self):
        """Build the fused kernel when this section can be expressed as two laws.

        Falls back silently to the NumPy path -- which stays the reference --
        whenever it cannot: no numba, a material without a scalar twin, or
        rebars that do not all share one law. `kernel_status` records which,
        so a run that is unexpectedly slow can be asked why rather than
        guessed at.

        A SECTION WITH NO REBARS IS THE EASY CASE, NOT AN EXCLUDED ONE. This
        used to open with `if not self.is_hybrid: return`, so every steel
        model in the library -- a tube, a box, anything homogeneous -- ran on
        the NumPy reference while every reinforced-concrete one got the
        kernel. That is backwards: the kernel inlines at most two law types,
        and a homogeneous section needs one and no rebar arrays at all. The
        gate the kernel actually has is about the LAWS, so that is what is
        asked here; `n_steel = 0` appears in the compiled body only as
        `for f in range(NS)`, which then does not execute.
        """
        self._kernel = None
        self.kernel_status = "not attempted"
        try:
            from ..kernels.numba_fiber_beam import make_kernel
        except Exception as exc:
            self.kernel_status = f"kernel module unavailable ({exc})"
            return

        # every concrete group of every section, and every rebar group: the
        # kernel inlines ONE law type per family, whatever the section
        concretes = [self.materials[g] for lay in self._layouts for g in lay.conc_groups]
        rebars = [self.materials[g] for lay in self._layouts for g in lay.rebar_groups]
        for m in concretes[1:]:
            if type(m) is not type(concretes[0]):
                self.kernel_status = ("cover and core are different law types; NumPy path"
                                      if len(self._layouts) == 1 else
                                      "the sections' concrete laws are of different types; NumPy path")
                return
        # Two inlined point functions means one law for every rebar. Same type
        # AND same parameters -- two steel02 of different grade are two laws.
        for m in rebars[1:]:
            if type(m) is not type(rebars[0]):
                self.kernel_status = "rebars use more than one material type; NumPy path"
                return
        try:
            cpoint = concretes[0].numba_uniaxial_fn()
            conc_params = {g: self.materials[g].numba_params()
                           for lay in self._layouts for g in lay.conc_groups}
            rebar_params = [m.numba_params() for m in rebars]
        except AttributeError:
            # NAME IT. `backend: numba` turns this string into the refusal the
            # user reads, and "a material" sent them looking through a section
            # with four laws in it.
            missing = [m.name for m in [concretes[0]] + list(rebars)
                       if not hasattr(m, "numba_uniaxial_fn")
                       or not hasattr(m, "numba_params")] or ["a material"]
            self.kernel_status = (f"{', '.join(dict.fromkeys(missing))} has no scalar twin "
                                  "(no numba_uniaxial_fn); NumPy path")
            return
        if len({len(p) for p in conc_params.values()}) != 1:
            self.kernel_status = "concrete laws disagree on parameter count; NumPy path"
            return
        if rebars:
            spoint = rebars[0].numba_uniaxial_fn()
            if len({len(p) for p in rebar_params}) != 1:
                self.kernel_status = "rebar laws disagree on parameter count; NumPy path"
                return
        else:
            # The kernel still has to be built around SOME second device
            # function, because the pair is baked in at compile time. Passing
            # the section's own law costs nothing: with n_steel = 0 the rebar
            # loop never runs, and using the same function means the two
            # kernels a mixed model and a homogeneous model compile share a
            # cache key whenever the law matches.
            spoint = cpoint
        if cpoint is None or spoint is None:
            self.kernel_status = "numba unavailable; NumPy path"
            return

        # Consolidated order PER SECTION: its concrete fibres grouped (cover,
        # then core), then its rebars, each padded to the widest section. The
        # kernel indexes every array [e, j], so a member only ever meets its
        # own section's row.
        n_c = max(lay.n_c for lay in self._layouts)
        n_s = self.n_steel
        n_gp = len(self.gauss_s)
        n_cp = len(next(iter(conc_params.values())))
        self._cy = np.zeros((self.nel, n_c)); self._cz = np.zeros((self.nel, n_c))
        self._ca = np.zeros((self.nel, n_c)); self._cpar = np.zeros((self.nel, n_c, n_cp))
        self._sy = np.zeros((self.nel, n_s)); self._sz = np.zeros((self.nel, n_s))
        self._sa = np.zeros((self.nel, n_s)); self._spre = np.zeros((self.nel, n_s))
        self._cpre = np.zeros((self.nel, n_c))
        self._c_order = []                    # per section: padded columns in consolidated order
        self._c_group = []                    # ... and the fibre group each of them belongs to
        for k, lay in enumerate(self._layouts):
            rows = self._rows[k]
            c_idx = np.concatenate([self.group_indices[g] for g in lay.conc_groups])
            c_group = np.concatenate([np.full(len(self.group_indices[g]), g) for g in lay.conc_groups])
            s_idx = (np.concatenate([self.group_indices[g] for g in lay.rebar_groups])
                     if lay.rebar_groups else np.zeros(0, dtype=np.int64))
            self._c_order.append(c_idx)
            #: the fibre GROUP of every consolidated column, so a per-group
            #: parameter (creep's strain scale) can be written back into
            #: `_cpar` after the pack
            self._c_group.append(c_group)
            m = len(c_idx)
            self._cy[np.ix_(rows, range(m))] = self.fiber_y[rows][:, c_idx]
            self._cz[np.ix_(rows, range(m))] = self.fiber_z[rows][:, c_idx]
            self._ca[np.ix_(rows, range(m))] = self.fiber_area[rows][:, c_idx]
            par = np.stack([conc_params[g] for g in c_group])
            # padded fibres carry the cover's parameters at zero area
            par = np.concatenate([par, np.repeat(par[:1], n_c - m, axis=0)]) if n_c > m else par
            self._cpar[rows] = par
            # A CONCRETE fibre can carry an imposed strain too. Until v0.4
            # the kernel had no array for it while the NumPy reference added
            # `fiber_prestrain` to EVERY fibre -- the two agreed only because
            # nothing ever set one on concrete. A temperature field does.
            self._cpre[np.ix_(rows, range(m))] = self._imposed[rows][:, c_idx]
            if lay.rebar_groups:
                ms = len(s_idx)
                self._sy[np.ix_(rows, range(ms))] = self.fiber_y[rows][:, s_idx]
                self._sz[np.ix_(rows, range(ms))] = self.fiber_z[rows][:, s_idx]
                self._sa[np.ix_(rows, range(ms))] = self.fiber_area[rows][:, s_idx]
                self._spre[np.ix_(rows, range(ms))] = self.fiber_prestrain[rows][:, s_idx]
        for name in ("_cy", "_cz", "_ca", "_cpar", "_sy", "_sz", "_sa", "_spre", "_cpre"):
            setattr(self, name, np.ascontiguousarray(getattr(self, name)))

        # The consolidated rebar state becomes the storage; the per-rebar
        # entries of fiber_states become views onto it, so checkpoint(),
        # restore() and output_fields() keep working on live data and the
        # kernel does not have to pack and unpack on every call.
        if rebars and self._spar is None:
            self.kernel_status = "rebar parameters unavailable; NumPy path"
            return
        if not rebars:
            # Shaped so the kernel's typing is the same in both cases; with
            # n_steel = 0 nothing indexes either of them.
            self._spar = np.zeros((self.nel, 0, n_cp))
        ns = (self.fiber_states[rebars_g0].shape[3]
              if (rebars_g0 := next((g for lay in self._layouts for g in lay.rebar_groups), None)) is not None
              else 0)
        self._state_s = np.zeros((self.nel, n_gp, n_s, ns))
        nc_state = self.fiber_states[self._layouts[0].conc_groups[0]].shape[3]
        self._state_c = np.zeros((self.nel, n_gp, n_c, nc_state))
        # every fibre, padded ones included, starts from the zero state the
        # laws all accept (a group's `fiber_states` start as zeros too)
        self._relink_kernel_states()

        self._edofs = np.array([self.dofs.element_dofs(c)
                                for c in self.mesh.elements], dtype=np.int64)
        self._x0 = np.ascontiguousarray(self.mesh.nodes[self.mesh.elements])
        self._gw = np.ascontiguousarray(self.gauss_w, dtype=np.float64)

        # `self._G0J0` is built with `self.J`, per element and per section;
        # nothing to recompute here.

        #: every fibre's stress at every station, consolidated column order
        #: (the concrete of `_c_order`, then the steel rows); the kernel
        #: writes it on every call and `fiber_stress_rows()` reads it back
        #: in the section's own order
        self._fstress = np.zeros((self.nel, n_gp, n_c + n_s))
        # elements across cores; 0 is every core numba sees
        from ..kernels.numba_frame import configure_threads
        self._kernel_threads = configure_threads(self.p.solver.threads)
        self._kernel = make_kernel(cpoint, spoint, n_gp, n_c, n_s, parallel=self._kernel_threads > 1)
        many = f", {len(self._layouts)} sections" if len(self._layouts) > 1 else ""
        if self._kernel_threads > 1:
            many += f", {self._kernel_threads} threads"
        if not rebars:
            self.kernel_status = (f"numba fused kernel ({n_c} fibres x {n_gp} "
                                  f"stations, one law, no rebars{many})")
            return
        # count the rows actually used, which strain penetration can multiply
        grades = len({tuple(r) for r in self._spar.reshape(-1, self._spar.shape[2])})
        lay0 = self._layouts[0]
        conf = (f", cover + confined core ({len(self.group_indices[lay0.conc_groups[0]])}/"
                f"{len(self.group_indices[lay0.conc_groups[1]])} fibres)" if self.is_confined else "")
        conf += many
        if getattr(self, "strain_penetration", None):
            conf += (f", strain penetration on {len(self.strain_penetration['elements'])}"
                     f" element(s) (scale {self.strain_penetration['scale']:.3f})")
        self.kernel_status = (f"numba fused kernel ({n_c} concrete + {n_s} rebar "
                              f"fibres x {n_gp} stations, {grades} rebar "
                              f"parameter set{'s' if grades > 1 else ''}{conf})")

    def _relink_kernel_states(self):
        """Point the per-group state entries at the consolidated arrays.

        Both fibre families are stored consolidated for the kernel, and the
        entries of `fiber_states` become views onto them, so checkpoint(),
        restore() and output_fields() keep seeing live data. restore() rebinds
        `fiber_states` to fresh copies, so it has to call this again or the
        kernel goes on reading an array nothing writes to any more.

        A group's columns are copied for ITS SECTION'S ROWS only: with several
        sections the same consolidated columns hold a different group on
        another section's rows, and copying a group over every
        row would overwrite the other section's state with this law's initial
        values. The view handed back spans every row; the rows outside the
        section are simply never read for that group.
        """
        if getattr(self, "_state_s", None) is None:
            return
        for k, lay in enumerate(self._layouts):
            rows = self._rows[k]
            whole = len(self._layouts) == 1
            for j, g in enumerate(lay.rebar_groups):
                if whole:
                    self._state_s[:, :, j, :] = self.fiber_states[g][:, :, 0, :]
                else:
                    self._state_s[rows, :, j, :] = self.fiber_states[g][rows, :, 0, :]
                self.fiber_states[g] = self._state_s[:, :, j:j + 1, :]
            start = 0
            for g in lay.conc_groups:
                n = self.fiber_states[g].shape[2]
                if whole:
                    self._state_c[:, :, start:start + n, :] = self.fiber_states[g]
                else:
                    self._state_c[rows, :, start:start + n, :] = self.fiber_states[g][rows]
                self.fiber_states[g] = self._state_c[:, :, start:start + n, :]
                start += n
        if not self.is_hybrid:
            self.fiber_state = self.fiber_states[0]       # the alias follows the live array
        self._sync_committed()

    def _sync_committed(self):
        """The converged fibre state, consolidated as the kernel stores it.

        A trial evaluation starts from the last converged state. Per group
        that was some three hundred small copies on the Hangzhou Bay unit --
        1.4 ms, as long as the kernel's own work, on every one of the Newton
        solver's force evaluations. The groups are views onto
        two arrays, so the same state is two copies. Called wherever the
        committed state is written: here (setup, restore) and at a commit."""
        if getattr(self, "_state_c", None) is not None and getattr(self, "_state_s", None) is not None:
            self._com_c = self._state_c.copy()
            self._com_s = self._state_s.copy()
        else:
            self._com_c = self._com_s = None

    def _commit_fibres(self):
        """The current fibre state becomes the converged one."""
        for g, c in enumerate(self.fiber_committed):
            if c.size:
                c[...] = self.fiber_states[g]
        if getattr(self, "_com_c", None) is not None:
            self._com_c[...] = self._state_c
            self._com_s[...] = self._state_s

    def _fiber_strain(self, e, eps0, kappa_y, kappa_z):
        # eps = eps0 - y*kappa_z + z*kappa_y  (plane sections remain plane),
        # plus a tendon's initial strain (zero on every other fibre)
        return eps0 - self.fiber_y[e] * kappa_z + self.fiber_z[e] * kappa_y + self._imposed[e]

    def _refresh_imposed(self, progress):
        """The imposed strain at this instant: the tendons' prestrain plus the
        temperature field scaled by its amplitude.

        It rides with the external force (this is called from `_force`)
        because it is refreshed exactly as often as one: once per stage under
        staged relaxation, once per step in a dynamic run. Without a
        temperature load `_imposed` stays the prestrain array itself and
        nothing here runs at all.
        """
        if not self._thermal and not self.time_dependent:
            # A STEP WITHOUT A TEMPERATURE ENDS THE LAST ONE'S. Step loads are
            # not inherited, so a field a previous step put in `_imposed` must
            # come out again; returning with it in place kept a step-level
            # temperature acting for the rest of the run (found in review,
            # v1.0). A model that never had one is untouched: the
            # array is still the prestrain itself.
            if self._imposed is not self.fiber_prestrain:
                self._imposed = self.fiber_prestrain
                self._prestrain_now = self.fiber_prestrain
                self._thermal_scale = 0.0
                self._imposed_to_kernel()
            return
        prestrain = self.fiber_prestrain
        age = None
        if self.time_dependent:
            self._current_age = age = creep.age_at(self._step_ages, progress)
        if self._relax:
            # RELAXATION IS A LOSS OF THE TENDON'S OWN PRESTRAIN, not a strain
            # added to it: `sigma_p = E_p (eps_section + eps_pre)`, so a stress
            # loss at held section strain IS `eps_pre` falling.
            # It therefore multiplies the prestrain BEFORE a temperature is
            # added to the same slot -- a warm tendon has not relaxed more.
            prestrain = prestrain.copy()
            for mask, spec, t0 in self._relax:
                r = relaxation.relaxation_loss(spec, max(0.0, age - t0) * 24.0)
                if r:
                    prestrain[mask] *= (1.0 - r)
        self._prestrain_now = prestrain
        self._imposed = prestrain.copy()
        for a, eps in self._thermal:
            self._imposed += eps * a(progress)
        if self._creep_spec is not None or self._shrink_spec is not None:
            # Creep and shrinkage ride in the same slot a temperature does,
            # for the same reason: they are a strain the fibre takes on that
            # is not a force. Two halves, and neither
            # works alone -- the recorded history lands here, the increment
            # this station is about to produce is carried by the scale on the
            # law's own input in `_age_scale`.
            if self._shifted():
                # each segment on its own clock (1.8): age +
                # shift, and its first increment dated from its birth
                phi = np.zeros(self.nel)
                if self._creep_spec is not None:
                    for rows, sh, floor in self._age_groups(np.arange(self.nel)):
                        prev = None if self._prev_station is None else max(self._prev_station + sh, floor)
                        phi[rows] = creep.station_phi(self._creep_spec, age + sh, prev)
                self._phi_cur = phi
            else:
                self._phi_cur = creep.station_phi(self._creep_spec, age,
                                                  self._prev_station)
            self._age_scale()
            self._imposed += self._time_strain(age)
        # THE SIZE OF A LOAD THAT HAS NO FORCE. The converged state of a free
        # thermal expansion has neither an external force nor an internal one,
        # so the relative residual would divide a small number by a smaller
        # one. The restraint force -- what it would take to suppress the field
        # entirely, sum(E alpha T A) over the section -- is the load's
        # magnitude whether or not any of it survives.
        thermal = self._imposed - self._prestrain_now
        built = getattr(self, "member_built", None) if self.plan is not None else None
        if built is not None:
            # a member not built (yet, or any more) restrains nothing: its
            # shrinkage and creep do not size the residual (found in review,
            # 1.8 -- the frames' k_local is zero there)
            thermal = np.where(np.asarray(built, bool)[:, None], thermal, 0.0)
        E = self.fiber_E if self.fiber_E is not None else self._modulus(self.materials[0])
        self._thermal_scale = float(np.abs(E * thermal * self.fiber_area).sum())
        self._imposed_to_kernel()

    def _imposed_to_kernel(self):
        """Copy `_imposed` into the fused kernel's per-family arrays."""
        if getattr(self, "_cpre", None) is not None:
            for k, lay in enumerate(self._layouts):
                rows = self._rows[k]
                c_idx = self._c_order[k]
                self._cpre[np.ix_(rows, range(len(c_idx)))] = self._imposed[rows][:, c_idx]
                if lay.rebar_groups:
                    s_idx = np.concatenate([self.group_indices[g] for g in lay.rebar_groups])
                    self._spre[np.ix_(rows, range(len(s_idx)))] = self._imposed[rows][:, s_idx]

    _relax_what = "fiber beam staged relaxation"

    def _member_forces(self):
        if getattr(self, "_kernel", None) is not None:
            out = self._internal_force_kernel()
        else:
            out = self._internal_force_true()
        if not getattr(self, "_probing", False) or getattr(self, "_closing", False):
            # the STEP's evaluation latches a fracture and takes out what a
            # removal has made due; a reading of the state (an output frame,
            # a progress check, a probe: `_reading`) does neither. It did:
            # a member whose bars ruptured in step n was latched by the
            # frame after step n and carried nothing in step n + 1, where
            # its own evaluation gives it the broken section's force -- and
            # the result depended on `output.every` and `progress_every`
            # (1.3 mm at the end of a run; review of v1.12, fixed in
            # v1.12). The run's last frame still does: no step follows it
            self._latch_fracture()
            self._check_failure()
        else:
            self._check_failure(observe=True)
        from . import soil
        if getattr(self, "soil", None) is not None:
            f, actions = out
            return soil.add_force(self, f), actions
        return out

    # ------------------------------------------------------ the compiled step
    # What the compiled step (core/compiled_step.FiberStepper) reads of this
    # solver, given by the solver: it read the private
    # arrays itself until then.
    def fused_kernel_kind(self):
        """The fused fibre kernel this model's sections run on (None: NumPy)."""
        return self._kernel

    def fused_kernel_refusal(self):
        """Why the fused kernel cannot take this model's step, or None."""
        if getattr(self, "_kernel", None) is None:
            return refuse("fibre_section", status=str(getattr(self, "kernel_status", "")))
        if len(getattr(self, "_connector_ids", ())) and self._connector_kernel() is None:
            return refuse("connectors")
        if getattr(self, "_imposed", None) is not getattr(self, "fiber_prestrain", None):
            return refuse("fibre_imposed")
        return None

    def split_pending(self) -> bool:
        return bool(getattr(self, "_pending_splits", None) or self._pending_debris)

    def in_trial(self) -> bool:
        return bool(self.trial_mode)

    def kernel_connectors(self):
        """(ids, kinds, compiled connectors or None) of the live connectors."""
        ck = self._connector_kernel()
        live = self._connector_ids[self._active[self._connector_ids].astype(bool)] if len(self._connector_ids) else \
            np.zeros(0, np.int64)
        return (np.ascontiguousarray(live, dtype=np.int64),
                np.ascontiguousarray(self._connector_of[live], dtype=np.int64), ck)

    def fused_kernel_prepare(self):
        """The rotation scratch, and the kernel's mask as `_internal_force_kernel`
        derives it on every call."""
        if getattr(self, "_rot_scratch", None) is None or self._rot_scratch.shape != (self.nn, 3):
            self._rot_scratch = np.zeros((self.nn, 3))
        np.multiply(self._active, self._is_fiber, out=self._kernel_active)

    def fused_kernel_section(self):
        """The fused kernel's member and section arguments, in its order: the
        rotation scratch through the birth offsets."""
        return (self._rot_scratch,
                self._edofs, self._x0, self.R0, self.length, self._gw, self._Bky, self._Bkz,
                self._cy, self._cz, self._ca, self._cpre, self._sy, self._sz, self._sa, self._spre,
                self._state_c, self._state_s, self._cpar, self._spar, self._G0J0, self._GA, self._Bgy, self._Bgz,
                self._kernel_active, self._fstress, self._kernel_birth())

    def fused_kernel_connector_state(self):
        """The connectors' history the kernel reads and writes."""
        return (self._conn_committed, self._conn_slip, self.connector_deformation)

    def shear_law_args(self):
        """The cracking shear law's arrays and whether any member has it on."""
        return (*self._shear_law_arrays(), bool(np.any(self._sl_on)))

    def fracture_kernel_arrays(self):
        """(t_j, t_eps, t_valid, point) for the fibre step's fracture latch
        (1.6): the tracked columns of `_point_failure` as steel slots of the
        kernel's `_state_s` with their rupture strains, or None when a tracked
        law is not a steel02 bar (the only kernel law with a failure criterion).
        The compiled step's (v1.18: compiled_step._fracture_arrays
        until then)."""
        fr = getattr(self, "fracture", None)
        if fr is None:
            return None
        cache = getattr(self, "_fracture_arrays_cache", None)
        if cache is not None and cache[0] is fr.latch.point:
            return cache[1]
        slot = {}
        for lay in self._layouts:
            for j, g in enumerate(lay.rebar_groups):
                slot[int(g)] = j
        t_j, t_eps = [], []
        for g, start, n in self._failing_groups:
            m = self.materials[g]
            eps = getattr(m, "eps_rupture", None)
            if int(g) not in slot or n != 1 or eps is None or type(m).__name__ != "Steel02":
                return None
            if self.fiber_states[g].shape[3] <= 7:
                return None
            t_j.append(slot[int(g)]); t_eps.append(float(eps))
        nt = len(t_j)
        tv = getattr(self, "_tracked_valid", None)
        valid = (np.ascontiguousarray(np.asarray(tv, float).reshape(self.nel, nt)) if tv is not None
                 else np.ones((self.nel, nt)))
        point = fr.latch.point
        if point.shape != (self.nel, nt * len(self.gauss_s)) or not point.flags.c_contiguous:
            return None
        out = (np.ascontiguousarray(t_j, dtype=np.int64), np.ascontiguousarray(t_eps, dtype=float), valid, point)
        self._fracture_arrays_cache = (point, out)
        return out

    def _kernel_birth(self):
        """(nel, 12) the members' birth offsets for the kernel (1.2): zeros
        when the steps build nothing, and the same array every call."""
        b = self.staging.birth_d if self.plan is not None else None
        if b is not None:
            return np.ascontiguousarray(b, dtype=float)
        z = getattr(self, "_kernel_zero_birth", None)
        if z is None or z.shape[0] != self.nel:
            z = self._kernel_zero_birth = np.zeros((self.nel, 12))
        return z

    #: the members' end offsets as the fused kernel takes them (core/member_geometry)
    _koffs = None

    def _kernel_offsets(self):
        """(offs (nel, 2, 3), has (nel,) uint8) for the fused kernel: zeros
        on a model without end offsets."""
        if self._koffs is None or self._koffs[0].shape[0] != self.nel:
            self._koffs = self.geometry.kernel_arrays(np.arange(self.nel))
        return self._koffs

    def _internal_force_kernel(self):
        """Fused path. `_internal_force_true` remains the reference it copies."""
        if self.trial_mode:
            if getattr(self, "_com_c", None) is not None:
                self._state_c[...] = self._com_c
                self._state_s[...] = self._com_s
            else:
                for g, c in enumerate(self.fiber_committed):
                    if c.size:
                        self.fiber_states[g][...] = c
        resume = self.__dict__.pop("_kernel_resume", None)
        if resume is not None:
            # the compiled step made this step's member forces (the fibre
            # kernel and the connectors, the state moved with them) and
            # handed the step over before its contact: a fibre point failed,
            # and the latch below comes first (compiled_step, status 5)
            at, out, actions, end_actions = resume
            if at != self.step or getattr(self, "_probing", False):
                raise RuntimeError("the compiled step's member forces of step "
                                   f"{at} were asked for at step {self.step}: this is a defect in vfife")
            np.multiply(self._active, self._is_fiber, out=self._kernel_active)
            self.end_actions = end_actions
            station_abs = np.where(self._kernel_active[:, None].astype(bool), actions[:, 6:10], np.nan)
            actions = np.ascontiguousarray(actions[:, :6])
            self._set_actions_abs(actions, station_abs)
            return out, actions
        out = np.zeros(self.ndof)
        # columns 6-9: the stations' plain magnitudes (`*_max_abs`, 1.0 re-review)
        actions = np.zeros((self.nel, 10))
        end_actions = np.zeros((self.nel, 12))
        # DERIVED ON EVERY CALL, NOT CACHED. `_refresh_kernel_mask` ran
        # wherever the solver itself wrote `_active` -- and anything else that
        # writes it (a test deactivating a member by hand, a caller cutting
        # one out) was ignored by the kernel while the NumPy reference honoured
        # it: tests/test_fracture.py's kernel-against-reference check failed
        # from v0.4 until the 0.4.0 release check caught it. One
        # in-place multiply of nel bytes makes the mask impossible to go stale.
        np.multiply(self._active, self._is_fiber, out=self._kernel_active)
        call = self._kernel
        n = getattr(self, "_kernel_threads", 1)
        if n > 1:
            from ..kernels.numba_frame import with_threads
            call = lambda *a: with_threads(n, self._kernel, *a)
        call(self._edofs, self._x0, self.q, self.orient, self.R0, self.length,
                     self._gw, self._Bky, self._Bkz,
                     self._cy, self._cz, self._ca, self._cpre,
                     self._sy, self._sz, self._sa, self._spre,
                     self._state_c, self._state_s,
                     self._cpar, self._spar, self._G0J0,
                     self._GA, self._Bgy, self._Bgz,
                     out, actions, end_actions, self._kernel_active, self._fstress,
                     self._kernel_birth(), *self._shear_law_arrays(), bool(getattr(self, "_sl_update", False)),
                     *self._kernel_offsets())
        # THE CONNECTORS ARE NOT IN THE KERNEL, and the mask is why: `_active`
        # is the fracture latch and it is WRITTEN by `_latch_fracture`, so a
        # connector cannot borrow it -- a fracture anywhere would revive it.
        # `_kernel_active` is `_active & _is_fiber`, refreshed wherever
        # `_active` changes.
        ck = self._connector_kernel()
        if ck is not None:
            # COMPILED, ALL OF THEM IN ONE CALL: two pot
            # bearings were 0.8 ms of the Hangzhou Bay seismic step in NumPy.
            # The same mechanics to round-off (kernels/numba_connector);
            # removed ones are left out, as below
            live = self._connector_ids[self._active[self._connector_ids].astype(bool)]
            if len(live):
                from ..kernels.numba_connector import respond
                enc = ck["enc"]
                respond(np.ascontiguousarray(live, dtype=np.int64),
                        np.ascontiguousarray(self._connector_of[live], dtype=np.int64),
                        # the connectivity and the frames read live: a split moves ends
                        np.ascontiguousarray(np.asarray(self.mesh.elements)[:, :2], dtype=np.int64),
                        self.mesh.nodes, self.dofs.view(self.q), self.orient, self.v,
                        np.ascontiguousarray(np.asarray(self.R0, dtype=float)), ck["spins"], *enc, self._conn_committed, self._conn_slip,
                        self.mode == "dynamic", out, actions, end_actions, self.connector_deformation, 6,
                        _NO_ROWS, False, False, _NO_BIRTH if self._conn_birth is None else self._conn_birth)
        else:
            for e in self._connector_ids:
                if not self._active[e]:
                    # removed (`removals`): it was reported failed and went on
                    # carrying 20 kN
                    continue
                conn = self.mesh.elements[e]
                dof = self.dofs.element_dofs(conn)
                fe, act, deform = self._connector_force(e, conn, dof)
                out[dof] += fe
                # the first six columns: the kernel's array has four more for the
                # stations' magnitudes (v1.0: a fibre model with a
                # connector crashed on numba)
                actions[e, :6] = np.abs(act)
                end_actions[e, :6] = -act; end_actions[e, 6:] = act
                self.connector_deformation[e] = deform
        self.end_actions = end_actions
        station_abs = np.where(self._kernel_active[:, None].astype(bool), actions[:, 6:10], np.nan)
        actions = np.ascontiguousarray(actions[:, :6])
        self._set_actions_abs(actions, station_abs)
        return out, actions

    def _internal_force_true(self):
        """True fibre section integration and Hermite virtual-work assembly."""
        if self.trial_mode:
            # Start from the last converged state, so an evaluation partway
            # through a stage cannot leave permanent history behind.
            for g, c in enumerate(self.fiber_committed):
                if c.size:
                    self.fiber_states[g][...] = c
        out = np.zeros(self.ndof); actions = np.empty((self.nel, 6))
        station_abs = np.full((self.nel, 4), np.nan)
        end_actions = np.zeros((self.nel, 12))
        G0J0_all = self._G0J0          # per element AND per section
        # the co-rotational frame and the local nodal rotations of every
        # member from the points' orientations (until then the
        # global displacement was rotated into the chord frame and the
        # Hermite functions were left to cancel the rigid part, which they do
        # to third order in the member's rotation)
        R_all, theta_loc, L_all = self._frames()
        birth = self.staging.birth_d if self.plan is not None else None
        sl_on = self._sl_on if (getattr(self, "_sl_on", None) is not None
                                and self._sl_on.any()) else None
        for e, conn in enumerate(self.mesh.elements):
            dof = self.dofs.element_dofs(conn)
            L = L_all[e]
            if L < 1e-14:
                raise ValueError("beam2_fiber member collapsed to zero length")
            if not self._is_fiber[e]:
                if not self._active[e]:
                    continue
                fe, act, deform = self._connector_force(e, conn, dof)
                out[dof] += fe
                actions[e] = np.abs(act)
                end_actions[e, :6] = -act; end_actions[e, 6:] = act
                self.connector_deformation[e] = deform
                continue
            if not self._active[e]:
                # A fractured member carries nothing, and its fibres are not
                # driven any further: their state is the one it broke in, which
                # is what the FAC output reports afterwards. Its MASS is
                # untouched -- lumped on the nodes at setup, and nothing here
                # removes it (core/fracture.py).
                actions[e] = 0.0
                continue
            R = R_all[e]; T = repeated_rotation(R); L0 = self.length[e]
            local = np.zeros(12); local[6] = L - L0
            local[3:6] = theta_loc[e, 0]; local[9:12] = theta_loc[e, 1]
            if birth is not None:
                local -= birth[e]          # born stress-free where it stood (1.2)
            eps0 = local[6] / L0
            shear_law = sl_on is not None and sl_on[e]
            if shear_law:
                # the bending sees the end rotations less the shear rotation
                # (_setup_shear_law)
                ri, rj = self._sl_rot[e]
                g_e = self._sl_gamma[e]
                local[ri] -= g_e; local[rj] -= g_e
                cracked_here = False
                conc = self._sl_conc[self._section_index(e)]
            # Peak magnitude over the integration points, which is what the other
            # element families report: beam_solver takes max(|fl_i|, |fl_j|).
            # This path used to accumulate `weight * L0 * [...]`
            # into these and report THAT -- an integral along the element, L0
            # times the section force, so a 400 mm element reported 0.4x. It
            # read plausibly (case_11 stored 819 kN under an applied 819 kN)
            # purely because a second, unrelated L0 error upstream cancelled it.
            fl = np.zeros(12); peak = np.zeros(4)
            # the plain magnitudes by the old rule: `*_max_abs` is the
            # pre-v1.0 field bit for bit (1.0 re-review)
            pabs = np.zeros(4)
            for gi, (s, weight) in enumerate(zip(self.gauss_s, self.gauss_w)):
                # the curvature rows: Hermite, or Reddy's Timoshenko rows when
                # the section states a shear stiffness (_setup_shear_interpolation)
                Bkz = self._Bkz[e, gi]
                Bky = self._Bky[e, gi]
                kz = Bkz @ local[[1, 5, 7, 11]]
                ky = Bky @ local[[2, 4, 8, 10]]
                strains = self._fiber_strain(e, eps0, ky, kz)
                stresses = self._fiber_stress_rows[e, gi]           # padded columns stay 0
                stresses[:] = 0.0
                dt = self.dt if hasattr(self, "dt") else 1.0
                lay = self._layouts[self._section_index(e)]
                scale_el = getattr(self, "_scale_el", None)
                if scale_el is not None:
                    # each segment's own creep scale (1.8, `_age_scale`)
                    for g_, base in self._base_scale.items():
                        self.materials[g_].strain_scale = base * float(scale_el[e])
                for j, g in enumerate(lay.conc_groups + lay.rebar_groups):
                    mat = self.materials[g]
                    idx = self.group_indices[g]
                    strain6 = np.zeros((len(idx), 6)); strain6[:, 0] = strains[idx]
                    # rebar groups may carry element-dependent parameters
                    par = (self._spar[e, j - len(lay.conc_groups)]
                           if (self._spar is not None and j >= len(lay.conc_groups))
                           else None)
                    if mat.name == "linear_elastic":
                        # A FIBRE IS UNIAXIAL STRESS, NOT UNIAXIAL STRAIN. The
                        # 3-D law handed [eps, 0, 0, 0, 0, 0] answers with
                        # C11 eps = E (1 - nu) / ((1 + nu)(1 - 2 nu)) eps: the
                        # constrained modulus, 7% stiff at nu = 1/6, 35% at
                        # 0.3. Every test had nu = 0 and never saw it; the
                        # Hangzhou Bay girder's static check did.
                        out6 = strain6 * mat.E
                    elif par is None:
                        out6 = mat.update(strain6, self.fiber_states[g][e, gi], dt)
                    else:
                        out6 = mat.update(strain6, self.fiber_states[g][e, gi], dt, par)
                    stresses[idx] = out6[:, 0]
                fa_e, fz_e, fy_e = self.fiber_area[e], self.fiber_z[e], self.fiber_y[e]
                if shear_law and self._sl_flex[e] and not cracked_here:
                    cracked_here = bool(np.max(strains[conc]) > self._sl_epscr[e])
                N = float(np.sum(stresses * fa_e))
                My = float(np.sum(stresses * fz_e * fa_e))
                Mz = float(np.sum(-stresses * fy_e * fa_e))
                # the largest station WITH its sign, a tie to the first
                # station (kernels/numba_fiber_beam.TIE)
                if abs(N) > abs(peak[0]) * _TIE:
                    peak[0] = N
                if abs(My) > abs(peak[2]) * _TIE:
                    peak[2] = My
                if abs(Mz) > abs(peak[3]) * _TIE:
                    peak[3] = Mz
                pabs[0] = max(pabs[0], abs(N))
                pabs[2] = max(pabs[2], abs(My))
                pabs[3] = max(pabs[3], abs(Mz))
                # dEps/du and dTwist/du, NOT [-1, 1].  eps0 = (L - L0)/L0 and
                # twist = (rx_j - rx_i)/L0, so both derivatives carry a 1/L0 that
                # was missing here. With `fl += weight * L0 * (...)` on the next
                # line the axial and torsional internal forces came out L0 times
                # too large, so equilibrium under an applied axial load P settled
                # at a section force of P/L0. Measured on case_11 before the fix:
                # 819 kN applied on 400 mm elements put N = 2048 kN through the
                # section -- 0.5 f'c Ag instead of the intended 0.2 -- which
                # raised the moment capacity from 291 to 321 kN.m and made the
                # column 20% too strong against Tanaka & Park. The bending terms
                # were always right: Bky/Bkz already carry their own 1/L0 powers.
                Ba = np.zeros(12); Ba[[0, 6]] = [-1.0 / L0, 1.0 / L0]
                Bt = np.zeros(12); Bt[[3, 9]] = [-1.0 / L0, 1.0 / L0]
                By = np.zeros(12); By[[2, 4, 8, 10]] = Bky
                Bz = np.zeros(12); Bz[[1, 5, 7, 11]] = Bkz
                twist = (local[9] - local[3]) / L0
                Mt = float(G0J0_all[e]) * twist
                if abs(Mt) > abs(peak[1]) * _TIE:
                    peak[1] = Mt
                pabs[1] = max(pabs[1], abs(Mt))
                fl += weight * L0 * (N * Ba + Mt * Bt + My * By + Mz * Bz)
            Vy = Vz = 0.0
            if shear_law:
                # the member's shear from its law, and one Newton step on
                # M_i + M_j = L0 V(gamma) for the next evaluation
                gcr = self._sl_gcr_c[e]
                if not np.isfinite(gcr) and cracked_here:
                    gcr = max(abs(g_e), 1e-30)      # cracks at the shear it carries now
                gmax = max(self._sl_gmax_c[e], abs(g_e))
                V, kv = self._shear_law(e, g_e, gcr, gmax)
                self._sl_gcr_t[e] = gcr; self._sl_gmax_t[e] = gmax
                self._sl_V[e] = V
                if self._sl_update:
                    r = fl[ri] + fl[rj] - L0 * V
                    self._sl_gamma[e] = g_e + r / (self._sl_kb[e] + L0 * kv)
                # held here as the GA rows' would be (Vy along +y, Vz minus
                # the z shear), so the one signing below serves both: the law's
                # V is (M_i + M_j)/L0, which is +S_z about y and -S_y about z
                # (core/member_actions)
                if ri == 4:
                    Vz = -V
                else:
                    Vy = -V
            if self._GA[e] > 0.0:
                # elastic member shear: constant along the
                # element, so its virtual work is L0 * V * Bg, no station loop
                gz = self._Bgz[e] @ local[[1, 5, 7, 11]]
                gy = self._Bgy[e] @ local[[2, 4, 8, 10]]
                Vy = self._GA[e] * gz; Vz = self._GA[e] * gy
                fl[[1, 5, 7, 11]] += L0 * Vy * self._Bgz[e]
                fl[[2, 4, 8, 10]] += L0 * Vz * self._Bgy[e]
            # signed (core/member_actions): GA gy is minus the z shear on the
            # +x face, GA gz is the y shear
            actions[e] = [peak[0], Vy, -Vz, peak[1], peak[2], peak[3]]
            station_abs[e] = pabs[[0, 1, 2, 3]]
            g = T.T @ fl
            if self.geometry.any:
                g = g.reshape(1, 12)
                self.geometry.forces_to_nodes([e], g, self.orient)
                g = g[0]
            out[dof] += g
            end_actions[e] = fl
        self.end_actions = end_actions
        self._set_actions_abs(actions, station_abs)
        return out, actions

    def _set_actions_abs(self, actions, station_abs):
        """`self.actions_abs` (nel, 6): what `*_max_abs` reports -- |N|, |Mt|,
        |My|, |Mz| at their largest station by the pre-v1.0 strictly-larger
        rule (so the field is the old unsigned one bit for bit, whatever the
        signed field's tie rule picks), |V| and a connector's |action| as they
        are (1.0 re-review). `station_abs` (nel, 4) is NaN where not a fibre
        member evaluated in this call."""
        a = np.abs(np.asarray(actions, dtype=float))
        rows = ~np.isnan(station_abs[:, 0])
        if rows.any():
            a[np.ix_(rows, [0, 3, 4, 5])] = station_abs[rows]
        self.actions_abs = a

    def _time_step_bound(self):
        """The explicit stability bound: dt_safety * sqrt(min m/k) over the
        free degrees of freedom that are still attached to something.

        THE ATTACHED PART MATTERS once particles can be reserved: a parked
        spare has no member on it, so its stiffness row is empty and its mass
        is the 1e-20 floor -- a ratio of 1, meaningless, and it would be the
        minimum in any model whose real ratios are the 1e-8 they usually are
        only by luck. A particle every one of whose members has fractured is
        the same case and drops out for the same reason: nothing about it can
        go unstable, because nothing pushes on it.
        """
        row = np.asarray(abs(self.K).sum(axis=1)).ravel()
        free = ~self._held_dofs()
        if getattr(self, "bonds", None) is not None:
            free = free & np.repeat(self.bonds.attachments() > 0, self.dofs.stride)
        ratio = np.where(free, self.mass / np.maximum(row, 1e-20), np.inf)
        root = float(np.sqrt(np.min(ratio)))
        # THE CONNECTOR DASHPOTS' OWN BOUND, dt < 2 m / c, on the free dofs,
        # in a dynamic run (the only one they act in). It was missing here:
        # a stiff bearing dashpot, c dt / m = 2.16, went non-finite.
        dash = 1.0
        if self.mode == "dynamic" and len(self._connector_ids):
            rows = connector.dashpot_rows(
                [e for e in self._connector_ids if self._active[e]], self.mesh.elements,
                self.mesh.nodes, self.R0,
                lambda e: self._connector_props[int(self._connector_of[e])], self.dofs, self.ndof)
            on = free & (rows > 1e-20)
            if np.any(on):
                dash = float(np.min(2.0 * self.mass[on] / rows[on]))
        # and what the EXPLICIT damping terms take off it: a Rayleigh beta of
        # 0.001 on the lattice dome of v0.1 cut the stable step by
        # half, and the bound handed out 0.6 / omega as if there were none
        self.damping_step_factor = self._damping_step_factor(1.0 / root)
        return float(self.p.solver.dt_safety
                     * min(root * self.damping_step_factor, dash))

    def _element_mass_share(self, e, rotated=True):
        """What ONE END of element `e` carries: [m, m, m, Jrot, Iy, Iz] x rho L/2.

        ONE DEFINITION, because it is used twice and the two uses have to agree
        exactly or the split stops conserving mass: the setup lumps it onto the
        member's two particles, and a split moves this same row off a particle
        and onto the fragment's own. Written as a function rather than repeated
        so that "conserved" is arithmetic, not a comment.
        """
        # each fibre at its own material's density
        L = float(self.length[e])
        mA, mIy, mIz = (float(self.mass_per_length[e]), float(self.rho_Iy[e]),
                        float(self.rho_Iz[e]))
        translational = mA * L / 2.0
        rotary = L * np.array([mIy + mIz, mIy, mIz]) / 2.0
        if rotated:
            # local [Ip, Iy, Iz] turned to global axes; the
            # unrotated row is the old rule the step is taken from
            rotary = mass_scaling.rotary_diagonal(self.R0[e], rotary)
        return np.concatenate([np.full(3, translational), rotary])

    def _setup_mass_and_time(self):
        sol = self.p.solver
        self.mode = sol.mode
        m = np.zeros((self.nn, 6))
        true_rot = np.zeros((self.nn, 3))
        for e, conn in enumerate(self.mesh.elements):
            m[conn] += self._element_mass_share(e, rotated=False)
            true_rot[conn] += self._element_mass_share(e)[3:]
        # nodal and hydrodynamic masses; a connector has no
        # section for water to act on
        kinds = getattr(self, "member_kind", None)
        add, rot, self.weightless_mass = added_mass.table(
            self.p, self._region, self.mesh.nodes[:self.nn], self.mesh.elements[:, :2],
            member_ok=None if kinds is None else np.asarray(kinds) != "connector2")
        self.line_mass_per_end = added_mass.member_line_mass(
            self.p.get("loads"), self._region, self.mesh.nodes[:self.nn], self.mesh.elements[:, :2],
            None if kinds is None else np.asarray(kinds) != "connector2", self.p.case.units)
        m[:, :3] += add; m[:, 3:] += rot; self.additional_mass = add
        self._add_zone_mass(m)                      # the rigid zones' own (1.21)
        true_rot += rot
        self.physical_mass = m.reshape(-1)
        # selective mass scaling, before anything reads the mass.
        # After a split the fragment's new particle gets only its member's
        # share, not a scaling of its own: the bound is re-evaluated then and
        # the step may drop below the target, which the log says.
        self.mass_scaling = mass_scaling.scale(
            self.physical_mass, np.asarray(abs(self.K).sum(axis=1)).ravel(), ~self._held_dofs(),
            sol.mass_scaling, sol.dt_safety)
        step_mass = self.physical_mass.copy()
        self._rotary_true = true_rot
        # DAMPING BEFORE THE STEP. `_time_step_bound` folds the explicit
        # damping terms into the bound, so they have to be known when it is
        # first called -- and it is called again after every particle split,
        # where the same factor applies.
        damp = sol.damping
        self.damp_kind = damp.type or ("kinetic" if self.mode == "relaxation" else "viscous")
        self.zeta, self.alpha, self.beta = float(damp.coeff), float(damp.alpha), float(damp.beta)
        row = np.asarray(abs(self.K).sum(axis=1)).ravel()
        if self.mode == "relaxation":
            self.mass = np.maximum(row / float(sol.relaxation_safety)**2, 1e-20)
            self.dt, self.n_steps, self.total_time = 1.0, int(sol.n_steps), float(sol.n_steps)
            self._n_steps_solver = self.n_steps
            self.damping_step_factor = 1.0
        else:
            self.mass = np.maximum(step_mass, 1e-20)      # the step's mass
            from ..elements.stability import aligned_step
            self.total_time = float(sol.total_time)
            # n_steps * dt is the run's duration exactly
            self.dt_bound = self._time_step_bound()
            self.stable_dt = self.dt_bound
            self.dt, self.n_steps = aligned_step(self.total_time, self.dt_bound)
            # what the damping costs in step, and the alpha/beta a `ratio`
            # became -- the line the manual promises; a fibre run printed none
            # and the step fell 10x without a word
            if self.damp_kind == "rayleigh" and callable(getattr(self, "log", None)):
                f = float(getattr(self, "damping_step_factor", 1.0))
                line = f"  damping       rayleigh alpha={self.alpha:.4g} 1/s, beta={self.beta:.4g} s"
                if f < 1.0:
                    line += f"; step x{f:.3f} for beta"
                    if hasattr(self, "omega_max_sharp"):
                        line += (f" at omega_max {self.omega_max_sharp:.4g} rad/s (power iteration; "
                                 f"Gershgorin {self.omega_max_gershgorin:.4g})")
                self.log(line)
        # the step is set: the true rotary inertia, floored where that step
        # needs it -- a connector dashpot's 2m/c too
        dash_floor = None
        if self.mode == "dynamic" and len(self._connector_ids):
            rows = connector.dashpot_rows(
                [e for e in self._connector_ids if self._active[e]], self.mesh.elements,
                self.mesh.nodes, self.R0,
                lambda e: self._connector_props[int(self._connector_of[e])], self.dofs, self.ndof)
            dash_floor = rows * self.dt / (2.0 * float(sol.dt_safety))
        self.physical_mass, self.rotary_floored = mass_scaling.true_rotary(
            step_mass, true_rot, np.asarray(abs(self.K).sum(axis=1)).ravel(), ~self._held_dofs(),
            dash_floor)
        if self.mode != "relaxation":
            self.mass = np.maximum(self.physical_mass, 1e-20)
        self.tolerance = float(sol.tolerance)
        self.kinetic_tolerance = float(sol.kinetic_tolerance)

    def _thermal_load(self, idx, spec):
        """(nel, n_fibres) of alpha * T at every fibre of the region's members,
        per unit amplitude.

        A fibre section is what lets a temperature field be more than two
        numbers: `profile` is sampled AT EACH FIBRE, so a nonlinear profile
        keeps the part of itself that plane sections cannot take on and turns
        it into self-equilibrated stress -- zero net force, zero net moment,
        and tension where a code's deck gradient puts it.
        """
        from .temperature import section_field
        elements = self._members_in(idx)
        if not len(elements):
            raise ValueError("a temperature load needs complete members; region contains none")
        out = np.zeros(self.fiber_y.shape)
        T = section_field(spec, self.fiber_y[elements], self.fiber_z[elements])
        # MINUS alpha T. What is added here is what the MATERIAL sees, and a
        # material sees the mechanical strain: the total strain the section's
        # plane gives it, less the part the temperature took for free. So a
        # fibre held at length while it is heated is in compression, which is
        # the sign a restrained beam must produce. (A tendon's `prestrain`
        # sits in the same array with the opposite sense -- it is a strain the
        # steel really carries.)
        out[elements] = -self.fiber_alpha[elements] * T
        if not np.any(out):
            raise ValueError(
                "a temperature load on a section whose materials have no "
                "`thermal_expansion`: "
                "every fibre would expand by zero. State the coefficient of thermal "
                "expansion in the material block: `thermal_expansion: 1.0e-5` for "
                "concrete, ~1.2e-5 for steel.")
        return out

    def _force(self, groups, ground, progress, t=None):
        if self.plan is not None and getattr(self, "_birth_raw", None) is not None:
            # the members born this step: their clocks, before the imposed
            # strains below are read on them (1.8; the frames' since 1.2)
            from . import analysis_steps as _as
            _as.birth_offsets(self)
        self._refresh_imposed(progress)
        f = np.zeros(self.ndof)
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

    def _running(self):
        """The shear rotations relax only inside a run: a probe outside one
        (the modal tangent, a strain reading) evaluates the forces at the
        gamma the run left."""
        @contextmanager
        def running():
            self._sl_update = True
            try:
                yield
            finally:
                self._sl_update = False
        return running()

    def _default_load_steps(self):
        """A nonlinear section takes 20 levels when nothing says (the
        default the fibre solver has had from the start)."""
        return 20 if (self.is_hybrid or self.material.name != "linear_elastic") else 0

    def _default_ramp(self, budget):
        """ONE ITERATION, the v0.1 rule, because it was measured: on
        case_11 (480 cyclic levels) ramps of 1, 50, 150, 300 and 600 change
        the peak shear by 0.3% and the hysteretic energy by 0.1%, and cost
        24, 24, 28, 36 and 56 s; on case_21 the frames' default fifth costs
        1.4x for 0.2%. A fibre level converges in a few hundred iterations
        and a ramp longer than that is spent waiting, not damping a local
        mode. `stage_ramp_iterations` turns it on."""
        return 1

    def _level_failures(self):
        return self._committed_failures()

    def _level_commit(self):
        self._commit_fibres()
        self.commit_state()

    def _level_latch(self, t):
        self._latch_fracture(committed=True, t=t)

    def compiled_pending(self):
        """A fracture found by a probe (the progress check's force
        evaluation) left a split for the end of the next step
        (compiled_step.drive)."""
        return bool(self._pending_splits or self._pending_debris)

    def compiled_event(self, k, local, F, data, after):
        """A fibre point failed in step `local`, which the kernel finished:
        the latch is set as the loop sets it after that step's forces, the
        split queued and applied at the step's end (1.6)."""
        self._latch_fracture()
        t = (local / self.n_steps) * self.total_time
        if (self._pending_splits or self._pending_debris) and self.apply_pending_splits(k*self.total_time + t):
            data[:] = self.step_data(k)
            self.current_fixed = data[0]
        after(local, F)

    @contextmanager
    def reading(self):
        """A driver's force evaluation that is not a step (the analysis steps'
        reactions and the like): the fibres', the kernel's and the shear
        laws' states as they were before it. The buckling and modal tangents
        read this solver in `trial_mode` instead (`TRIAL_STATE`)."""
        saved = {}
        for name in ("_state_c", "_state_s", "_conn_slip", "_sl_gcr_t", "_sl_gmax_t", "_sl_gamma"):
            v = getattr(self, name, None)
            if isinstance(v, np.ndarray):
                saved[name] = v.copy()
        fibres = [np.array(a, copy=True) for a in self.fiber_states]
        try:
            yield
        finally:
            for name, v in saved.items():
                getattr(self, name)[...] = v
            for a, b in zip(self.fiber_states, fibres):
                a[...] = b

    def _probe_hold(self):
        """What a probe (`BeamFamily._reading`) holds besides `_probing`: the
        shear laws' gamma is not relaxed by it -- `_record` gave gamma one more
        Newton step at every frame, so the answer depended on `output.every`
        (found in the review of 1.7; the frames' hinges had the same fault,
        v1.0) -- and it leaves the fibres as it found them. Returns
        what puts them back."""
        was = getattr(self, "_sl_update", False)
        self._sl_update = False
        # ... AND IT LEAVES THE FIBRES AS IT FOUND THEM. A law accepts
        # its last evaluation when it is evaluated again: steel02 reads
        # its rupture against the accepted maximum strain, so
        # the evaluation AFTER the one that passed the limit is the one
        # that cuts the bar. A frame or a progress check between the two
        # was that evaluation, and the bar was cut a step early: with a
        # frame after every step every rupture came a step early, the
        # girder of tests/test_compiled_breaking.py ended 17 mm from
        # where it ends with no frame at all. A staged
        # relaxation is in trial mode and restores the committed state
        # itself.
        saved = None
        if not self.trial_mode:
            if getattr(self, "_kernel", None) is not None and getattr(self, "_state_c", None) is not None:
                saved = ("kernel", self._state_c.copy(), self._state_s.copy())
            else:
                saved = ("groups", [st.copy() for st in self.fiber_states])

        def undo():
            self._sl_update = was
            if saved is not None:
                if saved[0] == "kernel":
                    self._state_c[...] = saved[1]; self._state_s[...] = saved[2]
                else:
                    for st, keep in zip(self.fiber_states, saved[1]):
                        st[...] = keep
        return undo

    def section_strains(self):
        """(nel, n_stations, 3): the generalised strains [eps0, kappa_y,
        kappa_z] of every station from the CURRENT state -- the same
        kinematics as the internal force, without the materials. What a
        strain reading at a point of the section is made of."""
        R_all, theta_loc, L_all = self._frames()
        out = np.zeros((self.nel, len(self.gauss_s), 3))
        for e in range(self.nel):
            if not self._is_fiber[e]:
                continue               # a connector has no section to strain
            L0 = self.length[e]
            local = np.zeros(12); local[6] = L_all[e] - L0
            local[3:6] = theta_loc[e, 0]; local[9:12] = theta_loc[e, 1]
            birth = self.staging.birth_d if self.plan is not None else None
            if birth is not None:
                local -= birth[e]
            sl = getattr(self, "_sl_on", None)
            if sl is not None and sl.any() and sl[e]:
                # the bending sees the end rotations less the shear rotation,
                # as in the force path; without it a strain
                # reading on a shear-law member counted the shear rotation as
                # curvature -- 6x the fibre's strain (found in review)
                ri, rj = self._sl_rot[e]
                local[ri] -= self._sl_gamma[e]; local[rj] -= self._sl_gamma[e]
            out[e, :, 0] = local[6] / L0
            for gi in range(len(self.gauss_s)):
                out[e, gi, 2] = self._Bkz[e, gi] @ local[[1, 5, 7, 11]]
                out[e, gi, 1] = self._Bky[e, gi] @ local[[2, 4, 8, 10]]
        return out

    def _fiber_monitor_station(self, n):
        """(element, station) of the station nearest node `n`, or None."""
        owners = np.flatnonzero((self.mesh.elements[:, :2] == n).any(axis=1) & self._is_fiber)
        best = None
        for e in owners:
            end = 0 if self.mesh.elements[e, 0] == n else 1
            d = np.abs(self.gauss_s - float(end))
            gi = int(np.argmin(d))
            if best is None or d[gi] < best[0]:
                best = (d[gi], int(e), gi)
        return None if best is None else best[1:]

    def _fiber_monitor_column(self, mon, e):
        """The fibre column a `field: stress` reading of element `e` takes:
        the live fibre nearest `at`, among the concrete or the steel fibres
        when `fiber_kind` says so."""
        y0, z0 = float(mon.at[0]), float(mon.at[1])
        k = int(self._section_of[e]) if self._section_of is not None else 0
        lay = self._layouts[k]
        cols = np.arange(self.fiber_y.shape[1])
        live = self.fiber_area[e] > 0.0
        if mon.fiber_kind:
            steel = np.zeros(len(cols), bool)
            steel[lay.n_c:lay.n_c + lay.n_s] = True
            live &= steel if mon.fiber_kind == "steel" else ~steel & (cols < lay.n_c)
        dist = np.where(live, (self.fiber_y[e] - y0) ** 2 + (self.fiber_z[e] - z0) ** 2, np.inf)
        return int(np.argmin(dist)) if np.isfinite(dist).any() else None

    def _check_fiber_monitors(self):
        """A `fiber` monitor's point inside the section it reads, and what it
        reads, said once. `at: [-0.6, 0]` on a 0.5 m wide,
        1.2 m deep girder -- y is across the width, z the depth -- read the
        side of the section at mid-depth, 4e-5 against the soffit's 1e-3,
        and the manual's own example wrote it so."""
        from .diagnostics import UserError
        from ..elements import fiber_section
        for name, mon in self.monitors.items():
            if mon.kind != "fiber":
                continue
            y0, z0 = float(mon.at[0]), float(mon.at[1])
            idx = np.unique(np.concatenate([self._region(r) for r in mon.regions]))
            first = None
            for n in idx:
                hit = self._fiber_monitor_station(n)
                if hit is None:
                    continue
                e, gi = hit
                k = int(self._section_of[e]) if self._section_of is not None else 0
                fsec = self._layouts[k].entry.get("fiber_section") or {}
                ftype = fsec.get("type", "rect")
                inside = fiber_section.outline_contains(
                    ftype, y0, z0, width=fsec.get("width"), height=fsec.get("height"),
                    diameter=fsec.get("diameter"), patches=fsec.get("patches"),
                    cells=fsec.get("cells"), exact=bool(fsec.get("section_size")))
                if inside is not None and not bool(np.all(inside)):
                    y_all, z_all = self.fiber_y[e][self.fiber_area[e] > 0], self.fiber_z[e][self.fiber_area[e] > 0]
                    raise UserError(
                        "FIBER_MONITOR_OUTSIDE",
                        f"monitor {name!r}: at = [{y0:g}, {z0:g}] is outside the section of element {e} "
                        f"({fiber_section.outline_name(ftype, bool(fsec.get('section_size')))}); `at` is [y, z] in the section's axes, "
                        f"y across the width and z through the depth, and its fibres span "
                        f"y {y_all.min():.4g}..{y_all.max():.4g}, z {z_all.min():.4g}..{z_all.max():.4g}",
                        "a gauge on the soffit of a rect section of height h is at [0, -h/2]")
                if first is None:
                    first = (e, gi)
            if first is None:
                continue
            e, gi = first
            read = {"read_element": e, "read_station": float(self.gauss_s[gi]), "read_nodes": int(len(idx))}
            if mon.field == "stress":
                c = self._fiber_monitor_column(mon, e)
                if c is None:
                    raise UserError("FIBER_MONITOR_NO_FIBRE",
                                    f"monitor {name!r}: the section of element {e} has no "
                                    f"{mon.fiber_kind} fibre to read", "drop fiber_kind, or point at the other kind")
                k = int(self._section_of[e]) if self._section_of is not None else 0
                lay = self._layouts[k]
                g = int(lay.fiber_group[c])
                kind = ("concrete" if c < lay.n_c else
                        "tendon" if any(c - lay.n_c == j for j, _t in lay.tendons) else "bar")
                read.update(read_y=float(self.fiber_y[e, c]), read_z=float(self.fiber_z[e, c]),
                            read_fiber=kind, read_material=str(getattr(self.materials[g], "name", "")))
            else:
                read.update(read_y=y0, read_z=z0, read_fiber="plane section (no tendon prestrain)")
            mon.read = read

    def _fiber_monitor_value(self, mon, idx, strains):
        """The section reading of a `fiber` monitor: at the station nearest
        each of the region's nodes (the mean over the nodes when several),
        the plane-section strain at (y, z) or the stress of the closest
        fibre (of `fiber_kind`, when given). A Lobatto rule has a station AT
        the node; Gauss's nearest is 0.21 L in, and the result's `read_station`
        attribute says which."""
        y0, z0 = float(mon.at[0]), float(mon.at[1])
        vals = []
        stress = self.fiber_stress_rows() if mon.field == "stress" else None
        for n in idx:
            hit = self._fiber_monitor_station(n)
            if hit is None:
                continue
            e, gi = hit
            if mon.field == "strain":
                eps0, ky, kz = strains[e, gi]
                vals.append(eps0 - y0 * kz + z0 * ky)
            else:
                c = self._fiber_monitor_column(mon, e)
                if c is not None:
                    vals.append(float(stress[e, gi, c]))
        return float(np.mean(vals)) if vals else float("nan")

    def fiber_stress_rows(self):
        """Every fibre's stress at every station, (nel, n_stations,
        n_fibers), columns in the order of `fiber_y` -- from the last
        internal-force evaluation on either path."""
        if self._kernel is None or self._fstress is None:
            return self._fiber_stress_rows
        out = self._fiber_stress_rows
        n_c = self._cy.shape[1]
        for k, lay in enumerate(self._layouts):
            rows = self._rows[k]
            c_idx = self._c_order[k]
            out[np.ix_(rows, range(out.shape[1]), c_idx)] = self._fstress[rows][:, :, :len(c_idx)]
            if lay.rebar_groups:
                s_idx = np.concatenate([self.group_indices[g] for g in lay.rebar_groups])
                out[np.ix_(rows, range(out.shape[1]), s_idx)] = self._fstress[rows][:, :, n_c:n_c + len(s_idx)]
        return out

    def fiber_damage_rows(self):
        """Every fibre's damage class at every station, (nel, n_stations,
        n_fibers) int8 in the order of `fiber_y`: 0 intact, 1 cracked,
        2 crushed, 3 yielded, 4 failed (materials/base.damage_class);
        0 on a law without the notion, and on padded columns."""
        out = np.zeros((self.nel, len(self.gauss_s), self.n_fibers), dtype=np.int8)
        for k, lay in enumerate(self._layouts):
            rows = self._rows[k]
            for g in lay.conc_groups + lay.rebar_groups:
                st = self.fiber_states[g]
                if st.shape[3] == 0:
                    continue
                cls = self.materials[g].damage_class(st[rows].reshape(-1, st.shape[3]))
                if cls is None:
                    continue
                idx = self.group_indices[g]
                out[np.ix_(rows, range(out.shape[1]), idx)] = cls.reshape(len(rows), st.shape[1], st.shape[2])
        return out

    def _nodal_fields(self, actions, recover_stress=False):
        out = {"displacement": self.u.copy(),
               "rotation": self.dofs.field(self.q, ROTATION).copy()}
        wanted = (self.p.get("output") or {}).get("fields") or []
        if "fiber_stress" in wanted:
            # asked for by name only: (nel, stations, fibres) per frame is
            # what `vfife sweep` paints the section with
            out["fiber_stress"] = self.fiber_stress_rows().copy()
        if "fiber_damage" in wanted:
            out["fiber_damage"] = self.fiber_damage_rows().astype(float)
        if getattr(self, "bonds", None) is not None:
            # a split rewires members onto their spare particles, so the
            # frame has to carry the connectivity the mesh group cannot
            out["connectivity"] = self.mesh.elements.astype(float)
        if hasattr(self, "last_reaction_force"):
            out["reaction_force"] = self.last_reaction_force
            out["reaction_moment"] = self.last_reaction_moment
        if actions is not None:
            # SIGNED (core/member_actions): a fibre member's
            # section resultants at its largest station, with their sign; a
            # connector's from its end actions. `*_max_abs` is the old field
            from . import member_actions
            signed = np.array(actions, dtype=float, copy=True)
            ends_now = getattr(self, "end_actions", None)
            if ends_now is not None and not np.all(self._is_fiber):
                s_c, _ = member_actions.from_end_actions(ends_now)
                signed[~self._is_fiber] = s_c[~self._is_fiber]
            names = ("axial_force", "shear_y", "shear_z", "torsion", "bending_y", "bending_z")
            # the member's end actions less the loads inside it.
            # The cell values are the sections' resultants of the element's
            # linear curvature, which cannot kink under an axle between its
            # nodes: there a fibre deck needs the mesh refined (documented)
            ends = self.end_actions
            from .moving_load import member_loads
            ml = member_loads(self, *self._member_load_at) if hasattr(self, "_member_load_at") else None
            if ml is not None:
                ends = ends - ml
            brk = getattr(self, "breaking", None)
            # `*_max_abs`: the stations' plain magnitudes by the old rule
            # (bit for bit the pre-v1.0 field), when they belong to these actions
            max_abs = getattr(self, "actions_abs", None)
            if max_abs is None or np.shape(max_abs) != np.shape(signed) or not np.allclose(
                    np.abs(np.asarray(actions, dtype=float)), max_abs, rtol=1e-6, atol=0.0):
                max_abs = np.abs(signed)          # actions from elsewhere: their own magnitudes
            member_actions.write(out, signed, max_abs, ends,
                                 failed=None if brk is None else ~brk.member_active)
        if self.has_connectors:
            # the same names the mixed_line solver writes, so a bearing reads
            # the same whichever solver is holding it
            mask = (~self._is_fiber).astype(float)
            out["connector_flag"] = mask
            names = ("force_x", "force_y", "force_z",
                     "moment_x", "moment_y", "moment_z")
            # signed, from the end actions (`actions` holds magnitudes for the
            # capacity checks)
            ends = getattr(self, "end_actions", None)
            signed = ends[:, 6:] if ends is not None else actions
            for i, name in enumerate(names):
                if signed is not None:
                    out[f"connector_{name}"] = np.where(~self._is_fiber, signed[:, i], 0.0)
                out[f"connector_deformation_{connector.COMPONENTS[i]}"] = \
                    self.connector_deformation[:, i]
                out[f"connector_slip_{connector.COMPONENTS[i]}"] = self._conn_slip[:, i]
        if getattr(self, "breaking", None) is not None:
            out["member_failed"] = (~self.breaking.member_active).astype(float)
        if self.plan is not None:
            # a member not built (yet, or any more) has not failed (1.2)
            out["member_failed"] = ((~self.breaking.member_active) & self.member_built).astype(float)
            out["member_built"] = self.member_built.astype(float)
        if self.fracture is not None:
            # BOTH, because they answer different questions: `fac` is the
            # fraction of the section that is gone and is the interesting
            # number for most of the run (Cai et al. Fig. 8 -- a member spends
            # a long time partially fractured), while `fractured` is the
            # latched 0/1 that says whether it still carries anything.
            out["fac"] = self.fracture.fac
            out["fractured"] = self.fracture.latch.fractured.astype(float)
        return out

    def _shear_checkpoint(self, ck):
        """The shear laws' rotations and history."""
        if getattr(self, "_sl_on", None) is not None and self._sl_on.any():
            ck["shear_gamma"] = self._sl_gamma.copy()
            ck["shear_crack"] = np.where(np.isfinite(self._sl_gcr_c), self._sl_gcr_c, -1.0)
            ck["shear_max"] = self._sl_gmax_c.copy()
        return ck

    def _checkpoint_members(self, out):
        # EVERY GROUP, FROM THE LIVE STATE, WHATEVER THE SECTION. A section
        # without rebars saved `self.fiber_state` -- an alias to group 0 that
        # the kernel's consolidated arrays had long since replaced, so the
        # checkpoint held zeros and a restart or an initial_state began from
        # a virgin section; and a cover + core section saved group 0 alone
        # (found in review). `fiber_state` stays as a key for
        # readers of old results.
        out = self._shear_checkpoint(out)
        if not self.is_hybrid:
            out["fiber_state"] = self.fiber_states[0].copy()
        for i, s in enumerate(self.fiber_states):
            out[f"fiber_state_{i}"] = s.copy()
        # AND THE CONVERGED STATE: inside a staged level or a Newton solve
        # every evaluation starts again from it, so a restart there must too
        for i, s in enumerate(self.fiber_committed):
            out[f"fiber_committed_{i}"] = s.copy()
        if not self._is_fiber.all():
            out["connector_slip"] = self._conn_committed.copy()
            if self.connector_materials:
                out["connector_material_state"] = self.connector_materials.checkpoint()
        # THE FIBRES' STRESSES, as the last evaluation left them: a member
        # that has fractured or been removed is not evaluated again, and its
        # row is the stress it broke at -- what `fiber_stress` and a fibre
        # monitor go on reporting. It was not here, and a restarted run
        # reported 0 for every broken member where the uninterrupted one
        # reported up to 404 MPa
        if getattr(self, "_fstress", None) is not None:
            out["fiber_stress_kernel"] = self._fstress.copy()
        out["fiber_stress_rows"] = self._fiber_stress_rows.copy()
        if self.fracture is not None:
            # a latch never un-sets, and a restart re-latched and logged the
            # fracture again at its own time
            out["fracture_point"] = self.fracture.latch.point.copy()
            out["fracture_fractured"] = self.fracture.latch.fractured.astype(np.int8)
            out["fracture_active"] = self.fracture.active.astype(np.int8)
            out["fracture_events"] = np.array(self.fracture.events, float).reshape(-1, 2)
        return out

    def _restore_members(self, ck):
        """The fibres, the shear laws, the connectors' slip and the fracture
        latch, once the broken members are restored."""
        if "shear_gamma" in ck and getattr(self, "_sl_on", None) is not None:
            self._sl_gamma = np.asarray(ck["shear_gamma"], float).copy()
            c = np.asarray(ck["shear_crack"], float)
            self._sl_gcr_c = np.where(c < 0, np.inf, c); self._sl_gcr_t = self._sl_gcr_c.copy()
            self._sl_gmax_c = np.asarray(ck["shear_max"], float).copy(); self._sl_gmax_t = self._sl_gmax_c.copy()
            self._sl_arrays = None
        if "fiber_state_0" in ck:
            self.fiber_states = [np.asarray(ck[f"fiber_state_{i}"]).copy()
                                 for i in range(len(self.fiber_states))]
        elif "fiber_states" in ck:                    # an older layout
            self.fiber_states = [np.asarray(s).copy() for s in ck["fiber_states"]]
        else:                                          # a rebar-less section, before v1.0
            self.fiber_states[0] = np.asarray(ck["fiber_state"]).copy()
        # A restored state IS a converged state; trial evaluations must start there.
        self._relink_kernel_states()
        if not self.is_hybrid:
            self.fiber_state = self.fiber_states[0]
        self.fiber_committed = [s.copy() for s in self.fiber_states]
        if "fiber_committed_0" in ck:
            self._restore_committed([np.asarray(ck[f"fiber_committed_{i}"])
                                     for i in range(len(self.fiber_states))])
        if "connector_slip" in ck:
            self._conn_committed = np.asarray(ck["connector_slip"], float).copy()
            self._conn_slip = self._conn_committed.copy()
        if self.connector_materials and "connector_material_state" in ck:
            self.connector_materials.restore(ck["connector_material_state"])
        # (a checkpoint from before v1.12 has neither: the broken members'
        # stresses then read 0, as they did)
        for key, name in (("fiber_stress_kernel", "_fstress"), ("fiber_stress_rows", "_fiber_stress_rows")):
            have = getattr(self, name, None)
            if key in ck and have is not None and np.shape(ck[key]) == have.shape:
                have[...] = np.asarray(ck[key], float)
        if self.fracture is not None and "fracture_point" in ck:
            L = self.fracture.latch
            L.point[...] = np.asarray(ck["fracture_point"], float)
            L.fractured[...] = np.asarray(ck["fracture_fractured"]).astype(bool)
            L.n_fractured = int(L.fractured.sum())
            self.fracture.active[...] = np.asarray(ck["fracture_active"]).astype(bool)
            self.fracture.events = [(float(t), int(e)) for t, e in np.asarray(ck["fracture_events"]).reshape(-1, 2)]
            gone = np.flatnonzero(L.fractured)
            if len(gone):
                self._active[gone] = 0
                self._refresh_kernel_mask()

    def _restore_committed(self, committed):
        """The converged fibre state, where it differs from the live one."""
        live = [s.copy() for s in self.fiber_states]
        for g, c in enumerate(committed):
            if self.fiber_states[g].size:
                self.fiber_states[g][...] = c
        self.fiber_committed = [s.copy() for s in self.fiber_states]
        self._sync_committed()
        for g, s in enumerate(live):
            if self.fiber_states[g].size:
                self.fiber_states[g][...] = s

    def summary(self):
        def steel(lay):
            # concrete fibres, then the bars and the tendons by name: the
            # line read "98 fibers + 2 rebars" for 96 + 1 bar + 1 tendon
            # (found in review)
            nt = len(lay.tendons); nb = lay.n_s - nt
            return ((f" + {nb} rebar{'s' * (nb != 1)}" if nb else "")
                    + (f" + {nt} tendon{'s' * (nt != 1)}" if nt else ""))
        extra = steel(self._layouts[0])

        def theory(rows):
            # THE MEMBER THEORY OF THESE ELEMENTS, not of the model: a shear
            # law on the piers labelled the deck too
            rows = np.asarray(rows, dtype=int)
            sl = getattr(self, "_sl_on", None)
            if sl is not None and np.any(np.asarray(sl)[rows]):
                return "Euler-Bernoulli + cracking shear law in series"
            if np.any(np.asarray(self._GA)[rows] > 0.0):
                return "Timoshenko (shear_stiffness)"
            return "Euler-Bernoulli"
        theories = [theory(r) for r in self._rows]
        mixed_theory = len(set(theories)) > 1
        if len(self._layouts) > 1:
            mats = "; ".join(
                f"section {k}: " + ", ".join(dict.fromkeys(self.materials[g].name for g in lay.conc_groups + lay.rebar_groups))
                for k, lay in enumerate(self._layouts))
        else:
            mats = ", ".join(m.name for m in self.materials) if self.is_hybrid else self.material.name
        integ = f"{len(self.gauss_s)}pt {'gauss' if len(self.gauss_s) == 2 else 'lobatto'}"
        frac = ""
        if self.fracture is not None:
            # naming the laws that CANNOT fail is the point of the line: a
            # section whose concrete is silent reaches FAC = 1 only when every
            # bar has ruptured, and that is a different model from the one an
            # analyst who wrote `fracture:` probably has in mind
            silent = (f", no criterion in {', '.join(self._fracture_silent)}"
                      if self._fracture_silent else "")
            action = ("deactivate + split (2 spare particles per element "
                      f"reserved, {self.bonds.n_components()} component(s) now)"
                      if self.bonds is not None else "deactivate, mass kept")
            of = (f"{self.n_fibers} fibres" if self._tracked_valid is None
                  else "each member's own fibres")
            frac = (f"\n  fracture      fibre latch, threshold "
                    f"{self.fracture.latch.threshold:g} of {of}"
                    f"{silent}; {action}")
        if self._section_of is None:
            size = (self._section_sizes[0] if getattr(self, "_section_sizes", None) else None)
            what = (f"girder {size[0]:.3g} x {size[1]:.3g} m (patches)" if size else
                    f"{self._section_specs[0][5]} "
                    f"{float(np.ptp(self.fiber_y[0])):.3g} x {float(np.ptp(self.fiber_z[0])):.3g} m")
            section = (f"  section       {what}, "
                       f"{self._layouts[0].n_c} fibers{extra}, A={float(self.total_area[0]):.3e} "
                       f"Iy={float(self.Iy[0]):.3e} Iz={float(self.Iz[0]):.3e}\n")
        else:
            lines = []
            for k, (y, z, a, _w, _h, stype) in enumerate(self._section_specs):
                lay = self._layouts[k]
                n_k = len(self._rows[k]); e0 = int(self._rows[k][0])
                bars = steel(lay)
                core = (f", cover/core {len(self.group_indices[lay.conc_groups[0]])}/"
                        f"{len(self.group_indices[lay.conc_groups[1]])}" if lay.confined else "")
                lines.append(f"  section {k:<5d} {stype} {float(np.ptp(y)):.3g} x {float(np.ptp(z)):.3g} m, "
                             f"{len(y)} fibers{bars}{core} on {n_k} element(s), A={float(self.total_area[e0]):.3e} "
                             f"Iy={float(self.Iy[e0]):.3e} Iz={float(self.Iz[e0]):.3e}"
                             + (f", {theories[k]}" if mixed_theory else ""))
            section = "\n".join(lines) + f"\n  (fibre arrays padded to {self.n_fibers})\n"
        if len(self.tendons) > 12:
            # A real girder has a hundred ducts and a line each is a wall of
            # text: group them by area and report the totals.
            groups = {}
            for t in self.tendons:
                g = groups.setdefault(round(float(t["area"]), 9), [0, [], []])
                g[0] += 1; g[1].append(float(t["sigma"].min())); g[2].append(float(t["sigma"].max()))
            P = sum(float(np.mean(t["sigma"])) * t["area"] for t in self.tendons)
            section += (f"  tendon        {len(self.tendons)} ducts, "
                        + ", ".join(f"{n} x {a * 1e4:.1f} cm2 at {min(lo) / 1e6:.0f}..{max(hi) / 1e6:.0f} MPa"
                                    for a, (n, lo, hi) in sorted(groups.items()))
                        + f"; total prestress {P / 1e6:.1f} MN\n")
        else:
            for t in self.tendons:
                sig = t["sigma"]; force = sig * t["area"]
                section += (f"  tendon        section {t['section']} slot {t['slot']}: A={t['area']:.3e} m2, "
                            f"y {float(t['y'].min()):+.3f}..{float(t['y'].max()):+.3f} m, "
                            f"z {float(t['z'].min()):+.3f}..{float(t['z'].max()):+.3f} m, "
                            f"sigma {float(sig.min())/1e6:.0f}..{float(sig.max())/1e6:.0f} MPa, "
                            f"P {float(force.min())/1e3:.0f}..{float(force.max())/1e3:.0f} kN\n")
        theory = "theory per section (below)" if mixed_theory else theories[0]
        if mats.count(",") > 10:
            names = {}
            for m in self.materials:
                names[m.name] = names.get(m.name, 0) + 1
            mats = ", ".join(f"{k} x{v}" if v > 1 else k for k, v in names.items())
        # the connectors (bearings) and what the model weighs: a bridge's
        # four bearings were not mentioned anywhere, and a total-mass line is
        # the first check of a model's weight
        conn = ""
        for k, props in enumerate(getattr(self, "_connector_props", ()) or ()):
            n_k = int(np.sum(self._connector_of == k))
            laws = props.get("laws") or {}
            K = np.diag(connector.matrix(props.get("stiffness"))) if props.get("stiffness") is not None \
                else np.zeros(6)
            comps = [f"{c} {laws[c]['type']}" if c in laws else f"{c} {K[i]:.3g}"
                     for i, c in enumerate(connector.COMPONENTS) if c in laws or K[i] != 0.0]
            conn += (f"  connector {k:<3d} region {props.get('region')}: {n_k} element(s), "
                     + (", ".join(comps) or "no stiffness") + "\n")
        mass_line = ""
        if hasattr(self, "physical_mass"):
            m_f = self.dofs.field(self.physical_mass, TRANSLATION)
            m_w = added_mass.weighed(m_f, getattr(self, "weightless_mass", None))
            # in the case's own units: it said kg and N at
            # g = 9.80665 in a mm / N / MPa case too
            from .units import to_case, mass_unit, complete
            units = complete(self.p.case.units)[0]
            g = to_case(9.80665, "acceleration", units) or 9.80665
            # t for N / mm and kN / m, kg for N / m (1.0 re-review: it said kg for kN / m)
            m_u = mass_unit(units) or "mass units"
            f_u = units.get("force", "N")
            mass_line = (f"  mass          {float(m_f[:, 0].sum()):.6g} {m_u}, of it members "
                         f"{float((self.mass_per_length * self.length).sum()):.6g} {m_u} (each fibre at its "
                         f"material's density); weight {float(m_w[:, 2].sum()) * g:.6g} {f_u} at g = {g:.6g}\n")
        return (f"  mesh          {self.nn} nodes, {self.nel} Line2 elements"
                + (f" ({int((~self._is_fiber).sum())} connectors)" if (~self._is_fiber).any() else "")
                + "\n"
                f"  element       beam2_fiber / fiber {theory}, 6 DOF/node\n"
                + section + conn + mass_line +
                f"  fiber         {mats}; {self.n_fibers} x {len(self.gauss_s)} points ({integ})\n"
                f"  mode          {self._mode_text()}"
                + mass_scaling.summary_line(getattr(self, "mass_scaling", None))
                + frac + member_axes.summary_lines(self))        # v1.0
