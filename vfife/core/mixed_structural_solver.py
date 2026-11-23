"""Mixed co-rotational truss/frame assembly on one Gmsh Line2 mesh."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .beam_solver import BeamSolver
from .member_failure import Breaking
from . import mass_scaling
from . import member_axes
from . import added_mass
from .dof import SparseAssembler, TRANSLATION, repeated_rotation
from ..elements import connector
from . import families
from .families import COLLAPSED


class MixedStructuralSolver(BeamSolver):
    """One six-DOF particle system containing axial bars and frame members."""

    def _group_materials(self, specs):
        """One material object per member group.

        A group without a `material` key gets the model's -- the very object,
        not a copy -- so a single-material model reads exactly the numbers it
        always did. A group with one gets its own, built by the same factory
        the continuum's material blocks go through, so a line material and a
        solid material are made the same way.

        Only `linear_elastic` is allowed, because that is what `BeamSolver`
        already demands of the model's material: a member group is not a door
        around that limit, and a case that tries is told so with the group's
        name rather than an AttributeError three frames down.
        """
        from ..materials.factory import create_material
        out, made = [], {}
        for spec in specs:
            props = spec.get("material")
            if props is None:
                out.append(self.material); continue
            # ONE OBJECT PER DISTINCT SPEC. Thirty-four stay groups naming the
            # same steel are one material, and `summary` counts objects: the
            # Xihoumen model printed "34 linear_elastic" for eight.
            key = repr(sorted(dict(props).items()))
            if key in made:
                out.append(made[key]); continue
            m = create_material(dict(props))
            made[key] = m
            if m.name != "linear_elastic":
                raise ValueError(
                    f"member group {spec['region']!r} names material {m.name!r}; line "
                    "members currently support linear_elastic only (the same limit the "
                    "model's own material is held to)")
            out.append(m)
        return out

    def _assemble_reference(self):
        nel = self.mesh.n_elem
        owner = np.full(nel, -1, dtype=int)
        specs = self.p.get("members") or []
        assigned = []
        for g, spec in enumerate(specs):
            self._region(spec["region"])          # so an unknown name still says so
            # The recorded physical group where the mesh has one; the old
            # every-node-inside rule otherwise. A building needs the first:
            # every column's two nodes are beam nodes as well, so the node
            # rule hands every column to the beams too.
            ids = self.mesh.region_elements(spec["region"], "truss2", self.mesh.elements)
            if ids is None or not len(ids):
                raise ValueError(f"member group {spec['region']!r} contains no complete Line2 elements")
            overlap = ids[owner[ids] >= 0]
            if len(overlap):
                raise ValueError(f"Line2 elements {overlap[:8].tolist()} belong to multiple member groups")
            owner[ids] = g; assigned.append(ids)
        missing = np.flatnonzero(owner < 0)
        self.member_owner = owner           # the group of each element
        if len(missing):
            raise ValueError(f"mixed_line has {len(missing)} unassigned Line2 elements; first ids: {missing[:8].tolist()}")

        group_materials = self._group_materials(specs)
        #: the material of every ELEMENT, read by `_material_of`
        self.member_material = [group_materials[owner[e]] for e in range(nel)]
        self.member_E = np.array([float(m.E) for m in self.member_material])
        self.member_rho = np.array([float(getattr(m, "rho", 0.0)) for m in self.member_material])
        #: True when every member reads the same material object -- what every
        #: case before v0.4 was, and what `summary` reports as one line
        self.one_material = all(m is self.material for m in self.member_material)

        assembly = SparseAssembler(self.ndof)
        self.T, self.R0, self.k_local, self.length, self._kg_ref = [], [], [], [], []
        self.member_kind, self.member_props = [], []
        for e, conn in enumerate(self.mesh.elements):
            props = specs[owner[e]]; kind = props["type"]
            reference = props.get("orientation")
            if kind in {"truss2", "connector2"} and reference is None:
                axis = self.mesh.nodes[conn[1]]-self.mesh.nodes[conn[0]]
                basis = np.eye(3); reference = basis[np.argmin(np.abs(basis@(axis/np.linalg.norm(axis))))]
            R, L = self._frame(self.mesh.nodes[conn[0]], self.mesh.nodes[conn[1]],
                               reference)
            if kind in self.BEAM_TYPES:
                kl = self._local_stiffness(L, props, material=group_materials[owner[e]])
                T = repeated_rotation(R); kg = T.T@kl@T
            elif kind == "connector2":
                kl = connector.reference_stiffness(self.mesh.nodes[conn], R, props)
                T = repeated_rotation(R); kg = kl
            else:
                kl = np.zeros((12, 12)); ka = self.member_E[e]*float(props["area"])/L
                kl[0,0] = kl[6,6] = ka; kl[0,6] = kl[6,0] = -ka
                T = repeated_rotation(R)
                if self._is_cable(props):
                    # A CABLE'S TRANSVERSE STIFFNESS IS ITS TENSION OVER ITS
                    # LENGTH, which at the reference state may be anything
                    # from zero (a slack chord) to E A eps / L. The reference
                    # K exists here to size the fictitious masses, and a mass
                    # sized on a zero row is 1e-20 -- the first transverse
                    # force blows up. So a cable member puts its axial
                    # stiffness on all three translations of both nodes: an
                    # upper bound on any stiffness the member can develop,
                    # hence stable, at the price of the transverse (sagging)
                    # motion being over-massed by 1/eps and converging slower.
                    # Not applied to plain trusses, whose
                    # masses stay what they were to the bit.
                    kg = np.zeros((12, 12))
                    for a_, b_ in ((0, 6), (6, 0)):
                        kg[a_:a_+3, a_:a_+3] += ka*np.eye(3); kg[a_:a_+3, b_:b_+3] -= ka*np.eye(3)
                else:
                    kg = T.T@kl@T
            dof = self.dofs.element_dofs(conn)
            assembly.add_matrix(dof, kg)
            self.T.append(T); self.R0.append(R); self.k_local.append(kl); self.length.append(L)
            self._kg_ref.append(kg)
            self.member_kind.append(kind); self.member_props.append(props)
        self.K = assembly.csr()
        self.T, self.R0, self.k_local, self.length = map(np.asarray,
            (self.T,self.R0,self.k_local,self.length))
        self.member_kind = np.asarray(self.member_kind)
        self._setup_cables(specs, owner)
        self._setup_failure(specs, owner)

    @staticmethod
    def _is_cable(props):
        return (props.get("type") == "truss2" and (props.get("prestress") is not None
                or props.get("unstressed_length") is not None
                or props.get("unstressed_lengths") is not None or props.get("tension_only")
                or props.get("jack") is not None))

    def _setup_cables(self, specs, owner):
        """A truss member's UNSTRESSED length, and whether it can go slack.

        Every truss member has an unstressed length L_u, its mesh length by
        default. `prestress` (the effective stress the bar carries in the
        mesh geometry) sets L_u = L_0 / (1 + sigma / E); `unstressed_length`
        states a group's total L_u and shares it over the group's elements in
        proportion to their mesh lengths, which is one cable stated by the
        length it was cut to. The axial strain is then (L - L_u) / L_u and
        the force E A times it, so a cable meshed on its design geometry
        carries its design tension at u = 0 -- the dead-load state of a
        cable-stayed or suspension bridge is a statement, not a search.
        `tension_only` clips the force at zero: a slack cable carries nothing.
        """
        self.member_Lu = np.array(self.length, dtype=float)
        self.member_tension_only = np.zeros(self.nel, dtype=bool)
        #: the force each bar is jacked to in its step (0: not jacked), and
        #: whether that step is behind it (core/analysis_steps.jack_*)
        self._jack_T = np.zeros(self.nel)
        self._jack_done = np.zeros(self.nel, dtype=bool)
        self._jack_p = 0.0
        for g, spec in enumerate(specs):
            ids = np.flatnonzero(owner == g)
            if spec.get("type") != "truss2":
                continue
            if spec.get("prestress") is not None:
                # the GROUP's E: `prestress` is a stress, and sigma/E is the
                # strain of this cable's own steel
                E = float(self.member_E[ids[0]]) if len(ids) else float(self.material.E)
                self.member_Lu[ids] = self.length[ids] / (1.0 + float(spec["prestress"]) / E)
            if spec.get("unstressed_length") is not None:
                total = float(np.sum(self.length[ids]))
                self.member_Lu[ids] = self.length[ids] * (float(spec["unstressed_length"]) / total)
            if spec.get("unstressed_lengths") is not None:
                lu = np.asarray(spec["unstressed_lengths"], dtype=float)
                if lu.shape != (len(ids),):
                    raise ValueError(
                        f"member group {spec['region']!r}: unstressed_lengths has {lu.size} "
                        f"values for {len(ids)} elements")
                self.member_Lu[ids] = lu             # ids ascend: the group's element order
            if spec.get("tension_only"):
                self.member_tension_only[ids] = True
            if spec.get("jack") is not None:
                self._jack_T[ids] = float(spec["jack"])
        self.has_cables = bool(np.any(self.member_Lu != self.length) or np.any(self.member_tension_only)
                               or np.any(self._jack_T > 0.0))

    # ------------------------------------------------------------------ #
    #  Elastic-brittle breaking and scheduled removal
    # ------------------------------------------------------------------ #
    def _setup_failure(self, specs, owner):
        """Capacities of every member that can break, and the removal times
        (core/member_failure.py holds the table; this solver's part is to
        rebuild its batches without the members that have gone).

        A broken member is REMOVED: it carries nothing from then on and its
        mass stays on its nodes. Removing it is what disconnects: a deck
        segment whose last member to the rest has gone is a free chain of
        particles and falls, with no splitting needed.
        """
        self.breaking = Breaking(self.nel, self.mesh, self.p.case.removals, self.log,
                                 split_default=self.SPLIT_DEFAULT)
        self.member_active = self.breaking.member_active
        self.capacity_ratio = self.breaking.capacity_ratio
        self.failure_events = self.breaking.failure_events
        self._reload_step = False
        self.splits = []
        self._fragment_parent = {}
        self._appendages = []
        # the slipped deformation of every connector component:
        # `_conn_slip` is what the last evaluation arrived at, `_conn_committed`
        # what the laws start from -- the last equilibrium in a static run,
        # the last step in a dynamic one
        self._conn_slip = np.zeros((self.nel, 6))
        self._conn_committed = np.zeros((self.nel, 6))
        #: (nel, 6) the deformation each connector was born with, once an
        #: analysis step has built one; None before
        self._conn_birth = None
        for g, spec in enumerate(specs):
            self.breaking.add(np.flatnonzero(owner == g), spec.get("failure"), spec, spec["type"])
        # a breakable beam twisted a quarter turn is torn: its G J / L
        beams_ = np.isin(self.member_kind, tuple(self.BEAM_TYPES))
        if beams_.any():
            self.breaking.twist_caps(np.where(beams_, np.asarray([np.asarray(k)[3, 3] if np.shape(k) == (12, 12) else 0.0
                                                                  for k in self.k_local]), 0.0))
        # PLASTIC HINGES: the ends of the members whose failure has a
        # rotation_capacity. Their plastic rotations ride in the eigen-
        # deformation slot a temperature uses (f = k (d - d_eigen)), so both
        # force paths -- the NumPy law and the numba kernel -- see them
        hinged = self.breaking.hinged & np.isin(self.member_kind, tuple(self.BEAM_TYPES))
        self._hinged = np.flatnonzero(hinged)
        if len(self._hinged):
            self._plastic_d = np.zeros((self.nel, 12))
            self._hinge_rotation = np.zeros((self.nel, 2))       # accumulated, per end
            cap = self.breaking._beam_cap
            self._hinge_cap = np.ascontiguousarray(cap[:, :3])      # N_p, M_y,p, M_z,p
            self.log(f"  hinges       {len(self._hinged)} member(s) with plastic hinges at both ends "
                     f"(rotation capacity {', '.join(sorted({f'{v:g}' for v in self.breaking.rotation_capacity[hinged]}))} rad)")
        else:
            self._plastic_d = None

    def _setup_breaking(self):
        """Done in `_setup_failure`, from the member groups."""

    def _breakable_specs(self):
        specs = self.p.get("members") or []
        groups = []
        for spec in specs:
            if spec.get("failure"):
                ids = self.mesh.region_elements(spec["region"], "truss2", self.mesh.elements)
                if ids is not None:
                    groups.append((np.asarray(ids, int), spec["failure"]))
        return groups, list(self.p.case.removals)

    def _element_mass_share(self, e):
        """The (6,) mass one of member e's two particles carries for it: the
        translational half of rho A L for a bar or a beam, the beam's rotary
        share on top; a connector has none."""
        out = np.zeros(6)
        if self.member_kind[e] == "connector2":
            return out
        rho = self.member_rho[e]; L = float(self.length[e]); props = self.member_props[e]
        A = float(props["area"])
        out[:3] = rho * A * L / 2.0
        # the true rotary inertia in global axes, as beam2's
        if self.member_kind[e] in self.BEAM_TYPES:
            Iy, Iz = float(props["iy"]), float(props["iz"])
            if self.member_kind[e] == "beam2":
                local = np.array([rho * (Iy + Iz) * L / 2.0] + [rho * A * L ** 3 / 24.0] * 2)
            else:
                local = rho * L * np.array([Iy + Iz, Iy, Iz]) / 2.0
            out[3:] = mass_scaling.rotary_diagonal(self.R0[e], local)
        return out

    def _deactivate(self, ids):
        """The batches are rebuilt from the active members on the next call."""
        if len(ids):
            self.rebatch()
            self._drop_from_reference_stiffness(ids)

    def _checkpoint(self):
        ck = super()._checkpoint()
        ck.update(self._checkpoint_member_state(breaking=False))
        return ck

    def _checkpoint_member_state(self, breaking=True):
        """What a member carries beyond the shared particles: slip, hinges,
        and (for a solver nested in another) the broken members."""
        ck = {"connector_slip": self._conn_committed.copy()}
        if getattr(self, "_plastic_d", None) is not None:
            ck["plastic_rotation"] = self._plastic_d.copy()
            ck["hinge_rotation"] = self._hinge_rotation.copy()
        if breaking and getattr(self, "breaking", None) is not None:
            ck = self.breaking.checkpoint(ck)
        return ck

    def _restore_member_state(self, ck):
        if getattr(self, "breaking", None) is not None and "member_active" in ck:
            self._deactivate(self.breaking.restore(ck))
        self._restore_extras(ck)

    def _restore(self, ck):
        super()._restore(ck)                      # a broken member stays broken
        self._restore_extras(ck)

    def _restore_extras(self, ck):
        if "connector_slip" in ck:               # and a slid bearing stays slid
            self._conn_committed = np.asarray(ck["connector_slip"], float).copy()
            self._conn_slip = self._conn_committed.copy()
        if getattr(self, "_plastic_d", None) is not None and "plastic_rotation" in ck:
            self._plastic_d = np.asarray(ck["plastic_rotation"], float).copy()
            self._hinge_rotation = np.asarray(ck["hinge_rotation"], float).copy()
            self.breaking.hinge_damage[...] = self._hinge_rotation / np.where(
                np.isfinite(self.breaking.rotation_capacity), self.breaking.rotation_capacity, np.inf)[:, None]
        self.rebatch()

    def _eigen_d(self):
        """The deformation the beams' law subtracts before the stiffness sees
        it: the thermal (and creep, shrinkage) one, plus the plastic hinge
        rotations."""
        th = getattr(self, "_thermal_d", None)
        pl = getattr(self, "_plastic_d", None)
        if pl is None:
            return th
        return pl if th is None else th + pl

    def _hinge_update(self, end_actions, actions, commit=True):
        """RETURN TO THE YIELD SURFACE, end by end. After the step's forces,
        an end of a hinged member whose |N|/N_p + |M_y|/M_y,p + |M_z|/M_z,p
        exceeds 1 has its two moments scaled back onto the surface (N is not
        plastic here: it breaks the end at N_p) and the difference becomes
        plastic rotation through the end's own rotational stiffness,
        dtheta_p = dM / k_ii. The coupling to the other end (2EI/L) is taken
        up on the next evaluation, as is the global force: in an explicit run
        the plastic increment of one step is small, and the moment overshoots
        the surface by one step's worth at most. The end actions the failure
        table reads are corrected now. Unloading is elastic: nothing changes
        below the surface. Perfectly plastic, no hardening.

        `commit` False is a probe (`_record`, a convergence measure): the
        end actions are returned to the surface as the step's would be, but
        no rotation is accumulated -- it was, at every output frame, so
        `output.every` changed when a hinge broke."""
        ids = self._hinged[self.member_active[self._hinged]]
        if not len(ids):
            return
        try:
            from ..kernels.numba_step import hinge_update as _hu
        except ImportError:                                   # pragma: no cover
            _hu = None
        if _hu is not None and getattr(self, "_beam_ids", None) is not None:
            # the compiled version (1.6), which the compiled step runs too:
            # one arithmetic for both paths, to the bit. `k_local` rows are
            # the batch's for the beam members
            where = {int(e): k for k, e in enumerate(self._beam_batch())}
            if all(int(e) in where for e in ids):
                h_el = np.ascontiguousarray(ids, dtype=np.int64)
                h_batch = np.ascontiguousarray([where[int(e)] for e in ids], dtype=np.int64)
                nh = len(ids)
                d_out = np.zeros((nh, 12)); rot_out = np.zeros((nh, 2)); dmg_out = np.zeros((nh, 2))
                _hu(h_el, h_batch, end_actions, actions, self.frame_family.kc, self._hinge_cap,
                    self.breaking.end_broken, self.member_active, self._plastic_d, self._hinge_rotation,
                    self.breaking.rotation_capacity, self.breaking.hinge_damage, d_out, rot_out, dmg_out,
                    bool(commit))
                return
        fl = end_actions[ids]
        k = self.k_local[ids]
        cap = self._hinge_cap[ids]
        broken = self.breaking.end_broken[ids]
        d = np.zeros((len(ids), 12))
        for a, (iN, iy, iz) in enumerate(((0, 4, 5), (6, 10, 11))):
            n = np.abs(fl[:, iN]) / cap[:, 0]
            m = np.abs(fl[:, iy]) / cap[:, 1] + np.abs(fl[:, iz]) / cap[:, 2]
            room = np.clip(1.0 - n, 0.0, 1.0)
            over = (m > room) & ~broken[:, a]
            if not over.any():
                continue
            s = np.where(over, room / np.maximum(m, 1e-300), 1.0)
            d[:, iy] = fl[:, iy] * (1.0 - s) / k[:, iy, iy]
            d[:, iz] = fl[:, iz] * (1.0 - s) / k[:, iz, iz]
            if commit:
                self._hinge_rotation[ids, a] += np.abs(d[:, iy]) + np.abs(d[:, iz])
        if not d.any():
            return
        end_actions[ids] = fl = fl - np.einsum("eij,ej->ei", k, d)
        actions[ids] = np.maximum(np.abs(fl[:, :6]), np.abs(fl[:, 6:]))
        if not commit:
            return
        self._plastic_d[ids] += d
        cap_u = self.breaking.rotation_capacity[ids]
        self.breaking.hinge_damage[ids] = self._hinge_rotation[ids] / cap_u[:, None]

    def _commit_members(self):
        """The connector laws' history the next evaluation starts from: called
        by the solver loops at equilibrium (relaxation) or every step (dynamic)."""
        self._conn_committed[...] = self._conn_slip

    def _truss_force(self, e, L):
        """Axial force of truss member e at current length L: E A times the
        strain from its unstressed length, clipped at zero when it is a
        tension-only cable."""
        Lu = self.member_Lu[e]
        dth = getattr(self, "_thermal_d", None)
        stretch = L - Lu - (dth[e, 6] if dth is not None else 0.0)
        N = self.member_E[e] * float(self.member_props[e]["area"]) * stretch / Lu
        kf = getattr(self, "_k_factor", 1.0)
        if np.ndim(kf):
            # creep's 1/(1 + phi) and ageing, as the beams get them through
            # k_local
            N *= kf[e]
        if self.member_tension_only[e] and N < 0.0:
            N = 0.0
        return N

    def _setup_mass_and_time(self):
        # Same particle mass contract as beam2, with each member's own area.
        sol = self.p.solver; self.mode = sol.mode
        m = np.zeros((self.nn, 6))
        true_rot = np.zeros((self.nn, 3))
        for e, (conn, L) in enumerate(zip(self.mesh.elements, self.length)):
            if self.member_kind[e] == "connector2":
                continue
            # the step from the old rule below; motion and modes from the
            # true inertia
            true_rot[conn] += self._element_mass_share(e)[3:]
            rho = self.member_rho[e]
            A = float(self.member_props[e]["area"]); mt = rho*A*L/2
            m[conn,:3] += mt
            if self.member_kind[e] == "beam2":
                m[conn,3:] += rho*A*L**3/24
            elif self.member_kind[e] == "beam2_timoshenko":
                props = self.member_props[e]
                m[conn,3:] += rho*L*np.array(
                    [float(props["j"]), float(props["iy"]), float(props["iz"])]
                )/2
        # nodal and hydrodynamic masses; a connector has no
        # section for water to act on
        kinds = np.asarray(self.member_kind)
        add, rot, self.weightless_mass = added_mass.table(
            self.p, self._region, self.mesh.nodes[:self.nn], self.mesh.elements[:, :2],
            member_ok=kinds != "connector2")
        self.line_mass_per_end = added_mass.member_line_mass(
            self.p.get("loads"), self._region, self.mesh.nodes[:self.nn], self.mesh.elements[:, :2],
            kinds != "connector2", self.p.case.units)
        m[:, :3] += add; m[:, 3:] += rot; self.additional_mass = add
        true_rot += rot
        self.physical_mass=m.reshape(-1); row=np.asarray(abs(self.K).sum(axis=1)).ravel()
        # selective mass scaling, before anything reads the mass
        self.mass_scaling = mass_scaling.scale(self.physical_mass, row, ~self._held_dofs(),
                                               sol.mass_scaling,
                                               sol.dt_safety)
        step_mass = self.physical_mass.copy()
        self._rotary_true = true_rot; self._step_mass = step_mass
        if self.mode=="relaxation":
            self.mass=np.maximum(row/float(sol.relaxation_safety)**2,1e-20)
            self.dt,self.n_steps,self.total_time=1.,int(sol.n_steps),float(sol.n_steps)
        else:
            self.mass=np.maximum(self.physical_mass,1e-20)
            # the FREE dofs with stiffness: a damper alone on the one free
            # dof of a model whose stiff components are all held is a pure-
            # dashpot model too (it crashed on an empty minimum; review of 1.10)
            active=(row>1e-20) & ~self._held_dofs()
            omega_max=0.0
            if np.any(active):
                if sol.target_dt:
                    # read only by a pure-dashpot model, which has no stiffness
                    # to take a step from; ignored here
                    raise ValueError("solver.target_dt sets the step of a model with no stiffness "
                                     "(pure dashpots); this model takes its own stable step -- "
                                     "use solver.mass_scaling to raise it")
                root=float(np.sqrt(np.min(self.mass[active]/row[active])))
                self.dt=float(sol.dt_safety*root); omega_max=1.0/root
            elif sol.target_dt:
                self.dt=float(sol.target_dt)
            else:
                raise ValueError("a pure-dashpot connector model requires solver.target_dt")
            # Explicit connector dashpots also impose dt < 2m/c.  Use the
            # assembled initial damping matrix as a conservative row estimate.
            damping_rows = connector.dashpot_rows(
                np.flatnonzero(np.asarray(self.member_kind) == "connector2"), self.mesh.elements,
                self.mesh.nodes, self.R0, lambda e: self.member_props[e], self.dofs, self.ndof)
            # FREE DOFS ONLY, as the stiffness bound above: a held dof has the
            # 1e-20 mass floor, and a dashpot to a clamped anchor node -- the
            # ground connection docs/connector2.md recommends -- set dt to
            # 1.9e-24 s and the run to 5e23 steps
            damp_active=(damping_rows>1e-20) & ~self._held_dofs()
            if np.any(damp_active):
                self.dt=min(self.dt, float(2*sol.dt_safety*np.min(
                    self.mass[damp_active]/damping_rows[damp_active])))
            self.n_steps=max(1,int(round(sol.total_time/self.dt))); self.total_time=float(sol.total_time)
        damp=sol.damping; self.damp_kind=damp.type or ("kinetic" if self.mode=="relaxation" else "viscous")
        self.zeta,self.alpha,self.beta=float(damp.coeff),float(damp.alpha),float(damp.beta)
        # The connector dashpots above were already given their explicit
        # bound (dt < 2m/c); the GLOBAL Rayleigh term never was, and it is the
        # same species of thing.
        dash_floor = None
        if self.mode != "relaxation" and np.any(damp_active):
            dash_floor = np.where(damp_active, damping_rows * self.dt / (2 * float(sol.dt_safety)), 0.0)
        self._shrink_step_for_damping(omega_max if self.mode!="relaxation" else 0.0)
        # the step is set: the true rotary inertia, floored where the old
        # step needs it
        self.physical_mass, self.rotary_floored = mass_scaling.true_rotary(
            step_mass, true_rot, row, ~self._held_dofs(), dash_floor)
        if self.mode != "relaxation":
            self.mass = np.maximum(self.physical_mass, 1e-20)
        self.tolerance=float(sol.tolerance); self.kinetic_tolerance=float(sol.kinetic_tolerance)

    # ------------------------------------------------------------------ #
    #  Internal force
    # ------------------------------------------------------------------ #
    #  The frame members are evaluated for ALL elements at once. The first
    #  version of this loop did one element at a time with a dozen small numpy
    #  calls each (cross, kron, moveaxis, ...), and nobody noticed because
    #  every beam case had one or two members. cases/case_16, Lee's frame, has
    #  forty, and ran at 120 steps/s -- 8 ms a step, on a problem whose
    #  relaxation needs 10^4-10^5 steps per load level. Trusses
    #  and connectors are few and stay in the loop below.

    def kernel_members(self):
        """The members' arrays for the compiled step: the frame members, the
        bars and the compiled connectors (core/protocol.KernelMembers,
        v1.18)."""
        from .protocol import KernelMembers
        dth = self._kernel_dth()
        self._beam_batch(); self._truss_batch()
        f = self.frame_family
        return KernelMembers(
            f=(np.ascontiguousarray(f.ids, dtype=np.int64), f.conn64, f.R0c,
               f.kc, f.L0c, f.dofs64, dth),
            collapse=COLLAPSED,
            t_ids=np.ascontiguousarray(self._truss_ids, dtype=np.int64),
            t_conn=np.ascontiguousarray(self.bar_family.conn, dtype=np.int64),
            t_EA=np.ascontiguousarray(self.bar_family.EA, dtype=float),
            threads=int(getattr(self, "_threads", 1) or 1), connectors=self._connector_kernel())

    # ------------------------------------------------------------------ #
    #  The member families (core/families)
    # ------------------------------------------------------------------ #
    #: the families, built on first use and again after `rebatch`
    frame_family = None
    bar_family = None
    connector_family = None

    def rebatch(self, frames=True, axial=True):
        """The members changed (a step built or removed some, a split moved
        an end, an age changed the frames' stiffness): the families named
        are gathered again at their next use."""
        super().rebatch(frames=frames, axial=axial)
        if frames:
            self.frame_family = None; self._beam_ids = None
        if axial:
            self.bar_family = None; self._truss_ids = None

    def _beam_batch(self):
        """Element indices of the frame members (their family gathered first
        when the members changed)."""
        if getattr(self, "_beam_ids", None) is None or self.frame_family is None:
            f = self.frame_family = families.FrameMembers(self)
            self._beam_ids, self._other_ids = f.ids, f.other_ids
        return self._beam_ids

    def _beam_forces_batched(self, out, actions, end_actions):
        self._beam_batch()
        self.frame_family.forces(self, out, actions, end_actions)

    def _current_rotations(self):
        """R for every frame member in the CURRENT configuration -- the same
        frame the force evaluation uses, without the forces."""
        self._beam_batch()
        return self.frame_family.rotations(self)

    def _turned_blocks(self):
        """The mixed solver's turned members (`BeamFamily._refresh_fictitious_mass`):
        the frame members turned into their current frame, the bars along
        their current axis (a cable's along every axis, `_assemble_reference`
        says why); the connectors keep their reference rows (their stiffness
        is small and does not rotate with a chord)."""
        self._beam_batch()
        live = np.asarray(self._live_members(), bool)
        f = self.frame_family
        if len(f.ids):
            for i, e, kg in f.turned_blocks(self):
                if live[e]:
                    yield f.dofs[i], kg, self._k_block(e), self._kg_ref[e]
        for e in self._other_ids:
            if live[e] and self.member_kind[e] == "truss2":
                dof, kg = families.AxialMembers.mass_block(self, e)
                yield dof, kg, self._k_block(e), self._kg_ref[e]

    def _truss_batch(self):
        """Element indices of the bars; the bars' and the connectors'
        families gathered first when the members changed."""
        if getattr(self, "_truss_ids", None) is None or self.bar_family is None:
            self.bar_family = families.AxialMembers(self)
            self.connector_family = families.ConnectorMembers(self)
            self._truss_ids, self._truss_T = self.bar_family.ids, self.bar_family.T
            self._connector_ids = self.connector_family.ids
        return self._truss_ids

    def _truss_forces_batched(self, out, actions, end_actions):
        self._truss_batch()
        self.bar_family.forces(self, out, actions, end_actions)

    def _member_forces(self):
        out=np.zeros(self.ndof); actions=np.zeros((self.nel,6)); end_actions=np.zeros((self.nel,12))
        connector_deformation=np.zeros((self.nel,6))
        if self._jack_T.any():
            from . import analysis_steps as _as
            _as.jack_lengths(self)
        self._beam_forces_batched(out, actions, end_actions)
        self._truss_forces_batched(out, actions, end_actions)     # builds the connectors' family too
        self.connector_family.forces(self, out, actions, end_actions, connector_deformation)
        self.end_actions=end_actions; self.connector_deformation=connector_deformation
        if getattr(self, "_plastic_d", None) is not None and self.mode == "dynamic":
            # advanced in the step's own evaluation only
            self._hinge_update(end_actions, actions, commit=not getattr(self, "_probing", False))
        self._check_failure()
        from . import soil
        return soil.add_force(self, out),actions

    # ------------------------------------------------------------------ #
    #  The forces of SOME members (the implicit solver's tangent, 1.15)
    # ------------------------------------------------------------------ #
    def _tangent_picks(self, groups):
        """For each group of nodes, the members that touch one of them, as
        the compiled kernels take them -- or None when a part of this model's
        force is not a sum over members the kernels evaluate one by one (the
        implicit solver then differences the whole force, as before).

        The gathered arrays are built once for a set of members (a step that
        builds or removes any rebuilds the batches, and these with them); what
        a step changes while it runs -- a jack's length, a temperature, a
        birth offset -- is read at every tangent (`_tangent_begin`)."""
        if type(self) is not MixedStructuralSolver or getattr(self, "backend", "numpy") != "numba" \
                or self.dofs.stride != 6:
            return None
        if ("internal_force" in vars(self) or "_member_forces" in vars(self) or self.components
                or getattr(self, "contact", None) is not None):
            # a force added to the members' (a component: a rigid contact
            # surface's penalty; or a script's own term on the instance) is
            # not a member's: found in
            # review, v1.15 -- the tangent missed the contact stiffness and
            # Newton took 1720 iterations to fail where it had taken 6
            return None
        if getattr(self, "_plastic_d", None) is not None:
            return None                       # hinges: the force reads a history beyond the members' own
        soil_ = getattr(self, "soil", None)
        if soil_ is not None and len(soil_):
            return None
        brk = getattr(self, "breaking", None)
        if brk is not None and brk.can_break:
            return None                       # `_check_failure` reads every evaluation
        bids = self._beam_batch(); tids = self._truss_batch()
        if len(tids) and list(self._truss_T) != [0, 1, 2]:
            return None
        ck = self._connector_kernel()
        if ck is None and len(self._connector_ids):
            return None
        cache = getattr(self, "_picks", None)
        if cache is not None and cache[0] is bids and cache[1] is tids and cache[2] is self._connector_ids \
                and cache[3] is groups:
            return cache[4]
        try:
            from ..kernels import numba_frame as nf
            if nf.njit is None:
                return None
        except ImportError:                                   # pragma: no cover
            return None
        nn = self.nn
        colour = np.full(nn, -1, np.int64)
        for gi, g in enumerate(groups):
            colour[g] = gi
        el = np.asarray(self.mesh.elements)[:, :2]

        def split(ids):
            """positions (in the batch) of the members touching each group"""
            if not len(ids):
                return [np.zeros(0, np.int64)] * len(groups)
            ca = colour[el[ids, 0]]; cb = colour[el[ids, 1]]
            pos = np.concatenate([np.arange(len(ids)), np.arange(len(ids))[cb != ca]])
            col = np.concatenate([ca, cb[cb != ca]])
            order = np.lexsort((pos, col))          # by group, in the batch's order within one
            pos = pos[order]; col = col[order]
            cuts = np.searchsorted(col, np.arange(len(groups) + 1))
            return [np.ascontiguousarray(pos[cuts[i]:cuts[i + 1]], dtype=np.int64) for i in range(len(groups))]
        bpos = split(bids); tpos = split(tids)
        cids = ck["ids"] if ck is not None else np.zeros(0, np.int64)
        cpos = split(cids)
        c = np.ascontiguousarray
        picks = []
        for gi in range(len(groups)):
            b = bpos[gi]; t = tpos[gi]; k = cpos[gi]
            d = {"b": b, "t": t, "c": k}
            if len(b):
                d.update(b_ids=c(self.frame_family.ids64[b]), b_conn=c(self.frame_family.conn64[b]), b_R0=c(self.frame_family.R0c[b]),
                         b_k=c(self.frame_family.kc[b]), b_L0=c(self.frame_family.L0c[b]), b_dofs=c(self.frame_family.dofs64[b]))
            if len(t):
                d.update(t_ids=c(np.asarray(tids, np.int64)[t]), t_conn=c(np.asarray(self.bar_family.conn, np.int64)[t]))
            if len(k):
                d.update(c_ids=c(ck["ids"][k]), c_kind=c(ck["kind"][k]))
            picks.append(d)
        self._picks = (bids, tids, self._connector_ids, groups, picks)
        self._pick_el = np.ascontiguousarray(el, dtype=np.int64)
        self._pick_R0 = np.ascontiguousarray(np.asarray(self.R0, dtype=float))
        return picks

    def _tangent_begin(self, picks):
        """What every evaluation of this tangent shares: the imposed
        deformations, and the cables' stiffness as it is now."""
        # (a jack's length is NOT among them: a bar being jacked holds a
        # force whatever its length, so its length follows every nudge --
        # `_force_of` reads it. Taken once here it put the bar's whole E A / L
        # into the tangent of a member that has none while it is jacked, and
        # Newton went linear: 232 iterations for 20. Found in review.)
        self._pick_jack = bool(self._jack_T.any())
        dth = self._eigen_d()
        th = getattr(self, "_thermal_d", None)
        kf = getattr(self, "_k_factor", 1.0)
        c = np.ascontiguousarray
        for d in picks:
            if len(d["b"]):
                d["b_dth"] = np.zeros((len(d["b"]), 12)) if dth is None else c(dth[d["b_ids"]], dtype=float)
            if len(d["t"]):
                ids = d["t_ids"]
                EA = self.bar_family.EA[d["t"]]
                d["t_EA"] = c(EA * kf[ids] if np.ndim(kf) else EA, dtype=float)
                d["t_dth"] = np.zeros(len(ids)) if th is None else c(th[ids, 6], dtype=float)
                d["t_only"] = c(self.member_tension_only[ids], dtype=np.bool_)
        self._pick_scratch = (np.zeros((self.nel, 6)), np.zeros((self.nel, 12)), np.zeros((self.nel, 6)))
        ck = self._connector_kernel()
        self._pick_birth = None if ck is None else (ck["no_birth"] if self._conn_birth is None else self._conn_birth)

    def _force_of(self, pick, out):
        """The forces of the members `pick` (one entry of `_tangent_picks`)
        added into `out`, by the kernels `internal_force` runs, member for
        member the same arithmetic."""
        from ..kernels.numba_frame import frame_forces_subset, truss_forces
        actions, end_actions, deformation = self._pick_scratch
        qv = self.dofs.view(self.q)
        if len(pick["b"]):
            if frame_forces_subset(pick["b_ids"], self.mesh.nodes, pick["b_conn"], qv, self.orient, pick["b_R0"],
                                   pick["b_k"], pick["b_L0"], pick["b_dofs"], pick["b_dth"],
                                   out, end_actions, actions):
                raise ValueError("mixed structural member collapsed to zero length")
        if len(pick["t"]):
            if self._pick_jack:
                from . import analysis_steps as _as
                _as.jack_lengths(self)
            if truss_forces(pick["t_conn"], self.mesh.nodes, qv, pick["t_EA"],
                            np.ascontiguousarray(self.member_Lu[pick["t_ids"]], dtype=float), pick["t_dth"],
                            pick["t_only"], pick["t_ids"], out, end_actions, actions):
                raise ValueError("mixed structural member collapsed to zero length")
        if len(pick["c"]):
            from ..kernels.numba_connector import respond
            ck = self._connector_kernel()
            if respond(pick["c_ids"], pick["c_kind"], self._pick_el, self.mesh.nodes, qv, self.orient, self.v,
                       self._pick_R0, ck["spins"], *ck["enc"], self._conn_committed, self._conn_slip,
                       self.mode == "dynamic", out, actions, end_actions, deformation, 6, ck["no_rows"],
                       False, True, self._pick_birth) >= 0:
                raise ValueError("mixed structural member collapsed to zero length")

    def _connector_kernel(self):
        """The compiled connectors' arrays under `backend: numba`
        (kernels/numba_connector), rebuilt when the connector list is, or
        None: no connectors, another backend, no numba."""
        if getattr(self, "_connector_ids", None) is None or self.connector_family is None:
            return None
        return self.connector_family.kernel(self)

    def _internal_force_reference(self):
        """The element-at-a-time version, kept for the test that pins the
        batched one to it bit for bit. Not used by the solver."""
        if self._jack_T.any():
            from . import analysis_steps as _as
            _as.jack_lengths(self)
        out=np.zeros(self.ndof); actions=np.zeros((self.nel,6)); end_actions=np.zeros((self.nel,12))
        connector_deformation=np.zeros((self.nel,6))
        for e,conn in enumerate(self.mesh.elements):
            dof=self.dofs.element_dofs(conn)
            qe=self.q[dof].reshape(2,6); x=self.mesh.nodes[conn]+qe[:,:3]
            axis=x[1]-x[0]; L=np.linalg.norm(axis)
            if L<1e-14: raise ValueError("mixed structural member collapsed to zero length")
            if self.member_kind[e]=="truss2":
                n=axis/L; N=self._truss_force(e, L)
                self.dofs.add_nodal(out, [conn[0]], -N*n, TRANSLATION)
                self.dofs.add_nodal(out, [conn[1]], N*n, TRANSLATION)
                end_actions[e,0]=-N; end_actions[e,6]=N
                actions[e,0]=abs(N); continue
            if self.member_kind[e]=="connector2":
                fe, act, deform=connector.response(
                    qe[:,:3], self.orient[conn], self.v[dof], self.mesh.nodes[conn],
                    self.R0[e], self.member_props[e], include_damping=self.mode=="dynamic",
                    slip_committed=self._conn_committed[e], slip_out=self._conn_slip[e],
                    offset=None if self._conn_birth is None else self._conn_birth[e])
                out[dof]+=fe; actions[e]=np.abs(act)
                end_actions[e,:6]=-act; end_actions[e,6:]=act
                connector_deformation[e]=deform
                continue
            # the same kinematics as `rotations.beam_frames`, written with
            # scipy's rotations so the two are independent implementations
            T0 = self.R0[e].T
            T1 = self.orient[conn[0]]; T2 = self.orient[conn[1]]
            Tm = T1 @ Rotation.from_rotvec(0.5 * Rotation.from_matrix(T1.T @ T2).as_rotvec()).as_matrix()
            e1 = axis / L; t2 = Tm @ self.R0[e, 1]
            e3 = np.cross(e1, t2); e3 /= np.linalg.norm(e3); e2 = np.cross(e3, e1)
            Te = np.stack((e1, e2, e3), axis=1); R = Te.T
            local=np.zeros(12); local[6]=L-self.length[e]
            local[3:6]=Rotation.from_matrix(Te.T @ T1 @ T0).as_rotvec()
            local[9:12]=Rotation.from_matrix(Te.T @ T2 @ T0).as_rotvec()
            fl=self.k_local[e]@local; end_actions[e]=fl; T=repeated_rotation(R); out[dof]+=T.T@fl
            actions[e]=[max(abs(fl[i]),abs(fl[i+6])) for i in range(6)]
        return out,actions

    def _nodal_fields(self, actions, recover_stress=False):
        out=super()._nodal_fields(actions,recover_stress)
        out["member_type"]=np.isin(self.member_kind, tuple(self.BEAM_TYPES)).astype(float)
        out["beam_formulation"]=(self.member_kind=="beam2_timoshenko").astype(float)
        if getattr(self, "_plastic_d", None) is not None:
            # `hinge_rotation`: the accumulated plastic rotation of the worse
            # end, in radians; `hinge_usage`: that over the rotation capacity,
            # 0 elastic, 1 about to tear. The ratio was written as
            # `hinge_rotation` until v1.0. NaN = no hinge
            dmg = np.full(self.nel, np.nan)
            dmg[self._hinged] = self.breaking.hinge_damage[self._hinged].max(axis=1)
            rot = np.full(self.nel, np.nan)
            rot[self._hinged] = self._hinge_rotation[self._hinged].max(axis=1)
            out["hinge_rotation"] = rot
            out["hinge_usage"] = dmg
        connector_mask=(self.member_kind=="connector2")
        out["connector_flag"]=connector_mask.astype(float)
        names=("force_x","force_y","force_z","moment_x","moment_y","moment_z")
        # SIGNED, from the end actions: `actions` holds magnitudes for the
        # capacity checks, and a bearing's compression and its uplift both
        # came out positive
        signed=getattr(self,"end_actions",None)
        signed=signed[:,6:] if signed is not None else actions
        for i,name in enumerate(names):
            out[f"connector_{name}"]=np.where(connector_mask,signed[:,i],0.0)
            out[f"connector_deformation_{connector.COMPONENTS[i]}"]=self.connector_deformation[:,i]
            out[f"connector_slip_{connector.COMPONENTS[i]}"]=self._conn_slip[:,i]
        return out

    def summary(self):
        neb=int(np.sum(self.member_kind=="beam2"))
        ntm=int(np.sum(self.member_kind=="beam2_timoshenko"))
        nc=int(np.sum(self.member_kind=="connector2"))
        nt=self.nel-neb-ntm-nc
        # A MULTI-MATERIAL MODEL HAS TO SAY SO. Printing one material name for
        # a model with five would be the same untruth the refusal used to
        # prevent; the distinct E-rho pairs are listed instead,
        # in member-group order, with how many elements each one drives.
        if self.one_material:
            mats = f"{self.material.name}"
        else:
            seen = {}
            for m in self.member_material:
                seen.setdefault(id(m), [m, 0])[1] += 1
            mats = f"{len(seen)} linear_elastic: " + ", ".join(
                f"E={v[0].E:.3g} rho={getattr(v[0], 'rho', 0.0):.4g} ({v[1]} el)"
                for v in seen.values())
        return (f"  mesh          {self.nn} nodes, {self.nel} Line2 elements\n"
                f"  elements      {nt} truss2 + {neb} beam2 + {ntm} beam2_timoshenko + {nc} connector2, 6 DOF/node\n"
                f"  material      {mats}; backend {getattr(self, 'backend', 'numpy')}\n"
                f"  mode          {self._mode_text()}"
                + mass_scaling.summary_line(getattr(self, "mass_scaling", None))
                + member_axes.summary_lines(self))        # v1.0
