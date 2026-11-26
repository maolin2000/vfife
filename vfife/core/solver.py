"""Explicit central-difference / dynamic-relaxation solver.

Two modes share one time loop:

  mode: dynamic
      Physical lumped mass, dt from the CFL limit computed with the TRUE
      element characteristic length V/A_max (the original used the nominal
      L/n, which is unsafe on distorted meshes and over-conservative on
      regular ones).  Physical time, viscous damping.

  mode: relaxation                      <- use this for quasi-static problems
      Fictitious Underwood masses m_i = 0.25 * sum_j |K_ij| with dt = 1, i.e.
      pseudo-time relaxation.  Combined with kinetic damping this converges to
      the static solution in far fewer steps than integrating the physical
      dynamics with an arbitrary viscous coefficient.

Kinetic damping (Barnes / Papadrakakis): monitor the total kinetic energy and,
the step after it peaks, stop dead -- zero the velocity and restart.  Each reset
removes one oscillation cycle, so the residual falls in bursts instead of
decaying slowly under a hand-tuned viscous coefficient.

The loop is written in LEAPFROG (velocity) form, which matters here: a
central-difference restart must use half the acceleration for its first
velocity, v_(1/2) = 0.5 a dt.  In displacement form that half factor is easy to
lose, and with kinetic damping restarting every few steps the resulting energy
injection compounds -- the displacement-form version of this loop reached
|u| ~ 1e140 by step 15000 on a mesh whose critical step it satisfied with a
factor of 1.8 to spare.

Integration is on the DISPLACEMENT field, not on absolute coordinates.  The
original integrated x directly, so on a 12 m beam with 0.08 m of displacement
roughly two significant digits were lost to cancellation on every step -- fatal
once the element kernels move to fp32.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .problem import Problem
from .. import kernels
from ..materials.points import initialise_points, validate_restored_state
from ..elements import continuum
from ..elements import library as continuum_library
from . import amplitude as amp
from . import selectors
from ..base.dof import DofLayout
from .components import HasComponents
from ..base.voigt import von_mises
from .mesh import FaceMoment, FollowerPressure, Mesh, scatter_add, surface_traction, surface_pressure, volume_body_force
from .diagnostics import (StopAnalysis, UserError,
                          start as start_diagnostics, finish as finish_diagnostics,
                          ConvergenceFailure, NonFiniteFailure, first_nonfinite,
                          worst_residual)


# one monitor record for every solver (core/monitors)
from .monitors import Monitor  # noqa: F401  (the name this module exported)
from . import monitors as monitor_records


#: this package's one Voigt order (see vfife/voigt.py)
VOIGT_INDEX = {"xx": 0, "yy": 1, "zz": 2, "xy": 3, "yz": 4, "zx": 5, "xz": 5}

REDUCTIONS = {
    "max": lambda v: float(np.max(v)),
    "min": lambda v: float(np.min(v)),
    "mean": lambda v: float(np.mean(v)),
    "absmax": lambda v: float(v[np.argmax(np.abs(v))]),
}


@dataclass
class Block:
    """One element type with one material: its geometry, its kernel, its state.

    A model used to be one element type, and `Solver` held that type's
    precompute, connectivity, material state and rotations directly.  A mesh
    with hexahedra in the regular regions and tetrahedra in the awkward ones --
    which is how anyone meshes a real geometry -- was refused at import.

    So the model is now a LIST of these, and a single-type mesh is a list of
    one.  There is no special case for the common path: the force is summed
    over blocks, the fictitious mass is summed over blocks, the stable time
    step is the minimum over blocks.  Blocks share the node set and the
    boundary conditions; they differ in shape functions and, since the case
    can assign several solid sections, in material.

    `elem_ids` says where this block's elements sit in the GLOBAL element
    order -- the mesh's own order, element types in sorted sequence -- which
    is the order the writer, the erosion mask and every per-element output
    use.  A block is a gather from that order and scatters back into it.
    """

    element: str
    conn: np.ndarray                 # (n_elem, nodes per element)
    pre: object                      # continuum.Precomputed
    x0_e: np.ndarray                 # centred reference coordinates
    quat: np.ndarray                 # (n_elem, 4) co-rotational frame
    material: object = None          # this block's own Material instance
    elem_ids: np.ndarray = None      # (n_elem,) indices into the global order
    name: str = ""                   # section name from the case, or ""
    # filled in after the material points are bound: the Numba backend bakes
    # per-point crack-band factors into its buffers, so the force evaluator
    # cannot exist until the mass, erosion and reinforcement setup has run
    force: object = None             # kernels.make_force(...)
    state: np.ndarray = None         # (n_elem * n_gp, n_state)
    points: object = None            # material point metadata

    @property
    def n_elem(self) -> int:
        return self.conn.shape[0]

    @property
    def n_gp(self) -> int:
        return self.pre.n_gp

    @property
    def label(self) -> str:
        return f"{self.name}:{self.element}" if self.name else self.element


def partition_elements(conn: np.ndarray, n_nodes: int, specs, region_nodes):
    """Which material block owns each element of one connectivity array.

    An element belongs to a block when EVERY one of its nodes is in the
    block's region -- the rule the line solvers already use for member
    groups, and the one that makes a Gmsh physical volume select exactly its
    own elements without carrying an element-to-volume map: a solid element
    with all its nodes on the interface to the next volume would be flat.

    Blocks apply in order and a later one takes what an earlier one also
    covers, which is Abaqus' section-assignment rule and lets a case say
    `all: cover` then `core: confined`.  Returns (owner (n_elem,) int with -1
    for unassigned, overrides [(from, to, count)]).
    """
    owner = np.full(conn.shape[0], -1, dtype=np.int64)
    overrides = []
    for i, spec in enumerate(specs):
        mark = np.zeros(n_nodes, dtype=bool)
        mark[region_nodes(spec["region"])] = True
        inside = mark[conn].all(axis=1)
        for j in range(i):
            n = int((inside & (owner == j)).sum())
            if n:
                overrides.append((j, i, n))
        owner[inside] = i
    return owner, overrides


class Solver(HasComponents):
    def __init__(self, problem, mesh: Mesh, material, log: Callable = print):
        problem = Problem.of(problem)        # typed, and its dict
        self.p = problem
        self.mesh = mesh
        self.material = material
        self.log = log

        pre_opts = problem.case.element             # typed
        self.nn = mesh.n_nodes
        self.dofs = DofLayout.translations(self.nn)

        # One entry per element type present.  `cell_blocks` carries them when
        # the mesh is mixed; otherwise the whole mesh is one block.  This is
        # the GLOBAL element order: sorted element types, the mesh's order
        # within each.
        declared = pre_opts.type
        blocks = getattr(mesh, "cell_blocks", None) or {}
        solid = {k: v for k, v in blocks.items()
                 if k in continuum_library.ELEMENTS} or {declared: mesh.elements}
        self._block_specs = [(name, np.ascontiguousarray(conn))
                             for name, conn in sorted(solid.items())]
        self.nel = int(sum(c.shape[0] for _n, c in self._block_specs))

        # the iteration counters exist before the boundary conditions, because
        # a step records where it started from
        self.step = 0
        self.step_start = 0
        self._resuming = False

        # the regions come first: material blocks are cut by region, and the
        # boundary conditions (gravity needs each block's density) come after
        self._resolve_regions()
        self.blocks: List[Block] = self._build_blocks(material, pre_opts)
        self._setup_monitors_and_first_step()
        self._setup_reinforcement()
        # after the reinforcement, because a `bar` monitor names one of its
        # sets and a name that matches nothing must fail here rather than
        # produce an empty series someone later reads as zero
        self._check_monitor_targets()
        self._register_staged()
        self._setup_mass_and_dt()
        self._setup_erosion()

        # the force evaluator is built AFTER bind_points, because the Numba
        # backend bakes the per-point crack-band factors into its buffers
        sol = problem.solver                       # typed
        self.backend = sol.backend
        if self.bonded and self.backend == "warp":
            # WARP, AND ONLY WARP. The first version of this guard refused
            # every backend but numpy, which was wrong by inspection: the
            # numba path is not device resident (only warp defines
            # `force_resident`), so it updates the host arrays every step and a
            # host-side bar joins it through F_int exactly as it does under
            # numpy. Refusing numba cost nothing but the ability to use it.
            #
            # Warp is different in kind: its time loop lives on the device and
            # never brings `u` back, so a bar computed on the host cannot see
            # the displacement it is supposed to slip against. Making it work
            # means a device-resident bar, which is the rest of stage C.
            raise UserError(
                "BONDED_BAR_BACKEND",
                "bond-slip reinforcement cannot run on the warp backend yet",
                "warp keeps the whole time loop on the device and the bonded "
                "bar lives on the host, so it would never see the displacement "
                "it slips against. Use backend = numba, which is not resident "
                "and runs the bar correctly, or drop the `bond` block for "
                "perfectly bonded bars, which warp does support.",
            )
        kw = {}
        if self.backend == "warp":
            kw = {"device": sol.device, "precision": sol.precision}

        for b in self.blocks:
            # State rows are element-major (index = e*n_gp + g), matching the
            # reshape in continuum.internal_force.
            b.state, b.points = initialise_points(
                b.material, b.n_elem * b.n_gp, kinematics="solid_3d",
                backend=self.backend,
                char_length=np.repeat(b.pre.char_len, b.n_gp),
            )
            b.force = kernels.make_force(b.pre, b.conn, b.x0_e, b.material,
                                         self.backend, n_nodes=self.nn, **kw)

        one = self.blocks[0]
        self.material_points = one.points
        # A backend that can keep the nodal state on its own device runs the
        # whole time loop there.  Otherwise every step would ship displacement,
        # material state and element force across the bus -- ~280 MB per
        # iteration at 10^6 elements, several times the kernel it feeds.
        #
        # With several blocks the FIRST block's evaluator owns the nodal state
        # (displacement, velocity, mass, boundary data) and the others attach
        # to it, the way embedded reinforcement does: their kernels read its
        # displacement and add into its force, on the device.
        self.resident = all(hasattr(b.force, "force_resident") for b in self.blocks)
        if self.resident:
            for b in self.blocks[1:]:
                one.force.attach_block(b.force)
        self.u = np.zeros((self.nn, 3))
        self.v = np.zeros((self.nn, 3))
        self.contact = None                       # set by rigid_contact.attach_contact
        self.ke_prev = 0.0
        self.n_resets = 0
        self._half = True          # the first step is a restart too

    # ------------------------------------------------------------------ #

    def _build_blocks(self, material, pre_opts) -> List[Block]:
        """Cut every element type by material block and precompute each.

        Without `blocks` in the case the whole mesh is `material`; each
        element type is one block and the first of them keeps the very
        object the caller passed (tests and the API inspect it).  Every
        other block gets its own instance: a material binds per-point data
        (the crack-band factors) when its points are initialised, and two
        blocks sharing one instance would each overwrite the other's.
        """
        from ..materials.factory import create_material, wants_auto_length

        specs = list(self.p.get("blocks") or [])
        formulation = pre_opts.formulation
        hourglass = pre_opts.hourglass_coeff

        # `char_length: auto` resolves against the whole mesh, the same probe
        # assemble.build uses, so a block material and the whole-mesh material see
        # one number
        base = self.p.get("material")
        lc = None
        if (any(wants_auto_length(sp["material"]) for sp in specs)
                or (not specs and len(self._block_specs) > 1
                    and isinstance(base, dict) and wants_auto_length(base))):
            lc = self.material_char_length()

        def make_block(name, conn, ids, mat, label, form, hg):
            pre = continuum.precompute(self.mesh.nodes, conn, element=name,
                                       formulation=form, hourglass_coeff=hg,
                                       material=mat)
            x0 = self.mesh.nodes[conn]
            quat = np.zeros((conn.shape[0], 4))
            quat[:, 0] = 1.0
            return Block(name, conn, pre,
                         np.ascontiguousarray(x0 - x0.mean(axis=1, keepdims=True)),
                         quat, material=mat, elem_ids=ids, name=label)

        out: List[Block] = []
        offset = 0
        if not specs:
            for i, (name, conn) in enumerate(self._block_specs):
                mat = material
                if i and isinstance(base, dict) and base.get("type"):
                    mat = create_material(base, lc)
                ids = offset + np.arange(conn.shape[0], dtype=np.int64)
                out.append(make_block(name, conn, ids, mat, "", formulation, hourglass))
                offset += conn.shape[0]
            return out

        mats = [create_material(sp["material"], lc) for sp in specs]
        counts = np.zeros(len(specs), dtype=np.int64)
        overrides_all: Dict[tuple, int] = {}
        unassigned = []
        for name, conn in self._block_specs:
            owner, overrides = partition_elements(conn, self.nn, specs, self._region)
            for j, i, n in overrides:
                overrides_all[(j, i)] = overrides_all.get((j, i), 0) + n
            lost = np.flatnonzero(owner < 0)
            if lost.size:
                unassigned.append((name, conn, offset + lost, lost))
            for i, sp in enumerate(specs):
                ids = np.flatnonzero(owner == i)
                counts[i] += ids.size
                if not ids.size:
                    continue
                out.append(make_block(
                    name, np.ascontiguousarray(conn[ids]), offset + ids, mats[i],
                    sp.get("name") or f"block{i + 1}",
                    sp.get("formulation") or formulation,
                    hourglass if sp.get("hourglass_coeff") is None else sp["hourglass_coeff"]))
            offset += conn.shape[0]

        if unassigned:
            n = int(sum(g.size for _n, _c, g, _l in unassigned))
            name, conn, gids, lids = unassigned[0]
            c = self.mesh.nodes[conn[lids[0]]].mean(axis=0)
            raise UserError(
                "ELEMENTS_UNASSIGNED",
                f"{n} element(s) are covered by no `assign` entry; the first is "
                f"{name} element {int(gids[0])} centred at "
                f"({c[0]:.4g}, {c[1]:.4g}, {c[2]:.4g})",
                "an element belongs to a section when EVERY one of its nodes is in "
                "the assigned region. Give the background section `region: all` "
                "first and the local ones after it, or widen the region so it "
                "holds the whole element.",
            )
        empty = [sp.get("name") or f"block{i + 1}" for i, sp in enumerate(specs) if counts[i] == 0]
        if empty:
            raise UserError(
                "SECTION_EMPTY",
                f"section(s) {', '.join(repr(e) for e in empty)} are assigned to a "
                "region that contains no whole element",
                "a region selects NODES; an element is in it only when all of its "
                "nodes are. A box one node wide selects nothing, and a later "
                "assignment may have taken every element this one covered.",
            )
        for (j, i), n in sorted(overrides_all.items()):
            self.log(f"    section {specs[i].get('name')!r} takes {n} element(s) "
                     f"also covered by {specs[j].get('name')!r} (later assignment wins)")
        return out

    def material_char_length(self) -> float:
        """The mesh's smallest V/A_max, what `char_length: auto` resolves to."""
        return min(float(continuum.precompute(self.mesh.nodes, conn, element=name,
                                              formulation="reduced").char_len.min())
                   for name, conn in self._block_specs)

    def _host_conn(self) -> Dict[str, np.ndarray]:
        """Every element type's full connectivity, for locating bars: a bar's
        host is geometry, whichever material block the element is in."""
        return {name: conn for name, conn in self._block_specs}

    # ---- single-block conveniences -------------------------------------- #
    # Most of the solver, the coupled path and the tests were written when a
    # model was one element type.  These keep that code working for a
    # single-block model and refuse loudly for a mixed one, so any path that
    # has not been converted fails at its own line with a sentence, instead of
    # quietly computing block 0 and reporting it as the whole model.

    def _only(self, what):
        if len(self.blocks) != 1:
            raise UserError(
                "MIXED_MESH_UNSUPPORTED",
                f"this model has {len(self.blocks)} element blocks "
                f"({', '.join(b.element for b in self.blocks)}) and `{what}` "
                "describes only one",
                "the feature reading it has not been extended to mixed meshes; "
                "mesh the volume with a single element type, or report which "
                "command you ran",
            )
        return self.blocks[0]

    @property
    def pre(self):
        return self._only("pre").pre

    @property
    def x0_e(self):
        return self._only("x0_e").x0_e

    @property
    def _force(self):
        """The nodal owner on the resident path: block 0's evaluator, to
        which every other block is attached (see `resident` in __init__)."""
        return self.blocks[0].force

    @property
    def n_gp(self):
        return self._only("n_gp").pre.n_gp

    @property
    def n_nodes_e(self):
        return self._only("n_nodes_e").pre.n_nodes

    @property
    def state(self):
        return self._only("state").state

    @state.setter
    def state(self, value):
        self._only("state").state = value

    @property
    def quat(self):
        return self._only("quat").quat

    @quat.setter
    def quat(self, value):
        self._only("quat").quat = value

    def _region(self, name: str) -> np.ndarray:
        """Node set by name, from the case file or from a Gmsh physical group."""
        if name not in self.mesh.regions:
            # offered but not yet built -- a boundary condition may name
            # `surface` without any region spec having referenced it
            thunk = getattr(self, "_lazy_regions", {}).get(name)
            if thunk is not None:
                self.mesh.regions[name] = thunk()
        try:
            return self.mesh.regions[name]
        except KeyError:
            have = ", ".join(sorted(set(self.mesh.regions)
                                    | set(getattr(self, "_lazy_regions", {}))))
            have = have or "(none)"
            raise ValueError(
                f"unknown region {name!r}. Available: {have}. Regions come "
                "from the case file's `regions:` block and, for a Gmsh mesh, "
                "from the file's physical groups."
            ) from None

    def _additional_nodal_mass(self) -> np.ndarray:
        """Nodal masses per translation (v1.0: directional and
        weightless ones too; no rotary inertia on a solid, no members for a
        hydrodynamic mass)."""
        from . import added_mass
        out, _rot, self.weightless_mass = added_mass.table(
            self.p, self._region, self.mesh.nodes, None, rotations=False)
        return out

    def _resolve_regions(self):
        """Resolve every named region once."""
        self.amplitudes = amp.build(self.p.case.amplitudes)

        predefined = dict(self.mesh.regions)
        predefined.setdefault("all", np.arange(self.nn, dtype=np.int64))
        # `surface` is offered, not computed: it costs a pass over every element
        # face, and a case that never names it should not pay for it.
        self._lazy_regions = {
            "surface": selectors.surface_provider(
                self.mesh, [n for n, _c in self._block_specs])}
        self.mesh.regions = selectors.resolve_all(
            self.mesh.nodes, self.p.get("regions") or {}, predefined,
            lazy=self._lazy_regions,
        )

    def _setup_monitors_and_first_step(self):
        """The monitors, then the first step's boundary conditions."""
        self.monitors: Dict[str, Monitor] = {}
        for name, spec in self.p.get("output", {}).get("monitors", {}).items():
            # element and bar monitors may name no region at all: then the
            # reduction runs over the whole model
            self.monitors[name] = monitor_records.from_spec(spec, reduce_default="max")

        self.steps = self.p.get("steps") or [{"name": "step1"}]
        self.apply_step(0)

    # ------------------------------------------------------------------ #
    #  Element and bar monitors
    # ------------------------------------------------------------------ #

    BAR_FIELDS = ("axial_stress", "axial_force", "plastic_strain")

    def _bar_sets(self) -> Dict[str, object]:
        """Every reinforcement set by name, embedded and bonded alike."""
        return {b.label: b for b in list(self.rebar) + list(self.bonded)}

    def _check_monitor_targets(self):
        """Fail at setup on a monitor that can never produce a number.

        A monitor that silently yields nothing is worse than one that raises:
        the run costs its half hour, the result file looks complete, and the
        contract reads a series that is empty or constant-zero.
        """
        bars = self._bar_sets()
        # what the materials can report, asked of a fresh state so this does
        # not depend on the run having started (line ~929 does the same)
        scalars = {"von_mises"}
        for b in self.blocks:
            scalars |= set(b.material.output_fields(b.material.init_state(1)))
        for name, mon in self.monitors.items():
            if mon.kind == "element":
                if mon.field != "stress" and mon.field not in scalars:
                    raise ValueError(
                        f"monitor {name!r} reads element field {mon.field!r}, "
                        f"which this model does not produce. Available: "
                        f"stress (with a component), "
                        f"{', '.join(sorted(scalars))}")
                if mon.regions and not self._element_mask(mon.regions[0]).any():
                    raise ValueError(
                        f"monitor {name!r}: region {mon.regions[0]!r} owns no "
                        "elements. An element belongs to a region only when "
                        "EVERY one of its nodes is inside it -- the rule a "
                        "section assignment uses -- so a region that is a "
                        "surface or a single plane of nodes owns none")
            elif mon.kind == "bar":
                if mon.of not in bars:
                    have = ", ".join(sorted(bars)) or "(none)"
                    raise ValueError(
                        f"monitor {name!r} reads reinforcement set {mon.of!r}, "
                        f"which does not exist. Available: {have}. The name is "
                        "the `name` on the assignment, not the reinforcement "
                        "geometry's key")
                if mon.field not in self.BAR_FIELDS:
                    raise ValueError(
                        f"monitor {name!r} reads bar field {mon.field!r}; "
                        f"available: {', '.join(self.BAR_FIELDS)}")
                if mon.regions and not self._bar_mask(bars[mon.of],
                                                      mon.regions[0]).any():
                    raise ValueError(
                        f"monitor {name!r}: no segment of {mon.of!r} lies in "
                        f"region {mon.regions[0]!r}")

    def _element_mask(self, region: Optional[str]) -> np.ndarray:
        """Which elements a region owns, in the global element order.

        Every node of the element inside -- `partition_elements`' rule, so a
        monitor and a section assignment cannot disagree about what a region
        contains. Eroded elements are excluded: a deleted element's damage is
        frozen at whatever it held when it left, and letting that decide a
        `max` would make the monitor report history rather than state.
        """
        mask = np.ones(self.nel, dtype=bool)
        if region is not None:
            mark = np.zeros(self.nn, dtype=bool)
            mark[self._region(region)] = True
            mask = np.zeros(self.nel, dtype=bool)
            for b in self.blocks:
                mask[b.elem_ids] = mark[b.conn].all(axis=1)
        if getattr(self, "erosion", None) is not None:
            for b in self.blocks:
                mask[b.elem_ids] &= b.pre.active.astype(bool)
        return mask

    def _bar_mask(self, emb, region: Optional[str]) -> np.ndarray:
        """Which segments of a bar set lie in a region.

        A bar's nodes are its OWN, not the host mesh's, so the node-set that
        `regions:` resolves to says nothing about them. The test is therefore
        geometric and reads the region's box straight from the case: both ends
        of the segment inside it. A region that is not a box -- a Gmsh
        physical group, an explicit node list -- has no box to test against
        and is refused rather than approximated by a bounding box, which would
        quietly select the wrong segments on any non-convex group.
        """
        n = emb.n_elem
        if region is None:
            return np.ones(n, dtype=bool)
        spec = (self.p.get("regions") or {}).get(region)
        box = (spec or {}).get("box") if isinstance(spec, dict) else None
        if box is None:
            raise ValueError(
                f"a bar monitor cannot use region {region!r}: it is not a "
                "`box`. Bar segments carry their own nodes, so membership is "
                "geometric; give the monitor a box region covering the length "
                "of bar you want")
        x0, x1, y0, y1, z0, z1 = (float(v) for v in box)
        ends = np.asarray(emb.x0_e, dtype=float).reshape(n, 2, 3)
        lo = np.array([min(x0, x1), min(y0, y1), min(z0, z1)])
        hi = np.array([max(x0, x1), max(y0, y1), max(z0, z1)])
        return ((ends >= lo - 1e-9) & (ends <= hi + 1e-9)).all(axis=(1, 2))

    def _element_series(self, mon: Monitor, efields: Dict[str, np.ndarray]) -> float:
        mask = self._element_mask(mon.regions[0] if mon.regions else None)
        if not mask.any():
            # every element in the region has eroded away; there is no state
            # left to report and a stale number would be a lie
            return float("nan")
        v = efields[mon.field]
        if mon.field == "stress":
            v = v[:, VOIGT_INDEX[mon.component]]
        return REDUCTIONS[mon.reduce](np.asarray(v)[mask])

    def _bar_series(self, mon: Monitor) -> float:
        emb = self._bar_sets()[mon.of]
        mask = self._bar_mask(emb, mon.regions[0] if mon.regions else None)
        if mon.field == "plastic_strain":
            v = np.asarray(emb.state)[:, 1]
        else:
            # read-only: `force()` would advance the plastic state, making the
            # answer depend on how often the monitor was sampled
            v = np.asarray(emb.stress(self.u))
            if mon.field == "axial_force":
                v = v * np.asarray(emb.pre.area, dtype=float)
        return REDUCTIONS[mon.reduce](v[mask])

    def _shared_faces(self):
        """(faces shared between blocks, whether there are several blocks)."""
        if len(self.blocks) < 2:
            return None, False
        if getattr(self, "_shared_face_keys", None) is None:
            from .mesh import shared_face_keys
            self._shared_face_keys = shared_face_keys([(b.conn, b.element) for b in self.blocks])
        return self._shared_face_keys, True

    def apply_step(self, k: int):
        """Install the boundary conditions of step `k`.

        Model-level `constraints` and `loads` are the base and apply in every
        step; a step's own lists are added to them.  Where the two prescribe the
        same (node, dof), the LATER declaration wins -- so a step can drive a
        degree of freedom that the base merely held, which is how "support it
        throughout, then push this face" is written without repeating the
        supports in every step.
        """
        step = self.steps[k]
        self.step_index = k
        self.step_name = step.get("name") or f"step{k + 1}"
        staged = bool(self._step_solver("load_steps"))
        # where this step's iteration count starts from -- restored rather than
        # reset when a run is picked up from a checkpoint mid-step
        # a step picked up mid-way keeps its start, its stage and the point
        # its prescribed motions blend in from
        self._mid_step = bool(getattr(self, "_resuming", False))
        if not self._mid_step:
            self.step_start = self.step
        self._resuming = False

        # A model-level boundary condition ramps on during the FIRST step and
        # stays on for the rest of the analysis, which is what "the structure
        # carries its own weight throughout" actually means.  Without this rule
        # a base load either ramps up again from zero in every step (if it kept
        # the `ramp` default) or arrives as a step change in the first one (if
        # it defaulted to `hold`); the first is wrong, the second is a shock.
        # An explicit `amplitude:` on a base BC overrides the rule everywhere.
        def base(items):
            out = []
            for it in items:
                if it.get("amplitude") is None and k > 0:
                    it = {**it, "amplitude": "hold"}
                out.append(it)
            return out

        constraints = base(self.p.get("constraints") or []) \
            + list(step.get("constraints") or [])
        loads = base(self.p.get("loads") or []) \
            + list(step.get("loads") or [])

        self.fixed = np.zeros((self.nn, 3), dtype=bool)
        self.motion_velocity = np.zeros((self.nn, 3))
        # (node index array, dof, value, amplitude), later entries overriding
        prescribed: Dict[tuple, tuple] = {}
        self.ground_groups = []
        self.follower_groups = []
        for c in constraints:
            if len(c["dofs"]) != 3:
                raise ValueError("solid-element constraints require three translational DOF flags [ux,uy,uz]")
            if c.get("record") is not None:
                raise ValueError("a constraint `record` (a support's own ground motion) is "
                                 "implemented for frame, fibre-beam and shell models; this is a solid model")
            idx = self._region(c["region"])
            a = amp.get(self.amplitudes, c.get("amplitude"),
                        f"constraint on {c['region']!r}", staged)
            val = c.get("value")
            motion = c.get("motion", "displacement")
            if motion != "displacement" and self.p.solver.mode != "dynamic":
                raise ValueError(f"{motion} constraints require solver.mode = dynamic")
            if motion != "displacement" and self.p.solver.backend == "warp":
                raise ValueError("velocity/acceleration constraints are not yet supported by backend = warp")
            for d in range(3):
                if not c["dofs"][d]:
                    continue
                self.fixed[idx, d] = True
                v = 0.0
                if val is not None:
                    v = float(val[d] if hasattr(val, "__len__") else val)
                for n in idx:
                    if v == 0.0:
                        prescribed.pop((int(n), d), None)
                    else:
                        prescribed[(int(n), d)] = (v, a, motion)

        # regroup into (nodes, dof, value, amplitude) blocks so the device path
        # uploads a handful of arrays rather than one entry per node
        blocks: Dict[tuple, list] = {}
        for (n, d), (v, a, motion) in prescribed.items():
            blocks.setdefault((d, v, a.name, motion), []).append(n)
        self.prescribed = [
            (np.array(sorted(ns), dtype=np.int64), d, v, self.amplitudes[an], motion)
            for (d, v, an, motion), ns in blocks.items()
        ]

        # where each prescribed node stands as this step begins -- what
        # `apply_bc`'s `approach` blends in from
        self._bc_start = [np.asarray(self.u)[idx, d].copy()
                          for idx, d, _v, _a, _m in self.prescribed] \
            if getattr(self, "u", None) is not None else \
            [np.zeros(len(idx)) for idx, *_ in self.prescribed]
        saved = getattr(self, "_restored_bc_start", None)
        if self._mid_step and saved is not None and len(saved) == len(self._bc_start) \
                and all(len(a) == len(b) for a, b in zip(saved, self._bc_start)):
            # blended in from where the step began, not from the restart
            self._bc_start = [np.array(a, dtype=float) for a in saved]
        self._restored_bc_start = None

        self._driven = np.zeros(self.nn, dtype=bool)
        for idx, _d, _v, _a, _motion in self.prescribed:
            self._driven[idx] = True

        # loads grouped by amplitude: the device holds one unit vector per
        # distinct curve, not one per load and not a fresh upload per iteration
        by_amp: Dict[str, np.ndarray] = {}
        for ld in loads:
            region = ld.get("region", "all")
            idx = self._region(region)
            a = amp.get(self.amplitudes, ld.get("amplitude"),
                        f"load on {region!r}", staged)
            if ld.get("moment") is not None:
                # on a solid a moment on a face is the load conjugate to the
                # face's rotation (mesh.FaceMoment): with `follower` the couple
                # turns with the face, without it the reference distribution
                # is a dead load
                if self.p.solver.backend == "warp":
                    raise ValueError("a face moment is not yet supported by backend = warp")
                evaluators = []
                for b in self.blocks:
                    try:
                        evaluators.append(FaceMoment(self.mesh.nodes, b.conn, b.element, idx, ld["moment"]))
                    except ValueError as e:
                        if "no complete exterior faces" not in str(e) or len(self.blocks) == 1:
                            raise
                if not evaluators:
                    raise ValueError("moment region contains no complete exterior faces")
                if ld.get("follower"):
                    self.follower_groups.append((a, evaluators))
                else:
                    F = by_amp.setdefault(a.name, np.zeros((self.nn, 3)))
                    for fm in evaluators:
                        F += fm.reference_forces()
                continue
            if ld.get("ground_motion") is not None:
                if self.p.solver.mode != "dynamic":
                    raise ValueError("ground_motion requires solver.mode = dynamic")
                if self.p.solver.backend == "warp":
                    raise ValueError("ground_motion is not yet supported by backend = warp")
                gm = ld["ground_motion"]
                data = np.loadtxt(gm["path"], dtype=float, encoding="utf-8")
                direction = np.asarray(gm.get("direction", [1, 0, 0]), dtype=float)
                if data.ndim != 2 or data.shape[1] < 2 or len(data) < 2 or direction.shape != (3,):
                    raise ValueError("ground_motion needs a two-column record and 3-component direction")
                if np.any(np.diff(data[:, 0]) <= 0) or np.linalg.norm(direction) == 0:
                    raise ValueError("ground_motion times must increase and direction must be non-zero")
                self.ground_groups.append((data[:, 0], data[:, 1] * float(gm.get("scale", 1.0)), direction))
                continue
            if ld.get("pressure") is not None and ld.get("follower"):
                # integrated on the current faces every step (mesh.FollowerPressure)
                if self.p.solver.backend == "warp":
                    raise ValueError("a follower pressure is not yet supported by backend = warp")
                shared, many = self._shared_faces()
                parts = [FollowerPressure(self.mesh.nodes, b.conn, b.element, idx, ld["pressure"],
                                          shared=shared, allow_empty=many) for b in self.blocks]
                if not any(p.faces for p in parts):
                    raise ValueError("pressure region contains no complete exterior faces")
                self.follower_groups.append((a, [p for p in parts if p.faces]))
                continue
            F = by_amp.setdefault(a.name, np.zeros((self.nn, 3)))
            if ld.get("traction") is not None or ld.get("pressure") is not None:
                # over the exterior of the WHOLE model: an interface between
                # two blocks is no surface, and a block with none of the
                # region's faces is no error
                shared, many = self._shared_faces()
                total = np.zeros((self.nn, 3))
                for b in self.blocks:
                    if ld.get("traction") is not None:
                        total += surface_traction(self.mesh.nodes, b.conn, b.element, idx,
                                                  ld["traction"], shared=shared, allow_empty=many)
                    else:
                        total += surface_pressure(self.mesh.nodes, b.conn, b.element, idx,
                                                  ld["pressure"], shared=shared, allow_empty=many)
                if not np.any(total):
                    raise ValueError(f"{'traction' if ld.get('traction') is not None else 'pressure'} "
                                     "region contains no complete exterior faces")
                F += total
            elif ld.get("body_force") is not None:
                for b in self.blocks:
                    F += volume_body_force(self.mesh.nodes, b.conn, idx,
                                           b.pre.shape_gp, b.pre.w_detJ,
                                           ld["body_force"])
            elif ld.get("gravity") is not None:
                for b in self.blocks:
                    F += volume_body_force(self.mesh.nodes, b.conn, idx,
                                           b.pre.shape_gp, b.pre.w_detJ,
                                           b.material.rho * np.asarray(ld["gravity"]))
                # the nodal masses in the region weigh too, as they do on every
                # frame solver; until v1.0 a solid's did not
                if self.p.case.nodal_masses:
                    from . import added_mass
                    extra = added_mass.weighed(self._additional_nodal_mass(), self.weightless_mass)
                    F[idx] += extra[idx] * np.asarray(ld["gravity"], dtype=float)
            elif ld.get("vector") is not None:
                vec = np.asarray(ld["vector"], dtype=float)
                if ld.get("per_node"):
                    F[idx] += vec
                else:
                    F[idx] += vec / idx.size
            else:
                # A LOAD THIS SOLVER DOES NOT OWN HAS TO SAY SO.
                # This branch used to be a bare `else` that assumed `vector`,
                # and `np.asarray(None, dtype=float)` is `nan` -- so a
                # `temperature` or `moving_load` on a solid model poisoned the
                # whole force vector instead of being refused. Every other
                # structural solver's dispatch ends in a sentence; this was the
                # one that ended in an assumption.
                kind = next((k for k in ("temperature", "moving_load", "line_load", "moment")
                             if ld.get(k) is not None), None)
                raise ValueError(
                    f"solid elements do not take a {kind!r} load" if kind else
                    "a load needs one of vector, traction, pressure, body_force, "
                    "gravity or ground_motion")
        self.load_groups = [(self.amplitudes[n], F) for n, F in by_amp.items()]

    def _step_solver(self, key, default=None):
        """A solver setting, overridden by the current step if it names one.

        The fallback chain is step -> solver block -> SCHEMA default; the last
        comes with the typed case (core/problem.py), which
        holds every setting the schema has, defaults included.

        `default` survives for the cases the schema cannot express -- a value
        computed from another setting, such as `n_steps` falling back to the
        model-level count. It must never be a restatement of a schema default.
        """
        step = self.steps[getattr(self, "step_index", 0)]
        if key in step and step[key] is not None:
            return step[key]
        v = getattr(self.p.solver, key, None)
        return default if v is None else v
    @staticmethod
    def _prestrain(name, spec, material, bar_nodes=None, bar_conn=None):
        """The effective prestress, as the strain offset the bar carries.

        eps_p = sigma / E, and the conversion is exact only while the bar is
        elastic there -- which is where a tendon is, and where the refusal
        below keeps it. A jacking stress at or past yield would need the
        material's own inverse, and would also mean a tendon with no elastic
        range left to load: it is a modelling error, not a case to support.

        WITHOUT `friction` this is one number for the whole set, and it is the
        EFFECTIVE prestress: elastic shortening comes out of equilibrium
        (elements/truss.py, PRESTRESS) and every other loss is the analyst's.

        WITH `friction` it is a value per bar element, `prestress` is the
        JACKING stress, and the program computes the profile along the tendon
        from mu, k and the draw-in (elements/truss.tendon_profile). Relaxation
        and shrinkage/creep are then all that is left outside.
        """
        sigma = spec.get("prestress")
        if not sigma:
            if spec.get("friction"):
                raise UserError(
                    "FRICTION_WITHOUT_PRESTRESS",
                    f"reinforcement {name!r} has a `friction` block but no `prestress`",
                    "friction is a loss along a tendon; there is no tendon force "
                    "for it to act on.",
                )
            return 0.0
        sigma = float(sigma)
        E = float(getattr(material, "E", 0.0) or 0.0)
        if E <= 0.0:
            raise UserError(
                "PRESTRESS_MATERIAL",
                f"reinforcement {name!r} sets prestress but its material "
                f"{material.name!r} has no elastic modulus",
                "prestress is converted to an initial strain sigma/E. Use a "
                "`rebar`, `steel02` or `linear_elastic` bar material.",
            )
        friction = spec.get("friction")
        if friction:
            sigma = Solver._friction_profile(name, spec, friction, sigma, E,
                                             bar_nodes, bar_conn)
        sy = getattr(material, "sy", None)
        if sy is not None and np.max(np.abs(sigma)) >= float(sy):
            peak = float(np.max(np.abs(sigma)))
            raise UserError(
                "PRESTRESS_YIELD",
                f"reinforcement {name!r} is prestressed to "
                f"{peak / 1e6:.0f} MPa, at or past its yield strength "
                f"{float(sy) / 1e6:.0f} MPa",
                "the strain offset is sigma/E, which is only the right strain "
                "while the bar is elastic -- and a tendon jacked to yield has "
                "no elastic range left to carry load. Give the EFFECTIVE "
                "prestress after losses (0.6-0.75 f_py is the usual range), "
                "or raise the bar's yield_strength to the strand's.",
            )
        return sigma / E

    @staticmethod
    def _friction_profile(name, spec, friction, sigma_con, E, bar_nodes, bar_conn):
        """`sigma_con exp(-(mu theta + k s))` in every element of the tendon.

        A tendon has an arc length and an accumulated angle only because it is
        ONE connected bar. Lines that merely touch are separate bars with
        separate nodes (io/schema.BarLineSpec), and a friction profile run over
        those would restart at every piece -- so that is refused here rather
        than producing a plausible wrong answer.
        """
        from ..elements import truss

        chains = truss.bar_chains(bar_conn)
        nodes = np.asarray(bar_nodes, dtype=float)
        touching = sum(
            1 for (_a, hi), (lo, _b) in zip(chains, chains[1:])
            if np.allclose(nodes[bar_conn[hi - 1, 1]], nodes[bar_conn[lo, 0]], atol=1e-9))
        if touching:
            raise UserError(
                "FRICTION_SPLIT_TENDON",
                f"reinforcement {name!r} asks for a friction profile but is laid as "
                f"{len(chains)} bars, {touching + 1} of which merely touch end to start",
                "friction accumulates ALONG a tendon, so the tendon has to be one "
                "bar -- over separate bars the profile would restart at each piece "
                "and read back as a plausible answer. Draw the route as a single "
                "line with `points: [[x,y,z], ...]`.",
            )
        for lo, hi in chains:
            if truss.is_closed(bar_conn, lo, hi):
                raise UserError(
                    "FRICTION_CLOSED_TENDON",
                    f"reinforcement {name!r} asks for a friction profile on a closed "
                    "route (a hoop)",
                    "friction is measured from the jacking end, and a closed loop "
                    "has none. Friction belongs on a tendon, not on a tie.",
                )
        try:
            return truss.tendon_profile(
                bar_nodes, bar_conn, sigma_con,
                mu=float(friction.get("mu", 0.0)), k=float(friction.get("k", 0.0)),
                jack=friction.get("jack", "start"),
                anchor_set=friction.get("anchor_set"), E=E)
        except ValueError as exc:
            raise UserError("ANCHOR_SET", f"reinforcement {name!r}: {exc}",
                            "state the draw-in the anchorage actually has, or jack "
                            "from the far end.") from exc

    def _setup_reinforcement(self):
        """Build the embedded bar blocks, if the case defines any.

        Each block is an independent element set over the SAME nodes as the
        concrete, so nothing about the mesh changes: the bars contribute force
        and mass through the host shape functions and go through the ordinary
        gather assembly.  See elements/truss.py for the formulation.
        """
        from ..elements import truss
        from ..elements.bonded import BondedTruss
        from ..materials import create

        self.rebar = []
        #: bars that may SLIP: their own dofs, coupled through an interface.
        #: A separate list, not a flag on the same objects, because the two
        #: have different unknowns and the time loop has to treat them apart.
        self.bonded = []
        specs = self.p.get("reinforcement") or []
        host_type = self.p.case.element.type
        damaging = [b for b in self.blocks
                    if b.material.damage_measure(b.material.init_state(1)) is not None]
        host_damages = bool(damaging)
        for i, spec in enumerate(specs):
            # the dump carries `name: None` for an unnamed set: the default
            # beside a `.get` was never taken, and two unnamed sets merged
            # under one key (the review of v1.19)
            name = spec.get("name") or f"rebar{i + 1}"
            mspec = dict(spec["material"])
            mat = create(mspec.pop("type"), **mspec)
            bar_nodes, bar_conn = truss.bars_from_spec(spec)
            prestrain = self._prestrain(name, spec, mat, bar_nodes, bar_conn)
            if (spec.get("host_deduction") or 0.0) > 0.0 and host_damages:
                # `host_deduction` subtracts E_c*eps*A from the bar to cancel
                # the concrete volume the bar displaces. That is right while
                # the concrete carries E_c*eps and wrong the moment it cracks:
                # the bar is then short by 15% of its force at yield, and past
                # yield its net tangent is E_hard - E_c, NEGATIVE -- a bar that
                # loses force as it hardens. elements/truss.py has said "leave
                # it off when the concrete is expected to damage" since the
                # option was written; case_12 carried it into a cracking beam
                # anyway and six stages of work chased the wall it built.
                # A warning in a docstring is not a guard.
                raise UserError(
                    "HOST_DEDUCTION_DAMAGING",
                    f"reinforcement {name!r} sets host_deduction with the "
                    f"damaging host material {damaging[0].material.name!r}",
                    "the deduction cancels concrete stiffness that a cracked "
                    "host no longer has, and turns the bar's post-yield "
                    "tangent negative. Remove `host_deduction` (the overlap "
                    "it corrects is under 3% of EA at rho = 3%), or use a "
                    "linear-elastic host for the elastic work it is meant for.",
                )
            bond_spec = spec.get("bond")
            if bond_spec:
                bspec = dict(bond_spec)
                bond = create(bspec.pop("type", "bond_ceb"), **bspec)
                dia = spec.get("diameter")
                if dia is None:
                    # A bond force is tau times the bar SURFACE, so the
                    # diameter is not optional here the way it is for a
                    # perfectly bonded bar, where only the area matters.
                    raise ValueError(
                        f"reinforcement {name!r} has a bond block but no "
                        "`diameter`; bond acts on the bar surface, so the "
                        "area alone does not determine it")
                bt = BondedTruss(
                    self.mesh.nodes,
                    self._host_conn(), host_type,
                    bar_nodes, bar_conn, mat, bond, spec["area"], dia,
                    transverse_penalty=spec.get("transverse_penalty"),
                    # `or 0.0`, not a get() default: the schema emits the key
                    # with its own default, so a get() default never fires
                    host_deduction=spec.get("host_deduction") or 0.0,
                    label=name,
                    anchorage=spec.get("anchorage"),
                    prestrain=prestrain,
                )
                self.bonded.append(bt)
                continue
            # A DEVICE that can do nothing is refused; a PROPERTY that this
            # formulation happens not to need is not.
            #   `transverse_penalty` is a numerical restraint on slip ACROSS
            #   the bar. Without an interface there is no slip and nothing
            #   reads it, so accepting it would tell the analyst their number
            #   had been used when nothing did -- the same defect as
            #   `host_deduction` being dropped by BondedTruss, pointing the
            #   other way.
            #
            #   `diameter` is deliberately NOT refused. It is a real property
            #   of the bar that the perfectly bonded formulation does not
            #   happen to need, and refusing it would forbid the one pattern
            #   that makes a controlled comparison readable: the same
            #   reinforcement spec, run once with a `bond` block and once
            #   without. tests/test_bonded_solver.py is written that way, and
            #   it is the right way to write it.
            if spec.get("transverse_penalty") is not None:
                raise ValueError(
                    f"reinforcement {name!r} sets `transverse_penalty` but has "
                    "no `bond` block. It restrains slip across the bar, a "
                    "perfectly bonded bar has no slip, and nothing would read "
                    "it. Add a `bond` block, or remove it.")
            if spec.get("anchorage") is not None:
                raise ValueError(
                    f"reinforcement {name!r} sets `anchorage` but has no `bond` "
                    "block: a perfectly bonded bar's ends are already tied to "
                    "the host, so nothing would read it.")
            # Every block, not just the largest: a bar in a mixed mesh runs
            # through whichever elements lie on its line, and `mesh.elements`
            # holds only one of them -- with `host_type` naming a different one,
            # which came out as a numpy shape error from inside the locator.
            emb = truss.EmbeddedTruss(
                self.mesh.nodes,
                self._host_conn(), host_type,
                bar_nodes, bar_conn, mat, spec["area"],
                host_deduction=spec.get("host_deduction") or 0.0,
                prestrain=prestrain,
            )
            emb.label = name
            self.rebar.append(emb)

    def _bar_kernel(self, emb):
        """The compiled force of an embedded bar set under `backend: numba`
        (kernels/numba_embedded), or None: another backend,
        a law without a compiled twin. (`VFIFE_COMPILED_BARS=0`, which
        switched it off, went in v1.18: `backend: numpy` is the
        reference.)"""
        if self.backend != "numba":
            return None
        if not (isinstance(self.u, np.ndarray) and self.u.dtype == np.float64
                and self.u.flags.c_contiguous and self.u.shape == (self.nn, 3)):
            return None
        try:
            from ..kernels.numba_embedded import BarForce
            return BarForce(emb)
        except (ImportError, TypeError):
            return None

    def _lumped_physical_mass(self):
        """rho*V/n_nodes lumped at the nodes, summed over every block."""
        m = np.zeros((self.nn, 3))
        for b in self.blocks:
            nn_e = b.pre.n_nodes
            m_e = np.repeat(
                (b.material.rho * b.pre.volume / nn_e)[:, None, None],
                nn_e, axis=1) * np.ones((1, 1, 3))
            m += scatter_add(m_e, b.conn, self.nn)
        return m

    def _assemble_mass(self, factor):
        """Underwood masses from the row sums, optionally degraded per element.

        K is linear in the material stiffness, so a per-element secant factor
        scales that element's row sums directly -- no B matrices are rebuilt.
        """
        s = self._rs_safety
        m = np.zeros((self.nn, 3))
        for i, b in enumerate(self.blocks):
            rs = self._rs_base[i]
            if factor is not None:
                rs = rs * factor[b.elem_ids, None, None]
            m += scatter_add(0.25 * rs / (s * s), b.conn, self.nn)
        m += 0.25 * self._rs_bar / (s * s)
        return np.maximum(m, 1e-30)


    def _setup_erosion(self):
        """Element deletion, after what Abaqus and LS-DYNA actually do.

        A softening continuum model has no answer for an element that has
        reached total degradation: the material point still exists, still has
        nodes, and still has whatever residual out-of-balance force those nodes
        carry, but no stiffness with which to resist it.  Both commercial codes
        answer the same way -- take the element out of the calculation.
        Abaqus: `*SECTION CONTROLS, ELEMENT DELETION=YES` with MAX DEGRADATION.
        LS-DYNA: `*MAT_CSCM` ERODE, "elements erode when damage exceeds 0.99
        and the maximum principal strain exceeds ERODE-1", plus the
        criterion-free `*CONTROL_TIMESTEP` ERODE on collapsing volume.

        Both criteria are implemented here, and the second is not optional
        garnish: a material threshold can be tuned wrong, an inverted element is
        wrong under every tuning.

        `points` is LS-DYNA's NUMFIP -- how many of the element's integration
        points must fail before it goes.  The LS-DYNA default is 1, which on a
        fully integrated hexahedron means one failed point out of eight kills
        the element; that is aggressive, and the default here is half the
        points, rounded up.
        """
        spec = self.p.solver.erosion
        self.erosion = None
        self.n_eroded = 0
        self.frozen = np.zeros(self.nn, dtype=bool)
        if spec is None:                  # (a model: `erosion: {}` is erosion with its defaults)
            return
        if all(b.material.damage_measure(b.material.init_state(1)) is None
               for b in self.blocks):
            raise ValueError(
                f"solver.erosion is configured but material {self.material.name!r} "
                "has no erosion_state(); it has no notion of failure, so nothing "
                "would ever erode. Remove the erosion block or use a material "
                "that reports damage (cdp)."
            )
        measure = spec.measure or "d"
        if measure != "d" and not any(
                measure in b.material.output_fields(b.material.init_state(1))
                for b in self.blocks):
            raise ValueError(
                f"solver.erosion.measure = {measure!r} but no material here reports "
                f"a field of that name; cdp reports dt and dc")
        # the schema's defaults, typed: a partial dict from Python is
        # validated too since v1.19 (the copy of the defaults that
        # stood here is gone)
        e = spec.model_dump()
        if e.get("points") is None:
            e["points"] = (min(b.n_gp for b in self.blocks) + 1) // 2
        self.erosion = e

    def _setup_mass_and_dt(self):
        sol = self.p.solver                        # typed
        mode = sol.mode
        # `or`, not a get() default: the schema emits `mass: None` for an
        # unset field, so the key IS present and get()'s default never fires.
        # That silently ran relaxation cases with physical mass and a CFL time
        # step -- real dynamics, with n_steps counted as if they were relaxation
        # iterations, and no error anywhere.
        mass_kind = sol.mass or (
            "physical" if mode == "dynamic" else "underwood")
        self.additional_mass = self._additional_nodal_mass()

        if mass_kind == "eigenvalue":
            # the rotation-invariant bound (continuum.stiffness_max_eigenvalues):
            # each element gives every one of its nodes lambda_max(K_e) / 4,
            # in every configuration. Expressed as per-element "row sums" so
            # the assembly below is the same code
            rs = []
            for b in self.blocks:
                lam = continuum.stiffness_max_eigenvalues(self.mesh.nodes, b.conn, b.pre, b.material.D0)
                rs.append(np.broadcast_to(lam[:, None, None], (len(lam), b.pre.n_nodes, 3)).copy())
        if mass_kind in ("underwood", "eigenvalue"):
            if mass_kind == "underwood":
                rs = [continuum.stiffness_row_sums(
                          self.mesh.nodes, b.conn, b.pre, b.material.D0)
                      for b in self.blocks]
            # Underwood's bound is m_ii >= (dt^2/4) sum_j |K_ij|.  With dt = 1
            # that is 0.25*sum|K|, i.e. exactly the stability boundary -- and a
            # marginally stable central-difference scheme diverges within a few
            # thousand steps (measured: NaN by step ~20k).  The safety factor
            # enters as 1/s^2, the same way dt_safety does in dynamic mode.
            s = float(sol.relaxation_safety)
            self._rs_base = rs
            self._rs_safety = s
            # bar row sums do not degrade -- steel does not lose stiffness the
            # way the concrete around it does -- so they are a fixed nodal term
            from ..elements.truss import _emb_row_sums
            self._rs_bar = sum(
                (_emb_row_sums(emb, self.nn) for emb in self.rebar),
                np.zeros((self.nn, 3)),
            )
            # A bonded bar hands the host only its INTERFACE stiffness -- a
            # two-point coupling whose row sum is elementary. That is the
            # difference from `_emb_row_sums`, which has to push a perfectly
            # bonded bar's whole axial stiffness onto host nodes it does not
            # belong to. (That bound is exact -- tests/test_stability_row_sums.py;
            # the "drift" once blamed on it was the relaxation exit firing on
            # the last step of the load ramp, see `settled_at_last_check`.)
            for bt in self.bonded:
                self._rs_bar = self._rs_bar + bt.host_row_sums(self.nn)
            self.mass = self._assemble_mass(None)
            self.mass += self.additional_mass
            #: the bars' own fictitious mass, from their own row sums
            self.mass_bar = [
                np.maximum(0.25 * bt.bar_row_sums() / (s * s), 1e-30)
                for bt in self.bonded
            ]
            self.dt = 1.0
            self.added_mass_frac = 0.0
        else:
            self.mass = self._lumped_physical_mass()
            for emb in self.rebar:
                self.mass += emb.lumped_mass(self.nn)
            self.mass_bar = [np.maximum(bt.bar_mass(), 1e-30)
                             for bt in self.bonded]
            self.mass += self.additional_mass
            self.mass = np.maximum(self.mass, 1e-30)
            # the binding limit is the smallest element of ANY block: a
            # tetrahedron in the transition region is typically finer than the
            # hexahedra it joins, and its CFL limit governs the whole model --
            # each block against its own wave speed, since a steel pad and
            # the concrete under it do not share one
            dt_cfl = (sol.dt_safety
                      * min(float(b.pre.char_len.min()) / b.material.wave_speed
                            for b in self.blocks))
            # a bar is stiff and light: its own CFL limit is usually the binding
            # one once reinforcement is present, and it is silently violated if
            # only the concrete is checked
            for emb in self.rebar:
                dt_bar = (sol.dt_safety * float(emb.pre.L0.min())
                          / emb.material.wave_speed)
                dt_cfl = min(dt_cfl, dt_bar)
            # A bonded bar's own dofs are held by the interface, and the
            # transverse penalty is usually the stiffest thing in the model.
            # dt <= 2 sqrt(m/k) is the explicit bound for a single dof; the
            # same dt_safety scales it. Leaving this out would let the penalty
            # -- a NUMERICAL device -- silently destabilise a run that the
            # physics could have carried.
            for i, bt in enumerate(self.bonded):
                k = bt.bar_row_sums()
                m_b = bt.bar_mass()
                dt_int = (sol.dt_safety
                          * 2.0 * float(np.sqrt(np.min(m_b / np.maximum(k, 1e-30)))))
                dt_cfl = min(dt_cfl, dt_int)
            target = sol.target_dt
            if target and target > dt_cfl:
                f = (target / dt_cfl) ** 2
                self.mass *= f
                self.added_mass_frac = f - 1.0
                self.dt = float(target)
            else:
                self.added_mass_frac = 0.0
                self.dt = float(dt_cfl)

        # Base excitation uses physical inertia even when relaxation builds a
        # fictitious Underwood mass. Embedded bars contribute their own mass.
        self.physical_mass = self._lumped_physical_mass()
        for emb in self.rebar:
            self.physical_mass += emb.lumped_mass(self.nn)
        self.physical_mass += self.additional_mass

        self.mode = mode
        self.mass_kind = mass_kind
        if mode == "relaxation":
            n = sol.n_steps
            if n is None:
                # derived from the steps: a staged step's budget is its
                # load_steps * max_steps_per_load, a continuous one its own
                # n_steps (the schema has checked that every step has one)
                n = 0
                for st in self.p.get("steps") or []:
                    if st.get("n_steps") is not None:
                        n += int(st["n_steps"])
                    else:
                        n += int(st["load_steps"]) * int(
                            st.get("max_steps_per_load")
                            or sol.max_steps_per_load)
            self.n_steps = int(n)
            self.total_time = float(self.n_steps)
        else:
            self.total_time = float(sol.total_time)
            self.n_steps = max(1, int(round(self.total_time / self.dt)))

        # staged relaxation: hold each load level until equilibrium
        self.load_steps = sol.load_steps
        self.max_steps_per_load = int(sol.max_steps_per_load)
        self.stage_ramp = int(sol.stage_ramp_iterations
                              or max(1, self.max_steps_per_load // 5))
        if self.load_steps and mode != "relaxation":
            # (the schema refuses it first since a solver's case is validated,
            # v1.19; kept as the second line)
            raise ValueError("solver.load_steps requires mode = relaxation")

        damp = sol.damping
        self.damp_kind = damp.type or ("kinetic" if mode == "relaxation" else "viscous")
        self.zeta = float(damp.coeff)
        #: `alpha` NOW APPLIES WHATEVER `type` IS, AND KINETIC + ALPHA IS THE
        #: COMBINATION IT WAS ADDED FOR.
        #:
        #: Kinetic damping zeroes the velocity when the kinetic energy PEAKS.
        #: That is what makes it the fastest relaxation there is, and it is
        #: also a blind spot with a precise shape: a mode whose kinetic energy
        #: grows MONOTONICALLY never presents a peak, so the reset never
        #: fires and nothing in the scheme opposes it.
        #:
        #: A softening element is exactly such a mode. Measured on the Tanaka
        #: column under a full +-32 mm reversal, one node grew by a constant
        #: factor of 1.0096 per iteration -- displacement and velocity
        #: together, smoothly, for as long as it was allowed to -- from 32 mm
        #: to 46 mm in 37 iterations, and on to 6e8 mm. Elements eroded AFTER
        #: that started, not before. Adding fictitious mass (relaxation_safety
        #: 0.3 -> 0.075, mass x16) made it fail EARLIER, which is what rules
        #: out a time-step bound and rules in a negative tangent stiffness.
        #:
        #: `alpha * m * v` is the one term that acts on such a mode: it
        #: opposes velocity whatever the energy is doing. alpha = 0.002 with
        #: kinetic damping carried that reversal to completion with no eroded
        #: elements, where the same run without it diverged.
        #:
        #: IT IS NOT FREE, AND THE NUMBER IS NOT UNIVERSAL. On a monotonic
        #: push that converges without any of this, alpha = 0.002 moved the
        #: base shear by 1.18% RMS and alpha = 0.005 by 6.57% -- not a smooth
        #: perturbation but a different branch of an ill-posed softening
        #: problem. Use the smallest value that runs, and report it; a result
        #: that needs alpha to exist is a result that depends on it.
        self.rayleigh_alpha = float(damp.alpha)
        self.rayleigh_beta = float(damp.beta)
        if self.damp_kind == "rayleigh" and self.rayleigh_beta:
            raise ValueError("solid Rayleigh beta is not yet available; use alpha only or truss2")
        # keyed off alpha, not off `type`: the term is applied host-side, so a
        # warp run would take `type: kinetic, alpha: ...` and silently drop it
        if self.rayleigh_alpha and sol.backend == "warp":
            raise ValueError("Rayleigh damping (damping.alpha) is not yet "
                             "supported by backend = warp")
        # `alpha m v` is applied explicitly from the lagged velocity, and an
        # explicit damping term cuts the stable step (elements.stability).
        # For a mass-proportional term the cut is alpha / (2 omega_max) --
        # negligible for any alpha anyone would use -- but the seven solvers
        # should agree on the rule, not on the coincidence that this one's
        # term is small. The viscous coefficient is NOT passed: `_run_step`
        # treats it by the averaged form, which needs no cut. omega_max is
        # taken as 2c/h from the CFL step the solver already chose.
        self.damping_step_factor = 1.0
        if (mode != "relaxation" and self.damp_kind == "rayleigh"
                and self.rayleigh_alpha and self.added_mass_frac == 0.0):
            from ..elements.stability import damped_step_factor
            omega_max = 2.0 * float(sol.dt_safety) / self.dt
            f = damped_step_factor(omega_max, alpha=self.rayleigh_alpha)
            if f < 1.0:
                self.damping_step_factor = f
                self.dt = float(self.dt * f)
        if mode != "relaxation":
            # n_steps * dt is the run's duration exactly
            from ..elements.stability import aligned_step
            self.stable_dt = self.dt
            self.dt, self.n_steps = aligned_step(self.total_time, self.dt)
        self.tol = float(sol.tolerance)
        self.rot_iters = int(sol.rotation_iterations)


    # ------------------------------------------------------------------ #

    def _loading_settled(self, progress: float) -> bool:
        """True once no amplitude in this step is still changing.

        A residual below tolerance while the load is still ramping is not
        equilibrium, it is a snapshot of a moving target -- so the early exit
        has to wait for the loading to stop moving, not merely for the step to
        end.
        """
        for _i, _d, _v, a, _motion in self.prescribed:
            if not a.settled(progress):
                return False
        for a, _F in self.load_groups:
            if not a.settled(progress):
                return False
        for a, _ev in self.follower_groups:
            if not a.settled(progress):
                return False
        return True

    def external_force(self, progress: float, physical_time: float | None = None) -> np.ndarray:
        F = np.zeros((self.nn, 3))
        for a, unit in self.load_groups:
            F += unit * a(progress)
        for a, evaluators in self.follower_groups:
            factor = a(progress)
            if factor != 0.0:
                x = self.mesh.nodes + self.u
                for fp in evaluators:
                    F += fp(x) * factor
        if physical_time is not None:
            for t, acc, direction in self.ground_groups:
                F -= self.physical_mass * (np.interp(physical_time, t, acc, left=0.0, right=0.0)
                                           * direction)[None, :]
        return F

    def _register_staged(self):
        """Collect the states that advance on CONVERGED points, not iterations.

        One registry, one policy per member, described once in core/staging.py.
        A path that registers nothing pays nothing -- which is deliberate for
        the continuum state, whose exclusion is a measurement rather than an
        oversight (see that module, and tests/test_staging.py).
        """
        from .staging import CallbackState, Registry, TrialState

        self.staged = Registry()
        for bt in self.bonded:
            # ORDER MATTERS. The cyclic law's commit banks energy from the
            # committed anchor to the current point and must run BEFORE the
            # snapshot that makes the current point the new anchor.
            commit = getattr(bt.bond, "commit", None)
            if commit is not None:
                self.staged.add(CallbackState(commit, bt.bond_state))
            # EVERY bond law here is anchored -- `bond_ceb` unloads from
            # (s_prev, tau_prev) and ratchets s_max -- so every one of them
            # needs the restore-then-integrate policy, not only the cyclic
            # one. tests/test_staging.py once recorded that bond_ceb "has no
            # irreversible accumulator"; s_max and the unloading anchor are
            # exactly that, and the iteration path moves them. Measured in
            # tests/test_bonded_convergence.py: a bar started 20 um off in a
            # rigid host settled at 300 um of PERMANENT slip with zero force,
            # every overshoot of the relaxation transient having moved the
            # anchor a little further. Restored to the committed point on
            # every evaluation, it slides back to zero.
            self.staged.add(TrialState(bt.bond_state))

    def _commit_staged(self):
        """A load level has converged. Advance every staged history once.

        WHY IT EXISTS
            Relaxation reaches one equilibrium point by taking thousands of
            explicit iterations whose path is not physical. A law that banked
            damage every time it was evaluated therefore read every
            convergence wiggle as a load reversal: on a MONOTONIC tension
            case, changing only `n_steps` from 60 000 to 200 000 moved the
            slip from 106.0 to 24.3 um against a perfectly-bonded 24.9. The
            fibre path had the same defect in a different shape and its own
            fix; core/staging.py is where the two were finally written once.
        """
        self.staged.commit()

    def apply_bc(self, u: np.ndarray, progress: float, approach: float = 1.0):
        """`approach` blends a prescribed displacement in from where the node
        already is, over the first `stage_ramp` iterations of a STEP.

        A prescribed level is absolute: `value: -0.0564` means -56.4 mm, not
        56.4 mm further down. The staging inside a step already ramps INTO each
        load level for exactly that reason -- 4 mm imposed in one iteration
        diverged to 1e100 (see the ramp in `run`). What it could not cover is a
        step that STARTS somewhere: the first level's ramp begins at
        `0/load_steps`, so a node the previous step left at +3.57 mm is dragged
        to 0 in a single iteration. On the post-tensioned T-beam, whose
        transfer step cambers the bearings 3.6 mm before the jacks touch it,
        that reached a kinetic energy of 1e269 within 200 iterations of the
        loading step -- and it is the first case in the project to prescribe a
        displacement on nodes an earlier step had moved, which is why it had
        never shown up.

        The blend changes nothing after those iterations, and nothing at all
        for a step whose nodes start at zero, which is every other case here.
        """
        u[self.fixed] = 0.0
        for i, (idx, d, v, a, motion) in enumerate(self.prescribed):
            target = v * a(progress)
            if motion == "displacement":
                if approach < 1.0 and i < len(self._bc_start):
                    u0 = self._bc_start[i]
                    target = u0 + (target - u0) * approach
                u[idx, d] = target
            elif motion == "velocity":
                u[idx, d] = self.u[idx, d] + self.dt * target
            else:
                old = self.motion_velocity[idx, d].copy()
                self.motion_velocity[idx, d] = old + self.dt * target
                u[idx, d] = self.u[idx, d] + self.dt * old + 0.5 * self.dt**2 * target

    # ------------------------------------------------------------------ #

    def check_erosion(self) -> int:
        """Apply the erosion criteria once. Returns how many elements died."""
        e = self.erosion
        n = 0
        for b in self.blocks:
            J, eps1, bad = continuum.strain_measures(b.pre, self.u[b.conn])
            dead = np.zeros(b.n_elem, dtype=bool)

            if e["measure"] == "d":
                d = b.material.damage_measure(b.state)
            else:
                d = b.material.output_fields(b.state).get(e["measure"])
            if d is not None:
                # both halves must hold, as in MAT_CSCM: damage says the material
                # is finished, the principal strain says the element has gone
                n_dmg = (d >= e["damage"]).reshape(b.n_elem, b.n_gp).sum(axis=1)
                dead |= (n_dmg >= e["points"]) & (eps1 >= e["strain"])

            # the safety net: an element whose volume has collapsed or inverted
            # goes whatever the constitutive model says about it
            dead |= bad | (J < e["min_volume_ratio"]) | (J > e["max_volume_ratio"])
            killed = continuum.erode(b.pre, dead)
            if killed:
                # WHERE, NOT JUST HOW MANY, AND IT USED TO SAY NEITHER.
                #   Erosion removed elements and logged nothing. On case_06 an
                #   erosion block that looked inert (the last frame's worst
                #   damage was 0.88, below its 0.98 threshold, because damage
                #   FALLS again when a crack closes) had quietly deleted four
                #   elements at the loaded head and moved the peak shear 2%.
                #   Removing part of the model is not a detail of the
                #   iteration; it is a change to the model, and it says so.
                c = self.mesh.nodes[b.conn[np.flatnonzero(dead)]].mean(axis=1)
                self.log(f"    {killed} {b.element} element(s) eroded near "
                         f"({c[0, 0]:.3g}, {c[0, 1]:.3g}, {c[0, 2]:.3g}) m"
                         + (f" and {killed - 1} more" if killed > 1 else ""))
            n += killed
        if n:
            self.n_eroded += n
            self._after_erosion()
        return n

    def _after_erosion(self):
        """Freeze whatever the erosion has left unsupported, and tell the device.

        LS-DYNA leaves the nodes of a deleted element in the model as free mass,
        which is right for a transient analysis -- a freed fragment simply flies
        off and is harmless.  It is NOT right here.  Under dynamic relaxation
        the mass is fictitious and the time step is pseudo-time, so a node with
        mass, no stiffness and no restoring force does not fly off; it coasts,
        for as many iterations as the run has left, and it sits in the residual
        norm the whole time.  So a node that no element and no bar still touches
        is frozen where it is and taken out of the free set.

        Frozen, not zeroed: the node keeps the displacement it had reached.
        Sending it back to the origin would be a discontinuity in the middle of
        a converged field, which is a far bigger disturbance than the drift.
        """
        support = np.zeros(self.nn, dtype=np.int64)
        for b in self.blocks:
            live = b.pre.active > 0
            support += np.bincount(b.conn[live].ravel(), minlength=self.nn)
        for emb in self.rebar:
            support += np.bincount(emb.conn.ravel(), minlength=self.nn)
        # A prescribed node is never frozen, however unsupported it is: it is
        # being DRIVEN, and holding it would silently cancel the loading. On the
        # bar case, eroding the last element leaves the loaded end unsupported,
        # and freezing it stopped the applied displacement dead -- the device
        # path reported max|u| = 0.00056 against a prescribed 0.020.
        newly = (support == 0) & ~self.frozen & ~self._driven
        if newly.any():
            self.frozen |= newly
            self.log(f"    {int(newly.sum())} node(s) left unsupported, frozen")
        if self.resident:
            for b in self.blocks:
                b.force.refresh_weights(b.pre)
            self._force.set_frozen(self.frozen)

    def summary(self) -> str:
        lines = [
            f"  mesh          {self.nn} nodes, {self.nel} elements",
            "  element       "
            + " + ".join(f"{b.label}({b.n_elem})" for b in self.blocks)
            + f" / {self.blocks[0].pre.formulation} ("
            + ", ".join(str(b.n_gp) for b in self.blocks)
            + " integration points)",
            "  material      " + "; ".join(
                (f"{b.name}: " if b.name else "") + f"{b.material.name} "
                f"({b.material.n_state} state vars/point)"
                for b in self._distinct_material_blocks()),
            f"  mode          {self.mode}, mass = {self.mass_kind}, "
            f"damping = {self.damp_kind}",
            f"  backend       {self.backend}" + (
                f" ({self.p.solver.device}, "
                f"{self.p.solver.precision}, "
                f"device-resident loop)" if self.backend == "warp" else ""),
            f"  time step     dt = {self.dt:.4e}, {self.n_steps} steps"
            + ("/step" if len(getattr(self, "steps", [])) > 1 else ""),
        ]
        if len(getattr(self, "steps", [])) > 1:
            names = ", ".join(st.get("name") or f"step{i+1}"
                              for i, st in enumerate(self.steps))
            lines.append(f"  steps         {len(self.steps)}: {names}")
        # every declared curve, not just the ones the first step happens to
        # use -- the summary is printed before the later steps are installed
        declared = sorted(self.p.case.amplitudes)
        if declared:
            lines.append("  amplitudes    " + "; ".join(
                self.amplitudes[n].describe() for n in declared))
        for emb in self.rebar:
            lines.append(
                f"  reinforcement {emb.label}: {emb.n_elem} bar segments, "
                f"A = {emb.pre.area[0] * 1e6:.1f} mm^2, "
                f"{emb.material.name} (embedded, perfect bond)"
            )
        if self.erosion:
            lines.append(
                f"  erosion       {self.erosion['measure']} >= {self.erosion['damage']} and eps_1 >= "
                f"{self.erosion['strain']} at >= {self.erosion['points']}/"
                f"{min(b.n_gp for b in self.blocks)} points, or volume ratio outside "
                f"[{self.erosion['min_volume_ratio']}, "
                f"{self.erosion['max_volume_ratio']}]; checked every "
                f"{self.erosion['every']} iterations"
            )
        if getattr(self, "load_steps", None):
            lines.append(
                f"  load steps    {self.load_steps} equilibrium points, "
                f"ramp {self.stage_ramp} + <= {self.max_steps_per_load} "
                f"iterations each, tol {self.tol:.1e}"
            )
        if self.added_mass_frac > 0:
            lines.append(f"  mass scaling  +{self.added_mass_frac * 100:.1f}% added mass")
        if np.any(self.additional_mass):
            lines.append(f"  nodal mass    {self.additional_mass[:, 0].sum():.6g} kg")
        if self.mode == "dynamic":
            lines.append(
                f"  char. length  min {min(float(b.pre.char_len.min()) for b in self.blocks):.4e} "
                "(wave speed " + ", ".join(f"{b.material.wave_speed:.1f}"
                                            for b in self._distinct_material_blocks())
                + " m/s)"
            )
        for b in self._distinct_material_blocks():
            eta = getattr(b.material, "eta", 0.0) or getattr(b.material, "mu", 0.0)
            if eta:
                lines.append(
                    f"  visc. regular. mu = {eta:.3e} ({eta / self.dt:.1f} x dt)"
                    + (f" [{b.name}]" if b.name else "")
                )
            extra = b.material.describe()
            if extra:
                lines.append(extra if not b.name else f"{extra}  [{b.name}]")
        return "\n".join(lines)

    def _distinct_material_blocks(self) -> List[Block]:
        """One block per material: a mixed-element mesh repeats the material
        per element type, and the summary should not."""
        seen, out = set(), []
        for b in self.blocks:
            if b.name not in seen:
                seen.add(b.name)
                out.append(b)
        return out

    # ------------------------------------------------------------------ #

    def run(self, on_output=None, output_every: int = 200,
            on_progress=None, progress_every: int = 500):
        """Run every analysis step in sequence.

        The steps share one model: displacement, velocity, material state,
        rotations and any erosion carry straight over. Only the boundary
        conditions change between them, which is the whole point -- a structure
        that has already cracked under its own weight is the structure the next
        step pushes on.
        """
        t_start = time.perf_counter()
        self.times, self.residuals, self.kinetic = [], [], []
        self.diagnostics = start_diagnostics(self.mode, self.backend, self.resident)
        from .diagnostics import resume_ledger
        resume_ledger(self)      # a restart carries the ledger on
        self.frame_equilibrium = None
        self._upload_state = True
        for k in range(getattr(self, "step_index", 0), len(self.steps)):
            self.apply_step(k)
            if len(self.steps) > 1:
                self.log(f"  --- step {k + 1}/{len(self.steps)}: "
                         f"{self.step_name} ---")
            self._run_step(on_output, output_every, on_progress, progress_every)
        status = "CONVERGED" if self.mode == "relaxation" else "COMPLETED_DYNAMIC"
        finish_diagnostics(self.diagnostics, status, step=self.step,
                           residual=self.residuals[-1] if self.residuals else None,
                           kinetic=self.kinetic[-1] if self.kinetic else None,
                           load_step=len(self.steps), eroded=self.n_eroded)
        self.wall_time = time.perf_counter() - t_start
        return self

    def _run_step(
        self,
        on_output: Optional[Callable] = None,
        output_every: int = 200,
        on_progress: Optional[Callable] = None,
        progress_every: int = 500,
    ):
        dt, m = self.dt, self.mass
        dt_m = dt / m
        if self.damp_kind == "viscous":
            c1 = 1.0 / (1.0 + 0.5 * self.zeta * dt)
            c2 = c1 * (1.0 - 0.5 * self.zeta * dt)
        else:
            c1 = c2 = 1.0
        free = ~self.fixed & ~self.frozen[:, None]
        self.step_cold = self.step + 1
        mid_step = self._step_prepare()

        n_steps, first_step, last_step, stage, stage_start = self._step_budget(mid_step)
        step_converged = self.mode == "dynamic"
        # the loading's progress on the previous iteration: a continuous
        # ramp may not declare convergence on the very step where it ends
        # (see the relaxation exit below)
        factor_prev = None

        # A restart whose checkpoint has already reached the budget iterates
        # zero times, and the reporting below then reads state the loop never
        # produced -- it surfaced as `UnboundLocalError: F_un`, a Python
        # internal name for what is really a plain scheduling mistake.
        #
        # And it is the natural mistake: a run ends with MAX_STEPS, so you
        # re-run the same command with `--restart` and nothing else changed.
        if self.step >= last_step:
            raise UserError(
                "RESTART_BUDGET_EXHAUSTED",
                f"this restart has no iterations left to run: the checkpoint "
                f"is already at step {self.step}, and solver.n_steps ({n_steps}) "
                f"allows up to step {last_step}",
                f"raise solver.n_steps above {self.step - first_step} and "
                "restart again -- the checkpoint keeps the state it reached, so "
                "nothing is repeated.",
            )

        F_un = None
        stop_now = False
        for step in range(self.step + 1, last_step + 1):
            factor, t = self._iteration_factor(step, first_step, n_steps, stage, stage_start, dt)

            want = (step % output_every == 0) or (step == last_step)
            # A staged run emits only when a load level has converged.  That
            # decision is made in the progress check below, so capture stress
            # on those candidate steps too.  Otherwise `output.every: 1000000`
            # would make a converged staged frame contain the stress from an
            # unrelated earlier iteration (or require every iteration to write
            # the stress buffer just to be safe).
            if self.load_steps and (
                step % progress_every == 0
                or (step - stage_start) >= self.max_steps_per_load
            ):
                want = True
            elif (not self.load_steps and self.mode == "relaxation"
                  and step % progress_every == 0):
                # A continuous relaxation run may converge long before its
                # nominal n_steps limit. Its only emitted frame is then this
                # progress candidate; request stress now so fields such as
                # von_mises are not silently omitted from the result.
                want = True
            # cold start: the quaternions begin at identity, so the first step
            # gets a full solve; afterwards the warm start makes 2 enough.
            iters = 30 if step == self.step_cold else self.rot_iters

            sig = None
            F_int = None
            # a dynamic step's load at its START, t_n, where the state is: read
            # at t_(n+1) it made the time history first-order in dt (1.1% at the
            # default step on a 200 Hz ground motion). The
            # boundary conditions below still take `factor`, the end of the step.
            if self.mode == "dynamic":
                physical_time = (step - 1 - first_step) * dt
                f_load = (step - 1 - first_step) / max(n_steps, 1)
            else:
                physical_time, f_load = None, factor
            self.last_external = self.external_force(f_load, physical_time)
            if self.resident:
                # nothing crosses the bus here: force, assembly, integrator and
                # boundary conditions all run on the device, and only the
                # kinetic energy comes back as a scalar
                self._force.force_resident(dt, rot_iters=iters, want_stress=want)
                self._force.integrate(factor, dt, c1, c2,
                                      1 if self._half else 0,
                                      min(1.0, (step - first_step)
                                          / max(self.stage_ramp, 1)))
                self._half = False
                self.step = step
                ke = self._force.kinetic_energy()
            else:
                sig, F_int, F_ext, F_un, bar_forces, ke, ke_audit = self._host_iteration(
                    step, first_step, factor, stage, dt, dt_m, m, c1, c2, iters, want)

            self._kinetic_damping(ke)

            # Erosion is checked periodically, not every step -- LS-DYNA's
            # dynamic relaxation checks convergence every NRCYCK = 250 cycles
            # for the same reason, and on the resident path the check costs a
            # state read across the bus.
            if self.erosion and step % self.erosion["every"] == 0:
                if self.resident:
                    for b in self.blocks:
                        b.state = b.force.get_state()
                    self.u = self._force.get_u()
                if self.check_erosion():
                    free = ~self.fixed & ~self.frozen[:, None]

            check = (step % progress_every == 0) or (step == last_step)
            if self.load_steps and (step - stage_start) >= self.max_steps_per_load:
                check = True
            if check:
                # Normalise by the internal force over ALL dofs, not just the
                # free ones.  Under pure displacement control F_ext is zero on
                # every free dof, so F_un[free] == -F_int[free] and a
                # free-dof-only ratio is identically 1 -- a convergence test
                # that can never fire.  Including the constrained dofs puts the
                # support reactions in the denominator, which is the right
                # scale for "how far from equilibrium is this".
                r, scale = self._progress_residual(step, stage, factor, free, F_un, F_int,
                                          F_ext if not self.resident else None,
                                          bar_forces if not self.resident else None, ke,
                                          ke_audit if not self.resident else None)
                if on_progress:
                    # a caller returning False is asking the solve to stop;
                    # the partial result is still written (api.run_case)
                    if on_progress(step, last_step, r / scale, ke,
                                   self.n_resets) is False:
                        # raised once this iteration's frame is written: the
                        # checkpoint is after it
                        stop_now = True
                if self.load_steps:
                    used = step - stage_start
                    # never declare convergence while still ramping into the level
                    done = used > self.stage_ramp and (
                        r / scale < self.tol or used >= self.max_steps_per_load
                    )
                    if done:
                        if used >= self.max_steps_per_load and r / scale >= self.tol:
                            message = (f"load step {stage}/{self.load_steps} reached its budget "
                                       f"({self.max_steps_per_load}) with residual {r/scale:.3e} "
                                       f"> tolerance {self.tol:.3e}")
                            location = ({"array": "device_residual", "index": [],
                                         "detail": "per-node location unavailable without a device-to-host field transfer"}
                                        if self.resident or F_un is None
                                        else worst_residual(F_un, free, 3))
                            finish_diagnostics(self.diagnostics, "MAX_STEPS", step=step,
                                               residual=r/scale, kinetic=ke, load_step=stage,
                                               message=message, location=location)
                            raise ConvergenceFailure(message, location=location)
                        self._commit_staged()
                        self.frame_equilibrium = True
                        self._emit(on_output, step, float(stage), sig, F_int)
                        if stage >= self.load_steps:
                            step_converged = True
                            self.log(
                                f"  all {self.load_steps} load steps converged "
                                f"({step} iterations, {self.n_resets} kinetic resets)"
                            )
                            break
                        stage += 1
                        stage_start = step
                        self._stage = (stage, stage_start)
                        # Start every load level from rest.  On the Warp path
                        # `self.v` is only the last host snapshot; the live
                        # leapfrog velocity remains in d_v and must be reset
                        # there too.  Forgetting that reset carries momentum
                        # across a cyclic reversal and can make the final
                        # return-to-zero level diverge although the preceding
                        # peak was equilibrated.
                        self.v.fill(0.0)
                        if self.resident:
                            self._force.zero_velocity()
                        self._half = True
                        self.ke_prev = 0.0
                elif (
                    self.mode == "relaxation"
                    and self._loading_settled(factor)
                    and (factor_prev is None or self._loading_settled(factor_prev))
                    and r / scale < self.tol
                ):
                    # `factor_prev`: the loading must have been settled on
                    # the previous iteration too, so the ramp ended BEFORE
                    # this check. On the check where it ends the state has
                    # been following a moving target until this very step,
                    # and a residual just under the tolerance there is the
                    # ramp's, not equilibrium's -- on a hex beam with an
                    # embedded bar it left the root reaction 0.1-0.24% off
                    # the load at tolerance 1e-4, written up for a long time
                    # as a fictitious-mass drift (tests/test_mixed_mesh.py).
                    # One more check interval takes that residual from 7e-5
                    # to 3e-8 and the reaction to 3e-7. Staged runs have the
                    # same rule in `stage_ramp`. (`factor_prev is None` is a
                    # resumed run whose first iteration is a check.)
                    self.log(
                        f"  converged at step {step}/{last_step} "
                        f"(residual {r / scale:.3e}, {self.n_resets} kinetic resets)"
                    )
                    self._commit_staged()
                    self.frame_equilibrium = True
                    self._emit(on_output, step, t, sig, F_int)
                    step_converged = True
                    break


            factor_prev = factor
            if want and not self.load_steps:
                self.frame_equilibrium = None if self.mode == "dynamic" else False
                self._emit(on_output, step, t, sig, F_int)
            if stop_now:
                raise StopAnalysis()

        if stop_now:
            raise StopAnalysis()
        if self.mode == "relaxation" and not step_converged:
            residual = self.residuals[-1] if self.residuals else None
            message = (f"relaxation reached {n_steps} iterations without convergence"
                       + (f": residual={residual:.3e} > tolerance={self.tol:.3e}" if residual is not None else ""))
            location = ({"array": "device_residual", "index": [],
                         "detail": "per-node location unavailable without a device-to-host field transfer"}
                        if self.resident or F_un is None
                                        else worst_residual(F_un, free, 3))
            finish_diagnostics(self.diagnostics, "MAX_STEPS", step=self.step, residual=residual,
                               kinetic=self.kinetic[-1] if self.kinetic else None,
                               load_step=self.step_index+1, message=message, location=location)
            raise ConvergenceFailure(message, location=location)
        if self.resident:
            self._sync_from_device(False)
        return self


    # -- the parts of `_run_step` (v1.18: complexity 123 before; the
    # same statements in the same order, taken out where they are one thing)

    def _step_prepare(self):
        """The device's fields and nodal data for this step, and a relaxation
        step started from rest. Returns whether a restart resumes mid-step.

        A restart that lands inside step k > 0 resumes it: the velocity is
        not zeroed and a restored state is still uploaded. Both used to key
        on `step_index == 0` (v1.0: a two-step J2 run restarted in
        its second step lost its velocity, 119% off in u)."""
        mid_step = bool(getattr(self, "_mid_step", False))
        upload = bool(getattr(self, "_upload_state", True))
        self._upload_state = False
        if self.resident:
            if not getattr(self, "_rebar_attached", False):
                for emb in self.rebar:
                    emb.device = self._force.attach_truss(emb)
                self._rebar_attached = True
            # the boundary conditions are what changes between steps, so this
            # is re-sent each time; the fields stay where they are
            self._force.set_nodal(self.mass, self.fixed.astype(np.int8),
                                  self.load_groups,
                                  [(idx, d, v, a) for idx, d, v, a, _motion in self.prescribed],
                                  # the device twin of `apply_bc`'s blend: the
                                  # same fix has to reach both paths, or the
                                  # GPU keeps the divergence the CPU no longer
                                  # has (this is the shape of defect
                                  # tests/test_reinforcement_fields.py exists
                                  # for -- one path fixed, one silently not)
                                  self._bc_start)
            if upload:
                self._force.set_u(self.u, self.v)
                for b in self.blocks:
                    b.force.set_state(b.state, b.quat)
            self._force.set_frozen(self.frozen)

        # each step starts from rest: whatever kinetic energy the previous one
        # ended with belongs to its loading, not to this one
        # -- IN A RELAXATION. A dynamic run's steps are one time history, and
        # zeroing the velocity at their boundary halved a truss's response to
        # a held load over two steps
        if self.step_index > 0 and not mid_step and self.mode == "relaxation":
            self.v.fill(0.0)
            if self.resident:
                self._force.zero_velocity()
            self._half = True
            self.ke_prev = 0.0
        return mid_step

    def _step_budget(self, mid_step):
        """(n_steps, first, last iteration, load level, the level's first
        iteration) of this step. Sets the staged-relaxation settings and
        `_stage`, and clears `_restored_stage`."""
        if self.mode == "dynamic":
            # the duration is total_time / dt; a solver-block `n_steps` is a
            # relaxation budget and must not truncate a physical run -- it
            # did, silently, and compressed the load history with it, since
            # the amplitude progress is per step (found by the free-flight
            # test of v0.1: a 1 ms pulse became 0.34 ms). A STEP
            # that names its own n_steps still gets it.
            n_steps = int(self.steps[self.step_index].get("n_steps") or self.n_steps)
        else:
            n_steps = int(self._step_solver("n_steps", self.n_steps))
        first_step = self.step_start
        last_step = first_step + n_steps

        # In a continuous ramp the intermediate output frames are NOT
        # equilibrium states -- the residual only reaches the tolerance at the
        # very end -- so a reaction-displacement curve read off them is inertia,
        # not structural response (measured on the beam case: 2341, 287, 2151,
        # 258 kN at successive frames).  Staged relaxation holds each load level
        # until the residual converges and emits ONE frame there, so every point
        # on the curve is a genuine equilibrium point.
        self.load_steps = self._step_solver("load_steps")
        self.max_steps_per_load = int(self._step_solver("max_steps_per_load"))
        self.stage_ramp = int(self._step_solver("stage_ramp_iterations")
                              or max(1, self.max_steps_per_load // 5))
        self.tol = float(self._step_solver("tolerance"))
        if self.load_steps:
            # `n_steps` is a useful safety ceiling for a continuous analysis,
            # but it cannot truncate a staged solve before every declared
            # equilibrium level has even had its allowed iteration budget.
            # This surfaced on refined RC meshes: 64 levels x 8000 iterations
            # needs 512k steps, while the legacy model-wide n_steps was 400k;
            # the loop silently returned after level 50 with a plausible HDF5
            # file.  A staged run's explicit contract is load_steps times its
            # per-level budget, so reserve at least that many iterations.
            n_steps = max(n_steps, int(self.load_steps) * self.max_steps_per_load)
            last_step = first_step + n_steps
        stage, stage_start = (1, self.step) if self.load_steps else (None, None)
        saved_stage = getattr(self, "_restored_stage", None)
        if self.load_steps and mid_step and saved_stage is not None:
            # the level the run was in, not level 1 again: a plastic model
            # would have been unloaded
            stage, stage_start = saved_stage
        self._restored_stage = None
        self._stage = (stage, stage_start)
        return n_steps, first_step, last_step, stage, stage_start

    def _iteration_factor(self, step, first_step, n_steps, stage, stage_start, dt):
        """(the loading's progress, the frame's time) at iteration `step`."""
        if self.load_steps:
            # Ramp INTO each level rather than jumping to it.  A prescribed
            # displacement applied as a step change is a shock: on the beam
            # case, 4 mm imposed in one iteration on an Underwood-mass system
            # with dt = 1 diverged to 1e100 within the first load level.
            # Ramping over the first `ramp` iterations costs nothing, since
            # the level then has to be relaxed to equilibrium anyway.
            f1 = stage / self.load_steps
            f0 = (stage - 1) / self.load_steps
            frac = min(1.0, (step - stage_start) / max(self.stage_ramp, 1))
            factor = f0 + (f1 - f0) * frac
            t = self.step_index + f1
        else:
            # progress through THIS step, 0 to 1 -- what every amplitude
            # curve is written against (see core/amplitude.py)
            factor = (step - first_step) / max(n_steps, 1)
            # the frame's `time`: seconds in a dynamic run, as every other
            # solver writes it (it was step + progress: a 2 ms run ended at
            # time = 1.0); step + progress in a relaxation
            t = (self.step_index * self.total_time + (step - first_step) * dt
                 if self.mode == "dynamic" else self.step_index + factor)
        return factor, t

    def _host_iteration(self, step, first_step, factor, stage, dt, dt_m, m, c1, c2, iters, want):
        """One iteration on the host: the element blocks, bars and contact
        forces, the ledger, the leapfrog update of the host and the bonded
        bars, the kinetic energy. Returns (stress, F_int, F_ext, F_un, the
        bonded bars' own forces, kinetic energy at n + 1/2, at n for the
        audit). Sets `self.step`, commits the bonded bars' history in a
        dynamic run, and raises NonFiniteFailure on a non-finite state."""
        # One evaluation per element type.  Each block owns its own
        # kernel, connectivity and material state; the assembly is a
        # sum over blocks in a fixed order, so the result is still
        # bit-reproducible.
        F_int = np.zeros((self.nn, 3))
        sigs = []
        for b in self.blocks:
            f_e, sig_b = b.force(self.u, b.quat, b.state, dt,
                                 rot_iters=iters, want_stress=want)
            if (not np.isfinite(f_e).all()
                    or not np.isfinite(b.state).all()
                    or (sig_b is not None and not np.isfinite(sig_b).all())):
                location = first_nonfinite(
                    element_force=f_e, material_state=b.state,
                    stress=sig_b if sig_b is not None else np.array([0.0]))
                location["element_block"] = b.label
                finish_diagnostics(self.diagnostics, "NONFINITE", step=step,
                                   load_step=stage or self.step_index+1,
                                   message="non-finite element/material response", location=location)
                raise NonFiniteFailure("non-finite element/material response; inspect diagnostics location", location=location)
            F_int += b.force.assemble(f_e)
            sigs.append(sig_b)
        sig = (sigs[0] if len(sigs) == 1
               else (None if sigs[0] is None else sigs))
        for emb in self.rebar:
            nb = emb.__dict__.get("_numba_force", False)
            if nb is False:
                nb = emb._numba_force = self._bar_kernel(emb)
            if nb is not None:
                # compiled, into F_int in place
                sb = nb(self.u, dt, F_int, want_stress=want)
            else:
                fb, sb = emb.force(self.u, dt, want_stress=want)
                F_int += scatter_add(fb, emb.conn, self.nn)
            if want:
                emb.last_stress = sb
        # A bonded bar returns two residuals: what it does to the host,
        # which joins F_int, and what acts on its OWN dofs, which the
        # bar's leapfrog below consumes. Computing them in one call is
        # what makes the pair equal and opposite by construction rather
        # than by a second calculation that could drift.
        bar_forces = []
        if self.bonded:
            # every evaluation integrates from the last COMMITTED
            # point; the iteration path leaves nothing in the state
            self.staged.restore_trial()
        for bt in self.bonded:
            f_bar, f_host, sb = bt.force(self.u, dt=dt, want_stress=want)
            F_int += bt.scatter_host(f_host, self.nn)
            bar_forces.append(f_bar)
            if want:
                bt.last_stress = sb
        if self.contact is not None:      # rigid surfaces
            F_int = F_int - self.contact.force(
                self.mesh.nodes + self.u, self.v if self.mode == "dynamic" else None,
                self.mass.min(axis=1), dt)
        F_ext = self.last_external
        F_un = F_ext - F_int
        # the energy ledger takes the state and the forces of the SAME
        # instant, before the update below moves u. It used to be fed
        # after the update, pairing the forces of step n with the
        # displacement increment of step n+1 -- a one-step lag whose
        # trapezoidal error drifts on any oscillating mode: on the
        # floating beam (case_19, flexible) the internal work ran to
        # -18 J of a 91 J budget while the kinetic energy stayed put
        self.diagnostics["energy"].update(self.u, F_ext, F_int)
        # Applied on `alpha`, not on `damp_kind`: kinetic damping has
        # a blind spot for monotonically growing modes and this is the
        # term that covers it. See the `rayleigh_alpha` comment in the
        # setup for the measurement that put it here.
        if self.rayleigh_alpha:
            F_un -= self.rayleigh_alpha * m * self.v
        # the velocity the damping terms below act on; `self.v` is
        # rebound, never written in place, so no copy is needed
        v_old = self.v
        restart_step = self._half

        # --- leapfrog velocity update -----------------------------
        # A central-difference restart must use HALF the acceleration:
        #     v_{1/2} = 0.5 a dt,  not  v_{1/2} = a dt.
        # Written in displacement form this is easy to get wrong, and
        # doing so injects energy on every restart.  With kinetic
        # damping restarting every few steps that compounds into
        # divergence -- the displacement-form version of this loop
        # reached 1e140 by step 15k.
        # Read the restart flag BEFORE the host branch clears it: the
        # bars need the same half-acceleration on the same steps, and a
        # second flag kept in sync by hand would have to be reset at
        # every one of the three places that restart the integrator.
        restart = self._half
        a = F_un * dt_m
        if self._half:
            self.v = 0.5 * a
            self._half = False
        else:
            self.v = c2 * self.v + c1 * a

        u_next = self.u + self.v * dt
        # a frozen node holds the displacement it had reached
        u_next[self.frozen] = self.u[self.frozen]
        self.apply_bc(u_next, factor,
                      min(1.0, (step - first_step)
                          / max(self.stage_ramp, 1)))
        # BCs move nodes directly, so keep v consistent
        self.v = (u_next - self.u) / dt
        self.u = u_next
        # what the damping did over this step, which the ledger
        # booked as error until v0.4: the explicit alpha M v
        # term, and the implicit-midpoint viscous update, whose force
        # is zeta m (v_old + v_new) / 2 (not applied on a restart)
        if self.rayleigh_alpha or (self.damp_kind == "viscous" and not restart_step):
            fd = 0.0
            if self.rayleigh_alpha:
                fd = fd + self.rayleigh_alpha * m * v_old
            if self.damp_kind == "viscous" and not restart_step:
                fd = fd + 0.5 * self.zeta * m * (v_old + self.v)
            self.diagnostics["energy"].damping(fd, self.v * dt)

        # The bars, on the identical scheme. No boundary conditions and
        # nothing frozen: a bar node is held only by its own bond, which
        # is the point of giving it dofs at all.
        for i, bt in enumerate(self.bonded):
            m_b = self.mass_bar[i]
            # `dt *`, AND IT WAS MISSING FOR A WHOLE STAGE OF WORK.
            #   The host above uses `a = F_un * dt_m` with
            #   `dt_m = dt / m`, so what both branches call `a` is a
            #   VELOCITY INCREMENT, not an acceleration -- which is
            #   why `v = c2 v + c1 a` and `u += v dt` are dimensionally
            #   right. The bar's version was `-f / m_b`, a true
            #   acceleration, short by exactly one factor of dt.
            #
            #   Under relaxation dt is 1.0 by construction, so the two
            #   are the same number and nothing showed: every bonded
            #   test and every bonded result to date is a relaxation
            #   run. In dynamic mode dt is 6.8e-6 s on this model, the
            #   bar's velocity update came out 1.5e5 times too large,
            #   and the run reached 1e146 mm in under forty steps --
            #   in the SAME forty steps whether dt_safety was 0.6 or
            #   0.03, which is the tell: an instability's onset scales
            #   with the time step, a wrong force does not.
            #
            #   tests/test_bonded_dynamic.py is the coverage that was
            #   missing, and it is missing no longer.
            f_un_b = -bar_forces[i]             # F_ext on a bar is zero
            if self.rayleigh_alpha:
                f_un_b = f_un_b - self.rayleigh_alpha * m_b * bt.v_bar
            a_b = f_un_b * dt / m_b
            if restart:
                bt.v_bar = 0.5 * a_b
            else:
                bt.v_bar = c2 * bt.v_bar + c1 * a_b
            bt.u_bar = bt.u_bar + bt.v_bar * dt
            bt.tie(self.u)              # anchored ends follow the host
        if self.bonded and self.mode == "dynamic":
            # in dynamics every step IS a physical point: the history
            # advances here, and the restore above is then a no-op
            self.staged.commit()
        self.step = step
        # THE BARS ARE PART OF THE STATE, AND FOR A WHOLE STAGE OF WORK
        # THEY WERE NOT PART OF THIS NUMBER. Kinetic damping decides
        # when to stop dead from the peak of this energy; with the bars
        # left out it watched the host settle against bars that were
        # still moving, and reset on the host's schedule rather than
        # the system's. The convergence residual below had the same
        # blind spot. Measured on case_12b, first load step: declared
        # converged at 37.1 kN with a 71.7% left-right reaction split
        # on a symmetric beam; run to a real equilibrium it is 66.8 kN
        # and 1.1%.
        # the leapfrog velocity sits at n + 1/2; the ledger's work terms
        # sit at n. For the audit the kinetic energy at n is the mean
        # of the two half-step values, second order; the damping logic
        # below keeps the half-step number, which is what it watches
        ke_half_prev = getattr(self, "_ke_half", 0.0)
        ke = 0.5 * float(np.sum(m * self.v * self.v))
        for i, bt in enumerate(self.bonded):
            ke += 0.5 * float(np.sum(self.mass_bar[i] * bt.v_bar * bt.v_bar))
        self._ke_half = ke
        ke_audit = ke if restart else 0.5 * (ke_half_prev + ke)
        if not np.isfinite(ke) or not np.isfinite(self.u).all():
            location = first_nonfinite(displacement=self.u, velocity=self.v)
            finish_diagnostics(self.diagnostics, "NONFINITE", step=step,
                               load_step=stage or self.step_index+1,
                               message="non-finite nodal integration state", location=location)
            raise NonFiniteFailure("non-finite nodal integration state; inspect diagnostics location", location=location)
        return sig, F_int, F_ext, F_un, bar_forces, ke, ke_audit

    def _progress_residual(self, step, stage, factor, free, F_un, F_int, F_ext, bar_forces, ke, ke_audit):
        """(residual, its scale) at a progress check. Appends the relative
        residual to `residuals` and the kinetic energy to `kinetic`, takes the
        energy ledger's snapshot; a non-finite one ends the run."""
        if self.resident:
            r, n_int, n_ext = self._force.residual(factor)
            scale = max(float(n_int), float(n_ext), 1e-30)
            r = float(r)
        else:
            # The bar dofs are free dofs with F_ext = 0, so their
            # residual is the bar force itself, and it belongs in the
            # same norm as the host's. Without it the host can sit in
            # equilibrium against a bar that is still moving, and the
            # load step is declared converged while the interface is
            # mid-transient -- see the kinetic-energy note above for
            # the measurement.
            r2 = float(np.sum(F_un[free] ** 2))
            for f_b in bar_forces:
                r2 += float(np.sum(f_b ** 2))
            r = float(np.sqrt(r2))
            scale = max(
                float(np.linalg.norm(F_int)),
                float(np.linalg.norm(F_ext)),
                1e-30,
            )
        self.residuals.append(r / scale)
        self.kinetic.append(ke)
        self.diagnostics["energy"].snapshot(step, ke_audit if not self.resident else ke)
        if not np.isfinite(r) or not np.isfinite(scale) or not np.isfinite(ke):
            location = {"array": "device_scalar" if self.resident else "residual", "index": []}
            finish_diagnostics(self.diagnostics, "NONFINITE", step=step,
                               residual=r/scale if np.isfinite(r/scale) else None,
                               kinetic=ke if np.isfinite(ke) else None,
                               load_step=stage or self.step_index+1,
                               message="non-finite convergence diagnostic", location=location)
            raise NonFiniteFailure("non-finite convergence diagnostic", location=location)
        return r, scale

    def _kinetic_damping(self, ke):
        """Kinetic damping: at a peak of the kinetic energy, stop dead and
        restart the integrator."""
        if self.damp_kind == "kinetic":
            if ke < self.ke_prev:
                # kinetic energy has peaked: stop dead and restart
                self.diagnostics["energy"].dissipate(self.ke_prev)
                if self.resident:
                    self._force.zero_velocity()
                else:
                    self.v.fill(0.0)
                self._half = True
                self.n_resets += 1
                self.ke_prev = 0.0
            else:
                self.ke_prev = ke

    # ------------------------------------------------------------------ #

    def _sync_from_device(self, want_stress):
        """Bring a consistent device snapshot back for output or restart.

        The Warp loop owns displacement, velocity, rotations and material
        state.  Leaving any of them on the device at an output/checkpoint
        boundary creates a restart file that looks valid but resumes from an
        earlier host copy.  Internal force and stress are returned to the
        caller for monitor and field output respectively.
        """
        self.u = self._force.get_u()
        self.v = self._force.get_v()
        sigs = []
        for b in self.blocks:
            b.quat = b.force.get_quat().astype(np.float64)
            b.state = b.force.get_state().astype(np.float64)
            sigs.append(b.force.get_stress() if want_stress else None)
        sig = None
        if want_stress:
            sig = sigs[0] if len(sigs) == 1 else sigs
        return sig, self._force.get_F_int()

    def _emit(self, on_output, step, t, sig, F_int):
        if self.resident:
            sig, F_int = self._sync_from_device(sig is None)
            # the bars live on the device too; their host state is only a
            # starting value until it is read back
            for emb in self.rebar:
                dev = getattr(emb, "device", None)
                if dev is not None:
                    emb.state = dev.d_state.numpy().astype(np.float64)
        self.last_reaction = np.zeros((self.nn, 3))
        reaction = F_int-self.last_external
        constrained = self.fixed | self.frozen[:, None]
        self.last_reaction[constrained] = reaction[constrained]
        # the per-element scalars, built once and shared by every element
        # monitor: `von_mises` and the material's own fields are not free, and
        # a case may reasonably have several monitors reading them
        efields = None
        if any(m.kind == "element" for m in self.monitors.values()):
            efields = self.nodal_fields(sig)
            missing = [n for n, m in self.monitors.items()
                       if m.kind == "element" and m.field not in efields]
            if missing:
                raise ValueError(
                    f"monitor(s) {', '.join(sorted(missing))} read a field the "
                    "solver did not produce on this frame. `stress` and "
                    "`von_mises` need the stress buffer, which is only filled "
                    "on output steps -- this is a bug, not a case error")
        for name, mon in self.monitors.items():
            if mon.kind == "element":
                mon.values.append(self._element_series(mon, efields))
                continue
            if mon.kind == "bar":
                mon.values.append(self._bar_series(mon))
                continue
            if mon.kind == "reaction":
                # The same array the `reaction_force` FIELD is built from, so
                # ParaView and the monitor cannot disagree about the answer.
                #
                # This used to be `-sum(F_int[...])`, which differed from the
                # field in two ways: it dropped the -F_ext term (wrong whenever
                # a load acts directly on a constrained node) and it carried
                # the opposite sign. A support holding a downward -1000 N load
                # was therefore reported as -1000 N when the reaction it exerts
                # is upward, +1000 N.
                #
                # Nothing caught it because every verification contract that
                # checks a reaction runs a structural element -- beam, truss,
                # shell -- and those solvers already used the right sign. The
                # one contract covering this path, case_06, had never been run.
                # over the UNION of the regions: summed region by region, a
                # node two regions share counted twice
                idx_r = np.unique(np.concatenate([self._region(r) for r in mon.regions]))
                v = float(self.last_reaction[idx_r, mon.dof].sum())
            else:
                # every region the monitor names, as the frame solvers read
                # them -- only the first was read here (found in review,
                # v1.0)
                idx = np.unique(np.concatenate([self._region(r) for r in mon.regions]))
                v = float(self.u[idx, mon.dof].mean())
                if mon.relative_to:
                    # the motion of one region seen from another: a free
                    # body's rotation shows in the ends' relative motion
                    # while its mass centre drifts
                    v -= float(self.u[self._region(mon.relative_to), mon.dof].mean())
            mon.values.append(v)
        self.times.append(t)
        if on_output is not None:
            on_output(self, step, t, sig)

    # ------------------------------------------------------------------ #

    def _nodal_fields(self, sig: Optional[np.ndarray], recover_stress: bool = False
                     ) -> Dict[str, np.ndarray]:
        """Fields for output: nodal displacement plus element-averaged scalars."""
        out = {"displacement": self.u.copy()}
        if hasattr(self, "last_reaction"): out["reaction_force"] = self.last_reaction
        if self.erosion is not None:
            out["active"] = self._scatter_elements(
                [b.pre.active.astype(np.float64) for b in self.blocks])
        if len(self.blocks) > 1:
            # which material block each element is in, for ParaView's
            # threshold filter; the summary lists the names in this order
            out["block"] = self._scatter_elements(
                [np.full(b.n_elem, float(i)) for i, b in enumerate(self.blocks)])
        if sig is not None:
            # blocks may differ in integration-point count, so the average is
            # taken per block and the ELEMENT values scatter into the global
            # element order the writer and every other output already use
            sig_e = (self._scatter_elements([s.mean(axis=1) for s in sig])
                     if isinstance(sig, list) else sig.mean(axis=1))
            out["von_mises"] = von_mises(sig_e)
            out["stress"] = sig_e
            if recover_stress:
                sig_n = self.recover_nodal_stress(sig)
                out["stress_nodal"] = sig_n
                out["sigma_xx"] = sig_n[:, 0]
                out["sigma_yy"] = sig_n[:, 1]
                out["sigma_zz"] = sig_n[:, 2]
        # the union over blocks: an elastic pad has no damage, and reports
        # zero of it rather than making the field disappear from the beam
        per_block = [b.material.output_fields(b.state) for b in self.blocks]
        keys: List[str] = []
        for d in per_block:
            keys += [k for k in d if k not in keys]
        for k in keys:
            out[k] = self._scatter_elements([
                (d[k].reshape(b.n_elem, b.n_gp).mean(axis=1) if k in d
                 else np.zeros(b.n_elem))
                for d, b in zip(per_block, self.blocks)])
        return out

    def _scatter_elements(self, per_block: List[np.ndarray]) -> np.ndarray:
        """Per-block element arrays -> one array in the global element order."""
        first = np.asarray(per_block[0])
        out = np.zeros((self.nel,) + first.shape[1:], dtype=first.dtype)
        for b, v in zip(self.blocks, per_block):
            out[b.elem_ids] = v
        return out

    def recover_nodal_stress(self, sig: np.ndarray) -> np.ndarray:
        """Volume-weighted linear patch recovery of Gauss stress to nodes."""
        ngp, nne = self.n_gp, self.pre.n_nodes
        pts = self.pre.gp_coords.reshape(-1, 3)
        vals = np.asarray(sig, dtype=float).reshape(-1, 6)
        weights = self.pre.w_detJ.reshape(-1)
        conn = self.mesh.elements
        order = np.argsort(conn.ravel(), kind="stable")
        node_ids = conn.ravel()[order]
        starts = np.searchsorted(node_ids, np.arange(self.nn), side="left")
        ends = np.searchsorted(node_ids, np.arange(self.nn), side="right")
        recovered = np.empty((self.nn, 6), dtype=float)
        gp_local = np.arange(ngp)
        for node in range(self.nn):
            elem_ids = order[starts[node]:ends[node]] // nne
            sample = (elem_ids[:, None] * ngp + gp_local).ravel()
            w = weights[sample]
            good = w > 0.0
            sample, w = sample[good], w[good]
            if not len(sample):
                recovered[node] = np.nan
                continue
            dx = pts[sample] - self.mesh.nodes[node]
            scale = max(float(np.linalg.norm(dx, axis=1).mean()), 1e-30)
            A = np.column_stack([np.ones(len(sample)), dx / scale])
            root_w = np.sqrt(w / w.sum())
            if len(sample) >= 4 and np.linalg.matrix_rank(A * root_w[:, None]) == 4:
                recovered[node] = np.linalg.lstsq(A * root_w[:, None],
                                                   vals[sample] * root_w[:, None],
                                                   rcond=None)[0][0]
            else:
                recovered[node] = np.average(vals[sample], axis=0, weights=w)
        return recovered

    def _checkpoint(self) -> Dict[str, np.ndarray]:
        """The state a restart resumes from, COPIED: it used to hand out the
        live arrays, so a checkpoint held in memory changed as the run went
        on."""
        ck = self._checkpoint_live()
        stage = getattr(self, "_stage", (None, None))
        if stage[0] is not None:
            ck["stage"] = np.array([int(stage[0]), int(stage[1])])
        for i, a in enumerate(getattr(self, "_bc_start", None) or []):
            ck[f"bc_start{i}"] = np.asarray(a, dtype=float)
        if getattr(self, "erosion", None):
            for i, b in enumerate(self.blocks):
                ck[f"active{i}"] = b.pre.active
            ck["frozen"] = self.frozen
            ck["n_eroded"] = np.array([self.n_eroded])
        ck["energy"] = self.diagnostics["energy"].checkpoint() if hasattr(self, "diagnostics") else np.zeros(0)
        # the last converged bond state a staged level re-drives from, which
        # inside a level is not the live one
        for j, item in enumerate(getattr(self, "staged", None) or []):
            if hasattr(item, "committed"):
                ck[f"staged_committed{j}"] = item.committed
        return {k: np.array(v, copy=True) for k, v in ck.items()}

    def _checkpoint_live(self) -> Dict[str, np.ndarray]:
        return {
            "u": self.u, "v": self.v,
            # one block writes the flat layout every existing restart file
            # has; several write one dataset per block, since their state
            # widths need not agree
            **({"quat": self.blocks[0].quat, "state": self.blocks[0].state}
               if len(self.blocks) == 1 else
               {k: v for i, b in enumerate(self.blocks)
                for k, v in ((f"quat{i}", b.quat), (f"state{i}", b.state))}),
            "step": np.array([self.step]),
            "ke_prev": np.array([self.ke_prev]),
            "half": np.array([1 if self._half else 0]),
            "step_index": np.array([self.step_index]),
            "step_start": np.array([self.step_start]),
            "material_signature": np.asarray(self.material_signature().encode()),
            **{f"rebar{i}": emb.state for i, emb in enumerate(self.rebar)},
            # a bonded bar carries three things a restart cannot rebuild: where
            # it is, how fast it is moving, and how far it has already slipped
            **{f"bonded{i}_u": bt.u_bar for i, bt in enumerate(self.bonded)},
            **{f"bonded{i}_v": bt.v_bar for i, bt in enumerate(self.bonded)},
            **{f"bonded{i}_s": bt.state for i, bt in enumerate(self.bonded)},
            **{f"bonded{i}_b": bt.bond_state for i, bt in enumerate(self.bonded)},
        }

    def material_signature(self) -> str:
        """What a checkpoint must match: every block's material-point contract."""
        return "+".join(b.points.signature for b in self.blocks)

    def _restore(self, ck: Dict[str, np.ndarray]):
        self.u = np.array(ck["u"], dtype=float, order="C")      # copies: the run
        self.v = np.array(ck["v"], dtype=float, order="C")      # must not write into ck
        if "material_signature" in ck:
            saved = bytes(np.asarray(ck["material_signature"]).item()).decode()
            if saved != self.material_signature():
                raise ValueError("checkpoint material-point contract does not match this model")
        if len(self.blocks) == 1:
            b = self.blocks[0]
            b.state = validate_restored_state(ck["state"], b.points, b.n_elem * b.n_gp)
            b.quat = np.ascontiguousarray(ck["quat"])
        elif "state" in ck:
            # a restart written before blocks had their own datasets: one
            # flat array in block-major order, every block the same width
            n_gp_total = int(sum(b.n_elem * b.n_gp for b in self.blocks))
            state = validate_restored_state(ck["state"], self.blocks[0].points, n_gp_total)
            quat = np.ascontiguousarray(ck["quat"])
            eo = so = 0
            for b in self.blocks:
                b.quat = np.ascontiguousarray(quat[eo:eo + b.n_elem])
                b.state = np.ascontiguousarray(state[so:so + b.n_elem * b.n_gp])
                eo += b.n_elem
                so += b.n_elem * b.n_gp
        else:
            for i, b in enumerate(self.blocks):
                if f"state{i}" not in ck:
                    raise ValueError(f"checkpoint has no state for block {i} ({b.label}); "
                                     "it was written by a model with a different block layout")
                b.state = validate_restored_state(ck[f"state{i}"], b.points, b.n_elem * b.n_gp)
                b.quat = np.ascontiguousarray(ck[f"quat{i}"])
        self.step = int(ck["step"][0])
        self.ke_prev = float(ck["ke_prev"][0])
        self._half = bool(ck["half"][0])
        if "step_index" in ck:
            self.step_index = int(ck["step_index"][0])
            self.step_start = int(ck["step_start"][0])
            self._resuming = True
        if "stage" in ck:
            self._restored_stage = (int(ck["stage"][0]), int(ck["stage"][1]))
        n_bc = len([k for k in ck if k.startswith("bc_start")])
        self._restored_bc_start = [np.asarray(ck[f"bc_start{i}"], float) for i in range(n_bc)] or None
        if "frozen" in ck and getattr(self, "erosion", None):
            # eroded elements stay eroded (they came back: v1.0)
            for i, b in enumerate(self.blocks):
                if f"active{i}" in ck:
                    continuum.erode(b.pre, np.asarray(ck[f"active{i}"]) == 0)
            self.frozen = np.asarray(ck["frozen"], dtype=bool).copy()
            self.n_eroded = int(ck["n_eroded"][0])
            if self.resident:
                for b in self.blocks:
                    b.force.refresh_weights(b.pre)
                self._force.set_frozen(self.frozen)
        self._restored_energy = ck.get("energy")
        for i, emb in enumerate(self.rebar):
            key = f"rebar{i}"
            if key in ck:
                emb.state = np.ascontiguousarray(ck[key])
        for i, bt in enumerate(self.bonded):
            for suffix, attr in (("u", "u_bar"), ("v", "v_bar"),
                                 ("s", "state"), ("b", "bond_state")):
                key = f"bonded{i}_{suffix}"
                if key in ck:
                    # WRITE THROUGH, do not rebind: `bond_state` is held by a
                    # staging TrialState (and by a CallbackState for the
                    # cyclic law), and a rebound attribute would leave those
                    # restoring and committing an orphan while the kernels
                    # read the new array.
                    target = getattr(bt, attr)
                    target[...] = np.asarray(ck[key], dtype=target.dtype)
        if self.bonded:
            self.staged.commit()          # the restored point IS a committed one
            for j, item in enumerate(self.staged):
                key = f"staged_committed{j}"
                if key in ck and hasattr(item, "committed") and "stage" in ck:
                    # ... unless it was taken inside a level
                    item.committed[...] = np.asarray(ck[key], dtype=item.committed.dtype)
