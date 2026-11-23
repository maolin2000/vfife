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
from dataclasses import dataclass
from typing import Dict, List
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
from . import creep
from . import fracture
from .member_failure import Breaking
from . import relaxation
from ..elements import connector
from . import topology
from .dof import (DofLayout, ROTATION, SparseAssembler, TRANSLATION,
                  repeated_rotation)
from ..elements.fiber_section import (box_fibers, circle_fibers,
                                      rect_fibers, tube_fibers)
from .beam_family import BeamFamily
from .capabilities import refuse

# one monitor record for every solver (core/monitors)
from .monitors import Monitor  # noqa: F401  (the name this module exported)


@dataclass
class _SectionLayout:
    """How one fibre section's row is made of material groups.

    `conc_groups` are the indices into `solver.materials` of the cover and
    (when confined) core law, `rebar_groups` one per bar in the section's
    bar order, and `fiber_group[f]` the group of column f of the section's
    row (concrete fibres in geometric order, then the bars). `n_c` and `n_s`
    are the row's real widths before padding.
    """
    conc_groups: list
    rebar_groups: list
    fiber_group: np.ndarray
    n_c: int
    n_s: int
    confined: bool
    rebars: list
    entry: dict
    #: (slot j among the section's steel rows, tendon spec) per prestressing
    #: tendon; the tendon's rebar group is rebar_groups[j]
    tendons: list

# Gauss along the beam axis for section integration.  2 points capture the
# bending-moment gradient; the fiber loop is cheap.
_GAUSS_S = np.array([0.211324865405187, 0.788675134594813])  # s = x/L in [0,1]
_GAUSS_W = np.array([0.5, 0.5])  # weights sum to 1, scaled by L later
_LOBATTO3_S = np.array([0.0, 0.5, 1.0])
_LOBATTO3_W = np.array([1.0/6.0, 2.0/3.0, 1.0/6.0])
# Gauss-Lobatto with 4 and 5 stations on [0, 1] (nodes +-1, +-1/sqrt5 and
# +-1, +-sqrt(3/7), 0 on [-1, 1]; weights 1/6, 5/6 and 1/10, 49/90, 32/45,
# halved). A plastic hinge is a station: five of them put one at the ends,
# the middle and a quarter-way in, so a member 2 h long resolves a hinge
# of 0.5 h without being split.
_LOBATTO4_S = np.array([0.0, 0.5 * (1.0 - 1.0 / np.sqrt(5.0)), 0.5 * (1.0 + 1.0 / np.sqrt(5.0)), 1.0])
_LOBATTO4_W = np.array([1.0 / 12.0, 5.0 / 12.0, 5.0 / 12.0, 1.0 / 12.0])
_LOBATTO5_S = np.array([0.0, 0.5 * (1.0 - np.sqrt(3.0 / 7.0)), 0.5, 0.5 * (1.0 + np.sqrt(3.0 / 7.0)), 1.0])
_LOBATTO5_W = np.array([1.0 / 20.0, 49.0 / 180.0, 16.0 / 45.0, 49.0 / 180.0, 1.0 / 20.0])
INTEGRATION_RULES = {"gauss2": (_GAUSS_S, _GAUSS_W), "lobatto3": (_LOBATTO3_S, _LOBATTO3_W),
                     "lobatto4": (_LOBATTO4_S, _LOBATTO4_W), "lobatto5": (_LOBATTO5_S, _LOBATTO5_W)}


def _piecewise_parabola(sp, v, x):
    """A tendon's `interpolation: quadratic`: one parabola
    through each three consecutive points 0-1-2, 2-3-4, ... -- the way a
    bridge tendon is specified (anchor, low point, anchor; then over the
    pier, low point, ...). Needs an odd number of points (checked on input).
    Outside the first and last point the end parabolas continue."""
    sp = np.asarray(sp, float); v = np.asarray(v, float)
    x = np.asarray(x, float)
    m = (len(sp) - 1) // 2
    knots = sp[0::2]
    seg = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, m - 1)
    x0, x1, x2 = sp[2 * seg], sp[2 * seg + 1], sp[2 * seg + 2]
    v0, v1, v2 = v[2 * seg], v[2 * seg + 1], v[2 * seg + 2]
    return (v0 * (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2))
            + v1 * (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2))
            + v2 * (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1)))


class FiberBeamSolver(BeamFamily):
    BEAM_TYPES = frozenset(("beam2_fiber",))
    #: the bond graph of a model whose members split (`_setup_fracture`)
    bonds = None
    #: a removal in the step that ended set something loose, to be looked
    #: at between two steps (`apply_pending_splits`)
    _pending_debris = False
    ELEMENT_NAME = "beam2_fiber"
    HOLD_PARKED = True
    SOLVER_WHAT = "fiber beam"
    STAGED_TRIALS = True
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
        #: the next velocity update is a central-difference RESTART and
        #: takes half the acceleration; set again wherever v is zeroed
        self._half = True
        self.ke_prev = 0.0
        self.n_resets = 0
        self.step = self.step_index = self.step_start = 0
        self.time = 0.0
        self.rebar = []
        self._pending_splits = getattr(self, "_pending_splits", [])
        if self.plan is not None:
            analysis_steps.install(self)

    def _reserve_split_particles(self):
        """Park a spare particle on every element end, before anything is sized.

        THE WHOLE OF THE PRE-ALLOCATION is here, and it happens between the
        regions and everything else for two reasons that are both about not
        being clever later:

        REGIONS FIRST, so that a spare is in no region. A spare sits exactly on
        top of the particle it may one day replace, so a geometric selector --
        `where: {x: 0}` -- would pick it up, and the fragment that finally uses
        it would be born clamped to the ground it was supposed to fall away
        from. Resolving the regions before the spares exist is the only way to
        get that right that does not involve special cases in the selector.

        EVERYTHING ELSE AFTER, so that no array is ever resized at run time:
        `q`, `v`, the mass, the orientations, the reference stiffness and the
        output mesh are all built at the final size. That is not a performance
        argument -- a handful of fractures could afford a reallocation -- it is
        that `self.orient` and `self.v` are REBOUND in the time loop (a
        multiplicative orientation update, a Rayleigh velocity update), so a
        design where the live arrays are windows onto a bigger buffer would
        lose the window silently on the first step after a split.

        The reservation is exactly two per element, indexed `base + 2e + k`,
        because an element end can separate once. There is no capacity to
        choose and none to run out of. The cost is 2 x n_elements particles'
        worth of state -- for a frame, about three times the original -- and
        it buys a run whose memory is known at setup.
        """
        self._spare_base = None
        spec = (self.p.get("element") or {}).get("fracture")
        if not spec or spec is True or not dict(spec).get("split"):
            return
        if self.p.solver.mode != "dynamic":
            from .diagnostics import UserError
            raise UserError(
                "SPLIT_NEEDS_DYNAMIC",
                "element.fracture.split needs solver.mode = dynamic",
                "a fragment that separates has nothing to relax to: it "
                "accelerates. Use the dynamic solver, or turn the split off "
                "and keep the deactivation-only fracture of v0.1")
        self._spare_base = self.nn
        # (n_elements, 2), element-major: the spare for end k of element e is
        # parked ON that end's particle, so its reference coordinates, and
        # therefore the element's undeformed geometry, do not change when the
        # bond moves onto it.
        self._spare_of = (self._spare_base
                          + 2 * np.arange(self.nel, dtype=np.int64)[:, None]
                          + np.arange(2, dtype=np.int64)[None, :])
        parked = self.mesh.nodes[self.mesh.elements].reshape(-1, 3)
        self._elements0 = self.mesh.elements[:, :2].copy()      # the connectivity as read
        self.mesh.nodes = np.vstack([self.mesh.nodes, parked])
        self.nn = self.mesh.n_nodes          # a property: the rows just added
        self.dofs = DofLayout.frame(self.nn)
        self.ndof = self.dofs.size

    @staticmethod
    def _discretise(sec):
        """(y, z, area, width, height, type) of one fibre-section spec."""
        stype = sec.get("type", "rect")
        owner = None
        w = h = None
        if stype == "rect":
            w = float(sec.get("width", 0.1))
            h = float(sec.get("height", 0.2))
            ny = int(sec.get("ny", 10))
            nz = int(sec.get("nz", 10))
            y, z, area = rect_fibers(w, h, ny, nz)
        elif stype == "circle":
            d = float(sec.get("diameter", 0.2))
            nr = int(sec.get("nr", 5))
            nt = int(sec.get("nt", 8))
            y, z, area = circle_fibers(d, nr, nt)
        elif stype == "tube":
            y, z, area = tube_fibers(float(sec["diameter"]), float(sec["thickness"]),
                                     int(sec.get("nt", 20)), int(sec.get("nr", 1)))
        elif stype == "box":
            y, z, area = box_fibers(float(sec["width"]), float(sec["height"]),
                                    float(sec["thickness"]),
                                    int(sec.get("ny", 4)), int(sec.get("nz", 4)),
                                    int(sec.get("nr", 1)))
        elif stype == "patches":
            from ..elements.fiber_section import patch_fibers
            y, z, area, _cells, owner = patch_fibers(sec["patches"], owner=True)
            w = float(np.ptp(y)) if len(y) else None
            h = float(np.ptp(z)) if len(z) else None
        else:
            raise ValueError(f"unknown fiber_section.type {stype!r}")
        return (np.asarray(y, float), np.asarray(z, float), np.asarray(area, float),
                w, h, stype, owner)

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

    @property
    def has_connectors(self):
        return bool(getattr(self, "_connector_props", ()))

    def _setup_section(self):
        """The fibre model of every member: geometry, laws, and how the two
        are joined.

        ONE LAYOUT PER SECTION (generalised in v0.2). A
        section's fibres are its concrete fibres in geometric order followed
        by its rebars, and the model's fibre arrays are (nel, n_max) with
        every member holding its own section's row, padded with zero-area
        fibres to the widest section. Each section owns its material GROUPS
        -- cover, confined core, one per rebar -- and `self.materials` is the
        concatenation over sections; a group belongs to one section and its
        `group_indices` address that section's row. A one-section model is
        one layout on every row, which is exactly the arithmetic this solver
        always had: checked to the bit against the previous code (v0.1 for
        the geometry, v0.2 for the groups).

        WHY GROUPS PER SECTION RATHER THAN PARAMETERS PER FIBRE: the NumPy
        reference path drives one material object per group, and a cover law
        of one section and a cover law of another may differ in parameters or
        even in type; the kernel, which inlines one law type per family and
        reads parameters per fibre, gets its per-element parameter rows from
        the same groups (`_setup_kernel`). Both paths therefore describe the
        same model, which is what makes the reference a reference.
        """
        from ..materials.base import create as mat_create
        e = self.p["element"]
        # selectable integration along the beam axis (default: 2-pt Gauss, Cai Lobatto optional)
        integ = (e.get("fiber_integration") or "gauss2").lower()
        if integ not in INTEGRATION_RULES:
            raise ValueError(f"unknown fiber_integration {integ!r} (expected one of "
                             f"{', '.join(INTEGRATION_RULES)})")
        self.gauss_s, self.gauss_w = INTEGRATION_RULES[integ]

        # `e` is rebound to an element index by the assembly loop further down,
        # so keep the spec itself for anything set up after that.
        self._elem_spec = e
        self._setup_connectors(e)
        many = e.get("fiber_sections")
        if many:
            entries = many
            owner = np.full(self.nel, -1, dtype=np.int64)
            block = next(iter(getattr(self.mesh, "cell_blocks", None) or {"line2": None}))
            for k, m in enumerate(many):
                ids = self.mesh.region_elements(m["region"], block, self.mesh.elements)
                if ids is None:
                    ids = self.mesh.region_elements(m["region"], "beam2_fiber", self.mesh.elements)
                if ids is None or len(ids) == 0:
                    raise ValueError(
                        f"fiber section for region {m['region']!r} covers no complete "
                        "Line2 element")
                taken = ids[owner[ids] >= 0]
                if len(taken):
                    raise ValueError(
                        f"fiber sections overlap: element(s) {taken[:8].tolist()} are in "
                        f"region {m['region']!r} and in region "
                        f"{many[int(owner[taken[0]])]['region']!r}")
                owner[ids] = k
            missing = np.flatnonzero((owner < 0) & self._is_fiber)
            if len(missing):
                raise ValueError(
                    f"element(s) {missing[:8].tolist()} lie in no fibre section's region; "
                    "every member needs a section (a `connector` member is the one "
                    "exception, and it is identified by its own region)")
            owner[~self._is_fiber] = 0          # a padded row; never read
            self._section_of = owner
            self._section_orient = [m.get("orientation") for m in many]
        else:
            entries = [{"fiber_section": e.get("fiber_section") or {},
                        "fiber_rebars": e.get("fiber_rebars") or [],
                        "fiber_tendons": e.get("fiber_tendons") or []}]
            owner = np.zeros(self.nel, dtype=np.int64)
            self._section_of = None if self._is_fiber.all() else owner
            self._section_orient = None if self._is_fiber.all() else [None]
        # a connector's element belongs to no section: it has no fibres, so it
        # must not appear in any layout's rows or it would be given some
        self._rows = [np.flatnonzero((owner == k) & self._is_fiber)
                      for k in range(len(entries))]

        # ---- each section: fibres, cover/core, rebars, its material groups ----
        self.materials = []
        self._layouts = []
        self._section_specs = []
        #: a girder's drawn (width, height), for `info` (1.2); None otherwise
        self._section_sizes = []
        rows_y, rows_z, rows_a = [], [], []
        for k, m in enumerate(entries):
            sec = m["fiber_section"] or {}
            y, z, area, w, h, stype, owner = self._discretise(sec)
            self._section_specs.append((y, z, area, w, h, stype))
            self._section_sizes.append(sec.get("section_size"))
            # Cover and confined core are the same law with different parameters.
            # `cover` is the thickness measured in from each face; a fibre whose
            # centroid lies outside the core rectangle belongs to the cover. Giving
            # `core_material` splits the concrete into two groups, which is what the
            # reference path needs -- one `update` call takes one parameter set --
            # while the kernel keeps a single inlined law and reads the parameters
            # per fibre. Without `core_material` nothing changes: one group, as before.
            cover_t = float(sec.get("cover", 0.0) or 0.0)
            core_spec = sec.get("core_material")
            confined = bool(core_spec) and cover_t > 0.0
            # A PATCH MAY NAME ITS OWN MATERIAL. `cover` +
            # `core_material` is the rect section's rule -- everything within
            # `cover` of a face is cover -- and it does not generalise to
            # patches, where the seam between two patches is not a free face
            # and a distance rule would call interior fibres cover. So a
            # patches section says it outright: the flange's top 20 mm is one
            # patch with the unconfined law, the rest is another with the
            # confined one. The two rules are alternatives, never both on one
            # section.
            patch_mats = ([q.get("material") for q in (sec.get("patches") or [])]
                          if stype == "patches" else [])
            # The test below is on `core_spec` and `cover_t`, NOT on `confined`:
            # `confined` needs both, and the lowering drops `cover` for every
            # shape but rect, so a patches section that wrote both rules would
            # otherwise reach here with `confined` False and have its
            # `core_material` silently ignored.
            if any(patch_mats) and (core_spec or cover_t > 0.0):
                raise ValueError(
                    "a fibre section uses BOTH `core_material` (with `cover`) and a "
                    "per-patch `material`; they are two ways to say the same thing -- "
                    "keep one")
            if core_spec and stype != "rect":
                raise ValueError("fiber_section.core_material currently needs type: rect")
            if confined:
                in_core = (np.abs(y) <= w / 2 - cover_t) & (np.abs(z) <= h / 2 - cover_t)
                if not in_core.any():
                    raise ValueError(
                        f"fiber_section.cover = {cover_t} leaves no core fibres in a "
                        f"{w} x {h} section; every fibre would be cover")
                if in_core.all():
                    raise ValueError(
                        f"fiber_section.cover = {cover_t} leaves no cover fibres; "
                        "with ny/nz this coarse no fibre centroid falls outside the core")
            else:
                in_core = np.zeros(len(y), dtype=bool)
            rebars = list(m.get("fiber_rebars") or [])
            # A PRESTRESSING TENDON IS A STEEL ROW WITH A POSITION OF ITS OWN
            # PER ELEMENT. It joins the section's steel rows
            # after the rebars -- one material group each, like a bar -- and
            # `_setup_tendons` writes its element-by-element (y, z) from the
            # profile and its initial strain from the prestress once the
            # fibre arrays exist. Here its coordinates are placeholders.
            tendon_specs = list(m.get("fiber_tendons") or [])
            tendons = []
            for t in tendon_specs:
                tendons.append((len(rebars), t))
                rebars.append({"y": 0.0, "z": 0.0, "area": t["area"], "material": t["material"]})
            # the section's own cover law, or the model's when it names none
            cover_mat = self.material
            if m.get("material"):
                cmat = m["material"]
                cover_mat = mat_create(cmat["type"], **{kk: v for kk, v in cmat.items() if kk != "type"})
            g0 = len(self.materials)
            self.materials.append(cover_mat)
            conc_groups = [g0]
            if confined:
                self.materials.append(mat_create(core_spec["type"],
                                                 **{kk: v for kk, v in core_spec.items() if kk != "type"}))
                conc_groups.append(g0 + 1)
            # one extra group per DISTINCT per-patch material; patches that
            # name none stay on the section's own law, which is group 0, so a
            # section whose patches name nothing is unchanged to the bit.
            patch_group = {}
            for spec in patch_mats:
                if not spec: continue
                key = repr(sorted(spec.items()))
                if key in patch_group: continue
                patch_group[key] = len(conc_groups)
                conc_groups.append(len(self.materials))
                self.materials.append(mat_create(
                    spec["type"], **{kk: v for kk, v in spec.items() if kk != "type"}))
            rebar_groups = []
            for rb in rebars:
                mtype = rb["material"]["type"]
                mkwargs = {kk: v for kk, v in rb["material"].items() if kk != "type"}
                rebar_groups.append(len(self.materials))
                self.materials.append(mat_create(mtype, **mkwargs))
            n_c = len(y)
            if rebars:
                ry = np.array([float(rb.get("y", 0.0)) for rb in rebars], dtype=float)
                rz = np.array([float(rb.get("z", 0.0)) for rb in rebars], dtype=float)
                ra = np.array([float(rb["area"]) for rb in rebars], dtype=float)
                row_y = np.concatenate([y, ry]); row_z = np.concatenate([z, rz])
                row_a = np.concatenate([area, ra])
            else:
                row_y, row_z, row_a = y, z, area
            # group ids per fibre of THIS section's row: cover 0 / core 1 among
            # the concrete groups, then one per rebar
            if confined:
                conc_local = np.where(in_core, 1, 0)
            elif any(patch_mats) and owner is not None:
                per_patch = np.array([0 if not spec else patch_group[repr(sorted(spec.items()))]
                                      for spec in patch_mats], dtype=int)
                conc_local = per_patch[owner]
            else:
                conc_local = np.zeros(n_c, int)
            fiber_group = np.concatenate([
                np.asarray([conc_groups[i] for i in conc_local], dtype=int),
                np.asarray(rebar_groups, dtype=int)]).astype(int)
            self._layouts.append(_SectionLayout(
                conc_groups=conc_groups, rebar_groups=rebar_groups,
                fiber_group=fiber_group, n_c=n_c, n_s=len(rebars),
                confined=confined, rebars=rebars, entry=m, tendons=tendons))
            rows_y.append(row_y); rows_z.append(row_z); rows_a.append(row_a)

        # ---- the model-wide view: padded rows, groups over sections ----
        n_max = max(len(r) for r in rows_y)
        self.n_fibers = n_max
        if len(entries) == 1:
            # the same row on every element, as it always was
            row_y, row_z, row_a = rows_y[0], rows_z[0], rows_a[0]
            self.fiber_y = np.ascontiguousarray(np.broadcast_to(row_y, (self.nel, len(row_y))))
            self.fiber_z = np.ascontiguousarray(np.broadcast_to(row_z, (self.nel, len(row_z))))
            self.fiber_area = np.ascontiguousarray(np.broadcast_to(row_a, (self.nel, len(row_a))))
            self.fiber_valid = np.ones(self.fiber_y.shape, dtype=bool)
        else:
            fy = np.zeros((self.nel, n_max)); fz = np.zeros((self.nel, n_max))
            fa = np.zeros((self.nel, n_max)); valid = np.zeros((self.nel, n_max), bool)
            for k in range(len(entries)):
                rows, n = self._rows[k], len(rows_y[k])
                fy[rows, :n] = rows_y[k]; fz[rows, :n] = rows_z[k]
                fa[rows, :n] = rows_a[k]; valid[rows, :n] = True
            self.fiber_y, self.fiber_z, self.fiber_area, self.fiber_valid = fy, fz, fa, valid
        #: initial strain per fibre column: zero everywhere but on tendons.
        #: Adding it to every fibre's strain leaves a model
        #: without tendons bit-identical (x + 0.0 == x).
        self.fiber_prestrain = np.zeros(self.fiber_y.shape)
        self.tendons = []
        self._tendon_force_scale = 0.0
        self._setup_tendons()
        #: per fibre: the coefficient of thermal expansion of ITS OWN
        #: material. Concrete is ~1.0e-5 and steel ~1.2e-5, and
        #: on a section that is both, one number for the section would be a
        #: choice nobody made. Zero unless the material block states `alpha`.
        self.fiber_alpha = np.zeros(self.fiber_y.shape)
        #: the total imposed (non-mechanical) strain per fibre: the tendons'
        #: prestrain plus whatever a temperature load adds this step. It IS
        #: `fiber_prestrain` -- the same object -- until something warms up,
        #: so a model without a temperature load is unchanged to the bit.
        self._imposed = self.fiber_prestrain
        self._thermal = []
        # a group's fibres, as columns of its own section's row
        self.group_indices = [None] * len(self.materials)
        self.group_section = np.zeros(len(self.materials), dtype=np.int64)
        for k, lay in enumerate(self._layouts):
            for g in lay.conc_groups + lay.rebar_groups:
                self.group_indices[g] = np.flatnonzero(lay.fiber_group == g)
                self.group_section[g] = k
        for k, lay in enumerate(self._layouts):
            a = np.array([float(getattr(self.materials[g], "thermal_expansion", 0.0) or 0.0)
                          for g in lay.fiber_group], dtype=float)
            self.fiber_alpha[np.ix_(self._rows[k], np.arange(len(a)))] = a
        lay0 = self._layouts[0]
        # the one-section names the rest of the file grew up with; on a
        # multi-section model they describe the first section, and every
        # consumer that must not assume one section reads `_layouts` instead
        self.is_confined = lay0.confined
        self.is_hybrid = any(lay.n_s > 0 for lay in self._layouts)
        self.rebar_specs = lay0.rebars
        self.n_conc_groups = len(lay0.conc_groups)
        self.rebar_mats = [self.materials[g] for g in lay0.rebar_groups]
        self.fiber_group = lay0.fiber_group
        self.n_concrete = lay0.n_c
        self.n_steel = max(lay.n_s for lay in self._layouts)
        self._check_fiber_monitors()
        #: per element: the effective shear stiffness kappa G A_s the section
        #: states, 0 where it states none (Euler-Bernoulli)
        self._GA = np.zeros(self.nel)
        for k, m in enumerate(entries):
            ga = (m["fiber_section"] or {}).get("shear_stiffness")
            if ga:
                self._GA[self._rows[k]] = float(ga)
        #: per section: its cracking shear law, if it states one
        self._shear_laws = {k: m["fiber_section"]["shear_law"] for k, m in enumerate(entries)
                            if (m["fiber_section"] or {}).get("shear_law")}
        #: per element: the section area and second moments (diagnostics, the
        #: reference stiffness, the lumped rotary mass)
        self.total_area = self.fiber_area.sum(axis=1)
        self.Iy = np.sum(self.fiber_area * self.fiber_z * self.fiber_z, axis=1)
        self.Iz = np.sum(self.fiber_area * self.fiber_y * self.fiber_y, axis=1)
        #: per element: the mass per length and the section's mass moments,
        #: every fibre at the density of ITS OWN material -- concrete, bars and
        #: tendons, per section and per patch. It was the
        #: first fibre section's cover density times the whole area: a pier's
        #: own concrete, every bar and every tendon weighed as that concrete
        #: (a 30+40+30 m bridge came out 1.4% light; a steel girder written
        #: first weighed the concrete piers as steel, 3.1x).
        self._fiber_rho = np.zeros(self.fiber_area.shape)
        for k, lay in enumerate(self._layouts):
            r = np.array([float(getattr(self.materials[g], "rho", 7850))
                          for g in lay.fiber_group], dtype=float)
            self._fiber_rho[np.ix_(self._rows[k], np.arange(len(r)))] = r
        _ra = self.fiber_area * self._fiber_rho
        self.mass_per_length = _ra.sum(axis=1)
        self.rho_Iy = np.sum(_ra * self.fiber_z * self.fiber_z, axis=1)
        self.rho_Iz = np.sum(_ra * self.fiber_y * self.fiber_y, axis=1)
        #: per element: Saint-Venant's torsion constant. The polar moment
        #: Iy + Iz unless the section states `torsion_constant` -- right
        #: for a circle, several times too stiff for a box girder or a
        #: rectangle
        self.J = self.Iy + self.Iz
        for k, m in enumerate(entries):
            j = (m["fiber_section"] or {}).get("torsion_constant")
            if j:
                self.J[self._rows[k]] = float(j)
        #: per element: the E and nu OF ITS OWN SECTION.
        #: Torsion is not a fibre quantity -- there is no fibre law to ask for
        #: it -- so G J was taken from `self.material`, which is the FIRST
        #: fibre section's cover law and nothing else. v0.2 had already given
        #: every section its own material, and `self.J` here is already per
        #: element; only the modulus multiplying it was not. Measured: a
        #: four-element beam, two concrete sections then two steel, one uniform
        #: twist rate -- concrete written first gave EVERY member Mt = 5000,
        #: steel written first gave EVERY member Mt = 32307.7. Swapping two
        #: lines of YAML changed every member's torque by 6.46x, silently.
        #:
        #: A single-section model reads section 0's cover law, which IS
        #: `self.material` (the input model writes a section's `material` key
        #: only when it differs from the first's), so it is unchanged to the bit.
        self.member_E = np.full(self.nel, float(getattr(self.material, "E", 200e9)))
        self.member_nu = np.full(self.nel, float(getattr(self.material, "nu", 0.3)))
        for k in range(len(entries)):
            cover = self.materials[self._layouts[k].conc_groups[0]]
            self.member_E[self._rows[k]] = float(getattr(cover, "E", 200e9))
            self.member_nu[self._rows[k]] = float(getattr(cover, "nu", 0.3))
        #: per element: G J, the Saint-Venant torsional rigidity actually used
        self.member_G = self.member_E / (2.0 * (1.0 + self.member_nu))
        self._G0J0 = np.ascontiguousarray(self.member_G * self.J)
        # per-group states: list of (nel, n_gauss, n_in_group, n_state). A
        # group's rows for elements outside its section are never read.
        self.fiber_states = []
        for g, mat in enumerate(self.materials):
            n_state = getattr(mat, "n_state", 0)
            n_in_g = len(self.group_indices[g])
            self.fiber_states.append(np.zeros((self.nel, len(self.gauss_s), n_in_g, n_state or 0)))
        # The CONVERGED state of the last load stage. Staged relaxation reaches
        # each equilibrium point by thousands of explicit iterations whose path
        # is not physical -- kinetic damping resets velocity whenever the energy
        # dips, so the strain at a fibre wanders up and down on the way in. The
        # materials here are path dependent and `update` mutates the state it is
        # given, so committing on every iteration records every one of those
        # wiggles as a load reversal. Measured on case_11: ~50 000 update calls
        # and 4 000-6 500 direction flips in a SINGLE load stage, which
        # re-anchors the Giuffre-Menegotto-Pinto curve away from the real
        # excursion and collapses the hysteresis loops -- the cyclic energy came
        # out 16x too small (1.13 kJ against 17.92 kJ for one +-42 mm cycle,
        # measured 14.7 kJ). Trial state during a stage, commit on convergence.
        # THIS IS THE RESTORE-THEN-INTEGRATE POLICY, described once in
        # core/staging.py along with the two measurements that forced it and
        # the accumulate-between-commits policy that bond damage needs
        # instead. It is NOT expressed with `staging.TrialState` on purpose:
        # `fiber_states[g]` is REBOUND -- to a view of `_state_c` when the
        # fused kernel is installed, and again on restore from a checkpoint --
        # so a helper holding the array from construction would quietly go
        # stale. Rewriting it to survive that would add the exact class of
        # trap the module exists to remove. New path-dependent state should
        # use staging.py; this one is the reference the module describes.
        self.fiber_committed = [s.copy() for s in self.fiber_states]
        self._trial_mode = False
        #: (nel, n_stations, n_fibers) in the section-row order of `fiber_y`:
        #: the NumPy path writes it directly, the kernel path through
        #: `fiber_stress_rows()`
        self._fiber_stress_rows = np.zeros((self.nel, len(self.gauss_s), self.n_fibers))
        self._fstress = None
        # legacy single-state for non-hybrid (kept for checkpoint compat)
        # we keep self.fiber_state as alias to first group's state when not hybrid
        if not self.is_hybrid:
            self.fiber_state = self.fiber_states[0]
        else:
            # for hybrid keep a flat alias for checkpoint (concatenated)
            self.fiber_state = None
        # for linear_elastic fiber, store E (only meaningful when single mat)
        if not self.is_hybrid and self.material.name == "linear_elastic":
            self.fiber_E = float(self.material.E)
        else:
            self.fiber_E = None

    @staticmethod
    def _modulus(mat):
        for name in ("E", "Ec", "E0"):
            v = getattr(mat, name, None)
            if v is not None:
                return float(v)
        raise ValueError(f"material {mat.name!r} has no modulus to derive a strain from")

    def _member_axis(self, rows):
        """(s_mid, s_lo, s_hi, member length, axis, s per element END) of
        the members `rows`: distance along the members' common axis from
        the start -- the end with the smaller global x (then y, then z).
        The last return is (n, 2) in the elements' OWN node order, which is
        what a per-element end value has to be written against."""
        conn = self.mesh.elements[rows]
        X = self.mesh.nodes[np.unique(conn)]
        c = X.mean(axis=0)
        _u, _sv, vt = np.linalg.svd(X - c, full_matrices=False)
        t = vt[0]
        if np.dot(t, [1.0, 1e-3, 1e-6]) < 0.0:
            t = -t
        proj = (self.mesh.nodes - c) @ t
        s0 = float(proj[np.unique(conn)].min())
        pe = proj[conn] - s0
        return (0.5 * pe.sum(axis=1), pe.min(axis=1), pe.max(axis=1),
                float(proj[np.unique(conn)].max() - s0), t, pe)

    def _setup_tendons(self):
        """Every tendon's position per element and its initial strain.

        The profile [[s, y, z], ...] is sampled at each element's midpoint
        along the member, so a draped tendon is a polyline of one
        eccentricity per element: the moment P e(x) then steps at the nodes,
        and those steps are the equivalent transverse load, lumped -- the
        discrete form of the balancing-load picture, converging with the
        mesh (tests/test_fiber_tendons.py). The initial strain is the stated
        stress over the tendon modulus, or, with `jack` and `friction`, the
        post-tensioning profile of elements/truss.tendon_profile evaluated on
        the tendon's own polyline. `elastic_shortening: compensate` adds the
        linear-elastic strain of the section at the tendon, from all the
        section's tendons, so that the stated stress is what the tendon
        carries after the concrete has shortened under it.
        """
        from ..elements import fiber_section, truss
        for k, lay in enumerate(self._layouts):
            if not lay.tendons:
                continue
            rows_all = self._rows[k]
            s_mid, s_lo, s_hi, L_m, _t, s_ends = self._member_axis(rows_all)
            placed = []
            for j, spec in lay.tendons:
                rows = rows_all
                col = lay.n_c + j
                prof = np.asarray(spec["profile"], dtype=float)
                sp = prof[:, 0] * (L_m if spec.get("normalised") else 1.0)
                # A TENDON ANCHORED INSIDE THE MEMBER: with
                # `extent: [s0, s1]` (in the profile's s units) the tendon
                # exists only in the elements whose midpoint lies between
                # the anchors -- the pier-top cap tendons of a continuous
                # girder, a span's own draped tendons when several spans
                # share one section. Outside, the row's area is zero: it
                # carries nothing, weighs nothing, and the sweep skips it.
                if spec.get("extent") is not None:
                    s0, s1 = (float(v) * (L_m if spec.get("normalised") else 1.0) for v in spec["extent"])
                    inside = (s_mid >= min(s0, s1)) & (s_mid <= max(s0, s1))
                    if not inside.any():
                        raise ValueError(f"tendon slot {j} of section {k}: no element midpoint lies in "
                                         f"extent {spec['extent']} (member length {L_m:.3g})")
                    self.fiber_area[rows[~inside], col] = 0.0
                    rows = rows[inside]; s_mid_j = s_mid[inside]
                    s_lo_j, s_hi_j = s_lo[inside], s_hi[inside]
                    s_ends_j = s_ends[inside]
                    order = np.argsort(s_mid_j)
                else:
                    s_mid_j, s_lo_j, s_hi_j, s_ends_j = s_mid, s_lo, s_hi, s_ends
                    order = np.argsort(s_mid)
                # THE PROFILE COVERS THE TENDON, as the prestress profile below
                # must: a profile that stops short was clamped (linear) or
                # extrapolated (quadratic -- 0.6 m outside a 0.4 m section)
                # without a word (found in review)
                tol_s = 1e-9 * max(1.0, L_m)
                if s_mid_j.min() < sp.min() - tol_s or s_mid_j.max() > sp.max() + tol_s:
                    raise ValueError(
                        f"tendon slot {j} of section {k}: the profile covers s = {sp.min():.6g} .. "
                        f"{sp.max():.6g} but the tendon has element midpoints from {s_mid_j.min():.6g} "
                        f"to {s_mid_j.max():.6g} (member length {L_m:.6g}; `normalised: true` reads s "
                        "as a fraction of it)")
                if spec.get("interpolation", "linear") == "quadratic":
                    # ONE PARABOLA PER THREE POINTS (0-1-2, 2-3-4, ...), each
                    # through its own points. It was ONE least-squares parabola
                    # through all of them: a five-point draped profile came out
                    # a straight tendon at z = -0.08 (found in review)
                    y_of = lambda x, sp=sp, v=prof[:, 1]: _piecewise_parabola(sp, v, x)
                    z_of = lambda x, sp=sp, v=prof[:, 2]: _piecewise_parabola(sp, v, x)
                else:
                    y_of = lambda x, sp=sp, v=prof[:, 1]: np.interp(x, sp, v)
                    z_of = lambda x, sp=sp, v=prof[:, 2]: np.interp(x, sp, v)
                self.fiber_y[rows, col] = y_of(s_mid_j)
                self.fiber_z[rows, col] = z_of(s_mid_j)
                # THE TENDON AS SAMPLED stays inside the section: at every
                # element midpoint (what is integrated) and densely along the
                # profile over the elements it is in (a parabola's vertex can
                # fall between the listed points). The input check sees only
                # the listed points (found in review)
                # Tested against the section's OUTER OUTLINE -- the patches'
                # union, the tube's circle -- not its bounding rectangle,
                # which let a tendon sit beside a T's web
                fsec = lay.entry.get("fiber_section") or {}
                ftype = fsec.get("type", "rect")
                s_a = max(float(sp.min()), float(s_lo_j.min()))
                s_b = min(float(sp.max()), float(s_hi_j.max()))
                dense = np.linspace(s_a, s_b, max(2, 50 * len(prof))) if s_b > s_a else np.array([s_a])
                s_chk = np.concatenate([s_mid_j, dense])
                yc, zc = y_of(s_chk), z_of(s_chk)
                inside = fiber_section.outline_contains(
                    ftype, yc, zc, width=fsec.get("width"), height=fsec.get("height"),
                    diameter=fsec.get("diameter"), patches=fsec.get("patches"),
                    cells=fsec.get("cells"), exact=bool(fsec.get("section_size")))
                if inside is not None and not inside.all():
                    i = int(np.flatnonzero(~inside)[0])
                    # s in the profile's own units: a fraction when `normalised`
                    s_user = s_chk[i] / L_m if spec.get("normalised") else s_chk[i]
                    raise ValueError(
                        f"tendon slot {j} of section {k}: the profile passes y = {yc[i]:.4g}, "
                        f"z = {zc[i]:.4g} at s = {s_user:.6g}{' (normalised)' if spec.get('normalised') else ''}, "
                        f"outside the section ({fiber_section.outline_name(ftype, bool(fsec.get('section_size')))})"
                        + ("; with `interpolation: quadratic` the curve between the points is what is placed"
                           if spec.get("interpolation", "linear") == "quadratic" else ""))
                E_p = self._modulus(self.materials[lay.rebar_groups[j]])
                if spec.get("prestress") is not None:
                    sigma = np.full(len(rows), float(spec["prestress"]))
                elif spec.get("prestress_profile") is not None:
                    # [[s, sigma], ...] sampled at the element midpoints.
                    # Not extrapolated: a table that stops
                    # short of an element the tendon is in would hand that
                    # element its last value without saying so.
                    tab = np.asarray(spec["prestress_profile"], dtype=float)
                    s_tab = tab[:, 0] * (L_m if spec.get("normalised") else 1.0)
                    tol = 1e-9 * max(1.0, L_m)
                    if s_mid_j.min() < s_tab[0] - tol or s_mid_j.max() > s_tab[-1] + tol:
                        raise ValueError(
                            f"tendon slot {j} of section {k}: the `prestress` profile covers "
                            f"s = {s_tab[0]:.6g} .. {s_tab[-1]:.6g} but the tendon has element "
                            f"midpoints from {s_mid_j.min():.6g} to {s_mid_j.max():.6g}")
                    sigma = np.interp(s_mid_j, s_tab, tab[:, 1])
                else:
                    fr = spec.get("friction") or {}
                    # the tendon's own polyline, element by element along s
                    bnd = np.concatenate([s_lo_j[order], [s_hi_j[order][-1]]])
                    pts = np.column_stack([bnd, y_of(bnd), z_of(bnd)])
                    conn = np.column_stack([np.arange(len(rows)), np.arange(1, len(rows) + 1)])
                    sig_sorted = truss.tendon_profile(
                        pts, conn, float(spec["jack"]), mu=float(fr.get("mu", 0.0)),
                        k=float(fr.get("k", 0.0)), jack=fr.get("jack", "start"),
                        anchor_set=fr.get("anchor_set"), E=E_p)
                    sigma = np.empty(len(rows)); sigma[order] = sig_sorted
                self.fiber_prestrain[rows, col] = sigma / E_p
                # THE PROFILE AT THE ELEMENT'S TWO ENDS, for output only:
                # the section integration samples the
                # tendon ONCE per element, at the midpoint, which is the
                # discretisation -- but drawing that value at both ends
                # turns a draped tendon into a staircase of horizontal
                # chords, which is what the sweep used to export and what
                # reads in ParaView as a straight tendon. The ends are the
                # tendon's declared geometry and nothing reads them but
                # the exporter.
                ends = (np.column_stack([y_of(s_ends_j[:, 0]), y_of(s_ends_j[:, 1])]),
                        np.column_stack([z_of(s_ends_j[:, 0]), z_of(s_ends_j[:, 1])]))
                placed.append((j, col, spec, sigma, E_p, rows, ends))
            if any(sp.get("elastic_shortening") == "compensate" for _j, _c, sp, _s, _E, _r, _e in placed):
                # gross, uncracked, all fibres: A, Iy, Iz of the section per
                # element; a bounded tendon's force is zero where it is absent
                rows = rows_all
                fa = self.fiber_area[rows]; fy = self.fiber_y[rows]; fz = self.fiber_z[rows]
                # THE TRANSFORMED SECTION'S 3x3 STIFFNESS ABOUT THE ORIGIN --
                # every fibre at its own law's modulus, first moments and the
                # product of inertia included -- so eps(y, z) = a + b z + c y
                # is the section's linear-elastic answer to the tendons. It was
                # N/A + M z/I with z and I about the origin and the concrete
                # modulus on every fibre: right only when the origin is the
                # centroid of a doubly symmetric section, and a T written with
                # its origin at the flange over-compensated by 20% (1227 MPa for
                # 1200; found in review).
                E_col = np.zeros(fa.shape[1])
                E_col[:len(lay.fiber_group)] = [self._modulus(self.materials[g]) for g in lay.fiber_group]
                ea = fa * E_col[None, :]
                basis = (np.ones_like(fa), fz, fy)
                K = np.empty((len(rows), 3, 3))
                for i in range(3):
                    for jj in range(3):
                        K[:, i, jj] = (ea * basis[i] * basis[jj]).sum(axis=1)
                # A DEGENERATE AXIS IS SOLVED AROUND, NOT THROUGH: a planar
                # section (`rect` with ny: 1, every fibre and tendon at y = 0)
                # has no stiffness about z and no tendon moment about it, and
                # `solve` raised "Singular matrix" (the 3x3 of v1.0; before
                # it, 0/0 = NaN; found in review). The
                # pseudo-inverse of the diagonally scaled matrix (scaled, so
                # N and N m^2 compare) answers the non-degenerate block and
                # leaves the empty direction at zero strain.
                dg = np.einsum("eii->ei", K)
                Ds = np.where(dg > 0.0, 1.0 / np.sqrt(np.where(dg > 0.0, dg, 1.0)), 0.0)
                lam, V = np.linalg.eigh(Ds[:, :, None] * K * Ds[:, None, :])
                inv = np.where(lam > 1e-12 * lam.max(axis=1, keepdims=True), 1.0 / np.where(lam > 0.0, lam, 1.0), 0.0)
                K_pinv = Ds[:, :, None] * np.einsum("eij,ej,ekj->eik", V, inv, V) * Ds[:, None, :]
                # the compensated prestrain p satisfies sigma / E_p = p + eps_t(p):
                # the force of the COMPENSATED tendons shortens the section
                # too, so it is solved for (a contraction: E_p A_p is small
                # against the section) rather than estimated once
                stated = {c: self.fiber_prestrain[rows, c].copy() for _j, c, _sp, _s, _E, _r, _e in placed}
                comp = [(c, np.isin(rows, r)) for _j, c, sp, _s, _E, r, _e in placed
                        if sp.get("elastic_shortening") == "compensate"]
                for _it in range(200):
                    P = [(self.fiber_prestrain[rows, c] * E * self.fiber_area[rows, c],
                          self.fiber_y[rows, c], self.fiber_z[rows, c])
                         for _j, c, sp, sig, E, _r, _e in placed]
                    F = np.column_stack([sum(p for p, _y, _z in P), sum(p * z for p, _y, z in P),
                                         sum(p * y for p, y, _z in P)])
                    abc = -np.einsum("eij,ej->ei", K_pinv, F)
                    change = 0.0
                    for c, keep in comp:
                        eps_t = abc[:, 0] + abc[:, 1] * self.fiber_z[rows, c] + abc[:, 2] * self.fiber_y[rows, c]
                        new = np.where(keep, stated[c] - eps_t, self.fiber_prestrain[rows, c])
                        change = max(change, float(np.max(np.abs(new - self.fiber_prestrain[rows, c]))))
                        self.fiber_prestrain[rows, c] = new
                    if change <= 1e-14 * max(float(np.max(np.abs(stated[c]))) for c, _k in comp):
                        break
            for j, c, sp, sig, E_p, rows, ends in placed:
                self._tendon_force_scale = max(self._tendon_force_scale, float(np.max(np.abs(sig))) * float(sp["area"]))
                self.tendons.append({"section": k, "slot": j, "column": c, "area": float(sp["area"]),
                                     "elements": rows, "sigma": sig,
                                     "strain": self.fiber_prestrain[rows, c].copy(),
                                     "y": self.fiber_y[rows, c].copy(), "z": self.fiber_z[rows, c].copy(),
                                     "y_ends": ends[0], "z_ends": ends[1]})

    def _section_index(self, e):
        return 0 if self._section_of is None else int(self._section_of[e])

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
            T = self.T[e]
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
            slip_committed=self._conn_committed[e], slip_out=self._conn_slip[e])

    def _connector_kernel(self):
        """The compiled connectors' arrays (kernels/numba_connector), built
        once, or None: no connectors, no numba, or not six dofs a node. Used
        wherever the fibre kernel is."""
        ck = getattr(self, "_conn_kernel", False)
        if ck is not False:
            return ck
        ck = None
        if len(getattr(self, "_connector_ids", ())) and self.dofs.stride == 6:
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
            R, L = self._frame(self.mesh.nodes[conn[0]], self.mesh.nodes[conn[1]], o)
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

    def _setup_shear_interpolation(self):
        """The curvature and shear-strain rows of every member.

        A member whose section states a `shear_stiffness` GA_s gets Reddy's
        interdependent Timoshenko interpolation: phi = 12 E I / (GA_s L0^2)
        per bending plane (E the section's reference modulus, I the fibre
        section's own), the curvature rows scaled and shifted by phi, and a
        constant shear strain gamma whose force V = GA_s gamma enters through
        its own rows. A member without one has phi = 0 and zero shear rows,
        and its curvature rows are the Hermite ones to the bit. Both force
        paths read these arrays; the reference path used to rebuild the
        Hermite rows inside its station loop.

        The fibres never see shear. What this adds is the elastic (or, with
        a cracked-effective GA_s such as ASCE 41's 0.4 E_c A_w, the
        service-level) member shear flexibility -- on Vecchio & Emara's frame
        the 20% of total deformation the test's authors measured and an
        Euler-Bernoulli fibre beam cannot have.
        """
        from ..kernels.numba_fiber_beam import (precompute_shape_derivatives,
                                                precompute_shear_rows)
        L0 = np.asarray(self.length, dtype=float)
        self._phi_y = np.zeros(self.nel)
        self._phi_z = np.zeros(self.nel)
        for k, lay in enumerate(self._layouts):
            rows = self._rows[k]
            if not np.any(self._GA[rows] > 0.0):
                continue
            mat = self.materials[lay.conc_groups[0]]
            E = float(self.fiber_E if self.fiber_E is not None else getattr(mat, "E", 200e9))
            ga = self._GA[rows]
            self._phi_y[rows] = 12.0 * E * self.Iy[rows] / (ga * L0[rows] ** 2)
            self._phi_z[rows] = 12.0 * E * self.Iz[rows] / (ga * L0[rows] ** 2)
        self._Bky, self._Bkz = precompute_shape_derivatives(
            L0, self.gauss_s, self._phi_y, self._phi_z)
        self._Bgy, self._Bgz = precompute_shear_rows(L0, self._phi_y, self._phi_z)
        self.has_shear = bool(np.any(self._GA > 0.0))
        self._setup_shear_law()

    def _setup_shear_law(self):
        """The cracking shear law of the sections that state one.

        IN SERIES WITH THE BENDING. A member whose section has a shear law
        keeps its Hermite (Euler-Bernoulli) curvature rows, and carries one
        more unknown per shear plane: the shear rotation gamma, the part of
        both end rotations that is shear and not bending. Its energy is

            U = U_bend(theta_i - gamma, theta_j - gamma) + L0 * int V(gamma) dgamma

        and the equilibrium in gamma is M_i + M_j = L0 V(gamma), the member's
        end moments against its shear. For an elastic V = GA gamma this is
        exactly the Timoshenko member (bending and shear flexibility in
        series); for the cracking law it lets the shear stiffness fall while
        the fibres keep their bending. gamma is relaxed once per force
        evaluation by a Newton step on that equation, with the bending
        tangent taken at its uncracked value (and doubled for the bars):
        never larger than the true step, so it converges from one side.
        """
        nel = self.nel
        self._sl_on = np.zeros(nel, bool)
        self._sl_update = False
        self._sl_arrays = None                 # the kernel's view, rebuilt on demand
        if not self._shear_laws:
            return
        self._sl_rot = np.zeros((nel, 2), int)          # the two end-rotation entries of `local`
        self._sl_GA0 = np.ones(nel); self._sl_GAcr = np.ones(nel)
        self._sl_Vcr = np.full(nel, np.inf); self._sl_Vy = np.full(nel, np.inf)
        self._sl_h = np.zeros(nel); self._sl_kb = np.ones(nel)
        self._sl_flex = np.zeros(nel, bool); self._sl_epscr = np.zeros(nel)
        self._sl_conc = {}
        L0 = np.asarray(self.length, float)
        for k, law in self._shear_laws.items():
            rows = self._rows[k]
            lay = self._layouts[k]
            cover = self.materials[lay.conc_groups[0]]
            Ec = float(getattr(cover, "E", 3.0e10))
            GA0 = float(law["stiffness"])
            if law.get("stirrups"):
                st = law["stirrups"]
                bw = float(law["web_width"])
                rho = float(st["area"]) / (bw * float(st["spacing"]))
                n = float(st["E"]) / Ec
                GAcr = rho / (1.0 + 4.0 * n * rho) * float(st["E"]) * bw * float(st["depth"])
            else:
                GAcr = float(law["cracked_stiffness"])
            if GAcr >= GA0:
                raise ValueError(f"fibre section {k}: the cracked shear stiffness {GAcr:.4g} N is not "
                                 f"below the uncracked {GA0:.4g} N")
            z = law.get("direction", "z") == "z"
            self._sl_on[rows] = True
            # a z shear bends about y: rows (2, 4, 8, 10), rotations 4 and 10;
            # a y shear bends about z: rows (1, 5, 7, 11), rotations 5 and 11
            self._sl_rot[rows] = (4, 10) if z else (5, 11)
            self._sl_GA0[rows] = GA0; self._sl_GAcr[rows] = GAcr
            if law.get("cracking", "flexural") == "flexural":
                self._sl_flex[rows] = True
                ft = float(getattr(cover, "ft", 0.0) or 0.0)
                self._sl_epscr[rows] = ft / Ec
            else:
                self._sl_Vcr[rows] = float(law["cracking"])
            if law.get("yield_force"):
                self._sl_Vy[rows] = float(law["yield_force"])
            self._sl_h[rows] = float(law.get("hardening", 0.01))
            I = self.Iy[rows] if z else self.Iz[rows]
            self._sl_kb[rows] = 2.0 * 12.0 * Ec * I / L0[rows]
            self._sl_conc[k] = np.concatenate([self.group_indices[g] for g in lay.conc_groups])
        #: the state: gamma (a displacement, carried), and the committed
        #: history -- the shear strain the section cracked at (inf: not yet)
        #: and the largest it has reached
        self._sl_gamma = np.zeros(nel)
        gcr0 = np.where(self._sl_flex, np.inf, self._sl_Vcr / self._sl_GA0)
        self._sl_gcr_c = gcr0.copy(); self._sl_gcr_t = gcr0.copy()
        self._sl_gmax_c = np.zeros(nel); self._sl_gmax_t = np.zeros(nel)
        self._sl_V = np.zeros(nel)

    def _shear_law_arrays(self):
        """The shear laws' arrays as the fused kernel takes them (1.7,
        kernels/numba_fiber_beam): the per-member parameters, the state
        (gamma, carried) and the history (committed and trial), or arrays
        of nothing when no section states a law."""
        a = getattr(self, "_sl_arrays", None)
        if a is not None:
            return a
        nel = self.nel
        on = getattr(self, "_sl_on", None)
        if on is None or not on.any():
            z = np.zeros(nel)
            a = (np.zeros(nel, np.uint8), np.zeros(nel, np.int64), np.zeros(nel, np.int64),
                 z, z, z, z, z, np.zeros(nel, np.bool_), z, z, z, z, z, z, z)
        else:
            c = np.ascontiguousarray
            a = (c(on.astype(np.uint8)), c(self._sl_rot[:, 0], dtype=np.int64), c(self._sl_rot[:, 1], dtype=np.int64),
                 c(self._sl_GA0, dtype=float), c(self._sl_GAcr, dtype=float), c(self._sl_Vy, dtype=float),
                 c(self._sl_h, dtype=float), c(self._sl_kb, dtype=float), c(self._sl_flex, dtype=np.bool_),
                 c(self._sl_epscr, dtype=float),
                 self._sl_gamma, self._sl_gcr_c, self._sl_gmax_c, self._sl_gcr_t, self._sl_gmax_t, self._sl_V)
            for arr, name in zip(a[10:], ("_sl_gamma", "_sl_gcr_c", "_sl_gmax_c", "_sl_gcr_t", "_sl_gmax_t", "_sl_V")):
                assert arr is getattr(self, name)       # the state arrays themselves, written in place
        self._sl_arrays = a
        return a

    def _shear_law(self, e, g, gcr, gmax):
        """(V, tangent) of element e's shear law at gamma g, with the history
        (the crack strain `gcr`, the largest |gamma| `gmax`)."""
        GA0, GAcr = self._sl_GA0[e], self._sl_GAcr[e]
        a = abs(g); sgn = 1.0 if g >= 0.0 else -1.0
        if not np.isfinite(gcr) or max(a, gmax) <= gcr:
            return GA0 * g, GA0
        Vc = GA0 * gcr
        gy = gcr + (self._sl_Vy[e] - Vc) / GAcr if np.isfinite(self._sl_Vy[e]) else np.inf

        def env(x):
            if x <= gcr:
                return GA0 * x, GA0
            if x <= gy:
                return Vc + GAcr * (x - gcr), GAcr
            h = self._sl_h[e] * GAcr
            return self._sl_Vy[e] + h * (x - gy), h
        if a >= gmax:
            v, k = env(a)
            return sgn * v, k
        # inside the envelope once cracked: towards the origin (the cracks
        # close under no shear), on the secant through the largest excursion
        v, _k = env(gmax)
        sec = v / gmax
        return sec * g, sec

    def _setup_fracture(self, spec_root):
        """`element.fracture: {threshold: 1.0}` -- off unless it is asked for.

        THE ELEMENT'S ONLY CONTRIBUTION to the criterion is the layout: this
        section has `n_fibers` fibres, each sampled at every Gauss station,
        and a fibre is gone when it is gone at ANY of them (Cai et al. eq. 20).
        What makes a fibre fail is the fibre material's business and is not
        mentioned here or anywhere else in this file (core/fracture.py).

        Which is why the check below is worth its lines: a section whose laws
        declare no failure at all can never reach the threshold, so asking for
        fracture and getting a run that never fractures would look exactly
        like a structure that is strong enough. Refuse it instead.
        """
        self.fracture = None
        self._active = np.ones(self.nel, dtype=np.uint8)
        self._refresh_kernel_mask()
        spec = (spec_root or {}).get("fracture")
        if not spec:
            return
        spec = {} if spec is True else dict(spec)
        silent = [type(m).__name__ for g, m in enumerate(self.materials)
                  if m.failure(m.init_state(1)) is None]
        if len(silent) == len(self.materials):
            raise ValueError(
                "element.fracture is on but no fibre material declares a "
                f"failure criterion ({', '.join(sorted(set(silent)))}); give "
                "the bars a `rupture_strain`, or use a law with damage")
        self._fracture_silent = sorted(set(silent))
        # WHICH FIBRES ARE WORTH WATCHING, decided once. A law that declares no
        # criterion returns None forever, so its fibres are structurally zero
        # in the gather -- on an RC section that is 64 of 68 columns whose
        # contents can never change. Only the rest are gathered and reduced,
        # and `fiber_layout(..., tracked=)` keeps the denominator at n_fibers
        # so FAC still means the fraction of the SECTION that is gone.
        self._failing_groups, n_tracked = [], 0
        for g, m in enumerate(self.materials):
            n = len(self.group_indices[g])
            if self.fiber_states[g].shape[3] == 0 or m.failure(m.init_state(1)) is None:
                continue
            self._failing_groups.append((g, n_tracked, n))
            n_tracked += n
        if not n_tracked:                  # a criterion on a law with no state
            raise ValueError(
                "element.fracture is on but no fibre carries a law that can "
                "report failure; give the bars a `rupture_strain`, or use a "
                "law with damage")
        self._tracked_failure = np.zeros(
            (self.nel, n_tracked, len(self.gauss_s)))
        # A PADDED FIBRE CANNOT BREAK. With several sections
        # the fibre arrays are padded to the widest one; a padded fibre has
        # zero area, sees the axial strain, and its law may well report
        # failure -- so its column is masked out of the gather, and the
        # fraction is taken over the member's OWN fibre count.
        self._tracked_valid = None
        over = None
        if len(self._layouts) > 1 or not bool(self.fiber_valid.all()):
            # a tracked column belongs to one section's group: it is live on
            # that section's rows and on real (unpadded) fibres only
            valid = np.zeros((self.nel, n_tracked), dtype=float)
            for g, start, n in self._failing_groups:
                rows = self._rows[int(self.group_section[g])]
                valid[rows, start:start + n] = self.fiber_valid[rows][:, self.group_indices[g]]
            self._tracked_valid = valid[:, :, None]
            over = self.fiber_valid.sum(axis=1).astype(float)
            # a member with no fibres at all (a connector, a bearing) has
            # nothing that can break and a fraction of 0 out of anything:
            # its 0 here was refused by the latch ("every element is out of
            # at least one thing"), and a fibre model with `fracture`, two
            # fibre sections and a bearing stopped at its first step (found
            # in the review of v1.12; there since the padded layouts)
            over[over < 1.0] = 1.0
        self._pending_splits: List[int] = []
        #: (time, element, parent particle, new particle) of every separation
        self.splits: List[tuple] = []
        #: spare particle -> the particle it was split from (the frame
        #: solvers' `_with_fragments`)
        self._fragment_parent: Dict[int, int] = {}
        self.bonds = None
        if self._spare_base is not None:
            self.bonds = topology.BondGraph(self.mesh.elements, self.nn,
                                            reserved=self._spare_of)
            self._components0 = self.bonds.n_components()
        self.fracture = fracture.Fracture(
            self.nel,
            # with several sections the tracked columns are the sections'
            # groups side by side, more than any one row holds; the fraction
            # is then taken over each member's own count (`over`), and the
            # layout's denominator is only its shape
            fracture.fiber_layout(max(self.n_fibers, n_tracked), len(self.gauss_s),
                                  tracked=n_tracked),
            threshold=float(spec.get("threshold", 1.0)),
            on_fracture=(self._queue_split if self.bonds is not None else None),
            over=over)

    # ------------------------------------------------------------------ #
    #  Scheduled removal, and the shared event list (core/member_failure)
    # ------------------------------------------------------------------ #
    def _setup_breaking(self):
        """A fibre member has no capacity criterion here -- its section knows
        its own strength, fibre by fibre, and `fracture` is where it breaks --
        but it can be REMOVED at a time (`removals`), and both go to
        the same `member_failures` list and `member_failed` field the other
        line solvers write."""
        self.breaking = Breaking(self.nel, self.mesh, self.p.case.removals, self.log)
        self.member_active = self.breaking.member_active
        self.failure_events = self.breaking.failure_events

    def _check_failure(self, observe=False):
        """After a dynamic force evaluation: the removals now due. `observe`:
        a reading -- the capacity ratios alone."""
        b = self.breaking
        if self.mode != "dynamic" or not b.can_break or getattr(self, "_trial_mode", False):
            return
        if observe:
            b.check(float(self.time), self.end_actions, observe=True)
            return
        gone = b.check(float(self.time), self.end_actions)
        if len(gone):
            self._deactivate(gone)
            b.diagnostics(getattr(self, "diagnostics", None))
            # what the removal set loose is looked at between two steps, as
            # a split is (`apply_pending_splits`): only where members can
            # split, the frame solvers' scope
            if self.bonds is not None:
                self._pending_debris = True

    def _deactivate(self, ids):
        ids = np.asarray(ids, int)
        if len(ids):
            self._active[ids] = 0
            self._refresh_kernel_mask()

    def _live_members(self):
        """(nel,) bool: neither removed nor fractured (a fracture without a
        `breaking` table is in `_active` alone)."""
        return self.breaking.member_active & self._active.astype(bool)

    def _queue_split(self, newly, t):
        """The `on_fracture` seam of v0.1, and all it does is write a note.

        NOT the split itself, because of WHERE this is called from: inside the
        internal-force evaluation, where `F`, `fint` and the boundary masks of
        the step in progress are already vectors of the current length. Moving
        a bond changes which particle carries a member's mass, and doing that
        halfway through a step would leave the step's own force vector
        describing a structure that no longer exists. The loop applies the
        queue at the end of the step instead (`apply_pending_splits`), which
        is the only instant at which the model is between two consistent
        states.
        """
        self._pending_splits.extend(int(e) for e in newly)

    def apply_pending_splits(self, t):
        """Move the fractured members' bonds onto their own particles.

        WHAT A SPLIT IS, in one line: the member's share of the mass moves
        from the frame's particle to the fragment's, and the fragment's
        particle starts with the frame particle's position and velocity.

        MASS IS CONSERVED because the same row leaves one particle and lands
        on another (`_element_mass_share`, one definition). MOMENTUM IS
        CONSERVED for the same reason and no other: the mass that moves takes
        the velocity it already had, so sum(m v) is unchanged at the instant of
        the split. Giving the fragment anything else -- a separation velocity,
        a zeroed velocity -- would be inventing an impulse, and the invariant
        test would be right to fail.

        WHAT THE FRAGMENT IS NOW: its two particles, and the dead member
        between them. The member carries no force, so under gravity
        alone both particles see the same acceleration and the fragment falls
        as a rigid segment; nothing holds it together against anything else,
        which is honest about what this stage does and does not have -- there
        is no contact yet, so it falls THROUGH the floor below it.
        """
        debris, self._pending_debris = self._pending_debris, False
        if not self._pending_splits:
            # a removal alone can set a piece of the frame loose too
            return bool(debris) and self._debris_isotropic(self._grounded(self._split_held()),
                                                           self._energy_ledger())
        pending, self._pending_splits = self._pending_splits, []
        pm = self.physical_mass.reshape(self.nn, self.dofs.stride)
        moved = []
        # a particle held in this step is the world's: an end there leaves it
        # even when it is the particle's only bond, or the fragment stays
        # clamped where it broke off (the frame solvers' rule)
        held = self._split_held()
        ledger = self._energy_ledger()
        for e in sorted(set(pending)):
            share = self._element_mass_share(e)
            for k, parent, new in self.bonds.detach(e, held=held):
                # the mass, the state and the ledger, as the frame solvers
                # move them (core/beam_family)
                self._move_end(e, parent, new, share, pm, ledger)
                moved.append((e, k, parent, new))
            self.mesh.elements[e] = self.bonds.bonds[e]
            if getattr(self, "_edofs", None) is not None:
                self._edofs[e] = self.dofs.element_dofs(self.mesh.elements[e])
                self._x0[e] = self.mesh.nodes[self.mesh.elements[e]]
        if not moved:
            return False
        # A CONNECTOR THAT HOLDS NOTHING UP GOES, and a particle left with
        # nothing that is a body stops where it is: a girder member that
        # fractured over its bearing left the bearing's particle on the
        # connector alone, with no mass, and the step bound at 5e-16 s (the
        # review of v1.18; the frame solvers' rules, v0.5)
        device = ~np.asarray(self._is_fiber, bool)
        if device.any():
            touched = {int(p) for _e, _k, p, _n in moved}
            hang = self._hanging_devices(device, held, device)
            if hang:
                self._deactivate(hang)
                b = self.breaking
                b.member_active[hang] = False
                b.member_failed[hang] = True
                b.failure_events.extend([float(t), e, 0.0, 2.0] for e in hang)
                touched.update(int(n) for n in self.mesh.elements[hang, :2].ravel())
                self.log(f"  {len(hang)} connector(s) left holding nothing up: removed")
            self._park_bodiless(touched, t, device)
        self.mass = np.maximum(self.physical_mass, 1e-20)
        # A fragment's particles are in no region -- a fragment that
        # inherited its parent's CONSTRAINT regions would be born clamped to
        # the support it just broke away from. Gravity reaches them through
        # their parent (`_with_fragments`, as the frame solvers do); a nodal
        # force stays on the nodes it was written on. They used to join
        # "all": gravity on a named region missed them (the debris hung in
        # the air) and a force on "all" was spread onto them.
        self._rebuild_reference_stiffness()
        self._floor_split_rotations(moved)
        # ON DEBRIS ONE ROTARY INERTIA FOR ALL THREE AXES: a piece the
        # fracture set loose tumbles out of the frame its per-axis inertia
        # was set in (the frame solvers' rule of v1.17, here from v1.18)
        self._debris_isotropic(self._grounded(held), ledger)
        before, self.dt_bound = self.dt_bound, self._time_step_bound()
        self._bc_cache = None
        self.splits.extend((float(t), e, p, n) for e, _k, p, n in moved)
        self.log(f"  split: {len(moved)} particle(s) at t = {t:.6g} s "
                 f"({self.bonds.n_components()} component(s), "
                 f"dt bound {before:.3e} -> {self.dt_bound:.3e} s)")
        # the run's dt is usually the bound itself, and a split that leaves
        # it where it was recomputes it to within a rounding: only a bound
        # that really fell is said (the warning came at every split)
        if self.dt_bound < self.dt * (1.0 - 1e-9):
            self.log(f"  WARNING: dt = {self.dt:.3e} s now exceeds the stability "
                     f"bound {self.dt_bound:.3e} s; the step is NOT adapted "
                     f"mid-run")
        return True

    def _floor_split_rotations(self, moved):
        """The particles a split touched keep a rotary inertia stable at the
        run's step.

        The share that moves is the member's TRUE rotary inertia,
        and the setup raised a rotation below what the step needs to the least
        stable value; the fragment's particle got the unfloored share and the
        bound fell under the step (0.90 dt, a bar along z). As at the setup,
        and as the frame solvers' split: m_rot >= row (dt / dt_safety / f)^2
        on a free rotation, added with its angular momentum kept (v scaled by
        m_old / m_new). A rotation does not weigh, so nothing is booked
        against gravity."""
        if getattr(self, "mode", None) != "dynamic" or not moved:
            return
        stride = self.dofs.stride
        pm = self.physical_mass.reshape(self.nn, stride)
        row = np.asarray(abs(self.K).sum(axis=1)).ravel().reshape(self.nn, stride)
        free = (~self._held_dofs()).reshape(self.nn, stride)
        f = float(getattr(self, "damping_step_factor", 1.0) or 1.0)
        need = (self.dt / (float(self.p.solver.dt_safety) * f)) ** 2
        touched = sorted({int(p) for _e, _k, p, _n in moved} | {int(n) for _e, _k, _p, n in moved})
        for n in touched:
            on = free[n, 3:] & (row[n, 3:] > 1e-20)
            old = pm[n, 3:].copy()
            new = np.where(on, np.maximum(old, row[n, 3:] * need), old)
            if not (new > old).any():
                continue
            rd = self.dofs.node_dofs(np.array([n])).ravel()[3:]
            self.v[rd] = self.v[rd] * np.where(new > 0.0, old / np.maximum(new, 1e-300), 1.0)
            pm[n, 3:] = new
        self.mass = np.maximum(self.physical_mass, 1e-20)

    def _committed_failures(self):
        """How many fibre points of the ACCEPTED state have failed, over every
        law that declares a criterion (fracture on or not). The staged
        relaxation compares it before and after a commit,
        the static solver too (`implicit._settle_ruptures`).

        THE COMMITTED STATE, not the live one. The two are the same wherever
        a commit is compared, except after a restart or an `initial_state`
        taken inside a level: the live state there is the level's TRIAL,
        with the points past their limit already failed, and the commit
        found nothing new -- the level was not solved again and its frame
        paired the intact positions with the cut forces, 372 kN against the
        uninterrupted run's 294 kN, in staged relaxation and in the static
        solver's fallback alike."""
        n = 0
        for g, m in enumerate(self.materials):
            st = self.fiber_committed[g]
            if st.shape[-1] == 0:
                continue
            f = m.failure(st.reshape(-1, st.shape[-1]))
            if f is not None:
                n += int(np.count_nonzero(f > 0.5))
        return n

    def _point_failure(self):
        """(nel, tracked_fibres * n_stations), fibre-major, straight from the laws.

        The gather is the same on both force paths because the kernel's
        consolidated state arrays are VIEWS: `fiber_states[g]` still addresses
        live data after `_relink_kernel_states` (see there), so this reads the
        fibres the kernel just wrote without knowing the kernel exists.

        ONLY THE FIBRES THAT CAN FAIL are gathered (`_failing_groups`, decided
        at setup), and into a buffer that is allocated once: every tracked
        column is written on every call, so there is nothing to zero. This runs
        at every step of every fracture run, and on an RC section it used to
        build a 68-column array of which 64 columns were a law answering None.
        """
        f = self._tracked_failure
        for g, start, n in self._failing_groups:
            st = self.fiber_states[g]
            fg = self.materials[g].failure(st.reshape(-1, st.shape[3]))
            f[:, start:start + n, :] = fg.reshape(st.shape[:3]).transpose(0, 2, 1)
        if self._tracked_valid is not None:
            f *= self._tracked_valid
        return f.reshape(len(f), -1)

    def _setup_strain_penetration(self, spec_root):
        """Per-element rebar parameters, so a hinge element can differ.

        `strain_penetration: {region: base, bar_diameter: 0.020}` gives the
        elements touching that region a bar strain reduced by
        Le / (Le + Lsp), with Lsp = 0.022 fy db (Priestley) -- the length over
        which the bar keeps straining past the member face, into the anchorage.
        Nothing is fitted: Lsp comes out of fy and db.

        The array is built whether or not the kernel is available, because the
        NumPy reference reads the same rows. A reference that could not
        express per-element parameters would stop being a reference for the
        thing the kernel actually computes.

        `_spar` is (nel, NS, n_par): rebar slot j of element e carries the
        parameters of the j-th bar of e's section; a section with fewer bars
        than the widest repeats its first row into the padded slots, which
        the kernel evaluates at zero area. With several
        sections each may name its own anchorage (`fiber_sections[k].anchorage`)
        and it applies to that section's members only.
        """
        self._spar = None
        self.strain_penetration = None
        wanted = bool(spec_root.get("strain_penetration")) or any(
            lay.entry.get("anchorage") for lay in self._layouts)

        def decline(why):
            # it returned early and the anchorage was dropped without a word
            raise ValueError(f"strain_penetration (anchorage) cannot be applied: {why}")
        if not self.is_hybrid:
            if wanted:
                decline("the section has no bars")
            return
        n_par = None
        rows_of = {}
        for g in range(len(self.materials)):
            lay = self._layouts[int(self.group_section[g])]
            if g not in lay.rebar_groups:
                continue
            try:
                rows_of[g] = np.asarray(self.materials[g].numba_params(), dtype=float)
            except AttributeError:
                if wanted:
                    decline(f"the bar law {self.materials[g].name!r} has no strain scale "
                           "(use steel02 for anchored bars)")
                return                   # a law without the array form; kernel declines too
            n_par = len(rows_of[g]) if n_par is None else n_par
            if len(rows_of[g]) != n_par:
                if wanted:
                    decline("the bars use laws with different parameter layouts")
                return                   # the kernel declines this too, by name
        self._spar = np.zeros((self.nel, self.n_steel, n_par))
        for k, lay in enumerate(self._layouts):
            rows = self._rows[k]
            if not lay.rebar_groups:
                continue
            for j in range(self.n_steel):
                g = lay.rebar_groups[min(j, lay.n_s - 1)]
                self._spar[rows, j, :] = rows_of[g]

        specs = []
        if spec_root.get("strain_penetration"):
            specs.append((spec_root["strain_penetration"], None))
        for k, lay in enumerate(self._layouts):
            if lay.entry.get("anchorage"):
                specs.append((lay.entry["anchorage"], k))
        if not specs:
            return
        all_touched, report = [], []
        for spec, k in specs:
            idx = set(self._region(spec["region"]).tolist())
            db = float(spec["bar_diameter"])
            touched = [t for t, conn in enumerate(self.mesh.elements)
                       if any(int(n) in idx for n in conn)
                       and (k is None or int(self._section_index(t)) == k)]
            if not touched:
                # Defensive: the region system already refuses a region that
                # selects no node, and in a line mesh every node belongs to an
                # element, so this needs a mesh with stray nodes to fire. Untested
                # for that reason -- see tests/test_strain_penetration.py.
                raise ValueError(
                    f"strain_penetration region {spec['region']!r} touches no element"
                    + ("" if k is None else f" of fibre section {k}"))
            for t in touched:
                lay = self._layouts[self._section_index(t)]
                le = float(self.length[t])
                for j, g in enumerate(lay.rebar_groups):
                    mat = self.materials[g]
                    if getattr(mat, "strain_scale", None) is None:
                        # column 8 is steel02's strain scale; it is another
                        # law's own parameter (steel_cyclic: b1)
                        decline(f"the bar law {mat.name!r} has no strain scale (use steel02)")
                    fy = float(getattr(mat, "Fy", getattr(mat, "yield_strength", 0.0)))
                    if fy <= 0:
                        raise ValueError("strain_penetration needs a rebar with a yield strength")
                    lsp = 0.022 * (fy / 1e6) * (db * 1e3) / 1e3          # Priestley, in m
                    self._spar[t, j, 8] = le / (le + lsp)
                    report.append(lsp)
            all_touched.extend(touched)
        # The scale multiplies the law's whole input, imposed strain included:
        # a tendon's prestrain or a bar's temperature strain in an anchored
        # element would be cut with the anchorage's share.
        # Refused until the kernel scales only the mechanical part.
        cols = np.concatenate([self.group_indices[g] for lay in self._layouts
                               for g in lay.rebar_groups]) if self.n_steel else np.zeros(0, int)
        pre = np.asarray(self.fiber_prestrain)
        if len(cols) and np.abs(pre[np.ix_(all_touched, cols)]).max(initial=0.0) > 0.0:
            decline("a tendon (a prestrained steel fibre) runs through an anchored element")
        loads = list(self.p.get("loads") or []) + [l for st in (self.p.get("steps") or [])
                                                    for l in (st.get("loads") or [])]
        if any(l.get("temperature") is not None for l in loads):
            decline("a temperature load would be scaled with the anchorage's share on the bars")
        self.strain_penetration = {
            "region": specs[0][0]["region"] if len(specs) == 1 else
                      [sp["region"] for sp, _k in specs],
            "elements": all_touched,
            "Lsp": float(np.mean(report)),
            "scale": float(self._spar[all_touched[0], 0, 8]),
        }

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

    # ---------------------------------------------------------------- time
    #
    #  Creep and shrinkage on a FIBRE section. The mechanism is
    #  v0.4's, one level down: the beam path subtracts a free deformation
    #  from the member's and divides the member stiffness by (1 + phi_cur);
    #  here the free strain goes into the fibre's own imposed-strain slot and
    #  the (1 + phi_cur) is a SCALE ON THE LAW'S INPUT, which is what
    #  `concrete02.strain_scale` exists for. The material is then
    #  evaluated at the elastic strain itself -- not at an effective modulus --
    #  so it answers with its own nonlinear stress there.
    #
    #  ONE JUDGEMENT, and the plan records it before it was made: the imposed
    #  slot is (nel, n_fibres) and has NO INTEGRATION-STATION INDEX, while a
    #  member's three stations carry different moments and so different creep.
    #  Creep is therefore evaluated ONCE PER ELEMENT, at the middle station.
    #  Not "for now": a draped tendon's prestrain, which varies along the
    #  element for the same reason, is already evaluated at the element
    #  midpoint and uses the same slot. Refining this means giving
    #  `_cpre` a station axis and changing the kernel signature, which is a
    #  different piece of work.
    def _setup_time_dependence(self):
        """The fibre materials' `creep` / `shrinkage` specs, read once.

        A model whose materials declare neither is untouched: `time_dependent`
        is False and nothing below ever runs.
        """
        self._creep_spec = self._shrink_spec = None
        self._creep_who = None
        #: [(fibre mask, spec, concrete age in days when it was stressed), ...]
        #: -- a LIST, not one law, because relaxation has no history: it is a
        #: pure function of the time since stressing, so several tendon steels
        #: in one model cost nothing. Creep cannot say that.
        self._relax = []
        self._prestrain_now = self.fiber_prestrain
        self._history = None
        self._prev_station = None
        self._phi_cur = 0.0
        self._scale_cur = None
        self._scale_el = None
        self._base_scale = {}
        self._creep_cols = None
        self._age0 = float(self.p.solver.age or 28.0)
        self._current_age = self._age0
        self._step_ages = (self._age0, None, "log")
        self.ages = []
        conc = {g for lay in self._layouts for g in lay.conc_groups}
        groups = []
        for g, mat in enumerate(self.materials):
            c, s = getattr(mat, "creep", None), getattr(mat, "shrinkage", None)
            if getattr(mat, "ageing", None) is not None:
                raise ValueError(
                    f"material {mat.name!r} declares `ageing` on a FIBRE "
                    "section, and this solver has none. Ageing changes Ec, "
                    "which on a nonlinear law is not one number -- the peak "
                    "strain and the strength age with it -- so it cannot ride "
                    "the strain scale creep uses. Declare the aged modulus "
                    "directly instead, as cases/case_28d_confederation_sdl.yaml "
                    "does on the beam path, and say in the case which age it is")
            if c is None and s is None:
                continue
            if g not in conc:
                raise ValueError(
                    f"material {mat.name!r} declares `creep` or `shrinkage` "
                    "but its fibres are REINFORCEMENT. Steel does not creep; "
                    "what it does is relax, which is a different law and is "
                    "not in this library (docs/creep.md)")
            if getattr(mat, "strain_scale", None) is None:
                raise ValueError(
                    f"material {mat.name!r} declares `creep` or `shrinkage` "
                    "but has no `strain_scale`, which is how a fibre station "
                    "carries the creep of the interval it is arriving in. "
                    "`concrete02` has one; `linear_elastic` "
                    "does not, and for a linear fibre the same thing is said "
                    "by declaring the modulus you mean")
            groups.append((g, c, s))
        self._read_relaxation(conc)
        if not groups:
            return
        specs = {(repr(sorted((c or {}).items())), repr(sorted((s or {}).items())))
                 for _, c, s in groups}
        if len(specs) > 1:
            raise ValueError(
                "two concrete fibre materials declare DIFFERENT `creep` or "
                "`shrinkage` specs. One law per model in this version -- the "
                "history and the station's own phi are kept once for the "
                "whole model, not per group. Several laws are buildable (the "
                "scale and the imposed strain are already per fibre); no case "
                "needs them yet, so the refusal is deliberate rather than an "
                "oversight")
        self._creep_spec, self._shrink_spec = groups[0][1], groups[0][2]
        if self._creep_spec is not None:
            mat = self.materials[groups[0][0]]
            # a dynamic run carries creep only on a clock that stands still:
            # the history an initial_state restored is then a frozen imposed
            # strain; a step that moves the clock is refused
            # (v1.0: `to_age` here, a moved `age` in `step_data`)
            self._creep_who = f"material {mat.name!r}"
            if self.p.solver.mode == "dynamic":
                for st in self.p.get("steps") or []:
                    if st.get("to_age") is not None:
                        creep.refuse_moving_clock(st, (float(st.get("age") or self._age0),
                                                       float(st["to_age"]), "log"), None,
                                                  self._creep_who)
            # mc90's phi is referenced to Ec28; `ageing` is
            # refused above, so it is the spec's own `E28`
            self._creep_spec = creep.bind_reference(
                self._creep_spec, self._modulus(mat), None, f"material {mat.name!r}")
        self._base_scale = {g: float(self.materials[g].strain_scale)
                            for g, _, _ in groups}
        # which fibre COLUMNS creep, in the section-row order of `fiber_y`
        self._creep_cols = np.zeros(self.fiber_y.shape, dtype=bool)
        for k, lay in enumerate(self._layouts):
            cols = np.concatenate([self.group_indices[g] for g, _, _ in groups
                                   if g in lay.conc_groups] or [np.zeros(0, int)])
            if len(cols):
                self._creep_cols[np.ix_(self._rows[k], cols)] = True
        if self._creep_spec is not None:
            self._history = creep.CreepHistory(self._creep_spec)
        #: the integration station(s) creep is read at -- the element's
        #: middle. `gauss_s` is s/L in [0, 1]: the middle is 0.5. It read
        #: argmin(|s|), which is the END station s = 0 of every Lobatto rule --
        #: a cantilever's creep came out 20% high (found in review,
        #: v1.0). An EVEN rule (gauss2, the default, and lobatto4)
        #: has no station there: its two middle stations straddle it
        #: symmetrically and their mean is the strain at the middle. Reading
        #: the first of them alone read 0.21 L from the element's
        #: own node 0, so the same cantilever crept +6.7% fixed at one end
        #: and -5.9% fixed at the other (found in review)
        d = np.abs(np.asarray(self.gauss_s, dtype=float) - 0.5)
        self._creep_stations = tuple(int(i) for i in np.flatnonzero(d <= d.min() + 1e-12))
        self._creep_station = self._creep_stations[0]

    def _read_relaxation(self, conc):
        """`relaxation` on the steels, one entry per tendon fibre group.

        It does nothing at all on a fibre whose prestrain is zero, so a
        material a section uses for BOTH tendons and ordinary bars is fine and
        needs no separate declaration. A spec that would touch no prestrained
        fibre at all is refused instead, because that is the case where it
        would silently do nothing.
        """
        for g, mat in enumerate(self.materials):
            spec = getattr(mat, "relaxation", None)
            if spec is None:
                continue
            if g in conc:
                raise ValueError(
                    f"material {mat.name!r} declares `relaxation` but its "
                    "fibres are CONCRETE. Concrete's version of this is "
                    "`creep`, and the two are different laws -- relaxation "
                    "here is the prestressing steel losing the strain it was "
                    "stretched to")
            relaxation.validate(dict(spec))
            mask = np.zeros(self.fiber_y.shape, dtype=bool)
            for k, lay in enumerate(self._layouts):
                if g in lay.rebar_groups:
                    mask[np.ix_(self._rows[k], self.group_indices[g])] = True
            mask &= self.fiber_prestrain != 0.0
            if not mask.any():
                raise ValueError(
                    f"material {mat.name!r} declares `relaxation` but no fibre "
                    "of it carries a PRESTRAIN, so there is nothing to relax "
                    "and the declaration would do nothing. Relaxation is the "
                    "loss of a tendon's initial strain; an ordinary bar's "
                    "stress comes from the structure and is not held")
            # default solver.age, and not after the analysis starts
            # (v1.0: relaxation.stressed_at)
            t0 = relaxation.stressed_at(spec, self._age0, self.p.get("steps") or [],
                                        f"material {mat.name!r}")
            self._relax.append((mask, dict(spec), t0))

    def _shifted(self) -> bool:
        """Whether a member is on a clock of its own (1.8): one born in an
        analysis step at its `activate_age` (core/analysis_steps)."""
        shift = self.staging.age_shift
        if shift is None:
            return False
        return bool(np.any(shift) or np.isfinite(self.staging.age_floor).any())

    @property
    def time_dependent(self):
        return getattr(self, "_creep_spec", None) is not None \
            or getattr(self, "_shrink_spec", None) is not None \
            or bool(getattr(self, "_relax", ()))

    def _time_strain(self, age):
        """(nel, n_fibres) of MINUS the free strain creep and shrinkage have
        imposed by `age`, zero on every fibre that does not creep.

        Minus, because what goes in the imposed slot is what the MATERIAL
        sees added to the section's plane strain, and a material sees the
        mechanical strain -- the same sign a temperature uses.

        Only the increments ALREADY RECORDED appear here. The one this station
        is about to produce is carried by `_age_scale` instead, because it is
        not known until the station is solved.
        """
        out = np.zeros(self.fiber_y.shape)
        shifted = self._shifted()
        if self._shrink_spec is not None:
            if shifted:
                # a segment born at `floor` shrank in its formwork before
                # then, which is no load: its shrinkage counts from its birth
                # (the frames subtract it from the birth offset, 1.2)
                for rows, sh, floor in self._age_groups(np.arange(self.nel)):
                    eps = creep.shrinkage_strain(self._shrink_spec, max(age + sh, floor))
                    if np.isfinite(floor):
                        eps -= creep.shrinkage_strain(self._shrink_spec, floor)
                    if eps:
                        out[rows] -= eps
            else:
                eps = creep.shrinkage_strain(self._shrink_spec, age)
                if eps:
                    out -= eps
        if self._history is not None:
            c = (creep.shifted_strain(self._history, age, self._age_groups) if shifted
                 else self._history.strain(age))
            if c is not None:
                out -= c
            # the current increment creeps from where the last station left
            # off, so its own creep is `phi_cur * (e - e_prev)`: the `e` half
            # is the scale in `_age_scale` and the `e_prev` half is an imposed
            # strain the other way
            phi = np.asarray(self._phi_cur, dtype=float)
            if np.any(phi) and self._history._last is not None:
                out += (phi[:, None] if phi.ndim else phi) * self._history._last
        return np.where(self._creep_cols, out, 0.0)

    def _age_scale(self):
        """`1 / (1 + phi_cur)` on the creeping fibres' own input.

        With the imposed strain of `_time_strain`, a fibre then satisfies

            eps_el (1 + phi_cur) = eps_section - eps_free_past + phi_cur eps_prev

        which is the increment's own creep solved for TOGETHER with it -- the
        implicit step. Leaving it out makes the scheme explicit, and on a held
        strain that is not merely first order but marginally UNSTABLE
        (v0.4 measured it; case_30d checks it here).

        Both paths have to be written: the reference path reads
        `mat.strain_scale` on every call (`numba_params` is rebuilt each
        time), the fused kernel reads column 7 of its own `_cpar`. A scale
        that reached only one of them would be invisible until a case ran, and
        would then look like a creep result.
        """
        phi = np.asarray(self._phi_cur, dtype=float)
        if phi.ndim:
            # one factor per member (1.8): each segment on its own clock. The
            # reference path sets the law's scale member by member
            # (`_scale_el`, read in `_internal_force_true`), the kernel reads
            # its per-member `_cpar`
            f_el = 1.0 / (1.0 + phi)
            if isinstance(self._scale_cur, np.ndarray) and self._scale_cur.shape == f_el.shape \
                    and np.all(np.abs(f_el - self._scale_cur) < 1e-14):
                return
            self._scale_cur = f_el
            self._scale_el = f_el
            for g, base in self._base_scale.items():
                self.materials[g].strain_scale = base
            if getattr(self, "_cpar", None) is not None:
                for k, lay in enumerate(self._layouts):
                    c_idx = self._c_order[k]
                    keep = self._creep_cols[self._rows[k][0], c_idx]
                    cols = np.flatnonzero(keep)
                    if len(cols):
                        base = np.array([self._base_scale[g]
                                         for g in self._c_group[k][cols]])
                        rows = self._rows[k]
                        self._cpar[np.ix_(rows, cols, [7])] = \
                            (base[None, :] * f_el[rows][:, None])[:, :, None]
            return
        f = 1.0 / (1.0 + float(phi))
        self._scale_el = None
        if self._scale_cur is not None and not isinstance(self._scale_cur, np.ndarray) \
                and abs(f - self._scale_cur) < 1e-14:
            return
        self._scale_cur = f
        for g, base in self._base_scale.items():
            self.materials[g].strain_scale = base * f
        if getattr(self, "_cpar", None) is not None:
            for k, lay in enumerate(self._layouts):
                c_idx = self._c_order[k]
                keep = self._creep_cols[self._rows[k][0], c_idx]
                cols = np.flatnonzero(keep)
                if len(cols):
                    base = np.array([self._base_scale[g]
                                     for g in self._c_group[k][cols]])
                    self._cpar[np.ix_(self._rows[k], cols, [7])] = \
                        (base * f)[None, :, None]

    def hold_time_strain(self) -> bool:
        """Hold the creep and shrinkage reached as an imposed strain at the
        instantaneous modulus -- the state a dynamic run from here starts in
        -- for a tangent that is asked for vibration (modes, a response
        spectrum).

        The last time station leaves `1 / (1 + phi_cur)` on the law's input:
        that is the IMPLICIT creep step of the interval just solved, a
        sustained-load stiffness, and the modes and spectrum at the end of a
        creep run were taken from it -- periods 20-35% long, and longer the
        fewer stations the run used. With the station recorded, phi_cur = 0
        and the history's own strain give the same elastic strain in every
        fibre (the equilibrium does not move) at the true stiffness.

        Returns True when the solver is time-dependent and the state is a
        recorded station (or has no creep history to hold), False when there
        is nothing to hold or the state is between stations.
        """
        if not self.time_dependent:
            return False
        age = float(self._current_age)
        prev = self._prev_station
        if self._creep_spec is not None and prev is not None and age > prev * (1 + 1e-12) + 1e-12:
            return False
        keep = self._step_ages
        self._step_ages = (age, None, "log")
        try:
            self._refresh_imposed(1.0)
        finally:
            self._step_ages = keep
        return True

    def _record_time_station(self, age):
        """Called once a time station has reached equilibrium: the ELASTIC
        strain every creeping fibre settled at goes into the history, and
        everything after it creeps from there.

        `epsP` -- state column 2 of `concrete02` -- is exactly that and needs
        no correction: it is the strain the MATERIAL saw, which is already
        `(eps_section - eps_free) / (1 + phi_cur)` because the scale was
        applied to the law's input. The beam path has to divide by
        `(1 + phi_cur)` by hand at this point; here the division has already
        happened, in the one place that could not forget it.
        """
        if self._history is None:
            return
        elastic = np.zeros(self.fiber_y.shape)
        gi = list(getattr(self, "_creep_stations", (self._creep_station,)))
        for k, lay in enumerate(self._layouts):
            rows = self._rows[k]
            for g in lay.conc_groups:
                if g not in self._base_scale:
                    continue
                st = self.fiber_committed[g]
                if st.size:
                    # the middle of the element: one station, or the mean of
                    # the two that straddle it
                    elastic[np.ix_(rows, self.group_indices[g])] = \
                        st[rows][:, gi][..., 2].mean(axis=1)
        built = getattr(self, "member_built", None) if self.plan is not None else None
        if built is not None:
            elastic[~np.asarray(built, bool)] = 0.0      # a member not built has no history (1.8)
        self._history.add(creep.station_date(age, self._prev_station),
                          np.where(self._creep_cols, elastic, 0.0))
        if self.staging.age_shift is not None:
            # the interval itself, so that a member on its own clock files the
            # increment where its station's phi dated it (1.2, the frames')
            if not hasattr(self._history, "windows") or len(self._history.windows) != len(self._history.ages) - 1:
                self._history.windows = [(None, a) for a in self._history.ages[:-1]]
            self._history.windows.append((self._prev_station, float(age)))
        self._prev_station = age

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

    def _latch_fracture(self, committed: bool = False, t: float = None):
        """Ask the laws what has failed, AFTER the forces are assembled.

        After, and not before, for a reason that decides what the first
        fractured step looks like: a fibre that ruptures during this step has
        already had its stress zeroed by its own law, so the section force
        this step is the one the broken section carries. Latching first would
        deactivate the element a step EARLIER than its own material did, and
        the two halves would then disagree about which step the member died in.

        Nothing happens during a staged relaxation. `_trial_mode` means the
        state being read is a trial that is about to be thrown away and
        re-driven from the last converged one (see `fiber_committed`), and a
        latch set from a trial path cannot be un-set.
        """
        if self.fracture is None:
            return
        if getattr(self, "_trial_mode", False) and not committed:
            return
        t = float(self.time if t is None else t)
        newly = self.fracture.update(self._point_failure(), t=t)
        if len(newly):
            self._active[newly] = 0
            self._refresh_kernel_mask()
            self.log(f"  fracture: element(s) {', '.join(map(str, newly[:8]))}"
                     f"{' ...' if len(newly) > 8 else ''} at t = {t:.6g} s "
                     f"({self.fracture.n_fractured}/{self.nel} fractured)")
            if getattr(self, "breaking", None) is not None:
                # the same event list a mixed_line break goes to: cause 0,
                # and the ratio is the fraction of the section that is gone
                b = self.breaking
                fresh = newly[~b.member_failed[newly]]
                b.failure_events.extend(
                    [float(t), int(e), float(self.fracture.fac[e]), 0.0] for e in fresh)
                # FAILED as well as inactive: a fractured member reached by
                # its removal later was reported failed a second time
                b.member_active[newly] = False
                b.member_failed[newly] = True
                b.end_broken[newly] = True
                b.diagnostics(getattr(self, "diagnostics", None))


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
        return bool(getattr(self, "_trial_mode", False))

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

    def _internal_force_kernel(self):
        """Fused path. `_internal_force_true` remains the reference it copies."""
        if getattr(self, "_trial_mode", False):
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
                     self._kernel_birth(), *self._shear_law_arrays(), bool(getattr(self, "_sl_update", False)))
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
                        _NO_ROWS, False, False, _NO_BIRTH)
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
        if getattr(self, "_trial_mode", False):
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
            out[dof] += T.T @ fl
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
            for ts, ac, direction in ground:
                ag = np.interp(t, ts, ac, left=0., right=0.)
                mass = self.dofs.field(self.physical_mass, TRANSLATION)
                self.dofs.add_nodal(f, np.arange(self.nn), -mass*ag*direction, TRANSLATION)
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
        if not getattr(self, "_trial_mode", False):
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
