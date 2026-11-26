"""The fibre beam's sections (1.22, out of `fiber_beam_solver`): how each
member's section is laid out in fibres and bars, its tendons, its shear law
and its strain penetration -- the setup the solver's force reads. A mixin of
`FiberBeamSolver`, which is where its state lives."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..elements.fiber_section import (box_fibers, circle_fibers,
                                      rect_fibers, tube_fibers)


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


class FiberSections:
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
