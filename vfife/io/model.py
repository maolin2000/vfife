"""The input model: materials -> sections -> assign, and reinforcement as geometry.

THE FIVE NOUNS AND THE ONE RULE
    materials       constitutive laws, by name
    sections        geometry + material -- the thing that gets assigned
    mesh, reinforcement   where things are: the meshed continuum, and bars
                    placed by drawing coordinates, independent of the mesh
    regions         named node sets on the mesh
    assign          the ONLY place a section meets geometry

    Anything with physics is a section; anything with a place is geometry or
    a region; the two meet only in `assign`; anything between two things is
    an interaction (reserved: `interactions`); anything kinematic is a
    constraint. That rule decides where a new feature goes.

    It is the Abaqus / OpenSees / Code_Aster structure -- material, section,
    assignment -- with the reinforcement treated the way DIANA and ANSYS
    treat it: as its own geometry, located inside whatever mesh exists.

WHAT THIS MODULE DOES
    Validates that form and LOWERS it to the solver's internal case (the
    `schema.Case` model): one element type and material for the continuum,
    `members` for line groups, `reinforcement` entries with area, perimeter
    and material resolved from their sections. The solver never sees a
    section name. The lowering is total -- every field of the internal case
    is produced here -- so the internal form can change without the input
    form moving.

SEVERAL SOLID SECTIONS
    Each `assign: {region, section}` with a solid section becomes a material
    block: the elements whose every node lies in the region carry that
    section's material and formulation. Assignments apply in order and a
    later one takes the elements an earlier one also covers, so
        - {region: all,  section: cover}
        - {region: core, section: confined}
    reads as "paint the beam, then paint the core" (Abaqus' rule). An element
    no assignment covers, or a section that covers no element, is refused by
    the solver with the element ids.

LIMITS, refused with a message rather than silently narrowed
    * a shell section is the whole mesh, and a coupled model (solid + line
      members) carries one solid material
    * every LINE section may name its own material: the one
      that differs from the model's travels down as `material` on its member
      group. Fiber sections and reinforcement have their own too
    * `path` reinforcement draws a route from `points` here; reading one out
      of the Gmsh file (`path: {group: ...}`) is not implemented
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from .. import sections as sec_geom
from .schema import ConnectorLawSpec, coerce_numbers
from ..prep.offsets import OffsetArmSpec
from ..prep.tendons import TendonSpec


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StirrupSpec(Strict):
    """Stirrups for the truss-model cracked shear stiffness."""
    #: area of one stirrup set, all legs (m^2)
    area: float = Field(gt=0.0)
    #: spacing along the member (m)
    spacing: float = Field(gt=0.0)
    #: effective depth d (m)
    depth: float = Field(gt=0.0)
    #: stirrup modulus: 2e11 Pa by default, converted to the declared `units` when they are not SI (2e5 in MPa)
    E: float = Field(2.0e11, gt=0.0)


class FiberShearSpec(Strict):
    """A cracking shear law for a fibre section, in series
    with the fibres' bending: V = GA0 gamma until the section cracks, then
    the cracked stiffness, then (optionally) a stirrup-yield plateau with a
    small hardening; unloading and reloading run to the origin once cracked.

    The cracked stiffness is given, or taken from the 45-degree truss of the
    stirrups (Park & Paulay 1975, sec. 7.3): GA_cr = rho_v / (1 + 4 n rho_v)
    E_s b_w d, rho_v = A_v / (b_w s), n = E_s / E_c. The section cracks in
    shear when its concrete first cracks in flexure (`cracking: flexural`,
    the flexure-shear cracks a frame member actually has), or at a stated
    shear force."""
    #: the direction the shear acts along: `z` (the section's depth, the plane of bending about y) or `y`
    direction: Literal["y", "z"] = "z"
    #: uncracked kappa G A (N)
    stiffness: float = Field(gt=0.0)
    #: cracked GA (N); or give `stirrups` for the truss model
    cracked_stiffness: Optional[float] = Field(None, gt=0.0)
    #: stirrups (the truss model's cracked stiffness)
    stirrups: Optional[StirrupSpec] = None
    #: the web width b_w for the truss model (m); by default the section's width across the shear
    web_width: Optional[float] = Field(None, gt=0.0)
    #: when it cracks: `flexural` (when the member's concrete first cracks in tension) or a shear force (N)
    cracking: Union[Literal["flexural"], float] = "flexural"
    #: the shear at which the stirrups yield (N); beyond it the stiffness is `hardening` times the cracked one
    yield_force: Optional[float] = Field(None, gt=0.0)
    #: post-yield stiffness ratio
    hardening: float = Field(0.01, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _one_cracked(self):
        if (self.cracked_stiffness is None) == (self.stirrups is None):
            raise ValueError("a fibre shear law needs `cracked_stiffness` OR `stirrups` (the truss model)")
        if not isinstance(self.cracking, str) and self.cracking <= 0.0:
            raise ValueError("shear `cracking` is `flexural` or a positive shear force")
        return self


# --------------------------------------------------------------------------- #
#  sections
# --------------------------------------------------------------------------- #

SectionType = Literal["solid", "shell", "beam", "bar", "fiber", "connector"]


#: old spelling -> the one name; read with a warning, rewritten by `vfife migrate`
DEPRECATED_SECTION_KEYS = {"b": "width", "h": "height", "d": "diameter", "torsion_constant": "j"}


class SectionSpec(Strict):
    #: a girder shape's faces once it is patches (1.2; not a key of the file)
    _girder: Optional[Dict[str, Any]] = PrivateAttr(None)
    #: what the section is, and so which element carries it: solid (the meshed continuum),
    #: shell, beam (beam2), fiber (beam2_fiber), bar (truss2: bars, cables), connector (connector2)
    type: SectionType
    #: a name in `materials`; optional for a connector; required on a fiber
    #: section (the law of every patch that names none of its own)
    material: Optional[str] = None
    # --- solid ---
    #: solid: volumetric locking treatment for hexa8 (`b-bar` default; `selective`, `reduced` with hourglass control, `full`)
    formulation: Optional[Literal["full", "reduced", "b-bar", "selective"]] = None
    #: solid, `reduced` only: hourglass stabilisation coefficient
    hourglass_coeff: Optional[float] = None
    # --- shell ---
    #: shell: plate thickness (m)
    thickness: Optional[float] = Field(None, gt=0.0)
    #: shell: transverse shear correction factor (5/6 default)
    shear_correction: Optional[float] = Field(None, gt=0.0)
    #: shell: drilling-rotation stiffness factor
    drilling_factor: Optional[float] = Field(None, gt=0.0)
    #: shell: a shell4 whose corners leave a plane by more than this fraction is refused
    warpage_tolerance: Optional[float] = Field(None, gt=0.0)
    # --- beam / bar: a shape, or the integrated properties directly ---
    #: `tube` (circular hollow) and `box` (rectangular/square hollow) are fiber
    #: sections only: a steel spatial structure is built of tubes, and a tube
    #: discretised as a solid puts fibres where there is no steel.
    shape: Optional[Literal["circle", "rect", "I", "general", "tube", "box", "patches",
                            "box_girder", "t_girder"]] = None
    #: circle (beam, bar) and fiber tube: outer diameter; a circle with `t` is a tube (the old spelling `d` is deprecated since v1.0)
    diameter: Optional[float] = Field(None, gt=0.0)
    #: wall thickness of a hollow shape (circle + t, rect + t; fiber tube/box)
    t: Optional[float] = Field(None, gt=0.0)
    #: rect (beam, fiber) and fiber box: width along the section's y axis, outer for a box (the old spelling `b` is deprecated since v1.0)
    width: Optional[float] = Field(None, gt=0.0)
    #: rect (beam, fiber) and fiber box: depth along z, outer for a box (the old spelling `h` is deprecated since v1.0)
    height: Optional[float] = Field(None, gt=0.0)
    # --- box_girder / t_girder (1.2): the dimensions on the drawing ---
    #: box_girder, t_girder: the whole deck, tip to tip
    top_width: Optional[float] = Field(None, gt=0.0)
    #: box_girder: the bottom slab, outside to outside; t_girder: the bottom flange's width (an I or a bulb-T)
    bottom_width: Optional[float] = Field(None, gt=0.0)
    #: box_girder: from a tip to the web's outer face, measured at the top of the deck
    cantilever: Optional[float] = Field(None, ge=0.0)
    #: box_girder: the top slab between the webs; t_girder: the flange
    top_thickness: Optional[float] = Field(None, gt=0.0)
    #: box_girder: the bottom slab; t_girder: the bottom flange
    bottom_thickness: Optional[float] = Field(None, gt=0.0)
    #: box_girder, t_girder: each web, measured HORIZONTALLY on an inclined web
    web_thickness: Optional[float] = Field(None, gt=0.0)
    #: the cantilever (flange outstand): one value or [at the web, at the tip]; default top_thickness
    cantilever_thickness: Optional[Union[float, List[float]]] = None
    #: box_girder: the fillet in each inside top corner of the cell, [width, height]
    top_haunch: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    #: box_girder: the fillet in each inside bottom corner, [width, height]
    bottom_haunch: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    #: box_girder: the number of cells (1 default); interior webs vertical, evenly between the outer webs' centre lines
    cells: Optional[int] = Field(None, ge=1)
    #: box_girder, t_girder fibre sections: fibres through each wall's thickness (3 default)
    layers: Optional[int] = Field(None, ge=1)
    #: box_girder, t_girder fibre sections: the longest a fibre runs along a wall (default height / 8)
    fiber_size: Optional[float] = Field(None, gt=0.0)
    #: box_girder, t_girder fibre sections: where the member line passes -- `centroid` (default), the top of the deck, the bottom of the girder; bars and tendons are written about it
    origin: Optional[Literal["centroid", "top", "bottom"]] = None
    #: I: flange width (along y)
    bf: Optional[float] = Field(None, gt=0.0)
    #: I: flange thickness
    tf: Optional[float] = Field(None, gt=0.0)
    #: I: clear web height between the flanges (total depth = hw + 2 tf)
    hw: Optional[float] = Field(None, gt=0.0)
    #: I: web thickness
    tw: Optional[float] = Field(None, gt=0.0)
    #: general: cross-section area (m^2); on a shape, overrides the derived value
    area: Optional[float] = Field(None, gt=0.0)
    #: second moment about the section y axis (fibres vary in z), m^4
    iy: Optional[float] = Field(None, gt=0.0)
    #: second moment about z, m^4
    iz: Optional[float] = Field(None, gt=0.0)
    #: Saint-Venant torsion constant, m^4; on a fibre section the polar moment of the fibres when absent (the fibre sections' old spelling `torsion_constant` is deprecated since v1.0)
    j: Optional[float] = Field(None, gt=0.0)
    #: Timoshenko: shear area for y-shear (default from the shape)
    ay: Optional[float] = Field(None, gt=0.0)
    #: Timoshenko: shear area for z-shear
    az: Optional[float] = Field(None, gt=0.0)
    #: bar: contact perimeter for bond (default from the shape)
    perimeter: Optional[float] = Field(None, gt=0.0)
    #: beam theory: Euler-Bernoulli (default) or Timoshenko (needs ay, az)
    theory: Optional[Literal["euler", "timoshenko"]] = None
    #: Timoshenko shear correction factor on ay (1 = use ay as given)
    kappa_y: Optional[float] = Field(None, gt=0.0)
    #: Timoshenko shear correction factor on az
    kappa_z: Optional[float] = Field(None, gt=0.0)
    # --- fiber (beam2_fiber): the existing fiber_section keys, plus bars ---
    #: fiber: number of fibres across the width (rect), or along the walls (box)
    ny: Optional[int] = Field(None, ge=1)
    #: fiber: number of fibres through the depth
    nz: Optional[int] = Field(None, ge=1)
    #: hollow fiber sections: fibres round the ring (`tube`) and through the
    #: wall thickness (both). `ny`/`nz` count along the walls of a `box`.
    nt: Optional[int] = Field(None, ge=3)
    #: fibres through the wall thickness of a tube or box
    nr: Optional[int] = Field(None, ge=1)
    #: fiber rect: cover thickness; the fibres inside it carry `material`, the core `core_material`
    cover: Optional[float] = Field(None, ge=0.0)
    #: fiber rect: material of the stirrup-confined core inside `cover` (a name in `materials`)
    core_material: Optional[str] = None
    #: fiber: stations along the member -- gauss2 (default) or lobatto3/4/5 (end points sampled; a hinge forms at the end)
    integration: Optional[str] = None
    #: `shape: patches`: a fibre section built of
    #: quadrilateral patches [{corners: [[y, z] x 4], ny, nz}], each
    #: subdivided bilinearly -- a box girder with inclined webs, haunches
    #: and tapered cantilevers. Coordinates in the section axes, as `bars`.
    patches: List[Dict[str, Any]] = []
    #: Effective shear stiffness kappa G A_s [N] of the fibre section.
    #: Absent, the member is Euler-Bernoulli; given, the
    #: bending interpolation becomes Timoshenko's (Reddy's interdependent
    #: element, phi = 12 E I / (GA_s L^2) per member) and a constant shear
    #: strain gamma = V / GA_s is added. The fibres do not see shear: this is
    #: the elastic (or cracked-effective, e.g. ASCE 41's 0.4 E_c A_w) member
    #: shear flexibility, stated by the analyst.
    shear_stiffness: Optional[float] = Field(None, gt=0.0)
    #: a cracking shear law for a fibre section, in series with the bending; exclusive with `shear_stiffness`; in the numba kernel since 1.7
    shear: Optional[FiberShearSpec] = None
    #: [{y, z, section: <bar section name>, count?}] -- OpenSees `layer` per
    #: bar; `count` lumps n identical bars at the point
    bars: List[Dict[str, Any]] = []
    #: bars laid in straight layers (1.2): [{section, from: [y, z], to: [y, z],
    #: count | spacing, z_from}] -- both ends included; `spacing` puts as many
    #: bars as fit at that spacing or closer (d16 @ 150 over 1 m is 8 bars);
    #: `z_from: top | bottom` measures z from that face, downward from the top
    bar_layers: List[Dict[str, Any]] = []
    #: PRESTRESSING TENDONS of a fibre section: each is
    #:   {section: <bar section>, profile: [[s, y, z], ...],
    #:    interpolation: linear | quadratic, normalised: false,
    #:    prestress: <Pa> | [[s, <Pa>], ...]  |  jack: <Pa>, friction: {mu, k, jack, anchor_set},
    #:    elastic_shortening: none | compensate}
    #: `profile` is the tendon's line: `s` the distance along the member from
    #: its start (the end with the smaller global x, then y, then z; a
    #: fraction of the member length when `normalised`), `y`/`z` the offsets
    #: in the section axes, as for `bars`; `quadratic` is one parabola through
    #: each three consecutive points (0-1-2, 2-3-4, ...: an odd count), and
    #: the curve as placed must stay inside the section. The tendon becomes a bonded steel
    #: fibre whose position is the profile's value at each element's
    #: midpoint -- a draped tendon is a polyline of one eccentricity per
    #: element -- carrying an initial strain: `prestress` is the stress the
    #: tendon holds with the section unstrained, or `jack` is the stress at
    #: the jack and `friction` is the post-tensioning loss profile the
    #: embedded tendons already use. What the relaxation then finds is the
    #: elastic shortening; `elastic_shortening: compensate` adds the
    #: linear-elastic estimate of it back (the gross section's 3x3 stiffness
    #: about the origin, so a T or a box need not be written about its
    #: centroid), so the stated stress is what the tendon carries against the
    #: hardened, already-shortened concrete.
    tendons: List[Dict[str, Any]] = []
    #: `fracture: {threshold: 1.0}` -- the member is severed when that fraction
    #: of its fibres has failed. On the SECTION and not on the assignment
    #: because it is the section that says how many fibres there are and which
    #: laws they carry; what makes a fibre fail is in the material block.
    fracture: Optional[Dict[str, Any]] = None
    # --- connector ---
    #: connector: six diagonal values [ux, uy, uz, rx, ry, rz] (N/m, N m/rad) or a symmetric 6x6 matrix (docs/connector2.md)
    stiffness: Optional[Any] = None
    #: connector: viscous damping in the same form as `stiffness` (N s/m, N m s/rad)
    connector_damping: Optional[Any] = None
    #: connector: per component a piecewise-linear force-displacement skeleton [[u, F], ...] replacing that component's stiffness (reversible, non-decreasing)
    curves: Dict[str, List[List[float]]] = {}
    #: connector: a path-dependent law per component -- gap, bilinear hysteresis, Coulomb friction (schema.ConnectorLawSpec)
    laws: Dict[str, ConnectorLawSpec] = {}

    @model_validator(mode="before")
    @classmethod
    def _canonical_names(cls, data):
        """One name per quantity. A beam's rectangle was
        `b`/`h` and a fibre section's `width`/`height`; a beam's circle `d`
        and a fibre tube's `diameter`; a beam's torsion constant `j` and a
        fibre section's `torsion_constant` -- the same quantity under two
        names depending on which element carried it, and the one the other
        element did not know was refused (or, for `j` on a fibre section,
        silently ignored). The old spellings are still read, with a
        preflight warning; `vfife migrate` rewrites them."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for old, new in DEPRECATED_SECTION_KEYS.items():
            if old in data:
                v = data.pop(old)
                if new in data and data[new] is not None and v is not None and data[new] != v:
                    raise ValueError(f"`{old}` (the old spelling) and `{new}` are both given, "
                                     f"and differ ({v!r} against {data[new]!r}); keep `{new}`")
                data.setdefault(new, v)
        return data

    @model_validator(mode="after")
    def _by_type(self):
        t = self.type
        self._girder_shapes()
        if t in ("solid", "shell", "beam", "bar") and self.material is None:
            raise ValueError(f"a {t} section needs a material")
        if t == "shell" and self.thickness is None:
            raise ValueError("a shell section needs thickness")
        if t in ("beam", "bar"):
            if self.shape is None and self.area is None:
                raise ValueError(f"a {t} section needs a shape (circle, rect, I, general) or area")
            if self.shape is None:
                self.shape = "general"
        if t == "bar" and self.shape not in (None, "circle", "general"):
            raise ValueError("a bar section is circle (diameter) or general (area, perimeter)")
        if self.shear is not None and (t != "fiber" or self.shear_stiffness is not None):
            raise ValueError("`shear` is a fibre section's cracking shear law, and it replaces "
                             "`shear_stiffness` (give one of the two)")
        if t == "fiber":
            if self.shape not in (None, "rect", "tube", "box", "patches"):
                raise ValueError(
                    "a fiber section is shape: rect (solid), tube (circular "
                    "hollow), box (rectangular hollow), patches (quadrilaterals), "
                    "box_girder or t_girder")
            if self.shape == "patches":
                if not self.patches:
                    raise ValueError("a patches fiber section needs `patches: [{corners, ny, nz}, ...]`")
                for k, pp in enumerate(self.patches):
                    c = pp.get("corners") if isinstance(pp, dict) else None
                    if not isinstance(c, list) or len(c) != 4 or any(
                            not isinstance(q, (list, tuple)) or len(q) != 2 for q in c):
                        raise ValueError(f"patch {k}: `corners` is four [y, z] points in order round the patch")
                    if pp.get("material") is not None and not isinstance(pp["material"], str):
                        raise ValueError(
                            f"patch {k}: `material` names one of the model's "
                            "materials, e.g. `material: confined`")
                    for h in (pp.get("holes") or []):
                        if not isinstance(h, (list, tuple)) or len(h) != 3 or float(h[2]) <= 0.0:
                            raise ValueError(f"patch {k}: each hole is [y, z, radius] with a positive radius")
            elif self.shape == "tube":
                if self.diameter is None or self.t is None:
                    raise ValueError(
                        "a tube fiber section needs diameter (outer) and t (wall)")
            elif self.shape == "box":
                if self.width is None or self.height is None or self.t is None:
                    raise ValueError(
                        "a box fiber section needs width, height (both outer) "
                        "and t (wall)")
            elif self.width is None or self.height is None:
                raise ValueError("a fiber section needs width and height")
            if self.material is None:
                raise ValueError("a fiber section needs a material (the cover concrete)")
            # WHAT EACH SHAPE READS: `nt` on a rect, `t` on a
            # rect (a beam's box spelling), `width` on a tube, `cover` on a
            # tube, a box or without `core_material` were accepted and did
            # nothing. A patch's and a bar's own keys likewise.
            shape = self.shape or "rect"
            reads = {"rect": {"width", "height", "ny", "nz", "cover", "core_material"},
                     "tube": {"diameter", "t", "nt", "nr"},
                     "box": {"width", "height", "t", "ny", "nz", "nr"},
                     "patches": {"patches"}}[shape]
            geometric = {"width", "height", "diameter", "t", "ny", "nz", "nt", "nr", "cover",
                         "core_material", "patches"}
            if shape == "patches" and any(isinstance(p, dict) and p.get("material") for p in self.patches):
                # a patch's own material against `cover` + `core_material` is
                # refused below in its own words ("two ways to say the same thing")
                reads = reads | {"cover", "core_material"}
            unread = sorted(k for k in self.model_fields_set & geometric
                            if k not in reads and getattr(self, k) not in (None, [], {}))
            if unread:
                raise ValueError(f"a {shape} fiber section does not read "
                                 f"{', '.join('`' + k + '`' for k in unread)} "
                                 f"(it reads {', '.join(sorted(reads))})")
            if self.cover is not None and self.core_material is None:
                raise ValueError("`cover` is where `core_material` begins; without a core_material "
                                 "every fibre is `material` and the cover does nothing")
            for k, pp in enumerate(self.patches):
                bad = sorted(set(pp) - {"corners", "ny", "nz", "material", "holes"}) if isinstance(pp, dict) else []
                if bad:
                    raise ValueError(f"patch {k} does not read {', '.join('`' + b + '`' for b in bad)} "
                                     "(a patch is {corners, ny, nz, material?, holes?}; `ny` divides "
                                     "corner 0 -> 1 and `nz` corner 1 -> 2)")
            self._lay_bars()
            for k, bar in enumerate(self.bars):
                bad = sorted(set(bar) - {"y", "z", "section", "count"}) if isinstance(bar, dict) else []
                if bad:
                    raise ValueError(f"bars[{k}] does not read {', '.join('`' + b + '`' for b in bad)}: "
                                     "a bar is {y, z, section, count?}, and its area and law come "
                                     "from the bar section it names")
            if self.bars and self.shape in ("tube", "box"):
                # the message spoke only of steel tubes and concrete-filled
                # ones; a concrete box girder is the common case and it has a
                # way to be written
                raise ValueError(
                    f"bars in a hollow fiber section (`shape: {self.shape}`) are not supported: "
                    "`tube` and `box` are for steel hollow members (a concrete-filled one would "
                    "need a confinement model this library does not have). A REINFORCED "
                    "concrete box girder is written as `shape: patches` -- one patch per slab "
                    "and web -- which takes `bars` and `tendons`")
        if self.bar_layers and t != "fiber":
            raise ValueError(f"`bar_layers` are bars in a fibre section, not in a {t} section")
        if self.fracture is not None and t != "fiber":
            raise ValueError(f"section fracture applies to a fiber section, not {t}")
        if t == "connector":
            self.stiffness = coerce_numbers(self.stiffness)
            self.connector_damping = coerce_numbers(self.connector_damping)
            self.curves = coerce_numbers(self.curves)
        elif self.laws:
            raise ValueError(f"connector laws apply to a connector section, not {t}")
        return self

    def _girder_shapes(self):
        """A box or T girder from its drawing's dimensions (1.2,
        section_shapes.py): a beam keeps the shape and integrates it in
        `geometry`; a fibre section becomes the `patches` it is made of, here,
        so that everything downstream -- the bars' outline check, the solver,
        the sweep -- reads the one form it already knows."""
        from .. import section_shapes as ss
        shape = self.shape
        girder_keys = set().union(*ss.GIRDER_KEYS.values()) - {"height"}
        stated = {k for k in self.model_fields_set
                  if getattr(self, k) not in (None, [], {})}
        if shape not in ss.GIRDER_SHAPES:
            extra = sorted(stated & (girder_keys | ss.FIBRE_KEYS))
            if extra:
                raise ValueError(f"{', '.join('`' + k + '`' for k in extra)} "
                                 f"{'is' if len(extra) == 1 else 'are'} for shape: box_girder "
                                 "or t_girder" + (f", not {shape}" if shape else ""))
            return
        if self.type not in ("beam", "fiber"):
            raise ValueError(f"shape {shape} is a beam or fiber section, not a {self.type} one")
        reads = ss.GIRDER_KEYS[shape] | (ss.FIBRE_KEYS if self.type == "fiber" else set())
        extra = sorted(stated & ((girder_keys | ss.FIBRE_KEYS | {"width", "t", "diameter", "bf", "tf",
                                                                   "hw", "tw", "ny", "nz", "nt", "nr",
                                                                   "cover", "core_material", "patches"})
                                 - reads))
        if extra:
            why = (" (a beam's properties are about its centroid, and the fibres are a fiber "
                   "section's)" if self.type == "beam" and set(extra) & ss.FIBRE_KEYS else "")
            raise ValueError(f"a {shape} {self.type} section does not read "
                             f"{', '.join('`' + k + '`' for k in extra)}{why} "
                             f"(it reads {', '.join(sorted(reads))})")
        spec = self._girder_spec()
        try:
            g = ss.patches(spec) if self.type == "fiber" else ss.girder(spec)
        except (ValueError, ZeroDivisionError) as e:
            raise ValueError(str(e)) from None
        if self.type == "fiber":
            self._girder = {"shape": shape, "top": g["top"], "bottom": g["bottom"],
                            "section_size": g["section_size"], "walls": g["walls"],
                            "voids": g["voids"]}
            if self.j is None:
                self.j = g["j"]
            self.patches = g["patches"]
            self.shape = "patches"
            for k in girder_keys | ss.FIBRE_KEYS | {"height"}:
                setattr(self, k, None)

    def _girder_spec(self):
        from .. import section_shapes as ss
        keys = set().union(*ss.GIRDER_KEYS.values()) | ss.FIBRE_KEYS
        spec = {k: getattr(self, k) for k in keys if getattr(self, k) is not None}
        spec["shape"] = self.shape
        return spec

    def _lay_bars(self):
        """`bar_layers` -> `bars` (1.2): one straight row each."""
        from .. import section_shapes as ss
        if not self.bar_layers:
            return
        if self.shape in ("tube", "box"):
            raise ValueError(f"bar_layers in a hollow fiber section (`shape: {self.shape}`): "
                             "`tube` and `box` take no bars; a reinforced box girder is "
                             "`shape: box_girder` or `patches`")
        top = bottom = None
        if self._girder is not None:
            top, bottom = self._girder["top"], self._girder["bottom"]
        elif (self.shape or "rect") == "rect" and self.height is not None:
            top, bottom = self.height / 2, -self.height / 2
        elif self.shape == "patches" and self.patches:
            zs = [float(c[1]) for p in self.patches for c in p.get("corners") or []]
            top, bottom = max(zs), min(zs)
        laid = []
        for k, layer in enumerate(self.bar_layers):
            if not isinstance(layer, dict):
                raise ValueError(f"bar_layers[{k}] is {{section, from, to, count | spacing}}")
            laid += ss.layer_bars(coerce_numbers(layer), top, bottom, f"bar_layers[{k}]")
        self.bars = list(self.bars) + laid
        self.bar_layers = []

    def geometry(self) -> Dict[str, float]:
        """Integrated properties for beam and bar sections."""
        from .. import section_shapes as ss
        if self.shape in ss.GIRDER_SHAPES:
            g = ss.properties(self._girder_spec())
            for k in ("area", "iy", "iz", "j", "ay", "az"):
                if getattr(self, k) is not None:
                    g[k] = float(getattr(self, k))
            return g
        spec = {k: getattr(self, k) for k in
                ("shape", "t", "bf", "tf", "hw", "tw",
                 "area", "iy", "iz", "j", "ay", "az", "perimeter")}
        # sections.py keeps its formula letters
        spec.update(d=self.diameter, b=self.width, h=self.height)
        return sec_geom.properties({k: v for k, v in spec.items() if v is not None})


# --------------------------------------------------------------------------- #
#  reinforcement geometry
# --------------------------------------------------------------------------- #

class LinePrim(Strict):
    #: first end of the bar, [x, y, z] in mesh coordinates
    start: List[float] = Field(min_length=3, max_length=3)
    #: second end
    end: List[float] = Field(min_length=3, max_length=3)
    #: number of segments along the bar; prefer `segment_length` in a member that cracks (see schema.BarLineSpec)
    divisions: Optional[int] = Field(None, ge=1)
    #: segment length in metres (a physical smearing length for a bonded bar), rounded to whole segments
    segment_length: Optional[float] = Field(None, gt=0.0)
    #: `{count, offset: [dx, dy, dz]}`: `count` copies of the line, each shifted by `offset` from the last
    repeat: Optional[Dict[str, Any]] = None


class LayerPrim(Strict):
    """OpenSees `layer straight`: `count` parallel bars from the first at
    `from`..`to`, each the next one `spacing` away."""
    #: the first bar, [x, y, z] of its start
    from_: List[float] = Field(alias="from", min_length=3, max_length=3)
    #: the first bar's end
    to: List[float] = Field(min_length=3, max_length=3)
    #: how many parallel bars
    count: int = Field(ge=1)
    #: offset from one bar to the next, [dx, dy, dz]
    spacing: List[float] = Field(default=[0.0, 0.0, 0.0], min_length=3, max_length=3)
    #: segments per bar (or `segment_length`)
    divisions: Optional[int] = Field(None, ge=1)
    #: segment length in metres
    segment_length: Optional[float] = Field(None, gt=0.0)
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class HoopPrim(Strict):
    """A closed rectangular tie in the plane normal to `axis`, repeated
    along it from `from` to `to` at `spacing`. `corners` are the two
    opposite corners in the plane's own two coordinates (the other two
    axes in x, y, z order)."""
    #: two opposite corners of the tie in the plane normal to `axis`, in that plane's own two coordinates
    corners: List[List[float]] = Field(min_length=2, max_length=2)
    #: the axis the ties are stacked along
    axis: Literal["x", "y", "z"]
    #: coordinate along `axis` of the first tie
    from_: float = Field(alias="from")
    #: coordinate of the last tie
    to: float
    #: tie spacing along `axis`
    spacing: float = Field(gt=0.0)
    #: segments per side of each tie (or `segment_length`)
    divisions: Optional[int] = Field(None, ge=1)
    #: segment length in metres
    segment_length: Optional[float] = Field(None, gt=0.0)
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class PathPrim(Strict):
    """A bar route through control points, rather than a single straight run.

    Two shapes, and the choice between them is not a smoothing preference:

    `polyline` (the default) lays a straight leg between each pair of points.
    It is exact -- the bar goes where the points say and nowhere else -- and it
    is what a bent-up bar, an L-bar, a hook and a HARPED pretensioned tendon
    actually are. Each leg is subdivided by `segment_length` (or `divisions`)
    the way an ordinary `line` is.

    `parabola` takes exactly three points and draws the unique quadratic
    through them, with the middle point at the PARAMETER midpoint -- so for the
    usual symmetric tendon (anchorage, low point at midspan, anchorage) the
    middle point is the low point, and for an asymmetric profile it is not what
    you meant and `polyline` is the honest way to say it. The curve is sampled
    at EQUAL ARC LENGTH, so `segment_length` means the same physical thing here
    as it does on a straight bar (see BarLineSpec: it is a smearing length, not
    a discretisation choice), and each chord becomes one bar segment.

    A path can `repeat`, exactly as a line can: several parallel tendons or a
    row of identical bent-up bars is one entry plus an offset.

    WHAT A CURVED TENDON DOES NOT BUY YOU. The geometry is right and the
    prestress acts along it -- so the vertical component of a draped tendon
    balances load, which is the whole point of draping one. Friction is still
    not computed: `prestress` remains the effective stress after losses. What
    preflight can now give you is the number a friction calculation needs, the
    tendon's total angular change, because a path has one.
    """
    #: at least two points, each [x, y, z]; three exactly, for `parabola`
    points: Optional[List[List[float]]] = None
    #: `polyline` (straight legs between the points) or `parabola` (three points)
    shape: Literal["polyline", "parabola"] = "polyline"
    #: a Gmsh curve as a bar route -- reserved, and refused with a message
    group: Optional[str] = None
    #: `polyline` subdivides each leg; `parabola` divides the whole curve
    divisions: Optional[int] = Field(None, ge=1)
    #: segment length in metres along each leg (polyline) or the whole curve (parabola)
    segment_length: Optional[float] = Field(None, gt=0.0)
    #: `{count, offset}` as for a `line`
    repeat: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _geometry(self):
        if (self.points is None) == (self.group is None):
            raise ValueError("a reinforcement `path` takes `points` or `group`, not both")
        if self.points is not None:
            if len(self.points) < 2:
                raise ValueError("a reinforcement `path` needs at least two points")
            if any(len(p) != 3 for p in self.points):
                raise ValueError("every point of a reinforcement `path` is [x, y, z]")
            if self.shape == "parabola" and len(self.points) != 3:
                raise ValueError(
                    f"a parabolic path takes exactly three points (start, the point "
                    f"the curve passes through at mid-parameter, end); got "
                    f"{len(self.points)}. A profile with more control points than "
                    "that is a polyline, or several paths assigned together.")
        return self


def _parabola_through(points, n: int):
    """`n + 1` points at equal arc length along the quadratic through three.

    The interpolating parabola, not a control polygon: p(0), p(1/2), p(1) ARE
    the three points given. Sampling the parameter uniformly would put longer
    chords where the curve is steepest -- a few percent on a shallow tendon,
    much more on a sharply draped one -- and `segment_length` is a physical
    length, so it is sampled by arc length instead.
    """
    P0, P1, P2 = (np.asarray(p, dtype=float) for p in points)

    def at(t):
        t = np.asarray(t, dtype=float)[:, None]
        return ((2.0 * (t - 0.5) * (t - 1.0)) * P0
                - (4.0 * t * (t - 1.0)) * P1
                + (2.0 * t * (t - 0.5)) * P2)

    fine = np.linspace(0.0, 1.0, 2001)
    xyz = at(fine)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
    if s[-1] <= 0.0:
        raise ValueError("a parabolic path has zero length")
    t_nodes = np.interp(np.linspace(0.0, s[-1], n + 1), s, fine)
    return at(t_nodes), float(s[-1])


class ReinforcementGeometry(Strict):
    """One named bar set: exactly one primitive, or a plain list of lines."""
    #: one straight bar (LinePrim)
    line: Optional[LinePrim] = None
    #: several straight bars in one set
    lines: Optional[List[LinePrim]] = None
    #: `count` parallel bars (LayerPrim)
    layer: Optional[LayerPrim] = None
    #: closed rectangular ties along an axis (HoopPrim)
    hoop: Optional[HoopPrim] = None
    #: a bent or curved bar through control points (PathPrim)
    path: Optional[PathPrim] = None

    @model_validator(mode="after")
    def _one(self):
        given = [k for k in ("line", "lines", "layer", "hoop", "path") if getattr(self, k) is not None]
        if len(given) != 1:
            raise ValueError("a reinforcement set is exactly one of line, lines, layer, hoop, path"
                             + ("; got " + ", ".join(given) if given else ""))
        return self

    def expand(self, name: str, segment_length: float | None = None,
               divisions: int | None = None) -> List[Dict[str, Any]]:
        """The primitive as a list of straight-line dicts the solver lays.

        `segment_length` / `divisions` are the ASSIGNMENT's fallbacks. Every
        other primitive takes them after expansion, in `lower`, because a
        straight leg is subdivided by the solver. A curve cannot: sampling it
        IS the discretisation, so the length has to be known here.
        """
        if self.line is not None:
            return [self.line.model_dump(exclude_none=True)]
        if self.lines is not None:
            return [ln.model_dump(exclude_none=True) for ln in self.lines]
        if self.layer is not None:
            L = self.layer
            out = []
            for k in range(L.count):
                off = [k * s for s in L.spacing]
                d = {"start": [a + o for a, o in zip(L.from_, off)],
                     "end": [b + o for b, o in zip(L.to, off)]}
                if L.divisions is not None:
                    d["divisions"] = L.divisions
                if L.segment_length is not None:
                    d["segment_length"] = L.segment_length
                out.append(d)
            return out
        if self.hoop is not None:
            H = self.hoop
            ax = {"x": 0, "y": 1, "z": 2}[H.axis]
            others = [i for i in range(3) if i != ax]
            (p0, q0), (p1, q1) = H.corners
            corners2d = [(p0, q0), (p1, q0), (p1, q1), (p0, q1)]
            n = int(math.floor((H.to - H.from_) / H.spacing + 1e-9)) + 1
            if n < 1:
                raise ValueError(f"hoop {name!r}: no ties between from and to")

            def pt(s, pq):
                v = [0.0, 0.0, 0.0]
                v[ax] = s
                v[others[0]], v[others[1]] = pq
                return v

            # ONE CLOSED ROUTE, not four legs. Four entries whose ends touch
            # are four bars with their own nodes: a bonded tie was four
            # independent stubs, each with two free ends, and `anchorage: ends`
            # tied all eight of them.
            d = {"points": [pt(H.from_, c) for c in corners2d], "closed": True,
                 "repeat": {"count": n,
                            "offset": [H.spacing if i == ax else 0.0 for i in range(3)]}}
            if H.divisions is not None:
                d["divisions"] = H.divisions
            if H.segment_length is not None:
                d["segment_length"] = H.segment_length
            return [d]
        P = self.path
        if P.group is not None:
            raise ValueError(
                f"reinforcement {name!r}: `path: {{group: ...}}` (a curve read from the "
                "Gmsh file as a bar route) is not supported yet; give the route as "
                "`points` here, or draw it with line, layer or hoop")
        seg = P.segment_length if P.segment_length is not None else segment_length
        div = P.divisions if P.divisions is not None else divisions
        common: Dict[str, Any] = {}
        if P.repeat is not None:
            common["repeat"] = P.repeat

        if P.shape == "polyline":
            # ONE route, laid with its legs sharing nodes: the bar runs through
            # the points and is a single bar, not a pile of touching stubs
            for a, b in zip(P.points, P.points[1:]):
                if np.allclose(a, b):
                    raise ValueError(
                        f"reinforcement {name!r}: a path repeats the point {list(a)}, "
                        "which would be a zero-length leg")
            d = {"points": [list(map(float, p)) for p in P.points], **common}
            if seg is not None:
                d["segment_length"] = seg
            elif div is not None:
                d["divisions"] = div
            return [d]

        # parabola: the sampling IS the discretisation, so it needs a length
        if seg is None and div is None:
            raise ValueError(
                f"reinforcement {name!r}: a parabolic path needs `segment_length` "
                "(or `divisions`), on the path or on its assignment -- sampling the "
                "curve is what turns it into bars, so nothing can supply it later")
        if seg is not None and div is not None and P.segment_length is not None \
                and P.divisions is not None:
            raise ValueError(f"reinforcement {name!r}: a path takes `segment_length` "
                             "or `divisions`, not both")
        if seg is not None:
            _probe, length = _parabola_through(P.points, 1)
            n = max(1, int(round(length / float(seg))))
        else:
            n = int(div)
        nodes, _length = _parabola_through(P.points, n)
        # ONE route through the sampled points, each chord one segment: the
        # node spacing already IS segment_length
        return [{"points": [list(map(float, q)) for q in nodes],
                 "divisions": 1, **common}]


# --------------------------------------------------------------------------- #
#  assign
# --------------------------------------------------------------------------- #

class AssignSpec(Strict):
    #: a name in `sections`
    section: str
    #: exactly one target: a mesh region, or one or more reinforcement sets
    region: Optional[str] = None
    #: one or more names in `reinforcement`: the bars this (bar) section describes
    reinforcement: Optional[Union[str, List[str]]] = None
    # beams / connectors on line regions
    #: beam/connector: a vector fixing the section's local y axis (not parallel to the member)
    orientation: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    #: beam end releases, from i_rx, i_ry, i_rz, j_rx, j_ry, j_rz (i = first node); a pin at both ends is [i_ry, i_rz, j_ry, j_rz]
    releases: List[str] = []
    #: rigid end offsets (1.21): {i: [x, y, z], j: [x, y, z], axes: local|global, per: line|member, axial: elastic|full}, the vectors from the nodes to the ends of the member's elastic part (`local`: the member's axes from its nodes' chord and `orientation`). A rigid end zone is an offset along the member, a section whose centroid is below its nodes one across it. `axial: full` takes a frame beam's axial and torsional stiffness over the length between the nodes; `elastic`, the default, over the elastic part
    end_offsets: Optional[Dict[str, Any]] = None
    #: connector: `local` (x along the chord, y from `orientation`) or `global` components
    coordinate_system: Optional[Literal["local", "global"]] = None
    # embedded bars
    #: embedded bars: `{type: bond_ceb | bond_cyclic, ...}` gives the bar its own dofs and a bond-slip law; absent = perfect bond
    bond: Optional[Dict[str, Any]] = None
    #: embedded bars: segment length (m) for every line of the set, when the geometry gives none
    segment_length: Optional[float] = Field(None, gt=0.0)
    #: embedded bars: segments per line, when the geometry gives none
    divisions: Optional[int] = Field(None, ge=1)
    #: bonded bars: stiffness holding the bar across its axis (N/m); default the bar's own E A / L_trib. A numerical device, not a dowel
    transverse_penalty: Optional[float] = Field(None, gt=0.0)
    #: subtract the host's E*eps from the bar (elastic hosts only; the solver
    #: refuses it with a damaging host -- see elements/truss.py)
    host_deduction: Optional[float] = Field(None, gt=0.0)
    #: fiber beams: strain penetration, {region, bar_diameter}; bonded bars:
    #: `ends` / `start` / `end`, the bar nodes tied to the host
    anchorage: Optional[Union[str, Dict[str, Any]]] = None
    #: pretensioning: the EFFECTIVE prestress in the bar after losses, Pa,
    #: positive in tension. The program computes the elastic-shortening loss
    #: (it is what equilibrium at transfer gives) and nothing else -- write
    #: sigma_pe here, not sigma_con.
    prestress: Optional[float] = None
    #: post-tensioning friction {mu, k, jack, anchor_set}. With it, `prestress`
    #: is the JACKING stress and the solver computes the profile along the
    #: tendon; without it, `prestress` is what the whole tendon carries.
    friction: Optional[Dict[str, Any]] = None
    #: a `bar` section on a line region as a CABLE: its total
    #: unstressed length (an alternative to `prestress`), and whether it
    #: carries no compression
    unstressed_length: Optional[float] = Field(None, gt=0.0)
    #: one per element of the region, in element order (form-finding output)
    unstressed_lengths: Optional[List[float]] = None
    #: the force (N) a cable is JACKED to: at the end of the analysis step that builds it (the first, when built from the start) each member carries exactly this, whatever the structure gave under it; from then on it is a bar of the unstressed length it ended with. Frame models (beams, bars, connectors) only
    jack: Optional[float] = Field(None, gt=0.0)
    #: cable: the member carries no compression (goes slack)
    tension_only: bool = False
    #: elastic-brittle breaking of a line member: see
    #: schema.FailureSpec
    failure: Optional[Dict[str, Any]] = None
    #: the reinforcement set's name for monitors (`of:`) and result fields; defaults to the section name
    name: Optional[str] = None

    @model_validator(mode="after")
    def _one_target(self):
        if (self.region is None) == (self.reinforcement is None):
            raise ValueError("an assignment targets exactly one of `region` or `reinforcement`")
        if isinstance(self.reinforcement, str):
            self.reinforcement = [self.reinforcement]
        return self


# --------------------------------------------------------------------------- #
#  the input case, and its lowering
# --------------------------------------------------------------------------- #

INPUT_KEYS = ("materials", "sections", "assign")


def is_input_form(d: Dict[str, Any]) -> bool:
    return any(k in d for k in INPUT_KEYS)


_LINE_SECTION_TYPES = {"bar": "truss2", "beam": "beam2", "connector": "connector2"}


#: WHAT EACH SECTION TYPE READS. Anything else written on a
#: section of that type was accepted and dropped by the lowering -- a
#: timoshenko `theory` on a fibre section, `tendons` on a beam -- so the
#: analyst believed a setting was active that was not. Now it is refused.
SECTION_KEYS = {
    "solid": {"formulation", "hourglass_coeff"},
    "shell": {"thickness", "shear_correction", "drilling_factor", "warpage_tolerance",
              "formulation", "hourglass_coeff"},
    "beam": {"shape", "diameter", "t", "width", "height", "bf", "tf", "hw", "tw", "area",
             "iy", "iz", "j", "ay", "az", "theory", "kappa_y", "kappa_z", "formulation",
             # box_girder / t_girder (1.2)
             "top_width", "bottom_width", "cantilever", "top_thickness", "bottom_thickness",
             "web_thickness", "cantilever_thickness", "top_haunch", "bottom_haunch", "cells"},
    "bar": {"shape", "diameter", "area", "perimeter", "formulation"},
    "fiber": {"shape", "width", "height", "diameter", "t", "ny", "nz", "nt", "nr", "cover",
              "core_material", "integration", "patches", "shear_stiffness", "shear", "j",
              "bars", "tendons", "fracture", "formulation",
              # box_girder / t_girder and bars in layers (1.2)
              "top_width", "bottom_width", "cantilever", "top_thickness", "bottom_thickness",
              "web_thickness", "cantilever_thickness", "top_haunch", "bottom_haunch", "cells",
              "layers", "fiber_size", "origin", "bar_layers"},
    "connector": {"stiffness", "connector_damping", "curves", "laws", "formulation"},
}

#: what an assignment reads, by what it assigns (same reason)
ASSIGN_KEYS = {
    "reinforcement": {"bond", "segment_length", "divisions", "transverse_penalty", "host_deduction",
                      "anchorage", "prestress", "friction"},
    "bar": {"prestress", "unstressed_length", "unstressed_lengths", "jack", "tension_only", "failure",
            "orientation"},
    "beam": {"orientation", "releases", "failure", "end_offsets"},
    "fiber": {"orientation", "anchorage", "end_offsets"},
    "connector": {"orientation", "coordinate_system", "failure"},
    "solid": set(),
    "shell": set(),
}


#: where a refused assignment key's intent is served instead
INSTEAD = {
    ("fiber", "failure"): "a fibre member fails through the section's `fracture` block",
    ("fiber", "releases"): "the fibre element has no end releases; use a beam section, or a connector",
    ("bar", "friction"): "friction losses along a tendon are fibre `tendons` with `jack` + `friction`",
    ("beam", "prestress"): "prestress is on a bar (cable) section, or fibre `tendons`",
}


class InputCase(Strict):
    """The user-facing case. Everything not listed here passes through to
    the internal case unchanged (regions, constraints, loads, steps, solver,
    output, amplitudes, nodal_masses, units, name, mesh)."""
    model_config = ConfigDict(extra="allow")
    materials: Dict[str, Dict[str, Any]] = {}
    sections: Dict[str, SectionSpec]
    reinforcement: Dict[str, ReinforcementGeometry] = {}
    assign: List[AssignSpec] = []
    #: a frame model's tendons, one entry each: line, area, jacking, friction, draw-in, relaxation; expanded on reading into the tendon's nodes, arms and jacked bars (`vfife expand` writes what it becomes)
    tendons: List[TendonSpec] = []
    #: merge these files' materials, sections, reinforcement, groups, tendons and tendon_defaults (paths relative to this file); a name in both is refused, or the case's wins with `{path: ..., override: true}`
    include: List[Union[str, Dict[str, Any]]] = []
    #: groups: name -> [mesh group or group names]; once read, a group of the mesh made of those pieces (their elements and nodes), usable wherever a region is named; lowers to `mesh.groups`
    groups: Dict[str, List[str]] = {}
    #: what every tendon has unless it says otherwise: a key a tendon does not write comes from here, a mapping (`stress`, `friction`, ...) merged key by key with the tendon's own winning; `null` in a tendon is its own value
    tendon_defaults: Dict[str, Any] = {}
    #: the arms of members set off their nodes: one rule for all, each arm's section worked out on reading from what the members set off carry
    offset_arms: Optional[OffsetArmSpec] = None

    @model_validator(mode="after")
    def _what_is_read(self):
        """Names that exist, and keys the section or assignment type reads
        (v1.0: a misspelt section name in `assign` was a KeyError
        traceback, and keys another type reads were dropped silently)."""
        for name, sec in self.sections.items():
            allowed = SECTION_KEYS.get(sec.type, set()) | {"type", "material"}
            extra = sorted(k for k in sec.model_fields_set if k not in allowed)
            if extra:
                raise ValueError(f"section {name!r} is a {sec.type} section, which does not read "
                                 f"{', '.join('`' + k + '`' for k in extra)}")
            if sec.type == "fiber" and sec.core_material is not None and sec.cover is None:
                raise ValueError(f"section {name!r}: `core_material` is the law inside `cover`; "
                                 "without a cover every fibre is `material` and the core law is unused")
        for i, a in enumerate(self.assign):
            where = f"assign[{i}]"
            if a.section not in self.sections:
                raise ValueError(f"{where} names section {a.section!r}, which is not in `sections` "
                                 f"({', '.join(sorted(self.sections)) or 'empty'})")
            for r in (a.reinforcement or []):
                if r not in self.reinforcement:
                    raise ValueError(f"{where} names reinforcement {r!r}, which is not in `reinforcement` "
                                     f"({', '.join(sorted(self.reinforcement)) or 'empty'})")
            kind = "reinforcement" if a.reinforcement is not None else self.sections[a.section].type
            allowed = ASSIGN_KEYS.get(kind, set()) | {"section", "region", "reinforcement", "name"}
            explicit = {k for k in a.model_fields_set
                        if not (k == "releases" and not a.releases)
                        and not (k == "tension_only" and not a.tension_only)}
            extra = sorted(k for k in explicit if k not in allowed)
            if extra:
                what = ("reinforcement" if kind == "reinforcement" else f"a {kind} section on a region")
                hints = [INSTEAD[(kind, k)] for k in extra if (kind, k) in INSTEAD]
                raise ValueError(f"{where} assigns {what}, which does not read "
                                 f"{', '.join('`' + k + '`' for k in extra)}"
                                 + ("; " + "; ".join(hints) if hints else ""))
            if kind == "fiber":
                sec = self.sections[a.section]
                # THE OUTER OUTLINE, not its bounding rectangle: a bar beside
                # a T's web or in a tube's square corner was accepted
                # (voids and holes count as inside)
                from ..elements.fiber_section import outline_contains, outline_name

                def outside(y, z, cells=False):
                    if sec._girder is not None:
                        # a girder's outline is exact; its cells hold external
                        # tendons but no bars (review, 1.2)
                        from ..section_shapes import contains
                        quads = sec._girder["walls"] + (sec._girder["voids"] if cells else [])
                        return not bool(contains(quads, [y], [z])[0])
                    hit = outline_contains(sec.shape, [y], [z], width=sec.width, height=sec.height,
                                           diameter=sec.diameter, patches=sec.patches)
                    return hit is not None and not bool(hit[0])
                where_ = outline_name(sec.shape)
                if sec._girder is not None:
                    where_ = f"the {sec._girder['shape']}'s walls"
                    where_t = where_ + " and its cells" if sec._girder["voids"] else where_
                else:
                    where_t = where_
                for j, bar in enumerate(sec.bars):
                    if not all(isinstance(bar.get(k), (int, float)) for k in ("y", "z")):
                        raise ValueError(f"section {a.section!r}: bars[{j}] needs numeric `y` and `z`")
                    if outside(bar["y"], bar["z"]):
                        raise ValueError(f"section {a.section!r}: bars[{j}] at y = {bar['y']}, z = {bar['z']} "
                                         f"lies outside the section ({where_})")
                for j, t in enumerate(sec.tendons):
                    for row in (t.get("profile") or []):
                        if (len(row) >= 3 and all(isinstance(v, (int, float)) for v in row[1:3])
                                and outside(row[1], row[2], cells=True)):
                            raise ValueError(f"section {a.section!r}: tendons[{j}] passes y = {row[1]}, "
                                             f"z = {row[2]}, outside the section ({where_t})")
        return self

    @staticmethod
    def _fiber_discretisation(a, s) -> Dict[str, Any]:
        """The fibre grid of one section, as the solver reads it.

        Factored out of the single-section lowering so that a model with
        several fibre sections builds each one through exactly
        the code the single-section case always used.
        """
        shape = s.shape or "rect"
        if shape == "tube":
            d = s.diameter
            if d is None or s.t is None:
                raise ValueError(
                    f"fiber section {a.section!r}: a tube needs "
                    "`diameter` (outer) and `t` (wall thickness)")
            fs: Dict[str, Any] = {"type": "tube", "diameter": d,
                                  "thickness": s.t}
            for k in ("nt", "nr"):
                if getattr(s, k) is not None:
                    fs[k] = getattr(s, k)
        elif shape == "box":
            if s.width is None or s.height is None or s.t is None:
                raise ValueError(
                    f"fiber section {a.section!r}: a box needs `width`, "
                    "`height` (both outer) and `t` (wall thickness)")
            fs = {"type": "box", "width": s.width, "height": s.height,
                  "thickness": s.t}
            for k in ("ny", "nz", "nr"):
                if getattr(s, k) is not None:
                    fs[k] = getattr(s, k)
        elif shape in ("rect", "general"):
            fs = {"type": "rect", "width": s.width, "height": s.height}
            for k in ("ny", "nz", "cover"):
                if getattr(s, k) is not None:
                    fs[k] = getattr(s, k)
        elif shape == "patches":
            if not s.patches:
                raise ValueError(f"fiber section {a.section!r}: shape patches needs `patches`")
            fs = {"type": "patches", "patches": [dict(pp) for pp in s.patches]}
        else:
            raise ValueError(
                f"fiber section {a.section!r}: shape {shape!r} has no "
                "fibre discretisation (rect, tube, box)")
        if s.shear_stiffness is not None:
            fs["shear_stiffness"] = s.shear_stiffness
        if s.shear is not None:
            law = s.shear.model_dump()
            if law["web_width"] is None:
                # the width ACROSS the shear: a z shear crosses the section's
                # width (y extent), a y shear its height
                w = s.width if law["direction"] == "z" else s.height
                if w is None and law["stirrups"] is not None:
                    raise ValueError(f"fiber section {a.section!r}: the truss model needs the web "
                                     "width; give shear.web_width")
                law["web_width"] = w
            fs["shear_law"] = law
        if s.j is not None:
            fs["torsion_constant"] = s.j
        if s._girder is not None:
            # for `info`'s note on a girder lying on its side (1.2)
            fs["section_size"] = list(s._girder["section_size"])
            if s._girder["voids"]:
                # a box's cells: the tendons' outline check counts them inside
                fs["cells"] = [list(map(list, c)) for c in s._girder["voids"]]
        return fs

    def _resolve_patch_materials(self, fs, mat, who):
        """A patch may name its own material: the flange's
        cover and its stirrup-confined core are two patches of one section.
        The name is resolved here, where the material table is in scope, so
        the solver sees a spec dict exactly as it does for `core_material`."""
        for k, pp in enumerate(fs.get("patches") or []):
            nm = pp.get("material")
            if nm is not None:
                pp["material"] = mat(nm, f"{who} patch {k}")
        return fs

    def _fiber_tendons(self, a, s, mat):
        """The tendons of one fibre section, as the solver reads them."""
        out = []
        for k, t in enumerate(s.tendons):
            who = f"fiber section {a.section!r} tendon {k}"
            bs = self.sections.get(t.get("section"))
            if bs is None or bs.type != "bar":
                raise ValueError(f"{who}: needs `section` naming a bar section")
            prof = t.get("profile")
            if not isinstance(prof, list) or len(prof) < 2 or any(
                    not isinstance(p, (list, tuple)) or len(p) != 3 for p in prof):
                raise ValueError(f"{who}: `profile` is a list of at least two [s, y, z] points")
            prof = [[float(v) for v in p] for p in prof]
            if any(p1[0] <= p0[0] for p0, p1 in zip(prof, prof[1:])):
                raise ValueError(f"{who}: profile `s` must increase along the member")
            interp = t.get("interpolation", "linear")
            if interp not in ("linear", "quadratic"):
                raise ValueError(f"{who}: interpolation is linear or quadratic, not {interp!r}")
            if interp == "quadratic" and (len(prof) < 3 or len(prof) % 2 == 0):
                # one parabola per three points, 0-1-2, 2-3-4, ...:
                # an even count leaves a last segment of two points, which no
                # parabola is defined by
                raise ValueError(f"{who}: a quadratic profile is one parabola per three consecutive "
                                 "points (0-1-2, 2-3-4, ...), so it needs at least three points and "
                                 f"an odd number of them; this one has {len(prof)}. Add the point "
                                 "between, or use interpolation: linear")
            has_pre = t.get("prestress") is not None
            has_jack = t.get("jack") is not None
            if has_pre == has_jack:
                raise ValueError(f"{who}: give ONE of `prestress` (the tendon's stress with the "
                                 "section unstrained) or `jack` (the stress at the jack, with "
                                 "`friction`)")
            entry = {"profile": prof, "interpolation": interp,
                     "normalised": bool(t.get("normalised", False)),
                     "area": bs.geometry()["area"],
                     "material": mat(bs.material, f"bar section {t['section']!r}"),
                     "elastic_shortening": t.get("elastic_shortening", "none")}
            if entry["elastic_shortening"] not in ("none", "compensate"):
                raise ValueError(f"{who}: elastic_shortening is none or compensate")
            if t.get("extent") is not None:
                ext = t["extent"]
                if not isinstance(ext, (list, tuple)) or len(ext) != 2:
                    raise ValueError(f"{who}: `extent` is [s_start, s_end], the anchors along the member")
                entry["extent"] = [float(ext[0]), float(ext[1])]
            if has_pre:
                pre = t["prestress"]
                if isinstance(pre, (list, tuple)):
                    # A STRESS PROFILE ALONG THE TENDON:
                    # [[s, sigma], ...] in the profile's s units, linear
                    # between the points and sampled at element midpoints.
                    # For an effective stress known point by point -- a
                    # friction profile computed once for a tendon that the
                    # case splits into pieces, or a stress measured on site.
                    tab = [[float(v) for v in p] for p in pre] if all(
                        isinstance(p, (list, tuple)) and len(p) == 2 for p in pre) else None
                    if not tab or len(tab) < 2:
                        raise ValueError(f"{who}: `prestress` is a stress, or a list of at "
                                         "least two [s, stress] points")
                    if any(b[0] <= a[0] for a, b in zip(tab, tab[1:])):
                        raise ValueError(f"{who}: `prestress` profile `s` must increase")
                    entry["prestress_profile"] = tab
                else:
                    entry["prestress"] = float(pre)
                if t.get("friction"):
                    raise ValueError(f"{who}: `friction` goes with `jack`, not `prestress`")
            else:
                entry["jack"] = float(t["jack"])
                fr = t.get("friction") or {}
                if not isinstance(fr, dict):
                    raise ValueError(f"{who}: friction is {{mu, k, jack, anchor_set}}")
                # a misspelt key dropped the wobble or the draw-in without a
                # word (found in review)
                bad = sorted(set(fr) - {"mu", "k", "jack", "anchor_set"})
                if bad:
                    raise ValueError(f"{who}: friction does not read {', '.join('`' + b + '`' for b in bad)}; "
                                     "it is {mu, k (wobble, 1/m), jack: start|end|both, anchor_set (m)}")
                entry["friction"] = {"mu": float(fr.get("mu", 0.0)), "k": float(fr.get("k", 0.0)),
                                     "jack": fr.get("jack", "start"),
                                     "anchor_set": fr.get("anchor_set")}
            unknown = set(t) - {"section", "profile", "interpolation", "normalised", "prestress",
                                "jack", "friction", "elastic_shortening", "extent"}
            if unknown:
                raise ValueError(f"{who}: unknown key(s) {sorted(unknown)}")
            out.append(entry)
        return out

    def _where(self, a) -> str:
        i = next((i for i, x in enumerate(self.assign) if x is a), None)
        what = (f"reinforcement {'+'.join(a.reinforcement)}" if a.reinforcement is not None
                else f"section {a.section!r} on {a.region!r}")
        return f"assign[{i}] ({what})"

    def lower(self) -> Dict[str, Any]:
        # where each lowered entry came from, so an error found in the
        # internal form names the user's `assign[i]`
        self._provenance = prov = {}
        mats = {k: coerce_numbers(v) for k, v in self.materials.items()}
        for name, m in mats.items():
            if "type" not in m:
                raise ValueError(f"material {name!r} has no type")

        def parts_of(spec, name, within):
            """A composite's parts (1.21), each a name in the table or a spec
            of its own, resolved here where the table is in scope, so the law
            is built from specs."""
            if spec.get("type") not in ("parallel", "series"):
                return spec
            parts = spec.get("of")
            if not isinstance(parts, list) or len(parts) < 2 \
                    or not all(isinstance(x, (str, dict)) for x in parts):
                raise ValueError(f"material {name!r}: `of` lists two or more materials (names in "
                                 f"`materials`, or specs with a `type`), not {parts!r}")
            spec["of"] = [mat(x, f"material {name!r}", within) if isinstance(x, str)
                          else parts_of(coerce_numbers(copy.deepcopy(x)), name, within) for x in parts]
            return spec

        def mat(name, who, _within=()):
            if name not in mats:
                raise ValueError(f"{who} names material {name!r}, which is not in `materials` "
                                 f"({', '.join(sorted(mats)) or 'empty'})")
            if name in _within:
                raise ValueError(f"material {name!r} is a part of itself ({' -> '.join(_within + (name,))})")
            return parts_of(copy.deepcopy(mats[name]), name, _within + (name,))

        # a connector's `material` law names a material too (1.21), resolved
        # where its entry is made
        self._mat = mat

        if self.tendons or self.include or self.groups or self.tendon_defaults or self.offset_arms:
            # expanded before the lowering (vfife/prep): here
            # only when the case came in without its mesh's path
            raise ValueError("`tendons`, `tendon_defaults`, `offset_arms`, `include` and `groups` are expanded when the case is read "
                             "(`schema.from_dict`); this case did not come through it")
        passthrough = {k: v for k, v in self.model_dump(mode="python", by_alias=True).items()
                       if k not in ("materials", "sections", "reinforcement", "assign", "tendons", "include", "groups",
                                    "tendon_defaults", "offset_arms")}
        out: Dict[str, Any] = dict(passthrough)

        assigns = list(self.assign)
        solid_sections = [n for n, s in self.sections.items() if s.type in ("solid", "shell")]
        region_assigns = [a for a in assigns if a.region is not None]
        # a single solid/shell section with no region assignment goes to `all`
        if not any(self.sections[a.section].type in ("solid", "shell") for a in region_assigns):
            if len(solid_sections) == 1:
                assigns.insert(0, AssignSpec(section=solid_sections[0], region="all"))
            elif len(solid_sections) > 1:
                raise ValueError("more than one solid/shell section but none assigned to a region; "
                                 "add `assign: - {region: ..., section: ...}`")
            region_assigns = [a for a in assigns if a.region is not None]

        continuum = [a for a in region_assigns if self.sections[a.section].type in ("solid", "shell")]
        line_assigns = [a for a in region_assigns if self.sections[a.section].type in ("bar", "beam", "connector", "fiber")]
        bar_assigns = [a for a in assigns if a.reinforcement is not None]

        shells = [a for a in continuum if self.sections[a.section].type == "shell"]
        if shells and len(continuum) > 1:
            raise ValueError("a shell section is the whole mesh; it cannot share the mesh with "
                             f"other continuum sections (got {', '.join(a.section for a in continuum)})")

        element: Dict[str, Any] = {}
        material: Optional[Dict[str, Any]] = None
        members: List[Dict[str, Any]] = []
        blocks: List[Dict[str, Any]] = []

        if continuum:
            material = self._lower_continuum(continuum, line_assigns, element, blocks, mat, prov)

        # line sections on mesh regions
        if line_assigns:
            material = self._lower_lines(line_assigns, continuum, element, blocks, members, material, mat, prov)
        if material is None:
            raise ValueError("nothing is assigned to the mesh: add a solid, shell, beam, bar or "
                             "fiber section and an `assign` entry for it")

        rein = self._lower_reinforcement(bar_assigns, mat, prov)
        out["element"] = element
        out["material"] = material
        if blocks:
            out["blocks"] = blocks
        if members:
            out["members"] = members
        if rein:
            out["reinforcement"] = rein
        return out

    # -- the parts of `lower` (v1.18: one function of complexity 135
    # before; the same code, in the same order, split where it branches)

    def _lower_continuum(self, continuum, line_assigns, element, blocks, mat, prov):
        """The solid or shell sections: the element's settings, and per-region
        material blocks. Returns the model's material of record."""
        a = continuum[0]
        s = self.sections[a.section]
        material = mat(s.material, f"section {a.section!r}")
        if s.type == "solid":
            if s.formulation is not None:
                element["formulation"] = s.formulation
            if s.hourglass_coeff is not None:
                element["hourglass_coeff"] = s.hourglass_coeff
            # Several solid sections, or one on part of the mesh, become
            # material BLOCKS: each region's elements (every node inside)
            # carry that section's material, in assignment order, a later
            # assignment taking precedence where two overlap. `material`
            # stays the first block's so every single-material reader
            # (preflight, summaries) still has something to read.
            if len(continuum) > 1 or a.region != "all":
                for b in continuum:
                    sb = self.sections[b.section]
                    if b.orientation is not None:
                        raise ValueError("orientation on a solid assignment is not supported")
                    blk = {"name": b.section, "region": b.region,
                           "material": mat(sb.material, f"section {b.section!r}")}
                    if sb.formulation is not None:
                        blk["formulation"] = sb.formulation
                    if sb.hourglass_coeff is not None:
                        blk["hourglass_coeff"] = sb.hourglass_coeff
                    prov[("blocks", len(blocks))] = self._where(b)
                    blocks.append(blk)
        else:
            element["type"] = "shell4"
            element["thickness"] = s.thickness
            # the shell's only formulation, stated: left out, the internal
            # schema's solid default (b-bar) was refused for a shell written
            # exactly as the manual shows (found in review)
            element["formulation"] = "selective"
            for k in ("shear_correction", "drilling_factor", "warpage_tolerance", "formulation", "hourglass_coeff"):
                if getattr(s, k) is not None:
                    element[k] = getattr(s, k)
            # The shell solver carries ONE thickness, so a shell section is
            # the whole shell mesh. In a shell_line model the shell mesh is
            # the quad4 block and the line members are the rest, so there
            # the section names the region that holds the quads -- which is
            # how a building's walls and its frame are written down.
            if a.region != "all" and not line_assigns:
                raise ValueError("a shell section is assigned to `all`")
        if a.orientation is not None:
            raise ValueError("orientation on a solid/shell assignment is not supported yet")

        return material

    def _lower_lines(self, line_assigns, continuum, element, blocks, members, material, mat, prov):
        """The line sections: fibre sections (one, or several on their regions)
        with the connectors that may sit beside them, or the plain line
        sections (one element block, or members). Returns the material of
        record."""
        fiber = [a for a in line_assigns if self.sections[a.section].type == "fiber"]
        for a in fiber:
            if a.failure is not None:
                raise ValueError(
                    f"assignment of fiber section {a.section!r} with `failure`: a fibre "
                    "member's section knows its own strength -- it breaks through the "
                    "section's `fracture` (the laws' failure), or is cut by `removals`")
        # A CONNECTOR MAY SIT IN A FIBRE MODEL, and it is
        # the one section type that may: a bearing is two nodes and six
        # stiffnesses, it has no section to integrate and no material to
        # drive, so it costs the fibre machinery nothing. Everything else
        # still cannot -- a truss or a plain beam beside a fibre section
        # would need a second member law, which the line solvers do not
        # carry. Without a fibre section a connector goes down the
        # `mixed_line` path it always did.
        conn = ([a for a in line_assigns
                 if self.sections[a.section].type == "connector"] if fiber else [])
        plain = [a for a in line_assigns
                 if self.sections[a.section].type != "fiber" and a not in conn]
        if fiber and (plain or continuum):
            raise ValueError("a fiber section is the whole model in phase 1; do not mix it "
                             "with other sections (a `connector` section may sit in a fibre "
                             "model -- v0.4 -- but nothing else)")
        if fiber and len(fiber) > 1:
            return self._lower_fibres(fiber, conn, element, mat)
        if fiber:
            return self._lower_fibre(fiber[0], conn, element, mat)
        return self._lower_plain(plain, continuum, blocks, element, members, material, mat, prov)

    def _lower_fibres(self, fiber, conn, element, mat):
        # SEVERAL FIBRE SECTIONS ON ONE MODEL. Each
        # `assign` names a region and a fibre section; the solver gives
        # every member the geometry of the section its region names.
        # Phase 2, and the limits are stated rather than discovered:
        # one fibre LAW for the whole model (the line solvers carry one
        # material), no rebars and no cover/core split (those make the
        # fibre groups differ between sections, which the consolidated
        # state arrays do not express yet), and the element-level
        # settings -- integration, fracture, formulation -- taken from
        # the first section and required to agree.
        first = self.sections[fiber[0].section]
        material = mat(first.material, f"section {fiber[0].section!r}")
        entries = []
        for a in fiber:
            s = self.sections[a.section]
            for key in ("integration", "fracture", "formulation"):
                if getattr(s, key) != getattr(first, key):
                    raise ValueError(
                        f"fiber section {a.section!r}: `{key}` differs from "
                        f"{fiber[0].section!r}; it is an element-level setting and "
                        "every fibre section on the model must agree")
            if a.region == "all" and len(fiber) > 1:
                raise ValueError(
                    f"fiber section {a.section!r} is assigned to `all` alongside "
                    "other fibre sections; give each section its own region")
            # Each section carries its own laws: the
            # cover law when it is not the model's, the confined
            # core, one material per bar, and the anchorage that
            # applies to its members. The solver builds one material
            # group per law per section, so a column and a beam can
            # differ in everything but the element-level settings.
            fs = self._fiber_discretisation(a, s)
            self._resolve_patch_materials(fs, mat, f"section {a.section!r}")
            if s.core_material is not None:
                fs["core_material"] = mat(s.core_material, f"section {a.section!r} core")
            entry = {"region": a.region, "fiber_section": fs}
            if s.material != first.material:
                entry["material"] = mat(s.material, f"section {a.section!r}")
            rebars = self._fiber_rebars(a, s, mat)
            if rebars:
                entry["fiber_rebars"] = rebars
            tendons = self._fiber_tendons(a, s, mat)
            if tendons:
                entry["fiber_tendons"] = tendons
            if a.anchorage is not None:
                if not isinstance(a.anchorage, dict):
                    raise ValueError("a fiber section's anchorage is {region, bar_diameter}")
                if not rebars:
                    raise ValueError(f"fiber section {a.section!r} has an anchorage but no bars")
                entry["anchorage"] = a.anchorage
            if a.orientation is not None:
                entry["orientation"] = a.orientation
            if a.end_offsets is not None:
                entry["end_offsets"] = dict(a.end_offsets)
            entries.append(entry)
        element["type"] = "beam2_fiber"
        element["formulation"] = first.formulation or "full"
        element["fiber_sections"] = entries
        if conn:
            element["fiber_connectors"] = [
                {"region": a.region, **self._line_element(a, self.sections[a.section])}
                for a in conn]
        if first.integration is not None:
            element["fiber_integration"] = first.integration
        if first.fracture is not None:
            element["fracture"] = first.fracture
        return material

    def _lower_fibre(self, a, conn, element, mat):
        """One fibre section, on the whole mesh or on one region."""
        s = self.sections[a.section]
        material = mat(s.material, f"section {a.section!r}")
        fs = self._fiber_discretisation(a, s)
        self._resolve_patch_materials(fs, mat, f"section {a.section!r}")
        if s.core_material is not None:
            fs["core_material"] = mat(s.core_material, f"section {a.section!r} core")
        element["type"] = "beam2_fiber"
        element["formulation"] = s.formulation or "full"
        element["fiber_section"] = fs
        if a.region != "all":
            # the region was dropped: a section assigned to half the
            # members, or to a region that did not exist, went on the
            # whole mesh without a word (found in review)
            element["fiber_region"] = a.region
        if s.integration is not None:
            element["fiber_integration"] = s.integration
        if s.fracture is not None:
            element["fracture"] = s.fracture
        rebars = self._fiber_rebars(a, s, mat)
        if rebars:
            element["fiber_rebars"] = rebars
        tendons = self._fiber_tendons(a, s, mat)
        if tendons:
            element["fiber_tendons"] = tendons
        if a.anchorage is not None:
            if not isinstance(a.anchorage, dict):
                raise ValueError("a fiber section's anchorage is {region, bar_diameter}")
            element["strain_penetration"] = a.anchorage
        if a.orientation is not None:
            element["orientation"] = a.orientation
        if a.end_offsets is not None:
            element["end_offsets"] = dict(a.end_offsets)
        if conn:
            element["fiber_connectors"] = [
                {"region": c.region, **self._line_element(c, self.sections[c.section])}
                for c in conn]
        return material

    def _lower_plain(self, plain, continuum, blocks, element, members, material, mat, prov):
        """Plain line sections: one on the whole mesh is the element block;
        several, or beside a continuum, are members (mixed_line, coupled,
        shell_line)."""
        # one line type on the whole mesh -> element block; several, or
        # alongside a continuum -> members (mixed_line / coupled)
        regions_all = all(a.region == "all" for a in plain)
        if len(plain) == 1 and regions_all and not continuum:
            a = plain[0]
            s = self.sections[a.section]
            material = mat(s.material, f"section {a.section!r}") if s.material else material
            if a.unstressed_length is not None or a.unstressed_lengths is not None:
                raise ValueError(
                    f"assignment of {a.section!r} to `all`: `unstressed_length` is one "
                    "cable's total length and needs a named line region; a standalone "
                    "truss model states `prestress` instead")
            element.update(self._line_element(a, s))
            if a.failure is not None:
                # one section on the whole mesh: the capacity of every
                # member, judged by the single-section solver
                element["failure"] = dict(a.failure)
        else:
            if not continuum:
                element["type"] = "mixed_line"
                element["formulation"] = "full"
            else:
                if blocks:
                    raise ValueError("a coupled model (solid + line members) carries one "
                                     "solid material; per-region solid sections are for "
                                     "the continuum solver")
                if self.sections[continuum[0].section].type == "shell":
                    # shell4 quads plus Line2 members on shared nodes:
                    # a shear wall building. The shell's own
                    # `formulation` (selective / full) is already in
                    # `element` and must not be overwritten with the
                    # solid's default.
                    element["type"] = "shell_line"
                else:
                    element["type"] = "coupled"
                    element.setdefault("formulation", "full")
            # EVERY MEMBER GROUP MAY NAME ITS OWN MATERIAL.
            # Until then the line solvers carried one
            # material for the whole model and this loop refused the
            # second one -- which made a cable-stayed or hybrid bridge
            # (steel cables, concrete tower, steel girder) impossible to
            # write down at all. The refusal was at least loud.
            #
            # `material` stays the FIRST line section's (or, in a
            # coupled / shell_line model, the continuum's), so every
            # single-material reader -- preflight, the summaries, the
            # solid and shell halves -- still has one to read, and a
            # model whose sections all name the same material writes no
            # `material` key on any member and is identical to the bit.
            if material is None:
                for a in plain:
                    s = self.sections[a.section]
                    if s.material is not None:
                        material = mat(s.material, f"section {a.section!r}")
                        break
            for a in plain:
                s = self.sections[a.section]
                entry = self._line_element(a, s)
                entry.pop("formulation", None)
                # A CONNECTOR'S `material` IS NOT ITS LAW (its stiffness
                # is): it was forwarded when it differed from the first
                # section's and then refused, so naming the first
                # material passed and the second failed.
                # It is read only as the model's material of record.
                if s.material is not None and s.type != "connector":
                    m = mat(s.material, f"section {a.section!r}")
                    if m != material:
                        entry["material"] = m
                prov[("members", len(members))] = self._where(a)
                members.append({"region": a.region, **entry})

        return material

    def _fiber_rebars(self, a, s, mat):
        """A fibre section's bars: position, area (`count` bars lumped), law."""
        rebars = []
        for bar in s.bars:
            bs = self.sections.get(bar.get("section"))
            if bs is None or bs.type != "bar":
                raise ValueError(f"fiber section {a.section!r}: each bar needs `section` naming a bar section")
            # `count`: n identical bars lumped at one point (the 105 deck bars of
            # a box girder as ten rows)
            count = int(bar.get("count", 1))
            if count < 1:
                raise ValueError(f"fiber section {a.section!r}: a bar's `count` is a positive integer")
            rebars.append({"y": bar["y"], "z": bar["z"], "area": bs.geometry()["area"] * count,
                           "material": mat(bs.material, f"bar section {bar['section']!r}")})
        return rebars

    def _lower_reinforcement(self, bar_assigns, mat, prov):
        """The embedded reinforcement: one entry per bar assignment."""
        rein: List[Dict[str, Any]] = []
        for a in bar_assigns:
            s = self.sections[a.section]
            if s.type != "bar":
                raise ValueError(f"reinforcement is assigned a {s.type} section {a.section!r}; it needs a bar section")
            geom = s.geometry()
            lines: List[Dict[str, Any]] = []
            for rname in a.reinforcement:
                if rname not in self.reinforcement:
                    raise ValueError(f"assignment names reinforcement {rname!r}, which is not defined "
                                     f"({', '.join(sorted(self.reinforcement)) or 'none'})")
                for ln in self.reinforcement[rname].expand(
                        rname, a.segment_length, a.divisions):
                    if ln.get("divisions") is None and ln.get("segment_length") is None:
                        if a.segment_length is not None:
                            ln["segment_length"] = a.segment_length
                        elif a.divisions is not None:
                            ln["divisions"] = a.divisions
                        else:
                            raise ValueError(f"reinforcement {rname!r} has no segment_length or divisions, "
                                             "on the set or on its assignment")
                    lines.append(ln)
            entry: Dict[str, Any] = {
                "name": a.name or "+".join(a.reinforcement),
                "area": geom["area"],
                "material": mat(s.material, f"bar section {a.section!r}"),
                "lines": lines,
            }
            d = s.diameter
            if d is None and geom.get("perimeter"):
                d = geom["perimeter"] / math.pi          # the perimeter-equivalent diameter
            if d is not None:
                entry["diameter"] = float(d)
            if a.host_deduction is not None:
                entry["host_deduction"] = a.host_deduction
            if a.prestress is not None:
                entry["prestress"] = a.prestress
            if a.friction is not None:
                if a.prestress is None:
                    raise ValueError(
                        f"assignment of {entry['name']!r} has a `friction` block but no "
                        "`prestress`: friction is a loss along a tendon, and there is "
                        "no tendon force for it to act on")
                entry["friction"] = a.friction
            if a.bond is not None:
                if d is None:
                    raise ValueError(
                        f"assignment of {entry['name']!r} has a bond block but section "
                        f"{a.section!r} has no surface: bond acts on the bar SURFACE, so give "
                        "the bar section `shape: circle` with a `diameter` (or a `perimeter`)")
                entry["bond"] = a.bond
                if a.transverse_penalty is not None:
                    entry["transverse_penalty"] = a.transverse_penalty
                if a.anchorage is not None:
                    if not isinstance(a.anchorage, str):
                        raise ValueError("a bonded bar's anchorage is ends, start or end")
                    entry["anchorage"] = a.anchorage
            elif a.transverse_penalty is not None:
                raise ValueError("transverse_penalty applies to a bonded assignment only")
            elif a.anchorage is not None:
                raise ValueError("anchorage applies to a bonded assignment only: a perfectly "
                                 "bonded bar's ends are already tied to the host")
            prov[("reinforcement", len(rein))] = self._where(a)
            rein.append(entry)

        return rein

    def _line_element(self, a: AssignSpec, s: SectionSpec) -> Dict[str, Any]:
        entry: Dict[str, Any] = {}
        if s.type == "connector":
            entry["type"] = "connector2"
            entry["stiffness"] = s.stiffness
            if s.connector_damping is not None:
                entry["connector_damping"] = s.connector_damping
            if s.curves:
                entry["curves"] = s.curves
            if s.laws:
                entry["laws"] = {k: v.model_dump() for k, v in s.laws.items()}
                for k, law in entry["laws"].items():
                    if law["type"] == "material":
                        law["material"] = self._mat(law["material"], f"section {a.section!r} law {k}")
        else:
            g = s.geometry()
            if s.type == "bar":
                entry["type"] = "truss2"
                entry["area"] = g["area"]
                if a.prestress is not None:
                    entry["prestress"] = a.prestress
                if a.unstressed_length is not None:
                    entry["unstressed_length"] = a.unstressed_length
                if a.unstressed_lengths is not None:
                    entry["unstressed_lengths"] = list(a.unstressed_lengths)
                if a.jack is not None:
                    entry["jack"] = a.jack
                if a.tension_only:
                    entry["tension_only"] = True
            else:
                timo = s.theory == "timoshenko"
                entry["type"] = "beam2_timoshenko" if timo else "beam2"
                for k in ("area", "iy", "iz", "j"):
                    if g.get(k) is None:
                        raise ValueError(f"beam section {a.section!r} has no {k}; give a shape or the value")
                    entry[k] = g[k]
                if timo:
                    for k in ("ay", "az"):
                        if g.get(k) is None:
                            raise ValueError(f"timoshenko section {a.section!r} needs {k}")
                        entry[k] = g[k]
                    for k in ("kappa_y", "kappa_z"):
                        if getattr(s, k) is not None:
                            entry[k] = getattr(s, k)
                if s.shape in ("rect", "box") and s.width is not None and s.height is not None:
                    # for `info`'s note on a height lying flat
                    entry["section_size"] = [float(s.width), float(s.height)]
                elif s.shape in ("box_girder", "t_girder"):
                    entry["section_size"] = [float(s.top_width), float(s.height)]
        # line elements have one formulation and the internal schema wants
        # it spelled out; the input form never asks for it
        entry["formulation"] = s.formulation or "full"
        if a.orientation is not None:
            entry["orientation"] = a.orientation
        if a.releases:
            entry["releases"] = a.releases
        if a.end_offsets is not None:
            entry["end_offsets"] = dict(a.end_offsets)
        if a.coordinate_system is not None:
            entry["coordinate_system"] = a.coordinate_system
        if a.failure is not None:
            entry["failure"] = dict(a.failure)
        return entry


def lower(d: Dict[str, Any]) -> Dict[str, Any]:
    """Input form dict -> internal case dict."""
    return InputCase.model_validate(d).lower()


def lower_with_provenance(d: Dict[str, Any]):
    """(internal case dict, {(list, index): "assign[i] (...)"})."""
    ic = InputCase.model_validate(d)
    out = ic.lower()
    prov = dict(ic._provenance)
    # the element is what the sections lowered to; say which sections
    region_secs = sorted({a.section for a in ic.assign if a.reinforcement is None})
    prov[("element",)] = "the sections " + ", ".join(repr(x) for x in region_secs) + " (as one element)"
    return out, prov


def translate_errors(errors, prov) -> list:
    """Pydantic error records of the internal case, located in the user's
    input form where the lowering made the location up."""
    lines = []
    for e in errors:
        loc = tuple(e.get("loc") or ())
        where = None
        if len(loc) >= 2 and (loc[0], loc[1]) in prov:
            rest = ".".join(str(x) for x in loc[2:])
            where = prov[(loc[0], loc[1])] + (f" > {rest}" if rest else "")
        elif loc and (loc[0],) in prov:
            rest = ".".join(str(x) for x in loc[1:])
            where = prov[(loc[0],)] + (f" > {rest}" if rest else "")
        lines.append((where, ".".join(str(x) for x in loc), e.get("msg", "")))
    return lines
