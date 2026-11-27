"""A reader for Midas Civil text models (`.mct`): what a construction-stage
model is made of, as plain data.

A bridge office's stage model is a Midas file more often than anything else,
and the first real bridge this solver was asked to follow through its
erection came as one: an arch rib hung from buckle cables, 66 stages. This
module READS such a file -- nodes, members, materials, sections, the
structure / boundary / load groups, supports, elastic and rigid links, the
load cases with their loads, and the construction stages -- and says what is
in it. Turning it into a case is `midas_case` (to come); nothing here builds
a solver.

What is read, and what is kept as text:

* `*NODE`, `*ELEMENT` (BEAM, TRUSS and their tension/compression-only
  variants as line members; PLATE and the rest are counted and kept);
* `*MATERIAL` (user values: E, nu, alpha, weight density; a standard grade
  keeps its code and grade, and E when the file gives it);
* `*SECTION`: the id, the name, the kind and shape and the numbers of the
  first line (a pipe's D and t, a box's sizes ...); further lines of a
  multi-line section are kept raw; a VALUE section's stated properties,
  a TAPERED section's two ends (sizes); `*SECT-PSCVALUE`: the properties
  Midas computed for a PSC, general or composite section, and for each end
  of a tapered one; `*TS-GROUP`: the elements a tapered section runs over;
* `*GROUP`, `*BNDR-GROUP`, `*LOAD-GROUP`; `*CONSTRAINT`; `*ELASTICLINK`
  (GEN with its six stiffnesses and its six "rigid in this direction"
  flags, RIGID, the others kept raw); `*RIGIDLINK`; `*SPRING` (linear
  point springs); `*LOCALAXIS` (a node's own axes, by angles);
  `*TDN-PROPERTY`, `*TDN-PROFILE` (2-D profiles on a straight or an
  element reference line), `*TDN-PRESTRESS`;
* `*STLDCASE` and, under each `*USE-STLD`, `*SELFWEIGHT`, `*PRETENSION`,
  `*CONLOAD`, `*BEAMLOAD` (uniform loads), `*SYSTEMPER`;
* `*STAGE`: per stage the element, boundary and load groups activated and
  deactivated, with the ages, the DEFORMED / ORIGINAL position and the day.

Everything else (`*CPOSECT4CS`, time-dependent materials, colours, design
data ...) is counted in `MidasModel.other`, so a summary can say what was
not understood instead of dropping it silently.

Lengths, forces and temperatures are the file's own (`*UNIT`); `si()` gives
the factors to metres and newtons.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_LENGTH = {"MM": 1e-3, "CM": 1e-2, "M": 1.0, "IN": 0.0254, "FT": 0.3048}
_FORCE = {"N": 1.0, "KN": 1e3, "KGF": 9.80665, "TONF": 9806.65, "LBF": 4.4482216152605, "KIPS": 4448.2216152605}
_LINE = {"BEAM", "TRUSS", "TENSTR", "COMPTR"}


@dataclass
class Element:
    kind: str
    material: int
    section: int
    nodes: Tuple[int, ...]
    angle: float = 0.0
    #: a tension- or compression-only truss's iSUB (TENSTR: 1 truss, 2 hook,
    #: 3 cable; COMPTR: 1 truss, 2 gap), EXVAL and EXVAL2 as the row has them
    sub: int = 0
    exval: Optional[float] = None
    exval2: int = 0


@dataclass
class Material:
    name: str
    kind: str                       # STEEL, CONC, USER, SRC
    E: Optional[float] = None       # force / length^2, the file's units
    nu: Optional[float] = None
    alpha: Optional[float] = None
    weight_density: Optional[float] = None     # force / length^3
    standard: str = ""              # "JTG3362-18(S) Strand1860" for a grade from a code
    raw: str = ""


@dataclass
class Section:
    name: str
    kind: str                       # DBUSER, VALUE, SRC, SOD, TAPERED, PSC ...
    shape: str = ""                 # P (pipe), B (box), SR (solid rectangle), H ...
    dims: Tuple[float, ...] = ()    # the sizes of a user-dimensioned shape
    db: str = ""                    # the name in a section database, when it is one
    lines: List[str] = field(default_factory=list)   # the record's further lines, raw
    #: the properties the file states (1.15): AREA, ASy, ASz, Ixx (torsion),
    #: Iyy, Izz in the file's units -- a VALUE section's, or a PSC / general
    #: section's from `*SECT-PSCVALUE`; of a TAPERED one, its I end's
    props: Tuple[float, ...] = ()
    props_j: Tuple[float, ...] = ()            # a tapered section's J end
    dims_j: Tuple[float, ...] = ()             # a tapered user shape's J end sizes
    offset: str = ""                           # CC (centroid), CT (centre top), LT, ...
    #: the centroid's distances to the section's edges as the file states them
    #: (CyP, CyM, CzP, CzM: to +y, -y, the top, the bottom), and a tapered one's J end's
    cent: Tuple[float, ...] = ()
    cent_j: Tuple[float, ...] = ()


@dataclass
class Link:
    kind: str                       # GEN, RIGID, TENS, COMP, SADDLE, MULTI LINEAR
    nodes: Tuple[int, int]
    stiffness: Tuple[float, ...] = ()          # SDx, SDy, SDz, SRx, SRy, SRz of a GEN link
    rigid: Tuple[bool, ...] = ()               # a GEN link's six "rigid in this direction" flags
    angle: float = 0.0
    group: str = ""
    raw: str = ""


@dataclass
class TendonType:
    name: str
    kind: str = "INTERNAL"
    material: int = 0
    area: float = 0.0                          # one tendon's
    friction: float = 0.0                      # mu, per radian
    wobble: float = 0.0                        # k, per length
    post: bool = True
    slip_begin: float = 0.0                    # the anchorage's draw-in, a length
    slip_end: float = 0.0
    bonded: bool = True
    # relaxation: bRELAX, its method RM, zeta (FT), over-stressing (bOSRF), fpk
    relax: bool = False
    relax_method: int = 0
    relax_zeta: float = 0.0
    overstressed: bool = False
    fpk: float = 0.0


@dataclass
class TendonProfile:
    """A tendon's line, as the file draws it (2-D: a plan Y(x) and an
    elevation Z(x), each a polyline whose corners are rounded by their
    radii), on a reference line: STRAIGHT from an insertion point along an
    axis, or ELEMENT from an end of a reference element along it. `y`, `z`
    are [(x, value, radius)]; `count` tendons of this one line."""
    name: str
    type: str = ""
    elements: List[int] = field(default_factory=list)
    curve: str = ""
    input: str = ""
    lines: List[str] = field(default_factory=list)
    count: int = 1
    reference: str = ""                        # STRAIGHT, ELEMENT ("" = not read)
    origin: Tuple[float, ...] = ()             # STRAIGHT: the insertion point
    axis: str = "X"                            # STRAIGHT: X, Y; ELEMENT: I-J, J-I
    ref_element: int = 0
    ref_end: str = "I"
    offset: Tuple[float, float] = (0.0, 0.0)   # ELEMENT: added to y and z
    turn: float = 0.0                          # about its axis, degrees
    y: List[Tuple[float, float, float]] = field(default_factory=list)
    z: List[Tuple[float, float, float]] = field(default_factory=list)
    fixed_angle: bool = False                  # a point whose tangent the file fixes (not followed)


def _tendon_shape(tp: "TendonProfile") -> None:
    """Fill a profile's reference line and its two polylines from its raw lines."""
    if len(tp.lines) < 3:
        return
    second = _fields(tp.lines[0])
    if len(second) > 5 and second[4].upper() == "YES":
        try:
            tp.count = max(1, int(float(second[5])))
        except ValueError:
            pass
    third = _fields(tp.lines[1])
    fourth = _fields(tp.lines[2])
    shape = third[0].upper() if third else ""
    if shape == "STRAIGHT" and len(third) >= 5:
        tp.reference, tp.origin, tp.axis = "STRAIGHT", tuple(_num(x) for x in third[1:4]), third[4].upper()
        tp.turn = _num(fourth[0]) if fourth else 0.0
    elif shape == "ELEMENT" and len(third) >= 4:
        tp.reference, tp.ref_end = "ELEMENT", ("J" if third[1].upper().endswith("J") else "I")
        tp.ref_element, tp.axis = int(_num(third[2])), third[3].upper()
        if len(fourth) >= 4:
            tp.turn, tp.offset = _num(fourth[0]), (_num(fourth[2]), _num(fourth[3]))
    for ln in tp.lines[3:]:
        head, _, rest = ln.partition("=")
        p = _fields(rest)
        if head.strip().upper() in ("Y", "Z") and len(p) >= 2:
            radius = _num(p[4]) if len(p) > 4 and p[4] else 0.0
            if len(p) > 2 and p[2].upper() == "YES":
                tp.fixed_angle = True
            (tp.y if head.strip().upper() == "Y" else tp.z).append((_num(p[0]), _num(p[1]), radius or 0.0))


@dataclass
class Stage:
    name: str
    duration: float = 0.0
    activate_elements: List[Tuple[str, float]] = field(default_factory=list)     # (group, age)
    deactivate_elements: List[Tuple[str, float]] = field(default_factory=list)   # (group, redistribution %)
    activate_boundaries: List[Tuple[str, str]] = field(default_factory=list)     # (group, DEFORMED | ORIGINAL)
    deactivate_boundaries: List[str] = field(default_factory=list)
    activate_loads: List[Tuple[str, str]] = field(default_factory=list)          # (group, FIRST | LAST | day)
    deactivate_loads: List[Tuple[str, str]] = field(default_factory=list)
    #: displacements a stage imposes on supports (a case's `*SPDISP`, a
    #: settlement group): (nodes, flags "001000", Dx..Rz), the file's units
    displacements: List[Tuple[List[int], str, Tuple[float, ...]]] = field(default_factory=list)
    #: a load case's temperatures (`cases`): ("system", None, rise), ("element",
    #: elements, T) or ("section", elements, (DIR, REF, layers))
    temperatures: List[Tuple[str, Optional[List[int]], Any]] = field(default_factory=list)


@dataclass
class MidasModel:
    path: str = ""
    units: Tuple[str, str] = ("N", "M")                      # (force, length)
    nodes: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    elements: Dict[int, Element] = field(default_factory=dict)
    materials: Dict[int, Material] = field(default_factory=dict)
    sections: Dict[int, Section] = field(default_factory=dict)
    groups: Dict[str, Tuple[List[int], List[int]]] = field(default_factory=dict)   # name -> (nodes, elements)
    boundary_groups: List[str] = field(default_factory=list)
    load_groups: List[str] = field(default_factory=list)
    constraints: List[Tuple[List[int], str, str]] = field(default_factory=list)    # (nodes, "111000", group)
    links: List[Link] = field(default_factory=list)
    rigid_links: List[Tuple[int, str, List[int], str]] = field(default_factory=list)   # (master, dofs, slaves, group)
    #: `*TDN-PROPERTY` by name, `*TDN-PROFILE` in order, `*TDN-PRESTRESS`
    #: as (load case, tendon, FORCE | STRESS, BEGIN | END | BOTH, at begin, at end, load group)
    tendon_types: Dict[str, "TendonType"] = field(default_factory=dict)
    tendons: List["TendonProfile"] = field(default_factory=list)
    tendon_stress: List[Tuple[str, str, str, str, float, float, str]] = field(default_factory=list)
    #: `*LOCALAXIS` by angles: node -> (about x, about y, about z), degrees
    local_axes: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    #: `*SPRING` (linear point springs): (nodes, fixed flags, SDx..SRz, group)
    springs: List[Tuple[List[int], Tuple[bool, ...], Tuple[float, ...], str]] = field(default_factory=list)
    load_cases: Dict[str, str] = field(default_factory=dict)                       # name -> type
    self_weight: List[Tuple[str, Tuple[float, float, float], str]] = field(default_factory=list)   # (case, factors, group)
    pretension: List[Tuple[str, int, float, str]] = field(default_factory=list)    # (case, element, tension, group)
    nodal_loads: List[Tuple[str, int, Tuple[float, ...], str]] = field(default_factory=list)       # (case, node, 6 values, group)
    #: (case, element, dir, w1 at the I end, w2 at the J end, group, projected): uniform or linearly
    #: varying over the whole member; `projected` (bPROJ) -- per unit of the length projected normal to dir
    beam_loads: List[Tuple[str, int, str, float, float, str, bool]] = field(default_factory=list)
    #: `*SECT-SCALE` by section: (AREA, ASy, ASz, Ixx, Iyy, Izz, weight factors, group, part, J end differs)
    section_scales: Dict[int, Tuple[Tuple[float, ...], str, int, bool]] = field(default_factory=dict)
    temperatures: List[Tuple[str, float, str]] = field(default_factory=list)       # (case, rise, group)
    stages: List[Stage] = field(default_factory=list)
    #: the structure group whose new members are assembled along the tangent
    #: of what stands (`*STAGE-CTRL`: initial tangent displacement), "ALL",
    #: or "" for none
    tangent_group: str = ""
    #: how the stages hold a pretension (`*STAGE-CTRL`, CPFC): "INTERNAL"
    #: (a lack of fit the structure then shares out), "EXTERNAL" (a force
    #: held in its stage), "" when the file does not say, or the word it
    #: writes there (upper case) when that is neither
    pretension_control: str = ""
    #: plate thicknesses by id (`*THICKNESS`, TYPE=VALUE), in the file's length unit
    thicknesses: Dict[int, float] = field(default_factory=dict)
    #: `*OFFSET` (beam end offsets) by element: ("ELEMENT", (RGDyi, RGDzi,
    #: RGDyj, RGDzj), group) -- rigid lengths along the member at its two
    #: ends, for bending about z and about y -- or ("GLOBAL", (RGDXi, RGDYi,
    #: RGDZi, RGDXj, RGDYj, RGDZj), group), vectors from the nodes; in the
    #: file's length unit
    offsets: Dict[int, Tuple[str, Tuple[float, ...], str]] = field(default_factory=dict)
    #: `*TS-GROUP`: the elements, in order, one tapered section runs over
    taper_groups: List[Tuple[str, List[int]]] = field(default_factory=list)
    #: `*TDM-TYPE` by name: the creep and shrinkage model, (CODE, the
    #: fields after it as written) -- JTG: STR, HU, MSIZE, BSC, AGE;
    #: China(JTG3362-2018): the same and FLYASH
    time_models: Dict[str, Tuple[str, List[str]]] = field(default_factory=dict)
    #: `*TDM-ELAST` by name: (TYPE, CODE, the fields after it)
    time_moduli: Dict[str, Tuple[str, str, List[str]]] = field(default_factory=dict)
    #: `*TDM-LINK`: material -> (creep/shrinkage model name, modulus model name), "" for none
    time_links: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    #: `*ELEM-DEPMATL`: element -> notional size h (the file's length unit), its TYPE NSM
    notional_sizes: Dict[int, float] = field(default_factory=dict)
    #: `*STAGE-CTRL` line 4: whether the stages take time-dependent effects
    #: (bINC-TDE), which ("CREEP", "SHRINKAGE" or "BOTH"; "NONE" when the
    #: creep and shrinkage box, bCNS, is off), and whether the
    #: strength grows with age (bVAR); None: the file has no stage control
    time_effects: Optional[Tuple[bool, str, bool]] = None
    #: `*SPDISP` (specified displacements of supports): (case, nodes, flags
    #: "101000", Dx..Rz in the file's units, group)
    displacements: List[Tuple[str, List[int], str, Tuple[float, ...], str]] = field(default_factory=list)
    #: `*ELTEMPER`: (case, elements, temperature change, group)
    element_temperatures: List[Tuple[str, List[int], float, str]] = field(default_factory=list)
    #: `*BSTEMPER`: [case, elements, DIR, REF, [(TYPE, B, H1, T1, H2, T2), ...], group, not read]: the
    #: temperature over a section's depth, layer by layer, linear in each, the
    #: depths H from the REF face, in the file's length unit
    section_temperatures: List[Tuple[str, List[int], str, str, List[Tuple[float, ...]], str]] = \
        field(default_factory=list)
    #: `*SM-GROUP`: settlement group -> (displacement, nodes), the file's length unit
    settlements: Dict[str, Tuple[float, List[int]]] = field(default_factory=dict)
    other: Dict[str, int] = field(default_factory=dict)      # blocks not read: keyword -> data lines

    def si(self) -> Tuple[float, float]:
        """(newtons per file force unit, metres per file length unit)."""
        return _FORCE[self.units[0]], _LENGTH[self.units[1]]


def expand(text: str) -> List[int]:
    """A Midas id list: `3 5 10to20 30to50by5` -> the ids."""
    out: List[int] = []
    for tok in text.replace(",", " ").split():
        m = re.fullmatch(r"(\d+)to(\d+)(?:by(\d+))?", tok, flags=re.I)
        if m:
            a, b, s = int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)
            out.extend(range(a, b + 1, s))
        elif tok.isdigit():
            out.append(int(tok))
        else:
            raise ValueError(f"not an id list: {text.strip()[:60]!r} (at {tok!r})")
    return out


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "gbk", "gb18030", "cp949", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _records(text: str):
    """(keyword, argument, data lines) of every block, in file order: a
    line ending in a backslash goes on in the next; `;` starts a comment."""
    key, arg, rows = None, "", []
    pending = ""
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("*"):
            if key is not None:
                yield key, arg, rows
            head = line[1:].split(",", 1)
            key, arg, rows, pending = head[0].strip().upper(), (head[1].strip() if len(head) > 1 else ""), [], ""
            continue
        if key is None:
            continue
        if line.rstrip().endswith("\\"):
            pending += line.rstrip()[:-1] + " "
            continue
        rows.append(pending + line)
        pending = ""
    if key is not None:
        yield key, arg, rows


def _fields(row: str) -> List[str]:
    return [p.strip() for p in row.split(",")]


def _num(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _material(p: List[str], raw: str) -> Material:
    m = Material(name=p[2], kind=p[1].upper(), raw=raw)
    data = p[9:]
    if not data:
        return m
    if data[0] == "2" and len(data) >= 5:                 # user values
        m.E, m.nu, m.alpha, m.weight_density = _num(data[1], None), _num(data[2], None), _num(data[3], None), _num(data[4], None)
    elif data[0] == "1":                                  # a grade from a code
        m.standard = " ".join(x for x in data[1:4] if x)
        if len(data) > 5 and data[5]:
            m.E = _num(data[5], None)
    return m


def _stage_pairs(text: str) -> List[Tuple[str, str]]:
    p = [x.strip() for x in text.split(",")]
    p = [x for x in p if x != ""] if len(p) % 2 else p
    return [(p[i], p[i + 1]) for i in range(0, len(p) - 1, 2)]


def read(path: str) -> MidasModel:
    """The model in the `.mct` file at `path`."""
    with open(path, "rb") as f:
        text = _decode(f.read())
    m = MidasModel(path=str(path))
    case = ""
    for key, arg, rows in _records(text):
        if key == "UNIT" and rows:
            p = _fields(rows[0])
            force, length = p[0].upper(), p[1].upper()
            if force not in _FORCE or length not in _LENGTH:
                raise ValueError(f"*UNIT {rows[0].strip()!r}: unknown force or length unit")
            m.units = (force, length)
        elif key == "NODE":
            for r in rows:
                p = _fields(r)
                m.nodes[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
        elif key == "ELEMENT":
            for r in rows:
                p = _fields(r)
                kind = p[1].upper()
                if kind in _LINE:
                    el = Element(kind, int(p[2]), int(p[3]), (int(p[4]), int(p[5])), _num(p[6]) if len(p) > 6 else 0.0)
                    if kind in ("TENSTR", "COMPTR"):
                        # ..., ANGLE, iSUB, EXVAL, EXVAL2, bLMT
                        el.sub = int(_num(p[7])) if len(p) > 7 else 0
                        if len(p) > 8 and p[8] != "":
                            el.exval = _num(p[8])
                        el.exval2 = int(_num(p[9])) if len(p) > 9 else 0
                    m.elements[int(p[0])] = el
                else:
                    n = [int(x) for x in p[4:12] if x.lstrip("-").isdigit() and int(x) > 0][:8]
                    if kind == "PLATE":
                        n = [int(x) for x in p[4:8] if x.isdigit() and int(x) > 0]
                    m.elements[int(p[0])] = Element(kind, int(p[2]), int(p[3]), tuple(n))
        elif key == "MATERIAL":
            for r in rows:
                p = _fields(r)
                if p[0].isdigit() and len(p) > 2:
                    m.materials[int(p[0])] = _material(p, r)
        elif key == "SECTION":
            last = None
            for r in rows:
                p = _fields(r)
                head = p[0].isdigit() and len(p) > 3 and re.fullmatch(r"[A-Z][A-Z0-9_-]*", p[1] or "") is not None
                if head and int(p[0]) in m.sections and m.sections[int(p[0])].props:
                    last = None                # `*SECT-PSCVALUE` stated it: that stands
                    continue
                if head:
                    s = Section(name=p[2], kind=p[1].upper(), offset=p[3].upper())
                    if s.kind == "TAPERED":
                        while p and not p[-1]:
                            p = p[:-1]                     # a row may end with a comma
                        # ..., SHAPE, iyVAR, izVAR, STYPE: counted from the end
                        # (a composite's row has one field more before them)
                        s.shape = p[-4].upper() if len(p) > 6 else ""
                        s.db = p[-1].upper()               # USER, DB, VALUE, PSC
                    elif s.kind == "DBUSER" and len(p) > 13:
                        s.shape = p[12].upper()
                        if p[13] == "2":
                            s.dims = tuple(_num(x) for x in p[14:22])
                        else:
                            s.db = ", ".join(x for x in p[14:16] if x)
                    elif len(p) > 12:
                        s.shape = p[12].upper()
                    m.sections[int(p[0])] = last = s
                elif last is not None:
                    last.lines.append(r.strip())
            for s in m.sections.values():
                nums = [[_num(x, None) for x in _fields(ln)] for ln in s.lines[:1]]
                if s.kind == "VALUE" and nums and len(nums[0]) >= 6 and None not in nums[0][:6]:
                    s.props = tuple(nums[0][:6])
                    second = [_num(x, None) for x in _fields(s.lines[1])] if len(s.lines) > 1 else []
                    if len(second) >= 4 and None not in second[:4]:
                        s.cent = tuple(second[:4])
                elif s.kind == "TAPERED" and s.db == "USER" and nums and len(nums[0]) >= 16:
                    v = [x or 0.0 for x in nums[0]]
                    s.dims, s.dims_j = tuple(v[:8]), tuple(v[8:16])
        elif key == "SECT-PSCVALUE":
            # the properties Midas computed for a PSC, a general or a
            # composite section, and for each end of a tapered one: stated,
            # so read rather than worked out again from the outline
            cur, body = None, []

            def close(cur=None, body=None):
                if cur is None:
                    return
                sid, sec = cur
                six = [[_num(x, None) for x in _fields(ln)] for ln in body[:5]]
                ok = lambda row: len(row) >= 6 and None not in row[:6]
                four = lambda row: tuple(row[:4]) if len(row) >= 4 and None not in row[:4] else ()
                if six and ok(six[0]):
                    sec.props = tuple(six[0][:6])
                    sec.cent = four(six[1]) if len(six) > 1 else ()
                if sec.kind == "TAPERED" and len(six) > 3 and ok(six[3]):
                    sec.props_j = tuple(six[3][:6])
                    sec.cent_j = four(six[4]) if len(six) > 4 else ()
                sec.lines = body
                m.sections[sid] = sec
            for r in rows:
                t = r.strip()
                if t.upper().startswith("SECT="):
                    close(cur, body)
                    p = _fields(t.split("=", 1)[1])
                    kind = p[1].upper() if len(p) > 1 else ""
                    shape = next((x.upper() for x in p[4:] if x.upper() in ("VALU", "GEN", "PSC", "CMP")), "")
                    cur = (int(p[0]), Section(name=p[2] if len(p) > 2 else "", kind=kind, shape=shape,
                                              offset=p[3].upper() if len(p) > 3 else ""))
                    body = []
                elif cur is not None:
                    body.append(t)
            close(cur, body)
        elif key == "TS-GROUP":
            for r in rows:
                p = _fields(r)
                if len(p) > 1:
                    m.taper_groups.append((p[0], expand(p[1])))
        elif key == "GROUP":
            for r in rows:
                p = _fields(r)
                m.groups[p[0]] = (expand(p[1]) if len(p) > 1 else [], expand(p[2]) if len(p) > 2 else [])
        elif key == "BNDR-GROUP":
            m.boundary_groups += [_fields(r)[0] for r in rows]
        elif key == "LOAD-GROUP":
            m.load_groups += [_fields(r)[0] for r in rows]
        elif key == "CONSTRAINT":
            for r in rows:
                p = _fields(r)
                m.constraints.append((expand(p[0]), p[1], p[2] if len(p) > 2 else ""))
        elif key == "ELASTICLINK":
            for r in rows:
                p = _fields(r)
                kind = p[3].upper()
                link = Link(kind, (int(p[1]), int(p[2])), angle=_num(p[4]), group=p[-1], raw=r.strip())
                if kind == "GEN" and len(p) >= 17:
                    link.stiffness = tuple(_num(x) for x in p[11:17])
                    link.rigid = tuple(x.strip().upper() == "YES" for x in p[5:11])
                m.links.append(link)
        elif key == "SECT-SCALE":
            # iSEC, AREA_SF, ASY_SF, ASZ_SF, IXX_SF, IYY_SF, IZZ_SF, WGT_SF, GROUP, iPart, bDiffIJ (and a J line)
            j_line = False
            for r in rows:
                p = _fields(r)
                if j_line:                                # the J end's factors of the row before
                    j_line = False
                    continue
                if len(p) >= 8 and p[0].isdigit():
                    j_line = len(p) > 10 and p[10].upper() == "YES"
                    if len(p) > 9 and p[9] and int(_num(p[9], 1.0)) != 1:
                        # a part of a composite section other than the first: its own row, not read
                        m.other["SECT-SCALE (a later part of a composite section)"] = \
                            m.other.get("SECT-SCALE (a later part of a composite section)", 0) + 1
                        continue
                    m.section_scales[int(p[0])] = (tuple(_num(x, 1.0) for x in p[1:8]), p[8] if len(p) > 8 else "",
                                                   int(_num(p[9], 1.0)) if len(p) > 9 and p[9] else 1,
                                                   len(p) > 10 and p[10].upper() == "YES")
        elif key == "OFFSET":
            # ELEM_LIST, TYPE, then RGDXi, RGDYi, RGDZi, RGDXj, RGDYj, RGDZj, GROUP (TYPE=GLOBAL)
            # or RGDyi(Mzi), RGDzi(Myi), RGDyj(Mzj), RGDzj(Myj), GROUP (TYPE=ELEMENT)
            for r in rows:
                p = _fields(r)
                kind = p[1].upper() if len(p) > 1 else ""
                n = {"GLOBAL": 6, "ELEMENT": 4}.get(kind)
                if n is None or len(p) < 2 + n:
                    m.other["OFFSET (not read)"] = m.other.get("OFFSET (not read)", 0) + 1
                    continue
                try:
                    values = tuple(float(x) for x in p[2:2 + n])
                except ValueError:                        # a field that is not a number: another layout
                    m.other["OFFSET (not read)"] = m.other.get("OFFSET (not read)", 0) + 1
                    continue
                for e in expand(p[0]):
                    m.offsets[e] = (kind, values, p[2 + n] if len(p) > 2 + n else "")
        elif key == "RIGIDLINK":
            for r in rows:
                p = _fields(r)
                m.rigid_links.append((int(p[0]), p[1], expand(p[2]), p[3] if len(p) > 3 else ""))
        elif key == "LOCALAXIS":
            # NODE_LIST, 1, the angles about x, y and z (degrees): a node's own axes, for its supports and springs
            for r in rows:
                p = _fields(r)
                if len(p) >= 5 and p[1] == "1":
                    for n in expand(p[0]):
                        m.local_axes[n] = (_num(p[2]), _num(p[3]), _num(p[4]))
                else:
                    m.other["LOCALAXIS (not by angles)"] = m.other.get("LOCALAXIS (not by angles)", 0) + 1
        elif key == "SPRING":
            # NODE_LIST, LINEAR, six "fixed in this direction" flags, SDx..SRz, ..., GROUP at the 22nd field
            for r in rows:
                p = _fields(r)
                if len(p) < 22 or p[1].upper() != "LINEAR" or any(x.upper() not in ("YES", "NO") for x in p[2:8]):
                    m.other["SPRING (not read)"] = m.other.get("SPRING (not read)", 0) + 1
                    continue
                m.springs.append((expand(p[0]), tuple(x.strip().upper() == "YES" for x in p[2:8]),
                                  tuple(_num(x) for x in p[8:14]), p[21] if len(p) > 21 else ""))
        elif key == "TDN-PROPERTY":
            # NAME, TYPE, MATL, AREA, DIA, RM, RC, FF (friction), WF (wobble), US, YS, LT, ASB, ASE (anchorage slip), bBONDED ...
            for r in rows:
                p = _fields(r)
                if len(p) >= 15:
                    m.tendon_types[p[0]] = TendonType(name=p[0], kind=p[1].upper(), material=int(p[2]), area=_num(p[3]),
                                                      friction=_num(p[7]), wobble=_num(p[8]), post=p[11].upper() == "POST",
                                                      slip_begin=_num(p[12]), slip_end=_num(p[13]),
                                                      bonded=p[14].upper() == "YES")
                    if len(p) >= 22:
                        # ..., ALPHA, bOSRF, FT, LR, FPK, ACHANGE, bRELAX, ...
                        t_ = m.tendon_types[p[0]]
                        t_.relax = p[21].strip().upper() == "YES"
                        t_.relax_method = int(_num(p[5]) or 0)
                        t_.relax_zeta = _num(p[17]) or 0.0
                        t_.overstressed = p[16].strip().upper() == "YES"
                        t_.fpk = _num(p[19]) or 0.0
        elif key == "TDN-PROFILE":
            cur = None
            for r in rows:
                t = r.strip()
                if t.upper().startswith("NAME="):
                    p = _fields(t.split("=", 1)[1])
                    cur = TendonProfile(name=p[0], type=p[1], elements=expand(p[2]) if len(p) > 2 else [],
                                        curve=p[5].upper() if len(p) > 5 else "", input=p[6].upper() if len(p) > 6 else "")
                    m.tendons.append(cur)
                    cur.lines = []
                elif cur is not None:
                    cur.lines.append(t)
            for tp in m.tendons:
                _tendon_shape(tp)
        elif key == "TDN-PRESTRESS":
            for r in rows:
                p = _fields(r)
                if len(p) >= 7:
                    m.tendon_stress.append((case, p[0], p[1].upper(), p[2].upper(), _num(p[3]), _num(p[4]), p[6]))
        elif key == "STLDCASE":
            for r in rows:
                p = _fields(r)
                m.load_cases[p[0]] = p[1] if len(p) > 1 else ""
        elif key == "USE-STLD":
            case = arg
        elif key == "SELFWEIGHT":
            for r in rows:
                p = _fields(r)
                m.self_weight.append((case, (_num(p[0]), _num(p[1]), _num(p[2])), p[3] if len(p) > 3 else ""))
        elif key == "PRETENSION":
            for r in rows:
                p = _fields(r)
                for e in expand(p[0]):
                    m.pretension.append((case, e, float(p[1]), p[2] if len(p) > 2 else ""))
        elif key == "ELTEMPER":
            for r in rows:
                p = _fields(r)
                if len(p) > 1 and p[0].strip():
                    m.element_temperatures.append((case, expand(p[0]), _num(p[1]), p[2].strip() if len(p) > 2 else ""))
        elif key == "BSTEMPER":
            cur = None
            for r in rows:
                p = _fields(r)
                if not p or not p[0].strip():
                    continue
                if p[0].strip().upper() in ("ELEMENT", "MATERIAL", "INPUT", "SECTION"):
                    # a layer: TYPE, ELAST, THERMAL, B, H1, T1, H2, T2 (the PSC form is counted)
                    if cur is not None and len(p) >= 8 and not cur[6]:
                        cur[4].append((p[0].strip().upper(),) + tuple(_num(x) for x in p[3:8]))
                    elif cur is not None:
                        cur[6] = True                  # a layer not read: the whole profile is not followed
                    continue
                psc = len(p) > 5 and p[5].strip().upper() == "YES"
                cur = [case, expand(p[0]), p[1].strip().upper(), p[2].strip().upper(), [],
                       p[4].strip() if len(p) > 4 else "", psc]
                m.section_temperatures.append(cur)
        elif key == "SPDISP":
            for r in rows:
                p = _fields(r)
                if len(p) < 8 or not p[0].strip():
                    continue
                m.displacements.append((case, expand(p[0]), p[1].strip(), tuple(_num(x) for x in p[2:8]),
                                        p[8].strip() if len(p) > 8 else ""))
        elif key == "SM-GROUP":
            for r in rows:
                p = _fields(r)
                if len(p) > 2 and p[0].strip():
                    m.settlements[p[0].strip()] = (_num(p[1]), expand(" ".join(p[2:])))
        elif key == "CONLOAD":
            for r in rows:
                p = _fields(r)
                for n in expand(p[0]):
                    m.nodal_loads.append((case, n, tuple(_num(x) for x in p[1:7]), p[7] if len(p) > 7 else ""))
        elif key == "BEAMLOAD":
            for r in rows:
                p = _fields(r)
                # ELEM_LIST, CMD, TYPE, DIR, bPROJ, bECCEN, ECCDIR, I-END, J-END, bJ-END, D1, P1, D2, P2, D3, P3, D4, P4, GROUP
                if len(p) > 18 and p[2].upper() == "UNILOAD" and p[5].upper() != "YES" \
                        and _num(p[10]) == 0.0 and _num(p[12]) == 1.0:
                    for e in expand(p[0]):
                        m.beam_loads.append((case, e, p[3].upper(), _num(p[11]), _num(p[13]), p[18],
                                             p[4].upper() == "YES"))
                else:
                    m.other["BEAMLOAD (not uniform)"] = m.other.get("BEAMLOAD (not uniform)", 0) + 1
        elif key == "SYSTEMPER":
            for r in rows:
                p = _fields(r)
                m.temperatures.append((case, _num(p[0]), p[1] if len(p) > 1 else ""))
        elif key == "STAGE":
            stage, last = None, None
            for r in rows:
                s = r.strip()
                tag = re.match(r"([A-Z]+)=(.*)", s)
                if tag:
                    last, body = tag.group(1), tag.group(2)
                else:
                    body = s                              # a line that goes on with the last keyword
                if last == "NAME":
                    p = _fields(body)
                    stage = Stage(p[0], _num(p[1]) if len(p) > 1 else 0.0)
                    m.stages.append(stage)
                elif stage is None:
                    continue
                elif last == "AELEM":
                    stage.activate_elements += [(g, _num(a)) for g, a in _stage_pairs(body)]
                elif last == "DELEM":
                    stage.deactivate_elements += [(g, _num(a)) for g, a in _stage_pairs(body)]
                elif last == "ABNDR":
                    stage.activate_boundaries += _stage_pairs(body)
                elif last == "DBNDR":
                    stage.deactivate_boundaries += [x for x in _fields(body) if x]
                elif last == "ALOAD":
                    stage.activate_loads += _stage_pairs(body)
                elif last == "DLOAD":
                    stage.deactivate_loads += _stage_pairs(body)
        elif key == "TDM-TYPE":
            for r in rows:
                head, _, rest = r.partition("=")
                p = _fields(rest)
                if head.strip().upper() == "NAME" and len(p) > 1:
                    m.time_models[p[0]] = (p[1].strip(), p[2:])
                else:                       # a USER model's curve lines, and the like
                    m.other[key] = m.other.get(key, 0) + 1
        elif key == "TDM-ELAST":
            for r in rows:
                head, _, rest = r.partition("=")
                p = _fields(rest)
                if head.strip().upper() == "NAME" and len(p) > 2:
                    m.time_moduli[p[0]] = (p[1].strip().upper(), p[2].strip(), p[3:])
                else:
                    m.other[key] = m.other.get(key, 0) + 1
        elif key == "TDM-LINK":
            for r in rows:
                p = _fields(r) + ["", ""]
                if p[0].isdigit():
                    m.time_links[int(p[0])] = (p[1].strip(), p[2].strip())
        elif key == "ELEM-DEPMATL":
            for r in rows:
                p = _fields(r)
                if len(p) > 2 and p[1].strip().upper() == "NSM":
                    for e in expand(p[0]):
                        m.notional_sizes[e] = _num(p[2])
                else:
                    m.other[key] = m.other.get(key, 0) + 1
        elif key == "THICKNESS":
            for row in rows:
                p = [x.strip() for x in row.split(",")]
                if "VALUE" in [x.upper() for x in p]:
                    i = [x.upper() for x in p].index("VALUE")
                    try:
                        # iTHK, VALUE, <n>, bSAME, THIK-IN, ...: the number after the first YES/NO
                        j = next(q for q in range(i + 1, len(p)) if p[q].upper() in ("YES", "NO"))
                        m.thicknesses[int(p[0])] = float(p[j + 1])
                    except (ValueError, IndexError, StopIteration):
                        m.other[key] = m.other.get(key, 0) + 1
                else:
                    m.other[key] = m.other.get(key, 0) + 1
        elif key in ("VERSION", "ENDDATA"):
            pass
        else:
            if rows:
                m.other[key] = m.other.get(key, 0) + len(rows)
                if key == "STAGE-CTRL" and rows:
                    # line 1: bLAST-FINAL, FINAL-STAGE, CPFC, bEXT_REPL, ...
                    first = [x.strip().upper() for x in rows[0].split(",")]
                    if len(first) > 2:
                        m.pretension_control = first[2]
                    # line 4: bINC-TDE, bCNS, TYPE, ... (lines 1-3 end where a
                    # row does not end in the continuation backslash)
                    # (no line 4: no time-dependent effects); bVAR, its 8th
                    # field, the growth of the strength (and modulus) with age
                    m.time_effects = (False, "", False)
                    if len(rows) > 3:
                        f4 = [x.strip().upper() for x in rows[3].split(",")] + [""] * 8
                        # bINC-TDE, bCNS, TYPE, ..., bVAR: bCNS is the dialog's
                        # "Creep & Shrinkage" box -- off, TYPE says nothing
                        # (and the strength may still grow with age)
                        m.time_effects = (f4[0] == "YES", f4[2] if f4[1] != "NO" else "NONE", f4[7] == "YES")
                    # ..., bITD, iITD, GROUP, ...: "YES, GROUP, <name>" or "YES, ALL"
                    p = [x.strip() for x in ",".join(rows).split(",")]
                    for i in range(len(p) - 1):
                        if p[i].upper() == "YES" and p[i + 1].upper() == "GROUP" and i + 2 < len(p):
                            m.tangent_group = p[i + 2]
                            break
                        if p[i].upper() == "YES" and p[i + 1].upper() == "ALL":
                            m.tangent_group = "ALL"
                            break
    return m


def with_stage_order(model: MidasModel, names) -> MidasModel:
    """The model with its construction stages in the order `names` gives
    (every stage's name, once). A set of results is sometimes of the same
    model run in another sequence -- two ribs' segments interleaved
    differently -- and is only comparable stage by stage in its own order."""
    import dataclasses
    by_name = {s.name.strip(): s for s in model.stages}
    names = [str(n).strip() for n in names]
    if len(by_name) != len(model.stages):
        raise ValueError("the model has two stages of one name: they cannot be ordered by name")
    if sorted(names) != sorted(by_name):
        missing = sorted(set(by_name) - set(names)); extra = sorted(set(names) - set(by_name))
        raise ValueError(f"the order must name every stage once: missing {missing[:3]}, not in the model {extra[:3]}"
                         + (", or a name twice" if not missing and not extra else ""))
    return dataclasses.replace(model, stages=[by_name[n] for n in names])


def check(m: MidasModel) -> List[str]:
    """What does not hang together in the model as read: an element on a
    node, a material or a section that is not there, a stage that names a
    group nobody defined. Empty when all is well."""
    out: List[str] = []
    bad = sum(1 for e in m.elements.values() if any(n not in m.nodes for n in e.nodes))
    if bad:
        out.append(f"{bad} element(s) on a node that is not in *NODE")
    lines = [e for e in m.elements.values() if e.kind in _LINE]
    bad = sorted({e.material for e in lines if e.material not in m.materials})
    if bad:
        out.append(f"line members name material(s) {bad[:8]} not in *MATERIAL")
    bad = sorted({e.section for e in lines if e.section not in m.sections})
    if bad:
        out.append(f"line members name section(s) {bad[:8]} not in *SECTION")
    for name, (nodes, elems) in m.groups.items():
        if any(n not in m.nodes for n in nodes) or any(e not in m.elements for e in elems):
            out.append(f"group {name!r} lists a node or an element that is not in the model")
    bset, lset = set(m.boundary_groups), set(m.load_groups)
    for s in m.stages:
        for g, _ in s.activate_elements + s.deactivate_elements:
            if g not in m.groups:
                out.append(f"stage {s.name!r}: element group {g!r} is not in *GROUP")
        for g in [g for g, _ in s.activate_boundaries] + s.deactivate_boundaries:
            if g not in bset:
                out.append(f"stage {s.name!r}: boundary group {g!r} is not in *BNDR-GROUP")
        for g, _ in s.activate_loads + s.deactivate_loads:
            if g not in lset:
                out.append(f"stage {s.name!r}: load group {g!r} is not in *LOAD-GROUP")
    for case, e, _t, _g in m.pretension:
        if e not in m.elements:
            out.append(f"*PRETENSION ({case}) on element {e}, which is not in the model")
            break
    return out


def summary(m: MidasModel) -> str:
    """One page on what the file holds."""
    kinds: Dict[str, int] = {}
    for e in m.elements.values():
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    xs = list(m.nodes.values())
    f, length = m.si()
    lines = [f"midas model   {m.path}",
             f"  units       {m.units[0]}, {m.units[1]}",
             f"  nodes       {len(m.nodes)}"]
    if xs:
        lo = [min(x[i] for x in xs) * length for i in range(3)]
        hi = [max(x[i] for x in xs) * length for i in range(3)]
        lines.append("  extent      " + " x ".join(f"{hi[i] - lo[i]:.1f}" for i in range(3)) + " m")
    lines.append("  elements    " + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])))
    lines.append(f"  materials   {len(m.materials)}"
                 f" ({sum(1 for x in m.materials.values() if x.E is None)} without E in the file)")
    shapes: Dict[str, int] = {}
    for s in m.sections.values():
        k = f"{s.kind} {s.shape}".strip()
        shapes[k] = shapes.get(k, 0) + 1
    lines.append(f"  sections    {len(m.sections)}: " + ", ".join(f"{n} {k}" for k, n in sorted(shapes.items(), key=lambda kv: -kv[1])))
    lines.append(f"  groups      {len(m.groups)} structure, {len(m.boundary_groups)} boundary, {len(m.load_groups)} load")
    link_kinds: Dict[str, int] = {}
    for k in m.links:
        link_kinds[k.kind] = link_kinds.get(k.kind, 0) + 1
    if m.offsets:
        lines.append(f"  offsets     {len(m.offsets)} beam end offset(s)")
    lines.append(f"  supports    {sum(len(c[0]) for c in m.constraints)} nodes in {len(m.constraints)} constraint(s)")
    lines.append("  links       " + (", ".join(f"{n} {k}" for k, n in sorted(link_kinds.items())) or "none")
                 + (f"; {len(m.rigid_links)} rigid link(s)" if m.rigid_links else ""))
    lines.append(f"  load cases  {len(m.load_cases)}: {len(m.self_weight)} self weight, {len(m.pretension)} pretensions, "
                 f"{len(m.nodal_loads)} nodal, {len(m.beam_loads)} member, {len(m.temperatures)} temperature")
    lines.append(f"  stages      {len(m.stages)}" + (f": {m.stages[0].name} ... {m.stages[-1].name}" if m.stages else ""))
    if m.other:
        lines.append("  not read    " + ", ".join(f"{k} ({n})" for k, n in sorted(m.other.items())))
    problems = check(m)
    lines.append("  check       " + ("consistent" if not problems else f"{len(problems)} problem(s)"))
    lines += [f"    {p}" for p in problems[:12]]
    return "\n".join(lines)
