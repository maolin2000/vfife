"""The arms of members set off their nodes: `offset_arms:`.

A Midas girder is drawn by the top of its section; the member is on its
centroid's line, a node of its own under each drawn node, joined to it by a
rigid arm. The arm's stiffness is `factor` times what the
members it carries do, along and in turning -- so each arm has a section of
its own, and a converted T-girder bridge had 846 of them. Here the rule is
written once:

    offset_arms:
      material: rigid          # the arms' material (its E)
      factor: 30.0             # times what the members set off carry
      limit: 1.0e13            # a member's share is capped here (N/m)
      regions: [E266, ...]     # the arms: one member each, drawn node -> centroid

and each arm's section is worked out when the case is read, from the mesh:
the beams at its centroid node whose other end is the centroid of another
listed arm are the members set off there. Each carries (the larger of EA/L
and 12EIy/L^3, capped at `limit`) along, and (4EIy/L, capped at limit L^2/3)
in turning, L the member's drawn length (between the two arms' drawn
nodes). The arm takes the largest of each, and over its own length r

    area = factor along r / E,   iy = iz = factor max(turning r, along r^3 / 12) / E,   j = 2 iy

-- what the Midas converter has worked out since 1.15, from the coordinates
as the mesh holds them. Its orientation is global Z, or X for an arm nearer
the vertical than 45 degrees.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from .tendons import _Strict

#: the section-name stem of an expanded arm
STEM = "offset_arm"


class OffsetArmSpec(_Strict):
    #: the arms' material (its E)
    material: str
    #: an arm's stiffness: this times what the members set off carry, along and in turning
    factor: float = Field(30.0, gt=0.0)
    #: the most a member hands an arm (force/length; none: 1e13 N/m in the case's `units`; in turning, limit L^2/3)
    limit: Optional[float] = Field(None, gt=0.0)
    #: the arms: one member per region, from the drawn node to the centroid node
    regions: List[str] = Field(min_length=1)


def free_axis(a, b) -> Tuple[float, float, float]:
    """Global Z, or global X for a member nearer the vertical than 45
    degrees (as the Midas converter orients a rigid arm)."""
    x = [b[i] - a[i] for i in range(3)]
    n = math.sqrt(sum(c * c for c in x)) or 1.0
    return (1.0, 0.0, 0.0) if abs(x[2]) / n > 0.7071 else (0.0, 0.0, 1.0)


class ArmError(ValueError):
    """What one arm cannot be: `region` names it."""

    def __init__(self, region, text):
        super().__init__(text)
        self.region = region


def arm_sections(d: Dict[str, Any], mesh, bad: Optional[List[str]] = None
                 ) -> List[Tuple[str, Dict[str, Any], List[float]]]:
    """[(region, section, orientation)] of each arm `d['offset_arms']` lists.
    With `bad` a list, an arm that cannot be worked out is put in it (by
    region) and left out, instead of refused (the converter keeps those as
    sections of their own)."""
    spec = OffsetArmSpec.model_validate(d["offset_arms"])
    limit = spec.limit
    if limit is None:
        from ..core.units import scales
        sc = scales(d.get("units"))                 # (a stiffness is a stress times a length)
        unit = sc["stress"] * sc["length"] if sc.get("stress") and sc.get("length") else 1.0
        limit = 1.0e13 / unit
    mats = d.get("materials") or {}
    sections = d.get("sections") or {}
    if spec.material not in mats or "E" not in (mats[spec.material] or {}):
        raise ValueError(f"`offset_arms`: material {spec.material!r} is not in `materials` with an `E`")
    e_arm = float(mats[spec.material]["E"])
    conn = (mesh.cell_blocks or {}).get("truss2", mesh.elements)
    xyz = mesh.nodes.tolist()

    def one(region):
        if region not in (mesh.element_regions or {}):
            raise ValueError(f"`offset_arms`: {region!r} is not a group of members in the mesh")
        rows = list((mesh.element_regions[region] or {}).get("truss2", ()))
        if len(rows) != 1:
            raise ValueError(f"`offset_arms`: {region!r} has {len(rows)} members; an arm is one")
        a, b = (int(q) for q in conn[rows[0]][:2])
        return a, b

    def problem(region, text):
        if bad is None:
            raise ArmError(region, text)
        bad.append(region)

    if len(set(spec.regions)) != len(spec.regions):
        raise ValueError("`offset_arms`: a region is listed twice")
    arms = []
    for r in spec.regions:
        try:
            arms.append((r,) + one(r))
        except ValueError as exc:
            problem(r, str(exc))
    drawn_of = {}                                   # centroid node -> drawn node
    for r, n, c in list(arms):
        if c in drawn_of and drawn_of[c] != n:
            problem(r, f"`offset_arms`: {r!r} ends at a centroid node another arm reaches from another node")
            arms.remove((r, n, c))
            continue
        drawn_of[c] = n

    # the beams (not of the arms' material) and their sections, by element row:
    # assigned to a group of the mesh (or of `mesh.groups`, or a node group as
    # the solver reads one), with `area`, `iy` and the material's `E` written
    mesh_groups = (d.get("mesh") or {}).get("groups") or {}

    def rows_of(region):
        if region in (mesh.element_regions or {}):
            return [int(q) for q in (mesh.element_regions[region] or {}).get("truss2", ())]
        if region in mesh_groups:
            return sorted({q for part in mesh_groups[region] for q in (rows_of(part) or ())})
        if region in (mesh.regions or {}):
            got = mesh.region_elements(region, "truss2", conn)
            return [] if got is None else [int(q) for q in got]
        return None

    beam_of: Dict[int, Dict[str, Any]] = {}
    unread: Dict[int, str] = {}                     # rows of beams whose stiffness is not written out
    for entry in d.get("assign") or []:
        sec = sections.get(entry.get("section"))
        if not isinstance(sec, dict) or sec.get("type") != "beam" or sec.get("material") == spec.material:
            continue
        region = entry.get("region")
        rows = rows_of(region)
        if rows is None:
            raise ValueError(f"`offset_arms` works its arms out from the beams' groups of the mesh; the beams "
                             f"of {region!r} (a region written in the case) cannot be placed: give them a group")
        mat = mats.get(sec.get("material")) or {}
        if "E" not in mat or not all(k in sec for k in ("area", "iy")):
            for row in rows:
                unread[row] = f"{entry.get('section')!r} (on {region!r})"
            continue
        for row in rows:
            beam_of[row] = {"E": float(mat["E"]), "area": float(sec["area"]), "iy": float(sec["iy"])}
    at_node: Dict[int, List[int]] = {}
    for row in list(beam_of) + list(unread):
        for q in conn[row][:2]:
            at_node.setdefault(int(q), []).append(row)

    out = []
    for region, n, c in arms:
        along = turning = 0.0
        found, why = False, None
        for row in at_node.get(c, ()):
            p, q = (int(v) for v in conn[row][:2])
            other = q if p == c else p
            if other not in drawn_of:
                continue                            # not a member set off (a pier drawn at the centroid)
            if row in unread:
                why = (f"`offset_arms`: a member set off at {region!r} has the section {unread[row]}, whose "
                       "`area`, `iy` or material's `E` is not written out")
                break
            found = True
            x_a, x_b = xyz[drawn_of[p]], xyz[drawn_of[q]]
            x_ = [x_b[i] - x_a[i] for i in range(3)]
            n_ = math.sqrt(sum(v * v for v in x_))
            m = beam_of[row]
            got = (max(m["E"] * m["area"] / n_, 12.0 * m["E"] * m["iy"] / n_ ** 3), 4.0 * m["E"] * m["iy"] / n_)
            got = (min(got[0], limit), min(got[1], limit * n_ * n_ / 3.0))
            along, turning = max(along, got[0]), max(turning, got[1])
        if why is None and not found:
            why = (f"`offset_arms`: no member set off is at the centroid end of {region!r} "
                   "(a beam to the centroid of another listed arm)")
        if why is not None:
            problem(region, why)
            continue
        reach = math.dist(xyz[n], xyz[c])
        i_arm = spec.factor * max(turning * reach, along * reach ** 3 / 12.0) / e_arm
        sec = {"type": "beam", "material": spec.material, "shape": "general",
               "area": spec.factor * along * reach / e_arm, "iy": i_arm, "iz": i_arm, "j": 2.0 * i_arm}
        out.append((region, sec, list(free_axis(xyz[n], xyz[c]))))
    return out


def expand(d: Dict[str, Any], mesh) -> Dict[str, Any]:
    """`d` with `offset_arms` written out as a section and an assignment per arm."""
    if d.get("offset_arms") is None:
        return d
    if not d.get("offset_arms"):
        raise ValueError("`offset_arms` is empty: give `material` and `regions` (or leave it out)")
    made = arm_sections(d, mesh)
    d = dict(d)
    d.pop("offset_arms")
    sections = dict(d.get("sections") or {})
    taken = [k for k in sections if re.fullmatch(STEM + r"_\d+", k)]
    if taken:
        raise ValueError(f"`offset_arms` names its sections {STEM}_1, ...; the case already has {taken[0]!r}")
    assign = list(d.get("assign") or [])
    listed = {a.get("region") for a in assign}
    same: Dict[str, str] = {}
    for region, sec, orient in made:
        if region in listed:
            raise ValueError(f"`offset_arms`: {region!r} also has an `assign` of its own")
        key = repr(sorted(sec.items()))
        name = same.get(key)
        if name is None:
            name = f"{STEM}_{len(same) + 1}"
            same[key] = name
            sections[name] = sec
        assign.append({"region": region, "section": name, "orientation": orient})
    d["sections"] = sections
    d["assign"] = assign
    return d
