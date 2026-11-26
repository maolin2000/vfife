"""`vfife view model.yaml`: a case, before it is run, as one web page.

    python -m vfife view bridge.yaml          -> bridge.model.html

A frame model of a few thousand members is checked by looking at it: which
members a stage builds, where the supports are and what they hold, where
the loads act, what a member's section is. The case says all of it, but in
a file of a thousand lines (or, before the high-level blocks, sixty
thousand). The page shows the model as the solver will read it -- the
high-level blocks expanded, the mesh with what `tendons:` adds, the
regions resolved as the solver resolves them -- step by step:

* the members built in the step, coloured by kind (beam, cable or bar,
  tendon, rigid arm, connector), section, build step or region; those not
  built yet (or removed) faint or hidden; those the step builds marked;
* the supports the step has, the connectors and the soil springs: one
  symbol each, with a code of what each degree of freedom or component is,
  on its own axes; the one clicked drawn dof by dof;
* the loads the step has (the model's, its own, those kept from earlier
  steps) as arrows: forces and moments at nodes, line loads at the members'
  middles; gravity and the others listed;
* with the switch "sections at size" (1.21), beams and bars as boxes of
  their sections' depth and width on their local axes, between their
  elastic ends (`end_offsets`), the rigid stretches as lines;
* a member clicked: its row, region, section and what it holds, material,
  the step that builds it and the one that removes it, its length.

Nothing is run: the case is read and its mesh built (`cli.build`), no
solver. A model the page cannot show (solids, shells) is refused.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

#: member kinds, in the order of the page's legend
KINDS = ("beam", "cable/bar", "tendon", "rigid arm", "connector")

#: a load's kinds the page draws as arrows, and the rest it lists
_ARROWS = ("vector", "moment", "line_load")
_OTHERS = ("gravity", "body_force", "traction", "pressure", "temperature", "moving_load", "vehicle",
           "ground_motion")


def _steps_of(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = list(d.get("steps") or [])
    return steps if steps else [{"name": "step 1"}]


def _rows(mesh, name: str, conn) -> np.ndarray:
    got = mesh.region_elements(name, "truss2", conn)
    return np.zeros(0, np.int64) if got is None else np.asarray(got, np.int64)


def _kind(sec: Dict[str, Any], region: str, tendon_regions: set, arm_material: Optional[str]) -> int:
    t = sec.get("type")
    if t == "connector":
        return 4
    if region in tendon_regions:
        return 2 if t == "bar" else 3
    if t == "bar":
        return 1
    if t == "beam" and (sec.get("material") in ("rigid", arm_material)):
        return 3
    return 0


def _section_text(name: str, sec: Dict[str, Any]) -> str:
    keys = [k for k in ("type", "material", "shape", "area", "iy", "iz", "j", "width", "height", "diameter",
                        "thickness") if k in sec]
    parts = []
    for k in keys:
        v = sec[k]
        parts.append(f"{k} {v:.6g}" if isinstance(v, float) else f"{k} {v}")
    return f"{name}: " + ", ".join(parts)


#: a connector component's drawing: the low 3 bits what carries it, bit 3 a damper beside it,
#: bit 4 one law with the other transverse component (uy and uz), bit 5 a coupled (6x6) matrix
SPRING, CURVE, GAP, SLIDE, DAMPER, PLANAR, COUPLED = 1, 2, 3, 4, 8, 16, 32
_COMPONENTS = ("ux", "uy", "uz", "rx", "ry", "rz")


def _connector(sec: Dict[str, Any]):
    """(codes (6,), words (6,)): what each of a connector's six components
    is, as the solver reads it: a stiffness (or a 6x6 matrix's diagonal), a
    curve, a law, a damping. A component with none of them is free."""
    from ..io.schema import _connector_matrix
    K = _connector_matrix(sec.get("stiffness"), "stiffness")
    C = _connector_matrix(sec.get("connector_damping"), "connector_damping")
    curves = sec.get("curves") or {}
    laws = sec.get("laws") or {}
    codes, words = [], []
    for i, c in enumerate(_COMPONENTS):
        code, w = 0, []
        law = laws.get(c)
        t = (law or {}).get("type")
        if c in curves:
            code = CURVE
            w.append(f"curve ({len(curves[c])} points)")
        elif t == "gap":
            code = GAP
            w.append("gap " + ", ".join(f"{k} {law[k]}" for k in ("lo", "hi", "stiffness") if law.get(k) is not None))
        elif t in ("friction", "pendulum"):
            code = SLIDE
            w.append(" ".join([t] + [f"{k} {law[k]}" for k in ("mu", "radius", "normal") if law.get(k) is not None]))
        elif t == "bilinear":
            code = CURVE
            w.append("bilinear " + ", ".join(f"{k} {law[k]}" for k in ("k1", "k2", "fy") if law.get(k) is not None))
        elif t == "material":
            # a law of the material table (1.21), named, or its spec once lowered
            code = CURVE
            m = law.get("material")
            name = m if isinstance(m, str) else (m or {}).get("type", "?")
            w.append(f"material {name}" + "".join(f", {k} {law[k]}" for k in ("area", "length")
                                                  if law.get(k) is not None))
        elif K[i, i] > 0.0:
            code = SPRING
            w.append(f"k {K[i, i]:.4g}")
        if t == "damper":
            code |= DAMPER
            if law.get("stiffness"):                # the spring a damper law has beside it (elements/connector.py)
                code |= SPRING
            w.append("damper " + ", ".join(f"{k} {law[k]}" for k in ("c", "exponent", "linear_below", "stiffness")
                                           if law.get(k) is not None))
        if C[i, i] > 0.0:
            code |= DAMPER
            w.append(f"c {C[i, i]:.4g}")
        codes.append(code)
        words.append("; ".join(w))
    # a friction, bilinear or pendulum law the same on uy and uz is ONE law in
    # the y-z plane, a circular yield surface (elements/connector.py): marked
    uy, uz = laws.get("uy"), laws.get("uz")
    if uy and uz and dict(uy) == dict(uz) and uy.get("type") in ("friction", "bilinear", "pendulum"):
        for i in (1, 2):
            codes[i] |= PLANAR
            words[i] += "; one law in the y-z plane (uy and uz)"
    for M, what in ((K, "stiffness"), (C, "connector_damping")):
        off = M - np.diag(np.diag(M))
        for i in np.flatnonzero(np.any(off != 0.0, axis=1)):
            codes[i] |= COUPLED
            words[i] += f"; coupled with another component (6x6 {what})"
    return codes, words


def _connector_axes(xi, xj, orientation, global_axes: bool) -> np.ndarray:
    """The connector's component axes as rows, as the solver takes them
    (core.mixed_structural_solver, beam_family._frame): x along the member,
    y the orientation made normal to it -- without one, the global axis
    most nearly normal to the member -- z = x × y. Global: the global axes."""
    if global_axes:
        return np.eye(3)
    from ..core.diagnostics import UserError
    ex = np.asarray(xj, float) - np.asarray(xi, float)
    L = float(np.linalg.norm(ex))
    if L <= 0.0:
        raise UserError("VIEW_MODEL", "a connector has zero length", "its two nodes must be apart")
    ex /= L
    if orientation is None:
        basis = np.eye(3)
        ref = basis[np.argmin(np.abs(basis @ ex))]
    else:
        ref = np.asarray(orientation, float)
    ey = ref - ex * float(ex @ ref)
    n = float(np.linalg.norm(ey))
    if n < 1e-10:
        raise UserError("VIEW_MODEL", f"a connector's orientation {list(ref)} is parallel to it",
                        "choose another vector for its `orientation`")
    ey /= n
    return np.vstack([ex, ey, np.cross(ex, ey)])


def _drawn_sections(d, assigns, sections, sec_names, member_sec, member_assign, kind, nodes, conn, mesh):
    """(ends (nel, 6), size (nel, 2), yz (nel, 6), members with offsets):
    for the page's sections at their size -- each member's elastic ends
    (its nodes moved by its `end_offsets`, as core/end_offsets puts them),
    its drawn depth (along local z) and width (along local y) as the result
    page takes them from the section (post/view._section_size; 0 where
    unknown, and for rigid arms and connectors, which are drawn as lines),
    capped at half its length, and its local y and z axes as the solver
    makes them (y the assignment's orientation made normal to the member,
    z = x cross y). Offsets the solver refuses are not drawn, and said.
    Nothing here is read back: it is a picture."""
    from types import SimpleNamespace
    from ..core import end_offsets
    from ..core.member_geometry import MemberGeometry
    from .view import _section_size
    nel = len(conn)
    materials = d.get("materials") or {}
    groups = []
    for ai, a in enumerate(assigns):
        if a.get("end_offsets") is not None:
            # (a connector is never an offset member: the fibre solver takes
            # connectors out of every section's rows whatever the order of
            # the assignments -- found in review)
            rows = np.flatnonzero((member_assign == ai) & (kind != 4))
            if len(rows):
                groups.append((rows, a["end_offsets"], a.get("orientation"), a.get("region")))
    note = ""
    try:
        geometry = MemberGeometry.build(SimpleNamespace(nodes=nodes, elements=conn), nel, groups)
    except ValueError as exc:
        # the page is where such a model is looked at: drawn without them, and said
        note = f"end offsets not drawn -- the solver refuses them: {exc}"
        geometry = MemberGeometry()
    off = geometry.offset
    ends = np.hstack([nodes[conn[:, 0]], nodes[conn[:, 1]]]).astype(np.float64)
    if off is not None:
        ends[:, :3] += off[:, 0]
        ends[:, 3:] += off[:, 1]
    size = np.zeros((nel, 2))
    yz = np.zeros((nel, 6))
    for e in range(nel):
        if member_sec[e] < 0 or kind[e] in (3, 4):
            continue
        sec = sections[sec_names[member_sec[e]]]
        mat = materials.get(sec.get("material")) if isinstance(sec, dict) else None
        got = _section_size(sec, float((mat or {}).get("E") or 0.0))
        if got is None:
            continue
        a = assigns[member_assign[e]] if member_assign[e] >= 0 else {}
        ref = a.get("orientation")
        xi, xj = ends[e, :3], ends[e, 3:]
        L = float(np.linalg.norm(xj - xi))
        if L <= 0.0:
            continue
        try:
            R = end_offsets._axes(xi, xj, ref)
        except ValueError:                           # (the solver refuses it; drawn about another axis)
            R = end_offsets._axes(xi, xj, [1.0, 0.0, 0.0])
        size[e] = np.minimum([got[0], got[1]], 0.5 * L)
        yz[e] = np.concatenate([R[1], R[2]])
    return ends, size, yz, len(geometry.members), note


def load(case_path: str) -> Dict[str, Any]:
    """The page's data: the model as the solver will read it, step by step."""
    from ..cli import build
    from ..core import selectors
    from ..core.diagnostics import UserError
    from ..io import schema
    from .. import prep

    case_path = os.path.abspath(case_path)
    case, raw = schema.load(case_path)
    mesh, _material, element = build(case, case_path)
    from .view import _LINE_FAMILIES
    if element not in _LINE_FAMILIES:
        raise UserError("VIEW_MODEL", f"{os.path.basename(case_path)} is a {element} model; the model page "
                        "shows frame models (beams, bars, cables, connectors)",
                        "run the case and view its result: `vfife view result.h5`")
    # the case in its input form, high-level blocks expanded: its sections,
    # assignments, steps, loads and constraints by the names the user wrote
    d = schema.safe_load_named(raw, case_path)
    arm_material = (d.get("offset_arms") or {}).get("material")     # (the expansion takes the block away)
    if prep.has_high_level(d):
        d, _overlay, _report = prep.expand_case(d, os.path.dirname(case_path))
    tendon_report = dict(case._tendon_report or {})
    tendon_of: Dict[str, str] = {}                  # a tendon's bars' and arms' regions -> its name
    for name, t in tendon_report.items():
        segs = list(t.get("segments") or ())
        for r, _p in segs:
            tendon_of[r] = name
        for j in range(len(segs) + 1):
            tendon_of[f"{name}/arm{j + 1}"] = name
    tendon_regions = set(tendon_of)

    conn = np.asarray((mesh.cell_blocks or {}).get("truss2", mesh.elements)[:, :2], np.int64)
    nel, nn = len(conn), len(mesh.nodes)
    nodes = np.asarray(mesh.nodes, np.float64)
    predefined = dict(mesh.regions)
    predefined.setdefault("all", np.arange(nn, dtype=np.int64))
    regions = selectors.resolve_all(mesh.nodes, d.get("regions") or {}, predefined)
    mesh.regions = dict(regions)                    # as the solver does: the case's regions and `all` hold members too

    def node_set(name):
        if name not in regions:
            raise UserError("VIEW_MODEL", f"region {name!r} is not a region of the model",
                            "the regions are the mesh's groups and the case's `regions:`")
        return np.asarray(regions[name], np.int64)

    def members_in(name):
        """The members with both nodes in the region's nodes (the solver's rule for line loads)."""
        inside = np.zeros(nn, bool)
        inside[node_set(name)] = True
        return np.flatnonzero(inside[conn[:, 0]] & inside[conn[:, 1]])

    # SECTIONS: each member's (the solver refuses two on one member; here the later is shown)
    sections = d.get("sections") or {}
    sec_names = list(sections)
    sec_index = {n: i for i, n in enumerate(sec_names)}
    member_sec = np.full(nel, -1, np.int64)
    member_reg = np.full(nel, -1, np.int64)
    member_assign = np.full(nel, -1, np.int64)
    reg_names: List[str] = []
    reg_rows: Dict[str, np.ndarray] = {}
    for ai, a in enumerate(d.get("assign") or []):
        region = a.get("region")
        if region is None or a.get("section") not in sec_index:
            continue
        rows = _rows(mesh, region, mesh.elements)
        if not len(rows):
            continue
        if region not in reg_names:
            reg_names.append(region)
            reg_rows[region] = np.unique(rows)
        if element == "beam2_fiber" and (sections[a["section"]] or {}).get("type") != "connector":
            # a fibre model's connectors are its connectors whatever comes
            # after (the fibre solver takes them out of every section's rows;
            # found in review)
            held = np.asarray([member_sec[r] >= 0 and (sections[sec_names[member_sec[r]]] or {}).get("type") == "connector"
                               for r in rows], bool)
            rows = rows[~held]
        member_sec[rows] = sec_index[a["section"]]
        member_reg[rows] = reg_names.index(region)
        member_assign[rows] = ai
    kind = np.zeros(nel, np.int64)
    for e in range(nel):
        if member_sec[e] >= 0:
            name = sec_names[member_sec[e]]
            sec = sections[name]
            kind[e] = 3 if (sec.get("type") == "beam" and name.startswith("offset_arm_")) else \
                _kind(sec, reg_names[member_reg[e]], tendon_regions, arm_material)

    assigns = d.get("assign") or []
    ends, size, yz, n_offset, offset_note = _drawn_sections(d, assigns, sections, sec_names, member_sec, member_assign,
                                               kind, nodes, conn, mesh)

    # CONNECTORS: each one's six components and the axes they are on
    cn_ids = np.flatnonzero(kind == 4)
    cn_codes = np.zeros((len(cn_ids), 6), np.uint8)
    cn_axes = np.zeros((len(cn_ids), 9), np.float32)
    cn_words: Dict[int, List[str]] = {}
    for c, e in enumerate(cn_ids.tolist()):
        si = int(member_sec[e])
        if si not in cn_words:
            cn_words[si] = list(_connector(sections[sec_names[si]]))
        cn_codes[c] = cn_words[si][0]
        a = assigns[member_assign[e]] if member_assign[e] >= 0 else {}
        sec = sections[sec_names[si]]
        glob = (a.get("coordinate_system") or sec.get("coordinate_system") or "local") == "global"
        cn_axes[c] = _connector_axes(nodes[conn[e, 0]], nodes[conn[e, 1]],
                                     a.get("orientation", sec.get("orientation")), glob).ravel()
    connector_words = {si: w[1] for si, w in cn_words.items()}

    # SOIL SPRINGS: the nodes each spec puts a spring on, as core.soil finds
    # them (the buried part of the region's members), and on which axes
    soil_nodes: List[int] = []
    soil_axes: List[int] = []
    soil_text: List[str] = []
    if d.get("soil_springs"):
        from ..core.soil import SoilSprings
        specs = [dict(sp, curves=[{"depth": 0.0, "p_y": [[0.0, 0.0], [1.0, 1.0]]}]) for sp in d["soil_springs"]]

        def members_of(ids):
            inside = np.zeros(nn, bool)
            inside[np.asarray(ids, np.int64)] = True
            return np.flatnonzero(inside[conn[:, 0]] & inside[conn[:, 1]])
        try:
            groups = SoilSprings(specs, nodes, conn, node_set, members_of).groups
        except ValueError as exc:
            raise UserError("VIEW_MODEL", f"soil_springs: {exc}", "") from None
        for sp, (ids, axes, _g, _p) in zip(d["soil_springs"], groups):
            bits = sum(1 << int(a) for a in axes) | (8 if len(axes) == 2 else 0)
            soil_nodes += ids.tolist()
            soil_axes += [bits] * len(ids)
            soil_text.append(f"{sp['region']}: {len(ids)} nodes, "
                             + ("radial in the horizontal plane" if len(axes) == 2 else f"along {'xyz'[axes[0]]}"))

    # BUILT, step by step, as `core.analysis_steps.plan` has it: a member
    # whose first mention is an `activate` is not there before it, the others
    # stand from the start; each step builds, then removes
    steps = _steps_of(d)
    n_steps = len(steps)
    first = np.zeros(nel, np.int8)                  # 0 not named, 1 first activated, 2 first deactivated
    changes = []
    for k, st in enumerate(steps):
        got = {}
        for key in ("activate", "deactivate"):
            ids = [_rows(mesh, r, mesh.elements) for r in st.get(key) or []]
            arr = np.unique(np.concatenate(ids)) if ids else np.zeros(0, np.int64)
            fresh = arr[first[arr] == 0]
            first[fresh] = 1 if key == "activate" else 2
            got[key] = arr
        changes.append(got)
    built0 = first != 1
    built = np.zeros((nel, n_steps), bool)
    cur = built0.copy()
    for k, got in enumerate(changes):
        cur[got["activate"]] = True
        cur[got["deactivate"]] = False
        built[:, k] = cur
    before = np.column_stack([built0, built[:, :-1]]) if n_steps else built
    new = built & ~before
    gone = before & ~built
    born = np.where(built0, -1, np.where(new.any(axis=1), new.argmax(axis=1), n_steps))

    # SUPPORTS, per step (core.analysis_steps.constrained_items): the model's
    # and the kept ones, less the nodes released since each began; the
    # step's own. 6 bits for what is held, the 7th for a prescribed motion
    # anywhere on the node; and 6 bits for the held dofs that move: a nonzero
    # `value` (one for all the flagged dofs, or one each), a `motion` other
    # than displacement, a `record` (it drives the flagged translations)
    from ..io.schema import coerce_numbers

    def mask_of(c):
        dofs = list(c.get("dofs") or [])
        dofs = dofs + [0] * (6 - len(dofs))
        flagged = [i for i, v in enumerate(dofs[:6]) if v]
        m = sum(1 << i for i in flagged)
        if c.get("record") is not None:
            moving = [i for i in flagged if i < 3]
        elif c.get("motion", "displacement") != "displacement":
            moving = flagged
        else:
            value = c.get("value")
            try:
                value = coerce_numbers(value)
            except Exception:                       # (a value the schema took: drawn as moving)
                pass
            if isinstance(value, (list, tuple)):
                moving = [i for i in flagged if i < len(value) and not (isinstance(value[i], (int, float)) and value[i] == 0)]
            else:
                moving = flagged if value is not None and not (isinstance(value, (int, float)) and value == 0) else []
        mv = sum(1 << i for i in moving)
        return m | (64 if mv else 0), mv

    released = [np.unique(np.concatenate([node_set(r) for r in st.get("release")]))
                if st.get("release") else np.zeros(0, np.int64) for st in steps]
    held = np.zeros((nn, n_steps), np.uint8)
    moves = np.zeros((nn, n_steps), np.uint8)
    for k, st in enumerate(steps):
        items = [(c, -1) for c in d.get("constraints") or []]
        items += [(c, j) for j in range(k) for c in steps[j].get("constraints") or [] if c.get("keep")]
        for c, origin in items:
            ids = node_set(c.get("region", "all"))
            freed = [released[j] for j in range(origin + 1, k + 1) if len(released[j])]
            if freed:
                ids = np.setdiff1d(ids, np.concatenate(freed))
            m, mv = mask_of(c)
            held[ids, k] |= m
            moves[ids, k] |= mv
        for c in st.get("constraints") or []:
            m, mv = mask_of(c)
            held[node_set(c.get("region", "all")), k] |= m
            moves[node_set(c.get("region", "all")), k] |= mv
    sup_nodes = np.flatnonzero(held.any(axis=1))

    # LOADS, per step: the model's, the step's own and those kept before. The
    # arrows are summed at each node (or member) and step, and written once
    # for each run of steps they stay the same in (an opposite load that
    # takes one off nets to nothing)
    loads_of_step: List[List[Dict[str, Any]]] = []
    for k, st in enumerate(steps):
        lds = [dict(ld, _how="model") for ld in d.get("loads") or []]
        lds += [dict(ld, _how="kept from an earlier step") for j in range(k) for ld in steps[j].get("loads") or []
                if ld.get("keep")]
        lds += [dict(ld, _how="this step's") for ld in st.get("loads") or []]
        loads_of_step.append(lds)
    net: Dict[tuple, np.ndarray] = {}
    listed: List[List[str]] = [[] for _ in range(n_steps)]
    for k, lds in enumerate(loads_of_step):
        for ld in lds:
            region, how = ld.get("region", "all"), ld["_how"]
            if "table" in ld:
                listed[k].append(f"table {ld['table']} ({how})")
                continue
            drew = False
            for key, code in (("vector", 0), ("moment", 1)):
                if ld.get(key) is not None:
                    ids = node_set(region)
                    v = np.asarray(ld[key], float) * (1.0 if ld.get("per_node") else 1.0 / max(len(ids), 1))
                    for n in ids.tolist():
                        net.setdefault((code, n), np.zeros((n_steps, 3)))[k] += v
                    drew = True
                    listed[k].append(f"{'force' if code == 0 else 'moment'} {list(ld[key])} on {region} "
                                     f"({'each node' if ld.get('per_node') else 'in all'}, {how})")
            if ld.get("line_load") is not None:
                rows = members_in(region)
                rows = rows[built[rows, k]]
                local = ld.get("coordinate_system") == "local"
                if not local:
                    v = np.asarray(ld["line_load"], float)
                    for e in rows.tolist():
                        net.setdefault((2, e), np.zeros((n_steps, 3)))[k] += v
                drew = True
                listed[k].append(f"line load {list(ld['line_load'])} per length on {region} ({len(rows)} members"
                                 + (", member axes: not drawn" if local else "") + f", {how})")
            if not drew:
                what = next((key for key in _OTHERS if ld.get(key) is not None), "load")
                val = ld.get(what)
                text = f"{what} {val}" if isinstance(val, (list, int, float)) else what
                listed[k].append(f"{text} on {region} ({how})")
    mid = 0.5 * (nodes[conn[:, 0]] + nodes[conn[:, 1]])
    arrows: List[List[float]] = []                  # [from step, to step, kind, x, y, z, vx, vy, vz]
    for (code, i), vs in net.items():
        at = mid[i] if code == 2 else nodes[i]
        k = 0
        while k < n_steps:
            if not np.any(vs[k]):
                k += 1
                continue
            j = k
            while j + 1 < n_steps and np.array_equal(vs[j + 1], vs[k]):
                j += 1
            arrows.append([k, j, code, *(at - 0), *vs[k]])
            k = j + 1

    # each step in words
    step_text = []
    for k, st in enumerate(steps):
        step_text.append({"name": str(st.get("name") or f"step {k + 1}"),
                          "procedure": str(st.get("procedure") or ""),
                          "built": int(new[:, k].sum()), "removed": int(gone[:, k].sum()),
                          "standing": int(built[:, k].sum()),
                          "prescribed": int(np.sum(held[:, k] & 64 > 0)),
                          "loads": listed[k], "released": list(st.get("release") or [])})

    centre = 0.5 * (nodes.min(axis=0) + nodes.max(axis=0)) if nn else np.zeros(3)
    diag = float(np.linalg.norm(nodes.max(axis=0) - nodes.min(axis=0))) if nn else 1.0
    tendons = [{"name": name, "bars": [r for r, _p in t.get("segments") or ()],
                "step": int(t.get("step", 0)), "jacking": [float(v) for v in t.get("jacking") or ()]}
               for name, t in tendon_report.items()]
    meta = {"name": str(case.name or os.path.splitext(os.path.basename(case_path))[0]),
            "case": os.path.basename(case_path), "units": dict(case.units or {}),
            "n_nodes": int(nn), "n_members": int(nel), "diag": diag,
            "centre": [float(c) for c in centre], "kinds": list(KINDS),
            "sections": [_section_text(n, sections[n]) for n in sec_names], "section_names": sec_names,
            "connector_words": {str(k): v for k, v in connector_words.items()}, "soil": soil_text,
            "regions": reg_names, "steps": step_text, "tendons": tendons,
            "sized": bool(np.any(size[:, 0] > 0.0)), "offset_members": int(n_offset), "offset_note": offset_note,
            "tendon_of_region": {reg_names.index(r): n for r, n in tendon_of.items() if r in reg_names},
            "gravity": [[k, list(ld["gravity"])] for k, lds in enumerate(loads_of_step) for ld in lds
                        if ld.get("gravity") is not None][:1]}
    arr = np.asarray(arrows, np.float64).reshape(-1, 9)
    arr[:, 3:6] -= centre
    arrays = {
        "nodes": (nodes - centre).astype(np.float32),
        "conn": conn.astype(np.int32),
        "kind": kind.astype(np.int8),
        "sec": member_sec.astype(np.int32),
        "reg": member_reg.astype(np.int32),
        "built0": built0.astype(np.uint8),
        "built": built.astype(np.uint8).reshape(nel, n_steps),
        "born": born.astype(np.int32),
        "sup_nodes": sup_nodes.astype(np.int32),
        "sup_masks": held[sup_nodes].astype(np.uint8).reshape(len(sup_nodes), n_steps),
        "sup_moving": moves[sup_nodes].astype(np.uint8).reshape(len(sup_nodes), n_steps),
        "arrows": arr.astype(np.float32),
        "cn_ids": cn_ids.astype(np.int32),
        "cn_codes": cn_codes.reshape(-1),
        "cn_axes": cn_axes.reshape(-1),
        # each assigned region's members (a member in several regions is in each)
        "reg_offsets": np.cumsum([0] + [len(reg_rows[r]) for r in reg_names]).astype(np.int32),
        "reg_members": (np.concatenate([reg_rows[r] for r in reg_names]) if reg_names
                        else np.zeros(0, np.int64)).astype(np.int32),
        "soil_nodes": np.asarray(soil_nodes, np.int32),
        # the members drawn at their sections' size: each one's elastic ends
        # (its nodes moved by its end offsets), depth and width, local y and z
        "ends": (ends - np.tile(centre, 2)).astype(np.float32),
        "size": size.astype(np.float32),
        "yz": yz.astype(np.float32),
        "soil_axes": np.asarray(soil_axes, np.uint8),
    }
    return {"meta": meta, "arrays": arrays}


def pack(arrays: Dict[str, np.ndarray]):
    """(index {name: [dtype, offset, count, shape]}, one blob), each array 8-byte aligned."""
    index, parts, offset = {}, [], 0
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        b = a.tobytes()
        pad = (-len(b)) % 8
        index[name] = [a.dtype.name, offset, int(a.size), list(a.shape)]
        parts.append(b + b"\0" * pad)
        offset += len(b) + pad
    return index, b"".join(parts)


def render(data: Dict[str, Any], title: Optional[str] = None, thickness: Optional[float] = None) -> str:
    from .model_page import PAGE
    meta = dict(data["meta"])
    if title:
        meta["name"] = title
    if thickness is not None:
        meta["thickness"] = float(thickness)
    index, blob = pack(data["arrays"])
    meta["index"] = index
    import html as _html
    return (PAGE.replace("__TITLE__", _html.escape(meta["name"]))
                .replace("__META__", json.dumps(meta, separators=(",", ":"), ensure_ascii=False)
                         .replace("<", "\\u003c"))
                .replace("__BLOB__", base64.b64encode(blob).decode("ascii")))


def write(case_path: str, output: Optional[str] = None, title: Optional[str] = None,
          thickness: Optional[float] = None, scripts: Optional[str] = None) -> str:
    """Write the model page; return its path."""
    from .view import inline_scripts, read_scripts
    if scripts:
        read_scripts(scripts)
    out = output or os.path.splitext(case_path)[0] + ".model.html"
    html = render(load(case_path), title=title, thickness=thickness)
    if scripts:
        html = inline_scripts(html, scripts)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out
