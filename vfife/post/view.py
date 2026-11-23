"""`vfife view`: a result file into one self-contained web page that replays
it in three dimensions.

    python -m vfife view results/case_34_brittle_cantilever.h5
    -> results/case_34_brittle_cantilever.html

WHY A WEB PAGE. Until now a result had one way out, ParaView, which is the
right tool for a mesh of a million cells and the wrong one for showing a
collaborator what a bridge did: it has to be installed, the XDMF has to
find its HDF5, and nothing about a collapse -- what broke when, where the
debris went -- is visible without building a pipeline. A page is a file
that opens anywhere and can be sent as a link; the D1 demo of the nsfc
project (a 1488 m bridge, three acts) was such a page, hand-built. This is
the same page for any result: line members, shells, solids, coupled models
and their reinforcement, every scalar field the run wrote, the monitors,
the breaks, a time axis.

WHAT GOES INTO THE PAGE. The mesh; the displacement of every frame,
quantised to 16 bits over its own range (a 32000-level grid, a fraction of
a millimetre on a bridge, invisible); every nodal and element scalar
field, quantised to 8 bits over its range across all frames (the page
colours by it, and 256 colours are more than a contour plot shows); the
monitors; `diagnostics/member_failures`. All of it as one base64 block in
the HTML, so the page is a single file with no data beside it. Three.js is
loaded from a CDN, which is the one thing the page needs a network for.

WHAT IS DRAWN. Line members as boxes of one thickness (`--thickness`,
default 1/300 of the model's diagonal) whose axis follows the two nodes;
a member the run marked `member_failed` is drawn grey, or hidden. Shell
elements as their faces, solids as their exterior faces (interior faces
are found once, by sorting, as `core/mesh` does for pressures), both with
the element's colour flat or the nodes' colour interpolated. A coupled
model draws both its solid and its members; a `rebar` group is drawn as
thin lines.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..io.writer import NODE_FIELDS

from ..core.mesh import _FACETS
from ..io.writer import open_result

#: how many nodes each XDMF mixed-topology code carries, for a `blocks` mesh
_MIXED_NODES = {1: 1, 2: 2, 4: 3, 5: 4, 6: 4, 7: 5, 8: 6, 9: 8, 38: 10, 40: 15, 41: 18, 48: 20, 50: 27}
_MIXED_KIND = {2: "line", 4: "shell3", 5: "shell4", 6: "tet4", 7: "pyramid5", 8: "wedge6", 9: "hexa8",
               38: "tet10", 40: "wedge15", 48: "hexa20"}
_LINE_FAMILIES = {"truss2", "beam2", "beam2_timoshenko", "beam2_fiber", "connector2", "mixed_line"}


def _exterior_faces(conn: np.ndarray, kind: str):
    """(faces (m, 3 or 4) of corner node ids, owner element) of the faces
    that belong to one element only."""
    out = []
    for local in _FACETS[kind]:
        corners = local[:3] if len(local) in (3, 6) else local[:4]
        out.append((conn[:, corners], np.arange(len(conn))))
    by_n: Dict[int, list] = {}
    for f, o in out:
        by_n.setdefault(f.shape[1], []).append((f, o))
    result = []
    for n, groups in by_n.items():
        faces = np.vstack([g[0] for g in groups]); owner = np.concatenate([g[1] for g in groups])
        key = np.sort(faces, axis=1)
        order = np.lexsort(key.T[::-1]); sk = key[order]
        change = np.r_[True, np.any(sk[1:] != sk[:-1], axis=1)]
        starts = np.flatnonzero(change); ends = np.r_[starts[1:], len(order)]
        ext = order[starts[ends - starts == 1]]
        result.append((faces[ext], owner[ext]))
    return result


def _blocks_from_mixed(flat: np.ndarray):
    """Split an XDMF mixed connectivity stream into (kind, conn, first element index)."""
    blocks, i, e = [], 0, 0
    cur_kind, cur_rows, cur_first = None, [], 0
    while i < len(flat):
        code = int(flat[i]); n = _MIXED_NODES[code]; kind = _MIXED_KIND.get(code, f"code{code}")
        if kind != cur_kind:
            if cur_rows:
                blocks.append((cur_kind, np.array(cur_rows), cur_first))
            cur_kind, cur_rows, cur_first = kind, [], e
        cur_rows.append(flat[i + 1:i + 1 + n]); i += 1 + n; e += 1
    if cur_rows:
        blocks.append((cur_kind, np.array(cur_rows), cur_first))
    return blocks


def _section_size(sec: dict, E: float) -> Optional[tuple]:
    """(depth along local z, width along local y, 1 for a bar else 0) of one input-form section,
    from its shape where it has one; for `general` from A, I_y, I_z with the
    thin-walled formula h = sqrt(4 I / A) for steel (E > 1e11) and the solid
    one sqrt(12 I / A) otherwise -- a box girder and a concrete pier both
    come out near their true depth, which a picture is all that asks. E and
    not rho tells steel from concrete: a model's densities are smeared and
    zeroed at will (a massless rigid arm, a deck with its ballast)."""
    if not isinstance(sec, dict):
        return None
    # a result carries the case as it was written: both spellings
    h = sec.get("height") or sec.get("h")
    b = sec.get("width") or sec.get("b")
    shape = sec.get("shape") or ("rect" if h else None)
    d = sec.get("diameter") or sec.get("d")
    if sec.get("type") == "connector":
        return None
    bar = 1.0 if sec.get("type") == "bar" else 0.0
    if shape in ("circle", "tube") or (shape is None and d):
        return (float(d), float(d), bar) if d else None
    if shape in ("rect", "box") and h and b:
        return (float(h), float(b), bar)
    if shape in ("box_girder", "t_girder") and h and sec.get("top_width"):     # 1.2
        return (float(h), float(sec["top_width"]), bar)
    if shape == "I" and sec.get("hw") and sec.get("bf"):
        return (float(sec["hw"]) + 2.0 * float(sec.get("tf") or 0.0), float(sec["bf"]), bar)
    A = sec.get("area")
    if not A:
        return None
    A = float(A)
    if sec.get("type") == "bar" or not sec.get("iy"):
        r = 2.0 * np.sqrt(A / np.pi)
        return (r, r, 1.0)
    k = 4.0 if E > 1.0e11 else 12.0
    return (float(np.sqrt(k * float(sec["iy"]) / A)), float(np.sqrt(k * float(sec.get("iz") or sec["iy"]) / A)), 0.0)


def _drawn_count(mesh) -> int:
    """How many elements a result of this mesh holds in `mesh/elements`:
    every block of a mixed VOLUME mesh (hexa8 + wedge6), and the solid or
    shell alone beside Line2 members (coupled, shell_line -- those go to the
    result's `members`). `mesh.n_elem` is the largest block only, and a
    mixed mesh's own case was refused as another model."""
    from ..elements.library import families
    fam = families()
    blocks = getattr(mesh, "cell_blocks", None) or {}
    bodies = [len(v) for k, v in blocks.items() if fam.get(k) != "line"]
    return int(sum(bodies)) if len(bodies) > 1 else int(mesh.n_elem)


def member_sizes(case_path: str, n_elements: int, hide=None):
    """(sizes, hidden): the (nel, 3) drawn depth and width of every line
    member of the case and whether it is a bar (NaN where unknown), the
    sizes capped at half the member's length, and the elements of the `hide` regions (rigid arms, links --
    modelling devices a picture is better without). A case that cannot be
    prepared, or is not the result's model, is refused, and so is a `hide`
    region with no members: all three were ignored in silence, and the page
    drawn without what was asked (found in review)."""
    import yaml
    from ..api import prepare_case
    from ..core.diagnostics import UserError
    try:
        with open(case_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        prep = prepare_case(case_path)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise UserError("VIEW_CASE", f"--case {case_path}: {exc}",
                        "give the case file the result was computed from, or leave --case out")
    mesh = prep.mesh
    n_case = _drawn_count(mesh)
    if n_case != n_elements:
        raise UserError("VIEW_CASE_MISMATCH",
                        f"--case {case_path} has {n_case} elements and the result "
                        f"{n_elements}: it is not the model of this result",
                        "give the case file the result was computed from")
    if mesh.elements.ndim != 2 or mesh.elements.shape[1] != 2:
        if hide and prep.element in ("coupled", "shell_line"):
            # it said "this model has none" of a model with truss or beam
            # members beside its solid or shell
            raise UserError("VIEW_HIDE", f"--hide is not supported for a {prep.element} model "
                            "(line members beside a solid or a shell)",
                            "leave --hide out; --hide works on models of line members only")
        if hide:
            raise UserError("VIEW_HIDE", "--hide hides line members; this model has none",
                            "leave --hide out")
        return None, None
    sections = raw.get("sections") or {}
    materials = raw.get("materials") or {}
    regions = raw.get("regions") or {}
    def E_of(sec):
        m = materials.get(sec.get("material")) if isinstance(sec, dict) else None
        return float((m or {}).get("E") or getattr(prep.material, "E", 0.0) or 0.0)
    def ids_of(region):
        if region == "all":
            return np.arange(n_elements)
        if region not in mesh.regions and isinstance(regions.get(region), dict):
            # a region the case defines (a box, a plane), not a Gmsh group
            try:
                mesh.resolve_region(region, regions[region])
            except Exception:
                return None
        ids = mesh.region_elements(region, "truss2", mesh.elements)
        return None if ids is None else np.asarray(ids, int)
    out = np.full((n_elements, 3), np.nan)
    for a in raw.get("assign") or []:
        sec = sections.get(a.get("section"))
        size = _section_size(sec, E_of(sec))
        ids = ids_of(a.get("region", "all"))
        if size is not None and ids is not None and len(ids):
            out[ids] = size
    length = np.linalg.norm(mesh.nodes[mesh.elements[:, 1]] - mesh.nodes[mesh.elements[:, 0]], axis=1)
    out[:, :2] = np.minimum(out[:, :2], 0.5 * length[:, None])
    hidden = []
    for region in hide or []:
        ids = ids_of(region)
        if ids is None or not len(ids):
            have = sorted(set(mesh.regions) | set(regions))
            raise UserError("VIEW_HIDE",
                            f"--hide {region!r}: " + ("no such region" if ids is None
                                                      else "the region holds no members")
                            + f" in {case_path}",
                            "regions of the case: " + ", ".join(have))
        hidden.extend(int(i) for i in ids)
    return (None if np.isnan(out).all() else out), sorted(set(hidden))


def member_profiles(case_path: str, n_elements: int, profiles: Dict[str, Any]):
    """(specs, profile_of): the cross-section OUTLINES to draw instead of a
    box, for the members of the named regions of the case -- a drawing
    option, for a page that should show a model's real sections (the D1
    demo's streamlined boxes and tower legs); nothing reads them but the
    page. `profiles` maps a region to

        {"outline": [[y, z], ...],     # metres, section-local: y across the
                                       # member's width, z along its depth
         "outboard": true}             # optional: +y points away from the
                                       # model's centreline (a fairing on
                                       # the outer edge of an edge girder)

    with depth and width as the page lays a member: depth as vertical as
    the axis allows (a deck box lies flat), along the bridge for a near-
    vertical member (a tower leg). Returns ([spec, ...], [index or -1 per
    element]); a region the case does not have is an error."""
    from ..api import prepare_case
    mesh = prepare_case(case_path).mesh
    if _drawn_count(mesh) != n_elements:
        raise ValueError(f"profiles: {case_path} does not match the result ({_drawn_count(mesh)} vs {n_elements} elements)")
    specs, of = [], np.full(n_elements, -1, int)
    for region, spec in profiles.items():
        pts = np.asarray(spec.get("outline"), float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
            raise ValueError(f"profiles[{region!r}]: 'outline' must be three or more [y, z] points")
        ids = mesh.region_elements(region, "truss2", mesh.elements)
        if ids is None or not len(ids):
            raise ValueError(f"profiles: no such region {region!r} in {case_path}")
        of[np.asarray(ids, int)] = len(specs)
        specs.append({"outline": pts.round(4).tolist(), "outboard": bool(spec.get("outboard", False))})
    return specs, of.tolist()


def _quantise16(a: np.ndarray):
    scale = float(np.abs(a).max()) / 32000.0 if a.size and np.abs(a).max() > 0 else 1.0
    return np.round(a / scale).astype(np.int16), scale


def _quantise8(a: np.ndarray, robust: bool = False, symmetric: bool = False):
    """0..254 over the field's range; 255 is "no value" (NaN: a member that
    has no capacity criterion, a field a solver did not evaluate), which
    the page draws in the neutral colour rather than at the bottom of the
    scale. `robust` takes the range from the 0.5th to the 99.5th percentile
    and clips outside it: a member force spans the towers' 1e9 and the
    hangers' 1e6, and a scale set by the towers paints every girder the
    bottom colour (the D1 typhoon act's bending_y)."""
    finite = np.isfinite(a)
    if robust and finite.sum() > 100:
        lo, hi = (float(np.percentile(a[finite], 0.5)), float(np.percentile(a[finite], 99.5)))
        if hi <= lo:
            lo, hi = float(a[finite].min()), float(a[finite].max())
    else:
        lo, hi = (float(a[finite].min()), float(a[finite].max())) if finite.any() else (0.0, 0.0)
    if symmetric and lo < 0.0 < hi:
        # a SIGNED member force: zero in the middle of the
        # scale, so tension and compression (sagging and hogging) read as
        # the two halves of it
        hi = max(-lo, hi); lo = -hi
    span = hi - lo if hi > lo else 1.0
    q = np.round((np.nan_to_num(a, nan=lo) - lo) / span * 254.0).clip(0, 254).astype(np.uint8)
    q[~finite] = 255
    return q, lo, hi


def _common_particles(paths) -> Optional[np.ndarray]:
    """The particle table several results of ONE model are drawn on.

    A run that can split reserves spare particles after the mesh's own nodes
    (two per breakable member end), so the dead-load state (50 nodes), the
    hanger cut (68) and the collapse with deck hinges (148) of one bridge had
    three node counts and were refused as "a different mesh".
    What has to agree is the mesh AS READ: the same members on the same
    nodes -- every node a member names as read in the same place in every
    table. The spares after them may be parked in a different order (they
    are drawn on the longest table, _load_one carrying the difference).
    Returns the longest table, or None when they all have the same length."""
    tables = []
    for p in paths:
        with open_result(p, "r") as f:
            if "mesh" not in f:
                return None                   # _load_one says what it is
            tables.append((p, np.asarray(f["mesh/nodes"], float), np.asarray(f["mesh/elements"])))
    if len({len(n) for _, n, _ in tables}) == 1:
        return None
    longest = max(tables, key=lambda t: len(t[1]))
    scale = max(float(np.abs(longest[1]).max()), 1.0)
    for p, nodes, elements in tables:
        named = int(np.max(elements)) + 1 if elements.size else 0      # the mesh's own nodes
        if elements.shape != longest[2].shape or not np.array_equal(elements, longest[2]) \
                or named > len(nodes) \
                or not np.allclose(nodes[:named], longest[1][:named], rtol=0.0, atol=1e-9 * scale):
            raise ValueError(f"{p}: a different mesh from {longest[0]} (the members or the nodes as "
                             "read differ, not only the spare particles a split reserves) -- one page "
                             "holds runs of ONE model")
    return longest[1]


def load(paths, every: int = 1, max_frames: int = 400, case: Optional[str] = None,
         names: Optional[List[str]] = None, hide: Optional[List[str]] = None,
         fields: Optional[List[str]] = None, profiles: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Everything the page needs: the geometry of the first result and one
    ACT per result file (a page can replay several runs of one model --
    the three acts of the D1 demo -- and switch between them)."""
    from ..core.diagnostics import UserError
    paths = [paths] if isinstance(paths, str) else list(paths)
    # a name, a field or a frame step that does not apply was dropped in
    # silence (found in review)
    if names is not None and len(names) != len(paths):
        raise UserError("VIEW_NAMES", f"--names gives {len(names)} name(s) for {len(paths)} "
                        "result(s)", "one name per result, in the order the results are given")
    if int(every) < 1:
        raise UserError("VIEW_EVERY", f"--every {every}: take every k-th frame, k >= 1")
    # a GROUP name (`end_actions`, as `output.fields` writes it) stands for
    # the components the result holds
    from ..elements.library import FIELD_GROUPS
    groups = {f: list(FIELD_GROUPS[f]) for f in (fields or []) if f in FIELD_GROUPS}
    keep = None if fields is None else list(dict.fromkeys(
        x for f in fields for x in [f] + groups.get(f, [])))
    acts, blobs, offset, geometry, available, held = [], [], 0, None, set(), set()
    ref_nodes = _common_particles(paths) if len(paths) > 1 else None
    for i, path in enumerate(paths):
        one = _load_one(path, every, max_frames, keep=keep, ref_nodes=ref_nodes)
        available |= one["available"]
        held |= one["held"]
        if geometry is None:
            geometry = one["geometry"]
        elif one["geometry"]["nodes"].shape != geometry["nodes"].shape or \
                one["geometry"]["n_elements"] != geometry["n_elements"]:
            raise ValueError(f"{path}: a different mesh from {paths[0]} -- one page holds runs of ONE model")
        a = one["act"]
        a["name"] = names[i] if names and i < len(names) and names[i] else a["name"]
        a["offset"] = offset
        for fld in a["fields"]:
            if "offset" in fld:
                fld["offset"] += offset
        if a.get("nodes"):
            a["nodes"]["offset"] += offset
        acts.append(a); blobs.append(one["blob"]); offset += len(one["blob"])
    if fields is not None:
        derived = {"displacement", "u_x", "u_y", "u_z", "|u|"}
        drawable = available | derived
        unknown = [f for f in fields if f not in drawable
                   and not (f in groups and drawable.intersection(groups[f]))]
        # held by the result but not drawable (a rotation, a fibre table):
        # "the result does not hold" it was false
        undrawn = [f for f in unknown if f in held]
        missing = [f for f in unknown if f not in held]
        if undrawn:
            raise UserError("VIEW_FIELDS", "--fields names " + ", ".join(map(repr, undrawn))
                            + ", which the result holds but `vfife view` does not draw: it draws "
                            "one value per node or element (a vector by its magnitude)",
                            "fields it can draw from this result: "
                            + (", ".join(sorted(available)) or "none"))
        if missing:
            raise UserError("VIEW_FIELDS", "--fields names " + ", ".join(map(repr, missing))
                            + ", which the result does not hold",
                            "fields it can draw from this result: "
                            + (", ".join(sorted(available)) or "none"))
    lo = geometry["nodes"].min(axis=0); hi = geometry["nodes"].max(axis=0)
    sizes, hidden = member_sizes(case, geometry["n_elements"], hide) if case else (None, None)
    if hide and not case:
        raise UserError("VIEW_HIDE", "--hide names regions of the case: give --case as well")
    if profiles and not case:
        raise ValueError("profiles name regions of the case: give the case as well")
    prof, prof_of = member_profiles(case, geometry["n_elements"], profiles) if profiles else (None, None)
    meta = {
        "name": acts[0]["name"] if len(acts) == 1 else os.path.splitext(os.path.basename(paths[0]))[0].rsplit("_", 1)[0],
        "family": geometry["family"], "nodes": geometry["nodes"].round(6).tolist(),
        "parts": [{"kind": p["kind"], "conn": np.asarray(p["conn"]).astype(int).tolist(),
                   "owner": np.asarray(p["owner"]).astype(int).tolist(), "group": p["group"]} for p in geometry["parts"]],
        "rebar": geometry["rebar"],
        "sizes": None if sizes is None else np.where(np.isnan(sizes), -1.0, sizes).round(4).tolist(),
        "hidden": hidden or [],
        "profiles": prof, "profile_of": prof_of,
        "bbox": [lo.tolist(), hi.tolist()], "n_elements": int(geometry["n_elements"]),
        "units": _units(paths[0]),
        "acts": acts,
    }
    return {"meta": meta, "blob": b"".join(blobs)}


def _units(path: str) -> Dict[str, str]:
    """The case's units (the axis labels say them); metres when the result
    does not carry its case."""
    try:
        import yaml
        with open_result(path, "r") as f:
            text = f.attrs.get("case_yaml")
        units = (yaml.safe_load(text) or {}).get("units") if text else None
        return {"length": str(units.get("length", "m"))} if isinstance(units, dict) else {"length": "m"}
    except Exception:
        return {"length": "m"}


def _mode(case_yaml: str, f) -> str:
    """The run's `solver.mode` (dynamic, relaxation, static): from the case
    the result carries, else its diagnostics; "" when neither says."""
    try:
        import yaml
        solver = (yaml.safe_load(case_yaml) or {}).get("solver") or {}
        if isinstance(solver, dict) and solver.get("mode"):
            return str(solver["mode"])
    except Exception:
        pass
    d = f["diagnostics"].attrs.get("mode", "") if "diagnostics" in f else ""
    return d.decode() if isinstance(d, bytes) else str(d)


def _load_one(path: str, every: int, max_frames: int, keep=None, ref_nodes=None) -> Dict[str, Any]:
    """One result file: its geometry and its act (frames, fields, monitors, breaks).

    `ref_nodes`: the particle table of the act with the most particles, when
    a page holds several runs of one model that reserved different numbers
    of spare particles for splitting; this file's table is
    its prefix and is drawn on it."""
    with open_result(path, "r") as f:
        if "mesh" not in f or "steps" not in f:
            what = "a modal result (frequencies and shapes; `vfife modal --paraview` writes those)" \
                if "shape" in f and "frequency" in f else "not a vfife result file"
            raise ValueError(f"{path}: {what} -- `vfife view` replays the frames of a run")
        family = str(f.attrs.get("element_family", ""))
        nodes = np.asarray(f["mesh/nodes"], float)
        # drawn on the common table: a particle whose reference position
        # differs there (a spare parked elsewhere) carries the difference in
        # its displacement, so the positions are this run's own
        shift = np.zeros_like(nodes)
        if ref_nodes is not None and len(ref_nodes) >= len(nodes):
            shift = nodes - np.asarray(ref_nodes, float)[:len(nodes)]
            nodes = np.asarray(ref_nodes, float)
        nn = len(nodes)
        elements = np.asarray(f["mesh/elements"])
        name = os.path.splitext(os.path.basename(path))[0]
        case_yaml = str(f.attrs.get("case_yaml", ""))
        for line in case_yaml.splitlines():
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip(); break
        # ------------------------------------------------ the drawn parts
        parts: List[Dict[str, Any]] = []          # {kind: line|face, conn, owner, group}
        if "blocks" in f["mesh"].attrs:
            blocks = _blocks_from_mixed(elements.ravel())
        elif elements.ndim == 2 and elements.shape[1] == 2:
            blocks = [("line", elements, 0)]
        else:
            kind = family
            if family == "shell_line":
                kind = "shell4"
            elif family == "coupled" or family not in _FACETS and family not in ("shell3", "shell4"):
                # the continuum of a coupled model: named by its width. It was
                # passed as "coupled", which draws nothing
                kind = {4: "tet4", 5: "pyramid5", 6: "wedge6", 8: "hexa8", 10: "tet10",
                        15: "wedge15", 20: "hexa20"}.get(elements.shape[1], family)
            blocks = [(kind, elements, 0)]
        for kind, conn, first in blocks:
            if kind == "line":
                parts.append({"kind": "line", "conn": conn[:, :2], "owner": np.arange(len(conn)) + first, "group": "elements"})
            elif kind in ("shell4", "shell3"):
                parts.append({"kind": "face", "conn": conn[:, :4 if kind == "shell4" else 3],
                              "owner": np.arange(len(conn)) + first, "group": "elements"})
            elif kind in _FACETS:
                for faces, owner in _exterior_faces(conn, kind):
                    parts.append({"kind": "face", "conn": faces, "owner": owner + first, "group": "elements"})
        if "members" in f:
            mconn = np.asarray(f["members/elements"])
            parts.append({"kind": "line", "conn": mconn[:, :2], "owner": np.arange(len(mconn)), "group": "members"})
        rebar = None
        if "rebar" in f:
            rebar = {"nodes": np.asarray(f["rebar/nodes"], float).round(6).tolist(),
                     "conn": np.asarray(f["rebar/elements"])[:, :2].astype(int).tolist()}
        # ------------------------------------------------------- frames
        # by the step NUMBER, not the name: the names are padded to one width
        # today, and a reader that trusts that breaks the day it changes
        every_key = sorted(f["steps"].keys(), key=int)
        keys = every_key[::max(1, every)] if every > 1 else list(every_key)
        if len(keys) > max_frames:
            pick = np.unique(np.linspace(0, len(keys) - 1, max_frames).round().astype(int))
            keys = [keys[i] for i in pick]
        if keys and keys[-1] != every_key[-1]:
            keys.append(every_key[-1])
        times = [float(f["steps"][k].attrs["time"]) for k in keys]
        nf = len(keys)
        disp = np.zeros((nf, nn, 3), np.float32)
        nodal: Dict[str, np.ndarray] = {}
        elem: Dict[str, np.ndarray] = {}
        member: Dict[str, np.ndarray] = {}
        nel_total = sum(len(c) for k, c, _ in blocks)
        conn_frames = None                    # (nf, nel, 2) once members move onto other particles
        held = set()                          # every dataset a frame holds
        for i, k in enumerate(keys):
            g = f["steps"][k]
            held |= set(g)
            if "displacement" in g:
                d = np.asarray(g["displacement"])[:, :3]
                disp[i, :len(d)] = d + shift[:len(d)]
            if "connectivity" in g and np.asarray(g["connectivity"]).shape == (nel_total, 2):
                if conn_frames is None:
                    conn_frames = np.repeat(elements[None, :, :2].astype(np.int64), nf, axis=0)
                conn_frames[i] = np.asarray(g["connectivity"]).astype(np.int64)
            for fname in g:
                a = np.asarray(g[fname])
                if fname == "displacement":
                    continue          # its components are derived in the page from the positions
                # by the field, not by its length: with as many elements as
                # nodes an element field went onto the nodes
                per_node = fname in NODE_FIELDS or (len(a) == nn and nn != nel_total)
                if a.ndim == 1 and len(a) == nn and per_node and fname != "connectivity":
                    nodal.setdefault(fname, np.zeros((nf, nn), np.float32))[i] = a
                elif a.ndim == 2 and a.shape[0] == nn and a.shape[1] == 3 and fname != "rotation":
                    nodal.setdefault(f"|{fname}|", np.zeros((nf, nn), np.float32))[i] = np.linalg.norm(a, axis=1)
                elif a.ndim == 1 and len(a) == nel_total and fname != "connectivity":
                    if fname == "capacity_ratio":
                        # exactly zero is "no value": a member with no criterion
                        # (NaN since v0.5; results before it wrote 0) or one
                        # carrying exactly nothing -- neutral either way, not the
                        # bottom of the scale
                        a = np.where(a == 0.0, np.nan, a)
                    elem.setdefault(fname, np.zeros((nf, nel_total), np.float32))[i] = a
            if "member_steps" in f and k in f["member_steps"]:
                mg = f["member_steps"][k]
                held |= set(mg)
                for fname in mg:
                    a = np.asarray(mg[fname])
                    if a.ndim == 1 and fname != "connectivity":
                        member.setdefault(fname, np.zeros((nf, len(a)), np.float32))[i] = a
        # ------------------------------------------------------ monitors
        monitors: Dict[str, list] = {}
        if "monitors" in f:
            n_t = len(f["monitors"]["time"]) if "time" in f["monitors"] else None
            for m in f["monitors"]:
                a = np.asarray(f["monitors"][m], float)
                # the residual and the kinetic energy are sampled at the
                # progress checks, not at the frames: drawn against the frame
                # times they were squeezed into part of the axis
                if a.ndim == 1 and m not in ("residual", "kinetic_energy") and (
                        n_t is None or len(a) == n_t):
                    monitors[m] = a.tolist()
        failures = np.asarray(f["diagnostics/member_failures"], float).tolist() \
            if "diagnostics" in f and "member_failures" in f["diagnostics"] else []
        # THE AXIS IS A LOAD FACTOR, NOT A TIME, for a relaxation or static
        # run: it was labelled in seconds, and an arc-length path (negative,
        # going back) was drawn against it off the chart (found in review,
        # v1.0). Such an act is drawn against the record's index.
        mode = _mode(case_yaml, f)
        axis = "time" if mode in ("dynamic", "") else "load"
        frame_mon = None
        if axis == "load" and "time" in monitors:
            tm, j, frame_mon = monitors["time"], 0, []
            for t in times:                    # frames match records in order
                while j < len(tm) and abs(tm[j] - t) > 1e-9 * max(1.0, abs(t)):
                    j += 1
                if j == len(tm):
                    frame_mon = None
                    break
                frame_mon.append(j)
        # A TIME-DEPENDENT RELAXATION (creep, shrinkage, ageing) runs on the
        # concrete's age in days, which its `age` record holds: it was
        # labelled a load factor. Drawn against the record
        # index as a load factor is, labelled by the age.
        ages = None
        if axis == "load" and frame_mon is not None and "age" in monitors \
                and len(monitors["age"]) == len(monitors.get("time", ())):
            axis = "age"
            ages = [float(monitors["age"][j]) for j in frame_mon]
    # ------------------------------------------------------ pack
    # ONLY THE PARTICLES THAT ARE EVER USED. A model that can split reserves
    # two spare particles per breakable member, and a run where
    # nothing splits never touches them: the Xihoumen typhoon act is 810
    # particles of motion and 1762 parked spares, which were 70% of its page
    used = np.zeros(nn, bool)
    for kind, conn_b, _first in blocks:
        used[np.asarray(conn_b, np.int64).ravel()] = True
    if conn_frames is not None:
        used[np.unique(conn_frames)] = True
    used |= np.any(disp != 0.0, axis=(0, 2))
    node_ids = None
    if used.sum() < nn:
        node_ids = np.flatnonzero(used)
        q16, scale = _quantise16(disp[:, node_ids])
    else:
        q16, scale = _quantise16(disp)
    buffers = [q16.tobytes()]
    offset = q16.nbytes
    if node_ids is not None:
        nb = node_ids.astype(np.int32).tobytes()
        node_ids_at = offset
        buffers.append(nb); offset += len(nb)
    fields = []
    # member_failed is the event list read frame by frame: when the run
    # wrote its events the page rebuilds the field from them
    if failures and "member_failed" in elem:
        elem.pop("member_failed")
        failed_from_events = True
    else:
        failed_from_events = False
    # what --fields may name: every field this result holds (checked in
    # `load`, over all the acts)
    available = set(nodal) | set(elem) | set(member)
    if failed_from_events:
        available.add("member_failed")
    available |= {k.strip("|") for k in nodal}          # a vector by its own name too
    if keep is not None:
        for table in (nodal, elem, member):
            for fname in list(table):
                if fname not in keep and fname.strip("|") not in keep and fname != "member_failed":
                    table.pop(fname)
    # the displacement's components and magnitude: ranges only, the page
    # reads the values off the positions it already holds
    if np.any(disp):
        mag = np.linalg.norm(disp, axis=2)
        for k, (nm, arr) in enumerate((("u_x", disp[..., 0]), ("u_y", disp[..., 1]), ("u_z", disp[..., 2]), ("|u|", mag))):
            fields.append({"name": nm, "on": "nodes", "lo": float(arr.min()), "hi": float(arr.max()), "derived": k, "n": nn})
    def pack(group, table):
        nonlocal offset
        for fname, arr in table.items():
            finite = arr[np.isfinite(arr)]
            if finite.size == 0 or (finite.min() == finite.max() and finite.size == arr.size):
                # a field that never changes (nothing failed in a typhoon run)
                # is its value, not 300 frames of it
                fields.append({"name": fname, "on": group, "lo": float(finite.min()) if finite.size else 0.0,
                               "hi": float(finite.max()) if finite.size else 0.0, "n": arr.shape[1],
                               "constant": float(finite.min()) if finite.size else None})
                continue
            robust = fname not in ("capacity_ratio", "member_failed")
            from ..core.member_actions import ACTIONS as _signed_actions
            q, lo, hi = _quantise8(arr, robust=robust, symmetric=fname in _signed_actions)
            entry = {"name": fname, "on": group, "lo": lo, "hi": hi, "offset": offset, "n": arr.shape[1]}
            if robust:
                finite = arr[np.isfinite(arr)]
                entry["clipped"] = bool(finite.size and (finite.max() > hi or finite.min() < lo))
            fields.append(entry)
            buffers.append(q.tobytes()); offset += q.nbytes
    pack("nodes", nodal); pack("elements", elem); pack("members", member)
    conn = None
    if conn_frames is not None and (conn_frames != conn_frames[0]).any():
        # a member split off sits on other particles from then on:
        # the page needs every CHANGE of connectivity, [frame, element, a, b]
        # -- a member splits once, so this is a few hundred rows where the
        # connectivity of every frame was a megabyte and a half
        base = conn_frames[0]
        changes = []
        prev = base
        for i in range(1, nf):
            diff = np.flatnonzero((conn_frames[i] != prev).any(axis=1))
            changes.extend([int(i), int(e), int(conn_frames[i][e, 0]), int(conn_frames[i][e, 1])] for e in diff)
            prev = conn_frames[i]
        conn = {"base": base.astype(int).tolist() if (base != elements[:, :2]).any() else None,
                "changes": changes}
    geometry = {"family": family, "nodes": nodes, "parts": parts, "rebar": rebar, "n_elements": int(nel_total)}
    act = {"name": name, "frames": nf, "times": times, "scale": scale, "fields": fields,
           "mode": mode, "axis": axis, "frame_mon": frame_mon, "ages": ages,
           "monitors": monitors, "failures": failures, "conn": conn,
           "failed_from_events": failed_from_events,
           "nodes": None if node_ids is None else {"offset": node_ids_at, "n": int(len(node_ids))}}
    return {"geometry": geometry, "act": act, "blob": b"".join(buffers), "available": available,
            "held": held}


#: the two scripts the page loads, by the file names `--scripts DIR` looks for
#: (three.js r128: build/three.min.js and examples/js/controls/OrbitControls.js)
SCRIPTS = (("three.min.js", "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"),
           ("OrbitControls.js", "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"))


def inline_scripts(html: str, folder: str) -> str:
    """The page with three.js written into it from `folder` instead of
    loaded from the two CDNs (v1.16: where those hosts are not reachable
    the page was black). `folder` holds three.min.js and OrbitControls.js
    of three.js r128 (in the npm package: build/ and
    examples/js/controls/); either may also sit under those subfolders."""
    for (name, url), code in zip(SCRIPTS, read_scripts(folder)):
        tag = f'<script src="{url}"></script>'
        if tag not in html:
            raise ValueError(f"the page has no {tag}")
        html = html.replace(tag, "<script>" + code + "</script>")
    return html


#: what each script must contain to be the r128 build the page is written for:
#: another three.js fails in the browser, in silence (an ES-module
#: OrbitControls from examples/jsm/ is a SyntaxError at its `import`)
_SCRIPT_MARKS = {"three.min.js": (('const e="128"', "REVISION"), "three.js r128 (build/three.min.js)"),
                 "OrbitControls.js": (("THREE.OrbitControls",), "the classic (non-module) OrbitControls of r128")}


def read_scripts(folder: str) -> List[str]:
    """The two scripts' text from `folder`, checked to be three.js r128's,
    with every "</script" written "<\\/script" so it cannot end the tag."""
    out = []
    for name, _url in SCRIPTS:
        cands = [os.path.join(folder, name), os.path.join(folder, "build", name),
                 os.path.join(folder, "examples", "js", "controls", name)]
        path = next((c for c in cands if os.path.isfile(c)), None)
        if path is None:
            raise FileNotFoundError(f"--scripts {folder}: no {name} there (three.js r128; in the npm package "
                                    "it is build/three.min.js and examples/js/controls/OrbitControls.js)")
        code = open(path, encoding="utf-8").read()
        marks, what = _SCRIPT_MARKS[name]
        miss = [m_ for m_ in marks if m_ not in code]
        if miss:
            raise ValueError(f"--scripts {folder}: {path} is not {what} (it has no {miss[0]!r}); the page is "
                             "written for three.js r128, npm package three@0.128.0")
        out.append(code.replace("</script", "<\\/script"))
    return out


def render(data: Dict[str, Any], thickness: Optional[float] = None, title: Optional[str] = None) -> str:
    from .view_page import PAGE
    meta = dict(data["meta"])
    if thickness is not None:
        meta["thickness"] = float(thickness)
    if title:
        meta["name"] = title
    b64 = base64.b64encode(data["blob"]).decode("ascii")
    import html as _html
    # (a name is text: escaped for the HTML, and no "<" in the script -- v1.20)
    return (PAGE.replace("__TITLE__", _html.escape(str(meta["name"])))
                .replace("__META__", json.dumps(meta, separators=(",", ":")).replace("<", "\\u003c"))
                .replace("__BLOB__", b64))


def write(result, output: Optional[str] = None, every: int = 1, max_frames: int = 400,
          thickness: Optional[float] = None, title: Optional[str] = None,
          case: Optional[str] = None, names: Optional[List[str]] = None,
          hide: Optional[List[str]] = None, fields: Optional[List[str]] = None,
          profiles: Optional[Dict[str, Any]] = None, scripts: Optional[str] = None) -> str:
    """Write the page; return its path. `scripts`: a folder with three.js
    r128's three.min.js and OrbitControls.js, written into the page instead
    of loaded from the CDNs (`inline_scripts`)."""
    first = result if isinstance(result, str) else result[0]
    if scripts:
        read_scripts(scripts)                      # a wrong folder fails before the result is read
    out = output or os.path.splitext(first)[0] + ".html"
    data = load(result, every=every, max_frames=max_frames, case=case, names=names, hide=hide, fields=fields,
                profiles=profiles)
    html = render(data, thickness=thickness, title=title)
    if scripts:
        html = inline_scripts(html, scripts)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out
