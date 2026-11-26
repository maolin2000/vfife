"""What model a checkpoint belongs to, and refusing one that belongs to
another.

A checkpoint is a set of arrays, and before this stage nothing said which
model they were the state of. `--restart` laid them onto whatever the case
built: a mesh of the same node count but another geometry was accepted and
answered wrong, a 4 x 4 fibre state went into a 6 x 6 section, a dynamic
state's physical mass replaced a relaxation's fictitious one and the run
went non-finite, and a different node count failed inside numpy with
"could not broadcast". The checkpoint file now carries a fingerprint of the
model (`model_fingerprint`), and `check` compares it with the model about to
be restored, naming what differs.

Two purposes, two rules:

* a RESTART continues the same run, so everything must be the same: the
  element family, the mode, the node and element counts, the mesh, every
  material and every section, what may break, and the loads, constraints,
  masses, damping and units;
* an INITIAL STATE starts a new run from another's end, and may change the
  mode (relaxation -> dynamic), the loads, the material constants and what
  may break (`fracture`, `failure`) -- only
  what cannot be mapped is refused: the element family, the element count,
  the connectivity, the nodes (the saved ones must be this mesh's first ones:
  a relaxed dead state has no spare particles, the dynamic run that may break
  has), and any state array whose shape differs (a fibre layout, a material
  with another number of state variables).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional

import numpy as np

from ..core.diagnostics import UserError

#: arrays whose size is the run's business, not the model's
_VARIABLE_PREFIXES = ("loop", "vehicle", "creep_", "static_", "step_ages", "bond_", "relax_",
                      "member_contact_", "fiber_stress_", "appendage_")
_VARIABLE_NAMES = {"energy", "fragment_parent", "pending_split", "pending_splits", "park_check",
                   "splits", "fracture_events", "member_failure_events", "material_signature",
                   "parked"}
#: node-sized arrays an initial state may hold FEWER rows of: the spare
#: particles a breakable model reserves are parked and at rest
_PADDED = {"q", "v", "orient", "mass", "physical_mass", "weightless_mass"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canon(x: Any) -> str:
    return _sha(json.dumps(x, sort_keys=True, default=str).encode())[:16]


def _elements_bytes(mesh) -> bytes:
    blocks = getattr(mesh, "cell_blocks", None) or {}
    parts = []
    if blocks:
        for name in sorted(blocks):
            parts.append(name.encode() + np.ascontiguousarray(np.asarray(blocks[name]), dtype=np.int64).tobytes())
    else:
        parts.append(np.ascontiguousarray(np.asarray(mesh.elements), dtype=np.int64).tobytes())
    return b"|".join(parts)


def nodes_sha(nodes, n: Optional[int] = None) -> str:
    a = np.ascontiguousarray(np.asarray(nodes, dtype=np.float64))
    return _sha((a if n is None else a[:n]).tobytes())


#: keys of the lowered model that are a material law (its constants may
#: change for an initial state, not for a restart)
_MATERIAL_KEYS = {"material", "core_material", "laws", "curves", "stiffness", "connector_damping"}
#: per-region settings that are not the layout: what may break, and the
#: initial strains -- an initial state may add or change them (a relaxed dead
#: state has no `fracture`, the dynamic run that may break has), a restart not
_SETTING_KEYS = {"failure", "fracture", "prestress", "unstressed_length", "unstressed_lengths", "jack"}
#: keys naming a region: replaced by the nodes and elements the region holds,
#: so a renamed group is the same model
_REGION_KEYS = {"region", "regions", "fiber_region"}
#: keys that are only names
_NAME_KEYS = {"name"}


def _region_key(mesh, name):
    """A region by what it holds: its nodes and its recorded elements."""
    if not isinstance(name, str):
        return name
    nodes = (getattr(mesh, "regions", None) or {}).get(name)
    elems = (getattr(mesh, "element_regions", None) or {}).get(name)
    if nodes is None and not elems:
        return name                                    # "all", or not resolved
    parts = [np.ascontiguousarray(np.sort(np.asarray(nodes, dtype=np.int64))).tobytes()
             if nodes is not None else b""]
    for block in sorted(elems or {}):
        parts.append(block.encode() + np.ascontiguousarray(
            np.sort(np.asarray(elems[block], dtype=np.int64))).tobytes())
    return "@" + _sha(b"|".join(parts))[:16]


def _split(x, mesh, materials, settings, where, detail=None):
    """The layout part of a lowered-model value; its material laws go to
    `materials` and its settings to `settings`, each under `where` (and, in
    full, to `detail`, for the message that names what changed)."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in _NAME_KEYS:
                continue
            if k == "end_offsets" and v is None:       # 1.21: a member without them is as before
                continue
            if k in _MATERIAL_KEYS:
                if v not in (None, {}, []):
                    materials[f"{where}.{k}"] = _canon(_plain(v, mesh))
                    if detail is not None:
                        detail[f"{where}.{k}"] = _plain(v, mesh)
                continue
            if k in _SETTING_KEYS:
                if v not in (None, {}, [], False):
                    settings[f"{where}.{k}"] = _canon(_plain(v, mesh))
                continue
            if k in _REGION_KEYS:
                out[k] = ([_region_key(mesh, r) for r in v] if isinstance(v, list)
                          else _region_key(mesh, v))
                continue
            out[k] = _split(v, mesh, materials, settings, f"{where}.{k}", detail)
        return out
    if isinstance(x, (list, tuple)):
        return [_split(v, mesh, materials, settings, f"{where}[{i}]", detail) for i, v in enumerate(x)]
    return x


#: keys added in a 1.x release, with the default that means "as in 1.0": left
#: out of the fingerprint at that default, so a checkpoint written by 1.0
#: still resumes under 1.x (the frozen checkpoint format 1). A case that USES
#: the new key is a different model and fingerprints differently.
_STEP_CHANGE_KEYS = ("procedure", "total_time", "activate", "birth", "activate_age",
                     "deactivate", "release")
_ADDED_SINCE_1_0 = {"with_mass": False, "keep": False, "end_offsets": None}


def _without_none(d, key):
    """`d` without `key` where it is None (a key added since 1.0 whose
    absence is the 1.0 model)."""
    if isinstance(d, dict) and key in d and d[key] is None:
        return {k: v for k, v in d.items() if k != key}
    return d


def _plain(x, mesh):
    """A value with its regions by content and its file paths by name: a
    restart on another machine reads the same record from another folder."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in _NAME_KEYS:
                continue
            if k in _ADDED_SINCE_1_0 and v == _ADDED_SINCE_1_0[k]:
                continue
            if k in _REGION_KEYS:
                out[k] = [_region_key(mesh, r) for r in v] if isinstance(v, list) else _region_key(mesh, v)
            elif k in ("path", "profile") and isinstance(v, str):
                out[k] = os.path.basename(v.replace("\\", "/"))
            else:
                out[k] = _plain(v, mesh)
        return out
    if isinstance(x, (list, tuple)):
        return [_plain(v, mesh) for v in x]
    return x


def _label(entry, kind, k):
    if isinstance(entry, dict):
        r = entry.get("region") or entry.get("fiber_region") or entry.get("regions")
        if isinstance(r, list):
            r = ",".join(map(str, r))
        if r:
            return f"{kind} on {r}"
    return f"{kind} {k}"


def model_fingerprint(solver, prepared, static_run: bool = False) -> Dict[str, Any]:
    """What a checkpoint file carries about its model.

    Taken once, right after the solver is built and before anything runs: a
    split moves the connectivity, and the fingerprint is of the model, not of
    its state. It is taken from the LOWERED model -- the validated case with
    every section and material written into the members that carry them --
    so names do not matter, an unused section does not matter, and a case
    given as a file, a dict, a `CaseDraft` or a `schema.Case` gives the same
    fingerprint (v1.0: it hashed the YAML text, and a `schema.Case`
    had no materials or sections at all)."""
    from ..assemble import _with_element
    mesh = getattr(solver, "mesh", None) or prepared.mesh
    nodes = np.ascontiguousarray(np.asarray(mesh.nodes, dtype=np.float64))
    blocks = getattr(prepared.mesh, "cell_blocks", None) or {}
    n_elem = int(sum(len(b) for b in blocks.values())) if blocks else int(len(prepared.mesh.elements))
    d = _with_element(prepared.case, prepared.element, prepared.case_path)
    pm = prepared.mesh
    materials: Dict[str, str] = {}
    settings: Dict[str, str] = {}
    layout: Dict[str, str] = {}
    el = dict(d.get("element") or {})
    # the element block carries a single-section fibre model's section
    el_label = f"section on {el['fiber_region']}" if el.get("fiber_region") else "element"
    detail: Dict[str, Any] = {}          # the laws in full, for the message
    layout[el_label] = _canon(_split(el, pm, materials, settings, el_label, detail))
    if d.get("material") not in (None, {}):
        materials["material"] = _canon(_plain(d["material"], pm))
        detail["material"] = _plain(d["material"], pm)
    for kind, key in (("section", "members"), ("block", "blocks"), ("reinforcement", "reinforcement")):
        for k, entry in enumerate(d.get(key) or []):
            label = _label(entry, kind, k)
            layout[label] = _canon(_split(entry, pm, materials, settings, label, detail))
    if d.get("removals"):
        settings["removals"] = _canon(_plain(d["removals"], pm))
    sol = dict(d.get("solver") or {})
    conditions = {
        "units": _canon(d.get("units") or {}),
        "constraints": _canon(_plain(d.get("constraints") or [], pm)),
        "loads": _canon(_plain(d.get("loads") or [], pm)),
        "nodal_masses": _canon(_plain(d.get("nodal_masses") or [], pm)),
        "hydrodynamic_mass": _canon(_plain(d.get("hydrodynamic_mass") or [], pm)),
        "diaphragms": _canon(_plain(d.get("diaphragms") or [], pm)),
        "amplitudes": _canon(_plain(d.get("amplitudes") or {}, pm)),
        "contact": _canon(_plain(d.get("contact"), pm)),
        "steps": _canon(_plain([{k: (st or {}).get(k) for k in ("loads", "constraints")}
                                for st in (d.get("steps") or [])], pm)),
        # (`regions`, 1.22: a model without them is as before)
        "damping": _canon(_plain(_without_none(sol.get("damping"), "regions"), pm)),
        "mass": _canon([sol.get("mass"), _plain(sol.get("mass_scaling"), pm)]),
    }
    # added in 1.2, and only when used: a 1.0 or 1.1 checkpoint still fits
    if d.get("soil_springs"):
        conditions["soil_springs"] = _canon(_plain(d["soil_springs"], pm))
    # the analysis steps' model changes and procedures (1.2), only when used
    changes = [{k: (st or {}).get(k) for k in _STEP_CHANGE_KEYS
                if (st or {}).get(k) not in (None, [], "tangent")}
               for st in (d.get("steps") or [])]
    if any(changes):
        conditions["step_changes"] = _canon(_plain(changes, pm))
    layout_json = json.dumps(layout, sort_keys=True)
    materials_json = json.dumps(materials, sort_keys=True)
    return {
        "element_family": str(prepared.element),
        "mode": "static" if static_run else str(solver.staging.base_mode or getattr(solver, "mode", "")),
        "backend": str(getattr(solver, "backend", "numpy")),
        "n_nodes": int(len(nodes)),
        "n_elements": n_elem,
        "nodes_sha256": nodes_sha(nodes),
        "elements_sha256": _sha(_elements_bytes(prepared.mesh)),
        # the layout as a whole (members in any order) and by part, for the message
        "layout_sha256": _canon(sorted(layout.values())),
        "layout_json": layout_json,
        "materials_sha256": _canon(sorted(materials.values())),
        "materials_json": materials_json,
        "materials_detail_json": json.dumps(detail, sort_keys=True, default=str),
        "settings_sha256": _canon(sorted(settings.values())),
        "conditions_json": json.dumps(conditions, sort_keys=True),
        # the reference coordinates themselves, compared with a tolerance
        # (a generated mesh on another CPU may differ in the last bit); a
        # dataset `fingerprint/nodes` in the checkpoint file
        "nodes": nodes,
    }


def own_state(solver) -> Dict[str, np.ndarray]:
    """This solver's own checkpoint before anything is restored, for the
    shapes; empty when it cannot give one yet (the shape check is skipped)."""
    try:
        return solver.checkpoint()
    except Exception:
        return {}


def _named(saved: str, now: str):
    """What differs, down to the key that changed and its two values:
    "material changed: material" did not say that it was `creep.rh`.
    At most six entries."""
    try:
        a, b = json.loads(saved), json.loads(now)
    except Exception:
        return ["?"]
    out = []

    def short(v):
        s = json.dumps(v, sort_keys=True)
        return s if len(s) <= 40 else s[:37] + "..."

    def walk(x, y, path):
        if len(out) >= 6 or x == y:
            return
        if isinstance(x, str) and isinstance(y, str):
            # a canonical JSON string inside the record: compare what it holds
            try:
                x2, y2 = json.loads(x), json.loads(y)
            except Exception:
                x2 = y2 = None
            if isinstance(x2, (dict, list)) and isinstance(y2, (dict, list)):
                walk(x2, y2, path)
                return
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y), key=str):
                if k not in x or k not in y:
                    out.append(f"{path + '.' if path else ''}{k} "
                               + ("added" if k not in x else "removed"))
                else:
                    walk(x[k], y[k], f"{path + '.' if path else ''}{k}")
            return
        if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (p, q) in enumerate(zip(x, y)):
                walk(p, q, f"{path}[{i}]")
            return
        out.append(f"{path or '(value)'} ({short(x)} there, {short(y)} here)")

    walk(a, b, "")
    return out or ["?"]


def _nodes_differ(saved_nodes, nodes, n) -> bool:
    """The first `n` reference coordinates against the saved ones, to a
    billionth of the model's size."""
    a = np.asarray(saved_nodes, dtype=float)
    b = np.asarray(nodes, dtype=float)[:n]
    if a.shape != b.shape:
        return True
    size = float(np.ptp(b, axis=0).max()) if len(b) else 1.0
    return not np.allclose(a, b, rtol=0.0, atol=1e-9 * max(1.0, size))


def check(saved_meta: dict, saved: Dict[str, np.ndarray], now: dict, fresh: Dict[str, np.ndarray],
          purpose: str, path: str, nodes=None) -> None:
    """Refuse a checkpoint that is not of this model (see the module text).
    `fresh` is this solver's own checkpoint before the restore, for the
    shapes; `nodes` this model's nodes, for an initial state's prefix."""
    restart = purpose == "restart"
    what = "--restart" if restart else "solver.initial_state"
    nodes = now.get("nodes") if nodes is None else nodes
    diffs = []
    if saved_meta:
        for key, label in (("element_family", "element family"), ("n_elements", "element count")):
            if key in saved_meta and str(saved_meta[key]) != str(now[key]):
                diffs.append(f"{label}: {saved_meta[key]} there, {now[key]} here")
        if "elements_sha256" in saved_meta and saved_meta["elements_sha256"] != now["elements_sha256"] \
                and not any(d.startswith("element count") for d in diffs):
            diffs.append("connectivity: the elements join other nodes")
        n_saved = int(saved_meta.get("n_nodes", now["n_nodes"]))
        n_now = int(now["n_nodes"]) if nodes is None else len(nodes)
        if (n_saved != n_now) if restart else (n_saved > n_now):
            diffs.append(f"node count: {n_saved} there, {n_now} here")
        elif "nodes" in saved_meta and nodes is not None:
            if _nodes_differ(saved_meta["nodes"], nodes, n_saved):
                diffs.append("a different mesh: the nodes are at other coordinates")
        elif "nodes_sha256" in saved_meta and nodes is not None \
                and nodes_sha(nodes, n_saved) != saved_meta["nodes_sha256"]:
            diffs.append("a different mesh: the nodes are at other coordinates")
        # both: the layout (an element's type, a section's shape and size, a
        # fibre arrangement, which elements carry which section) cannot be
        # mapped; names, unused sections, materials and settings are not in it
        if "layout_sha256" in saved_meta and saved_meta["layout_sha256"] != now["layout_sha256"]:
            names = _named(saved_meta.get("layout_json", "{}"), now["layout_json"])
            diffs.append(f"section layout changed: {', '.join(names) or 'the sections moved between elements'}")
        if restart:
            if saved_meta.get("mode", now["mode"]) != now["mode"]:
                diffs.append(f"mode: {saved_meta['mode']} there, {now['mode']} here "
                             "(a restart continues the same analysis; start a new one with solver.initial_state)")
            if "materials_sha256" in saved_meta and saved_meta["materials_sha256"] != now["materials_sha256"]:
                if "materials_detail_json" in saved_meta and "materials_detail_json" in now:
                    names = _named(saved_meta["materials_detail_json"], now["materials_detail_json"])
                else:
                    names = _named(saved_meta.get("materials_json", "{}"), now["materials_json"])
                diffs.append(f"material changed: {', '.join(names) or 'a law moved between elements'}")
            if "settings_sha256" in saved_meta and saved_meta["settings_sha256"] != now["settings_sha256"]:
                diffs.append("a per-region setting changed (failure, fracture, prestress, removals ...)")
            if "conditions_json" in saved_meta and saved_meta["conditions_json"] != now["conditions_json"]:
                names = _named(saved_meta["conditions_json"], now["conditions_json"])
                diffs.append(f"changed: {', '.join(names)}")
    # every state array the model sizes must be the size this model gives it
    # (said only when the counts agree: otherwise every array differs)
    counted = any(d.startswith(("element count", "node count")) for d in diffs)
    n_pad = (int(now["n_nodes"]) - int(saved_meta.get("n_nodes", now["n_nodes"]))) if saved_meta else None
    for key, a in ({} if counted else saved).items():
        if key not in fresh or key in _VARIABLE_NAMES or key.startswith(_VARIABLE_PREFIXES):
            continue
        a, b = np.asarray(a), np.asarray(fresh[key])
        if a.dtype.kind in "SUO" or a.ndim == 0 or a.shape == b.shape:
            continue
        if not restart and a.ndim == b.ndim and a.shape[0] < b.shape[0] and a.shape[1:] == b.shape[1:] \
                and (key in _PADDED or _is_node_sized(a, b, saved_meta, now)):
            continue
        diffs.append(f"state `{key}`: shape {tuple(a.shape)} there, {tuple(b.shape)} here")
    if diffs:
        raise UserError(
            "CHECKPOINT_MISMATCH",
            f"{what} {path} holds the state of a different model: " + "; ".join(diffs[:6]),
            ("a restart continues the run that wrote the checkpoint with the same model, loads, "
             "constraints, masses, damping and units; only the solver's budget (n_steps, "
             "total_time), the backend and the output may change") if restart else
            ("an initial state must come from the same mesh, elements and section layout; "
             "the mode, the loads, the material constants and what may break may differ"))


def _is_node_sized(a, b, saved_meta, now) -> bool:
    """A per-node (or per-dof) array of the saved model, one row per node
    (or `stride` rows per node): the spare particles a breakable model
    reserves come after the mesh's nodes and are parked at rest."""
    if not saved_meta or "n_nodes" not in saved_meta:
        return False
    n0, n1 = int(saved_meta["n_nodes"]), int(now["n_nodes"])
    if n0 <= 0 or n1 <= n0:
        return False
    for stride in (1, 3, 6, 7):
        if a.shape[0] == stride * n0 and b.shape[0] == stride * n1:
            return True
    return False
