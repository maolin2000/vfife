"""The high-level blocks that need no mesh: `include:`,
`tendon_defaults:`, `table:` and `groups:`. Each is a rewriting of the case
dict into what it says; `vfife.prep.expand_case` runs them in that order,
before the tendons.

* `include: [lib/materials.yaml, ...]` merges the files' `materials`,
  `sections`, `reinforcement`, `groups` and `tendons` into the case. A name
  defined twice is refused; `{path: ..., override: true}` lets the case's
  own definition win over the file's.
* `tendon_defaults: {friction: ..., stress: {ends: both}, ...}` is what every
  tendon has unless it says otherwise: a key a tendon does not write is
  taken from here, and a mapping (its `stress`, `friction`, ...) is merged
  key by key, the tendon's own keys winning. `null` in a tendon is its own
  value (`relaxation: null`: it does not relax).
* `table: file.csv` stands for rows a CSV file holds: a tendon's profile
  (`profile_z: {table: t.csv}`, columns x, z, radius -- or x, y, radius;
  with a `name` column, the rows of the tendon's own name unless the entry
  says `name:`),
  nodal loads (`loads: [{table: l.csv}]`, columns region or x, y, z, then
  fx .. mz) and nodal masses (`nodal_masses: [{table: m.csv}]`, region or
  x, y, z, then mass). Paths are relative to the case file.
* `groups: {name: [region, ...]}` names a set of the mesh's groups (or of
  other groups): it becomes `mesh.groups`, and the mesh as read gets a group
  of that name holding the parts' elements and nodes -- a Gmsh physical group
  of several pieces, the same everywhere a region is named (an `assign`'s
  cable length is the whole group's). The steps' `activate`, `deactivate` and
  `release` lists are written out as the parts here already, so the tendons'
  expansion sees the mesh's names.
"""
from __future__ import annotations

import copy
import csv
import os
from typing import Any, Dict, List, Optional

#: what an included file may hold
INCLUDABLE = ("materials", "sections", "reinforcement", "groups", "tendons", "tendon_defaults")


def _base(base_dir: Optional[str]) -> str:
    return base_dir or os.getcwd()


# --------------------------------------------------------------- include

def _tables_from(defaults: Dict[str, Any], folder: Optional[str]) -> Dict[str, Any]:
    """`tendon_defaults` with the profile tables it names as absolute paths."""
    out = dict(defaults)
    for key in ("profile_y", "profile_z"):
        spec = out.get(key)
        if isinstance(spec, dict) and isinstance(spec.get("table"), str):
            out[key] = dict(spec, table=os.path.abspath(os.path.join(_base(folder), spec["table"])))
    return out


def include(d: Dict[str, Any], base_dir: Optional[str], _seen=None) -> Dict[str, Any]:
    """`d` with its `include:` files merged in (recursively)."""
    import yaml
    entries = d.get("include")
    if not entries:
        return d
    d = dict(d)
    d.pop("include")
    seen = set(_seen or ())
    if isinstance(entries, (str, dict)):
        entries = [entries]
    for entry in entries:
        if isinstance(entry, str):
            path, override = entry, False
        elif isinstance(entry, dict) and set(entry) <= {"path", "override"} and "path" in entry:
            path, override = entry["path"], bool(entry.get("override", False))
        else:
            raise ValueError(f"an `include` entry is a path or {{path, override}}, not {entry!r}")
        full = path if os.path.isabs(path) else os.path.join(_base(base_dir), path)
        full = os.path.abspath(full)
        if full in seen:
            raise ValueError(f"`include: {path}` includes itself (through {len(seen)} file(s))")
        if not os.path.isfile(full):
            raise ValueError(f"`include: {path}`: there is no such file ({full})")
        try:
            with open(full, "r", encoding="utf-8") as fh:
                part = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"`include: {path}` is not valid YAML: {exc}") from None
        if not isinstance(part, dict):
            raise ValueError(f"`include: {path}` is not a mapping of blocks")
        part = include(part, os.path.dirname(full), seen | {full})
        part = tables(part, os.path.dirname(full))
        # a table its defaults name is read for each tendon later, from the case: its path from this file
        if isinstance(part.get("tendon_defaults"), dict):
            part["tendon_defaults"] = _tables_from(part["tendon_defaults"], os.path.dirname(full))
        extra = sorted(k for k in part if k not in INCLUDABLE)
        if extra:
            raise ValueError(f"`include: {path}` holds {', '.join('`' + k + '`' for k in extra)}; an included "
                             f"file holds only {', '.join('`' + k + '`' for k in INCLUDABLE)} (the steps and the "
                             "solver stay in the case, where they are read)")
        for block in INCLUDABLE:
            if block not in part:
                continue
            mine = d.get(block)
            theirs = part[block]
            if block == "tendons":
                if not all(isinstance(t, dict) for t in list(theirs) + list(mine or [])):
                    raise ValueError(f"`include: {path}`: each tendon is a mapping")
                names = {t.get("name"): t for t in (mine or [])}
                clash = sorted(t.get("name") for t in theirs if t.get("name") in names and names[t.get("name")] != t)
                if clash and not override:
                    raise ValueError(f"tendon {clash[0]!r} is in the case and in `include: {path}`; "
                                     "write `{path: ..., override: true}` for the case's to win")
                d[block] = [t for t in theirs if t.get("name") not in names] + list(mine or [])
                continue
            mine = dict(mine or {})
            if block == "tendon_defaults":
                mine = _tables_from(mine, base_dir)       # (theirs: from their file, above)
            clash = sorted(k for k in set(mine) & set(theirs) if mine[k] != theirs[k])
            if clash and not override:
                raise ValueError(f"{block} {clash[0]!r} is in the case and in `include: {path}`; "
                                 "write `{path: ..., override: true}` for the case's to win")
            merged = dict(theirs)
            merged.update(mine)
            d[block] = merged
    return d


# -------------------------------------------------------- tendon defaults

def tendon_defaults(d: Dict[str, Any]) -> Dict[str, Any]:
    """`d` with its `tendon_defaults:` written into each tendon."""
    from .tendons import TendonSpec
    if "tendon_defaults" not in d:
        return d
    d = dict(d)
    base = d.pop("tendon_defaults")
    if base is None:
        return d
    if not isinstance(base, dict):
        raise ValueError(f"`tendon_defaults` is a mapping of a tendon's keys, not {base!r}")
    known = set(TendonSpec.model_fields) - {"name"}
    extra = sorted(k for k in base if k not in known)
    if extra:
        raise ValueError(f"`tendon_defaults` has {', '.join('`' + str(k) + '`' for k in extra)}; it takes a "
                         f"tendon's keys but its name ({', '.join(sorted(known))})")
    out = []
    for j, t in enumerate(d.get("tendons") or []):
        if not isinstance(t, dict):
            raise ValueError(f"tendons[{j}] is a mapping, not {t!r}")
        t_ = copy.deepcopy(base)
        for k, v in t.items():
            if isinstance(v, dict) and isinstance(t_.get(k), dict) and "table" not in v:
                shared = dict(t_[k])
                if k == "stress" and ("stress" in v or "force" in v):
                    shared.pop("stress", None), shared.pop("force", None)   # one of the two, the tendon's
                t_[k] = {**shared, **copy.deepcopy(v)}
            else:
                t_[k] = copy.deepcopy(v)
        out.append({k: t_[k] for k in (["name"] if "name" in t_ else []) + [k for k in t_ if k != "name"]})
    d["tendons"] = out
    return d


# ----------------------------------------------------------------- tables

def _read_csv(path: str, base_dir: Optional[str], who: str, allowed=None) -> List[Dict[str, str]]:
    full = path if os.path.isabs(path) else os.path.join(_base(base_dir), path)
    if not os.path.isfile(full):
        raise ValueError(f"{who}: `table: {path}`: there is no such file ({full})")
    try:
        with open(full, "r", encoding="utf-8-sig", newline="") as fh:
            rows = [(i, r) for i, r in enumerate(csv.reader(fh), 1)
                    if r and any(c.strip() for c in r) and not r[0].lstrip().startswith("#")]
    except (csv.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"{who}: `table: {path}` cannot be read as CSV (UTF-8): {exc}") from None
    if len(rows) < 2:
        raise ValueError(f"{who}: `table: {path}` needs a header row and at least one row")
    head = [h.strip().lower() for h in rows[0][1]]
    if allowed is not None:
        extra = [h for h in head if h not in allowed]
        if extra:
            raise ValueError(f"{who}: `table: {path}` has the column(s) {', '.join(extra)}; it reads "
                             f"{', '.join(allowed)}")
    out = []
    for i, r in rows[1:]:
        if len(r) != len(head):
            raise ValueError(f"{who}: `table: {path}` line {i} has {len(r)} values, the header {len(head)}")
        out.append({h: v.strip() for h, v in zip(head, r)})
    return out


def _num(v: str, who: str, where: str) -> float:
    try:
        return float(v)
    except ValueError:
        raise ValueError(f"{who}: {where}: {v!r} is not a number") from None


def _profile(spec, base_dir, who, own=None):
    """[[x, v, r], ...] from `{table: file.csv, name?: ...}`; a table with a
    `name` column gives the rows of `name`, or of the tendon's own name."""
    if not (isinstance(spec, dict) and "table" in spec):
        return spec
    extra = sorted(set(spec) - {"table", "name"})
    if extra:
        raise ValueError(f"{who}: a profile table takes `table` and `name`, not {', '.join(extra)}")
    rows = _read_csv(spec["table"], base_dir, who, allowed=("name", "x", "y", "z", "radius"))
    if "name" not in spec and "name" in rows[0] and own is not None:
        # the rows of the tendon's own name; a table of one
        # name, another, is still all of it, as it was read before
        if any(r["name"] == str(own) for r in rows) or len({r["name"] for r in rows}) > 1:
            spec = dict(spec, name=own)
    if "name" in spec:
        if "name" not in rows[0]:
            raise ValueError(f"{who}: `name: {spec['name']}` picks rows by a `name` column, which "
                             f"{spec['table']} does not have")
        rows = [r for r in rows if r["name"] == str(spec["name"])]
        if not rows:
            raise ValueError(f"{who}: {spec['table']} has no row named {spec['name']!r}")
    cols = [c for c in ("x", "z", "y") if c in rows[0]]
    if "x" not in cols or len(cols) != 2:
        raise ValueError(f"{who}: a profile table has the columns x and one of z or y (and radius)")
    v = cols[1]
    return [[_num(r["x"], who, "x"), _num(r[v], who, v), _num(r.get("radius", "0") or "0", who, "radius")]
            for r in rows]


def _node_rows(entry, base_dir, who, regions, values):
    """The rows of a load or mass table: (region name, {column: number})."""
    extra = sorted(set(entry) - {"table"} - set(values.get("keys", ())))
    if extra:
        raise ValueError(f"{who}: a table entry takes `table`{''.join(', `' + k + '`' for k in values.get('keys', ()))},"
                         f" not {', '.join(extra)}")
    rows = _read_csv(entry["table"], base_dir, who, allowed=("region", "x", "y", "z") + tuple(values["columns"]))
    stem = os.path.splitext(os.path.basename(entry["table"]))[0]
    out = []
    for i, r in enumerate(rows, 1):
        if r.get("region"):
            name = r["region"]
        elif all(c in r for c in ("x", "y", "z")):
            name = f"{stem}#{i}"
            spec = {"nearest": [_num(r[c], who, c) for c in ("x", "y", "z")]}
            if regions.get(name, spec) != spec:
                raise ValueError(f"{who}: a region named {name!r} is already in `regions`, elsewhere")
            regions[name] = spec              # (the same table used again: the same places)
        else:
            raise ValueError(f"{who}: {entry['table']} row {i}: give `region`, or `x`, `y`, `z` (its nearest node)")
        out.append((name, r))
    return out


def tables(d: Dict[str, Any], base_dir: Optional[str]) -> Dict[str, Any]:
    """`d` with every `table:` replaced by the rows its CSV file holds."""
    d = copy.deepcopy(d)
    for t in d.get("tendons") or []:
        who = f"tendon {t.get('name')!r}"
        for key in ("profile_y", "profile_z"):
            if key in t:
                t[key] = _profile(t[key], base_dir, who, own=t.get("name"))
    regions = d.get("regions") or {}

    def loads(lst, where):
        out = []
        for j, ld in enumerate(lst or []):
            if not (isinstance(ld, dict) and "table" in ld):
                out.append(ld)
                continue
            who = f"{where}[{j}]"
            for name, r in _node_rows(ld, base_dir, who, regions,
                                      {"keys": ("keep", "amplitude"), "columns": ("fx", "fy", "fz", "mx", "my", "mz")}):
                f = [_num(r.get(c, "0") or "0", who, c) for c in ("fx", "fy", "fz")]
                m = [_num(r.get(c, "0") or "0", who, c) for c in ("mx", "my", "mz")]
                common = {k: ld[k] for k in ("keep", "amplitude") if k in ld}
                if any(f) or not any(m):
                    out.append({"region": name, "per_node": True, "vector": f, **common})
                if any(m):
                    out.append({"region": name, "per_node": True, "moment": m, **common})
        return out

    if d.get("loads"):
        d["loads"] = loads(d["loads"], "loads")
    for k, st in enumerate(d.get("steps") or []):
        if isinstance(st, dict) and st.get("loads"):
            st["loads"] = loads(st["loads"], f"steps[{k}].loads")
    if d.get("nodal_masses"):
        out = []
        for j, nm in enumerate(d["nodal_masses"]):
            if not (isinstance(nm, dict) and "table" in nm):
                out.append(nm)
                continue
            who = f"nodal_masses[{j}]"
            for name, r in _node_rows(nm, base_dir, who, regions, {"keys": ("weight", "components"), "columns": ("mass",)}):
                out.append({"region": name, "per_node": True, "mass": _num(r.get("mass", ""), who, "mass"),
                            **{k: nm[k] for k in ("weight", "components") if k in nm}})
        d["nodal_masses"] = out
    if regions:
        d["regions"] = regions
    return d


# ----------------------------------------------------------------- groups

def groups(d: Dict[str, Any]) -> Dict[str, Any]:
    """`d` with its `groups` as `mesh.groups` (nested groups flattened to the
    mesh's own) and the steps' lists of members written out."""
    gs = d.get("groups")
    if not gs:
        return d
    d = copy.deepcopy(d)
    gs = d.pop("groups")
    if not isinstance(gs, dict):
        raise ValueError("`groups` is a mapping: name -> [region, ...]")
    mesh = d.get("mesh") or {}
    if (mesh.get("type") or "gmsh") != "gmsh":
        raise ValueError("top-level `groups` name groups of a Gmsh mesh; a mesh of type "
                         f"{mesh.get('type')!r} writes `mesh.groups`" if mesh.get("type") == "lines" else
                         "top-level `groups` name groups of a Gmsh mesh")
    regions = d.get("regions") or {}
    for name, members in gs.items():
        if not isinstance(members, list) or not all(isinstance(m, str) for m in members) or not members:
            raise ValueError(f"group {name!r} is a list of region (or group) names")
        if name in regions or name == "all":
            raise ValueError(f"{name!r} is both a group and a region of the case")
        built = [m for m in members if m in regions]
        if built:
            raise ValueError(f"group {name!r} lists {built[0]!r}, a region of the case; a group joins the "
                             "mesh's groups (its elements), which a region written as a box or a plane has not")

    def flat(name, path=()):
        if name in path:
            raise ValueError(f"group {name!r} contains itself ({' > '.join(path + (name,))})")
        out = []
        for m in gs[name]:
            out.extend(flat(m, path + (name,)) if m in gs else [m])
        return out
    resolved = {g: list(dict.fromkeys(flat(g))) for g in gs}

    def names(lst):
        out = []
        for n in lst or []:
            out.extend(resolved.get(n, [n]))
        return list(dict.fromkeys(out))

    for k, st in enumerate(d.get("steps") or []):
        if not isinstance(st, dict):
            raise ValueError(f"steps[{k}] is a mapping")
        for key in ("activate", "deactivate", "release"):
            if st.get(key):
                st[key] = names(st[key])
    d["mesh"] = dict(mesh, groups=resolved)
    return d
