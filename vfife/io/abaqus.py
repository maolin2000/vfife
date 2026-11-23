"""Small, strict Abaqus INP mesh reader for the engineering interchange path.

Supported keywords: NODE, ELEMENT, NSET and ELSET. Loads, materials and steps
remain in the VFIFE case schema so one model has one unambiguous source of
analysis controls.
"""
from __future__ import annotations

import numpy as np

from ..core.mesh import Mesh

_TYPES = {
    "C3D8": ("hexa8", 8), "C3D8R": ("hexa8", 8),
    "C3D4": ("tet4", 4), "C3D10": ("tet10", 10),
    "C3D6": ("wedge6", 6), "T3D2": ("truss2", 2),
    "B31": ("truss2", 2),
}


def _options(line):
    parts = [p.strip() for p in line[1:].split(",")]
    out = {"keyword": parts[0].upper()}
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1); out[k.strip().lower()] = v.strip()
        elif part:
            out[part.lower()] = True
    return out


def read(path: str, scale: float = 1.0, coupled: bool = False):
    lines = open(path, encoding="utf-8-sig").read().splitlines()
    nodes = {}; blocks = {}; nsets = {}; elsets = {}; element_nodes = {}
    mode = None; opts = {}; current_type = None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("**"): continue
        if line.startswith("*"):
            opts = _options(line); mode = opts["keyword"]
            current_type = opts.get("type", "").upper() if mode == "ELEMENT" else None
            if mode == "ELEMENT" and current_type not in _TYPES:
                raise ValueError(f"unsupported Abaqus element type {current_type!r}")
            continue
        vals = [x.strip() for x in line.split(",") if x.strip()]
        if mode == "NODE":
            nodes[int(vals[0])] = [float(x)*scale for x in vals[1:4]]
        elif mode == "ELEMENT":
            name, nne = _TYPES[current_type]; eid = int(vals[0]); conn = [int(x) for x in vals[1:]]
            if len(conn) != nne: raise ValueError(f"{current_type} element {eid} needs {nne} nodes")
            blocks.setdefault(name, []).append((eid, conn)); element_nodes[eid] = conn
            if opts.get("elset"): elsets.setdefault(opts["elset"], []).append(eid)
        elif mode in {"NSET", "ELSET"}:
            target = nsets if mode == "NSET" else elsets; name = opts.get(mode.lower())
            if not name: raise ValueError(f"*{mode} requires {mode}=name")
            numbers = [int(x) for x in vals]
            if opts.get("generate"):
                if len(numbers) != 3: raise ValueError(f"generated {mode} needs start,end,increment")
                numbers = list(range(numbers[0], numbers[1]+1, numbers[2]))
            target.setdefault(name, []).extend(numbers)
    if not nodes or not blocks: raise ValueError("INP contains no supported nodes/elements")
    tags = np.array(sorted(nodes)); index = {tag: i for i, tag in enumerate(tags)}
    xyz = np.asarray([nodes[t] for t in tags], float)
    arrays = {k: np.asarray([[index[n] for n in conn] for _eid, conn in rows], np.int32)
              for k, rows in blocks.items()}
    regions = {name: np.asarray(sorted({index[n] for n in ids if n in index}), np.int64)
               for name, ids in nsets.items()}
    for name, ids in elsets.items():
        ns = {index[n] for eid in ids for n in element_nodes.get(eid, [])}
        if ns: regions[name] = np.asarray(sorted(ns), np.int64)
    solids = [k for k in arrays if k != "truss2"]
    if coupled:
        if len(solids) != 1 or "truss2" not in arrays:
            raise ValueError("coupled INP requires one solid topology plus T3D2/B31")
        primary = solids[0]; element = "coupled"
    else:
        if len(arrays) != 1: raise ValueError("mixed INP requires element.type: coupled")
        primary = next(iter(arrays)); element = primary
    used = np.unique(np.concatenate([a.ravel() for a in arrays.values()]))
    remap = np.full(len(xyz), -1, int); remap[used] = np.arange(len(used))
    xyz = xyz[used]; arrays = {k: remap[a].astype(np.int32) for k, a in arrays.items()}
    regions = {k: remap[v][remap[v] >= 0] for k, v in regions.items()}
    mesh = Mesh(xyz, arrays[primary], regions, arrays if coupled else {})
    return mesh, element
