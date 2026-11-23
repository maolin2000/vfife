"""Pre-processing: what a case writes in a few lines and the solver reads in
many. `expand_case` is called on every case dict
before it is lowered; a case with none of the high-level blocks passes
unchanged. `blocks`: include, tendon_defaults, table, groups; `offsets`: the arms of members set off their nodes; `tendons`: the tendon
block."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

#: the top-level keys this expands
HIGH_LEVEL = ("include", "groups", "tendons", "tendon_defaults", "offset_arms")


def _has_table(d: Dict[str, Any]) -> bool:
    def table_in(lst):
        return any(isinstance(e, dict) and "table" in e for e in (lst or []))
    return (table_in(d.get("loads")) or table_in(d.get("nodal_masses"))
            or any(isinstance(st, dict) and table_in(st.get("loads")) for st in (d.get("steps") or []))
            or any(isinstance(t, dict) and any(isinstance(t.get(k), dict) for k in ("profile_y", "profile_z"))
                   for t in (d.get("tendons") or [])))


def has_high_level(d: Any) -> bool:
    return isinstance(d, dict) and (any(d.get(k) for k in HIGH_LEVEL) or _has_table(d))


def mesh_path(d: Dict[str, Any], base_dir: Optional[str]) -> str:
    m = d.get("mesh") or {}
    path = m.get("path")
    if (m.get("type") or "gmsh") != "gmsh" or not path:
        raise ValueError("`tendons` and `offset_arms` expand onto a Gmsh mesh: give `mesh: {type: gmsh, path: ...}`")
    if not os.path.isabs(path):
        path = os.path.join(base_dir or os.getcwd(), path)
    return path


def expand_case(d: Dict[str, Any], base_dir: Optional[str] = None, warn=None
                ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
    """(the case with its high-level blocks expanded, the mesh overlay or
    None, the report). Reads the mesh the case names."""
    if not has_high_level(d):
        return d, None, {}
    from . import blocks
    # the ones that need no mesh first, in the order each can use the last:
    # an included file's tables, a table's regions in a group;
    # the tendons' defaults before the tables, which a default may name
    d = blocks.include(d, base_dir)
    d = blocks.tendon_defaults(d)
    d = blocks.tables(d, base_dir)
    d = blocks.groups(d)
    if not d.get("tendons") and d.get("offset_arms") is None:
        d = dict(d)
        d.pop("tendons", None)
        d.pop("offset_arms", None)
        return d, None, {}
    from ..io import gmsh
    from . import offsets, tendons
    path = mesh_path(d, base_dir)
    scale = float((d.get("mesh") or {}).get("scale") or 1.0)
    mesh, element = gmsh.read(path, scale=scale, verify=False)
    if element != "truss2":
        raise ValueError("`tendons` and `offset_arms` are for frame models (a mesh of Line2 members)")
    # the arms first: a tendon's expansion sees the sections they become
    d = offsets.expand(d, mesh)
    if not d.get("tendons"):
        d = dict(d)
        d.pop("tendons", None)
        return d, None, {}
    low, overlay, report = tendons.expand(d, mesh, warn=warn)
    return low, overlay, {"tendons": report}
