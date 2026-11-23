"""One place that turns a material spec (a dict from the case) into a Material.

`char_length: auto` / `l_ref: auto` resolve to the mesh's smallest element
size here, so the case never hard-codes a number that goes stale when the
mesh is refined. The CLI's `build` and the solver's per-block construction
both come through this function, so a block material and the whole-mesh
material are made the same way.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from . import base as mat_base

AUTO_KEYS = ("char_length", "l_ref")


def wants_auto_length(props: Dict[str, Any]) -> bool:
    return any(str(props.get(k, "")).lower() == "auto" for k in AUTO_KEYS)


def create_material(props: Dict[str, Any], char_length: Optional[float] = None):
    """Material object from its spec. `char_length` resolves any `auto`."""
    mprops = dict(props)
    for k in AUTO_KEYS:
        if str(mprops.get(k, "")).lower() == "auto":
            if char_length is None:
                raise ValueError(f"material {mprops.get('type')!r} asks for {k}: auto "
                                 "but no mesh length is available to resolve it")
            mprops[k] = float(char_length)
    return mat_base.create(mprops.pop("type"), **mprops)
