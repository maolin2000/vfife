"""The local axes of every line-member group, for `vfife info` and the run
summary.

A section's `width` runs along local y and its `height` (and a fibre's
`z`, the `at: [y, z]` of a monitor) along local z; local y follows the
assign's `orientation` (default global +Z, so a horizontal member's local z
is horizontal). Nothing printed this, and a rect/box section turned on its
side is a stiffness off by (h/b)^2 that no check can see: a note says so
for exactly those sections. Read from the frames the
solver built (`R0`, rows = local x, y, z in global coordinates), so the
line says what the run uses.
"""
from __future__ import annotations

import numpy as np

_BENDING = {"beam2", "beam2_timoshenko", "beam2_fiber"}
_TOL_SAME = 1.0          # deg: one direction for the whole group
_TOL_VERTICAL = 5.0      # deg: local z this close to the vertical is "vertical"


def vector_text(v) -> str:
    """`+X`, `-Y`, or `[0.71, 0, 0.71]`."""
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return "[0, 0, 0]"
    v = v / n
    i = int(np.argmax(np.abs(v)))
    if abs(v[i]) > np.cos(np.radians(0.5)):
        return ("+" if v[i] > 0 else "-") + "XYZ"[i]

    return "[" + ", ".join(_num(c) for c in v) + "]"


def vertical(problem) -> np.ndarray:
    """Opposite to the gravity load when the case has one, else global +Z."""
    p = problem or {}
    for loads in [p.get("loads") or []] + [s.get("loads") or [] for s in p.get("steps") or []]:
        for ld in loads:
            g = ld.get("gravity") if isinstance(ld, dict) else None
            if g is not None and np.linalg.norm(g) > 0.0:
                g = np.asarray(g, dtype=float)
                return -g / np.linalg.norm(g)
    return np.array([0.0, 0.0, 1.0])


def _angles(vecs, ref):
    return np.degrees(np.arccos(np.clip(vecs @ ref, -1.0, 1.0)))


def _axis_text(name, vecs) -> str:
    """One axis over a group: `= +X`, `within 12 deg of +Z`, or `varies`."""
    ref = vecs[0]
    if _angles(vecs, ref).max() < _TOL_SAME:
        return f"local {name} = {vector_text(ref)}"
    mean = vecs.mean(axis=0)
    if np.linalg.norm(mean) < 0.5:
        return f"local {name} varies"
    mean = mean / np.linalg.norm(mean)
    spread = float(_angles(vecs, mean).max())
    if spread > 45.0:
        return f"local {name} varies"
    return f"local {name} within {int(np.ceil(spread - 0.01))} deg of {vector_text(mean)}"


def _fibre_size(fs):
    """(width, height) of a rect/box fibre section; None for patches (the
    user drew y/z), circles and tubes."""
    fs = fs or {}
    if fs.get("section_size") is not None:            # a box or T girder (1.2)
        return tuple(float(v) for v in fs["section_size"])
    if fs.get("type", "rect") in ("rect", "box") and fs.get("height") is not None:
        return float(fs.get("width") or 0.0), float(fs["height"])
    return None


def groups(solver):
    """[(name, element type, element ids, coordinate_system, (width, height)
    of a rect/box section or None)] of the line members, from the solver's
    own grouping."""
    p = solver.p
    el = p.get("element") or {}
    nel = int(getattr(solver, "nel", 0))
    if hasattr(solver, "_rows") and hasattr(solver, "_layouts"):          # fibre
        many = el.get("fiber_sections")
        entries = many or [{"region": el.get("fiber_region") or "all",
                            "fiber_section": el.get("fiber_section")}]
        out = [(entries[k]["region"], "beam2_fiber", np.asarray(r, dtype=int), None,
                _fibre_size(entries[k].get("fiber_section")))
               for k, r in enumerate(solver._rows) if len(r)]
        for k, m in enumerate(el.get("fiber_connectors") or []):
            out.append((m["region"], "connector2", np.flatnonzero(solver._connector_of == k),
                        m.get("coordinate_system"), None))
        return out
    owner = getattr(solver, "member_owner", None)
    if owner is not None:                                                 # mixed_line & co.
        out = []
        for g, spec in enumerate(p.get("members") or []):
            ids = np.flatnonzero(owner == g)
            if len(ids):
                out.append((spec.get("region", "all"), spec.get("type", "?"), ids,
                            spec.get("coordinate_system"), spec.get("section_size")))
        return out
    return [("all", el.get("type", "?"), np.arange(nel), el.get("coordinate_system"),
             el.get("section_size"))]


def _num(c) -> str:
    s = f"{c:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _length_unit(problem) -> str:
    try:
        from .units import complete
        return complete((problem or {}).get("units") or {})[0].get("length", "m")
    except Exception:
        return "m"


def _frames(solver, ids):
    """(n, 3, 3) local axes of the members `ids`; a truss solver keeps no
    frames, so its local x is the chord."""
    R0 = getattr(solver, "R0", None)
    if R0 is not None and len(R0):
        return np.asarray(R0)[ids]
    conn = np.asarray(solver.mesh.elements)[ids, :2]
    x = solver.mesh.nodes[conn[:, 1]] - solver.mesh.nodes[conn[:, 0]]
    out = np.zeros((len(ids), 3, 3))
    out[:, 0] = x / np.linalg.norm(x, axis=1)[:, None]
    return out


def describe(solver, problem=None):
    """[(axes line, note or None)] per group, groups with the same type and
    the same axes merged into one line."""
    up = vertical(problem if problem is not None else solver.p)
    unit = _length_unit(problem if problem is not None else solver.p)
    rows = {}
    for name, kind, ids, csys, size in groups(solver):
        R = _frames(solver, ids)
        if kind == "truss2":
            axes = _axis_text("x", R[:, 0])
        elif csys == "global":
            axes = _axis_text("x", R[:, 0]) + " (global components)"
        else:
            axes = ", ".join(_axis_text(a, R[:, i]) for i, a in enumerate("xyz"))
        note = None
        # only a section with a named height that is not a square: patches
        # (y/z drawn by hand), circles, A/Iy/Iz, bars and connectors have no
        # height to lie down
        if kind in _BENDING and size is not None and abs(size[0] - size[1]) > 1e-9 * max(size):
            ex, ez = R[:, 0], R[:, 2]
            flat = np.abs(ex @ up) < np.sin(np.radians(45.0))             # horizontal-ish
            # the vertical as near as a member at this slope can have it
            v = up[None, :] - ex * (ex @ up)[:, None]
            v /= np.maximum(np.linalg.norm(v, axis=1), 1e-12)[:, None]
            off = flat & (np.degrees(np.arccos(np.clip(np.abs(np.sum(ez * v, axis=1)), 0, 1)))
                          > _TOL_VERTICAL)
            if off.any():
                e = int(np.flatnonzero(off)[0])
                # local y = orientation projected, z = x cross y: z is up
                # when y = up cross x
                y = np.cross(up, ex[e]); y /= np.linalg.norm(y)
                note = (f"section height {size[1]:g} {unit} lies along local z = "
                        f"{vector_text(ez[e])} (horizontal); if the depth should be vertical, "
                        "set orientation so local z is up (e.g. orientation: "
                        f"[{', '.join(_num(c) for c in y)}])")
        key = (kind, axes, note)
        rows.setdefault(key, []).append((name, len(ids)))
    out = []
    for (kind, axes, note), members in rows.items():
        names = [n for n, _ in members]
        shown = ", ".join(names[:4]) + (f" and {len(names) - 4} more" if len(names) > 4 else "")
        count = sum(n for _, n in members)
        what = (f"{kind}, {count} member{'s' * (count != 1)}" if len(names) == 1 else
                f"{kind}, {len(names)} groups, {count} members")
        label = names[0] if len(names) == 1 else shown
        out.append((f"{shown} ({what}): {axes}", f"{label}: {note}" if note else None))
    return out


def summary_lines(solver, problem=None) -> str:
    """The `axes` (and `note`) lines of a summary, each starting with a
    newline; empty when the solver has no line members."""
    try:
        rows = describe(solver, problem)
    except Exception:                       # never let a report line stop `info`
        return ""
    text = ""
    for axes, note in rows:
        text += f"\n  axes          {axes}"
        if note:
            text += f"\n  note          {note}"
    return text
