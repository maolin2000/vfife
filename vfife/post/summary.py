"""`vfife summary`: one page that says what a result is (1.1).

What an analyst checks first, before any plot, taken from the result file
alone (and, for a result written before 1.1, the case for the mass):

* did it finish, and is it an answer (status, convergence, frames);
* does the model weigh what it should: its mass and the weight its `gravity`
  loads apply, the line loads it carries, and the reactions at the last
  frame beside them -- under a static dead load the vertical reaction is
  weight plus line load, and a model that weighs 2.8 times its sections
  shows it on this line;
* where the largest displacement, rotation, member force and stress are,
  over every frame, with the node or member and its position and the time;
* how the members fared (failed, the highest capacity ratio), the
  connectors' largest forces;
* the energy balance and the monitors' ranges.

`summarize(path)` returns all of it as plain data (`--json`); `text(...)`
lays it out.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

#: member forces reported with their largest and smallest value (signed)
MEMBER_FORCES = ("axial_force", "shear_y", "shear_z", "torsion", "bending_y", "bending_z")
#: element fields reported with their largest and smallest value (signed)
ELEMENT_SIGNED = ("axial_stress",)
#: element fields reported with their largest value
ELEMENT_MAXIMA = ("von_mises", "von_mises_top", "von_mises_bottom", "capacity_ratio", "hinge_usage",
                  "hinge_rotation")
CONNECTOR_FORCES = ("connector_force_x", "connector_force_y", "connector_force_z",
                    "connector_moment_x", "connector_moment_y", "connector_moment_z")


def _units(attrs) -> Dict[str, str]:
    out = {}
    for k, v in attrs.items():
        if k.startswith("unit_"):
            out[k[5:]] = v.decode() if isinstance(v, bytes) else str(v)
    return out


def _unit_of(units: Dict[str, str], what: str) -> str:
    L, F = units.get("length", ""), units.get("force", "")
    if what == "length":
        return L
    if what == "force":
        return F
    if what == "moment":
        return f"{F}*{L}" if F and L else ""
    if what == "stress":
        return units.get("stress", "")
    if what == "mass":
        from ..core.units import mass_unit
        return mass_unit(units) if units else ""
    return ""


def _frame_keys(group) -> List[str]:
    return sorted(group, key=int)


def _element_midpoints(f, key: str) -> Optional[np.ndarray]:
    """Element centres from `mesh/elements` (fields in `steps/`) or
    `members/elements` (a coupled model's `member_steps/`), a mixed mesh's
    XDMF stream too."""
    if key not in f:
        return None
    nodes = np.asarray(f["mesh/nodes"])
    e = np.asarray(f[key])
    if e.ndim == 1:
        try:
            from ..io.writer import mixed_centroids
            return mixed_centroids(nodes, e)
        except Exception:
            return None
    if e.shape[1] < 2:
        return None
    return nodes[e].mean(axis=1)


class _Extreme:
    """The largest (and, signed, the smallest) value of one field over the
    frames, where and when."""

    def __init__(self, signed: bool):
        self.signed = signed
        self.hi = None
        self.lo = None
        self.source = "steps"

    def see(self, values: np.ndarray, t: float, frame: str):
        v = np.asarray(values, float)
        if not v.size or not np.isfinite(v).any():
            return
        v = np.where(np.isfinite(v), v, np.nan)
        i = int(np.nanargmax(v))
        if self.hi is None or v[i] > self.hi[0]:
            self.hi = (float(v[i]), i, float(t), frame)
        if self.signed:
            j = int(np.nanargmin(v))
            if self.lo is None or v[j] < self.lo[0]:
                self.lo = (float(v[j]), j, float(t), frame)


def summarize(path, case: Optional[str] = None) -> Dict[str, Any]:
    """Everything `vfife summary` prints, as plain data."""
    from ..io.writer import open_result
    from ..core.diagnostics import UserError
    path = str(path)
    out: Dict[str, Any] = {"result": path}
    with open_result(path) as f:
        if "mesh" not in f or "steps" not in f:
            what = ("a checkpoint" if "checkpoint_format" in f.attrs or path.endswith(".ckpt.h5")
                    else "not the result of a run (a modal or other file?)")
            raise UserError("NOT_A_RUN_RESULT", f"{path} is {what}: it has no mesh and frames",
                            "summary reads the HDF5 result `vfife run` writes (output.path)")
        a = f.attrs
        get = lambda k, d=None: (a[k].decode() if isinstance(a.get(k), bytes) else a.get(k, d))
        units = _units(a)
        out["units"] = units
        out["vfife_version"] = get("vfife_version")
        out["element_family"] = get("element_family")
        out["status"] = get("analysis_status")
        out["converged"] = None if a.get("analysis_converged") is None else bool(a["analysis_converged"])
        out["time_axis"] = get("time_axis")
        out["restarted_from"] = get("restarted_from")
        name = None
        try:
            import yaml
            name = (yaml.safe_load(get("case_yaml") or "") or {}).get("name")
        except Exception:
            pass
        out["case"] = name or Path(path).stem
        out["time_unit"] = units.get("time", "s")
        nodes = np.asarray(f["mesh/nodes"])
        out["nodes"] = int(len(nodes))
        diag = f["diagnostics"] if "diagnostics" in f else None
        if diag is not None:
            d = diag.attrs
            out["diagnostics"] = {k: (d[k].decode() if isinstance(d[k], bytes) else
                                      d[k].item() if hasattr(d[k], "item") else d[k])
                                  for k in ("mode", "backend", "step", "residual", "message", "status")
                                  if k in d}
            series = {}
            for k in ("relative_energy_error", "external_work", "kinetic_energy", "internal_work",
                      "dissipated_energy", "energy_imbalance"):
                if k in diag:
                    series[k] = np.asarray(diag[k], float)
            if "relative_energy_error" in series and series["relative_energy_error"].size:
                r = series["relative_energy_error"]
                fin = r[np.isfinite(r)]
                out["energy"] = {
                    "relative_error_final": float(r[-1]),
                    "relative_error_max": float(np.max(np.abs(fin))) if fin.size else None,
                    "external_work_final": (float(series["external_work"][-1])
                                            if "external_work" in series and series["external_work"].size
                                            else None),
                    "available": bool(d.get("energy_available", True)),
                }

        if not out["time_axis"]:
            # a format-2 result has no `time_axis`: the run's mode says it
            out["time_axis"] = ("time" if (out.get("diagnostics") or {}).get("mode") == "dynamic"
                                else "load_factor")

        # ---------------------------------------------------- the model's totals
        totals = {k: np.asarray(a[f"model_{k}"], float) for k in ("mass", "weight", "line_load", "point_load")
                  if f"model_{k}" in a}
        if not totals and case:
            totals = _totals_from_case(case)
            out["totals_from"] = "case"
        elif totals:
            out["totals_from"] = "result"
        out["totals"] = {k: v.tolist() for k, v in totals.items()}

        # ---------------------------------------------------- the frames
        steps = f["steps"] if "steps" in f else None
        keys = _frame_keys(steps) if steps is not None else []
        out["frames"] = len(keys)
        mids = {"steps": _element_midpoints(f, "mesh/elements"),
                "member_steps": _element_midpoints(f, "members/elements")}
        members = f["member_steps"] if "member_steps" in f else None
        if keys:
            first, last = steps[keys[0]], steps[keys[-1]]
            out["time_first"] = float(first.attrs.get("time", 0.0))
            out["time_last"] = float(last.attrs.get("time", 0.0))
            eq = [steps[k].attrs.get("equilibrium") for k in keys]
            eq = [e.decode() if isinstance(e, bytes) else e for e in eq]
            out["equilibrium_frames"] = {"true": eq.count("true"), "false": eq.count("false")}
            if "reaction_force" in last:
                out["reaction_last"] = np.asarray(last["reaction_force"], float).sum(axis=0).tolist()
            if "member_failed" in last:
                out["members_failed_last"] = int(np.nansum(np.asarray(last["member_failed"]) > 0.5))

        disp, rot = _Extreme(False), _Extreme(False)
        comp = [_Extreme(True) for _ in range(3)]
        # a shell's `shear_y` is a resultant (force per length), not a member's
        # force: member forces only where there are members
        has_members = bool(keys) and ("axial_force" in steps[keys[-1]] or (
            members is not None and keys[-1] in members and "axial_force" in members[keys[-1]]))
        el = {n: _Extreme(True) for n in (MEMBER_FORCES if has_members else ())}
        el.update({n: _Extreme(True) for n in ELEMENT_SIGNED})
        el.update({n: _Extreme(False) for n in ELEMENT_MAXIMA})
        conn = {n: _Extreme(True) for n in CONNECTOR_FORCES}
        first_failure = None
        for k in keys:
            g = steps[k]
            t = float(g.attrs.get("time", 0.0))
            if "displacement" in g:
                u = np.asarray(g["displacement"], float)
                disp.see(np.linalg.norm(u, axis=1), t, k)
                for c in range(min(3, u.shape[1])):
                    comp[c].see(u[:, c], t, k)
            if "rotation" in g:
                rot.see(np.abs(np.asarray(g["rotation"], float)).max(axis=1), t, k)
            mg = members[k] if members is not None and k in members else None
            for n, ext in list(el.items()) + list(conn.items()):
                src = g if n in g else (mg if mg is not None and n in mg else None)
                if src is not None:
                    ext.source = "steps" if src is g else "member_steps"
                    ext.see(np.asarray(src[n], float).reshape(len(src[n]), -1)[:, 0], t, k)
            if first_failure is None and "member_failed" in g:
                failed = np.flatnonzero(np.asarray(g["member_failed"]) > 0.5)
                if len(failed):
                    first_failure = {"time": t, "members": failed[:10].tolist(), "count": int(len(failed))}

        def where_node(i):
            return {"node": int(i), "at": nodes[i].tolist()}

        def where_el(i, source="steps"):
            w = {"element": int(i)}
            m = mids.get(source)
            if m is not None and i < len(m):
                w["at"] = m[i].tolist()
            return w

        ext = {}
        if disp.hi:
            v, i, t, _k = disp.hi
            ext["displacement"] = dict(value=v, time=t, **where_node(i))
            ext["displacement_components"] = []
            for c, e in enumerate(comp):
                if e.hi is None:
                    continue
                big = e.hi if abs(e.hi[0]) >= abs(e.lo[0]) else e.lo
                if big[0] == 0.0:
                    continue
                ext["displacement_components"].append(dict(axis="xyz"[c], value=big[0], time=big[2],
                                                           **where_node(big[1])))
        if rot.hi and rot.hi[0] != 0.0:
            v, i, t, _k = rot.hi
            ext["rotation"] = dict(value=v, time=t, **where_node(i))
        for n, e in el.items():
            if e.hi is None or (e.hi[0] == 0.0 and (e.lo is None or e.lo[0] == 0.0)):
                continue                       # not written, or zero throughout
            row = {"max": dict(value=e.hi[0], time=e.hi[2], **where_el(e.hi[1], e.source))}
            if e.signed and e.lo is not None and (e.lo[0], e.lo[1]) != (e.hi[0], e.hi[1]):
                row["min"] = dict(value=e.lo[0], time=e.lo[2], **where_el(e.lo[1], e.source))
            ext[n] = row
        for n, e in conn.items():
            if e.hi is None:
                continue
            big = e.hi if abs(e.hi[0]) >= abs(e.lo[0]) else e.lo
            if big[0] != 0.0:
                ext[n] = {"max_abs": dict(value=big[0], time=big[2], element=int(big[1]))}
        out["extremes"] = ext
        if "response_spectrum" in f:
            rs = f["response_spectrum"]
            spec = {k: (rs.attrs[k].item() if hasattr(rs.attrs[k], "item") else rs.attrs[k])
                    for k in rs.attrs}
            for k in ("period", "base_force"):
                if k in rs:
                    spec[k] = np.asarray(rs[k], float).ravel().tolist()
            if "displacement" in rs:
                du = np.linalg.norm(np.asarray(rs["displacement"], float), axis=1)
                i = int(np.argmax(du))
                spec["displacement"] = dict(value=float(du[i]), **where_node(i))
            for n in MEMBER_FORCES:
                if n in rs:
                    v = np.abs(np.asarray(rs[n], float))
                    if v.size and np.nanmax(v) > 0:
                        i = int(np.nanargmax(v))
                        spec[n] = dict(value=float(v[i]), **where_el(i))
            out["response_spectrum"] = spec
            out["extremes"] = {}               # the frame is the state the modes are taken at
        if first_failure:
            out["first_failure"] = first_failure

        # ---------------------------------------------------- the monitors
        mons = {}
        if "monitors" in f:
            mt = np.asarray(f["monitors/time"], float) if "time" in f["monitors"] else None
            for name, ds in f["monitors"].items():
                if name in ("time", "residual", "kinetic_energy", "age"):
                    continue
                v = np.asarray(ds, float)
                if not v.size:
                    continue
                fin = np.isfinite(v)
                u = ds.attrs.get("units")
                row = {"final": float(v[-1]), "units": (u.decode() if isinstance(u, bytes) else u)}
                if fin.any():
                    i, j = int(np.nanargmax(v)), int(np.nanargmin(v))
                    row.update(max=float(v[i]), min=float(v[j]))
                    if mt is not None and len(mt) == len(v):
                        row.update(time_max=float(mt[i]), time_min=float(mt[j]))
                mons[name] = row
        out["monitors"] = mons
    return out


def _totals_from_case(case: str) -> Dict[str, np.ndarray]:
    """For a result written before 1.1: build the model from its case."""
    from ..api import prepare_case
    from ..assemble import _with_element, make_solver
    from ..core.model_totals import totals
    prep = prepare_case(case)
    p = _with_element(prep.case, prep.element, prep.case_path)
    solver = make_solver(p, prep.mesh, prep.material)
    return {k: np.asarray(v, float) for k, v in totals(solver, p).items() if v is not None}


# --------------------------------------------------------------------------- text
def _g(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if x == 0:
        return "0"                              # not "-0"
    return f"{x:.4g}"


def _finite(x):
    """NaN and infinities as None, for strict JSON."""
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return _finite(x.tolist())
    return x


def _vec(v) -> str:
    return "[" + ", ".join(_g(x) for x in v) + "]"


def _at(row) -> str:
    where = f"node {row['node']}" if "node" in row else f"element {row['element']}"
    if row.get("at") is not None:
        where += " at " + _vec(row["at"])
    return where


def text(s: Dict[str, Any]) -> str:
    u = s.get("units") or {}
    L, F = _unit_of(u, "length"), _unit_of(u, "force")
    M, S, Ma = _unit_of(u, "moment"), _unit_of(u, "stress"), _unit_of(u, "mass")
    tname = "t" if s.get("time_axis") == "time" else "load factor"
    tu = f" {s.get('time_unit') or 's'}" if s.get("time_axis") == "time" else ""
    lines = [f"result: {s['result']}",
             f"  case          {s.get('case')} ({s.get('element_family')}, {s.get('nodes')} nodes"
             + (f"; units {', '.join(f'{k} {v}' for k, v in u.items())}" if u else "; no units declared")
             + f"; vfife {s.get('vfife_version')})"]
    status = s.get("status") or "not closed (the run was killed: frames only)"
    conv = s.get("converged")
    lines.append(f"  status        {status}" + ("" if conv is None else
                                               ("  -- an answer" if conv else "  -- NOT an answer")))
    d = s.get("diagnostics") or {}
    if d.get("message"):
        lines.append(f"  message       {d['message']}")
    if s.get("restarted_from"):
        lines.append(f"  continuation  of {s['restarted_from']} (this file holds the tail only)")
    if s.get("frames"):
        eqf = s.get("equilibrium_frames") or {}
        tail = (f"; equilibrium in {eqf['true']} of them" if eqf.get("true") or eqf.get("false") else "")
        lines.append(f"  frames        {s['frames']}, {tname} {_g(s.get('time_first'))} to "
                     f"{_g(s.get('time_last'))}{tu}{tail}")
    if d.get("residual") is not None and s.get("time_axis") != "time":
        lines.append(f"  residual      {_g(d['residual'])} at the end")

    tot = s.get("totals") or {}
    if tot:
        lines.append("")
        lines.append("what the model weighs and carries" + (" (built from the case)" if s.get("totals_from") == "case" else ""))
        if tot.get("mass") is not None:
            lines.append(f"  mass          {_vec(tot['mass'])} {Ma} in x, y, z".rstrip())
        if tot.get("weight") is not None and np.any(np.asarray(tot["weight"])):
            lines.append(f"  weight        {_vec(tot['weight'])} {F}  (the gravity loads at the step's end)")
        if tot.get("line_load") is not None and np.any(np.asarray(tot["line_load"])):
            lines.append(f"  line loads    {_vec(tot['line_load'])} {F}  (line_load, at the step's end)")
        if tot.get("point_load") is not None and np.any(np.asarray(tot["point_load"])):
            lines.append(f"  point loads   {_vec(tot['point_load'])} {F}  (nodal forces, at the step's end)")
    if s.get("reaction_last") is not None:
        r = s["reaction_last"]
        lines.append(f"  reactions     {_vec(r)} {F}  (sum at the last frame; they oppose the load)")
        applied = np.zeros(3)
        for k in ("weight", "line_load", "point_load"):
            if tot.get(k) is not None:
                applied += np.asarray(tot[k])
        t_last = s.get("time_last")
        # at full amplitude: a load-factor axis is completed steps + lambda,
        # whole at the end of a step
        full = t_last is not None and t_last >= 1.0 - 1e-9 and abs(t_last - round(t_last)) < 1e-6
        if np.any(applied) and s.get("time_axis") != "time" and full:
            ax = int(np.argmax(np.abs(applied)))
            if abs(applied[ax]) > 0:
                lines.append(f"  check         reaction {'xyz'[ax]} / (weight + line and point loads) = "
                             f"{-r[ax] / applied[ax]:.4f}"
                             "  (1 when these are all the loads and the frame is in equilibrium)")

    ext = s.get("extremes") or {}
    if ext:
        lines.append("")
        lines.append("largest over all frames")
        if "displacement" in ext:
            r = ext["displacement"]
            lines.append(f"  |u|           {_g(r['value'])} {L} at {_at(r)}, {tname} {_g(r['time'])}{tu}")
            for c in ext.get("displacement_components", []):
                lines.append(f"  u{c['axis']}            {_g(c['value'])} {L} at {_at(c)}, {tname} {_g(c['time'])}{tu}")
        if "rotation" in ext:
            r = ext["rotation"]
            lines.append(f"  rotation      {_g(r['value'])} rad at {_at(r)}, {tname} {_g(r['time'])}{tu}")
        for n in MEMBER_FORCES:
            if n in ext:
                unit = F if n in ("axial_force", "shear_y", "shear_z") else M
                hi, lo = ext[n]["max"], ext[n].get("min")
                lines.append(f"  {n:<13} max {_g(hi['value'])} {unit} at {_at(hi)}, {tname} {_g(hi['time'])}{tu}")
                if lo is not None:
                    lines.append(f"  {'':<13} min {_g(lo['value'])} {unit} at {_at(lo)}, {tname} {_g(lo['time'])}{tu}")
        for n in ELEMENT_SIGNED:
            if n in ext:
                hi, lo = ext[n]["max"], ext[n].get("min")
                lines.append(f"  {n:<13} max {_g(hi['value'])} {S} at {_at(hi)}, {tname} {_g(hi['time'])}{tu}")
                if lo is not None:
                    lines.append(f"  {'':<13} min {_g(lo['value'])} {S} at {_at(lo)}, {tname} {_g(lo['time'])}{tu}")
        for n in ELEMENT_MAXIMA:
            if n in ext:
                hi = ext[n]["max"]
                unit = S if n.startswith("von_mises") else ("rad" if n == "hinge_rotation" else "")
                lines.append(f"  {n:<13} max {_g(hi['value'])} {unit} at {_at(hi)}, {tname} {_g(hi['time'])}{tu}")
        for n in CONNECTOR_FORCES:
            if n in ext:
                r = ext[n]["max_abs"]
                unit = F if "force" in n else M
                lines.append(f"  {n:<19} {_g(r['value'])} {unit} at element {r['element']}, {tname} {_g(r['time'])}{tu}")
    rs = s.get("response_spectrum")
    if rs:
        lines.append("")
        lines.append(f"response spectrum ({rs.get('n_modes', '?')} modes, {rs.get('combination', '?')}, "
                     f"damping {_g(rs.get('damping'))}); combined peaks, magnitudes")
        if rs.get("period"):
            lines.append(f"  periods       {_vec(rs['period'][:6])}{' ...' if len(rs['period']) > 6 else ''} s")
        if rs.get("base_force"):
            lines.append(f"  base force    {_vec(rs['base_force'])} {F}")
        if rs.get("displacement"):
            r = rs["displacement"]
            lines.append(f"  |u|           {_g(r['value'])} {L} at {_at(r)}")
        for n in MEMBER_FORCES:
            if n in rs:
                unit = F if n in ("axial_force", "shear_y", "shear_z") else M
                lines.append(f"  {n:<13} {_g(rs[n]['value'])} {unit} at {_at(rs[n])}")
    if s.get("first_failure"):
        ff = s["first_failure"]
        lines.append(f"  first failure {tname} {_g(ff['time'])}{tu}: member(s) {ff['members']}"
                     + (f" ({ff['count']} at once)" if ff["count"] > len(ff["members"]) else ""))
    if s.get("members_failed_last") is not None:
        lines.append(f"  failed        {s['members_failed_last']} member(s) at the last frame")

    en = s.get("energy")
    if en:
        lines.append("")
        lines.append("energy")
        if not en.get("available", True):
            lines.append("  not booked for this model")
        else:
            lines.append(f"  balance       relative error {_g(en['relative_error_final'])} at the end, "
                         f"largest {_g(en['relative_error_max'])}"
                         + (f"; external work {_g(en['external_work_final'])} {M}".rstrip()
                            if en.get("external_work_final") is not None else ""))

    mons = s.get("monitors") or {}
    if mons:
        lines.append("")
        lines.append("monitors (final, min, max)")
        w = max(len(n) for n in mons)
        for n, r in mons.items():
            unit = f" {r['units']}" if r.get("units") else ""
            lines.append(f"  {n:<{w}}  {_g(r['final'])}{unit}   [{_g(r.get('min'))}, {_g(r.get('max'))}]")
    return "\n".join(lines)
