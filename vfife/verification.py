"""Contract-driven verification of VFIFE HDF5 results.

Verification contracts live outside the solver so reference answers cannot
silently influence the analysis.  A report records the executed backend and
precision from the case YAML embedded in the result file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py

from .io.writer import open_result
import numpy as np
import yaml
from .base.compat import trapezoid


_COMPONENT = {"x": 0, "y": 1, "z": 2, "xx": 0, "yy": 1, "zz": 2,
              "xy": 3, "yz": 4, "zx": 5, "xz": 5}   # zx is the name; xz read too


def load_contract(path: str | Path) -> dict[str, Any]:
    from .io.schema import safe_load_named
    data = safe_load_named(Path(path).read_text(encoding="utf-8"), path)
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("verification contract must be a mapping with version: 1")
    for key in ("case", "classification", "metrics"):
        if key not in data:
            raise ValueError(f"verification contract is missing {key!r}")
    if not data["metrics"]:
        raise ValueError("verification contract requires at least one metric")
    return data


def _case_metadata(h5: h5py.File) -> dict[str, Any]:
    raw = h5.attrs.get("case_yaml", "")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    case = yaml.safe_load(raw) if raw else {}
    solver = case.get("solver") or {}
    return {
        "case": case.get("name"),
        "backend": solver.get("backend", "numpy"),
        "device": solver.get("device"),
        "precision": solver.get("precision"),
    }


#: reductions that read a SECOND monitor (`against`): the load at a given
#: deflection, the area under a load-deflection record, the deflection at
#: the load peak. A contract that held only a peak could not tell a curve
#: with the right peak and the wrong shape from the right one.
PAIRED_REDUCTIONS = ("at", "energy", "at_peak_of")


def _reduce(values: np.ndarray, operation: str) -> float:
    a = np.asarray(values, dtype=float)
    if operation == "final": return float(a.reshape(-1)[-1])
    if operation == "first": return float(a.reshape(-1)[0])
    # what the LAST step added: the live load's share of a cable's tension
    # over the dead load's, or a deflection measured from the built state
    if operation == "delta": return float(a.reshape(-1)[-1] - a.reshape(-1)[0])
    if operation == "peak_abs": return float(np.max(np.abs(a)))
    if operation == "max": return float(np.max(a))
    if operation == "min": return float(np.min(a))
    if operation == "mean": return float(np.mean(a))
    if operation in PAIRED_REDUCTIONS:
        raise ValueError(f"reduction {operation!r} needs `against: <monitor>`")
    raise ValueError(f"unknown reduction {operation!r}")


def _reduce_against(values: np.ndarray, other: np.ndarray, spec: dict[str, Any],
                    other_name: str) -> float:
    """`values` reduced with respect to the record `other` of the same length.

    at          this monitor, linearly interpolated, where `other` FIRST
                reaches `value` (signed, as the monitor records it)
    energy      trapezoidal integral of this monitor over `other` -- the area
                under a load-deflection curve, in the units of their product.
                Reported as a magnitude, since a reaction and a deflection
                usually carry opposite signs; `signed: true` keeps the sign
                (a cyclic record's net dissipation)
    at_peak_of  this monitor at the frame where |other| is largest

    `from_origin: true` prepends (0, 0) to both records: monitors hold only
    the converged frames, so a load-deflection curve's first point is the
    first load step, not the origin. It is opt-in because a record that
    starts under a previous step's load does not start at zero.
    """
    y = np.asarray(values, dtype=float).reshape(-1)
    x = np.asarray(other, dtype=float).reshape(-1)
    if y.shape != x.shape:
        raise ValueError(f"monitor {spec['name']!r} has {y.size} frames and "
                         f"{other_name!r} has {x.size}; they must be the same record")
    if spec.get("from_origin"):
        x, y = np.concatenate([[0.0], x]), np.concatenate([[0.0], y])
    op = spec["reduction"]
    if op == "energy":
        area = float(trapezoid(y, x))
        return area if spec.get("signed") else abs(area)
    if op == "at_peak_of":
        return float(y[int(np.argmax(np.abs(x)))])
    # at
    if "value" not in spec:
        raise ValueError("reduction `at` needs `value`, the level of `against` to read at")
    v = float(spec["value"])
    if x.size and x[0] == v:
        return float(y[0])
    for i in range(x.size - 1):
        lo, hi = min(x[i], x[i + 1]), max(x[i], x[i + 1])
        if lo <= v <= hi and hi > lo:
            t = (v - x[i]) / (x[i + 1] - x[i])
            return float(y[i] + t * (y[i + 1] - y[i]))
    raise ValueError(
        f"monitor {other_name!r} never reaches {v:g}: its record runs "
        f"{x.min():g}..{x.max():g}" + ("" if spec.get("from_origin") else
                                       " (from_origin: true would include the origin)"))


def _monitor(h5: h5py.File, name: str, prefer_diagnostics: bool = False) -> np.ndarray:
    path = f"monitors/{name}"
    alt = f"diagnostics/{name}"
    if prefer_diagnostics and alt in h5 and isinstance(h5[alt], h5py.Dataset):
        # `source: diagnostic` means the energy audit: `kinetic_energy` is
        # also the name of the progress-sampled series in monitors/, and that
        # one was returned
        return h5[alt][...]
    if path not in h5:
        # the energy audit (external_work, internal_work, kinetic_energy,
        # dissipated_energy, energy_imbalance, relative_energy_error) lives
        # under diagnostics/, one row per snapshot
        alt = f"diagnostics/{name}"
        if alt in h5 and isinstance(h5[alt], h5py.Dataset): return h5[alt][...]
        raise KeyError(path)
    return h5[path][...]


def _metric_value(h5: h5py.File, spec: dict[str, Any]) -> float:
    source = spec["source"]
    reduction = spec.get("reduction", "final")
    if source in ("monitor", "diagnostic"):
        diag = source == "diagnostic"
        values = _monitor(h5, spec["name"], diag)
        if reduction in PAIRED_REDUCTIONS:
            if "against" not in spec:
                raise ValueError(f"reduction {reduction!r} needs `against: <monitor>`")
            return _reduce_against(values, _monitor(h5, spec["against"], diag), spec, spec["against"])
        return _reduce(values, reduction)
    if source == "spectrum":
        # the peaks of a response-spectrum analysis, one
        # dataset per quantity in the result's `response_spectrum` group
        path = f"response_spectrum/{spec['name']}"
        if path not in h5: raise KeyError(path)
        # `reaction` is a node quantity here, not the element field
        # `nearest` took it for
        a = _located(h5, np.asarray(h5[path], dtype=float), spec,
                     node=spec["name"] in _SPECTRUM_NODE)
        a = np.asarray(a)
        if spec.get("component") is None and reduction in ("final", "first", "delta") and (
                a.ndim > 1 or (spec["name"] in _SPECTRUM_VECTOR and a.ndim == 1 and a.size == 3)):
            # flattened, `final` was the z component
            raise ValueError(f"spectrum quantity {spec['name']!r} is a vector: name a `component` "
                             f"(x, y or z) for a `{reduction}` reduction")
        return _reduce(a.reshape(-1), reduction)
    if source != "field":
        raise ValueError(f"unknown metric source {source!r}")
    steps = sorted(h5["steps"], key=int)   # names are counters, not fixed width
    selected = steps if spec.get("frame") == "all" else [steps[-1]]
    chunks = []
    for step in selected:
        path = f"steps/{step}/{spec['name']}"
        group = "steps"
        if path not in h5 and f"member_steps/{step}/{spec['name']}" in h5:
            # a coupled or shell_line model keeps its members' fields apart
            # (v1.0: a contract could not read them)
            path, group = f"member_steps/{step}/{spec['name']}", "member_steps"
        if path not in h5: raise KeyError(path)
        raw = np.asarray(h5[path], dtype=float)
        a = np.asarray(_located(h5, raw, spec, group, step))
        if raw.ndim > 1 and spec.get("component") is None and reduction in ("final", "first", "delta"):
            # flattened, `final` was the z of the last node
            # -- and, at a `location`, the z of that node
            raise ValueError(f"field {spec['name']!r} has {raw.shape[-1]} components: name a "
                             "`component` (x, y, z or an index) for a `" + reduction + "` reduction"
                             + (" at a location" if spec.get("location") else ""))
        chunks.append(a.reshape(-1))
    return _reduce(np.concatenate(chunks), reduction)


def _centroids(h5: h5py.File, group: str, step: str | None) -> np.ndarray | None:
    """Each element's centre, in the element order its fields are written
    in: the frame's own connectivity once members have split, the members'
    for a coupled model's member fields, the flat stream for a mixed mesh."""
    nodes = np.asarray(h5["mesh/nodes"], dtype=float)
    if step is not None and f"{group}/{step}/connectivity" in h5:
        conn = np.asarray(h5[f"{group}/{step}/connectivity"]).astype(np.int64)
        return nodes[conn].mean(axis=1)
    key = "members/elements" if group == "member_steps" else "mesh/elements"
    if key not in h5:
        return None
    el = np.asarray(h5[key])
    if el.ndim == 2:
        return nodes[el.astype(np.int64)].mean(axis=1)
    from .io.writer import mixed_centroids
    return mixed_centroids(nodes, el)


#: the response-spectrum quantities that are one row per node, and those
#: that are x, y, z vectors
_SPECTRUM_NODE = {"displacement", "rotation", "reaction"}
_SPECTRUM_VECTOR = {"displacement", "rotation", "reaction", "base_force"}


def _located(h5: h5py.File, a: np.ndarray, spec: dict[str, Any],
             group: str = "steps", step: str | None = None, node: bool | None = None) -> np.ndarray:
    """The entry `location` names (an index, or the node -- or, for an
    element field, the element -- nearest a point) and the `component` of it."""
    location = spec.get("location")
    if location:
        if "index" in location:
            a = a[int(location["index"])]
        elif "nearest" in location:
            from .io.writer import NODE_FIELDS
            nodes = np.asarray(h5["mesh/nodes"], dtype=float)
            point = np.asarray(location["nearest"], dtype=float)
            centres = None
            if (spec.get("name") not in NODE_FIELDS) if node is None else not node:
                centres = _centroids(h5, group, step)
                if centres is not None and len(centres) != len(a):
                    centres = None
            if centres is None and len(a) != len(nodes):
                raise ValueError(f"field {spec.get('name')!r}: `nearest` cannot place its "
                                 f"{len(a)} entries; use `index`")
            # an element field was indexed by the nearest NODE's number
            where = centres if centres is not None else nodes
            a = a[int(np.argmin(np.linalg.norm(where - point, axis=1)))]
        else:
            raise ValueError("field location requires index or nearest")
    component = spec.get("component")
    if component is not None:
        j = _COMPONENT.get(component, component)
        a = np.asarray(a)[..., int(j)]
    return a


def _judge(value: float, expect: dict[str, Any]) -> tuple[bool, float | None]:
    if not np.isfinite(value): return False, None
    if expect.get("finite") is True and len(expect) == 1: return True, None
    if "upper" in expect and value > float(expect["upper"]): return False, value - float(expect["upper"])
    if "lower" in expect and value < float(expect["lower"]): return False, float(expect["lower"]) - value
    if "target" in expect:
        target = float(expect["target"])
        error = abs(value - target)
        limit = float(expect.get("atol", 0.0)) + float(expect.get("rtol", 0.0)) * abs(target)
        return error <= limit, error
    return True, None


def verify(contract_path: str | Path, result_path: str | Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    checks = []
    with open_result(result_path, "r") as h5:
        execution = _case_metadata(h5)
        if execution["case"] != contract["case"]:
            raise ValueError(f"result case {execution['case']!r} does not match contract {contract['case']!r}")
        analysis_valid = bool(h5.attrs.get("analysis_converged", True))
        analysis_status = str(h5.attrs.get("analysis_status", "legacy_result"))
        for spec in contract["metrics"]:
            try:
                value = _metric_value(h5, spec)
                passed, error = _judge(value, spec["expect"])
                message = None
            except (KeyError, ValueError, IndexError) as exc:
                value, error, passed, message = None, None, False, str(exc)
            checks.append({"id": spec["id"], "description": spec.get("description", ""),
                           "value": value, "expect": spec["expect"], "error": error,
                           "status": "PASS" if passed else "FAIL", "message": message})
    if not analysis_valid:
        checks.insert(0, {"id": "analysis_status", "description": "result must be converged/completed",
                         "value": None, "expect": {"converged": True}, "error": None,
                         "status": "FAIL", "message": analysis_status})
    return {"schema_version": 1, "case": contract["case"],
            "classification": contract["classification"],
            "reference": contract.get("reference"), "result": str(result_path),
            "execution": execution, "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
            "checks": checks}


def markdown(report: dict[str, Any]) -> str:
    e = report["execution"]
    lines = [f"# Verification: {report['case']}", "",
             f"**Status:** {report['status']}  ",
             f"**Classification:** {report['classification']}  ",
             f"**Execution:** backend={e['backend']}, device={e['device']}, precision={e['precision']}", "",
             "| Check | Value | Requirement | Status |", "|---|---:|---|:---:|"]
    for c in report["checks"]:
        value = "—" if c["value"] is None else f"{c['value']:.8g}"
        req = json.dumps(c["expect"], ensure_ascii=False)
        lines.append(f"| {c['id']} | {value} | `{req}` | {c['status']} |")
    if report.get("reference"):
        lines += ["", f"Reference: {report['reference']}"]
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    jp, mp = Path(json_path), Path(markdown_path)
    jp.parent.mkdir(parents=True, exist_ok=True); mp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    mp.write_text(markdown(report), encoding="utf-8")
