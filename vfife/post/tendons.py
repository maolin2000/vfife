"""The tendons of a result: what `tendons:` expanded to, and
the force in each tendon at the end of each step.

`read(path)` gives the tendons as the result's `tendons` group holds them;
`table(path)` one row per tendon and step -- the force along the tendon at
the step's last frame, least, mean and largest over its bars, and the mean
against the force the bars were jacked to; `bars(path)` one row per bar and
step. `vfife tendons result.h5` prints the table (`--csv` writes it)."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np


def read(path: str) -> List[Dict[str, Any]]:
    """[{name, count, area, jacking, stress_step, regions, elements, jack}]."""
    from ..io.writer import open_result
    with open_result(path) as f:
        if "tendons" not in f:
            return []
        g = f["tendons"]
        out = []
        for key in sorted(g):
            t = g[key]
            out.append({"name": str(t.attrs["name"]), "count": int(t.attrs["count"]),
                        "area": float(t.attrs["area"]), "jacking": [float(v) for v in t.attrs["jacking"]],
                        "stress_step": int(t.attrs["stress_step"]),
                        "regions": json.loads(t.attrs["regions_json"]),
                        "elements": np.asarray(t["elements"][()], dtype=np.int64),
                        "jack": np.asarray(t["jack"][()], dtype=np.float64)})
        return out


def _step_names(f) -> List[str]:
    try:
        return list(json.loads(f["tendons"].attrs["step_names_json"]))
    except (KeyError, ValueError):
        return []


def _last_frames(f):
    """[(analysis step, frame key, time)]: each step's last frame with axial forces."""
    last: Dict[int, Any] = {}
    for key in sorted(f["steps"], key=int):
        fr = f["steps"][key]
        if "axial_force" not in fr:
            continue
        k = int(fr.attrs.get("analysis_step", 0))
        last[k] = (k, key, float(fr.attrs.get("time", 0.0)))
    return [last[k] for k in sorted(last)]


def bars(path: str) -> List[Dict[str, Any]]:
    """One row per bar and step: tendon, bar (1 from its beginning), region,
    step, time, force, jack. A step's row is its last frame: the end of
    the step, or where a run that stopped part-way through it stopped."""
    from ..io.writer import open_result
    tendons = read(path)
    if not tendons:
        return []
    with open_result(path) as f:
        names = _step_names(f)
        frames = _last_frames(f)
        if not frames:
            status = str(f.attrs.get("analysis_status", "UNKNOWN"))
            if not len(f["steps"]):
                raise ValueError(f"{path}: the run wrote no frame (its status: {status})")
            raise ValueError(f"{path}: no frame has `axial_force` (add it to `output.fields`)")
        out = []
        for k, key, t in frames:
            force = np.asarray(f["steps"][key]["axial_force"][()], dtype=np.float64)
            step = names[k] if k < len(names) else f"step {k + 1}"
            for td in tendons:
                built = k >= td["stress_step"]
                for i, (row, region, jack) in enumerate(zip(td["elements"], td["regions"], td["jack"])):
                    out.append({"tendon": td["name"], "bar": i + 1, "region": region, "step": step,
                                "step_index": k, "time": t, "force": float(force[row]) if built else None,
                                "jack": float(jack)})
        return out


def table(path: str) -> List[Dict[str, Any]]:
    """One row per tendon and step: least, mean and largest force over its
    bars at the step's last frame, and the mean against the mean jacking
    force. Before the step that stresses it a tendon has no force (None)."""
    rows: Dict[tuple, Dict[str, Any]] = {}
    for b in bars(path):
        r = rows.setdefault((b["step_index"], b["tendon"]),
                            {"tendon": b["tendon"], "step": b["step"], "step_index": b["step_index"],
                             "time": b["time"], "forces": [], "jacks": []})
        if b["force"] is not None:
            r["forces"].append(b["force"])
        r["jacks"].append(b["jack"])
    out = []
    for r in rows.values():
        fs, js = r.pop("forces"), r.pop("jacks")
        if fs:
            mean = float(np.mean(fs))
            r.update(least=float(np.min(fs)), mean=mean, largest=float(np.max(fs)),
                     of_jack=mean / float(np.mean(js)) if np.mean(js) else None)
        else:
            r.update(least=None, mean=None, largest=None, of_jack=None)
        out.append(r)
    return out


def text(rows: List[Dict[str, Any]], unit: str = "") -> str:
    """The table as text, a block per step."""
    if not rows:
        return "no tendons (the case has no `tendons:`, or the result was written before v1.20)"
    lines = []
    width = max(len(r["tendon"]) for r in rows)
    for k in sorted({r["step_index"] for r in rows}):
        sub = [r for r in rows if r["step_index"] == k]
        lines.append(f"step {k + 1}: {sub[0]['step']}  (at {sub[0]['time']:g})")
        lines.append(f"  {'tendon':<{width}}  {'least':>12}  {'mean':>12}  {'largest':>12}  {'mean/jack':>9}"
                     + (f"   [{unit}]" if unit else ""))
        for r in sub:
            if r["mean"] is None:
                lines.append(f"  {r['tendon']:<{width}}  {'(not yet stressed)':>40}")
            else:
                lines.append(f"  {r['tendon']:<{width}}  {r['least']:12.6g}  {r['mean']:12.6g}  "
                             f"{r['largest']:12.6g}  {r['of_jack']:9.4f}")
    return "\n".join(lines)
