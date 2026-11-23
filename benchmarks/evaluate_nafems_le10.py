"""Evaluate the retained VFIFE NAFEMS LE10 result against its public target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

TARGET = -5.38e6
POINT_D = np.array([2.0, 0.0, 0.6])


def evaluate(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        nodes = np.asarray(h5["mesh/nodes"])
        elements = np.asarray(h5["mesh/elements"])
        step = sorted(h5["steps"], key=int)[-1]
        stress_name = "stress_nodal" if f"steps/{step}/stress_nodal" in h5 else "stress"
        stress = np.asarray(h5[f"steps/{step}/{stress_name}"])
        d = int(np.argmin(np.linalg.norm(nodes - POINT_D, axis=1)))
        incident = np.any(elements == d, axis=1)
        sigma_yy = (float(stress[d, 1]) if stress_name == "stress_nodal"
                    else float(stress[incident, 1].mean()))
        residual = float(h5["monitors/residual"][-1])
    return {
        "result": str(path), "step": int(step), "point_D_node": int(d),
        "point_D_coordinate": nodes[d].tolist(), "incident_elements": int(incident.sum()),
        "sigma_yy_pa": sigma_yy, "sigma_yy_mpa": sigma_yy / 1e6,
        "target_mpa": TARGET / 1e6,
        "relative_error_percent": abs(sigma_yy - TARGET) / abs(TARGET) * 100,
        "final_relative_residual": residual,
        "stress_recovery": ("volume-weighted linear Gauss-point patch recovery"
                            if stress_name == "stress_nodal" else
                            "mean of cell-centred stresses incident to D"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = evaluate(args.result)
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
