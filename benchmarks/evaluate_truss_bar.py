"""Evaluate the analytical standalone-truss bar benchmark result."""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)
    with h5py.File(args.result, "r") as f:
        key = sorted(f["steps"], key=int)[-1]
        u = np.asarray(f["steps"][key]["displacement"])
        stress = float(np.asarray(f["steps"][key]["axial_stress"])[0])
        force = float(np.asarray(f["steps"][key]["axial_force"])[0])
    report = {
        "loaded_ux_m": float(u[1, 0]), "target_ux_m": 5e-5,
        "axial_stress_pa": stress, "target_axial_stress_pa": 1e7,
        "axial_force_n": force, "target_axial_force_n": 1000.0,
    }
    report["relative_error_percent"] = abs(report["loaded_ux_m"] - 5e-5) / 5e-5 * 100
    print(json.dumps(report, indent=2))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as out:
            json.dump(report, out, indent=2)


if __name__ == "__main__":
    main()
