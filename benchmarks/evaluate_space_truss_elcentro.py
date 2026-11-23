"""Independent linear-Newmark check for the El Centro space-truss example.

The structural output is compared with a separate three-DOF apex model, not
with the same explicit force loop.  The public ground-motion record validates
the input; this script validates the linear structural response independently.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def reference(case_path):
    with open(case_path, encoding="utf-8") as fh:
        case = yaml.safe_load(fh)
    mesh_path = case["mesh"]["path"]
    if not mesh_path.startswith(("/", "\\")):
        mesh_path = os.path.join(os.path.dirname(case_path), mesh_path)
    # The example's 5-node/4-leg geometry is purposely simple enough that its
    # free apex has exactly three translational DOFs.
    from vfife.io import gmsh
    mesh, _ = gmsh.read(mesh_path)
    x, conn = mesh.nodes, mesh.elements
    apex = int(mesh.regions["apex"][0])
    legs = conn[np.any(conn == apex, axis=1)]
    E, rho, A = (float(case["material"]["E"]), float(case["material"]["rho"]),
                 float(case["element"]["area"]))
    K = np.zeros((3, 3)); m = 0.0
    for e in legs:
        other = e[0] if e[1] == apex else e[1]
        d = x[apex] - x[other]; L = np.linalg.norm(d); n = d / L
        K += E * A / L * np.outer(n, n)
        m += rho * A * L / 2.0
    for lump in case.get("nodal_masses") or []:
        ids = mesh.regions[lump.get("region", "all")]
        if apex in ids:
            m += float(lump["mass"]) if lump.get("per_node") else float(lump["mass"]) / len(ids)
    gm = case["loads"][0]["ground_motion"]
    record = np.loadtxt(os.path.join(os.path.dirname(case_path), gm["path"]), dtype=float, encoding="utf-8")
    direction = np.asarray(gm["direction"], dtype=float)
    dt = case["solver"]["dt_safety"] * min(np.linalg.norm(x[e[1]]-x[e[0]]) for e in legs) / np.sqrt(E/rho)
    total = case["solver"]["total_time"]; steps = int(round(total / dt))
    beta = 0.25; gamma = 0.5
    M = np.eye(3) * m; u = np.zeros(3); v = np.zeros(3)
    damp = case["solver"].get("damping") or {}
    C = float(damp.get("alpha", 0.0)) * M + float(damp.get("beta", 0.0)) * K
    force = lambda t: -m * np.interp(t, record[:, 0], record[:, 1] * gm["scale"], left=0, right=0) * direction
    a = np.linalg.solve(M, force(0.0) - C @ v - K @ u)
    peak = 0.0
    for i in range(1, steps + 1):
        up = u + dt*v + dt*dt*(0.5-beta)*a
        vp = v + dt*(1-gamma)*a
        anew = np.linalg.solve(M + gamma*dt*C + beta*dt*dt*K,
                               force(i*dt) - C @ vp - K @ up)
        u = up + beta*dt*dt*anew
        v = vp + gamma*dt*anew
        a = anew
        peak = max(peak, abs(u[0]))
    return peak, dt, m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("result", nargs="?", default="results/case_05_space_truss_seismic.h5")
    ap.add_argument("--case", default="cases/case_05_space_truss_seismic.yaml")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)
    ref, dt, mass = reference(args.case)
    with h5py.File(args.result) as f:
        frames = [f["steps"][k]["displacement"][4, 0] for k in sorted(f["steps"], key=int)]
    got = float(np.max(np.abs(frames)))
    report = {"reference": "independent linear Newmark-beta (average acceleration)",
              "physical_apex_mass_kg": mass, "time_step_s": dt,
              "vfife_peak_apex_ux_m": got, "newmark_peak_apex_ux_m": ref,
              "relative_difference_percent": abs(got-ref)/ref*100}
    print(json.dumps(report, indent=2))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh: json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
