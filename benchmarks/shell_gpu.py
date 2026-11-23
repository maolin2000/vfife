"""The shell4 backends against each other: numpy, numba, warp.

    python benchmarks/shell_gpu.py                    # sweep + case_17 on the device
    python benchmarks/shell_gpu.py --device cpu       # warp's CPU target, for a machine without a card
    python benchmarks/shell_gpu.py --no-case          # the sweep only
    python benchmarks/shell_gpu.py --n 16 32 64 128   # mesh sizes (elements = n*n)

WHAT IT MEASURES
----------------
Per iteration of the staged relaxation -- the whole thing, not the element
force alone: force, integrator, boundary conditions and the convergence
measures -- on the Sze roof at n x n elements, for each backend. The warp
column is the device-resident loop (`WarpShell.force_resident / integrate /
measures`), which is what a run actually does; the host columns drive the same
steps through the numpy/numba force.

Like benchmarks/gpu.py: a warm-up first (Warp compiles the module on the
first launch, tens of seconds on a first run of a machine), several repeats,
the MINIMUM reported and the spread shown. On a GPU the per-iteration cost
is launch-bound -- about eight launches and one synchronising readback -- so
expect it to be nearly flat in the element count until the card fills, and
do not size a run from the smallest mesh.

Then, unless --no-case, cases/case_17 itself on the device at `precision:
mixed`, with the contract applied to the result: the number that matters is
not the speed but whether the curve is still inside
verification/case_17_hinged_cylindrical_roof.yaml.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pathlib
import tempfile
import time

import numpy as np

from benchmarks.generate_sze_shells import roof
from vfife.api import prepare_case, run_case
from vfife.cli import _with_element, make_solver
from vfife.verification import verify


def roof_problem(n, backend, device, precision, target=-0.5):
    d = pathlib.Path(tempfile.mkdtemp())
    roof(d / "roof.msh", n)
    return {
        "name": "roof_bench", "units": {"length": "mm", "force": "N", "stress": "MPa"},
        "mesh": {"type": "gmsh", "path": str(d / "roof.msh")},
        "materials": {"roof": {"type": "linear_elastic", "E": 3102.75, "nu": 0.3, "rho": 3.1e-9}},
        "sections": {"shell": {"type": "shell", "material": "roof", "thickness": 12.7,
                               "shear_correction": 0.8333333333, "drilling_factor": 1.0,
                               "hourglass_coeff": 0.1, "formulation": "selective"}},
        "assign": [{"region": "all", "section": "shell"}],
        "constraints": [{"region": "sym_x", "dofs": [1, 0, 0, 0, 1, 1]},
                        {"region": "sym_y", "dofs": [0, 1, 0, 1, 0, 1]},
                        {"region": "hinged", "dofs": [1, 1, 1, 0, 0, 0]}],
        "steps": [{"name": "push", "constraints": [
            {"region": "crown", "dofs": [0, 0, 1, 0, 0, 0], "value": target}],
            "load_steps": 2, "max_steps_per_load": 20000, "tolerance": 1.0e-5}],
        "solver": {"mode": "relaxation", "backend": backend, "device": device, "precision": precision,
                   "n_steps": 40000, "damping": {"type": "kinetic"}, "tolerance": 1.0e-5},
        "output": {"path": str(d / "r.h5"), "every": 10**7,
                   "monitors": {"w_C": {"kind": "displacement", "region": "crown", "dof": "z"},
                                "P4": {"kind": "reaction", "region": "crown", "dof": "z"}}},
    }


def solver_for(problem):
    pr = prepare_case(problem)
    return make_solver(_with_element(pr.case, pr.element, pr.case_path), pr.mesh, pr.material)


def time_host(s, iters, repeats):
    """One iteration of BeamSolver's staged loop, with the numpy/numba force."""
    fixed, moving, prescribed, groups, ground = s.step_data(0)
    F = np.zeros(s.ndof); free = ~fixed
    starts = [np.zeros(len(d)) for d, *_ in prescribed]
    s.internal_force()                                  # numba's compile / cache load is not the iteration
    best, worst = np.inf, 0.0
    for _ in range(repeats):
        s.q[:] = 0.0; s.v[:] = 0.0
        t = time.perf_counter()
        for it in range(1, iters + 1):
            fint, _ = s.internal_force()
            acc = (F - fint) / s.mass; acc[fixed] = 0.0
            s.v += acc; s.q += s.v
            s._apply_bc(fixed, moving, prescribed, 1.0, starts, min(1.0, it / 200))
            ke = float(0.5 * np.sum(s.mass * s.v * s.v))
            s._convergence_measures(F, free, ke, fint)
        dt = (time.perf_counter() - t) / iters
        best, worst = min(best, dt), max(worst, dt)
    return best, worst


def time_device(s, iters, repeats):
    """One iteration of ShellSolver._run_warp: force, integrate, measures."""
    fixed, moving, prescribed, groups, ground = s.step_data(0)
    ws = s._make_warp_shell(); ws.set_mass(s.mass)
    ws.set_nodal(fixed, groups, [(d, v, a) for d, v, a, _m in prescribed],
                 [np.zeros(len(d)) for d, *_ in prescribed])
    best, worst = np.inf, 0.0
    for _ in range(repeats):
        ws.set_u(np.zeros(s.ndof), np.zeros(s.ndof))
        ws.force_resident(); ws.integrate(1.0, 1, 0.0); ws.measures(1.0)      # warm, and the half step
        t = time.perf_counter()
        for it in range(1, iters + 1):
            ws.force_resident(); ws.integrate(1.0, 0, min(1.0, it / 200)); ws.measures(1.0)
        dt = (time.perf_counter() - t) / iters
        best, worst = min(best, dt), max(worst, dt)
    return best, worst


def sweep(sizes, device, iters, repeats):
    import warp as wp
    print(f"warp {wp.config.version} on {wp.get_device(device)}; {iters} iterations x {repeats} repeats, minimum shown\n")
    print(f"{'elements':>9} {'numpy':>12} {'numba':>12} {'warp mixed':>12} {'warp fp64':>12}   numba/warp")
    for n in sizes:
        row = []
        for backend, precision in (("numpy", "fp64"), ("numba", "fp64"), ("warp", "mixed"), ("warp", "fp64")):
            s = solver_for(roof_problem(n, backend, device, precision))
            if backend == "numpy" and n > 64:
                row.append(None); continue                       # 12 ms an iteration at 4096; not the point
            best, worst = (time_device if backend == "warp" else time_host)(s, iters, repeats)
            row.append((best, worst))
        cells = []
        for r in row:
            if r is None:
                cells.append(f"{'--':>16}"); continue
            b, w = r
            spread = f"±{100 * (w - b) / b:.0f}%" if w > 1.05 * b else ""
            cells.append(f"{1e6 * b:9.0f} us {spread:>5}")
        ratio = row[1][0] / row[2][0]
        print(f"{n * n:9d} " + " ".join(cells) + f"   {ratio:5.1f}x")
    print()


def case_on_device(device, precision, out):
    import yaml
    root = pathlib.Path(__file__).resolve().parent.parent
    case = yaml.safe_load((root / "cases/case_17_hinged_cylindrical_roof.yaml").read_text(encoding="utf-8"))
    case["solver"].update(backend="warp", device=device, precision=precision)
    case["mesh"]["path"] = str(root / "benchmarks/assets/sze_roof_16_v22.msh")
    case["output"]["path"] = str(out)
    print(f"case_17 on warp/{device}/{precision} ...")
    t = time.perf_counter()
    r = run_case(case)
    print(f"  {r.status}; {r.diagnostics['step']} iterations in {time.perf_counter() - t:.1f} s "
          f"({r.diagnostics['step'] / (time.perf_counter() - t):.0f} it/s); n_resets in diagnostics")
    rep = verify(root / "verification/case_17_hinged_cylindrical_roof.yaml", out)
    print(f"  contract {rep['status']}")
    for c in rep["checks"]:
        target = c["expect"].get("target")
        rel = "" if c["error"] is None or not target else f"  ({100 * c['error'] / abs(target):.2f}% of target {target})"
        print(f"    {c['status']}  {c['id']}: {c['value']}{rel}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-case", action="store_true")
    ap.add_argument("--precision", default="mixed")
    a = ap.parse_args()
    sweep(a.n, a.device, a.iters, a.repeats)
    if not a.no_case:
        out = pathlib.Path("results") / f"case_17_warp_{a.device.replace(':', '')}_{a.precision}.h5"
        out.parent.mkdir(exist_ok=True)
        case_on_device(a.device, a.precision, out)
