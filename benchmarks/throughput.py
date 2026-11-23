"""Element-kernel throughput, both backends, and the projection to 10^6 elements.

    python benchmarks/throughput.py [--big]

The MATLAB original managed 600 elements x 5900 steps in 659 s = 0.0054 M
element-steps/s on the same problem; every row is measured against that.

A "step" here is force evaluation plus nodal assembly, i.e. everything the time
loop does per iteration except the (trivial) integrator update.
"""

import os
import sys

# Run me directly from anywhere: as a script, only this file's own directory
# lands on sys.path, not the repo root, so `import vfife` would need an install
# or PYTHONPATH without this.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import numpy as np

from vfife import kernels
from vfife.core.mesh import box_mesh
from vfife.elements import continuum
from vfife.materials import create

MATLAB_BASELINE = 600 * 5900 / 659.0 / 1e6      # M element-steps/s


def make_material(kind):
    if kind == "elastic":
        return create("linear_elastic", E=3e10, nu=0.2, rho=2400.0)
    if kind == "chaboche":
        return create("chaboche", E=3e10, nu=0.2, rho=2400.0,
                      yield_strength=3e6, Q_inf=1e6, b=20.0, H_iso=1e8,
                      C1=1e9, gamma1=200.0, C2=2e8, gamma2=20.0, n_newton=2)
    return create("cdp", E=3e10, nu=0.2, rho=2400.0, l_ref=0.02,
                  compression={"source": "gb50010", "fc": 30e6},
                  tension={"source": "fracture_energy", "ft": 3e6, "Gf": 120.0})


def bench(divs, formulation, kind, backend, n=15):
    mesh = box_mesh((12.0, 0.5, 1.0), divs)
    mat = make_material(kind)
    pre = continuum.precompute(mesh.nodes, mesh.elements,
                           formulation=formulation, material=mat)
    x0 = mesh.nodes[mesh.elements]
    x0_e = np.ascontiguousarray(x0 - x0.mean(axis=1, keepdims=True))
    mat.bind_points(np.repeat(pre.char_len, pre.n_gp))
    force = kernels.make_force(pre, mesh.elements, x0_e, mat, backend)

    state = mat.init_state(mesh.n_elem * pre.n_gp)
    quat = np.zeros((mesh.n_elem, 4))
    quat[:, 0] = 1.0
    u = np.zeros((mesh.n_nodes, 3))
    u[:, 2] = -3e-3 * mesh.nodes[:, 0] ** 2

    t0 = time.perf_counter()
    f_e, _ = force(u, quat, state, 1e-5, rot_iters=30)
    force.assemble(f_e)
    warm = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n):
        f_e, _ = force(u, quat, state, 1e-5, rot_iters=2)
        force.assemble(f_e)
    return mesh.n_elem, (time.perf_counter() - t0) / n, warm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true",
                    help="include a 150k-element mesh")
    args = ap.parse_args()

    sizes = [(60, 2, 5), (120, 8, 20)]
    if args.big:
        sizes.append((240, 16, 40))

    print(f"MATLAB baseline: {MATLAB_BASELINE:.4f} M element-steps/s")
    print("(force evaluation + nodal assembly, per step)\n")
    print(f"{'elements':>9} {'formulation':>11} {'material':>9} {'numpy ms':>9} "
          f"{'numba ms':>9} {'gain':>6} {'M el-st/s':>10} {'vs MATLAB':>10}")
    print("-" * 82)
    for divs in sizes:
        for form in ("reduced", "b-bar"):
            for kind in ("elastic", "chaboche", "cdp"):
                nel, t_np, _ = bench(divs, form, kind, "numpy")
                _, t_nb, warm = bench(divs, form, kind, "numba")
                rate = nel / t_nb / 1e6
                print(f"{nel:>9d} {form:>11s} {kind:>9s} {t_np * 1e3:>9.2f} "
                      f"{t_nb * 1e3:>9.2f} {t_np / t_nb:>5.1f}x {rate:>10.3f} "
                      f"{rate / MATLAB_BASELINE:>9.0f}x")
        print()

    print("One-time JIT cost per (material, formulation) pair, first call only:")
    for kind in ("elastic", "cdp"):
        _, _, warm = bench((60, 2, 5), "reduced", kind, "numba", n=1)
        print(f"  {kind:>9s}: {warm:5.1f} s")


if __name__ == "__main__":
    main()
