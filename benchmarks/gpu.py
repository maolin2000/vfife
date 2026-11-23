"""GPU throughput and the cost of 10^6 elements.

    python benchmarks/gpu.py                  # correctness + scaling sweep
    python benchmarks/gpu.py --million        # measure 10^6 elements directly
    python benchmarks/gpu.py --device cpu     # same kernel source, CPU target

WHAT THIS MEASURES, AND WHY IT IS BUILT THIS WAY
------------------------------------------------
The first version of this script reported one un-repeated timing per
configuration at 19 200 elements, and both of those choices produced misleading
numbers on a real card:

* **19 200 elements does not fill a GPU.** One thread per element is ~4 waves on
  a 4608-core part, so the kernel is latency-bound and every material comes out
  at the same time regardless of how much arithmetic it does. Measured on an
  RTX 5060 Ti: linear elastic 0.130 ms, Chaboche 0.131 ms, CDP 0.126 ms -- three
  materials with very different cost, indistinguishable, which is the signature
  of a card sitting idle. Extrapolating 10^6 elements from there is
  extrapolating from the wrong regime, and it lands pessimistic.
* **A single timing hides outliers.** The same run produced 5.83 ms for
  b-bar/Chaboche where b-bar/CDP -- a strictly more expensive material -- took
  0.405 ms. A 14x inversion like that is either a real effect (register
  spilling) or a one-off (clock drop, scheduler, cache load). One measurement
  cannot tell you which, so this version takes the minimum of several repeats
  and prints the spread.

So: a global warm-up before anything is timed, several repeats per point, the
minimum reported (the right statistic for throughput -- noise only ever adds
time), the spread shown so an unstable point is visible, and a scaling column so
you can see whether the card is saturated before trusting any projection.

`--million` skips the projection entirely and measures the real thing.
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
from vfife.core.mesh import box_mesh, scatter_add
from vfife.elements import continuum
from vfife.materials import create

MATLAB_BASELINE = 600 * 5900 / 659.0 / 1e6      # M element-steps/s

# roughly cube-ish meshes of a 12 x 0.5 x 1 m beam at increasing resolution
SWEEP = [(60, 2, 5), (120, 8, 20), (240, 16, 40)]
MILLION = (400, 25, 100)                         # 1 000 000 elements


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


def build(divs, formulation, kind, backend, **kw):
    mesh = box_mesh((12.0, 0.5, 1.0), divs)
    mat = make_material(kind)
    pre = continuum.precompute(mesh.nodes, mesh.elements,
                           formulation=formulation, material=mat)
    x0 = mesh.nodes[mesh.elements]
    x0_e = np.ascontiguousarray(x0 - x0.mean(axis=1, keepdims=True))
    mat.bind_points(np.repeat(pre.char_len, pre.n_gp))
    force = kernels.make_force(pre, mesh.elements, x0_e, mat, backend, **kw)
    u = np.zeros((mesh.n_nodes, 3))
    u[:, 2] = -3e-3 * mesh.nodes[:, 0] ** 2
    return mesh, pre, mat, force, u


# --------------------------------------------------------------------------- #


def check(device):
    """Same answer on this device as on the NumPy reference?"""
    print("correctness on this device (fp64, vs the NumPy reference)")
    worst = 0.0
    for kind in ("elastic", "chaboche", "cdp"):
        for form in ("reduced", "b-bar"):
            mesh, pre, mat, f_np, u = build((10, 3, 3), form, kind, "numpy")
            _, _, mat2, f_wp, _ = build((10, 3, 3), form, kind, "warp",
                                        device=device, precision="fp64")
            st1 = mat.init_state(mesh.n_elem * pre.n_gp)
            st2 = mat2.init_state(mesh.n_elem * pre.n_gp)
            q1 = np.zeros((mesh.n_elem, 4)); q1[:, 0] = 1.0
            q2 = q1.copy()
            a, _ = f_np(u, q1, st1, 1e-5, rot_iters=30)
            b, _ = f_wp(u, q2, st2, 1e-5, rot_iters=30)
            A = scatter_add(a, mesh.elements, mesh.n_nodes)
            B = f_wp.assemble(b)
            rel = np.abs(A - B).max() / max(np.abs(A).max(), 1e-30)
            worst = max(worst, rel)
            print(f"  {kind:>8s} {form:>8s}: {rel:.2e}  "
                  f"{'ok' if rel < 1e-9 else 'MISMATCH'}")
    print(f"  worst: {worst:.2e}\n")


def time_one(force, mesh, repeats, iters):
    """Minimum wall time per step over `repeats` runs of `iters` steps."""
    import warp as wp

    force.set_nodal(np.ones((mesh.n_nodes, 3)),
                    np.zeros((mesh.n_nodes, 3), dtype=np.int8), [], [])
    # warm up properly: compile, allocate, settle clocks
    for _ in range(5):
        force.force_resident(1e-5, rot_iters=2)
        force.integrate(1.0, 1e-5, 1.0, 1.0, 0)
    wp.synchronize_device(force.device)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            force.force_resident(1e-5, rot_iters=2)
            force.integrate(1.0, 1e-5, 1.0, 1.0, 0)
        wp.synchronize_device(force.device)
        times.append((time.perf_counter() - t0) / iters)
    return min(times), max(times) / min(times)


def bench(device, precision, sizes, repeats, iters, forms, kinds):
    print(f"throughput on '{device}', precision = {precision}, "
          f"min of {repeats} x {iters} steps")
    print("(force + assembly + integrator per step; nothing crosses the bus)\n")
    print(f"{'elements':>9} {'formulation':>11} {'material':>9} {'ms/step':>9} "
          f"{'M el-st/s':>10} {'vs MATLAB':>10} {'spread':>7} {'MB':>7}")
    print("-" * 80)
    table = {}
    for divs in sizes:
        for form in forms:
            for kind in kinds:
                mesh, pre, mat, force, u = build(
                    divs, form, kind, "warp", device=device, precision=precision)
                force.set_u(u)
                dt, spread = time_one(force, mesh, repeats, iters)
                rate = mesh.n_elem / dt / 1e6
                flag = "  <-- unstable" if spread > 1.5 else ""
                print(f"{mesh.n_elem:>9d} {form:>11s} {kind:>9s} {dt * 1e3:>9.3f} "
                      f"{rate:>10.2f} {rate / MATLAB_BASELINE:>9.0f}x "
                      f"{spread:>6.2f}x {force.bytes_resident() / 1e6:>7.1f}"
                      f"{flag}")
                table.setdefault((form, kind), []).append(
                    (mesh.n_elem, rate, force.bytes_resident() / mesh.n_elem))
                del force
        print()
    return table


def report_scaling(table):
    """Is the card saturated? If throughput still climbs, it is not."""
    print("scaling of throughput with problem size")
    print("(a card that is already full shows a flat rate; a rising rate means")
    print(" the smaller meshes were latency-bound and any projection from them")
    print(" is too pessimistic)\n")
    for (form, kind), rows in sorted(table.items()):
        if len(rows) < 2:
            continue
        rates = "  ".join(f"{n / 1000:.0f}k: {r:6.1f}" for n, r, _ in rows)
        growth = rows[-1][1] / rows[0][1]
        note = "saturated" if growth < 1.25 else f"still climbing ({growth:.1f}x)"
        print(f"  {form:>8s} {kind:>9s}  {rates}   -> {note}")
    print()


def project(table, steps):
    print(f"projection to 10^6 elements (from the LARGEST mesh measured above)")
    print(f"{'formulation':>11} {'material':>9} {'s/step':>9} {'GB':>7} "
          f"{str(steps) + ' steps':>13}")
    print("-" * 56)
    for (form, kind), rows in sorted(table.items()):
        n, rate, per_el = rows[-1]
        s_step = 1.0 / rate
        total = s_step * steps
        h = f"{total / 3600:.1f} h" if total > 3600 else f"{total / 60:.1f} min"
        print(f"{form:>11s} {kind:>9s} {s_step:>9.4f} {per_el * 1e6 / 1e9:>7.2f} "
              f"{h:>13s}")
    print("\nRun with --million to measure 10^6 directly instead of extrapolating.")


def million(device, precision, repeats, iters):
    print("measuring 10^6 elements directly (reduced integration)\n")
    print("host RAM note: the precompute peaks around 1.5-2 GB before anything")
    print("reaches the device.\n")
    print(f"{'elements':>9} {'material':>9} {'ms/step':>9} {'M el-st/s':>10} "
          f"{'spread':>7} {'GB':>6} {'70k steps':>11}")
    print("-" * 68)
    for kind in ("elastic", "chaboche", "cdp"):
        mesh, pre, mat, force, u = build(MILLION, "reduced", kind, "warp",
                                         device=device, precision=precision)
        force.set_u(u)
        dt, spread = time_one(force, mesh, repeats, iters)
        rate = mesh.n_elem / dt / 1e6
        total = dt * 70000
        h = f"{total / 3600:.1f} h" if total > 3600 else f"{total / 60:.1f} min"
        print(f"{mesh.n_elem:>9d} {kind:>9s} {dt * 1e3:>9.3f} {rate:>10.2f} "
              f"{spread:>6.2f}x {force.bytes_resident() / 1e9:>6.2f} {h:>11s}")
        del force
    print("\nThe step count assumes physical-time dynamics; a quasi-static run")
    print("in relaxation mode with staged loading needs far fewer.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="mixed", choices=("mixed", "fp64"))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--steps", type=int, default=70000,
                    help="step count used for the projected run time")
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--million", action="store_true",
                    help="measure 10^6 elements directly (reduced only)")
    ap.add_argument("--small", action="store_true",
                    help="stop at 19k elements (quick, but too small to saturate)")
    ap.add_argument("--only", default=None,
                    help="restrict to one material, e.g. --only chaboche")
    args = ap.parse_args()

    import warp as wp
    wp.init()
    dev = wp.get_device(args.device)
    print(f"device: {getattr(dev, 'name', dev)}  (cuda: {dev.is_cuda})")
    if dev.is_cuda:
        print(f"        {dev.total_memory / 1e9:.1f} GB total, "
              f"{dev.free_memory / 1e9:.1f} GB free")
    else:
        print("        NOT a GPU -- these numbers say nothing about CUDA")
        print("        throughput, only that the same kernel source runs and")
        print("        agrees with the reference.")
    print(f"MATLAB baseline: {MATLAB_BASELINE:.4f} M element-steps/s\n")

    if not args.skip_check:
        check(args.device)

    kinds = (args.only,) if args.only else ("elastic", "chaboche", "cdp")
    if args.million:
        million(args.device, args.precision, args.repeats, args.iters)
        return

    sizes = SWEEP[:2] if args.small else SWEEP
    table = bench(args.device, args.precision, sizes, args.repeats, args.iters,
                  ("reduced", "b-bar"), kinds)
    report_scaling(table)
    project(table, args.steps)


if __name__ == "__main__":
    main()
