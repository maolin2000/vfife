"""Where does the GPU start to win? Numba and Warp on one axis.

    python benchmarks/crossover.py                 # the sweep
    python benchmarks/crossover.py --device cpu    # no card: numba vs warp-cpu
    python benchmarks/crossover.py --only elastic  # one material, faster

WHY THIS SCRIPT EXISTS
----------------------
`throughput.py` measures numpy and numba. `gpu.py` measures numpy and warp.
Neither puts numba and warp-cuda on the same axis, so the question that
actually decides a backend for a given job -- *at what problem size does the
card start to pay for itself?* -- had no measurement behind it. `cases/case_06`
hard-codes `device: cuda`, and every element now declares all three backends,
so the choice is real and needs a number, not a guess.

The answer is not "the GPU is faster". Below some element count it is slower,
because a kernel launch costs the same whether it does 500 elements or 500 000,
and a VFIFE step launches several. `gpu.py` documents the same effect from the
other side: at 19 200 elements three materials of very different arithmetic
cost all came out at 0.13 ms, which is what a card sitting idle looks like.
This script finds where the lines cross.

HOW IT MEASURES
---------------
Following the corrections already made in `gpu.py`, and for the same reasons:

* **Warm up before timing.** Numba pays a JIT cost on the first call and warp
  compiles its modules on first use -- on a new architecture (sm_120) that is
  minutes, not milliseconds. Timing a first call measures the compiler.
* **Minimum of several repeats**, with the spread printed. Noise only ever adds
  time, so the minimum is the right statistic; the spread makes an unstable
  point visible rather than silently averaged in.
* **Synchronize the device before stopping the clock.** CUDA launches are
  asynchronous; without this the GPU column measures how fast Python can queue
  work, which is a much prettier and entirely fictional number.
* **The sweep starts small on purpose**, around the size of the meshes in
  `tests/test_backends.py`, so the table covers the regime where a launch
  cannot be amortised as well as the one where it can.

A caution learned the hard way, on an RTX 5060 Ti: the first cuda run of that
suite took 3693 s and the second took 207 s, against 944 s for warp's CPU
target on the same machine. Ninety-four percent of the first number was
one-time sm_120 compilation into warp's kernel cache. Any timing taken before
that cache is warm measures the compiler, and a conclusion drawn from it will
be backwards -- it was, once, and the wrong conclusion was nearly written down
here.
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

# WHAT THIS SCRIPT MEASURES, AND WHAT IT MUST NOT BE USED FOR
#   It times ONE ELEMENT-KERNEL CALL. That is a real quantity and it is the
#   right one for judging a kernel. It is NOT the right one for choosing a
#   backend, and it was used that way: case_06 carried a long comment saying
#   "below ~1 300 elements use numba", derived from these numbers, and the
#   whole-step measurement puts the crossover at or below 256 elements
#   instead. On an RTX 5060 Ti at 240 elements the kernel is 0.045 ms while
#   the numba STEP is 0.522 ms -- nine tenths of a step is not the kernel, and
#   in the warp path that nine tenths is on the device as well.
#   For a backend decision use benchmarks/reinforced_step_cost.py, which times
#   the whole step. Use this script to ask whether a KERNEL got faster.
#
# Spans the latency-bound regime through to a mesh big enough to start filling
# a card. The first entry is deliberately tiny -- the size the test suite uses.
# Whether the card loses there is the question; do not assume the answer.
# 40 / 240 / 600 / 2 560 / 6 000 / 19 200 / 153 600 elements. The two middle
# points exist because the first run of this script put the crossover at
# ~2 800 by interpolating across a 32x gap between 600 and 19 200 -- a
# headline number resting on nothing measured anywhere near it.
SWEEP = [(10, 2, 2), (30, 2, 4), (60, 2, 5), (80, 4, 8), (120, 5, 10),
         (120, 8, 20), (240, 16, 40)]


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


def time_host(divs, form, kind, backend, repeats, iters):
    """Per-step wall time for a host backend (numpy / numba)."""
    mesh, pre, mat, force, u = build(divs, form, kind, backend)
    state = mat.init_state(mesh.n_elem * pre.n_gp)
    quat = np.zeros((mesh.n_elem, 4))
    quat[:, 0] = 1.0

    # First call compiles; never time it.
    f_e, _ = force(u, quat, state, 1e-5, rot_iters=30)
    force.assemble(f_e)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            f_e, _ = force(u, quat, state, 1e-5, rot_iters=2)
            force.assemble(f_e)
        times.append((time.perf_counter() - t0) / iters)
    return mesh.n_elem, min(times), max(times) / min(times)


def time_warp(divs, form, kind, device, precision, repeats, iters):
    """Per-step wall time on a warp device, resident (nothing crosses the bus)."""
    import warp as wp

    mesh, pre, mat, force, u = build(divs, form, kind, "warp",
                                     device=device, precision=precision)
    force.set_u(u)
    force.set_nodal(np.ones((mesh.n_nodes, 3)),
                    np.zeros((mesh.n_nodes, 3), dtype=np.int8), [], [])

    # Compile, allocate, let the clocks settle -- none of it timed.
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
        # Without this the clock stops before the GPU has finished.
        wp.synchronize_device(force.device)
        times.append((time.perf_counter() - t0) / iters)
    out = mesh.n_elem, min(times), max(times) / min(times)
    del force
    return out


def crossover(rows):
    """First element count at which warp overtakes numba, interpolated."""
    prev = None
    for nel, t_nb, t_wp in rows:
        if t_wp < t_nb:
            if prev is None:
                return nel, "at or below the smallest mesh measured"
            n0, a0, b0 = prev
            # log-linear interpolation on the ratio, which crosses 1.0 here
            r0, r1 = a0 / b0, t_nb / t_wp
            if r1 == r0:
                return nel, ""
            f = (1.0 - r0) / (r1 - r0)
            return int(round(n0 * (nel / n0) ** f)), ""
        prev = (nel, t_nb, t_wp)
    return None, "not reached within the sweep"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="fp64", choices=("mixed", "fp64"))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--only", default=None, help="one material, e.g. --only elastic")
    ap.add_argument("--forms", default="reduced",
                    help="comma-separated formulations (default: reduced)")
    ap.add_argument("--small", action="store_true", help="stop before the big mesh")
    args = ap.parse_args()

    import warp as wp
    wp.init()
    dev = wp.get_device(args.device)
    print(f"warp device: {getattr(dev, 'name', dev)}  (cuda: {dev.is_cuda})")
    if not dev.is_cuda:
        print("             NOT a GPU -- this run compares numba against warp's")
        print("             CPU target and says nothing about a card.")
    print(f"precision:   {args.precision}")
    print(f"timing:      min of {args.repeats} x {args.iters} steps, after warm-up\n")

    kinds = (args.only,) if args.only else ("elastic", "chaboche", "cdp")
    forms = tuple(f.strip() for f in args.forms.split(","))
    sizes = SWEEP[:-1] if args.small else SWEEP

    print(f"{'elements':>9} {'form':>8} {'material':>9} {'numba ms':>9} "
          f"{'warp ms':>9} {'warp/numba':>11} {'winner':>8} {'spread':>7}")
    print("-" * 78)
    table = {}
    for divs in sizes:
        for form in forms:
            for kind in kinds:
                nel, t_nb, s_nb = time_host(divs, form, kind, "numba",
                                            args.repeats, args.iters)
                _, t_wp, s_wp = time_warp(divs, form, kind, args.device,
                                          args.precision, args.repeats, args.iters)
                ratio = t_wp / t_nb
                win = "warp" if ratio < 1.0 else "numba"
                spread = max(s_nb, s_wp)
                flag = "  <-- unstable" if spread > 1.5 else ""
                print(f"{nel:>9d} {form:>8s} {kind:>9s} {t_nb * 1e3:>9.3f} "
                      f"{t_wp * 1e3:>9.3f} {ratio:>10.2f}x {win:>8s} "
                      f"{spread:>6.2f}x{flag}")
                table.setdefault((form, kind), []).append((nel, t_nb, t_wp))
        print()

    print("crossover -- the mesh size at which warp overtakes numba")
    print("(below it, prefer numba; the launch overhead of a step is fixed and")
    print(" a small mesh cannot amortise it)\n")
    for (form, kind), rows in sorted(table.items()):
        nel, note = crossover(rows)
        if nel is None:
            print(f"  {form:>8s} {kind:>9s}:  {note}")
        else:
            print(f"  {form:>8s} {kind:>9s}:  ~{nel:,} elements"
                  f"{'  (' + note + ')' if note else ''}")
    print()
    print("Note: warp's CPU target is not numba. It is the slower of the two by")
    print("a wide margin, so 'warp beat warp-cpu' says nothing about whether the")
    print("card beats the fused CPU kernel -- which is the comparison above, and")
    print("the only one that should decide a backend for a real job.")


if __name__ == "__main__":
    main()
