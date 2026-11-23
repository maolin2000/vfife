"""Where the warp relaxation iteration's time actually goes.

    python benchmarks/warp_loop_cost.py                     # the default sweep on cuda
    python benchmarks/warp_loop_cost.py --device cpu        # no graph capture there; the rest still runs
    python benchmarks/warp_loop_cost.py --n 16 32 64 128
    python benchmarks/warp_loop_cost.py --block 100         # iterations per readback / per captured graph

THE QUESTION
------------
`benchmarks/shell_gpu.py` measured 367 us an iteration at 256 elements and
532 us at 4096 -- a factor of sixteen in work for a factor of 1.4 in time.
That is not a solver that is doing arithmetic; it is a solver paying a fixed
per-iteration toll. This file asks WHICH toll, because the three candidates
have three different fixes and only one of them is cheap:

    ~10 kernel launches   ->  capture the block as a CUDA graph
    1 synchronising readback of six scalars (`measures`) -> check convergence
                              every N iterations instead of every one
    the `measures` kernel itself (a pass over the dofs with five atomics)
                          -> fold it into the integrator

FIVE ARMS, AND WHAT EACH GAP MEANS
----------------------------------
    A  force + integrate + measures + readback     what a run does today
    B  force + integrate + measures                A minus the readback
    C  force + integrate                           B minus the measures kernel
    D  C, captured as a graph and replayed         C minus the launch overhead
    E  B, captured, one readback per block         the realistic target

    A - B   the cost of synchronising every iteration
    B - C   the cost of the measures kernel
    C - D   the cost of launching ~9 kernels from Python
    E       what the loop would cost after both changes

Every arm runs the same number of iterations of the same problem, so the
columns are directly comparable. A, B and C are checked to leave the same state:
`measures` writes only to the scalar buffer, so diluting or removing the
readback cannot move the trajectory. "The same" and not "bit-identical",
because the warp path accumulates element forces with atomics and two runs
of the SAME arm already differ -- 5e-3 of a 1e0 displacement over 200
iterations on the CPU target. The check is against that floor, and the floor
is printed, because it is worth knowing on its own: the numpy and numba
paths are bit-reproducible by construction (`mesh.scatter_add` is a
bincount) and this one is not.

What CAN move the trajectory is the kinetic-damping reset, which today is a
host decision taken on the value the readback returns -- that is a separate
question, settled by running the case against its contract, not by this
file.

CAPTURE AND THE RAMP. A captured graph bakes its kernel arguments in, and
`integrate(progress, half, approach)` passes `approach` as a scalar that
changes every iteration while a load level ramps in. The captured arms
therefore hold `approach = 1.0`, which is what all but the first
`stage_ramp_iterations` of a level do anyway. Making the real loop
capturable means moving that scalar into a device array -- one of the two
changes this measurement is meant to justify or rule out.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import numpy as np

from benchmarks.shell_gpu import roof_problem, solver_for


def _shell(s):
    """The device-resident shell of a prepared solver, wired for step 0."""
    fixed, moving, prescribed, groups, ground = s.step_data(0)
    ws = s._make_warp_shell()
    ws.set_mass(s.mass)
    ws.set_nodal(fixed, groups, [(d, v, a) for d, v, a, _m in prescribed],
                 [np.zeros(len(d)) for d, *_ in prescribed])
    return ws


def _measures_launch(ws):
    """`WarpShell.measures` without the readback."""
    ws.measures_launch(1.0)


def _reset(ws, ndof):
    ws.set_u(np.zeros(ndof), np.zeros(ndof))
    ws._progress = None                       # so _set_factors runs once, as in a real level
    ws.force_resident(); ws.integrate(1.0, 1, 1.0); ws.measures(1.0)


def _state(ws):
    return ws.get_u().copy(), ws.get_v().copy()


def _diff(a, b):
    return max(float(np.abs(a[0] - b[0]).max()), float(np.abs(a[1] - b[1]).max()))


def _time(fn, iters, repeats, sync):
    best, worst = np.inf, 0.0
    for _ in range(repeats):
        sync()
        t = time.perf_counter()
        fn(iters)
        sync()
        dt = (time.perf_counter() - t) / iters
        best, worst = min(best, dt), max(worst, dt)
    return best, worst


def arms(s, iters, repeats, block, device):
    import warp as wp

    ws = _shell(s)
    ndof = s.ndof
    dev = wp.get_device(device)
    sync = lambda: wp.synchronize_device(dev)
    out = {}
    states = {}

    def run_A(n):
        for _ in range(n):
            ws.force_resident(); ws.integrate(1.0, 0, 1.0); ws.measures(1.0)

    def run_B(n):
        for i in range(n):
            ws.force_resident(); ws.integrate(1.0, 0, 1.0); _measures_launch(ws)
            if (i + 1) % block == 0:
                ws.d_scalars.numpy()

    def run_C(n):
        for _ in range(n):
            ws.force_resident(); ws.integrate(1.0, 0, 1.0)

    for name, fn in (("A full", run_A), ("B no sync", run_B), ("C no measures", run_C)):
        _reset(ws, ndof)
        out[name] = _time(fn, iters, repeats, sync)
        _reset(ws, ndof)
        fn(iters)
        states[name] = _state(ws)
    # the floor: the same arm, twice. The warp path accumulates the element
    # forces with atomics, whose summation order is not fixed, so two runs of
    # the SAME arm already differ -- and by more, on the CPU target, than the
    # arms differ from each other. Comparing the arms against zero would
    # therefore report a difference that is not about the arms at all.
    _reset(ws, ndof)
    run_A(iters)
    states["A again"] = _state(ws)

    if dev.is_cuda:
        def capture(body):
            _reset(ws, ndof)
            with wp.ScopedCapture(dev) as cap:
                for _ in range(block):
                    body()
            return cap.graph

        g_C = capture(lambda: (ws.force_resident(), ws.integrate(1.0, 0, 1.0)))
        g_B = capture(lambda: (ws.force_resident(), ws.integrate(1.0, 0, 1.0),
                               _measures_launch(ws)))

        def replay(graph, readback):
            def run(n):
                for i in range(0, n, block):
                    wp.capture_launch(graph)
                    if readback:
                        ws.d_scalars.numpy()
            return run

        _reset(ws, ndof)
        out["D graph, no measures"] = _time(replay(g_C, False), iters, repeats, sync)
        _reset(ws, ndof)
        out["E graph + measures"] = _time(replay(g_B, True), iters, repeats, sync)
        _reset(ws, ndof)
        replay(g_C, False)(iters)
        states["D graph, no measures"] = _state(ws)

    return out, states


def per_kernel(s, repeats, block, device):
    """Every launch of one iteration, timed on its own inside a graph.

    Arm D of the sweep -- force and integrate, captured, nothing read back --
    measured 154 us at 256 elements and 159 us at 4096: sixteen times the
    work for the same time, with no Python in the loop at all. Something in
    those nine launches costs about 150 us regardless of how little there is
    to do, and a per-launch breakdown is the cheapest way to find out which.
    Values are meaningless afterwards (the same kernel is replayed on its own
    output); only the times mean anything.
    """
    import warp as wp

    ws = _shell(s)
    ndof = s.ndof
    dev = wp.get_device(device)
    if not dev.is_cuda:
        print("  per-kernel timing needs a CUDA device (graph capture); skipped")
        return
    n, d, dim = ws.nodal, ws.device, (ws.rows, 3)
    items = [
        ("zero F, scalars", lambda: wp.launch(ws.zero_kernel, dim=dim, device=d,
                                              inputs=[ws.d_F, ws.d_scalars])),
        ("shell4 frame", lambda: wp.launch(
            ws.frame_kernel, dim=ws.nel, device=d,
            inputs=[ws.d_nodes, ws.d_conn, ws.d_u, ws.d_local0,
                    np.float64(ws.warp_tol), ws.d_lq, ws.d_frame, ws.d_flag, ws.d_scalars])),
        ("shell4 node", lambda: wp.launch(
            ws.node_kernel, dim=(ws.nel, 4), device=d,
            inputs=[ws.d_conn, ws.d_quat, ws.d_frame0, ws.d_frame, ws.d_flag, ws.d_lq])),
        ("shell4 apply", lambda: wp.launch(
            ws.apply_kernel, dim=(ws.nel, 8), device=d,
            inputs=[ws.d_conn, ws.d_lq, ws.d_frame, ws.d_K, ws.d_Bres,
                    ws.d_flag, ws.d_F, ws.d_result])),
        ("copy2d", lambda: wp.launch(n["copy2d"], dim=dim, device=d,
                                     inputs=[ws.d_u, ws.d_u_prev])),
        ("integrate", lambda: wp.launch(
            n["integrate"], dim=dim, device=d,
            inputs=[ws.d_F, ws.d_F_ext_stack, ws.d_factors, ws.d_mass, ws.d_u, ws.d_v,
                    np.float64(1.0), np.float64(1.0), np.float64(1.0), 0])),
        ("zero_fixed", lambda: wp.launch(n["zero_fixed"], dim=dim, device=d,
                                         inputs=[ws.d_fixed, ws.d_u])),
        ("sync_velocity", lambda: wp.launch(
            n["sync_velocity"], dim=dim, device=d,
            inputs=[ws.d_u_prev, ws.d_u, ws.d_v, np.float64(1.0)])),
        ("measures partial", lambda: wp.launch(
            ws.measures_partial, dim=ws.groups, device=d,
            inputs=[ws.d_F, ws.d_F_ext_stack, ws.d_factors, ws.d_mass,
                    ws.d_u, ws.d_v, ws.d_fixed, ws.d_partials])),
        ("measures mid", lambda: wp.launch(
            ws.measures_mid, dim=(5, ws.d_mids.shape[1]), device=d,
            inputs=[ws.d_partials, ws.d_mids])),
        ("measures finish", lambda: wp.launch(ws.measures_finish, dim=5, device=d,
                                              inputs=[ws.d_mids, ws.d_scalars])),
    ]
    if ws._pres is not None:
        rows, dims, vals, aidx, pfac, pstart, npres = ws._pres
        items.insert(5, ("apply_prescribed", lambda: wp.launch(
            n["apply_prescribed"], dim=npres, device=d,
            inputs=[rows, dims, vals, aidx, pfac, pstart, np.float64(1.0), ws.d_u])))
    sync = lambda: wp.synchronize_device(dev)
    print(f"    {'launch':>18} {'us / call':>12}")
    total = 0.0
    for name, call in items:
        _reset(ws, ndof)
        try:
            with wp.ScopedCapture(dev) as cap:
                for _ in range(block):
                    call()
            graph = cap.graph
        except Exception as exc:                                   # noqa: BLE001
            print(f"    {name:>18} {'not capturable':>12}   {type(exc).__name__}: {exc}")
            continue
        best = _time(lambda k: [wp.capture_launch(graph) for _ in range(k // block)],
                     block * 20, repeats, sync)[0]
        total += best
        print(f"    {name:>18} {1e6 * best:11.1f}")
    print(f"    {'sum':>18} {1e6 * total:11.1f}")


def report(sizes, device, iters, repeats, block, precision, breakdown=False):
    import warp as wp

    dev = wp.get_device(device)
    print(f"warp {wp.config.version} on {dev}; {iters} iterations x {repeats} repeats, "
          f"minimum shown; block = {block}\n")
    if not dev.is_cuda:
        print("!! not a CUDA device: arms D and E (graph capture) are skipped, and the\n"
              "   launch/synchronisation costs this file is about barely exist on the CPU\n"
              "   target. The numbers below are a smoke test of the harness, not an answer.\n")
    names = ["A full", "B no sync", "C no measures"] + (
        ["D graph, no measures", "E graph + measures"] if dev.is_cuda else [])
    print(f"{'elements':>9} " + " ".join(f"{n:>21}" for n in names))
    rows = {}
    for n in sizes:
        s = solver_for(roof_problem(n, "warp", device, precision))
        out, states = arms(s, iters, repeats, block, device)
        rows[n * n] = out
        ref = states["A full"]
        scale = max(float(np.abs(ref[0]).max()), 1e-30)
        floor = _diff(ref, states["A again"])
        for k, v in states.items():
            if k in ("A full", "A again"):
                continue
            d = _diff(ref, v)
            if d > max(10.0 * floor, 1e-12 * scale):
                print(f"  !! {k} left a different state than A: max |diff| {d:.3e}, "
                      f"against a run-to-run floor of {floor:.3e}")
        if floor > 1e-12 * scale:
            print(f"  (run-to-run difference of the SAME arm: {floor:.3e} of a "
                  f"displacement of {scale:.3e} -- the atomics are not reproducible)")
        cells = []
        for name in names:
            b, w = out[name]
            spread = f"±{100 * (w - b) / b:.0f}%" if w > 1.05 * b else ""
            cells.append(f"{1e6 * b:14.0f} us {spread:>5}")
        print(f"{n * n:9d} " + " ".join(cells))
        if breakdown:
            per_kernel(s, repeats, block, device)

    print("\nwhat the gaps say, per iteration:")
    print(f"{'elements':>9} {'sync (A-B)':>14} {'measures (B-C)':>16} "
          f"{'launches (C-D)':>16} {'target E':>12} {'A/E':>7}")
    for nel, out in rows.items():
        a = out["A full"][0]; b = out["B no sync"][0]; c = out["C no measures"][0]
        d = out.get("D graph, no measures", (float("nan"),))[0]
        e = out.get("E graph + measures", (float("nan"),))[0]
        print(f"{nel:9d} {1e6 * (a - b):11.0f} us {1e6 * (b - c):13.0f} us "
              f"{1e6 * (c - d):13.0f} us {1e6 * e:9.0f} us {a / e:6.1f}x")
    print("\nread it like this:")
    print("  A-B large  -> the synchronising readback dominates: check convergence every")
    print("                N iterations and decide the damping reset on the device")
    print("  C-D large  -> Python-side launch overhead dominates: capture the block")
    print("  B-C large  -> the measures kernel itself is worth folding into the integrator")
    print("  none large -> the loop is already compute-bound and the win is in the")
    print("                iteration COUNT, not in the iteration cost (Chebyshev/Anderson)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--precision", default="mixed")
    ap.add_argument("--per-kernel", action="store_true",
                    help="also time every launch of the iteration on its own")
    a = ap.parse_args()
    if a.iters % a.block:
        a.iters = a.block * max(1, a.iters // a.block)
        print(f"iterations rounded to a whole number of blocks: {a.iters}")
    report(a.n, a.device, a.iters, a.repeats, a.block, a.precision, a.per_kernel)
