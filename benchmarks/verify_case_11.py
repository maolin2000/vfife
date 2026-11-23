"""Check the fibre-beam kernel on your own machine, on cases/case_11.

    python benchmarks/verify_case_11.py            # checks + timing, ~1 min
    python benchmarks/verify_case_11.py --full     # also solve the whole case
    python benchmarks/verify_case_11.py --full --reference
                                                   # ...and solve it the slow
                                                   # way too, for a real
                                                   # end-to-end comparison
                                                   # (this one takes ~45 min)

WHAT IT CHECKS, AND WHY EACH ONE IS HERE
----------------------------------------
1. The kernel is actually running. It falls back to the NumPy path silently by
   design -- no numba, a material without a scalar twin, rebars that do not
   share a law. Silence is right for correctness and wrong for you: the run
   still gives the same answer, twenty times slower, with nothing to say why.
   So the first thing printed is which path is live and, if it is the slow one,
   the reason.

2. The two paths agree. Both laws are path dependent, so this walks a
   displacement history and compares at every step -- internal force, element
   actions, and the fibre state, which is what the next step reads. Two
   implementations can agree on force for a while and still be drifting
   underneath.

3. How fast each one is here. Numbers measured on this project's machine are
   in docs/fiber_beam.md; yours will differ and that is fine. What matters is
   the ratio and that the kernel is the one running.

The first call compiles, which takes a few seconds and is not timed.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE = os.path.join(ROOT, "cases", "case_11_fiber_rc_column_cyclic.yaml")

OK = "  [ok]  "
BAD = "  [!!]  "


def build(kernel=True):
    from vfife.api import prepare_case
    from vfife.core.fiber_beam_solver import FiberBeamSolver
    prep = prepare_case(CASE)
    problem = prep.case.model_dump(mode="python")
    solver = FiberBeamSolver(problem, prep.mesh, prep.material)
    if not kernel:
        solver._kernel = None
        solver.kernel_status = "forced off for comparison"
    return solver


def section_1_which_path():
    print("1. 走的是哪条路径")
    try:
        import numba
        print(f"      numba {numba.__version__}")
    except ImportError:
        print(f"{BAD}numba 未安装 —— 只能走 NumPy 参照路径，本脚本后面的对比无意义")
        return None
    solver = build(kernel=True)
    print(f"      kernel_status: {solver.kernel_status}")
    if solver._kernel is None:
        print(f"{BAD}内核未启用。上面这行说明了原因；"
              f"若原因不是预期的，算例或材料有变化。")
        return None
    print(f"{OK}熔合内核已启用")
    return solver


def section_2_equivalence(steps=40, seed=7):
    print(f"\n2. 内核与参照实现是否等价（走 {steps} 步历程，状态逐步累积）")
    fast, ref = build(True), build(False)
    if fast._kernel is None:
        print(f"{BAD}内核未启用，跳过")
        return False
    rng = np.random.default_rng(seed)
    worst = {"内力": 0.0, "单元内力": 0.0, "材料状态": 0.0}
    for _ in range(steps):
        dq = rng.normal(0.0, 3e-4, fast.ndof)
        fast.q[:] += dq
        ref.q[:] += dq
        f1, a1 = fast.internal_force()
        f2, a2 = ref.internal_force()
        worst["内力"] = max(worst["内力"],
                            np.max(np.abs(f1 - f2)) / max(np.max(np.abs(f2)), 1.0))
        worst["单元内力"] = max(worst["单元内力"],
                                np.max(np.abs(a1 - a2)) / max(np.max(np.abs(a2)), 1.0))
        s1 = np.concatenate([s.ravel() for s in fast.fiber_states])
        s2 = np.concatenate([s.ravel() for s in ref.fiber_states])
        worst["材料状态"] = max(worst["材料状态"],
                                np.max(np.abs(s1 - s2)) / max(np.max(np.abs(s2)), 1e-30))
    good = True
    for name, value in worst.items():
        mark = OK if value < 1e-10 else BAD
        print(f"{mark}{name:8s} 最大相对偏差 {value:.3e}")
        good = good and value < 1e-10
    if not good:
        print("      偏差应在 1e-10 以下。超过说明两条路径已经分岔，"
              "结果不可用，请把这段输出发回。")
    return good


def section_3_timing(reps=200):
    print(f"\n3. 单次内力评估耗时（{reps} 次取平均，首次编译不计入）")
    times = {}
    for label, use_kernel in (("熔合内核", True), ("NumPy 参照", False)):
        solver = build(kernel=use_kernel)
        if use_kernel and solver._kernel is None:
            continue
        solver.q[:] = np.random.default_rng(0).normal(0, 1e-4, solver.ndof)
        solver.internal_force()
        t0 = time.perf_counter()
        for _ in range(reps):
            solver.internal_force()
        times[label] = (time.perf_counter() - t0) / reps
        print(f"      {label:12s} {times[label] * 1e3:8.3f} ms")
    if len(times) == 2:
        ratio = times["NumPy 参照"] / times["熔合内核"]
        print(f"{OK}内核快 {ratio:.0f} 倍"
              f"（本项目机器上是 101 倍，你的硬件不同属正常）")
    return times


def section_4_full(with_reference):
    print("\n4. 完整算例")
    from vfife.api import run_case
    results = {}
    runs = [("熔合内核", "case_11_kernel.h5", False)]
    if with_reference:
        runs.append(("NumPy 参照", "case_11_reference.h5", True))
        print("      注意：参照路径大约需要 45 分钟")
    for label, name, force_off in runs:
        if force_off:
            import vfife.core.fiber_beam_solver as F
            original = F.FiberBeamSolver._setup_kernel

            def disabled(self, _o=original):
                _o(self)
                self._kernel = None
                self.kernel_status = "forced off"
            F.FiberBeamSolver._setup_kernel = disabled
        out = os.path.join(ROOT, "results", name)
        print(f"      跑 {label} ...", flush=True)
        t0 = time.perf_counter()
        outcome = run_case(CASE, output=out)
        elapsed = time.perf_counter() - t0
        if force_off:
            F.FiberBeamSolver._setup_kernel = original
        mark = OK if outcome.converged else BAD
        print(f"{mark}{label:12s} {elapsed/60:6.2f} 分钟   "
              f"status={getattr(outcome, 'status', '?')}")
        results[label] = (out, elapsed, outcome.converged)

    if len(results) == 2:
        import h5py
        (pa, _, _), (pb, _, _) = results["熔合内核"], results["NumPy 参照"]
        with h5py.File(pa) as fa, h5py.File(pb) as fb:
            worst = 0.0
            for name in ("tip_disp", "base_shear"):
                x = np.asarray(fa[f"monitors/{name}"])
                y = np.asarray(fb[f"monitors/{name}"])
                if len(x) != len(y):
                    print(f"{BAD}{name} 点数不同：{len(x)} vs {len(y)}")
                    continue
                d = np.max(np.abs(x - y)) / max(np.max(np.abs(y)), 1e-30)
                worst = max(worst, d)
                print(f"{OK if d < 1e-9 else BAD}{name:12s} 偏差 {d:.3e}")
        speed = results["NumPy 参照"][1] / results["熔合内核"][1]
        print(f"{OK}端到端快 {speed:.1f} 倍（本项目机器上是 18.7 倍）")

    _plot(results["熔合内核"][0])


def _plot(result_h5):
    """Draw the measured-vs-computed hysteresis and print the comparison.

    Run as a subprocess rather than imported: the plotting script owns its own
    matplotlib backend choice, and if matplotlib is missing here the failure
    should be one clear message, not a traceback out of the middle of a check
    that otherwise passed.
    """
    import subprocess
    print("\n5. 与实验对比")
    script = os.path.join(ROOT, "benchmarks", "plot_tanaka_hysteresis.py")
    png = os.path.join(ROOT, "results", "tanaka_hysteresis.png")
    proc = subprocess.run([sys.executable, script, result_h5, "-o", png],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        print(f"{BAD}出图失败：")
        for line in tail:
            print(f"        {line}")
        if "matplotlib" in (proc.stderr or ""):
            print("        装一下： conda install matplotlib")
        return
    for line in proc.stdout.strip().splitlines():
        print(f"      {line}")
    print(f"{OK}图已写到 {png}")
    print("      期望（本项目机器）：峰值力比 1.16，割线刚度 1.37，耗能 1.30")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="并求解完整算例（约 1 分钟）")
    ap.add_argument("--reference", action="store_true",
                    help="连 NumPy 参照路径也跑一遍（约 45 分钟）")
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    print(f"case: {CASE}\n")
    solver = section_1_which_path()
    if solver is None:
        print("\n内核未启用，后面的对比没有意义，先解决第 1 项。")
        return 1
    good = section_2_equivalence(args.steps)
    section_3_timing()
    if args.full:
        os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
        section_4_full(args.reference)
    else:
        existing = os.path.join(ROOT, "results", "case_11_kernel.h5")
        if os.path.exists(existing):
            print("\n（用已有的结果文件出图；要重新求解请加 --full）")
            _plot(existing)
        else:
            print("\n滞回曲线需要先求解完整算例：加 --full 重跑（约 1 分钟）。")
    print("\n" + ("全部通过。" if good else "等价性未通过 —— 见上方 [!!] 行。"))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
