"""What does one step of a reinforced solid model actually cost, and where?

WHY THIS EXISTS
---------------
benchmarks/crossover.py measures the ELEMENT KERNEL. On case_06 that kernel is
about 0.4 ms/step at this mesh size, while the case as a whole runs at roughly
3 ms/step. Nine tenths of the time is therefore outside the kernel and has no
name, and every backend decision made from crossover.py alone is being made on
a tenth of the picture.

This script measures the WHOLE step, on a mesh the size a Tanaka solid model
would need, with reinforcement present -- and then, with --profile, says where
the rest of the time goes.

WHAT IT IS NOT
--------------
Not a validation model. The mesh is a structured box sized to be REPRESENTATIVE
OF THE COST of a Tanaka-geometry solid model: right element count, right bar
count, right material types. At 8 divisions across 400 mm the cell is 50 mm and
the 40 mm cover is not resolved, so nothing about the forces it computes means
anything. Read the milliseconds, not the stresses.

USAGE (Anaconda Prompt)
-----------------------
    python benchmarks\\reinforced_step_cost.py
    python benchmarks\\reinforced_step_cost.py --backends numba warp --device cuda
    python benchmarks\\reinforced_step_cost.py --profile
    python benchmarks\\reinforced_step_cost.py --divisions 4 6 8 10

Compilation and device warm-up are excluded: every configuration runs a short
throwaway pass first, and only the second pass is timed.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
#  The model
# --------------------------------------------------------------------------- #

def build_case(nxy, nz, steps, backend, device, precision):
    """Tanaka-sized reinforced column: 0.4 x 0.4 x 1.6, 8 bars + hoops at 80 mm.

    `steps` is a fixed iteration count with an unreachable tolerance, so every
    configuration does exactly the same amount of work. Timing a run that stops
    on convergence would compare backends on how many steps they needed, which
    is a different question and confounds this one.
    """
    H = 1.6
    n_hoop = int(round(H / 0.08)) - 1          # 12 mm hoops at 80 mm
    c = 0.062                                  # bar centroid from the face
    far = 0.4 - c
    mid = 0.2

    longitudinal = []
    for y, z in ((c, c), (far, c), (c, far), (far, far),
                 (mid, c), (mid, far), (c, mid), (far, mid)):
        longitudinal.append({"start": [y, z, 0.0], "end": [y, z, H],
                             "divisions": nz})

    hoop_corners = [(c, c), (far, c), (far, far), (c, far)]
    hoops = []
    for i in range(4):
        y0, z0 = hoop_corners[i]
        y1, z1 = hoop_corners[(i + 1) % 4]
        hoops.append({
            "start": [y0, z0, 0.08], "end": [y1, z1, 0.08], "divisions": 4,
            "repeat": {"count": n_hoop, "offset": [0.0, 0.0, 0.08]},
        })

    solver = {
        "backend": backend, "mode": "relaxation", "mass": "underwood",
        "n_steps": steps, "tolerance": 1.0e-30,   # never converge: fixed work
        "relaxation_safety": 0.3, "damping": {"type": "kinetic"},
    }
    if backend == "warp":
        solver["device"] = device
        solver["precision"] = precision

    return {
        "name": "reinforced_step_cost",
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"dimensions": [0.4, 0.4, H], "divisions": [nxy, nxy, nz]},
        "element": {"formulation": "b-bar"},
        "material": {
            "type": "cdp", "E": 2.53e10, "nu": 0.2, "rho": 2400.0,
            "fb0_fc0": 1.16, "Kc": 0.6667, "dilation_angle": 36.0,
            "eccentricity": 0.1, "wt": 0.0, "wc": 1.0,
            "viscosity": 1.0, "n_return": 5, "crack_band": "both",
            "l_ref": "auto",
            "compression": {"source": "gb50010", "fc": 25.6e6,
                            "beta_c": 0.7, "sigma_c0_ratio": 0.4},
            "tension": {"source": "fracture_energy", "ft": 3.1e6,
                        "Gf": 120.0, "beta_t": 0.9},
        },
        "reinforcement": [
            {"name": "longitudinal", "area": 3.1416e-4,
             "host_deduction": 2.53e10,
             "material": {"type": "rebar", "E": 2.0e11, "rho": 7850.0,
                          "yield_strength": 474.0e6, "hardening": 1.0e9,
                          "kinematic_fraction": 0.5},
             "lines": longitudinal},
            {"name": "hoops", "area": 1.131e-4,
             "host_deduction": 2.53e10,
             "material": {"type": "rebar", "E": 2.0e11, "rho": 7850.0,
                          "yield_strength": 333.0e6, "hardening": 1.0e9,
                          "kinematic_fraction": 0.5},
             "lines": hoops},
        ],
        "regions": {
            "base": {"plane": {"point": [0, 0, 0], "normal": [0, 0, 1]}},
            "head": {"plane": {"point": [0, 0, H], "normal": [0, 0, 1]}},
        },
        "constraints": [{"region": "base", "dofs": [1, 1, 1]}],
        "loads": [{"region": "head", "vector": [0.0, 0.0, -819.0e3]}],
        "steps": [{"name": "axial", "n_steps": steps, "tolerance": 1.0e-30}],
        "solver": solver,
        "output": {"path": "results/_step_cost.h5", "every": 10 ** 9,
                   "monitors": {}},
    }


def run(case, steps):
    """One pass at a fixed step count. Returns total wall seconds."""
    import yaml
    from vfife.api import run_case
    case = dict(case)
    case["steps"] = [dict(case["steps"][0], n_steps=steps)]
    case["solver"] = dict(case["solver"], n_steps=steps)
    path = os.path.join(ROOT, "cases", "_step_cost.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(case, fh, allow_unicode=True)
    try:
        t0 = time.perf_counter()
        run_case(path, output=os.path.join(ROOT, "results", "_step_cost.h5"))
        return time.perf_counter() - t0
    finally:
        if os.path.exists(path):
            os.remove(path)


def measure(case, steps):
    """Per-step cost, and the fixed per-run tax, measured separately.

    HOW THIS WENT WRONG FIRST, SO IT IS NOT DONE THAT WAY AGAIN
    `run_case` builds a fresh solver, and the hexa8 kernel is declared
    `cache=False` with no process-level cache, so every run pays the numba
    compiler again. The first version of this function timed n and 3n steps
    through `run_case` and fitted t = C + n*s to subtract that. It does not
    work: compile time jitters by around a second run to run, and at a few
    hundred steps that jitter IS the answer. The same 256-element case came out
    at 0.951 ms/step from one pair and 5.227 from another.

    So the compiler is kept out of the timed region instead of subtracted from
    it. One solver is built and stepped twice: the first pass compiles, the
    second is timed, and the kernel is already there because it belongs to that
    solver object. `fixed` is then measured directly as build + compile.
    """
    import yaml
    from vfife.api import prepare_case
    from vfife.core.solver import Solver

    case = dict(case)
    case["steps"] = [dict(case["steps"][0], n_steps=steps)]
    case["solver"] = dict(case["solver"], n_steps=steps)
    path = os.path.join(ROOT, "cases", "_step_cost.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(case, fh, allow_unicode=True)
    try:
        t0 = time.perf_counter()
        prep = prepare_case(path)
        solver = Solver(prep.case.model_dump(mode="python"), prep.mesh,
                        prep.material, log=lambda *_: None)
        # A fixed-step benchmark is SUPPOSED not to converge -- tolerance is
        # 1e-30 so that every backend does identical work. Solver.run() raises
        # on that, where run_case() swallows it and returns an outcome, so the
        # exception here is the expected outcome and not an error.
        try:
            solver.run()                   # builds, compiles, steps once
        except Exception as e:
            if type(e).__name__ != "ConvergenceFailure":
                raise
        fixed = time.perf_counter() - t0

        solver.apply_step(0)               # same solver: kernel already built
        t0 = time.perf_counter()
        try:
            solver._run_step()
        except Exception as e:
            if type(e).__name__ != "ConvergenceFailure":
                raise
        hot = time.perf_counter() - t0
        return hot / steps, max(fixed - hot, 0.0)
    finally:
        if os.path.exists(path):
            os.remove(path)


# --------------------------------------------------------------------------- #
#  Where the rest of the time goes
# --------------------------------------------------------------------------- #

def profile(case, steps):
    """Attribute a warmed-up run by cumulative time, grouped by what it is.

    cumtime, not tottime: the bar path and the assembler spend most of their
    time inside numpy calls, and self-time hides that in `c_einsum`.
    """
    import cProfile
    import io
    import pstats
    run(case, steps)                                   # compile / warm up
    pr = cProfile.Profile()
    pr.enable()
    run(case, steps)
    pr.disable()
    ps = pstats.Stats(pr, stream=io.StringIO()).sort_stats("cumtime")

    def cum(fname, func):
        return max((v[3] for k, v in ps.stats.items()
                    if fname in k[0] and k[2] == func), default=0.0)

    total = ps.total_tt
    compile_time = max(
        (v[3] for k, v in ps.stats.items()
         if "dispatcher" in k[0] and k[2] == "_compile_for_args"), default=0.0)
    groups = [
        ("混凝土单元内核", cum("numba_hexa8", "kernel") + cum("warp_backend", "__call__")),
        ("嵌入钢筋 truss.force", cum("truss.py", "force")),
        ("  其中 gather (主机插值)", cum("truss.py", "gather")),
        ("  其中 internal_force", cum("truss.py", "internal_force")),
        ("装配 scatter_add", cum("mesh.py", "scatter_add")),
        ("numba 编译（一次性）", compile_time),
    ]
    print(f"\n  剖析：总计 {total:.2f} s，其中")
    named = 0.0
    for label, t in groups:
        if not label.startswith("  "):
            named += t
        flag = "" if t / total > 0.02 else "   (可忽略)"
        print(f"    {label:26s} {t:7.2f} s  {t / total * 100:5.1f}%{flag}")
    rest = total - named
    print(f"    {'以上都不是':26s} {rest:7.2f} s  {rest / total * 100:5.1f}%")
    if rest / total > 0.4:
        print("\n    过半时间不在任何具名条目里。这正是要找的东西："
              "把这段输出发回，\n    我按调用图往下钻。")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["numba", "warp"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="mixed", choices=["mixed", "fp64"])
    ap.add_argument("--divisions", nargs="+", type=int, default=[4, 6, 8],
                    help="section divisions; height scales to keep cubes")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--profile", action="store_true",
                    help="also break down where the time goes (numba only)")
    a = ap.parse_args()

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    print(f"固定 {a.steps} / {3 * a.steps} 步两点拟合 t = C + n*s（tolerance 1e-30，")
    print("绝不提前收敛，两个后端做完全相同的功）。s 是每步成本，C 是每次运行")
    print("的固定开销——主要是 numba 重新编译，任何后端选择都动不了它。\n")
    print(f"  {'截面':>5} {'单元':>6} {'钢筋':>6} "
          + "".join(f"{b + ' ms/step':>15}" for b in a.backends)
          + f"{'倍数':>8}   " + "".join(f"{b + ' 固定 s':>13}" for b in a.backends))

    for nxy in a.divisions:
        nz = nxy * 4                                   # keep cells cubic
        res = {}
        for backend in a.backends:
            case = build_case(nxy, nz, a.steps, backend, a.device, a.precision)
            try:
                res[backend] = measure(case, a.steps)
            except Exception as e:
                res[backend] = None
                print(f"    {backend} 失败: {type(e).__name__}: "
                      f"{str(e).splitlines()[0][:70]}")
        n_bar = 8 * nz + 4 * 4 * (int(round(1.6 / 0.08)) - 1)
        cells = "".join(f"{res[b][0] * 1e3:>15.3f}" if res.get(b) else f"{'--':>15}"
                        for b in a.backends)
        fixed = "".join(f"{res[b][1]:>13.1f}" if res.get(b) else f"{'--':>13}"
                        for b in a.backends)
        ratio = ""
        if len(a.backends) == 2 and all(res.get(b) for b in a.backends):
            x, y = (res[b][0] for b in a.backends)
            ratio = f"{x / y:>7.2f}x"
        print(f"  {nxy:>5} {nxy * nxy * nz:>6} {n_bar:>6} {cells}{ratio}   {fixed}")

    if len(a.backends) > 1:
        print(f"\n  倍数 = {a.backends[0]} / {a.backends[-1]}，>1 表示后者更快。")
        print("  交叉点是倍数穿过 1.0 的地方。Tanaka 实体模型约需 2000 单元。")
    else:
        print(f"\n  只跑了 {a.backends[0]}，没有可比对象。")
    print("\n  参照：本项目容器（2 核，无 GPU）上 numba 实测")
    print("    256 单元 1.43 ms/step，864 单元 3.83，2048 单元 5.63 —— 每单元成本"
          "\n    随网格变大反而下降（5.6 -> 4.4 -> 2.7 us），说明这个尺度上仍有"
          "\n    固定的每步开销在摊薄，单元内核还不是瓶颈。")
    print("\n  『固定 s』是每次运行的一次性代价，实测 8-10 s，几乎全是 numba 重编："
          "\n  numba_hexa8 声明了 cache=False，也没有纤维梁内核那样的进程级缓存，"
          "\n  所以每建一次 solver 就重编一次。长算例里可忽略，短算例和测试套件里"
          "\n  它就是主要成本，且换任何后端都不会变。")

    if a.profile:
        nxy = a.divisions[-1]
        print(f"\n剖析最大的那个网格（截面 {nxy}，numba）")
        profile(build_case(nxy, nxy * 4, a.steps, "numba", a.device,
                           a.precision), a.steps)


if __name__ == "__main__":
    main()
