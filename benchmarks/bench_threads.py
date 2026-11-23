"""How much `solver.threads` buys on this machine.

Runs one case, cut to a short `--time`, once per thread count with
`backend: numba`, and reports the wall time of each run and whether the
answers are identical (they should be, to the bit: every element writes its
own row and the rows are summed in element order).

Each thread count first runs a warm-up of a twentieth of `--time`, so the
numba compilation (10-30 s per kernel) is not in the timed run.

    python benchmarks/bench_threads.py cases/case_24d_hzb_girder_piers.yaml --time 1.0
    python benchmarks/bench_threads.py cases/_hzb_seismic.yaml --time 2.0

(the seismic case starts from case_24f's result through its own
`solver.initial_state`; `--initial-state` points it elsewhere)

`--threads 1 2 4 8 0` picks the counts (0 is every core numba sees).

A frame or frame-and-cable model and a fibre model with
its bearings run through the compiled whole step when they can; `VFIFE_COMPILED_STEP=0 python benchmarks/...`
times the step-by-step loop instead, for comparison.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _variant(case: Path, threads: int, total_time: float, tag: str,
             initial_state: Path | None = None) -> Path:
    """The case with numba, `threads` and a short run, written next to the
    original so that its relative paths (mesh, records) still resolve."""
    d = yaml.safe_load(case.read_text(encoding="utf-8"))
    sol = d.setdefault("solver", {})
    if sol.get("mode", "relaxation") != "dynamic":
        raise SystemExit("bench_threads times explicit dynamic runs (solver.mode: dynamic)")
    sol["backend"] = "numba"
    sol["threads"] = threads
    sol["total_time"] = total_time
    if initial_state is not None:
        sol["initial_state"] = str(initial_state.resolve())
    out = d.setdefault("output", {})
    out["fields"] = ["displacement"]          # time the solve, not the HDF5 writer
    out["every"] = max(int(out.get("every") or 1), 1000)
    path = case.with_name(f"_bench_{tag}_{threads}.yaml")
    path.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _monitors(h5path: Path) -> dict:
    with h5py.File(h5path, "r") as f:
        return {k: np.asarray(v) for k, v in f["monitors"].items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path)
    ap.add_argument("--time", type=float, default=1.0, help="simulated seconds per timed run (default 1.0)")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 0], help="thread counts (0 = all cores)")
    ap.add_argument("--initial-state", "--restart", dest="initial_state", type=Path, default=None,
                    help="start from this result (a seismic run's dead state), as solver.initial_state; "
                         "`--restart` is the pre-1.0 name -- a restart continues the same analysis, "
                         "and a relaxation's state is refused for a dynamic run")
    ap.add_argument("--out", type=Path, default=Path("results/bench_threads"))
    args = ap.parse_args()

    # the warm-up is short; a fibre model would keep the loop there and
    # compile its step inside the timed run
    os.environ.setdefault("VFIFE_COMPILED_STEP", "1")
    import numba
    from vfife.api import run_case

    print(f"numba {numba.__version__}, {numba.config.NUMBA_NUM_THREADS} threads available, "
          f"os.cpu_count() = {os.cpu_count()}")
    args.out.mkdir(parents=True, exist_ok=True)
    rows, answers = [], {}
    for n in args.threads:
        created = []
        try:
            warm = _variant(args.case, n, args.time / 20.0, "warm", args.initial_state)
            created.append(warm)
            run_case(str(warm), output=str(args.out / f"warm_{n}.h5"))
            case = _variant(args.case, n, args.time, "timed", args.initial_state); created.append(case)
            h5 = args.out / f"threads_{n}.h5"
            t0 = time.perf_counter()
            outcome = run_case(str(case), output=str(h5))
            wall = time.perf_counter() - t0
        finally:
            for p in created:
                p.unlink(missing_ok=True)
        used = numba.config.NUMBA_NUM_THREADS if n == 0 else n
        rows.append((n, used, wall, outcome.status))
        answers[n] = _monitors(h5)
        print(f"  threads {n} ({used} used): {wall:8.1f} s  {outcome.status}", flush=True)

    base_n, _u, base_wall, _s = rows[0]
    print(f"\n{'threads':>8} {'used':>5} {'wall s':>9} {'speed-up':>9}  same answer as threads={base_n}")
    for n, used, wall, _status in rows:
        same = all(np.array_equal(answers[base_n][k], answers[n][k]) for k in answers[base_n])
        print(f"{n:>8} {used:>5} {wall:>9.1f} {base_wall / wall:>9.2f}  {'bit-identical' if same else 'DIFFERENT'}")
    full = float(yaml.safe_load(args.case.read_text(encoding="utf-8"))["solver"].get("total_time") or args.time)
    print(f"\nthe whole {full:g} s run, scaled from {args.time:g} s:")
    for n, used, wall, _status in rows:
        print(f"  threads {n}: about {wall * full / args.time / 60:.1f} min")


if __name__ == "__main__":
    main()
