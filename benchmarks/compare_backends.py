"""Run a case on CPU reference and accelerated backends, then compare fields.

This is a *backend-equivalence* benchmark: NumPy is the transparent reference
implementation and Numba is the fused CPU implementation.  It does not claim
an analytical reference solution for an arbitrary CAD model.  Both solved
cases, their ParaView sidecars and a compact JSON report are retained.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml


def _last_step(h5: h5py.File) -> str:
    steps = sorted(h5["steps"].keys(), key=int)
    if not steps:
        raise RuntimeError(f"{h5.filename} has no output frame")
    return steps[-1]


def _summary(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        step = _last_step(h5)
        out = {
            "step": int(step),
            "displacement": np.asarray(h5[f"steps/{step}/displacement"]),
        }
        for key, value in h5.get("monitors", {}).items():
            if key != "time" and len(value):
                out[f"monitor:{key}"] = float(value[-1])
    return out


def report_existing(case_path: Path, backends: list[str], work_dir: Path) -> dict:
    """Compare retained HDF5 results without rerunning a solver."""
    results = {}
    for backend in backends:
        path = work_dir / f"{case_path.stem}_{backend}.h5"
        results[backend] = {"result": str(path), **_summary(path)}
    report = {"reference": backends[0], "runs": {}}
    u_ref = results[backends[0]]["displacement"]
    for backend, result in results.items():
        entry = dict(result)
        u = entry.pop("displacement")
        entry["max_abs_displacement_difference"] = float(np.max(np.abs(u - u_ref)))
        entry["relative_displacement_difference"] = float(
            entry["max_abs_displacement_difference"] / max(np.max(np.abs(u_ref)), 1e-30)
        )
        report["runs"][backend] = entry
    report_path = work_dir / f"{case_path.stem}_backend_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nBackend comparison")
    for backend, entry in report["runs"].items():
        print(f"  {backend:>6}: rel |du| = {entry['relative_displacement_difference']:.3e}")
    print(f"  report: {report_path}")
    return report


def run(case_path: Path, backends: list[str], work_dir: Path) -> dict:
    source = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    # Generated cases live below ``results/``.  Resolve the source mesh before
    # moving the YAML, otherwise its original relative reference would point to
    # a different directory during the benchmark run.
    mesh_path = Path(source["mesh"]["path"])
    if not mesh_path.is_absolute():
        source["mesh"]["path"] = str((case_path.parent / mesh_path).resolve())
    work_dir.mkdir(parents=True, exist_ok=True)
    for backend in backends:
        case = copy.deepcopy(source)
        case["solver"]["backend"] = backend
        h5_path = (work_dir / f"{case_path.stem}_{backend}.h5").resolve()
        case["output"]["path"] = str(h5_path)
        generated = work_dir / f"{case_path.stem}_{backend}.yaml"
        generated.write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
        print(f"\n== {backend}: {generated} ==", flush=True)
        subprocess.run([sys.executable, "-m", "vfife", "run", str(generated)],
                       check=True)
    return report_existing(case_path, backends, work_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--backends", nargs="+", default=["numpy", "numba"])
    parser.add_argument("--work-dir", type=Path, default=Path("results/backend_benchmark"))
    parser.add_argument("--report-only", action="store_true",
                        help="compare existing HDF5 results without solving")
    args = parser.parse_args()
    if len(args.backends) < 2:
        parser.error("give at least two backends; the first is the reference")
    action = report_existing if args.report_only else run
    action(args.case.resolve(), args.backends, args.work_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
