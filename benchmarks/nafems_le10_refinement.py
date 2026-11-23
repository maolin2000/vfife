"""Generate and solve a reproducible Tet10 refinement study for NAFEMS LE10."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import yaml

from evaluate_nafems_le10 import evaluate


def _mesh(gmsh: str, geo: Path, target: Path, lc: float) -> None:
    if gmsh == "python":
        try:
            import gmsh as gmsh_api
        except ImportError as exc:
            raise RuntimeError(
                "--gmsh python requires the Gmsh Python API (pip install gmsh)"
            ) from exc
        gmsh_api.initialize()
        try:
            gmsh_api.option.setNumber("General.Terminal", 0)
            gmsh_api.open(str(geo))
            gmsh_api.option.setNumber("Mesh.CharacteristicLengthMin", lc)
            gmsh_api.option.setNumber("Mesh.CharacteristicLengthMax", lc)
            gmsh_api.model.mesh.generate(3)
            gmsh_api.write(str(target))
        finally:
            gmsh_api.finalize()
        return
    subprocess.run([gmsh, "-3", str(geo), "-setnumber", "lc", str(lc),
                    "-o", str(target), "-format", "msh41"], check=True)


def run(source_case: Path, geo: Path, gmsh: str, levels: list[float],
        work_dir: Path, backend: str) -> list[dict]:
    source = yaml.safe_load(source_case.read_text(encoding="utf-8"))
    work_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for lc in levels:
        tag = f"lc_{lc:g}".replace(".", "p")
        mesh = (work_dir / f"nafems_le10_{tag}.msh").resolve()
        print(f"\n== {tag}: mesh ==", flush=True)
        _mesh(gmsh, geo, mesh, lc)
        case = copy.deepcopy(source)
        case["mesh"]["path"] = str(mesh)
        case["solver"]["backend"] = backend
        result = (work_dir / f"nafems_le10_{tag}.h5").resolve()
        case["output"]["path"] = str(result)
        case_path = work_dir / f"nafems_le10_{tag}.yaml"
        case_path.write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
        print(f"== {tag}: solve ==", flush=True)
        started = time.perf_counter()
        subprocess.run([sys.executable, "-m", "vfife", "run", str(case_path)],
                       check=True)
        report = evaluate(result)
        with h5py.File(result, "r") as h5:
            report["nodes"] = int(h5["mesh/nodes"].shape[0])
            report["elements"] = int(h5["mesh/elements"].shape[0])
        report.update({"lc": lc, "wall_seconds": time.perf_counter() - started,
                       "result": str(result)})
        rows.append(report)
        print(f"  {report['elements']} elements; sigma_yy(D) = "
              f"{report['sigma_yy_mpa']:.5f} MPa; "
              f"error = {report['relative_error_percent']:.2f}%")
    return rows


def write_report(rows: list[dict], work_dir: Path) -> None:
    columns = ["lc", "nodes", "elements", "sigma_yy_mpa", "target_mpa",
               "relative_error_percent", "final_relative_residual", "wall_seconds",
               "stress_recovery", "result"]
    (work_dir / "nafems_le10_refinement.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    with (work_dir / "nafems_le10_refinement.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=Path("cases/case_04_nafems_le10.yaml"))
    parser.add_argument("--geo", type=Path, default=Path("benchmarks/assets/nafems_le10.geo"))
    parser.add_argument("--gmsh", default="gmsh",
                        help="Gmsh executable/path, or 'python' for its Python API")
    parser.add_argument("--levels", type=float, nargs="+", default=[0.45, 0.30, 0.22])
    parser.add_argument("--backend", choices=["numpy", "numba", "warp"], default="numba")
    parser.add_argument("--work-dir", type=Path, default=Path("results/nafems_le10_refinement"))
    args = parser.parse_args()
    rows = run(args.case.resolve(), args.geo.resolve(), args.gmsh, args.levels,
               args.work_dir.resolve(), args.backend)
    write_report(rows, args.work_dir.resolve())
    print(f"\nreports: {args.work_dir.resolve() / 'nafems_le10_refinement.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
