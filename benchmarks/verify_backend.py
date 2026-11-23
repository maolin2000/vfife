"""Run a numbered case on a backend the case file does not name, and hold
the result to the case's own contract.

    python benchmarks/verify_backend.py cases/case_17_hinged_cylindrical_roof.yaml \
        --backend warp --device cuda --precision mixed

WHY THIS EXISTS. `vfife run` takes no `--backend`: a case file names the
backend it was written for, which is right -- the backend is part of what
the contract was measured against. But a backend under repair has to be
held to the same contract as the one that works, and editing a tracked case
file to do it is how an experiment ends up committed by accident.

So this writes the override into a copy of the case in memory, sends the
result to `results/<case>_<backend><device>.h5` so it cannot clobber the
case's own, and then runs `vfife.verification.verify` against
`verification/<case>.yaml` -- the same contract, the same checks, the same
tolerances. A backend that passes here is a backend the case can be moved
to; one that does not has broken something the tests did not cover.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy

import yaml

from vfife.api import prepare_case, run_case
from vfife.verification import verify


def main(case_path, backend, device, precision, tolerance=None):
    raw = yaml.safe_load(open(case_path, encoding="utf-8"))
    stem = os.path.splitext(os.path.basename(case_path))[0]
    tag = backend + (("_" + str(device).replace(":", "")) if device else "")

    case = copy.deepcopy(raw)
    sol = case.setdefault("solver", {})
    sol["backend"] = backend
    if device:
        sol["device"] = device
    if precision:
        sol["precision"] = precision
    if tolerance:
        sol["tolerance"] = float(tolerance)
        for step in case.get("steps") or []:
            if step.get("tolerance"):
                step["tolerance"] = float(tolerance)
    out = os.path.join("results", f"{stem}_{tag}.h5")
    case.setdefault("output", {})["path"] = out
    # a mesh path in a case file is relative to that file; a case handed over
    # as a dict has no file, so resolve it here rather than write a temporary
    # copy into cases/
    mesh = case.get("mesh") or {}
    if mesh.get("path"):
        mesh["path"] = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(case_path)), mesh["path"]))
    prepared = prepare_case(case)
    print(f"case: {stem}  backend={backend} device={device} precision={precision}")
    result = run_case(prepared, progress_every=10 ** 9)
    print(f"  analysis      {result.status}")
    if result.result_path is None:
        print("  no result written; nothing to verify")
        return 2
    report = verify(os.path.join("verification", stem + ".yaml"), result.result_path)
    print(f"  verification  {report['status']} ({len(report['checks'])} checks)")
    for check in report["checks"]:
        print(f"    {check['status']:4s}  {check['id']}: {check['value']}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--backend", default="warp")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="mixed")
    ap.add_argument("--tolerance", help="override the convergence tolerance")
    a = ap.parse_args()
    raise SystemExit(main(a.case, a.backend, a.device, a.precision, a.tolerance))
