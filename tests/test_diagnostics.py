"""Termination status, failure location and HDF5 audit regressions."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from vfife.cli import _with_element, build, make_solver
from vfife.core.diagnostics import (ConvergenceFailure, EnergyLedger,
                                    serialise, start, worst_residual)
from vfife.io import schema
from vfife.io.writer import ResultWriter


ROOT = Path(__file__).resolve().parents[1]


def test_worst_residual_reports_node_and_dof():
    got = worst_residual([[0, 2, 0], [0, -7, 3]],
                         [[False, True, False], [True, True, True]], 3)
    assert got == {"array": "residual", "index": [4], "node": 1,
                   "dof": 1, "value": -7.0}


def test_energy_ledger_is_finite_and_serialisable():
    record = start("relaxation", "numpy")
    record["energy"].update([0.0], [1.0], [0.0])
    record["energy"].update([2.0], [1.0], [1.0])
    record["energy"].snapshot(2, 0.25)
    report = serialise(record)
    assert report["energy_available"] is True
    assert np.isfinite(list(report["energy_history"][0].values())).all()


def test_beam_iteration_exhaustion_is_failure_with_location():
    case_path = ROOT / "cases" / "case_01_beam_cantilever.yaml"
    case, _ = schema.load(str(case_path))
    p = _with_element(case, "beam2", str(case_path))
    p["solver"] = {**p["solver"], "n_steps": 1, "tolerance": 1e-30}
    mesh, material, _ = build(case, str(case_path))
    solver = make_solver(p, mesh, material)
    with pytest.raises(ConvergenceFailure):
        solver.run(output_every=1, progress_every=1)
    assert solver.diagnostics["status"] == "MAX_STEPS"
    assert solver.diagnostics["converged"] is False
    assert solver.diagnostics["location"]["node"] == 1


def test_result_writer_persists_status_equilibrium_and_energy(tmp_path):
    case_path = ROOT / "cases" / "case_02_truss_bar.yaml"
    case, _ = schema.load(str(case_path))
    mesh, _, _ = build(case, str(case_path))
    path = tmp_path / "audit.h5"
    writer = ResultWriter(str(path), mesh, element="truss2")
    writer.write_frame(1, 1.0, {"displacement": np.zeros((mesh.n_nodes, 3))},
                       equilibrium=False)
    record = start("relaxation", "numpy")
    record["status"] = "MAX_STEPS"
    writer.write_diagnostics(serialise(record))
    writer.close()
    with h5py.File(path, "r") as h5:
        assert not bool(h5.attrs["analysis_converged"])
        assert h5.attrs["analysis_status"] == "MAX_STEPS"
        # by index, not by a literal name: the zero padding is a writer detail
        # and widening it (which had to happen once the iteration counter
        # passed 99999) should not break a test about equilibrium flags
        frames = sorted(h5["steps"], key=int)
        assert [int(k) for k in frames] == [1], frames
        # always a string since result format 3
        assert h5[f"steps/{frames[0]}"].attrs["equilibrium"] == "false"
        assert "relative_energy_error" in h5["diagnostics"]
