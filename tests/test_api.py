from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import pytest

from vfife.api import case_schema, prepare_case, run_case
from vfife import api, cli
from vfife.io import schema

ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT/"cases"/"case_02_truss_bar.yaml"


def test_prepare_case_is_solver_free_and_structured():
    p=prepare_case(CASE)
    assert p.preflight["status"]=="PASS"
    assert p.element=="truss2" and p.mesh.n_elem==1
    assert "properties" in case_schema()


def test_gui_api_runs_and_returns_paths_diagnostics_and_monitors(tmp_path):
    out=tmp_path/"api_bar.h5"
    import yaml
    source=yaml.safe_load(open(CASE,encoding="utf-8"))      # the input form, as written
    source["mesh"]["path"]=str((ROOT/"benchmarks/assets/beam_cantilever_v22.msh").resolve())
    source["output"]["fields"] += ["reaction_force"]
    result=run_case(source,output=str(out),progress_every=999999)
    assert result.status=="CONVERGED" and result.converged
    assert result.monitors["loaded_ux"][-1] > 0
    assert all(Path(p).exists() for p in result.sidecars)
    with h5py.File(out,"r") as h5:
        from vfife.io.writer import RESULT_FORMAT, read_checkpoint
        assert h5.attrs["result_schema_version"]==RESULT_FORMAT
        from vfife import __version__
        assert h5.attrs["vfife_version"]==__version__
        assert bool(h5.attrs["analysis_converged"])
        assert "material_signature" in read_checkpoint(str(out))      # its own file
        frame=h5["steps"][next(iter(h5["steps"]))]
        assert "reaction_force" in frame
        # 2e-7: the truss relaxation's 500 iterations with kinetic damping that
        # resets at the energy peaks, not on every step
        assert frame["reaction_force"][0,0] == pytest.approx(-1000.0, rel=1e-6)


def test_cli_run_delegates_to_application_service(monkeypatch, capsys):
    """The CLI must not grow a second solver/writer lifecycle again."""
    prepared = SimpleNamespace(
        case=SimpleNamespace(name="delegation"),
        preflight={"status": "PASS", "issues": [],
                   "counts": {"ERROR": 0, "WARNING": 0}},
        element="truss2",
    )
    solver = SimpleNamespace(
        step=10, nel=1, wall_time=0.5, u=np.zeros((2, 3)), monitors={},
        # every attribute the CLI's summary reads, and no more: the fake is
        # the record of what that summary is allowed to depend on
        n_eroded=0,
        summary=lambda: "  solver        fake",
    )
    calls = []

    monkeypatch.setattr(api, "prepare_case", lambda source: prepared)

    def fake_run(source, **kwargs):
        calls.append((source, kwargs))
        kwargs["on_ready"](solver, prepared)
        # the fields the CLI's summary reads off an outcome, and no more
        return SimpleNamespace(converged=True, status="CONVERGED", message=None,
                               result_path="delegated.h5",
                               diagnostics={"residual": 1.0e-7, "load_step": 1})

    monkeypatch.setattr(api, "run_case", fake_run)
    args = SimpleNamespace(case="case.yaml", output=None, restart=None,
                           progress_every=200)
    assert cli.cmd_run(args) == 0
    assert calls and calls[0][0] is prepared
    out = capsys.readouterr().out
    assert "delegated.h5" in out
    assert "CONVERGED" in out          # the first word an engineer reads
