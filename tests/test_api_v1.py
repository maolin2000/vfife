"""The Python API, version 1 (route 0.9 item 2).

`vfife.api.__all__` is the promise: within 1.x the names stay, their
parameters stay (new ones may be added, with defaults), and the reference
page is what the docstrings say. The parameter study -- change a value,
run again -- is `load_case` / `CaseDraft.with_changes` / `study`, and a
changed copy of a case still finds its mesh by the paths its file wrote.
"""
import csv
import inspect
import os
import pathlib
import shutil

import pytest

from vfife import api, reference

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data" / "result_format_2"

#: the API as frozen at version 1: name -> parameters it must keep
FROZEN = {
    "load_case": ["path"],
    "prepare_case": ["source"],
    "run_case": ["source", "output", "restart", "on_ready", "on_progress", "on_frame", "progress_every"],
    "study": ["case", "changes", "output_dir", "combine", "extract", "on_run", "quiet"],
    "case_schema": [],
    "read_monitors": ["path"],
    "verify": ["contract", "result"],
    "CaseDraft": ["data", "base", "name"],
}
OUTCOME_FIELDS = ["status", "converged", "result_path", "sidecars", "preflight", "diagnostics",
                  "monitors", "wall_time", "message", "times"]


def test_the_api_reference_on_disk_is_what_the_code_generates():
    on_disk = (ROOT / "docs" / "API参考.md").read_text(encoding="utf-8")
    assert on_disk == reference.generate_api(), \
        "docs/API参考.md is stale: run `python -m vfife reference --api`"


def test_nothing_frozen_at_version_1_has_gone():
    assert api.API_VERSION == 1
    for name, params in FROZEN.items():
        assert name in api.__all__, name
        have = list(inspect.signature(getattr(api, name)).parameters)
        assert [p for p in params if p not in have] == [], (name, have)
    import dataclasses
    fields = [f.name for f in dataclasses.fields(api.AnalysisOutcome)]
    assert fields[:len(OUTCOME_FIELDS)] == OUTCOME_FIELDS       # positional order kept
    for name in api.__all__:
        assert hasattr(api, name), name


def test_a_draft_reads_and_changes_by_dotted_keys():
    case = api.load_case(DATA / "cantilever.yaml")
    assert case.get("sections.beam.width") == 0.1
    assert case.get("loads[0].vector[2]") == -1000.0
    assert case.get("loads.0.vector.2") == -1000.0
    new = case.with_changes({"loads[0].vector[2]": -2000.0, "solver.threads": 1})
    assert new.get("loads[0].vector[2]") == -2000.0 and new.get("solver.threads") == 1
    assert case.get("loads[0].vector[2]") == -1000.0            # the original untouched
    with pytest.raises(KeyError, match="has no 'bem'.*beam"):
        case.get("sections.bem.width")
    with pytest.raises(KeyError, match="has no 'bem'"):
        case.with_changes({"sections.bem.width": 0.2})           # only the last part may be new
    with pytest.raises(KeyError, match="1 entries"):
        case.set("loads[3].vector[2]", 0.0)


def test_a_changed_draft_finds_its_mesh_from_anywhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                                  # not the case's folder
    case = api.load_case(DATA / "cantilever.yaml")
    out = api.run_case(case.with_changes({"loads[0].vector[2]": -2000.0}),
                       output=str(tmp_path / "twice.h5"))
    assert out.status == "COMPLETED_DYNAMIC"
    assert len(out.times) == len(out.monitors["w_tip"])
    ref = api.read_monitors(DATA / "cantilever.h5")["w_tip"].values
    # a linear material and 2 mm on 2 m: twice the load, twice the deflection
    assert out.monitors["w_tip"][-1] == pytest.approx(2 * ref[-1], rel=2e-3)


def test_a_study_runs_every_combination_and_keeps_going_past_a_bad_one(tmp_path):
    case = api.load_case(DATA / "cantilever.yaml")
    rows = api.study(case, {"loads[0].vector[2]": [-1000.0, -3000.0],
                            "sections.beam.height": [0.1, -0.2]},
                     output_dir=tmp_path / "study",
                     extract=lambda o: {"w": o.monitors["w_tip"][-1]})
    assert [r["index"] for r in rows] == [0, 1, 2, 3]
    good = [r for r in rows if r["sections.beam.height"] == 0.1]
    bad = [r for r in rows if r["sections.beam.height"] == -0.2]
    assert all(r["status"] == "COMPLETED_DYNAMIC" for r in good)
    assert all(r["status"] == "ERROR" and "height" in r["message"] for r in bad)
    assert good[1]["w"] == pytest.approx(3 * good[0]["w"], rel=5e-3)
    assert os.path.exists(good[0]["result"])
    with open(tmp_path / "study" / "study.csv", newline="", encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
    assert len(table) == 4 and "w" in table[0]
    # zip, and a misspelt key refused before anything runs
    rows = api.study(case, {"loads[0].vector[2]": [-1000.0, -2000.0],
                            "solver.total_time": [0.005, 0.005]},
                     output_dir=tmp_path / "zip", combine="zip")
    assert len(rows) == 2 and all(r["converged"] is not None for r in rows)
    with pytest.raises(KeyError):
        api.study(case, {"sections.bem.height": [0.1]}, output_dir=tmp_path / "never")
    assert not (tmp_path / "never" / "study.csv").exists()
