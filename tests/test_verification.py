"""Formal result-verification contract tests."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from vfife.verification import load_contract, markdown, verify, write_reports


def _result(path: Path, ux=5e-5):
    with h5py.File(path, "w") as h5:
        h5.attrs["case_yaml"] = "name: demo\nsolver: {backend: numpy}\n"
        mesh = h5.create_group("mesh")
        mesh.create_dataset("nodes", data=[[0., 0., 0.], [1., 0., 0.]])
        mesh.create_dataset("elements", data=[[0, 1]])
        step = h5.create_group("steps").create_group("00001")
        step.create_dataset("displacement", data=[[0., 0., 0.], [ux, 0., 0.]])
        mon = h5.create_group("monitors")
        mon.create_dataset("residual", data=[1e-8])


def _contract(path: Path):
    path.write_text("""version: 1
case: demo
classification: analytical
metrics:
  - id: tip
    source: field
    name: displacement
    location: {nearest: [1, 0, 0]}
    component: x
    expect: {target: 5.0e-5, rtol: 1.0e-6}
  - id: residual
    source: diagnostic
    name: residual
    expect: {upper: 1.0e-6}
""", encoding="utf-8")


def test_contract_generates_json_and_markdown_reports(tmp_path):
    result, contract = tmp_path / "result.h5", tmp_path / "contract.yaml"
    _result(result); _contract(contract)
    report = verify(contract, result)
    assert report["status"] == "PASS"
    assert report["execution"]["backend"] == "numpy"
    jp, mp = tmp_path / "report.json", tmp_path / "report.md"
    write_reports(report, jp, mp)
    assert jp.exists() and "PASS" in markdown(report) and mp.exists()


def test_failed_metric_fails_report_and_case_mismatch_is_rejected(tmp_path):
    result, contract = tmp_path / "result.h5", tmp_path / "contract.yaml"
    _result(result, ux=1e-3); _contract(contract)
    assert verify(contract, result)["status"] == "FAIL"
    text = contract.read_text(encoding="utf-8").replace("case: demo", "case: other")
    contract.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        verify(contract, result)


def test_contract_schema_rejects_missing_metrics(tmp_path):
    path = tmp_path / "bad.yaml"; path.write_text("version: 1\ncase: demo\nclassification: analytical\n", encoding="utf-8")
    with pytest.raises(ValueError, match="metrics"):
        load_contract(path)


def test_every_numbered_case_has_a_matching_contract():
    root = Path(__file__).resolve().parents[1]
    cases = sorted((root / "cases").glob("case_*.yaml"))
    # A lower bound, not an equality. The equality was `== 11` and it fired the
    # day case_12 and case_12b were added, which is noise: the loop BELOW is
    # what actually protects -- every case found must have a contract whose
    # `case` field matches its own `name` and its stem. The count exists only
    # so a glob that matched nothing cannot make that loop vacuously pass.
    assert len(cases) >= 13
    for case_path in cases:
        contract = load_contract(root / "verification" / case_path.name)
        import yaml
        case = yaml.safe_load(case_path.read_text(encoding="utf-8"))
        assert contract["case"] == case["name"] == case_path.stem
