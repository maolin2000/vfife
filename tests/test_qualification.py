from pathlib import Path

import yaml

from vfife import cli
from vfife.qualification import Qualification, assess, load, template


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_verified_manifests_pass_their_gates():
    for name in ("hexa8", "beam2", "beam2_timoshenko", "connector2", "chaboche", "cdp",
                 "piecewise_uniaxial", "shell4"):
        report = assess(load(ROOT/"qualification"/f"{name}.yaml"), root=ROOT)
        assert report["status"] == "PASS", report


def test_template_is_complete_but_todo_references_do_not_pass(tmp_path):
    raw = template("material", "cdp", "production")
    record = Qualification.model_validate(raw)
    report = assess(record, root=tmp_path)
    assert report["status"] == "INCOMPLETE"
    assert report["invalid_references"]
    assert set(report["missing_backend_coverage"]) == {"numpy", "numba", "warp"}
    categories = {item["category"] for item in raw["evidence"]}
    assert {"mesh_objectivity", "external_reference", "failure_path"} <= categories


def test_cli_qualification_check_and_template_are_machine_readable(capsys):
    assert cli.main(["qualify", "check", str(ROOT/"qualification"/"beam2.yaml"),
                     "--format", "json"]) == 0
    assert '"status": "PASS"' in capsys.readouterr().out
    assert cli.main(["qualify", "template", "element", "truss2",
                     "--level", "verified"]) == 0
    emitted = yaml.safe_load(capsys.readouterr().out)
    assert emitted["name"] == "truss2"
    assert any(x["category"] == "mixed_assembly" for x in emitted["evidence"])


def test_missing_gate_is_reported_not_silently_downgraded():
    record = Qualification(version=1, kind="element", name="beam2",
                           target_level="verified", evidence=[])
    report = assess(record)
    assert report["status"] == "INCOMPLETE"
    assert set(report["missing"]) == {
        "rigid_body", "fundamental_response", "equilibrium", "mixed_assembly"
    }


# --------------------------------------------------------------------------- #
#  evidence that names nothing is not evidence
# --------------------------------------------------------------------------- #


def test_a_reference_to_a_test_that_does_not_exist_is_invalid(tmp_path):
    """`assess` used to check only that the FILE before `::` existed.

    So a manifest could cite `tests/test_x.py::test_renamed_last_year` and
    still be reported PASS -- the qualification would keep asserting evidence
    that had stopped existing, which is the one thing a qualification record
    must not do.
    """
    from vfife.qualification import _names_a_real_test

    src = tmp_path / "test_thing.py"
    src.write_text("def test_real():\n    pass\n\n"
                   "async def test_async_real():\n    pass\n", encoding="utf-8")
    assert _names_a_real_test(src, "test_real")
    assert _names_a_real_test(src, "test_async_real")
    assert _names_a_real_test(src, "test_real[case-1]")      # parametrised
    assert _names_a_real_test(src, "TestClass::test_real")   # class path
    assert not _names_a_real_test(src, "test_gone")
    assert not _names_a_real_test(src, "")
    # a name that only appears in a comment or a string is not a definition
    src.write_text("# test_ghost\nX = 'test_ghost'\n", encoding="utf-8")
    assert not _names_a_real_test(src, "test_ghost")


def test_every_shipped_manifest_cites_evidence_that_exists():
    """The manifests are honest today. This is what keeps them honest when a
    test is renamed."""
    import glob
    import os

    from vfife.qualification import assess, load

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifests = sorted(glob.glob(os.path.join(root, "qualification", "*.yaml")))
    assert manifests, "no manifests found -- this test would be vacuous"
    for path in manifests:
        report = assess(load(path), root=root)
        assert not report["invalid_references"], (
            f"{os.path.basename(path)} cites {report['invalid_references']}")


def test_the_check_does_not_claim_the_evidence_passes():
    """Deliberate scope: `qualify check` verifies that a citation points at
    something real, not that it is green. Running the suite from here would
    make it a test runner, which is a different tool -- and a check that
    silently became a slow one would stop being run."""
    import inspect

    from vfife import qualification

    source = inspect.getsource(qualification._names_a_real_test)
    assert "does NOT run" in source


# --------------------------------------------------------------------------- #
#  Contract hygiene, found by auditing the gate rather than the models
# --------------------------------------------------------------------------- #

def test_every_contract_threshold_is_a_number_not_a_string():
    """`-5.38e6` is a STRING to PyYAML -- scientific notation needs a signed
    exponent -- and five contract thresholds were written that way.

    Nothing was broken by it, because `_judge` casts with `float()`. That is
    the reason to pin it rather than to shrug: the contracts survived only
    because one consumer happened to be defensive, while the JSON report
    already carried the string through verbatim. Anything that later reads a
    target and does arithmetic on it would have failed on five of eleven
    contracts, and the failure would have looked like a verification bug.
    """
    import glob

    import yaml

    offenders = []
    for path in sorted(glob.glob(str(ROOT / "verification" / "*.yaml"))):
        with open(path, encoding="utf-8") as fh:
            contract = yaml.safe_load(fh)
        for metric in contract.get("metrics", []):
            for key, value in (metric.get("expect") or {}).items():
                if key in ("target", "upper", "lower", "rtol", "atol") \
                        and isinstance(value, str):
                    offenders.append(
                        f"{Path(path).name}:{metric.get('id')}:{key}"
                        f"={value!r}")
    assert not offenders, (
        "thresholds parsed as strings (write the exponent signed, 1.0e+7): "
        + ", ".join(offenders))
