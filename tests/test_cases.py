"""The shipped cases, and the contracts that are supposed to police them.

`cases/` is what a new user copies and `verification/` is what says whether the
answers are still right. Eleven cases had eleven contracts -- and nothing ran
them. Six of the eleven cases were touched by no test at all, including the
NAFEMS benchmark and the RC column that is this project's flagship nonlinear
example. They could break and no one would find out.

That is the same shape as the hexa20 bug: the thing existed, the path was never
walked.

Solving eleven cases takes minutes, so the full verification is marked `slow`
and deselected by default. Everything that can be checked WITHOUT solving runs
every time, because a case file rots long before its physics does: a renamed
region, a monitor the contract no longer matches, a field name that no longer
exists. Those are the failures this file is for.
"""

import os

import pytest
import yaml

from vfife.api import available_fields, check_output_fields
from vfife.cli import build, make_solver, _with_element
from vfife.io import schema
from vfife.preflight import check as preflight_check

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE_DIR = os.path.join(ROOT, "cases")
CONTRACT_DIR = os.path.join(ROOT, "verification")

# A LEADING UNDERSCORE IS NOT A CASE. Several test helpers and benchmark
# sweeps write a patched copy of a case beside the original so that its mesh's
# relative path still resolves, and delete it in a `finally`; a run that is
# killed leaves one behind, and every contract test in this file then fails on
# a file nobody wrote. Five confusing failures, none of them about the library.
CASES = sorted(f for f in os.listdir(CASE_DIR)
               if f.endswith(".yaml") and not f.startswith("_"))
STEMS = [os.path.splitext(f)[0] for f in CASES]


def load(stem):
    path = os.path.join(CASE_DIR, stem + ".yaml")
    case, _raw = schema.load(path)
    return case, path


def contract(stem):
    with open(os.path.join(CONTRACT_DIR, stem + ".yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
#  the contracts exist and point at the right things
# --------------------------------------------------------------------------- #


def test_there_are_cases_to_check():
    """Guards the rest of the file: a glob that silently matches nothing would
    make every test below vacuously pass."""
    assert len(CASES) >= 13


@pytest.mark.parametrize("stem", STEMS)
def test_every_case_has_a_verification_contract(stem):
    """A case with no contract is a case nobody can tell is wrong."""
    assert os.path.exists(os.path.join(CONTRACT_DIR, stem + ".yaml"))


def test_every_contract_belongs_to_a_case():
    """And the other direction, so a renamed case leaves no orphan behind."""
    contracts = {os.path.splitext(f)[0] for f in os.listdir(CONTRACT_DIR)
                 if f.endswith(".yaml")}
    assert contracts == set(STEMS)


@pytest.mark.parametrize("stem", STEMS)
def test_a_contract_only_names_monitors_its_case_records(stem):
    """The check that costs nothing and catches renaming.

    A contract metric with `source: monitor` reads a series out of the result
    file by name. Rename the monitor in the case and the contract still looks
    fine -- it fails only at the end of a full run, with a missing-key error
    rather than a wrong answer.
    """
    case, _ = load(stem)
    # `time` is not a monitor anyone declares: the writer puts it beside them
    # in every result file (io/writer.write_monitors), so a contract may read
    # it -- `reduction: at, against: time` is how a metric names an instant --
    # and so may `residual` / `kinetic_energy`, which are written the same way.
    # `age` joins them for a time-dependent model: the writer puts the
    # concrete age of every frame beside the monitors, which is the axis a
    # creep result is read on.
    declared = set(case.output.monitors or {}) | {"time", "residual",
                                                  "kinetic_energy", "age"}
    wanted = {m["name"] for m in contract(stem)["metrics"]
              if m.get("source") == "monitor"}
    missing = wanted - declared
    assert not missing, (
        f"{stem}: the contract reads monitor(s) {sorted(missing)}, which the "
        f"case does not record. It records {sorted(declared)}."
    )


@pytest.mark.parametrize("stem", STEMS)
def test_a_contract_states_what_it_is_evidence_of(stem):
    """`classification` is what stops a project baseline being presented as a
    public benchmark; the README defines exactly four values."""
    c = contract(stem)
    assert c["classification"] in {
        "analytical", "public_benchmark", "independent_reference", "regression"}
    assert c.get("reference"), f"{stem}: a contract with no reference"
    assert c["case"] == stem
    for metric in c["metrics"]:
        assert metric.get("expect"), f"{stem}: metric {metric['id']} expects nothing"


# --------------------------------------------------------------------------- #
#  the cases themselves still work, as far as can be told without solving
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stem", STEMS)
def test_every_case_parses_imports_and_preflights_clean(stem):
    """No solver, no backend: this must pass on a machine with no GPU.

    Six of these cases had no test of any kind. A renamed Physical group or a
    moved mesh file would have been found by whoever ran the case next.
    """
    case, path = load(stem)
    mesh, _material, element = build(case, path)
    report = preflight_check(case, mesh, element)
    assert report["status"] == "PASS", [
        i for i in report["issues"] if i["severity"] == "ERROR"]


@pytest.mark.parametrize("stem", STEMS)
def test_every_case_asks_only_for_fields_it_can_produce(stem):
    """`output.fields` is filtered by name at write time, so a stale name is
    dropped in silence -- discovered in ParaView, after the run.

    The probe runs on numpy whatever the case asks for: which fields exist is a
    property of the element and the material, not of the backend, and forcing
    numpy is what lets this run on a machine with no CUDA.
    """
    case, path = load(stem)
    if not case.output.fields:
        return
    mesh, material, element = build(case, path)
    problem = _with_element(case, element, path)
    problem = {**problem, "solver": {**problem["solver"], "backend": "numpy"}}
    solver = make_solver(problem, mesh, material)
    check_output_fields(solver, case.output.fields)     # raises if stale


@pytest.mark.parametrize("stem", STEMS)
def test_a_contract_checks_an_answer_and_not_only_convergence(stem):
    """`residual` says the iteration settled, not that it settled on the right
    number. A contract made only of that would pass a confidently wrong run.

    (Not "every case must record a monitor": case_03 verifies from `field`
    metrics and needs none. What matters is that something about the ANSWER is
    checked, whatever it is read from.)
    """
    metrics = contract(stem)["metrics"]
    answers = [m for m in metrics if m.get("name") != "residual"]
    assert answers, f"{stem}: the contract only checks the residual"


# --------------------------------------------------------------------------- #
#  the real thing
# --------------------------------------------------------------------------- #


def _cpu_backend(declared: str) -> str:
    """The backend a case is solved on for its contract: the one it declares,
    with `warp` mapped to `numba` (the CI has no GPU, and a check that runs
    only on hardware the runner lacks reports green by skipping) and `numba`
    mapped to `numpy` when numba is not installed. This used to force every
    case onto numpy, so that the contracts held on the readable path; since
    the fast tier asserts the compiled paths bit-identical to it on every
    push, the nightly runs the cases as they are shipped and at their
    shipped speed -- case_15 alone was 105 minutes on numpy.
    """
    if declared == "warp":
        declared = "numba"
    if declared == "numba":
        try:
            import numba  # noqa: F401
        except ImportError:
            return "numpy"
    return declared


@pytest.mark.slow
@pytest.mark.parametrize("stem", STEMS)
def test_a_case_solved_satisfies_its_contract(stem, tmp_path):
    """The verification that was written and never wired to anything.

    Deselected by default -- `pytest -m slow` runs it. Solved on the backend
    the case declares, CPU-mapped (`_cpu_backend`), so it neither skips on a
    machine without CUDA nor spends an hour on the readable path.
    """
    from vfife import api
    from vfife.verification import verify

    case, path = load(stem)
    # The case as a draft: it keeps the folder the file is
    # in, so the mesh, a ground-motion record, a support's own record,
    # a roughness profile and an initial_state all resolve
    # the way the file wrote them. Until then this test resolved the mesh and
    # the ground motion by hand, and a support record -- case_36 -- would have
    # been looked for relative to wherever pytest was started.
    source = api.load_case(path)
    source.set("solver.backend", _cpu_backend(source.get("solver").get("backend", "numpy")))
    source.get("solver").pop("device", None)
    # tools/check_all.py gives the longest case threads of its own (1.7): the
    # threaded kernels give the serial answer to the bit, so the contract is
    # the same whatever this says
    threads = os.environ.get("VFIFE_CASE_THREADS")
    if threads:
        source.set("solver.threads", int(threads))
    out = str(tmp_path / (stem + ".h5"))

    outcome = api.run_case(source, output=out)
    assert outcome.converged, outcome.message

    report = verify(os.path.join(CONTRACT_DIR, stem + ".yaml"), out)
    failed = [c for c in report["checks"] if c["status"] != "PASS"]
    assert report["status"] == "PASS", failed


# --------------------------------------------------------------------------- #
#  handing a case back as a dict
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stem", STEMS)
def test_a_case_survives_a_round_trip_through_a_dict(stem):
    """Load, serialise, validate back: the case must say the same things.

    `as_dict` fills in every default, so a Case that never mentioned
    `element.type` came back claiming the default `hexa8` -- and the solver
    then refuses it against any mesh that is not hexahedral, which is most of
    them, since Gmsh's default 3-D output is tetrahedra. That is the GUI path:
    load a case, show it, send it back.
    """
    case, _ = load(stem)
    back = schema.Case.model_validate(case.as_written())
    assert ("type" in case.element.model_fields_set) == (
        "type" in back.element.model_fields_set), (
        f"{stem}: the round trip changed whether element.type was stated")
    assert back.element.type == case.element.type


def test_as_dict_still_fills_in_the_defaults_the_solver_reads():
    """The two forms are for different jobs and must not be merged: the solver
    reads values, the round trip preserves claims."""
    case, _ = load(STEMS[0])
    full = case.as_dict()
    assert "hourglass_coeff" in full["element"]        # a default, present
    written = case.as_written()
    assert set(written) <= set(full)
    assert len(str(written)) < len(str(full))


def test_the_mismatch_error_names_the_round_trip_as_a_cause(tmp_path):
    """The message used to accuse the user of writing something they never
    wrote, with no way to tell where it came from."""
    from vfife.core.diagnostics import UserError

    case, path = load("case_04_nafems_le10")          # a tet10 mesh
    fabricated = schema.Case.model_validate(case.as_dict())
    with pytest.raises(UserError) as exc:
        build(fabricated, path)
    assert exc.value.code == "ELEMENT_TYPE_MISMATCH"
    assert "as_written" in exc.value.hint
