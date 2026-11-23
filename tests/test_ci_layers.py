"""What CI installs is itself an engineering decision, so it is asserted here.

WHY
    `.github/workflows/ci.yml` says, in its own comment, that installing the
    optional extras is deliberate because "一个被跳过的测试保护不了任何东西".
    It then installed `.[dev,plot]` and nothing else for four stages, so the
    numba extra was absent and 44 assertions -- every check that the fused
    CPU kernel agrees bit for bit with the scalar law, that a fibre section
    stays on that kernel, that a declared scalar twin actually compiles --
    skipped on every push. Nobody saw it: a skip is green.

    That is the failure mode a comment cannot prevent, because the next
    person to edit the install line will not read it. So the policy lives
    here, where a change that drops an extra fails the suite that the extra
    protects.

WHAT IS ASSERTED
    The fast tier installs every extra a GPU-less runner can use -- derived
    from pyproject, not hardcoded, so a NEW extra is a failure until someone
    decides which tier it belongs to. `gpu` is excluded by name and with a
    reason: warp needs hardware GitHub does not rent us.

    The nightly tier installs the same. It used to install
    LESS, so that the contracts held on the NumPy reference path -- but "the
    compiled path is bit-identical to it" is what the fast tier asserts on
    every push, and running the contracts on the readable path a second
    time bought nothing but a 105-minute case_15. The cases now run at their
    shipped backend, CPU-mapped (tests/test_cases.py::_cpu_backend).

    The nightly tier is one job PER CASE. Twenty full solves
    in one job blew through its 120-minute budget, and a timed-out check is
    as green as a skipped one -- nobody saw the answers stop being checked.
    The matrix is a literal list in the workflow, so it is asserted here to
    equal `cases/*.yaml` minus the cases skipped BY NAME with a reason: a
    new case that nobody added to the matrix fails the fast tier, and a
    skipped case that no longer exists fails it too.
"""
import pathlib
import re

import pytest
import yaml

# 3.11 stdlib; the package still supports 3.10, where it is the pip package.
tomllib = pytest.importorskip("tomllib", reason="python < 3.11 without tomli")

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
PYPROJECT = ROOT / "pyproject.toml"

#: warp only: it needs a GPU, and a check that runs only on hardware the CI
#: does not have is a check that reports green by skipping.
NEEDS_HARDWARE_CI_LACKS = {"gpu"}

#: cases deliberately left out of the nightly matrix, each with its reason.
#: case_15 is 105 minutes on two threads and no faster with
#: numba (v0.2: the bonded-reinforcement path's per-step Python overhead
#: dominates, not the kernel); it comes back when that path is faster or it
#: has a cheaper verification form (a coarser mesh, a contract to cracking).
NOT_IN_THE_NIGHTLY_MATRIX = {
    "case_15_chalmers_prc": "~100 min on either path; the reinforcement path, not the kernel",
}


def _install_extras(job):
    """The extras in the `pip install -e ".[a,b]"` line of a workflow job."""
    steps = [s for s in job["steps"] if "pip install -e" in s.get("run", "")]
    assert len(steps) == 1, "one install step per job, so there is one place to look"
    found = re.findall(r'pip install -e "\.\[([^\]]*)\]"', steps[0]["run"])
    assert len(found) == 1
    return {name.strip() for name in found[0].split(",")}


@pytest.fixture(scope="module")
def jobs():
    if not (WORKFLOW.exists() and PYPROJECT.exists()):   # not a checkout
        pytest.skip("no workflow file next to the tests")
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def test_the_fast_tier_installs_every_extra_the_runner_can_use(jobs):
    declared = set(tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
                   ["project"]["optional-dependencies"])
    assert NEEDS_HARDWARE_CI_LACKS <= declared, "the excluded extra still exists"
    assert _install_extras(jobs["fast"]) == declared - NEEDS_HARDWARE_CI_LACKS


def test_the_fast_tier_really_gets_the_compiled_path(jobs):
    """The one that would have caught it: `cpu` by name, plus the reason.

    Measured on the twenty test files that mention numba at all: 270 pass
    with it, 226 without (28 turn into skips, 16 are parametrised on the
    available backends and are never collected). Those 44 are not a
    performance detail -- they are the whole of tests/test_uniaxial_numba.py,
    the fused fibre kernel comparison and the frame kernel, i.e. every
    assertion that the fast path computes what the readable path computes.
    """
    assert "cpu" in _install_extras(jobs["fast"])


def test_the_nightly_tier_solves_on_the_compiled_path(jobs):
    """Since v0.2 the nightly installs numba too -- see this module's
    docstring; a nightly without it would solve case_15 for 105 minutes."""
    assert "cpu" in _install_extras(jobs["verify"])
    assert "cpu" in _install_extras(jobs["verify_rest"])


def test_the_nightly_matrix_is_every_case_minus_the_named_skips(jobs):
    cases = {p.stem for p in (ROOT / "cases").glob("case_*.yaml")}
    assert cases, "no cases next to the tests"
    missing_skips = set(NOT_IN_THE_NIGHTLY_MATRIX) - cases
    assert not missing_skips, f"skipped by name but no such case: {missing_skips}"
    matrix = set(jobs["verify"]["strategy"]["matrix"]["case"])
    assert matrix == cases - set(NOT_IN_THE_NIGHTLY_MATRIX), (
        f"not in the matrix: {sorted(cases - set(NOT_IN_THE_NIGHTLY_MATRIX) - matrix)}; "
        f"in the matrix but not a case: {sorted(matrix - cases)}")
    assert jobs["verify"]["strategy"].get("fail-fast") is False, \
        "one red case must not cancel the others"


def test_the_nightly_matrix_runs_exactly_its_case(jobs):
    """The job must select ONE case's test by node id, not `-m slow` at large."""
    run = [s["run"] for s in jobs["verify"]["steps"] if "pytest" in s.get("run", "")]
    assert len(run) == 1
    assert "test_a_case_solved_satisfies_its_contract[${{ matrix.case }}]" in run[0]
    rest = [s["run"] for s in jobs["verify_rest"]["steps"] if "pytest" in s.get("run", "")]
    assert len(rest) == 1 and "--ignore=tests/test_cases.py" in rest[0]


def test_no_tier_pretends_to_have_a_gpu(jobs):
    for name, job in jobs.items():
        assert not (_install_extras(job) & NEEDS_HARDWARE_CI_LACKS), name


def test_nothing_runs_on_a_timer():
    """The nightly matrix ran every shipped case every night -- sixty jobs of
    up to two hours -- and used a private repository's month of Actions
    minutes in days. The solving layers run when started by
    hand, and `tools/check_all.py` runs the same checks on a local machine;
    a schedule put back would spend the minutes again without anyone
    deciding to."""
    if not WORKFLOW.exists():
        pytest.skip("no workflow file next to the tests")
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    on = wf.get("on", wf.get(True))            # PyYAML reads the key `on` as True
    assert "schedule" not in on
    for name, job in wf["jobs"].items():
        if name != "fast":
            assert job.get("if") == "github.event_name == 'workflow_dispatch'", name
    assert (WORKFLOW.parents[2] / "tools" / "check_all.py").exists()


def test_every_heavy_test_still_exists():
    """tests/heavy_tests.txt names tests the GitHub fast tier leaves out.
    A renamed test would silently move back into it, and a
    stale line would hide nothing -- so each line must name a test function
    that is still defined in its file."""
    lines = [l.strip() for l in (ROOT / "tests" / "heavy_tests.txt").read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    assert lines
    missing = []
    for line in lines:
        path, name = line.split("::", 1)
        func = name.split("[", 1)[0]
        src = (ROOT / path)
        if not src.exists() or not re.search(rf"^def {re.escape(func)}\(", src.read_text(encoding="utf-8"), re.M):
            missing.append(line)
    assert not missing, f"heavy_tests.txt names tests that are gone: {missing[:5]}"


def test_the_fast_tier_leaves_out_the_heavy_tests_and_runs_single_threaded(jobs):
    step = next(s for s in jobs["fast"]["steps"] if "pytest" in s.get("run", ""))
    assert '-m "not slow and not heavy"' in step["run"] and "-n auto" in step["run"]
    assert step.get("env", {}).get("NUMBA_NUM_THREADS") == "1"
