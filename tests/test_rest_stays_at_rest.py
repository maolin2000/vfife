"""A model at rest, given a machine-precision kick, must stay at rest.

WHY THIS IS THE INVARIANT AND NOT SOMETHING CLEVERER
    An unstable explicit step does not need a load to show itself: it
    amplifies whatever is there, and round-off is always there. So the test
    is the cheapest one imaginable -- no load, a 1e-12 displacement seed, a
    few hundred steps -- and it is a razor, because a stable scheme leaves the
    seed where it is (or decays it) while an unstable one turns it into 1e100
    within a few hundred steps. There is no middle.

    It would have caught v0.1 the day it happened: a lattice dome
    with Rayleigh beta = 0.001 diverged from rest at the default step, and
    "diverged" was first read as "collapsed at 500 gal". Every dynamic solver
    set its step from the UNDAMPED bound and read the damping coefficients on
    the next line. beta is stiffness-proportional, so zeta = beta omega / 2 is
    largest at the top of the spectrum, where the step is decided, and the
    damped limit 2(sqrt(1+zeta^2) - zeta)/omega had collapsed to 0.31/omega
    against the 0.6/omega handed out.

WHAT IS COVERED
    Every element family with a dynamic path, times the damping kinds the
    schema offers: none, viscous, Rayleigh alpha, and a Rayleigh beta chosen
    per model to put zeta_top near 3 -- the regime that broke. The beta cases
    are also run at the OLD step, restored by dividing out the factor the
    solver applied, and must diverge: an invariant that cannot fail is not
    one.
"""
import numpy as np
import pytest

from vfife.api import prepare_case
from vfife.cli import make_solver
from vfife.core.diagnostics import NonFiniteFailure
from vfife.core.dof import TRANSLATION
from vfife.elements.stability import damped_step_factor

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

MESHES = "benchmarks/assets/"
STEEL = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}
CONCRETE = {"type": "linear_elastic", "E": 2.0e10, "nu": 0.2, "rho": 2400.0}
BOX = {"dimensions": [0.4, 0.3, 0.2], "divisions": [2, 2, 2]}
LINE = {"type": "gmsh", "path": MESHES + "beam_cantilever_v22.msh"}
SHELL = {"type": "gmsh", "path": MESHES + "clamped_square_shell4_v22.msh"}

#: (id, section, mesh, material, how the seed is read back). The fibre beam
#: is the one that found the defect, so it is here with the law it found it
#: under; the others are the linear families.
FAMILIES = [
    ("hexa8", {"type": "solid", "formulation": "b-bar"}, BOX, CONCRETE),
    ("truss2", {"type": "bar", "shape": "general", "area": 1.0e-4}, LINE, STEEL),
    ("beam2", {"type": "beam", "shape": "general", "area": 1.0e-4,
               "iy": 8.0e-10, "iz": 8.0e-10, "j": 1.6e-9}, LINE, STEEL),
    ("shell4", {"type": "shell", "formulation": "selective", "thickness": 0.01},
     SHELL, STEEL),
    ("beam2_fiber", {"type": "fiber", "shape": "tube", "diameter": 0.048,
                     "t": 0.003, "nt": 8}, LINE,
     {"type": "steel_cyclic", "grade": "Q235B", "E": 2.06e11, "rho": 7850.0}),
]

#: the damping kinds, with the beta filled in per model below
DAMPINGS = {
    "none": {"type": "none"},
    "viscous": {"type": "viscous", "coeff": 50.0},
    "rayleigh_alpha": {"type": "rayleigh", "alpha": 5.0},
    "rayleigh_beta": {"type": "rayleigh", "alpha": 0.0, "beta": None},
}


def _case(section, mesh, material, damping, total_time):
    case = {
        "name": "rest",
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": mesh,
        "materials": {"main": material},
        "sections": {"main": {**section, "material": "main"}},
        "assign": [{"region": "all", "section": "main",
                    **({"orientation": [0, 0, 1]} if section["type"] in ("beam", "fiber") else {})}],
        "loads": [],
        "solver": {"backend": "numpy", "mode": "dynamic", "mass": "physical",
                   "total_time": total_time, "damping": damping},
        "output": {"path": "unused.h5", "monitors": {}},
    }
    # the solid mesh is generated and has no named regions to hold; the others
    # are free bodies too -- a support is not what keeps a stable scheme still
    return case


def _build(section, mesh, material, damping, total_time=1.0):
    prepared = prepare_case(_case(section, mesh, material, damping, total_time))
    solver = make_solver(prepared.case.model_dump(mode="python"),
                         prepared.mesh, prepared.material)
    solver.log = lambda *_: None
    return solver


def _displacement(solver):
    a = np.asarray(solver.u)
    if a.ndim == 2:
        return a[:, :3]
    return solver.dofs.field(a, TRANSLATION)


def _kick(solver, seed=1e-12):
    rng = np.random.default_rng(3)
    arr = solver.q if hasattr(solver, "q") else solver.u
    arr[...] += rng.normal(0.0, seed, arr.shape)


def _omega_max(solver):
    """The top frequency the solver's own bound implies, from its undamped step."""
    safety = float(solver.p["solver"].get("dt_safety", 0.6))
    # the bound itself: `dt` is shortened to divide the run's duration
    dt0 = getattr(solver, "stable_dt", solver.dt) / solver.damping_step_factor
    # row-sum solvers set dt = safety / omega; the CFL ones set dt = safety * 2 / omega
    return (2.0 if type(solver).__name__ in ("Solver", "TrussSolver") else 1.0) * safety / dt0


def _strong_beta(section, mesh, material):
    """beta that puts zeta at the top of the spectrum near 3 for THIS model."""
    probe = _build(section, mesh, material, {"type": "none"}, total_time=1e-6)
    return 6.0 / _omega_max(probe)


def _run_n(solver, n_steps):
    solver.n_steps = int(n_steps)
    solver.total_time = solver.n_steps * solver.dt
    solver.run(on_progress=lambda *_: None, progress_every=10 ** 9,
               output_every=10 ** 9)


@pytest.mark.parametrize("damping_id", list(DAMPINGS))
@pytest.mark.parametrize("family", FAMILIES, ids=[f[0] for f in FAMILIES])
def test_a_kicked_model_stays_at_rest(family, damping_id):
    fid, section, mesh, material = family
    if fid == "hexa8" and damping_id == "rayleigh_beta":
        pytest.skip("the continuum refuses Rayleigh beta by design")
    damping = dict(DAMPINGS[damping_id])
    if damping.get("beta", 0) is None:
        damping["beta"] = _strong_beta(section, mesh, material)
    solver = _build(section, mesh, material, damping)
    _kick(solver)
    _run_n(solver, 600)
    u = _displacement(solver)
    assert np.all(np.isfinite(u))
    grown = float(np.abs(u).max())
    assert grown < 1.0e-9, (
        f"{fid} with {damping}: a 1e-12 kick grew to {grown:.3e} in 600 steps "
        f"(dt {solver.dt:.3e}, damping factor {solver.damping_step_factor:.3f})")


@pytest.mark.parametrize("family", [f for f in FAMILIES if f[0] != "hexa8"],
                         ids=[f[0] for f in FAMILIES if f[0] != "hexa8"])
def test_the_invariant_has_teeth_at_the_old_step(family):
    """Restore the undamped step under strong beta and watch it go.

    This is the defect, reproduced on purpose: the factor the solver now
    applies is divided back out, which is exactly the step every solver used
    to hand out. If this stops diverging, either the factor has become 1 for
    a case it should not, or the kick is being swallowed -- and the invariant
    above has quietly stopped meaning anything.

    WHY IT SWEEPS. beta is chosen from the solver's OWN omega_max, which is
    Gershgorin's and over-estimates the true one by a factor r that depends
    on the element. Under the old step a = 0.6 r and zeta_true = zeta_est r,
    so divergence needs zeta_est > 1 / (0.6 r^2): at r = 1 anything past 0.33
    goes, at r = 0.5 it takes 6.7, at r = 0.3 it takes 18.5. The shell's r is
    small enough that zeta_est = 3 sat below the line -- so the level at
    which each family finally goes is recorded, and it is a measurement of
    how loose its row-sum bound is.
    """
    fid, section, mesh, material = family
    omega = _omega_max(_build(section, mesh, material, {"type": "none"}, total_time=1e-6))
    went_at = None
    for zeta_est in (3.0, 10.0, 30.0, 100.0):
        beta = 2.0 * zeta_est / omega
        solver = _build(section, mesh, material,
                        {"type": "rayleigh", "alpha": 0.0, "beta": beta})
        assert solver.damping_step_factor < 0.5, (
            f"beta {beta:.3e} was meant to bite: factor {solver.damping_step_factor:.3f}")
        solver.dt = solver.dt / solver.damping_step_factor      # the old step
        _kick(solver)
        try:
            _run_n(solver, 600)
            grown = float(np.abs(_displacement(solver)).max())
        except NonFiniteFailure:
            grown = np.inf
        except ValueError as exc:
            # the shell notices its elements have been torn before the
            # numbers overflow: "reference face is too warped" IS divergence
            if "warped" not in str(exc) and "degenerate" not in str(exc):
                raise
            grown = np.inf
        if not np.isfinite(grown) or grown > 1.0e-3:
            went_at = zeta_est
            break
    assert went_at is not None, (
        f"{fid}: the old step did not diverge up to zeta_est = 100; "
        "the invariant has lost its teeth")
    # r >= sqrt(1 / (0.6 zeta_est)) at the level it went: a record, not a limit
    print(f"{fid}: old step diverges at zeta_est = {went_at:g} "
          f"(Gershgorin looseness r >= {np.sqrt(1/(0.6*went_at)):.2f})")


def test_the_factor_is_exactly_one_when_nothing_is_damped():
    """So an undamped run's step does not move by a bit."""
    assert damped_step_factor(1.0e4) == 1.0
    assert damped_step_factor(1.0e4, alpha=0.0, beta=0.0, coeff=0.0) == 1.0
    assert damped_step_factor(0.0, beta=1.0) == 1.0
    # and the derivation's two limits
    assert damped_step_factor(1.0e4, beta=2.0e-4) == pytest.approx(np.sqrt(2.0) - 1.0)
    z = 50.0
    assert damped_step_factor(1.0e4, beta=2 * z / 1.0e4) == pytest.approx(1 / (2 * z), rel=1e-3)
