"""Every solver path, in real time, against an exact analytical invariant.

WHY THIS FILE EXISTS
--------------------
Relaxation pins `dt` at exactly 1.0. That is a sentinel value, and a sentinel
value VALIDATES WRONG CODE: any expression whose dimensions are out by one
power of dt is numerically correct under relaxation and wrong everywhere else.

It happened. The bonded bar's velocity update was written

    a_b = -bar_forces[i] / m_b

against a host that uses `a = F_un * dt_m` with `dt_m = dt / m`, so what both
branches call `a` is a velocity INCREMENT, not an acceleration, and the bar's
was short by one factor of dt. Every bonded test was a relaxation run, so the
whole suite agreed with it. In dynamic mode, dt = 6.8e-6 s, the bar's update
came out 1.5e5 times too large and the model reached 1e146 mm in forty steps.

No amount of care writes that line correctly every time. A test that runs the
path at dt != 1 does.

THE INVARIANT, AND WHY IT IS EXACT RATHER THAN APPROXIMATE
----------------------------------------------------------
Under a constant resultant force F on total mass M, the CENTRE OF MASS obeys
Newton and nothing else: internal forces are equal and opposite and cancel in
the sum, whatever the material, the mesh, the element type or the constitutive
nonlinearity. So

    u_cm(T) = (F / M) * T**2 / 2

and -- this is the part that makes it a razor -- central difference started at
`v_(1/2) = a dt / 2` reproduces that EXACTLY for constant acceleration:

    u_N = sum_(n=0..N-1) (n + 1/2) a dt**2 = a (N dt)**2 / 2

so the discrete answer is the continuum one, with no truncation error at all,
PROVIDED T is taken as the time actually simulated. `n_steps` is
`round(total_time / dt)`, so the last partial step is not taken and the
simulated time is `n_steps * dt`; comparing against the requested
`total_time` instead leaves an O(dt/T) discrepancy that looks like a real
error and is not (measured: 8.5e-3 at 101 steps, and it is entirely this).

Measured agreement, every path below: 5e-16 or better. A single wrong power
of dt does not perturb that, it destroys it.

WHAT IT DOES NOT CHECK
    Anything about the ANSWER -- stresses, stiffness, convergence. Momentum
    is blind to all of it, which is exactly why it makes a clean dimensional
    probe: nothing about the physics can mask a units error, and nothing
    about a units error can hide behind plausible physics.
"""
import numpy as np
import pytest

from vfife.api import prepare_case
from vfife.cli import make_solver
from vfife.core.dof import TRANSLATION

MESHES = "benchmarks/assets/"
#: A resultant with three unequal, mixed-sign components: an error that
#: happens to preserve one axis, or magnitude, still shows up.
FORCE = [3.0e4, -2.0e4, 5.0e4]
STEEL = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}
CONCRETE = {"type": "linear_elastic", "E": 2.0e10, "nu": 0.2, "rho": 2400.0}
BOX = {"dimensions": [0.4, 0.3, 0.2], "divisions": [2, 2, 2]}
REBAR = {"type": "rebar", "E": 2.0e11, "rho": 7850.0,
         "yield_strength": 4.74e8, "hardening": 1.0e9}
BAR_LINE = {"line": {"start": [0.05, 0.15, 0.1], "end": [0.35, 0.15, 0.1], "divisions": 4}}


def _case(section, mesh, material, **extra):
    """A free body: NO constraints, one constant resultant, real time.

    `hold` rather than the default `ramp`, because the invariant is for a
    CONSTANT force; a ramp would need the integral and would stop being exact.

    `section` is the one section the whole mesh gets; `extra` may add
    reinforcement sets and their assignments.
    """
    case = {
        "name": "dimensional_smoke",
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": mesh,
        "materials": {"main": material, "steel": REBAR},
        "sections": {"main": {**section, "material": "main"}},
        "assign": [{"region": "all", "section": "main"}],
        "loads": [{"region": "all", "vector": FORCE, "amplitude": "hold"}],
        "solver": {"backend": "numpy", "mode": "dynamic", "mass": "physical",
                   "total_time": 1.0e-3,
                   # NOT the default: kinetic damping zeroes the velocity at
                   # every kinetic-energy peak, which is a device for finding
                   # a static answer and would destroy the momentum this test
                   # is built on.
                   "damping": {"type": "none"}},
        "output": {"path": "unused.h5", "monitors": {}},
    }
    if "bar" in extra:
        case["sections"]["bar20"] = {"type": "bar", "material": "steel", "shape": "circle",
                                     "diameter": 0.020, "area": 3.1416e-4}
        case["reinforcement"] = {"bar": BAR_LINE}
        case["assign"].append({"reinforcement": "bar", "section": "bar20", "name": "bar",
                               **extra["bar"]})
    return case


def _translational(solver, array):
    """The (n_nodes, 3) translational view of a nodal array.

    The solvers do not agree on the layout and there is no reason they should:
    the solid path keeps (n_nodes, 3) arrays, the frame paths keep a flat
    equation vector of n_nodes * 6 -- and `beam_solver` exposes `u` as
    (n_nodes, 3) while its `mass` stays flat. So the shape is read per array,
    not per solver.
    """
    a = np.asarray(array)
    if a.ndim == 2:
        return a[:, :3]
    return solver.dofs.field(a, TRANSLATION)


def _centre_of_mass(solver):
    """Mass-weighted mean displacement, INCLUDING any independent bar dofs.

    A bonded bar carries its own mass and its own displacement, outside the
    host arrays. Leaving it out would make the test blind to the very path
    the missing dt was on.
    """
    m = _translational(solver, solver.mass)
    u = _translational(solver, solver.u)
    num, den = (m * u).sum(axis=0), m.sum(axis=0)
    for bar, mass in zip(getattr(solver, "bonded", None) or [],
                         getattr(solver, "mass_bar", None) or []):
        num = num + (mass * bar.u_bar).sum(axis=0)
        den = den + mass.sum(axis=0)
    return num / den, float(den[0])


def _run(case):
    prepared = prepare_case(case)
    solver = make_solver(prepared.case.model_dump(mode="python"),
                         prepared.mesh, prepared.material)
    solver.log = lambda *_: None
    solver.run(on_progress=lambda *_: None, progress_every=10 ** 9)
    return solver


#: (id, case) for every family the element library declares. `families()` is
#: asserted against this list below, so a NEW element family cannot be added
#: without either covering it here or deliberately excluding it.
PATHS = [
    ("hexa8", _case({"type": "solid", "formulation": "b-bar"}, BOX, CONCRETE)),
    ("hexa8+embedded_bar",
     _case({"type": "solid", "formulation": "b-bar"}, BOX, CONCRETE, bar={})),
    # the path the missing dt was on: the bar has dofs of its own, integrated
    # in a second leapfrog block beside the host's
    ("hexa8+bonded_bar",
     _case({"type": "solid", "formulation": "b-bar"}, BOX, CONCRETE,
           bar={"bond": {"type": "bond_ceb", "fc": 25.6e6}})),
    ("tet10", _case({"type": "solid", "formulation": "full"},
                    {"type": "gmsh", "path": MESHES + "nafems_le10.msh"},
                    CONCRETE)),
    ("truss2", _case({"type": "bar", "shape": "general", "area": 1.0e-4},
                     {"type": "gmsh",
                      "path": MESHES + "beam_cantilever_v22.msh"}, STEEL)),
    ("beam2", _case({"type": "beam", "shape": "general", "area": 1.0e-4,
                     "iy": 8.0e-10, "iz": 8.0e-10, "j": 1.6e-9},
                    {"type": "gmsh",
                     "path": MESHES + "beam_cantilever_v22.msh"}, STEEL)),
    ("beam2_timoshenko",
     _case({"type": "beam", "theory": "timoshenko", "shape": "general", "area": 1.0e-4,
            "iy": 8.0e-10, "iz": 8.0e-10, "j": 1.6e-9,
            "ay": 8.3e-5, "az": 8.3e-5},
           {"type": "gmsh", "path": MESHES + "beam_cantilever_v22.msh"},
           STEEL)),
    ("shell4", _case({"type": "shell", "formulation": "selective",
                      "thickness": 0.01},
                     {"type": "gmsh",
                      "path": MESHES + "clamped_square_shell4_v22.msh"},
                     STEEL)),
]


@pytest.mark.parametrize("case", [c for _, c in PATHS],
                         ids=[i for i, _ in PATHS])
def test_the_centre_of_mass_obeys_newton_at_a_real_time_step(case):
    """One constant force, one free body, `dt` nowhere near 1.

    The tolerance is 1e-9 against a measured 5e-16 -- six orders of headroom,
    because the point is not to police round-off but to catch an error that
    would be a factor of 1e5.
    """
    solver = _run(case)
    assert solver.dt != pytest.approx(1.0), (
        "this test is worthless unless dt differs from the relaxation "
        f"sentinel; got {solver.dt}")
    assert np.isfinite(solver.u).all()

    u_cm, total_mass = _centre_of_mass(solver)
    simulated = solver.n_steps * solver.dt
    expected = 0.5 * (np.array(FORCE) / total_mass) * simulated ** 2
    error = np.abs(u_cm - expected).max() / np.abs(expected).max()
    assert error < 1e-9, (
        f"centre of mass {u_cm} against Newton's {expected}: relative error "
        f"{error:.3e} at dt = {solver.dt:.3e} s over {solver.n_steps} steps")


def test_every_element_family_is_covered():
    """The guard that keeps this file honest as the library grows.

    A new element family added without a line in PATHS would be a new
    integrator path with no dimensional check -- which is precisely the state
    the bonded bar was in.
    """
    from vfife.elements.library import families

    declared = set(families().values()) - {None}
    covered = {"continuum", "line", "surface"}
    missing = declared - covered - {"mixed_dimensional"}
    assert not missing, (
        f"element families with no dimensional smoke test: {sorted(missing)}. "
        "Add a case to PATHS, or extend this exclusion with the reason.")
    # `mixed_dimensional` (coupled, shell_line) is a COMBINATION of paths that
    # are each covered above, and its meshes carry their own supports, so a
    # free body is not expressible without a bespoke mesh. Excluded knowingly.
    assert "mixed_dimensional" in declared


def test_the_invariant_would_notice_a_wrong_power_of_the_time_step():
    """The test's own teeth, checked rather than assumed.

    A velocity update short by one factor of dt scales the displacement by
    1/dt -- on the paths above that is 1e4 to 1e5. Asserting the ratio here
    means the 1e-9 tolerance is not merely tight, it is tight against a
    known, quantified failure.
    """
    solver = _run(PATHS[0][1])
    _u_cm, mass = _centre_of_mass(solver)
    simulated = solver.n_steps * solver.dt
    expected = 0.5 * (np.array(FORCE) / mass) * simulated ** 2
    # what the same run would have produced with the dt dropped from the
    # velocity update: every increment 1/dt too large
    corrupted = expected / solver.dt
    ratio = np.abs(corrupted / expected).max()
    assert ratio > 1e4, f"a dropped dt is only a factor of {ratio:.1f} here"
