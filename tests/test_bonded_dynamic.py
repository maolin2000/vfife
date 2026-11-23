"""A bonded bar in `mode: dynamic` -- the coverage that was missing.

WHY THIS FILE EXISTS
    Every bonded-bar test and every bonded result in this project is a
    RELAXATION run, and relaxation fixes dt at exactly 1.0. The bar's velocity
    update was written as

        a_b = -bar_forces[i] / m_b

    against a host that uses `a = F_un * dt_m` with `dt_m = dt / m` -- so what
    both branches call `a` is a velocity INCREMENT and the bar's was short by
    one factor of dt. At dt = 1.0 the two are the same number, so nothing in
    the suite could see it. In dynamic mode dt is 6.8e-6 s on the RC column,
    the bar's velocity update came out 1.5e5 times too large, and the run
    reached 1e146 mm in under forty steps.

    The tell was that it failed in the SAME forty steps whether dt_safety was
    0.6 or 0.03: an instability's onset scales with the time step, a wrong
    force does not.

WHAT WAS OPEN, AND IS NOT
    `test_zero_load_leaves_a_bonded_model_at_rest` carried an xfail: on the
    RC column a bonded bar left the origin under NO load, 0.29 mm in 0.02 s,
    growing ~1.2x per step and then saturating, immune to the time step. See
    that test for what it was.
"""
import numpy as np
import pytest
import yaml

from vfife.api import prepare_case
from vfife.core.solver import Solver

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

BOND = {"type": "bond_ceb", "fc": 25.6e6}


def _case(bond=BOND, total_time=0.02, dt_safety=0.6, load=2.0e5):
    """The block in tension from tests/test_bonded_solver.py, in real time."""
    case = {
        "name": "bonded_dynamic",
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"dimensions": [0.2, 0.2, 1.0], "divisions": [2, 2, 10]},
        "materials": {
            "concrete": {"type": "linear_elastic", "E": 2.53e10, "nu": 0.2, "rho": 2400.0},
            "steel": {"type": "rebar", "E": 2.0e11, "rho": 7850.0,
                      "yield_strength": 4.74e8, "hardening": 1.0e9},
        },
        "sections": {
            "body": {"type": "solid", "material": "concrete", "formulation": "b-bar"},
            "bar20": {"type": "bar", "material": "steel", "shape": "circle",
                      "diameter": 0.020, "area": 3.1416e-4},
        },
        "reinforcement": {"bar": {"line": {"start": [0.1, 0.1, 0.0], "end": [0.1, 0.1, 1.0],
                                           "divisions": 20}}},
        "assign": [{"reinforcement": "bar", "section": "bar20", "name": "bar"}],
        "regions": {
            "base": {"plane": {"point": [0, 0, 0], "normal": [0, 0, 1]}},
            "top": {"plane": {"point": [0, 0, 1.0], "normal": [0, 0, 1]}},
        },
        "constraints": [{"region": "base", "dofs": [1, 1, 1]}],
        "loads": [{"region": "top", "vector": [0.0, 0.0, load],
                   "amplitude": "ramp"}],
        "solver": {"backend": "numpy", "mode": "dynamic", "mass": "physical",
                   "total_time": total_time, "dt_safety": dt_safety,
                   # NOT the default: DampingSpec.type is "kinetic", which
                   # zeroes the velocity at every kinetic-energy peak. That is
                   # a device for finding a STATIC answer and it removes the
                   # dynamics it is applied to.
                   "damping": {"type": "none"}},
        "output": {"path": "unused.h5", "monitors": {}},
    }
    if bond is not None:
        case["assign"][0]["bond"] = bond
    return case


def _solve(tmp_path, case, name):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(case, allow_unicode=True), encoding="utf-8")
    prep = prepare_case(str(path))
    solver = Solver(prep.case.model_dump(mode="python"), prep.mesh,
                    prep.material, log=lambda *_: None)
    try:
        solver.run()
    except Exception:
        pass
    return solver


def test_a_bonded_bar_stays_finite_in_dynamic_mode(tmp_path):
    """The regression, in the bluntest possible form.

    Before the fix this reached 1e146 mm. `np.isfinite` is the whole
    assertion: a missing factor of dt does not produce a slightly wrong
    answer, it produces an overflow.
    """
    s = _solve(tmp_path, _case(), "dyn")
    assert np.isfinite(s.u).all()
    assert np.isfinite(s.bonded[0].u_bar).all()
    assert np.abs(s.u).max() < 1.0e-3, "a 200 kN pull moved the block a metre"


def test_the_dynamic_answer_walks_to_the_static_one_as_the_step_shrinks(
        tmp_path):
    """A ramped load rings, so a single dynamic run is not the static answer.
    Refining the time step must move it TOWARDS the static one rather than
    somewhere else -- which is what says the dt now enters the bar's update
    with the right power.

    Measured, host tip displacement against a relaxation reference of
    216.5 um: 262.7 um at dt_safety 0.6, 209.6 at 0.2, 216.5 at 0.05.
    """
    tips = []
    for f in (0.6, 0.2, 0.05):
        s = _solve(tmp_path, _case(dt_safety=f), f"dt{f}")
        assert np.isfinite(s.u).all()
        tips.append(float(np.abs(s.u[:, 2]).max()))
    static = 216.5e-6
    assert abs(tips[-1] - static) < abs(tips[0] - static), (
        f"refining the step moved away from the static answer: {tips}")
    assert abs(tips[-1] - static) / static < 0.05


def test_perfect_bond_was_never_affected(tmp_path):
    """The control that says the defect was in the bonded branch and not in
    dynamic mode generally. `EmbeddedTruss` has no dofs of its own, so it
    never went near the broken line."""
    s = _solve(tmp_path, _case(bond=None), "perfect")
    assert np.isfinite(s.u).all()
    assert float(np.abs(s.u[:, 2]).max()) == pytest.approx(217e-6, rel=0.1)


def test_zero_load_leaves_a_bonded_model_at_rest(tmp_path):
    """CLOSED, and the closing measurement is on the case that showed it.

    Zero load, zero initial conditions: every displacement in the model must
    stay at round-off. Measured on the RC column, 0.02 s
    of real time, linear-elastic host so no constitutive story could be
    blamed:

        no reinforcement at all      7.9e-24 mm
        perfect bond                 1.3e-13 mm
        BONDED bar                   2.9e-01 mm      <- five orders up
        ... after the bonded row sums were corrected      6.6e-02
        ... after v0.1                                6.1e-14   round-off

    It was never a time-step instability -- dt_safety 0.6 and 0.15 both gave
    0.29 mm, halving the duration gave 0.30 -- and this file recorded that
    "something else is also wrong". It was the bond law's own anchor. The
    unloading slope was the s1 secant, 63x SOFTER than the loading tangent:
    a round-off slip loaded the interface at k_initial, and on the way back
    the soft line held nearly all of that stress, so every step left a net
    push behind. A ratchet with a fixed ratio per step is exactly what a
    fixed growth of ~1.2x that ignores dt looks like. With the unloading
    slope at the loading tangent and the anchor restored to the committed
    point (per step, in dynamics) the interface is elastic at round-off and
    the drift is gone. materials/bond.py and core/staging.py carry the
    measurements; this test is the one that was waiting for them.
    """
    case = _case(load=0.0, total_time=0.02)
    s = _solve(tmp_path, case, "zero")
    assert np.isfinite(s.u).all()
    assert float(np.abs(s.u).max()) < 1.0e-9, (
        f"zero load moved the model {np.abs(s.u).max() * 1e3:.4g} mm")
