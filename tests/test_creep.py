"""Creep, shrinkage and ageing (cases/case_29*).

The three of them are one mechanism -- a strain that arrives for reasons that
are not force -- and the tests below are in the order the mechanism is built:

  * the model functions, against the code documents they come from;
  * the superposition integral, against the one creep law whose relaxation
    modulus is exact algebra;
  * and the solver's own step-by-step scheme, which is where two plausible
    ways of dating an increment both fail and the third works.
"""
import math
import pathlib
import sys

import numpy as np
import pytest

from vfife.api import _with_element, prepare_case
from vfife.cli import make_solver
from vfife.core import creep as C

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))
ROOT = pathlib.Path(__file__).resolve().parents[1]

PHI_INF, TAU, T0 = 2.0, 40.0, 28.0
ZENER = {"model": "curve",
         "curve": [[float(d), PHI_INF * (1.0 - math.exp(-d / TAU))]
                   for d in np.r_[0.0, np.exp(np.linspace(math.log(1e-4),
                                                          math.log(1e+6), 400))]]}


def zener_relaxation(d):
    """R(d)/E for phi(d) = phi_inf (1 - exp(-d/tau)): the Zener model's own
    relaxation modulus, which is why that creep function was chosen."""
    return (1.0 + PHI_INF * math.exp(-d * (1.0 + PHI_INF) / TAU)) / (1.0 + PHI_INF)


# --------------------------------------------------------------------------- #
#  Ageing
# --------------------------------------------------------------------------- #
def test_the_modulus_grows_with_age_and_is_1_at_28_days():
    assert C.modulus_factor(28.0) == pytest.approx(1.0)
    assert C.modulus_factor(7.0) < 1.0 < C.modulus_factor(365.0)


def test_s_applies_to_the_modulus_directly_only_when_it_was_fitted_to_one():
    """MC90's s is a STRENGTH coefficient and E follows the square root of
    strength, so the default halves it. An analyst who fitted s to measured
    moduli -- Ghali et al. for the Confederation Bridge -- means the other
    one, and at 7 days the two readings of the same s differ by 13%."""
    strength = C.modulus_factor(7.0, s=0.25, of="strength")
    modulus = C.modulus_factor(7.0, s=0.25, of="modulus")
    assert modulus == pytest.approx(strength ** 2)
    assert strength / modulus == pytest.approx(1.133, rel=0.01)


def test_an_age_of_zero_is_an_error_and_not_a_division():
    with pytest.raises(ValueError, match="must be positive"):
        C.modulus_factor(0.0)


# --------------------------------------------------------------------------- #
#  The creep coefficient
# --------------------------------------------------------------------------- #
def test_nothing_has_crept_at_the_instant_of_loading():
    for spec in (ZENER, {"model": "mc90", "notional_size": 200.0}):
        assert C.creep_coefficient(spec, 28.0, 28.0) == 0.0
        assert C.creep_coefficient(spec, 27.0, 28.0) == 0.0   # before, too


def test_mc90_reaches_about_2_for_a_normal_member_loaded_at_28_days():
    """h0 = 200 mm, RH 70%, fcm 38 MPa: MC90 gives phi(inf, 28) = 2.0, which
    is the number the code's own tables give and the reason 2.0 is the
    default anyone reaches for."""
    spec = {"model": "mc90", "notional_size": 200.0, "rh": 70.0, "fcm": 38.0e+6}
    assert C.creep_coefficient(spec, 1.0e+5, 28.0) == pytest.approx(2.0, rel=0.05)


def test_loading_later_creeps_less():
    spec = {"model": "mc90", "notional_size": 200.0}
    early = C.creep_coefficient(spec, 1.0e+4, 7.0)
    late = C.creep_coefficient(spec, 1.0e+4, 365.0)
    assert early > late


def test_a_vector_of_loading_ages_is_the_scalar_answer_element_by_element():
    """This is what makes the superposition affordable -- n vectorised calls
    instead of n^2 scalar ones -- so it has to give the same numbers."""
    spec = {"model": "mc90", "notional_size": 200.0}
    ages = np.array([7.0, 28.0, 90.0, 400.0, 1200.0])
    got = C.creep_coefficient(spec, 1000.0, ages)
    want = [C.creep_coefficient(spec, 1000.0, float(a)) for a in ages]
    assert got == pytest.approx(want)
    assert got[-1] == 0.0                     # loaded after t: no creep yet


def test_a_measured_curve_is_read_on_a_log_axis():
    """Linear in t would put the whole first decade in one segment. The
    midpoint of [1, 100] in log time is 10, and that is where the value
    halfway up the curve must sit."""
    spec = {"model": "curve", "curve": [[1.0, 0.0], [100.0, 2.0]]}
    assert C.creep_coefficient(spec, 28.0 + 10.0, 28.0) == pytest.approx(1.0)


def test_several_measured_curves_blend_in_log_loading_age():
    spec = {"model": "curve",
            "curve": {10.0: [[1.0, 0.0], [100.0, 2.0]],
                      1000.0: [[1.0, 0.0], [100.0, 1.0]]}}
    # log-halfway between the two loading ages, ten days after loading: the
    # 10-day curve reads 1.0 there and the 1000-day curve 0.5
    at_100 = C.creep_coefficient(spec, 100.0 + 10.0, 100.0)
    assert at_100 == pytest.approx(0.75)


def test_an_unknown_model_says_what_the_choices_are():
    with pytest.raises(ValueError, match="mc90"):
        C.creep_coefficient({"model": "b3"}, 100.0, 28.0)


def test_a_model_that_needs_a_notional_size_says_so():
    with pytest.raises(ValueError, match="notional_size"):
        C.creep_coefficient({"model": "mc90"}, 100.0, 28.0)


# --------------------------------------------------------------------------- #
#  Shrinkage
# --------------------------------------------------------------------------- #
def test_shrinkage_is_negative_and_starts_when_drying_starts():
    spec = {"model": "mc90", "notional_size": 200.0, "drying_start": 7.0}
    assert C.shrinkage_strain(spec, 5.0) == 0.0
    assert C.shrinkage_strain(spec, 7.0) == 0.0
    assert C.shrinkage_strain(spec, 1000.0) < 0.0


def test_mc90_shrinkage_against_the_formula_written_out_by_hand():
    """eps_s = [160 + 10 x 5 x (9 - 3.8)] 1e-6 = 420e-6;
    beta_rh = -1.55 (1 - 0.7^3) = -1.01835;
    beta_s(993) = sqrt(993 / (1400 + 993)) = 0.64417."""
    spec = {"model": "mc90", "notional_size": 200.0, "rh": 70.0,
            "fcm": 38.0e+6, "cement": 5.0, "drying_start": 7.0}
    want = -420.0e-6 * 1.01835 * 0.64417
    assert C.shrinkage_strain(spec, 1000.0) == pytest.approx(want, rel=1e-3)


def test_a_sealed_member_at_99_percent_rh_barely_shrinks():
    wet = {"model": "mc90", "notional_size": 200.0, "rh": 99.0}
    dry = {"model": "mc90", "notional_size": 200.0, "rh": 40.0}
    assert abs(C.shrinkage_strain(wet, 1000.0)) < 0.2 * abs(
        C.shrinkage_strain(dry, 1000.0))


def test_aci209_shrinkage_is_the_hyperbola_it_says_it_is():
    spec = {"model": "aci209", "ultimate": 780.0e-6, "drying_start": 7.0}
    assert C.shrinkage_strain(spec, 7.0 + 35.0) == pytest.approx(-390.0e-6)


# --------------------------------------------------------------------------- #
#  The superposition
# --------------------------------------------------------------------------- #
def test_the_history_is_the_sum_over_increments_and_not_over_states():
    """Two stations, two INCREMENTS. A history that superposed the states
    would double-count the first one."""
    h = C.CreepHistory(ZENER)
    h.add(28.0, np.full((1, 12), 1.0))
    h.add(68.0, np.full((1, 12), 3.0))         # increment of 2, not 3
    got = float(h.strain(1028.0)[0, 0])
    want = (C.creep_coefficient(ZENER, 1028.0, 28.0) * 1.0
            + C.creep_coefficient(ZENER, 1028.0, 68.0) * 2.0)
    assert got == pytest.approx(want)


def test_an_empty_history_imposes_nothing():
    assert C.CreepHistory(ZENER).strain(1000.0) is None


# --------------------------------------------------------------------------- #
#  Relaxation: the closed form, and the stability of the recursion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("t", [38.0, 68.0, 128.0, 428.0, 1028.0])
def test_relaxation_reproduces_the_zener_modulus(t):
    assert C.relaxation(ZENER, t, T0, n=2000) == pytest.approx(
        zener_relaxation(t - T0), abs=2e-4)


def test_what_is_left_of_that_error_is_the_curve_and_not_the_integral():
    """The creep function reaches the solver as a table read log-linearly,
    and a 400-point table of a smooth function is worth about 1e-4. Four
    times the points and the disagreement with the algebra falls by more
    than an order of magnitude, which is what says the remaining error is
    the interpolation and not the superposition."""
    def worst(npts):
        d = np.r_[0.0, np.exp(np.linspace(math.log(1e-4), math.log(1e+6), npts))]
        spec = {"model": "curve",
                "curve": [[float(x), PHI_INF * (1.0 - math.exp(-x / TAU))]
                          for x in d]}
        return max(abs(C.relaxation(spec, t, T0, n=2000) - zener_relaxation(t - T0))
                   for t in (38.0, 68.0, 128.0, 428.0, 1028.0))
    assert worst(400) < 2e-4
    assert worst(1600) < worst(400) / 8.0


def test_relaxation_is_not_the_effective_modulus_guess():
    """1/(1 + phi) is the answer only once phi has saturated. At 38 days it
    is 0.69 and the truth is 0.65 -- a 6% error on the way, in the direction
    that matters, and the reason this library solves the integral."""
    phi = C.creep_coefficient(ZENER, 38.0, T0)
    guess = 1.0 / (1.0 + phi)
    exact = zener_relaxation(38.0 - T0)
    assert exact < guess
    assert guess / exact == pytest.approx(1.06, rel=0.02)


@pytest.mark.parametrize("n", [6, 12, 24, 48, 200, 2000])
def test_the_recursion_is_stable_in_the_number_of_stations(n):
    """It was not. Dating each increment to the END of its interval made the
    recursion explicit, and at twenty-four stations it oscillated into a
    NEGATIVE stress (-0.05). The fix is the classical rule -- the increment
    arrives at the geometric middle of its interval and is solved for
    together with its own creep -- which is right from six stations on."""
    assert C.relaxation(ZENER, 1028.0, T0, n=n) == pytest.approx(1.0 / 3.0,
                                                                abs=2e-4)


def test_relaxation_before_the_load_is_applied_is_the_whole_of_it():
    assert C.relaxation(ZENER, 20.0, T0) == 1.0


# --------------------------------------------------------------------------- #
#  The solver's own time axis
# --------------------------------------------------------------------------- #
def _bar_case(steps, monitors, **material):
    """The 10 m bar of cases/case_29, held at the left end."""
    return {
        "name": "creep_unit",
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh",
                 "path": str(ROOT / "benchmarks/assets/creep_bar_8.msh")},
        "materials": {"concrete": {"type": "linear_elastic", "E": 3.0e+10,
                                   "nu": 0.2, "rho": 2400.0, **material}},
        "sections": {"bar": {"type": "beam", "material": "concrete",
                             "shape": "general", "area": 0.01, "iy": 8.33e-6,
                             "iz": 8.33e-6, "j": 1.41e-5}},
        "assign": [{"region": "bar", "section": "bar", "orientation": [0, 0, 1]}],
        "constraints": [{"region": "all", "dofs": [0, 1, 1, 1, 1, 1]},
                        {"region": "left", "dofs": [1, 1, 1, 1, 1, 1]}] + monitors,
        "steps": steps,
        "solver": {"mode": "relaxation", "backend": "numpy", "n_steps": 100000,
                   "age": 28.0, "max_steps_per_load": 20000,
                   "damping": {"type": "kinetic"}, "tolerance": 1e-6},
        "output": {"path": "results/_unit_creep.h5", "every": 1,
                   "fields": ["displacement"], "monitors": {}},
    }


def _run(case):
    prep = prepare_case(case)
    s = make_solver(_with_element(prep.case, prep.element, prep.case_path),
                    prep.mesh, prep.material)
    s.run(on_progress=lambda *a, **k: None)
    return s


def _axial(solver):
    """The axial force in the first element, from the member's own state.

    `end_actions` and not the `actions` the solver reports as fields: that
    one is max(|at i|, |at j|), a magnitude, and the sign is half the point
    here (docs/structural_framework.md).
    """
    solver.internal_force()
    return float(solver.end_actions[0][6])


HOLD = [{"region": "right", "dofs": [1, 1, 1, 1, 1, 1],
         "value": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]}]


@pytest.mark.parametrize("nstations", [6, 12])
def test_a_held_bar_relaxes_to_the_closed_form(nstations):
    """The whole chain: the material's creep spec, the history, the imposed
    strain, the station's own creep in its stiffness. 300 kN at age 28 and
    300 x R = 100 kN a thousand days later, to better than 1% on six
    stations."""
    s = _run(_bar_case([{"name": "stretch", "age": 28.0, "load_steps": 1},
                        {"name": "relax", "to_age": 1028.0,
                         "load_steps": nstations}],
                       HOLD, creep=ZENER))
    assert _axial(s) == pytest.approx(1.0e+5, rel=0.01)


def test_the_relaxing_bar_never_carries_more_than_it_started_with():
    """The regression that matters. The first version of this scheme dated
    each increment to the middle of its interval WITHOUT putting the current
    increment's creep in the station's stiffness, which is pure
    over-relaxation: the recorded increments ran +1.25, -0.55, ... +0.36,
    -0.73, +1.45 mm and the bar ended up carrying 332 kN, more than the
    300 kN it was stretched to. Relaxation cannot do that."""
    s = _run(_bar_case([{"name": "stretch", "age": 28.0, "load_steps": 1},
                        {"name": "relax", "to_age": 1028.0, "load_steps": 6}],
                       HOLD, creep=ZENER))
    n = np.abs(np.asarray(s.monitors["N"].values)) if "N" in s.monitors else None
    assert abs(_axial(s)) <= 3.0e+5 * 1.001
    assert n is None or np.all(n <= 3.0e+5 * 1.001)


def test_the_solver_keeps_the_age_of_every_frame():
    """A creep run's own axis is days. Without this the result file can only
    be plotted against a step counter, on which log-spaced stations are
    meaningless."""
    s = _run(_bar_case([{"name": "stretch", "age": 28.0, "load_steps": 1},
                        {"name": "relax", "to_age": 1028.0, "load_steps": 6}],
                       HOLD, creep=ZENER))
    assert len(s.ages) == len(s.times)
    assert s.ages[0] == pytest.approx(28.0)
    assert s.ages[-1] == pytest.approx(1028.0)
    assert np.all(np.diff(s.ages) >= 0.0)


def test_a_model_with_no_time_spec_is_untouched():
    s = _run(_bar_case([{"name": "stretch", "age": 28.0, "load_steps": 1}], HOLD))
    assert not s.time_dependent
    assert s.ages == []
    assert _axial(s) == pytest.approx(3.0e+5, rel=1e-3)


def test_ageing_makes_the_same_bar_stiffer_later():
    """Ec(t) reaches the whole local stiffness through one factor. A bar
    stretched at 7 days carries Ec(7)/Ec(28) of what it carries at 28 --
    which for s = 0.25 on strength is exp(-0.125) = 0.882: at a quarter of
    28 days the modulus is 12% short of its reference value."""
    ageing = {"ageing": {"s": 0.25, "of": "strength", "reference_age": 28.0}}
    young = _run(_bar_case([{"name": "stretch", "age": 7.0, "load_steps": 1}],
                           HOLD, **ageing))
    old = _run(_bar_case([{"name": "stretch", "age": 28.0, "load_steps": 1}],
                         HOLD, **ageing))
    ratio = _axial(young) / _axial(old)
    assert ratio == pytest.approx(C.modulus_factor(7.0, 0.25, "strength"),
                                  rel=1e-3)
    assert ratio == pytest.approx(0.882, rel=0.01)
