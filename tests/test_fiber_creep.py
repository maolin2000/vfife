"""Creep and shrinkage on a FIBRE section.

The mechanism is v0.4's, one level down. The beam path subtracts a free
DEFORMATION from the member's and divides the member stiffness by
`(1 + phi_cur)`; a fibre section has no member law to divide, so the free
STRAIN goes into the fibre's imposed-strain slot -- the one a temperature and
a tendon's prestrain already use -- and the implicit half of the step is a
scale on the law's own input (`concrete02.strain_scale`).

Nothing but `core.creep` is shared between the two paths. That is what makes
the agreement below evidence rather than a tautology, and it is the first
thing this file asserts.
"""
import copy
import pathlib
import sys

import numpy as np
import pytest
import yaml

from vfife.api import _with_element, prepare_case
from vfife.core import creep
from vfife.core.structural import make_structural_solver

CASES = pathlib.Path(__file__).resolve().parents[1] / "cases"
T0, T_END, P0 = 28.0, 1028.0, 3.0e+5


def _solve(path, patch=None, tmp_path=None):
    """Solve a case, optionally with a patched copy written beside it."""
    if patch is not None:
        case = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
        patch(case)
        # beside the original, so the mesh's relative path still resolves
        # a name of its own: parallel workers (pytest -n) wrote the same
        # _tmp_ file and solved each other's patch
        import os, uuid
        path = pathlib.Path(path).with_name(f"_tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}_"
                                            + pathlib.Path(path).name)
        path.write_text(yaml.safe_dump(case), encoding="utf-8")
    try:
        prep = prepare_case(str(path))
        p = _with_element(prep.case, prep.element, prep.case_path)
        s = make_structural_solver(p, prep.mesh, prep.material)
        s.run()
        return s
    finally:
        if patch is not None:
            pathlib.Path(path).unlink(missing_ok=True)


def _monitor(s, name):
    return np.asarray(s.monitors[name].values, float)


def _spec(path):
    case = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    return case["materials"]["concrete"]["creep"]


# ------------------------------------------------- the two paths on one problem
@pytest.mark.slow
def test_the_fibre_path_and_the_beam_path_relax_identically():
    """case_29b is a linear-elastic MEMBER law with `1/(1 + phi_cur)` on the
    member stiffness; case_29e is sixteen concrete02 FIBRES with the same
    factor as a scale on the law's own input. Different slot for the free
    strain, different mechanism for the implicit step, different place the
    elastic history is read from -- `epsP` in the fibre state against the
    member's local deformation.

    They land on the same thirteen numbers to a millionth. A number that
    survives that is not an artefact of either implementation.
    """
    beam = _solve(CASES / "case_29b_relaxation.yaml")
    fibre = _solve(CASES / "case_29e_fiber_relaxation.yaml")
    assert np.allclose(beam.ages, fibre.ages, rtol=1e-12)
    a, b = _monitor(beam, "N_left"), _monitor(fibre, "N_left")
    assert len(a) == 13
    assert np.max(np.abs(b / a - 1.0)) < 1e-5


# ------------------------------------------------------------ against algebra
@pytest.mark.slow
@pytest.mark.parametrize("n,worst", [(6, 0.035), (12, 0.015), (24, 0.008)])
def test_refining_the_time_axis_moves_towards_the_integral(n, worst):
    """The whole decay against `core.creep.relaxation`, not just its end.

    THE END IS EASY: every scheme reaches the asymptote, and `1/(1 + phi)`
    would reach the same 100 kN. The middle is where the implicit step is
    worth having, and v0.4 measured the explicit one there getting WORSE as
    the axis is refined -- twelve stations landing on 0.317 of the initial
    force where the answer is 1/3, twenty-four worse still. Here: 2.8%, 1.0%,
    0.4%.
    """
    path = CASES / "case_29e_fiber_relaxation.yaml"
    spec = _spec(path)

    def patch(case):
        case["steps"][1]["load_steps"] = n

    s = _solve(path, patch)
    ages = np.asarray(s.ages, float)
    got = -_monitor(s, "N_left")
    exact = np.array([P0 * creep.relaxation(spec, float(t), T0) for t in ages])
    err = np.max(np.abs(got / exact - 1.0))
    assert err < worst, f"{n} stations: worst {100 * err:.2f}%"
    # and the endpoint is exact whatever the axis
    assert got[-1] == pytest.approx(1.0e+5, rel=2e-3)


@pytest.mark.slow
def test_a_determinate_fibre_bar_creeps_freely_and_carries_nothing_new():
    """The other half, and the one that catches the commonest mistake: creep
    put in as a load. The elongation triples and the axial force does not move
    by a newton -- on a section whose sixteen fibres each keep their own
    history."""
    s = _solve(CASES / "case_29d_fiber_creep_bar.yaml")
    u = _monitor(s, "u_right")
    N = _monitor(s, "N_left")
    assert u[0] == pytest.approx(0.010, rel=2e-3)
    assert u[-1] == pytest.approx(0.030, rel=2e-3)
    assert np.max(np.abs(N / N[0] - 1.0)) < 1e-4
    # the SECTION strain tripled while the material stayed at the elastic one
    assert _monitor(s, "eps_fibre")[-1] == pytest.approx(0.003, rel=2e-3)


@pytest.mark.slow
def test_the_fused_kernel_and_the_reference_path_creep_identically():
    """The scale has to be written TWICE -- into `concrete02.strain_scale`,
    which the reference path rebuilds its parameters from on every call, and
    into column 7 of the kernel's own `_cpar`. A scale that reached only one
    of them would be invisible until a case ran, and would then look like a
    creep result rather than a bug. Thirteen stations, agreeing to 4e-14.
    """
    def solve(force_reference):
        prep = prepare_case(str(CASES / "case_29e_fiber_relaxation.yaml"))
        p = _with_element(prep.case, prep.element, prep.case_path)
        s = make_structural_solver(p, prep.mesh, prep.material)
        if force_reference:
            s._kernel = None
        s.run()
        return s

    fast = solve(False)
    if fast.kernel_status.startswith("numba fused"):
        ref = solve(True)
        a, b = _monitor(fast, "N_left"), _monitor(ref, "N_left")
        assert np.max(np.abs(b / a - 1.0)) < 1e-10
    # and both halves of the scale hold the same number at the last station
    f = 1.0 / (1.0 + fast._phi_cur)
    assert float(fast.materials[0].strain_scale) == pytest.approx(f, rel=1e-14)
    if getattr(fast, "_cpar", None) is not None:
        assert float(fast._cpar[0, 0, 7]) == pytest.approx(f, rel=1e-14)


# --------------------------------------------------------------- the two clocks
def test_the_two_solvers_put_their_stations_at_the_same_ages():
    """`core.creep` owns the clock -- `read_step_ages`, `age_at`,
    `station_phi`, `station_date` -- because two solvers run on it and two
    copies would drift. This is the assertion that keeps it single-sourced:
    the step spec a case file writes must produce the same station ages and
    the same phi sequence whichever solver reads it.
    """
    step = {"age": 28.0, "to_age": 1028.0}
    ages = creep.read_step_ages(step, None, 28.0)
    assert ages == (28.0, 1028.0, "log")
    # a step with no `age` continues from where the last one left off
    assert creep.read_step_ages({"to_age": 90.0}, 55.0, 28.0)[0] == 55.0
    stations = [creep.age_at(ages, (i + 1) / 12) for i in range(12)]
    assert stations[-1] == pytest.approx(1028.0)
    assert np.all(np.diff(stations) > 0)
    # log spacing: equal RATIOS, not equal differences
    r = np.asarray(stations[1:]) / np.asarray(stations[:-1])
    assert np.ptp(r) < 1e-12
    # the increment of a station is dated at the geometric middle, and that is
    # the same date `station_phi` assumed while the station was being solved
    prev = 100.0
    assert creep.station_date(400.0, prev) == pytest.approx(200.0)
    assert creep.station_phi(None, 400.0, prev) == 0.0
    assert creep.station_phi({"model": "aci209"}, 400.0, None) == 0.0


def test_the_first_station_of_an_analysis_carries_no_creep():
    """Nothing has elapsed before it, so the instant a load arrives is elastic
    -- which is what makes `reduction: first` an exact metric in both
    contracts."""
    spec = {"model": "aci209", "ultimate": 2.0}
    assert creep.station_phi(spec, 28.0, None) == 0.0
    assert creep.station_phi(spec, 28.0, 28.0) == 0.0
    assert creep.station_phi(spec, 128.0, 28.0) > 0.0


# ------------------------------------------------------------- what it refuses
def _refuses(patch, match):
    with pytest.raises(ValueError, match=match):
        _solve(CASES / "case_29d_fiber_creep_bar.yaml", patch)


def test_ageing_on_a_fibre_section_is_refused_and_says_what_to_do_instead():
    """Ageing changes Ec, and on a nonlinear law that is not one number -- the
    peak strain and the strength age with it -- so it cannot ride the strain
    scale creep uses. Silently ignoring a declared `ageing` would be the worst
    outcome: the case would read as if it had one."""
    def patch(case):
        case["materials"]["concrete"]["ageing"] = {"s": 0.25, "of": "modulus"}
    _refuses(patch, "ageing")


def test_creep_on_a_steel_fibre_group_is_refused():
    """Steel does not creep; it relaxes, which is a different law and is not
    in this library. Accepting the key would produce a plausible-looking
    prestress loss computed by the wrong physics."""
    def patch(case):
        case["materials"]["strand"] = {"type": "steel02", "E": 1.95e+11,
                                       "Fy": 1.6e+9, "b": 0.01, "R0": 20.0,
                                       "rho": 7850,
                                       "creep": {"model": "aci209"}}
        case["sections"]["s1"] = {"type": "bar", "material": "strand",
                                  "shape": "general", "area": 1.0e-4}
        case["sections"]["bar"]["bars"] = [{"y": 0.0, "z": 0.0, "section": "s1"}]
    _refuses(patch, "REINFORCEMENT")


def test_creep_on_a_law_with_no_strain_scale_is_refused():
    """A fibre station carries the creep of the interval it is arriving in by
    scaling the law's own input. `linear_elastic` has no such knob, and for a
    linear fibre the same thing is said by declaring the modulus you mean."""
    def patch(case):
        c = case["materials"]["concrete"]
        case["materials"]["concrete"] = {"type": "linear_elastic", "E": 3.0e+10,
                                         "nu": 0.2, "rho": 2400.0,
                                         "creep": c["creep"]}
    _refuses(patch, "strain_scale")


def test_two_different_creep_laws_in_one_model_are_refused():
    """Buildable -- the scale and the imposed strain are already per fibre --
    but the history and the station's own phi are kept once for the model. The
    refusal is there so that a model with two laws cannot quietly run on the
    first one."""
    def patch(case):
        c = case["materials"]["concrete"]
        core = copy.deepcopy(c)
        core["creep"] = {"model": "aci209", "ultimate": 1.5}
        case["materials"]["core"] = core
        case["sections"]["bar"]["cover"] = 0.02
        case["sections"]["bar"]["core_material"] = "core"
    _refuses(patch, "DIFFERENT")


# ------------------------------------------------------------- the null result
def test_a_fibre_model_with_no_creep_is_untouched():
    """Every fibre case in the library goes through the new code path and must
    come out bit-identical. `time_dependent` is False, `_refresh_imposed`
    returns where it always did, and `_cpar` is never written."""
    path = CASES / "case_29d_fiber_creep_bar.yaml"

    def patch(case):
        case["materials"]["concrete"].pop("creep")
        case["steps"] = [{"name": "load", "load_steps": 1,
                          "max_steps_per_load": 40000}]
    s = _solve(path, patch)
    assert s.time_dependent is False
    assert s.ages == []
    # the elastic answer, and nothing after it
    assert _monitor(s, "u_right")[-1] == pytest.approx(0.010, rel=2e-3)
    assert float(s.materials[0].strain_scale) == 1.0


# ---------------------------------------------------- relaxation
def test_the_relaxation_law_is_its_own_definition():
    """`r(t) = rho_1000 (t/1000 h)^k`, zero at and before t = 0, and equal to
    `rho_1000` at 1000 hours whatever `k` is -- which is what makes rho_1000
    the number a code quotes."""
    from vfife.core import relaxation as R
    spec = {"model": "power", "rho_1000": 0.031, "k": 0.21}
    assert R.relaxation_loss(spec, 0.0) == 0.0
    assert R.relaxation_loss(spec, -5.0) == 0.0
    assert R.relaxation_loss(spec, 1000.0) == pytest.approx(0.031, rel=1e-12)
    for k in (0.1, 0.21, 0.4):
        s = dict(spec, k=k)
        assert R.relaxation_loss(s, 1000.0) == pytest.approx(0.031, rel=1e-12)
    # the reduction factor is a plain multiplier and nothing else
    assert R.relaxation_loss(dict(spec, reduction=0.8), 4000.0) == pytest.approx(
        0.8 * R.relaxation_loss(spec, 4000.0), rel=1e-12)


def test_neither_coefficient_has_a_default():
    """A default here would be a number nobody chose, and it would come out as
    a plausible prestress loss rather than as an error. This is the assertion
    that keeps the absence deliberate."""
    from vfife.core import relaxation as R
    for bad in ({"model": "power"}, {"model": "power", "k": 0.21},
                {"model": "power", "rho_1000": 0.031}):
        with pytest.raises(ValueError, match="rho_1000"):
            R.relaxation_loss(bad, 1000.0)


def test_the_power_law_reproduces_the_published_curve_it_was_fitted_to():
    """Ghali (2000) Fig. 7 plots CEB-FIP MC90 for low-relaxation steel at
    0.8 f_pu, for the Confederation Bridge's own strand. Digitised it gives
    2.20% at 250 h, 3.11% at 1000 and 4.44% at 6000; `rho_1000 = 0.031,
    k = 0.21` hits all three within 5.3%, which is the digitisation's own
    accuracy. That is the only provenance these coefficients have, and it is
    why the case file says so."""
    from vfife.core import relaxation as R
    spec = {"model": "power", "rho_1000": 0.031, "k": 0.21}
    for t, read in ((250.0, 0.0220), (1000.0, 0.0311), (6000.0, 0.0444)):
        assert R.relaxation_loss(spec, t) == pytest.approx(read, rel=0.06)


@pytest.mark.slow
def test_a_tendon_held_at_its_strain_loses_exactly_r():
    """case_29f: both ends held, so the section strain is zero, so the support
    reaction IS the tendon force. The RATIO is the near-exact statement -- the
    absolute force sits 0.45% below A_p sigma_pi because steel02's
    Menegotto-Pinto curve has already begun to soften at 0.89 of yield strain
    (0.23% before v1.0, when steel02 was not yet OpenSees' Steel02), and
    that offset is nearly constant, so it cancels out of the ratio to 2.3e-4:
    the loss is taken on a curve whose tangent there is 0.91 E."""
    from vfife.core import relaxation as R
    s = _solve(CASES / "case_29f_tendon_relaxation.yaml")
    spec = {"model": "power", "rho_1000": 0.031, "k": 0.21}
    ages = np.asarray(s.ages, float)
    N = -_monitor(s, "N_left")
    ratio = N / N[0]
    exact = np.array([1.0 - R.relaxation_loss(spec, (t - 28.0) * 24.0) for t in ages])
    assert np.max(np.abs(ratio - exact)) < 3.0e-4
    # and the absolute value is 0.45% under the algebra, by the same amount
    # at every station -- a material effect, not a creep one
    off = N / (1.488e+9 * 1.40e-4 * exact) - 1.0
    assert np.all(off < 0.0) and np.ptp(off) < 3.0e-4


@pytest.mark.slow
def test_the_three_losses_are_less_than_their_sum():
    """case_29g's physics, measured on the solver rather than on the hand
    method: each loss relieves the concrete and so reduces the next. Run the
    same bar with one law at a time and with all three, and the total has to
    come out SMALLER than the sum of the parts."""
    path = CASES / "case_29g_prestress_loss.yaml"

    def only(*keep):
        def patch(case):
            if "creep" not in keep:
                case["materials"]["concrete"].pop("creep")
            if "shrinkage" not in keep:
                case["materials"]["concrete"].pop("shrinkage")
            if "relaxation" not in keep:
                case["materials"]["strand"].pop("relaxation")
        return patch

    base = _monitor(_solve(path, only()), "sig_p")
    assert np.ptp(base) < 1.0e-3 * base[0]        # nothing time-dependent left
    parts = {nm: base[0] - _monitor(_solve(path, only(nm)), "sig_p")[-1]
             for nm in ("creep", "shrinkage", "relaxation")}
    total = base[0] - _monitor(_solve(path, only("creep", "shrinkage",
                                                 "relaxation")), "sig_p")[-1]
    assert all(v > 0 for v in parts.values())
    assert total < sum(parts.values())
    # and the interaction is a few percent, not a rounding error
    assert 0.02 < sum(parts.values()) / total - 1.0 < 0.12


@pytest.mark.slow
def test_the_step_by_step_answer_stays_inside_the_hand_methods_bracket():
    """chi = 0 and chi = 1 bracket the integral by construction. The measured
    effective chi runs from 0.22 at 38 days to 0.99 at 1028 -- it is NOT the
    textbook constant 0.8, and that is the finding case_29g records."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))
    import generate_creep_bar as cb          # noqa: E402
    s = _solve(CASES / "case_29g_prestress_loss.yaml")
    ages = np.asarray(s.ages, float)
    sig = _monitor(s, "sig_p")
    for t, v in zip(ages[1:], sig[1:]):
        lo = cb.prestress_loss(float(t), chi=0.0)[1]
        hi = cb.prestress_loss(float(t), chi=1.0)[1]
        assert lo <= v <= hi * 1.001, f"at {t:.0f} d: {v:.3e} outside [{lo:.3e}, {hi:.3e}]"
    early = cb.prestress_loss(float(ages[1]), chi=0.5)[1]
    assert sig[1] < early                      # effective chi below 0.5 early on
    late = cb.prestress_loss(float(ages[-1]), chi=0.95)[1]
    assert sig[-1] > late                      # and above 0.95 late


def test_relaxation_on_concrete_is_refused():
    """Concrete's version of this is `creep`, and they are different laws."""
    def patch(case):
        case["materials"]["concrete"]["relaxation"] = {"model": "power",
                                                       "rho_1000": 0.03, "k": 0.2}
    with pytest.raises(ValueError, match="CONCRETE"):
        _solve(CASES / "case_29g_prestress_loss.yaml", patch)


def test_relaxation_on_a_steel_that_carries_no_prestrain_is_refused():
    """An ordinary bar's stress comes from the structure and is not held, so
    there is no initial stress to relax from -- the declaration would silently
    do nothing, which is the worst of the three outcomes."""
    def patch(case):
        case["materials"]["rebar"] = {"type": "steel02", "E": 2.0e+11,
                                      "Fy": 4.0e+8, "b": 0.01, "R0": 20.0,
                                      "rho": 7850,
                                      "relaxation": {"model": "power",
                                                     "rho_1000": 0.03, "k": 0.2}}
        case["sections"]["r1"] = {"type": "bar", "material": "rebar",
                                  "shape": "general", "area": 1.0e-4}
        case["sections"]["bar"]["bars"] = [{"y": 0.03, "z": 0.0, "section": "r1"}]
    with pytest.raises(ValueError, match="PRESTRAIN"):
        _solve(CASES / "case_29g_prestress_loss.yaml", patch)
