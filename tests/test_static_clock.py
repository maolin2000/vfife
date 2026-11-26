"""The implicit static solver runs the concrete clock:
every converged increment of a step with `to_age` is a time station, as
every level of a staged relaxation is, so creep, shrinkage and the
relaxation of prestressed bars come out the same in both -- and against
their closed forms."""
import copy

import numpy as np
import pytest

import test_creep as TC
import test_review4_creep as T4
from vfife.core import relaxation as R

STATIC = {"mode": "static", "age": 28.0, "tolerance": 1e-9}


def _go(case):
    """Run a case the way `vfife run` does: a static one by `implicit.run`
    (the solvers' own `run` is the relaxation and the dynamics)."""
    from vfife import implicit
    s = T4._solver(case)
    if case["solver"]["mode"] == "static":
        implicit.run(s, output_every=1)
    else:
        s.run(on_progress=lambda *a, **k: None)
    return s


def _static(case, backend="numpy"):
    c = copy.deepcopy(case)
    c["solver"] = dict(STATIC, backend=backend)
    return c


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("nstations", [6, 12])
def test_a_held_bar_relaxes_to_the_closed_form_in_static_steps_too(nstations, backend):
    """300 kN at age 28 held a thousand days on a Zener law: 100 kN, and
    the same number as the staged relaxation to 1e-9."""
    case = TC._bar_case([{"name": "stretch", "age": 28.0, "load_steps": 1},
                         {"name": "relax", "to_age": 1028.0, "load_steps": nstations}],
                        TC.HOLD, creep=TC.ZENER)
    s = _go(_static(case, backend))
    assert TC._axial(s) == pytest.approx(1.0e+5, rel=0.01)
    assert TC._axial(s) == pytest.approx(TC._axial(_go(case)), rel=1e-9)
    assert s.ages[0] == pytest.approx(28.0) and s.ages[-1] == pytest.approx(1028.0)


def test_a_prestressed_bar_loses_r_of_its_force_in_static_steps():
    spec = {"model": "power", "rho_1000": 0.031, "k": 0.21}
    s = _go(_static(T4._prestressed_bar(spec)))
    N = -T4._mon(s, "N")
    P0 = 1.488e9 * 1.4e-4
    want = [P0 * (1.0 - R.relaxation_loss(spec, (a - 28.0) * 24.0)) for a in np.asarray(s.ages)]
    assert N == pytest.approx(want, rel=1e-6)
    assert N[-1] < 0.96 * P0


def test_creep_under_a_held_load_is_the_relaxation_runs_answer():
    """A cantilevered bar under 300 kN for 300 days: its end creeps by phi
    times the elastic shortening, the same in both procedures."""
    case = T4._bar_case({"creep": TC.ZENER})
    s = _go(_static(case))
    r = _go(case)
    assert T4._mon(s, "u")[-1] == pytest.approx(T4._mon(r, "u")[-1], rel=1e-6)
    assert T4._mon(s, "u")[-1] > 1.5 * T4._mon(s, "u")[0]


def test_a_static_model_with_no_time_spec_has_no_clock():
    case = TC._bar_case([{"name": "stretch", "age": 28.0, "load_steps": 1}], TC.HOLD)
    s = _go(_static(case))
    assert not s.time_dependent and s.ages == []
    assert TC._axial(s) == pytest.approx(3.0e+5, rel=1e-6)


def test_creep_with_arc_length_is_refused():
    case = _static(TC._bar_case([{"name": "stretch", "age": 28.0, "load_steps": 1},
                                 {"name": "relax", "to_age": 1028.0, "load_steps": 6}],
                                TC.HOLD, creep=TC.ZENER))
    case["solver"]["static"] = {"arc_length": {"initial": 0.1}}
    with pytest.raises(ValueError, match="arc_length"):
        _go(case)


# ------------------------------------------------ a jacked bar's own clock
SPEC = {"model": "power", "rho_1000": 0.031, "k": 0.21}


def _jacked(tmp_path, steps, backend="numpy", stay=None):
    """test_jack's cantilever and stay with the tip held too, so the stay's
    strain is held and its force falls by r exactly; the stay's steel
    relaxes."""
    import test_jack as TJ
    c = TJ._case(tmp_path, steps, stay or {"jack": TJ.T}, backend)
    c["materials"]["strand"] = dict(c["materials"]["steel"], relaxation=dict(SPEC))
    c["sections"]["c"]["material"] = "strand"
    c["constraints"].append({"region": "B", "dofs": [1, 1, 1, 1, 1, 1]})
    c["solver"]["age"] = 28.0
    from vfife.api import run_case
    got = {}
    out = run_case(c, on_ready=lambda s, _p: got.__setitem__("s", s))
    assert out.status == "CONVERGED", out.message
    s = got["s"]
    e = int(np.flatnonzero(np.asarray(s.member_kind) == "truss2")[0])
    return s, float(s.end_actions[e, 6]), TJ.T


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_a_jacked_bar_relaxes_from_its_lock_off(tmp_path, backend):
    """Jacked at 28 days, held to 1028: T (1 - r(1000 days)) -- the
    relaxation dated from the lock-off and from the strain the jack left
    (refused before v1.16: a jacked bar has no `prestress`)."""
    s, N, T = _jacked(tmp_path, [{"name": "stress", "load_steps": 1},
                                 {"name": "hold", "to_age": 1028.0, "load_steps": 4}], backend)
    assert N == pytest.approx(T * (1.0 - R.relaxation_loss(SPEC, 1000.0 * 24.0)), rel=1e-6)


def test_a_bar_jacked_later_relaxes_from_then(tmp_path):
    """Built and jacked at 100 days: its clock starts at 100, not at the
    analysis start, and it has lost nothing before it."""
    s, N, T = _jacked(tmp_path, [{"name": "wait", "to_age": 100.0, "load_steps": 2},
                                 {"name": "stress", "activate": ["stay"], "load_steps": 1},
                                 {"name": "hold", "to_age": 1100.0, "load_steps": 4}])
    assert N == pytest.approx(T * (1.0 - R.relaxation_loss(SPEC, 1000.0 * 24.0)), rel=1e-6)


def test_a_prestressed_bar_born_later_relaxes_from_its_birth(tmp_path):
    """A stay with `prestress` built at 100 days: from 100, and it has not
    lost what the model's stressing age would have taken (refused before
    v1.16)."""
    import test_jack as TJ
    s, N, T = _jacked(tmp_path, [{"name": "wait", "to_age": 100.0, "load_steps": 2},
                                 {"name": "build", "activate": ["stay"], "load_steps": 1},
                                 {"name": "hold", "to_age": 1100.0, "load_steps": 4}],
                      stay={"prestress": TJ.T / TJ.A})
    assert N == pytest.approx(T * (1.0 - R.relaxation_loss(SPEC, 1000.0 * 24.0)), rel=1e-6)


@pytest.mark.parametrize("case", ["case_29d_fiber_creep_bar", "case_29g_prestress_loss"])
def test_a_fibre_models_creep_in_static_steps_is_the_relaxation_runs(tmp_path, case):
    """Fibre sections creep, shrink and relax in Newton steps too (1.22;
    refused until then): every converged increment is a time station, as
    every level of the staged relaxation is, and the two give the same
    history station by station."""
    import os
    import yaml
    from vfife.api import run_case
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cases")
    base = yaml.safe_load(open(os.path.join(root, f"{case}.yaml"), encoding="utf-8"))
    if base["mesh"].get("path"):
        base["mesh"]["path"] = os.path.normpath(os.path.join(root, base["mesh"]["path"]))
    got = {}
    for mode in ("relaxation", "static"):
        d = copy.deepcopy(base)
        d["solver"]["mode"] = mode
        if mode == "static":
            for st in d.get("steps") or []:
                st.pop("max_steps_per_load", None)
        d["output"]["path"] = str(tmp_path / f"{mode}.h5")
        out = run_case(d)
        assert out.status == "CONVERGED", out.message
        got[mode] = {k: np.asarray(v, float) for k, v in out.monitors.items()}
    for k, v in got["relaxation"].items():
        if k in ("residual", "kinetic_energy", "time"):
            continue
        scale = np.abs(v).max() or 1.0
        np.testing.assert_allclose(got["static"][k], v, atol=2e-5 * scale, err_msg=k)
