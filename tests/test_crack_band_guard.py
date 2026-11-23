"""`crack_band: both` holds the compression band factor near one, and says so.

WHY
    The tension curve is built FROM `l_ref`, so the crack-band factor
    k = l_e/l_ref is a change of variable on it and l_ref cancels. The
    compression curve is a code curve on a fixed strain axis with no
    reference length the solver knows, and `crack_band: both` multiplies the
    same k onto it with nothing behind it. On the Alberta beam `l_ref: 0.15`
    over a 15 mm mesh (k = 0.10) stretched the crushing branch tenfold; the
    compression zone never softened and the curve went 3% OVER the measured
    peak with no descent -- a wrong answer that looked better than the right
    one. A tenfold error was completely silent: the tension
    side has had a snap-back check since the crack band was written, the
    compression side had nothing.

    So k_c is held in [1/2, 2]: refused by the material when its points are
    bound, and by preflight before that, with the ratio and the fix.
"""
import numpy as np
import pytest
import yaml

from vfife.api import prepare_case
from vfife.core.diagnostics import UserError
from vfife.materials import create
from vfife.materials.cdp import COMPRESSION_BAND_MAX, COMPRESSION_BAND_MIN

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _cdp(**over):
    kw = dict(E=3.0e10, nu=0.2, rho=2400.0, l_ref=0.15, crack_band="both",
              compression={"source": "gb50010", "fc": 44e6},
              tension={"source": "fracture_energy", "ft": 3.42e6, "Gf": 140.0})
    kw.update(over)
    return create("cdp", **kw)


def test_a_reference_length_far_from_the_element_size_is_refused_with_the_ratio():
    m = _cdp()
    with pytest.raises(UserError) as exc:
        m.bind_points(np.full(8, 0.015))       # the xfine mesh: k = 0.10
    assert exc.value.code == "COMPRESSION_BAND_RATIO"
    assert "0.1..0.1" in exc.value.message
    assert "l_ref: auto" in exc.value.hint


def test_the_band_is_inclusive_and_named():
    m = _cdp()
    m.bind_points(np.array([0.15 * COMPRESSION_BAND_MIN, 0.15 * COMPRESSION_BAND_MAX]))
    assert "k_c = 0.5..2" in m.describe()
    with pytest.raises(UserError):
        m.bind_points(np.array([0.15 * COMPRESSION_BAND_MAX * 1.01]))


def test_a_graded_mesh_is_told_to_use_the_tension_band_only():
    """`auto` puts l_ref at the smallest element; a 3:1 mesh cannot be
    brought inside [1/2, 2] by it, and the hint must not send the user
    round in a circle."""
    m = _cdp(l_ref=0.1)
    with pytest.raises(UserError) as exc:
        m.bind_points(np.array([0.1, 0.3]))
    assert "graded" in exc.value.hint and "crack_band: tension" in exc.value.hint
    _cdp(l_ref=0.1, crack_band="tension").bind_points(np.array([0.1, 0.3]))   # allowed


def test_tension_only_and_none_are_untouched():
    for band in ("tension", "none"):
        _cdp(crack_band=band).bind_points(np.full(4, 0.015))


def _beam(tmp_path, l_ref):
    case = {
        "name": "g", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"dimensions": [1.0, 0.2, 0.2], "divisions": [8, 2, 2]},
        "materials": {"c": {"type": "cdp", "E": 3.0e10, "nu": 0.2, "rho": 2400.0,
                            "crack_band": "both", "l_ref": l_ref,
                            "compression": {"source": "gb50010", "fc": 44e6},
                            "tension": {"source": "fracture_energy", "ft": 3.42e6, "Gf": 140.0}}},
        "sections": {"body": {"type": "solid", "material": "c"}},
        "regions": {"fix": {"where": {"x": 0.0}}},
        "constraints": [{"region": "fix", "dofs": [1, 1, 1]}],
        "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 5},
        "output": {"path": str(tmp_path / "g.h5"), "monitors": {}},
    }
    path = tmp_path / "g.yaml"
    path.write_text(yaml.safe_dump(case), encoding="utf-8")
    return prepare_case(str(path))


def test_preflight_says_when_the_compression_band_has_no_length_scale(tmp_path):
    """Not an error -- it is what every early version did --
    but never silent: without Gc the peak load moves with the element size."""
    p = _beam(tmp_path, "auto")
    info = [i for i in p.preflight["issues"] if i["code"] == "COMPRESSION_UNREGULARISED"]
    assert info and info[0]["severity"] == "INFO"
    assert p.preflight["status"] == "PASS"


def test_preflight_reports_the_band_factor_before_the_solver_refuses_it(tmp_path):
    p = _beam(tmp_path, 0.15)                 # elements are 0.1 m: k = 0.67, allowed
    info = [i for i in p.preflight["issues"] if i["code"] == "CRACK_BAND"]
    assert info and info[0]["details"]["k_min"] == pytest.approx(0.1 / 0.15, rel=1e-6)
    p = _beam(tmp_path, 1.0)
    assert p.preflight["status"] == "FAIL"
    err = [i for i in p.preflight["issues"] if i["code"] == "COMPRESSION_BAND_RATIO"]
    assert err and err[0]["details"]["k_max"] == pytest.approx(0.1, rel=1e-6)
    p = _beam(tmp_path, "auto")
    assert p.preflight["status"] == "PASS"


# --------------------------------------------------------------------------- #
#  `units:` is read, not just stored
# --------------------------------------------------------------------------- #

def _units_case(tmp_path, units, E, rho):
    case = {
        "name": "u", "units": units,
        "mesh": {"dimensions": [1.0, 0.2, 0.2], "divisions": [4, 2, 2]},
        "materials": {"s": {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": rho}},
        "sections": {"body": {"type": "solid", "material": "s"}},
        "regions": {"fix": {"where": {"x": 0.0}}},
        "constraints": [{"region": "fix", "dofs": [1, 1, 1]}],
        "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 5},
        "output": {"path": str(tmp_path / "u.h5"), "monitors": {}},
    }
    path = tmp_path / f"{abs(hash((str(units), E, rho)))}.yaml"
    path.write_text(yaml.safe_dump(case), encoding="utf-8")
    return [i for i in prepare_case(str(path)).preflight["issues"]
            if i["code"] in ("UNIT_MAGNITUDE", "UNKNOWN_LENGTH_UNIT")]


SI = {"length": "m", "force": "N", "stress": "Pa"}
MM = {"length": "mm", "force": "N", "stress": "MPa"}


def test_a_consistent_case_says_nothing_in_either_system(tmp_path):
    """There is no conversion here and there should not be: the solver takes
    one consistent system, whichever it is. Both of these are consistent."""
    assert _units_case(tmp_path, SI, 2.1e11, 7850.0) == []
    assert _units_case(tmp_path, MM, 2.1e5, 7.85e-9) == []


def test_si_numbers_declared_as_millimetres_are_caught(tmp_path):
    """The wave speed is the invariant: sqrt(E/rho) read in the DECLARED
    length unit is 5.2 m/s here, which no structural solid has."""
    issues = _units_case(tmp_path, MM, 2.1e11, 7850.0)
    assert len(issues) == 1 and issues[0]["severity"] == "WARNING"
    assert "5.172 m/s" in issues[0]["message"]
    assert "units.length: m" in issues[0]["message"]      # and what to write instead


def test_megapascals_mixed_with_kilograms_per_cubic_metre_are_caught(tmp_path):
    """No length unit reconciles these, so the message must not pretend one
    does -- `units.length: km` puts the arithmetic in the band and would be
    worse advice than none."""
    issues = _units_case(tmp_path, SI, 2.1e5, 7850.0)
    assert len(issues) == 1
    assert "No length unit reconciles them" in issues[0]["message"]
    assert "km" not in issues[0]["message"]


def test_an_unknown_length_unit_is_reported_rather_than_guessed(tmp_path):
    issues = _units_case(tmp_path, {"length": "cubits"}, 2.1e11, 7850.0)
    assert len(issues) == 1 and issues[0]["code"] == "UNKNOWN_LENGTH_UNIT"


def test_a_case_that_declares_no_units_is_not_second_guessed(tmp_path):
    assert _units_case(tmp_path, {}, 2.1e11, 7850.0) == []
