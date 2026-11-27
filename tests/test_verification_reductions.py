"""Contract reductions that read a second monitor: at, energy, at_peak_of.

WHY
    Every RC-beam contract in this project held one number against the
    measurement: the peak load. A model with the right peak and the wrong
    shape -- stiffer before yield, no descent after -- passed it. Version 0.1
    found exactly that: an l_ref mistake put the xfine curve 3% OVER the
    measured peak with no descending branch, and a peak-only contract would
    have called it the best result yet. The load at a given deflection, the
    area under the curve and the deflection at the peak are what tell the
    shape, and they need the OTHER monitor to be defined.
"""
from pathlib import Path

import h5py
import numpy as np
import pytest

from vfife.verification import _reduce_against, verify
from vfife.base.compat import trapezoid


def _spec(name, reduction, **kw):
    return {"name": name, "reduction": reduction, **kw}


# a load-deflection record the way the monitors hold it: converged frames
# only, reaction positive, deflection negative, first frame not the origin
MID = np.array([-1.0, -2.0, -3.0, -4.0, -5.0]) * 1e-3
P = np.array([100.0, 180.0, 220.0, 210.0, 190.0]) * 1e3


def test_at_interpolates_this_monitor_where_the_other_first_reaches_a_level():
    v = _reduce_against(P, MID, _spec("P", "at", value=-2.5e-3), "mid")
    assert v == pytest.approx(200.0e3)
    # exactly on a frame
    assert _reduce_against(P, MID, _spec("P", "at", value=-3.0e-3), "mid") == pytest.approx(220e3)
    # the first frame itself
    assert _reduce_against(P, MID, _spec("P", "at", value=-1.0e-3), "mid") == pytest.approx(100e3)


def test_at_takes_the_first_crossing_of_a_record_that_comes_back():
    x = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    y = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    assert _reduce_against(y, x, _spec("y", "at", value=0.5), "x") == pytest.approx(5.0)
    assert _reduce_against(y, x, _spec("y", "at", value=1.5), "x") == pytest.approx(15.0)


def test_at_refuses_a_level_the_record_never_reaches_and_names_the_range():
    with pytest.raises(ValueError, match="never reaches -0.01"):
        _reduce_against(P, MID, _spec("P", "at", value=-1.0e-2), "mid")
    with pytest.raises(ValueError, match="from_origin"):
        _reduce_against(P, MID, _spec("P", "at", value=-0.5e-3), "mid")
    # ... and from_origin makes the segment before the first frame readable
    v = _reduce_against(P, MID, _spec("P", "at", value=-0.5e-3, from_origin=True), "mid")
    assert v == pytest.approx(50.0e3)
    with pytest.raises(ValueError, match="needs `value`"):
        _reduce_against(P, MID, _spec("P", "at"), "mid")


def test_energy_is_the_area_under_the_curve_as_a_magnitude():
    # by hand: trapezoids of P over |mid| between the frames
    area = trapezoid(P, -MID)
    assert _reduce_against(P, MID, _spec("P", "energy"), "mid") == pytest.approx(area)
    assert _reduce_against(P, MID, _spec("P", "energy", signed=True), "mid") == pytest.approx(-area)
    # from the origin adds the first triangle, 0.5 * 100 kN * 1 mm = 50 J
    v = _reduce_against(P, MID, _spec("P", "energy", from_origin=True), "mid")
    assert v == pytest.approx(area + 50.0)


def test_at_peak_of_reads_this_monitor_where_the_other_is_largest():
    assert _reduce_against(MID, P, _spec("mid", "at_peak_of"), "P") == pytest.approx(-3.0e-3)


def test_records_of_different_length_are_refused():
    with pytest.raises(ValueError, match="same record"):
        _reduce_against(P[:3], MID, _spec("P", "energy"), "mid")


def test_the_contract_carries_them_end_to_end(tmp_path):
    result = tmp_path / "r.h5"
    with h5py.File(result, "w") as h5:
        h5.attrs["case_yaml"] = "name: demo\nsolver: {backend: numpy}\n"
        mon = h5.create_group("monitors")
        mon.create_dataset("P", data=P)
        mon.create_dataset("mid", data=MID)
    contract = tmp_path / "c.yaml"
    contract.write_text("""version: 1
case: demo
classification: regression
metrics:
  - id: load_at_2p5mm
    source: monitor
    name: P
    reduction: at
    against: mid
    value: -2.5e-3
    expect: {target: 200.0e3, rtol: 1.0e-9}
  - id: energy
    source: monitor
    name: P
    reduction: energy
    against: mid
    from_origin: true
    expect: {lower: 800.0}
  - id: deflection_at_peak
    source: monitor
    name: mid
    reduction: at_peak_of
    against: P
    expect: {target: -3.0e-3, rtol: 1.0e-9}
  - id: forgot_against
    source: monitor
    name: P
    reduction: at
    value: -2.5e-3
    expect: {finite: true}
""", encoding="utf-8")
    report = verify(contract, result)
    by_id = {c["id"]: c for c in report["checks"]}
    assert by_id["load_at_2p5mm"]["status"] == "PASS"
    assert by_id["energy"]["status"] == "PASS" and by_id["energy"]["value"] == pytest.approx(805.0)
    assert by_id["deflection_at_peak"]["status"] == "PASS"
    assert by_id["forgot_against"]["status"] == "FAIL"
    assert "against" in by_id["forgot_against"]["message"]


def test_first_and_delta_read_the_start_of_a_record_and_what_the_last_step_added():
    """A staged run's monitors hold one frame per converged step: `first` is
    the built (dead-load) state and `delta` is the last step's increment
    over it -- a live load's share of a cable's tension."""
    from vfife.verification import _reduce
    H = np.array([-3.75e7, -4.45e7])
    assert _reduce(H, "first") == pytest.approx(-3.75e7)
    assert _reduce(H, "delta") == pytest.approx(-7.0e6)
    assert _reduce(H, "final") == pytest.approx(-4.45e7)
    assert _reduce(np.array([[1.0], [3.0], [6.0]]), "delta") == pytest.approx(5.0)
    with pytest.raises(ValueError, match="unknown reduction"):
        _reduce(H, "last")
