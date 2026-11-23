"""Monitors that read elements and bars, so a contract can reach them.

THE LIMIT THIS EXISTS FOR
    A verification contract reads three sources: `monitor`, `field` (a
    reduction over the WHOLE model) and `diagnostic`. Two things a
    reinforced-concrete case is built around fell outside all three.

    Bar stress. cases/case_15 applies 679 MPa of prestrain so the tendon
    carries the measured 633 MPa, and its contract cannot check that: bar
    stress is written to `rebar_steps/`, which the contract layer does not
    read. It asserts the camber instead and says so in a comment.

    A field inside a region. The same case wants "the constant-moment region
    has not crushed". `source: field` can only say "no element anywhere has",
    and the global maximum lives at the anchorage (dc = 0.46) rather than in
    the span (0.14), so the threshold had to be loosened to 0.60 and the
    comment records that it is held globally because a contract cannot
    restrict an element field to a region.

    Both are monitor-shaped. So they are monitors.
"""
import numpy as np
import pytest
import yaml

from vfife.api import run_case
from vfife.io import schema
from vfife.io.model import InputCase

E, FT, AREA = 30.0e9, 3.0e6, 200.0e-6


def _case(tmp_path, monitors, *, divisions=(8, 2, 2), pull=2.0e-4, prestress=None):
    """A concrete bar in tension with one embedded rebar down its middle.

    Small on purpose: every assertion below is about WHERE a monitor looks,
    not about the mechanics, and the mechanics of a stretched prism are known
    in closed form.
    """
    case = {
        "name": "mon", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"dimensions": [0.8, 0.2, 0.2], "divisions": list(divisions)},
        "materials": {
            "c": {"type": "cdp", "E": E, "nu": 0.2, "rho": 2400.0,
                  "fb0_fc0": 1.16, "Kc": 0.6667, "dilation_angle": 36.0,
                  "eccentricity": 0.1, "wt": 0.0, "wc": 1.0, "viscosity": 1.0,
                  "n_return": 5, "crack_band": "tension", "l_ref": "auto",
                  "damage_max": 0.8,
                  "compression": {"source": "gb50010", "fc": 30.0e6,
                                  "beta_c": 0.7, "sigma_c0_ratio": 0.4},
                  "tension": {"source": "fracture_energy", "ft": FT,
                              "Gf": 120.0, "beta_t": 0.9}},
            "s": {"type": "rebar", "E": 200.0e9, "rho": 7850.0,
                  "yield_strength": 500.0e6, "hardening": 1.0e9}},
        "sections": {"body": {"type": "solid", "material": "c",
                              "formulation": "b-bar"},
                     "b": {"type": "bar", "material": "s", "shape": "circle",
                           "diameter": 0.016, "area": AREA}},
        "reinforcement": {"rod": {"layer": {
            "from": [0.0, 0.1, 0.1], "to": [0.8, 0.1, 0.1],
            "count": 1, "spacing": [0.0, 0.0, 0.0]}}},
        "assign": [{"section": "body", "region": "all"},
                   {"section": "b", "reinforcement": "rod", "name": "rod",
                    "segment_length": 0.1,
                    **({"prestress": prestress} if prestress else {})}],
        "regions": {
            "left": {"box": [-1e-3, 1e-3, -1, 1, -1, 1]},
            "right": {"box": [0.8 - 1e-3, 0.8 + 1e-3, -1, 1, -1, 1]},
            # a BOX that owns whole elements, and one that owns none
            "near_end": {"box": [-1e-3, 0.2 + 1e-3, -1, 1, -1, 1]},
            "far_end": {"box": [0.6 - 1e-3, 0.8 + 1e-3, -1, 1, -1, 1]},
            "everything": {"box": [-1.0, 2.0, -1.0, 2.0, -1.0, 2.0]},
        },
        "constraints": [{"region": "left", "dofs": [1, 1, 1]}],
        "steps": [{"name": "pull", "constraints": [
            {"region": "right", "dofs": [1, 0, 0], "value": pull}]}],
        "solver": {"backend": "numpy", "mode": "relaxation",
                   "n_steps": 60000, "tolerance": 1.0e-6},
        "output": {"path": str(tmp_path / "r.h5"), "every": 1000000,
                   "fields": ["displacement", "dt"], "monitors": monitors},
    }
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(case), encoding="utf-8")
    return path


def _run(tmp_path, monitors, **kw):
    return run_case(str(_case(tmp_path, monitors, **kw)))


# --------------------------------------------------------------------------- #
#  The bar
# --------------------------------------------------------------------------- #

def test_a_bar_monitor_reads_the_stress_a_contract_could_not_reach(tmp_path):
    """The whole point: 633 MPa in a tendon becomes a number in `monitors/`."""
    r = _run(tmp_path, {
        "rod_sigma": {"kind": "bar", "of": "rod", "field": "axial_stress",
                      "reduce": "mean"}})
    sigma = r.monitors["rod_sigma"][-1]
    # An exact invariant, not a range. The bar runs end to end and its ends
    # follow the host, so however the concrete cracks between them the bar's
    # MEAN strain is fixed by the two prescribed displacements: 2e-4 over
    # 0.8 m. Elastic and below yield, the mean stress is therefore
    # 2.5e-4 * 200 GPa. (The bar is far from uniform -- 10 to 233 MPa along
    # its length -- which is what makes the mean a real check and not a
    # restatement of one segment.)
    assert np.isclose(sigma, 2.5e-4 * 200.0e9, rtol=1e-6), sigma


def test_axial_force_is_the_stress_times_the_area(tmp_path):
    r = _run(tmp_path, {
        "s": {"kind": "bar", "of": "rod", "field": "axial_stress", "reduce": "mean"},
        "n": {"kind": "bar", "of": "rod", "field": "axial_force", "reduce": "mean"}})
    assert np.allclose(np.asarray(r.monitors["n"]),
                       np.asarray(r.monitors["s"]) * AREA, rtol=1e-12)


def test_prestress_shows_up_in_the_bar_monitor(tmp_path):
    """A pretensioned bar reads its prestress back -- the number cases/case_15
    is built on and had to proxy with the camber.

    Held between two fixed ends the prism cannot shorten, so there is no
    elastic-shortening loss and the answer is the applied 300 MPa exactly,
    with nothing to calibrate.
    """
    r = _run(tmp_path, {
        "s": {"kind": "bar", "of": "rod", "field": "axial_stress", "reduce": "mean"},
        "lo": {"kind": "bar", "of": "rod", "field": "axial_stress", "reduce": "min"},
        "hi": {"kind": "bar", "of": "rod", "field": "axial_stress", "reduce": "max"}},
        pull=0.0, prestress=3.0e8)
    for k in ("s", "lo", "hi"):
        assert np.isclose(r.monitors[k][-1], 3.0e8, rtol=1e-6), (k, r.monitors[k][-1])


def test_a_bar_region_selects_segments_by_geometry(tmp_path):
    """Bar nodes are the bar's own, so the node set a region resolves to says
    nothing about them; the box does.

    Two claims, and both are needed: a box that covers the model reproduces
    the unrestricted answer to the bit, and a box that covers a quarter of the
    span does NOT -- this bar runs from 10 to 233 MPa along its length, so a
    mask that quietly fell back to "everything" would be visible.
    """
    r = _run(tmp_path, {
        "all": {"kind": "bar", "of": "rod", "field": "axial_stress", "reduce": "mean"},
        "boxed": {"kind": "bar", "of": "rod", "field": "axial_stress",
                  "reduce": "mean", "region": "everything"},
        "near": {"kind": "bar", "of": "rod", "field": "axial_stress",
                 "reduce": "mean", "region": "near_end"}})
    assert r.monitors["boxed"][-1] == r.monitors["all"][-1]
    assert not np.isclose(r.monitors["near"][-1], r.monitors["all"][-1], rtol=0.2)


def test_a_bar_monitor_naming_nothing_fails_at_setup(tmp_path):
    with pytest.raises(Exception) as e:
        _run(tmp_path, {"x": {"kind": "bar", "of": "tendon",
                              "field": "axial_stress"}})
    assert "tendon" in str(e.value) and "rod" in str(e.value)


def test_a_bar_region_that_is_not_a_box_is_refused(tmp_path):
    """`all` is a predefined node set with no box. Approximating it by a
    bounding box would silently select the wrong segments on any region whose
    nodes are not a solid brick, so it raises instead."""
    p = _case(tmp_path, {"x": {"kind": "bar", "of": "rod",
                              "field": "axial_stress", "region": "all"}})
    with pytest.raises(Exception) as e:
        run_case(str(p))
    assert "box" in str(e.value)


# --------------------------------------------------------------------------- #
#  The elements
# --------------------------------------------------------------------------- #

def test_an_element_monitor_restricted_to_a_region_sees_less_than_the_model(tmp_path):
    """The case_15 shape: a global maximum decided by one corner of the model,
    and the question that was actually being asked is about a region."""
    r = _run(tmp_path, {
        "dt_all": {"kind": "element", "field": "dt", "reduce": "max"},
        "dt_far": {"kind": "element", "field": "dt", "reduce": "max",
                   "region": "far_end"}}, pull=6.0e-4)
    everywhere, far = r.monitors["dt_all"][-1], r.monitors["dt_far"][-1]
    assert everywhere > 0.0
    assert far <= everywhere + 1e-12
    # and the region is a strict subset: the restrained end cracks first, so
    # the far half cannot be where the model maximum sits
    assert far < everywhere


def test_a_stress_component_needs_its_component_and_reads_the_right_one(tmp_path):
    """A bar pulled along x: sigma_xx is tension, sigma_zz is not."""
    r = _run(tmp_path, {
        "sxx": {"kind": "element", "field": "stress", "component": "xx",
                "reduce": "mean"},
        "szz": {"kind": "element", "field": "stress", "component": "zz",
                "reduce": "mean"}})
    assert r.monitors["sxx"][-1] > 0.0
    assert abs(r.monitors["szz"][-1]) < 0.2 * r.monitors["sxx"][-1]


def test_stress_without_a_component_is_refused_by_the_schema():
    with pytest.raises(Exception) as e:
        schema.MonitorSpec(kind="element", field="stress")
    assert "component" in str(e.value)


def test_a_component_on_a_scalar_field_is_refused():
    with pytest.raises(Exception) as e:
        schema.MonitorSpec(kind="element", field="dt", component="xx")
    assert "scalar" in str(e.value)


def test_an_unknown_element_field_names_what_is_available(tmp_path):
    with pytest.raises(Exception) as e:
        _run(tmp_path, {"x": {"kind": "element", "field": "damage"}})
    msg = str(e.value)
    assert "damage" in msg and "dt" in msg and "dc" in msg


def test_a_region_that_owns_no_elements_fails_at_setup(tmp_path):
    """`left` is one plane of nodes. No element has ALL its nodes there, so
    the monitor would reduce over nothing -- and an empty reduction that
    reports 0.0 reads, in a contract, as "undamaged"."""
    with pytest.raises(Exception) as e:
        _run(tmp_path, {"x": {"kind": "element", "field": "dt",
                              "region": "left"}})
    assert "no elements" in str(e.value)


def test_the_reductions_agree_with_numpy(tmp_path):
    r = _run(tmp_path, {
        "mx": {"kind": "element", "field": "stress", "component": "xx", "reduce": "max"},
        "mn": {"kind": "element", "field": "stress", "component": "xx", "reduce": "min"},
        "av": {"kind": "element", "field": "stress", "component": "xx", "reduce": "mean"},
        "am": {"kind": "element", "field": "stress", "component": "xx", "reduce": "absmax"},
    })
    mx, mn, av, am = (r.monitors[k][-1] for k in ("mx", "mn", "av", "am"))
    assert mn <= av <= mx
    assert abs(am) == max(abs(mx), abs(mn))


# --------------------------------------------------------------------------- #
#  The schema keeps the four kinds apart
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("extra", [{"field": "dt"}, {"of": "rod"},
                                   {"component": "xx"}])
def test_a_nodal_monitor_rejects_element_keys(extra):
    with pytest.raises(Exception) as e:
        schema.MonitorSpec(kind="displacement", region="right", **extra)
    assert "kind: element" in str(e.value)


def test_an_element_monitor_rejects_a_union_of_regions():
    with pytest.raises(Exception) as e:
        schema.MonitorSpec(kind="element", field="dt", regions=["a", "b"])
    assert "not 'regions'" in str(e.value)


def test_an_element_or_bar_monitor_may_name_no_region_at_all():
    schema.MonitorSpec(kind="element", field="dt")
    schema.MonitorSpec(kind="bar", of="rod", field="axial_stress")


def test_a_bar_monitor_without_of_says_where_the_name_comes_from():
    with pytest.raises(Exception) as e:
        schema.MonitorSpec(kind="bar", field="axial_stress")
    assert "assignment" in str(e.value)
