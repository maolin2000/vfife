"""`precision: mixed` has a resolution, and it depends on the model's size.

THE RUN THIS EXISTS FOR
    The warp backend's `mixed` precision keeps element data and the element
    kernel in fp32 and the nodal displacement and force accumulation in fp64.
    A segment vector is a difference of two coordinates, so what survives the
    cancellation scales with (coordinate magnitude)/(segment length) -- the
    model's extent divided by its element size, which is not a number anyone
    thinks about when choosing a mesh.

    cases/case_15 is 8.5 m long with 50 mm elements, a ratio of 170. At
    `mixed` its transfer step spent 150000 iterations with 44581 kinetic
    resets -- one every 3.4 steps -- and ended at a relative residual of
    2.2e-1, where the same model converges to 1e-5 in 34000 iterations on the
    CPU. The tendon carried 1.5e-5 of strain error, an eighth of the cracking
    strain, which at 195 GPa is 2.9 MPa of stress noise. Relaxation cannot
    remove that, because it is not out of balance: it is the arithmetic.

    Every ingredient of that is known before the run starts, and it cost an
    afternoon and two hours of card time. So preflight computes it.
"""
import numpy as np
import pytest
import yaml

from vfife.cli import build
from vfife.io import schema
from vfife.io.model import InputCase
from vfife.preflight import _fp32_strain_resolution, check

EC, FT = 36.0e9, 3.7e6


def _case(length, divisions, backend="warp", precision="mixed", bars=True):
    """A concrete bar of `length` with a tendon down it. Only the LENGTH and
    the element size change between the variants below."""
    seg = length / divisions[0]
    case = {
        "name": "res", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"dimensions": [length, 0.3, 0.3], "divisions": list(divisions)},
        "materials": {
            "c": {"type": "cdp", "E": EC, "nu": 0.2, "rho": 2400.0,
                  "fb0_fc0": 1.16, "Kc": 0.6667, "dilation_angle": 36.0,
                  "eccentricity": 0.1, "wt": 0.0, "wc": 1.0, "viscosity": 1.0,
                  "n_return": 5, "crack_band": "tension", "l_ref": "auto",
                  "damage_max": 0.8,
                  "compression": {"source": "gb50010", "fc": 50.0e6,
                                  "beta_c": 0.7, "sigma_c0_ratio": 0.4},
                  "tension": {"source": "fracture_energy", "ft": FT,
                              "Gf": 150.0, "beta_t": 0.9}},
            "s": {"type": "rebar", "E": 195.0e9, "rho": 7850.0,
                  "yield_strength": 1640.0e6, "hardening": 8.3e9}},
        "sections": {"body": {"type": "solid", "material": "c",
                              "formulation": "b-bar"},
                     "t": {"type": "bar", "material": "s", "shape": "circle",
                           "diameter": 0.016, "area": 150.0e-6}},
        "assign": [{"section": "body", "region": "all"}],
        "regions": {"end": {"box": [-1e-3, 1e-3, -1, 1, -1, 1]}},
        "constraints": [{"region": "end", "dofs": [1, 1, 1]}],
        "solver": {"backend": backend, "precision": precision,
                   "mode": "relaxation", "n_steps": 10},
        "output": {"path": "unused.h5", "monitors": {}},
    }
    if bars:
        # a route, not a straight line: its nodes land on arbitrary coordinates,
        # which is what makes a tendon the worst-resolved thing in a model
        case["reinforcement"] = {"t": {"path": {
            "shape": "parabola", "segment_length": seg,
            "points": [[0.0, 0.15, 0.25], [length / 2, 0.15, 0.06],
                       [length, 0.15, 0.25]]}}}
        case["assign"].append({"section": "t", "reinforcement": "t",
                               "name": "tendon"})
    lowered = InputCase(**case).lower()
    c = schema.Case(**lowered)
    mesh, _material, element = build(c, "unused.yaml")
    return check(c, mesh, element)


def _codes(report):
    return {i["code"]: i for i in report["issues"]}


# --------------------------------------------------------------------------- #
#  The number itself
# --------------------------------------------------------------------------- #

def test_the_resolution_is_the_cancellation_and_scales_with_the_model(tmp_path):
    """Two identical meshes, one ten times longer. The strain error is not a
    property of the element; it is a property of where the element SITS."""
    conn = np.array([[0, 1]], dtype=np.int64)
    near = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    far = near + np.array([8.4, 0.0, 0.0])
    _r_near, m_near = _fp32_strain_resolution(near + 1e-3, conn)
    _r_far, m_far = _fp32_strain_resolution(far + 1e-3, conn)
    assert m_far > 20 * max(m_near, 1e-12), (
        f"the same 50 mm element gives {m_near:.2e} at the origin and "
        f"{m_far:.2e} at x = 8.4 m; if those are the same the check is not "
        "measuring the cancellation")


# --------------------------------------------------------------------------- #
#  When it fires and when it does not
# --------------------------------------------------------------------------- #

def test_a_long_model_with_small_elements_is_warned():
    issues = _codes(_case(8.5, [170, 2, 2]))
    w = issues["MIXED_PRECISION_RESOLUTION"]
    assert w["severity"] == "WARNING"
    assert w["details"]["ratio"] > 0.05
    assert "fp64" in w["message"] and "tendon" in w["message"]


def test_a_short_model_is_not_warned():
    """The check has to stay quiet where mixed precision is fine, or it is
    noise and gets ignored on the one case that needed it."""
    issues = _codes(_case(0.6, [12, 2, 2]))
    assert issues["MIXED_PRECISION_RESOLUTION"]["severity"] == "INFO"


def test_fp64_says_nothing():
    assert "MIXED_PRECISION_RESOLUTION" not in _codes(
        _case(8.5, [170, 2, 2], precision="fp64"))


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_a_cpu_backend_says_nothing(backend):
    """`precision` exists on every case; it only means anything on warp."""
    assert "MIXED_PRECISION_RESOLUTION" not in _codes(
        _case(8.5, [170, 2, 2], backend=backend))


def test_the_reinforcement_is_measured_and_not_only_the_mesh():
    """On cases/case_15 the mesh itself was the good part -- its extruded,
    axis-aligned edges cancel exactly in two of three components, 4.9e-08 rms
    -- and the TENDON carried 3.8e-06, because a route's nodes land on
    arbitrary coordinates. A check that looked only at the mesh would have
    said nothing about the run that failed, so both are measured and both are
    named."""
    with_bars = _codes(_case(8.5, [170, 2, 2]))["MIXED_PRECISION_RESOLUTION"]
    without = _codes(_case(8.5, [170, 2, 2], bars=False))["MIXED_PRECISION_RESOLUTION"]
    assert "tendon" in with_bars["message"] and "mesh" in with_bars["message"]
    assert "tendon" not in without["message"]
    # and the bars only ever make it worse, never better
    assert with_bars["details"]["worst"] >= without["details"]["worst"]
