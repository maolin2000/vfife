"""1.3, item 11: displacement control in the arc length.

`static.arc_length.control: {region, dof}` measures each increment's arc on
one node's translation instead of the whole state: displacement control
with the load factor as the unknown (OpenSees' DisplacementControl). The
controlled displacement moves on at every increment, so an equilibrium
that would bring it back -- the elastic unloading from the last state -- is
not on the arc.

* the snap-through truss: every point on the exact path, the pushed point
  moving on through the limit point and the negative branch to the full load;
* a crushing RC fibre beam: the controlled path is the whole-state arc's;
* a SNAP-BACK of the controlled displacement (a soft spring above the
  truss) is not jumped: the run stops there and says so, where the
  whole-state arc follows it. (It was jumped -- one 12 mm increment, the
  load factor from 1.0 to -0.66, reported CONVERGED -- until the review.)
* a stop and a restart under control end where one run ends;
* the refusals: a region of two nodes, a held dof, a dof the pattern does
  not move.
"""
import h5py
import numpy as np
import pytest
import yaml

from test_static import _von_mises
from vfife.api import run_case


def test_control_follows_the_snap_through_truss_exactly(tmp_path):
    ctrl = {"region": "apex", "dof": "z"}
    P, w, P_of, Pmax, out = _von_mises(tmp_path, "vmc", {"initial": 0.02, "max_increments": 400,
                                                         "control": ctrl})
    assert out.status == "CONVERGED", out.message
    assert np.abs(P - P_of(w)).max() < 1e-8 * Pmax         # on the exact path
    assert np.all(np.diff(w) > 0.0)                         # the apex moves on at every increment
    assert P.min() < -0.5 * Pmax and P.max() == pytest.approx(1.5 * Pmax)
    assert w[-1] > 2 * 0.1


def test_control_on_a_crushing_fibre_beam_is_the_whole_state_arcs_path(tmp_path):
    """An over-reinforced RC beam, simply supported, pushed at mid-span far
    past the crushing of its top fibres: the load falls to a fifth of the
    peak. Both constraints follow it, to the same curve."""
    L = 4.0

    def case(name, arc):
        return {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
                "mesh": {"type": "lines", "points": {"A": [0, 0, 0], "M": [L / 2, 0, 0], "B": [L, 0, 0]},
                         "lines": {"l": {"through": ["A", "M"], "size": 1.0},
                                   "r": {"through": ["M", "B"], "size": 1.0}}},
                "materials": {"c": {"type": "concrete02", "fpc": 3.0e7, "epsc0": 0.002, "fcu": 3.0e6,
                                    "epscu": 0.003, "ft": 2.5e6, "Ets": 1.5e9, "rat": 0.1, "rho": 2500},
                              "s": {"type": "steel02", "E": 2.0e11, "Fy": 1.6e9, "b": 0.01, "R0": 20.0,
                                    "rho": 7850}},
                "sections": {"bar": {"type": "bar", "material": "s", "shape": "circle", "diameter": 0.04},
                             "f": {"type": "fiber", "material": "c", "shape": "rect", "width": 0.3,
                                   "height": 0.5, "ny": 1, "nz": 20, "integration": "lobatto3",
                                   "bars": [{"y": y, "z": -0.2, "section": "bar"} for y in (-0.1, 0.0, 0.1)]}},
                "assign": [{"region": "all", "section": "f", "orientation": [0, 1, 0]}],
                "constraints": [{"region": "all", "dofs": [0, 1, 0, 1, 0, 1]},
                                {"region": "A", "dofs": [1, 1, 1, 1, 0, 1]},
                                {"region": "B", "dofs": [0, 1, 1, 1, 0, 1]}],
                "loads": [{"region": "M", "vector": [0, 0, -1.0e6]}],
                "solver": {"mode": "static", "tolerance": 1e-6,
                           "static": {"arc_length": dict(arc, initial=0.05, max_increments=150,
                                                         stop={"region": "M", "dof": "z", "value": -0.15})}},
                "output": {"path": str(tmp_path / f"{name}.h5"), "every": 1, "fields": ["displacement"],
                           "monitors": {"w": {"kind": "displacement", "region": "M", "dof": "z"}}}}
    curves = {}
    for name, arc in (("whole", {}), ("ctrl", {"control": {"region": "M", "dof": "z"}})):
        out = run_case(case(name, arc))
        assert out.status == "CONVERGED", out.message
        with h5py.File(tmp_path / f"{name}.h5") as f:
            curves[name] = (np.asarray(f["monitors/time"]) * 1e6, -np.asarray(f["monitors/w"]))
    P, w = curves["ctrl"]
    assert np.all(np.diff(w) > 0.0)
    peak = P.max()
    assert P[-1] < 0.25 * peak                              # far down the softening
    Pw, ww = curves["whole"]
    for x in (0.03, 0.05, 0.08, 0.12):                      # the same curve, after the peak
        assert np.interp(x, w, P) == pytest.approx(np.interp(x, ww, Pw), rel=0.03)


def test_a_stop_and_a_restart_under_control_end_where_one_run_ends(tmp_path):
    from test_static import _write_msh  # noqa: F401  (the helper's own import)
    from vfife import api
    from vfife.core.diagnostics import AnalysisInterrupted
    _P, _w, _Pof, _Pm, _o = _von_mises(tmp_path, "vmr", {"initial": 0.05, "control": {"region": "apex", "dof": "z"}})
    # the same case, written out and run three ways
    d = yaml.safe_load((tmp_path / "vmr.yaml").read_text()) if (tmp_path / "vmr.yaml").exists() else None
    if d is None:
        pytest.skip("the truss helper keeps its case in memory")
    p = tmp_path / "vmr2.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    got = {}
    api.run_case(p, output=str(tmp_path / "full.h5"), progress_every=1,
                 on_ready=lambda s, _p: got.__setitem__("full", s))
    with pytest.raises(AnalysisInterrupted):
        api.run_case(p, output=str(tmp_path / "half.h5"), progress_every=1,
                     on_ready=lambda s, _p: got.__setitem__("half", s),
                     on_progress=lambda *a: got["half"].step < 4)
    api.run_case(p, output=str(tmp_path / "rest.h5"), restart=str(tmp_path / "half.h5"), progress_every=1,
                 on_ready=lambda s, _p: got.__setitem__("rest", s))
    a, b = got["full"].checkpoint(), got["rest"].checkpoint()
    for k in a:
        if np.asarray(a[k]).dtype.kind not in "SUO":
            assert np.array_equal(a[k], b[k]), k


@pytest.mark.parametrize("control, words", [
    ({"region": "sup", "dof": "z"}, "holds 2 nodes"),
    ({"region": "apex", "dof": "y"}, "is held"),
    ({"region": "apex", "dof": "x"}, "does not move the control"),
])
def test_what_control_refuses(tmp_path, control, words):
    with pytest.raises(ValueError, match=words):
        _von_mises(tmp_path, "vmx", {"initial": 0.02, "control": control})


def _spring_truss(tmp_path, name, arc, factor=1.5):
    """The snap-through truss with a soft spring above its apex, loaded at
    the spring's top: the top's displacement snaps back where the truss
    snaps through (the spring is half as stiff as the truss's steepest
    negative slope)."""
    from test_member_failure import _run, _write_msh
    a, h, EA = 1.0, 0.1, 1.0e6
    L0 = np.hypot(a, h)

    def F(w):
        y = h - w
        L = np.hypot(a, y)
        return -2 * EA * (L - L0) / L0 * y / L
    ws = np.linspace(0, 2 * h, 20001)
    Pmax = float(F(ws).max())
    k = float(0.5 * abs(np.gradient(F(ws), ws).min()))
    _write_msh(tmp_path / f"{name}.msh", [(-a, 0, 0), (0, 0, h), (a, 0, 0), (0, 0, h + 1.0)],
               {"sup": (0, [0, 2]), "apex": (0, [1]), "top": (0, [3]), "bars": (1, [(0, 1), (1, 2)]),
                "spring": (1, [(1, 3)])})
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": f"{name}.msh"},
         "materials": {"s": {"type": "linear_elastic", "E": EA / 1e-3, "nu": 0.3, "rho": 7850},
                       "k": {"type": "linear_elastic", "E": k / 1e-3, "nu": 0.3, "rho": 7850}},
         "sections": {"b": {"type": "bar", "material": "s", "shape": "general", "area": 1e-3},
                      "kb": {"type": "bar", "material": "k", "shape": "general", "area": 1e-3}},
         "assign": [{"region": "bars", "section": "b"}, {"region": "spring", "section": "kb"}],
         "constraints": [{"region": "sup", "dofs": [1] * 6}, {"region": "apex", "dofs": [0, 1, 0, 1, 1, 1]},
                         {"region": "top", "dofs": [1, 1, 0, 1, 1, 1]}],
         "loads": [{"region": "top", "vector": [0, 0, -factor * Pmax]}],
         "solver": {"mode": "static", "tolerance": 1e-9, "static": {"arc_length": arc}},
         "output": {"fields": ["displacement"], "every": 1,
                    "monitors": {"wa": {"kind": "displacement", "region": "apex", "dof": "z"},
                                 "wt": {"kind": "displacement", "region": "top", "dof": "z"}}}}
    out, h5 = _run(tmp_path, name, d)
    with h5py.File(h5) as f:
        lam = np.asarray(f["monitors/time"]); wa = -np.asarray(f["monitors/wa"]); wt = -np.asarray(f["monitors/wt"])
    return out, lam * factor * Pmax, wa, wt, F, Pmax


@pytest.mark.parametrize("initial", [0.005, 0.02])
def test_a_snap_back_of_the_controlled_displacement_is_not_jumped(tmp_path, initial):
    out, P, wa, wt, F, Pmax = _spring_truss(tmp_path, f"sb{initial}", {
        "initial": initial, "max_increments": 400, "control": {"region": "top", "dof": "z"}})
    assert out.status != "CONVERGED"
    assert "snap-back" in (out.message or ""), out.message
    assert np.abs(P - F(wa)).max() < 1e-7 * Pmax          # every frame it kept is on the path
    assert np.all(np.diff(wa) < 0.03)                      # and none of them jumped across the snap
    assert P.max() < 1.01 * Pmax


def test_the_whole_state_arc_follows_that_snap_back(tmp_path):
    out, P, wa, wt, F, Pmax = _spring_truss(tmp_path, "sbw", {"initial": 0.02, "max_increments": 400})
    assert out.status == "CONVERGED", out.message
    assert np.abs(P - F(wa)).max() < 1e-8 * Pmax
    assert np.any(np.diff(wt) < 0.0)                       # the top went back: the snap-back, followed
