"""1.2, item 6: soil springs on piles (p-y curves by depth, nonlinear elastic).

Checked against Hetenyi's beam on an elastic foundation: a long pile in soil
of modulus k (a linear p-y curve p = k y) with a lateral load H at its free
head deflects y0 = 2 H beta / k there, beta = (k / 4 EI)^(1/4). Then: the
soil's force is a reaction (the head load comes back as the sum of the
springs), the spring is radial (a push at 45 degrees meets the same soil),
the ultimate resistance caps it, the curves are interpolated by depth, and
what does not make sense is refused by name.
"""
import numpy as np
import pytest
import yaml
import h5py

from vfife.api import prepare_case, run_case
from vfife.cli import _with_element, make_solver
from vfife.io import schema

E, D = 3.0e10, 1.0
I = np.pi * D ** 4 / 64
A = np.pi * D ** 2 / 4
K_SOIL = 2.0e7                      # N/m per m of pile: a linear p-y curve p = k y


def _pile(tmp_path, name, H=(1.0e5, 0.0), L=30.0, n=60, curves=None, direction="horizontal",
          mode="static", extra=None):
    d = {"version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "lines", "points": {"head": [0, 0, 0], "tip": [0, 0, -L]},
                  "lines": {"pile": {"through": ["head", "tip"], "divisions": n}}},
         "materials": {"c": {"type": "linear_elastic", "E": E, "nu": 0.2, "rho": 2500.0}},
         "sections": {"p": {"type": "beam", "material": "c", "shape": "general", "area": A,
                            "iy": I, "iz": I, "j": 2 * I}},
         "assign": [{"region": "pile", "section": "p", "orientation": [1, 0, 0]}],
         "constraints": [{"region": "tip", "dofs": [0, 0, 1, 0, 0, 1]}],
         "loads": [{"region": "head", "vector": [H[0], H[1], 0.0]}],
         "soil_springs": [{"region": "pile", "ground": 0.0, "direction": direction,
                           "curves": curves or [{"depth": 0.0, "p_y": [[0, 0], [1.0, K_SOIL]]}]}],
         "solver": {"backend": "numpy", "mode": mode} if mode == "static" else
                   {"backend": "numpy", "mode": "relaxation", "n_steps": 200000, "tolerance": 1e-8},
         "output": {"path": str(tmp_path / f"{name}.h5"),
                    "monitors": {"ux": {"kind": "displacement", "region": "head", "dof": "x"},
                                 "uy": {"kind": "displacement", "region": "head", "dof": "y"},
                                 "Rx": {"kind": "reaction", "region": "pile", "dof": "x"}}}}
    d.update(extra or {})
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return p


def _final(path):
    out = run_case(str(path))
    assert out.converged, out.message
    return {k: float(v[-1]) for k, v in out.monitors.items()}


def test_a_long_pile_deflects_as_hetenyi_says(tmp_path):
    """The springs are lumped at the nodes, half a member each side: the
    error is second order in the element length (-1.9%, -0.48%, -0.12% on
    30, 60, 120 elements)."""
    H = 1.0e5
    beta = (K_SOIL / (4 * E * I)) ** 0.25
    assert beta * 30.0 > 6                                   # a long pile
    y0 = 2 * H * beta / K_SOIL
    m = _final(_pile(tmp_path, "het", H=(H, 0.0)))
    fine = _final(_pile(tmp_path, "het2", H=(H, 0.0), n=120))
    assert m["ux"] == pytest.approx(y0, rel=6e-3)
    assert abs(fine["ux"] / y0 - 1) == pytest.approx(abs(m["ux"] / y0 - 1) / 4, rel=0.05)
    # the soil is a support: its force on the pile is the reaction
    assert m["Rx"] == pytest.approx(-H, rel=1e-6)


def test_the_spring_is_radial(tmp_path):
    a = _final(_pile(tmp_path, "ax", H=(1.0e5, 0.0)))
    c, s = float(np.cos(np.pi / 4)), float(np.sin(np.pi / 4))
    b = _final(_pile(tmp_path, "diag", H=(1.0e5 * c, 1.0e5 * s),
                     curves=[{"depth": 0.0, "p_y": [[0, 0], [0.002, 4e4], [0.01, 8e4], [0.05, 9e4]]}]))
    a2 = _final(_pile(tmp_path, "ax2", H=(1.0e5, 0.0),
                      curves=[{"depth": 0.0, "p_y": [[0, 0], [0.002, 4e4], [0.01, 8e4], [0.05, 9e4]]}]))
    assert np.hypot(b["ux"], b["uy"]) == pytest.approx(abs(a2["ux"]), rel=1e-6)
    assert b["ux"] == pytest.approx(b["uy"], rel=1e-6)
    assert a["ux"] < a2["ux"]                     # the softer curve gives more


def test_the_ultimate_resistance_caps_it(tmp_path):
    """A pile whose soil gives at most p_ult per metre carries at most about
    p_ult L sideways; a head load past that has nothing to stand on (a static
    solve that cannot converge, or a large displacement)."""
    s = _solver(_pile(tmp_path, "cap", curves=[{"depth": 0, "p_y": [[0, 0], [0.01, 1e5], [1.0, 1e5]]}]))
    u = np.zeros((s.nn, 3)); u[:, 0] = 0.5                        # every node 0.5 m over: on the plateau
    f = s.soil.force(u)
    assert f[:, 0].sum() == pytest.approx(1e5 * 30.0, rel=1e-9)  # p_ult times the pile length
    u[:, 0] = 2.0                                                  # past the last point: still p_ult
    assert s.soil.force(u)[:, 0].sum() == pytest.approx(1e5 * 30.0, rel=1e-9)


def _solver(path):
    prep = prepare_case(str(path))
    return make_solver(_with_element(prep.case, prep.element, prep.case_path), prep.mesh, prep.material)


def test_the_curves_are_interpolated_by_depth_and_shared_by_length(tmp_path):
    s = _solver(_pile(tmp_path, "dep", L=10.0, n=10,
                      curves=[{"depth": 2.0, "p_y": [[0, 0], [1.0, 1e6]]},
                              {"depth": 6.0, "p_y": [[0, 0], [1.0, 5e6]]}]))
    u = np.zeros((s.nn, 3)); u[:, 0] = 0.01
    f = s.soil.force(u)[:, 0]
    z = s.mesh.nodes[:, 2]
    for n in range(s.nn):
        depth = -z[n]
        k = np.interp(depth, [2.0, 6.0], [1e6, 5e6])
        L = 0.5 if depth in (0.0, 10.0) else 1.0
        assert f[n] == pytest.approx(k * L * 0.01, rel=1e-9), depth


def test_the_springs_are_in_the_modes(tmp_path):
    """Above the ground nothing; below it the pile on its springs vibrates
    faster than the same pile without them."""
    from vfife import modal
    with_ = modal.analyse(_solver(_pile(tmp_path, "m1", mode="relaxation")), n_modes=1).frequencies[0]
    p = _pile(tmp_path, "m0", mode="relaxation")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    d.pop("soil_springs")
    d["constraints"] = [{"region": "tip", "dofs": [1, 1, 1, 0, 0, 1]}]
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    without = modal.analyse(_solver(p), n_modes=1).frequencies[0]
    assert with_ > 3 * without


def test_what_is_refused():
    with pytest.raises(Exception, match="starts at"):
        schema.PYCurveSpec(depth=0, p_y=[[0.001, 0], [0.01, 1e4]])
    with pytest.raises(Exception, match="increases"):
        schema.PYCurveSpec(depth=0, p_y=[[0, 0], [0.01, 1e4], [0.01, 2e4]])
    with pytest.raises(Exception, match="vertical axis"):
        schema.SoilSpringSpec(region="pile", direction="z",
                              curves=[{"depth": 0, "p_y": [[0, 0], [1, 1]]}])


def test_a_truss_pile_is_refused(tmp_path):
    p = _pile(tmp_path, "tr")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    d["sections"] = {"p": {"type": "bar", "material": "c", "shape": "general", "area": A}}
    d["assign"] = [{"region": "all", "section": "p"}]
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="soil_springs act on frame members"):
        _solver(p)


def test_a_fibre_pile_takes_them_too(tmp_path):
    """The same pile as a steel fibre section, solved on its own tangent."""
    p = _pile(tmp_path, "fib")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    d["materials"]["c"] = {"type": "steel02", "E": 2e11, "Fy": 4e8, "b": 0.01, "rho": 7850}
    d["sections"] = {"p": {"type": "fiber", "material": "c", "shape": "rect", "width": 0.6, "height": 0.6,
                           "ny": 8, "nz": 8}}
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    m = _final(p)
    EI = 2e11 * 0.6 ** 4 / 12
    beta = (K_SOIL / (4 * EI)) ** 0.25
    assert m["ux"] == pytest.approx(2 * 1e5 * beta / K_SOIL, rel=2e-3)
    assert m["Rx"] == pytest.approx(-1e5, rel=1e-9)


def test_in_a_dynamic_run_the_soil_reaction_is_the_soil_s_force(tmp_path):
    """Not the residual at the springs' nodes, which holds the inertia: a
    ramped head load read -99.7 kN of soil reaction where the soil carried
    -16 kN (review of 1.2)."""
    p = _pile(tmp_path, "dynr", n=30, mode="relaxation")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    d["solver"] = {"backend": "numpy", "mode": "dynamic", "total_time": 0.01}
    d["output"]["fields"] = ["displacement", "reaction_force"]
    d["output"]["every"] = 200
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    run_case(str(p))
    with h5py.File(tmp_path / "dynr.h5") as f:
        k = sorted(f["steps"], key=int)[-1]
        u = np.asarray(f["steps"][k]["displacement"], float)
        R = np.asarray(f["steps"][k]["reaction_force"], float)
    s = _solver(p)
    soil = s.soil.force(u)
    springs = s.soil.node_ids
    np.testing.assert_allclose(R[springs, 0], -soil[springs, 0], rtol=1e-5, atol=1e-6 * abs(soil).max())
    assert abs(R[:, 0].sum()) < 0.5 * 1e5                     # the soil has not taken the whole load yet


def test_a_ground_surface_inside_an_element_still_converges_as_the_square(tmp_path):
    """The buried part of the member crossing the ground is shared as its
    shape functions share it (+23% / +8.7% / +2.3% on 30 / 60 / 120 elements
    when the end above ground got nothing)."""
    curves = [{"depth": 0.0, "p_y": [[0, 0], [1.0, K_SOIL]]}]
    errs = []
    ref = None
    for n in (30, 60, 120):
        p = _pile(tmp_path, f"g{n}", n=n, curves=curves)
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        d["soil_springs"][0]["ground"] = -0.13
        p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
        errs.append(_final(p)["ux"])
    # Richardson: successive differences shrink four times
    assert (errs[0] - errs[1]) / (errs[1] - errs[2]) == pytest.approx(4.0, rel=0.15)


def test_a_curve_without_initial_stiffness_and_moving_supports_are_refused(tmp_path):
    with pytest.raises(Exception, match="rises from"):
        schema.PYCurveSpec(depth=0, p_y=[[0, 0], [0.01, 0], [0.05, 2e5]])
    p = _pile(tmp_path, "rec")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    (tmp_path / "g.dat").write_text("0 0\n1 0.01\n", encoding="utf-8")
    d["constraints"].append({"region": "tip", "dofs": [1, 0, 0, 0, 0, 0],
                             "record": {"path": "g.dat", "quantity": "displacement"}})
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    with pytest.raises(Exception, match="support `record`"):
        schema.load(str(p))


def test_preflight_takes_the_springs_as_holding_sideways_and_nothing_else(tmp_path):
    """A pile in soil is no free body; and a buried pile under a uniform
    ground motion is not 'fully restrained' (the springs were counted as
    constraints by every check, review of 1.2)."""
    p = _pile(tmp_path, "gm", mode="relaxation")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    (tmp_path / "acc.dat").write_text("0 0\n0.1 1\n0.2 0\n", encoding="utf-8")
    d["loads"] = [{"ground_motion": {"path": "acc.dat", "direction": [1, 0, 0]}}]
    d["solver"] = {"backend": "numpy", "mode": "dynamic", "total_time": 0.05}
    d["soil_springs"][0]["ground"] = 1.0                       # the head 1 m below the ground
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    codes = {i["code"] for i in prepare_case(str(p)).preflight["issues"]}
    assert "RIGID_BODY_MODE" not in codes and "GROUND_MOTION_FULLY_RESTRAINED" not in codes
