"""Added mass: directional, weightless and rotary nodal
masses, and the Morison added mass of members under water.

Checked against what the added mass should do and nothing else: a mass on x
alone lowers the x frequency by exactly sqrt(k / (m + dm)) and leaves y
alone; a weightless mass leaves the dead-load reactions alone; a rotary
inertia gives a torsional pendulum its closed-form frequency; the
hydrodynamic mass of a pile is C_a rho_w A times the wet length, across the
pile, lumped by the lever rule.
"""
import h5py
import numpy as np
import pytest
import yaml

from test_member_failure import _run, _write_msh
from vfife.api import prepare_case, _with_element
from vfife.core.structural import make_structural_solver
from vfife.core import added_mass
from vfife.core.dof import TRANSLATION

E, G_MOD = 2.0e11, 8.0e10
STEEL = {"type": "linear_elastic", "E": E, "nu": 0.25, "rho": 7850.0}
LIGHT = {"type": "linear_elastic", "E": E, "nu": 0.25, "rho": 1.0e-3}


def _column_case(tmp_path, name, masses=None, hydro=None, loads=None, material=LIGHT, n=4, H=4.0,
                 mode="relaxation", monitors=None, sec=None):
    nodes = [(0, 0, H * i / n) for i in range(n + 1)]
    _write_msh(tmp_path / f"{name}.msh", nodes,
               {"base": (0, [0]), "top": (0, [n]), "col": (1, [(i, i + 1) for i in range(n)])})
    sec = sec or {"type": "beam", "material": "m", "shape": "general", "area": 1e-2,
                  "iy": 2e-5, "iz": 1e-5, "j": 3e-5}
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": f"{name}.msh"},
        "materials": {"m": material},
        "sections": {"c": sec},
        "assign": [{"region": "col", "section": "c", "orientation": [1, 0, 0]}],
        "nodal_masses": masses or [],
        "constraints": [{"region": "base", "dofs": [1] * 6}],
        "loads": loads or [],
        "solver": {"backend": "numpy", "mode": mode, "n_steps": 20000, "tolerance": 1e-9}
        if mode == "relaxation" else {"backend": "numpy", "mode": "dynamic", "total_time": 0.01},
        "output": {"fields": ["displacement"], "every": 1000, "monitors": monitors or {}},
    }
    if hydro:
        d["hydrodynamic_mass"] = hydro
    p = tmp_path / f"{name}.yaml"
    d["output"]["path"] = str(tmp_path / f"{name}.h5")
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return p, d


def _solver(path):
    prep = prepare_case(str(path))
    return make_structural_solver(_with_element(prep.case, prep.element, prep.case_path),
                                  prep.mesh, prep.material)


def _modes(path, n=6):
    from vfife import modal
    s = _solver(path)
    res = modal.analyse(s, n_modes=n)
    return res, s


def _mode_along(res, axis):
    """The frequency of the lowest mode whose participation is along `axis`."""
    g = np.abs(res.gamma[:, axis])
    i = int(np.flatnonzero(g > 1e-6 * g.max())[0])
    return res.omega[i]


def test_a_mass_on_x_alone_lowers_x_and_leaves_y(tmp_path):
    """Small-displacement bending in x sees only the x masses: a 3000 kg
    mass on x alone gives the x mode of an isotropic 4000 kg lump and the y
    mode of the 1000 kg one. (A steel column: with near-massless members the
    eigenproblem's conditioning, 1e10 between the lump and the members'
    rotary inertia, costs the lowest mode a percent.)"""
    M, dM = 1000.0, 3000.0
    p1, _ = _column_case(tmp_path, "iso1", [{"region": "top", "mass": M}], material=STEEL)
    p4, _ = _column_case(tmp_path, "iso4", [{"region": "top", "mass": M + dM}], material=STEEL)
    pd, _ = _column_case(tmp_path, "dir", [{"region": "top", "mass": M},
                                           {"region": "top", "mass": dM, "components": [1, 0, 0]}],
                         material=STEEL)
    r1, _ = _modes(p1); r4, _ = _modes(p4); rd, sd = _modes(pd)
    assert _mode_along(rd, 0) == pytest.approx(_mode_along(r4, 0), rel=1e-8)
    assert _mode_along(rd, 1) == pytest.approx(_mode_along(r1, 1), rel=1e-8)
    assert _mode_along(rd, 0) < 0.6 * _mode_along(r1, 0)
    tip = sd.dofs.field(sd.physical_mass, TRANSLATION)[4]
    own = 7850.0 * 1e-2 * 1.0 / 2
    assert tip.tolist() == pytest.approx([M + dM + own, M + own, M + own], rel=1e-12)
    assert sd.weightless_mass[4].tolist() == pytest.approx([dM, 0.0, 0.0])


def test_added_mass_does_not_weigh(tmp_path):
    """The dead-load reaction of a column with a weighing lump and a
    weightless one is the weighing lump's weight, and the column's own."""
    g = 9.80665
    mons = {"rz": {"kind": "reaction", "region": "base", "dof": "z"}}
    loads = [{"region": "all", "gravity": [0, 0, -g]}]
    lumps = [{"region": "top", "mass": 1000.0},
             {"region": "top", "mass": 5000.0, "components": [1, 1, 0]},
             {"region": "col", "mass": 2000.0, "components": [1, 1, 1], "weight": False}]
    p, d = _column_case(tmp_path, "w", lumps, loads=loads, material=STEEL, monitors=mons)
    out, h5 = _run(tmp_path, "w", d)
    with h5py.File(h5) as f:
        rz = float(np.asarray(f["monitors/rz"])[-1])
    own = 7850.0 * 1e-2 * 4.0
    assert rz == pytest.approx((1000.0 + own) * g, rel=1e-6)


def test_a_rotary_inertia_makes_a_torsion_pendulum(tmp_path):
    """A 4 m shaft, G J / L, with a disc of inertia I_z on top:
    omega = sqrt(G J / (L I_z)) (the shaft's own inertia 1e-9 of it)."""
    Iz = 50.0
    p, _ = _column_case(tmp_path, "tor", [{"region": "top", "mass": 10.0, "inertia": [1e-3, 1e-3, Iz],
                                          "per_node": True}])
    res, s = _modes(p, n=12)
    J = 3e-5
    w = np.sqrt(G_MOD * J / (4.0 * Iz))
    assert np.min(np.abs(res.omega - w)) / w < 1e-4


def test_the_hydrodynamic_mass_of_a_pile_is_the_wet_length_across_it():
    # a vertical pile from z = -10 to +5, water at 0, D = 2
    nodes = np.array([[0, 0, -10.0], [0, 0, -2.5], [0, 0, 5.0]])
    members = np.array([[0, 1], [1, 2]])
    spec = {"region": "p", "diameter": 2.0, "coefficient": 1.0, "density": 1025.0, "water_level": 0.0}
    m, light = added_mass.hydrodynamic([spec], lambda r: np.arange(3), nodes, members)
    pl = 1025.0 * np.pi
    # member 1 fully wet (7.5 m): half each end; member 2 wet from -2.5 to 0
    # (2.5 m, s in [0, 1/3], centroid 1/6): 5/6 to node 1, 1/6 to node 2
    a = pl * 7.5 / 2 + 0.0
    assert m[:, 2].tolist() == [0.0, 0.0, 0.0]                  # nothing along the pile
    assert m[0, 0] == pytest.approx(a) and m[0, 1] == pytest.approx(a)
    assert m[1, 0] == pytest.approx(a + pl * 2.5 * 5 / 6)
    assert m[2, 0] == pytest.approx(pl * 2.5 / 6)
    assert m.sum(axis=0)[0] == pytest.approx(pl * 10.0)
    assert np.array_equal(light, m)                              # all of it weightless
    # an inclined member is refused, a dry region too
    tilted = np.array([[0, 0, -10.0], [3.0, 0, 0.0]])
    with pytest.raises(ValueError, match="off the nearest global axis"):
        added_mass.hydrodynamic([spec], lambda r: np.arange(2), tilted, np.array([[0, 1]]))
    with pytest.raises(ValueError, match="no member is below"):
        added_mass.hydrodynamic([dict(spec, water_level=-20.0)], lambda r: np.arange(3), nodes, members)


def test_a_pile_in_water_equals_its_directional_nodal_masses_and_weighs_the_same(tmp_path):
    D = 0.5
    hyd = [{"region": "col", "diameter": D, "water_level": 10.0}]
    p_w, d_w = _column_case(tmp_path, "wet", [{"region": "top", "mass": 500.0}], hydro=hyd, material=STEEL)
    pl = 1025.0 * np.pi * D ** 2 / 4
    per = [{"region": "base", "mass": pl * 0.5, "components": [1, 1, 0]},
           {"region": "top", "mass": pl * 0.5, "components": [1, 1, 0]}]
    # the three interior nodes get a full element each
    _write = _column_case(tmp_path, "eq", [{"region": "top", "mass": 500.0}] + per, material=STEEL)
    s_w = _solver(p_w)
    s_e = _solver(_write[0])
    mw = s_w.dofs.field(s_w.physical_mass, TRANSLATION)
    me = s_e.dofs.field(s_e.physical_mass, TRANSLATION)
    # interior nodes: one full element's worth
    me[1:4, :2] += pl * 1.0
    assert mw == pytest.approx(me, rel=1e-9, abs=1e-6)
    assert mw[:, 2] == pytest.approx(s_w.dofs.field(_solver(_column_case(
        tmp_path, "dry", [{"region": "top", "mass": 500.0}], material=STEEL)[0]).physical_mass,
        TRANSLATION)[:, 2])


def test_a_truss_nodal_mass_now_weighs_and_an_inertia_is_refused(tmp_path):
    g = 9.80665
    _write_msh(tmp_path / "t.msh", [(0, 0, 0), (0, 0, -1.0)],
               {"top": (0, [0]), "tip": (0, [1]), "bar": (1, [(0, 1)])})
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": "t.msh"},
         "materials": {"s": STEEL},
         "sections": {"b": {"type": "bar", "material": "s", "shape": "general", "area": 1e-4}},
         "assign": [{"region": "all", "section": "b"}],
         "nodal_masses": [{"region": "tip", "mass": 100.0},
                          {"region": "tip", "mass": 900.0, "components": [1, 1, 0]}],
         "constraints": [{"region": "top", "dofs": [1, 1, 1]}, {"region": "tip", "dofs": [1, 1, 0]}],
         "loads": [{"region": "all", "gravity": [0, 0, -g]}],
         "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 20000, "tolerance": 1e-9},
         "output": {"fields": ["displacement"], "every": 1000,
                    "monitors": {"rz": {"kind": "reaction", "region": "top", "dof": "z"}}}}
    _out, h5 = _run(tmp_path, "t", d)
    with h5py.File(h5) as f:
        rz = float(np.asarray(f["monitors/rz"])[-1])
    assert rz == pytest.approx((100.0 + 7850.0 * 1e-4) * g, rel=1e-5)
    d["nodal_masses"] = [{"region": "tip", "mass": 1.0, "inertia": [1, 1, 1]}]
    with pytest.raises(Exception, match="rotary inertia"):
        _run(tmp_path, "t2", d)


def test_the_schema_refuses_what_would_mean_nothing():
    from vfife.io.schema import NodalMassSpec, HydrodynamicMassSpec
    with pytest.raises(ValueError, match="cannot weigh"):
        NodalMassSpec(region="a", mass=1.0, components=[1, 0, 0], weight=True)
    with pytest.raises(ValueError, match="needs a `mass`"):
        NodalMassSpec(region="a")
    with pytest.raises(ValueError, match="0/1 flags"):
        NodalMassSpec(region="a", mass=1.0, components=[0, 0, 0])
    assert NodalMassSpec(region="a", inertia=[1, 2, 3]).mass == 0.0
    assert added_mass.weighs({"components": [1, 1, 0]}) is False
    assert added_mass.weighs({}) is True
    with pytest.raises(ValueError, match="diameter` OR an `area"):
        HydrodynamicMassSpec(region="p")
    with pytest.raises(ValueError, match="diameter` OR an `area"):
        HydrodynamicMassSpec(region="p", diameter=1.0, area=1.0)


def test_case_37_meets_its_contract(tmp_path):
    from pathlib import Path
    from vfife.api import run_case
    from vfife.verification import verify
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "case_37.h5"
    run_case(str(root / "cases" / "case_37_pier_in_water.yaml"), output=str(out))
    report = verify(root / "verification" / "case_37_pier_in_water.yaml", out)
    assert report["status"] == "PASS", [c for c in report["checks"] if c["status"] != "PASS"]
