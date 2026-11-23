"""v1.0's fixes, reviewed again (1.0 re-review).

One test per finding of the independent review of the signed member forces,
the unit-carrying defaults and the per-fibre mass:

  M1  `bond_cyclic` defaults converted like `bond_ceb`'s
  M2  a `units` declaration with `stress` but not `length` or `force` is
      completed from the other two, or refused (UNITS_INCOMPLETE)
  minors: PCHIP slopes for displacement records, a fibre member's
  `*_max_abs` by the old strictly-larger rule, the mass unit `info` prints,
  plot units of member-force monitors.
"""
import copy
import pathlib

import h5py
import numpy as np
import pytest
import yaml

from vfife import api
from vfife.core import units as U

STEEL02 = {"type": "steel02", "E": 2.0e11, "Fy": 3.45e9, "b": 0.01, "R0": 20.0, "rho": 7850.0}


def _line_mesh(path, length, n=4, direction=(1.0, 0.0, 0.0)):
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    pts = [d * length * k / n for k in range(n + 1)]
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "3",
             '0 1 "root"', '0 2 "tip"', '1 3 "beam"', "$EndPhysicalNames", "$Nodes", str(n + 1)]
    lines += [f"{k + 1} {float(p[0])!r} {float(p[1])!r} {float(p[2])!r}" for k, p in enumerate(pts)]
    lines += ["$EndNodes", "$Elements", str(n + 2), "1 15 2 1 1 1", f"2 15 2 2 2 {n + 1}"]
    lines += [f"{e + 3} 1 2 3 3 {e + 1} {e + 2}" for e in range(n)]
    lines += ["$EndElements"]
    pathlib.Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _run(tmp_path, d, name):
    d = copy.deepcopy(d)
    d["name"] = name
    d.setdefault("output", {})["path"] = str(tmp_path / f"{name}.h5")
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return api.run_case(str(p)), tmp_path / f"{name}.h5"


# ---------------------------------------------------------------- M1
@pytest.mark.parametrize("where", ["material", "assign"])
def test_bond_cyclic_defaults_are_converted_like_bond_ceb(where):
    """`bond_cyclic` inherits bond_ceb's MC2010 envelope and its SI defaults;
    only bond_ceb was converted, so in a mm / N / MPa case switching law made
    the bond a thousand times stronger and its slips a thousand times
    shorter (tau_max 13 693 against 13.69), with no message."""
    from vfife.materials.bond import BondCEB
    from vfife.materials.bond_cyclic import BondCyclic
    got = {}
    for law in ("bond_ceb", "bond_cyclic"):
        d = {"units": {"length": "mm", "force": "N", "stress": "MPa"}, "materials": {}}
        if where == "material":
            d["materials"]["b"] = {"type": law, "fc": 30.0}
        else:
            d["assign"] = [{"region": "r", "section": "s", "bond": {"type": law, "fc": 30.0}}]
        out, filled = U.fill_defaults(d)
        props = out["materials"]["b"] if where == "material" else out["assign"][0]["bond"]
        assert props["type"] == law
        assert {w.rsplit(".", 1)[-1] for w, _v, _d in filled} >= {"rho", "tau_max", "s1", "s2", "s3"}
        cls = BondCEB if law == "bond_ceb" else BondCyclic
        m = cls(**{k: v for k, v in props.items() if k != "type"})
        got[law] = (m.tau_max, m.s1, m.s2, m.s3, m.rho)
    assert got["bond_cyclic"] == pytest.approx(got["bond_ceb"], rel=1e-12)
    assert got["bond_ceb"][0] == pytest.approx(2.5 * np.sqrt(30.0), rel=1e-12)   # MPa
    assert got["bond_ceb"][1] == pytest.approx(1.0)                               # mm


# ---------------------------------------------------------------- M2
def test_a_declaration_with_stress_is_completed_from_the_other_two():
    assert U.complete({"force": "N", "stress": "MPa"})[:2] == (
        {"force": "N", "stress": "MPa", "length": "mm"}, ["length"])
    assert U.complete({"length": "m", "stress": "kPa"})[0]["force"] == "kN"
    assert U.complete({"length": "mm", "stress": "MPa"})[0]["force"] == "N"
    assert U.complete({"stress": "Pa"})[2] == []                 # m and N
    assert U.complete({"stress": "MPa"})[2] == ["length", "force"]
    assert U.complete({"force": "kN", "stress": "MPa"})[2] == ["length"]   # a 31.6 mm length
    assert U.complete({"length": "mm"}) == ({"length": "mm"}, [], [])      # N by default, as before
    sc = U.scales({"force": "N", "stress": "MPa"})
    assert sc["length"] == pytest.approx(1e-3) and sc["density"] == pytest.approx(1e12)   # t/mm^3 in kg/m^3
    assert U.fill_defaults({"units": {"force": "N", "stress": "MPa"},
                            "materials": {"s": {"type": "steel02"}}})[0]["materials"]["s"]["E"] == \
        pytest.approx(2.0e5)


def _cantilever(tmp_path, units, mm):
    s = 1e3 if mm else 1.0
    return {"units": units,
            "mesh": {"type": "gmsh", "path": _line_mesh(tmp_path / f"c{int(mm)}.msh", 4.0 * s)},
            "materials": {"s02": {"type": "steel02"}},          # E, Fy and rho: defaults
            "sections": {"s": {"type": "fiber", "material": "s02", "shape": "rect", "width": 0.3 * s,
                               "height": 0.5 * s, "ny": 4, "nz": 8, "j": 0.01 * s ** 4}},
            "assign": [{"region": "beam", "section": "s", "orientation": [0, 1, 0]}],
            "constraints": [{"region": "root", "dofs": [1] * 6}],
            "loads": [{"region": "tip", "vector": [0, 0, -1.0e4]},
                      {"region": "beam", "gravity": [0, 0, -9.80665 * s]}],
            "solver": {"mode": "static", "backend": "numpy", "tolerance": 1e-6},
            "output": {"fields": ["displacement"],
                       "monitors": {"R": {"kind": "reaction", "region": "root", "dof": "z"},
                                    "w": {"kind": "displacement", "region": "tip", "dof": "z"}}}}


def test_force_and_stress_without_length_run_as_millimetres(tmp_path):
    """`{force: N, stress: MPa}` on a mm model was read as m: nothing
    converted, steel02's E = 2e11 and rho = 7850 went into a mm / MPa model,
    the cantilever deflected 311 137 instead of 0.96 and weighed 4.6e16 N,
    preflight PASS with no issue."""
    si, _ = _run(tmp_path, _cantilever(tmp_path, {"length": "m", "force": "N", "stress": "Pa"}, False), "si")
    mm, _ = _run(tmp_path, _cantilever(tmp_path, {"force": "N", "stress": "MPa"}, True), "mm")
    assert si.converged and mm.converged, (si.message, mm.message)
    assert mm.monitors["R"][-1] == pytest.approx(si.monitors["R"][-1], rel=1e-6)
    assert mm.monitors["w"][-1] == pytest.approx(1e3 * si.monitors["w"][-1], rel=1e-5)
    codes = {i["code"] for i in mm.preflight["issues"]}
    assert "UNITS_INFERRED" in codes and "UNIT_DEFAULT_CONVERTED" in codes


@pytest.mark.parametrize("units, missing", [({"stress": "MPa"}, {"length", "force"}),
                                            ({"force": "kN", "stress": "MPa"}, {"length"})])
def test_a_stress_that_fixes_no_system_is_refused(tmp_path, units, missing):
    prep = api.prepare_case(_write_case(tmp_path, _cantilever(tmp_path, units, True)))
    assert prep.preflight["status"] == "FAIL"
    bad = [i for i in prep.preflight["issues"] if i["code"] == "UNITS_INCOMPLETE"]
    assert {i["details"]["key"] for i in bad} == missing
    assert all(i["severity"] == "ERROR" for i in bad)


def _write_case(tmp_path, d, name="case"):
    d = copy.deepcopy(d)
    d["name"] = name
    d.setdefault("output", {})["path"] = str(tmp_path / f"{name}.h5")
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return str(p)


def test_the_magnitude_checks_use_the_completed_system(tmp_path):
    """GRAVITY_MAGNITUDE needed units.length: SI gravity in a case declared
    `{force: N, stress: MPa}` (a mm model) was not questioned."""
    d = _cantilever(tmp_path, {"force": "N", "stress": "MPa"}, True)
    d["loads"][1]["gravity"] = [0, 0, -9.80665]
    prep = api.prepare_case(_write_case(tmp_path, d))
    assert "GRAVITY_MAGNITUDE" in {i["code"] for i in prep.preflight["issues"]}


# ---------------------------------------------------------------- minor 1
def test_a_displacement_record_does_not_overshoot_a_coarse_step(tmp_path):
    """The three-point knot slopes put the support 7.4% past a coarse step on
    each side (and stepping back before it stepped forward)."""
    from vfife.core.ground import RecordMotion, pchip_slopes
    rec = tmp_path / "step.txt"
    rec.write_text("0 0\n1 0\n2 1\n3 1\n4 1\n", encoding="utf-8")
    a = RecordMotion({"path": str(rec), "quantity": "displacement"}, 4.0, 0.0, {})
    p = np.linspace(0.0, 1.0, 4001)
    u = np.array([a(x) for x in p])
    v = np.array([a.rate(x) for x in p])
    assert u.min() >= 0.0 and u.max() <= 1.0
    assert np.all(np.diff(u) >= -1e-15)                         # monotone
    num = np.gradient(u, 4.0 * p)
    assert np.abs(num - v)[5:-5].max() < 2e-3                   # rate is the derivative
    for x in (0.1, 0.37, 0.5, 0.81):                            # through the samples' range only
        assert a(x) == pytest.approx(np.interp(4 * x, [0, 1, 2, 3, 4], [0, 0, 1, 1, 1]), abs=0.5)
    # scipy's PchipInterpolator, written out: weighted harmonic means, zero at
    # an extremum, the shape-preserving end slope
    t = np.array([0.0, 0.3, 1.1, 2.0, 2.2, 3.5]); y = np.array([0.0, 1.0, 0.5, 0.7, 2.0, 1.9])
    m = pchip_slopes(t, y)
    assert m[1] == 0.0 and m[4] == 0.0
    h = np.diff(t); s = np.diff(y) / h
    w1, w2 = 2 * h[3] + h[2], h[3] + 2 * h[2]
    assert m[3] == pytest.approx((w1 + w2) / (w1 / s[2] + w2 / s[3]))
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:
        return
    np.testing.assert_allclose(m, PchipInterpolator(t, y).derivative()(t), rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------- minor 2
def test_a_fibre_members_max_abs_is_the_station_maximum_of_the_magnitude(tmp_path):
    """`*_max_abs` was |signed|, and the signed field picks a later station
    only beyond 1e-9: in a near-tie the envelope was not the old field bit
    for bit. It is the stations' max |value| by the old rule, on both paths."""
    from vfife.cli import _with_element, build, make_solver
    from vfife.io import schema
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": _line_mesh(tmp_path / "k.msh", 4.0, direction=(1, 2, -0.7))},
         "materials": {"s02": STEEL02},
         "sections": {"s": {"type": "fiber", "material": "s02", "shape": "rect", "width": 1.0, "height": 1.0,
                            "ny": 4, "nz": 4, "shear_stiffness": 1.0e10, "j": 0.1}},
         "assign": [{"region": "beam", "section": "s", "orientation": [0, 0, 1]}],
         "constraints": [{"region": "root", "dofs": [1] * 6}],
         "loads": [{"region": "tip", "vector": [1e4, 2e4, -3e4]}],
         "solver": {"mode": "static", "backend": "numpy", "tolerance": 1e-6},
         "output": {"fields": ["displacement"]}}
    p = _write_case(tmp_path, d, "k")
    case, _ = schema.load(p)
    mesh, material, element = build(case, p)
    s = make_solver(_with_element(case, element, p), mesh, material)
    rng = np.random.default_rng(5)
    s.q[:] = 1e-4 * rng.standard_normal(s.q.shape)
    s.q[s.dofs.node_dofs(np.array([0]), ("ux", "uy", "uz", "rx", "ry", "rz")).ravel()] = 0.0
    s._kernel = None
    signed = np.array(s.internal_force()[1], copy=True)
    ref = np.array(s.actions_abs, copy=True)
    # the stations' resultants from the fibre stresses
    stress = s.fiber_stress_rows()
    fa, fy, fz = s.fiber_area, s.fiber_y, s.fiber_z
    N = np.einsum("egf,ef->eg", stress, fa)
    My = np.einsum("egf,ef->eg", stress, fa * fz)
    Mz = -np.einsum("egf,ef->eg", stress, fa * fy)
    for col, R in ((0, N), (4, My), (5, Mz)):
        np.testing.assert_allclose(ref[:, col], np.abs(R).max(axis=1), rtol=1e-10)
    np.testing.assert_allclose(ref, np.abs(signed), rtol=1e-8)
    fields = s.nodal_fields(signed)
    np.testing.assert_array_equal(fields["bending_y_max_abs"], ref[:, 4])
    np.testing.assert_array_equal(fields["shear_z_max_abs"], np.abs(signed[:, 2]))
    # actions handed in from elsewhere (not this evaluation's) report their own magnitudes
    other = 2.0 * signed
    np.testing.assert_array_equal(s.nodal_fields(other)["axial_force_max_abs"], np.abs(other[:, 0]))


# ---------------------------------------------------------------- minors 4 and 6
def test_the_consistent_mass_unit_has_its_name():
    assert U.mass_unit({"length": "m", "force": "kN"}) == "t"
    assert U.mass_unit({"length": "mm", "force": "N", "stress": "MPa"}) == "t"
    assert U.mass_unit({"length": "m", "force": "N"}) == "kg"
    assert U.mass_unit({"force": "N", "stress": "MPa"}) == "t"          # mm, inferred
    assert U.mass_unit({}) == "kg"


def test_member_force_monitors_carry_force_and_moment_units(tmp_path):
    from vfife.io import schema
    d = _cantilever(tmp_path, {"length": "m", "force": "kN", "stress": "kPa"}, False)
    d["output"]["monitors"] = {
        n: {"kind": "element", "region": "beam", "field": n}
        for n in ("axial_force", "shear_z_max_abs", "bending_y", "torsion_i", "bending_z_j")}
    case, _ = schema.load(_write_case(tmp_path, d))
    got = {n: api._monitor_unit(case, n) for n in d["output"]["monitors"]}
    assert got == {"axial_force": "kN", "shear_z_max_abs": "kN", "bending_y": "kN*m",
                   "torsion_i": "kN*m", "bending_z_j": "kN*m"}


def test_info_names_tonnes_for_a_kn_m_fibre_model(tmp_path):
    from vfife.cli import _with_element, build, make_solver
    from vfife.io import schema
    d = _cantilever(tmp_path, {"length": "m", "force": "kN", "stress": "kPa"}, False)
    p = _write_case(tmp_path, d)
    case, _ = schema.load(p)
    mesh, material, element = build(case, p)
    s = make_solver(_with_element(case, element, p), mesh, material)
    line = next(l for l in s.summary().splitlines() if l.strip().startswith("mass"))
    assert " t, of it members" in line and "kg" not in line and "kN at g = 9.80665" in line


def test_the_fibre_kernel_reports_the_same_station_magnitudes(tmp_path):
    from vfife.cli import _with_element, build, make_solver
    from vfife.io import schema
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": _line_mesh(tmp_path / "k.msh", 4.0, direction=(1, 2, -0.7))},
         "materials": {"s02": STEEL02},
         "sections": {"s": {"type": "fiber", "material": "s02", "shape": "rect", "width": 1.0, "height": 1.0,
                            "ny": 4, "nz": 4, "shear_stiffness": 1.0e10, "j": 0.1}},
         "assign": [{"region": "beam", "section": "s", "orientation": [0, 0, 1]}],
         "constraints": [{"region": "root", "dofs": [1] * 6}],
         "loads": [{"region": "tip", "vector": [1e4, 2e4, -3e4]}],
         "solver": {"mode": "static", "backend": "numba", "tolerance": 1e-6},
         "output": {"fields": ["displacement"]}}
    p = _write_case(tmp_path, d, "kn")
    case, _ = schema.load(p)
    mesh, material, element = build(case, p)
    s = make_solver(_with_element(case, element, p), mesh, material)
    if getattr(s, "_kernel", None) is None:
        pytest.skip("no fused kernel here: " + str(getattr(s, "kernel_status", "")))
    rng = np.random.default_rng(7)
    s.q[:] = 1e-4 * rng.standard_normal(s.q.shape)
    s.q[s.dofs.node_dofs(np.array([0]), ("ux", "uy", "uz", "rx", "ry", "rz")).ravel()] = 0.0
    a_kernel = np.array(s.internal_force()[1], copy=True)
    abs_kernel = np.array(s.actions_abs, copy=True)
    assert a_kernel.shape == (s.nel, 6)
    s._kernel = None
    s.internal_force()
    np.testing.assert_allclose(abs_kernel, s.actions_abs, rtol=1e-7, atol=1e-6 * np.abs(abs_kernel).max())
    np.testing.assert_allclose(abs_kernel, np.abs(a_kernel), rtol=1e-8)
