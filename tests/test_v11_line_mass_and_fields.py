"""1.1, item 1: a `line_load` that carries its mass, and the output fields a
run writes when its case names none.

A superimposed dead load written as a `line_load` weighed and had no inertia,
so a seismic run, the modes and a response spectrum left it out of the mass
(the bridge scenario). `with_mass: true` gives the load its
mass |q|/g, half of each member's to each end, all translations, weightless.

Until 1.1 a case without `output.fields` got the solid's [displacement,
von_mises]: a frame wrote its displacements and nothing else.
"""
import h5py
import numpy as np
import pytest
import yaml

from test_added_mass import STEEL, _column_case, _modes, _solver
from test_member_failure import _run
from vfife.core import added_mass
from vfife.io import schema

G = 9.80665
Q = 2.0e4                     # N/m: a 2 t/m superimposed dead load, nearly


def _line(q=Q, **kw):
    return dict({"region": "col", "line_load": [0.0, 0.0, -q], "with_mass": True}, **kw)


# --------------------------------------------------------------------------- #
#  the mass
# --------------------------------------------------------------------------- #
def test_the_mass_is_q_over_g_lumped_to_the_ends_and_weighs_nothing(tmp_path):
    p, _ = _column_case(tmp_path, "m", loads=[_line()])
    s = _solver(p)
    add = s.additional_mass
    total = Q * 4.0 / G
    assert add[:, 0].sum() == pytest.approx(total, rel=1e-12)
    np.testing.assert_allclose(add[:, 0], add[:, 2])          # every translation
    # four 1 m members: a quarter-share at each end, a half at the inner nodes
    np.testing.assert_allclose(add[:, 0], total / 4 * np.array([0.5, 1, 1, 1, 0.5]), rtol=1e-12)
    np.testing.assert_allclose(s.weightless_mass, add)


def test_without_with_mass_a_line_load_has_no_mass(tmp_path):
    p, _ = _column_case(tmp_path, "n", loads=[_line(with_mass=False)])
    assert not np.any(_solver(p).additional_mass)


def test_gravity_does_not_weigh_it_a_second_time(tmp_path):
    """The column's own weight under `gravity`, plus q H from the load --
    not 2 q H."""
    mons = {"rz": {"kind": "reaction", "region": "base", "dof": "z"}}
    loads = [{"region": "all", "gravity": [0, 0, -G]}, _line()]
    _p, d = _column_case(tmp_path, "g", loads=loads, material=STEEL, monitors=mons)
    _out, h5 = _run(tmp_path, "g", d)
    with h5py.File(h5) as f:
        rz = float(np.asarray(f["monitors/rz"])[-1])
    assert rz == pytest.approx(7850.0 * 1e-2 * 4.0 * G + Q * 4.0, rel=1e-6)


def test_the_modes_see_it_as_a_heavier_member(tmp_path):
    """A column with the load's mass sways like one whose density carries
    q / (g A) more -- to 1.1 % on four members: the heavier member's lumped
    rotary inertia rho A L^3 / 24 carries the extra density too, and the
    load's mass is translational only."""
    p1, _ = _column_case(tmp_path, "a", loads=[_line()], material=STEEL)
    heavier = dict(STEEL, rho=7850.0 + Q / (G * 1e-2))
    p2, _ = _column_case(tmp_path, "b", material=heavier)
    r1, _ = _modes(p1)
    r2, _ = _modes(p2)
    for axis in (0, 1):
        w1 = np.min(r1.omega[np.abs(r1.gamma[:, axis]) > 1e-6 * np.abs(r1.gamma[:, axis]).max()])
        w2 = np.min(r2.omega[np.abs(r2.gamma[:, axis]) > 1e-6 * np.abs(r2.gamma[:, axis]).max()])
        assert w1 == pytest.approx(w2, rel=0.02)
    r0, _ = _modes(_column_case(tmp_path, "c", material=STEEL)[0])
    assert r1.omega[0] < 0.5 * r0.omega[0]          # the load is 25 times the column


def test_g_is_in_the_declared_units():
    assert added_mass.line_load_g(None) == G
    assert added_mass.line_load_g({"length": "mm", "force": "N", "stress": "MPa"}) == pytest.approx(G * 1e3)
    assert added_mass.line_load_g({"length": "m", "force": "kN", "stress": "kPa"}) == pytest.approx(G)


def test_it_is_refused_where_it_means_nothing():
    base = {"region": "col"}
    with pytest.raises(ValueError, match="with_mass applies to line_load only"):
        schema.LoadSpec(**base, vector=[0, 0, -1.0], with_mass=True)
    with pytest.raises(ValueError, match="with_mass is the model's"):
        schema.StepSpec(loads=[dict(base, line_load=[0, 0, -1.0], with_mass=True)])
    schema.StepSpec(loads=[dict(base, line_load=[0, 0, -1.0])])           # without it: fine


def test_a_checkpoint_from_1_0_still_fits(tmp_path):
    """`with_mass: false` is how 1.0 read every load: the fingerprint leaves
    it out, so a 1.0 checkpoint resumes; `true` is another model."""
    from vfife.io import fingerprint as fp
    from vfife.api import prepare_case

    def fprint(load):
        p, _ = _column_case(tmp_path, "f", loads=[load])
        return fp.model_fingerprint(_solver(p), prepare_case(str(p)))["conditions_json"]

    plain = {"region": "col", "line_load": [0.0, 0.0, -Q]}
    assert fprint(plain) == fprint(dict(plain, with_mass=False))
    assert fprint(plain) != fprint(dict(plain, with_mass=True))


# --------------------------------------------------------------------------- #
#  the default output fields
# --------------------------------------------------------------------------- #
def _written(tmp_path, name, d):
    _out, h5 = _run(tmp_path, name, d)
    with h5py.File(h5) as f:
        last = sorted(f["steps"])[-1]
        names = set(f["steps"][last])
    return names


def _short_column(tmp_path, name, **out):
    _p, d = _column_case(tmp_path, name, loads=[{"region": "top", "vector": [1e3, 0, 0]}],
                         material=STEEL)
    d["solver"] = {"backend": "numpy", "mode": "dynamic", "total_time": 0.002}
    d["output"] = dict(out, every=50)
    return d


def test_a_frame_without_fields_writes_its_rotations_forces_and_reactions(tmp_path):
    names = _written(tmp_path, "fr", _short_column(tmp_path, "fr"))
    for f in ("displacement", "rotation", "reaction_force", "reaction_moment", "axial_force",
              "shear_y", "shear_z", "torsion", "bending_y", "bending_z"):
        assert f in names, f
    # not the end actions, the envelopes or the failure state of a member
    # that cannot fail
    assert not {"axial_force_i", "bending_y_max_abs", "member_failed"} & names


def test_named_fields_are_written_as_named(tmp_path):
    names = _written(tmp_path, "nm", _short_column(tmp_path, "nm", fields=["displacement"]))
    assert "displacement" in names and "axial_force" not in names and "rotation" not in names


def test_a_member_that_can_fail_writes_its_failure_state(tmp_path):
    d = _short_column(tmp_path, "mf")
    d["assign"][0]["failure"] = {"yield_strength": 3.0e12, "extreme_fibre": [0.1, 0.1]}
    names = _written(tmp_path, "mf", d)
    assert {"member_failed", "capacity_ratio"} <= names


def test_preflight_says_a_dynamic_line_load_has_no_mass(tmp_path):
    from vfife.api import prepare_case

    def codes(load, material):
        _p, d = _column_case(tmp_path, "pf", loads=[load], material=material, mode="dynamic")
        p = tmp_path / "pf.yaml"
        p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
        return {i["code"] for i in prepare_case(str(p)).preflight["issues"]}

    assert "LINE_LOAD_WITHOUT_MASS" in codes(_line(with_mass=False), STEEL)
    assert "LINE_LOAD_WITHOUT_MASS" not in codes(_line(), STEEL)


def test_a_member_that_splits_off_takes_its_load_s_mass_with_it(tmp_path):
    """A 2 m beam carrying 2 kN/m `with_mass`, pinned at one end and hung
    from a light bar at the other; both cut at 0.05 s. The fragment carries
    its load AND the load's mass, so it falls at g -- not at 3.6 g, the
    load's force on the beam's own 78.5 kg/m, which is what the mass left
    behind on the old particles gave (review of 1.1)."""
    from test_member_failure import _write_msh
    L = 2.0
    _write_msh(tmp_path / "hang.msh", [(0, 0, 0), (L, 0, 0), (L, 0, 5.0)],
               {"pin": (0, [0]), "tip": (0, [1]), "top": (0, [2]),
                "beam": (1, [(0, 1)]), "cable": (1, [(1, 2)])})
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": "hang.msh"},
         "materials": {"steel": STEEL, "light": dict(STEEL, rho=1.0)},
         "sections": {"b": {"type": "beam", "material": "steel", "shape": "general", "area": 0.01,
                            "iy": 1e-5, "iz": 1e-5, "j": 2e-5},
                      "c": {"type": "bar", "material": "light", "shape": "general", "area": 0.1}},
         "assign": [{"region": "beam", "section": "b", "orientation": [0, 0, 1]},
                    {"region": "cable", "section": "c", "tension_only": True}],
         "constraints": [{"region": "pin", "dofs": [1, 1, 1, 1, 0, 1]},
                         {"region": "top", "dofs": [1] * 6}],
         "loads": [{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"},
                   {"region": "beam", "line_load": [0, 0, -2000.0], "with_mass": True,
                    "amplitude": "hold"}],
         "removals": [{"region": "beam", "time": 0.05}, {"region": "cable", "time": 0.05}],
         "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.25,
                    "damping": {"type": "none"}},
         "output": {"fields": ["displacement"], "every": 50}}
    _out, h5 = _run(tmp_path, "hang", d)
    with h5py.File(h5) as f:
        keys = sorted(f["steps"])
        t = np.array([f["steps"][k].attrs["time"] for k in keys])
        U = np.array([np.asarray(f["steps"][k]["displacement"]) for k in keys])
        conn = np.asarray(f["steps"][keys[-1]]["connectivity"]).astype(int)
    late = t > 0.1
    for n in conn[0]:
        accel = 2.0 * np.polyfit(t[late], U[late, n, 2], 2)[0]
        assert accel == pytest.approx(-G, rel=1e-3)


def test_a_shell_default_is_not_a_member_s(tmp_path):
    """A shell's `shear_y` is half of its own shear resultants: the member
    forces' default must not pick it out alone (review of 1.1)."""
    from pathlib import Path
    from vfife.api import _output_fields, load_case, prepare_case
    from vfife.cli import _with_element, make_solver
    root = Path(__file__).resolve().parents[1]
    draft = load_case(root / "cases" / "case_10_clamped_square_shell.yaml")
    draft.get("output").pop("fields")
    prep = prepare_case(draft)
    s = make_solver(_with_element(prep.case, prep.element, prep.case_path), prep.mesh, prep.material)
    fields = _output_fields(s, prep.case)
    assert {"displacement", "rotation", "von_mises_top", "von_mises_bottom"} <= set(fields)
    assert "shear_y" not in fields
