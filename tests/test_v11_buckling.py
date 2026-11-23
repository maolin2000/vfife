"""1.1, item 4: linear buckling (`vfife buckling`).

Checked against closed forms: the Euler load of a cantilever, pi^2 EI /
(4 L^2), about two axes, with or without a lateral load beside the axial one
(the first version, a difference of tangents, read the lateral load's
member rotations as softening); the sway load of a fixed-base portal from
the stability functions; a model far from the origin; a mechanism; and the
same factor read from a loaded state, where it is the load ADDED to it.
"""
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from test_member_failure import _write_msh
from vfife import buckling
from vfife.api import prepare_case
from vfife.cli import _with_element, make_solver

E, I, A = 2.0e11, 1.0e-6, 1.0e-3
ROOT = Path(__file__).resolve().parents[1]


def _solver(path):
    prep = prepare_case(str(path))
    return make_solver(_with_element(prep.case, prep.element, prep.case_path), prep.mesh, prep.material)


def _column(tmp_path, n=8, L=5.0, P=-1000.0, tol=1e-9, H=0.0, offset=(0.0, 0.0)):
    nodes = [(offset[0], offset[1], L * i / n) for i in range(n + 1)]
    _write_msh(tmp_path / f"c{n}.msh", nodes,
               {"base": (0, [0]), "top": (0, [n]), "col": (1, [(i, i + 1) for i in range(n)])})
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": f"c{n}.msh"},
         "materials": {"s": {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": 7850}},
         "sections": {"c": {"type": "beam", "material": "s", "shape": "general", "area": A,
                            "iy": I, "iz": 2 * I, "j": 2 * I}},
         "assign": [{"region": "col", "section": "c", "orientation": [1, 0, 0]}],
         "constraints": [{"region": "base", "dofs": [1] * 6}],
         "loads": [{"region": "top", "vector": [0, H, P]}],
         "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 200000, "tolerance": tol},
         "output": {"fields": ["displacement", "rotation"], "every": 100000}}
    p = tmp_path / f"c{n}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return p


def test_a_cantilever_buckles_at_euler_s_load_about_each_axis(tmp_path):
    L = 5.0
    pcr = np.pi ** 2 * E * I / (4 * L ** 2)
    for n in (4, 8):
        r = buckling.analyse(_solver(_column(tmp_path, n, L)), n_modes=3)
        ratio = r.factors * 1000.0 / pcr
        # the consistent geometric matrix: 3e-5 high on four elements
        assert ratio[0] == pytest.approx(1.0, abs=5e-5 if n == 4 else 5e-6)
        assert ratio[1] == pytest.approx(2.0 * ratio[0], rel=1e-6)     # iz = 2 iy
        assert ratio[2] == pytest.approx(9.0, rel=3e-3)                # the second mode, 9 P_cr
        assert r.method == "reference"
        np.testing.assert_allclose(r.axial_force, -1000.0, rtol=1e-6)
    # the first shape: the top swings, the base does not, a unit largest translation
    sh = r.shapes[0].reshape(-1, 6)
    assert abs(sh[-1, :3]).max() == pytest.approx(1.0) and not sh[0].any()


@pytest.mark.parametrize("H", [20.0, 300.0])
def test_a_lateral_load_beside_the_axial_one_changes_nothing(tmp_path, H):
    """The lateral load bends the column but puts no axial force in it: the
    factor is Euler's. The difference of two tangents read 0.36 and 0.09 of
    it, and reading the axial force at the full linear state (where a bent
    member is longer by L theta^2 / 2) read 44 times it."""
    pcr = np.pi ** 2 * E * I / (4 * 25.0)
    r = buckling.analyse(_solver(_column(tmp_path, 8, H=H)), n_modes=1)
    assert r.factors[0] * 1000.0 / pcr == pytest.approx(1.0, abs=1e-4)


def test_far_from_the_origin_and_a_mechanism(tmp_path):
    """Survey coordinates (5e5 m east) do not move the factor; a column on a
    pin is refused by name, not answered with 5e-8."""
    from vfife.core.diagnostics import UserError
    near = buckling.analyse(_solver(_column(tmp_path, 8)), n_modes=1).factors[0]
    far = buckling.analyse(_solver(_column(tmp_path, 8, offset=(3e5, 4e5))), n_modes=1).factors[0]
    assert far == pytest.approx(near, rel=1e-4)
    path = _column(tmp_path, 8)
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    d["constraints"] = [{"region": "base", "dofs": [1, 1, 1, 1, 0, 1]}]          # a pin about y
    path.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    with pytest.raises(UserError) as exc:
        buckling.analyse(_solver(path), n_modes=1)
    assert exc.value.code == "BUCKLING_BASE_NOT_STABLE"


def test_survey_coordinates_do_not_move_the_modes(tmp_path):
    """`vfife modal` and `spectrum` took the tangent's step from the largest
    coordinate: the column 5e5 m from the origin came out 1.8 to 3.7 times
    stiffer (1.4.1). The step is the model's size now, as buckling's."""
    from vfife import modal
    near = modal.analyse(_solver(_column(_mk(tmp_path / "n"), 8)), n_modes=3).frequencies
    far = modal.analyse(_solver(_column(_mk(tmp_path / "f"), 8, offset=(3e5, 4e5))), n_modes=3).frequencies
    assert far == pytest.approx(near, rel=1e-4)


def test_a_portal_sways_at_the_stability_function_load(tmp_path):
    """Two 4 m columns fixed at the base, a 6 m beam of the same EI, 1 kN on
    each column top: the sway load from the slope-deflection equations with
    stability functions is 0.66944 pi^2 EI / H^2 per column."""
    H, B, n = 4.0, 6.0, 8
    nodes = ([(0, 0, H * i / n) for i in range(n + 1)] + [(B, 0, H * i / n) for i in range(n + 1)]
             + [(B * i / n, 0, H) for i in range(1, n)])
    left, right = list(range(n + 1)), list(range(n + 1, 2 * n + 2))
    top = [left[-1]] + list(range(2 * n + 2, 3 * n + 1)) + [right[-1]]
    _write_msh(tmp_path / "portal.msh", nodes,
               {"bases": (0, [left[0], right[0]]), "tops": (0, [left[-1], right[-1]]),
                "cols": (1, [(left[i], left[i + 1]) for i in range(n)] + [(right[i], right[i + 1]) for i in range(n)]),
                "beam": (1, [(top[i], top[i + 1]) for i in range(n)])})
    sec = {"type": "beam", "material": "s", "shape": "general", "area": A, "iy": I, "iz": I, "j": 2 * I}
    d = {"units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": "portal.msh"},
         "materials": {"s": {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": 7850}},
         "sections": {"c": sec},
         "assign": [{"region": "cols", "section": "c", "orientation": [0, 1, 0]},
                    {"region": "beam", "section": "c", "orientation": [0, 1, 0]}],
         "constraints": [{"region": "bases", "dofs": [1] * 6}, {"region": "all", "dofs": [0, 1, 0, 1, 0, 1]}],
         "loads": [{"region": "tops", "vector": [0, 0, -2000.0]}],
         "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 1000}}
    p = tmp_path / "portal.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    r = buckling.analyse(_solver(p), n_modes=2)
    per_column = r.factors[0] * 1000.0
    assert per_column / (np.pi ** 2 * E * I / H ** 2) == pytest.approx(0.669441, rel=5e-4)


def test_a_bar_in_tension_does_not_buckle(tmp_path):
    r = buckling.analyse(_solver(_column(tmp_path, 4, P=+1000.0)), n_modes=3)
    assert not len(r.factors) and "stiffen" in r.note
    assert r.n_negative >= 1          # reversed, it is the column above


def test_from_a_loaded_state_the_factor_is_the_load_added(tmp_path):
    """Run to equilibrium under its load first, the column has one load
    fewer to go: the factor from that state is the one from the reference
    configuration less one."""
    path = _column(tmp_path, 4)
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    d["solver"] = {"backend": "numpy", "mode": "static"}          # Newton: a second, not minutes
    path.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    lin = buckling.analyse(_solver(path), n_modes=1).factors[0]
    h5 = tmp_path / "relaxed.h5"
    out = subprocess.run([sys.executable, "-m", "vfife", "run", str(path), "-o", str(h5)], cwd=tmp_path,
                         capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stdout + out.stderr
    res = subprocess.run([sys.executable, "-m", "vfife", "buckling", str(path), "--result", str(h5),
                          "--modes", "1", "-o", str(tmp_path / "b.h5")], cwd=tmp_path,
                         capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, res.stdout + res.stderr
    with h5py.File(tmp_path / "b.h5") as f:
        at_state = float(f["load_factor"][0])
        assert f.attrs["method"] == "state"
    assert at_state == pytest.approx(lin - 1.0, rel=1e-3)


def test_the_command_writes_the_factors_and_the_shapes(tmp_path):
    path = _column(tmp_path, 4)
    res = subprocess.run([sys.executable, "-m", "vfife", "buckling", str(path), "--modes", "2",
                          "-o", str(tmp_path / "b.h5"), "--paraview", str(tmp_path / "pv")],
                         cwd=tmp_path, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "load factor" in res.stdout
    with h5py.File(tmp_path / "b.h5") as f:
        assert f.attrs["buckling_format"] == 1
        assert f["shape"].shape == (2, 5, 6) and len(f["load_factor"]) == 2
    assert (tmp_path / "pv" / "buckling_all.pvd").exists()
    assert "load factor" in (tmp_path / "pv" / "buckling.txt").read_text(encoding="utf-8")


def test_a_shell_is_refused_by_name():
    from vfife.core.diagnostics import UserError
    with pytest.raises(UserError) as exc:
        buckling.analyse(_solver(ROOT / "cases" / "case_10_clamped_square_shell.yaml"), n_modes=1)
    assert exc.value.code == "BUCKLING_FAMILY"


def test_a_prestressed_reference_is_refused_by_name():
    """case_24f's tendons are an internal force at the reference
    configuration: linear pre-buckling from there buckled the girder at 1.7
    times its dead load."""
    from vfife.core.diagnostics import UserError
    s = _solver(ROOT / "cases" / "case_24f_hzb_girder_bearings.yaml")
    with pytest.raises(UserError) as exc:
        buckling.analyse(s, n_modes=1)
    assert exc.value.code == "BUCKLING_BASE_NOT_AT_REST"


def _mk(path):
    path.mkdir(parents=True, exist_ok=True)
    return path
