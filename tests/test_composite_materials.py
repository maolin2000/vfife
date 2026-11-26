"""The composite one-dimensional laws `parallel` and `series`, and `gap`
(1.21), against their closed forms, and in the two places a case puts a
one-dimensional law: a bar and a fibre section."""

import pathlib

import numpy as np
import pytest
import yaml

from vfife.core.diagnostics import UserError
from vfife.materials import create

STEEL = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}
SOFT = {"type": "linear_elastic", "E": 1.0e11, "nu": 0.3, "rho": 7850.0}
PLATEAU = {"type": "piecewise_uniaxial", "E": 2.0e11, "rho": 7850.0, "tension": [[0.0, 2.0e8], [1.0, 2.0e8]]}
STEEL02 = {"type": "steel02", "E": 2.0e11, "Fy": 4.0e8, "b": 0.01, "rho": 7850.0}


def follow(mat, history, state=None):
    """Stress along a strain history, committing after each point."""
    state = mat.init_state(1) if state is None else state
    out = []
    for e in history:
        eps = np.zeros((1, 6)); eps[0, 0] = e
        trial = state.copy()
        out.append(mat.update(eps, trial, 1.0)[0, 0])
        state = trial
    return np.array(out), state


HISTORY = [0.5e-3, 2.0e-3, 5.0e-3, 3.0e-3, -3.0e-3, 0.0, 4.0e-3]


def test_parallel_adds_the_stresses_and_a_part_is_a_uniaxial_stress():
    """E1 + E2 to the bit; `linear_elastic` read as E eps, not as the
    constrained modulus its 3-D form gives [eps, 0, ...] (35% stiff at nu = 0.3)."""
    m = create("parallel", of=[STEEL, SOFT])
    s, _ = follow(m, HISTORY)
    assert np.array_equal(s, 2.0e11 * np.array(HISTORY) + 1.0e11 * np.array(HISTORY))
    assert m.E == 3.0e11 and m.rho == 2 * 7850.0
    half = create("parallel", of=[STEEL, SOFT], factors=[0.5, 2.0])
    assert follow(half, [1.0e-3])[0][0] == pytest.approx(0.5 * 2.0e8 + 2.0 * 1.0e8, rel=1e-15)


def test_an_elastic_part_beside_a_plastic_one_is_the_bilinear_law():
    """k1 = E1 + E2, k2 = E2, yield at the plastic part's fy: the connector's
    `bilinear` law, round a cycle (kinematic: the plastic part's yield stays
    +-fy about the elastic part's line)."""
    m = create("parallel", of=[PLATEAU, SOFT])
    hist = np.r_[np.linspace(0, 5e-3, 11), np.linspace(5e-3, -5e-3, 21)[1:], np.linspace(-5e-3, 2e-3, 15)[1:]]
    s, _ = follow(m, hist)
    from vfife.elements import connector
    law = {"type": "bilinear", "k1": 3.0e11, "k2": 1.0e11, "fy": 2.0e8 * 3.0 / 2.0}
    slip = 0.0; ref = []
    for e in hist:
        f, slip = connector.law_force(law, float(e), slip, 0.0)
        ref.append(f)
    assert np.allclose(s, ref, rtol=1e-12, atol=1e-3)


def test_series_adds_the_compliances_and_yields_at_the_weaker_part():
    el, _ = follow(create("series", of=[STEEL, SOFT]), HISTORY)
    E = 1.0 / (1.0 / 2.0e11 + 1.0 / 1.0e11)
    assert np.allclose(el, E * np.array(HISTORY), rtol=1e-10, atol=1e-3)
    # elastic in series with a perfectly plastic part: E/2 to 200 MPa, and
    # back down the series stiffness
    s, state = follow(create("series", of=[STEEL, PLATEAU]), HISTORY)
    assert np.allclose(s / 1e6, [50.0, 200.0, 200.0, 0.0, -200.0, 100.0, 200.0], atol=1e-5)
    # three parts: two elastic links and the plateau
    s3, _ = follow(create("series", of=[STEEL, STEEL, PLATEAU]), HISTORY)
    assert np.allclose(s3 / 1e6, [100 / 3, 400 / 3, 200.0, 200 / 3, -200.0, 0.0, 200.0], atol=1e-5)


def test_two_perfectly_plastic_parts_in_series_and_a_trial_leaves_nothing_behind():
    weak = {**PLATEAU, "tension": [[0.0, 1.5e8], [1.0, 1.5e8]]}
    m = create("series", of=[PLATEAU, weak])
    s, state = follow(m, [1.0e-3, 4.0e-3, 2.0e-3])
    assert np.allclose(s / 1e6, [100.0, 150.0, 150.0 - 2.0e-3 * 1.0e11 / 1e6], atol=1e-5)
    # the iteration works on copies: handed the committed state itself, the
    # law leaves in it exactly what a run on a copy arrives at -- nothing of
    # the iterates it tried on the way
    eps = np.zeros((1, 6)); eps[0, 0] = 6.0e-3
    on_copy = state.copy()
    a = m.update(eps, on_copy, 1.0)[0, 0]
    b = m.update(eps, state, 1.0)[0, 0]
    assert a == b and np.array_equal(state, on_copy)


def test_points_that_converge_at_different_iterations_and_a_part_that_ruptures():
    """Several points at once, one elastic and one far past yield, each its
    own answer; and a series whose steel ruptures carries nothing from there
    on instead of refusing the step (found in review: the bracket was not
    widened far enough to find the root near zero)."""
    m = create("series", of=[STEEL, PLATEAU])
    eps = np.zeros((3, 6)); eps[:, 0] = [0.5e-3, 5.0e-3, -1.0e-4]
    sig = m.update(eps, m.init_state(3), 1.0)[:, 0]
    assert np.allclose(sig / 1e6, [50.0, 200.0, -10.0], atol=1e-6)
    brittle = create("series", of=[STEEL, {**STEEL02, "rupture_strain": 0.01}])
    s, _ = follow(brittle, np.linspace(0.0, 0.02, 401))
    assert np.all(np.isfinite(s)) and abs(s[-1]) < 1.0 and s.max() > 3.5e8


def test_density_failure_and_poisson_of_a_composite():
    """Side by side the densities add (with the factors), end to end it is
    the mean over every part; `rho` says otherwise when given. A part's
    failure is the series' and, only when every part has failed, the
    parallel's. Poisson's ratio (a fibre section's torsion) is the first part's."""
    assert create("parallel", of=[STEEL, SOFT]).rho == 2 * 7850.0
    assert create("parallel", of=[STEEL, SOFT], rho=7850.0).rho == 7850.0
    light = {**SOFT, "rho": 1000.0}
    assert create("series", of=[STEEL, STEEL, light]).rho == pytest.approx((2 * 7850.0 + 1000.0) / 3)
    assert create("series", of=[STEEL, SOFT]).nu == 0.3
    rupt = {**STEEL02, "rupture_strain": 0.002}
    ser = create("series", of=[STEEL, rupt]); par = create("parallel", of=[STEEL, rupt])
    assert create("parallel", of=[STEEL, SOFT]).failure(np.zeros((1, 0))) is None
    _, st_s = follow(ser, [0.004, 0.006]); _, st_p = follow(par, [0.004])
    assert ser.failure(st_s)[0] == 1.0 and par.failure(st_p)[0] == 0.0


def test_a_part_written_in_place_takes_the_units_a_named_one_does():
    """In a mm case a named steel02 gets its density and yield in N, mm; one
    written inside `of` did not (found in review: 7850 and 400e6 kept)."""
    from vfife.core.units import fill_defaults
    d = {"units": {"length": "mm", "force": "N"},
         "materials": {"steel": {"type": "steel02", "E": 2.0e5},
                       "two": {"type": "parallel", "of": ["steel", {"type": "steel02", "E": 2.0e5}]}}}
    d, _ = fill_defaults(d)
    named, inline = d["materials"]["steel"], d["materials"]["two"]["of"][1]
    assert inline["rho"] == pytest.approx(named["rho"]) and inline["Fy"] == pytest.approx(named["Fy"])
    assert inline["Fy"] == pytest.approx(400.0)


def test_a_split_that_does_not_converge_is_an_error():
    m = create("series", of=[STEEL02, PLATEAU], max_iterations=1)
    eps = np.zeros((1, 6)); eps[0, 0] = 5.0e-3
    with pytest.raises(UserError) as e:
        m.update(eps, m.init_state(1), 1.0)
    assert e.value.code == "SERIES_NOT_CONVERGED"


def test_a_gap_and_a_stop_made_of_one():
    g = create("gap", E=1.0e9, rho=1.0, lo=-0.01, hi=0.02)
    s, _ = follow(g, [-0.02, 0.0, 0.01, 0.03])
    assert np.allclose(s, [-1.0e7, 0.0, 0.0, 1.0e7], rtol=1e-12, atol=0.0)
    # slack in series with a spring: free to the gap, then the spring's stiffness in series
    stop, _ = follow(create("series", of=[{"type": "gap", "E": 1.0e9, "rho": 1.0, "hi": 0.01},
                                          {**STEEL, "E": 1.0e9}]), [0.005, 0.03])
    assert stop[0] == pytest.approx(0.0, abs=1e-3) and stop[1] == pytest.approx(0.02 * 0.5e9, rel=1e-8)
    # beside a soft spring: the spring alone inside the travel, both past it
    side, _ = follow(create("parallel", of=[{"type": "gap", "E": 1.0e9, "rho": 1.0, "hi": 0.01},
                                            {**STEEL, "E": 1.0e7}]), [0.005, 0.03])
    assert np.allclose(side, [1.0e7 * 0.005, 1.0e7 * 0.03 + 1.0e9 * 0.02])
    for bad in ({}, {"lo": 0.1, "hi": 0.0}):
        with pytest.raises(ValueError):
            create("gap", E=1.0e9, rho=1.0, **bad)


def test_what_a_composite_refuses():
    with pytest.raises(ValueError, match="two or more"):
        create("parallel", of=[STEEL])
    with pytest.raises(ValueError, match="one-dimensional"):
        create("series", of=[STEEL, {"type": "j2", "E": 2e11, "nu": 0.3, "rho": 7850.0,
                                     "hardening": {"source": "linear", "yield_strength": 3e8, "H": 1e9}}])
    with pytest.raises(ValueError, match="creep"):
        create("parallel", of=[STEEL, {**STEEL, "creep": {"model": "curve", "curve": [[0, 0], [1, 1]]}}])
    with pytest.raises(ValueError, match="factors"):
        create("parallel", of=[STEEL, SOFT], factors=[1.0])


# ------------------------------------------------------------- in a case

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _bar_case(tmp_path, materials, section_material):
    d = yaml.safe_load((ROOT / "cases" / "case_02_truss_bar.yaml").read_text(encoding="utf-8"))
    d["mesh"]["path"] = str((ROOT / "cases" / d["mesh"]["path"]).resolve())
    d["materials"] = materials
    d["sections"]["bar"]["material"] = section_material
    d["output"] = {"path": str(tmp_path / "bar.h5"), "every": 10 ** 6,
                   "monitors": d["output"]["monitors"]}
    return d


def test_a_bar_of_a_series_named_in_the_table(tmp_path):
    """The parts are names in `materials`, resolved where the table is read:
    case_02's 5e-5 m with E, three times it with E in series with E/2."""
    from vfife.api import run_case
    mats = {"steel": STEEL, "link": SOFT, "both": {"type": "series", "of": ["steel", "link"]}}
    out = run_case(_bar_case(tmp_path, mats, "both"))
    assert out.converged
    assert float(out.monitors["loaded_ux"][-1]) == pytest.approx(1.5e-4, rel=2e-3)


def test_a_composite_that_names_itself_or_nothing_is_refused(tmp_path):
    from vfife.api import prepare_case
    for mats, what in (({"steel": STEEL, "a": {"type": "series", "of": ["steel", "b"]},
                         "b": {"type": "parallel", "of": ["a", "steel"]}}, "part of itself"),
                       ({"steel": STEEL, "a": {"type": "series", "of": ["steel", "nothing"]}}, "not in `materials`")):
        with pytest.raises(Exception, match=what):
            prepare_case(_bar_case(tmp_path, mats, "a"))


def _fiber_case(tmp_path, materials, section_material):
    msh = tmp_path / "fiber.msh"
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n2\n1 0 0 0\n2 1 0 0\n$EndNodes\n"
                   "$Elements\n1\n1 1 2 1 1 1 2\n$EndElements\n", encoding="utf-8")
    return {
        "materials": materials,
        "sections": {"rc": {"type": "fiber", "material": section_material, "shape": "rect",
                            "width": 0.05, "height": 0.10, "ny": 4, "nz": 4}},
        "assign": [{"region": "all", "section": "rc"}],
        "mesh": {"type": "gmsh", "path": str(msh)},
        "regions": {"fixed": {"box": [-0.01, 0.01, -0.01, 0.01, -0.01, 0.01]},
                    "tip": {"box": [0.99, 1.01, -0.01, 0.01, -0.01, 0.01]}},
        "constraints": [{"region": "fixed", "dofs": [True] * 6}],
        "loads": [{"region": "tip", "vector": [50000.0, 0, 0]}],
        "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 1200, "tolerance": 1.0e-3,
                   "load_steps": 3, "max_steps_per_load": 1500},
        "output": {"path": str(tmp_path / "fiber.h5"), "every": 10 ** 6,
                   "monitors": {"tip": {"region": "tip", "dof": "x", "kind": "displacement"}}},
    }


def test_a_fibre_section_of_a_composite(tmp_path):
    """P L / (E A) = 5e-5 m with E = 200 GPa; a parallel of two such is half
    of it, a series of two twice it. The section runs on the NumPy path."""
    from vfife.api import run_case
    for kind, expect in (("parallel", 2.5e-5), ("series", 1.0e-4)):
        mats = {"steel": STEEL, "two": {"type": kind, "of": ["steel", "steel"]}}
        out = run_case(_fiber_case(tmp_path, mats, "two"))
        assert out.converged
        assert float(out.monitors["tip"][-1]) == pytest.approx(expect, rel=0.03), kind
