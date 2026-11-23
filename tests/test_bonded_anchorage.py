"""`anchorage` on a bonded bar: its free ends tied to the host.

WHY
    A bonded bar that stops inside the mesh is, under the bond law alone,
    a free end: the last node holds by the bond on its own half-segment and
    pulls out at tau_max. Real bars do not end like that -- they run past
    the support, hook, or bear on a plate (the Alberta beam has 216 mm plates
    over the bars at each support), and the bond law was never meant to
    represent that. `anchorage: ends` (or `start` / `end`) makes the chosen
    end nodes move with the host, exactly as a perfectly bonded end does:
    no dofs of their own, the segment force handed to the host through the
    interpolation weights, the interface switched off there.

WHAT IS PINNED
    * an anchored end carries the host strain; a free end carries none
    * the coupling is conservative: the support reaction is the load
    * a fully anchored bar in a uniformly strained host IS the perfectly
      bonded bar
    * which nodes `ends` / `start` / `end` select, and the refusals
"""
import numpy as np
import pytest
import yaml

from vfife.api import prepare_case
from vfife.core.diagnostics import UserError
from vfife.core.solver import Solver
from vfife.elements.bonded import BondedTruss
from vfife.materials import create

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

L = 1.0
E_HOST, E_BAR, A_BAR = 3.0e10, 2.0e11, 2.0e-4
DELTA = 2.0e-4                                     # host strain 2e-4


def _case(tmp_path, bond=True, anchorage=None, bar_from=0.0, bar_to=L):
    assign = {"reinforcement": "bar", "section": "bar16", "name": "bar", "divisions": 8}
    if bond:
        assign["bond"] = {"type": "bond_ceb", "fc": 40.0e6}
    if anchorage is not None:
        assign["anchorage"] = anchorage
    case = {
        "name": "anch", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"dimensions": [L, 0.2, 0.2], "divisions": [8, 2, 2]},
        "materials": {"c": {"type": "linear_elastic", "E": E_HOST, "nu": 0.0, "rho": 2400.0},
                      "s": {"type": "rebar", "E": E_BAR, "rho": 7850.0,
                            "yield_strength": 5.0e8, "hardening": 2.0e9}},
        "sections": {"body": {"type": "solid", "material": "c", "formulation": "b-bar"},
                     "bar16": {"type": "bar", "material": "s", "shape": "circle",
                               "diameter": 0.016, "area": A_BAR}},
        "reinforcement": {"bar": {"line": {"start": [bar_from, 0.1, 0.1], "end": [bar_to, 0.1, 0.1]}}},
        "assign": [assign],
        "regions": {"fix": {"where": {"x": 0.0}}, "tip": {"where": {"x": L}}},
        "constraints": [{"region": "fix", "dofs": [1, 1, 1]}],
        "steps": [{"name": "pull", "constraints": [{"region": "tip", "dofs": [1, 0, 0], "value": DELTA}],
                   "load_steps": 2, "max_steps_per_load": 20000, "tolerance": 1.0e-6}],
        "solver": {"backend": "numpy", "mode": "relaxation", "mass": "underwood",
                   "relaxation_safety": 0.3, "damping": {"type": "kinetic"}},
        "output": {"path": str(tmp_path / "a.h5"), "monitors": {
            "R": {"kind": "reaction", "region": "fix", "dof": "x"}}},
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "a.yaml"
    path.write_text(yaml.safe_dump(case), encoding="utf-8")
    return path


def _run(path):
    p = prepare_case(str(path))
    s = Solver(p.case.model_dump(mode="python"), p.mesh, p.material, log=lambda *_: None)
    s.run(on_progress=lambda *_: None, progress_every=500)
    return s


def _end_stresses(s):
    """Axial stress in the bar's first and last segment."""
    bar = s.bonded[0] if s.bonded else s.rebar[0]
    if s.bonded:
        _, _, sig = bar.force(s.u, dt=1.0, want_stress=True)
    else:
        _, sig = bar.force(s.u, 1.0, want_stress=True)
    sig = np.asarray(sig).reshape(bar.n_elem, -1)[:, 0]
    return float(sig[0]), float(sig[-1]), sig


def test_an_anchored_end_carries_the_host_strain_and_a_free_end_carries_none(tmp_path):
    """The bar stops 0.1 m short of the loaded face on both sides, so its
    ends are inside the mesh. Free, the end segments are nearly unloaded;
    tied, they carry E * eps_host like the rest of the bar."""
    eps = DELTA / L
    free = _run(_case(tmp_path / "free", bar_from=0.1, bar_to=0.9))
    tied = _run(_case(tmp_path / "tied", bar_from=0.1, bar_to=0.9, anchorage="ends"))
    s0_free, s1_free, _ = _end_stresses(free)
    s0_tied, s1_tied, all_tied = _end_stresses(tied)
    # 2-3% under E * eps: the tie hands the end force to the host at one
    # point, which strains the host a little more there
    assert s0_tied == pytest.approx(E_BAR * eps, rel=0.05)
    assert s1_tied == pytest.approx(E_BAR * eps, rel=0.05)
    # the free end holds by the bond on its last half-segment alone: with
    # MC2010's stiff early branch that is still 2/3 of the full strain here,
    # and it is the bond law, not the host, that sets it
    assert abs(s0_free) < 0.8 * E_BAR * eps and abs(s1_free) < 0.8 * E_BAR * eps
    assert s0_tied > 1.2 * abs(s0_free)
    _, slip_free = free.bonded[0].slip(free.u)
    assert np.abs(slip_free).max() > 1e-2 * DELTA
    assert np.allclose(all_tied, E_BAR * eps, rtol=0.05)
    # and no slip anywhere on the tied bar
    _, slip = tied.bonded[0].slip(tied.u)
    assert np.abs(slip).max() < 1e-2 * DELTA
    assert tied.bonded[0].anchored.sum() == 2


def test_the_tie_is_conservative(tmp_path):
    """Whatever the bar hands the host through its tied ends is internal
    force: the support reaction must equal the total elastic pull of host
    plus bar."""
    s = _run(_case(tmp_path, anchorage="ends"))
    R = s.monitors["R"].values[-1]
    expected = (E_HOST * 0.04 + (E_BAR - 0.0) * A_BAR) * DELTA / L
    assert abs(R) == pytest.approx(expected, rel=2e-3)


def test_a_fully_anchored_bar_in_a_uniform_host_is_the_perfectly_bonded_bar(tmp_path):
    bonded = _run(_case(tmp_path / "b", anchorage="ends"))
    perfect = _run(_case(tmp_path / "p", bond=False))
    _, _, sb = _end_stresses(bonded)
    _, _, sp = _end_stresses(perfect)
    np.testing.assert_allclose(sb, sp, rtol=2e-3)
    np.testing.assert_allclose(bonded.monitors["R"].values, perfect.monitors["R"].values, rtol=2e-3)


def test_which_nodes_start_end_and_ends_select():
    nodes = np.array([[0.0, 0.1, 0.1], [0.5, 0.1, 0.1], [1.0, 0.1, 0.1],
                      [0.0, 0.1, 0.15], [1.0, 0.1, 0.15]])
    conn = np.array([[0, 1], [1, 2], [3, 4]])          # a two-segment line and a one-segment line
    host = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 0.2, 0], [0.0, 0.2, 0],
                     [0.0, 0, 0.2], [1.0, 0, 0.2], [1.0, 0.2, 0.2], [0.0, 0.2, 0.2]])
    hconn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    steel = create("rebar", E=2.0e11, rho=7850.0, yield_strength=5e8, hardening=2e9)
    bond = create("bond_ceb", fc=40e6)

    def anchored(kind):
        return BondedTruss(host, hconn, "hexa8", nodes, conn, steel, bond, 2e-4, 0.016,
                           anchorage=kind).anchored.tolist()

    assert anchored(None) == [False] * 5
    assert anchored("ends") == [True, False, True, True, True]
    assert anchored("start") == [True, False, False, True, False]
    assert anchored("end") == [False, False, True, False, True]
    tri = np.array([[0.2, 0.05, 0.05], [0.8, 0.05, 0.05], [0.5, 0.15, 0.15]])
    hoop = np.array([[0, 1], [1, 2], [2, 0]])
    with pytest.raises(ValueError, match="no free end"):
        BondedTruss(host, hconn, "hexa8", tri, hoop, steel, bond, 2e-4, 0.016, anchorage="ends")


def test_anchorage_without_a_bond_block_is_refused(tmp_path):
    with pytest.raises(ValueError, match="bonded assignment only"):
        prepare_case(str(_case(tmp_path, bond=False, anchorage="ends")))


def test_in_bending_an_anchored_bar_still_reproduces_perfect_bond(tmp_path):
    """The tension tests above passed with the interface force dropped from
    the free nodes of an anchored bar, because a uniformly strained host has
    no slip to speak of. Bending has: the first version of the tie gave a
    NEGATIVE support reaction on this beam. So the judge is a beam."""
    def case(bond, anchorage):
        assign = {"reinforcement": "bar", "section": "bar16", "name": "bar", "segment_length": 0.1}
        if bond:
            assign["bond"] = {"type": "bond_ceb", "fc": 44.0e6}
        if anchorage:
            assign["anchorage"] = anchorage
        d = {"name": "bend", "units": {"length": "m", "force": "N", "stress": "Pa"},
             "mesh": {"dimensions": [2.0, 0.2, 0.3], "divisions": [12, 2, 3]},
             "materials": {"c": {"type": "linear_elastic", "E": E_HOST, "nu": 0.2, "rho": 2400.0},
                           "s": {"type": "rebar", "E": E_BAR, "rho": 7850.0,
                                 "yield_strength": 5e8, "hardening": 2e9}},
             "sections": {"body": {"type": "solid", "material": "c", "formulation": "b-bar"},
                          "bar16": {"type": "bar", "material": "s", "shape": "circle",
                                    "diameter": 0.016, "area": A_BAR}},
             "reinforcement": {"bar": {"line": {"start": [0.0, 0.1, 0.05], "end": [2.0, 0.1, 0.05]}}},
             "assign": [assign],
             "regions": {"sup_L": {"where": {"x": 0.0, "z": 0}}, "sup_R": {"where": {"x": 2.0, "z": 0}},
                         "pin": {"where": {"x": 0.0, "y": 0.1, "z": 0}},
                         "pin_R": {"where": {"x": 2.0, "y": 0.1, "z": 0}},
                         "load": {"where": {"x": 1.0, "z": 0.3}}, "mid": {"where": {"x": 1.0, "z": 0}}},
             "constraints": [{"region": "sup_L", "dofs": [0, 0, 1]}, {"region": "sup_R", "dofs": [0, 0, 1]},
                             {"region": "pin", "dofs": [1, 1, 0]}, {"region": "pin_R", "dofs": [0, 1, 0]}],
             "steps": [{"name": "push", "constraints": [{"region": "load", "dofs": [0, 0, 1], "value": -0.002}],
                        "load_steps": 1, "max_steps_per_load": 20000, "tolerance": 1.0e-5}],
             "solver": {"backend": "numpy", "mode": "relaxation", "mass": "underwood",
                        "relaxation_safety": 0.3, "damping": {"type": "kinetic"}},
             "output": {"path": str(tmp_path / "b.h5"), "monitors": {
                 "R": {"kind": "reaction", "region": "sup_L", "dof": "z"},
                 "mid": {"kind": "displacement", "region": "mid", "dof": "z"}}}}
        path = tmp_path / f"{int(bond)}{anchorage}.yaml"
        path.write_text(yaml.safe_dump(d), encoding="utf-8")
        return _run(path)

    perfect = case(False, None)
    tied = case(True, "ends")
    assert tied.monitors["R"].values[-1] == pytest.approx(perfect.monitors["R"].values[-1], rel=2e-3)
    assert tied.monitors["mid"].values[-1] == pytest.approx(perfect.monitors["mid"].values[-1], rel=2e-3)
    assert perfect.monitors["R"].values[-1] > 0
