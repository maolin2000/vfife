"""Debris that gained energy without bound.

A progressive-collapse run of a real steel truss threw a dozen
fragments off at 1e13 J after about a second on the ground. A generic
through truss of 60 m (`benchmarks/debris_truss.py`, own geometry) losing an
end post did the same, and showed three causes, each fixed here:

1. the rotary inertia (and its stability floor) is a diagonal fixed in
   global axes from the reference frame, and the step has no gyroscopic
   term: a tumbling fragment carried its stiff bending onto the axis whose
   small torsional inertia or floor was set for it (omega dt = 5.5 there) --
   on debris (particles joined to no support) the three are now made one,
   the largest, with the angular momentum kept; the standing structure
   keeps them per axis;
2. (folded into 1: a particle on one member is debris's commonest case);
3. a member twisted past half a turn end to end: the co-rotational frame
   cannot follow that, the torque reverses and a spinning fragment gains at
   every turn -- a breakable member twisted a quarter turn is now removed.
"""
import os
import sys

import h5py
import numpy as np
import pytest

from test_member_failure import STEEL, _write_msh
from vfife.api import run_case
from vfife.core import mass_scaling
from vfife.core.member_failure import TWIST_LIMIT

HERE = os.path.dirname(os.path.abspath(__file__))


def test_the_standing_structure_keeps_its_rotary_floor_per_axis():
    # the floor is per axis: made one on all three everywhere (the first
    # version of this fix), it halved a beam's torsional frequency in a
    # modal analysis (test_review4_seismic, review)
    ms = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    row = np.array([1e6, 1e6, 1e6, 1e4, 1e3, 1e3])
    out, n = mass_scaling.true_rotary(ms, np.full((1, 3), 1e-9), row, np.ones(6, bool))
    assert np.allclose(out[3:], np.array([1e4, 1e3, 1e3]) * 1e-6)
    assert n == 3


class _Dofs:
    stride = 6

    @staticmethod
    def node_dofs(n):
        return np.arange(6 * n, 6 * n + 6)


def test_on_debris_the_three_rotary_inertias_are_made_one_with_the_spin_kept():
    from types import SimpleNamespace
    from vfife.core.beam_solver import BeamSolver
    pm = np.array([[5.0, 5.0, 5.0, 1.0, 4.0, 2.0],      # on debris
                   [5.0, 5.0, 5.0, 1.0, 4.0, 2.0],      # standing
                   [5.0, 5.0, 5.0, 3.0, 3.0, 3.0]])     # on debris, already one
    v = np.zeros(18); v[3:6] = (2.0, 1.0, -1.0); v[9:12] = (2.0, 1.0, -1.0)
    booked = []
    s = SimpleNamespace(physical_mass=pm.reshape(-1).copy(), dofs=_Dofs(), v=v.copy(), nn=3,
                        _appendages=[], _parked={})
    ledger = SimpleNamespace(dissipate=booked.append)
    L0 = pm[0, 3:] * v[3:6]
    BeamSolver._debris_isotropic(s, np.array([False, True, False]), ledger)
    m = s.physical_mass.reshape(3, 6)
    assert np.allclose(m[0, 3:], 4.0) and np.allclose(m[0, :3], 5.0)
    assert np.allclose(m[1], pm[1]) and np.allclose(m[2], pm[2])
    assert np.allclose(m[0, 3:] * s.v[3:6], L0)                  # the angular momentum kept
    assert np.allclose(s.v[9:12], v[9:12])
    ke0 = 0.5 * np.sum(pm[0, 3:] * v[3:6] ** 2); ke1 = 0.5 * np.sum(m[0, 3:] * s.v[3:6] ** 2)
    assert len(booked) == 1 and booked[0] == pytest.approx(ke0 - ke1) and booked[0] > 0


def _twisted(tmp_path, name, compiled):
    """A cantilever of one breakable member twisted by a torque at its tip
    that grows past what turns it a quarter turn: it is removed, whole."""
    L, J = 2.0, 1e-6
    _write_msh(tmp_path / f"{name}.msh", [(0, 0, 0), (L, 0, 0)], {"root": (0, [0]), "tip": (0, [1]), "b": (1, [(0, 1)])})
    k_t = STEEL["E"] / (2 * (1 + STEEL["nu"])) * J / L          # G J / L
    T_quarter = TWIST_LIMIT * k_t
    d = {"name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": str(tmp_path / f"{name}.msh")},
         "materials": {"steel": STEEL},
         "sections": {"b": {"type": "beam", "material": "steel", "shape": "general",
                            "area": 1e-2, "iy": 1e-6, "iz": 1e-6, "j": J}},
         # strong enough never to break on N and M; only the twist removes it
         "assign": [{"region": "b", "section": "b", "orientation": [0, 0, 1],
                     "failure": {"yield_strength": 1e12, "extreme_fibre": [0.1, 0.1]}}],
         "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]}],
         "amplitudes": {"a": [[0, 0], [1.0, 2.0]]},
         "loads": [{"region": "tip", "moment": [T_quarter, 0, 0], "amplitude": "a"}],
         "solver": {"backend": "numba", "mode": "dynamic", "total_time": 2.0,
                    "damping": {"type": "viscous", "coeff": 2000.0}},
         "output": {"path": str(tmp_path / f"{name}.h5"), "every": 200, "fields": ["rotation", "member_failed"]}}
    old = os.environ.get("VFIFE_COMPILED_STEP")
    os.environ["VFIFE_COMPILED_STEP"] = "1" if compiled else "0"
    try:
        run_case(d)
    finally:
        os.environ.pop("VFIFE_COMPILED_STEP", None) if old is None else os.environ.update(VFIFE_COMPILED_STEP=old)
    with h5py.File(tmp_path / f"{name}.h5", "r") as f:
        return np.asarray(f["diagnostics/member_failures"])


@pytest.mark.parametrize("compiled", [False, True])
def test_a_member_twisted_a_quarter_turn_is_removed(tmp_path, compiled):
    pytest.importorskip("numba")
    ev = _twisted(tmp_path, f"tw{int(compiled)}", compiled)
    assert len(ev) == 1
    t, e, ratio, cause = ev[0]
    assert cause == 3 and e == 0
    # the torque ramps 0 -> 2 T_quarter over 2 s and the member follows it
    # quasi-statically: it goes at T = T_quarter, t = 1 s
    assert t == pytest.approx(1.0, abs=0.02)
    assert ratio == pytest.approx(1.0, abs=0.02)


def test_the_loop_and_the_compiled_step_remove_it_at_the_same_instant(tmp_path):
    pytest.importorskip("numba")
    a = _twisted(tmp_path, "twa", False)
    b = _twisted(tmp_path, "twb", True)
    assert np.array_equal(a, b)


@pytest.mark.slow
def test_a_collapsing_truss_keeps_its_energy_on_the_ground(tmp_path):
    # the generic truss of v1.17 losing an end post, 6 s: it fell, broke
    # into some 90 pieces and from about 4.5 s threw them off at 1e13 J
    sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))
    import debris_truss
    case = debris_truss.case(total_time=6.0, output=str(tmp_path / "debris.h5"))
    run_case(case)
    with h5py.File(tmp_path / "debris.h5", "r") as f:
        err = np.asarray(f["diagnostics/relative_energy_error"])
        ke = np.asarray(f["diagnostics/kinetic_energy"])
        st = f["steps"]; last = sorted(st, key=lambda k: float(st[k].attrs["time"]))[-1]
        u = np.asarray(st[last]["displacement"])
        failed = int(np.asarray(st[last]["member_failed"]).sum())
    assert failed > 60                                   # it did collapse
    assert np.nanmax(err) < 0.03
    assert ke.max() < 2e7                                # its weight times its fall, about 1.5e7 J
    assert np.abs(u).max() < 20.0                        # nothing thrown off


def test_the_loop_and_the_kernel_use_one_fragment_limit():
    pytest.importorskip("numba")
    from vfife.core.member_failure import FRAGMENT_LIMIT
    from vfife.kernels import numba_step
    assert numba_step.FRAGMENT_LIMIT == FRAGMENT_LIMIT


def _one_beam():
    from vfife.core.member_failure import Breaking
    b = Breaking(1, None, None, None, split_default=True)
    b.beams([0], {"yield_strength": 1.0, "extreme_fibre": [1.0, 1.0], "split": True}, 1.0, 1.0, 1.0)
    b.twist_caps(np.array([1.0]))
    return b


def _actions(n=0.0, m=0.0, t=0.0):
    a = np.zeros((1, 12))
    a[0, [0, 6]] = n; a[0, [4, 10]] = m; a[0, [3, 9]] = t
    return a


def test_a_member_that_parts_now_is_split_off_not_removed_in_the_same_breath():
    # found in review: the fragment rule read the ends this very call had
    # broken, and a hinge member parting at 2.5 times its capacity was removed
    # whole, its break written twice
    b = _one_beam()
    gone = b.check(0.1, _actions(n=2.5))
    assert list(gone) == [0]
    assert b.member_active[0] and b.end_broken[0].all()           # split off, still a piece
    assert len(b.failure_events) == 1
    # the next evaluation finds a free piece at 2.5 times its capacity: removed
    gone = b.check(0.2, _actions(n=2.5))
    assert list(gone) == [0] and not b.member_active[0]
    assert [ev[3] for ev in b.failure_events] == [0.0, 0.0]


def test_a_free_piece_below_twice_its_capacity_stays_a_piece():
    b = _one_beam()
    b.check(0.1, _actions(n=1.2))
    b.check(0.2, _actions(n=1.9))
    assert b.member_active[0] and len(b.failure_events) == 1


def test_the_twist_rule_writes_one_event_for_a_member():
    b = _one_beam()
    gone = b.check(0.1, _actions(t=TWIST_LIMIT * 1.0 * 1.01))      # G J / L = 1
    assert list(gone) == [0] and not b.member_active[0]
    assert len(b.failure_events) == 1 and b.failure_events[0][3] == 3.0


# ------------------------------------------------- the fibre beam's debris:
# a fracture that splits goes through the frame solvers'
# separation (core/beam_family) -- the ledger relabelled, debris given one
# rotary inertia, the mass never below zero

def test_a_fibre_split_books_no_work_over_the_relabelling(tmp_path):
    # the RC girder of test_compiled_breaking, 1.5 s: four splits, the halves
    # fall. The spare's displacement jumping from 0 to its parent's was
    # booked as work: 1.7 % of the energy at the end before v1.18 and v0.1.1 % now
    pytest.importorskip("numba")
    sys.path.insert(0, HERE)
    from test_compiled_breaking import _fibre_girder, _run
    d = _fibre_girder(tmp_path, "fsplit", total=1.5)
    out, s = _run(d, False)
    assert out.status == "COMPLETED_DYNAMIC"
    assert len(s.splits) >= 2
    with h5py.File(d["output"]["path"], "r") as f:
        err = np.asarray(f["diagnostics/relative_energy_error"])
    assert np.nanmax(err) < 0.005


def test_a_piece_a_fibre_fracture_set_loose_has_one_rotary_inertia(tmp_path):
    pytest.importorskip("numba")
    sys.path.insert(0, HERE)
    from test_compiled_breaking import _fibre_girder, _run
    d = _fibre_girder(tmp_path, "floose", total=1.5)
    _out, s = _run(d, False)
    pm = s.physical_mass.reshape(s.nn, 6)
    spread = np.ptp(pm[pm[:, 0] > 0, 3:], axis=1)
    # a rectangle along x: on the girder still standing the three inertias
    # differ (rho [Ip, Iy, Iz] L / 2); on the halves that fell they are one
    assert (spread == 0.0).any() and (spread > 0.0).any()
    assert len(s.splits) >= 2


def _girder_on_bearings(tmp_path, name):
    from test_compiled_breaking import _fibre_girder
    d = _fibre_girder(tmp_path, name, total=2.0)
    d["mesh"]["points"].update({"GA": [0, 0, -1.0], "GB": [8.0, 0, -1.0]})
    d["mesh"]["lines"]["ba"] = {"through": ["GA", "A"], "size": 2.0}
    d["mesh"]["lines"]["bb"] = {"through": ["B", "GB"], "size": 2.0}
    d["sections"]["slide"] = {"type": "connector", "stiffness": [5e9, 5e9, 5e9, 1e8, 1e8, 1e8]}
    d["assign"] += [{"region": "ba", "section": "slide", "orientation": [1, 0, 0]},
                    {"region": "bb", "section": "slide", "orientation": [1, 0, 0]}]
    d["constraints"] = [{"region": "GA", "dofs": [1] * 6}, {"region": "GB", "dofs": [1] * 6}]
    # a load on the bearing node: once it is parked it pushed it to 1e22 m/s
    # (the review of v1.18) -- a parked particle is held in place
    d["loads"].append({"region": "A", "vector": [0, 0, -1.0e3], "amplitude": "hold"})
    return d


def test_a_bearing_left_holding_nothing_up_is_removed_on_both_paths(tmp_path):
    # the review of v1.18: the girder member over a bearing
    # fractured, its particle kept the connector alone and no mass, and the
    # step bound fell to 5e-16 s -- non-finite at 0.86 s. The frame solvers'
    # rule now applies: the connector goes (cause 2) and the
    # particle left with nothing is parked
    pytest.importorskip("numba")
    sys.path.insert(0, HERE)
    from test_compiled_breaking import _run
    ends = {}
    for compiled in (False, True):
        d = _girder_on_bearings(tmp_path, f"brg{int(compiled)}")
        out, s = _run(d, compiled)
        assert out.status == "COMPLETED_DYNAMIC", out.message
        ev = np.asarray(s.breaking.failure_events)
        assert (ev[:, 3] == 2.0).any()                       # a bearing removed
        gone = ev[ev[:, 3] == 2.0, 1].astype(int)
        assert set(gone) <= {s.nel - 2, s.nel - 1}           # the two connectors, last
        with h5py.File(d["output"]["path"], "r") as f:
            assert np.nanmax(np.asarray(f["diagnostics/relative_energy_error"])) < 0.005
        assert np.abs(s.dofs.view(s.q)[0, :3]).max() < 1.0        # node A, parked
        ends[compiled] = s
    a, b = ends[False], ends[True]
    for k in ("q", "v", "physical_mass"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k


def test_a_fragment_end_in_a_staged_run_is_not_held_as_unbuilt(tmp_path):
    # the review of v1.18: the spares are particles no built member
    # reaches, so a staged run held them; a member's end that moved onto one
    # stayed clamped where it broke while the rest of the piece fell
    pytest.importorskip("numba")
    sys.path.insert(0, HERE)
    from test_compiled_breaking import _fibre_girder, _run
    d = _fibre_girder(tmp_path, "stg", total=2.0)
    d["mesh"]["points"].update({"X0": [0, 5, 0], "X1": [2, 5, 0]})
    d["mesh"]["lines"]["x"] = {"through": ["X0", "X1"], "size": 1.0}
    d["assign"].append({"region": "x", "section": "girder", "orientation": [0, 1, 0]})
    d["constraints"].append({"region": "X0", "dofs": [1] * 6})
    d["steps"] = [{"name": "one", "procedure": "dynamic", "total_time": 2.0, "activate": ["g"]},
                  {"name": "two", "procedure": "dynamic", "total_time": 0.2, "activate": ["x"]}]
    out, s = _run(d, False)
    assert out.status == "COMPLETED_DYNAMIC", out.message
    assert s.splits
    qv = s.dofs.view(s.q)
    for _t, _e, _p, n in s.splits:
        assert not s.staging.dormant[n]
        assert qv[n, 2] < -1.0                    # it fell with its piece
