"""Member failure, fracture and particle splitting: the fourth review's
findings, one test each.

1. the fibre beam's clock restarted in every dynamic step
2. a member breaking at a driven support was not split
3. a fibre fragment was weighed and loaded through region "all" only
4. the split's stability floor was per dof, weighed, and cut v per dof
5. a split queued by the frame's own force evaluation was lost on restart
6. a restart after a split kept the setup's beta K
7. a removal's `split` was overwritten by the member's failure `split`
8. hinge rotation grew in `_record` and the convergence probes
9. a removal after the run's end was accepted silently
10. a fractured fibre member was reported failed again at its removal
"""
import pathlib
import tempfile

import h5py
import numpy as np
from vfife.io.writer import read_checkpoint  # the checkpoint is its own file
import pytest
import yaml

from vfife.api import prepare_case, run_case, _with_element
from vfife.core import added_mass
from vfife.core.dof import TRANSLATION
from vfife.core.structural import make_structural_solver

from test_member_failure import STEEL, _run, _write_msh
import test_particle_split as tps

G = 9.80665
BEAM = {"type": "beam", "material": "steel", "shape": "general",
        "area": 0.01, "iy": 1e-5, "iz": 1e-5, "j": 2e-5}
UNITS = {"length": "m", "force": "N", "stress": "Pa"}


def _solver_of(tmp_path, name, d):
    p = tmp_path / f"{name}.yaml"
    d = dict(d, name=name)
    d["output"] = dict(d.get("output", {}), path=str(tmp_path / f"{name}.h5"))
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    prep = prepare_case(str(p))
    return make_structural_solver(_with_element(prep.case, prep.element, prep.case_path),
                                  prep.mesh, prep.material)


def _events(h5):
    with h5py.File(h5) as f:
        d = f["diagnostics"]
        ev = np.asarray(d["member_failures"]) if "member_failures" in d else np.zeros((0, 4))
        sp = np.asarray(d["member_splits"]) if "member_splits" in d else np.zeros((0, 4))
    return ev, sp


# ---------------------------------------------------------------- 1
def test_a_fibre_removal_in_the_second_step_fires_at_its_run_time(tmp_path):
    """Two dynamic steps of T; a removal at 1.5 T. The fibre beam's clock
    was the step's (0 at every step start), so it never fired."""
    _write_msh(tmp_path / "f1.msh", [(0, 0, 0), (0.5, 0, 0), (1, 0, 0)],
               {"base": (0, [0]), "m0": (1, [(0, 1)]), "m1": (1, [(1, 2)])})
    T = 2e-3
    d = {"units": UNITS, "mesh": {"type": "gmsh", "path": "f1.msh"},
         "materials": {"steel": {"type": "rebar", "E": 2e11, "yield_strength": 4e8, "hardening": 2e9,
                                 "rho": 7850.0, "rupture_strain": 0.05}},
         "sections": {"rc": {"type": "fiber", "material": "steel", "shape": "rect",
                             "width": 0.05, "height": 0.05, "ny": 3, "nz": 3}},
         "assign": [{"region": "m0", "section": "rc"}, {"region": "m1", "section": "rc"}],
         "constraints": [{"region": "base", "dofs": [1, 1, 1, 1, 1, 1]}],
         "loads": [{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}],
         "removals": [{"region": "m0", "time": 1.5 * T}],
         "steps": [{"name": "a"}, {"name": "b"}],
         "solver": {"backend": "numpy", "mode": "dynamic", "total_time": T, "damping": {"type": "none"}},
         "output": {"fields": ["displacement", "member_failed"], "every": 1000}}
    out, h5 = _run(tmp_path, "f1", d)
    ev, _ = _events(h5)
    assert len(ev) == 1 and ev[0, 1] == 0 and ev[0, 3] == 1.0, ev
    assert 1.5 * T <= ev[0, 0] < 1.5 * T + 1e-5, "logged on the run's clock, at its time"


def test_a_fibre_fracture_in_the_second_step_is_logged_on_the_run_clock():
    with tempfile.TemporaryDirectory() as td:
        s = tps._solver(tps._falling_case(td))
        s.run(output_every=10 ** 9, progress_every=10 ** 9)
        (t_frac, e), = s.fracture.events
        # the push step lasts 0.01 s and the base member ruptures in the
        # second (it was logged as 0.00146 s)
        assert t_frac > s.total_time
        assert s.splits[0][0] >= t_frac
        assert s.breaking.failure_events[0][0] == t_frac


# ---------------------------------------------------------------- 2
def test_a_member_breaking_at_a_driven_support_leaves_it(tmp_path):
    """The tip is driven down by a prescribed displacement; member b breaks
    at that end. It kept the tip's particle (the split's `held` skipped a
    constraint with a value) and the reaction went on to 5.5x capacity."""
    xs = np.linspace(0, 2.0, 3)
    _write_msh(tmp_path / "f2.msh", [(x, 0, 0) for x in xs],
               {"root": (0, [0]), "tip": (0, [2]), "a": (1, [(0, 1)]), "b": (1, [(1, 2)])})
    d = {"units": UNITS, "mesh": {"type": "gmsh", "path": "f2.msh"},
         "materials": {"steel": STEEL}, "sections": {"s": BEAM},
         "assign": [{"region": "a", "section": "s", "orientation": [0, 0, 1],
                     "failure": {"yield_strength": 1e12, "extreme_fibre": [0.1, 0.1]}},
                    {"region": "b", "section": "s", "orientation": [0, 0, 1],
                     "failure": {"yield_strength": 3e8, "extreme_fibre": [0.1, 0.1]}}],
         "amplitudes": {"ramp": [[0, 0], [1, 1]]},
         "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]},
                         {"region": "tip", "dofs": [1, 1, 1, 1, 1, 1], "value": [0, 0, -0.05, 0, 0, 0],
                          "amplitude": "ramp"}],
         "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.1,
                    "damping": {"type": "viscous", "coeff": 200.0}},
         "output": {"fields": ["displacement"], "every": 400,
                    "monitors": {"R": {"kind": "reaction", "region": "tip", "dof": "z"}}}}
    out, h5 = _run(tmp_path, "f2", d)
    ev, sp = _events(h5)
    assert len(ev) and ev[0, 1] == 1
    assert len(sp) == 1 and sp[0, 1] == 1 and sp[0, 2] == 2, "b left the driven tip"
    with h5py.File(h5) as f:
        t = np.asarray(f["monitors/time"]); R = np.asarray(f["monitors/R"])
        el = np.asarray(read_checkpoint(str(h5))["elements"])
    assert el[1, 1] != 2
    m_cap = 3e8 * 1e-5 / 0.1
    assert np.all(np.abs(R[t > ev[0, 0] + 0.01]) < 1e-6 * m_cap), R


# ---------------------------------------------------------------- 3
def test_a_fibre_fragment_weighs_through_its_parent_and_keeps_off_nodal_forces():
    """Gravity on a NAMED region reaches the fragment's particle; a nodal
    force on "all" stays on the mesh's nodes. The fragment used to join
    region "all" and nothing else."""
    with tempfile.TemporaryDirectory() as td:
        yp = pathlib.Path(tps._falling_case(td))
        s_ = yp.read_text()
        s_ = s_.replace("regions:\n", "regions:\n  girder: {box: [-0.01, 1.01, -0.01, 0.01, -0.01, 0.01]}\n")
        s_ = s_.replace("loads: [{region: all, gravity: [0, 0, -9.81], amplitude: hold}]",
                        "loads: [{region: girder, gravity: [0, 0, -9.81], amplitude: hold}, "
                        "{region: all, vector: [0, 0, -90.0], amplitude: hold}]")
        yp.write_text(s_, encoding="utf-8")
        s = tps._solver(str(yp))
        s.run(output_every=10 ** 9, progress_every=10 ** 9)
        # both ends: the one at the clamped root leaves it too (1.0 re-review)
        news = [n for _t, _e, _p, n in s.splits]
        assert len(news) == 2
        _f, _m, _p, groups, ground = s.step_data(s.step_index)
        Fz = s.dofs.view(s._force(groups, ground, 1.0, 0.0))[:, 2]
        mz = s.physical_mass.reshape(s.nn, 6)[:, 2]
        for new in news:
            assert Fz[new] == pytest.approx(-9.81 * mz[new], rel=1e-12), "the fragment is weightless"
        spares = [n for n in range(3, s.nn) if n not in news]
        assert np.all(Fz[spares] == 0.0)
        # the nodal force: 30 N on each of the three mesh nodes, before and after
        assert Fz[:3] + 9.81 * mz[:3] == pytest.approx([-30.0] * 3, rel=1e-12)
        assert "girder" in s.mesh.regions and not set(news) & set(s.mesh.regions["all"].tolist())


# ---------------------------------------------------------------- 4
def test_the_stability_floor_is_one_weightless_mass_that_keeps_momentum(tmp_path):
    """A light member between two heavy ones split off: its particles need
    far more mass than its share at the run's step. The floor used to be
    per dof (36.6 kg in x, 7.9 kg in y and z), weighed, cut v per dof and
    was logged as the sum over x, y, z."""
    Lb = 0.25
    xs = [0, 1.0, 1.0 + Lb, 2.0 + Lb]
    _write_msh(tmp_path / "f4.msh", [(x, 0, 0) for x in xs],
               {"root": (0, [0]), "heavy": (1, [(0, 1), (2, 3)]), "light": (1, [(1, 2)])})
    heavy = dict(BEAM, area=0.02, iy=2e-5, iz=2e-5, j=4e-5)
    light = dict(BEAM, area=0.004, iy=4e-6, iz=4e-6, j=8e-6)
    d = {"units": UNITS, "mesh": {"type": "gmsh", "path": "f4.msh"},
         "materials": {"steel": STEEL}, "sections": {"h": heavy, "l": light},
         "assign": [{"region": "heavy", "section": "h", "orientation": [0, 0, 1]},
                    {"region": "light", "section": "l", "orientation": [0, 0, 1]}],
         "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]}],
         "loads": [{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}],
         "removals": [{"region": "light", "time": 0.001}],
         "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.003, "damping": {"type": "none"}},
         "output": {"fields": ["displacement"], "every": 10 ** 6}}
    s = _solver_of(tmp_path, "f4", d)
    logs = []
    s.log = logs.append
    rng = np.random.default_rng(3)
    s.v = rng.normal(size=s.ndof)
    s.v[s.dofs.node_dofs(np.arange(4, s.nn)).ravel()] = 0.0      # spares at rest
    pm = s.physical_mass.reshape(s.nn, 6)
    P0 = (pm[:, :3] * s.dofs.view(s.v)[:, :3]).sum(axis=0)
    v_parent = s.dofs.view(s.v)[[1, 2], :3].copy()
    s.breaking.mark([2], 0.0, 0.0, 1, split=True)
    s._split_off(np.array([2]))
    pm = s.physical_mass.reshape(s.nn, 6)
    share = s._element_mass_share(2)[0]
    new = [n for _t, _e, _p, n in s.splits]
    assert len(new) == 2
    for n in new:
        assert pm[n, 0] == pm[n, 1] == pm[n, 2] > 5 * share, "one translational mass"
    # momentum, and the fragment moves the way its parent did
    P1 = (pm[:, :3] * s.dofs.view(s.v)[:, :3]).sum(axis=0)
    assert P1 == pytest.approx(P0, rel=1e-12, abs=1e-12)
    for n, vp in zip(new, v_parent):
        vn = s.dofs.view(s.v)[n, :3]
        assert vn == pytest.approx(vp * share / pm[n, 0], rel=1e-12)
    # gravity sees the member's share, not the floor
    weighed = added_mass.weighed(s.dofs.field(s.physical_mass, TRANSLATION), s.weightless_mass)
    for n in new:
        assert weighed[n] == pytest.approx([share] * 3, rel=1e-12)
    added = sum(pm[n, 0] - share for n in new)
    line = next(x for x in logs if "kg added" in x)
    assert f"{added:.3g} kg added" in line, line


# ---------------------------------------------------------------- 5
def _chain(tmp_path, name, removal, damping=None, total=0.01, every=1):
    n = 4
    xs = np.linspace(0, 2.0, n + 1)
    _write_msh(tmp_path / f"{name}.msh", [(x, 0, 0) for x in xs],
               {"root": (0, [0]), "tip": (0, [n]), "cut": (1, [removal[0]]),
                "all_": (1, [(k, k + 1) for k in range(n)])})
    d = {"version": 1, "name": name, "units": UNITS, "mesh": {"type": "gmsh", "path": f"{name}.msh"},
         "materials": {"steel": STEEL}, "sections": {"b": BEAM},
         "assign": [{"region": "all", "section": "b", "orientation": [0, 0, 1]}],
         "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]}],
         "loads": [{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}],
         "removals": [{"region": "cut", "time": removal[1]}],
         "solver": {"backend": "numpy", "mode": "dynamic", "total_time": total,
                    "damping": damping or {"type": "none"}},
         "output": {"fields": ["displacement", "member_failed"], "every": every,
                    "path": str(tmp_path / f"{name}.h5")}}
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return str(p)


def _last(h5):
    with h5py.File(h5) as f:
        keys = sorted(f["steps"], key=lambda q: f["steps"][q].attrs["time"])
        return np.asarray(f["steps"][keys[-1]]["displacement"]), np.asarray(read_checkpoint(str(h5))["elements"])


def test_a_stop_at_the_step_of_a_removal_resumes_to_the_same_run(tmp_path):
    """Until v1.12 the break was found in the output frame's force
    evaluation, one step before its own, the split waited for the end of the
    next step, and a run stopped in between carried the queue in its
    checkpoint (`pending_split`). A frame only reads now: the step finds the
    removal and splits at its own end, nothing is ever left waiting at a
    stop -- and the stopped and resumed run is the uninterrupted one."""
    box = {}
    prep = prepare_case(_chain(tmp_path, "probe", ((3, 4), 1.0)))
    box["dt"] = make_structural_solver(_with_element(prep.case, prep.element, prep.case_path),
                                       prep.mesh, prep.material).dt
    k = 60
    tr = (k + 0.5) * box["dt"]
    run_case(_chain(tmp_path, "full", ((3, 4), tr)))
    for at in (k, k + 1, k + 2):
        stop = _chain(tmp_path, f"stop{at}", ((3, 4), tr))
        with pytest.raises(Exception):
            run_case(stop, on_progress=lambda st, *a, _at=at: st < _at, progress_every=1)
        ck = read_checkpoint(str(tmp_path / f"stop{at}.h5"))
        assert not len(np.atleast_1d(ck.get("pending_split", []))), at
        run_case(stop, restart=str(tmp_path / f"stop{at}.h5"), output=str(tmp_path / f"resumed{at}.h5"))
        u_full, el_full = _last(tmp_path / "full.h5")
        u_res, el_res = _last(tmp_path / f"resumed{at}.h5")
        assert el_res.tolist() == el_full.tolist() != [[3, 4], [0, 1], [1, 2], [2, 3]]
        assert np.array_equal(u_res, u_full), at


def test_191_a_removal_does_not_depend_on_where_the_frames_fall(tmp_path):
    """The state at the end, with a frame after every step, after every
    seventh, and only at the end: the same bits. With frames that broke
    members the removed member carried nothing in the step it should have
    carried its last force in, whenever a frame fell on the step before."""
    dt = make_structural_solver(*(lambda p: (_with_element(p.case, p.element, p.case_path), p.mesh, p.material))(
        prepare_case(_chain(tmp_path, "dt", ((3, 4), 1.0))))).dt
    tr = 60.5 * dt
    ends = []
    for every in (1, 7, 10 ** 6):
        run_case(_chain(tmp_path, f"ev{every}", ((3, 4), tr), every=every))
        ends.append(_last(tmp_path / f"ev{every}.h5"))
        ev, _sp = _events(tmp_path / f"ev{every}.h5")
        assert len(ev) == 1 and ev[0, 0] == pytest.approx(61 * dt, rel=1e-12)     # the first step at or after tr
    for u, el in ends[1:]:
        assert np.array_equal(u, ends[0][0]) and el.tolist() == ends[0][1].tolist()


# ---------------------------------------------------------------- 6
def test_a_restart_after_a_split_damps_on_the_split_connectivity(tmp_path):
    beta = {"type": "rayleigh", "alpha": 0.0, "beta": 2e-5}
    run_case(_chain(tmp_path, "bfull", ((1, 2), 0.002), beta, total=0.02, every=10))
    stop = _chain(tmp_path, "bstop", ((1, 2), 0.002), beta, total=0.02, every=10)
    with pytest.raises(Exception):
        run_case(stop, on_progress=lambda st, *a: st < 300, progress_every=10)
    run_case(stop, restart=str(tmp_path / "bstop.h5"), output=str(tmp_path / "bres.h5"))
    u_full, el = _last(tmp_path / "bfull.h5")
    u_res, _ = _last(tmp_path / "bres.h5")
    assert el[1].tolist() != [1, 2], "the member split before the stop"
    # 0.37 |u| before
    assert np.abs(u_res - u_full).max() <= 1e-12 * np.abs(u_full).max()


# ---------------------------------------------------------------- 7
@pytest.mark.parametrize("fsplit,rsplit", [(False, True), (True, False)])
def test_a_removal_splits_as_its_own_spec_says(tmp_path, fsplit, rsplit):
    n = 4
    xs = np.linspace(0, 2.0, n + 1)
    _write_msh(tmp_path / "f7.msh", [(x, 0, 0) for x in xs],
               {"root": (0, [0]), "cut": (1, [(2, 3)]), "all_": (1, [(k, k + 1) for k in range(n)])})
    d = {"units": UNITS, "mesh": {"type": "gmsh", "path": "f7.msh"},
         "materials": {"steel": STEEL}, "sections": {"b": BEAM},
         "assign": [{"region": "all", "section": "b", "orientation": [0, 0, 1],
                     "failure": {"yield_strength": 3e12, "extreme_fibre": [0.1, 0.1], "split": fsplit}}],
         "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]}],
         "loads": [{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}],
         "removals": [{"region": "cut", "time": 0.005, "split": rsplit}],
         "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.01, "damping": {"type": "none"}},
         "output": {"fields": ["displacement", "member_failed"], "every": 1000}}
    out, h5 = _run(tmp_path, f"f7_{fsplit}", d)
    ev, sp = _events(h5)
    with h5py.File(h5) as f:
        active = np.asarray(read_checkpoint(str(h5))["member_active"]).astype(bool)
    cut = int(ev[0, 1])
    assert ev[:, 3].tolist() == [1.0]
    if rsplit:
        assert active[cut] and len(sp) == 2 and set(sp[:, 1]) == {cut}
    else:
        assert not active[cut] and len(sp) == 0


# ---------------------------------------------------------------- 8
@pytest.mark.parametrize("backend", ["numpy"])
def test_the_hinge_rotation_does_not_depend_on_the_output_frequency(tmp_path, backend):
    """`_record` and the convergence measure evaluate the forces again; each
    evaluation added plastic rotation (4.96e-3 rad over 2022 frames, a
    tenth of the capacity)."""
    from test_plastic_hinge import FY, C, I, A, M_P, L
    rot = {}
    for every in (7, 10 ** 6):
        n = 4
        xs = np.linspace(0, L, n + 1)
        _write_msh(tmp_path / "f8.msh", [(x, 0, 0) for x in xs],
                   {"root": (0, [0]), "tip": (0, [n]), "first": (1, [(0, 1)]),
                    "rest": (1, [(k, k + 1) for k in range(1, n)])})
        beam = dict(BEAM, area=A, iy=I, iz=I, j=2 * I)
        fail = {"yield_strength": FY, "extreme_fibre": [C, C], "rotation_capacity": 0.05}
        d = {"units": UNITS, "mesh": {"type": "gmsh", "path": "f8.msh"},
             "materials": {"steel": STEEL}, "sections": {"b": beam},
             "assign": [{"region": "first", "section": "b", "orientation": [0, 0, 1], "failure": fail},
                        {"region": "rest", "section": "b", "orientation": [0, 0, 1]}],
             "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]}],
             "amplitudes": {"a": [[0, 0], [1, 1.5]]},
             "loads": [{"region": "tip", "vector": [0, 0, -M_P / L], "amplitude": "a"}],
             "solver": {"backend": backend, "mode": "dynamic", "total_time": 0.45,
                        "damping": {"type": "viscous", "coeff": 50.0}},
             "output": {"fields": ["displacement", "hinge_rotation"], "every": every}}
        s = _solver_of(tmp_path, f"f8_{every}", d)
        s.run(output_every=every, progress_every=13)
        rot[every] = s._hinge_rotation.copy()
    assert rot[7][0, 0] > 1e-3, "the root yielded"
    assert rot[7].tolist() == rot[10 ** 6].tolist()


# ---------------------------------------------------------------- 9
# at the end itself the last frame reaches it and it is logged (1.0 re-review):
# only a removal strictly after the end is warned of
@pytest.mark.parametrize("time,warned", [(0.0199, False), (0.02, False), (0.0201, True), (5.0, True)])
def test_a_removal_after_the_run_s_end_is_warned_of(tmp_path, time, warned):
    p = _chain(tmp_path, "f9", ((3, 4), time))
    d = yaml.safe_load(open(p, encoding="utf-8"))
    d["steps"] = [{"name": "a"}, {"name": "b"}]                 # two steps of 0.01 s
    pathlib.Path(p).write_text(yaml.safe_dump(d), encoding="utf-8")
    prep = prepare_case(p)
    codes = [i["code"] for i in prep.preflight["issues"] if i["severity"] == "WARNING"]
    assert ("REMOVAL_AFTER_END" in codes) == warned


# ---------------------------------------------------------------- 10
def test_a_fractured_fibre_member_is_not_failed_again_by_its_removal():
    with tempfile.TemporaryDirectory() as td:
        yp = pathlib.Path(tps._falling_case(td, push=0.3))
        s_ = yp.read_text()
        s_ = s_.replace("regions:\n", "regions:\n  first: {box: [0.49, 1.01, -0.01, 0.01, -0.01, 0.01]}\n")
        head, rest = s_.split("steps:\n", 1)
        push = rest.split("  - name: fall\n")[0]
        s_ = head + "removals: [{region: first, time: 0.008}]\nsteps:\n" + push + rest.split("  - name: fall\n")[1]
        yp.write_text(s_, encoding="utf-8")
        s = tps._solver(str(yp))
        s.run(output_every=10 ** 9, progress_every=10 ** 9)
        assert [e for _t, e in s.fracture.events] == [1]
        assert [int(ev[1]) for ev in s.breaking.failure_events] == [1]
        assert s.breaking.member_failed[1] and not s.breaking.member_active[1]


# =========================================================================== #
#  1.0 re-review: the review of the v1.0 fixes
# =========================================================================== #
def _shored(tmp_path, name, removal_time):
    """A cantilever chain whose tip is shored in step 1 only; the tip member
    is removed (split) at `removal_time`."""
    xs = np.linspace(0, 2.0, 5)
    _write_msh(tmp_path / f"{name}.msh", [(x, 0, 0) for x in xs],
               {"root": (0, [0]), "shore": (0, [4]), "cut": (1, [(3, 4)]),
                "all_": (1, [(k, k + 1) for k in range(4)])})
    d = {"units": UNITS, "mesh": {"type": "gmsh", "path": f"{name}.msh"},
         "materials": {"steel": STEEL}, "sections": {"b": BEAM},
         "assign": [{"region": "all", "section": "b", "orientation": [0, 0, 1]}],
         "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]}],
         "loads": [{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}],
         "steps": [{"name": "shored", "constraints": [{"region": "shore", "dofs": [1, 1, 1, 0, 0, 0]}]},
                   {"name": "released"}],
         "removals": [{"region": "cut", "time": removal_time, "split": True}],
         "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.02, "damping": {"type": "none"}},
         "output": {"fields": ["displacement"], "every": 10 ** 6}}
    s = _solver_of(tmp_path, name, d)
    s.log = lambda *a: None
    s.run(output_every=10 ** 6)
    return s


def test_147_a_support_released_in_an_earlier_step_is_free_when_its_member_splits(tmp_path):
    """Counted as a support in every step, a shore released in step 1 had
    the member split off it in step 2 and was left as a massless particle
    drifting at its last velocity, never parked."""
    s = _shored(tmp_path, "shore2", 0.03)
    cut = int(s.splits[0][1])
    assert 4 in s.mesh.elements[cut, :2].tolist(), "the free tip stays the fragment's end"
    assert s.physical_mass.reshape(s.nn, 6)[4, 0] > 1.0
    assert [p for _t, _e, p, _n in s.splits] == [3]
    # while the shore holds, the member leaves it (the v1.0 rule)
    s = _shored(tmp_path, "shore1", 0.005)
    assert sorted(p for _t, _e, p, _n in s.splits) == [3, 4]
    assert 4 not in s.mesh.elements[int(s.splits[0][1]), :2].tolist()


def test_147_a_removed_member_under_a_line_load_reports_no_end_actions(tmp_path):
    p = _chain(tmp_path, "dead", ((1, 2), 0.002))
    d = yaml.safe_load(open(p, encoding="utf-8"))
    d["removals"][0]["split"] = False
    d["loads"] = [{"region": "all", "line_load": [0, 0, -1000.0], "amplitude": "hold"}]
    d["solver"]["damping"] = {"type": "viscous", "coeff": 20.0}
    pathlib.Path(p).write_text(yaml.safe_dump(d), encoding="utf-8")
    prep = prepare_case(p)
    s = make_structural_solver(_with_element(prep.case, prep.element, prep.case_path),
                               prep.mesh, prep.material)
    s.log = lambda *a: None
    s.run(output_every=10 ** 6)
    f = s.nodal_fields(s._cell_actions())
    gone = np.flatnonzero(~s.breaking.member_active)
    assert len(gone) == 1
    names = ("axial_force", "shear_y", "shear_z", "torsion", "bending_y", "bending_z")
    for n in names:
        assert f[f"{n}_i"][gone].tolist() == f[f"{n}_j"][gone].tolist() == f[n][gone].tolist() == [0.0]
    live = np.flatnonzero(s.breaking.member_active)
    assert np.abs(f["bending_z_i"][live]).max() > 1.0         # the others still corrected


def test_147_a_removal_at_the_run_s_end_is_logged_and_not_warned_of(tmp_path):
    p = _chain(tmp_path, "atend", ((3, 4), 0.01))
    prep = prepare_case(p)
    assert "REMOVAL_AFTER_END" not in [i["code"] for i in prep.preflight["issues"]]
    run_case(p)
    ev, _sp = _events(tmp_path / "atend.h5")
    assert ev[:, 0].tolist() == [0.01]


def test_147_the_modal_tangent_does_not_advance_a_hinge(tmp_path):
    from test_plastic_hinge import FY, C, I, A, M_P, L
    from vfife import modal
    n = 4
    xs = np.linspace(0, L, n + 1)
    _write_msh(tmp_path / "mh.msh", [(x, 0, 0) for x in xs],
               {"root": (0, [0]), "tip": (0, [n]), "first": (1, [(0, 1)]),
                "rest": (1, [(k, k + 1) for k in range(1, n)])})
    beam = dict(BEAM, area=A, iy=I, iz=I, j=2 * I)
    fail = {"yield_strength": FY, "extreme_fibre": [C, C], "rotation_capacity": 0.05}
    d = {"units": UNITS, "mesh": {"type": "gmsh", "path": "mh.msh"},
         "materials": {"steel": STEEL}, "sections": {"b": beam},
         "assign": [{"region": "first", "section": "b", "orientation": [0, 0, 1], "failure": fail},
                    {"region": "rest", "section": "b", "orientation": [0, 0, 1]}],
         "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]}],
         "amplitudes": {"a": [[0, 0], [1, 1.2]]},
         "loads": [{"region": "tip", "vector": [0, 0, -M_P / L], "amplitude": "a"}],
         "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.3,
                    "damping": {"type": "viscous", "coeff": 50.0}},
         "output": {"fields": ["displacement"], "every": 10 ** 6}}
    s = _solver_of(tmp_path, "mh", d)
    s.log = lambda *a: None
    s.run(output_every=10 ** 6)
    before = (s._hinge_rotation.copy(), s._plastic_d.copy())
    assert before[0][0, 0] > 1e-3, "the root yielded"
    modal.analyse(s, n_modes=2)
    assert s._hinge_rotation.tolist() == before[0].tolist()
    assert s._plastic_d.tolist() == before[1].tolist()


def test_147_a_fibre_member_fracturing_at_its_clamped_root_leaves_it(tmp_path):
    _write_msh(tmp_path / "fr.msh", [(0, 0, 0), (0.5, 0, 0), (1.0, 0, 0)],
               {"root": (0, [0]), "tip": (0, [2]), "all_": (1, [(0, 1), (1, 2)])})
    d = {"units": UNITS, "mesh": {"type": "gmsh", "path": "fr.msh"},
         "materials": {"steel": {"type": "rebar", "E": 200.0e9, "yield_strength": 4.0e8,
                                 "hardening": 2.0e9, "rho": 7850.0, "rupture_strain": 3.0e-3}},
         "sections": {"s": {"type": "fiber", "material": "steel", "shape": "rect", "width": 0.05,
                            "height": 0.05, "ny": 3, "nz": 3, "fracture": {"threshold": 0.3, "split": True}}},
         "assign": [{"region": "all", "section": "s", "orientation": [0, 1, 0]}],
         "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]}],
         "amplitudes": {"r": [[0, 0], [1, 1]]},
         "loads": [{"region": "tip", "vector": [0, 0, -4.0e4], "amplitude": "r"}],
         "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.05,
                    "damping": {"type": "viscous", "coeff": 50.0}},
         "output": {"fields": ["displacement"], "every": 10 ** 6}}
    s = _solver_of(tmp_path, "fr", d)
    s.log = lambda *a: None
    s.run(output_every=10 ** 6)
    assert [e for _t, e in s.fracture.events] == [0]
    assert 0 not in s.mesh.elements[0, :2].tolist(), "the fragment left the clamped root"
    assert sorted(p for _t, _e, p, _n in s.splits) == [0, 1]
