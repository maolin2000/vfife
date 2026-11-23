"""Elastic-brittle breaking and scheduled removal of line members.

Each test is a closed form a broken member has to reproduce: the dynamic
overshoot of a suddenly loaded cable, the tip load at which a cantilever's
root reaches its moment capacity, free fall once the last member is gone.
"""
import h5py
import numpy as np
from vfife.io.writer import read_checkpoint  # the checkpoint is its own file
import pytest
import yaml

from vfife.api import prepare_case, run_case, _with_element
from vfife.core.structural import make_structural_solver

G = 9.80665


def _write_msh(path, nodes, groups):
    """groups: name -> (dim, [node] or [(a, b)])"""
    names = list(groups)
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(names))]
    out += [f'{groups[n][0]} {i + 1} "{n}"' for i, n in enumerate(names)]
    out += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    out += [f"{i + 1} {x:.12g} {y:.12g} {z:.12g}" for i, (x, y, z) in enumerate(nodes)]
    els = []
    for i, n in enumerate(names):
        dim, items = groups[n]
        for it in items:
            els.append((15, i + 1, [it + 1]) if dim == 0 else (1, i + 1, [it[0] + 1, it[1] + 1]))
    out += ["$EndNodes", "$Elements", str(len(els))]
    out += [f"{k + 1} {t} 2 {g} {g} " + " ".join(map(str, c)) for k, (t, g, c) in enumerate(els)]
    out += ["$EndElements", ""]
    path.write_text("\n".join(out), encoding="utf-8")


STEEL = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}


def _run(tmp_path, name, d):
    p = tmp_path / f"{name}.yaml"
    d = dict(d, name=name)
    d["output"] = dict(d.get("output", {}), path=str(tmp_path / f"{name}.h5"))
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return run_case(str(p)), tmp_path / f"{name}.h5"


# --------------------------------------------------------------------------- #
#  a weight on a cable, the cable a beam's end: T = m g statically, 2 m g
#  when the weight is dropped onto it
# --------------------------------------------------------------------------- #
def _hanging(tmp_path, strength_over_static, removal=None, total=0.25):
    # a stiff short beam (the support) and a 1 m cable below its tip carrying a
    # 1000 kg point mass
    _write_msh(tmp_path / "hang.msh",
               [(0, 0, 0), (1, 0, 0), (1, 0, -1)],
               {"root": (0, [0]), "tip": (0, [1]), "weight": (0, [2]),
                "arm": (1, [(0, 1)]), "cable": (1, [(1, 2)])})
    A = 1e-4
    T_static = 1000.0 * G
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": "hang.msh"},
        "materials": {"steel": STEEL},
        "sections": {"arm": {"type": "beam", "material": "steel", "shape": "general",
                             "area": 0.1, "iy": 1e-2, "iz": 1e-2, "j": 1e-2},
                     "cable": {"type": "bar", "material": "steel", "shape": "general", "area": A}},
        "assign": [{"region": "arm", "section": "arm", "orientation": [0, 0, 1]},
                   {"region": "cable", "section": "cable", "tension_only": True,
                    "failure": {"tensile_strength": strength_over_static * T_static / A}}],
        "nodal_masses": [{"region": "weight", "mass": 1000.0}],
        "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]},
                        {"region": "weight", "dofs": [1, 1, 0, 1, 1, 1]}],
        # the weight is DROPPED onto the cable: a step load, held
        "loads": [{"region": "weight", "vector": [0, 0, -T_static], "amplitude": "hold"}],
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": total,
                   "damping": {"type": "none"}},
        "output": {"fields": ["displacement"], "every": 50,
                   "monitors": {"w": {"kind": "displacement", "region": "weight", "dof": "z"}}},
    }
    if removal is not None:
        d["removals"] = [{"region": "cable", "time": removal}]
    return d


def test_a_suddenly_loaded_cable_breaks_on_the_overshoot_and_not_on_the_static_force(tmp_path):
    # 1.5 x static: the static force is safe, the dynamic peak (2 x) is not
    out, h5 = _run(tmp_path, "weak", _hanging(tmp_path, 1.5))
    with h5py.File(h5) as f:
        ev = np.asarray(f["diagnostics/member_failures"])
    assert ev.shape == (1, 4) and ev[0, 3] == 0.0          # one break, on capacity
    assert 1.0 <= ev[0, 2] < 1.05                          # caught at the crossing
    # after it the weight is in free fall: w(t) accelerates at g
    w = np.asarray(out.monitors["w"])
    assert w[-1] < -0.1
    # 2.5 x static: survives the doubled force, nothing breaks
    out2, h52 = _run(tmp_path, "strong", _hanging(tmp_path, 2.5))
    with h5py.File(h52) as f:
        assert "member_failures" not in f["diagnostics"]
    assert np.min(out2.monitors["w"]) > -0.01


def test_a_scheduled_removal_happens_at_its_time_and_the_weight_then_falls_freely(tmp_path):
    t_cut = 0.05
    out, h5 = _run(tmp_path, "cut", _hanging(tmp_path, 100.0, removal=t_cut, total=0.35))
    with h5py.File(h5) as f:
        ev = np.asarray(f["diagnostics/member_failures"])
        t = np.asarray(f["monitors/time"])
    assert ev.shape == (1, 4) and ev[0, 3] == 1.0
    assert ev[0, 0] == pytest.approx(t_cut, abs=2e-4)
    w = np.asarray(out.monitors["w"])
    w_cut = np.interp(t_cut, t, w)
    late = t > t_cut + 0.1
    # free fall from w_cut with the velocity it had there (small): a = -g
    fit = np.polyfit(t[late], w[late], 2)
    assert 2 * fit[0] == pytest.approx(-G, rel=0.02)
    assert w[-1] < w_cut - 0.3


# --------------------------------------------------------------------------- #
#  a cantilever with a slowly growing tip load: the root breaks at M_cap / L
# --------------------------------------------------------------------------- #
def test_a_cantilever_root_breaks_at_its_moment_capacity(tmp_path):
    L, n = 2.0, 4
    xs = np.linspace(0, L, n + 1)
    _write_msh(tmp_path / "cant.msh", [(x, 0, 0) for x in xs],
               {"root": (0, [0]), "tip": (0, [n]),
                "first": (1, [(0, 1)]), "rest": (1, [(k, k + 1) for k in range(1, n)])})
    fy, c, I, A = 3.0e8, 0.1, 1e-5, 1e-2
    beam = {"type": "beam", "material": "steel", "shape": "general",
            "area": A, "iy": I, "iz": I, "j": 2 * I}
    fail = {"yield_strength": fy, "extreme_fibre": [c, c]}
    M_cap = fy * I / c                       # 30 kN m
    P_break = M_cap / L                      # 15 kN
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": "cant.msh"},
        "materials": {"steel": STEEL},
        "sections": {"b": beam},
        "assign": [{"region": "first", "section": "b", "orientation": [0, 0, 1], "failure": fail},
                   {"region": "rest", "section": "b", "orientation": [0, 0, 1]}],
        "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]}],
        "amplitudes": {"grow": [[0.0, 0.0], [1.0, 2.0]]},
        # 0 -> 2 P_break over 4 s, heavily damped: quasi-static
        "loads": [{"region": "tip", "vector": [0, 0, -P_break], "amplitude": "grow"}],
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 4.0,
                   "damping": {"type": "viscous", "coeff": 50.0}},
        "output": {"fields": ["displacement"], "every": 200,
                   "monitors": {"w": {"kind": "displacement", "region": "tip", "dof": "z"}}},
    }
    out, h5 = _run(tmp_path, "cant", d)
    with h5py.File(h5) as f:
        ev = np.asarray(f["diagnostics/member_failures"])
    assert ev.shape[0] >= 1 and ev[0, 1] == 0                # element 0 is the root member
    P_at_break = 2 * P_break * ev[0, 0] / 4.0
    assert P_at_break == pytest.approx(P_break, rel=0.03)


# --------------------------------------------------------------------------- #
#  refusals, and a broken member stays broken across a checkpoint
# --------------------------------------------------------------------------- #
def test_breaking_is_refused_where_it_would_be_ignored(tmp_path):
    base = _hanging(tmp_path, 1.5)
    relax = dict(base, solver={"backend": "numpy", "mode": "relaxation", "tolerance": 1e-6,
                               "n_steps": 1000})
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(dict(relax, name="r", output={"path": str(tmp_path / "r.h5")})), encoding="utf-8")
    with pytest.raises(Exception, match="mode = dynamic"):
        prepare_case(str(p))
    bad = _hanging(tmp_path, 1.5)
    bad["assign"][1]["failure"] = {"yield_strength": 3e8, "extreme_fibre": [0.1, 0.1]}
    p.write_text(yaml.safe_dump(dict(bad, name="b", output={"path": str(tmp_path / "b.h5")})), encoding="utf-8")
    with pytest.raises(Exception, match="tensile_strength"):
        prepare_case(str(p))


def test_the_broken_state_is_in_the_checkpoint(tmp_path):
    # the default: the snapped cable SPLITS OFF -- it stays active
    # between its own particles: its tip end moved onto a spare because the
    # arm is still on that particle, its weight end because the weight's
    # particle is constrained (a support's particle is never the fragment's)
    out, h5 = _run(tmp_path, "weak", _hanging(tmp_path, 1.5))
    with h5py.File(h5) as f:
        active = np.asarray(read_checkpoint(str(h5))["member_active"]).astype(bool)
        failed = np.asarray(read_checkpoint(str(h5))["member_failed"]).astype(bool)
        conn = np.asarray(read_checkpoint(str(h5))["elements"])
        nn = len(f["mesh/nodes"])
    assert active.tolist() == [True, True] and failed.tolist() == [False, True]
    assert nn == 5 and conn[1].tolist() == [3, 4]          # 3 nodes + 2 spares, both ends on theirs
    p = tmp_path / "weak.yaml"
    prep = prepare_case(str(p))
    s = make_structural_solver(_with_element(prep.case, prep.element, prep.case_path),
                               prep.mesh, prep.material)
    s.restore(read_checkpoint(str(h5)))
    assert s.member_active.tolist() == [True, True] and s.breaking.member_failed.tolist() == [False, True]
    assert s.mesh.elements[1].tolist() == [3, 4] and s._truss_batch().tolist() == [1]
    # split: false is v0.4's rule: deactivated, mass left on the nodes
    d = _hanging(tmp_path, 1.5)
    d["assign"][1]["failure"]["split"] = False
    out2, h52 = _run(tmp_path, "weak_off", d)
    with h5py.File(h52) as f:
        assert np.asarray(read_checkpoint(str(h52))["member_active"]).astype(bool).tolist() == [True, False]
        assert len(f["mesh/nodes"]) == 3 and "elements" not in read_checkpoint(str(h52))
    prep = prepare_case(str(tmp_path / "weak_off.yaml"))
    s = make_structural_solver(_with_element(prep.case, prep.element, prep.case_path),
                               prep.mesh, prep.material)
    s.restore(read_checkpoint(str(h52)))
    assert s._truss_batch().tolist() == []                 # nothing left to carry


def test_a_split_member_takes_its_mass_and_its_weight_with_it(tmp_path):
    """A beam hanging from a cable at one end, pinned at the other: cut the
    cable and the beam swings; cut the BEAM at its pinned end instead and the
    whole beam is a free body -- with its own mass, under its own weight,
    falling at g -- and the pin's node keeps only what was never the beam's."""
    L = 2.0
    _write_msh(tmp_path / "hang2.msh",
               [(0, 0, 0), (L, 0, 0), (L, 0, 5.0)],
               {"pin": (0, [0]), "tip": (0, [1]), "top": (0, [2]),
                "beam": (1, [(0, 1)]), "cable": (1, [(1, 2)])})
    A_b = 0.01
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": "hang2.msh"},
        "materials": {"steel": STEEL},
        "sections": {"b": {"type": "beam", "material": "steel", "shape": "general",
                           "area": A_b, "iy": 1e-5, "iz": 1e-5, "j": 2e-5},
                     "c": {"type": "bar", "material": "steel", "shape": "general", "area": 1e-4}},
        "assign": [{"region": "beam", "section": "b", "orientation": [0, 0, 1]},
                   {"region": "cable", "section": "c", "tension_only": True}],
        "constraints": [{"region": "pin", "dofs": [1, 1, 1, 1, 0, 1]},
                        {"region": "top", "dofs": [1, 1, 1, 1, 1, 1]}],
        "loads": [{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}],
        "removals": [{"region": "beam", "time": 0.05}, {"region": "cable", "time": 0.05}],
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.4,
                   "damping": {"type": "none"}},
        "output": {"fields": ["displacement", "member_failed"], "every": 50,
                   "monitors": {"w": {"kind": "displacement", "region": "tip", "dof": "z"}}},
    }
    out, h5 = _run(tmp_path, "split", d)
    with h5py.File(h5) as f:
        ev = np.asarray(f["diagnostics/member_failures"])
        sp = np.asarray(f["diagnostics/member_splits"])
        keys = sorted(f["steps"])
        tf = np.array([f["steps"][k].attrs["time"] for k in keys])
        U = np.array([np.asarray(f["steps"][k]["displacement"]) for k in keys])
        conn = np.asarray(f["steps"][keys[-1]]["connectivity"]).astype(int)
        nn = len(f["mesh/nodes"])
    assert ev.shape[0] == 2 and nn == 3 + 4                  # both members reserved two spares
    # the beam left the pin and the cable the top -- a constrained particle
    # is duplicated even when the member was its only bond -- and the beam
    # left the tip's node, which the cable, alone on it by then, kept: three
    # ends moved, two free bodies sharing no particle
    beam, cable = conn[0], conn[1]
    assert beam.min() >= 3 and cable[1] >= 3 and not set(beam) & set(cable)
    assert sp.shape[0] == 3
    # the beam is a free body under gravity: both its particles fall at g --
    # at g m / (m + dm), where dm is the split's stability floor, which has
    # inertia and no weight and is the same on x, y and z (v1.0:
    # 1.5 kg on 78.5 kg here; per dof it had gone on x only)
    with h5py.File(h5) as f:
        pm = np.asarray(read_checkpoint(str(h5))["physical_mass"]).reshape(-1, 6)
        w = np.asarray(read_checkpoint(str(h5))["weightless_mass"])
    late = tf > 0.05 + 0.05
    for n in beam:
        g_eff = G * (pm[n, 2] - w[n, 2]) / pm[n, 2]
        assert g_eff > 0.97 * G
        fit = np.polyfit(tf[late], U[late, n, 2], 2)
        assert 2 * fit[0] == pytest.approx(-g_eff, rel=0.02)
    assert np.abs(U[-1, 0]).max() < 1e-9                     # the pin's node stayed put
    assert U[-1, beam[1], 2] < -0.3                          # and the beam is gone from it
    # the energy ledger closes across the split: moving an end onto its
    # spare is a relabelling, not a displacement the forces did work over
    with h5py.File(h5) as f:
        err = np.asarray(f["diagnostics/relative_energy_error"])
    assert err.max() < 1e-2


# --------------------------------------------------------------------------- #
#  the same closed forms on the single-section solvers:
#  beam2 on both force paths, the standalone truss, a fibre member removed
# --------------------------------------------------------------------------- #
def _numba():
    try:
        from vfife.kernels import numba_frame
        return numba_frame.available()
    except Exception:
        return False


@pytest.mark.parametrize("backend", ["numpy", pytest.param(
    "numba", marks=pytest.mark.skipif(not _numba(), reason="numba not installed"))])
def test_a_single_section_beam2_cantilever_breaks_at_its_root(tmp_path, backend):
    # one section on the whole mesh: `element.failure`, every member can break,
    # and the root (the largest moment) is the one that does, at M_cap / L
    L, n = 2.0, 4
    xs = np.linspace(0, L, n + 1)
    _write_msh(tmp_path / "cant1.msh", [(x, 0, 0) for x in xs],
               {"root": (0, [0]), "tip": (0, [n]), "all_": (1, [(k, k + 1) for k in range(n)])})
    fy, c, I, A = 3.0e8, 0.1, 1e-5, 1e-2
    M_cap = fy * I / c; P_break = M_cap / L
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": "cant1.msh"},
        "materials": {"steel": STEEL},
        "sections": {"b": {"type": "beam", "material": "steel", "shape": "general",
                           "area": A, "iy": I, "iz": I, "j": 2 * I}},
        "assign": [{"region": "all", "section": "b", "orientation": [0, 0, 1],
                    "failure": {"yield_strength": fy, "extreme_fibre": [c, c]}}],
        "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]}],
        "amplitudes": {"grow": [[0.0, 0.0], [1.0, 2.0]]},
        "loads": [{"region": "tip", "vector": [0, 0, -P_break], "amplitude": "grow"}],
        "solver": {"backend": backend, "mode": "dynamic", "total_time": 4.0,
                   "damping": {"type": "viscous", "coeff": 50.0}},
        "output": {"fields": ["displacement", "capacity_ratio"], "every": 200,
                   "monitors": {"w": {"kind": "displacement", "region": "tip", "dof": "z"}}},
    }
    out, h5 = _run(tmp_path, "cant1_" + backend, d)
    with h5py.File(h5) as f:
        ev = np.asarray(f["diagnostics/member_failures"])
        active = np.asarray(read_checkpoint(str(h5))["member_active"]).astype(bool)
        failed = np.asarray(read_checkpoint(str(h5))["member_failed"]).astype(bool)
    assert ev[0, 1] == 0                                     # the root member
    assert 2 * P_break * ev[0, 0] / 4.0 == pytest.approx(P_break, rel=0.03)
    assert active[0] and failed[0]                           # split off: broken, still a beam of its own
    # it broke AT THE ROOT: its root end left the support first.
    # The release of M_cap at once sends a bending wave down the free
    # cantilever, and 0.2 ms later the same member's other end (at 0.75 M_cap
    # before) goes too: the elastic-brittle zipper, which is why a brittle
    # criterion over-fragments a collapse
    with h5py.File(h5) as f:
        broken_ends = np.asarray(read_checkpoint(str(h5))["end_broken"]).astype(bool)
    assert ev[0, 0] < ev[-1, 0] and broken_ends[0, 0]
    # once the root is gone the rest is a free body under the tip load and the
    # tip runs away: the member carries nothing
    w = np.asarray(out.monitors["w"])
    assert w[-1] < -1.0


def test_a_beam_parts_at_the_end_that_broke_and_stays_welded_at_the_other(tmp_path):
    """v0.5: a fracture is at a SECTION. A two-member cantilever whose
    root member alone can break: it leaves the support at its root end and
    stays welded to the tip member at the other. (Before, the whole member
    was cut out at both ends at once, and a girder unzipped into single
    elements.) The brittle release then overloads that other end within a
    millisecond -- a free end with a lumped mass swinging on it -- and it may
    go too: that is the elastic-brittle criterion, recorded as a second event,
    not the split taking both ends at once."""
    L, n = 2.0, 2
    xs = np.linspace(0, L, n + 1)
    _write_msh(tmp_path / "c2.msh", [(x, 0, 0) for x in xs],
               {"root": (0, [0]), "tip": (0, [n]), "m0": (1, [(0, 1)]), "m1": (1, [(1, 2)])})
    fy, c, I, A = 3.0e8, 0.1, 1e-5, 1e-2
    M_cap = fy * I / c; P = 1.2 * M_cap / L
    sec = {"type": "beam", "material": "steel", "shape": "general", "area": A, "iy": I, "iz": I, "j": 2 * I}
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": "c2.msh"},
        "materials": {"steel": STEEL},
        "sections": {"b": sec},
        "assign": [{"region": "m0", "section": "b", "orientation": [0, 0, 1],
                    "failure": {"yield_strength": fy, "extreme_fibre": [c, c]}},
                   {"region": "m1", "section": "b", "orientation": [0, 0, 1]}],
        "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]}],
        "amplitudes": {"grow": [[0.0, 0.0], [1.0, 1.0]]},
        "loads": [{"region": "tip", "vector": [0, 0, -P], "amplitude": "grow"}],
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 1.2,
                   "damping": {"type": "viscous", "coeff": 50.0}},
        "output": {"fields": ["displacement"], "every": 200, "monitors": {}},
    }
    out, h5 = _run(tmp_path, "c2", d)
    with h5py.File(h5) as f:
        ev = np.asarray(f["diagnostics/member_failures"])
        conn = np.asarray(read_checkpoint(str(h5))["elements"]).astype(int)
        broken_ends = np.asarray(read_checkpoint(str(h5))["end_broken"]).astype(bool)
        splits = np.asarray(f["diagnostics/member_splits"])
    assert int(ev[0, 1]) == 0 and broken_ends[0, 0]
    first = splits[splits[:, 0] == splits[0, 0]]
    assert len(first) == 1 and int(first[0, 2]) == 0     # the first split moved the ROOT end only
    assert conn[0, 0] != 0                               # off the support


def test_a_standalone_truss_bar_breaks_on_the_overshoot(tmp_path):
    # the hanging weight of the first test as a truss2-only model: T = m g
    # statically, 2 m g when dropped; strength 1.5 m g breaks on the overshoot
    _write_msh(tmp_path / "bar.msh", [(0, 0, 0), (0, 0, -1)],
               {"top": (0, [0]), "weight": (0, [1]), "bar": (1, [(0, 1)])})
    A = 1e-4; T_static = 1000.0 * G
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": "bar.msh"},
        "materials": {"steel": STEEL},
        "sections": {"bar": {"type": "bar", "material": "steel", "shape": "general", "area": A}},
        "assign": [{"region": "all", "section": "bar", "tension_only": True,
                    "failure": {"tensile_strength": 1.5 * T_static / A}}],
        "nodal_masses": [{"region": "weight", "mass": 1000.0}],
        "constraints": [{"region": "top", "dofs": [1, 1, 1]}, {"region": "weight", "dofs": [1, 1, 0]}],
        "loads": [{"region": "weight", "vector": [0, 0, -T_static], "amplitude": "hold"}],
        # the weight's own step (v1.0: the truss step now sees the lumped
        # mass; the bar's wave speed alone gave a step 100 times finer, which
        # is what resolved the break to 5% before)
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.25,
                   "dt_safety": 0.02, "damping": {"type": "none"}},
        "output": {"fields": ["displacement", "member_failed"], "every": 50,
                   "monitors": {"w": {"kind": "displacement", "region": "weight", "dof": "z"}}},
    }
    out, h5 = _run(tmp_path, "bar", d)
    with h5py.File(h5) as f:
        ev = np.asarray(f["diagnostics/member_failures"])
        active = np.asarray(read_checkpoint(str(h5))["member_active"]).astype(bool)
    assert ev.shape == (1, 4) and ev[0, 3] == 0.0 and 1.0 <= ev[0, 2] < 1.05
    assert active.tolist() == [False]
    assert np.asarray(out.monitors["w"])[-1] < -0.1          # in free fall since
    # and the strong bar survives the doubled force
    d["assign"][0]["failure"]["tensile_strength"] = 2.5 * T_static / A
    out2, h52 = _run(tmp_path, "bar_strong", d)
    with h5py.File(h52) as f:
        assert "member_failures" not in f["diagnostics"]
    assert np.min(out2.monitors["w"]) > -0.01


def test_a_fibre_member_is_removed_at_its_time_and_the_rest_falls_freely(tmp_path):
    # a fibre cantilever whose root member is cut: the rest is a free body
    # under gravity and its tip falls at g
    L, n = 2.0, 4
    xs = np.linspace(0, L, n + 1)
    _write_msh(tmp_path / "fib.msh", [(x, 0, 0) for x in xs],
               {"root": (0, [0]), "tip": (0, [n]), "first": (1, [(0, 1)]),
                "all_": (1, [(k, k + 1) for k in range(n)])})
    t_cut = 0.05
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": "fib.msh"},
        "materials": {"steel": {"type": "steel_cyclic", "grade": "Q235B", "E": 2.06e11, "rho": 7850.0}},
        "sections": {"tube": {"type": "fiber", "material": "steel", "shape": "tube",
                              "diameter": 0.2, "t": 0.01, "nt": 8}},
        "assign": [{"region": "all", "section": "tube", "orientation": [0, 0, 1]}],
        "constraints": [{"region": "root", "dofs": [1, 1, 1, 1, 1, 1]}],
        "loads": [{"region": "all_", "gravity": [0, 0, -G], "amplitude": "hold"}],
        "removals": [{"region": "first", "time": t_cut}],
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.35,
                   "damping": {"type": "none"}},
        "output": {"fields": ["displacement", "member_failed"], "every": 100,
                   "monitors": {"w": {"kind": "displacement", "region": "tip", "dof": "z"}}},
    }
    out, h5 = _run(tmp_path, "fib", d)
    with h5py.File(h5) as f:
        ev = np.asarray(f["diagnostics/member_failures"])
        t = np.asarray(f["monitors/time"])
        active = np.asarray(read_checkpoint(str(h5))["member_active"]).astype(bool)
    assert ev.shape == (1, 4) and ev[0, 3] == 1.0 and ev[0, 1] == 0
    assert ev[0, 0] == pytest.approx(t_cut, abs=2e-3)
    assert active.tolist() == [False, True, True, True]
    w = np.asarray(out.monitors["w"])
    late = t > t_cut + 0.1
    fit = np.polyfit(t[late], w[late], 2)
    assert 2 * fit[0] == pytest.approx(-G, rel=0.03)


def test_a_massless_link_left_connecting_nothing_is_removed(tmp_path):
    """Xihoumen's crossbeams carry no mass of their own (rho = 1, their
    weight is in the girders'). When both girder segments at a deck node
    split off, the crossbeam there ties nothing: left in place it was a
    spring with an artificial mass on its end, whipping round the sea floor
    at 350 m/s. It is removed (cause 2) and nothing runs away."""
    _write_msh(tmp_path / "t.msh", [(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0), (1, 2, 0)],
               {"base": (0, [4]), "girders": (1, [(0, 1), (1, 2)]), "gnodes": (0, [0, 1, 2]),
                "link": (1, [(1, 3)]), "post": (1, [(3, 4)])})
    light = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 1.0}
    beam = {"type": "beam", "shape": "general", "area": 1e-2, "iy": 1e-5, "iz": 1e-5, "j": 2e-5}
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": "t.msh"},
        "materials": {"steel": STEEL, "light": light},
        "sections": {"g": dict(beam, material="steel"), "l": dict(beam, material="light"),
                     "p": dict(beam, material="steel")},
        "assign": [{"region": "girders", "section": "g", "orientation": [0, 0, 1]},
                   {"region": "link", "section": "l", "orientation": [0, 0, 1]},
                   {"region": "post", "section": "p", "orientation": [0, 0, 1]}],
        "constraints": [{"region": "base", "dofs": [1, 1, 1, 1, 1, 1]}],
        "loads": [{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}],
        "removals": [{"region": "gnodes", "time": 0.02}],
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.1, "damping": {"type": "none"},
                   "mass_scaling": {"target_dt": 2.0e-5, "max_added_fraction": 0.1}},
        "output": {"fields": ["displacement"], "every": 100, "monitors": {}},
    }
    out, h5 = _run(tmp_path, "orphan", d)
    with h5py.File(h5) as f:
        ev = np.asarray(f["diagnostics/member_failures"])
        err = np.asarray(f["diagnostics/relative_energy_error"])
        u = np.asarray(f["steps"][sorted(f["steps"])[-1]]["displacement"])
    causes = {int(e): int(c) for _t, e, _r, c in ev}
    # the link's other end is on the post, which is massive: an outrigger
    # still attached to something massive stays -- as an APPENDAGE carried
    # rigidly by the post's top, its free end exactly 1 m from it
    assert causes[0] == 1 and causes[1] == 1 and 2 not in causes
    with h5py.File(h5) as f:
        X = np.asarray(f["mesh/nodes"])
        conn = np.asarray(f["steps"][sorted(f["steps"])[-1]]["connectivity"]).astype(int)
    link = conn[2]
    xa, xb = X[link[0]] + u[link[0]], X[link[1]] + u[link[1]]
    assert np.linalg.norm(xa - xb) == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(u).all() and np.abs(u).max() < 1.0
    assert err.max() < 1e-2


def test_a_cluster_of_massless_links_attached_to_nothing_massive_is_removed(tmp_path):
    """Two massless links in a row between two girder ends: when both
    girders split off the links hang from nothing massive anywhere, and the
    whole cluster goes (cause 2)."""
    _write_msh(tmp_path / "c.msh", [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0)],
               {"base": (0, [0]), "tip": (0, [5]), "g": (1, [(1, 2), (3, 4)]), "gnodes": (0, [1, 2, 3, 4]),
                "links": (1, [(2, 3)]), "posts": (1, [(0, 1), (4, 5)])})
    light = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 1.0}
    beam = {"type": "beam", "shape": "general", "area": 1e-2, "iy": 1e-5, "iz": 1e-5, "j": 2e-5}
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": "c.msh"},
        "materials": {"steel": STEEL, "light": light},
        "sections": {"g": dict(beam, material="steel"), "l": dict(beam, material="light"), "p": dict(beam, material="light")},
        "assign": [{"region": "g", "section": "g", "orientation": [0, 0, 1]},
                   {"region": "links", "section": "l", "orientation": [0, 0, 1]},
                   {"region": "posts", "section": "p", "orientation": [0, 0, 1]}],
        "constraints": [{"region": "base", "dofs": [1, 1, 1, 1, 1, 1]}],
        "loads": [{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}],
        "removals": [{"region": "g", "time": 0.02}],
        # the light members' particles take mass scaling up to the girders'
        # step, as Xihoumen's do: that is what makes them devices
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.05, "damping": {"type": "none"},
                   "mass_scaling": {"target_dt": 2.0e-5, "max_added_fraction": 0.1}},
        "output": {"fields": ["displacement"], "every": 100, "monitors": {}},
    }
    out, h5 = _run(tmp_path, "cluster", d)
    with h5py.File(h5) as f:
        ev = np.asarray(f["diagnostics/member_failures"])
    causes = {int(e): int(c) for _t, e, _r, c in ev}
    # elements: 0 posts(0-1), ... find by region order in the mesh: girders removed (1),
    # the middle link and the far post (attached to nothing massive, not held) go (2);
    # the near post is held at the base and stays
    assert sorted(c for c in causes.values()).count(1) == 2
    assert sorted(c for c in causes.values()).count(2) == 2


def test_a_massless_link_on_a_piece_of_debris_is_removed_and_its_joint_stops(tmp_path):
    """An outrigger rides its girder only while the girder is part of the
    standing structure. On a piece of debris the anchor is the end of a free
    segment whose roll has none of the outrigger's rotary inertia (its mass
    is in the girder's density), and Xihoumen's k = 3 run swung 9 m arms
    round at 150 m/s on such ends: there the link is removed (cause 2) and
    its free joint -- no body, only the stability floor -- is parked where
    it was, instead of flying off with the last velocity it had."""
    _write_msh(tmp_path / "d.msh", [(0, 0, 0), (0, 0, -1), (1, 0, -1), (1, 1, -1)],
               {"base": (0, [0]), "hanger": (1, [(0, 1)]), "girder": (1, [(1, 2)]), "link": (1, [(2, 3)])})
    light = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 1.0}
    beam = {"type": "beam", "shape": "general", "area": 1e-2, "iy": 1e-5, "iz": 1e-5, "j": 2e-5}
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": "d.msh"},
        "materials": {"steel": STEEL, "light": light},
        "sections": {"g": dict(beam, material="steel"), "l": dict(beam, material="light")},
        "assign": [{"region": "hanger", "section": "g", "orientation": [1, 0, 0]},
                   {"region": "girder", "section": "g", "orientation": [0, 0, 1]},
                   {"region": "link", "section": "l", "orientation": [0, 0, 1]}],
        "constraints": [{"region": "base", "dofs": [1, 1, 1, 1, 1, 1]}],
        "loads": [{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}],
        "removals": [{"region": "hanger", "time": 0.02}],
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.1, "damping": {"type": "none"},
                   "mass_scaling": {"target_dt": 2.0e-5, "max_added_fraction": 0.1}},
        "output": {"fields": ["displacement"], "every": 100, "monitors": {}},
    }
    out, h5 = _run(tmp_path, "debris", d)
    with h5py.File(h5) as f:
        ev = np.asarray(f["diagnostics/member_failures"])
        steps = sorted(f["steps"])
        T = np.array([f["steps"][k].attrs["time"] for k in steps])
        U = np.array([np.asarray(f["steps"][k]["displacement"]) for k in steps])
        conn = np.asarray(f["steps"][steps[-1]]["connectivity"]).astype(int)
    causes = {int(e): int(c) for _t, e, _r, c in ev}
    assert causes == {0: 1, 2: 2}
    tip = int(conn[2][1])                               # the link's free end
    # the girder falls at g; the link's free joint stays where it was left
    late = T > 0.03
    assert np.isfinite(U).all()
    assert np.ptp(U[late][:, tip], axis=0).max() < 1e-9
    g_end = int(conn[1][0])
    drop = U[-1][g_end, 2] - U[np.searchsorted(T, 0.03)][g_end, 2]
    t0, t1 = T[np.searchsorted(T, 0.03)], T[-1]
    assert drop < -0.5 * G * ((t1 - 0.02) ** 2 - (t0 - 0.02) ** 2) * 0.9


def test_a_massless_bearing_goes_when_the_girder_on_it_does(tmp_path):
    """v0.5: a massless bar is a support between two bodies. A girder on
    a bearing (a massless bar to the ground) and a post: when the girder is
    removed the bearing holds nothing up, and is removed (cause 2) -- left,
    bearings and crossbeams at a tower whose girders had gone were a pinned
    mechanism of stability masses."""
    _write_msh(tmp_path / "b.msh", [(0, 0, 0), (0, 0, 1), (2, 0, 1), (2, 0, 0)],
               {"base": (0, [0, 3]), "bearing": (1, [(0, 1)]), "girder": (1, [(1, 2)]), "post": (1, [(2, 3)])})
    light = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 1.0}
    beam = {"type": "beam", "shape": "general", "area": 1e-2, "iy": 1e-5, "iz": 1e-5, "j": 2e-5}
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": "b.msh"},
        "materials": {"steel": STEEL, "light": light},
        "sections": {"g": dict(beam, material="steel"),
                     "b": {"type": "bar", "material": "light", "shape": "general", "area": 1e-2}},
        "assign": [{"region": "girder", "section": "g", "orientation": [0, 0, 1]},
                   {"region": "post", "section": "g", "orientation": [1, 0, 0]},
                   {"region": "bearing", "section": "b"}],
        "constraints": [{"region": "base", "dofs": [1, 1, 1, 1, 1, 1]}],
        "loads": [{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}],
        "removals": [{"region": "girder", "time": 0.02}],
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.06, "damping": {"type": "none"},
                   "mass_scaling": {"target_dt": 2.0e-5, "max_added_fraction": 0.1}},
        "output": {"fields": ["displacement"], "every": 100, "monitors": {}},
    }
    out, h5 = _run(tmp_path, "bearing", d)
    with h5py.File(h5) as f:
        ev = np.asarray(f["diagnostics/member_failures"])
        u = np.asarray(f["steps"][sorted(f["steps"])[-1]]["displacement"])
    causes = {int(e): int(c) for _t, e, _r, c in ev}
    assert causes[0] == 2 and causes[1] == 1
    assert np.isfinite(u).all()


def test_a_carriageway_survives_a_member_splitting_off_under_the_vehicle(tmp_path):
    """The Xihoumen run died the moment a deck member split: the moving load
    rebuilt its path from the current connectivity and the road was in two
    pieces. The road is the chain as it was laid; the axles over a member
    that fell ride the fragment."""
    L, n = 4.0, 4
    xs = np.linspace(0, L, n + 1)
    _write_msh(tmp_path / "road.msh", [(x, 0, 0) for x in xs],
               {"left": (0, [0]), "right": (0, [n]), "mid": (0, [1, 2]),
                "deck": (1, [(k, k + 1) for k in range(n)])})
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": "road.msh"},
        "materials": {"steel": STEEL},
        "sections": {"b": {"type": "beam", "material": "steel", "shape": "general",
                           "area": 1e-2, "iy": 1e-5, "iz": 1e-5, "j": 2e-5}},
        "assign": [{"region": "all", "section": "b", "orientation": [0, 0, 1]}],
        "constraints": [{"region": "left", "dofs": [1, 1, 1, 1, 0, 0]}, {"region": "right", "dofs": [0, 1, 1, 1, 0, 0]}],
        "loads": [{"region": "deck", "moving_load": {"axles": [[0.0, 1000.0]], "start": 0.0, "speed": 10.0}}],
        "removals": [{"region": "mid", "time": 0.05}],
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.3, "damping": {"type": "viscous", "coeff": 20.0}},
        "output": {"fields": ["displacement"], "every": 100,
                   "monitors": {"w": {"kind": "displacement", "region": "right", "dof": "z"}}},
    }
    out, h5 = _run(tmp_path, "road", d)
    with h5py.File(h5) as f:
        keys = sorted(f["steps"])
        conn = np.asarray(f["steps"][keys[-1]]["connectivity"]).astype(int)
        u = np.asarray(f["steps"][keys[-1]]["displacement"])
    assert conn[1].min() >= 5                    # the middle member is on its own particles
    assert u[conn[1], 2].min() < -0.01           # and it fell, with the axle that was over it


def test_failure_on_a_fibre_member_points_to_fracture(tmp_path):
    d = {
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": "benchmarks/assets/beam_cantilever_v22.msh"},
        "materials": {"steel": {"type": "steel_cyclic", "grade": "Q235B", "E": 2.06e11, "rho": 7850.0}},
        "sections": {"tube": {"type": "fiber", "material": "steel", "shape": "tube",
                              "diameter": 0.2, "t": 0.01, "nt": 8}},
        "assign": [{"region": "all", "section": "tube", "orientation": [0, 0, 1],
                    "failure": {"yield_strength": 3e8, "extreme_fibre": [0.1, 0.1]}}],
        "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.1},
        "output": {"path": str(tmp_path / "x.h5")},
    }
    with pytest.raises(Exception, match="fracture"):
        prepare_case(dict(d, name="x"))


# --------------------------------------------------------------------------- #
#  many time-history loads are summed as one matrix product
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("same_grid", [True, False])
def test_stacked_load_groups_give_the_same_force_as_one_by_one(tmp_path, same_grid):
    d = _hanging(tmp_path, 3.0)
    rng = np.random.default_rng(0)
    grid = np.linspace(0, 1, 41)
    d["amplitudes"] = {}
    for k in range(12):
        g = grid if same_grid else np.sort(np.r_[0.0, rng.uniform(0, 1, 30), 1.0])
        d["amplitudes"][f"a{k}"] = [[float(p), float(v)] for p, v in zip(g, rng.normal(size=len(g)))]
        d["loads"].append({"region": ["tip", "weight"][k % 2],
                           "vector": [float(v) for v in rng.normal(size=3)], "amplitude": f"a{k}"})
    p = tmp_path / "many.yaml"
    p.write_text(yaml.safe_dump(dict(d, name="many", output={"path": str(tmp_path / "m.h5")})), encoding="utf-8")
    prep = prepare_case(str(p))
    s = make_structural_solver(_with_element(prep.case, prep.element, prep.case_path),
                               prep.mesh, prep.material)
    groups = s.step_data(0)[3]
    assert s._stacked_groups(groups) is not None
    for prog in (0.0, 0.137, 0.5, 0.999, 1.0):
        f_stacked = s._force(groups, [], prog)
        s._stack_cache = (groups, None)                    # force the one-by-one path
        f_loop = s._force(groups, [], prog)
        s._stack_cache = None
        np.testing.assert_allclose(f_stacked, f_loop, rtol=1e-12, atol=1e-9)


# --------------------------------------------------------------------------- #
#  a platoon of axles is evaluated at once, and its offsets follow the travel
# --------------------------------------------------------------------------- #
def test_many_axles_at_once_equal_one_by_one_and_trail_whichever_way_they_go():
    from types import SimpleNamespace
    from vfife.core.dof import DofLayout, repeated_rotation
    from vfife.core.moving_load import LoadPath, MovingLoad, consistent_point_load
    n = 21
    xs = np.linspace(0.0, 40.0, n) + 0.3 * np.sin(np.arange(n))       # uneven members
    nodes = np.c_[xs, np.zeros(n), np.zeros(n)]
    elements = np.c_[np.arange(n - 1), np.arange(1, n)]
    mesh = SimpleNamespace(nodes=nodes, elements=elements)
    dofs = DofLayout.frame(n)
    length = np.diff(xs)
    R0 = np.repeat(np.eye(3)[None], n - 1, axis=0)
    path = LoadPath(mesh, range(n - 1))
    rng = np.random.default_rng(3)
    axles = [[-float(o), float(f)] for o, f in zip(np.sort(rng.uniform(0, 30, 25)),
                                                   rng.uniform(1e4, 2e5, 25))]
    for speed, start in ((7.0, 5.0), (-7.0, 35.0)):
        ml = MovingLoad({"axles": axles, "start": start, "speed": speed},
                        path, dofs, mesh, length, R0, 10.0)
        for t in (0.0, 0.4, 1.3, 3.0):
            got = ml(0.0, t, np.zeros(dofs.size))
            ref = np.zeros(dofs.size)
            way = np.sign(speed)
            for off, f in axles:
                hit = path.locate(start + speed * t + way * off)
                if hit is None:
                    continue
                e, xi = hit
                fl = consistent_point_load(length[e], R0[e], xi, f * ml.direction)
                ref[dofs.element_dofs(elements[e])] += repeated_rotation(R0[e]).T @ fl
            np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-6)
    # one lead axle and one trailing axle 4 m behind it, driven towards -x:
    # the trailing one is at the LARGER x
    back = MovingLoad({"axles": [[0.0, 1.0], [-4.0, 2.0]], "start": 20.0, "speed": -1.0},
                      path, dofs, mesh, length, R0, 10.0)
    f = back(0.0, 0.0, np.zeros(dofs.size)).reshape(n, 6)[:, 2]
    assert np.sum(f * xs) / np.sum(f) > 20.0 + 1.0      # load centroid behind the lead axle
