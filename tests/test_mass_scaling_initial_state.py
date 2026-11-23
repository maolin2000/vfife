"""Selective mass scaling and starting a run from an earlier state.

The model is the smallest one that has the Xihoumen defect: a 20 m simply
supported beam with a short, very stiff, nearly massless link hanging off its
midspan -- the shape of a stay anchor on a rigid arm. That link sets the
explicit step by five orders of magnitude, and nothing about the beam depends
on it.
"""
import numpy as np
import pytest
import yaml

from vfife.api import prepare_case, run_case, _with_element
from vfife.core.structural import make_structural_solver

G = 9.80665


def _mesh(path):
    # 11 beam nodes along x, a link from the midspan node down 0.5 m
    xs = np.linspace(0.0, 20.0, 11)
    nodes = [(x, 0.0, 0.0) for x in xs] + [(10.0, 0.0, -0.5)]
    groups = {"left": (0, [0]), "right": (0, [10]), "mid": (0, [5]), "tip": (0, [11]),
              "beam": (1, [(k, k + 1) for k in range(10)]), "link": (1, [(5, 11)])}
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


def _case(tmp_path, name, solver, loads=None, steps=None, link_rho=1.0):
    _mesh(tmp_path / "beam.msh")
    d = {
        "name": name,
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": "beam.msh"},
        "materials": {"steel": {"type": "linear_elastic", "E": 2.1e11, "nu": 0.3, "rho": 7850.0},
                      "rigid": {"type": "linear_elastic", "E": 2.1e12, "nu": 0.3, "rho": link_rho}},
        "sections": {"box": {"type": "beam", "material": "steel", "shape": "general",
                             "area": 0.05, "iy": 2e-3, "iz": 2e-3, "j": 3e-3},
                     # stiff and nearly massless, but not so stiff that it
                     # binds the (heavy) beam node it hangs from
                     "rigid": {"type": "beam", "material": "rigid", "shape": "general",
                               "area": 0.01, "iy": 1e-4, "iz": 1e-4, "j": 1e-4}},
        "assign": [{"region": "beam", "section": "box", "orientation": [0, 0, 1]},
                   {"region": "link", "section": "rigid", "orientation": [1, 0, 0]}],
        "constraints": [{"region": "left", "dofs": [1, 1, 1, 1, 0, 0]},
                        {"region": "right", "dofs": [0, 1, 1, 0, 0, 0]}],
        "loads": loads or [],
        "solver": solver,
        "output": {"path": str(tmp_path / f"{name}.h5"), "fields": ["displacement"],
                   "monitors": {"w": {"kind": "displacement", "region": "mid", "dof": "z"},
                                "R": {"kind": "reaction", "region": "left", "dof": "z"}}},
    }
    if steps:
        d["steps"] = steps
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return p


def _solver(path):
    prep = prepare_case(str(path))
    return make_structural_solver(_with_element(prep.case, prep.element, prep.case_path),
                                  prep.mesh, prep.material)


DYN = {"backend": "numpy", "mode": "dynamic", "total_time": 0.01, "damping": {"type": "none"}}
MS = {"target_dt": 1e-5}


# --------------------------------------------------------------------------- #
#  mass scaling
# --------------------------------------------------------------------------- #

def test_the_light_stiff_link_sets_the_step_and_scaling_lifts_it_to_the_target(tmp_path):
    plain = _solver(_case(tmp_path, "plain", DYN))
    assert plain.dt < 1e-6                                  # the link's own bound
    target = 1e-5
    s = _solver(_case(tmp_path, "scaled", dict(DYN, mass_scaling={"target_dt": target})))
    assert s.dt == pytest.approx(target, rel=1e-9)          # exactly what was asked
    rep = s.mass_scaling
    # the only node whose mass changed is the link's tip -- the beam is untouched
    assert rep["nodes"] == 1 and rep["added_per_node"][11] > 0
    dm = (s.physical_mass - plain.physical_mass).reshape(-1, 6)
    assert np.all(dm[:11, :3] == 0.0)
    assert np.allclose(dm[11, :3], dm[11, 0])               # isotropic: a node's mass stays a scalar


def test_a_target_the_model_already_meets_adds_nothing(tmp_path):
    plain = _solver(_case(tmp_path, "p", DYN, link_rho=7850.0))
    s = _solver(_case(tmp_path, "s", dict(DYN, mass_scaling={"target_dt": plain.dt / 2}),
                            link_rho=7850.0))
    assert s.mass_scaling["added_mass"] == 0.0
    assert s.dt == plain.dt
    np.testing.assert_array_equal(s.physical_mass, plain.physical_mass)


def test_a_target_that_needs_too_much_mass_is_refused(tmp_path):
    with pytest.raises(ValueError, match="max_added_fraction"):
        _solver(_case(tmp_path, "cap", dict(DYN, mass_scaling={"target_dt": 1e-2})))


def test_the_added_mass_is_weighed_and_relaxation_adds_the_same(tmp_path):
    """Gravity sees the added mass (it is real mass), and a relaxation run of
    the same case adds exactly what the dynamic run adds -- so a dead state
    relaxed with it is in equilibrium under the weight a dynamic run carries."""
    relax = {"backend": "numpy", "mode": "relaxation", "tolerance": 1e-8, "mass_scaling": MS}
    p = _case(tmp_path, "w", relax, loads=[{"region": "all", "gravity": [0, 0, -G]}],
              steps=[{"name": "dead", "load_steps": 1, "max_steps_per_load": 200000}])
    s_relax = _solver(p)
    s_dyn = _solver(_case(tmp_path, "d", dict(DYN, mass_scaling=MS)))
    np.testing.assert_array_equal(s_relax.physical_mass, s_dyn.physical_mass)
    out = run_case(str(p))
    assert out.converged
    total = s_relax.physical_mass.reshape(-1, 6)[:, 2].sum()
    R_left = out.monitors["R"][-1]
    # symmetric beam, the link at midspan: each support carries half the weight
    assert R_left == pytest.approx(0.5 * total * G, rel=1e-5)


def test_mass_scaling_is_refused_where_no_solver_reads_it():
    from vfife.core.structural import check_mass_scaling
    with pytest.raises(ValueError, match="mass_scaling"):
        check_mass_scaling({"element": {"type": "truss2"},
                            "solver": {"mass_scaling": {"target_dt": 1e-4}}})
    check_mass_scaling({"element": {"type": "mixed_line"},
                        "solver": {"mass_scaling": {"target_dt": 1e-4}}})


# --------------------------------------------------------------------------- #
#  initial_state
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def dead(tmp_path_factory):
    """The relaxed dead state, computed once for the tests below."""
    d = tmp_path_factory.mktemp("dead")
    p = _case(d, "dead", {"backend": "numpy", "mode": "relaxation", "tolerance": 1e-9,
                          "mass_scaling": MS},
              loads=[{"region": "all", "gravity": [0, 0, -G]}],
              steps=[{"name": "dead", "load_steps": 1, "max_steps_per_load": 400000}])
    assert run_case(str(p)).converged
    return d


def _rest(where, dead_dir, start=True, name="rest"):
    solver = dict(DYN, total_time=0.2, mass_scaling=MS)
    if start:
        solver["initial_state"] = str(dead_dir / "dead.h5")
    p = _case(where, name, solver,
              loads=[{"region": "all", "gravity": [0, 0, -G], "amplitude": "hold"}])
    return run_case(str(p))


def test_a_dynamic_run_started_from_the_relaxed_dead_state_stays_there(tmp_path, dead):
    import h5py
    with h5py.File(dead / "dead.h5") as f:
        dead_w = float(f["monitors"]["w"][-1])
    out = _rest(tmp_path, dead)
    w = np.asarray(out.monitors["w"])
    assert dead_w < -1e-3                                   # it did sag
    assert np.abs(w - dead_w).max() < 1e-6 * abs(dead_w) + 1e-9
    # and the run it replaces: from the mesh, the same load rings about the
    # dead state with an amplitude of the whole sag
    out_mesh = _rest(tmp_path, dead, start=False, name="mesh")
    assert np.abs(np.asarray(out_mesh.monitors["w"]) - dead_w).max() > 0.5 * abs(dead_w)


def test_an_initial_state_from_another_mesh_is_refused(tmp_path, dead):
    other = tmp_path / "other"
    other.mkdir()
    p = _case(other, "o", dict(DYN, initial_state=str(dead / "dead.h5")))
    # the same topology moved 1 m: a different set of particles
    msh = (other / "beam.msh").read_text(encoding="utf-8").replace("\n12 10 0 -0.5", "\n12 10 0 -1.5")
    (other / "beam.msh").write_text(msh, encoding="utf-8")
    with pytest.raises(ValueError, match="different mesh"):
        run_case(str(p))


def test_resuming_and_starting_over_are_not_both_asked_for(tmp_path, dead):
    # A restart ignores `initial_state` (v1.0: the stopped run of a
    # case that starts from a dead state resumes with that same case); what
    # it resumes must still be this run -- the relaxation's checkpoint is
    # another mode, refused by the fingerprint
    p = _case(tmp_path, "both", dict(DYN, initial_state=str(dead / "dead.h5")))
    with pytest.raises(ValueError, match="mode: relaxation there, dynamic here"):
        run_case(str(p), restart=str(dead / "dead.h5"))
