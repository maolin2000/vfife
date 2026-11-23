"""Several solid sections on one mesh: material blocks.

WHY
    A reinforced-concrete beam is not one material. The core inside the
    stirrups is confined and keeps its strength past the cover's crushing
    strain; the steel plates under the loads spread them; a joint has two
    concretes. Every commercial code assigns sections to element sets. Until
    this stage the solver carried ONE material for the whole continuum, and
    the 12.5% the Alberta beam was short of its measured capacity was, in
    part, the confinement the model could not express.

THE RULE
    An element belongs to the section whose region holds EVERY one of its
    nodes; assignments apply in order and a later one takes what an earlier
    one also covers. Anything left over, or a section that takes nothing, is
    refused with the element ids -- never silently narrowed.

WHAT IS PINNED
    * a mesh split into two blocks of the SAME material is the same model,
      to the last bit that assembly order allows
    * two elastic halves in series are the series spring, exactly (nu = 0)
    * the fused and the device-resident backends agree with numpy on a
      two-block model; the resident loop keeps every block on the device
    * output fields are the union over blocks, a block that has no damage
      reports zero of it, and `block` says which section an element is in
    * a checkpoint carries every block's state and comes back
    * the refusals, and what preflight says first
"""
import copy

import numpy as np
import pytest
import yaml

from vfife.api import prepare_case, run_case
from vfife.core.diagnostics import UserError
from vfife.core.solver import Solver, partition_elements

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

L, B, H = 1.0, 0.2, 0.2
E1, E2 = 3.0e10, 1.0e10
SOFT = {"type": "linear_elastic", "E": E1, "nu": 0.0, "rho": 2400.0}
SOFTER = {"type": "linear_elastic", "E": E2, "nu": 0.0, "rho": 1200.0}
CDP = {"type": "cdp", "E": 2.98e10, "nu": 0.2, "rho": 2400.0,
       "fb0_fc0": 1.16, "Kc": 0.6667, "dilation_angle": 36.0, "eccentricity": 0.1,
       "wt": 0.0, "wc": 1.0, "viscosity": 1.0, "n_return": 5,
       "crack_band": "both", "l_ref": "auto", "damage_max": 0.80,
       "compression": {"source": "gb50010", "fc": 44e6, "beta_c": 0.7, "sigma_c0_ratio": 0.4},
       "tension": {"source": "fracture_energy", "ft": 3.42e6, "Gf": 140.0, "beta_t": 0.9}}


def _case(tmp_path, materials, sections, assign, *, divisions=(6, 2, 2), load=-2.0e5,
          backend="numpy", steps=6, extra_solver=None, fields=None, regions=None):
    """A prism fixed at x = 0, a face load at x = L; regions for both halves."""
    solver = {"backend": backend, "mode": "relaxation", "mass": "underwood",
              "tolerance": 1.0e-5, "relaxation_safety": 0.3,
              "damping": {"type": "kinetic"}, "n_steps": steps * 6000 + 10}
    solver.update(extra_solver or {})
    case = {
        "name": "blocks", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"dimensions": [L, B, H], "divisions": list(divisions)},
        "materials": materials, "sections": sections, "assign": assign,
        "regions": {
            "fix": {"where": {"x": 0.0}},
            "tip": {"where": {"x": L}},
            "left": {"where": {"x": {"max": L / 2 + 1e-9}}},
            "right": {"where": {"x": {"min": L / 2 - 1e-9}}},
            **(regions or {})},
        "constraints": [{"region": "fix", "dofs": [1, 1, 1]}],
        "steps": [{"name": "push",
                   "loads": [{"region": "tip", "vector": [load, 0.0, 0.0]}],
                   "load_steps": steps, "max_steps_per_load": 6000,
                   "tolerance": 1.0e-5}],
        "output": {"path": str(tmp_path / "blocks.h5"), "every": 100000,
                   "fields": fields or ["displacement", "von_mises"],
                   "monitors": {"R": {"kind": "reaction", "region": "fix", "dof": "x"},
                                "u": {"kind": "displacement", "region": "tip", "dof": "x"}}},
        "solver": solver,
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "blocks.yaml"
    path.write_text(yaml.safe_dump(case, allow_unicode=True), encoding="utf-8")
    return path


def _solver(path):
    p = prepare_case(str(path))
    return Solver(p.case.model_dump(mode="python"), p.mesh, p.material, log=lambda *_: None)


# --------------------------------------------------------------------------- #
#  the partition
# --------------------------------------------------------------------------- #

def test_an_element_is_in_a_region_only_when_every_node_is_and_later_wins():
    conn = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]])
    regions = {"all": np.arange(8), "left": np.array([0, 1, 2, 3, 4, 5]),
               "mid": np.array([2, 3, 4, 5])}
    owner, over = partition_elements(conn, 8, [{"region": "all"}, {"region": "left"}],
                                     lambda r: regions[r])
    assert owner.tolist() == [1, 1, 0]                # element 2 has node 6, 7 outside
    assert over == [(0, 1, 2)]
    owner, over = partition_elements(conn, 8, [{"region": "mid"}], lambda r: regions[r])
    assert owner.tolist() == [-1, 0, -1]


def test_a_mesh_split_into_two_blocks_of_one_material_is_the_same_model(tmp_path):
    """Same material, two sections, `all` then `right`: the answer must be
    the single-block answer. This is the invariant everything else stands
    on -- a block is bookkeeping, not physics."""
    one = _case(tmp_path / "one", {"c": SOFT}, {"body": {"type": "solid", "material": "c"}},
                [{"region": "all", "section": "body"}])
    two = _case(tmp_path / "two", {"c": SOFT, "c2": SOFT},
                {"body": {"type": "solid", "material": "c"},
                 "other": {"type": "solid", "material": "c2"}},
                [{"region": "all", "section": "body"}, {"region": "right", "section": "other"}])
    s1, s2 = _solver(one), _solver(two)
    assert len(s1.blocks) == 1 and len(s2.blocks) == 2
    assert [b.name for b in s2.blocks] == ["body", "other"]
    assert s2.blocks[0].n_elem == s2.blocks[1].n_elem == s1.nel // 2
    ids = np.sort(np.concatenate([b.elem_ids for b in s2.blocks]))
    assert ids.tolist() == list(range(s1.nel))        # a partition of the mesh
    # the Underwood masses, built from every block's row sums, agree
    np.testing.assert_allclose(s2.mass, s1.mass, rtol=1e-12)
    s1.run(on_progress=lambda *_: None, progress_every=500)
    s2.run(on_progress=lambda *_: None, progress_every=500)
    np.testing.assert_allclose(s2.u, s1.u, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(s2.monitors["R"].values, s1.monitors["R"].values, rtol=1e-9)


def test_two_elastic_halves_in_series_are_the_series_spring(tmp_path):
    """E1 on the left, E2 on the right, nu = 0 and the tip face pushed as
    one: the stress is uniform and the exact reaction is
    R = delta A / ((L/2)(1/E1 + 1/E2))."""
    delta = -1.0e-4
    path = _case(tmp_path, {"c1": SOFT, "c2": SOFTER},
                 {"s1": {"type": "solid", "material": "c1"},
                  "s2": {"type": "solid", "material": "c2"}},
                 [{"region": "all", "section": "s1"}, {"region": "right", "section": "s2"}])
    case = yaml.safe_load(path.read_text(encoding="utf-8"))
    case["steps"][0].pop("loads")
    case["steps"][0]["constraints"] = [{"region": "tip", "dofs": [1, 0, 0], "value": delta}]
    path.write_text(yaml.safe_dump(case), encoding="utf-8")
    s = _solver(path)
    assert [b.material.E for b in s.blocks] == [E1, E2]
    s.run(on_progress=lambda *_: None, progress_every=500)
    A = B * H
    exact = -delta * A / ((L / 2) * (1.0 / E1 + 1.0 / E2))
    assert s.monitors["R"].values[-1] == pytest.approx(exact, rel=1e-4)
    # the stress is one number across both halves; the strain is not
    sig = [b.force(s.u, b.quat, b.state, 1.0, rot_iters=30, want_stress=True)[1]
           for b in s.blocks]
    assert np.allclose(sig[0][:, :, 0], -exact / A, rtol=1e-4)
    assert np.allclose(sig[1][:, :, 0], -exact / A, rtol=1e-4)
    eps = [b.force(s.u, b.quat, b.state, 1.0, rot_iters=30, want_stress=True)[1][:, :, 0] / b.material.E
           for b in s.blocks]
    assert np.mean(eps[0]) == pytest.approx(np.mean(eps[1]) * E2 / E1, rel=1e-4)


def test_gravity_and_the_time_step_use_each_blocks_own_material(tmp_path):
    """A block twice as dense weighs twice as much, and the CFL step is the
    slowest element of any block against ITS wave speed."""
    path = _case(tmp_path, {"c1": SOFT, "c2": SOFTER},
                 {"s1": {"type": "solid", "material": "c1"},
                  "s2": {"type": "solid", "material": "c2"}},
                 [{"region": "all", "section": "s1"}, {"region": "right", "section": "s2"}],
                 extra_solver={"mode": "dynamic", "mass": "physical", "n_steps": 2,
                               "total_time": 1.0e-4, "damping": {"type": "none"}})
    case = yaml.safe_load(path.read_text(encoding="utf-8"))
    case["steps"][0]["loads"] = [{"region": "all", "gravity": [0.0, 0.0, -9.81]}]
    del case["steps"][0]["load_steps"]
    path.write_text(yaml.safe_dump(case), encoding="utf-8")
    s = _solver(path)
    W = sum(float(F[:, 2].sum()) for _a, F in s.load_groups)
    V = L * B * H / 2
    assert W == pytest.approx(-9.81 * (2400.0 * V + 1200.0 * V), rel=1e-12)
    assert s.mass[:, 0].sum() == pytest.approx(2400.0 * V + 1200.0 * V, rel=1e-12)
    h = [float(b.pre.char_len.min()) for b in s.blocks]
    c = [b.material.wave_speed for b in s.blocks]
    assert c[0] != c[1] and h[0] == pytest.approx(h[1])
    bound = s._step_solver("dt_safety") * min(h[0] / c[0], h[1] / c[1])
    # the step is the bound, shortened so that a whole number of steps makes
    # the run's duration
    from vfife.elements.stability import aligned_step
    assert s.dt == pytest.approx(aligned_step(1.0e-4, bound)[0], rel=1e-9)
    assert s.dt <= bound * (1 + 1e-12) and s.n_steps * s.dt == pytest.approx(1.0e-4, rel=1e-12)


# --------------------------------------------------------------------------- #
#  backends
# --------------------------------------------------------------------------- #

def _two_block_prism(tmp_path, backend, **extra):
    return _case(tmp_path, {"c1": SOFT, "c2": SOFTER},
                 {"s1": {"type": "solid", "material": "c1"},
                  "s2": {"type": "solid", "material": "c2"}},
                 [{"region": "all", "section": "s1"}, {"region": "right", "section": "s2"}],
                 backend=backend, extra_solver=extra)


def test_the_fused_backend_agrees_with_numpy_on_two_blocks(tmp_path):
    pytest.importorskip("numba")
    ref = _solver(_two_block_prism(tmp_path / "np", "numpy"))
    fused = _solver(_two_block_prism(tmp_path / "nb", "numba"))
    for s in (ref, fused):
        s.run(on_progress=lambda *_: None, progress_every=500)
    np.testing.assert_allclose(fused.u, ref.u, rtol=1e-8, atol=1e-14)


def test_the_resident_loop_keeps_every_block_on_the_device(tmp_path):
    """Warp on its CPU device is the same code path as the GPU: block 0 owns
    the nodal state and block 1 is attached to it. The answer is numpy's."""
    pytest.importorskip("warp")
    ref = _solver(_two_block_prism(tmp_path / "np", "numpy"))
    dev = _solver(_two_block_prism(tmp_path / "wp", "warp", device="cpu", precision="fp64"))
    assert dev.resident
    assert len(dev.blocks) == 2
    assert dev._force.attached == [dev.blocks[1].force]
    for s in (ref, dev):
        s.run(on_progress=lambda *_: None, progress_every=500)
    np.testing.assert_allclose(dev.u, ref.u, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(dev.monitors["R"].values, ref.monitors["R"].values, rtol=1e-8)
    # the state came back per block
    for b in dev.blocks:
        assert b.quat.shape == (b.n_elem, 4)
        assert np.allclose(b.quat[:, 0], 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
#  output and restart
# --------------------------------------------------------------------------- #

def test_output_fields_are_the_union_and_a_block_without_damage_reports_none(tmp_path):
    """CDP on the left, elastic on the right. `d` and `dt` exist for the
    whole mesh -- zero where the material has no such thing -- and `block`
    says which section every element is in."""
    path = _case(tmp_path, {"c": CDP, "pad": SOFT},
                 {"concrete": {"type": "solid", "material": "c"},
                  "plate": {"type": "solid", "material": "pad"}},
                 [{"region": "all", "section": "concrete"}, {"region": "right", "section": "plate"}],
                 steps=2, load=2.0e4, fields=["displacement", "d", "dt", "block", "von_mises"])
    s = _solver(path)
    assert s.blocks[0].material.name == "cdp" and s.blocks[1].material.name == "linear_elastic"
    assert s.blocks[0].state.shape[1] != s.blocks[1].state.shape[1]
    s.run(on_progress=lambda *_: None, progress_every=500)
    sig = [b.force(s.u, b.quat, b.state, 1.0, rot_iters=2, want_stress=True)[1] for b in s.blocks]
    f = s.nodal_fields(sig)
    for k in ("d", "dt", "dc", "block", "von_mises"):
        assert f[k].shape == (s.nel,)
    assert np.all(f["block"][s.blocks[0].elem_ids] == 0)
    assert np.all(f["block"][s.blocks[1].elem_ids] == 1)
    assert np.all(f["d"][s.blocks[1].elem_ids] == 0.0)
    assert np.isfinite(f["von_mises"]).all() and f["von_mises"].max() > 0
    # and the file writer takes it: the requested names all exist
    out = run_case(str(path))
    assert out.status == "CONVERGED"
    import h5py
    with h5py.File(out.result_path if hasattr(out, "result_path") else out.path) as h:
        frames = [k for k in h if k.startswith("frame")]
        frame = h[sorted(frames)[-1]] if frames else h[[k for k in h if "step" in k][-1]]
        names = set()
        frame.visit(lambda n: names.add(n.split("/")[-1]))
    assert {"d", "dt", "block"} <= names


def test_a_checkpoint_carries_every_blocks_state_and_comes_back(tmp_path):
    path = _case(tmp_path, {"c": CDP, "pad": SOFT},
                 {"concrete": {"type": "solid", "material": "c"},
                  "plate": {"type": "solid", "material": "pad"}},
                 [{"region": "all", "section": "concrete"}, {"region": "right", "section": "plate"}],
                 steps=2, load=2.0e4)
    s = _solver(path)
    s.run(on_progress=lambda *_: None, progress_every=500)
    ck = s.checkpoint()
    assert "state0" in ck and "state1" in ck and "state" not in ck
    fresh = _solver(path)
    fresh.restore(ck)
    for a, b in zip(s.blocks, fresh.blocks):
        np.testing.assert_array_equal(a.state, b.state)
        np.testing.assert_array_equal(a.quat, b.quat)
    np.testing.assert_array_equal(fresh.u, s.u)
    # a checkpoint from a different block layout is refused, not misread
    other = _solver(_case(tmp_path / "one", {"c": SOFT},
                          {"body": {"type": "solid", "material": "c"}},
                          [{"region": "all", "section": "body"}]))
    with pytest.raises(ValueError, match="contract"):
        other.restore(ck)


# --------------------------------------------------------------------------- #
#  refusals
# --------------------------------------------------------------------------- #

def test_an_element_no_section_covers_is_refused_with_its_id(tmp_path):
    path = _case(tmp_path, {"c": SOFT}, {"s": {"type": "solid", "material": "c"}},
                 [{"region": "left", "section": "s"}])
    with pytest.raises(UserError) as exc:
        _solver(path)
    assert exc.value.code == "ELEMENTS_UNASSIGNED"
    assert "12 element(s)" in exc.value.message
    assert "EVERY one of its nodes" in exc.value.hint


def test_a_section_that_takes_no_whole_element_is_refused(tmp_path):
    """A region one node plane wide holds nodes, not elements."""
    path = _case(tmp_path, {"c": SOFT, "c2": SOFTER},
                 {"s": {"type": "solid", "material": "c"},
                  "thin": {"type": "solid", "material": "c2"}},
                 [{"region": "all", "section": "s"}, {"region": "tip", "section": "thin"}])
    with pytest.raises(UserError) as exc:
        _solver(path)
    assert exc.value.code == "SECTION_EMPTY"
    assert "'thin'" in exc.value.message


def test_a_later_assignment_takes_what_an_earlier_one_covered_and_says_so(tmp_path):
    path = _case(tmp_path, {"c": SOFT, "c2": SOFTER},
                 {"s": {"type": "solid", "material": "c"},
                  "core": {"type": "solid", "material": "c2"}},
                 [{"region": "all", "section": "s"}, {"region": "right", "section": "core"}])
    p = prepare_case(str(path))
    lines = []
    Solver(p.case.model_dump(mode="python"), p.mesh, p.material, log=lines.append)
    assert any("'core' takes 12 element(s) also covered by 's'" in ln for ln in lines)
    codes = {i["code"] for i in p.preflight["issues"]}
    assert "SECTION_OVERRIDE" in codes and "SECTION_ELEMENTS" in codes
    assert p.preflight["status"] == "PASS"


def test_preflight_reports_an_uncovered_element_before_the_solver_does(tmp_path):
    path = _case(tmp_path, {"c": SOFT}, {"s": {"type": "solid", "material": "c"}},
                 [{"region": "left", "section": "s"}])
    p = prepare_case(str(path))
    assert p.preflight["status"] == "FAIL"
    assert any(i["code"] == "ELEMENTS_UNASSIGNED" and i["details"]["count"] == 12
               for i in p.preflight["issues"])


def test_a_bar_through_two_blocks_is_located_by_geometry(tmp_path):
    """The host of an embedded bar is whichever element it lies in; the
    material block is not the bar's concern. With a damaging block anywhere
    `host_deduction` is still refused."""
    case = yaml.safe_load(_case(tmp_path, {"c": CDP, "pad": SOFT, "steel": {
        "type": "rebar", "E": 2.0e11, "rho": 7850.0, "yield_strength": 4.5e8, "hardening": 2.0e9}},
        {"concrete": {"type": "solid", "material": "c"},
         "plate": {"type": "solid", "material": "pad"},
         "bar": {"type": "bar", "material": "steel", "shape": "circle", "diameter": 0.016, "area": 2.0e-4}},
        [{"region": "all", "section": "concrete"}, {"region": "right", "section": "plate"}],
        steps=1).read_text(encoding="utf-8"))
    case["reinforcement"] = {"main": {"line": {"start": [0.0, 0.1, 0.1], "end": [L, 0.1, 0.1],
                                               "divisions": 12}}}
    case["assign"].append({"reinforcement": "main", "section": "bar", "name": "main"})
    path = tmp_path / "bar.yaml"
    path.write_text(yaml.safe_dump(case), encoding="utf-8")
    s = _solver(path)
    assert len(s.rebar) == 1 and s.rebar[0].n_elem == 12
    case["assign"][-1]["host_deduction"] = 2.98e10
    path.write_text(yaml.safe_dump(case), encoding="utf-8")
    with pytest.raises(UserError) as exc:
        _solver(path)
    assert exc.value.code == "HOST_DEDUCTION_DAMAGING"
