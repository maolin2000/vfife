"""A restart resumes the run it came from.

The property every test here checks: N steps in one run, and N/2 steps,
interrupted, then `--restart`, end in the SAME state -- bit for bit, the
integrator's memory, the material state and the energy ledger included.
Before this stage only the solid solver passed it, and only for one dynamic
step: the frame, fibre, truss and implicit solvers restored the state and
then ran the whole analysis again on top of it, from t = 0 and load factor 0.
"""
import copy
import pathlib
import sys

import numpy as np
import pytest
import yaml

from vfife import api
from vfife.core.diagnostics import AnalysisInterrupted, UserError

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from test_member_failure import _write_msh  # noqa: E402

STEEL = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}


def _write(tmp_path, d, name):
    p = tmp_path / f"{name}.yaml"
    d = copy.deepcopy(d)
    d["name"] = name
    d.setdefault("output", {})["path"] = str(tmp_path / f"{name}.h5")
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return p


def _split(tmp_path, d, stop, progress_every=10, name="c"):
    """(full solver, restarted solver): one uninterrupted run, and one
    stopped where `stop(solver, local)` says and restarted from its file."""
    p = _write(tmp_path, d, name)
    got = {}
    api.run_case(p, output=str(tmp_path / "full.h5"), progress_every=progress_every,
                 on_ready=lambda s, _p: got.__setitem__("full", s))

    def prog(local, *_):
        return not stop(got["half"], local)
    with pytest.raises(AnalysisInterrupted):
        api.run_case(p, output=str(tmp_path / "half.h5"), progress_every=progress_every,
                     on_ready=lambda s, _p: got.__setitem__("half", s), on_progress=prog)
    api.run_case(p, output=str(tmp_path / "rest.h5"), restart=str(tmp_path / "half.h5"),
                 progress_every=progress_every, on_ready=lambda s, _p: got.__setitem__("rest", s))
    return got["full"], got["half"], got["rest"]


def _same(a, b, skip=()):
    ca, cb = a.checkpoint(), b.checkpoint()
    assert set(ca) == set(cb)
    for k in ca:
        if k in skip or np.asarray(ca[k]).dtype.kind in "SUO":
            continue
        assert np.array_equal(np.asarray(ca[k]), np.asarray(cb[k])), k


def _cantilever(tmp_path, n=8, L=2.0):
    _write_msh(tmp_path / "c.msh", [(L * i / n, 0, 0) for i in range(n + 1)],
               {"root": (0, [0]), "tip": (0, [n]), "bar": (1, [(i, i + 1) for i in range(n)])})
    return {"version": 1, "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "gmsh", "path": str(tmp_path / "c.msh")},
            "materials": {"steel": dict(STEEL)},
            "sections": {"beam": {"type": "beam", "material": "steel", "shape": "rect",
                                  "width": 0.1, "height": 0.1}},
            "assign": [{"region": "bar", "section": "beam", "orientation": [0, 1, 0]}],
            "constraints": [{"region": "root", "dofs": [1] * 6}],
            "loads": [{"region": "tip", "vector": [0, 0, -1.0e3]}],
            "output": {"every": 10 ** 6, "fields": ["displacement"],
                       "monitors": {"w": {"kind": "displacement", "region": "tip", "dof": "z"}}}}


# ------------------------------------------------------------ frame family
def test_a_dynamic_frame_run_resumes_its_clock(tmp_path):
    d = _cantilever(tmp_path)
    d["solver"] = {"backend": "numpy", "mode": "dynamic", "total_time": 0.02, "damping": {"type": "none"}}
    full, half, rest = _split(tmp_path, d, lambda s, local: s.step >= 300)
    assert half.step < full.step and rest.step == full.step
    _same(full, rest)
    assert rest.times[0] > half.times[-1] if half.times else True


def test_a_staged_frame_relaxation_resumes_inside_its_level(tmp_path):
    """Stopped in level 2 of 4 with a prescribed tip displacement: the level
    blends in from where IT began, and the run does not reload from level 1."""
    d = _cantilever(tmp_path)
    d["solver"] = {"backend": "numpy", "mode": "relaxation", "load_steps": 4, "tolerance": 1e-7,
                   "n_steps": 100000}
    d["constraints"].append({"region": "tip", "dofs": [0, 0, 1, 0, 0, 0], "value": -0.05})
    d["loads"] = []
    d["output"]["monitors"] = {"R": {"kind": "reaction", "region": "root", "dof": "z"}}
    full, half, rest = _split(tmp_path, d, lambda s, stage: stage >= 2, progress_every=50)
    assert rest.times == pytest.approx(full.times[len(full.times) - len(rest.times):])
    assert rest.times[0] == 0.5                       # level 2, not level 1 again
    _same(full, rest)


def test_a_restart_from_a_frame_checkpoint_goes_on_from_the_frame(tmp_path):
    """The checkpoint written with a frame (`checkpoint_every`), not the one
    an interrupt writes: a level that was recorded is not solved again."""
    d = _cantilever(tmp_path)
    d["solver"] = {"backend": "numpy", "mode": "relaxation", "load_steps": 4, "tolerance": 1e-7,
                   "n_steps": 100000}
    d["output"]["every"] = 1
    prepared = api.prepare_case(api.CaseDraft(d, base=tmp_path))
    from vfife.cli import make_solver
    held = {}

    def grab(s, step, t, sig):
        if abs(t - 0.5) < 1e-12:
            held["ck"] = s.checkpoint()
    full = make_solver(api._with_element(prepared.case, prepared.element, prepared.case_path),
                       prepared.mesh, prepared.material)
    full.run(on_output=grab)
    again = make_solver(api._with_element(prepared.case, prepared.element, prepared.case_path),
                        prepared.mesh, prepared.material)
    again.restore(held["ck"])
    again.run()
    assert again.times == [0.75, 1.0]
    _same(full, again)


def test_an_implicit_static_run_resumes_at_its_load_factor(tmp_path):
    d = _cantilever(tmp_path)
    d["solver"] = {"backend": "numpy", "mode": "static", "load_steps": 6, "tolerance": 1e-6}
    d["output"]["every"] = 1
    full, half, rest = _split(tmp_path, d, lambda s, level: level >= 3, progress_every=1)
    assert rest.times[0] == pytest.approx(4 / 6)
    _same(full, rest)


def test_a_restart_with_nothing_left_to_run_is_refused(tmp_path):
    d = _cantilever(tmp_path)
    d["solver"] = {"backend": "numpy", "mode": "dynamic", "total_time": 0.005}
    p = _write(tmp_path, d, "done")
    api.run_case(p)
    with pytest.raises(UserError, match="nothing left to run"):
        api.run_case(p, output=str(tmp_path / "again.h5"), restart=str(tmp_path / "done.h5"))


@pytest.mark.parametrize("case, stop", [
    ("case_05_space_truss_seismic", 100),       # truss: the ground motion's clock
    ("case_38_sprung_mass_crossing", 300),      # a vehicle mid-crossing
])
def test_shipped_dynamic_cases_resume(tmp_path, case, stop):
    d = api.load_case(ROOT / "cases" / f"{case}.yaml")
    data = copy.deepcopy(d.data)
    for key in ("mesh",):
        if data[key].get("path"):
            data[key]["path"] = str((ROOT / "cases" / data[key]["path"]).resolve())
    for c in data.get("loads", []) + data.get("constraints", []):
        for k in ("ground_motion", "record"):
            if isinstance(c.get(k), dict) and c[k].get("path"):
                c[k]["path"] = str((ROOT / "cases" / c[k]["path"]).resolve())
    data["solver"]["total_time"] = min(float(data["solver"].get("total_time", 1.0)), 0.8)
    data["output"] = {**data["output"], "every": 10 ** 6}
    data["output"].pop("checkpoint_every", None)
    full, half, rest = _split(tmp_path, data, lambda s, local: s.step >= stop, progress_every=stop)
    _same(full, rest)


# ---------------------------------------------------------------- solids
def _solid(tmp_path, material, **solver):
    return {"version": 1, "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"dimensions": [1.0, 0.05, 0.05], "divisions": [8, 1, 2]},
            "materials": {"m": material},
            "sections": {"body": {"type": "solid", "material": "m"}},
            "regions": {"clamp": {"plane": {"point": [0, 0, 0], "normal": [1, 0, 0]}},
                        "tip": {"plane": {"point": [1.0, 0, 0], "normal": [1, 0, 0]}}},
            "constraints": [{"region": "clamp", "dofs": [1, 1, 1]}],
            "loads": [],
            "solver": dict({"backend": "numpy"}, **solver),
            "output": {"every": 10 ** 6, "fields": ["displacement"]}}


def test_a_solid_restarted_in_its_second_step_keeps_its_velocity_and_its_ledger(tmp_path):
    j2 = {"type": "j2", "E": 2.1e11, "nu": 0.3, "rho": 7800.0,
          "hardening": {"source": "linear", "yield_strength": 2.0e8, "H": 1e9}}
    d = _solid(tmp_path, j2, mode="dynamic", total_time=0.002, damping={"type": "none"})
    d["steps"] = [{"name": "a", "loads": [{"region": "tip", "vector": [0, 0, -30000.0]}]},
                  {"name": "b", "loads": [{"region": "tip", "vector": [0, 0, 30000.0]}]}]
    full, half, rest = _split(tmp_path, d, lambda s, step: s.step_index == 1 and s.step >= s.step_start + 200,
                              progress_every=100)
    assert half.step_index == 1
    _same(full, rest)
    e_full, e_rest = full.diagnostics["energy"].history[-1], rest.diagnostics["energy"].history[-1]
    assert e_rest["external_work"] == e_full["external_work"] != 0.0


def test_a_staged_solid_relaxation_resumes_its_level(tmp_path):
    d = _solid(tmp_path, dict(STEEL), mode="relaxation", n_steps=400000, load_steps=4,
               max_steps_per_load=20000, tolerance=1e-5, damping={"type": "kinetic"})
    d["mesh"]["divisions"] = [6, 1, 1]
    d["sections"]["body"]["formulation"] = "full"      # an input-form case says it on its section
    d["loads"] = [{"region": "tip", "vector": [0, 0, -6000.0]}]
    d["output"]["every"] = 1
    full, half, rest = _split(tmp_path, d, lambda s, step: step >= 7000, progress_every=100)
    assert rest.times == full.times[-len(rest.times):] and len(rest.times) < len(full.times)
    _same(full, rest)


def test_eroded_elements_stay_eroded(tmp_path):
    import test_erosion as te
    s, _R = te.bar_case({"damage": 0.9, "strain": 1e-3, "points": 4, "every": 200})
    assert s.n_eroded > 0
    ck = s.checkpoint()
    from vfife.core.mesh import box_mesh
    from vfife.core.solver import Solver
    from vfife.materials import create
    prob = copy.deepcopy(s.p)
    m = dict(te.CDP)
    fresh = Solver(prob, box_mesh((1.0, 0.1, 0.1), (10, 1, 1)), create(m.pop("type"), **m),
                   log=lambda *_: None)
    fresh.restore(ck)
    assert fresh.n_eroded == s.n_eroded
    assert np.array_equal(fresh.blocks[0].pre.active, s.blocks[0].pre.active)
    assert np.array_equal(fresh.frozen, s.frozen)


def test_a_checkpoint_is_a_copy():
    from test_solver import cantilever_case, solve
    held = {}

    def grab(s, step, t, sig):
        if step == 400:
            held["ck"] = s.checkpoint()
            held["copy"] = {k: np.array(v, copy=True) for k, v in held["ck"].items()}
    solve(cantilever_case(n_steps=1000, tol=0.0), on_output=grab, output_every=400, allow_max_steps=True)
    for k, v in held["copy"].items():
        assert np.array_equal(np.asarray(held["ck"][k]), v), k


# ----------------------------------------------------------- fibre beams
def test_a_fibre_member_that_split_stays_split(tmp_path):
    from test_particle_split import _falling_case
    y = _falling_case(str(tmp_path))
    d = yaml.safe_load(open(y, encoding="utf-8"))
    full, half, rest = _split(tmp_path, d, lambda s, local: s.step_index == 1 and local >= 400,
                              progress_every=10)
    assert len(half.splits) == 2          # both ends; the base one since 1.0 re-review
    assert rest.splits == full.splits and rest.fracture.events == full.fracture.events
    _same(full, rest)


def test_a_creep_continuation_keeps_the_concrete_s_memory(tmp_path):
    """From the end state of case_29d as an initial_state, the member goes on
    at the deformation it had crept to -- not back to the elastic value."""
    d = api.load_case(ROOT / "cases" / "case_29d_fiber_creep_bar.yaml")
    first = api.run_case(d, output=str(tmp_path / "a.h5"))
    u_end = first.monitors["u_right"][-1]
    last_age = first.times and d.get("steps")[-1].get("to_age")
    nxt = d.with_changes({"solver.initial_state": str(tmp_path / "a.h5"),
                          "loads[0].amplitude": "held"})
    nxt.data["amplitudes"] = {"held": [[0.0, 1.0], [1.0, 1.0]]}      # the load stays on
    nxt.data["steps"] = [{"name": "hold", "age": float(last_age), "to_age": float(last_age) * 1.05,
                          "load_steps": 2}]
    then = api.run_case(nxt, output=str(tmp_path / "b.h5"))
    assert then.monitors["u_right"][0] == pytest.approx(u_end, rel=1e-3)


def test_starting_from_a_state_stops_the_bonded_bars_too(tmp_path):
    from test_bonded_solver import _case
    c = _case({"type": "bond_ceb", "fc": 25.6e6}, n_steps=300)
    c["output"]["path"] = str(tmp_path / "src.h5")
    p = tmp_path / "src.yaml"
    p.write_text(yaml.safe_dump(c), encoding="utf-8")
    try:
        api.run_case(p)
    except Exception:
        pass
    d = copy.deepcopy(c)
    d["solver"] = {"backend": "numpy", "mode": "dynamic", "total_time": 1e-6,
                   "initial_state": str(tmp_path / "src.h5"), "damping": {"type": "none"}}
    d["output"]["path"] = str(tmp_path / "dyn.h5")
    p2 = tmp_path / "dyn.yaml"
    p2.write_text(yaml.safe_dump(d), encoding="utf-8")
    seen = {}
    api.run_case(p2, on_ready=lambda s, _p: seen.setdefault(
        "v", (np.abs(s.v).max(), np.abs(s.bonded[0].v_bar).max(), s.ke_prev)))
    assert seen["v"] == (0.0, 0.0, 0.0)


def test_a_checkpoint_without_a_loop_position_is_not_resumed(tmp_path):
    """One written before v1.0: resuming it replayed the run from step
    0 on top of its state. It is refused; as an initial_state it still works."""
    import h5py
    d = _cantilever(tmp_path)
    d["solver"] = {"backend": "numpy", "mode": "dynamic", "total_time": 0.01}
    p = _write(tmp_path, d, "old")
    got = {}

    def stop(local, *_):
        return local < 100
    with pytest.raises(AnalysisInterrupted):
        api.run_case(p, progress_every=10, on_progress=stop)
    with h5py.File(tmp_path / "old.ckpt.h5", "a") as f:      # its own file
        del f["checkpoint/loop"]
    with pytest.raises(UserError, match="cannot be resumed"):
        api.run_case(p, output=str(tmp_path / "r.h5"), restart=str(tmp_path / "old.h5"))
    d2 = copy.deepcopy(d)
    d2["solver"]["initial_state"] = str(tmp_path / "old.h5")
    out = api.run_case(_write(tmp_path, d2, "new"))
    assert out.status == "COMPLETED_DYNAMIC"
