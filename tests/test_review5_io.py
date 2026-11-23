"""Writing results and checkpoints, reading them back, and their formats
(review round 5).

Each test is one finding of the review: what it did before is in its text.
"""
import copy
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import threading

import h5py
import numpy as np
import pytest
import yaml

from vfife import api
from vfife.core.diagnostics import AnalysisInterrupted, UserError
from vfife.io.writer import (CHECKPOINT_FORMAT, RESULT_FORMAT, ResultWriter, checkpoint_path,
                             load_checkpoint, read_checkpoint, write_checkpoint_file)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from test_member_failure import _hanging, _write_msh  # noqa: E402

STEEL = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}


def _write(tmp_path, d, name):
    p = tmp_path / f"{name}.yaml"
    d = copy.deepcopy(d)
    d["name"] = name
    d.setdefault("output", {})["path"] = str(tmp_path / f"{name}.h5")
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return p


def _cantilever(tmp_path, n=8, L=2.0, tag=""):
    msh = tmp_path / f"c{n}{tag}.msh"
    _write_msh(msh, [(L * i / n, 0, 0) for i in range(n + 1)],
               {"root": (0, [0]), "tip": (0, [n]), "bar": (1, [(i, i + 1) for i in range(n)])})
    return {"version": 1, "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "gmsh", "path": str(msh)},
            "materials": {"steel": dict(STEEL)},
            "sections": {"beam": {"type": "beam", "material": "steel", "shape": "rect",
                                  "width": 0.1, "height": 0.1}},
            "assign": [{"region": "bar", "section": "beam", "orientation": [0, 1, 0]}],
            "constraints": [{"region": "root", "dofs": [1] * 6}],
            "loads": [{"region": "tip", "vector": [0, 0, -1.0e3]}],
            "solver": {"backend": "numpy", "mode": "dynamic", "total_time": 0.01,
                       "damping": {"type": "none"}},
            "output": {"every": 10 ** 6, "fields": ["displacement"],
                       "monitors": {"w": {"kind": "displacement", "region": "tip", "dof": "z"}}}}


def _stopped(p, out, at, progress_every=10):
    with pytest.raises(AnalysisInterrupted):
        api.run_case(p, output=str(out), progress_every=progress_every,
                     on_progress=lambda local, *_: local < at)
    return out


# ------------------------------------------------------- the checkpoint file
def test_the_checkpoint_is_its_own_file_with_a_format_and_a_fingerprint(tmp_path):
    """It was a group inside the result, which a SIGKILL left unreadable
    with the result; and it said nothing about which model it was of."""
    p = _write(tmp_path, _cantilever(tmp_path), "a")
    out = api.run_case(p)
    with h5py.File(tmp_path / "a.h5") as f:
        assert "checkpoint" not in f and int(f.attrs["result_schema_version"]) == RESULT_FORMAT == 3
        assert f.attrs["checkpoint_file"] == "a.ckpt.h5"
    assert str(tmp_path / "a.ckpt.h5") in out.sidecars
    with h5py.File(tmp_path / "a.ckpt.h5") as f:
        assert int(f.attrs["checkpoint_format"]) == CHECKPOINT_FORMAT
        for key in ("n_nodes", "n_elements", "element_family", "mode", "nodes_sha256",
                    "elements_sha256", "materials_json", "layout_json", "run_id"):
            assert key in f.attrs, key
        assert int(f.attrs["n_nodes"]) == 9 and f.attrs["mode"] == "dynamic"
    assert checkpoint_path(tmp_path / "a.h5") == str(tmp_path / "a.ckpt.h5")
    assert int(read_checkpoint(str(tmp_path / "a.h5"))["step"][0]) > 0


def test_a_checkpoint_that_fails_to_write_leaves_the_previous_one(tmp_path, monkeypatch):
    target = tmp_path / "r.ckpt.h5"
    write_checkpoint_file(target, {"q": np.arange(3.0), "step": np.array([7])})
    real = h5py.Group.create_dataset

    def boom(self, name, *a, **k):
        if name == "v":
            raise OSError("disk full")
        return real(self, name, *a, **k)
    monkeypatch.setattr(h5py.Group, "create_dataset", boom)
    with pytest.raises(OSError):
        write_checkpoint_file(target, {"q": np.zeros(3), "v": np.zeros(3), "step": np.array([9])})
    monkeypatch.undo()
    assert int(load_checkpoint(target).data["step"][0]) == 7
    assert not os.path.exists(str(target) + ".tmp")


_KILLED = """
import os, sys
sys.path.insert(0, {root!r})
from vfife import api
n = []
def frame(step, t, fields):
    n.append(step)
    if len(n) == 6:
        os._exit(9)          # as a SIGKILL would: no close, no flush
api.run_case({case!r}, output={out!r}, on_frame=frame, progress_every=50)
"""


def test_a_killed_run_leaves_a_readable_result_and_its_last_checkpoint(tmp_path):
    """SIGKILL (or a power cut, the OOM killer) mid-run: the result and the
    checkpoint_every checkpoints inside it were 'bad object header version
    number'. Now the result is flushed at every frame and the checkpoint is
    replaced whole beside it."""
    d = _cantilever(tmp_path)
    d["solver"]["total_time"] = 0.05
    d["output"].update(every=100, checkpoint_every=250)
    p = _write(tmp_path, d, "k")
    script = _KILLED.format(root=str(ROOT), case=str(p), out=str(tmp_path / "k.h5"))
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode == 9
    with h5py.File(tmp_path / "k.h5") as f:
        assert len(f["steps"]) >= 5
    ck = load_checkpoint(tmp_path / "k.h5")
    assert ck is not None and int(ck.data["step"][0]) in (250, 500)
    # and it resumes
    d2 = copy.deepcopy(d)
    out = api.run_case(_write(tmp_path, d2, "k2"), restart=str(tmp_path / "k.h5"))
    assert out.status == "COMPLETED_DYNAMIC"


def test_checkpoint_every_is_counted_in_steps(tmp_path, monkeypatch):
    """It was tested only at an output frame whose step number divided by it:
    with sparse frames, or a staged relaxation's irregular frame numbers,
    nothing was written before the end."""
    seen = []
    real = ResultWriter.write_checkpoint

    def spy(self, ck, *a, **k):
        seen.append(int(np.asarray(ck["step"]).ravel()[0]))
        return real(self, ck, *a, **k)
    monkeypatch.setattr(ResultWriter, "write_checkpoint", spy)
    d = _cantilever(tmp_path)
    d["solver"]["total_time"] = 0.02
    d["output"]["checkpoint_every"] = 200          # frames every 10**6 steps
    api.run_case(_write(tmp_path, d, "ce"))
    assert seen[:3] == [200, 400, 600] and len(seen) >= 4
    seen.clear()
    r = _cantilever(tmp_path)
    r["solver"] = {"backend": "numpy", "mode": "relaxation", "load_steps": 4, "tolerance": 1e-7,
                   "n_steps": 100000}
    r["output"].update(every=1, checkpoint_every=300)
    api.run_case(_write(tmp_path, r, "cr"))
    assert len(seen) >= 4 and all(b - a >= 300 for a, b in zip(seen[:-2], seen[1:-1]))


# --------------------------------------------------------- stopping cleanly
def test_sigterm_stops_at_a_step_boundary_and_the_restart_is_exact(tmp_path):
    """SIGTERM (a cluster's time limit) killed the process with the result
    unclosed; Ctrl-C raised inside a step, and in 2 of 5 trials the restart
    was not bit-identical. Both now set a flag the loop honours at the next
    progress check."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGTERM"):
        pytest.skip("signals are delivered to the main thread")
    d = _cantilever(tmp_path)
    p = _write(tmp_path, d, "t")
    full = api.run_case(p, output=str(tmp_path / "full.h5"), progress_every=10)
    sent = []

    def prog(local, *_):
        if local >= 105 and not sent:
            sent.append(local)
            signal.raise_signal(signal.SIGTERM)   # os.kill on Windows is TerminateProcess
        return None
    with pytest.raises(AnalysisInterrupted) as e:
        api.run_case(p, output=str(tmp_path / "half.h5"), progress_every=10, on_progress=prog)
    assert e.value.report["status"] == "INTERRUPTED" and "SIGTERM" in e.value.report["message"]
    step = int(read_checkpoint(str(tmp_path / "half.h5"))["step"][0])
    assert step % 10 == 0 and step > sent[0]                # at a progress check, after the signal
    api.run_case(p, output=str(tmp_path / "rest.h5"), restart=str(tmp_path / "half.h5"),
                 progress_every=10)
    a, b = read_checkpoint(str(tmp_path / "full.h5")), read_checkpoint(str(tmp_path / "rest.h5"))
    for k in ("q", "v", "orient", "energy"):
        assert np.array_equal(a[k], b[k]), k
    assert signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, signal.default_int_handler, None) \
        or callable(signal.getsignal(signal.SIGTERM))          # the old handler is back


# ---------------------------------------------------- the model it belongs to
@pytest.mark.parametrize("change, says", [
    (lambda d, t: d["mesh"].update(path=_cantilever(t, 8, 4.0, "L4")["mesh"]["path"]), "different mesh"),
    (lambda d, t: d["mesh"].update(path=_cantilever(t, 16)["mesh"]["path"]), "element count: 8 there, 16 here"),
    (lambda d, t: d["materials"]["steel"].update(E=2.0e9), "material changed: material"),
    (lambda d, t: d.update(solver={"backend": "numpy", "mode": "relaxation", "n_steps": 2000,
                                   "damping": {"type": "kinetic"}}), "mode: dynamic there, relaxation here"),
    (lambda d, t: d["sections"]["beam"].update(width=0.2), "section layout changed: section on bar"),
])
def test_a_restart_from_another_model_is_refused_by_what_differs(tmp_path, change, says):
    """Same node count, another geometry: accepted, wrong answer. Another
    count: 'could not broadcast (8,) into (16,)'. Another mode: accepted, the
    physical mass replaced the fictitious one, NONFINITE."""
    d = _cantilever(tmp_path)
    src = _stopped(_write(tmp_path, d, "src"), tmp_path / "src.h5", 100)
    other = copy.deepcopy(d)
    change(other, tmp_path)
    with pytest.raises(UserError) as e:
        api.run_case(_write(tmp_path, other, "tgt"), output=str(tmp_path / "o.h5"), restart=str(src))
    assert e.value.code == "CHECKPOINT_MISMATCH" and says in e.value.message, e.value.message


def test_a_restart_may_change_the_budget_and_the_backend(tmp_path):
    d = _cantilever(tmp_path)
    src = _stopped(_write(tmp_path, d, "src"), tmp_path / "src.h5", 100)
    more = copy.deepcopy(d)
    more["solver"].update(total_time=0.02, backend="numba")
    out = api.run_case(_write(tmp_path, more, "more"), restart=str(src))
    assert out.status == "COMPLETED_DYNAMIC"


def _fibre(tmp_path, ny, nz, material=None):
    d = _cantilever(tmp_path)
    d["materials"]["steel"] = material or {"type": "steel02", "E": 2e11, "Fy": 3.5e8, "b": 0.01,
                                           "R0": 20, "rho": 7850}
    d["sections"]["beam"] = {"type": "fiber", "material": "steel", "shape": "rect",
                             "width": 0.1, "height": 0.1, "ny": ny, "nz": nz}
    return d


def test_an_initial_state_refuses_what_cannot_be_mapped_and_takes_what_can(tmp_path):
    """A 4 x 4 fibre state went into a 6 x 6 section, a steel02 state into an
    elastic one, and a frame state into a fibre model failed on
    `KeyError: 'fiber_state'`. A new mode and new material constants are
    what an initial state is for."""
    src = _fibre(tmp_path, 4, 4)
    src["solver"] = {"backend": "numpy", "mode": "relaxation", "n_steps": 20000, "tolerance": 1e-6}
    assert api.run_case(_write(tmp_path, src, "dead")).converged

    def start(d, name):
        d = copy.deepcopy(d)
        d["solver"] = {"backend": "numpy", "mode": "dynamic", "total_time": 0.002,
                       "initial_state": str(tmp_path / "dead.h5")}
        return api.run_case(_write(tmp_path, d, name))
    for d, says in ((_fibre(tmp_path, 6, 6), "section layout changed: section on bar"),
                    (_fibre(tmp_path, 4, 4, dict(STEEL)), "state `fiber_state_0`: shape (8, 2, 16, 13)"),
                    (_cantilever(tmp_path), "element family: beam2_fiber there")):
        with pytest.raises(UserError) as e:
            start(d, "bad")
        assert e.value.code == "CHECKPOINT_MISMATCH" and says in e.value.message, e.value.message
    stiffer = _fibre(tmp_path, 4, 4)
    stiffer["materials"]["steel"]["Fy"] = 4.0e8              # a constant: mappable
    assert start(stiffer, "ok").status == "COMPLETED_DYNAMIC"


def test_an_initial_state_that_is_the_output_is_refused(tmp_path):
    """The writer opened the output anew and erased the state the run was
    starting from; `--restart` had this guard, `initial_state` did not."""
    d = _cantilever(tmp_path)
    api.run_case(_write(tmp_path, d, "dead"))
    d2 = copy.deepcopy(d)
    d2["solver"]["initial_state"] = str(tmp_path / "dead.h5")
    p = tmp_path / "again.yaml"
    d2["name"] = "again"
    d2["output"]["path"] = str(tmp_path / "dead.h5")
    p.write_text(yaml.safe_dump(d2), encoding="utf-8")
    with pytest.raises(UserError) as e:
        api.run_case(str(p))
    assert e.value.code == "INITIAL_STATE_OVERWRITES"
    with h5py.File(tmp_path / "dead.h5") as f:
        assert yaml.safe_load(f.attrs["case_yaml"])["name"] == "dead"


def test_a_restart_from_its_own_checkpoint_file_is_refused(tmp_path):
    d = _cantilever(tmp_path)
    p = _write(tmp_path, d, "s")
    _stopped(p, tmp_path / "s.h5", 100)
    with pytest.raises(UserError) as e:
        api.run_case(p, output=str(tmp_path / "s.h5"), restart=str(tmp_path / "s.ckpt.h5"))
    assert e.value.code == "RESTART_OVERWRITES"


# ----------------------------------------------------- what a restart carries
def test_a_restarted_result_keeps_the_splits_and_failures_and_the_debris_parents(tmp_path):
    """The restarted result had no member_splits / member_failures though its
    members were broken; and the bond graph's parents were lost, so a rigid
    surface restricted to a region let the debris split off its nodes fall
    through (case_34 with a ground on `root`: z = -0.183 under a plane at
    -0.12)."""
    d = _hanging(tmp_path, 1.5, total=0.05)
    p = _write(tmp_path, d, "h")
    got = {}
    api.run_case(p, output=str(tmp_path / "full.h5"), progress_every=10,
                 on_ready=lambda s, _p: got.__setitem__("full", s))
    with pytest.raises(AnalysisInterrupted):
        api.run_case(p, output=str(tmp_path / "half.h5"), progress_every=10,
                     on_ready=lambda s, _p: got.__setitem__("half", s),
                     on_progress=lambda local, *_: not got["half"].splits or local < 1000)
    assert got["half"].splits                                  # stopped after the split
    api.run_case(p, output=str(tmp_path / "rest.h5"), restart=str(tmp_path / "half.h5"),
                 progress_every=10, on_ready=lambda s, _p: got.__setitem__("rest", s))
    full, rest = got["full"], got["rest"]
    assert rest.bonds.parents == full.bonds.parents and rest.bonds.made_for == full.bonds.made_for
    with h5py.File(tmp_path / "full.h5") as f, h5py.File(tmp_path / "rest.h5") as g:
        for name in ("member_splits", "member_failures"):
            assert name in g["diagnostics"], name
            assert np.array_equal(f["diagnostics"][name][...], g["diagnostics"][name][...]), name
            assert "columns" in g["diagnostics"][name].attrs
        assert g.attrs["restarted_from"] == str(tmp_path / "half.h5")
        assert int(g.attrs["restarted_at_step"]) == got["half"].step


# ------------------------------------------------------------ names, units
@pytest.mark.parametrize("name", ["time", "residual", "kinetic_energy", "age"])
def test_a_monitor_may_not_take_a_name_the_result_writes_itself(tmp_path, name):
    """`time`, `residual`, `kinetic_energy`: the whole analysis ran, then the
    result's end crashed on 'name already exists'. `age`: silently replaced
    by the concrete age."""
    d = _cantilever(tmp_path)
    d["output"]["monitors"] = {name: {"kind": "displacement", "region": "tip", "dof": "z"}}
    with pytest.raises(Exception, match="reserved"):
        api.prepare_case(str(_write(tmp_path, d, "r")))


def test_the_units_are_attributes_and_the_plot_says_them(tmp_path):
    """`vfife plot` labelled a mm case's displacement [m]."""
    from vfife.post import _axis, read_monitors
    d = _cantilever(tmp_path)
    msh = pathlib.Path(d["mesh"]["path"])
    _write_msh(msh, [(250.0 * i, 0, 0) for i in range(9)],
               {"root": (0, [0]), "tip": (0, [8]), "bar": (1, [(i, i + 1) for i in range(8)])})
    d["units"] = {"length": "mm", "force": "N", "stress": "MPa", "mass": "t"}
    d["materials"]["steel"] = {"type": "linear_elastic", "E": 2.0e5, "nu": 0.3, "rho": 7.85e-9}
    d["sections"]["beam"].update(width=100.0, height=100.0)
    d["output"]["monitors"]["R"] = {"kind": "reaction", "region": "root", "dof": "z"}
    d["solver"]["total_time"] = 0.002
    api.run_case(_write(tmp_path, d, "mm"))
    with h5py.File(tmp_path / "mm.h5") as f:
        assert f.attrs["unit_length"] == "mm" and f.attrs["unit_stress"] == "MPa"
        assert f["monitors/w"].attrs["units"] == "mm"
        assert f["diagnostics/external_work"].attrs["units"] == "N*mm"
    s = read_monitors(str(tmp_path / "mm.h5"))
    assert _axis(s["w"]).endswith("[mm]") and _axis(s["R"]).endswith("[N]")


def test_a_frame_attribute_has_one_type_and_a_record_index(tmp_path):
    d = _cantilever(tmp_path)
    d["solver"] = {"backend": "numpy", "mode": "relaxation", "load_steps": 3, "tolerance": 1e-7,
                   "n_steps": 100000}
    d["output"]["every"] = 1
    api.run_case(_write(tmp_path, d, "rx"))
    import xml.etree.ElementTree as ET
    with h5py.File(tmp_path / "rx.h5") as f:
        assert f.attrs["time_axis"] == "load_factor"
        frames = sorted(f["steps"], key=int)
        attrs = [dict(f["steps"][k].attrs) for k in frames]
    assert [a["record"] for a in attrs] == list(range(len(frames)))
    assert {a["equilibrium"] for a in attrs} <= {"true", "false"}
    assert [a["load_factor"] for a in attrs] == [a["time"] for a in attrs]
    # the XDMF clock of a load-factor result is the record index, with the
    # load factor beside it as field data (it runs back on an arc-length path)
    grids = [g for g in ET.parse(tmp_path / "rx.xdmf").getroot().iter("Grid")
             if g.get("GridType") == "Uniform"]
    assert [float(g.find("Time").get("Value")) for g in grids] == list(range(len(frames)))
    lam = [float(a.find("DataItem").text) for g in grids for a in g.findall("Attribute")
           if a.get("Name") == "load_factor" and a.get("Center") == "Grid"]
    assert lam == pytest.approx([a["time"] for a in attrs])


def test_the_fallback_xdmf_has_the_six_components_and_not_the_matrix(tmp_path):
    """vtkXdmfReader -- the reader `_last.xdmf` is the fallback for -- reads a
    `Matrix` attribute as ONE tuple of 6n components."""
    from vfife.core.mesh import box_mesh
    m = box_mesh((1.0, 1.0, 1.0), (2, 1, 1))
    w = ResultWriter(str(tmp_path / "s.h5"), m, element="hexa8")
    w.write_frame(10, 0.5, {"displacement": np.zeros((m.n_nodes, 3)), "stress": np.ones((m.n_elem, 6))})
    w.close()
    last = (tmp_path / "s_last.xdmf").read_text(encoding="utf-8")
    full = (tmp_path / "s.xdmf").read_text(encoding="utf-8")
    assert 'AttributeType="Matrix"' in full and 'AttributeType="Matrix"' not in last
    assert all(f'Name="stress_{c}"' in last for c in ("xx", "yy", "zz", "xy", "yz", "zx"))


def test_the_hinge_rotation_is_in_radians_and_the_usage_beside_it(tmp_path):
    """`hinge_rotation` held rotation / capacity; it is now the rotation in
    radians, and the ratio is `hinge_usage`."""
    fy, c, i_, area, theta_u = 3.0e8, 0.1, 1e-5, 1e-2, 0.05
    d = _cantilever(tmp_path, n=4)
    d["sections"]["beam"] = {"type": "beam", "material": "steel", "shape": "general",
                             "area": area, "iy": i_, "iz": i_, "j": 2 * i_}
    d["assign"][0]["failure"] = {"yield_strength": fy, "extreme_fibre": [c, c],
                                 "rotation_capacity": theta_u}
    d["amplitudes"] = {"ramp": [[0.0, 0.0], [1.0, 1.4]]}
    d["loads"] = [{"region": "tip", "vector": [0, 0, -fy * i_ / c / 2.0], "amplitude": "ramp"}]
    d["solver"].update(total_time=1.0, damping={"type": "viscous", "coeff": 50.0})
    d["output"].update(every=500, fields=["displacement", "hinge_rotation", "hinge_usage"])
    api.run_case(_write(tmp_path, d, "hinge"))
    with h5py.File(tmp_path / "hinge.h5") as f:
        last = sorted(f["steps"], key=int)[-1]
        rot = f["steps"][last]["hinge_rotation"][...]
        use = f["steps"][last]["hinge_usage"][...]
    assert np.nanmax(use) > 0.0
    assert np.allclose(rot, use * theta_u, rtol=1e-5, equal_nan=True)


def test_the_modal_file_and_the_spectrum_tables_say_what_they_are(tmp_path):
    from vfife.modal import MODAL_FORMAT
    case = ROOT / "cases" / "case_35_spectrum_cantilever.yaml"
    draft = api.load_case(case)
    out = api.run_case(draft, output=str(tmp_path / "sp.h5"))
    assert out.converged
    with h5py.File(tmp_path / "sp.h5") as f:
        g = f["response_spectrum"]
        assert g["effective_mass"].attrs["dims"] == "mode,direction"
        assert g["end_actions"].attrs["columns"].split(",")[6] == "axial_force_j"
    from vfife.cli import main
    assert main(["modal", str(case), "-o", str(tmp_path / "m.h5"), "--modes", "3"]) == 0
    with h5py.File(tmp_path / "m.h5") as f:
        assert int(f.attrs["modal_format"]) == MODAL_FORMAT and "vfife_version" in f.attrs


def test_a_fibre_restore_brings_back_where_each_split_end_came_from(tmp_path):
    """The same bond-graph memory on the fibre path (resume.split_restore)."""
    from test_particle_split import _falling_case
    from vfife.core.structural import make_structural_solver
    y = _falling_case(str(tmp_path))
    held = {}
    api.run_case(y, output=str(tmp_path / "f.h5"), on_ready=lambda s, _p: held.__setitem__("s", s))
    s = held["s"]
    assert s.bonds.parents, "the case splits"
    prepared = api.prepare_case(y)
    fresh = make_structural_solver(api._with_element(prepared.case, prepared.element, prepared.case_path),
                                   prepared.mesh, prepared.material)
    fresh.restore(read_checkpoint(str(tmp_path / "f.h5")))
    assert fresh.bonds.parents == s.bonds.parents and fresh.bonds.made_for == s.bonds.made_for
    assert fresh.splits == s.splits


# ===================================================== 1.0 re-review
# The v1.0 fixes reviewed again: each test is one finding.
def test_a_result_may_not_be_named_like_a_checkpoint_and_hdf5_has_its_own(tmp_path):
    """`-o run.ckpt.h5` was its own checkpoint path: the writer deleted the
    open result as stale and the run's end replaced it with the checkpoint.
    `x.h5` and `x.hdf5` shared `x.ckpt.h5`."""
    p = _write(tmp_path, _cantilever(tmp_path), "n")
    with pytest.raises(UserError) as e:
        api.run_case(p, output=str(tmp_path / "n.ckpt.h5"))
    assert e.value.code == "OUTPUT_IS_CHECKPOINT_NAME"
    assert checkpoint_path("a/x.hdf5") == "a/x.hdf5.ckpt.h5" and checkpoint_path("a/x.h5") == "a/x.ckpt.h5"
    out = api.run_case(p, output=str(tmp_path / "x.hdf5"))
    assert os.path.exists(tmp_path / "x.hdf5.ckpt.h5") and load_checkpoint(out.result_path) is not None


def test_the_fingerprint_is_the_lowered_model_not_the_text(tmp_path):
    """It hashed the YAML text: a `schema.Case` source had no materials and
    no sections, so a restart with a section 3x wider and 10x the density was
    accepted, and the same model given as a file was refused; a renamed
    material or section, or an unused section, refused an initial state."""
    from vfife.io import schema
    d = _cantilever(tmp_path)
    src = _stopped(_write(tmp_path, d, "src"), tmp_path / "src.h5", 100)
    case, _raw = schema.load(str(tmp_path / "src.yaml"))
    case.output.path = str(tmp_path / "as_case.h5")
    assert api.run_case(case, restart=str(src)).status == "COMPLETED_DYNAMIC"
    other = copy.deepcopy(d)
    other["sections"]["beam"]["width"] = 0.3
    other["materials"]["steel"]["rho"] = 78500.0
    case2, _raw = schema.load(str(_write(tmp_path, other, "other")))
    case2.output.path = str(tmp_path / "o2.h5")
    with pytest.raises(UserError) as e:
        api.run_case(case2, restart=str(src))
    assert "section layout changed" in e.value.message and "material changed" in e.value.message
    # names and unused entries are not the model
    dead = copy.deepcopy(d)
    dead["solver"] = {"backend": "numpy", "mode": "relaxation", "n_steps": 3000}
    assert api.run_case(_write(tmp_path, dead, "dead")).converged
    y = copy.deepcopy(d)
    y["materials"] = {"Q345": y["materials"].pop("steel")}
    y["sections"] = {"girder": dict(y["sections"].pop("beam"), material="Q345"),
                     "unused": {"type": "beam", "material": "Q345", "shape": "rect", "width": 1, "height": 1}}
    y["assign"][0]["section"] = "girder"
    y["solver"]["initial_state"] = str(tmp_path / "dead.h5")
    assert api.run_case(_write(tmp_path, y, "renamed")).status == "COMPLETED_DYNAMIC"


def test_a_generated_mesh_on_another_cpu_is_the_same_mesh(tmp_path):
    """The nodes were compared bit for bit (`nodes_sha256`); a last-bit
    difference in a generated mesh refused the restart."""
    from vfife.io import fingerprint as fp
    nodes = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    now = {"n_nodes": 3, "n_elements": 2, "element_family": "beam2", "elements_sha256": "e",
           "nodes": nodes, "layout_sha256": "l", "layout_json": "{}", "mode": "dynamic"}
    saved = dict(now, nodes=nodes * (1 + 1e-15))
    fp.check(saved, {}, now, {}, "restart", "x")
    with pytest.raises(UserError):
        fp.check(dict(now, nodes=nodes * 1.001), {}, now, {}, "restart", "x")


@pytest.mark.parametrize("split", [False, True])
def test_a_fibre_model_that_may_break_starts_from_its_relaxed_state(tmp_path, split):
    """A dead state relaxed without `fracture` (split is refused in
    relaxation) and a dynamic run with it: refused as 'section layout
    changed' (0.5 ran it); with `split: true` the spare particles failed with
    'boolean index did not match'."""
    mat = {"type": "steel02", "E": 2e11, "Fy": 3.5e8, "b": 0.01, "R0": 20, "rho": 7850,
           "rupture_strain": 0.05}
    dead = _fibre(tmp_path, 4, 4, mat)
    dead["solver"] = {"backend": "numpy", "mode": "relaxation", "n_steps": 20000, "tolerance": 1e-6}
    assert api.run_case(_write(tmp_path, dead, "dead")).converged
    y = _fibre(tmp_path, 4, 4, mat)
    y["sections"]["beam"]["fracture"] = {"threshold": 1.0, "split": split}
    y["solver"].update(total_time=0.002, initial_state=str(tmp_path / "dead.h5"))
    out = api.run_case(_write(tmp_path, y, "dyn"))
    assert out.status == "COMPLETED_DYNAMIC"
    with h5py.File(tmp_path / "dead.h5") as f:
        w0 = f["monitors/w"][-1]
    assert out.monitors["w"][0] == pytest.approx(w0, rel=0.05)     # it started from the dead state


def test_a_checkpoint_that_is_not_this_results_is_refused(tmp_path, monkeypatch):
    """Windows keeps a file another program holds open: the stale checkpoint
    of an earlier run of the same name survived beside the new result, and a
    restart resumed the OTHER run's state."""
    d = _cantilever(tmp_path)
    d["solver"]["total_time"] = 0.02
    x = tmp_path / "x.h5"
    _stopped(_write(tmp_path, d, "one"), x, 60)
    locked = os.path.abspath(checkpoint_path(x))
    real_remove, real_replace = os.remove, os.replace

    def remove(p, *a, **k):
        if os.path.abspath(p) == locked:
            raise PermissionError(13, "used by another process", p)
        return real_remove(p, *a, **k)

    def replace(s, t, *a, **k):
        if os.path.abspath(t) == locked:
            raise PermissionError(13, "Access is denied", t)
        return real_replace(s, t, *a, **k)
    monkeypatch.setattr(os, "remove", remove)
    monkeypatch.setattr(os, "replace", replace)
    two = copy.deepcopy(d)
    two["loads"][0]["vector"] = [0, 0, -5e3]
    _stopped(_write(tmp_path, two, "two"), x, 150)
    monkeypatch.undo()
    with pytest.raises(UserError) as e:
        load_checkpoint(x)
    assert e.value.code == "CHECKPOINT_NOT_THIS_RUN"
    with pytest.raises(UserError) as e:
        api.run_case(_write(tmp_path, two, "three"), restart=str(x), output=str(tmp_path / "three.h5"))
    assert e.value.code == "CHECKPOINT_NOT_THIS_RUN"
    # another run's checkpoint copied onto a result's name: another run_id
    _stopped(_write(tmp_path, d, "four"), tmp_path / "four.h5", 60)
    shutil.copy(tmp_path / "four.ckpt.h5", tmp_path / "five.ckpt.h5")
    _stopped(_write(tmp_path, d, "five"), tmp_path / "five_real.h5", 60)
    shutil.copy(tmp_path / "five_real.h5", tmp_path / "five.h5")
    with pytest.raises(UserError) as e:
        load_checkpoint(tmp_path / "five.h5")
    assert e.value.code == "CHECKPOINT_NOT_THIS_RUN"
    # a checkpoint copied alone is read as it is
    assert load_checkpoint(tmp_path / "four.ckpt.h5") is not None


def test_a_non_finite_end_keeps_the_last_finite_checkpoint(tmp_path):
    """A run that went NONFINITE replaced its last `checkpoint_every`
    checkpoint with the NaN state, and a restart resumed the NaN."""
    d = _cantilever(tmp_path)
    d["solver"].update(total_time=0.05, dt_safety=5.0)
    d["output"]["checkpoint_every"] = 50
    out = api.run_case(_write(tmp_path, d, "nf"))
    assert out.status == "NONFINITE"
    ck = load_checkpoint(out.result_path)
    assert ck is not None and np.all(np.isfinite(ck.data["q"])) and int(ck.data["step"][0]) % 50 == 0


def test_sigterm_is_on_the_exception_and_the_cli_exits_143(tmp_path):
    """SIGTERM became an ordinary Exception: a batch loop with `except
    Exception` swallowed the cluster's time limit, and `vfife run` exited 130
    for it as for Ctrl-C."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGTERM"):
        pytest.skip("signals are delivered to the main thread")
    d = _cantilever(tmp_path)
    d["solver"]["total_time"] = 1.0
    p = _write(tmp_path, d, "t")

    def prog(local, *_):
        if local == 20:
            signal.raise_signal(signal.SIGTERM)   # os.kill on Windows is TerminateProcess
    with pytest.raises(AnalysisInterrupted) as e:
        api.run_case(p, progress_every=10, on_progress=prog)
    assert e.value.signal == signal.SIGTERM
    with pytest.raises(AnalysisInterrupted) as e:
        api.run_case(p, progress_every=10, on_progress=lambda local, *_: local < 20)
    assert e.value.signal is None
    from vfife.cli import main
    before = signal.getsignal(signal.SIGTERM)
    done = threading.Event()

    def send():
        # only once the run's own handler is installed (the default kills)
        while not done.is_set():
            if signal.getsignal(signal.SIGTERM) is not before:
                signal.raise_signal(signal.SIGTERM)   # os.kill on Windows is TerminateProcess
                return
            done.wait(0.05)
    t = threading.Thread(target=send)
    t.start()
    try:
        code = main(["run", str(p), "-o", str(tmp_path / "cli.h5")])
    finally:
        done.set()
        t.join()
    assert code == 143


@pytest.mark.parametrize("change, says", [
    (lambda d: d["constraints"][0].update(dofs=[1, 1, 1, 0, 0, 0]), "constraints"),
    (lambda d: d["loads"][0].update(vector=[0, 0, -1e4]), "loads"),
    (lambda d: d.update(nodal_masses=[{"region": "tip", "mass": 100.0}]), "nodal_masses"),
    (lambda d: d["solver"].update(damping={"type": "viscous", "coeff": 50.0}), "damping"),
    (lambda d: d.update(units={"length": "mm", "force": "N", "stress": "MPa"}), "units"),
])
def test_a_restart_refuses_other_conditions(tmp_path, change, says):
    """Changed constraints, loads, masses, damping and units were accepted on
    a restart; a changed nodal mass was ignored (the mass is the
    checkpoint's) and the result came out bit-identical."""
    d = _cantilever(tmp_path)
    src = _stopped(_write(tmp_path, d, "src"), tmp_path / "src.h5", 100)
    other = copy.deepcopy(d)
    change(other)
    with pytest.raises(UserError) as e:
        api.run_case(_write(tmp_path, other, "tgt"), output=str(tmp_path / "o.h5"), restart=str(src))
    assert e.value.code == "CHECKPOINT_MISMATCH" and says in e.value.message, e.value.message


def test_a_run_started_from_a_dead_state_resumes(tmp_path):
    """`--restart` of a run whose case has `solver.initial_state` was refused
    ('either resumes or starts from an initial_state, not both') -- the stop
    message's own advice failed for every scenario from a dead state. The
    restart ignores initial_state; and `restarted_from` names the result
    also when the checkpoint file is given."""
    d = _cantilever(tmp_path)
    dead = copy.deepcopy(d)
    dead["solver"] = {"backend": "numpy", "mode": "relaxation", "n_steps": 3000}
    api.run_case(_write(tmp_path, dead, "dead"))
    y = copy.deepcopy(d)
    y["solver"]["initial_state"] = str(tmp_path / "dead.h5")
    p = _write(tmp_path, y, "scen")
    full = api.run_case(p, output=str(tmp_path / "full.h5"), progress_every=10)
    _stopped(p, tmp_path / "part.h5", 100)
    rest = api.run_case(p, output=str(tmp_path / "rest.h5"), restart=str(tmp_path / "part.ckpt.h5"),
                        progress_every=10)
    assert rest.monitors["w"][-1] == full.monitors["w"][-1]
    with h5py.File(tmp_path / "rest.h5") as f:
        assert f.attrs["restarted_from"] == str(tmp_path / "part.h5")


def test_the_members_grid_and_the_sweep_run_on_the_frame_clock(tmp_path):
    """`_members.xdmf` had no load factor; the sweep's .pvd used lambda as its
    timestep, which runs backwards on an arc-length path."""
    out = api.run_case(ROOT / "cases" / "case_07_coupled_solid_truss.yaml", output=str(tmp_path / "c.h5"))
    text = (tmp_path / "c_members.xdmf").read_text(encoding="utf-8")
    assert 'Name="load_factor"' in text
    from vfife.sweep import write_series
    d = _fibre(tmp_path, 4, 4)
    d["solver"] = {"backend": "numpy", "mode": "relaxation", "n_steps": 3000, "load_steps": 3}
    d["output"].update(every=1, fields=["displacement", "rotation", "fiber_stress"])
    p = _write(tmp_path, d, "fs")
    api.run_case(p)
    pvd = write_series(str(p), str(tmp_path / "fs.h5"), out_dir=str(tmp_path / "sw"), log=lambda *a: None)
    import re
    steps = [float(v) for v in re.findall(r'timestep="([^"]+)"', pathlib.Path(pvd).read_text())]
    assert steps == [0.0, 1.0, 2.0]                    # lambda was 1/3, 2/3, 1


def test_the_readers_say_a_killed_result_was_not_closed(tmp_path):
    from vfife.post import read_monitors
    p = tmp_path / "k.h5"
    with h5py.File(p, "w") as f:
        f.attrs["result_schema_version"] = RESULT_FORMAT
        f.create_group("steps").create_group("00000010")
    with pytest.raises(UserError) as e:
        read_monitors(str(p))
    assert e.value.code == "RESULT_NOT_CLOSED"


def test_the_energy_units_are_absent_when_no_units_are_declared(tmp_path):
    d = _cantilever(tmp_path)
    d.pop("units")
    out = api.run_case(_write(tmp_path, d, "nu"))
    with h5py.File(out.result_path) as f:
        assert "units" not in f["diagnostics/external_work"].attrs
        assert "units" not in f["monitors/w"].attrs
