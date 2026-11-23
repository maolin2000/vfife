"""Reading results back, as a second reader found it: the
number the user reads must be the number the solver computed. One test per
reproduced defect."""
import copy
import pathlib
import re
import sys

import h5py
import numpy as np
import pytest
import yaml

from vfife import api, post, verification
from vfife.core.diagnostics import AnalysisInterrupted, UserError

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from test_member_failure import _write_msh  # noqa: E402

STEEL = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}


def _write(tmp_path, d, name):
    d = copy.deepcopy(d)
    d["name"] = name
    d.setdefault("output", {})["path"] = str(tmp_path / f"{name}.h5")
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return p


def _cantilever(tmp_path, n=4, L=2.0):
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
            "solver": {"backend": "numpy", "mode": "static", "load_steps": 1},
            "output": {"every": 1, "fields": ["displacement", "bending_y", "end_actions"],
                       "monitors": {"w": {"kind": "displacement", "region": "tip", "dof": "z"},
                                    "R": {"kind": "reaction", "region": "root", "dof": "z"},
                                    "R2": {"kind": "reaction", "regions": ["root", "bar"], "dof": "z"}}}}


# ------------------------------------------------------------------- XDMF
def test_a_stress_tensor_is_not_labelled_in_an_order_it_is_not_in(tmp_path):
    from vfife.core.mesh import Mesh
    from vfife.io.writer import ResultWriter
    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                      [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    mesh = Mesh(nodes, np.arange(8, dtype=np.int32).reshape(1, 8), {"all": np.arange(8)})
    w = ResultWriter(str(tmp_path / "t.h5"), mesh, element="hexa8")
    w.write_frame(0, 0.0, {"displacement": np.zeros((8, 3)),
                           "stress": np.array([[11., 22., 33., 12., 23., 13.]])})
    w.close()
    x = (tmp_path / "t.xdmf").read_text(encoding="utf-8")
    assert "Tensor6" not in x
    for comp, col in (("xx", 0), ("yy", 1), ("zz", 2), ("xy", 3), ("yz", 4), ("zx", 5)):
        m = re.search(rf'Name="stress_{comp}".*?Format="XML">0 (\d) 1 1', x, re.S)
        assert m and int(m.group(1)) == col, comp


def test_an_element_field_is_on_the_elements_when_they_number_the_nodes(tmp_path):
    _write_msh(tmp_path / "tri.msh", [(0, 0, 0), (2, 0, 0), (1, 0, 1)],
               {"a": (0, [0]), "b": (0, [1]), "apex": (0, [2]), "bars": (1, [(0, 1), (1, 2), (2, 0)])})
    d = {"version": 1, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": str(tmp_path / "tri.msh")}, "materials": {"s": dict(STEEL)},
         "sections": {"bar": {"type": "bar", "material": "s", "shape": "general", "area": 1e-4}},
         "assign": [{"region": "all", "section": "bar"}],
         "constraints": [{"region": "a", "dofs": [1, 1, 1]}, {"region": "b", "dofs": [0, 1, 1]},
                         {"region": "apex", "dofs": [0, 1, 0]}],
         "loads": [{"region": "apex", "vector": [0, 0, -1000.0]}],
         "solver": {"mode": "relaxation", "n_steps": 20000, "tolerance": 1e-8},
         "output": {"every": 100000, "fields": ["displacement", "axial_force"],
                    "monitors": {"w": {"kind": "displacement", "region": "apex", "dof": "z"},
                                 "R": {"kind": "reaction", "regions": ["a", "b"], "dof": "z"},
                                 "R2": {"kind": "reaction", "regions": ["a", "b", "all"], "dof": "z"}}}}
    out = api.run_case(_write(tmp_path, d, "tri"))
    x = (tmp_path / "tri.xdmf").read_text(encoding="utf-8")
    assert re.search(r'Name="axial_force" AttributeType="Scalar" Center="Cell"', x)
    assert re.search(r'Name="displacement" AttributeType="Vector" Center="Node"', x)
    # overlapping regions: every node once
    assert out.monitors["R2"][-1] == pytest.approx(out.monitors["R"][-1], rel=1e-6)   # was 2x


def test_a_split_member_is_drawn_on_its_own_particles(tmp_path):
    from test_particle_split import _falling_case
    d = yaml.safe_load(open(_falling_case(str(tmp_path)), encoding="utf-8"))
    d["output"] = {"every": 400, "fields": ["displacement"], "monitors": {}}
    api.run_case(_write(tmp_path, d, "fall"))
    x = (tmp_path / "fall_last.xdmf").read_text(encoding="utf-8")
    assert re.search(r"fall\.h5:/steps/\d+/connectivity", x)
    with h5py.File(tmp_path / "fall.h5") as f:
        last = sorted(f["steps"], key=int)[-1]
        assert f["steps"][last]["connectivity"].dtype.kind == "i"


# --------------------------------------------------------- restart, series
def test_a_stopped_and_resumed_run_loses_no_frame(tmp_path):
    d = _cantilever(tmp_path)
    d["solver"] = {"backend": "numpy", "mode": "dynamic", "total_time": 0.01,
                   "damping": {"type": "none"}}
    d["output"]["every"] = 100
    p = _write(tmp_path, d, "dyn")
    full = api.run_case(p, output=str(tmp_path / "full.h5"), progress_every=100)

    def stop(local, *_):
        return local < 200
    with pytest.raises(AnalysisInterrupted):
        api.run_case(p, output=str(tmp_path / "part.h5"), progress_every=100, on_progress=stop)
    with pytest.raises(UserError, match="write over the result"):
        api.run_case(p, output=str(tmp_path / "part.h5"), restart=str(tmp_path / "part.h5"))
    rest = api.run_case(p, output=str(tmp_path / "rest.h5"), restart=str(tmp_path / "part.h5"))
    part = post.read_monitors(str(tmp_path / "part.h5"))["time"].values.tolist()
    assert part + rest.times == pytest.approx(full.times)


def test_the_convergence_series_keep_their_own_axis(tmp_path):
    d = _cantilever(tmp_path)
    d["solver"] = {"backend": "numpy", "mode": "dynamic", "total_time": 0.01}
    d["output"]["every"] = 100
    out = api.run_case(_write(tmp_path, d, "s"), progress_every=50)
    series = post.read_monitors(out.result_path)
    assert len(series["residual"].values) != len(series["time"].values)
    csv = post.to_csv(out.result_path, str(tmp_path / "all.csv"))
    head = open(csv, encoding="utf-8").readline().strip().split(",")
    assert "residual" not in head and head[0] == "time"
    conv = post.to_csv(out.result_path, str(tmp_path / "r.csv"), ["residual"])
    assert open(conv, encoding="utf-8").readline().strip() == "check,residual"
    with pytest.raises(UserError, match="progress checks"):
        post.to_csv(out.result_path, str(tmp_path / "x.csv"), ["w", "residual"])


# ---------------------------------------------------------------- contracts
def _metric(result, spec):
    with h5py.File(result) as h5:
        return verification._metric_value(h5, spec)


def test_nearest_on_an_element_field_finds_the_element(tmp_path):
    out = api.run_case(_write(tmp_path, _cantilever(tmp_path), "m"))
    # |M_y| at the element whose centre is nearest: P (L - x_end) over the
    # element at 1.75 m is 500 at its root end, over the one at 0.25 m 2000
    tip = _metric(out.result_path, {"source": "field", "name": "bending_y",
                                    "location": {"nearest": [1.9, 0, 0]}, "reduction": "final"})
    root = _metric(out.result_path, {"source": "field", "name": "bending_y",
                                     "location": {"nearest": [0.1, 0, 0]}, "reduction": "final"})
    assert abs(tip) == pytest.approx(500.0, rel=1e-3) and abs(root) == pytest.approx(2000.0, rel=1e-3)
    assert out.monitors["R2"][-1] == pytest.approx(out.monitors["R"][-1], rel=1e-6)   # was 2x


def test_a_vector_field_needs_a_component_to_be_final(tmp_path):
    out = api.run_case(_write(tmp_path, _cantilever(tmp_path), "v"))
    with pytest.raises(ValueError, match="name a `component`"):
        _metric(out.result_path, {"source": "field", "name": "displacement", "reduction": "final"})


def test_a_contract_reads_the_members_of_a_coupled_model(tmp_path):
    out = api.run_case(api.load_case(ROOT / "cases" / "case_07_coupled_solid_truss.yaml"),
                       output=str(tmp_path / "c07.h5"))
    v = _metric(out.result_path, {"source": "field", "name": "axial_force", "reduction": "max"})
    assert v > 0.0
    from vfife.post import view
    parts = view.load(out.result_path)["meta"]["acts"][0]
    assert any(f["on"] == "elements" for f in parts["fields"])


def test_a_diagnostic_is_the_energy_audit_not_the_progress_series(tmp_path):
    d = _cantilever(tmp_path)
    d["solver"] = {"backend": "numpy", "mode": "dynamic", "total_time": 0.01}
    out = api.run_case(_write(tmp_path, d, "e"), progress_every=37)
    with h5py.File(out.result_path) as h5:
        audit = np.asarray(h5["diagnostics/kinetic_energy"])
    v = _metric(out.result_path, {"source": "diagnostic", "name": "kinetic_energy", "reduction": "final"})
    assert v == audit[-1]
