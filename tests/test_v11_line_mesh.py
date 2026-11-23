"""1.1, item 5: a line mesh written in the case (`mesh: {type: lines}`).

What it builds must be what a Gmsh Line2 file gives, so a case moved from
one to the other runs the same: case_03 (a beam and a bar) and case_22 (a
40-element cable with an unstressed length) written both ways give the same
monitors. Then the geometry -- divisions and sizes, a parabola and a circle
through their ends at their rise -- and the rules: named points are nodes,
two named points at one place are two nodes, and what is ambiguous or
meaningless is refused by name.
"""
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from vfife.api import run_case
from vfife.io import line_mesh, schema

ROOT = Path(__file__).resolve().parents[1]


def _mesh(**spec):
    return line_mesh.build(schema.MeshSpec(type="lines", **spec))


def _monitors(path):
    with h5py.File(path) as f:
        return {k: float(np.asarray(v)[-1]) for k, v in f["monitors"].items()
                if k not in ("time", "residual", "kinetic_energy")}


def _both_ways(tmp_path, stem, lines_mesh):
    d = yaml.safe_load((ROOT / "cases" / f"{stem}.yaml").read_text(encoding="utf-8"))
    d["mesh"]["path"] = str((ROOT / "cases" / d["mesh"]["path"]).resolve())
    gmsh_case, lines_case = tmp_path / "gmsh.yaml", tmp_path / "lines.yaml"
    d["output"]["path"] = str(tmp_path / "gmsh.h5")
    gmsh_case.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    d["mesh"] = lines_mesh
    d["output"]["path"] = str(tmp_path / "lines.h5")
    lines_case.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    a, b = run_case(str(gmsh_case)), run_case(str(lines_case))
    assert a.converged and b.converged
    return _monitors(tmp_path / "gmsh.h5"), _monitors(tmp_path / "lines.h5")


def test_case_03_written_as_lines_runs_as_its_gmsh_file(tmp_path):
    g, l = _both_ways(tmp_path, "case_03_mixed_frame", {
        "type": "lines",
        "points": {"fixed": [0, 0, 0], "corner": [1, 0, 0], "loaded": [1, 0, 1]},
        "lines": {"beam_members": {"through": ["fixed", "corner"]},
                  "truss_members": {"through": ["corner", "loaded"]}}})
    for k in g:
        assert l[k] == pytest.approx(g[k], rel=1e-9, abs=1e-12), k


def test_case_22_s_cable_written_as_lines_hangs_as_its_gmsh_file(tmp_path):
    g, l = _both_ways(tmp_path, "case_22_cable_catenary", {
        "type": "lines",
        "points": {"left": [0, 0, 0], "mid": [50, 0, 0], "right": [100, 0, 0]},
        "lines": {"cable": {"through": ["left", "mid", "right"], "size": 2.5}}})
    for k in g:
        assert l[k] == pytest.approx(g[k], rel=1e-5, abs=1e-9 * max(abs(v) for v in g.values())), k


def test_divisions_size_and_shared_points():
    m = _mesh(points={"A": [0, 0, 0], "B": [10, 0, 0], "P": [10, 0, -6]},
              lines={"girder": {"through": ["A", "B", [25, 0, 0]], "size": 2.0},
                     "pier": {"through": ["P", "B"], "divisions": 3}})
    girder = m.element_regions["girder"]["truss2"]
    assert len(girder) == 5 + 8                     # 10 m in 2 m, 15 m in ceil(7.5)
    assert len(m.element_regions["pier"]["truss2"]) == 3
    B = int(m.regions["B"][0])
    assert B in m.regions["girder"] and B in m.regions["pier"]       # the pier meets the girder at B
    assert m.nodes.shape == (3 + 1 + 4 + 7 + 2, 3)
    L = np.linalg.norm(m.nodes[m.elements[:, 1]] - m.nodes[m.elements[:, 0]], axis=1)
    np.testing.assert_allclose(sorted(L[girder])[:5], 15.0 / 8)
    # a member chain from A, in order
    first = m.elements[girder[0]]
    assert first[0] == int(m.regions["A"][0])


def test_lines_meet_only_at_named_points_and_groups_join_lines():
    """Two points at one place, a line over part of another, a line attached
    at another's division node: refused in the mesh's terms (preflight said
    DUPLICATE_NODE in Gmsh's words, or ran two members side by side). A part
    of a girder is its own line, and `groups` makes an element region of
    several lines, as a Gmsh group of several pieces is."""
    with pytest.raises(ValueError, match="at one place"):
        _mesh(points={"G": [0, 0, 0], "T": [0, 0, 0], "E": [5, 0, 0]},
              lines={"deck": {"through": ["G", "E"]}, "pier": {"through": [[0, 0, -5], "T"]}})
    pts = {"A": [0, 0, 0], "B": [10, 0, 0], "C": [20, 0, 0]}
    with pytest.raises(ValueError, match="side by side"):
        _mesh(points=pts, lines={"girder": {"through": ["A", "B", "C"]}, "main": {"through": ["B", "C"]}})
    with pytest.raises(ValueError, match="without sharing it"):
        _mesh(points=pts, lines={"girder": {"through": ["A", "B", "C"], "divisions": 2},
                                 "main": {"through": ["B", "C"], "divisions": 2}})
    with pytest.raises(ValueError, match="without sharing it"):
        _mesh(points=pts, lines={"girder": {"through": ["A", "C"], "divisions": 2},
                                 "hanger": {"through": [[10, 0, 5], [10, 0, 0]]}})
    m = _mesh(points=pts, lines={"side": {"through": ["A", "B"], "divisions": 2},
                                 "main": {"through": ["B", "C"], "divisions": 3}},
              groups={"girder": ["side", "main"]})
    np.testing.assert_array_equal(m.element_regions["girder"]["truss2"], np.arange(5))
    assert len(m.regions["girder"]) == 6


@pytest.mark.parametrize("shape", ["parabola", "circle"])
def test_an_arc_rises_to_its_rise_between_its_ends(shape):
    m = _mesh(points={"A": [0, 0, 0], "C": [60, 0, 0]},
              lines={"arch": {"through": ["A", "C"], "divisions": 24,
                              "arc": {"rise": 12.0, "shape": shape}}})
    x = m.nodes[m.regions["arch"]]
    assert x[:, 2].max() == pytest.approx(12.0, rel=1e-12)
    top = x[np.argmax(x[:, 2])]
    assert top[0] == pytest.approx(30.0)
    if shape == "parabola":
        np.testing.assert_allclose(x[:, 2], 4 * 12.0 * x[:, 0] / 60 * (1 - x[:, 0] / 60), atol=1e-9)
    else:
        R = (30.0 ** 2 + 12.0 ** 2) / 24.0
        np.testing.assert_allclose(np.hypot(x[:, 0] - 30.0, x[:, 2] - (12.0 - R)), R, rtol=1e-12)
        L = np.linalg.norm(np.diff(x[np.argsort(x[:, 0])], axis=0), axis=1)
        np.testing.assert_allclose(L, L[0], rtol=1e-12)              # evenly by angle


def test_what_is_refused():
    with pytest.raises(Exception, match="not in mesh.points"):
        schema.MeshSpec(type="lines", lines={"a": {"through": ["X", [1, 0, 0]]}})
    with pytest.raises(Exception, match="both a point and a line"):
        schema.MeshSpec(type="lines", points={"a": [0, 0, 0]}, lines={"a": {"through": ["a", [1, 0, 0]]}})
    with pytest.raises(Exception, match="`divisions` or `size`"):
        schema.MeshSpec(type="lines", lines={"a": {"through": [[0, 0, 0], [1, 0, 0]], "divisions": 2,
                                                   "size": 0.5}})
    with pytest.raises(Exception, match="two points"):
        schema.MeshSpec(type="lines", lines={"a": {"through": [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                                                   "arc": {"rise": 1.0}}})
    with pytest.raises(Exception, match="for type 'lines'"):
        schema.MeshSpec(type="box", dimensions=[1, 1, 1], divisions=[1, 1, 1], lines={})
    with pytest.raises(ValueError, match="on no line"):
        _mesh(points={"A": [0, 0, 0], "B": [1, 0, 0], "lost": [5, 5, 5]}, lines={"a": {"through": ["A", "B"]}})
    with pytest.raises(ValueError, match="same node"):
        _mesh(points={"A": [0, 0, 0], "B": [1, 0, 0]}, lines={"a": {"through": ["A", [0, 0, 0], "B"]}})
    with pytest.raises(Exception, match="every mesh has"):
        schema.MeshSpec(type="lines", points={"all": [0, 0, 0]}, lines={"a": {"through": ["all", [1, 0, 0]]}})
    with pytest.raises(Exception, match="not in mesh.lines"):
        schema.MeshSpec(type="lines", lines={"a": {"through": [[0, 0, 0], [1, 0, 0]]}}, groups={"g": ["b"]})
    with pytest.raises(Exception, match="finite"):
        schema.MeshSpec(type="lines", points={"A": [0, float("nan"), 0]}, lines={"a": {"through": ["A", [1, 0, 0]]}})
    with pytest.raises(ValueError, match="mesh.lines.arch: the arc's `up`"):
        _mesh(points={"A": [0, 0, 0], "C": [10, 0, 0]},
              lines={"arch": {"through": ["A", "C"], "divisions": 4, "arc": {"rise": 1.0, "up": [1, 0, 0]}}})


def test_a_case_region_may_not_take_a_point_s_name(tmp_path):
    from vfife.api import prepare_case
    from vfife.core.diagnostics import UserError
    d = yaml.safe_load((ROOT / "cases" / "case_03_mixed_frame.yaml").read_text(encoding="utf-8"))
    d["mesh"] = {"type": "lines", "points": {"fixed": [0, 0, 0], "corner": [1, 0, 0], "loaded": [1, 0, 1]},
                 "lines": {"beam_members": {"through": ["fixed", "corner"]},
                           "truss_members": {"through": ["corner", "loaded"]}}}
    d["regions"] = {"corner": {"where": {"z": 0.0}}}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    with pytest.raises(UserError) as exc:
        prepare_case(str(p))
    assert exc.value.code == "REGION_NAME_CLASH"


def test_a_load_on_a_name_the_mesh_does_not_have_is_refused_before_the_run(tmp_path):
    d = yaml.safe_load((ROOT / "cases" / "case_03_mixed_frame.yaml").read_text(encoding="utf-8"))
    d["mesh"] = {"type": "lines", "points": {"fixed": [0, 0, 0], "corner": [1, 0, 0], "tip": [1, 0, 1]},
                 "lines": {"beam_members": {"through": ["fixed", "corner"]},
                           "truss_members": {"through": ["corner", "tip"]}}}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    with pytest.raises(Exception, match="unknown region 'loaded'"):
        schema.load(str(p))
