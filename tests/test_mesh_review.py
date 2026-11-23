"""Mesh import, regions and units as a second reader found them.
One test per reproduced defect."""
import pathlib
import sys

import numpy as np
import pytest

from vfife import api
from vfife.core.diagnostics import UserError
from vfife.io import gmsh

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mshwrite import hex_bar, write22, write41  # noqa: E402

STEEL = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.0, "rho": 7850.0}


def _solid_case(msh, **over):
    d = {"version": 1, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": str(msh)},
         "materials": {"s": dict(STEEL)},
         "sections": {"body": {"type": "solid", "material": "s"}},
         "assign": [{"region": "all", "section": "body"}],
         "constraints": [{"region": "bottom", "dofs": [1, 1, 1]}],
         "loads": [{"region": "top", "traction": [0, 0, -1e6]}],
         "solver": {"mode": "relaxation", "backend": "numpy", "n_steps": 40000,
                    "tolerance": 1e-8, "damping": {"type": "kinetic"}},
         "output": {"every": 10 ** 6, "fields": ["displacement"],
                    "monitors": {"w": {"kind": "displacement", "region": "top", "dof": "z"}}}}
    d.update(over)
    return d


def _codes(tmp_path, d):
    p = api.prepare_case(api.CaseDraft(d, base=tmp_path))
    return {i["code"] for i in p.preflight["issues"] if i["severity"] in ("ERROR", "WARNING")}


# ------------------------------------------------------------------ reader
@pytest.mark.parametrize("fmt", ["22", "41"])
def test_a_physical_tag_repeated_across_dimensions_is_two_groups(tmp_path, fmt):
    """Physical Volume(1) "body" and Physical Surface(1) "top": the volume
    vanished and "top" took every node and both hexes."""
    nodes, hexes = hex_bar(2)
    top, bot = [9, 10, 11, 12], [1, 2, 3, 4]
    names = [(3, 1, "body"), (2, 1, "top"), (2, 2, "bottom")]
    f = tmp_path / f"m{fmt}.msh"
    if fmt == "22":
        write22(f, nodes, [(5, 1, h) for h in hexes] + [(3, 1, top), (3, 2, bot)], names)
    else:
        write41(f, nodes, [(3, 1, 5, hexes), (2, 1, 3, [top]), (2, 2, 3, [bot])],
                names, {(3, 1): [1], (2, 1): [1], (2, 2): [2]})
    m, _ = gmsh.read(str(f))
    assert sorted(m.regions["top"]) == [8, 9, 10, 11]
    assert len(m.regions["body"]) == 12
    assert sorted(m.element_regions["body"]["hexa8"]) == [0, 1]


def test_a_22_cell_listed_under_two_groups_is_in_both(tmp_path):
    nodes, hexes = hex_bar(3)
    f = tmp_path / "g.msh"
    write22(f, nodes, [(5, 1, hexes[0]), (5, 1, hexes[1]), (5, 2, hexes[1]), (5, 1, hexes[2])],
            [(3, 1, "body"), (3, 2, "zone")])
    m, _ = gmsh.read(str(f))
    assert m.n_elem == 3
    assert sorted(m.element_regions["body"]["hexa8"]) == [0, 1, 2]
    assert sorted(m.element_regions["zone"]["hexa8"]) == [1]


def _plate():
    return [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)]


def test_triangles_left_in_a_quad_mesh_are_refused(tmp_path):
    f = tmp_path / "mix.msh"
    write22(f, _plate(), [(3, 1, [1, 2, 5, 4]), (2, 1, [2, 3, 6]), (2, 1, [2, 6, 5])], [(2, 1, "plate")])
    with pytest.raises(UserError, match="quadrilateral and 2 triangular"):
        gmsh.read(str(f))


def test_a_triangle_plate_with_a_support_curve_is_a_shell_mesh(tmp_path):
    f = tmp_path / "tri.msh"
    write22(f, _plate(), [(2, 1, [1, 2, 5]), (2, 1, [1, 5, 4]), (2, 1, [2, 3, 6]), (2, 1, [2, 6, 5]),
                          (1, 2, [1, 4])], [(2, 1, "plate"), (1, 2, "edge")])
    m, element = gmsh.read(str(f))
    assert element == "shell3" and m.n_elem == 4
    assert sorted(m.regions["edge"]) == [0, 3]


@pytest.mark.parametrize("problem, match", [
    ("missing", "does not define"),
    ("duplicate", "more than once"),
])
def test_a_malformed_node_section_is_refused(tmp_path, problem, match):
    nodes, hexes = hex_bar(1)
    tags = list(range(1, 9))
    if problem == "missing":
        hexes = [[1, 2, 3, 4, 5, 6, 7, 99]]
    else:
        tags[7] = 7
    f = tmp_path / "bad.msh"
    write22(f, nodes, [(5, 1, hexes[0])], [(3, 1, "body")], tags=tags)
    with pytest.raises(UserError, match=match):
        gmsh.read(str(f))


def test_parametric_nodes_are_refused_by_name(tmp_path):
    nodes, hexes = hex_bar(1)
    f = tmp_path / "par.msh"
    write41(f, nodes, [(3, 1, 5, hexes)], [(3, 1, "body")], {(3, 1): [1]}, param=True)
    with pytest.raises(UserError, match="parametric"):
        gmsh.read(str(f))


def test_an_inverted_pyramid_is_turned_over_at_import():
    x = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0.5, 1)], float)
    conn = gmsh._fix_orientation(x, np.array([[0, 3, 2, 1, 4]]), "pyramid5")
    n = np.cross(x[conn[0, 1]] - x[conn[0, 0]], x[conn[0, 3]] - x[conn[0, 0]])
    assert n @ (x[conn[0, 4]] - x[conn[0, 0]]) > 0


# ------------------------------------------------------------ the models
def test_a_traction_on_a_multi_section_solid_runs(tmp_path):
    """A pad between two concrete blocks: the traction was looked for in
    each block, and the pad (no loaded face) raised."""
    nodes, hexes = hex_bar(3, L=3.0)
    f = tmp_path / "sand.msh"
    write22(f, nodes, [(5, 1, hexes[0]), (5, 2, hexes[1]), (5, 1, hexes[2]),
                       (3, 3, [1, 2, 3, 4]), (3, 4, [13, 14, 15, 16])],
            [(3, 1, "concrete"), (3, 2, "pad"), (2, 3, "bottom"), (2, 4, "top")])
    d = _solid_case(f)
    d["materials"] = {"stiff": {"type": "linear_elastic", "E": 3e10, "nu": 0.0, "rho": 2500.0},
                      "soft": {"type": "linear_elastic", "E": 3e8, "nu": 0.0, "rho": 1500.0}}
    d["sections"] = {"pad": {"type": "solid", "material": "soft"},
                     "concrete": {"type": "solid", "material": "stiff"}}
    d["assign"] = [{"region": "concrete", "section": "concrete"}, {"region": "pad", "section": "pad"}]
    out = api.run_case(api.CaseDraft(d, base=tmp_path), output=str(tmp_path / "s.h5"))
    assert out.monitors["w"][-1] == pytest.approx(-1e6 * (2 / 3e10 + 1 / 3e8), rel=1e-5)


def test_a_case_region_may_not_take_a_physical_group_s_name(tmp_path):
    nodes, hexes = hex_bar(2)
    f = tmp_path / "n.msh"
    write22(f, nodes, [(5, 1, hexes[0]), (5, 1, hexes[1]), (3, 3, [1, 2, 3, 4]),
                       (3, 4, [9, 10, 11, 12])],
            [(3, 1, "body"), (2, 3, "bottom"), (2, 4, "top")])
    d = _solid_case(f, regions={"body": {"box": [0, 1, 0, 1, 0, 0.5]}})
    with pytest.raises(UserError, match="both a physical group"):
        api.prepare_case(api.CaseDraft(d, base=tmp_path))


def test_a_pressure_on_shells_facing_both_ways_is_refused(tmp_path):
    f = tmp_path / "sh.msh"
    write22(f, _plate(), [(3, 1, [1, 2, 5, 4]), (3, 1, [2, 5, 6, 3]), (1, 2, [1, 4])],
            [(2, 1, "plate"), (1, 2, "edge")])
    d = {"version": 1, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "gmsh", "path": str(f)}, "materials": {"s": dict(STEEL)},
         "sections": {"p": {"type": "shell", "material": "s", "thickness": 0.01}},
         "assign": [{"region": "all", "section": "p"}],
         "constraints": [{"region": "edge", "dofs": [1] * 6}],
         "loads": [{"region": "plate", "pressure": 1000.0}],
         "solver": {"mode": "relaxation", "n_steps": 1000}}
    assert "SHELL_NORMALS_INCONSISTENT" in _codes(tmp_path, d)


# ---------------------------------------------------------- regions, units
def _box(tmp_path, **over):
    d = {"version": 1, "units": {"length": "m", "force": "N", "stress": "Pa"},
         "mesh": {"type": "box", "dimensions": [1.0, 0.1, 0.1], "divisions": [5, 1, 1]},
         "materials": {"s": dict(STEEL)}, "sections": {"b": {"type": "solid", "material": "s"}},
         "assign": [{"region": "all", "section": "b"}],
         "regions": {"root": {"where": {"x": 0.0}}, "tip": {"where": {"x": 1.0}}},
         "constraints": [{"region": "root", "dofs": [1, 1, 1]}],
         "loads": [{"region": "tip", "vector": [0, 0, -1.0]}],
         "solver": {"mode": "relaxation", "n_steps": 1000}}
    for k, v in over.items():
        d[k] = v
    return d


@pytest.mark.parametrize("region, match", [
    ({"node_ids": [-1]}, "outside"),
    ({"node_ids": [9999]}, "outside"),
    ({"sphere": {"center": [0, 0, 0], "radius": 0.1, "extra": 1}}, "unknown extra"),
    ({"where": {"X": 0.0}}, "axes x, y and z"),
])
def test_a_region_written_wrongly_is_a_sentence(tmp_path, region, match):
    d = _box(tmp_path)
    d["regions"]["pt"] = region
    d["loads"] = [{"region": "pt", "vector": [0, 0, -1.0]}]
    with pytest.raises((ValueError, UserError), match=match):
        p = api.prepare_case(api.CaseDraft(d, base=tmp_path))
        errs = [i["message"] for i in p.preflight["issues"] if i["severity"] == "ERROR"]
        raise ValueError("; ".join(errs))


def test_the_documented_sphere_spelling_is_read(tmp_path):
    d = _box(tmp_path)
    d["regions"]["ball"] = {"sphere": {"center": [1.0, 0.0, 0.0], "radius": 0.01}}
    p = api.prepare_case(api.CaseDraft(d, base=tmp_path))
    assert len(p.mesh.regions["ball"]) == 1


def test_where_snaps_its_pinned_axes_together(tmp_path):
    from vfife.core import selectors
    x = np.array([[0.98, 0, 1.0], [1.0, 0, 0.0], [0.0, 0, 0.0], [0.98, 0, 0.0]])
    notes = []
    got = selectors.resolve_all(x, {"s": {"where": {"x": 0.99, "z": 1.0}}}, notes=notes)
    assert got["s"].tolist() == [0]
    assert "mm" not in notes[0]


def test_a_box_mesh_refuses_a_scale(tmp_path):
    d = _box(tmp_path)
    d["mesh"]["scale"] = 0.001
    with pytest.raises(UserError, match="mesh.scale"):
        api.prepare_case(api.CaseDraft(d, base=tmp_path))


@pytest.mark.parametrize("units, gravity, code", [
    ({"length": "mm", "force": "N", "stress": "MPa"}, 9.81, "GRAVITY_MAGNITUDE"),
    ({"length": "m", "force": "N", "stress": "MPa"}, None, "UNITS_INCONSISTENT"),
    ({"lenght": "m", "force": "N", "stress": "Pa"}, None, "UNITS_UNKNOWN_KEY"),
])
def test_the_unit_declaration_is_checked(tmp_path, units, gravity, code):
    d = _box(tmp_path, units=units)
    if gravity is not None:
        d["materials"]["s"] = {"type": "linear_elastic", "E": 2.0e5, "nu": 0.0, "rho": 7.85e-9}
        d["mesh"]["dimensions"] = [1000.0, 100.0, 100.0]
        d["regions"] = {"root": {"where": {"x": 0.0}}, "tip": {"where": {"x": 1000.0}}}
        d["loads"] = [{"region": "all", "gravity": [0, 0, -gravity]}]
    assert code in _codes(tmp_path, d)


def test_nodes_a_nanometre_apart_are_duplicates(tmp_path):
    nodes, hexes = hex_bar(2)
    extra = [(x, y, z + 1e-9) for (x, y, z) in nodes[4:8]]
    nodes = nodes + extra
    h2 = [hexes[0], [13, 14, 15, 16, 9, 10, 11, 12]]
    f = tmp_path / "gap.msh"
    write22(f, nodes, [(5, 1, h2[0]), (5, 1, h2[1]), (3, 3, [1, 2, 3, 4]), (3, 4, [9, 10, 11, 12])],
            [(3, 1, "body"), (2, 3, "bottom"), (2, 4, "top")])
    assert "DUPLICATE_NODE" in _codes(tmp_path, _solid_case(f))
