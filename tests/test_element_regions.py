"""Physical groups that select ELEMENTS, not only nodes.

A region has always been a set of nodes, and an element belonged to it when
all of its nodes did. That rule cannot separate the beams of a building from
its columns: every column's two nodes are also beam nodes, so both groups
claim every column and the model is refused as over-assigned. A Gmsh physical
group knows which elements it holds; this is that information kept rather than
thrown away at import.

The other half of the file is the input form the same building needed: a shell
section on the wall region and line sections on the frame, which is a
`shell_line` model and could not be written down before.
"""

import textwrap

import numpy as np
import pytest

from vfife.core.mesh import Mesh
from vfife.io import gmsh


def _msh(tmp_path):
    """Two columns and a beam between their tops, plus one wall panel.

    Every node of both columns is also a node of the beam's physical group in
    the node-set sense, which is the whole difficulty.
    """
    text = textwrap.dedent("""\
        $MeshFormat
        2.2 0 8
        $EndMeshFormat
        $PhysicalNames
        3
        1 1 "columns"
        1 2 "beams"
        2 3 "walls"
        $EndPhysicalNames
        $Nodes
        6
        1 0 0 0
        2 0 0 3
        3 4 0 0
        4 4 0 3
        5 0 5 0
        6 0 5 3
        $EndNodes
        $Elements
        4
        1 1 2 1 1 1 2
        2 1 2 1 1 3 4
        3 1 2 2 2 2 4
        4 3 2 3 3 1 5 6 2
        $EndElements
        """)
    path = tmp_path / "frame.msh"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_the_import_records_which_elements_each_group_holds(tmp_path):
    mesh, element = gmsh.read_coupled(_msh(tmp_path))
    assert element == "shell_line"
    assert set(mesh.cell_blocks) == {"truss2", "shell4"}
    assert mesh.element_regions["columns"]["truss2"].tolist() == [0, 1]
    assert mesh.element_regions["beams"]["truss2"].tolist() == [2]
    assert mesh.element_regions["walls"]["shell4"].tolist() == [0]


def test_the_node_rule_could_not_have_separated_them(tmp_path):
    """The control, and the reason the recorded groups exist: by nodes alone
    the beam group claims nothing and the column group claims the beam."""
    mesh, _ = gmsh.read_coupled(_msh(tmp_path))
    lines = mesh.cell_blocks["truss2"]
    by_nodes = Mesh(mesh.nodes, lines, dict(mesh.regions))     # no element_regions
    columns = by_nodes.region_elements("columns", "truss2", lines)
    assert sorted(columns.tolist()) == [0, 1, 2], columns      # the beam too
    with_groups = mesh.region_elements("columns", "truss2", lines)
    assert with_groups.tolist() == [0, 1]


def test_a_region_with_no_recorded_group_still_uses_the_node_rule(tmp_path):
    """Regions written as a box or a plane have no elements of their own, and
    everything that consumed them before must keep working."""
    mesh, _ = gmsh.read_coupled(_msh(tmp_path))
    lines = mesh.cell_blocks["truss2"]
    mesh.regions["ground"] = np.array([0, 2, 4])               # the z = 0 nodes
    assert mesh.region_elements("ground", "truss2", lines).tolist() == []
    mesh.regions["left"] = np.array([0, 1])
    assert mesh.region_elements("left", "truss2", lines).tolist() == [0]


def test_an_unknown_region_is_none_not_an_empty_answer(tmp_path):
    mesh, _ = gmsh.read_coupled(_msh(tmp_path))
    assert mesh.region_elements("nope", "truss2", mesh.cell_blocks["truss2"]) is None


def test_a_shell_section_on_a_region_plus_line_sections_is_a_shell_line_model():
    """The input form for a shear-wall building: walls are a shell section on
    their own region, the frame is line sections on theirs. Before v0.1 a
    shell section had to be the whole mesh, so this could not be written.
    """
    from vfife.io.model import InputCase

    lowered = InputCase(
        materials={"c": {"type": "linear_elastic", "E": 3.0e4, "nu": 0.2, "rho": 2.4e-9}},
        sections={
            "wall": {"type": "shell", "material": "c", "thickness": 200.0},
            "col": {"type": "beam", "material": "c", "shape": "general",
                    "area": 1.6e5, "iy": 2.1e9, "iz": 2.1e9, "j": 3.6e9},
        },
        assign=[{"region": "walls", "section": "wall"},
                {"region": "columns", "section": "col", "orientation": [1, 0, 0]}],
    ).lower()
    assert lowered["element"]["type"] == "shell_line"
    assert lowered["element"]["thickness"] == 200.0
    assert [m["region"] for m in lowered["members"]] == ["columns"]


def test_a_shell_section_on_a_region_alone_is_still_refused():
    """The rule it relaxes, still in force where it means something: with no
    line members the shell IS the mesh, and one thickness is all there is."""
    from vfife.io.model import InputCase

    with pytest.raises(ValueError, match="assigned to `all`"):
        InputCase(
            materials={"c": {"type": "linear_elastic", "E": 3.0e4, "nu": 0.2, "rho": 2.4e-9}},
            sections={"wall": {"type": "shell", "material": "c", "thickness": 200.0}},
            assign=[{"region": "walls", "section": "wall"}],
        ).lower()
