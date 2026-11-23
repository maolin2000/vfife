"""The input model: materials -> sections -> assign, and its lowering.

WHY
    The old case form had no section layer: one `material` was implicitly
    the whole mesh's, section geometry lived in four places (`element`,
    `members`, `reinforcement`, `fiber_section`), and a bar's steel could
    only be written inline because there was nowhere to name it. The input
    form (io/model.py) has the Abaqus / OpenSees / Code_Aster structure and
    treats reinforcement the way DIANA and ANSYS do -- as its own geometry.

    The solver still reads the old layout, INTERNALLY. `from_dict` lowers
    the input form to it; the old layout is refused at the boundary and
    `vfife migrate` converts a file. These tests pin the boundary, the
    lowering of each section type, and the migration's own check.
"""
import glob
import os

import pytest
import yaml

from vfife.core.diagnostics import UserError
from vfife.io import schema
from vfife.io.migrate import check_round_trip, migrate
from vfife.io.model import lower

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STEEL = {"type": "linear_elastic", "E": 2.0e11, "nu": 0.3, "rho": 7850.0}


def _base(**over):
    d = {"name": "t", "mesh": {"dimensions": [1.0, 0.2, 0.3], "divisions": [4, 2, 3]},
         "materials": {"steel": STEEL},
         "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 10}}
    d.update(over)
    return d


def test_the_old_form_is_refused_at_the_boundary_and_names_the_tool():
    old = {"name": "t", "mesh": {"dimensions": [1, 1, 1], "divisions": [1, 1, 1]},
           "element": {"formulation": "b-bar"}, "material": STEEL,
           "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 10}}
    with pytest.raises(UserError) as exc:
        schema.from_dict(old)
    assert exc.value.code == "OLD_CASE_FORM"
    assert "vfife migrate" in exc.value.hint


def test_a_lone_solid_section_goes_to_the_whole_mesh_without_an_assignment():
    case = schema.from_dict(_base(sections={"body": {"type": "solid", "material": "steel",
                                                     "formulation": "full"}}))
    assert case.material == STEEL
    assert case.element.formulation == "full"
    assert case.element.type == "hexa8"


def test_two_solid_sections_become_material_blocks_in_assignment_order():
    """`all: a` then `left: b` -- the internal case carries both as blocks,
    in that order, with `material` still the first one's for every reader
    that knows only one material."""
    soft = {"type": "linear_elastic", "E": 1.0e10, "nu": 0.2, "rho": 2400.0}
    d = _base(materials={"steel": STEEL, "soft": soft},
              sections={"a": {"type": "solid", "material": "steel"},
                        "b": {"type": "solid", "material": "soft", "formulation": "reduced"}},
              regions={"left": {"where": {"x": {"max": 0.5}}}},
              assign=[{"region": "all", "section": "a"}, {"region": "left", "section": "b"}])
    case = schema.from_dict(d)
    assert case.material == STEEL
    assert [b.name for b in case.blocks] == ["a", "b"]
    assert [b.region for b in case.blocks] == ["all", "left"]
    assert case.blocks[1].material == soft
    assert case.blocks[1].formulation == "reduced"
    assert case.blocks[0].formulation is None


def test_one_solid_section_on_part_of_the_mesh_is_a_block_too():
    d = _base(sections={"a": {"type": "solid", "material": "steel"}},
              regions={"left": {"where": {"x": {"max": 0.5}}}},
              assign=[{"region": "left", "section": "a"}])
    case = schema.from_dict(d)
    assert [b.region for b in case.blocks] == ["left"]


def test_a_shell_section_cannot_share_the_mesh():
    d = _base(sections={"a": {"type": "solid", "material": "steel"},
                        "s": {"type": "shell", "material": "steel", "thickness": 0.01}},
              regions={"left": {"where": {"x": {"max": 0.5}}}},
              assign=[{"region": "all", "section": "a"}, {"region": "left", "section": "s"}])
    with pytest.raises(ValueError, match="shell section is the whole mesh"):
        schema.from_dict(d)


def test_a_material_a_section_names_must_exist():
    with pytest.raises(ValueError, match="names material 'steal'"):
        schema.from_dict(_base(sections={"body": {"type": "solid", "material": "steal"}}))


def test_a_beam_section_derives_its_properties_from_a_shape():
    d = _base(mesh={"type": "gmsh", "path": os.path.join(ROOT, "benchmarks/assets/beam_cantilever_v22.msh")},
              sections={"col": {"type": "beam", "material": "steel", "shape": "rect", "b": 0.3, "h": 0.5}},
              assign=[{"region": "all", "section": "col", "orientation": [0, 0, 1]}])
    case = schema.from_dict(d)
    assert case.element.type == "beam2"
    assert case.element.area == pytest.approx(0.15)
    assert case.element.iy == pytest.approx(0.3 * 0.5 ** 3 / 12)
    assert case.element.orientation == [0, 0, 1]


def test_reinforcement_layer_and_hoop_expand_to_the_bars_a_drawing_shows():
    d = _base(sections={"body": {"type": "solid", "material": "steel"},
                        "bar": {"type": "bar", "material": "steel", "shape": "circle", "diameter": 0.016}},
              reinforcement={
                  "row": {"layer": {"from": [0, 0.05, 0.05], "to": [1.0, 0.05, 0.05],
                                    "count": 3, "spacing": [0, 0.05, 0]}},
                  "ties": {"hoop": {"corners": [[0.05, 0.05], [0.15, 0.25]], "axis": "x",
                                    "from": 0.1, "to": 0.9, "spacing": 0.2, "divisions": 4}}},
              assign=[{"reinforcement": ["row", "ties"], "section": "bar", "segment_length": 0.1}])
    case = schema.from_dict(d)
    lines = case.reinforcement[0].lines
    # three straight bars, and the tie as ONE CLOSED ROUTE rather than four
    # touching legs -- a route is a bar, and a hoop drawn as four bars has
    # eight free ends where a real tie has none
    assert len(lines) == 3 + 1
    assert [ln.start[1] for ln in lines[:3]] == pytest.approx([0.05, 0.10, 0.15])
    assert all(ln.segment_length == 0.1 for ln in lines[:3])
    tie = lines[3]
    assert tie.closed and len(tie.points) == 4
    assert tie.repeat.count == 5                          # 0.1, 0.3, ..., 0.9
    assert case.reinforcement[0].area == pytest.approx(3.14159 * 0.008 ** 2, rel=1e-4)
    assert case.reinforcement[0].diameter == 0.016


def test_a_bonded_assignment_needs_a_bar_with_a_surface():
    d = _base(sections={"body": {"type": "solid", "material": "steel"},
                        "bar": {"type": "bar", "material": "steel", "shape": "general", "area": 2e-4}},
              reinforcement={"b": {"line": {"start": [0, 0.1, 0.1], "end": [1, 0.1, 0.1], "divisions": 4}}},
              assign=[{"reinforcement": "b", "section": "bar", "bond": {"type": "bond_ceb", "fc": 30e6}}])
    with pytest.raises(ValueError, match="diameter"):
        schema.from_dict(d)


def test_a_bare_section_key_that_is_not_a_field_is_refused_by_name():
    with pytest.raises(ValueError, match="Iz"):
        schema.from_dict(_base(sections={"body": {"type": "solid", "material": "steel", "Iz": 1.0}}))


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(ROOT, "tests", "data", "old_form", "*.yaml"))))
def test_every_archived_old_case_migrates_and_lowers_back_to_itself(path):
    """The thirteen cases as they were written before the input form, kept
    as fixtures: `migrate` must turn each into an input-form case whose
    lowering is field-for-field the original."""
    old = yaml.safe_load(open(path, encoding="utf-8"))
    new = migrate(old)
    assert "sections" in new and "materials" in new
    assert check_round_trip(old, new) == []
