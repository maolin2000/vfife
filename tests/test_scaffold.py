"""Regression tests for the generic mesh-driven case initializer."""

import yaml

from vfife.scaffold import GenericGmshDraft, generic_gmsh_case, physical_regions
from vfife.io import schema


def test_initializer_offers_boundary_groups_not_whole_volume():
    regions = {
        "all": [0, 1, 2, 3], "surface": [0, 1, 2],
        "solid": [0, 1, 2, 3], "support": [0, 1], "load_pad": [2, 3],
    }
    assert physical_regions(regions) == ["load_pad", "support"]


def test_generic_gmsh_scaffold_is_a_valid_case():
    text = generic_gmsh_case("bracket", "../mesh/bracket.msh", "bolt_holes",
                             "load_pad", [0.0, 0.0, -25000.0])
    data = yaml.safe_load(text)
    assert "sections" in data and "materials" in data      # the input form
    case = schema.from_dict(data)
    assert case.mesh.type == "gmsh"
    assert case.constraints[0].region == "bolt_holes"
    assert case.loads[0].vector == [0.0, 0.0, -25000.0]


def test_draft_represents_multiple_boundary_conditions_for_a_gui():
    draft = GenericGmshDraft(
        "frame", "frame.msh",
        constraints=[{"region": "left", "dofs": [1, 1, 1]},
                     {"region": "guide", "dofs": [0, 1, 0]}],
        loads=[{"region": "top", "vector": [0.0, 0.0, -25000.0]},
               {"region": "wind", "vector": [12000.0, 0.0, 0.0]}],
        amplitudes={"cycle": {"table": [[0.0, 0.0], [1.0, 1.0]]}},
        steps=[{"name": "load", "n_steps": 2000}],
    )
    case = draft.validate()
    assert len(case.constraints) == 2
    assert len(case.loads) == 2
    assert case.steps[0].name == "load"


# --------------------------------------------------------------------------- #
#  A scaffolded case has something to plot
# --------------------------------------------------------------------------- #

def test_the_scaffold_writes_a_load_displacement_pair():
    """It used to write no monitors at all, so `vfife plot` on the result of
    a scaffolded run could draw the residual and the kinetic energy -- the
    solver's own diagnostics -- and nothing about the structure. The first
    thing anyone wants is how far the loaded end moved and what the support
    carried."""
    from vfife.scaffold import GenericGmshDraft

    d = GenericGmshDraft("t", "m.msh",
                         constraints=[{"region": "root", "dofs": [1, 1, 1]}],
                         loads=[{"region": "tip", "vector": [0.0, 0.0, -5.0e3]}])
    mon = d.as_dict()["output"]["monitors"]
    assert mon["u"] == {"kind": "displacement", "region": "tip", "dof": "z"}
    assert mon["R"] == {"kind": "reaction", "region": "root", "dof": "z"}


def test_the_monitored_dof_follows_the_load():
    """`plot --x u --y R` is a load-deflection curve only if both read the
    direction the load acts along."""
    from vfife.scaffold import GenericGmshDraft

    d = GenericGmshDraft("t", "m.msh",
                         constraints=[{"region": "root", "dofs": [1, 1, 1]}],
                         loads=[{"region": "tip", "vector": [-2.0e3, 0.0, 0.0]}])
    mon = d.as_dict()["output"]["monitors"]
    assert mon["u"]["dof"] == "x" and mon["R"]["dof"] == "x"


def test_a_case_with_no_load_still_validates():
    """The monitors are a convenience, not a requirement: a case with only
    constraints must still produce a valid file."""
    from vfife.scaffold import GenericGmshDraft

    d = GenericGmshDraft("t", "m.msh",
                         constraints=[{"region": "root", "dofs": [1, 1, 1]}])
    mon = d.as_dict()["output"]["monitors"]
    assert "u" not in mon and mon["R"]["dof"] == "z"
