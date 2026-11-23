"""A patch may name its own material.

WHY
    `rect` sections have had `cover` + `core_material` since v0.2: everything
    within `cover` of a face is the cover law, the rest is the confined one.
    That rule does not generalise to `patches`, where the seam between two
    patches is not a free face -- a distance-to-boundary test would call
    interior fibres cover. A patches section says it outright instead, which
    is also more expressive: a deck of different concrete from the web, a
    repaired zone, a flange core that really is confined.

    The tests below hold the three things that can silently go wrong: a
    section that names nothing must be unchanged TO THE BIT, the fibres must
    land in the group their own patch names, and the two rules must not be
    usable together on one section.
"""
import pathlib

import numpy as np
import pytest

from vfife.elements.fiber_section import patch_fibers

TWO = [{"corners": [[0.0, -0.1], [0.0, 0.1], [0.05, 0.1], [0.05, -0.1]], "ny": 3, "nz": 4},
       {"corners": [[-0.2, -0.03], [-0.2, 0.03], [0.0, 0.03], [0.0, -0.03]], "ny": 6, "nz": 2}]


def test_owner_is_opt_in_and_does_not_move_the_fibres():
    a = patch_fibers(TWO)
    b = patch_fibers(TWO, owner=True)
    assert len(a) == 4 and len(b) == 5
    for x, y in zip(a, b[:4]):
        assert np.array_equal(x, y)
    owner = b[4]
    assert owner.shape == a[0].shape
    assert (owner[:12] == 0).all() and (owner[12:] == 1).all()


def test_owner_survives_a_hole_that_deletes_fibres():
    """A cell wholly inside a void produces no fibre, so the count per patch
    is not ny*nz and ownership cannot be reconstructed afterwards -- which is
    the whole reason the array is returned rather than recomputed."""
    # The radius has to swallow WHOLE cells: `_hole_weight` keeps a cut cell
    # at reduced area and drops only one whose every sample point is inside.
    # Patch 1's `ny` divides corner0 -> corner1, which here runs along z, so
    # its cells are 0.1 wide in y and 0.01 deep in z; this hole eats four of
    # the six in the y < -0.1 column.
    holed = [dict(TWO[0]), dict(TWO[1], holes=[[-0.15, 0.0, 0.05]])]
    y, z, area, _c, owner = patch_fibers(holed, owner=True)
    assert len(owner) == len(y)
    assert (owner == 1).sum() == 6          # the hole ate four of patch 1's cells
    assert (owner == 0).sum() == 12         # patch 0 is untouched


MSH = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
3
0 1 "left"
0 2 "right"
1 3 "bar"
$EndPhysicalNames
$Nodes
3
1 0 0 0
2 1 0 0
3 2 0 0
$EndNodes
$Elements
4
1 15 2 1 1 1
2 15 2 2 2 3
3 1 2 3 3 1 2
4 1 2 3 3 2 3
$EndElements
"""


def _model(tmp, patch_mats=None, core=False):
    """A two-element beam whose fibre section is the two patches above."""
    mpath = pathlib.Path(tmp) / "bar.msh"
    mpath.write_text(MSH, encoding="utf-8")
    pat = [dict(q) for q in TWO]
    if patch_mats:
        for q, m in zip(pat, patch_mats):
            if m: q["material"] = m
    sec = {"type": "fiber", "material": "plain", "shape": "patches", "patches": pat,
           "integration": "lobatto3", "torsion_constant": 1.0e-5}
    if core:
        sec["cover"] = 0.01
        sec["core_material"] = "conf"
    return {
        "name": "patch_material_test",
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "gmsh", "path": str(mpath)},
        "materials": {
            "plain": {"type": "concrete02", "fpc": 3.0e7, "epsc0": 0.002, "fcu": 6.0e6,
                      "epscu": 0.004, "ft": 3.0e6, "Ets": 1.5e9, "rho": 2400},
            "conf": {"type": "concrete02", "fpc": 3.6e7, "epsc0": 0.003, "fcu": 1.5e7,
                     "epscu": 0.009, "ft": 3.0e6, "Ets": 1.5e9, "rho": 2400}},
        "sections": {"s": sec},
        "assign": [{"region": "all", "section": "s", "orientation": [0, 0, 1]}],
        "constraints": [{"region": "all", "dofs": [0, 1, 0, 1, 0, 1]},
                        {"region": "left", "dofs": [1, 1, 1, 1, 0, 1]},
                        {"region": "right", "dofs": [0, 1, 1, 1, 0, 1]}],
        "solver": {"backend": "numpy", "mode": "relaxation", "n_steps": 10,
                   "max_steps_per_load": 10, "tolerance": 1.0e-3,
                   "damping": {"type": "kinetic"}},
        "output": {"path": "results/_patch_material_test.h5", "every": 10**7},
    }


def _solver(doc):
    from vfife.api import _with_element, prepare_case
    from vfife.core.structural import make_structural_solver
    prep = prepare_case(doc)
    p = _with_element(prep.case, prep.element, prep.case_path)
    return make_structural_solver(p, prep.mesh, prep.material)


def test_naming_no_patch_material_changes_nothing(tmp_path):
    """A section whose patches name nothing keeps ONE concrete group and the
    same fibre-to-group map it had before this feature existed."""
    a = _solver(_model(tmp_path))
    b = _solver(_model(tmp_path, patch_mats=[None, None]))
    la, lb = a._layouts[0], b._layouts[0]
    assert len(la.conc_groups) == 1
    assert np.array_equal(la.fiber_group, lb.fiber_group)


def test_each_fibre_lands_in_the_group_its_own_patch_names(tmp_path):
    s = _solver(_model(tmp_path, patch_mats=["conf", None]))
    lay = s._layouts[0]
    assert len(lay.conc_groups) == 2, "one group for the section's law, one for `conf`"
    g_plain, g_conf = lay.conc_groups
    n0 = 12                                   # patch 0 is 3 x 4 cells
    assert (lay.fiber_group[:n0] == g_conf).all()
    assert (lay.fiber_group[n0:lay.n_c] == g_plain).all()
    # and the two groups really carry different laws
    assert s.materials[g_conf].epscu != s.materials[g_plain].epscu


def test_two_patches_naming_the_same_material_share_one_group(tmp_path):
    s = _solver(_model(tmp_path, patch_mats=["conf", "conf"]))
    lay = s._layouts[0]
    assert len(lay.conc_groups) == 2
    assert (lay.fiber_group[:lay.n_c] == lay.conc_groups[1]).all()


def test_the_two_rules_cannot_be_combined(tmp_path):
    with pytest.raises(ValueError, match="two ways to say the same thing"):
        _solver(_model(tmp_path, patch_mats=["conf", None], core=True))


def test_a_patch_material_must_be_a_name(tmp_path):
    doc = _model(tmp_path)
    doc["sections"]["s"]["patches"][0]["material"] = {"type": "concrete02"}
    with pytest.raises(ValueError, match="names one of the model's materials"):
        _solver(doc)
