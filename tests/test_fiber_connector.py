"""A member of a fibre mesh that is not a fibre beam.

`FiberBeamSolver` had exactly one element type. A bearing is a `connector2`,
so `case_24d`'s piers had to be welded to the girder, and that weld doubled
the middle-span dead-load deflection.

The MECHANICS of this feature is `elements/connector.py`, which is verified by
`test_connector2.py` and `case_09` and which answers with the fibre solver's
own `q`, `orient`, `v` and `R0` unchanged. Nothing here re-tests it. What is
tested here is the BOOKKEEPING, because that is all that was written: an
element with no section, no fibres, no mass of its own and no place in the
fracture latch has to travel through nine places that were written assuming
every element was a fibre beam, and the fused kernel has to not see it at all.

The model below is the bridge in miniature -- a fibre beam on a "bearing" that
hangs it off a fixed point -- small enough that every array can be looked at
whole.
"""
import pathlib

import numpy as np
import pytest

from vfife.api import prepare_case
from vfife.core.fiber_beam_solver import FiberBeamSolver


CASE = """
name: fiber_connector_check
units: {length: m, force: N, stress: Pa}
mesh: {type: gmsh, path: MESH}
materials:
  concrete: {type: concrete02, fpc: 25.6e6, epsc0: 0.002, fcu: 15.0e6, epscu: 0.02,
             ft: 3.1e6, Ets: 1.2e9, rat: 0.1, rho: 2400}
  steel: {type: steel02, E: 200.0e9, Fy: 400.0e6, b: 0.01, R0: 20.0, rho: 7850}
sections:
  rc:
    type: fiber
    material: concrete
    shape: rect
    width: 0.3
    height: 0.3
    ny: 4
    nz: 4
    integration: lobatto3
    bars:
      - {y: 0.1, z: 0.1, section: bar}
      - {y: -0.1, z: -0.1, section: bar}
  bar: {type: bar, material: steel, shape: general, area: 3.1416e-4}
  pot:
    type: connector
    material: concrete
    stiffness: [5.0e+9, 1.0e+8, 1.0e+9, 1.0e+9, 1.0e+9, 1.0e+6]
assign:
  - {region: all, section: rc, orientation: [0, 1, 0]}
  - {region: LINKREGION, section: pot, orientation: [1, 0, 0], coordinate_system: local}
regions:
  root: {box: [-0.01, 0.01, -0.01, 0.01, -0.01, 0.01]}
  tip: {box: [1.19, 1.21, -0.01, 0.01, -0.01, 0.01]}
  ground: {box: [0.39, 0.41, -0.01, 0.01, -0.21, -0.19]}
  link: {box: [0.39, 0.41, -0.01, 0.01, -0.21, 0.01]}
  toowide: {box: [-0.01, 0.41, -0.01, 0.01, -0.21, 0.01]}
constraints:
  - {region: root, dofs: [1, 1, 1, 1, 1, 1]}
  - {region: ground, dofs: [1, 1, 1, 1, 1, 1]}
loads:
  - {region: tip, vector: [0.0, 0.0, -2.0e4]}
steps:
  - {name: push, load_steps: 2}
solver: {backend: numpy, mode: relaxation, tolerance: 1.0e-4, n_steps: 40}
output:
  path: out.h5
  monitors:
    tip: {kind: displacement, region: tip, dof: z}
"""


def _write(tmp_path, link_region="link", extra_assign=""):
    """Three fibre beams along x, and a 0.2 m link from the middle node down to
    a fixed point. The link is element 3 and is the only connector."""
    d = pathlib.Path(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    msh = d / "m.msh"
    nodes = ["1 0 0 0", "2 0.4 0 0", "3 0.8 0 0", "4 1.2 0 0", "5 0.4 0 -0.2"]
    elems = ["1 1 2 1 1 1 2", "2 1 2 1 1 2 3", "3 1 2 1 1 3 4", "4 1 2 1 1 2 5"]
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n"
                   f"{len(nodes)}\n" + "\n".join(nodes) + "\n$EndNodes\n$Elements\n"
                   f"{len(elems)}\n" + "\n".join(elems) + "\n$EndElements\n", encoding="utf-8")
    text = (CASE.replace("MESH", str(msh)).replace("LINKREGION", link_region))
    if extra_assign:
        text = text.replace("regions:", extra_assign + "regions:")
    p = d / "case.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def _solver(path, kernel=True):
    prep = prepare_case(path)
    sol = FiberBeamSolver(prep.case.model_dump(mode="python"), prep.mesh, prep.material)
    if not kernel:
        sol._kernel = None
    return sol


LINK = 3          # the connector's element index


# --------------------------------------------------------------------------- #
#  the mask
# --------------------------------------------------------------------------- #
def test_the_connector_is_marked_and_nothing_else_is(tmp_path):
    sol = _solver(_write(tmp_path))
    assert sol.has_connectors
    assert list(sol._is_fiber) == [True, True, True, False]
    assert list(sol._connector_of) == [-1, -1, -1, 0]
    assert list(sol._connector_ids) == [LINK]


def test_the_connector_is_in_no_section_row(tmp_path):
    """`_rows` is what hands each section its members, and a connector has no
    section. It reaches the solver through an `assign` like any other member,
    so the filter has to be in the row builder itself -- not in the input."""
    sol = _solver(_write(tmp_path))
    for row in (sol._rows or []):
        assert LINK not in list(row)
    assert sorted(int(e) for row in (sol._rows or []) for e in row) == [0, 1, 2]


def test_the_fused_kernel_is_not_shown_the_connector(tmp_path):
    """`_kernel_active` is `_active & _is_fiber`. The kernel integrates fibres
    over a section; handed a connector it would integrate the padded row."""
    sol = _solver(_write(tmp_path))
    assert list(sol._kernel_active) == [1, 1, 1, 0]


def test_a_fracture_cannot_revive_the_connector(tmp_path):
    """THE REASON THERE ARE TWO MASKS. `_active` is the fracture latch and the
    solver writes it; `_is_fiber` is a property of the mesh and nothing writes
    it. One flag doing both jobs would mean a broken member and a bearing were
    the same thing, and the next `_latch_fracture` would put the bearing back
    into the kernel."""
    sol = _solver(_write(tmp_path))
    sol._active[0] = False
    sol._refresh_kernel_mask()
    assert list(sol._kernel_active) == [0, 1, 1, 0]
    assert bool(sol._active[LINK]) is True          # a bearing is not broken
    assert list(sol._is_fiber) == [True, True, True, False]


# --------------------------------------------------------------------------- #
#  the arithmetic
# --------------------------------------------------------------------------- #
def test_the_kernel_and_the_reference_path_agree_with_a_connector_present(tmp_path):
    """The whole point of the mask. The kernel loops the fibre members and the
    connectors are added after it in Python; the reference path branches per
    element. Both have to produce the same vector, over a history, or the mask
    is being applied in one place and not the other."""
    pytest.importorskip("numba")
    fast = _solver(_write(tmp_path), kernel=True)
    ref = _solver(_write(tmp_path), kernel=False)
    assert fast._kernel is not None, fast.kernel_status

    rng = np.random.default_rng(3)
    worst_f = worst_a = 0.0
    for _ in range(25):
        dq = rng.normal(0.0, 2e-4, fast.ndof)
        fast.q[:] += dq
        ref.q[:] += dq
        f1, a1 = fast.internal_force()
        f2, a2 = ref.internal_force()
        worst_f = max(worst_f, np.max(np.abs(f1 - f2)) / max(np.max(np.abs(f2)), 1.0))
        worst_a = max(worst_a, np.max(np.abs(a1 - a2)) / max(np.max(np.abs(a2)), 1.0))
    assert worst_f < 1e-10, f"force drifted {worst_f:.2e}"
    assert worst_a < 1e-10, f"actions drifted {worst_a:.2e}"


def test_the_connector_answers_with_its_own_declared_stiffness(tmp_path):
    """The link runs from node 2 DOWN to node 5, so its local x is vertical and
    `stiffness[0]` is what resists a vertical pull. Lift the top node by d and
    the link stretches by d and pulls back with 5e9 d -- this is the one check
    that the props, the nodes and the frame were handed over in the right
    ORDER rather than merely in the right shapes."""
    sol = _solver(_write(tmp_path), kernel=False)
    d = 1e-6
    sol.q[1 * 6 + 2] += d                       # node 2's z, the link's top
    _f, actions = sol.internal_force()
    assert actions[LINK, 0] == pytest.approx(5.0e9 * d, rel=1e-6)
    assert sol.connector_deformation[LINK, 0] == pytest.approx(d, rel=1e-6)
    # and the fibre members, which were not moved, stay silent
    assert np.max(np.abs(sol.connector_deformation[:LINK])) == 0.0


def test_the_reference_stiffness_of_a_connector_is_the_connectors_own(tmp_path):
    """`_reference_k` sets the initial K, the relaxation mass and the stable
    step. Falling through to the fibre branch would have asked a connector for
    `total_area`, `Iy` and `J` -- which exist, as the padded row, so it would
    not have raised. It would have returned a 12x12 for a 0.3 m concrete
    section, and the bearing would have been given a beam's mass."""
    sol = _solver(_write(tmp_path))
    k = sol._reference_k(LINK)
    assert k.shape == (12, 12)
    # local x is global -z here, so the 5e9 sits on the global z pair
    assert k[2, 2] == pytest.approx(5.0e9, rel=1e-8)
    assert k[2, 8] == pytest.approx(-5.0e9, rel=1e-8)
    # the fibre members' own reference is unrelated to it
    assert sol._reference_k(0)[0, 0] > 1e9


def test_section_strains_and_the_output_fields_know_the_difference(tmp_path):
    """`section_strains()` asks every element for its section, and the output
    writer asks every element for its fibres. Both have to skip the connector,
    and `connector_flag` is how a reader of the result file tells which member
    the connector columns belong to."""
    sol = _solver(_write(tmp_path))
    sol.q[1 * 6 + 2] += 1e-5
    _f, actions = sol.internal_force()
    eps = sol.section_strains()
    assert np.all(np.asarray(eps)[LINK] == 0.0)
    fields = sol.nodal_fields(actions)
    assert list(fields["connector_flag"]) == [0.0, 0.0, 0.0, 1.0]
    assert fields["connector_deformation_ux"][LINK] != 0.0
    assert np.all(fields["connector_deformation_ux"][:LINK] == 0.0)


# --------------------------------------------------------------------------- #
#  the model without one is the model it always was
# --------------------------------------------------------------------------- #
def test_a_mesh_with_no_connector_is_left_alone(tmp_path):
    """The feature costs a mask multiply per fracture and nothing else. If a
    plain fibre model came out of this with `_section_of` set or a row missing,
    every case in `cases/` would be a different model than it was."""
    path = _write(tmp_path)
    text = pathlib.Path(path).read_text(encoding="utf-8").replace(
        "  - {region: link, section: pot, orientation: [1, 0, 0], "
        "coordinate_system: local}\n", "")
    pathlib.Path(path).write_text(text, encoding="utf-8")
    sol = _solver(path)
    assert not sol.has_connectors
    assert sol._is_fiber.all()
    assert list(sol._connector_ids) == []
    assert list(sol._kernel_active) == [1] * sol.nel
    assert "connector_flag" not in sol.nodal_fields(sol.internal_force()[1])


# --------------------------------------------------------------------------- #
#  the refusals
# --------------------------------------------------------------------------- #
def test_a_region_that_owns_no_complete_element_is_refused(tmp_path):
    """`ground` is one node -- the link's lower end -- and a region of one node
    owns no element. The failure mode this closes is an empty `ids` and a
    connector list that silently does nothing: the bearing would simply not be
    there, and the answer would quietly be the monolithic one, which is the
    exact number this whole feature exists to move."""
    with pytest.raises(ValueError, match="covers no complete Line2"):
        _solver(_write(tmp_path, link_region="ground"))


def test_overlapping_connector_regions_are_refused(tmp_path):
    """Two assigns claiming one element: the second would silently win, and
    which one that is depends on the order of the `assign` list."""
    extra = ("  - {region: link, section: pot, orientation: [0, 1, 0], "
             "coordinate_system: local}\n")
    with pytest.raises(ValueError, match="connector regions overlap"):
        _solver(_write(tmp_path, extra_assign=extra))


def test_a_connector_region_that_swallows_fibre_members_takes_them(tmp_path):
    """NOT a refusal, and worth pinning as such: a region is a region. The
    `toowide` box owns the first fibre beam as well as the link, and both
    become connectors. There is nothing wrong with that -- a run of bearings
    is a legitimate model -- but it means a sloppy box turns beams into springs
    silently, which is what `connector_flag` in the output is for."""
    path = _write(tmp_path, link_region="toowide")
    pathlib.Path(path).write_text(pathlib.Path(path).read_text(encoding="utf-8").replace(
        "section: pot, orientation: [1, 0, 0]", "section: pot, orientation: [0, 1, 0]"), encoding="utf-8")
    sol = _solver(path)
    assert list(sol._is_fiber) == [False, True, True, False]
    assert list(sol._kernel_active) == [0, 1, 1, 0]


# --------------------------------------------------------------------------- #
#  preflight: a bearing has to be bearing on something
# --------------------------------------------------------------------------- #
def _preflight(path):
    from vfife.cli import build
    from vfife.io import schema
    from vfife.preflight import check
    case, _raw = schema.load(path)
    mesh, _material, element = build(case, path)
    return check(case, mesh, element)


def test_preflight_passes_the_model_that_works(tmp_path):
    report = _preflight(_write(tmp_path))
    assert report["status"] == "PASS", [i["code"] for i in report["issues"]]


def test_preflight_refuses_a_connector_node_that_carries_nothing_else(tmp_path):
    """THE TRAP an earlier evaluation flagged and this closes. The link's lower node is held by its constraint, not by a
    member; remove the constraint and it hangs off six springs, three of which
    a real pot bearing releases. A zero row in K has zero pseudo-mass, so the
    run does not stop -- it either NaNs at step one or flies that node away
    while everything else looks reasonable."""
    path = _write(tmp_path)
    text = pathlib.Path(path).read_text(encoding="utf-8").replace(
        "  - {region: ground, dofs: [1, 1, 1, 1, 1, 1]}\n", "")
    pathlib.Path(path).write_text(text, encoding="utf-8")
    report = _preflight(path)
    assert report["status"] == "FAIL"
    issue = next(i for i in report["issues"] if i["code"] == "CONNECTOR_NODE_UNHELD")
    assert issue["details"]["first"] == [4]      # node 5, zero-based
