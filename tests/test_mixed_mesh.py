"""Hexahedra where the geometry is regular, tetrahedra where it is not.

The solver used to run one element type per model. That was a property of the
solver, not of meshes: nobody meshes a real structure with a single type, and
Gmsh will not even let you -- it caps every recombined quadrilateral face that
borders a tetrahedral volume with a pyramid, so one `Recombine Surface` line
produces a file the solver refused to read.

The model is now a list of element blocks sharing one node set. These tests are
the evidence that the blocks are really one body: that force, mass and time step
are aggregated rather than taken from block zero, and above all that the
interface between two element types transmits stress exactly.
"""

import numpy as np
import pytest

from vfife.core.diagnostics import UserError
from vfife.core.mesh import Mesh
from vfife.core.solver import Solver
from vfife.materials import create

E, NU, RHO = 3.0e10, 0.2, 2400.0

#: Freudenthal's 6-tetrahedron split of a hexahedron. Conforming: each face of
#: the cube is cut along the same diagonal from either side, so the tetrahedra
#: match the hexahedra across the interface node for node -- which is what makes
#: the patch test below a test of the solver rather than of the mesh.
FREUDENTHAL = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
               (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]


def grid(nx, ny, nz, L=4.0, B=1.0, H=1.0):
    xs, ys, zs = (np.linspace(0, L, nx + 1), np.linspace(0, B, ny + 1),
                  np.linspace(0, H, nz + 1))
    nid, nodes = {}, []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                nid[(i, j, k)] = len(nodes)
                nodes.append((x, y, z))
    return np.array(nodes, float), nid


def build(nx, ny, nz, kind, **dims):
    """`kind` is "hex", "tet", or "mixed" (hexahedra in x < L/2)."""
    nodes, nid = grid(nx, ny, nz, **dims)
    hexes, tets = [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [nid[(i, j, k)], nid[(i + 1, j, k)],
                     nid[(i + 1, j + 1, k)], nid[(i, j + 1, k)],
                     nid[(i, j, k + 1)], nid[(i + 1, j, k + 1)],
                     nid[(i + 1, j + 1, k + 1)], nid[(i, j + 1, k + 1)]]
                if kind == "tet" or (kind == "mixed" and i >= nx // 2):
                    tets += [[c[a], c[b], c[cc], c[d]]
                             for a, b, cc, d in FREUDENTHAL]
                else:
                    hexes.append(c)
    blocks = {}
    if hexes:
        blocks["hexa8"] = np.array(hexes, np.int32)
    if tets:
        blocks["tet4"] = np.array(tets, np.int32)
    primary = max(blocks, key=lambda k: len(blocks[k]))
    return Mesh(nodes=nodes, elements=blocks[primary], cell_blocks=blocks)


def solver_for(mesh, **solver_opts):
    opts = {"mode": "relaxation", "n_steps": 1, "backend": "numpy"}
    opts.update(solver_opts)
    problem = {"element": {"type": "hexa8", "formulation": "full"},
               "solver": opts}
    return Solver(problem, mesh, create("linear_elastic", E=E, nu=NU, rho=RHO),
                  log=lambda *_: None)


# --------------------------------------------------------------------------- #
#  the interface
# --------------------------------------------------------------------------- #


def test_a_linear_field_is_in_equilibrium_across_the_block_interface():
    """The patch test, which is the whole question in one assertion.

    Impose u = A.x for a constant A: the strain is then constant everywhere, so
    every interior node must be in exact equilibrium -- including the nodes on
    the plane where hexahedra meet tetrahedra. If the two blocks disagreed about
    that plane by so much as a shape function, the residual would show up here
    and nowhere else, because every other test is a smooth field that hides a
    small inconsistency inside its own discretisation error.
    """
    mesh = build(6, 4, 4, "mixed")
    s = solver_for(mesh)
    assert {b.element for b in s.blocks} == {"hexa8", "tet4"}

    A = np.array([[1e-4, 2e-5, 3e-5], [0.0, -3e-5, 1e-5], [0.0, 0.0, 2e-5]])
    s.u = mesh.nodes @ A.T

    F = np.zeros((s.nn, 3))
    for b in s.blocks:
        f_e, _ = b.force(s.u, b.quat, b.state, 1.0, rot_iters=30)
        F += b.force.assemble(f_e)

    x = mesh.nodes
    on_boundary = np.zeros(len(x), bool)
    for axis, hi in enumerate((4.0, 1.0, 1.0)):
        on_boundary |= (np.abs(x[:, axis]) < 1e-12) | (np.abs(x[:, axis] - hi) < 1e-12)
    interior = ~on_boundary
    assert interior.sum() > 10

    # and the interface plane itself is genuinely in the interior set
    interface = np.abs(x[:, 0] - 2.0) < 1e-12
    assert (interface & interior).sum() > 0

    assert np.abs(F[interior]).max() / np.abs(F).max() < 1e-10


def test_the_blocks_are_one_body_and_not_two():
    """A block whose assembly returned only its own nodes would make the two
    halves independent, and the beam would fold at the interface."""
    mesh = build(4, 2, 2, "mixed")
    s = solver_for(mesh)
    for b in s.blocks:
        f_e, _ = b.force(s.u, b.quat, b.state, 1.0)
        # each block assembles into the WHOLE model's node vector, so the sum
        # over blocks is well defined; sizing this from the block's own
        # connectivity is what broke first
        assert b.force.assemble(f_e).shape == (mesh.n_nodes, 3)


# --------------------------------------------------------------------------- #
#  aggregation: nothing may be read off block zero
# --------------------------------------------------------------------------- #


def test_the_mass_is_the_sum_over_blocks_not_the_first_block():
    """The whole model's mass is the whole model's mass whichever way it is
    cut, so the three meshes of the same box must weigh the same."""
    total = {}
    for kind in ("hex", "tet", "mixed"):
        s = solver_for(build(4, 2, 2, kind), mode="dynamic", mass="physical",
                        total_time=1.0)
        total[kind] = s.physical_mass[:, 0].sum()
    expected = RHO * 4.0 * 1.0 * 1.0
    for kind, m in total.items():
        assert m == pytest.approx(expected, rel=1e-12), kind


def test_the_time_step_is_the_smallest_block_not_the_first():
    """A tetrahedron in the transition region is finer than the hexahedra it
    joins; taking block zero's CFL limit would run the model unstable."""
    mixed = solver_for(build(6, 3, 3, "mixed"), mode="dynamic", total_time=1.0)
    tet = solver_for(build(6, 3, 3, "tet"), mode="dynamic", total_time=1.0)
    hexa = solver_for(build(6, 3, 3, "hex"), mode="dynamic", total_time=1.0)
    assert mixed.dt == pytest.approx(min(tet.dt, hexa.dt), rel=1e-12)
    assert mixed.dt < hexa.dt


def test_the_element_count_covers_every_block():
    s = solver_for(build(4, 2, 2, "mixed"))
    assert s.nel == sum(b.n_elem for b in s.blocks)
    assert s.nel == 8 + 6 * 8


def test_output_fields_are_element_major_over_all_blocks():
    """Cell data must line up with the cells the writer emits, in block order,
    or ParaView colours the wrong elements."""
    s = solver_for(build(4, 2, 2, "mixed"))
    sig = [np.zeros((b.n_elem, b.n_gp, 6)) for b in s.blocks]
    for i, a in enumerate(sig):
        a[...] = i + 1.0
    out = s.nodal_fields(sig)
    assert out["stress"].shape == (s.nel, 6)
    # the first block's elements come first, and every block is represented
    assert out["stress"][0, 0] == pytest.approx(1.0)
    assert out["stress"][-1, 0] == pytest.approx(float(len(s.blocks)))


def test_a_checkpoint_round_trips_every_block():
    s = solver_for(build(4, 2, 2, "mixed"))
    for i, b in enumerate(s.blocks):
        b.state[...] = i + 1.0
        b.quat[:, 1] = i + 1.0
    ck = s.checkpoint()

    fresh = solver_for(build(4, 2, 2, "mixed"))
    fresh.restore(ck)
    for i, b in enumerate(fresh.blocks):
        assert np.allclose(b.state, i + 1.0)
        assert np.allclose(b.quat[:, 1], i + 1.0)


# --------------------------------------------------------------------------- #
#  the answer
# --------------------------------------------------------------------------- #


@pytest.mark.slow
# 夹逼是物理性质。界面本身的正确性由同文件里的分片试验守着，那条是秒级的。
def test_a_mixed_cantilever_is_bracketed_by_its_single_type_references():
    """tet4 is a constant-strain element and locks in bending; hexa8 with a
    full quadrature does not. A half-and-half mesh must therefore land between
    the two, and it must land there for a physical reason rather than by
    averaging -- so the reference values are computed, not written down.
    """
    L, B, H, P = 4.0, 1.0, 1.0, -1.0e5
    tip_of = {}
    for kind in ("hex", "tet", "mixed"):
        mesh = build(12, 3, 3, kind, L=L, B=B, H=H)
        x = mesh.nodes
        mesh.regions["root"] = np.flatnonzero(np.abs(x[:, 0]) < 1e-12)
        mesh.regions["tip"] = np.flatnonzero(np.abs(x[:, 0] - L) < 1e-12)
        problem = {
            "element": {"type": "hexa8", "formulation": "full"},
            "constraints": [{"region": "root", "dofs": [1, 1, 1]}],
            "loads": [{"region": "tip", "vector": [0.0, 0.0, P]}],
            # (`tol: 1e-10` stood here, a misspelling the solver never read:
            # this ran at the default tolerance, which the bracket needs no
            # more than; the schema refuses the misspelling since a solver's
            # case is validated)
            "solver": {"mode": "relaxation", "n_steps": 60000, "backend": "numpy"},
        }
        s = Solver(problem, mesh,
                   create("linear_elastic", E=E, nu=NU, rho=RHO),
                   log=lambda *_: None)
        s.run()
        tip_of[kind] = float(s.u[mesh.regions["tip"], 2].mean())

    euler = P * L ** 3 / (3 * E * (B * H ** 3 / 12))
    # the two references first, or the bracket would prove nothing
    assert 0.9 < tip_of["hex"] / euler < 1.05          # accurate
    assert tip_of["tet"] / euler < 0.85                # locked, as expected
    lo, hi = sorted((tip_of["hex"], tip_of["tet"]))
    assert lo <= tip_of["mixed"] <= hi


# --------------------------------------------------------------------------- #
#  what it still will not do
# --------------------------------------------------------------------------- #


def test_the_fused_backend_runs_a_mixed_mesh_and_agrees_with_numpy():
    """This used to be refused (MIXED_MESH_BACKEND). The reason was never the
    kernels -- each block compiles its own -- but the assembler, which sized
    the nodal force from `conn.max() + 1` of ITS block and so returned a
    short vector for any block that did not touch the last node. It takes
    the model's node count now, and the sum over blocks is
    the same force numpy assembles."""
    pytest.importorskip("numba")
    mesh = build(4, 2, 2, "mixed")
    ref = solver_for(mesh)
    fused = solver_for(mesh, backend="numba")
    u = np.zeros((mesh.n_nodes, 3))
    u[:, 0] = 1e-4 * mesh.nodes[:, 0]
    u[:, 2] = 5e-5 * mesh.nodes[:, 0] ** 2
    F = []
    for s in (ref, fused):
        tot = np.zeros((s.nn, 3))
        for b in s.blocks:
            f_e, _ = b.force(u, b.quat, b.state, 1.0, rot_iters=30, want_stress=False)
            tot += b.force.assemble(f_e)
        F.append(tot.copy())
    assert F[0].shape == F[1].shape == (mesh.n_nodes, 3)
    np.testing.assert_allclose(F[1], F[0], rtol=1e-9, atol=1e-9 * np.abs(F[0]).max())


def test_a_single_type_mesh_is_the_one_block_case_of_the_same_path():
    """There is no special case for the common model: it is a list of one."""
    s = solver_for(build(4, 2, 2, "hex"))
    assert len(s.blocks) == 1
    assert s.pre is s.blocks[0].pre           # the aliases still resolve
    assert s.nel == s.blocks[0].n_elem


def test_the_single_block_aliases_refuse_a_mixed_model():
    """So an unconverted path fails at its own line with a sentence, instead of
    quietly computing block zero and reporting it as the whole model."""
    s = solver_for(build(4, 2, 2, "mixed"))
    with pytest.raises(UserError) as exc:
        s.pre
    assert exc.value.code == "MIXED_MESH_UNSUPPORTED"
    assert "hexa8" in exc.value.message and "tet4" in exc.value.message


# --------------------------------------------------------------------------- #
#  ParaView
# --------------------------------------------------------------------------- #


def test_the_result_is_one_mixed_grid_that_paraview_can_read(tmp_path):
    """XDMF `Mixed` puts both element types in ONE grid with one set of fields,
    rather than two grids the user has to overlay by hand."""
    meshio = pytest.importorskip("meshio")
    import re

    from vfife.io.writer import ResultWriter

    mesh = build(4, 2, 2, "mixed")
    nel = sum(len(v) for v in mesh.cell_blocks.values())
    w = ResultWriter(str(tmp_path / "m.h5"), mesh, element="hexa8")
    w.write_frame(0, 0.0, {"displacement": np.zeros((mesh.n_nodes, 3)),
                           "von_mises": np.arange(nel, dtype=float)})
    w.close()

    xdmf = (tmp_path / "m_last.xdmf").read_text(encoding="utf-8")
    assert 'TopologyType="Mixed"' in xdmf
    # meshio's reader does not accept a <Time> tag on a static grid; ParaView
    # does, and it is what makes the frame timestamped, so strip it here rather
    # than write a file that is poorer for the tool people actually use
    probe = tmp_path / "probe.xdmf"
    probe.write_text(re.sub(r"\s*<Time [^>]*/>", "", xdmf), encoding="utf-8")

    m = meshio.read(str(probe))
    assert {c.type for c in m.cells} == {"hexahedron", "tetra"}
    assert sum(len(c.data) for c in m.cells) == nel
    # and the cell data is still in step with the cells, block by block
    assert np.array_equal(np.concatenate(m.cell_data["von_mises"]),
                          np.arange(nel, dtype=float))


# --------------------------------------------------------------------------- #
#  cells written more than once
# --------------------------------------------------------------------------- #


def _msh(tmp_path, name, nodes, element_lines, physical=""):
    p = tmp_path / name
    body = "\n".join(f"{i + 1} {x} {y} {z}"
                      for i, (x, y, z) in enumerate(nodes))
    p.write_text(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n" + physical
        + f"$Nodes\n{len(nodes)}\n{body}\n$EndNodes\n"
        + f"$Elements\n{len(element_lines)}\n"
        + "".join(l + "\n" for l in element_lines) + "$EndElements\n", encoding="utf-8")
    return p


CUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1), (0.5, 0.5, 2)]


def test_a_cell_listed_under_two_groups_becomes_one_element(tmp_path):
    """MSH 2.2 writes an element once per Physical group it belongs to, so a
    volume named twice arrives twice. Both entries are the same cell: keeping
    both would double that region's stiffness and mass, and the run would look
    perfectly normal while being wrong.
    """
    from vfife.io import gmsh

    # the same tetrahedron, under physical 1 (entity 1) and physical 2 (entity 2)
    p = _msh(tmp_path, "twice.msh", CUBE,
             ["1 4 2 1 1 1 2 3 5", "2 4 2 2 2 1 2 3 5"],
             '$PhysicalNames\n2\n3 1 "solid"\n3 2 "whole"\n$EndPhysicalNames\n')
    mesh, _ = gmsh.read(str(p))
    assert len(mesh.cell_blocks["tet4"]) == 1
    # and it still belongs to BOTH groups, which is the point of naming both
    assert "solid" in mesh.regions and "whole" in mesh.regions
    note, = mesh.import_notes
    assert note["merged"] == 1
    assert note["across_entities"] == 1 and note["within_entity"] == 0


def test_a_cell_repeated_inside_one_volume_is_merged_but_reported(tmp_path):
    """Gmsh does this when `Recombine Surface {:}` hits an internal interface:
    it prints `non-manifold quad boundaries not supported yet`, hundreds of
    times, and then writes the file anyway. No physical group explains it, so
    the merge is right but silent is not."""
    from vfife.io import gmsh

    p = _msh(tmp_path, "inside.msh", CUBE,
             ["1 4 2 1 1 1 2 3 5",
              "2 4 2 1 1 3 1 5 2",      # same four nodes, different order
              "3 4 2 1 1 1 2 4 5"])
    mesh, _ = gmsh.read(str(p))
    assert len(mesh.cell_blocks["tet4"]) == 2
    note, = mesh.import_notes
    assert note["within_entity"] == 1 and note["across_entities"] == 0


def test_the_preflight_says_which_kind_of_repeat_it_was(tmp_path):
    """Two causes, two severities: a group choice the user made is an INFO, a
    mesher failure they did not ask for is a WARNING."""
    from vfife.io import gmsh
    from vfife.preflight import _import_notes

    for lines, code, severity in (
        (["1 4 2 1 1 1 2 3 5", "2 4 2 2 2 1 2 3 5"],
         "MERGED_SHARED_ELEMENTS", "INFO"),
        (["1 4 2 1 1 1 2 3 5", "2 4 2 1 1 3 1 5 2"],
         "REPEATED_ELEMENTS_IN_ONE_VOLUME", "WARNING"),
    ):
        mesh, _ = gmsh.read(str(_msh(tmp_path, f"{code}.msh", CUBE, lines)))
        items = []
        _import_notes(mesh, items)
        assert [i["code"] for i in items] == [code]
        assert items[0]["severity"] == severity

    # and the warning carries the fix, because the user cannot deduce it
    mesh, _ = gmsh.read(str(_msh(tmp_path, "w.msh", CUBE,
                                 ["1 4 2 1 1 1 2 3 5", "2 4 2 1 1 3 1 5 2"])))
    items = []
    _import_notes(mesh, items)
    assert "Recombine Surface" in items[0]["details"]["hint"]


def test_a_clean_mesh_is_untouched_and_says_nothing(tmp_path):
    from vfife.io import gmsh

    p = _msh(tmp_path, "clean.msh", CUBE,
             ["1 4 2 1 1 1 2 3 5", "2 4 2 1 1 1 2 4 5"])
    mesh, _ = gmsh.read(str(p))
    assert len(mesh.cell_blocks["tet4"]) == 2
    assert mesh.import_notes == []
    # the element order is the file's own, so element ids still mean something
    assert mesh.cell_blocks["tet4"][0].tolist() == [0, 1, 2, 4]


# --------------------------------------------------------------------------- #
#  a body is a body across its blocks
# --------------------------------------------------------------------------- #


def test_connectivity_is_computed_over_every_block():
    """The check that produced 845 false rigid-body errors.

    `mesh.elements` is the largest block. A boundary-layer model whose
    hexahedra are the minority block had every one of their interior nodes
    counted as its own connected component, and preflight reported one
    unconstrained rigid-body mode for each.
    """
    from vfife.preflight import _components

    mesh = build(6, 3, 3, "mixed")
    assert len(_components(mesh)) == 1

    # and the largest block alone really is disconnected, so the test is not
    # passing for the wrong reason
    single = Mesh(nodes=mesh.nodes, elements=mesh.cell_blocks["tet4"])
    assert len(_components(single)) > 1


def test_a_node_only_the_smaller_block_uses_is_not_an_orphan():
    """ORPHAN_NODE is a property of the model, not of one block."""
    from vfife.preflight import _mesh_checks

    mesh = build(6, 3, 3, "mixed")
    items = []
    _mesh_checks(mesh, "tet4", items)
    assert [i for i in items if i["code"] == "ORPHAN_NODE"] == []


# --------------------------------------------------------------------------- #
#  reinforcement that crosses the interface
# --------------------------------------------------------------------------- #


def bar_along(mesh, z, n=24, L=4.0, y=0.5):
    """A straight bar running the whole length, so it must cross the interface."""
    pts = np.stack([np.linspace(0.0, L, n + 1),
                    np.full(n + 1, y), np.full(n + 1, z)], axis=1)
    conn = np.column_stack([np.arange(n), np.arange(1, n + 1)]).astype(np.int32)
    return pts, conn


def embed(mesh, **kw):
    from vfife.elements import truss

    pts, conn = bar_along(mesh, kw.pop("z", 0.12), L=kw.pop("L", 4.0))
    steel = create("rebar", E=2.0e11, rho=7850.0,
                   yield_strength=4.0e8, hardening=2.0e9)
    return truss.EmbeddedTruss(mesh.nodes, dict(mesh.cell_blocks), None,
                               pts, conn, steel, area=kw.pop("area", 1.0e-3),
                               **kw)


def test_a_bar_can_run_from_one_element_block_into_another():
    """It could not.

    `EmbeddedTruss` was handed `mesh.elements` -- the largest block -- together
    with the element type the case declared. On a mixed mesh those disagree, and
    the locator died inside numpy with

        matmul: Input operand 1 has a mismatch in its core dimension 0
                (size 4 is different from 8)

    Concrete with reinforcement in a geometry that needs both hexahedra and
    tetrahedra is the ordinary case in this field, so the two features were
    unusable exactly where they are most wanted.
    """
    mesh = build(6, 3, 3, "mixed")
    emb = embed(mesh)

    hexa = set(np.unique(mesh.cell_blocks["hexa8"]).tolist())
    tets = set(np.unique(mesh.cell_blocks["tet4"]).tolist())
    used = set(np.unique(emb.conn).tolist())
    assert used & hexa, "the bar reaches no hexahedron"
    assert used & tets, "the bar reaches no tetrahedron -- the test is vacuous"

    # rows are padded to the widest block; the padding must carry no weight
    assert emb.n_host_nodes == 8
    assert np.abs(emb.w.sum(axis=1) - 1.0).max() < 1e-12, (
        "shape functions must still be a partition of unity in every block")


def test_a_uniform_axial_strain_reaches_every_bar_segment_equally():
    """The patch test for the coupling, and the sharp one.

    Impose u = (eps*x, 0, 0) on the host: every segment must read exactly
    eps, whichever block interpolates it. A wrong cross-block interpolation
    would show up here as a jump at the interface and nowhere else -- the
    bracketing test below would still pass, because a small coupling error is
    invisible next to the difference between hexahedra and tetrahedra.
    """
    mesh = build(6, 3, 3, "mixed")
    emb = embed(mesh)

    eps = 1.0e-4
    u = np.zeros_like(mesh.nodes)
    u[:, 0] = eps * mesh.nodes[:, 0]
    _f, sig = emb.force(u, 1.0, want_stress=True)

    axial = np.asarray(sig).reshape(emb.n_elem, -1)[:, 0]
    expected = 2.0e11 * eps
    assert axial.min() == pytest.approx(expected, rel=1e-9)
    assert (axial.max() - axial.min()) / abs(axial.mean()) < 1e-9, (
        "the segments disagree about a constant strain, so the host "
        "interpolation differs across the block boundary")


def test_the_host_weights_come_from_the_block_the_point_is_actually_in():
    """Each bar node interpolates from ONE element, and it must be an element
    that really contains it -- not the nearest one in whichever block happens
    to be searched first."""
    from vfife.elements import truss

    mesh = build(6, 3, 3, "mixed")
    pts, _conn = bar_along(mesh, 0.12)
    host_nodes, w = truss.locate_in_blocks(
        mesh.nodes, list(mesh.cell_blocks.items()), pts)

    # the interpolated position must be the point itself
    got = np.einsum("pa,pai->pi", w, mesh.nodes[host_nodes])
    assert np.abs(got - pts).max() < 1e-9, (
        "a host element was accepted that does not contain its point")


def test_a_bar_outside_the_mesh_is_still_refused_by_name():
    """The locator must not silently fall back to a nearest element just
    because there are now several blocks to look in."""
    from vfife.elements import truss

    mesh = build(4, 2, 2, "mixed")
    outside = np.array([[2.0, 0.5, 99.0]])
    with pytest.raises(ValueError) as exc:
        truss.locate_in_blocks(mesh.nodes, list(mesh.cell_blocks.items()), outside)
    assert "outside the mesh" in str(exc.value)
    assert "hexa8" in str(exc.value) and "tet4" in str(exc.value)


@pytest.mark.slow
# Solving to convergence three times. The patch test above already pins the
# coupling exactly; this one pins that the RESULT is physically ordered.
def test_reinforcement_stiffens_a_mixed_mesh_between_its_two_references():
    """Same bar, three meshes. The mixed result must land between all-hex and
    all-tet, as the unreinforced deflections do."""
    gain = {}
    for kind in ("hex", "mixed", "tet"):
        deflection = {}
        for reinforced in (False, True):
            mesh = build(6, 3, 3, kind)
            x = mesh.nodes
            mesh.regions["root"] = np.flatnonzero(np.abs(x[:, 0]) < 1e-12)
            mesh.regions["tip"] = np.flatnonzero(np.abs(x[:, 0] - 4.0) < 1e-12)
            problem = {
                "element": {"type": "hexa8", "formulation": "full"},
                "constraints": [{"region": "root", "dofs": [1, 1, 1]}],
                "loads": [{"region": "tip", "vector": [0.0, 0.0, -5.0e4]}],
                "solver": {"mode": "relaxation", "n_steps": 40000,
                           "backend": "numpy"},
                "output": {"monitors": {
                    "w": {"kind": "displacement", "region": "tip", "dof": "z"},
                    "R": {"kind": "reaction", "region": "root", "dof": "z"}}},
            }
            if reinforced:
                problem["reinforcement"] = [{
                    "name": "bar", "area": 5.0e-3, "host_deduction": 3.0e10,
                    "lines": [{"start": [0.0, 0.5, 0.08],
                               "end": [4.0, 0.5, 0.08], "divisions": 24}],
                    "material": {"type": "rebar", "E": 2.0e11, "rho": 7850.0,
                                 "yield_strength": 4.0e8, "hardening": 2.0e9}}]
            s = Solver(problem, mesh,
                       create("linear_elastic", E=3.0e10, nu=0.2, rho=2400.0),
                       log=lambda *_: None)
            s.run()
            deflection[reinforced] = s.monitors["w"].values[-1]
            # Global equilibrium is checked separately, and loosely, because it
            # is knowingly broken on one of these six combinations -- see
            # test_reinforced_hex_equilibrium_drifts below. A tight assertion
            # here would take the mesh-bracketing check down with it, and that
            # check is this test's actual subject and works.
            assert s.monitors["R"].values[-1] == pytest.approx(5.0e4, rel=5e-3)
        gain[kind] = 1.0 - deflection[True] / deflection[False]
        assert gain[kind] > 0.0, f"{kind}: the bar made it softer"

    lo, hi = sorted((gain["hex"], gain["tet"]))
    assert lo <= gain["mixed"] <= hi, (
        f"stiffening {gain} is not bracketed by the single-type meshes")


@pytest.mark.slow
def test_reinforced_hex_holds_equilibrium_at_the_default_tolerance():
    """The root reaction equals the applied load. It did not, and for a long
    time this test was an xfail that blamed the embedded bar's fictitious mass.

    WHAT IT WAS
        Nothing drifted. A continuous relaxation run ramps its load over the
        first half of the budget and may declare convergence at the first
        check after the ramp ends. On that check the state has been following
        a moving target until that very step; with the bar slowing the
        transient, the residual there was 7e-5 -- under the 1e-4 tolerance --
        and the run stopped with the root reaction 0.10% (40 000 steps) or
        0.24% (80 000) off the load. The plain hex converges far below the
        tolerance during the ramp, so it never showed it. Every fact in the
        old write-up fits: the "drift" grew with n_steps because a slower
        ramp ends closer to the tolerance; tolerance 1e-30 never stopped and
        so was exact; heavier bar mass slowed the ramp transient the other
        way; the bonded bar's own dofs converge faster.

        The exit now requires the loading to have settled at the PREVIOUS
        iteration as well (core/solver.py, `factor_prev`), the rule the
        staged path always had in `stage_ramp`. One interval later the
        residual is 3e-8 and the reaction is exact to 3e-7 at every budget.
        `_emb_row_sums` was right all along, as tests/test_stability_row_sums.py
        had already established.
    """
    mesh = build(6, 3, 3, "hex")
    x = mesh.nodes
    mesh.regions["root"] = np.flatnonzero(np.abs(x[:, 0]) < 1e-12)
    mesh.regions["tip"] = np.flatnonzero(np.abs(x[:, 0] - 4.0) < 1e-12)
    problem = {
        "element": {"type": "hexa8", "formulation": "full"},
        "constraints": [{"region": "root", "dofs": [1, 1, 1]}],
        "loads": [{"region": "tip", "vector": [0.0, 0.0, -5.0e4]}],
        "solver": {"mode": "relaxation", "n_steps": 40000, "backend": "numpy"},
        "output": {"monitors": {
            "R": {"kind": "reaction", "region": "root", "dof": "z"}}},
        "reinforcement": [{
            "name": "bar", "area": 5.0e-3, "host_deduction": 3.0e10,
            "lines": [{"start": [0.0, 0.5, 0.08],
                       "end": [4.0, 0.5, 0.08], "divisions": 24}],
            "material": {"type": "rebar", "E": 2.0e11, "rho": 7850.0,
                         "yield_strength": 4.0e8, "hardening": 2.0e9}}],
    }
    s = Solver(problem, mesh,
               create("linear_elastic", E=3.0e10, nu=0.2, rho=2400.0),
               log=lambda *_: None)
    s.run()
    assert s.monitors["R"].values[-1] == pytest.approx(5.0e4, rel=1e-4)
