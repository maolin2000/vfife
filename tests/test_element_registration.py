"""One element, one spelling.

An element used to be named in five independent places: `elements/library.py`,
the `Literal` in `io/schema.py`, `_TOPOLOGY` in `io/writer.py`, `_VOLUME` in
`io/gmsh.py`, and a hard-coded set in `cli.make_solver`. Nothing derived any of
them from any other, so adding an element was five chances to forget, and the
failure was silent.

It had already happened. `wedge15` was filed as XDMF `"Wedge"` -- which the
XDMF specification defines as the SIX-node prism -- while writing fifteen nodes
per element, so ParaView misread every 15-node prism result. Its neighbours
`hexa20` and `tet10` carry the `_20` / `_10` suffix correctly, which is what
makes it a copying mistake rather than a misunderstanding.

These tests fail if the spellings ever diverge again, and the node-count check
is the one that would have caught the original bug.
"""

import os
import numpy as np
import pytest

from vfife.cli import make_solver  # noqa: F401  (import must not break)
from vfife.elements import library
from vfife.io import gmsh, writer
from vfife.io.schema import ElementSpec

DEFS = library.element_definitions()

#: nodes per element implied by each XDMF topology name, from the XDMF3 model.
#: A topology name carries its node count, so a mismatch here is exactly the
#: wedge15 bug.
XDMF_NODES = {
    "Polyline": None,          # the count travels in NodesPerElement
    "Triangle": 3,
    "Quadrilateral": 4,
    "Tetrahedron": 4,
    "Tetrahedron_10": 10,
    "Pyramid": 5,
    "Wedge": 6,
    "Wedge_15": 15,
    "Wedge_18": 18,
    "Hexahedron": 8,
    "Hexahedron_20": 20,
    "Hexahedron_27": 27,
}


def test_every_element_the_schema_accepts_is_defined():
    """And nothing else is. The Literal is built from the library."""
    schema_names = set(
        ElementSpec.model_json_schema()["properties"]["type"]["enum"])
    assert schema_names == set(DEFS)


def test_the_writer_topology_table_is_derived():
    assert writer._TOPOLOGY == library.xdmf_topologies()
    for name, (topo, nne, _extra) in writer._TOPOLOGY.items():
        assert DEFS[name].xdmf == (topo, nne, _extra)


@pytest.mark.parametrize("name", sorted(DEFS))
def test_xdmf_topology_name_matches_its_node_count(name):
    """The check that would have caught wedge15.

    `Wedge` is 6 nodes in XDMF; writing 15 under that name produces a file
    ParaView reads without complaint and draws wrongly.
    """
    spec = DEFS[name].xdmf
    if spec is None:
        return
    topo, nne, extra = spec
    assert topo in XDMF_NODES, f"{name}: unknown XDMF topology {topo!r}"
    expected = XDMF_NODES[topo]
    if expected is None:                      # Polyline states its own count
        assert f'NodesPerElement="{nne}"' in extra
    else:
        assert nne == expected, (
            f"{name} is written as XDMF {topo!r}, which is {expected} nodes, "
            f"but declares {nne}"
        )


def test_the_gmsh_reader_tables_are_derived():
    both = {**gmsh._VOLUME, **gmsh._STRUCTURAL}
    assert both == library.gmsh_types()
    # and the split is by family, not by a second hand-written list
    for t, (name, _perm) in gmsh._VOLUME.items():
        assert DEFS[name].family == "continuum"
    for t, (name, _perm) in gmsh._STRUCTURAL.items():
        assert DEFS[name].family != "continuum"


@pytest.mark.parametrize("name", sorted(DEFS))
def test_a_gmsh_permutation_is_a_permutation_of_the_right_length(name):
    d = DEFS[name]
    if d.gmsh_type is None:
        assert d.gmsh_order is None
        return
    n = d.xdmf[1] if d.xdmf else len(d.gmsh_order)
    assert sorted(d.gmsh_order) == list(range(n)), (
        f"{name}: gmsh_order is not a permutation of 0..{n - 1}"
    )


def test_gmsh_type_ids_are_unique():
    ids = [d.gmsh_type for d in DEFS.values() if d.gmsh_type is not None]
    assert len(ids) == len(set(ids))


def test_the_solver_chooser_reads_the_family():
    """A new structural element must not run through the solid solver.

    The membership test used to be a literal set in `cli.make_solver`; an
    element added to the library but forgotten there ran silently on the
    continuum path.
    """
    fams = library.families()
    assert set(fams) == set(DEFS)
    assert fams["hexa8"] == "continuum"
    assert fams["beam2"] != "continuum"
    # every continuum element is in ELEMENTS, and only those
    assert {n for n, f in fams.items() if f == "continuum"} == set(library.ELEMENTS)


def test_a_continuum_element_carries_everything_it_needs():
    """The registration facts are not optional for a mesh-file element."""
    for name, elem in library.ELEMENTS.items():
        assert elem.xdmf is not None, f"{name} has no XDMF spelling"
        assert elem.gmsh_type is not None, f"{name} has no Gmsh type"
        assert elem.xdmf[1] == elem.n_nodes, (
            f"{name} declares {elem.n_nodes} nodes but writes {elem.xdmf[1]}"
        )


def test_wedge15_writes_the_fifteen_node_topology():
    """The specific bug, pinned so it cannot come back."""
    assert library.ELEMENTS["wedge15"].xdmf == ("Wedge_15", 15, "")
    assert writer._TOPOLOGY["wedge15"][0] != "Wedge"


def test_every_xdmf_topology_has_a_mixed_type_code():
    """A mixed mesh writes a flat `[type code, nodes...]` stream, so an element
    without a code cannot be written at all -- and the failure would be an
    unhandled KeyError deep in the writer, at the end of an analysis."""
    for name, spec in library.xdmf_topologies().items():
        assert spec[0] in writer._MIXED_CODE, (
            f"{name} is written as XDMF {spec[0]!r}, which has no Mixed "
            f"topology code; add it to writer._MIXED_CODE"
        )


def test_the_mixed_type_codes_are_the_ones_readers_expect():
    """Checked against meshio's table, which is the same one ParaView uses.

    A wrong code here is the wedge15 bug again in a new place: the file reads
    without complaint and draws the wrong cells.
    """
    meshio_common = pytest.importorskip("meshio.xdmf.common")
    theirs = meshio_common.xdmf_idx_to_meshio_type
    expected = {
        "Triangle": "triangle", "Quadrilateral": "quad",
        "Tetrahedron": "tetra", "Pyramid": "pyramid", "Wedge": "wedge",
        "Hexahedron": "hexahedron", "Tetrahedron_10": "tetra10",
        "Wedge_15": "wedge15", "Wedge_18": "wedge18",
        "Hexahedron_20": "hexahedron20", "Hexahedron_27": "hexahedron27",
    }
    for topo, cell in expected.items():
        assert theirs[writer._MIXED_CODE[topo]] == cell, topo


# --------------------------------------------------------------------------- #
#  the permutation has to be the RIGHT permutation
# --------------------------------------------------------------------------- #

#: Gmsh's own edge order for each second-order element, from the Gmsh manual's
#: node-ordering figures. Gmsh numbers edges by (lowest corner, then partner),
#: which is NOT the order VTK/XDMF uses.
GMSH_EDGES = {
    "hexa20": [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3),
               (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)],
    "wedge15": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4),
                (2, 5), (3, 4), (3, 5), (4, 5)],
    "tet10": [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)],
}

#: The order the solver's own shape functions expect, i.e. VTK/XDMF.  Verified
#: against the shape functions themselves, not assumed: placing a straight
#: tet10's mid-side nodes by this table gives the exact volume 1/6, and by the
#: other candidate ordering gives a non-positive Jacobian.  The GMSH_EDGES
#: tables above were read back out of real Gmsh 4.15 output the same way.
INTERNAL_EDGES = {
    "hexa20": [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
               (0, 4), (1, 5), (2, 6), (3, 7)],
    "wedge15": [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3),
                (0, 3), (1, 4), (2, 5)],
    "tet10": [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)],
}


@pytest.mark.parametrize("name", sorted(GMSH_EDGES))
def test_the_permutation_maps_gmsh_edges_onto_the_internal_ones(name):
    """The check `is it a permutation` could not catch this, because the
    identity is a permutation.

    `hexa20` and `wedge15` both declared the identity, so every mid-side node
    landed on the wrong edge and the element came out inside out: `128
    integration points of hexa20 have non-positive Jacobian` on the very first
    Gmsh file, for every mesh. Two of the seven solid elements could not be
    read at all, and nothing in the suite noticed.
    """
    order = library.ELEMENTS[name].gmsh_order
    n_corner = len(set(sum(GMSH_EDGES[name], ())))
    assert list(order[:n_corner]) == list(range(n_corner)), (
        f"{name}: corners must map to themselves")
    for slot, edge in enumerate(INTERNAL_EDGES[name]):
        gmsh_slot = order[n_corner + slot]
        got = GMSH_EDGES[name][gmsh_slot - n_corner]
        assert set(got) == set(edge), (
            f"{name}: internal slot {n_corner + slot} should hold edge {edge}, "
            f"but the permutation takes Gmsh slot {gmsh_slot}, which is edge {got}"
        )


@pytest.mark.parametrize("name", sorted(GMSH_EDGES))
def test_a_permuted_element_has_its_midside_nodes_on_its_edges(name):
    """The same statement in geometry, which is what actually goes wrong.

    Build a straight-sided element with Gmsh's node order, apply the declared
    permutation, and every mid-side node must land on the midpoint of the edge
    the internal shape functions expect it on.
    """
    corners = {
        "hexa20": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                   (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        "wedge15": [(0, 0, 0), (1, 0, 0), (0, 1, 0),
                    (0, 0, 1), (1, 0, 1), (0, 1, 1)],
        "tet10": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
    }[name]
    x = np.array(corners, dtype=float)
    n = len(corners)
    gmsh_nodes = np.vstack(
        [x] + [0.5 * (x[a] + x[b]) for a, b in GMSH_EDGES[name]])

    placed = gmsh_nodes[list(library.ELEMENTS[name].gmsh_order)]
    for slot, (a, b) in enumerate(INTERNAL_EDGES[name]):
        assert np.allclose(placed[n + slot], 0.5 * (x[a] + x[b])), (
            f"{name}: node {n + slot} should be the midpoint of edge {(a, b)}"
        )


# --------------------------------------------------------------------------- #
#  faces
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(library.ELEMENTS))
def test_every_node_of_a_solid_element_lies_on_one_of_its_faces(name):
    """None of these elements has an interior node, so the faces must account
    for all of them.

    `tet10.faces` was a copy of tet4's four CORNER triangles. The `surface`
    region is built by counting faces, so on a tet10 model it returned corners
    only: 14 of the 50 boundary nodes of a unit cube, 72% missing. A constraint
    or a pressure on `surface` then applied to a quarter of the face it named,
    and nothing said so.
    """
    elem = library.ELEMENTS[name]
    covered = set()
    for face in elem.faces:
        covered |= set(face)
    assert covered == set(range(elem.n_nodes)), (
        f"{name}: nodes {sorted(set(range(elem.n_nodes)) - covered)} appear on "
        f"no face, so the `surface` region cannot see them"
    )


@pytest.mark.parametrize("name", sorted(library.ELEMENTS))
def test_a_face_carries_the_right_number_of_nodes_for_its_order(name):
    """A quadratic element has quadratic faces: 6-node triangles and 8-node
    quadrilaterals, never the 3 and 4 of its corner outline."""
    elem = library.ELEMENTS[name]
    quadratic = name in ("tet10", "hexa20", "wedge15")
    for face in elem.faces:
        assert len(face) in ((6, 8) if quadratic else (3, 4)), (
            f"{name}: a face with {len(face)} nodes in a "
            f"{'quadratic' if quadratic else 'linear'} element"
        )


# --------------------------------------------------------------------------- #
#  files Gmsh actually wrote
# --------------------------------------------------------------------------- #

#: Every solid element, and a real Gmsh 4.1 file containing it. This mapping is
#: the point of the whole section: `hexa20` and `wedge15` had unit tests, shape
#: function tests, quadrature tests -- and no file Gmsh had ever written. Their
#: fixtures were hand-typed with the mid-side nodes in VTK order inside a file
#: labelled Gmsh type 17/18, which Gmsh never emits. The reader's permutation
#: was the identity, the two errors cancelled, and both elements were unusable
#: on any real mesh for as long as they had existed.
#:
#: A path that no real file has travelled is a path that has not been tested.
GMSH_FIXTURES = {
    "tet4": "box_tet4_v41.msh",
    "tet10": "box_tet10_v41.msh",
    "hexa8": "box_hexa8_v41.msh",
    "hexa20": "box_hexa20_v41.msh",
    "wedge6": "box_wedge6_v41.msh",
    "wedge15": "box_wedge15_v41.msh",
    "pyramid5": "box_pyramid5_v41.msh",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_every_solid_element_has_a_real_gmsh_fixture():
    """Not a hand-typed one. The set must not shrink and must not lag the
    library: a new element without a real file is an untested import."""
    assert set(GMSH_FIXTURES) == set(library.ELEMENTS)
    for name, fname in GMSH_FIXTURES.items():
        assert os.path.exists(os.path.join(DATA_DIR, fname)), name


@pytest.mark.parametrize("name", sorted(GMSH_FIXTURES))
def test_an_element_read_from_gmsh_is_the_right_way_out(name):
    """Positive Jacobians and the exact volume of a unit cube.

    This is the assertion the identity permutation could not survive: it gave
    `128 integration points of hexa20 have non-positive Jacobian` on the first
    element of the first file.
    """
    from vfife.elements import continuum
    from vfife.io import gmsh as gmsh_io
    from vfife.materials import create

    mesh, _primary = gmsh_io.read(os.path.join(DATA_DIR, GMSH_FIXTURES[name]))
    conn = mesh.cell_blocks[name]
    mat = create("linear_elastic", E=3.0e10, nu=0.2, rho=2400.0)

    pre = continuum.precompute(mesh.nodes, conn, element=name,
                               formulation="full", material=mat)
    assert (pre.volume > 0).all(), f"{name}: a negative element volume"

    # Every fixture is a box, so the elements must add up to the box -- taken
    # from the file's own extent rather than written down here, so a new
    # fixture of any size needs no magic number.
    #
    # The pyramid5 fixture is a MIXED mesh (its pyramids are the Gmsh
    # transition layer under one recombined face), so its pyramids fill only
    # part of the box and every block has to be summed.
    lo, hi = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)
    box = float(np.prod(hi - lo))
    total = sum(float(continuum.precompute(
                    mesh.nodes, c, element=n, formulation="full",
                    material=mat).volume.sum())
                for n, c in mesh.cell_blocks.items())
    assert total == pytest.approx(box, rel=1e-12), (
        f"{name}: the elements add up to {total}, not the {box} their own "
        f"bounding box spans, so the node ordering out of the reader is wrong"
    )


@pytest.mark.parametrize("name", sorted(GMSH_FIXTURES))
def test_an_element_read_from_gmsh_passes_a_patch_test(name):
    """A constant strain must come out constant.

    The volume check alone can pass for a wrong-but-symmetric ordering; this
    one cannot. Together they pin the permutation.
    """
    from vfife.elements import continuum
    from vfife.io import gmsh as gmsh_io
    from vfife.materials import create
    from vfife.materials.points import initialise_points

    mesh, _primary = gmsh_io.read(os.path.join(DATA_DIR, GMSH_FIXTURES[name]))
    conn = mesh.cell_blocks[name]
    mat = create("linear_elastic", E=3.0e10, nu=0.2, rho=2400.0)
    pre = continuum.precompute(mesh.nodes, conn, element=name,
                               formulation="full", material=mat)

    A = np.array([[1e-4, 2e-5, 3e-5], [0.0, -3e-5, 1e-5], [0.0, 0.0, 2e-5]])
    u = mesh.nodes @ A.T
    x0 = mesh.nodes[conn]
    x0_e = np.ascontiguousarray(x0 - x0.mean(axis=1, keepdims=True))
    state, _pts = initialise_points(
        mat, conn.shape[0] * pre.n_gp, kinematics="solid_3d", backend="numpy",
        char_length=np.repeat(pre.char_len, pre.n_gp))
    quat = np.zeros((conn.shape[0], 4))
    quat[:, 0] = 1.0

    _f, sig = continuum.internal_force(pre, x0_e, u[conn], quat, mat, state,
                                       1.0, rot_iters=30, want_stress=True)
    s = sig.reshape(-1, 6)
    spread = np.abs(s.max(axis=0) - s.min(axis=0)).max()
    assert spread / max(np.abs(s).max(), 1e-30) < 1e-8, (
        f"{name}: a constant strain field came out non-constant, so the nodes "
        f"are not where the shape functions think they are"
    )
