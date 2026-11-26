"""Mesh container, structured box generator, and named region resolution.

Regions are named once and resolved to node index arrays once, at setup.
The original code called `find_nodes_in_region` -- a full sweep over every node
-- inside the time loop, for every constraint and every load, 5900 times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import math

import numpy as np

from ..base.scatter import scatter_add  # noqa: F401  (`core.mesh.scatter_add` before 1.22)

# VTK_HEXAHEDRON node ordering, matching elements/hexa8.py
_OFF = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
)


@dataclass
class Mesh:
    nodes: np.ndarray                 # (n_nodes, 3)
    elements: np.ndarray              # (n_elem, 8) int32
    regions: Dict[str, np.ndarray] = field(default_factory=dict)
    # Optional additional topology blocks for conforming multi-family models.
    # The primary ``elements`` block remains for backwards compatibility.
    cell_blocks: Dict[str, np.ndarray] = field(default_factory=dict)
    #: region name -> block name -> element indices INTO THAT BLOCK.
    #
    # A region has always been a set of nodes, and an element belonged to it
    # when all of its nodes did. That rule cannot separate the beams of a
    # building from its columns: every column's two nodes are also beam nodes,
    # so both groups claim every column and the model is refused as
    # over-assigned. A Gmsh physical group knows perfectly well
    # which ELEMENTS it holds; this is that information, kept rather than
    # thrown away at import. Empty for meshes built in memory, and every
    # consumer falls back to the node rule, so nothing that worked before
    # depends on it.
    element_regions: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)

    def region_elements(self, name: str, block: str, elements: np.ndarray):
        """Element indices of region `name` within `block`.

        The recorded groups win where they exist; otherwise the old rule --
        every node of the element inside the region -- which is what a region
        written as a box or a plane can mean at all.
        """
        recorded = self.element_regions.get(name, {}).get(block)
        if recorded is not None:
            return np.asarray(recorded, dtype=np.int64)
        nodes = self.regions.get(name)
        if nodes is None:
            return None
        inside = np.zeros(self.n_nodes, dtype=bool)
        inside[np.asarray(nodes, dtype=np.int64)] = True
        return np.flatnonzero(inside[elements].all(axis=1))

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elem(self) -> int:
        return self.elements.shape[0]

    def resolve_region(self, name: str, spec: dict, tol: float = 1e-9) -> np.ndarray:
        """Resolve a region spec to a sorted node index array and cache it."""
        if "box" in spec:
            b = np.asarray(spec["box"], dtype=float)
            if b.shape != (6,):
                raise ValueError(
                    f"region {name!r}: 'box' must be "
                    "[xmin,xmax,ymin,ymax,zmin,zmax]"
                )
            x, y, z = self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2]
            m = (
                (x >= b[0] - tol) & (x <= b[1] + tol)
                & (y >= b[2] - tol) & (y <= b[3] + tol)
                & (z >= b[4] - tol) & (z <= b[5] + tol)
            )
        elif "node_ids" in spec:
            m = np.zeros(self.n_nodes, dtype=bool)
            m[np.asarray(spec["node_ids"], dtype=np.int64)] = True
        else:
            raise ValueError(f"region {name!r}: expected 'box' or 'node_ids'")

        idx = np.flatnonzero(m).astype(np.int64)
        if idx.size == 0:
            raise ValueError(f"region {name!r} selected no nodes")
        self.regions[name] = idx
        return idx


# Kuhn/Freudenthal subdivision of a hexahedron along its 0-6 diagonal.  Six
# tetrahedra, no new nodes, and consistent across neighbours so faces match.
_HEX_TO_TET = np.array([
    [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
    [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6],
])
# two prisms, cutting the 0-2 diagonal below and 4-6 above
_HEX_TO_WEDGE = np.array([[0, 1, 2, 4, 5, 6], [0, 2, 3, 4, 6, 7]])
# mid-edge nodes of a tet10, in the order library.py expects
_TET10_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


def box_mesh(
    dimensions, divisions, origin=(0.0, 0.0, 0.0), element: str = "hexa8"
) -> Mesh:
    """Structured box. dimensions = (Lx,Ly,Lz), divisions = (nx,ny,nz).

    `element` selects what the hexahedra are emitted as: `hexa8` directly,
    `tet4`/`tet10` via a Kuhn subdivision into six tetrahedra, `wedge6` via two
    prisms.  Having tets without needing Gmsh is what makes them testable
    against the same patch tests and convergence studies as the hexahedron.
    """
    Lx, Ly, Lz = (float(v) for v in dimensions)
    nx, ny, nz = (int(v) for v in divisions)
    ox, oy, oz = (float(v) for v in origin)

    xs = np.linspace(ox, ox + Lx, nx + 1)
    ys = np.linspace(oy, oy + Ly, ny + 1)
    zs = np.linspace(oz, oz + Lz, nz + 1)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    # node id = k*(nx+1)*(ny+1) + j*(nx+1) + i
    nodes = np.stack(
        [X.transpose(2, 1, 0).ravel(),
         Y.transpose(2, 1, 0).ravel(),
         Z.transpose(2, 1, 0).ravel()],
        axis=1,
    )

    i, j, k = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    base = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)   # (nel, 3)
    ijk = base[:, None, :] + _OFF[None, :, :]                    # (nel, 8, 3)
    conn = (
        ijk[:, :, 2] * (nx + 1) * (ny + 1)
        + ijk[:, :, 1] * (nx + 1)
        + ijk[:, :, 0]
    ).astype(np.int32)

    mesh = Mesh(nodes=np.ascontiguousarray(nodes),
                elements=np.ascontiguousarray(conn))
    if element == "hexa8":
        return mesh
    if element == "wedge6":
        return _split(mesh, _HEX_TO_WEDGE)
    if element in ("tet4", "tet10"):
        tet = _split(mesh, _HEX_TO_TET)
        return tet if element == "tet4" else add_midside_nodes(tet)
    raise ValueError(f"box_mesh cannot emit {element!r}")


def _split(mesh: Mesh, pattern: np.ndarray) -> Mesh:
    """Replace every hexahedron by `pattern`, then fix any inverted element.

    A subdivision pattern that is right-handed for one hexahedron ordering can
    be left-handed for another, and an inverted element is a negative Jacobian
    that the precompute rejects.  Rather than trust the table, check the signed
    volume and swap two nodes where it is negative -- cheap, once, at setup.
    """
    conn = mesh.elements[:, pattern].reshape(-1, pattern.shape[1])
    x = mesh.nodes[conn]
    if pattern.shape[1] == 4:
        vol = np.einsum("ei,ei->e", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]),
                        x[:, 3] - x[:, 0])
    else:                                   # prism: sign of the bottom triangle
        n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
        vol = np.einsum("ei,ei->e", n, x[:, 3] - x[:, 0])
    flip = vol < 0
    if flip.any():
        if pattern.shape[1] == 4:
            conn[flip, 1], conn[flip, 2] = conn[flip, 2].copy(), conn[flip, 1].copy()
        else:
            for a, b in ((1, 2), (4, 5)):
                conn[flip, a], conn[flip, b] = (conn[flip, b].copy(),
                                                conn[flip, a].copy())
    return Mesh(nodes=mesh.nodes, elements=np.ascontiguousarray(conn, np.int32))


def add_midside_nodes(mesh: Mesh) -> Mesh:
    """tet4 -> tet10. Mid-edge nodes are shared between neighbouring elements."""
    conn4 = mesh.elements
    edges = np.stack([conn4[:, list(e)] for e in _TET10_EDGES], axis=1)  # (nel,6,2)
    key = np.sort(edges, axis=2)
    flat = key.reshape(-1, 2)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    mid = 0.5 * (mesh.nodes[uniq[:, 0]] + mesh.nodes[uniq[:, 1]])
    nodes = np.vstack([mesh.nodes, mid])
    conn10 = np.hstack([conn4, (mesh.n_nodes + inv).reshape(-1, 6)])
    return Mesh(nodes=np.ascontiguousarray(nodes),
                elements=np.ascontiguousarray(conn10, np.int32))


# Boundary-face node order follows the element library's corner and mid-side
# order.  Tet10 faces carry all six nodes so a traction reaches its quadratic
# shape functions, rather than being (incorrectly) spread over every node in a
# named surface group.
_FACETS = {
    "tet4": ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
    "tet10": ((0, 2, 1, 6, 5, 4), (0, 1, 3, 4, 8, 7),
              (1, 2, 3, 5, 9, 8), (2, 0, 3, 6, 7, 9)),
    "hexa8": ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
               (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
    "wedge6": ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3),
               (1, 2, 5, 4), (2, 0, 3, 5)),
    # The quadratic and pyramidal solids had no entry at all, and all three
    # declared `surface_pressure` in the element catalogue regardless: a
    # `pressure:` on a hexa20 model raised `KeyError: 'hexa20'` from inside
    # the face walk, which is not a message anybody can act on. A face lists
    # its corner nodes first and then its edge mid-nodes in the same cyclic
    # order, which is what the shape functions below assume.
    "pyramid5": ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
    "hexa20": ((0, 3, 2, 1, 11, 10, 9, 8), (4, 5, 6, 7, 12, 13, 14, 15),
               (0, 1, 5, 4, 8, 17, 12, 16), (1, 2, 6, 5, 9, 18, 13, 17),
               (2, 3, 7, 6, 10, 19, 14, 18), (3, 0, 4, 7, 11, 16, 15, 19)),
    "wedge15": ((0, 2, 1, 8, 7, 6), (3, 4, 5, 9, 10, 11),
                (0, 1, 4, 3, 6, 13, 9, 12), (1, 2, 5, 4, 7, 14, 10, 13),
                (2, 0, 3, 5, 8, 12, 11, 14)),
}


#: A point strictly inside each element, for deciding which way a face's
#: normal points. It used to be the mean of the element's first FOUR nodes,
#: which for a hexahedron is the centre of its own bottom face: the outward
#: test for that face was then the sign of a quantity that is exactly zero in
#: the reference configuration and whatever rounding makes it once the mesh
#: moves. On a structured box it came out right, which is why the whole case
#: library passed; distort the same box by 2% of an element and a uniform
#: pressure over the CLOSED body acquired a net resultant of 19% of the load
#: it applies, because some faces were being pushed outward
#: (found by the closed-body check in tests/test_invariants.py). The mean of
#: every node is interior for any convex element, linear or quadratic.
def _interior_points(x: np.ndarray, elements: np.ndarray) -> np.ndarray:
    return x[elements].mean(axis=1)


def surface_traction(nodes: np.ndarray, elements: np.ndarray, element: str,
                     region: np.ndarray, traction, shared=None,
                     allow_empty: bool = False) -> np.ndarray:
    """Consistent nodal force for a constant traction on an exterior region.

    ``region`` is normally a named Gmsh Physical Surface.  Only exterior faces
    whose *complete* interpolation node set lies in that group are loaded.  A
    uniform pressure thus integrates over geometry rather than depending on
    the number or distribution of surface nodes.  Tet10 uses a 7-point
    triangle rule and its six quadratic face functions; linear triangles and
    bilinear quadrilaterals use their matching Gaussian rules.
    """
    if element not in _FACETS:
        raise ValueError(f"surface traction is not implemented for {element!r}")
    force = np.zeros((nodes.shape[0], 3), dtype=float)
    for face, _owner in _region_exterior_faces(nodes, elements, element, region, shared):
        if face.shape[1] == 8:
            local_force = _quad8_traction(nodes[face], traction)
        elif face.shape[1] == 6:
            local_force = _tri6_traction(nodes[face], traction)
        elif face.shape[1] == 3:
            xyz = nodes[face]
            area = 0.5 * np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0],
                                                 xyz[:, 2] - xyz[:, 0]), axis=1)
            local_force = area[:, None, None] * np.asarray(traction) / 3.0
            local_force = np.broadcast_to(local_force, (len(face), 3, 3))
        else:
            local_force = _quad4_traction(nodes[face], traction)
        force += scatter_add(local_force, face, nodes.shape[0])
    if not np.any(force) and not allow_empty:
        raise ValueError("surface traction region contains no complete exterior faces")
    return force


def surface_pressure(nodes: np.ndarray, elements: np.ndarray, element: str,
                     region: np.ndarray, pressure: float, shared=None,
                     allow_empty: bool = False) -> np.ndarray:
    """Consistent inward pressure force for exterior faces in ``region``."""
    if pressure < 0:
        raise ValueError("pressure is positive in compression; use traction for signed loading")
    force = np.zeros((nodes.shape[0], 3), dtype=float)
    centres = _interior_points(nodes, elements)
    for face, owner in _region_exterior_faces(nodes, elements, element, region, shared):
        xyz = nodes[face]
        if face.shape[1] == 8:
            local_force = _quad8_pressure(xyz, centres[owner], pressure)
        elif face.shape[1] == 6:
            local_force = _tri6_pressure(xyz, centres[owner], pressure)
        elif face.shape[1] == 3:
            normal = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
            outward = xyz[:, :3].mean(axis=1) - centres[owner]
            normal *= np.where((normal * outward).sum(axis=1)[:, None] < 0, -1.0, 1.0)
            # ONE force per face, broadcast to its three nodes. Without the
            # broadcast this is (nf, 1, 3) and `scatter_add` walks 3 nf node
            # indices against nf values -- numpy's "The weights and list don't
            # have the same length", out of a bincount, for every pressure
            # ever applied to a triangular exterior face: a tet4 model, or the
            # end of a wedge6 one. `surface_traction` broadcasts on the same
            # branch; the quad faces of the same meshes worked; no case in the
            # library puts a pressure on a triangle, so nothing ever saw it.
            local_force = np.broadcast_to(-pressure * normal[:, None, :] / 6.0,
                                          (len(xyz), 3, 3))
        else:
            local_force = _quad4_pressure(xyz, centres[owner], pressure)
        force += scatter_add(local_force, face, nodes.shape[0])
    if not np.any(force) and not allow_empty:
        raise ValueError("pressure region contains no complete exterior faces")
    return force


class FollowerPressure:
    """A pressure that follows the surface it acts on.

    `surface_pressure` integrates a pressure over the REFERENCE faces once,
    which fixes both the direction and the area of the load for the whole
    analysis. A pressure is normal to the surface it acts on, and on a body
    that rotates the two differ: the end pressures of Wu & Ting's floating
    beam turn with the beam, and the moment they exert about its centre stays
    the patch offset -- whereas a fixed-direction force on the same faces
    gains a lever arm of L/2 sin(theta) as the beam turns and spins it more
    than twice as fast. The exterior faces of the region are found once,
    here; `__call__` integrates the pressure over them at their current
    positions.
    """

    def __init__(self, nodes, elements, element, region, pressure, shared=None,
                 allow_empty: bool = False):
        self.pressure = float(pressure)
        if self.pressure < 0:
            raise ValueError("pressure is positive in compression; use traction for signed loading")
        self.faces = [(face, owner) for face, owner in
                      _region_exterior_faces(nodes, elements, element, region, shared)]
        if not self.faces and not allow_empty:
            raise ValueError("pressure region contains no complete exterior faces")
        self.elements = elements
        self.n_nodes = nodes.shape[0]

    def __call__(self, x):
        """The nodal force vector (n_nodes, 3) at the current positions x."""
        force = np.zeros((self.n_nodes, 3), dtype=float)
        centres = _interior_points(x, self.elements)
        for face, owner in self.faces:
            xyz = x[face]
            if face.shape[1] == 8:
                local_force = _quad8_pressure(xyz, centres[owner], self.pressure)
            elif face.shape[1] == 6:
                local_force = _tri6_pressure(xyz, centres[owner], self.pressure)
            elif face.shape[1] == 3:
                normal = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
                outward = xyz[:, :3].mean(axis=1) - centres[owner]
                normal *= np.where((normal * outward).sum(axis=1)[:, None] < 0, -1.0, 1.0)
                local_force = np.broadcast_to(
                    -self.pressure * normal[:, None, :] / 6.0, (len(xyz), 3, 3))
            else:
                local_force = _quad4_pressure(xyz, centres[owner], self.pressure)
            force += scatter_add(local_force, face, self.n_nodes)
        return force


class FaceMoment:
    """A moment on a face of a solid, as a couple about a fixed axis with a
    potential.

    A solid has no rotational dof to hang a moment on. What a moment on a
    face IS, on a solid, is a traction distribution with resultant zero and
    resultant couple M. Let R(x) be the face's best-fit rigid rotation (the
    area-weighted Kabsch/polar rotation of its nodes about their centroid),
    and theta(x) the angle by which R has turned a reference in-plane
    vector e0 ABOUT THE AXIS OF M -- the angle of the projection of R e0
    onto the plane normal to M. The load is the potential

        V = -|M| theta(x)

    whose gradient is the nodal force f_i = A_i R (g x rho_i) with rho_i the
    node's reference offset from the face centroid, A_i its tributary area,
    G = tr(S) I - S the "inertia" of the face's second-moment matrix
    S = sum A_i rho_i rho_i^T, and g = G^-1 R^T m with m = |M| q the couple
    conjugate to the face's spatial spin: q = M_hat - p (e'.M_hat)/|p|^2,
    e' = R e0, p its projection normal to M_hat. Whenever the face has not
    tilted out of its plane of rotation, q = M_hat and the couple is exactly
    M; in the reference configuration f_i = A_i (G^-1 M) x rho_i, which is
    the textbook linear bending traction M z / I for a bending component
    and the linear shear of Saint-Venant torsion for a component along the
    normal. Resultant zero and couple M through any rotation about M's own
    axis -- the cantilever of Wu & Ting's example 2 rolls into a full
    circle under a couple that stays a couple about its own section -- and
    no state: theta's unwrapping affects the potential, not its gradient.

    THE POINT IS THE POTENTIAL. The first version re-evaluated the textbook
    traction on the current face -- centroid, normal and S every step. That
    force is not the gradient of anything: its tangent is almost purely
    skew (|K - K^T| ~ 2|K|), and though a thousandth of the element
    stiffness it is a thousand times the softest mode of the rolled beam,
    where it turned the tangent circulatory. Kinetic damping, which is a
    descent method, then thrashed (resets every 150 steps from level 16 of
    40 on) and stalled at a residual of 1e-3 at m = 0.65, and without a
    symmetry plane an out-of-plane flutter grew from round-off at m = 0.4.
    Neither was the element: its tangent is symmetric to 1e-9 at 10% strain
    and its stiffest mode sat at half the stability limit.

    WHY THE ANGLE ABOUT A FIXED AXIS AND NOT THE ROTATION VECTOR. The
    second version was -M . phi with phi the face's rotation vector. Its
    gradient jumps at 180 degrees (the principal branch flips) unless the
    axis is exactly M's, which a continuous lift of phi cured; but the
    tangent map is singular at a full turn, where the transverse part of M
    is divided by |2 sin(theta/2)/theta| -- 33 at 350 degrees -- and that
    is a transverse stiffness the explicit step cannot carry: the rolled
    cantilever's tip tilt oscillated with alternating sign, growing 25% a
    step, from a 1e-4 wobble at 349 degrees. A couple of fixed direction is
    not conservative in three dimensions, and any potential pays for that
    somewhere; this one pays only where the face turns edge-on to M
    (|p| -> 0), which a bending roll-up never approaches.

    `reference_forces()` is the dead-load version: the reference
    distribution frozen, conservative too, but a couple that decays with
    the cosine of the face rotation. Quad4 faces only (hexa8/wedge6 quads).
    """

    def __init__(self, nodes, elements, element, region, moment):
        self.moment = np.asarray(moment, dtype=float).reshape(3)
        faces = [face for face, _ in _region_exterior_faces(nodes, elements, element, region)]
        if not faces:
            raise ValueError("moment region contains no complete exterior faces")
        if any(face.shape[1] != 4 for face in faces):
            raise ValueError("a face moment needs quadrilateral faces (hexa8 or wedge6 quads)")
        self.n_nodes = nodes.shape[0]
        fn = np.concatenate(faces)                                    # (nf, 4)
        # tributary areas and the mean outward normal from the 2 x 2 rule on
        # the reference faces
        g = 1.0 / np.sqrt(3.0); rs = [(-g, -g), (g, -g), (g, g), (-g, g)]
        N = np.array([[0.25 * (1 - r) * (1 - t), 0.25 * (1 + r) * (1 - t),
                       0.25 * (1 + r) * (1 + t), 0.25 * (1 - r) * (1 + t)] for r, t in rs])
        dr = np.array([[-0.25 * (1 - t), 0.25 * (1 - t), 0.25 * (1 + t), -0.25 * (1 + t)] for r, t in rs])
        ds = np.array([[-0.25 * (1 - r), -0.25 * (1 + r), 0.25 * (1 + r), 0.25 * (1 - r)] for r, t in rs])
        xyz = nodes[fn]
        normal = np.cross(np.einsum("gi,fij->fgj", dr, xyz), np.einsum("gi,fij->fgj", ds, xyz))
        w = np.linalg.norm(normal, axis=2)
        local = np.einsum("gi,fg->fi", N, w)                          # (nf, 4)
        self.idx = np.unique(fn)
        self.area = np.zeros(self.n_nodes); np.add.at(self.area, fn.ravel(), local.ravel())
        self.area = self.area[self.idx]
        X = nodes[self.idx]
        self.centroid = (self.area[:, None] * X).sum(axis=0) / self.area.sum()
        self.rho = X - self.centroid
        S = np.einsum("i,ij,ik->jk", self.area, self.rho, self.rho)
        if np.linalg.matrix_rank(S, tol=1e-12 * max(np.trace(S), 1e-300)) < 2:
            raise ValueError("the loaded face's nodes are collinear: no plane to apply a moment in")
        self.G0 = np.trace(S) * np.eye(3) - S
        self._wt = self.area / self.area.sum()
        self._arho = self.area[:, None] * self.rho
        # the axis of the couple and the reference vector whose turning about
        # it is the angle: the face normal, or for a torque any in-plane one
        mag = np.linalg.norm(self.moment)
        self.axis = self.moment / mag if mag > 0 else np.array([0.0, 0.0, 1.0])
        n0 = normal.sum(axis=(0, 1)); n0 /= np.linalg.norm(n0)
        e0 = n0 - (n0 @ self.axis) * self.axis
        if np.linalg.norm(e0) < 0.1:
            e0 = np.cross(self.axis, self.rho[np.argmax(np.linalg.norm(self.rho, axis=1))])
        self.e0 = e0 / np.linalg.norm(e0)

    def reference_forces(self):
        """The reference distribution as a dead load, (n_nodes, 3)."""
        g = np.linalg.solve(self.G0, self.moment)
        out = np.zeros((self.n_nodes, 3))
        out[self.idx] = self.area[:, None] * np.cross(g, self.rho)
        return out

    def rotation(self, x):
        """(R, S): the face's best-fit rotation at positions x and the
        symmetric factor of its cross-covariance H = R S."""
        xi = x[self.idx]
        c = (self._wt @ xi)
        H = (self.area[:, None] * (xi - c)).T @ self.rho
        V, _, Wt = np.linalg.svd(H)
        R = V @ Wt
        if np.linalg.det(R) < 0:
            V[:, -1] *= -1.0
            R = V @ Wt
        S = R.T @ H
        return R, 0.5 * (S + S.T)

    def angle(self, x):
        """The face's rotation angle about the couple's axis, in (-pi, pi]."""
        R, _ = self.rotation(x)
        e = R @ self.e0
        p = e - (e @ self.axis) * self.axis
        return math.atan2(np.cross(self.e0, p) @ self.axis, self.e0 @ p)

    def __call__(self, x):
        """The nodal force vector (n_nodes, 3) at positions x: minus the
        gradient of -|M| theta(x)."""
        R, S = self.rotation(x)
        G = np.trace(S) * np.eye(3) - S
        e = R @ self.e0
        c = e @ self.axis
        p = e - c * self.axis
        q = self.axis - p * (c / (p @ p))
        g = np.linalg.solve(G, R.T @ (np.linalg.norm(self.moment) * q))
        B = R @ np.array([[0.0, -g[2], g[1]], [g[2], 0.0, -g[0]], [-g[1], g[0], 0.0]])
        out = np.zeros((self.n_nodes, 3))
        out[self.idx] = self._arho @ B.T                              # A_i R (g x rho_i)
        return out


def volume_body_force(nodes: np.ndarray, elements: np.ndarray, region: np.ndarray,
                      shape_gp: np.ndarray, w_detJ: np.ndarray, force_density) -> np.ndarray:
    """Consistent nodal force for a constant body-force density (N/m³)."""
    member = np.zeros(nodes.shape[0], dtype=bool)
    member[np.asarray(region, dtype=np.int64)] = True
    active = member[elements].all(axis=1)
    local = np.einsum("eg,gn,c->enc", w_detJ * active[:, None], shape_gp,
                      np.asarray(force_density, dtype=float))
    return scatter_add(local, elements, nodes.shape[0])


def shared_face_keys(blocks) -> set:
    """Corner-node keys of faces that two cells share across BLOCKS.

    A face is exterior when one cell has it. Asked block by block, the
    interface between two sections (or two element types) is exterior to
    both, and a surface load looked for its faces in each block separately --
    a block with none of them raised, so a traction on a multi-section solid
    never ran. `blocks` is [(connectivity, element)]."""
    from collections import Counter
    counts: Counter = Counter()
    for conn, element in blocks:
        seen = set()
        for local in _FACETS[element]:
            f = np.asarray(conn)[:, local]
            nc = 3 if f.shape[1] in (3, 6) else 4
            for row in map(tuple, np.sort(f[:, :nc], axis=1)):
                if row not in seen:
                    seen.add(row)
                    counts[row] += 1
        # a face shared inside one block is already interior to it
    return {k for k, c in counts.items() if c > 1}


def _region_exterior_faces(nodes, elements, element, region, shared=None):
    member = np.zeros(nodes.shape[0], dtype=bool)
    member[np.asarray(region, dtype=np.int64)] = True
    faces = []
    # Faces of different shape (triangles/quads for wedges) cannot share a
    # single sortable key, so group by corner count before finding exterior.
    for local in _FACETS[element]:
        faces.append((elements[:, local], np.arange(len(elements), dtype=np.int64)))
    by_corners = {}
    for f, owner in faces:
        ncorner = 3 if f.shape[1] in (3, 6) else 4
        by_corners.setdefault(ncorner, []).append((f, owner))
    for ncorner, groups in by_corners.items():
        all_faces = np.vstack([g[0] for g in groups])
        all_owner = np.concatenate([g[1] for g in groups])
        key = np.sort(all_faces[:, :ncorner], axis=1)
        order = np.lexsort(key.T[::-1])
        sorted_key = key[order]
        change = np.r_[True, np.any(sorted_key[1:] != sorted_key[:-1], axis=1)]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(order)]
        exterior = order[starts[ends - starts == 1]]
        face = all_faces[exterior]
        keep = member[face].all(axis=1)
        if shared:
            keep &= np.array([tuple(r) not in shared
                              for r in np.sort(face[:, :ncorner], axis=1)], dtype=bool)
        if keep.any():
            yield face[keep], all_owner[exterior][keep]


#: The outward sense of a face, decided ONCE for the whole face.
#:
#: It used to be decided at every quadrature point, from the sign of that
#: point's normal against the vector from the element's interior point. On a
#: curved quadratic face that dot product goes through zero near a sliver
#: corner, so different points of the SAME face could be given opposite
#: senses: a uniform pressure over the closed boundary of a tet10 body kept a
#: net resultant of 0.7% once the mesh was distorted 2%, and its follower
#: form was fully circulatory -- tangent asymmetry 1.0 where the linear
#: elements scored 1e-9. A face has one orientation. This
#: takes it from the face's own area-weighted normal, which is the integral
#: of the pointwise ones and cannot be dominated by a single bad point.
def _face_sense(x: np.ndarray, centres: np.ndarray, normals) -> np.ndarray:
    total = sum(n for n in normals)
    out = x.mean(axis=1) - centres
    return np.where((total * out).sum(axis=1) < 0.0, -1.0, 1.0)[:, None]


def _tri6_traction(x: np.ndarray, traction) -> np.ndarray:
    # Dunavant degree-5 rule on the reference triangle.  Its weights below sum
    # to one; the reference triangle area is one half.
    q = [(1 / 3, 1 / 3, 0.225),
         (0.470142064105, 0.470142064105, 0.132394152789),
         (0.470142064105, 0.0597158717898, 0.132394152789),
         (0.0597158717898, 0.470142064105, 0.132394152789),
         (0.101286507323, 0.101286507323, 0.125939180544),
         (0.101286507323, 0.797426985353, 0.125939180544),
         (0.797426985353, 0.101286507323, 0.125939180544)]
    out = np.zeros((len(x), 6, 3), dtype=float)
    t = np.asarray(traction, dtype=float)
    for r, s, w in q:
        l1, l2, l3 = 1 - r - s, r, s
        N = np.array([l1 * (2*l1 - 1), l2 * (2*l2 - 1), l3 * (2*l3 - 1),
                      4*l1*l2, 4*l2*l3, 4*l3*l1])
        dr = np.array([1 - 4*l1, 4*l2 - 1, 0, 4*(l1-l2), 4*l3, -4*l3])
        ds = np.array([1 - 4*l1, 0, 4*l3 - 1, -4*l2, 4*l2, 4*(l1-l3)])
        jac = np.linalg.norm(np.cross(np.einsum("i,eij->ej", dr, x),
                                      np.einsum("i,eij->ej", ds, x)), axis=1)
        out += (0.5 * w * jac)[:, None, None] * N[None, :, None] * t
    return out


#: 3 x 3 Gauss on the reference square, for the eight-node faces of hexa20
#: and wedge15
_G3 = np.sqrt(0.6)
_QUAD8_Q = [(r, s, wr * ws)
            for r, wr in zip((-_G3, 0.0, _G3), (5 / 9, 8 / 9, 5 / 9))
            for s, ws in zip((-_G3, 0.0, _G3), (5 / 9, 8 / 9, 5 / 9))]


def _quad8_shape(r, s):
    """Serendipity shape functions and their derivatives at (r, s), in the
    node order corners then mid-sides (0,-1), (1,0), (0,1), (-1,0)."""
    xi = np.array([-1.0, 1.0, 1.0, -1.0])
    et = np.array([-1.0, -1.0, 1.0, 1.0])
    N = np.empty(8); dr = np.empty(8); ds = np.empty(8)
    N[:4] = 0.25 * (1 + xi * r) * (1 + et * s) * (xi * r + et * s - 1)
    dr[:4] = 0.25 * xi * (1 + et * s) * (2 * xi * r + et * s)
    ds[:4] = 0.25 * et * (1 + xi * r) * (xi * r + 2 * et * s)
    N[4] = 0.5 * (1 - r * r) * (1 - s); dr[4] = -r * (1 - s); ds[4] = -0.5 * (1 - r * r)
    N[5] = 0.5 * (1 + r) * (1 - s * s); dr[5] = 0.5 * (1 - s * s); ds[5] = -s * (1 + r)
    N[6] = 0.5 * (1 - r * r) * (1 + s); dr[6] = -r * (1 + s); ds[6] = 0.5 * (1 - r * r)
    N[7] = 0.5 * (1 - r) * (1 - s * s); dr[7] = -0.5 * (1 - s * s); ds[7] = -s * (1 - r)
    return N, dr, ds


def _quad8_normals(x: np.ndarray):
    """Per-quadrature-point (N, n dA) for a batch of eight-node faces."""
    for r, s, w in _QUAD8_Q:
        N, dr, ds = _quad8_shape(r, s)
        a = np.einsum("i,fij->fj", dr, x)
        b = np.einsum("i,fij->fj", ds, x)
        yield N, w * np.cross(a, b)


def _quad8_pressure(x: np.ndarray, centres: np.ndarray, pressure: float) -> np.ndarray:
    out = np.zeros((len(x), 8, 3))
    shape = list(_quad8_normals(x))
    sense = _face_sense(x, centres, [n for _, n in shape])
    for N, ndA in shape:
        out += -pressure * N[None, :, None] * (ndA * sense)[:, None, :]
    return out


def _quad8_traction(x: np.ndarray, traction) -> np.ndarray:
    t = np.asarray(traction, dtype=float)
    out = np.zeros((len(x), 8, 3))
    for N, ndA in _quad8_normals(x):
        area = np.linalg.norm(ndA, axis=1)
        out += area[:, None, None] * N[None, :, None] * t[None, None, :]
    return out


def _tri6_pressure(x: np.ndarray, centres: np.ndarray, pressure: float) -> np.ndarray:
    q = [(1 / 3, 1 / 3, 0.225), (0.470142064105, 0.470142064105, 0.132394152789),
         (0.470142064105, 0.0597158717898, 0.132394152789),
         (0.0597158717898, 0.470142064105, 0.132394152789),
         (0.101286507323, 0.101286507323, 0.125939180544),
         (0.101286507323, 0.797426985353, 0.125939180544),
         (0.797426985353, 0.101286507323, 0.125939180544)]
    out = np.zeros((len(x), 6, 3), dtype=float)
    shape = []
    for r, s, w in q:
        l1, l2, l3 = 1-r-s, r, s
        N = np.array([l1*(2*l1-1), l2*(2*l2-1), l3*(2*l3-1), 4*l1*l2, 4*l2*l3, 4*l3*l1])
        dr = np.array([1-4*l1, 4*l2-1, 0, 4*(l1-l2), 4*l3, -4*l3])
        ds = np.array([1-4*l1, 0, 4*l3-1, -4*l2, 4*l2, 4*(l1-l3)])
        normal = np.cross(np.einsum("i,eij->ej", dr, x), np.einsum("i,eij->ej", ds, x))
        shape.append((w, N, normal))
    sense = _face_sense(x, centres, [w * n for w, _, n in shape])
    for w, N, normal in shape:
        out -= 0.5*w*pressure*(normal*sense)[:, None, :]*N[None, :, None]
    return out


def _quad4_traction(x: np.ndarray, traction) -> np.ndarray:
    out = np.zeros((len(x), 4, 3), dtype=float)
    t = np.asarray(traction, dtype=float)
    g = 1.0 / np.sqrt(3.0)
    for r in (-g, g):
        for s in (-g, g):
            N = 0.25 * np.array([(1-r)*(1-s), (1+r)*(1-s),
                                 (1+r)*(1+s), (1-r)*(1+s)])
            dr = 0.25 * np.array([-(1-s), 1-s, 1+s, -(1+s)])
            ds = 0.25 * np.array([-(1-r), -(1+r), 1+r, 1-r])
            jac = np.linalg.norm(np.cross(np.einsum("i,eij->ej", dr, x),
                                          np.einsum("i,eij->ej", ds, x)), axis=1)
            out += jac[:, None, None] * N[None, :, None] * t
    return out


def _quad4_pressure(x: np.ndarray, centres: np.ndarray, pressure: float) -> np.ndarray:
    out = np.zeros((len(x), 4, 3), dtype=float)
    g = 1.0 / np.sqrt(3.0)
    shape = []
    for r in (-g, g):
        for s in (-g, g):
            N = 0.25*np.array([(1-r)*(1-s), (1+r)*(1-s), (1+r)*(1+s), (1-r)*(1+s)])
            dr = 0.25*np.array([-(1-s), 1-s, 1+s, -(1+s)])
            ds = 0.25*np.array([-(1-r), -(1+r), 1+r, 1-r])
            normal = np.cross(np.einsum("i,eij->ej", dr, x), np.einsum("i,eij->ej", ds, x))
            shape.append((N, normal))
    sense = _face_sense(x, centres, [n for _, n in shape])
    for N, normal in shape:
        out -= pressure*(normal*sense)[:, None, :]*N[None, :, None]
    return out
