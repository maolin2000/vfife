"""Element definitions: shape derivatives, quadrature, topology.

Every isoparametric continuum element in this solver goes through the same
machinery -- co-rotational frame, strain from the displacement gradient, the
material call, f = dN/dx^T . T -- and differs only in three things: how many
nodes it has, what dN/dxi looks like, and where the integration points are.

So they are data, not code.  Adding an element is adding an entry to
`ELEMENTS` below; the precompute, the three backends and the whole test
apparatus pick it up without changes.  That is the difference between a solver
with a hexahedron in it and a solver with an element library.

NODE ORDERING follows VTK / Abaqus, so meshes from Gmsh and meshio import
without renumbering:

    hexa8   0-3 bottom face counter-clockwise, 4-7 the same on top
    tet4    0-3, node 3 above the 0-1-2 face
    tet10   0-3 corners, then mid-edge 0-1, 1-2, 2-0, 0-3, 1-3, 2-3
    wedge6  0-2 bottom triangle, 3-5 the same on top

WHICH FORMULATIONS EACH ELEMENT SUPPORTS is a property of the element, not a
free choice, and asking for an unsupported one is an error rather than a silent
fallback:

  * `reduced` + hourglass control is hexa8 only.  The Flanagan-Belytschko modes
    below are the four hourglass modes of a trilinear hexahedron; a wedge or a
    tetrahedron has its own (or, for tet4, none at all) and guessing would
    produce forces that quietly pollute the answer.
  * `b-bar` needs more than one integration point, so it is available on
    everything except tet4 -- whose strain is constant over the element already,
    which is also exactly why tet4 locks (see `ELEMENTS["tet4"].caveat`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ElementCapability:
    """Machine-readable contract shared by CLI, API and a future GUI."""

    name: str
    family: str
    topology: tuple[str, ...]
    dofs: tuple[str, ...]
    formulations: tuple[str, ...]
    material_kinematics: tuple[str, ...]
    compatible_materials: tuple[str, ...]
    backends: tuple[str, ...]
    analysis_modes: tuple[str, ...]
    loads: tuple[str, ...]
    output_fields: tuple[str, ...]
    geometric_nonlinearity: str
    verification: str
    caveat: str = ""
    #: XDMF (TopologyType, nodes per element, extra topology attributes).
    #: None means this element is never written to an XDMF grid.
    xdmf: tuple[str, int, str] | None = None
    #: Gmsh MSH element type id, or None if it has no mesh-file form
    gmsh_type: int | None = None
    #: Gmsh -> VTK node permutation, `ours[i] = gmsh[order[i]]`
    gmsh_order: tuple[int, ...] | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ElementDef:
    name: str
    n_nodes: int
    #: (npt, 3) natural coordinates -> (npt, 3, n_nodes) shape derivatives
    dN_dxi: Callable[[np.ndarray], np.ndarray]
    #: scheme name -> (points (npt,3), weights (npt,))
    quadrature: Dict[str, Tuple[np.ndarray, np.ndarray]]
    #: natural coordinates of the element centre, for b-bar and the FB projection
    centroid: np.ndarray
    #: node indices of each face, for the characteristic length V / A_max
    faces: Sequence[Sequence[int]]
    #: which scheme integrates the volume accurately
    volume_scheme: str
    formulations: Tuple[str, ...]
    #: (4, n_nodes) hourglass base vectors, or None if reduced is unsupported
    hourglass: np.ndarray | None = None
    caveat: str = ""
    #: How this element is spelled outside the solver.  Declared here, beside
    #: the shape functions, because every other spelling of it -- the case
    #: schema, the XDMF writer, the Gmsh reader, the solver chooser -- used to
    #: keep its own hand-written copy, and a name typed into four tables is a
    #: name that can be typed wrong in one.  It was: wedge15 shipped as XDMF
    #: "Wedge", which is the 6-node topology, so ParaView misread every
    #: 15-node prism result it was given.
    xdmf: tuple[str, int, str] | None = None
    gmsh_type: int | None = None
    gmsh_order: tuple[int, ...] | None = None

    def gauss(self, formulation: str):
        if formulation not in self.formulations:
            raise ValueError(
                f"{self.name} does not support formulation {formulation!r}; "
                f"it supports {self.formulations}. "
                + (f"({self.caveat})" if self.caveat else "")
            )
        scheme = "reduced" if formulation == "reduced" else "full"
        return self.quadrature[scheme]


# --------------------------------------------------------------------------- #
#  hexa8 -- trilinear hexahedron
# --------------------------------------------------------------------------- #

_HX = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
_HY = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
_HZ = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])


def _dN_hexa8(pts: np.ndarray) -> np.ndarray:
    xi, eta, zeta = pts[:, 0:1], pts[:, 1:2], pts[:, 2:3]
    d = np.empty((pts.shape[0], 3, 8))
    d[:, 0] = 0.125 * _HX * (1 + eta * _HY) * (1 + zeta * _HZ)
    d[:, 1] = 0.125 * _HY * (1 + xi * _HX) * (1 + zeta * _HZ)
    d[:, 2] = 0.125 * _HZ * (1 + xi * _HX) * (1 + eta * _HY)
    return d


def _N_hexa8(pts: np.ndarray) -> np.ndarray:
    xi, eta, zeta = pts[:, 0:1], pts[:, 1:2], pts[:, 2:3]
    return 0.125 * (1 + xi * _HX) * (1 + eta * _HY) * (1 + zeta * _HZ)


_G = 1.0 / np.sqrt(3.0)
_HEX_FULL = np.stack(
    np.meshgrid([-_G, _G], [-_G, _G], [-_G, _G], indexing="ij"), axis=-1
).reshape(-1, 3)

# the four true hourglass modes of a trilinear hexahedron
_HEX_HG = np.stack([_HX * _HY, _HY * _HZ, _HX * _HZ, _HX * _HY * _HZ])


# --------------------------------------------------------------------------- #
#  tet4 -- constant-strain tetrahedron
# --------------------------------------------------------------------------- #


def _dN_tet4(pts: np.ndarray) -> np.ndarray:
    d = np.zeros((pts.shape[0], 3, 4))
    d[:, :, 0] = -1.0
    d[:, 0, 1] = 1.0
    d[:, 1, 2] = 1.0
    d[:, 2, 3] = 1.0
    return d


def _N_tet4(pts: np.ndarray) -> np.ndarray:
    return np.column_stack([1.0 - pts.sum(axis=1), pts])


# --------------------------------------------------------------------------- #
#  tet10 -- quadratic tetrahedron
# --------------------------------------------------------------------------- #


def _dN_tet10(pts: np.ndarray) -> np.ndarray:
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    L0 = 1.0 - x - y - z
    n = pts.shape[0]
    d = np.empty((n, 3, 10))
    # corners: dN_i/dL_i = 4 L_i - 1, and dL0/dxi = -1
    for a, (Lv, sgn) in enumerate(((L0, -1.0), (x, 1.0), (y, 1.0), (z, 1.0))):
        g = 4.0 * Lv - 1.0
        if a == 0:
            d[:, 0, 0] = d[:, 1, 0] = d[:, 2, 0] = -g
        else:
            d[:, :, a] = 0.0
            d[:, a - 1, a] = g
    # mid-edge nodes: 4=(0,1) 5=(1,2) 6=(2,0) 7=(0,3) 8=(1,3) 9=(2,3)
    # dL0 = (-1,-1,-1), dL1 = (1,0,0), dL2 = (0,1,0), dL3 = (0,0,1)
    dL = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    L = np.stack([L0, x, y, z], axis=1)
    for node, (i, j) in enumerate(((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)),
                                  start=4):
        d[:, :, node] = 4.0 * (L[:, i, None] * dL[j] + L[:, j, None] * dL[i])
    return d


def _N_tet10(pts: np.ndarray) -> np.ndarray:
    L0 = 1.0 - pts.sum(axis=1)
    L = np.column_stack([L0, pts])
    out = np.empty((len(pts), 10))
    out[:, :4] = L * (2.0 * L - 1.0)
    for node, (i, j) in enumerate(((0, 1), (1, 2), (2, 0),
                                   (0, 3), (1, 3), (2, 3)), start=4):
        out[:, node] = 4.0 * L[:, i] * L[:, j]
    return out


_A = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
_B = (5.0 - np.sqrt(5.0)) / 20.0
_TET4PT = np.array([[_B, _B, _B], [_A, _B, _B], [_B, _A, _B], [_B, _B, _A]])
_TET4W = np.full(4, 1.0 / 24.0)


# --------------------------------------------------------------------------- #
#  wedge6 -- linear triangular prism
# --------------------------------------------------------------------------- #


def _dN_wedge6(pts: np.ndarray) -> np.ndarray:
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    L = np.stack([1.0 - x - y, x, y], axis=1)          # (n, 3)
    dL = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])   # d/dxi, d/deta
    n = pts.shape[0]
    d = np.empty((n, 3, 6))
    for a in range(3):
        lo = 0.5 * (1.0 - z)
        hi = 0.5 * (1.0 + z)
        d[:, 0, a] = dL[a, 0] * lo
        d[:, 1, a] = dL[a, 1] * lo
        d[:, 2, a] = -0.5 * L[:, a]
        d[:, 0, a + 3] = dL[a, 0] * hi
        d[:, 1, a + 3] = dL[a, 1] * hi
        d[:, 2, a + 3] = 0.5 * L[:, a]
    return d


def _N_wedge6(pts: np.ndarray) -> np.ndarray:
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    L = np.column_stack([1.0 - x - y, x, y])
    return np.hstack([0.5 * L * (1.0 - z[:, None]),
                      0.5 * L * (1.0 + z[:, None])])


_TRI3 = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
_WEDGE_PTS = np.array([[t[0], t[1], s] for s in (-_G, _G) for t in _TRI3])
_WEDGE_W = np.full(6, 1.0 / 6.0)

# --------------------------------------------------------------------------- #
#  pyramid5 -- linear pyramid (quad base + apex)
#  Reference: base at z=0 (xi,eta in [-1,1]), apex at (0,0,1).
#  Rational shape functions N_i = 0.25*(a+xi_i*xi)*(a+eta_i*eta)/a , a=1-zeta, N_apex=zeta
#  Quadrature: 2x2 Gauss in scaled (xi/a, eta/a) x 2 Gauss in zeta, weights scaled by a^2.
# --------------------------------------------------------------------------- #

_PYR_XI = np.array([-1.0, 1.0, 1.0, -1.0])
_PYR_ETA = np.array([-1.0, -1.0, 1.0, 1.0])

def _dN_pyramid5(pts: np.ndarray) -> np.ndarray:
    xi, eta, zeta = pts[:, 0], pts[:, 1], pts[:, 2]
    a = 1.0 - zeta
    # avoid singularity at apex for quadrature (a>0); clamp for safety
    a_safe = np.where(np.abs(a) < 1e-12, 1e-12, a)
    n = pts.shape[0]
    d = np.empty((n, 3, 5))
    for i in range(4):
        A = a + _PYR_XI[i] * xi
        B = a + _PYR_ETA[i] * eta
        d[:, 0, i] = 0.25 * _PYR_XI[i] * B / a_safe
        d[:, 1, i] = 0.25 * _PYR_ETA[i] * A / a_safe
        dNda = 0.25 * ((B + A) * a_safe - A * B) / (a_safe * a_safe)
        d[:, 2, i] = -dNda
    d[:, 0, 4] = 0.0
    d[:, 1, 4] = 0.0
    d[:, 2, 4] = 1.0
    # apex limit: base derivatives -> -0.25 when evaluated exactly at apex
    apex_mask = (np.abs(a) < 1e-12) & (np.abs(xi) < 1e-12) & (np.abs(eta) < 1e-12)
    if np.any(apex_mask):
        for i in range(4):
            d[apex_mask, 0, i] = 0.0
            d[apex_mask, 1, i] = 0.0
            d[apex_mask, 2, i] = -0.25
    return d


def _N_pyramid5(pts: np.ndarray) -> np.ndarray:
    xi, eta, zeta = pts[:, 0], pts[:, 1], pts[:, 2]
    a = 1.0 - zeta
    a_safe = np.where(np.abs(a) < 1e-12, 1e-12, a)
    n = pts.shape[0]
    out = np.empty((n, 5))
    for i in range(4):
        A = a + _PYR_XI[i] * xi
        B = a + _PYR_ETA[i] * eta
        out[:, i] = 0.25 * A * B / a_safe
    out[:, 4] = zeta
    apex_mask = (np.abs(a) < 1e-12) & (np.abs(xi) < 1e-12) & (np.abs(eta) < 1e-12)
    if np.any(apex_mask):
        out[apex_mask, 0:4] = 0.0
        out[apex_mask, 4] = 1.0
    return out


# 8-point full integration for pyramid5 (see derivation above)
_pyr_zeta = np.array([(1 - _G) / 2.0, (1 + _G) / 2.0])
_pyr_wz = np.array([0.5, 0.5])
_PYRAMID_PTS = []
_PYRAMID_W = []
for _zeta_g in _pyr_zeta:
    _a = 1.0 - _zeta_g
    _wz = 0.5 if _zeta_g == _pyr_zeta[0] else 0.5  # both 0.5
    for _xr in (-_G, _G):
        for _yr in (-_G, _G):
            _xi = _a * _xr
            _eta = _a * _yr
            _PYRAMID_PTS.append([_xi, _eta, _zeta_g])
            _PYRAMID_W.append(_wz * _a * _a)
_PYRAMID_PTS = np.array(_PYRAMID_PTS)
_PYRAMID_W = np.array(_PYRAMID_W)

# --------------------------------------------------------------------------- #
#  hexa20 -- 20-node serendipity hexahedron (8 corners + 12 edge mids)
#  Reference cube [-1,1]^3.  Full integration 27 points (3x3x3 Gauss).
# --------------------------------------------------------------------------- #

_H20_XI = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 0, 1, 0, -1, 0, 1, 0, -1, -1, 1, 1, -1], dtype=float)
_H20_ETA = np.array([-1, -1, 1, 1, -1, -1, 1, 1, -1, 0, 1, 0, -1, 0, 1, 0, -1, -1, 1, 1], dtype=float)
_H20_ZETA = np.array([-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=float)

# corner sign arrays reuse _HX/_HY/_HZ for 0-7
_H20_G = np.sqrt(3.0 / 5.0)
_H20_PT1D = np.array([-_H20_G, 0.0, _H20_G])
_H20_W1D = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
_HEXA20_PTS = np.array([[x,y,z] for x in _H20_PT1D for y in _H20_PT1D for z in _H20_PT1D])
_HEXA20_W = np.array([wx*wy*wz for wx in _H20_W1D for wy in _H20_W1D for wz in _H20_W1D])
# ensure volume 8
_HEXA20_W = _HEXA20_W * 1.0  # already sums to 8

def _N_hexa20(pts: np.ndarray) -> np.ndarray:
    xi, eta, zeta = pts[:,0], pts[:,1], pts[:,2]
    n = pts.shape[0]
    out = np.empty((n, 20))
    # corners 0-7
    for i in range(8):
        xi_i, eta_i, zeta_i = _HX[i], _HY[i], _HZ[i]
        A = 1 + xi_i*xi
        B = 1 + eta_i*eta
        C = 1 + zeta_i*zeta
        S = xi_i*xi + eta_i*eta + zeta_i*zeta - 2
        out[:, i] = 0.125 * A * B * C * S
    # edges along xi: 8,10,12,14 (eta,zeta = +/-1)
    for i, (eta_i, zeta_i) in [ (8, (-1,-1)), (10, (1,-1)), (12, (-1,1)), (14, (1,1))]:
        out[:, i] = 0.25 * (1 - xi*xi) * (1 + eta_i*eta) * (1 + zeta_i*zeta)
    # edges along eta: 9,11,13,15
    for i, (xi_i, zeta_i) in [ (9, (1,-1)), (11, (-1,-1)), (13, (1,1)), (15, (-1,1))]:
        out[:, i] = 0.25 * (1 - eta*eta) * (1 + xi_i*xi) * (1 + zeta_i*zeta)
    # edges along zeta: 16,17,18,19
    for i, (xi_i, eta_i) in [ (16, (-1,-1)), (17, (1,-1)), (18, (1,1)), (19, (-1,1))]:
        out[:, i] = 0.25 * (1 - zeta*zeta) * (1 + xi_i*xi) * (1 + eta_i*eta)
    return out

def _dN_hexa20(pts: np.ndarray) -> np.ndarray:
    xi, eta, zeta = pts[:,0], pts[:,1], pts[:,2]
    n = pts.shape[0]
    d = np.empty((n, 3, 20))
    # corners
    for i in range(8):
        xi_i, eta_i, zeta_i = _HX[i], _HY[i], _HZ[i]
        A = 1 + xi_i*xi
        B = 1 + eta_i*eta
        C = 1 + zeta_i*zeta
        S = xi_i*xi + eta_i*eta + zeta_i*zeta - 2
        # dN/dxi =0.125*xi_i*B*C*(S+A)
        d[:,0,i] = 0.125 * xi_i * B * C * (S + A)
        d[:,1,i] = 0.125 * eta_i * A * C * (S + B)
        d[:,2,i] = 0.125 * zeta_i * A * B * (S + C)
    # xi edges
    for i, (eta_i, zeta_i) in [ (8, (-1,-1)), (10, (1,-1)), (12, (-1,1)), (14, (1,1))]:
        B = 1 + eta_i*eta
        C = 1 + zeta_i*zeta
        d[:,0,i] = -0.5 * xi * B * C
        d[:,1,i] = 0.25 * (1 - xi*xi) * eta_i * C
        d[:,2,i] = 0.25 * (1 - xi*xi) * B * zeta_i
    # eta edges
    for i, (xi_i, zeta_i) in [ (9, (1,-1)), (11, (-1,-1)), (13, (1,1)), (15, (-1,1))]:
        A = 1 + xi_i*xi
        C = 1 + zeta_i*zeta
        d[:,0,i] = 0.25 * (1 - eta*eta) * xi_i * C
        d[:,1,i] = -0.5 * eta * A * C
        d[:,2,i] = 0.25 * (1 - eta*eta) * A * zeta_i
    # zeta edges
    for i, (xi_i, eta_i) in [ (16, (-1,-1)), (17, (1,-1)), (18, (1,1)), (19, (-1,1))]:
        A = 1 + xi_i*xi
        B = 1 + eta_i*eta
        d[:,0,i] = 0.25 * (1 - zeta*zeta) * xi_i * B
        d[:,1,i] = 0.25 * (1 - zeta*zeta) * A * eta_i
        d[:,2,i] = -0.5 * zeta * A * B
    return d


# --------------------------------------------------------------------------- #
#  wedge15 -- 15-node quadratic prism (6 corners + 9 mids, 9-pt Gauss)
# --------------------------------------------------------------------------- #

_W15_NODES = np.array([[0,0,-1],[1,0,-1],[0,1,-1],[0,0,1],[1,0,1],[0,1,1],[0.5,0,-1],[0.5,0.5,-1],[0,0.5,-1],[0.5,0,1],[0.5,0.5,1],[0,0.5,1],[0,0,0],[1,0,0],[0,1,0]], dtype=float)

def _wedge15_basis(pts: np.ndarray) -> np.ndarray:
    xi, eta, zeta = pts[:,0], pts[:,1], pts[:,2]
    return np.column_stack([np.ones_like(xi), xi, eta, zeta, xi*xi, eta*eta, xi*eta, xi*zeta, eta*zeta, zeta*zeta, xi*xi*zeta, eta*eta*zeta, xi*eta*zeta, xi*zeta*zeta, eta*zeta*zeta])

def _wedge15_dbasis(pts: np.ndarray) -> np.ndarray:
    xi, eta, zeta = pts[:,0], pts[:,1], pts[:,2]
    n = pts.shape[0]
    d = np.empty((n,3,15))
    d[:,0,0]=0; d[:,0,1]=1; d[:,0,2]=0; d[:,0,3]=0; d[:,0,4]=2*xi; d[:,0,5]=0; d[:,0,6]=eta; d[:,0,7]=zeta; d[:,0,8]=0; d[:,0,9]=0; d[:,0,10]=2*xi*zeta; d[:,0,11]=0; d[:,0,12]=eta*zeta; d[:,0,13]=zeta*zeta; d[:,0,14]=0
    d[:,1,0]=0; d[:,1,1]=0; d[:,1,2]=1; d[:,1,3]=0; d[:,1,4]=0; d[:,1,5]=2*eta; d[:,1,6]=xi; d[:,1,7]=0; d[:,1,8]=zeta; d[:,1,9]=0; d[:,1,10]=0; d[:,1,11]=2*eta*zeta; d[:,1,12]=xi*zeta; d[:,1,13]=0; d[:,1,14]=zeta*zeta
    d[:,2,0]=0; d[:,2,1]=0; d[:,2,2]=0; d[:,2,3]=1; d[:,2,4]=0; d[:,2,5]=0; d[:,2,6]=0; d[:,2,7]=xi; d[:,2,8]=eta; d[:,2,9]=2*zeta; d[:,2,10]=xi*xi; d[:,2,11]=eta*eta; d[:,2,12]=xi*eta; d[:,2,13]=2*xi*zeta; d[:,2,14]=2*eta*zeta
    return d

_W15_B = _wedge15_basis(_W15_NODES)
_W15_C = np.linalg.inv(_W15_B)

def _N_wedge15(pts: np.ndarray) -> np.ndarray:
    return _wedge15_basis(pts) @ _W15_C

def _dN_wedge15(pts: np.ndarray) -> np.ndarray:
    return np.einsum('nki,ij->nkj', _wedge15_dbasis(pts), _W15_C)

_W15_G = np.sqrt(3.0/5.0)
_W15_ZPTS = np.array([-_W15_G, 0.0, _W15_G])
_W15_ZW = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
_WEDGE15_PTS = np.array([[t[0], t[1], z] for z in _W15_ZPTS for t in _TRI3])
_WEDGE15_W = np.array([1.0/6.0 * w for w in _W15_ZW for _ in _TRI3])


# --------------------------------------------------------------------------- #

ELEMENTS: Dict[str, ElementDef] = {
    "hexa8": ElementDef(
        name="hexa8", n_nodes=8, dN_dxi=_dN_hexa8,
        quadrature={"full": (_HEX_FULL, np.ones(8)),
                    "reduced": (np.zeros((1, 3)), np.array([8.0]))},
        centroid=np.zeros(3),
        faces=[(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
               (3, 2, 6, 7), (0, 3, 7, 4), (1, 2, 6, 5)],
        volume_scheme="full",
        formulations=("full", "reduced", "b-bar"),
        hourglass=_HEX_HG,
        xdmf=("Hexahedron", 8, ""), gmsh_type=5,
        gmsh_order=tuple(range(8)),
    ),
    "tet4": ElementDef(
        name="tet4", n_nodes=4, dN_dxi=_dN_tet4,
        quadrature={"full": (np.full((1, 3), 0.25), np.array([1.0 / 6.0]))},
        centroid=np.full(3, 0.25),
        faces=[(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)],
        volume_scheme="full",
        formulations=("full",),
        caveat="its strain is constant over the element, so there is no "
               "volumetric part to average and no hourglass mode to control; "
               "that same constancy is why it locks -- use tet10 where "
               "plasticity or near-incompressibility matters",
        xdmf=("Tetrahedron", 4, ""), gmsh_type=4,
        gmsh_order=tuple(range(4)),
    ),
    "tet10": ElementDef(
        name="tet10", n_nodes=10, dN_dxi=_dN_tet10,
        quadrature={"full": (_TET4PT, _TET4W)},
        centroid=np.full(3, 0.25),
        # Six-node faces, like hexa20 and wedge15 -- this used to be a copy of
        # tet4's four CORNER triangles, so the `surface` region on a tet10 model
        # returned corners only and silently left every mid-side node off the
        # outside of the mesh: 72% of the boundary nodes of a unit cube.  A
        # constraint or a pressure on `surface` then applied to a quarter of
        # the face it named.  Corners first, then the mid-edge node of each
        # corner pair in the same order, which is the tri6 convention.
        faces=[(0, 2, 1, 6, 5, 4), (0, 1, 3, 4, 8, 7),
               (1, 2, 3, 5, 9, 8), (0, 3, 2, 7, 9, 6)],
        volume_scheme="full",
        formulations=("full", "b-bar"),
        xdmf=("Tetrahedron_10", 10, ""), gmsh_type=11,
        gmsh_order=(0, 1, 2, 3, 4, 5, 6, 7, 9, 8),
    ),
    "wedge6": ElementDef(
        name="wedge6", n_nodes=6, dN_dxi=_dN_wedge6,
        quadrature={"full": (_WEDGE_PTS, _WEDGE_W)},
        centroid=np.array([1 / 3, 1 / 3, 0.0]),
        faces=[(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
        volume_scheme="full",
        formulations=("full", "b-bar"),
        caveat="hourglass control for a prism needs its own mode vectors, "
               "which are not the hexahedron's; use full or b-bar",
        xdmf=("Wedge", 6, ""), gmsh_type=6,
        gmsh_order=tuple(range(6)),
    ),
    "pyramid5": ElementDef(
        name="pyramid5", n_nodes=5, dN_dxi=_dN_pyramid5,
        quadrature={"full": (_PYRAMID_PTS, _PYRAMID_W)},
        centroid=np.array([0.0, 0.0, 0.25]),
        faces=[(0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
        volume_scheme="full",
        formulations=("full",),
        caveat="5-node linear pyramid for hex-tet transition; "
               "reference base at z=0 apex at (0,0,1), rational shape functions",
        xdmf=("Pyramid", 5, ""), gmsh_type=7,
        gmsh_order=tuple(range(5)),
    ),
    "hexa20": ElementDef(
        name="hexa20", n_nodes=20, dN_dxi=_dN_hexa20,
        quadrature={"full": (_HEXA20_PTS, _HEXA20_W)},
        centroid=np.zeros(3),
        faces=[(0, 1, 2, 3, 8, 9, 10, 11), (4, 5, 6, 7, 12, 13, 14, 15),
               (0, 1, 5, 4, 8, 17, 12, 16), (3, 2, 6, 7, 10, 18, 14, 19),
               (0, 3, 7, 4, 11, 19, 15, 16), (1, 2, 6, 5, 9, 18, 13, 17)],
        volume_scheme="full",
        formulations=("full",),
        caveat="20-node serendipity hexahedron; full 27-point Gauss, quadratic edges",
        xdmf=("Hexahedron_20", 20, ""), gmsh_type=17,
        # Gmsh numbers the 12 edges in its own order, not VTK's, so the
        # identity that used to sit here read every 20-node hexahedron with its
        # mid-side nodes on the wrong edges: `128 integration points of hexa20
        # have non-positive Jacobian` on the very first file, for every mesh.
        # VTK edge k -> the Gmsh slot holding that same edge.
        gmsh_order=(0, 1, 2, 3, 4, 5, 6, 7,
                    8, 11, 13, 9,          # bottom  0-1 1-2 2-3 3-0
                    16, 18, 19, 17,        # top     4-5 5-6 6-7 7-4
                    10, 12, 14, 15),       # verticals 0-4 1-5 2-6 3-7
    ),
    "wedge15": ElementDef(
        name="wedge15", n_nodes=15, dN_dxi=_dN_wedge15,
        quadrature={"full": (_WEDGE15_PTS, _WEDGE15_W)},
        centroid=np.array([1/3, 1/3, 0.0]),
        faces=[(0, 1, 2, 6, 7, 8), (3, 4, 5, 9, 10, 11),
               (0, 1, 4, 3, 6, 13, 9, 12), (1, 2, 5, 4, 7, 14, 10, 13), (2, 0, 3, 5, 8, 12, 11, 14)],
        volume_scheme="full",
        formulations=("full",),
        caveat="15-node quadratic prism; 9-point Gauss, quadratic triangle + vertical",
        xdmf=("Wedge_15", 15, ""), gmsh_type=18,
        # Same as hexa20: Gmsh's edge order is not VTK's, and the identity made
        # every 15-node prism inside out.
        gmsh_order=(0, 1, 2, 3, 4, 5,
                    6, 9, 7,               # bottom triangle 0-1 1-2 2-0
                    12, 14, 13,            # top triangle    3-4 4-5 5-3
                    8, 10, 11),            # verticals       0-3 1-4 2-5
    ),
}


def get(name: str) -> ElementDef:
    if name not in ELEMENTS:
        raise KeyError(
            f"unknown element {name!r}; available: {sorted(ELEMENTS)}"
        )
    return ELEMENTS[name]


def shape_functions(name: str, points: np.ndarray) -> np.ndarray:
    """Interpolation functions at natural coordinates, ``(n_point,n_node)``."""
    return {"hexa8": _N_hexa8, "tet4": _N_tet4,
            "tet10": _N_tet10, "wedge6": _N_wedge6, "pyramid5": _N_pyramid5,
            "hexa20": _N_hexa20, "wedge15": _N_wedge15}[name](points)


def available() -> list[str]:
    return sorted(element_definitions())


_COMMON_CONTINUUM_OUTPUTS = (
    "displacement", "stress", "strain",
    "von_mises", "stress_nodal", "reaction_force",
)

#: Output names that stand for several written fields. They were listed as
#: fields here while the writer matched names one by one, so asking for
#: `end_actions` -- as thirteen shipped cases did -- wrote no member force
#: at all, and nothing said so (found in review). A group
#: name in `output.fields` now writes every member of the group.
_ACTIONS = ("axial_force", "shear_y", "shear_z", "torsion", "bending_y", "bending_z")
FIELD_GROUPS: Dict[str, tuple] = {
    "end_actions": tuple(f"{n}_{e}" for n in _ACTIONS for e in ("i", "j")),
    "connector_force": ("connector_force_x", "connector_force_y", "connector_force_z"),
    "connector_moment": ("connector_moment_x", "connector_moment_y", "connector_moment_z"),
    "connector_deformation": tuple(f"connector_deformation_{c}" for c in ("ux", "uy", "uz", "rx", "ry", "rz")),
    # the connector laws' slipped deformation (v1.0: the manual named
    # `connector_slip_*`, and the group name was refused)
    "connector_slip": tuple(f"connector_slip_{c}" for c in ("ux", "uy", "uz", "rx", "ry", "rz")),
    "membrane_resultants": ("membrane_x", "membrane_y", "membrane_xy"),
    "bending_resultants": ("moment_x", "moment_y", "moment_xy"),
    "shear_resultants": ("shear_x", "shear_y"),
}


#: the member force fields a frame, truss or fibre model writes beside its end
#: actions: signed and the unsigned envelope `*_max_abs`,
#: with the failure fields -- listed so that `describe` names every field a
#: run can write (it named `end_actions` alone)
_MEMBER_FIELDS = _ACTIONS + tuple(f"{n}_max_abs" for n in _ACTIONS) + ("capacity_ratio", "member_failed",
                                                                          "member_built")


def expand_fields(names):
    """`names` with every group name replaced by (and kept beside) its members."""
    out = set()
    for n in names or ():
        out.add(n)
        out.update(FIELD_GROUPS.get(n, ()))
    return out


LINE_ELEMENTS: Dict[str, ElementCapability] = {
    "truss2": ElementCapability(
        "truss2", "line", ("line2",), ("ux", "uy", "uz"), ("axial",),
        # every UNIAXIAL law, because that is what TrussSolver accepts: it
        # asks the material for `uniaxial`, not for its name. The list used
        # to stop at three while the solver took seven, so preflight refused
        # a steel02 bar the solver would have run.
        ("solid_3d", "uniaxial"),
        ("linear_elastic", "rebar", "piecewise_uniaxial", "steel02", "steel_cyclic",
         "concrete01", "concrete02"),
        ("numpy",), ("dynamic", "relaxation"),
        ("nodal_force", "gravity", "body_force", "ground_motion"),
        ("displacement", "axial_force", "axial_force_max_abs",
         "axial_stress", "von_mises", "capacity_ratio", "member_failed", "reaction_force"),
        "co_rotational_vfife", "analytical",
        "Co-rotational axial bar; nonlinear uniaxial laws are advanced at one material point per member. "
        "As a cable: prestress or unstressed_length, and tension_only (slack). "
        "Standalone truss2 is NumPy only; inside mixed_line / coupled models the members run on the "
        "per-member loop whatever the backend.",
        xdmf=("Polyline", 2, ' NodesPerElement="2"'), gmsh_type=1,
        gmsh_order=(0, 1),
    ),
    "beam2": ElementCapability(
        "beam2", "line", ("line2",),
        ("ux", "uy", "uz", "rx", "ry", "rz"), ("euler_bernoulli_3d",),
        ("solid_3d",), ("linear_elastic",), ("numpy", "numba"), ("dynamic", "relaxation"),
        ("nodal_force", "nodal_moment", "line_load", "gravity", "ground_motion"),
        ("displacement", "rotation", "end_actions") + _MEMBER_FIELDS
        + ("reaction_force", "reaction_moment"),
        "co_rotational_vfife", "analytical",
        "Linear elastic Euler-Bernoulli beam; shear deformation is not represented.",
        xdmf=("Polyline", 2, ' NodesPerElement="2"'), gmsh_type=None,
        gmsh_order=None,
    ),
    "beam2_timoshenko": ElementCapability(
        "beam2_timoshenko", "line", ("line2",),
        ("ux", "uy", "uz", "rx", "ry", "rz"), ("timoshenko_3d",),
        ("solid_3d",), ("linear_elastic",), ("numpy", "numba"), ("dynamic", "relaxation"),
        ("nodal_force", "nodal_moment", "line_load", "gravity", "ground_motion"),
        ("displacement", "rotation", "end_actions") + _MEMBER_FIELDS
        + ("reaction_force", "reaction_moment"),
        "co_rotational_vfife", "analytical",
        "Linear elastic Timoshenko beam with independent local-y/local-z shear areas and correction factors.",
        xdmf=("Polyline", 2, ' NodesPerElement="2"'), gmsh_type=None,
        gmsh_order=None,
    ),
    "beam2_fiber": ElementCapability(
        "beam2_fiber", "line", ("line2",),
        ("ux", "uy", "uz", "rx", "ry", "rz"), ("fiber_euler_bernoulli_3d",),
        ("uniaxial",), ("linear_elastic", "rebar", "steel02", "steel_cyclic", "piecewise_uniaxial", "concrete01", "concrete02"),
        # `numpy` is the readable path, promoted to the fused numba kernel
        # whenever the section qualifies and numba is installed; `numba`
        # REQUIRES the kernel and is refused with the kernel's own reason
        # when it cannot be built. The table used to say
        # numpy only while the solver was already running the kernel.
        ("numpy", "numba"), ("dynamic", "relaxation"),
        ("nodal_force", "nodal_moment", "line_load", "gravity", "ground_motion"),
        # the fibre fields and the connectors a fibre model carries (its
        # bearings) were missing from what `describe` listed
        ("displacement", "rotation", "end_actions",
         "reaction_force", "reaction_moment") + _ACTIONS + tuple(f"{n}_max_abs" for n in _ACTIONS)
        + ("member_failed", "member_built", "fiber_stress", "fiber_damage",
         "connector_flag", "connector_force", "connector_moment", "connector_deformation",
         "connector_slip"),
        "co_rotational_vfife", "constitutive_path",
        "Fiber Euler-Bernoulli beam: rect, tube or box fibre sections (several per model), rebars, cover/core, "
        "gauss2 or lobatto3/4/5 stations, shear from moment gradient. backend numpy is promoted to the numba "
        "kernel when the model qualifies; backend numba requires it.",
        xdmf=("Polyline", 2, ' NodesPerElement="2"'), gmsh_type=None,
        gmsh_order=None,
    ),
    "shell4": ElementCapability(
        "shell4", "surface", ("quad4",),
        ("ux", "uy", "uz", "rx", "ry", "rz"),
        ("mindlin_reissner_3d", "selective_shear"),
        ("solid_3d",), ("linear_elastic",), ("numpy", "numba", "warp"), ("dynamic", "relaxation"),
        ("nodal_force", "nodal_moment", "surface_pressure", "surface_traction", "gravity", "ground_motion"),
        ("displacement", "rotation", "membrane_resultants", "bending_resultants",
         "shear_resultants", "shell_stress_top", "shell_stress_bottom",
         "von_mises_top", "von_mises_bottom", "reaction_force", "reaction_moment"),
        "co_rotational_vfife", "analytical",
        "EXPERIMENTAL, not covered by 1.0: the default hourglass term "
        "stiffens bending by 2-30% (with element aspect ratio) and refinement does not remove "
        "it; distorted quads lock. Homogeneous linear-elastic flat Quad4.",
        xdmf=("Quadrilateral", 4, ""), gmsh_type=3,
        gmsh_order=(0, 1, 2, 3),
    ),
    "shell3": ElementCapability(
        "shell3", "surface", ("tri3",),
        ("ux", "uy", "uz", "rx", "ry", "rz"),
        ("mindlin_reissner_3d",),
        ("solid_3d",), ("linear_elastic",), ("numpy",), ("dynamic", "relaxation"),
        ("nodal_force", "nodal_moment", "surface_pressure", "surface_traction", "gravity", "ground_motion"),
        ("displacement", "rotation", "membrane_resultants", "bending_resultants",
         "shear_resultants", "shell_stress_top", "shell_stress_bottom",
         "von_mises_top", "von_mises_bottom", "reaction_force", "reaction_moment"),
        "co_rotational_vfife", "analytical",
        "EXPERIMENTAL, not covered by 1.0: this constant-strain Mindlin "
        "triangle LOCKS in shear -- a thin plate deflects 0.2-20% of the true value. Use quads.",
        xdmf=("Triangle", 3, ""), gmsh_type=2,
        gmsh_order=(0, 1, 2),
    ),
    "shell_line": ElementCapability(
        "shell_line", "mixed_dimensional", ("quad4", "line2"),
        ("ux", "uy", "uz", "rx", "ry", "rz"), ("shared_node",),
        ("solid_3d",), ("linear_elastic",), ("numpy", "numba"), ("dynamic", "relaxation"),
        ("nodal_force", "nodal_moment", "surface_pressure", "surface_traction", "gravity", "ground_motion"),
        ("displacement", "rotation", "shell_stress_top", "shell_stress_bottom",
         "end_actions", "reaction_force", "reaction_moment"),
        "co_rotational_vfife", "independent_reference",
        "EXPERIMENTAL (its shell is shell4's). "
        "Conforming selected Line2 members; line_load currently uses equivalent nodal loads. "
        "Under backend numba the SHELL block uses the compiled kernel and the line members stay on numpy.",
        xdmf=None, gmsh_type=None,
        gmsh_order=None,
    ),
    "connector2": ElementCapability(
        "connector2", "line", ("line2",),
        ("ux", "uy", "uz", "rx", "ry", "rz"), ("six_component_connector",),
        ("solid_3d",), ("linear_elastic",), ("numpy",), ("dynamic", "relaxation"),
        # NOT gravity: a connector has no mass and no volume, so the load
        # assembles to exactly zero rather than being refused. Declaring it
        # would promise the user something that silently does nothing.
        ("nodal_force", "nodal_moment", "ground_motion"),
        ("displacement", "rotation",
         "connector_force", "connector_moment", "connector_deformation",
         "reaction_force", "reaction_moment"),
        "co_rotational_vfife", "analytical",
        "Per component: a linear stiffness, a reversible piecewise-linear curve, or a law (gap, bilinear hysteresis, Coulomb friction).",
        xdmf=("Polyline", 2, ' NodesPerElement="2"'), gmsh_type=None,
        gmsh_order=None,
    ),
    "mixed_line": ElementCapability(
        "mixed_line", "line", ("line2",),
        ("ux", "uy", "uz", "rx", "ry", "rz"), ("truss_beam_mixed",),
        ("solid_3d",), ("linear_elastic",), ("numpy", "numba"), ("dynamic", "relaxation"),
        ("nodal_force", "nodal_moment", "line_load", "gravity", "ground_motion"),
        ("displacement", "rotation", "member_type", "beam_formulation", "connector_flag",
         "connector_force", "connector_moment", "connector_deformation", "connector_slip",
         "end_actions") + _MEMBER_FIELDS + ("reaction_force", "reaction_moment"),
        "co_rotational_vfife", "analytical",
        xdmf=("Polyline", 2, ' NodesPerElement="2"'), gmsh_type=None,
        gmsh_order=None,
    ),
    "coupled": ElementCapability(
        "coupled", "mixed_dimensional", ("solid", "line2"),
        ("ux", "uy", "uz", "rx", "ry", "rz"), ("shared_node",),
        ("solid_3d",), ("linear_elastic",), ("numpy", "numba"), ("dynamic", "relaxation"),
        ("nodal_force", "nodal_moment", "line_load", "surface_pressure",
         "surface_traction",
         "gravity", "body_force", "ground_motion"),
        _COMMON_CONTINUUM_OUTPUTS + ("rotation", "member_type", "beam_formulation", "connector_flag",
                                    "connector_force", "connector_moment", "connector_deformation", "end_actions",
                                    "reaction_moment"),
        "co_rotational_vfife", "independent_reference",
        "Shared translations do not by themselves transfer a beam end moment into a solid.",
        xdmf=None, gmsh_type=None,
        gmsh_order=None,
    ),
}


def element_definitions() -> dict[str, ElementCapability]:
    """All continuum and structural elements through one metadata protocol."""
    out = dict(LINE_ELEMENTS)
    for name, elem in ELEMENTS.items():
        # Every continuum element runs on every backend. The fused kernels are
        # element-generic by construction: `make_kernel(..., n_nodes=...)` bakes
        # the node count in as a compile-time constant and the element supplies
        # only dN/dx and the quadrature.
        #
        # pyramid5, hexa20 and wedge15 were excluded here as a first-delivery
        # caution and never revisited. The list was enforced -- preflight refused
        # them with CAPABILITY_MISMATCH -- so those three ran ~3x slower than
        # necessary with no way to opt out, while the kernels computed them
        # correctly all along: verified against the numpy reference to <1e-11
        # across numba and warp, over a displacement history with rigid rotation
        # and plastic yielding, on exactly the evidence that justifies the other
        # four (tests/test_backends.py).
        backends = ("numpy", "numba", "warp")
        out[name] = ElementCapability(
            name=name, family="continuum", topology=(name,),
            dofs=("ux", "uy", "uz"), formulations=tuple(elem.formulations),
            material_kinematics=("solid_3d",),
            compatible_materials=("linear_elastic", "orthotropic", "viscoelastic", "j2", "chaboche", "cdp"),
            backends=backends,
            analysis_modes=("dynamic", "relaxation"),
            # A solid takes a traction as well as a pressure -- it always
            # did, in `Solver.apply_step`, and this tuple simply never said
            # so -- and the quadrilateral-faced ones take a `moment` on a
            # face. The tuple is what `vfife capabilities`
            # and `vfife describe` show, so a load missing from it is a
            # feature the user has no way to find; a load in it that does
            # not work is worse. tests/test_invariants.py now runs the round
            # trip both ways over every element, which is how
            # the four wrong entries here were found.
            loads=(("nodal_force", "surface_pressure", "surface_traction")
                   # a face moment is a couple carried by a quadrilateral
                   # face; the quadratic solids' faces are quad8 and tri6,
                   # which mesh.FaceMoment refuses
                   + (("face_moment",) if name in ("hexa8", "wedge6", "pyramid5") else ())
                   + ("gravity", "body_force", "ground_motion")),
            output_fields=_COMMON_CONTINUUM_OUTPUTS,
            geometric_nonlinearity="co_rotational_vfife",
            verification="patch_and_backend", caveat=elem.caveat,
            xdmf=elem.xdmf, gmsh_type=elem.gmsh_type,
            gmsh_order=elem.gmsh_order,
        )
    return {name: out[name] for name in sorted(out)}


def element_names() -> tuple[str, ...]:
    """Every element the case schema will accept, in sorted order."""
    return tuple(element_definitions())


def xdmf_topologies() -> dict[str, tuple[str, int, str]]:
    """{name: (TopologyType, nodes/element, extra attrs)} for the writer."""
    return {n: c.xdmf for n, c in element_definitions().items()
            if c.xdmf is not None}


def gmsh_types() -> dict[int, tuple[str, tuple[int, ...]]]:
    """{Gmsh MSH type: (our name, node permutation)} for the reader."""
    return {c.gmsh_type: (n, c.gmsh_order)
            for n, c in element_definitions().items()
            if c.gmsh_type is not None}


def families() -> dict[str, str]:
    """{name: family}. `continuum` runs the solid solver, everything else the
    structural one -- which is how the solver is chosen."""
    return {n: c.family for n, c in element_definitions().items()}


def element_definition(name: str) -> ElementCapability:
    definitions = element_definitions()
    if name not in definitions:
        raise KeyError(f"unknown element {name!r}; available: {sorted(definitions)}")
    return definitions[name]


def face_areas(coords: np.ndarray, elem: ElementDef) -> np.ndarray:
    """Area of every face, (n_elem, n_faces).

    Triangles exactly; quadrilaterals by the cross product of the diagonals,
    which is exact when planar and close enough for a characteristic length.
    Curved faces (tet10) use their corner nodes -- again, this feeds a stable
    time step, not a stress.
    """
    out = np.empty((coords.shape[0], len(elem.faces)))
    for f, nodes in enumerate(elem.faces):
        p = coords[:, list(nodes)]
        if len(nodes) == 3:
            a = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            out[:, f] = 0.5 * np.linalg.norm(a, axis=-1)
        elif len(nodes) == 6:
            a = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            out[:, f] = 0.5 * np.linalg.norm(a, axis=-1)
        elif len(nodes) == 8:
            # serendipity quad with mids: area from corner quad (first 4)
            p4 = p[:, :4]
            a = np.cross(p4[:, 2] - p4[:, 0], p4[:, 3] - p4[:, 1])
            out[:, f] = 0.5 * np.linalg.norm(a, axis=-1)
        else:
            a = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
            out[:, f] = 0.5 * np.linalg.norm(a, axis=-1)
    return out
