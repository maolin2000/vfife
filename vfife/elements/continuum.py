"""Isoparametric continuum elements: precompute (cold) and internal force (hot).

One implementation for every element in `library.ELEMENTS`.  Nothing below
mentions a node count: the co-rotational frame, the strain from the displacement
gradient, the material call and f = dN/dx^T . T are the same algebra whether the
element has 4 nodes or 10, and the shape derivatives and quadrature come from the
element definition.  Adding tet4, tet10 and wedge6 needed no changes here at all
-- only entries in `library.py`.

Two structural choices, unchanged from when this file was hexa8-only:

1.  COLD / HOT SPLIT.  `precompute()` runs once and stores dN/dx, w*detJ, the
    centroid operator and (hexa8 only) the Flanagan-Belytschko hourglass
    vectors.  The MATLAB original recomputed all of it every step for every
    element and every Gauss point, though it depends only on the reference
    configuration.

2.  NO B MATRIX IS EVER FORMED in the time loop.  Working with the displacement
    gradient instead of a 6x3n matrix is algebraically identical and removes
    most of the arithmetic:

        grad[i,j] = dN/dx[i,k] q[k,j]                    # 3x3
        eps       = [g00, g11, g22, g01+g10, g12+g21, g20+g02]
        f[k,j]    = w detJ dN/dx[i,k] T[i,j]             # T = stress as 3x3

    (`stiffness_row_sums` below is the one place a B matrix appears, and it runs
    once, only when Underwood relaxation masses are requested.)

Rotation extraction uses the quaternion iteration of Mueller et al. instead of
an SVD, WARM-STARTED from the previous step's quaternion: in explicit dynamics
the element rotation barely changes between steps, so one or two iterations
suffice where SVD pays full price every time.

Hourglass control (hexa8 reduced integration) uses properly projected FB vectors
and a stiffness that does NOT depend on dt.  The original's
`hg_stiffness = alpha_h/dt` meant that changing the time step -- or mass
scaling -- silently changed the hourglass stiffness, i.e. changed the answer.
"""

from __future__ import annotations

import numpy as np

from . import library
from .library import ElementDef


class Precomputed:
    """Reference-configuration data. Built once, then read every step."""

    __slots__ = (
        "elem", "formulation", "n_nodes", "n_gp", "dNdx", "dNdxT_w", "w_detJ",
        "dNdx0", "dNdx0T", "shape_gp", "gp_coords", "gamma", "gammaT", "gamma_norm", "volume",
        "char_len", "k_hg", "n_elem", "active",
    )

    def __init__(self, elem: ElementDef, formulation: str, n_gp: int):
        self.elem = elem
        self.formulation = formulation
        self.n_nodes = elem.n_nodes
        self.n_gp = n_gp
        self.k_hg = None

    @property
    def name(self) -> str:
        return self.elem.name


def precompute(
    nodes: np.ndarray,
    conn: np.ndarray,
    element: str = "hexa8",
    formulation: str = "b-bar",
    hourglass_coeff: float = 0.1,
    material=None,
    dtype=np.float64,
) -> Precomputed:
    elem = library.get(element)
    formulation = formulation.lower()
    nn = elem.n_nodes
    if conn.shape[1] != nn:
        raise ValueError(
            f"{element} needs {nn} nodes per element, connectivity has "
            f"{conn.shape[1]}"
        )

    if material is not None and getattr(material, "uniaxial", False):
        raise ValueError(
            f"material {material.name!r} is a one-dimensional law and cannot be "
            f"used in a {element} continuum element -- it has no shear "
            "stiffness at all, which would look like a converged answer. Use it "
            "with a truss element."
        )

    coords = nodes[conn]                       # (nel, nn, 3)
    nel = coords.shape[0]
    gp, weights = elem.gauss(formulation)
    dref = elem.dN_dxi(gp)                     # (ngp, 3, nn)

    J = np.einsum("gik,ekj->egij", dref, coords)
    detJ = np.linalg.det(J)
    if np.any(detJ <= 0):
        bad = int(np.sum(detJ <= 0))
        raise ValueError(
            f"{bad} integration points of {element} have non-positive "
            "Jacobian -- check the element node ordering in the mesh "
            "(this library follows the VTK/Abaqus convention)"
        )
    dNdx = np.einsum("egij,gjk->egik", np.linalg.inv(J), dref)

    pre = Precomputed(elem, formulation, gp.shape[0])
    pre.n_elem = nel
    # 1 while the element is in the calculation, 0 once it has been eroded
    pre.active = np.ones(nel, dtype=np.int8)
    pre.dNdx = np.ascontiguousarray(dNdx, dtype=dtype)
    pre.shape_gp = np.ascontiguousarray(library.shape_functions(element, gp), dtype=dtype)
    pre.gp_coords = np.ascontiguousarray(
        np.einsum("gn,enj->egj", pre.shape_gp, coords), dtype=dtype
    )
    pre.w_detJ = np.ascontiguousarray(detJ * weights[None, :], dtype=dtype)
    # dN/dx transposed and pre-multiplied by w*detJ: exactly the array the force
    # assembly needs, built once instead of allocated every step.
    pre.dNdxT_w = np.ascontiguousarray(
        (pre.dNdx * pre.w_detJ[..., None, None]).transpose(0, 1, 3, 2), dtype=dtype
    )

    # centroid operator: needed by b-bar and by the hourglass projection
    d0 = elem.dN_dxi(elem.centroid[None, :])
    J0 = np.einsum("gik,ekj->egij", d0, coords)[:, 0]
    pre.dNdx0 = np.ascontiguousarray(
        np.einsum("eij,jk->eik", np.linalg.inv(J0), d0[0]), dtype=dtype
    )
    pre.dNdx0T = np.ascontiguousarray(pre.dNdx0.transpose(0, 2, 1), dtype=dtype)

    # volume by the element's own accurate rule, whatever scheme is in use
    vp, vw = elem.quadrature[elem.volume_scheme]
    J_vol = np.einsum("gik,ekj->egij", elem.dN_dxi(vp), coords)
    pre.volume = np.ascontiguousarray(
        (np.linalg.det(J_vol) * vw[None, :]).sum(axis=1), dtype=dtype
    )

    # characteristic length = V / A_max, the quantity the CFL limit actually needs
    area = library.face_areas(coords, elem)
    pre.char_len = np.ascontiguousarray(
        pre.volume / np.maximum(area.max(axis=1), 1e-30), dtype=dtype
    )

    if formulation == "reduced":
        if elem.hourglass is None:
            raise ValueError(f"{element} has no hourglass modes defined")
        # Flanagan-Belytschko projection: remove the linear part of each mode
        H = elem.hourglass
        proj = np.einsum("ak,ekj->eaj", H, coords)           # (nel, 4, 3)
        gamma = H[None] - np.einsum("eaj,ejk->eak", proj, pre.dNdx0)
        pre.gamma = np.ascontiguousarray(gamma, dtype=dtype)
        pre.gammaT = np.ascontiguousarray(gamma.transpose(0, 2, 1), dtype=dtype)
        pre.gamma_norm = np.ascontiguousarray(
            np.maximum((gamma ** 2).sum(axis=2), 1e-30), dtype=dtype
        )
        if material is not None:
            # Scale with the SHEAR modulus, not the P-wave / bulk modulus.
            # M = E(1-nu)/((1+nu)(1-2nu)) diverges as nu -> 1/2, so a bulk-based
            # hourglass stiffness makes near-incompressible elements rigid --
            # measured on a cantilever at nu = 0.4999, the tip deflection came
            # out 64x too small, i.e. hourglass control reintroduced exactly the
            # locking that reduced integration exists to avoid.  G stays finite.
            G = material.E / (2.0 * (1.0 + material.nu))
            pre.k_hg = np.ascontiguousarray(
                hourglass_coeff * G * pre.volume ** (2.0 / 3.0)
                / (4.0 * pre.char_len),
                dtype=dtype,
            )
    else:
        pre.gamma = None
        pre.gammaT = None
        pre.gamma_norm = None

    return pre


def erode(pre, dead: np.ndarray) -> int:
    """Remove elements from the calculation. Returns how many newly died.

    Erosion is implemented by ZEROING THE QUADRATURE WEIGHTS rather than by
    compacting the connectivity or branching in the kernel, and that choice is
    what makes it work on all three backends without touching a single kernel:
    every path -- the deviatoric assembly (`dNdxT_w`), the b-bar volumetric term
    (`w_detJ`) and the hourglass term (`k_hg`) -- carries the element's weight
    as a factor, so a zeroed weight makes the internal force *identically* zero,
    not merely small.  The element stays in the arrays, keeps its node ids, and
    costs its share of the kernel; it simply stops pushing on anything.

    Paying for a dead element's arithmetic is the deliberate trade.  Compacting
    would break the deterministic gather (the inverse connectivity is built
    once, and its order is what makes CPU and GPU results comparable bit for
    bit) and would force a re-upload of every element array on the GPU each time
    one element died.  In the runs this was built for, erosion touches a few
    percent of the mesh.

    Erosion is ONE-WAY.  Nothing here restores an element, so the arrays are
    zeroed in place and no pristine copy is kept -- at 10^6 elements `dNdxT_w`
    alone is 1.5 GB in fp64, and a shadow copy of it would cost more than the
    feature.
    """
    dead = np.asarray(dead, dtype=bool)
    fresh = dead & (pre.active > 0)
    n = int(fresh.sum())
    if n == 0:
        return 0
    pre.active[fresh] = 0
    pre.w_detJ[fresh] = 0.0
    pre.dNdxT_w[fresh] = 0.0
    if pre.k_hg is not None:
        pre.k_hg[fresh] = 0.0
    return n


def strain_measures(pre, u_e: np.ndarray):
    """(det F, max principal Green-Lagrange strain) at each element centre.

    Both erosion criteria read from the same deformation gradient, so it is
    formed once.  The principal strain is the measure LS-DYNA's MAT_CSCM
    actually erodes on -- "damage exceeds 0.99 and the maximum principal strain
    exceeds ERODE-1" -- and using it rather than the material's own equivalent
    plastic strain matters more than it sounds:

    A plastic strain is scaled by whatever the constitutive model does with the
    element size.  Under crack-band regularisation the tensile plastic strain at
    full damage is O(2 Gf / (ft l)), which on the reinforced beam is 6e-3, so a
    threshold of 0.05 -- FHWA's own number for the principal strain -- never
    fires at all.  Measured: 56 elements at d = 0.99 and not one eroded.  The
    principal strain is a kinematic quantity with no such rescaling, so a
    threshold carries over between models and meshes.
    """
    F = np.matmul(pre.dNdx0, u_e).transpose(0, 2, 1)
    F[:, 0, 0] += 1.0
    F[:, 1, 1] += 1.0
    F[:, 2, 2] += 1.0

    # A safety net has to survive the thing it is catching.  By the time this
    # runs, the element it exists to remove may already hold inf or NaN, and
    # `eigvalsh` does not return from that -- it raises "Eigenvalues did not
    # converge" and takes the run down, which is a worse outcome than the
    # divergence it was called to stop.  Non-finite elements are substituted out
    # of the arithmetic and reported dead, which is what they are.
    bad = ~np.isfinite(F).all(axis=(1, 2))
    if bad.any():
        F[bad] = np.eye(3)
    J = np.linalg.det(F)
    E = 0.5 * (np.matmul(F.transpose(0, 2, 1), F)
               - np.eye(3)[None, :, :])
    return J, np.linalg.eigvalsh(E)[:, 2], bad


def volume_ratio(pre, u_e: np.ndarray) -> np.ndarray:
    """det(F) at each element centre -- the current volume over the reference.

    This is the criterion-free safety net, and it is the one both LS-DYNA and
    Abaqus keep alongside their material-based criteria: `*CONTROL_TIMESTEP`
    ERODE with a non-zero DTMIN deletes elements on a collapsing time step or a
    negative volume regardless of what the constitutive model thinks, and
    Abaqus/Explicit's distortion control exists for the same reason.  A material
    criterion can be tuned wrong; an inverted element is wrong under every
    tuning.
    """
    return strain_measures(pre, u_e)[0]


def stiffness_row_sums(
    nodes: np.ndarray, conn: np.ndarray, pre: Precomputed, D0: np.ndarray,
    chunk: int = 20000,
) -> np.ndarray:
    """sum_j |K_ij| per element dof, for Underwood relaxation masses.

    This is the one place a B matrix is built, and it runs once.
    Returns (n_elem, n_nodes, 3).
    """
    nel = conn.shape[0]
    nn = pre.n_nodes
    ndof = 3 * nn
    out = np.zeros((nel, nn, 3))
    for a in range(0, nel, chunk):
        b = min(a + chunk, nel)
        dNdx = pre.dNdx[a:b]                       # (n, ngp, 3, nn)
        w = pre.w_detJ[a:b]
        n = b - a
        K = np.zeros((n, ndof, ndof))
        for g in range(pre.n_gp):
            B = np.zeros((n, 6, ndof))
            dx = dNdx[:, g]
            for k in range(nn):
                B[:, 0, 3 * k + 0] = dx[:, 0, k]
                B[:, 1, 3 * k + 1] = dx[:, 1, k]
                B[:, 2, 3 * k + 2] = dx[:, 2, k]
                B[:, 3, 3 * k + 0] = dx[:, 1, k]
                B[:, 3, 3 * k + 1] = dx[:, 0, k]
                B[:, 4, 3 * k + 1] = dx[:, 2, k]
                B[:, 4, 3 * k + 2] = dx[:, 1, k]
                B[:, 5, 3 * k + 0] = dx[:, 2, k]
                B[:, 5, 3 * k + 2] = dx[:, 0, k]
            K += w[:, g, None, None] * np.einsum(
                "nji,jk,nkl->nil", B, D0, B
            )
        out[a:b] = np.abs(K).sum(axis=2).reshape(n, nn, 3)
    return out


def stiffness_max_eigenvalues(
    nodes: np.ndarray, conn: np.ndarray, pre: Precomputed, D0: np.ndarray,
    chunk: int = 20000,
) -> np.ndarray:
    """lambda_max of each element's stiffness, for the rotation-invariant
    relaxation mass `mass: eigenvalue`.

    Underwood's row sums bound the stiffest frequency of the REFERENCE
    configuration: a member that has rotated spreads its stiffness across
    other global rows and the row sums change. An element's eigenvalues do
    not change under rotation, and Hughes' element-eigenvalue inequality --
    max eig(K, M) <= max_e max eig(K_e, M_e) for K and M both assembled from
    element matrices -- makes the lumped mass m_e = lambda_max(K_e) dt^2 / 4
    per element node a bound that holds in every configuration. The
    hourglass stiffness of a reduced element adds k_hg, the eigenvalue of
    each of its modes. Returns (n_elem,).

    It was written on suspicion that the row sums were what stalled the
    rolled-up cantilever of case_20 at 225 degrees of tip rotation. They
    were not: the finite-difference tangent there put the stiffest mode at
    omega^2 = 1.94 against the limit of 4 under the row-sum mass, and the
    stall was the non-conservative face moment (mesh.FaceMoment). This mass
    is kept as the option it is -- a guarantee where the row sums are only
    a measurement of the reference state -- at the price of a smaller step:
    lambda_max is typically 2-4 times the largest row sum.
    """
    nel = conn.shape[0]
    nn = pre.n_nodes
    ndof = 3 * nn
    out = np.zeros(nel)
    for a in range(0, nel, chunk):
        b = min(a + chunk, nel)
        dNdx = pre.dNdx[a:b]
        w = pre.w_detJ[a:b]
        n = b - a
        K = np.zeros((n, ndof, ndof))
        for g in range(pre.n_gp):
            B = np.zeros((n, 6, ndof))
            dx = dNdx[:, g]
            for k in range(nn):
                B[:, 0, 3 * k + 0] = dx[:, 0, k]
                B[:, 1, 3 * k + 1] = dx[:, 1, k]
                B[:, 2, 3 * k + 2] = dx[:, 2, k]
                B[:, 3, 3 * k + 0] = dx[:, 1, k]
                B[:, 3, 3 * k + 1] = dx[:, 0, k]
                B[:, 4, 3 * k + 1] = dx[:, 2, k]
                B[:, 4, 3 * k + 2] = dx[:, 1, k]
                B[:, 5, 3 * k + 0] = dx[:, 2, k]
                B[:, 5, 3 * k + 2] = dx[:, 0, k]
            K += w[:, g, None, None] * np.einsum("nji,jk,nkl->nil", B, D0, B)
        out[a:b] = np.linalg.eigvalsh(K)[:, -1]
    if pre.formulation == "reduced" and pre.k_hg is not None:
        out += np.asarray(pre.k_hg, dtype=float)
    return out


# --------------------------------------------------------------------------- #
#  Rotation extraction (quaternion, warm-started)
# --------------------------------------------------------------------------- #


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(n,4) [w,x,y,z] -> (n,3,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty(q.shape[:1] + (3, 3), dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )


# Mueller's iteration normalises by tr(R^T A) = tr(S).  The exact Newton step
# needs (tr(S) I - S)^-1, which for a near-spherical stretch S ~ s I is
# 1/(2s) rather than 1/(3s) -- i.e. the plain step is 2/3 of Newton and
# converges linearly at ratio 1/3.  Scaling by 3/2 recovers the Newton step for
# spherical S and, since FEM stretches are close to the identity, cuts the
# error after two warm-started iterations by more than two orders of magnitude
# (measured: 0.113 -> 0.00085 of the per-step rotation increment).  For strongly
# distorted A it is still a contraction, so cold starts converge either way.
_OMEGA_SCALE = 1.5


def extract_rotation(A: np.ndarray, q: np.ndarray, max_iter: int = 2,
                     tol: float = 1e-12) -> np.ndarray:
    """Polar rotation of A (n,3,3), warm-started from quaternion q (n,4).

    q is updated in place and returned. Always yields a proper rotation
    (det = +1) by construction, so no reflection fix-up is needed.
    """
    for _ in range(max_iter):
        R = quat_to_mat(q)
        # sum_c (R_col_c x A_col_c), written out to avoid np.cross overhead
        omega = np.empty((q.shape[0], 3), dtype=q.dtype)
        omega[:, 0] = (R[:, 1] * A[:, 2] - R[:, 2] * A[:, 1]).sum(axis=1)
        omega[:, 1] = (R[:, 2] * A[:, 0] - R[:, 0] * A[:, 2]).sum(axis=1)
        omega[:, 2] = (R[:, 0] * A[:, 1] - R[:, 1] * A[:, 0]).sum(axis=1)
        denom = np.abs((R * A).sum(axis=(1, 2))) + 1e-12   # = |tr(R^T A)|
        omega *= _OMEGA_SCALE / denom[:, None]
        w = np.linalg.norm(omega, axis=1)
        if np.all(w < tol):
            break
        ang = np.where(w < 1e-30, 0.0, w)
        axis = omega / np.maximum(w, 1e-30)[:, None]
        s = np.sin(0.5 * ang)
        dq = np.concatenate([np.cos(0.5 * ang)[:, None], axis * s[:, None]], axis=1)
        q[:] = _quat_mul(dq, q)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


# --------------------------------------------------------------------------- #
#  Hot section: called every time step
# --------------------------------------------------------------------------- #


def internal_force(
    pre: Precomputed,
    x0_e: np.ndarray,       # (nel, 8, 3) reference coords, centred
    u_e: np.ndarray,        # (nel, 8, 3) displacements
    quat: np.ndarray,       # (nel, 4) previous rotation, updated in place
    material,
    state: np.ndarray,      # (nel*ngp, n_state), updated in place
    dt: float,
    rot_iters: int = 2,
    want_stress: bool = False,
):
    """Element internal force. Returns (f_e (nel, n_nodes, 3), stress or None).

    Nothing here depends on the node count: every operation is a batched matmul
    over whatever second dimension the arrays carry.  That is why the element
    library is data.
    """
    nel = x0_e.shape[0]
    ngp = pre.n_gp

    # --- co-rotational frame ------------------------------------------------
    # every contraction below is a batched matmul so it lands in BLAS;
    # np.einsum without optimize=True runs a naive C loop and was measured to
    # cost more than half of the whole time step.
    xn = x0_e + u_e
    xn = xn - xn.mean(axis=1, keepdims=True)

    # The rotation comes from the polar decomposition of the DEFORMATION
    # GRADIENT at the element centre, not from a Kabsch fit of the node
    # positions.  Those are not the same thing: sum(xn (x) x0) = F.M, where
    # M = sum(x0 (x) x0) is the reference second-moment tensor, and
    # polar(F.M) = polar(F) only when M is isotropic.  A cube's M is isotropic,
    # so the Kabsch fit was exact for hexa8 by accident of its geometry -- but a
    # tetrahedron's has anisotropy 1/3 and a wedge's 4/9, and there the fitted
    # "rotation" absorbs part of a pure strain.  Measured on a 1e-4 pure-strain
    # patch: spurious rotation 2.3e-5 (tet) and 3.8e-5 (wedge), and a patch test
    # that failed at 1e-5 instead of round-off.  F costs the same to form and is
    # the standard co-rotational definition.
    F = np.matmul(pre.dNdx0, u_e).transpose(0, 2, 1)    # dU_i/dX_j
    F[:, 0, 0] += 1.0
    F[:, 1, 1] += 1.0
    F[:, 2, 2] += 1.0
    extract_rotation(F, quat, max_iter=rot_iters)
    R = quat_to_mat(quat)                               # reference -> current
    q = np.matmul(xn, R) - x0_e                         # de-rotated displacement

    # --- strain from the displacement gradient ------------------------------
    grad = np.matmul(pre.dNdx, q[:, None])              # (nel,ngp,3,3)
    eps = np.empty((nel, ngp, 6))
    eps[..., 0] = grad[..., 0, 0]
    eps[..., 1] = grad[..., 1, 1]
    eps[..., 2] = grad[..., 2, 2]
    eps[..., 3] = grad[..., 0, 1] + grad[..., 1, 0]
    eps[..., 4] = grad[..., 1, 2] + grad[..., 2, 1]
    eps[..., 5] = grad[..., 2, 0] + grad[..., 0, 2]

    if pre.formulation == "b-bar":
        grad0 = np.matmul(pre.dNdx0, q)
        ev_bar = np.trace(grad0, axis1=1, axis2=2)
        ev = eps[..., 0] + eps[..., 1] + eps[..., 2]
        corr = (ev_bar[:, None] - ev) / 3.0
        eps[..., 0] += corr
        eps[..., 1] += corr
        eps[..., 2] += corr

    # --- constitutive update ------------------------------------------------
    sig = material.update(eps.reshape(nel * ngp, 6), state, dt).reshape(nel, ngp, 6)

    # --- internal force: f[k,j] = w*detJ * dNdx[i,k] * T[i,j] ---------------
    T = np.empty((nel, ngp, 3, 3))
    T[..., 0, 0] = sig[..., 0]
    T[..., 1, 1] = sig[..., 1]
    T[..., 2, 2] = sig[..., 2]
    T[..., 0, 1] = T[..., 1, 0] = sig[..., 3]
    T[..., 1, 2] = T[..., 2, 1] = sig[..., 4]
    T[..., 2, 0] = T[..., 0, 2] = sig[..., 5]

    if pre.formulation == "b-bar":
        # B_bar^T sigma = sum_g w_g dNdx_g^T (T_g - tr_g I) + (sum_g w_g tr_g) dNdx0^T
        # i.e. the ordinary assembly on the DEVIATORIC stress plus one
        # volume-averaged pressure term.  Written this way the correction costs
        # a diagonal subtraction instead of a second full-size contraction.
        tr = (sig[..., 0] + sig[..., 1] + sig[..., 2]) / 3.0
        T[..., 0, 0] -= tr
        T[..., 1, 1] -= tr
        T[..., 2, 2] -= tr

    # accumulate over Gauss points instead of forming an (nel, ngp, 8, 3)
    # temporary and reducing it -- same result, a fraction of the memory traffic
    f_loc = np.matmul(pre.dNdxT_w[:, 0], T[:, 0])
    for g in range(1, ngp):
        f_loc += np.matmul(pre.dNdxT_w[:, g], T[:, g])

    if pre.formulation == "b-bar":
        f_loc += (pre.w_detJ * tr).sum(axis=1)[:, None, None] * pre.dNdx0T

    if pre.formulation == "reduced" and pre.k_hg is not None:
        qm = np.matmul(pre.gamma, q)                                 # (nel,4,3)
        qm /= pre.gamma_norm[:, :, None]
        f_loc += pre.k_hg[:, None, None] * np.matmul(pre.gammaT, qm)

    # --- rotate back to the global frame ------------------------------------
    f_e = np.matmul(f_loc, R.transpose(0, 2, 1))
    return f_e, (sig if want_stress else None)
