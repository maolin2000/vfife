"""Fused Numba element kernel for hexa8.

The NumPy backend is the readable reference: it expresses each stage as a
batched tensor operation, which is clear but memory-bound -- every stage
allocates and rewrites an array the size of the mesh, and the arithmetic
intensity per byte moved is tiny.

This kernel does the whole element in one pass with scalar locals: gather,
polar decomposition, strain, constitutive update, internal force, rotate back.
Nothing intermediate reaches memory.  That is also exactly the shape a GPU
kernel has to be, so `make_kernel` is the prototype for the Warp backend, not a
detour on the way to it.

The material enters as a device function (`Material.numba_point_fn()`) closed
over its own parameters and tables, and is inlined here.  Adding a material
still means adding one file.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numba import njit, prange

from .assembly import GatherAssembler
from .scalar import extract_rotation, quat_to_mat

_FULL, _REDUCED, _BBAR = 0, 1, 2

_FORMULATIONS = {"full": _FULL, "reduced": _REDUCED, "b-bar": _BBAR}


_KERNEL_CACHE = {}


def _closure_key(point_fn):
    """Identify a material point function by WHAT IT CAPTURES, not by object.

    `Material.numba_point_fn()` builds a fresh njit device function on every
    call -- measured: two calls on the same material object are not the same
    object -- so keying the cache on the function itself never hits, and every
    solver built from the same material recompiles the element kernel. That is
    8-12 s each time, paid again on every `run_case`.

    Keying on a hand-listed parameter set was the obvious alternative and it is
    the dangerous one: the list drifts the moment someone captures a new
    variable, and the failure mode is two DIFFERENT materials silently sharing
    a kernel. `py_func.__closure__` is by construction exactly the set of values
    the kernel can see, so it cannot omit one.

    Objects that are neither arrays nor scalars fall back to `id()`. That is
    safe in the conservative direction -- a new object misses the cache and
    recompiles -- and the one hazard, an id reused after garbage collection, is
    closed by the cache itself: the stored kernel closes over `point_fn`, which
    keeps its own cells alive, so nothing keyed in the cache can be collected.

    Measured on tests/test_backends + test_cases + test_erosion + test_kinematics
    + test_cdp: 33 kernel builds, 27 distinct keys, 6 recompiles avoided out of
    a 328 s run.
    """
    py = getattr(point_fn, "py_func", point_fn)
    cells = getattr(py, "__closure__", None) or ()
    h = hashlib.blake2b(digest_size=16)
    h.update(getattr(py, "__qualname__", "?").encode())
    for c in cells:
        try:
            v = c.cell_contents
        except ValueError:                  # empty cell, recursive closure
            h.update(b"<empty>")
            continue
        if isinstance(v, np.ndarray):
            h.update(b"a")
            h.update(str(v.dtype).encode())
            h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v, (int, float, bool, str, bytes)):
            h.update(b"s")
            h.update(repr(v).encode())
        else:
            h.update(b"o")
            h.update(f"{type(v).__name__}:{id(v)}".encode())
    return h.hexdigest()


def make_kernel(point_fn, formulation: str, n_gp: int, n_nodes: int = 8,
                parallel: bool = True):
    """Build the fused element kernel for one material, formulation and element.

    `formulation`, `n_gp` and `n_nodes` are baked in as compile-time constants,
    so the branches disappear, the Gauss-point loop unrolls and the local
    scratch arrays are stack-allocated at fixed size.  One source serves hexa8,
    tet4, tet10 and wedge6 -- the element only supplies dN/dx and quadrature.
    """
    key = (_closure_key(point_fn), formulation, int(n_gp), int(n_nodes),
           bool(parallel))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit

    form = _FORMULATIONS[formulation]
    NGP = int(n_gp)
    NN = int(n_nodes)
    IS_BBAR = form == _BBAR
    IS_REDUCED = form == _REDUCED

    # cache=False deliberately: the kernel closes over a freshly built material
    # device function, so Numba's on-disk cache never hits (measured: 7.3 s on
    # the first call whether or not caching is on).  That one-time compile is
    # the price of inlining the material into the element loop, and it buys
    # ~5x on this case -- 47 s saved against 7 s spent on the beam benchmark.
    @njit(cache=False, fastmath=True, parallel=parallel, nogil=True)
    def kernel(conn, x0_e, u, dNdx, dNdxT_w, w_detJ, dNdx0, dNdx0T,
               gamma, gamma_norm, k_hg, quat, state, kband, dt,
               rot_iters, rot_tol, f_e, sig_out, want_stress):
        nel = conn.shape[0]
        for e in prange(nel):
            # --- gather and centre the current configuration ---------------
            xn = np.empty((NN, 3))
            u_loc = np.empty((NN, 3))
            for a in range(NN):
                n = conn[e, a]
                for i in range(3):
                    u_loc[a, i] = u[n, i]
                    xn[a, i] = x0_e[e, a, i] + u[n, i]
            for i in range(3):
                c = 0.0
                for a in range(NN):
                    c += xn[a, i]
                c /= NN
                for a in range(NN):
                    xn[a, i] -= c

            # --- co-rotational frame ------------------------------------
            # Polar decomposition of the DEFORMATION GRADIENT at the element
            # centre, not a Kabsch fit of node positions: sum(xn (x) x0) is
            # F.M with M the reference second-moment tensor, and polar(F.M)
            # equals polar(F) only when M is isotropic.  True for a cube, false
            # for a tet or a wedge -- see elements/continuum.py.
            A = np.empty((3, 3))
            for i in range(3):
                for j in range(3):
                    s = 0.0
                    for k in range(NN):
                        s += dNdx0[e, j, k] * u_loc[k, i]
                    A[i, j] = s
                A[i, i] += 1.0

            qv = np.empty(4)
            for i in range(4):
                qv[i] = quat[e, i]
            extract_rotation(A, qv, rot_iters, rot_tol)
            for i in range(4):
                quat[e, i] = qv[i]
            R = np.empty((3, 3))
            quat_to_mat(qv[0], qv[1], qv[2], qv[3], R)

            qd = np.empty((NN, 3))
            for a in range(NN):
                for j in range(3):
                    s = 0.0
                    for i in range(3):
                        s += xn[a, i] * R[i, j]
                    qd[a, j] = s - x0_e[e, a, j]

            # --- b-bar needs the centroid dilatation before the gp loop -----
            ev_bar = 0.0
            if IS_BBAR:
                for i in range(3):
                    s = 0.0
                    for k in range(NN):
                        s += dNdx0[e, i, k] * qd[k, i]
                    ev_bar += s

            f_loc = np.zeros((NN, 3))
            eps = np.empty(6)
            sig = np.empty(6)
            wtr_sum = 0.0

            for g in range(NGP):
                # strain from the displacement gradient -- no B matrix
                grad = np.empty((3, 3))
                for i in range(3):
                    for j in range(3):
                        s = 0.0
                        for k in range(NN):
                            s += dNdx[e, g, i, k] * qd[k, j]
                        grad[i, j] = s
                eps[0] = grad[0, 0]
                eps[1] = grad[1, 1]
                eps[2] = grad[2, 2]
                eps[3] = grad[0, 1] + grad[1, 0]
                eps[4] = grad[1, 2] + grad[2, 1]
                eps[5] = grad[2, 0] + grad[0, 2]

                if IS_BBAR:
                    corr = (ev_bar - (eps[0] + eps[1] + eps[2])) / 3.0
                    eps[0] += corr
                    eps[1] += corr
                    eps[2] += corr

                p = e * NGP + g
                point_fn(eps, state[p], dt, kband[p], sig)
                if want_stress:
                    for i in range(6):
                        sig_out[e, g, i] = sig[i]

                # stress as a 3x3 tensor; b-bar uses its deviator plus one
                # volume-averaged pressure term added after the loop
                T = np.empty((3, 3))
                tr = (sig[0] + sig[1] + sig[2]) / 3.0
                if IS_BBAR:
                    T[0, 0] = sig[0] - tr
                    T[1, 1] = sig[1] - tr
                    T[2, 2] = sig[2] - tr
                    wtr_sum += w_detJ[e, g] * tr
                else:
                    T[0, 0] = sig[0]
                    T[1, 1] = sig[1]
                    T[2, 2] = sig[2]
                T[0, 1] = sig[3]
                T[1, 0] = sig[3]
                T[1, 2] = sig[4]
                T[2, 1] = sig[4]
                T[2, 0] = sig[5]
                T[0, 2] = sig[5]

                for k in range(NN):
                    for j in range(3):
                        s = 0.0
                        for i in range(3):
                            s += dNdxT_w[e, g, k, i] * T[i, j]
                        f_loc[k, j] += s

            if IS_BBAR:
                for k in range(NN):
                    for j in range(3):
                        f_loc[k, j] += wtr_sum * dNdx0T[e, k, j]

            if IS_REDUCED:
                for a in range(4):
                    inv = 1.0 / gamma_norm[e, a]
                    for j in range(3):
                        qm = 0.0
                        for k in range(NN):
                            qm += gamma[e, a, k] * qd[k, j]
                        qm *= inv * k_hg[e]
                        for k in range(NN):
                            f_loc[k, j] += gamma[e, a, k] * qm

            # --- rotate the element force back to the global frame ---------
            for a in range(NN):
                for i in range(3):
                    s = 0.0
                    for j in range(3):
                        s += f_loc[a, j] * R[i, j]
                    f_e[e, a, i] = s

    _KERNEL_CACHE[key] = kernel
    return kernel


class NumbaForce:
    """Callable with the same contract as the NumPy `continuum.internal_force`.

    Buffers are allocated once and reused, so a time step does no allocation at
    all on this path -- which is most of why it is faster, not the JIT alone.
    """

    def __init__(self, pre, conn, x0_e, material, rot_tol: float = 1e-12,
                 parallel: bool = True, n_nodes=None):
        if not hasattr(material, "numba_point_fn"):
            raise TypeError(
                f"material {material.name!r} has no numba_point_fn(); "
                "the Numba backend needs a scalar twin of update(). "
                "Use backend: numpy, or add one (see materials/elastic.py)."
            )
        self.pre = pre
        self.conn = np.ascontiguousarray(conn, dtype=np.int32)
        self.x0_e = np.ascontiguousarray(x0_e)
        self.material = material
        self.rot_tol = rot_tol
        nel, ngp, nn = pre.n_elem, pre.n_gp, pre.n_nodes

        self.kernel = make_kernel(material.numba_point_fn(), pre.formulation,
                                  ngp, n_nodes=nn, parallel=parallel)

        z2 = np.zeros((nel, 3, nn))
        self.gamma = pre.gamma if pre.gamma is not None else np.zeros((nel, 4, nn))
        self.gamma_norm = (pre.gamma_norm if pre.gamma_norm is not None
                           else np.ones((nel, 4)))
        self.k_hg = pre.k_hg if pre.k_hg is not None else np.zeros(nel)
        self.dNdx0 = pre.dNdx0 if pre.dNdx0 is not None else z2
        self.dNdx0T = pre.dNdx0T

        k = getattr(material, "k_band", 1.0)
        self.kband = (np.ascontiguousarray(k, dtype=np.float64)
                      if np.ndim(k) else np.full(nel * ngp, float(k)))
        if self.kband.size != nel * ngp:
            raise ValueError("crack-band factor array has the wrong length")

        # `n_nodes` is the MODEL's node count: a material block or one element
        # type of a mixed mesh need not touch the highest-numbered node, and
        # its assembled force must still be the full nodal vector
        self._assembler = GatherAssembler(
            self.conn, int(n_nodes) if n_nodes is not None else int(self.conn.max()) + 1)
        self.f_e = np.zeros((nel, nn, 3))
        self.sig = np.zeros((nel, ngp, 6))
        self._dummy = np.zeros((1, 1, 6))

    def assemble(self, f_e):
        return self._assembler(f_e)

    def __call__(self, u, quat, state, dt, rot_iters=2, want_stress=False):
        self.kernel(self.conn, self.x0_e, u, self.pre.dNdx, self.pre.dNdxT_w,
                    self.pre.w_detJ, self.dNdx0, self.dNdx0T,
                    self.gamma, self.gamma_norm, self.k_hg,
                    quat, state, self.kband, dt, rot_iters, self.rot_tol,
                    self.f_e, self.sig if want_stress else self._dummy,
                    want_stress)
        return self.f_e, (self.sig if want_stress else None)
