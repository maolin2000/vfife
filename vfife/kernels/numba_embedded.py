"""Embedded reinforcement under `backend: numba` (2.0 route item 1,
v1.10).

`elements/truss.EmbeddedTruss.force` is the readable reference: the bar ends
gathered from the host nodes with two einsums, the strain, the uniaxial law
on the whole set as NumPy arrays, the axial force, and the force spread back
with the same weights -- a dozen temporaries the size of the bar set, and
then `mesh.scatter_add`. On the U of Alberta beam (case_12) that was 1.2 ms of
a 4.2 ms relaxation iteration, more than the solid kernel's Python around it.

This kernel does the same in one pass per bar, the law entering as the
material's own `numba_point_fn` (the device function the solid kernel
inlines), and adds the bar's nodal forces into the model's nodal force in
place. The same arithmetic as the reference, in the reference's order where
it has one; the law's compiled twin differs from its NumPy `update` at
round-off (the solid materials' twins have since v0.1), so the
answer of a numba run moves in the last places.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .numba_hexa8 import _closure_key

_CACHE: dict = {}


def make_bar_kernel(point_fn, n_host: int):
    """The force kernel of one bar set: `point_fn` is the law's device
    function (an embedded bar's law is one-dimensional: EmbeddedTruss
    refuses the others), `n_host` the host nodes per bar end, baked in."""
    key = (_closure_key(point_fn), int(n_host))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    NNH = int(n_host)
    law = point_fn

    # cache=False: the kernel closes over the law's freshly built device
    # function (see numba_hexa8.make_kernel); a bar kernel compiles in about
    # a second, once per process and law
    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def kernel(u, conn, w, x0_e, L0, area, pre_arr, pre_s, use_arr, host_ded, tonly,
               state, dt, F, sig_out, want):
        nel = conn.shape[0]
        strain = np.zeros(6)
        stress = np.zeros(6)
        d = np.empty(3)
        for e in range(nel):
            # the bar ends, interpolated from their host nodes (EmbeddedTruss.gather)
            x0 = 0.0; y0 = 0.0; z0 = 0.0
            x1 = 0.0; y1 = 0.0; z1 = 0.0
            for a in range(NNH):
                wa = w[e, 0, a]
                n = conn[e, a]
                x0 += wa * u[n, 0]; y0 += wa * u[n, 1]; z0 += wa * u[n, 2]
            for a in range(NNH):
                wa = w[e, 1, a]
                n = conn[e, NNH + a]
                x1 += wa * u[n, 0]; y1 += wa * u[n, 1]; z1 += wa * u[n, 2]
            d[0] = (x0_e[e, 1, 0] + x1) - (x0_e[e, 0, 0] + x0)
            d[1] = (x0_e[e, 1, 1] + y1) - (x0_e[e, 0, 1] + y0)
            d[2] = (x0_e[e, 1, 2] + z1) - (x0_e[e, 0, 2] + z0)
            L = np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            Ld = L if L > 1e-30 else 1e-30
            eps_mech = (L - L0[e]) / L0[e]
            ps = pre_arr[e] if use_arr else pre_s
            strain[0] = eps_mech + ps
            law(strain, state[e], dt, 1.0, stress)
            s0 = stress[0]
            N = (s0 - host_ded * eps_mech) * area[e]
            if tonly and N < 0.0:
                N = 0.0
            if want:
                for i in range(6):
                    sig_out[e, i] = stress[i]
            f0 = N * (d[0] / Ld); f1 = N * (d[1] / Ld); f2 = N * (d[2] / Ld)
            # spread back with the same weights: -f at end 0, +f at end 1
            for a in range(NNH):
                wa = w[e, 0, a]
                n = conn[e, a]
                F[n, 0] += wa * (-f0); F[n, 1] += wa * (-f1); F[n, 2] += wa * (-f2)
            for a in range(NNH):
                wa = w[e, 1, a]
                n = conn[e, NNH + a]
                F[n, 0] += wa * f0; F[n, 1] += wa * f1; F[n, 2] += wa * f2
        return 0

    _CACHE[key] = kernel
    return kernel


class BarForce:
    """`EmbeddedTruss.force` + the scatter, compiled: adds the bars' nodal
    force into `F` (nn, 3) and returns the stress (nel, 1, 6) when asked."""

    def __init__(self, emb):
        m = emb.material
        self.emb = emb
        fn = m.numba_point_fn() if getattr(m, "uniaxial", False) and hasattr(m, "numba_point_fn") else None
        if fn is None:
            # steel02 has none (its update is a Python loop over a compiled
            # point): those bars keep the reference path
            raise TypeError(f"{m.name} has no compiled one-dimensional twin")
        self.kernel = make_bar_kernel(fn, emb.n_host_nodes)
        self.conn = np.ascontiguousarray(emb.conn, dtype=np.int64)
        self.w = np.ascontiguousarray(emb.weights, dtype=np.float64)
        self.x0_e = np.ascontiguousarray(emb.x0_e, dtype=np.float64)
        self.sig = np.zeros((emb.n_elem, 6))
        self._z = np.zeros(1)

    def __call__(self, u, dt, F, want_stress=False):
        e = self.emb
        p = e.pre
        ps = p.prestrain
        arr = np.ndim(ps) > 0
        pre_arr = np.ascontiguousarray(ps, dtype=np.float64) if arr else self._z
        if arr and pre_arr.shape != (e.n_elem,):
            # the kernel reads one per bar; anything else would be misread
            pre_arr = np.ascontiguousarray(np.broadcast_to(pre_arr, (e.n_elem,)), dtype=np.float64)
        st = e.state
        if not (isinstance(st, np.ndarray) and st.dtype == np.float64 and st.flags.c_contiguous
                and st.ndim == 2):
            st = e.state = np.ascontiguousarray(st, dtype=np.float64)
        self.kernel(u, self.conn, self.w, self.x0_e, p.L0, p.area, pre_arr,
                    0.0 if arr else float(ps), arr, float(p.host_deduction),
                    bool(getattr(p, "tension_only", False)), st, float(dt), F, self.sig,
                    bool(want_stress))
        return self.sig.reshape(e.n_elem, 1, 6).copy() if want_stress else None
