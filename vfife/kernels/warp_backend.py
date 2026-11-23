"""Warp GPU backend -- the same fused element kernel, on the device.

This is a transcription of `numba_hexa8.py`, not a redesign, and that was the
point of writing the Numba kernel first: gather, warm-started polar
decomposition, strain from the displacement gradient, constitutive update,
internal force, rotate back, all in one pass with fixed-size locals and nothing
intermediate reaching memory.  `prange` becomes `wp.tid()`, `np.empty((8,3))`
becomes a `wp.mat` type, and the material device function is passed in exactly
as before.

PRECISION
---------
`precision="fp64"` runs everything in double and is what the port is checked
against the NumPy reference with (1e-12).  It is not the production mode: a
consumer GeForce runs fp64 at 1/64 of fp32, so a double-precision element
kernel throws away 98% of the card.

`precision="mixed"` is the production mode and follows the plan's §05: element
data, material state and the element kernel in fp32; nodal displacement,
velocity and the assembled force in fp64.  The nodal integration is a tiny
fraction of the arithmetic and is bandwidth-bound anyway, so fp64 there costs
almost nothing -- and it is where precision actually matters, because a 70 000
step run accumulates displacement increments that are small compared with the
coordinates they sit on.

RESIDENCY
---------
The nodal state lives on the device and stays there.  That is not an
optimisation, it is the difference between a GPU backend and a demo: with the
displacement, velocity, material state and element force copied across the bus
every step, a 10^6-element CDP case moves about 280 MB per iteration, which on
PCIe 5.0 x8 is ~14 ms against a 3-8 ms kernel.  The transfers would cost more
than the compute they exist to feed.

So the integrator runs on the device too (`integrate`, `apply_bc`), and the
host only sees scalars: kinetic energy for the damping decision each step, and
residual norms at progress intervals.  Fields come back only when a frame is
written or a checkpoint is taken.

MATERIAL INTERFACE
------------------
A material hands the device exactly two buffers whatever its shape: `prm`
(scalars) and `tab` (packed curve tables), plus a `wp.func` that reads them.
Structs with array members work too, but two flat buffers keep the kernel
signature identical for every material, so the factory below stays generic.
"""

# NOTE: no `from __future__ import annotations` here on purpose.  Warp resolves
# kernel argument annotations by evaluating them, and deferred (string)
# annotations cannot see the factory's closure variables -- `wtype` in
# particular.  Keeping annotations eager is what lets one kernel source compile
# for both fp32 and fp64.

from typing import Any

import numpy as np

from .warp_scalar import curve_eval, polar_quat, principal  # noqa: F401

_PRECISIONS = ("fp64", "mixed")


def _types(wtype, n_nodes):
    import warp as wp

    return {
        "vec6": wp.types.vector(length=6, dtype=wtype),
        "matN3": wp.types.matrix(shape=(int(n_nodes), 3), dtype=wtype),
        "mat33": wp.types.matrix(shape=(3, 3), dtype=wtype),
        "quat": wp.types.quaternion(dtype=wtype),
    }


def make_kernel(point_fn, formulation: str, n_gp: int, wtype, n_nodes: int = 8):
    """Build the element kernel for one material, formulation, element, precision.

    `n_nodes` sizes the local scratch matrices, so one source compiles for
    hexa8, tet4, tet10 and wedge6 -- the element supplies only dN/dx and the
    quadrature, both of which arrive as arrays.
    """
    import warp as wp

    T = _types(wtype, n_nodes)
    vec6, matN3, mat33, quatT = T["vec6"], T["matN3"], T["mat33"], T["quat"]
    NGP = wp.constant(int(n_gp))
    NN = wp.constant(int(n_nodes))
    # a separate constant, so using the node count in a float expression cannot
    # unify NN itself to a float and break `range(NN)`.  Its precision does not
    # matter: an error in the centroid shifts every qd by the same vector, and
    # sum_k dN/dx[i,k] = 0, so a rigid translation contributes no strain at all.
    INV_NN = wp.constant(1.0 / float(n_nodes))
    IS_BBAR = wp.constant(formulation == "b-bar")
    IS_REDUCED = wp.constant(formulation == "reduced")

    @wp.kernel
    def kernel(conn: wp.array2d(dtype=wp.int32),
               x0_e: wp.array3d(dtype=wtype),
               u: wp.array2d(dtype=wp.float64),
               dNdx: wp.array4d(dtype=wtype),
               w_detJ: wp.array2d(dtype=wtype),
               dNdx0: wp.array3d(dtype=wtype),
               gamma: wp.array3d(dtype=wtype),
               gamma_norm: wp.array2d(dtype=wtype),
               k_hg: wp.array(dtype=wtype),
               quat: wp.array2d(dtype=wtype),
               state: wp.array2d(dtype=wtype),
               kband: wp.array(dtype=wtype),
               prm: wp.array(dtype=wtype),
               tab: wp.array(dtype=wtype),
               dt: wtype,
               rot_iters: int,
               rot_tol: wtype,
               f_e: wp.array3d(dtype=wtype),
               sig_out: wp.array3d(dtype=wtype),
               want_stress: int):
        e = wp.tid()
        zero = wtype(0.0)

        # --- gather and centre the current configuration -------------------
        xn = matN3()
        u_loc = matN3()
        for a in range(NN):
            n = conn[e, a]
            for i in range(3):
                u_loc[a, i] = wtype(u[n, i])
                xn[a, i] = x0_e[e, a, i] + u_loc[a, i]
        for i in range(3):
            c = zero
            for a in range(NN):
                c += xn[a, i]
            c *= wtype(INV_NN)
            for a in range(NN):
                xn[a, i] -= c

        # --- co-rotational frame, warm-started -----------------------------
        # Polar decomposition of the DEFORMATION GRADIENT at the element centre.
        # A Kabsch fit of node positions gives polar(F.M), M the reference
        # second-moment tensor, which equals polar(F) only for an isotropic M --
        # true for a cube, false for a tet or a wedge. See elements/continuum.py.
        A = mat33()
        for i in range(3):
            for j in range(3):
                s = zero
                for k in range(NN):
                    s += dNdx0[e, j, k] * u_loc[k, i]
                A[i, j] = s
            A[i, i] += wtype(1.0)

        # state stores (w, x, y, z); wp.quat is (x, y, z, w)
        q = quatT(quat[e, 1], quat[e, 2], quat[e, 3], quat[e, 0])
        q = polar_quat(A, q, rot_iters, rot_tol)
        quat[e, 0] = q[3]
        quat[e, 1] = q[0]
        quat[e, 2] = q[1]
        quat[e, 3] = q[2]
        R = wp.quat_to_matrix(q)

        qd = matN3()
        for a in range(NN):
            for j in range(3):
                s = zero
                for i in range(3):
                    s += xn[a, i] * R[i, j]
                qd[a, j] = s - x0_e[e, a, j]

        ev_bar = zero
        if IS_BBAR:
            for i in range(3):
                s = zero
                for k in range(NN):
                    s += dNdx0[e, i, k] * qd[k, i]
                ev_bar += s

        f_loc = matN3()
        for a in range(NN):
            for j in range(3):
                f_loc[a, j] = zero
        wtr_sum = zero

        for g in range(NGP):
            grad = mat33()
            for i in range(3):
                for j in range(3):
                    s = zero
                    for k in range(NN):
                        s += dNdx[e, g, i, k] * qd[k, j]
                    grad[i, j] = s

            e0 = grad[0, 0]
            e1 = grad[1, 1]
            e2 = grad[2, 2]
            if IS_BBAR:
                corr = (ev_bar - (e0 + e1 + e2)) / wtype(3.0)
                e0 += corr
                e1 += corr
                e2 += corr
            eps = wp.vector(e0, e1, e2,
                            grad[0, 1] + grad[1, 0],
                            grad[1, 2] + grad[2, 1],
                            grad[2, 0] + grad[0, 2])

            p = e * NGP + g
            sig = point_fn(prm, tab, eps, state, p, dt, kband[p])
            if want_stress == 1:
                for i in range(6):
                    sig_out[e, g, i] = sig[i]

            T3 = mat33()
            tr = (sig[0] + sig[1] + sig[2]) / wtype(3.0)
            if IS_BBAR:
                T3[0, 0] = sig[0] - tr
                T3[1, 1] = sig[1] - tr
                T3[2, 2] = sig[2] - tr
                wtr_sum += w_detJ[e, g] * tr
            else:
                T3[0, 0] = sig[0]
                T3[1, 1] = sig[1]
                T3[2, 2] = sig[2]
            T3[0, 1] = sig[3]
            T3[1, 0] = sig[3]
            T3[1, 2] = sig[4]
            T3[2, 1] = sig[4]
            T3[2, 0] = sig[5]
            T3[0, 2] = sig[5]

            # index dNdx transposed rather than uploading a transposed copy:
            # the NumPy and Numba paths cache dNdxT_w for cache locality, but on
            # the device that is a second full-size array to hold and stream
            w = w_detJ[e, g]
            for k in range(NN):
                for j in range(3):
                    s = zero
                    for i in range(3):
                        s += dNdx[e, g, i, k] * T3[i, j]
                    f_loc[k, j] += w * s

        if IS_BBAR:
            for k in range(NN):
                for j in range(3):
                    f_loc[k, j] += wtr_sum * dNdx0[e, j, k]

        if IS_REDUCED:
            for a in range(4):
                inv = wtype(1.0) / gamma_norm[e, a]
                for j in range(3):
                    qm = zero
                    for k in range(NN):
                        qm += gamma[e, a, k] * qd[k, j]
                    qm *= inv * k_hg[e]
                    for k in range(NN):
                        f_loc[k, j] += gamma[e, a, k] * qm

        for a in range(NN):
            for i in range(3):
                s = zero
                for j in range(3):
                    s += f_loc[a, j] * R[i, j]
                f_e[e, a, i] = s

    return kernel


def make_nodal_kernels():
    """Integrator, boundary conditions and reductions -- all in fp64, on device.

    The nodal side is a small fraction of the arithmetic and is bandwidth-bound,
    so double precision here costs almost nothing and is where it actually
    matters: a 70 000-step run accumulates displacement increments that are
    small compared with the coordinates they sit on.
    """
    import warp as wp

    f64 = wp.float64

    @wp.kernel
    def integrate(F_int: wp.array2d(dtype=f64),
                  F_ext: wp.array3d(dtype=f64),
                  factors: wp.array(dtype=f64),
                  mass: wp.array2d(dtype=f64),
                  u: wp.array2d(dtype=f64),
                  v: wp.array2d(dtype=f64),
                  dt: f64, c1: f64, c2: f64, half: int):
        n, i = wp.tid()
        # one unit vector per distinct amplitude curve, combined here: this is
        # what lets each load carry its own curve without re-uploading the whole
        # external force every iteration
        fe = wp.float64(0.0)
        for k in range(factors.shape[0]):
            fe += F_ext[k, n, i] * factors[k]
        a = (fe - F_int[n, i]) * dt / mass[n, i]
        if half == 1:
            # a central-difference restart takes HALF the acceleration; losing
            # that factor injects energy on every kinetic-damping reset
            v[n, i] = wp.float64(0.5) * a
        else:
            v[n, i] = c2 * v[n, i] + c1 * a
        u[n, i] = u[n, i] + v[n, i] * dt

    @wp.kernel
    def zero_fixed(fixed: wp.array2d(dtype=wp.int8), u: wp.array2d(dtype=f64)):
        n, i = wp.tid()
        if fixed[n, i] != wp.int8(0):
            u[n, i] = wp.float64(0.0)

    @wp.kernel
    def sync_velocity(u_prev: wp.array2d(dtype=f64), u: wp.array2d(dtype=f64),
                      v: wp.array2d(dtype=f64), dt: f64):
        # after every position change, including the ones a boundary condition
        # imposed directly
        n, i = wp.tid()
        v[n, i] = (u[n, i] - u_prev[n, i]) / dt

    @wp.kernel
    def apply_prescribed(nodes: wp.array(dtype=wp.int32),
                         dims: wp.array(dtype=wp.int32),
                         values: wp.array(dtype=f64),
                         amp_index: wp.array(dtype=wp.int32),
                         factors: wp.array(dtype=f64),
                         start: wp.array(dtype=f64),
                         approach: f64,
                         u: wp.array2d(dtype=f64)):
        # `start` is where this node stood when the step began and `approach`
        # walks from it to the absolute target over the first stage_ramp
        # iterations. Both are the device twin of Solver.apply_bc: a prescribed
        # level is absolute, so a step that starts somewhere would otherwise
        # drag the node to zero in one iteration. See the docstring there.
        p = wp.tid()
        target = values[p] * factors[amp_index[p]]
        u[nodes[p], dims[p]] = start[p] + (target - start[p]) * approach

    @wp.kernel
    def copy2d(src: wp.array2d(dtype=f64), dst: wp.array2d(dtype=f64)):
        n, i = wp.tid()
        dst[n, i] = src[n, i]

    @wp.kernel
    def zero_velocity(v: wp.array2d(dtype=f64)):
        n, i = wp.tid()
        v[n, i] = wp.float64(0.0)

    @wp.kernel
    def kinetic(mass: wp.array2d(dtype=f64), v: wp.array2d(dtype=f64),
                out: wp.array(dtype=f64)):
        n, i = wp.tid()
        wp.atomic_add(out, 0, wp.float64(0.5) * mass[n, i] * v[n, i] * v[n, i])

    @wp.kernel
    def norms(F_int: wp.array2d(dtype=f64), F_ext: wp.array3d(dtype=f64),
              factors: wp.array(dtype=f64),
              frozen: wp.array(dtype=wp.int8),
              fixed: wp.array2d(dtype=wp.int8),
              out: wp.array(dtype=f64)):
        n, i = wp.tid()
        fi = F_int[n, i]
        fe = wp.float64(0.0)
        for k in range(factors.shape[0]):
            fe += F_ext[k, n, i] * factors[k]
        # a frozen node is out of the free set for the same reason a fixed one
        # is: nothing there is trying to reach equilibrium, so counting it would
        # make the residual report a disequilibrium the solver cannot act on
        if fixed[n, i] == wp.int8(0) and frozen[n] == wp.int8(0):
            r = fe - fi
            wp.atomic_add(out, 0, r * r)
        wp.atomic_add(out, 1, fi * fi)
        wp.atomic_add(out, 2, fe * fe)

    @wp.kernel
    def hold_frozen(frozen: wp.array(dtype=wp.int8),
                    u_prev: wp.array2d(dtype=f64),
                    u: wp.array2d(dtype=f64)):
        n, i = wp.tid()
        if frozen[n] != wp.int8(0):
            u[n, i] = u_prev[n, i]

    return dict(integrate=integrate, zero_fixed=zero_fixed,
                hold_frozen=hold_frozen,
                sync_velocity=sync_velocity,
                apply_prescribed=apply_prescribed, copy2d=copy2d,
                zero_velocity=zero_velocity, kinetic=kinetic, norms=norms)


def make_gather(wtype):
    """Node-parallel assembly: each node sums its own slots. No atomics.

    Accumulates in fp64 even when the element force is fp32 -- that is the whole
    mixed-precision idea, and it costs nothing because this kernel is bandwidth
    bound either way.
    """
    import warp as wp

    @wp.kernel
    def gather(values: wp.array2d(dtype=wtype),
               order: wp.array(dtype=wp.int32),
               offsets: wp.array(dtype=wp.int32),
               out: wp.array2d(dtype=wp.float64)):
        n = wp.tid()
        s0 = wp.float64(0.0)
        s1 = wp.float64(0.0)
        s2 = wp.float64(0.0)
        for p in range(offsets[n], offsets[n + 1]):
            i = order[p]
            s0 += wp.float64(values[i, 0])
            s1 += wp.float64(values[i, 1])
            s2 += wp.float64(values[i, 2])
        out[n, 0] = s0
        out[n, 1] = s1
        out[n, 2] = s2

    return gather


class WarpForce:
    """Same contract as the NumPy and Numba backends, on a Warp device.

    Device buffers are allocated once.  A time step moves the nodal
    displacement in and the assembled force out; everything else stays resident,
    which is what makes the 10^6-element case fit in the plan's memory budget.
    """

    name = "warp"

    def __init__(self, pre, conn, x0_e, material, device: str = "cuda",
                 precision: str = "mixed", rot_tol: float = 1e-12, n_nodes=None):
        import warp as wp

        from .assembly import inverse_connectivity

        if precision not in _PRECISIONS:
            raise ValueError(
                f"unknown precision {precision!r}; expected one of {_PRECISIONS}"
            )
        for hook in ("warp_point_fn", "warp_params"):
            if not hasattr(material, hook):
                raise TypeError(
                    f"material {material.name!r} has no {hook}(); the Warp "
                    "backend needs a device twin of update(). Use backend: "
                    "numpy or numba, or add one (see materials/elastic.py)."
                )

        wp.init()
        self.device = wp.get_device(device)
        self.precision = precision
        self.wtype = wp.float64 if precision == "fp64" else wp.float32
        self.np_t = np.float64 if precision == "fp64" else np.float32
        self.pre = pre
        self.material = material
        self.rot_tol = float(rot_tol)

        # `nne` is nodes per ELEMENT, `nn` is nodes in the MESH -- two very
        # different numbers that both want to be called n_nodes
        nel, ngp, nne = pre.n_elem, pre.n_gp, pre.n_nodes
        # `n_nodes` is the MODEL's node count: a material block, or one element
        # type of a mixed mesh, need not touch the highest-numbered node
        nn = int(n_nodes) if n_nodes is not None else int(conn.max()) + 1
        self.nel, self.ngp, self.nn, self.nne = nel, ngp, nn, nne
        d = self.device
        wt, nt = self.wtype, self.np_t

        def arr(a, dtype=None, npt=None):
            return wp.array(np.ascontiguousarray(a, dtype=npt or nt),
                            dtype=dtype or wt, device=d)

        self.conn = arr(conn, wp.int32, np.int32)
        self.x0_e = arr(x0_e)
        self.dNdx = arr(pre.dNdx)
        self.w_detJ = arr(pre.w_detJ)

        # unused-by-this-formulation buffers are one element, not one per
        # element: dNdx0 is 96 B/element in fp32 and only b-bar reads it
        needs_h = pre.formulation == "reduced"
        # dNdx0 is read every step now -- the rotation comes from the
        # deformation gradient at the centre, not just b-bar
        self.dNdx0 = arr(pre.dNdx0)
        self.gamma = arr(pre.gamma if needs_h else np.zeros((1, 4, nne)))
        self.gamma_norm = arr(pre.gamma_norm if needs_h else np.ones((1, 4)))
        self.k_hg = arr(pre.k_hg if needs_h else np.zeros(1))

        k = getattr(material, "k_band", 1.0)
        kb = (np.asarray(k, dtype=nt) if np.ndim(k)
              else np.full(nel * ngp, float(k), dtype=nt))
        self.kband = arr(kb)

        prm, tab = material.warp_params(nt)
        self.prm = arr(prm)
        self.tab = arr(tab)

        self.d_u = wp.zeros((nn, 3), dtype=wp.float64, device=d)
        # identity rotation, stored (w, x, y, z): a zeroed quaternion is not a
        # rotation at all, and the resident path -- unlike the host-array
        # contract -- never gets one pushed in before the first step
        q0 = np.zeros((nel, 4), dtype=nt)
        q0[:, 0] = 1.0
        self.d_quat = wp.array(q0, dtype=wt, device=d)
        self.d_state = wp.zeros((nel * ngp, max(material.n_state, 1)),
                                dtype=wt, device=d)
        self.d_f_e = wp.zeros((nel, nne, 3), dtype=wt, device=d)
        self.d_sig = wp.zeros((nel, ngp, 6), dtype=wt, device=d)
        self.d_F = wp.zeros((nn, 3), dtype=wp.float64, device=d)

        order, offsets = inverse_connectivity(conn, nn)
        self.order = arr(order, wp.int32, np.int32)
        self.offsets = arr(offsets, wp.int32, np.int32)

        self.kernel = make_kernel(material.warp_point_fn(), pre.formulation,
                                  ngp, wt, n_nodes=nne)
        self.gather = make_gather(wt)
        self.gather_add = make_gather_add(wt)
        self.nodal = make_nodal_kernels()
        self._sig_host = np.zeros((nel, ngp, 6))

        # --- device-resident nodal state (see RESIDENCY in the module doc) ---
        self.d_v = wp.zeros((nn, 3), dtype=wp.float64, device=d)
        self.d_u_prev = wp.zeros((nn, 3), dtype=wp.float64, device=d)
        self.d_mass = wp.ones((nn, 3), dtype=wp.float64, device=d)
        # replaced per step by set_nodal: one unit vector per amplitude curve
        self.d_F_ext_stack = wp.zeros((1, nn, 3), dtype=wp.float64, device=d)
        self.d_factors = wp.zeros(1, dtype=wp.float64, device=d)
        self._amps = [None]
        self._pres_amps = []
        self.d_fixed = wp.zeros((nn, 3), dtype=wp.int8, device=d)
        self.d_scalars = wp.zeros(3, dtype=wp.float64, device=d)
        self._pres = None
        self.d_frozen = wp.zeros(nn, dtype=wp.int8, device=d)
        self._any_frozen = False
        self.trusses = []          # embedded reinforcement blocks, if any
        self.attached = []         # other element blocks sharing this nodal state

    def refresh_weights(self, pre):
        """Re-upload the quadrature weights after elements have been eroded.

        Erosion zeroes `w_detJ` and `k_hg` on the host copy; the device holds
        its own.  Only these two arrays change -- `dNdx`, `dNdx0`, `gamma` and
        the connectivity are untouched, because an eroded element keeps its
        geometry and its node ids and merely stops contributing.  At 10^6
        elements this moves ~32 MB in fp32, and only on an iteration where
        something actually died.
        """
        self.w_detJ.assign(np.ascontiguousarray(pre.w_detJ, dtype=self.np_t))
        if pre.formulation == "reduced" and pre.k_hg is not None:
            self.k_hg.assign(np.ascontiguousarray(pre.k_hg, dtype=self.np_t))

    def set_frozen(self, frozen):
        """Nodes to hold at their current displacement (see Solver._after_erosion)."""
        self.d_frozen.assign(np.ascontiguousarray(frozen, dtype=np.int8))
        self._any_frozen = bool(np.any(frozen))

    def attach_truss(self, emb):
        """Add an embedded reinforcement block to the device-resident loop."""
        t = WarpTruss(emb, self.wtype, self.np_t, self.device, self.nn)
        self.trusses.append(t)
        return t

    def attach_block(self, other: "WarpForce"):
        """Add another element block (a second material, or a second element
        type of a mixed mesh) to this evaluator's resident loop.

        This object stays the owner of the nodal state -- displacement,
        velocity, mass, boundary data, assembled force.  The attached block
        keeps its own elements, kernel, rotations and material state, reads
        the owner's displacement and ADDS its element forces into the owner's
        force through the same node-parallel gather the reinforcement uses,
        so the sum stays bit-reproducible.  Its own nodal buffers are unused.
        """
        if other.device != self.device or other.wtype != self.wtype:
            raise ValueError("an attached block must share the owner's device and precision")
        if other.nn != self.nn:
            raise ValueError("an attached block must be built over the same node set")
        self.attached.append(other)

    def _launch_kernel(self, d_u, dt, rot_iters, want_stress):
        import warp as wp

        wp.launch(
            self.kernel, dim=self.nel, device=self.device,
            inputs=[self.conn, self.x0_e, d_u, self.dNdx,
                    self.w_detJ, self.dNdx0, self.gamma, self.gamma_norm,
                    self.k_hg, self.d_quat, self.d_state, self.kband,
                    self.prm, self.tab, self.np_t(dt), int(rot_iters),
                    self.np_t(self.rot_tol), self.d_f_e, self.d_sig,
                    1 if want_stress else 0],
        )

    # ------------------------------------------------------------------ #
    #  device-resident time loop
    # ------------------------------------------------------------------ #

    def set_nodal(self, mass, fixed, load_groups, prescribed, starts=None):
        """Upload the nodal boundary data for one analysis step.

        Loads arrive already grouped by amplitude curve, and each group keeps
        its own unit vector on the device.  That grouping is what lets per-BC
        amplitudes work without breaking residency: the alternative is to
        re-upload the whole external force whenever a factor changes, which at
        10^6 nodes is 24 MB every iteration -- more traffic than the element
        kernel it feeds.  Real cases have one or two distinct curves, so this is
        one or two extra vectors, uploaded once per step.
        """
        import warp as wp

        d = self.device
        self.d_mass.assign(np.ascontiguousarray(mass, dtype=np.float64))
        self.d_fixed.assign(np.ascontiguousarray(fixed, dtype=np.int8))

        if load_groups:
            self._amps = [a for a, _F in load_groups]
            stack = np.stack([np.ascontiguousarray(F, dtype=np.float64)
                              for _a, F in load_groups])
        else:
            self._amps = [None]
            stack = np.zeros((1, self.nn, 3))
        self.d_F_ext_stack = wp.array(stack, dtype=wp.float64, device=d)
        self.d_factors = wp.zeros(len(self._amps), dtype=wp.float64, device=d)

        nodes, dims, vals, aidx, starts_flat = [], [], [], [], []
        self._pres_amps = []
        for j, (idx, dim, val, a) in enumerate(prescribed):
            k = len(self._pres_amps)
            self._pres_amps.append(a)
            nodes.extend(np.asarray(idx).tolist())
            dims.extend([dim] * len(idx))
            vals.extend([val] * len(idx))
            aidx.extend([k] * len(idx))
            u0 = (np.zeros(len(idx)) if starts is None
                  else np.broadcast_to(np.asarray(starts[j], dtype=float), (len(idx),)))
            starts_flat.extend(u0.tolist())
        if nodes:
            self._pres = (
                wp.array(np.array(nodes, dtype=np.int32), dtype=wp.int32, device=d),
                wp.array(np.array(dims, dtype=np.int32), dtype=wp.int32, device=d),
                wp.array(np.array(vals, dtype=np.float64), dtype=wp.float64, device=d),
                wp.array(np.array(aidx, dtype=np.int32), dtype=wp.int32, device=d),
                wp.zeros(len(self._pres_amps), dtype=wp.float64, device=d),
                wp.array(np.array(starts_flat, dtype=np.float64),
                         dtype=wp.float64, device=d),
                len(nodes),
            )
        else:
            self._pres = None

    def set_u(self, u, v=None):
        self.d_u.assign(np.ascontiguousarray(u, dtype=np.float64))
        if v is not None:
            self.d_v.assign(np.ascontiguousarray(v, dtype=np.float64))

    def get_u(self):
        return self.d_u.numpy()

    def get_v(self):
        return self.d_v.numpy()

    def get_quat(self):
        """Return the current co-rotational frames for output/checkpointing."""
        return self.d_quat.numpy()

    def force_resident(self, dt, rot_iters=2, want_stress=False):
        """Force + assembly with nothing crossing the bus."""
        import warp as wp

        self._launch_kernel(self.d_u, dt, rot_iters, want_stress)
        wp.launch(self.gather, dim=self.nn, device=self.device,
                  inputs=[self.d_f_e.reshape((self.nel * self.nne, 3)),
                          self.order, self.offsets, self.d_F])
        # the other blocks and the reinforcement add into the same F_int, on
        # the device, after it
        for o in self.attached:
            o._launch_kernel(self.d_u, dt, rot_iters, want_stress)
            wp.launch(o.gather_add, dim=self.nn, device=self.device,
                      inputs=[o.d_f_e.reshape((o.nel * o.nne, 3)),
                              o.order, o.offsets, self.d_F])
        for t in self.trusses:
            t.add_force(self.d_u, self.d_F, dt, want_stress=want_stress)

    def _set_factors(self, progress):
        """Evaluate every amplitude at this progress and push the scalars."""
        self.d_factors.assign(np.array(
            [1.0 if a is None else a(progress) for a in self._amps],
            dtype=np.float64))
        if self._pres is not None:
            self._pres[4].assign(np.array([a(progress) for a in self._pres_amps],
                                          dtype=np.float64))

    def integrate(self, factor, dt, c1, c2, half, approach=1.0):
        """Leapfrog update plus boundary conditions, on device."""
        import warp as wp

        n = self.nodal
        dev = self.device
        wp.launch(n["copy2d"], dim=(self.nn, 3), device=dev,
                  inputs=[self.d_u, self.d_u_prev])
        self._set_factors(factor)
        wp.launch(n["integrate"], dim=(self.nn, 3), device=dev,
                  inputs=[self.d_F, self.d_F_ext_stack, self.d_factors,
                          self.d_mass, self.d_u,
                          self.d_v, np.float64(dt),
                          np.float64(c1), np.float64(c2), int(half)])
        # order matters and it is the reverse of what reads naturally: every
        # constrained dof is zeroed FIRST, then the prescribed ones are written.
        # Prescribing before zeroing wipes the imposed displacement -- the
        # symptom is a run where nothing moves at all.
        if self._any_frozen:
            # BEFORE the boundary conditions, so a prescribed node can still be
            # driven even if erosion left it unsupported. The solver also
            # refuses to freeze a driven node; this ordering makes the two
            # paths agree by construction rather than by agreement.
            wp.launch(n["hold_frozen"], dim=(self.nn, 3), device=dev,
                      inputs=[self.d_frozen, self.d_u_prev, self.d_u])
        wp.launch(n["zero_fixed"], dim=(self.nn, 3), device=dev,
                  inputs=[self.d_fixed, self.d_u])
        if self._pres is not None:
            nodes, dims, vals, aidx, pfac, pstart, npres = self._pres
            wp.launch(n["apply_prescribed"], dim=npres, device=dev,
                      inputs=[nodes, dims, vals, aidx, pfac, pstart,
                              np.float64(approach), self.d_u])
        wp.launch(n["sync_velocity"], dim=(self.nn, 3), device=dev,
                  inputs=[self.d_u_prev, self.d_u, self.d_v, np.float64(dt)])

    def kinetic_energy(self):
        import warp as wp

        self.d_scalars.zero_()
        wp.launch(self.nodal["kinetic"], dim=(self.nn, 3), device=self.device,
                  inputs=[self.d_mass, self.d_v, self.d_scalars])
        return float(self.d_scalars.numpy()[0])

    def zero_velocity(self):
        import warp as wp

        wp.launch(self.nodal["zero_velocity"], dim=(self.nn, 3),
                  device=self.device, inputs=[self.d_v])

    def frozen_mask(self):
        return self.d_frozen.numpy().astype(bool)

    def residual(self, factor):
        """(||F_unbalanced on free dofs||, ||F_int||, ||F_ext||)."""
        import warp as wp

        self.d_scalars.zero_()
        self._set_factors(factor)
        wp.launch(self.nodal["norms"], dim=(self.nn, 3), device=self.device,
                  inputs=[self.d_F, self.d_F_ext_stack, self.d_factors,
                          self.d_frozen, self.d_fixed, self.d_scalars])
        return np.sqrt(self.d_scalars.numpy())

    def get_state(self):
        return self.d_state.numpy()

    def set_state(self, state, quat):
        if state.size:
            self.d_state.assign(np.ascontiguousarray(state, dtype=self.np_t))
        self.d_quat.assign(np.ascontiguousarray(quat, dtype=self.np_t))

    def get_stress(self):
        self._sig_host[:] = self.d_sig.numpy()
        return self._sig_host

    def get_F_int(self):
        return self.d_F.numpy()

    # ------------------------------------------------------------------ #

    def bytes_resident(self) -> int:
        """Device memory held by the persistent buffers."""
        import warp as wp

        total = 0
        for v in vars(self).values():
            if isinstance(v, wp.array):
                total += v.size * wp.types.type_size_in_bytes(v.dtype)
        return total

    def __call__(self, u, quat, state, dt, rot_iters=2, want_stress=False):
        """Host-array contract, matching the NumPy and Numba backends.

        Copies in and out on every call, so it is for the equivalence tests and
        small runs -- the solver uses the resident path above.
        """
        import warp as wp

        self.d_u.assign(np.ascontiguousarray(u, dtype=np.float64))
        self.d_quat.assign(np.ascontiguousarray(quat, dtype=self.np_t))
        if state.size:
            self.d_state.assign(np.ascontiguousarray(state, dtype=self.np_t))

        self._launch_kernel(self.d_u, dt, rot_iters, want_stress)
        quat[:] = self.d_quat.numpy()
        if state.size:
            state[:] = self.d_state.numpy()
        f_e = self.d_f_e.numpy().astype(np.float64)
        if want_stress:
            self._sig_host[:] = self.d_sig.numpy()
        return f_e, (self._sig_host if want_stress else None)

    def assemble(self, f_e):
        import warp as wp

        wp.launch(self.gather, dim=self.nn, device=self.device,
                  inputs=[self.d_f_e.reshape((self.nel * self.nne, 3)),
                          self.order, self.offsets, self.d_F])
        return self.d_F.numpy()


# --------------------------------------------------------------------------- #
#  Embedded reinforcement on the device
# --------------------------------------------------------------------------- #


def make_gather_add(wtype):
    """Node-parallel assembly that ADDS into an existing force vector.

    Reinforcement is a second element block over the same nodes, so its force
    has to join the concrete's.  Adding through a second inverse-connectivity
    gather keeps the property that makes the first one worth having: each node
    is touched by exactly one thread, in a fixed order, so the sum is
    bit-reproducible.  An atomic scatter would be shorter and would quietly make
    every run differ in the last bits.
    """
    import warp as wp

    @wp.kernel
    def gather_add(values: wp.array2d(dtype=wtype),
                   order: wp.array(dtype=wp.int32),
                   offsets: wp.array(dtype=wp.int32),
                   out: wp.array2d(dtype=wp.float64)):
        n = wp.tid()
        s0 = wp.float64(0.0)
        s1 = wp.float64(0.0)
        s2 = wp.float64(0.0)
        for p in range(offsets[n], offsets[n + 1]):
            i = order[p]
            s0 += wp.float64(values[i, 0])
            s1 += wp.float64(values[i, 1])
            s2 += wp.float64(values[i, 2])
        out[n, 0] = out[n, 0] + s0
        out[n, 1] = out[n, 1] + s1
        out[n, 2] = out[n, 2] + s2

    return gather_add


def make_truss_kernel(point_fn, wtype, n_host: int):
    """One thread per bar: interpolate, stretch, uniaxial law, distribute.

    The bar has no rotation to extract -- its current axis IS its rotation --
    so this is much shorter than the continuum kernel: two weighted gathers, a
    length, one scalar constitutive call, and the force written back with the
    same weights that read the displacement.  That symmetry is what makes the
    embedding conservative; it is a Galerkin projection, not an interpolation
    trick, and it is why the assembled force self-equilibrates exactly.
    """
    import warp as wp

    NH = wp.constant(int(n_host))
    vec6 = wp.types.vector(length=6, dtype=wtype)

    @wp.kernel
    def kernel(conn: wp.array2d(dtype=wp.int32),
               weights: wp.array3d(dtype=wtype),
               x0_e: wp.array3d(dtype=wtype),
               L0: wp.array(dtype=wtype),
               area: wp.array(dtype=wtype),
               u: wp.array2d(dtype=wp.float64),
               state: wp.array2d(dtype=wtype),
               prm: wp.array(dtype=wtype),
               tab: wp.array(dtype=wtype),
               dt: wtype,
               host_deduction: wtype,
               prestrain: wp.array(dtype=wtype),
               f_e: wp.array3d(dtype=wtype),
               sig_out: wp.array2d(dtype=wtype),
               want_stress: int):
        b = wp.tid()
        zero = wtype(0.0)

        # --- bar-end positions, interpolated from the host nodes ------------
        x0 = wp.vector(zero, zero, zero)
        x1 = wp.vector(zero, zero, zero)
        for a in range(NH):
            w0 = weights[b, 0, a]
            w1 = weights[b, 1, a]
            n0 = conn[b, a]
            n1 = conn[b, a + NH]
            for i in range(3):
                x0[i] += w0 * wtype(u[n0, i])
                x1[i] += w1 * wtype(u[n1, i])
        for i in range(3):
            x0[i] += x0_e[b, 0, i]
            x1[i] += x0_e[b, 1, i]

        d = x1 - x0
        L = wp.sqrt(wp.dot(d, d))
        n = d / wp.max(L, wtype(1.0e-30))

        e_mech = (L - L0[b]) / L0[b]
        e = vec6()
        # total strain to the material, mechanical strain to the deduction
        # -- see PRESTRESS in elements/truss.py
        e[0] = e_mech + prestrain[b]
        sig = point_fn(prm, tab, e, state, b, dt, wtype(1.0))
        if want_stress != 0:
            sig_out[b, 0] = sig[0]
        N = (sig[0] - host_deduction * e_mech) * area[b]

        # --- distribute with the same weights that gathered -----------------
        for a in range(NH):
            w0 = weights[b, 0, a]
            w1 = weights[b, 1, a]
            for i in range(3):
                f_e[b, a, i] = -w0 * N * n[i]
                f_e[b, a + NH, i] = w1 * N * n[i]

    return kernel


class WarpTruss:
    """Device twin of `elements.truss.EmbeddedTruss`.

    Exists so that reinforcement does not break residency.  Evaluating the bars
    on the host would mean pulling `u` down and pushing a force back every
    single step -- at 10^6 elements that is the transfer cost the whole
    device-resident design was built to avoid, paid for a few thousand bars.
    """

    def __init__(self, emb, wtype, np_t, device, nn):
        import warp as wp

        from .assembly import inverse_connectivity

        mat = emb.material
        for hook in ("warp_point_fn", "warp_params"):
            if not hasattr(mat, hook):
                raise TypeError(
                    f"reinforcement material {mat.name!r} has no {hook}(); use "
                    "backend numpy or numba for this material"
                )
        d = device
        self.device = d
        self.n_elem = emb.n_elem
        self.nh = emb.n_host_nodes
        self.host_deduction = float(emb.pre.host_deduction)
        # per element: a tendon with friction carries a different
        # stress in every segment (elements/truss.tendon_profile)
        self._prestrain = np.broadcast_to(
            np.asarray(emb.pre.prestrain, dtype=float), (self.n_elem,))
        self.nn = nn

        def arr(a, dtype=None, npt=None):
            return wp.array(np.ascontiguousarray(a, dtype=npt or np_t),
                            dtype=dtype or wtype, device=d)

        self.prestrain = arr(self._prestrain)
        self.conn = arr(emb.conn, wp.int32, np.int32)
        self.weights = arr(emb.weights)
        self.x0_e = arr(emb.x0_e)
        self.L0 = arr(emb.pre.L0)
        self.area = arr(emb.pre.area)
        prm, tab = mat.warp_params(np_t)
        self.prm = arr(prm)
        self.tab = arr(tab)
        self.d_state = wp.zeros((self.n_elem, max(mat.n_state, 1)),
                                dtype=wtype, device=d)
        self.d_f_e = wp.zeros((self.n_elem, 2 * self.nh, 3), dtype=wtype,
                              device=d)
        self.d_sig = wp.zeros((self.n_elem, 1), dtype=wtype, device=d)

        order, offsets = inverse_connectivity(emb.conn, nn)
        self.order = arr(order, wp.int32, np.int32)
        self.offsets = arr(offsets, wp.int32, np.int32)

        self.kernel = make_truss_kernel(mat.warp_point_fn(), wtype, self.nh)
        self.gather_add = make_gather_add(wtype)
        self.np_t = np_t

    def add_force(self, d_u, d_F, dt, want_stress=False):
        import warp as wp

        wp.launch(self.kernel, dim=self.n_elem, device=self.device,
                  inputs=[self.conn, self.weights, self.x0_e, self.L0,
                          self.area, d_u, self.d_state, self.prm, self.tab,
                          self.np_t(dt), self.np_t(self.host_deduction),
                          self.prestrain,
                          self.d_f_e, self.d_sig,
                          1 if want_stress else 0])
        wp.launch(self.gather_add, dim=self.nn, device=self.device,
                  inputs=[self.d_f_e.reshape((self.n_elem * 2 * self.nh, 3)),
                          self.order, self.offsets, d_F])

    def bytes_resident(self):
        return sum(a.size * a.dtype._length_ * np.dtype(self.np_t).itemsize
                   if hasattr(a.dtype, "_length_") else a.size * 4
                   for a in (self.conn, self.weights, self.x0_e, self.d_f_e,
                             self.d_state))
