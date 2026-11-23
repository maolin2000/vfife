"""Warp backend for shell4: the element kernel and a device-resident
relaxation for the six-dof structural system.

THE LAYOUT THAT MAKES THE SOLID BACKEND'S NODAL SIDE REUSABLE
    The structural solvers keep one flat vector of 6 dofs per node, node-
    major: dof = 6 n + c. Viewed as an array of shape (2 nn, 3), row 2n is
    node n's translation and row 2n+1 its rotation, and every nodal kernel
    the solid backend already has -- integrate, zero_fixed, apply_prescribed,
    kinetic, norms -- works on it unchanged. So they are reused, not copied:
    `make_nodal_kernels()` from warp_backend, launched over (2 nn, 3).

THE ELEMENT KERNEL
    One thread per element, the same arithmetic as `kernels/numba_shell4`:
    current frame, each node's triad relative to it as a rotation vector,
    local deformation, K_local @ q_local, end actions rotated back and
    scattered with atomic adds. The triads are the points' orientations,
    kept on the device as quaternions and advanced multiplicatively by the
    step's spin. Geometry is done in fp64 whatever the precision
    setting; `precision: mixed` stores the 24x24 local stiffness in fp32 and
    accumulates the matvec in fp64, so what fp32 costs is the entries of K
    (1e-7 relative), not the cancellation in the coordinates that hurt the
    solid backend at mixed precision.

WHAT IS NOT ON THE DEVICE
    Ground motion, velocity- and acceleration-prescribed dofs, and the
    dynamic mode: the resident loop is the staged relaxation and the
    continuous relaxation as one stage. Anything else raises at setup and
    names the numba backend.
"""

import numpy as np

try:
    import warp as wp
except ImportError:                                   # pragma: no cover
    wp = None


def available() -> bool:
    return wp is not None


def make_shell4_kernels(wtype):
    """The shell4 internal force, in two launches.

    ONE THREAD PER ELEMENT WAS THE FLOOR OF THE WHOLE GPU PATH. The single
    kernel this replaces read K[e, 24, 24] and Bres[e, 8, 24] -- 768 values
    -- and accumulated them in thirty-two dependent chains, with one thread
    per element. At 256 elements that is 256 threads on a 36-SM card and
    nothing whatever to hide 768 dependent global loads behind, and it
    measured 128 us; at 1024 and at 4096 elements it measured 130 us, the
    same time for sixteen times the work. Changing the precision did not
    move it (121.6 us in fp64 against 128.0 in mixed), which is what ruled
    out the arithmetic and pointed at the memory
    (benchmarks/warp_loop_cost.py --per-kernel). At 16384 elements, where
    there are finally enough threads to hide the latency, precision DOES
    matter -- 780 us in fp64 against 459 in mixed, near enough the factor of
    two in the bytes of K -- which is the same diagnosis seen from the other
    end.

    So the force is three launches, each with as many threads as its work
    has independent pieces. `frame` is one thread per element: the element
    axes, the two degeneracy checks and the translational part of the local
    deformation, all of which need the four corners at once. `node` is one
    thread per element NODE: the triad relative to that frame, as a rotation
    vector -- four independent chains of quat_to_matrix, acos, sin and a
    branchy log map, which as one thread per element measured 51 us flat
    from 256 to 4096 elements, the same disease one level down. `apply` is
    one thread per element BLOCK, eight per element: the matrix products,
    which are the 768 loads and are trivially parallel, 96 loads per thread
    instead of 768.

    The arithmetic is untouched: every row of K . lq and of Bres . lq is
    still summed over j in the same order by a single thread, so the force
    is bit-for-bit what the one-kernel version produced. Only the scatter's
    atomics reorder, and they were never ordered.
    """
    import warp as wp

    f64 = wp.float64
    vec24 = wp.types.vector(length=24, dtype=f64)

    @wp.kernel
    def frame_of(nodes: wp.array2d(dtype=f64),
                 conn: wp.array2d(dtype=wp.int32),
                 u: wp.array2d(dtype=f64),
                 local0: wp.array3d(dtype=f64),
                 warp_tol: f64,
                 lq_out: wp.array2d(dtype=f64),
                 frame: wp.array(dtype=wp.mat33d),
                 flag: wp.array(dtype=wp.int32),
                 scalars: wp.array(dtype=f64)):
        e = wp.tid()
        # current nodal positions
        n0 = conn[e, 0]; n1 = conn[e, 1]; n2 = conn[e, 2]; n3 = conn[e, 3]
        x0 = wp.vec3d(nodes[n0, 0] + u[2 * n0, 0], nodes[n0, 1] + u[2 * n0, 1], nodes[n0, 2] + u[2 * n0, 2])
        x1 = wp.vec3d(nodes[n1, 0] + u[2 * n1, 0], nodes[n1, 1] + u[2 * n1, 1], nodes[n1, 2] + u[2 * n1, 2])
        x2 = wp.vec3d(nodes[n2, 0] + u[2 * n2, 0], nodes[n2, 1] + u[2 * n2, 1], nodes[n2, 2] + u[2 * n2, 2])
        x3 = wp.vec3d(nodes[n3, 0] + u[2 * n3, 0], nodes[n3, 1] + u[2 * n3, 1], nodes[n3, 2] + u[2 * n3, 2])
        c = (x0 + x1 + x2 + x3) * f64(0.25)
        ex = (x1 - x0) + (x2 - x3)
        nrm = wp.cross(x2 - x0, x3 - x1)
        lex = wp.length(ex); lnr = wp.length(nrm)
        if lex <= f64(1e-14) or lnr <= f64(1e-14):
            flag[e] = 2
            wp.atomic_add(scalars, 5, f64(1.0))
            return
        ex = ex / lex
        ez = nrm / lnr
        ey = wp.normalize(wp.cross(ez, ex))
        R = wp.mat33d(ex[0], ex[1], ex[2], ey[0], ey[1], ey[2], ez[0], ez[1], ez[2])   # rows = axes
        l0 = R * (x0 - c); l1 = R * (x1 - c); l2 = R * (x2 - c); l3 = R * (x3 - c)
        span = wp.max(wp.max(wp.max(l0[0], l1[0]), wp.max(l2[0], l3[0])) - wp.min(wp.min(l0[0], l1[0]), wp.min(l2[0], l3[0])),
                      wp.max(wp.max(l0[1], l1[1]), wp.max(l2[1], l3[1])) - wp.min(wp.min(l0[1], l1[1]), wp.min(l2[1], l3[1])))
        span = wp.max(span, f64(1e-30))
        dz = wp.max(wp.max(l0[2], l1[2]), wp.max(l2[2], l3[2])) - wp.min(wp.min(l0[2], l1[2]), wp.min(l2[2], l3[2]))
        if dz > warp_tol * span:
            flag[e] = 1
            wp.atomic_add(scalars, 5, f64(1.0))
            return
        # the translational part of the local deformation, z projected away
        for a in range(4):
            la = l0
            if a == 1:
                la = l1
            elif a == 2:
                la = l2
            elif a == 3:
                la = l3
            lq_out[e, 6 * a + 0] = la[0] - local0[e, a, 0]
            lq_out[e, 6 * a + 1] = la[1] - local0[e, a, 1]
            lq_out[e, 6 * a + 2] = f64(0.0) - local0[e, a, 2]
        frame[e] = R
        flag[e] = 0

    @wp.kernel
    def node_rotation(conn: wp.array2d(dtype=wp.int32),
                      quat: wp.array(dtype=wp.quatd),
                      frame0: wp.array(dtype=wp.mat33d),
                      frame: wp.array(dtype=wp.mat33d),
                      flag: wp.array(dtype=wp.int32),
                      lq_out: wp.array2d(dtype=f64)):
        e, a = wp.tid()
        if flag[e] != 0:
            return
        n = conn[e, a]
        # the node's triad relative to the face frame, as a rotation vector
        M = frame[e] * wp.quat_to_matrix(quat[n]) * wp.transpose(frame0[e])
        cang = f64(0.5) * (wp.trace(M) - f64(1.0))
        cang = wp.min(wp.max(cang, f64(-1.0)), f64(1.0))
        angle = wp.acos(cang)
        axial = wp.vec3d(f64(0.5) * (M[2, 1] - M[1, 2]), f64(0.5) * (M[0, 2] - M[2, 0]), f64(0.5) * (M[1, 0] - M[0, 1]))
        rl = wp.vec3d(f64(0.0), f64(0.0), f64(0.0))
        if angle < f64(1e-6):
            rl = axial * (f64(1.0) + angle * angle / f64(6.0))
        elif f64(3.141592653589793) - angle < f64(1e-7):
            best = int(0); bn = f64(-1.0)
            for j in range(3):
                cn = f64(0.0)
                for i in range(3):
                    v = M[i, j]
                    if i == j:
                        v = v + f64(1.0)
                    cn += v * v
                if cn > bn:
                    bn = cn; best = j
            ax = wp.vec3d(M[0, best], M[1, best], M[2, best])
            ax[best] = ax[best] + f64(1.0)
            rl = wp.normalize(ax) * angle
        else:
            rl = axial * (angle / wp.sin(angle))
        lq_out[e, 6 * a + 3] = rl[0]
        lq_out[e, 6 * a + 4] = rl[1]
        lq_out[e, 6 * a + 5] = rl[2]

    @wp.kernel
    def apply(conn: wp.array2d(dtype=wp.int32),
              lq_in: wp.array2d(dtype=f64),
              frame: wp.array(dtype=wp.mat33d),
              K: wp.array3d(dtype=wtype),
              Bres: wp.array3d(dtype=f64),
              flag: wp.array(dtype=wp.int32),
              F: wp.array2d(dtype=f64),
              result: wp.array2d(dtype=f64)):
        e, b = wp.tid()
        if flag[e] != 0:                       # refused faces contribute nothing
            return
        i0 = 3 * b
        s0 = f64(0.0); s1 = f64(0.0); s2 = f64(0.0); rb = f64(0.0)
        for j in range(24):
            q = lq_in[e, j]
            s0 += f64(K[e, i0, j]) * q
            s1 += f64(K[e, i0 + 1, j]) * q
            s2 += f64(K[e, i0 + 2, j]) * q
            rb += Bres[e, b, j] * q
        result[e, b] = rb
        # back to global and scattered: block b is node b//2, translation or rotation
        R = frame[e]
        g = wp.transpose(R) * wp.vec3d(s0, s1, s2)
        row = 2 * conn[e, b // 2] + (b % 2)
        wp.atomic_add(F, row, 0, g[0])
        wp.atomic_add(F, row, 1, g[1])
        wp.atomic_add(F, row, 2, g[2])

    return frame_of, node_rotation, apply


#: The three-level fan-in of the convergence measures.
#: MEASURE_GROUPS threads each stride over the dofs; MEASURE_MID threads per
#: quantity then fold those; five threads fold the rest. All three partitions
#: are fixed, so the summation order is fixed and the reduction is
#: bit-reproducible.
#:
#: TWO LEVELS WAS NOT ENOUGH, and the reason is worth writing down because it
#: is the same mistake in a different place. The first version summed 1024
#: partials with FIVE threads, one per quantity: 1024 strided global loads on
#: a serial dependency chain, nothing to hide the latency behind, and it
#: measured a flat 38 us -- more than every kernel of the iteration except
#: the element force, and enough to eat the whole gain at 256 and 1024
#: elements. A reduction is only as good as its narrowest level.
MEASURE_GROUPS = 1024
MEASURE_MID = 32
#: dofs per group of the first level. 1024 groups is 49 dofs each at 16384
#: elements and 1024 threads on a card that wants tens of thousands: the
#: first level measured 61.7 us there against 3.1 us at 256 elements. The
#: group count now grows with the problem, which keeps the partition fixed
#: for a given mesh -- and so keeps the reduction reproducible -- while
#: giving the card something to fill itself with.
MEASURE_CHUNK = 16
MEASURE_GROUPS_MAX = 65536


def measure_groups(total: int) -> int:
    """First-level groups for `total` degrees of freedom: a multiple of
    MEASURE_MID, at least MEASURE_GROUPS, at most MEASURE_GROUPS_MAX."""
    want = -(-total // (MEASURE_CHUNK * MEASURE_MID)) * MEASURE_MID
    return int(min(max(want, MEASURE_GROUPS), MEASURE_GROUPS_MAX))


def make_zero_kernel():
    """F and the element kernel's slot of the scalar buffer, in ONE launch.

    `d_F.zero_()` and `d_scalars.zero_()` measured 16 to 21 us together on an
    RTX 5060 Ti and did not move with the mesh size: two API round trips, one
    for a few hundred kilobytes and one for six doubles. They cannot be
    folded into a kernel that also scatters into F -- there is no ordering
    between the threads of one launch -- and they cannot move to the end of
    the previous iteration, because `measures` reads F after the integrator.
    One kernel is what is left, and it halves the cost.
    """
    import warp as wp

    f64 = wp.float64

    @wp.kernel
    def zero_force(F: wp.array2d(dtype=f64), scalars: wp.array(dtype=f64)):
        n, i = wp.tid()
        F[n, i] = f64(0.0)
        if n == 0 and i == 0:
            scalars[5] = f64(0.0)

    return zero_force


def make_measures_kernels():
    """Everything the convergence check needs: [kinetic energy,
    |F_ext - F_int|^2 on the free dofs, |F_int|^2 over all dofs, |F_ext|^2 on
    the free dofs, |u|^2 on the free dofs] -- the five numbers
    BeamSolver._convergence_measures is made of. Slot 5 is the element
    kernel's: the number of faces it refused this iteration, so a folded
    element surfaces in the same readback instead of silently contributing
    nothing until the level ends.

    A FAN-IN, BECAUSE FIVE ATOMICS WERE THE MOST EXPENSIVE THING IN THE LOOP.
    The original gave every dof its own `wp.atomic_add` onto the same five
    addresses: at 16384 elements that is fifty thousand threads contending
    for five cache lines, a quarter of a million serialised atomics, and it
    measured 555 us an iteration -- MORE than the element force and the
    integrator together, and the largest single item in the profile
    (benchmarks/warp_loop_cost.py --per-kernel).

    The partials are stored quantity-major, (5, MEASURE_GROUPS), so the
    level above reads them contiguously; that layout is the difference
    between a fold that costs a launch and one that costs 38 us.

    It also makes the reduction DETERMINISTIC, where the atomics gave a
    different answer every run -- the same loop differed by 1.3e-2 of a 5e-1
    displacement over 200 iterations, because the kinetic energy that decides
    a damping reset was itself noise at that level. (The element force still
    accumulates into F with atomics, so the whole path is not reproducible
    yet; this is one of the two places.)
    """
    import warp as wp

    f64 = wp.float64

    @wp.kernel
    def partial(F_int: wp.array2d(dtype=f64), F_ext: wp.array3d(dtype=f64),
                factors: wp.array(dtype=f64), mass: wp.array2d(dtype=f64),
                u: wp.array2d(dtype=f64), v: wp.array2d(dtype=f64),
                fixed: wp.array2d(dtype=wp.int8), partials: wp.array2d(dtype=f64)):
        g = wp.tid()
        groups = partials.shape[1]
        total = F_int.shape[0] * 3
        ke = f64(0.0); rr = f64(0.0); ii = f64(0.0); ee = f64(0.0); uu = f64(0.0)
        for t in range(g, total, groups):
            n = t // 3
            i = t - n * 3
            fi = F_int[n, i]
            fe = f64(0.0)
            for k in range(factors.shape[0]):
                fe += F_ext[k, n, i] * factors[k]
            ke += f64(0.5) * mass[n, i] * v[n, i] * v[n, i]
            ii += fi * fi
            if fixed[n, i] == wp.int8(0):
                r = fe - fi
                rr += r * r
                ee += fe * fe
                uu += u[n, i] * u[n, i]
        partials[0, g] = ke
        partials[1, g] = rr
        partials[2, g] = ii
        partials[3, g] = ee
        partials[4, g] = uu

    @wp.kernel
    def mid(partials: wp.array2d(dtype=f64), mids: wp.array2d(dtype=f64)):
        c, j = wp.tid()
        span = partials.shape[1] // mids.shape[1]
        s = f64(0.0)
        for g in range(j * span, (j + 1) * span):
            s += partials[c, g]
        mids[c, j] = s

    @wp.kernel
    def finish(mids: wp.array2d(dtype=f64), out: wp.array(dtype=f64)):
        # five threads, one per quantity; slot 5 is the element kernel's and
        # is left alone
        c = wp.tid()
        s = f64(0.0)
        for j in range(mids.shape[1]):
            s += mids[c, j]
        out[c] = s

    return partial, mid, finish


def make_spin_kernel():
    """The multiplicative orientation update of v0.1 on the device:
    per structural point, quat <- exp(omega dt) quat with omega the
    (already masked) spin of the step, then the rotation rows of u become
    the logarithm. A point with a prescribed rotation component takes that
    component from the value apply_prescribed wrote and rebuilds the
    quaternion from the mixed vector, exactly as BeamSolver._apply_bc does.
    """
    import warp as wp

    f64 = wp.float64

    @wp.func
    def quat_exp(w: wp.vec3d) -> wp.quatd:
        a = wp.length(w)
        if a < f64(1e-9):
            return wp.normalize(wp.quatd(f64(0.5) * w[0], f64(0.5) * w[1], f64(0.5) * w[2], f64(1.0)))
        s = wp.sin(f64(0.5) * a) / a
        return wp.quatd(s * w[0], s * w[1], s * w[2], wp.cos(f64(0.5) * a))

    @wp.func
    def quat_log(q: wp.quatd) -> wp.vec3d:
        x = q[0]; y = q[1]; z = q[2]; w = q[3]
        if w < f64(0.0):
            x = -x; y = -y; z = -z; w = -w
        n = wp.sqrt(x * x + y * y + z * z)
        if n < f64(1e-12):
            return wp.vec3d(f64(2.0) * x / w, f64(2.0) * y / w, f64(2.0) * z / w)
        f = f64(2.0) * wp.atan2(n, w) / n
        return wp.vec3d(f * x, f * y, f * z)

    @wp.kernel
    def spin(v: wp.array2d(dtype=f64), dt: f64, pres_rot: wp.array(dtype=wp.int32),
             quat: wp.array(dtype=wp.quatd), u: wp.array2d(dtype=f64)):
        n = wp.tid()
        r = 2 * n + 1
        w = wp.vec3d(v[r, 0] * dt, v[r, 1] * dt, v[r, 2] * dt)
        q = wp.normalize(quat_exp(w) * quat[n])
        vec = quat_log(q)
        mask = pres_rot[n]
        if mask != 0:
            if (mask & 1) != 0:
                vec[0] = u[r, 0]
            if (mask & 2) != 0:
                vec[1] = u[r, 1]
            if (mask & 4) != 0:
                vec[2] = u[r, 2]
            q = quat_exp(vec)
        quat[n] = q
        u[r, 0] = vec[0]; u[r, 1] = vec[1]; u[r, 2] = vec[2]

    return spin


class WarpShell:
    """The shell4 model resident on a Warp device: element force, nodal
    integrator, boundary conditions and reductions.

    Arrays are (2 nn, 3) fp64 on the nodal side (see the module docstring)
    and per-element on the element side. The host solver hands over the
    reference geometry once, the boundary data once per load level, and
    reads back u, F_int and the resultants when a level converges.
    """

    def __init__(self, nodes, conn, frame0, local0, K, Bres, warp_tol,
                 device="cuda", precision="mixed"):
        import warp as wp
        from .warp_backend import make_nodal_kernels

        wp.init()
        self.device = wp.get_device(device)
        self.precision = precision
        wtype = wp.float64 if precision == "fp64" else wp.float32
        np_t = np.float64 if precision == "fp64" else np.float32
        d = self.device
        self.nn = int(nodes.shape[0]); self.nel = int(conn.shape[0]); self.rows = 2 * self.nn
        self.d_nodes = wp.array(np.ascontiguousarray(nodes, dtype=np.float64), dtype=wp.float64, device=d)
        self.d_conn = wp.array(np.ascontiguousarray(conn, dtype=np.int32), dtype=wp.int32, device=d)
        self.d_frame0 = wp.array(np.ascontiguousarray(frame0, dtype=np.float64), dtype=wp.mat33d, device=d)
        self.d_local0 = wp.array(np.ascontiguousarray(local0, dtype=np.float64), dtype=wp.float64, device=d)
        self.d_K = wp.array(np.ascontiguousarray(K, dtype=np_t), dtype=wtype, device=d)
        self.d_Bres = wp.array(np.ascontiguousarray(Bres, dtype=np.float64), dtype=wp.float64, device=d)
        self.warp_tol = float(warp_tol)
        self.d_u = wp.zeros((self.rows, 3), dtype=wp.float64, device=d)
        self.d_u_prev = wp.zeros((self.rows, 3), dtype=wp.float64, device=d)
        self.d_v = wp.zeros((self.rows, 3), dtype=wp.float64, device=d)
        self.d_F = wp.zeros((self.rows, 3), dtype=wp.float64, device=d)
        self.d_result = wp.zeros((self.nel, 8), dtype=wp.float64, device=d)
        self.d_flag = wp.zeros(self.nel, dtype=wp.int32, device=d)
        self.d_mass = wp.zeros((self.rows, 3), dtype=wp.float64, device=d)
        self.d_fixed = wp.zeros((self.rows, 3), dtype=wp.int8, device=d)
        self.d_frozen = wp.zeros(self.rows, dtype=wp.int8, device=d)      # never frozen: shells do not erode
        self.d_scalars = wp.zeros(6, dtype=wp.float64, device=d)
        self.d_quat = wp.array(np.tile([0.0, 0.0, 0.0, 1.0], (self.nn, 1)), dtype=wp.quatd, device=d)
        self.d_pres_rot = wp.zeros(self.nn, dtype=wp.int32, device=d)
        self.frame_kernel, self.node_kernel, self.apply_kernel = make_shell4_kernels(wtype)
        self.d_lq = wp.zeros((self.nel, 24), dtype=wp.float64, device=d)
        self.d_frame = wp.zeros(self.nel, dtype=wp.mat33d, device=d)
        self.spin_kernel = make_spin_kernel()
        self.measures_partial, self.measures_mid, self.measures_finish = make_measures_kernels()
        self.groups = measure_groups(self.rows * 3)
        self.d_partials = wp.zeros((5, self.groups), dtype=wp.float64, device=d)
        self.d_mids = wp.zeros((5, MEASURE_MID), dtype=wp.float64, device=d)
        self.zero_kernel = make_zero_kernel()
        self.nodal = make_nodal_kernels()
        self._pres = None; self._amps = [None]

    # ---- geometry / boundary data --------------------------------------- #

    def set_mass(self, mass):
        self.d_mass.assign(np.ascontiguousarray(mass, dtype=np.float64).reshape(self.rows, 3))

    def set_nodal(self, fixed, load_groups, prescribed, starts=None):
        """fixed: bool (ndof,); load_groups: [(amplitude, F_unit (ndof,))];
        prescribed: [(dofs, value, amplitude)] with flat dof indices."""
        import warp as wp

        d = self.device
        self._progress = None
        self.d_fixed.assign(np.ascontiguousarray(fixed, dtype=np.int8).reshape(self.rows, 3))
        if load_groups:
            self._amps = [a for a, _F in load_groups]
            stack = np.stack([np.ascontiguousarray(F, dtype=np.float64).reshape(self.rows, 3)
                              for _a, F in load_groups])
        else:
            self._amps = [None]
            stack = np.zeros((1, self.rows, 3))
        self.d_F_ext_stack = wp.array(stack, dtype=wp.float64, device=d)
        self.d_factors = wp.zeros(len(self._amps), dtype=wp.float64, device=d)
        rows, dims, vals, aidx, st = [], [], [], [], []
        self._pres_amps = []
        pres_rot = np.zeros(self.nn, dtype=np.int32)
        for j, (dofs, val, a) in enumerate(prescribed):
            k = len(self._pres_amps); self._pres_amps.append(a)
            dofs = np.asarray(dofs, dtype=np.int64)
            rot = dofs[(dofs % 6) >= 3]
            np.bitwise_or.at(pres_rot, rot // 6, 1 << ((rot % 6) - 3))
            rows.extend((dofs // 3).tolist()); dims.extend((dofs % 3).tolist())
            vals.extend([float(val)] * len(dofs)); aidx.extend([k] * len(dofs))
            u0 = np.zeros(len(dofs)) if starts is None else np.broadcast_to(np.asarray(starts[j], dtype=float), (len(dofs),))
            st.extend(u0.tolist())
        self.d_pres_rot.assign(pres_rot)
        if rows:
            self._pres = (wp.array(np.array(rows, dtype=np.int32), dtype=wp.int32, device=d),
                          wp.array(np.array(dims, dtype=np.int32), dtype=wp.int32, device=d),
                          wp.array(np.array(vals, dtype=np.float64), dtype=wp.float64, device=d),
                          wp.array(np.array(aidx, dtype=np.int32), dtype=wp.int32, device=d),
                          wp.zeros(len(self._pres_amps), dtype=wp.float64, device=d),
                          wp.array(np.array(st, dtype=np.float64), dtype=wp.float64, device=d),
                          len(rows))
        else:
            self._pres = None

    def set_u(self, q, v=None, orient=None):
        from scipy.spatial.transform import Rotation
        self.d_u.assign(np.ascontiguousarray(q, dtype=np.float64).reshape(self.rows, 3))
        if v is not None:
            self.d_v.assign(np.ascontiguousarray(v, dtype=np.float64).reshape(self.rows, 3))
        if orient is not None:
            self.d_quat.assign(np.ascontiguousarray(Rotation.from_matrix(orient).as_quat(), dtype=np.float64))

    def get_orient(self):
        from scipy.spatial.transform import Rotation
        return Rotation.from_quat(self.d_quat.numpy()).as_matrix()

    def get_u(self):
        return self.d_u.numpy().reshape(-1)

    def get_v(self):
        return self.d_v.numpy().reshape(-1)

    def get_F_int(self):
        return self.d_F.numpy().reshape(-1)

    def get_result(self):
        return self.d_result.numpy()

    # ---- one iteration ---------------------------------------------------- #

    def force_resident(self):
        import warp as wp

        wp.launch(self.zero_kernel, dim=(self.rows, 3), device=self.device,
                  inputs=[self.d_F, self.d_scalars])
        wp.launch(self.frame_kernel, dim=self.nel, device=self.device,
                  inputs=[self.d_nodes, self.d_conn, self.d_u, self.d_local0,
                          np.float64(self.warp_tol), self.d_lq, self.d_frame,
                          self.d_flag, self.d_scalars])
        wp.launch(self.node_kernel, dim=(self.nel, 4), device=self.device,
                  inputs=[self.d_conn, self.d_quat, self.d_frame0, self.d_frame,
                          self.d_flag, self.d_lq])
        wp.launch(self.apply_kernel, dim=(self.nel, 8), device=self.device,
                  inputs=[self.d_conn, self.d_lq, self.d_frame, self.d_K, self.d_Bres,
                          self.d_flag, self.d_F, self.d_result])

    def check_flags(self):
        f = self.d_flag.numpy()
        if (f == 2).any():
            raise ValueError("shell4 has a degenerate midsurface")
        if (f == 1).any():
            raise ValueError("shell4 reference face is too warped; refine the surface mesh")

    def _set_factors(self, progress):
        if getattr(self, "_progress", None) == progress:
            return                                     # a staged level holds its factors
        self._progress = progress
        self.d_factors.assign(np.array([1.0 if a is None else a(progress) for a in self._amps], dtype=np.float64))
        if self._pres is not None:
            self._pres[4].assign(np.array([a(progress) for a in self._pres_amps], dtype=np.float64))

    def integrate(self, progress, half, approach=1.0):
        """Leapfrog with kinetic damping (c1 = c2 = 1), dt = 1, then the
        boundary conditions: fixed dofs zeroed, prescribed ones approached."""
        import warp as wp

        n, dev, dim = self.nodal, self.device, (self.rows, 3)
        wp.launch(n["copy2d"], dim=dim, device=dev, inputs=[self.d_u, self.d_u_prev])
        self._set_factors(progress)
        wp.launch(n["integrate"], dim=dim, device=dev,
                  inputs=[self.d_F, self.d_F_ext_stack, self.d_factors, self.d_mass, self.d_u, self.d_v,
                          np.float64(1.0), np.float64(1.0), np.float64(1.0), int(half)])
        wp.launch(n["zero_fixed"], dim=dim, device=dev, inputs=[self.d_fixed, self.d_u])
        if self._pres is not None:
            rows, dims, vals, aidx, pfac, pstart, npres = self._pres
            wp.launch(n["apply_prescribed"], dim=npres, device=dev,
                      inputs=[rows, dims, vals, aidx, pfac, pstart, np.float64(approach), self.d_u])
        wp.launch(n["sync_velocity"], dim=dim, device=dev,
                  inputs=[self.d_u_prev, self.d_u, self.d_v, np.float64(1.0)])
        # the host shell loop zeroes the velocity of every constrained dof,
        # prescribed ones included, so the kinetic energy that drives the
        # damping resets is the free dofs' alone; the sync above would have
        # credited the driven node with its ramp increment
        wp.launch(n["zero_fixed"], dim=dim, device=dev, inputs=[self.d_fixed, self.d_v])
        # the orientations: multiplicative, from the masked spin
        wp.launch(self.spin_kernel, dim=self.nn, device=dev,
                  inputs=[self.d_v, np.float64(1.0), self.d_pres_rot, self.d_quat, self.d_u])

    def measures_launch(self, progress):
        """The convergence measures into `d_scalars`, WITHOUT reading them
        back: two launches, no synchronisation, capturable into a graph.

        Split out from `measures` because the readback is the second most
        expensive thing in the iteration -- 148 to 249 us, near enough
        constant in the mesh size, since waiting on six scalars drains the
        whole pipeline every step (benchmarks/warp_loop_cost.py). A loop
        that decides its damping reset on the device can call this every
        iteration and read back every N."""
        import warp as wp

        self._set_factors(progress)
        wp.launch(self.measures_partial, dim=self.groups, device=self.device,
                  inputs=[self.d_F, self.d_F_ext_stack, self.d_factors, self.d_mass,
                          self.d_u, self.d_v, self.d_fixed, self.d_partials])
        wp.launch(self.measures_mid, dim=(5, MEASURE_MID), device=self.device,
                  inputs=[self.d_partials, self.d_mids])
        wp.launch(self.measures_finish, dim=5, device=self.device,
                  inputs=[self.d_mids, self.d_scalars])

    def measures(self, progress):
        """(kinetic energy, |F_ext - F_int|_free, |F_int|, |F_ext|_free,
        |u|_free, refused faces), from the internal force of the LAST
        force_resident call -- one iteration stale in the check, as on the
        host."""
        self.measures_launch(progress)
        s = self.d_scalars.numpy()
        return (float(s[0]), float(np.sqrt(s[1])), float(np.sqrt(s[2])), float(np.sqrt(s[3])),
                float(np.sqrt(s[4])), int(s[5]))

    def zero_velocity(self):
        import warp as wp

        wp.launch(self.nodal["zero_velocity"], dim=(self.rows, 3), device=self.device, inputs=[self.d_v])
