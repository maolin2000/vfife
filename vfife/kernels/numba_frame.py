"""Numba element kernel for the elastic frame members.

The three compiled kernels this package had -- solid, shell, fibre beam --
were each written when a case made the NumPy path the bottleneck. The elastic
frame never was: a beam element is cheap, and the largest frame model in the
library was Lee's frame with forty members. The eleven-storey building of
v0.1 has 759, and they cost 3.70 ms of a 4.4 ms step -- 97% of it, with the
55 shell walls beside them at 0.08 ms because the shell kernel already exists.

The NumPy path in `core/beam_solver.py` and `core/mixed_structural_solver.py`
is already batched over elements, so what this removes is not Python
overhead but temporaries: (ne, 12, 12) rotation blocks, (ne, 3, 3, 3) einsum
contractions and a `np.add.at` scatter, allocated and traversed every
iteration. Here one element lives in scalar locals and writes straight into
the global vector.

The element is geometrically nonlinear and materially linear, so the whole of
the constitutive law is frozen into `k_local` at assembly and the kernel needs
no material at all. What it does per element is the co-rotational kinematics
(`numba_rotations.member_frame`, which the fibre-beam kernel already uses --
the same mean-triad frame and the same local nodal rotations), one 12x12
matvec, and the rotation of the end actions back to the global axes.

SERIAL, like the other two, and for the same reason: the per-element work is
about a microsecond, threads would swamp it, and a serial loop can accumulate
straight into `out` instead of scattering afterwards.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
except ImportError:                                   # pragma: no cover
    njit = None

from .numba_rotations import member_frame


def available() -> bool:
    return njit is not None and member_frame is not None


if njit is not None:

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def frame_forces_subset(ids, nodes, conn, q, orient, R0, klocal, L0, dofs, dth,
                            out, end_actions, actions):
        """Internal force of the frame members `ids`, accumulated into `out`
        -- every member of a frame model (ids 0..n-1) or the frame members of
        a mixed one (core/families.FrameMembers).

        ids    (ne,)        each member's element number, the row it writes
                            in `end_actions` and `actions`
        nodes  (nn, 3)      reference coordinates
        conn   (ne, 2)      the two nodes of each member
        q      (nn, 6)      state; only the translations are read (the
                            rotations live in `orient` since v0.1)
        orient (nn, 3, 3)   each point's triad
        R0     (ne, 3, 3)   the member's reference frame, rows = axes
        klocal (ne, 12, 12) the local stiffness, materially linear
        L0     (ne,)        reference length
        dofs   (ne, 12)     global dof of each of the twelve local ones
        dth    (ne, 12)     the imposed deformation (thermal, creep, a
                            hinge's), subtracted from the local one so the
                            member force is k (d - d_th) and the actions it
                            reports are the ones it carries
                            (core/temperature.py). Zeros when there is none.

        All but `nodes`, `q`, `orient` are gathered to the members. The
        chord is checked BEFORE the frame: `member_frame` normalises by the
        chord and by the triad cross product, and on a collapsed member the
        second divides by zero (since v1.3 the kernels divide as
        IEEE does, error_model="numpy": a member that has turned inside out
        gives non-finite forces, as the NumPy path's, and the loop stops on
        them as NONFINITE instead of a traceback).

        Returns 0, or the position (in `ids`) of the first member that
        collapsed to zero length, plus one. Until v1.19 the frame
        solver ran a twin of this, `frame_forces`, over all its members:
        the same statements with ids[k] = k, removed when the frame solver
        took its members from the family."""
        ne = ids.shape[0]
        R = np.empty((3, 3))
        loc = np.empty((2, 3))
        x1 = np.empty(3)
        x2 = np.empty(3)
        local = np.empty(12)
        fl = np.empty(12)
        for k in range(ne):
            e = ids[k]
            n1 = conn[k, 0]
            n2 = conn[k, 1]
            for i in range(3):
                x1[i] = nodes[n1, i] + q[n1, i]
                x2[i] = nodes[n2, i] + q[n2, i]
            dx = x2[0] - x1[0]; dy = x2[1] - x1[1]; dz = x2[2] - x1[2]
            if dx * dx + dy * dy + dz * dz < 1e-28:
                return k + 1
            L = member_frame(x1, x2, orient[n1], orient[n2], R0[k], R, loc)
            for i in range(12):
                local[i] = 0.0
            local[3] = loc[0, 0]; local[4] = loc[0, 1]; local[5] = loc[0, 2]
            local[6] = L - L0[k]
            local[9] = loc[1, 0]; local[10] = loc[1, 1]; local[11] = loc[1, 2]
            # gathered like every other argument here: dth[k], not dth[e]
            # every component of the imposed deformation, creep's end twists too
            for i in range(12):
                local[i] -= dth[k, i]
            for i in range(12):
                acc = 0.0
                for j in range(12):
                    acc += klocal[k, i, j] * local[j]
                fl[i] = acc
                end_actions[e, i] = fl[i]
            for blk in range(4):
                b0 = fl[3 * blk]; b1 = fl[3 * blk + 1]; b2 = fl[3 * blk + 2]
                for i in range(3):
                    out[dofs[k, 3 * blk + i]] += (R[0, i] * b0 + R[1, i] * b1
                                                  + R[2, i] * b2)
            for i in range(6):
                a = fl[i] if fl[i] >= 0.0 else -fl[i]
                b = fl[i + 6] if fl[i + 6] >= 0.0 else -fl[i + 6]
                actions[e, i] = a if a > b else b
        return 0

    # ------------------------------------------------------------------ #
    # THREADED, v1.0. The note at the top was right for the
    # eleven-storey building and is wrong for a cable bridge: the Xihoumen
    # model's 851 members are 0.62 ms of a 1.0 ms step, and at 0.7 us each
    # the work divides well once the per-element scratch is not allocated
    # per element (six small arrays, some 0.3 us). So the members are cut
    # into `chunks` contiguous runs, each run allocates its scratch once,
    # each member writes its global forces into its own row of `ge`, and the
    # rows are added into `out` afterwards in member order -- the order the
    # serial kernel adds them in, so the answer is the serial one to the bit.
    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy", parallel=True)
    def frame_forces_subset_threaded(ids, nodes, conn, q, orient, R0, klocal, L0, dofs, dth,
                                     out, end_actions, actions, chunks):
        ne = ids.shape[0]
        ge = np.empty((ne, 12))
        bad = np.zeros(ne, dtype=np.uint8)
        nch = min(max(chunks, 1), max(ne, 1))
        for c in prange(nch):
            lo = c * ne // nch
            hi = (c + 1) * ne // nch
            R = np.empty((3, 3))
            loc = np.empty((2, 3))
            x1 = np.empty(3)
            x2 = np.empty(3)
            local = np.empty(12)
            fl = np.empty(12)
            for k in range(lo, hi):
                e = ids[k]
                n1 = conn[k, 0]
                n2 = conn[k, 1]
                for i in range(3):
                    x1[i] = nodes[n1, i] + q[n1, i]
                    x2[i] = nodes[n2, i] + q[n2, i]
                dx = x2[0] - x1[0]; dy = x2[1] - x1[1]; dz = x2[2] - x1[2]
                if dx * dx + dy * dy + dz * dz < 1e-28:
                    bad[k] = 1
                    for i in range(12):
                        ge[k, i] = 0.0
                    continue
                L = member_frame(x1, x2, orient[n1], orient[n2], R0[k], R, loc)
                for i in range(12):
                    local[i] = 0.0
                local[3] = loc[0, 0]; local[4] = loc[0, 1]; local[5] = loc[0, 2]
                local[6] = L - L0[k]
                local[9] = loc[1, 0]; local[10] = loc[1, 1]; local[11] = loc[1, 2]
                # every component of the imposed deformation, creep's end twists too
                for i in range(12):
                    local[i] -= dth[k, i]
                for i in range(12):
                    acc = 0.0
                    for j in range(12):
                        acc += klocal[k, i, j] * local[j]
                    fl[i] = acc
                    end_actions[e, i] = fl[i]
                for blk in range(4):
                    b0 = fl[3 * blk]; b1 = fl[3 * blk + 1]; b2 = fl[3 * blk + 2]
                    for i in range(3):
                        ge[k, 3 * blk + i] = R[0, i] * b0 + R[1, i] * b1 + R[2, i] * b2
                for i in range(6):
                    a = fl[i] if fl[i] >= 0.0 else -fl[i]
                    b = fl[i + 6] if fl[i + 6] >= 0.0 else -fl[i + 6]
                    actions[e, i] = a if a > b else b
        for k in range(ne):
            if bad[k]:
                return k + 1
        for k in range(ne):
            for j in range(12):
                out[dofs[k, j]] += ge[k, j]
        return 0

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def truss_forces(conn, nodes, q, EA, Lu, dth6, tension_only, ids, out, end_actions, actions):
        """Axial members: `_truss_forces_batched` element by
        element, its arithmetic in its order (E A first, then the strain),
        without the temporaries and the two `np.add.at` scatters that were a
        quarter of the Xihoumen step. `q` is the (nn, 6) state; `dth6` the
        free thermal elongation of each member (zeros when nothing is warm).
        Returns 0, or the first collapsed member plus one."""
        n = conn.shape[0]
        for k in range(n):
            a = conn[k, 0]
            b = conn[k, 1]
            ax = nodes[b, 0] + q[b, 0] - (nodes[a, 0] + q[a, 0])
            ay = nodes[b, 1] + q[b, 1] - (nodes[a, 1] + q[a, 1])
            az = nodes[b, 2] + q[b, 2] - (nodes[a, 2] + q[a, 2])
            L = np.sqrt(ax * ax + ay * ay + az * az)
            if L < 1e-14:
                return k + 1
            N = EA[k] * (L - Lu[k] - dth6[k]) / Lu[k]
            if tension_only[k] and N < 0.0:
                N = 0.0
            c = N / L
            fx = c * ax; fy = c * ay; fz = c * az
            out[6 * a] -= fx; out[6 * a + 1] -= fy; out[6 * a + 2] -= fz
            out[6 * b] += fx; out[6 * b + 1] += fy; out[6 * b + 2] += fz
            e = ids[k]
            end_actions[e, 0] = -N
            end_actions[e, 6] = N
            actions[e, 0] = N if N >= 0.0 else -N
        return 0

else:                                                  # pragma: no cover
    frame_forces_subset = frame_forces_subset_threaded = truss_forces = None


def configure_threads(threads) -> int:
    """`solver.threads` to a thread count: 1 serial, 0
    every core numba sees, n that many (capped there). Returns the count; 1
    means the serial kernels. Nothing is set here: numba's thread count is
    process-wide per calling thread, so a solver setting it once at setup
    changed every later kernel's -- the solid and shell ones too -- and a
    solver run from another thread did not see it (found in review,
    v1.0). `with_threads` sets it around each call instead."""
    t = 1 if threads is None else int(threads)
    if t == 1 or njit is None:
        return 1
    import numba
    top = numba.config.NUMBA_NUM_THREADS
    return top if t == 0 else min(t, top)


def with_threads(n: int, fn, *args):
    """`fn(*args)` with numba's thread count at `n`, restored afterwards."""
    import numba
    prev = numba.get_num_threads()
    if prev == n:
        return fn(*args)
    numba.set_num_threads(n)
    try:
        return fn(*args)
    finally:
        numba.set_num_threads(prev)
