"""The compiled whole step of the explicit dynamic frame loop.

`BeamSolver.run` takes one explicit step as about twenty NumPy operations
and Python calls around two compiled kernels: the external force, the member
forces, the damping force, the energy ledger, the velocity update, the
position and orientation update, the supports, the non-finite check. On the
Xihoumen model (810 points, 851 frame members, 356 cables) the kernels were
about two thirds of a 0.9 ms step and the rest was the glue -- each of its
operations a few microseconds of overhead on a 4860-vector, and fifty
thousand steps per second of bridge.

`dynamic_chunk` runs `n_do` whole steps inside one compiled call, between
two points where the Python loop has something to do (a progress check, an
output frame, the end of the step). Each operation is the Python loop's own,
element by element and in its order, so the state it leaves -- positions,
velocities, orientations -- is the Python loop's to the bit:

* the external force: `f = 0`, then the stacked groups' sparse product (a
  CSC matrix-vector product in SciPy's order), then each group's vector
  times its factor, then the ground's -m a_g d -- `BeamSolver._force`;
* the member forces: the frame kernel's arithmetic (`numba_frame`), each
  member's global forces into its own row, the rows added in member order,
  then the axial members as `numba_frame.truss_forces`;
* the damping force alpha m v + beta (K v) (CSR in SciPy's order) or
  zeta m v; the ledger's trapezoidal work with `np.dot`;
* v += (h dt) a with h = 1/2 on a restart, x += dt v, R <- exp(w dt) R
  (`numba_rotations.advance_node`), and the held dofs.

The ledger's sums go through `np.dot` as the Python ledger's do; numba's
`np.dot` is the same BLAS call, and a different BLAS would change them in
the last place only -- they are a diagnostic, not the state.

ONE SOURCE, TWO COMPILATIONS, as the fibre kernel: `prange`
is `range` in the serial one. Only the members and the points are divided
between threads; both are written per member / per point, and the members'
rows are added afterwards in member order, so the threaded answer is the
serial one to the bit.

Division is IEEE's (error_model="numpy", as the frame kernels since this
stage): a member that has turned inside out gives non-finite forces, the
state goes non-finite, and the run stops as NONFINITE -- the NumPy path's
behaviour, where the kernels used to raise ZeroDivisionError (from inside a
parallel region numba reports that as a SystemError, and only when the
region ends).

Returns (status, steps done, half): status 0 when all `n_do` steps were
taken, 1 when the state became non-finite in the last step taken (the
caller raises), 2 + i when member i (of the frames, then the axial members, then the connectors)
had collapsed to zero length at the start of the step after the last one
taken, 3 when a member reached its breaking capacity in the step after the
last one taken (an event, 1.6: the loop takes that step).
"""
from __future__ import annotations

import types

import numpy as np

try:
    from numba import njit, prange
except ImportError:                                   # pragma: no cover
    njit = None

from .numba_rotations import member_frame, advance_node, log3
from .numba_connector import respond, NO_BIRTH
from .numba_member_contact import contact_step, _keep_list
from .numba_loads import axle_actions, vehicle_step

#: a piece broken at both ends is removed at this multiple of its capacity;
#: the same number as core/member_failure.FRAGMENT_LIMIT
FRAGMENT_LIMIT = 2.0


def available() -> bool:
    return njit is not None and member_frame is not None


if njit is not None:
    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def _dot(a, b):
        """`np.dot`, compiled on its own: inside the threaded compilation numba
        would turn it into a parallel reduction, and the ledger's sums would
        then depend on the thread count in the last place."""
        return np.dot(a, b)


if njit is not None:
    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def _frame_rows(c_lo, c_hi, f_ids, f_conn, nodes, q2, orient, f_R0, f_k, f_L0, f_dth,
                    ge, bad, end_actions, actions, cR, cloc, cx1, cx2, clocal, cfl):
        """Members c_lo .. c_hi - 1: each one's global forces into its row of
        `ge` (numba_frame.frame_forces_subset_threaded). A function of its
        own: inlined into the serial chunk, LLVM made it half as fast."""
        for ck in range(c_lo, c_hi):
            ce = f_ids[ck]
            cn1 = f_conn[ck, 0]
            cn2 = f_conn[ck, 1]
            for ci in range(3):
                cx1[ci] = nodes[cn1, ci] + q2[cn1, ci]
                cx2[ci] = nodes[cn2, ci] + q2[cn2, ci]
            cdx = cx2[0] - cx1[0]; cdy = cx2[1] - cx1[1]; cdz = cx2[2] - cx1[2]
            if cdx * cdx + cdy * cdy + cdz * cdz < 1e-28:
                bad[ck] = 1
                for ci in range(12):
                    ge[ck, ci] = 0.0
                continue
            cL = member_frame(cx1, cx2, orient[cn1], orient[cn2], f_R0[ck], cR, cloc)
            # the local deformation as twelve scalars: each entry from 0.0
            # less its imposed part, as the array was filled; `f_dth` is per
            # ELEMENT (the plastic hinge rotations, 1.6), zeros when none
            dk = f_dth[ce]
            l0 = 0.0 - dk[0]; l1 = 0.0 - dk[1]; l2 = 0.0 - dk[2]
            l3 = cloc[0, 0] - dk[3]; l4 = cloc[0, 1] - dk[4]; l5 = cloc[0, 2] - dk[5]
            l6 = (cL - f_L0[ck]) - dk[6]; l7 = 0.0 - dk[7]; l8 = 0.0 - dk[8]
            l9 = cloc[1, 0] - dk[9]; l10 = cloc[1, 1] - dk[10]; l11 = cloc[1, 2] - dk[11]
            kk = f_k[ck]
            for ci in range(12):
                csum = 0.0
                csum += kk[ci, 0] * l0; csum += kk[ci, 1] * l1; csum += kk[ci, 2] * l2
                csum += kk[ci, 3] * l3; csum += kk[ci, 4] * l4; csum += kk[ci, 5] * l5
                csum += kk[ci, 6] * l6; csum += kk[ci, 7] * l7; csum += kk[ci, 8] * l8
                csum += kk[ci, 9] * l9; csum += kk[ci, 10] * l10; csum += kk[ci, 11] * l11
                cfl[ci] = csum
                end_actions[ce, ci] = csum
            for cb in range(4):
                cb0 = cfl[3 * cb]; cb1 = cfl[3 * cb + 1]; cb2 = cfl[3 * cb + 2]
                for ci in range(3):
                    ge[ck, 3 * cb + ci] = cR[0, ci] * cb0 + cR[1, ci] * cb1 + cR[2, ci] * cb2
            for ci in range(6):
                ca = cfl[ci] if cfl[ci] >= 0.0 else -cfl[ci]
                cbb = cfl[ci + 6] if cfl[ci + 6] >= 0.0 else -cfl[ci + 6]
                actions[ce, ci] = ca if ca > cbb else cbb

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def _truss_rows(c_lo, c_hi, t_conn, nodes, q2, t_EA, t_Lu, t_dth6, t_tonly, t_ids,
                    tg, tbad, end_actions, actions):
        """Axial members c_lo .. c_hi - 1 (numba_frame.truss_forces): the
        force on the first end into tg[k, :3], on the second into tg[k, 3:].
        `out[a] -= f` there is `out[a] += (-f)` here, the same number."""
        for tk in range(c_lo, c_hi):
            ta_ = t_conn[tk, 0]
            tb_ = t_conn[tk, 1]
            ax = nodes[tb_, 0] + q2[tb_, 0] - (nodes[ta_, 0] + q2[ta_, 0])
            ay = nodes[tb_, 1] + q2[tb_, 1] - (nodes[ta_, 1] + q2[ta_, 1])
            az = nodes[tb_, 2] + q2[tb_, 2] - (nodes[ta_, 2] + q2[ta_, 2])
            tL = np.sqrt(ax * ax + ay * ay + az * az)
            if tL < 1e-14:
                tbad[tk] = 1
                for ci in range(6):
                    tg[tk, ci] = 0.0
                continue
            tN = t_EA[tk] * (tL - t_Lu[tk] - t_dth6[tk]) / t_Lu[tk]
            if t_tonly[tk] and tN < 0.0:
                tN = 0.0
            cN = tN / tL
            fx = cN * ax; fy = cN * ay; fz = cN * az
            tg[tk, 0] = -fx; tg[tk, 1] = -fy; tg[tk, 2] = -fz
            tg[tk, 3] = fx; tg[tk, 4] = fy; tg[tk, 5] = fz
            te = t_ids[tk]
            end_actions[te, 0] = -tN
            end_actions[te, 6] = tN
            actions[te, 0] = tN if tN >= 0.0 else -tN

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def _gather_block(p_lo, p_hi, n, F_, fint_, damp_, tmp_, s_mode, g_F, g_amp, m_trans, m_ag, m_dir,
                      rows, inc_ptr, inc_idx, dkind, alpha, beta, zeta, K_indptr, K_indices, K_data,
                      v_, mass_, q_, led_on, led_prev, u_prev_, fe_prev_, fi_prev_, du_, tae_, tai_,
                      driven_, g_kind, g_idx, mv_, sub_on, sub_, gather=True):
        """Points p_lo .. p_hi - 1, dof by dof: the external force
        (BeamSolver._force), the member forces gathered from their rows in
        the order the serial kernels add them (frames by member, then the
        axial members), the damping force, and the ledger's terms.

        Plain loops over SLICES, one operation each: a loop from a variable
        start may meet a negative index, numba then wraps every access, and
        LLVM vectorises nothing (a copy of 4860 numbers took 10 us instead of
        1); one fused loop with its branches was slower still."""
        d0 = 6 * p_lo
        d1 = 6 * p_hi
        F = F_[d0:d1]; fint = fint_[d0:d1]; damp = damp_[d0:d1]; tmp = tmp_[d0:d1]
        v = v_[d0:d1]; mass = mass_[d0:d1]; q = q_[d0:d1]
        u_prev = u_prev_[d0:d1]; fe_prev = fe_prev_[d0:d1]; fi_prev = fi_prev_[d0:d1]
        du = du_[d0:d1]; tae = tae_[d0:d1]; tai = tai_[d0:d1]
        m = d1 - d0
        for i in range(m):
            F[i] = 0.0
        if s_mode != 0:
            for i in range(m):
                F[i] += tmp[i]
        for g in range(g_kind.shape[0]):
            # the load groups in the order the loop adds them: a fixed vector
            # times its factor, or a moving load's vector of this step (1.6)
            ag_ = g_amp[n, g]
            gF = g_F[g_idx[g], d0:d1] if g_kind[g] == 0 else mv_[g_idx[g], d0:d1]
            for i in range(m):
                F[i] += gF[i] * ag_
        for gm in range(m_dir.shape[0]):
            ag = m_ag[n, gm]
            for pp in range(p_hi - p_lo):
                for c in range(3):
                    F[6 * pp + c] += ((-m_trans[p_lo + pp, c]) * ag) * m_dir[gm, c]
        if gather:
            # (the fibre step's kernel adds into `fint` itself: gather=False)
            ip = inc_ptr[d0:d1 + 1]
            for i in range(m):
                fi = 0.0
                for jj in range(ip[i], ip[i + 1]):
                    fi += rows[inc_idx[jj]]
                fint[i] = fi
        if sub_on:
            # the contact between members, taken from the
            # member forces as the loop takes it: out[:, :3] -= f
            sub = sub_[d0:d1]
            for i in range(m):
                fint[i] -= sub[i]
        if dkind == 1:
            kp = K_indptr[d0:d1 + 1]
            for i in range(m):
                sk = 0.0
                for jj in range(kp[i], kp[i + 1]):
                    sk += K_data[jj] * v_[K_indices[jj]]
                damp[i] = (alpha * mass[i]) * v[i] + beta * sk
        elif dkind == 2:
            for i in range(m):
                damp[i] = (zeta * mass[i]) * v[i]
        if led_on:
            # the ledger's external force (orientation._ledger_force): at a
            # support that follows its own record the structure receives
            # fint + damp through the reaction (1.6 in the
            # compiled step). `damp` is zero where no damping is set
            driven = driven_[d0:d1]
            if led_prev:
                for i in range(m):
                    du[i] = q[i] - u_prev[i]
                for i in range(m):
                    tae[i] = .5 * (fe_prev[i] + ((fint[i] + damp[i]) if driven[i] else F[i]))
                for i in range(m):
                    tai[i] = .5 * (fi_prev[i] + fint[i])
            for i in range(m):
                u_prev[i] = q[i]
            for i in range(m):
                fe_prev[i] = (fint[i] + damp[i]) if driven[i] else F[i]
            for i in range(m):
                fi_prev[i] = fint[i]

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def _update_block(p_lo, p_hi, hd, dt, F_, fint_, damp_, acc_, dkind, mass_, q_, v_, w, orient, rot,
                      E, M, fixed_, fixed_zero_, clamped_mask, kin_v_, seed_kin, tav_, want_tav):
        """Points p_lo .. p_hi - 1: the velocity, the position and the
        orientation (orientation._advance), the held dofs
        (orientation._apply_bc, nothing prescribed). Returns False when a
        number came out non-finite. Slices and plain loops, as `_gather_block`."""
        d0 = 6 * p_lo
        d1 = 6 * p_hi
        m = d1 - d0
        F = F_[d0:d1]; fint = fint_[d0:d1]; damp = damp_[d0:d1]; acc = acc_[d0:d1]
        mass = mass_[d0:d1]; q = q_[d0:d1]; v = v_[d0:d1]
        fixed = fixed_[d0:d1]; fixed_zero = fixed_zero_[d0:d1]
        kin_v = kin_v_[d0:d1]; tav = tav_[d0:d1]
        if dkind == 0:
            for i in range(m):
                acc[i] = (F[i] - fint[i]) / mass[i]
        else:
            for i in range(m):
                acc[i] = (F[i] - fint[i] - damp[i]) / mass[i]
        for i in range(m):
            if fixed[i]:
                acc[i] = 0.0
                v[i] = 0.0
        for i in range(m):
            v[i] += hd * acc[i]
        npnt = p_hi - p_lo
        for pp in range(npnt):
            for c in range(3):
                q[6 * pp + c] += dt * v[6 * pp + c]
        for p in range(p_lo, p_hi):
            advance_node(p, w, dt, orient, rot, E, M)
        for pp in range(npnt):
            for c in range(3):
                q[6 * pp + 3 + c] = rot[p_lo + pp, c]
        for i in range(m):
            if fixed_zero[i]:
                v[i] = 0.0
        for pp in range(npnt):
            for c in range(3):
                if fixed_zero[6 * pp + c]:
                    q[6 * pp + c] = 0.0
        for pp in range(npnt):
            if clamped_mask[p_lo + pp]:
                for r in range(3):
                    for c in range(3):
                        orient[p_lo + pp, r, c] = 1.0 if r == c else 0.0
                    q[6 * pp + 3 + r] = 0.0
        if seed_kin:
            for i in range(m):
                kin_v[i] = v[i]
        finite = True
        for i in range(m):
            if not (np.isfinite(q[i]) and np.isfinite(v[i])):
                finite = False
        if want_tav:
            for i in range(m):
                tav[i] = dt * v[i]
        return finite


if njit is not None:

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def hinge_update(h_el, h_batch, end_actions, actions, f_k, cap, end_broken, active,
                     plastic_d, hinge_rot, rot_cap, hinge_damage, d_out, rot_out, dmg_out, commit):
        """MixedStructuralSolver._hinge_update, member by member (1.6): the
        return to the yield surface of each end of the hinged members `h_el`
        (batch rows `h_batch` of `f_k`), the corrected end actions and
        actions, and the increments of plastic rotation. With `commit` the
        increments go into `plastic_d`, `hinge_rot` and `hinge_damage`;
        without it they are left in `d_out` (nh, 12), `rot_out` (nh, 2) and
        the damage the commit would give in `dmg_out` (nh, 2), for the
        compiled step to read the failure check from and commit after it.
        The same arithmetic as the NumPy method, in the same order; the loop
        calls this too, so the two paths are one to the bit."""
        nh = h_el.shape[0]
        for r in range(nh):
            e = h_el[r]
            for i in range(12):
                d_out[r, i] = 0.0
            rot_out[r, 0] = hinge_rot[e, 0]; rot_out[r, 1] = hinge_rot[e, 1]
            dmg_out[r, 0] = hinge_damage[e, 0]; dmg_out[r, 1] = hinge_damage[e, 1]
            if not active[e]:
                continue
            kb = h_batch[r]
            any_d = False
            for a in range(2):
                iN = 6 * a; iy = 6 * a + 4; iz = 6 * a + 5
                n = abs(end_actions[e, iN]) / cap[e, 0]
                m = abs(end_actions[e, iy]) / cap[e, 1] + abs(end_actions[e, iz]) / cap[e, 2]
                room = 1.0 - n
                if room < 0.0:
                    room = 0.0
                if room > 1.0:
                    room = 1.0
                if m > room and not end_broken[e, a]:
                    mm = m if m > 1e-300 else 1e-300
                    sc = room / mm
                    dy = end_actions[e, iy] * (1.0 - sc) / f_k[kb, iy, iy]
                    dz = end_actions[e, iz] * (1.0 - sc) / f_k[kb, iz, iz]
                    d_out[r, iy] = dy; d_out[r, iz] = dz
                    rot_out[r, a] = hinge_rot[e, a] + (abs(dy) + abs(dz))
                    any_d = True
            if not any_d:
                continue
            # fl - k d: the four columns d is nonzero in, in ascending order
            # (the NumPy einsum adds the zeros of the others, which change
            # nothing)
            for i in range(12):
                acc = 0.0
                for j in range(12):
                    acc += f_k[kb, i, j] * d_out[r, j]
                end_actions[e, i] = end_actions[e, i] - acc
            for i in range(6):
                fa = abs(end_actions[e, i]); fb = abs(end_actions[e, 6 + i])
                actions[e, i] = fa if fa >= fb else fb
            rc = rot_cap[e]
            dmg_out[r, 0] = rot_out[r, 0] / rc; dmg_out[r, 1] = rot_out[r, 1] / rc
            if commit:
                for i in range(12):
                    plastic_d[e, i] += d_out[r, i]
                hinge_rot[e, 0] = rot_out[r, 0]; hinge_rot[e, 1] = rot_out[r, 1]
                hinge_damage[e, 0] = dmg_out[r, 0]; hinge_damage[e, 1] = dmg_out[r, 1]

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def hinge_commit(h_el, plastic_d, hinge_rot, hinge_damage, d_out, rot_out, dmg_out):
        """The commit `hinge_update` left out."""
        for r in range(h_el.shape[0]):
            e = h_el[r]
            for i in range(12):
                plastic_d[e, i] += d_out[r, i]
            hinge_rot[e, 0] = rot_out[r, 0]; hinge_rot[e, 1] = rot_out[r, 1]
            hinge_damage[e, 0] = dmg_out[r, 0]; hinge_damage[e, 1] = dmg_out[r, 1]

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def _nanmax(a, b):
        """max(a, b) as np.maximum / ndarray.max take it: a NaN wins."""
        if a != a or b != b:
            return np.nan
        return a if a >= b else b

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def break_check(end_actions, bar_cap, beam_on, beam_cap, failed, active, end_broken, hinged,
                    hinge_damage, h_el, dmg_out, capacity_ratio):
        """Breaking.check's criterion, without its bookkeeping (1.6): the
        capacity ratios of this step into `capacity_ratio`, as `check` writes
        them, and whether any member has reached 1 -- the compiled step then
        hands the step to the loop, whose `check` marks the break. A hinged
        member's damage is read from `dmg_out` where it is one of `h_el`
        (the update not yet committed), else from `hinge_damage`."""
        hit = False
        nel = end_actions.shape[0]
        for e in range(nel):
            if np.isfinite(bar_cap[e]) and not failed[e]:
                ratio = end_actions[e, 6] / bar_cap[e]
                # np.maximum(ratio, 0.0): a NaN stays a NaN (the step that
                # goes non-finite leaves the same ratios as the loop does)
                capacity_ratio[e] = ratio if ratio > 0.0 or ratio != ratio else 0.0
                if ratio >= 1.0:
                    hit = True
            if beam_on[e] and active[e] and not (end_broken[e, 0] and end_broken[e, 1]):
                r_i = abs(end_actions[e, 0]) / beam_cap[e, 0] + abs(end_actions[e, 4]) / beam_cap[e, 1] \
                    + abs(end_actions[e, 5]) / beam_cap[e, 2]
                r_j = abs(end_actions[e, 6]) / beam_cap[e, 0] + abs(end_actions[e, 10]) / beam_cap[e, 1] \
                    + abs(end_actions[e, 11]) / beam_cap[e, 2]
                if end_broken[e, 0]:
                    r_i = 0.0
                if end_broken[e, 1]:
                    r_j = 0.0
                capacity_ratio[e] = _nanmax(r_i, r_j)
                if hinged[e]:
                    d0 = hinge_damage[e, 0]; d1 = hinge_damage[e, 1]
                    for r in range(h_el.shape[0]):
                        if h_el[r] == e:
                            d0 = dmg_out[r, 0]; d1 = dmg_out[r, 1]
                    a0 = abs(end_actions[e, 0]) / beam_cap[e, 0]
                    a1 = abs(end_actions[e, 6]) / beam_cap[e, 0]
                    c0 = _nanmax(d0, a0)
                    c1 = _nanmax(d1, a1)
                    r_i = 0.0 if end_broken[e, 0] else c0
                    r_j = 0.0 if end_broken[e, 1] else c1
                if r_i >= 1.0 or r_j >= 1.0:
                    hit = True
            # a piece broken at both ends, judged whole
            if beam_on[e] and active[e] and end_broken[e, 0] and end_broken[e, 1]:
                r_i = abs(end_actions[e, 0]) / beam_cap[e, 0] + abs(end_actions[e, 4]) / beam_cap[e, 1] \
                    + abs(end_actions[e, 5]) / beam_cap[e, 2]
                r_j = abs(end_actions[e, 6]) / beam_cap[e, 0] + abs(end_actions[e, 10]) / beam_cap[e, 1] \
                    + abs(end_actions[e, 11]) / beam_cap[e, 2]
                if r_i >= FRAGMENT_LIMIT or r_j >= FRAGMENT_LIMIT:
                    hit = True
            # twisted a quarter turn (beam_cap[:, 3]): any
            # active breakable beam, a fragment broken at both ends included
            if beam_on[e] and active[e]:
                tq = abs(end_actions[e, 3])
                if abs(end_actions[e, 9]) > tq:
                    tq = abs(end_actions[e, 9])
                if tq >= beam_cap[e, 3]:
                    hit = True
        return hit

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def carry_appendages(q2, v2, orient, nodes, tips, anc, rel, o0, far, d0, parked_idx, parked_Q, E):
        """BeamSolver._carry_appendages (1.6): the parked particles held
        where they stopped, and every appendage's tips placed where its
        anchor carries them, x = x_anchor + R (x0 - x0_anchor), R the least
        rotation from the reference chord d0 to the chord now, the velocity
        the anchor's, the orientation R o0 and its logarithm into q. Returns
        False when a tip came out non-finite. The loop calls this too."""
        for k in range(parked_idx.shape[0]):
            p = parked_idx[k]
            for i in range(6):
                q2[p, i] = parked_Q[k, i]
                v2[p, i] = 0.0
        finite = True
        for k in range(tips.shape[0]):
            n = tips[k]; a = anc[k]; f = far[k]
            xa0 = nodes[a, 0] + q2[a, 0]; xa1 = nodes[a, 1] + q2[a, 1]; xa2 = nodes[a, 2] + q2[a, 2]
            if f >= 0:
                dx = nodes[f, 0] + q2[f, 0] - xa0; dy = nodes[f, 1] + q2[f, 1] - xa1; dz = nodes[f, 2] + q2[f, 2] - xa2
            else:
                dx = d0[k, 0]; dy = d0[k, 1]; dz = d0[k, 2]
            nrm = np.sqrt(dx * dx + dy * dy + dz * dz)
            if nrm < 1e-30:
                nrm = 1e-30
            dx /= nrm; dy /= nrm; dz /= nrm
            e0 = d0[k, 0]; e1 = d0[k, 1]; e2 = d0[k, 2]
            w0 = e1 * dz - e2 * dy; w1 = e2 * dx - e0 * dz; w2 = e0 * dy - e1 * dx
            c = e0 * dx + e1 * dy + e2 * dz
            den = 1.0 + c
            if den < 1e-9:
                den = 1e-9
            # R = I + W + W W / (1 + c), W = [w x]
            E[0, 0] = 0.0; E[0, 1] = -w2; E[0, 2] = w1
            E[1, 0] = w2; E[1, 1] = 0.0; E[1, 2] = -w0
            E[2, 0] = -w1; E[2, 1] = w0; E[2, 2] = 0.0
            R00 = 1.0 + E[0, 0] + (E[0, 0] * E[0, 0] + E[0, 1] * E[1, 0] + E[0, 2] * E[2, 0]) / den
            R01 = E[0, 1] + (E[0, 0] * E[0, 1] + E[0, 1] * E[1, 1] + E[0, 2] * E[2, 1]) / den
            R02 = E[0, 2] + (E[0, 0] * E[0, 2] + E[0, 1] * E[1, 2] + E[0, 2] * E[2, 2]) / den
            R10 = E[1, 0] + (E[1, 0] * E[0, 0] + E[1, 1] * E[1, 0] + E[1, 2] * E[2, 0]) / den
            R11 = 1.0 + E[1, 1] + (E[1, 0] * E[0, 1] + E[1, 1] * E[1, 1] + E[1, 2] * E[2, 1]) / den
            R12 = E[1, 2] + (E[1, 0] * E[0, 2] + E[1, 1] * E[1, 2] + E[1, 2] * E[2, 2]) / den
            R20 = E[2, 0] + (E[2, 0] * E[0, 0] + E[2, 1] * E[1, 0] + E[2, 2] * E[2, 0]) / den
            R21 = E[2, 1] + (E[2, 0] * E[0, 1] + E[2, 1] * E[1, 1] + E[2, 2] * E[2, 1]) / den
            R22 = 1.0 + E[2, 2] + (E[2, 0] * E[0, 2] + E[2, 1] * E[1, 2] + E[2, 2] * E[2, 2]) / den
            r0 = rel[k, 0]; r1 = rel[k, 1]; r2 = rel[k, 2]
            q2[n, 0] = xa0 + (R00 * r0 + R01 * r1 + R02 * r2) - nodes[n, 0]
            q2[n, 1] = xa1 + (R10 * r0 + R11 * r1 + R12 * r2) - nodes[n, 1]
            q2[n, 2] = xa2 + (R20 * r0 + R21 * r1 + R22 * r2) - nodes[n, 2]
            for i in range(6):
                v2[n, i] = v2[a, i]
            o = o0[k]
            for i in range(3):
                Ri0 = R00 if i == 0 else (R10 if i == 1 else R20)
                Ri1 = R01 if i == 0 else (R11 if i == 1 else R21)
                Ri2 = R02 if i == 0 else (R12 if i == 1 else R22)
                for j in range(3):
                    orient[n, i, j] = Ri0 * o[0, j] + Ri1 * o[1, j] + Ri2 * o[2, j]
            tmp = E[0]
            log3(orient[n], tmp)
            q2[n, 3] = tmp[0]; q2[n, 4] = tmp[1]; q2[n, 5] = tmp[2]
            for i in range(6):
                if not np.isfinite(q2[n, i]):
                    finite = False
        return finite

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def fracture_points(state_s, t_j, t_eps, t_valid, point):
        """FiberBeamSolver._point_failure and FractureLatch.update's maximum,
        for the steel fibres the fused kernel carries (1.6): a tracked column
        c is steel slot t_j[c] with rupture strain t_eps[c], and its point at
        station s has failed (1.0) where the committed maximum strain
        (state 7 of steel02) has reached it, masked by t_valid (a padded
        fibre, another section's rows). `point` (nel, n_tracked * n_gp) takes
        the maximum, as the latch does; returns True when a point rose, which
        the compiled step treats as an event (the loop then latches it)."""
        nel = state_s.shape[0]
        n_gp = state_s.shape[1]
        changed = False
        for e in range(nel):
            for c in range(t_j.shape[0]):
                for st in range(n_gp):
                    f = 1.0 if state_s[e, st, t_j[c], 7] >= t_eps[c] else 0.0
                    f = f * t_valid[e, c]
                    if f > point[e, c * n_gp + st]:
                        point[e, c * n_gp + st] = f
                        changed = True
        return changed

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def moving_step(n, q, dt, mv, mk, ma, mon, ax_group, ax_p, P_R, P_L, P_edofs, P_rel, P_has,
                    V_group, V_na, V_n, V_nf, V_nb, V_fi, V_flex, V_m, V_B, V_rigid, V_ks, V_cs, V_kt, V_ct,
                    V_weight, V_up, V_mk, V_ma, V_mon, V_rough, V_x, V_v, V_a, V_contact, V_surf, V_has, V_cnt,
                    scr_l, scr_g, V_fl, V_fg, V_sa, V_sf):
        """The load groups that are callables, at step `n` of a chunk (1.6):
        the moving loads' axles (core/moving_load.MovingLoad) and the vehicles
        (core/vehicle.Vehicle), each group into its row of `mv`, the axles
        in order, as `np.add.at` adds them. The paths' members are flattened
        into P_* (axes, lengths, dofs, releases); `mk`, `ma`, `mon` (n_do,
        n_axles) place the moving loads' axles, `ax_group`/`ax_p` give each
        its group and force. The vehicles' arrays are padded to the largest
        (V_na, V_n axles and dofs of each); their state advances in place
        and V_cnt[n] takes the contact losses for the loop's bookkeeping."""
        ndof = q.shape[0]
        for gm in range(mv.shape[0]):
            for i in range(ndof):
                mv[gm, i] = 0.0
        for ax in range(mk.shape[1]):
            if not mon[n, ax]:
                continue
            kk = mk[n, ax]
            axle_actions(ma[n, ax], P_L[kk], P_R[kk], ax_p[ax], P_rel[kk] if P_has[kk] else P_rel[0],
                         P_has[kk], scr_l, scr_g)
            gm = ax_group[ax]
            for i in range(12):
                mv[gm, P_edofs[kk, i]] += scr_g[i]
        for vi in range(V_group.shape[0]):
            na = V_na[vi]
            nv = V_n[vi]
            nf = V_nf[vi]
            t, r = vehicle_step(q, P_R, P_L, P_rel, P_has, P_edofs, V_mk[n, vi, :na], V_ma[n, vi, :na],
                                V_mon[n, vi, :na], V_rough[n, vi, :na], V_up[vi], V_nb[vi], V_fi[vi, :nf],
                                V_flex[vi, :na], V_m[vi, :nv], V_B[vi, :na, :nv], V_rigid[vi, :na],
                                V_ks[vi, :na], V_cs[vi, :na], V_kt[vi, :na], V_ct[vi, :na], V_weight[vi, :nv],
                                dt, V_x[vi, :nv], V_v[vi, :nv], V_a[vi, :nv], V_contact[vi, :na],
                                V_surf[vi, :na], V_has[vi:vi + 1], V_fl[:na], V_fg[:na], mv[V_group[vi]],
                                V_sa[:, :na], V_sf[:nv])
            V_cnt[n, vi, 0] = t
            V_cnt[n, vi, 1] = r

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def _prescribe(n, dt, half_dt2, q, v, kin_v, p_dof, p_entry, p_kind, p_tq, p_tv, tav, want_tav):
        """The prescribed motions of step n (orientation._apply_bc, after the
        held dofs; 1.6): per dof, its entry's kind -- 0 a displacement
        (q = target, v = its rate), 1 a support's velocity or acceleration
        record integrated by the amplitude (q, v and the kinematic velocity
        from it), 2 a velocity (q += dt v), 3 an acceleration (q += dt kv +
        dt^2/2 a, kv += dt a), and the damping work's dt v where it is
        wanted. Returns False when a number is not finite."""
        finite = True
        for j in range(p_dof.shape[0]):
            d = p_dof[j]
            e = p_entry[j]
            k = p_kind[e]
            if k == 0:
                q[d] = p_tq[n, e]
                v[d] = p_tv[n, e]
            elif k == 1:
                q[d] = p_tq[n, e]
                v[d] = p_tv[n, e]
                kin_v[d] = p_tv[n, e]
            elif k == 2:
                t = p_tq[n, e]
                q[d] += dt * t
                v[d] = t
                kin_v[d] = t
            else:
                t = p_tq[n, e]
                q[d] += dt * kin_v[d] + half_dt2 * t
                kin_v[d] += dt * t
                v[d] = kin_v[d]
            if want_tav:
                # the damping work is booked on the velocity the step leaves,
                # the prescribed one included (the loop's dt * v after _apply_bc)
                tav[d] = dt * v[d]
            if not (np.isfinite(q[d]) and np.isfinite(v[d])):
                finite = False
        return finite

else:                                                  # pragma: no cover
    _prescribe = None


def _dynamic_chunk(n_do, half, dt, q, v, mass, orient, rot, nodes,
                   f_ids, f_conn, f_R0, f_k, f_L0, f_dth, nch,
                   t_conn, t_EA, t_Lu, t_dth6, t_tonly, t_ids,
                   inc_ptr, inc_idx,
                   c_ids, c_kind, c_conn, c_R0, c_spins, cK, cC, clocal, chas_c, cncur, ccx, ccy, claw, clp,
                   cplanar, committed, slip, cdeform, include_damping,
                   g_F, g_amp,
                   s_mode, s_indptr, s_indices, s_data, s_A, s_i, s_w, s_direct,
                   m_trans, m_ag, m_dir,
                   dkind, alpha, beta, zeta, K_indptr, K_indices, K_data,
                   fixed, fixed_zero, clamped_mask, kin_v, seed_kin,
                   driven, half_dt2, p_dof, p_entry, p_kind, p_tq, p_tv,
                   h_el, h_batch, h_cap, b_end_broken, b_active, plastic_d, hinge_rot, rot_cap, hinge_damage,
                   brk_on, bar_cap, beam_on, beam_cap, b_failed, b_hinged, capacity_ratio,
                   a_tips, a_anc, a_rel, a_o0, a_far, a_d0, parked_idx, parked_Q,
                   g_kind, g_idx, mv, mk, ma, mon, ax_group, ax_p, P_R, P_L, P_edofs, P_rel, P_has,
                   V_group, V_na, V_n, V_nf, V_nb, V_fi, V_flex, V_m, V_B, V_rigid, V_ks, V_cs, V_kt, V_ct,
                   V_weight, V_up, V_mk, V_ma, V_mon, V_rough, V_x, V_v, V_a, V_contact, V_surf, V_has, V_cnt,
                   mc_on, mc_conn, mc_rad, mc_rad_s, mc_gid, mc_nel, mc_mass, mc_inertia, mc_k, mc_mu, mc_zeta,
                   mc_keys, mc_vals, mc_slack, mc_ci, mc_cj, mc_lstate, mc_ref, mc_f, mc_mom, mc_pen, mc_stats,
                   led_on, led_prev, u_prev, fe_prev, fi_prev, led_sums,
                   F, fint, damp, end_actions, actions, pos):
    ndof = q.shape[0]
    nn = ndof // 6
    nh = h_el.shape[0]
    mc_n = nn if mc_on else 1
    mc_x = np.zeros((mc_n, 3)); mc_v = np.zeros((mc_n, 3)); mc_w = np.zeros((mc_n, 3))
    mc_dof = np.zeros(ndof if mc_on else 1)
    movers = mk.shape[1] > 0 or V_group.shape[0] > 0
    scr_l = np.zeros(12); scr_g = np.zeros(12)
    na_max = V_mk.shape[2]; nv_max = V_x.shape[1]
    V_fl = np.zeros((na_max, 12)); V_fg = np.zeros((na_max, 12))
    V_sa = np.zeros((6, na_max)); V_sf = np.zeros(nv_max)
    d_out = np.zeros((max(nh, 1), 12)); rot_out = np.zeros((max(nh, 1), 2)); dmg_out = np.zeros((max(nh, 1), 2))
    aE = np.empty((3, 3))
    carry = a_tips.shape[0] > 0 or parked_idx.shape[0] > 0
    q2 = q.reshape((nn, 6))
    v2 = v.reshape((nn, 6))
    w = v2[:, 3:]
    nf = f_ids.shape[0]
    nt = t_conn.shape[0]
    ns = s_indptr.shape[0] - 1
    # every member writes its own row of `rows`: frames first (12 each), then
    # the axial members (6 each); `inc_ptr`/`inc_idx` gather them per dof
    nc = c_ids.shape[0]
    rows = np.zeros(nf * 12 + nt * 6 + nc * 12)
    ge = rows[:nf * 12].reshape((nf, 12))
    tg = rows[nf * 12:nf * 12 + nt * 6].reshape((nt, 6))
    # the connectors after the axial members, as the mixed
    # solver adds them
    cg = rows[nf * 12 + nt * 6:].reshape((nc, 12))
    bad = np.zeros(max(nf, 1), dtype=np.uint8)
    tbad = np.zeros(max(nt, 1), dtype=np.uint8)
    # chunks of frames, then chunks of axial members, in one parallel loop;
    # scratch once per call
    nfc = min(max(nch, 1), nf) if nf else 0
    ntc = min(max(nch, 1), nt) if nt else 0
    nsc = max(nfc, 1)
    ncc = max(nfc, ntc)
    sR = np.empty((nsc, 3, 3)); sloc = np.empty((nsc, 2, 3))
    sx1 = np.empty((nsc, 3)); sx2 = np.empty((nsc, 3))
    slocal = np.empty((nsc, 12)); sfl = np.empty((nsc, 12))
    nblk = min(max(nch, 1), max(nn, 1))
    bE = np.empty((nblk, 3, 3)); bM = np.empty((nblk, 3, 3))
    bok = np.ones(nblk, dtype=np.bool_)
    vals = np.empty(max(ns, 1))
    tmp = np.zeros(ndof)
    tae = np.empty(ndof)
    tai = np.empty(ndof)
    du = np.empty(ndof)
    tav = np.empty(ndof)
    acc = np.empty(ndof)
    want_tav = led_on and dkind != 0
    for n in range(n_do):
        # the steps completed so far, for the caller to read if an exception
        # ends the call (inside a step, or a KeyboardInterrupt as it returns)
        pos[0] = n
        # ---- the stacked load groups: SciPy's CSC product, serially
        if s_mode != 0:
            if s_mode == 1:
                ii0 = s_i[n]
                wn = s_w[n]
                for j in range(ns):
                    vals[j] = s_A[j, ii0] * (1.0 - wn) + s_A[j, ii0 + 1] * wn
            else:
                for j in range(ns):
                    vals[j] = s_direct[n, j]
            for i in range(ndof):
                tmp[i] = 0.0
            for j in range(ns):
                xj = vals[j]
                for jj in range(s_indptr[j], s_indptr[j + 1]):
                    tmp[s_indices[jj]] += s_data[jj] * xj
        # ---- the moving loads and the vehicles of this step (1.6): their
        # vectors, in the axles' order, each into its group's row of `mv`
        if movers:
            moving_step(n, q, dt, mv, mk, ma, mon, ax_group, ax_p, P_R, P_L, P_edofs, P_rel, P_has,
                        V_group, V_na, V_n, V_nf, V_nb, V_fi, V_flex, V_m, V_B, V_rigid, V_ks, V_cs, V_kt, V_ct,
                        V_weight, V_up, V_mk, V_ma, V_mon, V_rough, V_x, V_v, V_a, V_contact, V_surf, V_has,
                        V_cnt, scr_l, scr_g, V_fl, V_fg, V_sa, V_sf)
        # ---- the members, each into its own row
        # chunk ch takes a run of frames AND a run of axial members: numba
        # gives each thread a contiguous range of chunks, and frames-then-
        # bars chunks put all the frames on the first thread
        for ch in prange(ncc):
            if ch < nfc:
                _frame_rows(ch * nf // nfc, (ch + 1) * nf // nfc, f_ids, f_conn, nodes, q2,
                            orient, f_R0, f_k, f_L0, f_dth, ge, bad, end_actions, actions,
                            sR[ch], sloc[ch], sx1[ch], sx2[ch], slocal[ch], sfl[ch])
            if ch < ntc:
                _truss_rows(ch * nt // ntc, (ch + 1) * nt // ntc, t_conn, nodes, q2, t_EA, t_Lu,
                            t_dth6, t_tonly, t_ids, tg, tbad, end_actions, actions)
        for k in range(nf):
            if bad[k]:
                return 2 + k, n, half
        for k in range(nt):
            if tbad[k]:
                return 2 + nf + k, n, half
        if nc:
            cb = respond(c_ids, c_kind, c_conn, nodes, q2, orient, v, c_R0, c_spins, cK, cC, clocal,
                         chas_c, cncur, ccx, ccy, claw, clp, cplanar, committed, slip, include_damping,
                         fint, actions, end_actions, cdeform, 6, cg, True, True, NO_BIRTH)
            if cb >= 0:
                return 2 + nf + nt + cb, n, half
        # ---- the plastic hinges (MixedStructuralSolver._hinge_update) and
        # the failure criterion (Breaking.check), 1.6: a member that has
        # reached its capacity is an EVENT -- the step is handed back to the
        # loop, whose bookkeeping breaks it; the hinge increments of a step
        # that goes on are committed after the check, as the loop commits
        # them before it reads the damage
        if nh:
            hinge_update(h_el, h_batch, end_actions, actions, f_k, h_cap, b_end_broken, b_active,
                         plastic_d, hinge_rot, rot_cap, hinge_damage, d_out, rot_out, dmg_out, False)
        if brk_on:
            if break_check(end_actions, bar_cap, beam_on, beam_cap, b_failed, b_active, b_end_broken, b_hinged,
                           hinge_damage, h_el, dmg_out, capacity_ratio):
                return 3, n, half
        # ---- the contact between members (core/member_contact):
        # the one function the loop calls, on the positions and the
        # velocities the loop gives it. A neighbour list that has outgrown
        # its arrays is an EVENT: nothing has been changed, the loop takes
        # the step (and makes them longer). Before the hinges are committed:
        # an event hands the step back whole
        if mc_on:
            for i in range(nn):
                for c in range(3):
                    mc_x[i, c] = nodes[i, c] + q2[i, c]
                    mc_v[i, c] = v2[i, c]
                    mc_w[i, c] = v2[i, 3 + c]
            if contact_step(mc_x, mc_v, mc_w, mc_conn, mc_rad, mc_rad_s, mc_gid, mc_nel, mc_mass, mc_inertia,
                            dt, mc_k, mc_mu, mc_zeta, True, mc_keys, mc_vals, mc_slack, mc_ci, mc_cj,
                            mc_lstate, mc_ref, mc_f, mc_mom, mc_pen, mc_stats) < 0:
                return 3, n, half
            for i in range(nn):
                for c in range(3):
                    mc_dof[6 * i + c] = mc_f[i, c]
                    mc_dof[6 * i + 3 + c] = mc_mom[i, c]
        if nh:
            hinge_commit(h_el, plastic_d, hinge_rot, hinge_damage, d_out, rot_out, dmg_out)
        # ---- per dof: the loads, the member forces, damping, the ledger's
        # terms; then per point: velocity, position, orientation, supports.
        # One parallel loop for both, block by block, unless the damping
        # reads every velocity (beta K v): then all of it before any update
        hd = (0.5 if half else 1.0) * dt
        if dkind == 1:
            for bl in prange(nblk):
                _gather_block(bl * nn // nblk, (bl + 1) * nn // nblk, n, F, fint, damp, tmp, s_mode,
                              g_F, g_amp, m_trans, m_ag, m_dir, rows, inc_ptr, inc_idx,
                              dkind, alpha, beta, zeta, K_indptr, K_indices, K_data,
                              v, mass, q, led_on, led_prev, u_prev, fe_prev, fi_prev, du, tae, tai, driven,
                              g_kind, g_idx, mv, mc_on, mc_dof)
            for bl in prange(nblk):
                bok[bl] = _update_block(bl * nn // nblk, (bl + 1) * nn // nblk, hd, dt, F, fint, damp,
                                        acc, dkind, mass, q, v, w, orient, rot, bE[bl], bM[bl],
                                        fixed, fixed_zero, clamped_mask, kin_v, seed_kin, tav, want_tav)
        else:
            for bl in prange(nblk):
                _gather_block(bl * nn // nblk, (bl + 1) * nn // nblk, n, F, fint, damp, tmp, s_mode,
                              g_F, g_amp, m_trans, m_ag, m_dir, rows, inc_ptr, inc_idx,
                              dkind, alpha, beta, zeta, K_indptr, K_indices, K_data,
                              v, mass, q, led_on, led_prev, u_prev, fe_prev, fi_prev, du, tae, tai, driven,
                              g_kind, g_idx, mv, mc_on, mc_dof)
                bok[bl] = _update_block(bl * nn // nblk, (bl + 1) * nn // nblk, hd, dt, F, fint, damp,
                                        acc, dkind, mass, q, v, w, orient, rot, bE[bl], bM[bl],
                                        fixed, fixed_zero, clamped_mask, kin_v, seed_kin, tav, want_tav)
        half = False
        # ---- the prescribed motions, after the held dofs (orientation._apply_bc)
        pok = True
        if p_dof.shape[0]:
            pok = _prescribe(n, dt, half_dt2, q, v, kin_v, p_dof, p_entry, p_kind, p_tq, p_tv, tav, want_tav)
        # ---- the parked particles and the appendages (_carry_appendages)
        if carry:
            if not carry_appendages(q2, v2, orient, nodes, a_tips, a_anc, a_rel, a_o0, a_far, a_d0,
                                    parked_idx, parked_Q, aE):
                pok = False
            if want_tav:
                for k in range(parked_idx.shape[0]):
                    for i in range(6):
                        tav[6 * parked_idx[k] + i] = dt * v[6 * parked_idx[k] + i]
                for k in range(a_tips.shape[0]):
                    for i in range(6):
                        tav[6 * a_tips[k] + i] = dt * v[6 * a_tips[k] + i]
        if nc:
            # the connectors' history, committed every dynamic step (commit_state)
            for e in range(committed.shape[0]):
                for i in range(6):
                    committed[e, i] = slip[e, i]
        # ---- the ledger (EnergyLedger.update): the forces and the state of
        # one instant -- `du` and the mean forces were taken before the update
        if led_on:
            if led_prev:
                led_sums[0] += _dot(tae, du)
                led_sums[1] += _dot(tai, du)
            led_prev = True
        seed_kin = False
        if not pok:
            return 1, n + 1, half
        for bl in range(nblk):
            if not bok[bl]:
                return 1, n + 1, half
        # ---- the damping force's work over the step (EnergyLedger.damping)
        if want_tav:
            wd = _dot(damp, tav)
            if np.isfinite(wd):
                led_sums[2] += wd
        # this step is complete: an interrupt delivered when the call returns
        # still finds the count of the steps the state has taken
        pos[0] = n + 1
    return 0, n_do, half


def _named(fn, name):
    """A copy of `fn` under its own name, so the serial and the threaded
    compilations keep separate on-disk caches."""
    g = types.FunctionType(fn.__code__, fn.__globals__, name, fn.__defaults__, fn.__closure__)
    g.__qualname__ = name
    g.__doc__ = fn.__doc__
    return g


if njit is not None:

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def _pairwise_leaf(a, lo, n):
        if n < 8:
            res = 0.0
            for i in range(n):
                res += a[lo + i]
            return res
        r0 = a[lo]; r1 = a[lo + 1]; r2 = a[lo + 2]; r3 = a[lo + 3]
        r4 = a[lo + 4]; r5 = a[lo + 5]; r6 = a[lo + 6]; r7 = a[lo + 7]
        m = n - (n % 8)
        i = 8
        while i < m:
            r0 += a[lo + i]; r1 += a[lo + i + 1]; r2 += a[lo + i + 2]; r3 += a[lo + i + 3]
            r4 += a[lo + i + 4]; r5 += a[lo + i + 5]; r6 += a[lo + i + 6]; r7 += a[lo + i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[lo + i]
            i += 1
        return res

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def _pairwise(a, lo, n):
        """NumPy's pairwise sum of a[lo:lo+n] (numpy/_core/src/umath/
        loops_utils.h.src, `pairwise_sum`): eight accumulators to 128, halves
        rounded to a multiple of eight above -- `np.sum` of a contiguous
        float array, to the bit. The recursion on an explicit stack (a
        cached recursive numba function crashes on loading)."""
        if n <= 128:
            return _pairwise_leaf(a, lo, n)
        s_lo = np.empty(64, np.int64); s_n = np.empty(64, np.int64)
        s_st = np.zeros(64, np.int64); s_acc = np.empty(64)
        top = 0
        s_lo[0] = lo; s_n[0] = n; s_st[0] = 0
        res = 0.0
        while True:
            cl = s_lo[top]; cn = s_n[top]
            if cn <= 128:
                res = _pairwise_leaf(a, cl, cn)
                top -= 1
                # hand the value up: a left half waits for its right half
                while top >= 0:
                    if s_st[top] == 0:
                        s_acc[top] = res
                        s_st[top] = 1
                        n2 = s_n[top] // 2
                        n2 -= n2 % 8
                        top += 1
                        s_lo[top] = s_lo[top - 1] + n2; s_n[top] = s_n[top - 1] - n2; s_st[top] = 0
                        break
                    res = s_acc[top] + res
                    top -= 1
                if top < 0:
                    return res
            else:
                n2 = cn // 2
                n2 -= n2 % 8
                top += 1
                s_lo[top] = cl; s_n[top] = n2; s_st[top] = 0

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def kinetic_energy(mass, v, tmp):
        """0.5 * np.sum(mass * v * v), the relaxation's kinetic energy (1.9):
        the loop (BeamSolver._relax_stage under backend numba) and the
        compiled relaxation call this one function."""
        for i in range(v.shape[0]):
            tmp[i] = mass[i] * v[i] * v[i]
        return 0.5 * _pairwise(tmp, 0, tmp.shape[0])

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def norm2(x):
        """np.linalg.norm of a 1-D float array: sqrt(x . x)."""
        return np.sqrt(np.dot(x, x))

    @njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
    def free_norm(x, free_idx, tmp):
        """np.linalg.norm(x[free]): the free entries gathered in order."""
        n = free_idx.shape[0]
        for k in range(n):
            tmp[k] = x[free_idx[k]]
        return np.sqrt(np.dot(tmp[:n], tmp[:n]))


def _relax_chunk(it_first, it_last, ramp, tol, kin_tol, half, dt, c1, c2, cv,
                 q, v, mass, orient, rot, nodes,
                 f_ids, f_conn, f_R0, f_k, f_L0, f_dth, nch,
                 t_conn, t_EA, t_Lu, t_dth6, t_tonly, t_ids,
                 inc_ptr, inc_idx,
                 c_ids, c_kind, c_conn, c_R0, c_spins, cK, cC, clocal, chas_c, cncur, ccx, ccy, claw, clp,
                 cplanar, committed, slip, cdeform,
                 F, F_norm, free_idx, fixed, fixed_zero, clamped_mask, kin_v, seed_kin,
                 p_dof, p_ptr, p_target, p_start, p_vel, has_start,
                 led_on, led_prev, u_prev, fe_prev, fi_prev, led_sums,
                 fst, ist, meas, fint, end_actions, actions, pos):
    """Iterations `it_first` .. `it_last` of one level of a staged relaxation
    (BeamSolver._relax_stage, 1.9) in one call: the members
    (the dynamic step's own rows, `_frame_rows`, `_truss_rows`, `respond`
    without dashpots), the ledger, the velocity under kinetic damping, the
    position and the orientation, the held and the prescribed dofs, the
    kinetic reset, and the convergence measures. Stops at the first
    iteration that converges.

    fst = [ke_prev, force_scale, motion_scale], ist = [n_resets], both
    advanced in place; meas = [resid, kr, ke, resid_prev, kr_prev, ke_prev_m]
    (this iteration's measures and the one before's). Returns (status,
    iterations taken, converged, half): status 0, 1 non-finite, 2 + k a
    member collapsed (before the iteration's update)."""
    ndof = q.shape[0]
    nn = ndof // 6
    q2 = q.reshape((nn, 6))
    v2 = v.reshape((nn, 6))
    w = v2[:, 3:]
    nf = f_ids.shape[0]
    nt = t_conn.shape[0]
    nc = c_ids.shape[0]
    rows = np.zeros(nf * 12 + nt * 6 + nc * 12)
    ge = rows[:nf * 12].reshape((nf, 12))
    tg = rows[nf * 12:nf * 12 + nt * 6].reshape((nt, 6))
    cg = rows[nf * 12 + nt * 6:].reshape((nc, 12))
    bad = np.zeros(max(nf, 1), dtype=np.uint8)
    tbad = np.zeros(max(nt, 1), dtype=np.uint8)
    nfc = min(max(nch, 1), nf) if nf else 0
    ntc = min(max(nch, 1), nt) if nt else 0
    nsc = max(nfc, 1)
    ncc = max(nfc, ntc)
    sR = np.empty((nsc, 3, 3)); sloc = np.empty((nsc, 2, 3))
    sx1 = np.empty((nsc, 3)); sx2 = np.empty((nsc, 3))
    slocal = np.empty((nsc, 12)); sfl = np.empty((nsc, 12))
    E = np.empty((3, 3)); M = np.empty((3, 3))
    acc = np.empty(ndof)
    v_o = np.empty(ndof)
    tmp = np.empty(ndof)
    tmp2 = np.empty(ndof)
    du = np.empty(ndof)
    n_do = it_last - it_first + 1
    for n in range(n_do):
        pos[0] = n
        it = it_first + n
        approach = min(1.0, it / ramp)
        # ---- the members, each into its own row (as the dynamic step)
        for ch in prange(ncc):
            if ch < nfc:
                _frame_rows(ch * nf // nfc, (ch + 1) * nf // nfc, f_ids, f_conn, nodes, q2,
                            orient, f_R0, f_k, f_L0, f_dth, ge, bad, end_actions, actions,
                            sR[ch], sloc[ch], sx1[ch], sx2[ch], slocal[ch], sfl[ch])
            if ch < ntc:
                _truss_rows(ch * nt // ntc, (ch + 1) * nt // ntc, t_conn, nodes, q2, t_EA, t_Lu,
                            t_dth6, t_tonly, t_ids, tg, tbad, end_actions, actions)
        for k in range(nf):
            if bad[k]:
                return 2 + k, n, False, half
        for k in range(nt):
            if tbad[k]:
                return 2 + nf + k, n, False, half
        if nc:
            cb = respond(c_ids, c_kind, c_conn, nodes, q2, orient, v, c_R0, c_spins, cK, cC, clocal,
                         chas_c, cncur, ccx, ccy, claw, clp, cplanar, committed, slip, False,
                         fint, actions, end_actions, cdeform, 6, cg, True, True, NO_BIRTH)
            if cb >= 0:
                return 2 + nf + nt + cb, n, False, half
        for i in range(ndof):
            fi = 0.0
            for jj in range(inc_ptr[i], inc_ptr[i + 1]):
                fi += rows[inc_idx[jj]]
            fint[i] = fi
        # ---- the ledger (EnergyLedger.update), forces and state of one instant
        if led_on:
            if led_prev:
                for i in range(ndof):
                    du[i] = q[i] - u_prev[i]
                for i in range(ndof):
                    tmp[i] = .5 * (fe_prev[i] + F[i])
                led_sums[0] += _dot(tmp, du)
                for i in range(ndof):
                    tmp[i] = .5 * (fi_prev[i] + fint[i])
                led_sums[1] += _dot(tmp, du)
            for i in range(ndof):
                u_prev[i] = q[i]
                fe_prev[i] = F[i]
                fi_prev[i] = fint[i]
            led_prev = True
        # ---- the velocity under kinetic damping (and the mass-proportional term)
        for i in range(ndof):
            acc[i] = (F[i] - fint[i]) / mass[i]
        for i in range(ndof):
            if fixed[i]:
                acc[i] = 0.0
                v[i] = 0.0
        h = 0.5 if half else 1.0
        if cv > 0.0:
            for i in range(ndof):
                v_o[i] = v[i]
            hc = h * c1 * dt
            for i in range(ndof):
                v[i] = c2 * v[i] + hc * acc[i]
            if led_on:
                hm = 0.5 * cv
                for i in range(ndof):
                    tmp[i] = hm * mass[i] * (v_o[i] + v[i])
                for i in range(ndof):
                    tmp2[i] = dt * v[i]
                wd = _dot(tmp, tmp2)
                if np.isfinite(wd):
                    led_sums[2] += wd
        else:
            hd = h * dt
            for i in range(ndof):
                v[i] += hd * acc[i]
        half = False
        # ---- position and orientation (orientation._advance)
        for p in range(nn):
            for c in range(3):
                q[6 * p + c] += dt * v[6 * p + c]
        for p in range(nn):
            advance_node(p, w, dt, orient, rot, E, M)
        for p in range(nn):
            for c in range(3):
                q[6 * p + 3 + c] = rot[p, c]
        # ---- the supports (orientation._apply_bc with start and approach)
        for i in range(ndof):
            if fixed_zero[i]:
                v[i] = 0.0
        for p in range(nn):
            for c in range(3):
                if fixed_zero[6 * p + c]:
                    q[6 * p + c] = 0.0
        for p in range(nn):
            if clamped_mask[p]:
                for r in range(3):
                    for c in range(3):
                        orient[p, r, c] = 1.0 if r == c else 0.0
                    q[6 * p + 3 + r] = 0.0
        if seed_kin:
            for i in range(ndof):
                kin_v[i] = v[i]
            seed_kin = False
        # entry by entry, its displacements then its velocities, as the
        # loop writes them (q[dofs] = target; v[dofs] = ...)
        for e in range(p_ptr.shape[0] - 1):
            for k in range(p_ptr[e], p_ptr[e + 1]):
                t = p_target[k]
                if has_start and approach < 1.0:
                    t = p_start[k] + (t - p_start[k]) * approach
                q[p_dof[k]] = t
            for k in range(p_ptr[e], p_ptr[e + 1]):
                v[p_dof[k]] = p_vel[k]
        # ---- kinetic energy, finiteness, the kinetic reset
        ke = kinetic_energy(mass, v, tmp)
        finite = np.isfinite(ke)
        for i in range(ndof):
            if not np.isfinite(q[i]):
                finite = False
        if not finite:
            return 1, n + 1, False, half
        if ke < fst[0] and fst[0] > 0.0:
            if led_on and np.isfinite(fst[0]):
                led_sums[2] += fst[0] if fst[0] > 0.0 else 0.0
            for i in range(ndof):
                v[i] = 0.0
            half = True
            ist[0] += 1
            ke = 0.0
        fst[0] = ke
        # ---- the convergence measures (BeamSolver._convergence_measures),
        # the force of this iteration's start
        fn = norm2(fint)
        now = F_norm
        if fn > now:
            now = fn
        if now > fst[1]:
            fst[1] = now
        scale = now
        if 1e-6 * fst[1] > scale:
            scale = 1e-6 * fst[1]
        if 1e-30 > scale:
            scale = 1e-30
        for i in range(ndof):
            tmp2[i] = F[i] - fint[i]
        resid = free_norm(tmp2, free_idx, tmp) / scale
        motion = free_norm(q, free_idx, tmp)
        if motion > fst[2]:
            fst[2] = motion
        mm = motion
        if 1e-6 * fst[2] > mm:
            mm = 1e-6 * fst[2]
        if 1e-12 > mm:
            mm = 1e-12
        den = scale * mm
        if 1e-30 > den:
            den = 1e-30
        kr = ke / den
        meas[3] = meas[0]; meas[4] = meas[1]; meas[5] = meas[2]
        meas[0] = resid; meas[1] = kr; meas[2] = ke
        pos[0] = n + 1
        if it > ramp and resid <= tol and kr <= kin_tol:
            return 0, n + 1, True, half
    return 0, n_do, False, half


if njit is not None:
    relax_chunk = njit(cache=True, fastmath=False, nogil=True, error_model="numpy")(_named(_relax_chunk, "relax_chunk"))
    relax_chunk_threaded = njit(cache=True, fastmath=False, nogil=True, error_model="numpy", parallel=True)(
        _named(_relax_chunk, "relax_chunk_threaded"))
else:                                                  # pragma: no cover
    relax_chunk = relax_chunk_threaded = kinetic_energy = norm2 = free_norm = None


if njit is not None:
    dynamic_chunk = njit(cache=True, fastmath=False, nogil=True, error_model="numpy")(_named(_dynamic_chunk, "dynamic_chunk"))
    dynamic_chunk_threaded = njit(cache=True, fastmath=False, nogil=True, error_model="numpy", parallel=True)(
        _named(_dynamic_chunk, "dynamic_chunk_threaded"))
else:                                                  # pragma: no cover
    dynamic_chunk = dynamic_chunk_threaded = None


# --------------------------------------------------------------------------- #
#  THE FIBRE BEAM'S STEP. The fibre kernel is built per
#  section and pair of laws (kernels/numba_fiber_beam.make_kernel, not cached
#  on disk), so the chunk that calls it is too, and kept per kernel. Its
#  members are the kernel's and the compiled connectors'; the rest of the
#  step is the frame chunk's own blocks, in the fibre loop's order.
# --------------------------------------------------------------------------- #
_FIBER_CHUNKS: dict = {}


def fiber_chunk_ready(kernel) -> bool:
    """Whether this process has compiled the fibre step for `kernel` already."""
    hit = _FIBER_CHUNKS.get(id(kernel))
    return hit is not None and hit[0] is kernel


def fiber_chunk(kernel):
    """The compiled step for fibre models whose force is `kernel`. It is
    compiled in every new process (about 3 s): the fibre kernel it calls is
    built per model, which numba's cache cannot key on."""
    hit = _FIBER_CHUNKS.get(id(kernel))
    if hit is not None and hit[0] is kernel:
        return hit[1]
    fkernel = kernel

    @njit(cache=False, fastmath=False, nogil=True, error_model="numpy")
    def chunk(n_do, half, dt, q, v, mass, orient, rot,
              edofs, x0, R0, length, gw, Bky, Bkz, cy, cz, ca, cpre, sy, sz, sa, spre,
              state_c, state_s, cpar, spar, G0J0, GA, Bgy, Bgz, kactive, fstress, birth,
              c_ids, c_kind, c_conn, nodes, spins, cK, cC, clocal, chas_c, cncur, ccx, ccy, claw, clp,
              cplanar, committed, slip, cdeform, include_damping,
              g_F, g_amp, m_trans, m_ag, m_dir,
              dkind, alpha, beta, zeta, K_indptr, K_indices, K_data,
              fixed, fixed_zero, clamped_mask, kin_v, seed_kin,
              driven, half_dt2, p_dof, p_entry, p_kind, p_tq, p_tv,
              fr_on, t_j, t_eps, t_valid, point,
              sl_on, sl_ri, sl_rj, sl_GA0, sl_GAcr, sl_Vy, sl_h, sl_kb, sl_flex, sl_epscr,
              sl_gamma, sl_gcr_c, sl_gmax_c, sl_gcr_t, sl_gmax_t, sl_V, sl_any,
              g_kind, g_idx, mv, mk, ma, mon, ax_group, ax_p, P_R, P_L, P_edofs, P_rel, P_has,
              V_group, V_na, V_n, V_nf, V_nb, V_fi, V_flex, V_m, V_B, V_rigid, V_ks, V_cs, V_kt, V_ct,
              V_weight, V_up, V_mk, V_ma, V_mon, V_rough, V_x, V_v, V_a, V_contact, V_surf, V_has, V_cnt,
              mc_on, mc_conn, mc_rad, mc_rad_s, mc_gid, mc_nel, mc_mass, mc_inertia, mc_k, mc_mu, mc_zeta,
              mc_keys, mc_vals, mc_slack, mc_ci, mc_cj, mc_lstate, mc_ref, mc_f, mc_mom, mc_pen, mc_stats,
              led_on, led_prev, u_prev, fe_prev, fi_prev, led_sums,
              F, fint, damp, actions, end_actions, pos):
        ndof = q.shape[0]
        nn = ndof // 6
        mc_n = nn if mc_on else 1
        mc_x = np.zeros((mc_n, 3)); mc_v = np.zeros((mc_n, 3)); mc_w = np.zeros((mc_n, 3))
        mc_dof = np.zeros(1)
        q2 = q.reshape((nn, 6))
        v2 = v.reshape((nn, 6))
        w = v2[:, 3:]
        nc = c_ids.shape[0]
        movers = mk.shape[1] > 0 or V_group.shape[0] > 0
        scr_l = np.zeros(12); scr_g = np.zeros(12)
        na_max = V_mk.shape[2]; nv_max = V_x.shape[1]
        V_fl = np.zeros((na_max, 12)); V_fg = np.zeros((na_max, 12))
        V_sa = np.zeros((6, na_max)); V_sf = np.zeros(nv_max)
        rows = np.zeros(1)
        no_rows = np.zeros((1, 12))
        inc_ptr = np.zeros(ndof + 1, np.int64)
        inc_idx = np.zeros(1, np.int64)
        tmp = np.zeros(1)
        tae = np.empty(ndof); tai = np.empty(ndof); du = np.empty(ndof)
        tav = np.empty(ndof); acc = np.empty(ndof)
        E = np.empty((3, 3)); M = np.empty((3, 3))
        want_tav = led_on and dkind != 0
        for n in range(n_do):
            pos[0] = n
            # ---- the contact between members, its neighbour
            # list first: one that has outgrown its arrays hands the step
            # back to the loop (status 4) before anything of it is done --
            # the fibre kernel below moves the fibres' state
            mc_swept = 0
            if mc_on:
                for i in range(nn):
                    for c in range(3):
                        mc_x[i, c] = nodes[i, c] + q2[i, c]
                        mc_v[i, c] = v2[i, c]
                        mc_w[i, c] = v2[i, 3 + c]
                mc_swept = mc_lstate[1]
                if _keep_list(mc_x, mc_conn, mc_rad_s, mc_ci, mc_cj, mc_lstate, mc_ref, mc_stats) < 0:
                    return 4, n, half
                mc_swept = mc_lstate[1] - mc_swept
            # ---- the member forces at the start of the step: the fibre
            # kernel, then the connectors, into fresh arrays (as
            # FiberBeamSolver._internal_force_kernel)
            for i in range(ndof):
                fint[i] = 0.0
            for e in range(actions.shape[0]):
                for i in range(actions.shape[1]):
                    actions[e, i] = 0.0
                for i in range(12):
                    end_actions[e, i] = 0.0
            fkernel(edofs, x0, q, orient, R0, length, gw, Bky, Bkz, cy, cz, ca, cpre, sy, sz, sa, spre,
                    state_c, state_s, cpar, spar, G0J0, GA, Bgy, Bgz, fint, actions, end_actions,
                    kactive, fstress, birth,
                    sl_on, sl_ri, sl_rj, sl_GA0, sl_GAcr, sl_Vy, sl_h, sl_kb, sl_flex, sl_epscr,
                    sl_gamma, sl_gcr_c, sl_gmax_c, sl_gcr_t, sl_gmax_t, sl_V, True)
            if nc:
                respond(c_ids, c_kind, c_conn, nodes, q2, orient, v, R0, spins, cK, cC, clocal, chas_c,
                        cncur, ccx, ccy, claw, clp, cplanar, committed, slip, include_damping,
                        fint, actions, end_actions, cdeform, 6, no_rows, False, False, NO_BIRTH)
            # ---- the fracture latch's points (FiberBeamSolver._latch_fracture,
            # 1.6): a point that fails in this step is an EVENT -- the step is
            # finished as the loop finishes it (the forces of the step are the
            # broken section's own), and the loop then latches the member,
            # queues its split and rebuilds the compiled step
            fr_hit = False
            if fr_on:
                fr_hit = fracture_points(state_s, t_j, t_eps, t_valid, point)
            if fr_hit and mc_on:
                # ... and with contact between members the step is handed to
                # the loop HERE, half done (status 5): the loop
                # latches a fractured member -- it leaves the active set --
                # BEFORE the contact of this step, and whether a member goes
                # is the latch's to say. The fibres' state and the member
                # forces of this step are made; the loop takes them from
                # here instead of making them again. A list made above is
                # the loop's to make: its count is given back
                if mc_swept > 0:
                    mc_lstate[1] -= 1
                    mc_lstate[2] = 1
                return 5, n, half
            if mc_on:
                # the forces, from the member forces as the loop takes them
                # (the list was made above: it is kept here)
                contact_step(mc_x, mc_v, mc_w, mc_conn, mc_rad, mc_rad_s, mc_gid, mc_nel, mc_mass, mc_inertia,
                             dt, mc_k, mc_mu, mc_zeta, True, mc_keys, mc_vals, mc_slack, mc_ci, mc_cj,
                             mc_lstate, mc_ref, mc_f, mc_mom, mc_pen, mc_stats)
                for i in range(nn):
                    for c in range(3):
                        fint[6 * i + c] -= mc_f[i, c]
                        fint[6 * i + 3 + c] -= mc_mom[i, c]
            # ---- the moving loads and the vehicles of this step (1.6)
            if movers:
                moving_step(n, q, dt, mv, mk, ma, mon, ax_group, ax_p, P_R, P_L, P_edofs, P_rel, P_has,
                            V_group, V_na, V_n, V_nf, V_nb, V_fi, V_flex, V_m, V_B, V_rigid, V_ks, V_cs, V_kt,
                            V_ct, V_weight, V_up, V_mk, V_ma, V_mon, V_rough, V_x, V_v, V_a, V_contact, V_surf,
                            V_has, V_cnt, scr_l, scr_g, V_fl, V_fg, V_sa, V_sf)
            # ---- the loads, the damping, the ledger's terms (fint given)
            _gather_block(0, nn, n, F, fint, damp, tmp, 0, g_F, g_amp, m_trans, m_ag, m_dir,
                          rows, inc_ptr, inc_idx, dkind, alpha, beta, zeta, K_indptr, K_indices, K_data,
                          v, mass, q, led_on, led_prev, u_prev, fe_prev, fi_prev, du, tae, tai, driven,
                          g_kind, g_idx, mv, False, mc_dof, False)
            if led_on:
                if led_prev:
                    led_sums[0] += _dot(tae, du)
                    led_sums[1] += _dot(tai, du)
                led_prev = True
            hd = (0.5 if half else 1.0) * dt
            half = False
            # ---- the connectors' and the shear laws' history is committed
            # every dynamic step (commit_state), before the next evaluation
            # reads it
            if sl_any:
                for e in range(sl_gcr_c.shape[0]):
                    sl_gcr_c[e] = sl_gcr_t[e]
                    sl_gmax_c[e] = sl_gmax_t[e]
            if nc:
                for e in range(committed.shape[0]):
                    for i in range(6):
                        committed[e, i] = slip[e, i]
            finite = _update_block(0, nn, hd, dt, F, fint, damp, acc, dkind, mass, q, v, w, orient, rot,
                                   E, M, fixed, fixed_zero, clamped_mask, kin_v, seed_kin, tav, want_tav)
            seed_kin = False
            if p_dof.shape[0]:
                if not _prescribe(n, dt, half_dt2, q, v, kin_v, p_dof, p_entry, p_kind, p_tq, p_tv, tav, want_tav):
                    finite = False
            if not finite:
                return 1, n + 1, half
            if want_tav:
                wd = _dot(damp, tav)
                if np.isfinite(wd):
                    led_sums[2] += wd
            pos[0] = n + 1
            if fr_hit:
                return 3, n + 1, half
        return 0, n_do, half

    _FIBER_CHUNKS[id(kernel)] = (kernel, chunk)
    return chunk
