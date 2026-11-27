"""The implicit static solver: Newton on a sparse finite-difference tangent.

Dynamic relaxation reaches a static state by running an explicit dynamics
with fictitious masses and kinetic damping until it stops. It needs no
tangent and it survives anything, which is why it is this library's static
solver -- and it takes thousands of force evaluations per load level, which
on a fibre frame is minutes per level (case_21: 22 minutes for a pushover).

This is the other way: solve r(x) = F(lambda) - f(x) = 0 by Newton, load
level by load level, with the tangent K = df/dx built by finite differences
of the SAME internal force the explicit solvers use -- so every element,
section law and co-rotational frame the library has is in it, with no
element stiffness written twice.

THE TANGENT COSTS A FEW DOZEN FORCE EVALUATIONS, NOT 2n. A column of K is
the change of the force when one dof moves, and that change is confined to
the elements touching its node. Two nodes that share no element and no
neighbour can therefore be moved TOGETHER in one evaluation and their
columns read apart afterwards (Curtis, Powell & Reid 1974). A distance-2
colouring of the node graph groups them; the cost is 2 x (dofs per node) x
(colours) evaluations, which for a frame is 36-100 however large the model
is.

Rotations are perturbed and updated as SPINS, orient <- exp(w) orient: the
coordinates the internal force is conjugate to (testing/state.py), so the
tangent is the true one and a Newton step on a rotation is a rotation.

Path-dependent materials are evaluated in trial mode (every evaluation starts
from the last converged state, as staged relaxation already does), and the
state is committed once per converged increment. An increment that does not
converge is halved and retried; the load level it failed at is where a limit
point would be, which this solver (load control) cannot pass -- that is the
arc-length method, the next stage.
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .core.diagnostics import (ConvergenceFailure, finish as finish_diagnostics,
                               start as start_diagnostics, StopAnalysis)
from .base.rotations import exp_map, log_map

import os as _os
_DEBUG = bool(_os.environ.get("VFIFE_NEWTON_DEBUG"))
#: under arc_length.control, how much farther than its predictor an increment
#: may move the state before it counts as a jump to another branch
_JUMP = float(_os.environ.get("VFIFE_ARC_JUMP", "1.5"))
#: the tangent from the members at a colour's nodes where the solver can give
#: them; False differences the whole force, as every other
#: model does -- the tests compare the two. (The environment switch
#: `VFIFE_TANGENT_MEMBERS=0` went in v1.18.)
_MEMBERS = True


# ----------------------------------------------------------------- the model
def check(solver) -> None:
    """Refuse what this solver cannot do yet, by name."""
    from .elements.library import takes, taking
    element = solver.p.case.element.type
    if not takes(element, "static"):
        raise ValueError(f"solver.mode = static runs frame and fibre-beam models ({', '.join(taking('static'))}); this is a "
                         f"{element} model ({type(solver).__name__}). Use mode: relaxation")
    p = solver.p
    why = []
    # rigid diaphragms (1.7): a linear multi-point constraint,
    # taken by a transformation of the Newton system (`Newton.T`); member
    # failure (the same): a member that reaches its capacity in an accepted
    # equilibrium is taken out and the level solved again. What a static
    # solution has no clock or motion for stays refused.
    b = getattr(solver, "breaking", None)
    if b is not None and b.can_break:
        if b._removals:
            why.append("removals at a time (a static step has no clock; give the step a `deactivate`)")
    if getattr(solver, "member_contact", None) is not None:
        # found in the review of 1.11: Newton stalled at the load where two
        # members met and reported a limit point
        why.append("contact between members (the tangent's colouring does not know the pairs)")
    if getattr(solver, "fracture", None) is not None:
        why.append("fibre fracture")
    if len(getattr(solver, "_hinged", ())) > 0:
        why.append("plastic hinges (rotation_capacity)")
    if getattr(solver, "time_dependent", False):
        # THE CLOCK RUNS HERE TOO: every converged increment of a step with
        # `to_age` is a time station, as every level of a staged relaxation
        # is -- on the frame and the fibre paths alike (each solver's own
        # `_record_time_station`; the fibre path's from 1.22). An arc-length
        # path, whose increments are not levels, does not.
        arc_ = p.solver.static.arc_length              # typed
        on_ = arc_.steps if arc_ is not None else None
        timed_ = [st for st in solver.steps if st.get("to_age") is not None
                  and (not on_ or st.get("name") in on_)]
        if arc_ and (timed_ or not on_):
            why.append("creep, shrinkage, ageing or relaxation together with arc_length (in a step of time)")
    sl = getattr(solver, "_sl_on", None)
    if sl is not None and np.any(sl):
        why.append("a cracking shear law (its shear rotation relaxes with the explicit solution)")
    if why:
        raise ValueError("solver.mode = static does not take " + ", ".join(why) +
                         " yet; use mode: relaxation")
    arc = p.solver.static.arc_length
    arc = arc.model_dump() if arc is not None else None       # (read as a dict below)
    if arc:
        names = [st.get("name") for st in solver.steps]
        wanted = arc.get("steps")
        if wanted:
            # a misspelt step name used to match nothing, and every step ran
            # by load control -- jumping the snap it was written for
            # (found in review)
            unknown = [w for w in wanted if w not in names]
            if unknown:
                raise ValueError(f"static.arc_length.steps names {unknown}, which no step is called; "
                                 f"the steps are {names}")
        on = [i for i, n in enumerate(names) if not wanted or n in wanted]
        if arc.get("stop") and on and on[0] < len(names) - 1 and any(i < len(names) - 1 for i in on):
            # a step that ends at its `stop` ends short of its full load, and
            # the next step would start from its loads held at their full
            # value: a jump nobody asked for
            raise ValueError("static.arc_length.stop ends a step short of its full load, so it may only "
                             "apply to the last step; list the last step alone in arc_length.steps")


def node_colours(elements: np.ndarray, nn: int) -> List[np.ndarray]:
    """Groups of nodes no two of which share an element or a neighbour: a
    greedy distance-2 colouring of the node graph."""
    nbr = [set([i]) for i in range(nn)]
    for row in np.asarray(elements):
        row = [int(a) for a in row if a >= 0]
        for a in row:
            nbr[a].update(row)
    colour = np.full(nn, -1)
    # the most connected first: the greedy bound is then tighter
    for a in sorted(range(nn), key=lambda i: -len(nbr[i])):
        taken = set()
        for b in nbr[a]:
            for c in nbr[b]:
                if colour[c] >= 0:
                    taken.add(colour[c])
        k = 0
        while k in taken:
            k += 1
        colour[a] = k
    groups = [np.flatnonzero(colour == k) for k in range(colour.max() + 1)]
    return groups, nbr


class Newton:
    """Load-controlled Newton-Raphson with a sparse finite-difference tangent."""

    def __init__(self, solver, spec: Dict):
        from .core import constraints
        #: the solver's multi-point constraints (core/constraints), None for one outside the beam family
        self.ties = constraints.of(solver)
        self.s = solver
        self.spec = spec
        s = solver
        self.stride = s.dofs.stride
        self.nn = s.nn
        el = np.asarray(s.mesh.elements)
        el = el[:, :2] if el.ndim == 2 and el.shape[1] >= 2 else el
        self.groups, self.nbr = node_colours(el, self.nn)
        self.node_dofs = np.arange(self.nn * self.stride).reshape(self.nn, self.stride)
        # THE STEP IS SCALED BY THE ELEMENT, NOT BY THE MODEL. It was the
        # largest coordinate: 3e-5 m on the 580 m Hangzhou Bay unit, where a
        # fibre's strain moved by some 1e-6 either side -- enough to straddle
        # the cracking kink of the concrete in many fibres at once. The
        # central difference then averages two branches, the tangent is
        # wrong by a little, and near the peak a little is everything:
        # Newton went linear (ratio 0.4 per iteration), the arc shrank by a
        # fifth every increment and never reached the peak in 200 of them.
        # What a fibre feels is h over the element's length.
        el = np.asarray(s.mesh.elements)
        if len(el) and el.ndim == 2 and el.shape[1] >= 2:
            X = np.asarray(s.mesh.nodes, float)
            ok = (el[:, 0] >= 0) & (el[:, 1] >= 0)
            lengths = np.linalg.norm(X[el[ok, 1]] - X[el[ok, 0]], axis=1)
            size = float(np.median(lengths)) if lengths.size else 1.0
        else:
            size = 1.0
        self.h_t = float(spec.get("perturbation", 1e-7)) * (size if size > 0 else 1.0)
        self.h_r = float(spec.get("perturbation", 1e-7)) * 10.0
        self.trial = hasattr(s, "_trial_mode")
        self.evaluations = 0
        self.tangents = 0
        self.iterations = 0
        #: increments the stabilised Newton took (`solve_stabilised`)
        self.stabilised = 0
        # rows of each node's neighbourhood, for reading a column apart
        self._rows = [self.node_dofs[sorted(self.nbr[a])].ravel() for a in range(self.nn)]
        #: the model's size, for telling a round-off correction from a real one
        self.size = float(np.ptp(np.asarray(s.mesh.nodes, float), axis=0).max()) if s.nn else 1.0
        self.size = self.size if self.size > 0 else 1.0
        # THE ROUND-OFF OF A COORDINATE IS eps TIMES ITS DISTANCE FROM THE
        # ORIGIN, not times the model's extent: a 2 m cantilever placed at
        # UTM coordinates (5e5, 3e6) carries 7e-10 m of it, and a correction
        # of 1e-11 x 2 m is far below what its arithmetic can resolve. A
        # rotation's is that over the element.
        far = float(np.abs(np.asarray(s.mesh.nodes, float)).max()) if s.nn else 1.0
        far = max(far, self.size, 1e-300)
        self.roundoff_t = 1e3 * np.finfo(float).eps * far
        self.roundoff_r = 1e3 * np.finfo(float).eps * max(1.0, far / (size if size > 0 else 1.0))
        #: max(|F_free|, |f|, restraint) at the last `measure`, for `commit`
        self._now = 0.0
        #: whether the last failed `solve_level` stalled at round-off
        self.stalled = False
        #: the multi-point constraints' transformation q = T q_m, while it
        #: does not change (floors alone); see `T`
        self._T_held = None
        #: how many increments fell back to relaxation (1.7)
        self.fallbacks = 0

    @property
    def T(self):
        """The multi-point constraints as a transformation of the Newton
        system, q = T q_m (`ConstraintLayer.map`, 1.7 for the floors, 1.21
        for the rigid links); None without any. The floors' is linear and
        kept; with rigid links it is the current configuration's, made
        again each time it is asked for."""
        ties = self.ties
        if ties is None or self.stride != 6 or ties.empty():
            return None
        if self._T_held is not None:
            return self._T_held
        s = self.s
        T, own, m, slave = ties.map(s.ndof, s.physical_mass)
        self._master_of_own = (own, m)
        self._slave = slave
        if ties.links is None:
            self._T_held = T
        return T

    def move(self, dx: np.ndarray):
        """`advance`, then a rigid link's slaves set exactly from their
        masters (the update is the linear map's, the constraint is not)."""
        self.advance(dx)
        if self.ties is not None and self.ties.links is not None:
            self.ties.correct()

    def _free_master(self, free):
        """The free mask on the master dofs: a floor's masters are free (its
        in-plane dofs may not be held: `BeamFamily._tie_floors` refuses the
        always-held ones, and a step's own constraint is refused here)."""
        own, m = self._master_of_own
        if not np.all(free[self._slave]):
            self.ties.check_free(free)
        fm = np.ones(m, bool)
        fm[:len(own)] = free[own]
        return fm

    def residual_norm(self, r, free):
        """|r| over the free dofs -- of the master system when there are
        diaphragms: at a floor's node the nodal residual is the constraint
        force, and only the floor's resultants have to vanish."""
        T = self.T
        if T is None:
            return float(np.linalg.norm(r[free]))
        rm = T.T @ r
        return float(np.linalg.norm(rm[self._free_master(free)]))

    # ------------------------------------------------------------ measure
    def measure(self, F, f, free, r=None):
        """The relative residual, on the scale the relaxation solvers use
        (`BeamSolver._convergence_measures`).

        It was max(|F_free|, |f|) alone, and an equilibrium that carries no
        force -- a free member bowing under a temperature gradient, a
        prestressed cantilever under its own tendons, a support released in
        a later step -- has both go to zero at the answer: the residual was
        round-off over round-off, 1.000, and a simply supported girder under a
        gradient failed as "a limit point" (found in review).
        The restraint force of the imposed deformation, the tendons' force
        and a millionth of the largest force this analysis has carried join
        the scale, as they do in relaxation."""
        s = self.s
        if r is None:
            r = F - f
        if hasattr(s, "_thermal_d"):
            # the frame solvers: `_thermal_scale` is current only while a
            # deformation is imposed
            restraint = float(getattr(s, "_thermal_scale", 0.0)) if s._thermal_d is not None else 0.0
        else:
            restraint = float(getattr(s, "_thermal_scale", 0.0) or 0.0)
        now = max(float(np.linalg.norm(F[free])), float(np.linalg.norm(f)), restraint)
        # THE PEAK IS READ HERE, AND RAISED ONLY BY AN ACCEPTED STATE
        # (`commit`). It was raised by every iterate: the linear predictor of
        # a string-like member put 9.8e11 N in it against a 1e4 N load, and the
        # floor then accepted 82 N out of balance on a later 10 N load -- twice
        # the deflection, reported as converged
        self._now = now
        peak = float(getattr(s, "_force_scale", 0.0) or 0.0)
        # A JACKED BAR'S FORCE JOINS THE FLOOR TOO (1.15): a
        # girder under its own tendons and nothing else has no external load,
        # and at the answer the bars' pull and the girder's push cancel at
        # every node -- 1.000 of round-off over round-off, as for a gradient.
        # A millionth of it, as of the peak: the whole of it let a later
        # step's small load stop 0.24% short (found in review)
        jt = getattr(s, "_jack_T", None)
        jacks = float(np.max(np.abs(jt))) if jt is not None and len(jt) else 0.0
        scale = max(now, 1e-6 * max(peak, jacks), float(getattr(s, "_tendon_force_scale", 0.0) or 0.0), 1e-30)
        return self.residual_norm(r, free) / scale

    def accept_scale(self):
        """The state just measured is an accepted equilibrium: its force
        joins the peak the floor is a millionth of."""
        s = self.s
        carried = float(self._now)
        # AND THE FORCES ITS FIBRES CARRY, EVEN WHERE THEY CANCEL. A member
        # released after yielding holds +-Fy in its fibres with no resultant:
        # the residual's round-off is that of those fibre forces, 2e-7 N on a
        # 0.1 m steel section, and against a floor from the external load
        # alone the release "stalled" at 2.3e-6 for a tolerance of 1e-6
        # (it converged in v1.0 only because the floor was raised by a
        # trial iterate)
        rows = getattr(s, "fiber_stress_rows", None)
        area = getattr(s, "fiber_area", None)
        if rows is not None and area is not None:
            try:
                fa = np.abs(np.asarray(rows(), float)) * np.asarray(area, float)[:, None, :]
                carried = max(carried, float(np.linalg.norm(fa.sum(axis=2).max(axis=1))))
            except (ValueError, TypeError, AttributeError):
                pass
        s._force_scale = max(float(getattr(s, "_force_scale", 0.0) or 0.0), carried)

    def at_roundoff(self, dx) -> bool:
        """A Newton correction no larger than the round-off of the
        coordinates: the residual left is the arithmetic's, not the path's."""
        d = np.asarray(dx).reshape(self.nn, self.stride)
        t = float(np.abs(d[:, :3]).max()) if d.size else 0.0
        w = float(np.abs(d[:, 3:]).max()) if self.stride > 3 and d.size else 0.0
        return t <= self.roundoff_t and w <= self.roundoff_r

    # ------------------------------------------------------------- state
    def get(self):
        s = self.s
        return s.q.copy(), (s.orient.copy() if hasattr(s, "orient") else None)

    def put(self, state):
        s = self.s
        q, orient = state
        s.q[:] = q
        if orient is not None:
            s.orient[...] = orient

    def advance(self, dx: np.ndarray):
        """Translations added, rotations applied as spins."""
        s = self.s
        qv = s.dofs.view(s.q); dv = dx.reshape(self.nn, self.stride)
        qv[:, :3] += dv[:, :3]
        if self.stride == 6:
            w = dv[:, 3:]
            moved = np.flatnonzero(np.abs(w).max(axis=1) > 0.0)
            if len(moved):
                s.orient[moved] = np.einsum("nij,njk->nik", exp_map(w[moved]), s.orient[moved])
                qv[moved, 3:] = log_map(s.orient[moved])

    def force(self) -> np.ndarray:
        s = self.s
        if self.trial:
            s._trial_mode = True
        self.evaluations += 1
        f, _ = s.internal_force()
        return np.asarray(f, float).copy()

    # ----------------------------------------------------------- tangent
    def _pattern(self):
        """The tangent's sparsity, once (1.15): for every
        colour, the rows read for its nodes; and where each value read goes
        in the CSC matrix. The pattern is the mesh's -- it was rebuilt, one
        small array a node, at every tangent: a quarter of a tangent's time
        on a 17 000-member model."""
        pat = getattr(self, "_pat", None)
        if pat is not None:
            return pat
        n = self.s.ndof
        rows_g, lens_g = [], []
        rows_all, cols_all = [], []
        for group in self.groups:
            rows = [self._rows[a] for a in group]
            lens = np.array([len(r) for r in rows], np.int64)
            rc = np.concatenate(rows) if rows else np.zeros(0, np.int64)
            rows_g.append(rc); lens_g.append(lens)
            for comp in range(self.stride):
                rows_all.append(rc)
                cols_all.append(np.repeat(self.node_dofs[group, comp], lens))
        rows_all = np.concatenate(rows_all); cols_all = np.concatenate(cols_all)
        # the order `coo.tocsc()` puts the values in: by column, then by row
        # (each (row, column) is read once, so nothing is summed)
        order = sp.coo_matrix((np.arange(1, len(rows_all) + 1, dtype=np.float64), (rows_all, cols_all)),
                              shape=(n, n)).tocsc()
        perm = order.data.astype(np.int64) - 1
        assert len(perm) == len(rows_all)
        pat = self._pat = (rows_g, perm, order.indices.copy(), order.indptr.copy(), len(rows_all))
        return pat

    def tangent(self, free: np.ndarray) -> sp.csc_matrix:
        s = self.s
        rows_g, perm, indices, indptr, nnz = self._pattern()
        vals = np.empty(nnz)
        fast = getattr(s, "_tangent_picks", None) if _MEMBERS else None
        picks = fast(self.groups) if fast is not None else None
        if getattr(self, "_said", None) != (picks is not None):
            self._said = picks is not None
            n = 2 * self.stride
            s.log("  static        a tangent is " + (
                f"{2 * n} evaluations of each member (the ones at a colour's nodes)" if picks is not None
                else f"{n * len(self.groups)} evaluations of the whole force"))
        if picks is not None:
            self._tangent_of_members(picks, rows_g, vals)
        else:
            base = self.get()
            at = 0
            for gi, group in enumerate(self.groups):
                r = rows_g[gi]
                for comp in range(self.stride):
                    rot = comp >= 3
                    h = self.h_r if rot else self.h_t
                    # every column, the held dofs' too: the predictor needs K_fc,
                    # the coupling of the free dofs to the ones a constraint
                    # drives, and moving them costs no extra evaluation
                    cols = self.node_dofs[group, comp]
                    dx = np.zeros(s.ndof); dx[cols] = h
                    self.put(base); self.advance(dx); fp = self.force()
                    self.put(base); self.advance(-dx); fm = self.force()
                    df = (fp - fm) / (2.0 * h)
                    vals[at:at + len(r)] = df[r]; at += len(r)
            self.put(base)
        self.tangents += 1
        n = s.ndof
        return sp.csc_matrix((vals[perm], indices, indptr), shape=(n, n))

    def _tangent_of_members(self, picks, rows_g, vals):
        """THE TANGENT FROM THE MEMBERS THAT FEEL EACH NUDGE (1.15). Moving the nodes of one colour changes the force of
        the members that touch them and of no other; the difference of two
        whole-model forces is then a difference of two sums in which almost
        every term is the same -- 17 colours of them on the arch, each member
        evaluated 17 x 6 x 2 times for the 2 x 6 x 2 that tell anything.
        Here only those members are evaluated: an eighth of the arithmetic
        there (each member has two ends, of two colours), and none of the
        round-off of the terms that cancel -- a stiff member far from the
        nudge no longer leaves 1e-16 of its force in the column.

        The nudges and the central difference are `tangent`'s; the state is
        put back node by node."""
        s = self.s
        s._tangent_begin(picks)
        q0 = s.q.copy(); o0 = s.orient.copy()
        qv = s.dofs.view(s.q)
        out = np.zeros(s.ndof)
        at = 0
        for gi, group in enumerate(self.groups):
            r = rows_g[gi]; pick = picks[gi]
            base_q = qv[group].copy(); base_o = s.orient[group].copy()
            for comp in range(self.stride):
                rot = comp >= 3
                h = self.h_r if rot else self.h_t
                f = []
                for sign in (1.0, -1.0):
                    if rot:
                        w = np.zeros((len(group), 3)); w[:, comp - 3] = sign * h
                        s.orient[group] = np.einsum("nij,njk->nik", exp_map(w), base_o)
                        qv[group, 3:] = log_map(s.orient[group])
                    else:
                        qv[group, comp] = base_q[:, comp] + sign * h
                    out[r] = 0.0
                    s._force_of(pick, out)
                    f.append(out[r].copy())
                    qv[group] = base_q; s.orient[group] = base_o
                vals[at:at + len(r)] = (f[0] - f[1]) / (2.0 * h); at += len(r)
        self.evaluations += 2 * self.stride * 2      # as whole-model evaluations: each member twice a (component, sign)
        s.q[:] = q0; s.orient[...] = o0

    # --------------------------------------------------------- predictor
    def factor(self, K, free, mu=0.0, D=None):
        """The factored tangent on the free dofs (`_Factor`); on the master
        dofs, T^T K T, when there are diaphragms. `mu` > 0 adds the springs
        mu diag(D) (the stabilised Newton, `solve_stabilised`), through T.
        Raises RuntimeError when singular, as `splu` does."""
        return _Factor(self, K, free, mu, D)

    def predict(self, apply_bc, F_now, F_next, fixed):
        """The tangent predictor for the next load level: with K at the
        converged state, the free dofs move by K_ff^-1 (dF - K_fc dc), where
        dc is what the constraints impose over the increment. Without it a
        displacement-controlled increment starts with the whole jump in the
        one element next to the driven node -- a strain the concrete cracks
        and softens under, and Newton spends the increment undoing it."""
        s = self.s
        free = ~fixed
        base = self.get()
        K = self.tangent(free)
        apply_bc()
        dc = np.zeros(s.ndof)
        dc[fixed] = s.q[fixed] - base[0][fixed]
        # a prescribed rotation is written as its logarithm; small increments
        # of it are spins to first order
        rhs = (F_next - F_now) - K @ dc
        try:
            fac = self.factor(K, free)
        except RuntimeError:
            return K
        dx = fac.solve(rhs)
        if np.all(np.isfinite(dx)):
            self.move(np.where(free, dx, 0.0))
        return K

    # --------------------------------------------------------- increment
    def solve_level(self, F_of, fixed: np.ndarray, tol: float, K0=None, max_iterations=None):
        """Newton at one load level from the current state. Returns
        (converged, residual, iterations). `max_iterations` overrides the
        settings' for this call (fewer after a fallback, 1.7)."""
        spec = self.spec
        if max_iterations is None:
            max_iterations = int(spec.get("max_iterations", 30))
        free = ~fixed
        K = lu = None
        res = np.inf
        history = []
        self.stalled = False
        tiny = 0
        if K0 is not None and spec.get("tangent", "full") == "modified":
            try:
                lu = self.factor(K0, free)
            except RuntimeError:
                lu = None
        for it in range(int(max_iterations)):
            F = F_of()
            f = self.force()
            r = F - f
            r[fixed] = 0.0
            res = self.measure(F, f, free, r)
            if _DEBUG:
                print(f"      newton it {it}: residual {res:.3e}")
            if not np.isfinite(res):
                return False, res, it
            if res <= tol:
                return True, res, it
            if tiny >= 2 and res < 1e-2 and history and res > 0.5 * history[-1]:
                # two corrections in a row at the round-off of the
                # coordinates, the residual no longer falling, and still
                # above the tolerance: that is the arithmetic's floor, not a
                # limit point, and a smaller increment will not lower it.
                # Only for a SMALL residual: on a perfectly
                # plastic plateau the corrections vanish too, with the
                # residual at 0.2 -- that is a limit point. And only when the
                # residual has STOPPED FALLING: a modified tangent converges
                # linearly with corrections of 1e-12 m on a 2 m cantilever,
                # and was stopped at 2.8e-10 on its way to 1e-11
                self.stalled = True
                return False, res, it
            if lu is None or spec.get("tangent", "full") == "full":
                K = self.tangent(free)
                try:
                    lu = self.factor(K, free)
                except RuntimeError:
                    return False, res, it                 # singular: a mechanism, or past a limit
            dx = lu.solve(r)
            if not np.all(np.isfinite(dx)):
                return False, res, it
            tiny = tiny + 1 if self.at_roundoff(dx) else 0
            self.iterations += 1
            if _DEBUG:
                prev = getattr(self, "_dbg_dx", None)
                cos = float(np.dot(dx, prev) / (np.linalg.norm(dx) * np.linalg.norm(prev) + 1e-300)) if prev is not None else 0.0
                self._dbg_dx = dx.copy()
                top = np.argsort(-np.abs(r))[:3]
                print(f"        |dx| {np.linalg.norm(dx):.3e} cos(prev) {cos:+.3f}  top |r| dofs {top.tolist()} {np.abs(r)[top].round(1).tolist()}")
            if spec.get("line_search", True):
                # backtracking: the first step length that reduces the
                # residual norm -- or the energy along the step, |dx . r| --
                # failing both, the one that grew the norm least.
                #
                # THE ENERGY TEST IS WHAT LETS A BENT MEMBER STRAIGHTEN. A
                # cantilever released from a 30 mm tip deflection springs back
                # in one Newton step for w and the rotations, but its nodes
                # keep the arc-shortened x for one iteration more, and the
                # axial force that leaves (EA / l times w^2 / l) makes the
                # residual NORM fifteen times larger at the full step. The
                # norm test then took 1/8 of every step, crawled 12% per
                # iteration and failed the release of an elastic beam; the
                # energy along the step is near zero there, as for any full
                # Newton step on a near-quadratic energy (Crisfield's line
                # search; found in review).
                base = self.get()
                norm0 = self.residual_norm(r, free)
                s0 = abs(float(np.dot(dx, r)))
                alpha, best = 1.0, (np.inf, 1.0)
                for _ls in range(6):
                    self.put(base); self.move(alpha * dx)
                    Ft = F_of(); ft = self.force(); rt = Ft - ft; rt[fixed] = 0.0
                    nt = self.residual_norm(rt, free)
                    if np.isfinite(nt) and nt < best[0]:
                        best = (nt, alpha)
                    if np.isfinite(nt) and (nt < norm0 or abs(float(np.dot(dx, rt))) <= 0.5 * s0):
                        break
                    alpha *= 0.5
                else:
                    self.put(base); self.move(best[1] * dx)
                if _DEBUG:
                    print(f"        line search alpha {alpha if alpha >= 1/64 else best[1]:.4f}")
            else:
                self.move(dx)
            # fail fast: a Newton that has not halved the residual in eight
            # iterations is not going to converge from here; a smaller
            # increment is cheaper than the remaining iterations
            history.append(res)
            if it >= 8 and res > 0.5 * min(history[:3]):
                return False, res, it + 1
        return False, res, int(max_iterations)

    def solve_stabilised(self, F_of, fixed: np.ndarray, tol: float, max_rounds=None):
        """Newton by pseudo-time (1.23): a spring of mu times the tangent's
        own diagonal ties every dof to where the round began,

            K + mu D,    r = F - f(q) - mu D (q - q0),

        and the round is solved by Newton to the tolerance; the next round
        starts from there with the spring's anchor moved up to it and mu
        quartered. The springs carry less and less each round and the rounds
        stop when the residual WITHOUT them is within the tolerance: the
        answer is the plain equilibrium, the springs only the way to it
        (pseudo-transient continuation; Abaqus' viscous `stabilize` is the
        same idea with the spring on the increment's velocity).

        FOR A TANGENT THAT IS RIGHT AND ILL-CONDITIONED. A suspension
        bridge's deck whose end bays twist with next to no torsional
        stiffness (a steel box given J = 1e-4 m4, between the tower and the
        first hanger) has a tangent accurate to four digits whose softest
        directions turn a residual into a correction of a hundred metres; the
        spring holds them while the stiff ones settle, and gives them up as
        mu falls. Taken by the static step only after the plain Newton has
        failed an increment, before it is halved: a model that converged
        before converges as before.

        OFF BY DEFAULT, BECAUSE IT CROSSES LIMIT POINTS: a shallow arch whose
        snap-through Newton refuses under load control is carried by the
        rounds onto the far branch, which is the jump a load-controlled
        static analysis must report, not take. It is for a model whose
        stiffness is there but tiny in some direction; the Midas converter
        turns it on for the bridges it writes. Returns (converged, residual,
        iterations), as `solve_level`."""
        spec = self.spec
        if max_rounds is None:
            max_rounds = int(spec.get("stabilise_rounds", 40))
        free = ~fixed
        mu = float(spec.get("stabilise_mu", 1.0e-6))
        D = K = None
        res, its = np.inf, 0
        for _round in range(max_rounds):
            base = self.get()
            q0 = self.s.q.copy()
            if D is None:
                # the springs' stiffness: the start's tangent diagonal, frozen
                # (that tangent is the first iteration's too)
                K = self.tangent(free)
                D = np.abs(K.diagonal())
            ok = False
            first = None
            # a round's own budget: the springs make it a well-posed problem,
            # and `max_iterations` is the plain Newton's
            for _it in range(30):
                F = F_of(); f = self.force()
                spring = mu * D * (self.s.q - q0)
                r = F - f - spring; r[fixed] = 0.0
                plain = F - f; plain[fixed] = 0.0
                res = self.measure(F, f, free, plain)
                if _round and _it == 0 and res <= tol:
                    return True, res, its                # the springs are slack: equilibrium
                if not np.isfinite(res):
                    break
                sres = self.measure(F, f + spring, free, r)
                if sres <= tol:
                    ok = True
                    break
                first = sres if first is None else first
                if _it >= 8 and sres > 0.5 * first:
                    break                                # this round is not converging: a stiffer spring
                try:
                    # the factor's spring is the residual's, mu D, through the
                    # floors' and links' map when there is one
                    lu = self.factor(K if K is not None else self.tangent(free), free, mu, D)
                except RuntimeError:
                    break
                K = None
                dx = lu.solve(r)
                if not np.all(np.isfinite(dx)):
                    break
                self.move(dx)
                self.iterations += 1
                its += 1
            if _DEBUG:
                print(f"      stabilised round {_round}: mu {mu:.1e}, residual without the springs {res:.3e}")
            if not ok:
                # the round itself did not converge: back to its start, a stiffer spring
                self.put(base)
                mu *= 10.0
                if mu > 1.0:
                    return False, res, its
                continue
            mu = max(mu / 4.0, 1.0e-12)
        return False, res, its

    def commit(self):
        s = self.s
        # the last measure was of the converged state: its force joins the
        # peak of the residual floor
        self.accept_scale()
        # the last evaluation was at the converged state: its trial state
        # is the one to keep
        if hasattr(s, "_commit_fibres"):
            s._commit_fibres()
        else:
            for g, c in enumerate(getattr(s, "fiber_committed", []) or []):
                if c.size:
                    c[...] = s.fiber_states[g]
        s.commit_state()
        if self.trial:
            s._trial_mode = False


class _Factor:
    """A factored tangent: `solve(rhs)` gives the correction of the free
    dofs for a residual `rhs` (both over the full dofs). A free dof nothing
    is attached to (a bar's rotation, a parked particle) has an empty row
    and stays where it is. With diaphragms the system is the master one,
    T^T K T on the free masters, and the correction is T times its own."""

    def __init__(self, newton, K, free, mu=0.0, D=None):
        T = newton.T
        self.T = T
        if mu and D is not None:
            # the springs of the stabilised Newton, mu diag(D), on the full
            # dofs: the same springs its residual carries, so that with
            # floors or links they reach the masters as T^T mu D T (review:
            # mu on the master's own diagonal missed what cancels inside a
            # floor, 44 times weaker, and the rounds diverged)
            K = (K + mu * sp.diags(np.asarray(D, float))).tocsc()
            mu = 0.0
        if T is None:
            Km, fm = K, free
        else:
            Km = (T.T @ K @ T).tocsc()
            fm = newton._free_master(free)
        Kf = Km[fm][:, fm]
        d = np.abs(Kf.diagonal())
        live = d > 1e-12 * max(float(d.max()) if d.size else 0.0, 1e-300)
        self.idx = np.flatnonzero(fm)[live]
        self.n = Km.shape[0]
        # SuperLU in its symmetric mode (1.22): the tangent is symmetric to the
        # finite differences' round-off, and the ordering of A + A^T with
        # diagonal pivots (off the diagonal only where a pivot is under 0.1 of
        # its column) fills it about a tenth as much as the default (COLAMD
        # with partial pivoting): on a converted rigid-frame bridge of 16,000
        # dofs 0.03 s a factorisation instead of 0.39 -- where its Newton spent
        # 60% of its time. SCALED TO A UNIT DIAGONAL first, S K S with S =
        # |diag K|^-1/2: a frame in millimetres (or of 5 cm members) has its
        # rotation-translation couplings far above its translational
        # diagonals, the diagonal pivots were refused, and the fill was ten
        # times the default's instead of a tenth (found in review); scaled, the
        # fill is the same whatever the units
        self.scale = 1.0 / np.sqrt(d[live])
        S = sp.diags(self.scale)
        A = (S @ Km[self.idx][:, self.idx] @ S).tocsc()
        if mu:
            # K + mu diag|K|: on the unit diagonal, A + mu I
            A = (A + mu * sp.identity(A.shape[0], format="csc")).tocsc()
        self.lu = spla.splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.1,
                            options={"SymmetricMode": True})

    def solve(self, rhs):
        r = rhs if self.T is None else self.T.T @ rhs
        x = np.zeros(self.n)
        x[self.idx] = self.scale * self.lu.solve(self.scale * r[self.idx])
        return x if self.T is None else self.T @ x


def _failures(s) -> int:
    f = getattr(s, "_committed_failures", None)
    return f() if f is not None else 0


def stall_message(where: str, res: float, tol: float) -> str:
    """Newton's corrections at the round-off of the coordinates and the
    residual standing still above the tolerance: the floor of the arithmetic,
    not a limit point, wherever the solve was."""
    return (f"{where}: Newton stalled with residual {res:.3e} against a tolerance of {tol:.1e}: its "
            "corrections are at the round-off of the coordinates, so this is the floor of the "
            f"arithmetic, not a limit point. Loosen solver.tolerance (or the step's) to about {10 * res:.0e}")


def _settle_ruptures(newton, F_of, fixed, tol, n_before, log, where):
    """After a commit: a steel fibre that passed its rupture strain in the
    ACCEPTED state is cut by the commit (steel02 reads rupture against the
    accepted state), and the positions just solved no
    longer balance the forces. Solve the same level again, and again while
    that cascades, as staged relaxation does; only an equilibrium is
    recorded (found in review, v1.0: the static solver recorded
    the cut forces against the intact positions, 70 kN out of balance)."""
    s = newton.s
    count = getattr(s, "_committed_failures", None)
    if count is None:
        return
    n = count()
    redo = 0
    while n > n_before:
        redo += 1
        if redo > 50:
            raise ConvergenceFailure(f"{where}: fibre ruptures still cascading after 50 re-solves")
        log(f"  rupture       {where}: {n} point(s) failed; re-solving the level without them (pass {redo})")
        ok, res, _its = newton.solve_level(F_of, fixed, tol)
        if not ok and newton.stalled:
            raise ConvergenceFailure(stall_message(where, res, tol))
        if not ok:
            raise ConvergenceFailure(f"{where}: no equilibrium after {n} fibre point(s) ruptured "
                                     f"(residual {res:.3e}) -- the section has lost the load")
        newton.commit()
        n_before, n = n, count()


def relax_increment(newton, fixed, moving, prescribed, groups, ground, target, tol, spec, log, where,
                    step=None, report=None):
    """The fallback of an increment Newton could not take (1.7):
    from the last converged state, the level `target` is reached by a
    dynamic relaxation -- the explicit scheme with fictitious masses and
    kinetic damping of `mode: relaxation`, the solver's own `_relax_stage`
    -- and Newton goes on from the relaxed state. A localisation (concrete
    crushing concentrating into one element near a pushover's peak) is a
    jump in the equilibrium path that no tangent follows and a real motion
    passes; the relaxation is that motion. Returns
    (converged, residual, iterations). The solver's static settings (its
    physical mass, step, damping, counters) are put back afterwards."""
    s = newton.s
    # rigid diaphragms are taken since 1.9: the relaxation's residual is the
    # floors' own equations (`ConstraintLayer.residual_norm`), as
    # Newton's is (`residual_norm`)
    sol = s.p.solver
    step = step or {}
    safety = float(sol.relaxation_safety)
    budget = int(spec.get("fallback_iterations") or step.get("max_steps_per_load")
                 or sol.max_steps_per_load)
    # the frames' default ramp is a fifth of the budget, measured for a
    # whole staged run; a fallback blends its level in over 200 at most
    ramp = int(step.get("stage_ramp_iterations") or sol.stage_ramp_iterations or
               (1 if hasattr(s, "_commit_fibres") else min(max(budget // 5, 1), 200)))
    keep = {k: getattr(s, k, None) for k in ("mass", "dt", "damp_kind", "_half", "ke_prev", "n_resets",
                                             "step", "_loop", "_loop_measures", "_stage_start_q")}
    # the scales as the instance holds them: absent (None) is kept by the
    # relaxation, as it was before the frame solvers gave them a class
    # default of 0.0
    keep.update({k: vars(s).get(k) for k in ("_force_scale", "_motion_scale")})
    # the ledger does not audit a fictitious transient (a static step among
    # dynamic ones has it on)
    led = (s.diagnostics or {}).get("energy") if isinstance(getattr(s, "diagnostics", None), dict) else None
    led_on = getattr(led, "available", False)
    # the relaxation samples its residual at its progress checks; a static
    # run's series are Newton's, one per increment (1.9)
    n_series = (len(s.residuals), len(s.kinetic))
    log(f"  fallback      {where}: Newton could not take the increment; relaxing this level "
        f"(up to {budget} iterations)")
    # A STOP OR A CHECKPOINT IN THE MIDDLE OF THE FALLBACK (found in the
    # review of 1.9): its reports reach the progress callback, where the API
    # writes checkpoints and takes a stop. Mid-relaxation the solver holds a
    # half-relaxed q, the fictitious mass and the relaxation's iteration
    # count; a checkpoint of that resumed Newton from the wrong state. No
    # checkpoint is written while `_in_fallback` is set, and a stop puts the
    # last converged increment back before it goes on up.
    base = newton.get()
    s._in_fallback = True
    try:
        if led_on:
            led.available = False
        row = np.asarray(abs(s.K).sum(axis=1)).ravel()
        s.mass = np.maximum(row / safety ** 2, 1e-20)
        s.dt = 1.0
        s.damp_kind = "kinetic"
        s.v[:] = 0.0; s._half = True; s.ke_prev = 0.0
        if not hasattr(s, "n_resets") or s.n_resets is None:
            s.n_resets = 0
        if hasattr(s, "_reorthonormalize"):
            s._reorthonormalize()
        if hasattr(s, "_refresh_fictitious_mass"):
            s._refresh_fictitious_mass()           # the line solvers: the current geometry's rows
        if newton.trial:
            s._trial_mode = True
        F = s._force(groups, ground, target)
        start = [s.q[dofs].copy() for dofs, _v, _a, _m in prescribed]
        converged, resid, _kr, _ke = s._relax_stage(F, fixed, moving, prescribed, target, start, tol, budget,
                                                    # asked every 500 iterations, as a staged
                                                    # relaxation asks: a long fallback can be
                                                    # stopped (1.9)
                                                    ramp, 0, 500 if report is not None else budget + 1,
                                                    report, 0.0, 0,
                                                    what="static fallback relaxation")
        its = int(s.step - keep["step"])
    except StopAnalysis:
        newton.put(base)
        raise
    finally:
        s._in_fallback = False
        for k, val in keep.items():
            if val is not None or k in ("_loop", "_stage_start_q"):
                setattr(s, k, val)
        s.v[:] = 0.0
        del s.residuals[n_series[0]:], s.kinetic[n_series[1]:]
        if led_on:
            led.available = True
    newton.fallbacks += 1
    return converged, float(resid), its


def _polish(newton, F_of, fixed, tol, log, where):
    """After a relaxed level (1.7): the relaxation converged on a residual
    one iteration stale, and Newton's own measure of the relaxed state is
    a few times the tolerance. A few Newton iterations from there stay on
    the branch the relaxation landed on and bring it under; if they do
    not, the relaxed state is kept and the residual said. Returns the
    residual recorded for the level."""
    s = newton.s
    base = newton.get()
    if newton.trial:
        s._trial_mode = True
    ok, res, _its = newton.solve_level(F_of, fixed, tol)
    if ok:
        return res
    newton.put(base)
    if newton.trial:
        s._trial_mode = True
    F = F_of(); f = newton.force(); r = F - f; r[fixed] = 0.0
    res = newton.measure(F, f, ~fixed, r)
    if res > tol:
        log(f"  fallback      {where}: Newton does not polish the relaxed state (residual {res:.2e} "
            f"against {tol:.1e}); the level is kept on the relaxation's own criterion")
    return res


def _settle_breaks(newton, F_of, fixed, tol, t, log, where):
    """After a commit (1.7): a member whose demand reached
    its capacity in the ACCEPTED equilibrium breaks -- it is taken out, as
    a dynamic step takes it out -- and the same level is solved again
    without it, and again while that cascades (the alternate-path analysis
    of a progressive collapse, one member at a time). No equilibrium after
    a break is the structure losing its load, and said so. `t` is the
    event's place on the result's time axis (the step plus its progress)."""
    s = newton.s
    b = getattr(s, "breaking", None)
    if b is None or not b.can_break or not hasattr(s, "_break"):
        return 0
    broke = 0
    for redo in range(1, 52):
        # `end_actions` are those of the last force evaluation: the accepted
        # state's (solve_level returns right after it)
        gone = b.check(float(t), s.end_actions, removals=False, whole=True)
        if not len(gone):
            return broke
        if redo > 50:
            raise ConvergenceFailure(f"{where}: members still breaking after 50 re-solves")
        broke += len(gone)
        s._break(gone)
        if hasattr(s, "apply_pending_splits"):
            s.apply_pending_splits(float(t))
        b.diagnostics(getattr(s, "diagnostics", None))
        log(f"  failure       {where}: member(s) {', '.join(map(str, gone[:8]))}"
            f"{' ...' if len(gone) > 8 else ''} reached their capacity; re-solving the level without them "
            f"(pass {redo})")
        if newton.trial:
            s._trial_mode = True
        ok, res, _its = newton.solve_level(F_of, fixed, tol)
        if not ok and newton.stalled:
            raise ConvergenceFailure(stall_message(where, res, tol))
        if not ok:
            raise ConvergenceFailure(f"{where}: no equilibrium after member(s) {', '.join(map(str, gone[:8]))} "
                                     f"broke (residual {res:.3e}) -- the structure has lost its load")
        newton.commit()
    return broke


def amplitude_note(s, groups, lam, pattern: bool = False) -> str:
    """What the step's loads have reached at progress (or arc factor) `lam`,
    in the units the case wrote them in: "load factor
    0.725" was the step's PROGRESS, and with `push: [[0, 0], [1, 12]]` the
    loads stood at 8.7 times their written value. Under arc length the
    loads are one pattern, the step's full loads times lambda."""
    amps = getattr(s, "amplitudes", None) or {}
    parts = []
    # a dict keyed by the amplitude's name (frames), or (amplitude, load)
    # pairs (fibre beams)
    items = list(groups.keys()) if isinstance(groups, dict) else [
        g[0] if isinstance(g, (tuple, list)) and g else g for g in (groups or [])]
    seen = set()
    for key in items:
        if isinstance(key, str):
            name, a = key, amps.get(key)
        else:
            name, a = getattr(key, "name", None), key
        if a is None or not callable(a) or name in (None, "hold") or name in seen:
            continue
        seen.add(name)
        try:
            v = float(a(1.0)) * float(lam) if pattern else float(a(lam))
        except Exception:
            continue
        parts.append(f"{name} x{v:.4g}")
    return f" (step loads at {', '.join(parts)} of their written value)" if parts else ""


#: an arc this far below the first one, accepted this many times in a row
#: without a cutback, is doubled
_ARC_STALL_FRACTION, _ARC_STALL_RUN = 0.25, 3


def resize_arc(dl, its, cuts, target, good, dl0):
    """The next arc after an accepted increment: (dl, good).

    sqrt(target / taken) iterations, as before, but with two changes.
    An increment within ONE iteration of the target keeps
    its arc: on a fibre section past yield every increment took 5 or 6
    Newton iterations whatever its size, sqrt(5/6) shrank the arc at every
    other increment, and nothing ever grew it back -- the corroded pushover
    of the scenario review crawled 1.2e-5 m per increment for 150 increments
    and stopped at max_increments. And when the arc has fallen below a
    quarter of the first one while increment after increment converges
    without a cutback, the iteration count is not answering to the arc (it
    took 6 to 8 at every size there): the arc is doubled (a failure halves it
    again, as any)."""
    scale = float(np.clip(np.sqrt(target / max(its, 1)), 0.5, 1.0 if cuts else 2.0))
    if not cuts and its <= target + 1:
        scale = max(scale, 1.0)
    good = 0 if cuts else good + 1
    if dl0 and good >= _ARC_STALL_RUN and dl * scale < _ARC_STALL_FRACTION * dl0:
        scale, good = 2.0, 0
    return dl * scale, good


def arc_length_step(newton, k, fixed, moving, prescribed, groups, ground, spec, tol, record, log,
                    resume=None):
    """One analysis step by Crisfield's cylindrical arc length.

    The step's load is taken as a pattern, F(lambda) = F(0) + lambda (F(1) -
    F(0)), and so are the displacements its constraints impose on the
    translations -- the two things `load_steps` would ramp -- and lambda is
    an unknown beside the displacements. Each increment moves a fixed
    distance dl along the path:

        | W dx |^2 = dl^2          (cylindrical: the load factor is not in it)

    with W weighting a rotation by the model's size, so a metre and a radian
    of a 10 m frame count alike -- or, with `control` (1.3), W picking out one
    node's translation: displacement control with lambda unknown. The predictor follows the tangent in the
    direction of the last increment (so the path turns back at a limit point
    instead of reversing); every Newton correction splits into a residual
    part and a load part, dx = dx_r + dlam dx_t, and dlam is the root of the
    constraint that keeps the increment pointing where it was going. The arc
    grows or shrinks by sqrt(target / taken) iterations; an increment that
    does not converge is retried on half the arc.

    The step ends when lambda reaches 1 (the last increment lands on it by
    load control) or when `stop` is reached; running out of
    `max_increments` first is a failure. Returns the number of increments,
    or None when the pattern moves nothing (only held loads, a prescribed
    rotation) and the step is to be run by load control instead."""
    s = newton.s
    arc = spec["arc_length"]
    if any(callable(F) for _a, F in groups):
        # a travelling load is not a pattern: F(lambda) mixed its first and
        # last stations instead of moving the axles
        raise ValueError("a moving_load under static.arc_length: its factor is its position, not "
                         "its size; sweep it under load control")
    free = ~fixed
    F0 = s._force(groups, ground, 0.0)
    F1 = s._force(groups, ground, 1.0)
    dF = F1 - F0
    base = newton.get()
    s._apply_bc(fixed, moving, prescribed, 0.0); u0 = s.q.copy()
    s._apply_bc(fixed, moving, prescribed, 1.0); u1 = s.q.copy()
    newton.put(base)
    trans = np.zeros(s.ndof, bool)
    trans[s.dofs.node_dofs(np.arange(s.nn), (s.dofs.components[0], s.dofs.components[1],
                                              s.dofs.components[2])).ravel()] = True
    drive = fixed & trans
    du = np.zeros(s.ndof); du[drive] = (u1 - u0)[drive]
    size = float(np.ptp(s.mesh.nodes, axis=0).max()) or 1.0
    W = np.where(trans, 1.0, size)
    W[fixed] = 0.0
    W_state = W
    control = arc.get("control")
    if control:
        # DISPLACEMENT CONTROL (1.3): the arc measured on one
        # node's translation, OpenSees' DisplacementControl with the load
        # factor unknown. Every increment moves that point on by dl, so an
        # equilibrium that brings it back -- the elastic unloading from the
        # last state, which the whole-state arc may settle on past a peak
        # where fibres crush -- is not on the arc. A snap-back of that
        # displacement cannot be followed (see the jump test below).
        cnodes = np.asarray(s._region(control["region"]), int)
        if len(cnodes) != 1:
            raise ValueError(f"arc length, step {k + 1}: control.region {control['region']!r} holds "
                             f"{len(cnodes)} nodes; the control is one node's displacement")
        cdof = s.dofs.node_dofs(cnodes, (s.dofs.components["xyz".index(control.get("dof", "z"))],)).ravel()
        if fixed[cdof].any():
            raise ValueError(f"arc length, step {k + 1}: the control displacement ({control['region']}, "
                             f"{control.get('dof', 'z')}) is held by a constraint in this step; control "
                             "a point the load pattern pushes")
        W = np.zeros(s.ndof)
        W[cdof] = 1.0

    def set_lambda(lam):
        c = min(max(lam, 0.0), 1.0)
        # a temperature or an imposed strain rides in `_force` (it refreshes
        # the solver's imposed deformation as a side effect): without this
        # the one refreshed last -- at progress 1, building F1 -- stayed on
        # for the whole arc (found in review)
        s._force(groups, ground, c)
        s._apply_bc(fixed, moving, prescribed, c)
        s.q[drive] = u0[drive] + lam * du[drive]
        if newton.ties is not None and newton.ties.links is not None:
            # a driven link master's slaves follow it where it now is
            newton.ties.correct()

    def solve_both(K):
        lu = newton.factor(K, free)
        qv = dF - K @ du
        return lu, lu.solve(qv)

    # THE PATH STARTS FROM AN EQUILIBRIUM. A step can open out of balance
    # at lambda = 0 -- a prestress built into the fibres, the previous
    # step's loads under a changed support -- and the arc's first predictor
    # would then be spent on that. Settle it at lambda = 0 first. (Not on a
    # restart: the path is under way, and settling at 0 would unload it.)
    if resume is None:
        set_lambda(0.0)
        ok0, res0, _its0 = newton.solve_level(lambda: F0.copy(), fixed, tol)
        if not ok0 and newton.stalled:
            raise ConvergenceFailure(stall_message(f"arc length, step {k + 1}, load factor 0", res0, tol))
        if not ok0:
            raise ConvergenceFailure(f"arc length, step {k + 1}: no equilibrium at load factor 0 to start "
                                     f"from (residual {res0:.3e})")
        newton.commit()
    stop = arc.get("stop")
    if stop:
        sidx = s.dofs.node_dofs(s._region(stop["region"]),
                                (s.dofs.components["xyz".index(stop.get("dof", "z"))],)).ravel()
        # the stop is reached when the displacement CROSSES `value` from the
        # side the step starts on -- not when it merely lies beyond it, which
        # stopped a step after one increment that started past the value
        # (found in review)
        side0 = (np.sign(float(s.q[sidx].mean()) - stop["value"]) if resume is None
                 else float(resume["side0"]))
        if side0 == 0.0:
            raise ValueError(f"arc length, step {k + 1}: the stop value {stop['value']} is where the "
                             "step starts")
    else:
        side0 = 0.0
    lam, dl, prev, inc, cuts = 0.0, None, None, 0, 0
    rate_prev = 0.0                      # the state's speed per unit of control
    good, dl0 = 0, 0.0                   # the stall recovery's (resize_arc)
    land = False
    target_its = int(arc.get("target_iterations", 5))
    if spec.get("tangent", "full") == "modified":
        # a modified Newton converges linearly: it takes about twice the
        # iterations for the same arc, and holding it to the full Newton's
        # target shrank the arc to a tenth
        target_its *= 2
    max_it = int(spec.get("max_iterations", 30))
    if resume is not None:
        # where the path was when the checkpoint was taken: just after an
        # accepted increment, before its arc was resized
        lam, dl, inc = float(resume["lam"]), float(resume["dl"]), int(resume["inc"])
        prev = resume.get("prev")
        if control and prev is not None:
            # the speed the restart carries on from: the checkpoint's last increment
            rate_prev = float(np.linalg.norm(W_state * prev)) / max(float(np.linalg.norm(W * prev)), 1e-300)
        its_r, cuts_r = int(resume["its"]), int(resume["cuts"])
        good, dl0 = int(resume.get("good", 0) or 0), float(resume.get("dl0", 0.0) or 0.0)
        if abs(lam - 1.0) <= 1e-9:
            return inc
        dl, good = resize_arc(dl, its_r, cuts_r, target_its, good, dl0)
        if stop and np.sign(float(s.q[sidx].mean()) - stop["value"]) != side0:
            return inc
    while inc < int(arc.get("max_increments", 200)):
        saved = newton.get()
        n_before = _failures(s)
        K = newton.tangent(free)
        try:
            lu, dxt = solve_both(K)
        except RuntimeError:
            raise ConvergenceFailure(f"arc length, step {k + 1}: the tangent is singular at load factor "
                                     f"{lam:.6g} (a bifurcation, or a mechanism)")
        nt = float(np.linalg.norm(W * dxt))
        if dl is None:
            whole = float(np.linalg.norm(W_state * dxt))
            if control and whole > 1e-300 and not nt > 1e-10 * whole:
                # relative: a symmetric model moves the dof it should not by
                # round-off, 1e-19 against 0.07 (found in review)
                raise ValueError(
                    f"arc length, step {k + 1}: the load pattern does not move the control displacement "
                    f"({control['region']}, {control.get('dof', 'z')}); control a point it pushes")
            if not nt > 1e-300:
                # nothing free moves under the pattern: held loads only, a
                # prescribed rotation (not in the pattern), no free dofs.
                # An arc of length zero "converged" at lambda = 0 for
                # max_increments and reported CONVERGED (found in review,
                # v1.0); load control is the answer here
                log(f"  arc length    step {k + 1}: the load pattern moves no free dof; "
                    "the step runs by load control")
                return None
            dl = float(arc.get("initial", 0.1)) * nt
            dl0 = dl
        sign = 1.0
        if prev is not None:
            dot = float(np.dot(W * dxt, W * prev))
            sign = -1.0 if dot < 0.0 else 1.0
        dlam = sign * dl / max(nt, 1e-300)
        if _DEBUG and stop:
            print(f"    predictor: lambda {lam:.6g}, dlam {dlam:.3e}, dw {dlam * float(dxt[sidx].mean()):.3e}")
        if land or (lam < 1.0 and lam + dlam >= 1.0 - 1e-12):
            # the last increment lands on the full load by load control
            F_now = F0 + lam * dF
            newton.predict(lambda: set_lambda(1.0), F_now, F1, fixed)
            set_lambda(1.0)
            ok, res, its = newton.solve_level(lambda: F1.copy(), fixed, tol)
            land = False
            if ok:
                newton.commit(); lam = 1.0; inc += 1
                _settle_ruptures(newton, lambda: F1.copy(), fixed, tol, n_before, log,
                                 f"step {k + 1}, load factor 1")
                _settle_breaks(newton, lambda: F1.copy(), fixed, tol, k + 1.0, log, f"step {k + 1}, load factor 1")
                s._static_state = {"kind": 1, "lam": 1.0, "dl": dl, "inc": inc, "its": its,
                                   "cuts": cuts, "side0": side0}
                record(newton.s._frame_time(k, lam) if hasattr(newton.s, "_frame_time") else k + lam, res, F1)
                return inc
            newton.put(saved)
            if newton.trial:
                s._trial_mode = True
            if newton.stalled:
                # a shorter arc lands on the same load factor 1 at the same
                # floor
                raise ConvergenceFailure(stall_message(f"arc length, step {k + 1}, landing on load factor 1",
                                                       res, tol))
            dl *= 0.5
            cuts += 1
            if cuts > int(spec.get("cutbacks", 10)):
                raise ConvergenceFailure(f"arc length, step {k + 1}: could not land on load factor 1")
            continue
        Dx = dlam * dxt
        Dlam = dlam
        newton.move(np.where(free, Dx, 0.0)); set_lambda(lam + Dlam)
        converged, res, its = False, np.inf, 0
        history = []
        for it in range(max_it):
            F = F0 + (lam + Dlam) * dF
            f = newton.force()
            r = F - f; r[fixed] = 0.0
            res = newton.measure(F, f, free, r)      # v1.0
            if _DEBUG:
                print(f"      arc it {it}: residual {res:.3e}, lambda {lam + Dlam:.6g}")
            if not np.isfinite(res):
                break
            if res <= tol:
                converged, its = True, it
                break
            # fail fast, as the load-controlled Newton does: eight iterations
            # that have not halved the residual will not converge from here
            history.append(res)
            if it >= 8 and res > 0.5 * min(history[:3]):
                break
            # `modified`: the increment's predictor tangent, kept while it
            # still halves the residual each iteration and refreshed when it
            # stops doing so (and with every new increment or retry). It read
            # `it > 0 or full` -- a fresh tangent at every iteration but the
            # first, so `modified` was `full` with one stale step and cost 4.5
            # times the evaluations (found in review)
            slow = len(history) >= 2 and history[-1] > 0.5 * history[-2]
            if spec.get("tangent", "full") == "full" or slow:
                K = newton.tangent(free)
                try:
                    lu, dxt = solve_both(K)
                except RuntimeError:
                    break
            dr = lu.solve(r)
            a = float(np.dot(W * dxt, W * dxt))
            e = W * (Dx + dr)
            b = 2.0 * float(np.dot(e, W * dxt))
            c = float(np.dot(e, e)) - dl * dl
            disc = b * b - 4.0 * a * c
            if a <= 0.0 or disc < 0.0:
                break                                  # no point of the arc: a shorter one
            roots = ((-b + np.sqrt(disc)) / (2.0 * a), (-b - np.sqrt(disc)) / (2.0 * a))
            # the root that keeps the increment pointing where it was going
            pick = max(roots, key=lambda g: float(np.dot(W * (Dx + dr + g * dxt), W * Dx)))
            d = dr + pick * dxt
            Dx = Dx + d; Dlam += pick
            newton.iterations += 1
            newton.move(np.where(free, d, 0.0)); set_lambda(lam + Dlam)
            its = it + 1
        turned = False
        if converged and prev is not None and float(np.dot(W * Dx, W * prev)) <= 0.0:
            # AN EQUILIBRIUM BEHIND US IS NOT THE NEXT POINT OF THE PATH. On a
            # path-dependent section the corrector can wander (residual near 1
            # for several iterations) and settle on the elastic UNLOADING
            # from the last accepted state -- a true equilibrium, and the
            # path back down. case_24g did that at its limit point with the
            # OpenSees steel02 and unloaded to lambda < 0.
            # The predictor always points along the previous increment; an
            # accepted increment must too. A shorter arc instead.
            converged, turned = False, True
        jumped = 0.0
        if converged and control:
            # A JUMP IS NOT A STEP ALONG THE PATH. Under control the arc holds
            # one displacement, and where the path SNAPS BACK in it there is
            # no equilibrium on the arc near the last state; the corrector can
            # then converge on another branch far away -- a 12 mm increment of
            # a spring above a snap-through truss moved its apex 135 mm and
            # the load factor from 1.0 to -0.66, and case_24g's second peak
            # was "passed" by one 6.8 mm increment that 2 mm increments show
            # to be a jump (found in review). Per unit of the
            # controlled displacement the state may move a few times as fast
            # as it did over the last increment, not more; at a snap-back
            # that rate grows without bound, and the arc is halved until the
            # cutbacks run out, which is the answer there.
            rate = float(np.linalg.norm(W_state * Dx)) / max(float(np.linalg.norm(W * Dx)), 1e-300)
            if _DEBUG:
                print(f"    control: the state moved {rate / max(rate_prev, 1e-300):.3g} x as fast as before")
            if rate_prev > 0.0 and rate > _JUMP * rate_prev:
                converged, jumped = False, rate / rate_prev
        if not converged:
            newton.put(saved)
            if newton.trial:
                s._trial_mode = True
            cuts += 1
            if cuts > int(spec.get("cutbacks", 10)):
                raise ConvergenceFailure(f"arc length, step {k + 1}: no convergence at load factor "
                                         f"{lam:.6g}{amplitude_note(s, groups, lam, pattern=True)} "
                                         f"after {cuts - 1} cutbacks (residual {res:.3e})"
                                         + (f"; with `control` on {control['region']} ({control.get('dof', 'z')} "
                                            f"= {float(s.q[cdof].mean()):.6g}) this is where the controlled "
                                            "displacement itself may turn back (a snap-back), which displacement "
                                            "control cannot follow" if control else ""))
            log(f"  cutback       step {k + 1}, arc length at load factor {lam:.4g}: "
                + ("the corrector found the path back, not the next point; " if turned
                   else (f"the state moved {jumped:.3g} times as fast per unit of the control as over the "
                         "last increment (a jump to another branch); "
                         if jumped else f"residual {res:.2e}; ")) + "arc halved")
            dl *= 0.5
            continue
        if lam < 1.0 - 1e-12 and lam + Dlam > 1.0 + 1e-9:
            # the corrector carried the increment past the full load (on a
            # stiffening branch a short arc is a large load step): go back
            # and land on it by load control instead
            newton.put(saved)
            if newton.trial:
                s._trial_mode = True
            land = True
            continue
        if _DEBUG and stop:
            print(f"    converged: Dlam {Dlam:.3e}, Dw {float(Dx[sidx].mean()):.3e}, its {its}")
        newton.commit()
        lam += Dlam; prev = Dx; inc += 1
        if control:
            rate_prev = float(np.linalg.norm(W_state * Dx)) / max(float(np.linalg.norm(W * Dx)), 1e-300)
        lam_now = lam
        _settle_ruptures(newton, lambda: F0 + lam_now * dF, fixed, tol, n_before, log,
                         f"step {k + 1}, load factor {lam:.6g}"
                         + amplitude_note(s, groups, lam, pattern=True))
        _settle_breaks(newton, lambda: F0 + lam_now * dF, fixed, tol, k + min(max(lam, 0.0), 1.0), log,
                       f"step {k + 1}, load factor {lam:.6g}" + amplitude_note(s, groups, lam, pattern=True))
        s._static_state = {"kind": 1, "lam": lam, "dl": dl, "inc": inc, "its": its, "cuts": cuts,
                           "side0": side0, "prev": np.array(prev, float), "good": good, "dl0": dl0}
        record(newton.s._frame_time(k, lam) if hasattr(newton.s, "_frame_time") else k + lam, res, F0 + lam * dF)
        if abs(lam - 1.0) <= 1e-9:
            # accepted within the overshoot tolerance: this IS the full load,
            # and the loop would otherwise carry on beyond it
            return inc
        # an arc that has just been halved does not grow straight back: the
        # length that failed would fail again, and every other increment
        # would be a wasted one
        dl, good = resize_arc(dl, its, cuts, target_its, good, dl0)
        cuts = 0
        if stop:
            now = float(s.q[sidx].mean())
            if np.sign(now - stop["value"]) != side0:
                log(f"  arc length    step {k + 1}: stop reached ({now:.6g}) at load factor {lam:.6g}")
                return inc
    # neither the full load nor the stop: that is not an answer (it was
    # reported CONVERGED, and the next step jumped to the full held load --
    # found in review)
    raise ConvergenceFailure(
        f"arc length, step {k + 1}: {inc} increments (max_increments) reached load factor {lam:.6g}, "
        + (f"neither the full load nor the stop ({stop.get('dof', 'z')} = {float(s.q[sidx].mean()):.6g} "
           f"of {stop['value']:.6g} at {stop['region']}); raise max_increments or the initial arc"
           + ("; under `control` a point where the controlled displacement itself turns back (a "
              "snap-back) cannot be passed -- stop before it, or measure the arc on the whole state"
              if control else "")
           if stop else
           "and the full load was not reached; raise max_increments or the initial arc, or set a stop"
           + ("; under `control` a snap-back of the controlled displacement cannot be passed"
              if control else "")))


def _static_step(s, newton, k, r, ctx):
    """One step of the static solution (the body of `run`'s loop, a function
    since 1.2 so that a static step can sit among dynamic ones)."""
    sol = s.p.solver; spec = ctx["spec"]; tol = ctx["tol"]; every = ctx["every"]
    on_output, on_progress = ctx["on_output"], ctx["on_progress"]
    rst = (r.static or None) if r is not None else None
    s.step_index = k
    fixed, moving, prescribed, groups, ground = s.step_data(k)
    s.current_fixed = fixed
    if r is None:
        s.step_start = s.step
        s._static_state = None
    else:
        r.after_step_data(s)
    s._loop = [k, s.step_start, 0, s.step_start, 0]
    step = s.steps[k]
    # the step's own tolerance first, as every relaxation loop reads it;
    # this one read the solver's alone (found in review)
    tol_k = float(step.get("tolerance") or tol)
    arc = spec.get("arc_length")
    if arc and (not arc.get("steps") or step.get("name") in arc["steps"]):
        # the progress callback was told (ctx["count"], ctx["count"]): 100% from the
        # first increment (found in review). Out of this
        # step's budget of increments -- an arc-length path has no
        # other end to ctx["count"] to.
        budget = ctx["count"] + int(arc.get("max_increments", 200))
        first, last_res = ctx["count"], [0.0]
        framed = [True]

        def record(t, res, F):
            ctx["count"] += 1; s.step += 1
            last_res[0] = res
            if s._static_state is not None:
                s._static_state["count"] = ctx["count"]
            s.residuals.append(res); s.kinetic.append(0.0)
            s.frame_equilibrium = True
            # monitors at every increment, a frame every `output.every`
            # (it wrote every increment whatever `every` said,
            # v1.0); the step's last increment gets one below
            framed[0] = ctx["count"] % every == 0
            s._record(t, F, on_output if framed[0] else None)
            # asked after the frame is written
            if on_progress and on_progress(ctx["count"], budget, res, 0.0, ctx["count"]) is False:
                from .core.diagnostics import StopAnalysis
                raise StopAnalysis()
        done = arc_length_step(newton, k, fixed, moving, prescribed, groups, ground, spec, tol_k,
                               record, s.log,
                               resume=rst if rst and rst.get("kind") == 1 else None)
        if done is not None and not framed[0] and on_output and s.times:
            on_output(s, s.step, s.times[-1], s._cell_actions())
            framed[0] = True
        if done is not None:
            s._static_state = None
            # the step is done: said as done. Its path ended before the
            # budget of increments, and the last report stood at
            # ctx["count"] / budget -- 15% for a finished Lee frame
            if on_progress and ctx["count"] > first and ctx["count"] < budget \
                    and on_progress(ctx["count"], ctx["count"], last_res[0], 0.0, ctx["count"]) is False:
                from .core.diagnostics import StopAnalysis
                raise StopAnalysis()
            return
    n_inc = int(step.get("load_steps") or sol.load_steps or 10)
    lam, dl, cuts = 0.0, 1.0 / n_inc, 0
    inc = 0
    # NO INCREMENT SMALLER THAN `cutbacks` HALVINGS OF THE STATED ONE.
    # The budget of halvings is reset by every success, and approaching
    # a limit point a success always comes: the increment shrank across
    # successes to 4e-17, lam + dl rounded to lam, that "increment"
    # converged, the next doubled one failed, and so on for ever -- a
    # monitor row each time (Lee's frame, a two-bar truss under load
    # control). The smallest increment the budget allows
    # one increment is the floor for all of them.
    dl_min = (1.0 / n_inc) * 0.5 ** int(spec.get("cutbacks", 10))
    if rst and rst.get("kind") == 0:
        lam, dl, inc = float(rst["lam"]), float(rst["dl"]), int(rst["inc"])
        relaxing = bool(rst.get("relaxing", 0))
    else:
        relaxing = False
    max_cuts = int(spec.get("cutbacks", 10))
    report = None
    if on_progress is not None:
        def report(resid, ke):
            # where the step is (the stated levels done), as Newton reports
            return on_progress(int(round(lam * n_inc)), n_inc, resid, ke, inc)
    stab_tried = set()
    while lam < 1.0 - 1e-12:
        # never step over a stated level: after a cutback the increments
        # climb back to the grid, so the frames are the levels asked for
        level = (np.floor(lam * n_inc + 1e-9) + 1.0) / n_inc
        target = level if lam + dl >= level - 1e-9 else lam + dl
        target = min(1.0, target)
        saved = newton.get()
        n_before = _failures(s)
        if relaxing:
            # THE RELAXATION HAS TAKEN THE STEP OVER (1.7): once Newton could
            # not take a level, the levels after it are relaxed too. Newton
            # converges past a localisation jump -- to the branch it finds,
            # which on a softening section is not the one the transient
            # follows (case_25's peak came out 36% high on it); the
            # relaxation's path is the dynamics', level after level
            target = level
            where = f"step {k + 1}, load factor {target:.6g}" + amplitude_note(s, groups, target)
            ok, res, its = relax_increment(newton, fixed, moving, prescribed, groups, ground, target,
                                           tol_k, spec, s.log, where, step, report)
            if not ok:
                newton.put(saved)
                if newton.trial:
                    s._trial_mode = True
                msg = (f"static step {k + 1}: the relaxation that took over from Newton did not converge "
                       f"at load factor {target:.6g}{amplitude_note(s, groups, target)} in {its} iterations "
                       f"(residual {res:.3e}, tolerance {tol_k:.1e}); raise solver.max_steps_per_load "
                       "(static.fallback_iterations) or use mode: relaxation")
                finish_diagnostics(s.diagnostics, "MAX_STEPS", step=s.step, residual=res,
                                   load_step=k + 1, message=msg)
                raise ConvergenceFailure(msg)
            s.log(f"  fallback      {where}: relaxed in {its} iterations (residual {res:.2e})")
            res = _polish(newton, lambda: s._force(groups, ground, target), fixed, tol_k, s.log, where)
        else:
            F_now = s._force(groups, ground, lam)
            F_next = s._force(groups, ground, target)
            K0 = newton.predict(lambda: s._apply_bc(fixed, moving, prescribed, target),
                                F_now, F_next, fixed)
            s._apply_bc(fixed, moving, prescribed, target)
            ok, res, its = newton.solve_level(lambda: s._force(groups, ground, target), fixed, tol_k, K0)
            if not ok and not newton.stalled and spec.get("stabilise", False) and level not in stab_tried:
                # (once per stated level: at a real limit point the rounds
                # would otherwise run before every cutback -- review)
                stab_tried.add(level)
                # an ill-conditioned tangent before a limit point: the
                # increment again, from where it started, by pseudo-time
                # (1.23, `Newton.solve_stabilised`)
                newton.put(saved)
                if newton.trial:
                    s._trial_mode = True
                s._apply_bc(fixed, moving, prescribed, target)
                ok, res, its2 = newton.solve_stabilised(lambda: s._force(groups, ground, target), fixed, tol_k)
                its += its2
                if ok:
                    newton.stabilised += 1
                else:
                    newton.put(saved)
        if not ok:
            newton.put(saved)
            if newton.trial:
                s._trial_mode = True      # the committed fibre state is untouched
            if not newton.stalled and (cuts + 1 > max_cuts or dl / 2.0 < dl_min * (1.0 - 1e-9)) \
                    and spec.get("fallback", "none") == "relaxation":
                # A LIMIT POINT UNDER LOAD CONTROL, OR A JUMP IN THE PATH:
                # the level is taken by a relaxation instead (1.7) -- the
                # STATED level, not the halved increment: a relaxation does
                # not care how far the level is -- and so is the rest of the
                # step (`relaxing`)
                tried = target - lam
                target = level
                where = f"step {k + 1}, load factor {target:.6g}" + amplitude_note(s, groups, target)
                ok, res, its = relax_increment(newton, fixed, moving, prescribed, groups, ground, target,
                                               tol_k, spec, s.log, where, step, report)
                relaxing = ok
                if not ok:
                    newton.put(saved)
                    if newton.trial:
                        s._trial_mode = True
                    msg = (f"static step {k + 1}: no convergence at load factor {target:.6g}"
                           f"{amplitude_note(s, groups, target)}: Newton's increment halved to "
                           f"{tried:.3g} (static.cutbacks), and the fallback relaxation did not "
                           f"converge in {its} iterations (residual {res:.3e}, tolerance {tol_k:.1e}). A "
                           "limit point -- the peak of a pushover, a snap-through -- cannot be passed "
                           "under load control; add static.arc_length, drive it by a displacement, "
                           "or use mode: relaxation")
                    finish_diagnostics(s.diagnostics, "MAX_STEPS", step=s.step, residual=res,
                                       load_step=k + 1, message=msg)
                    raise ConvergenceFailure(msg)
                s.log(f"  fallback      {where}: relaxed in {its} iterations (residual {res:.2e})")
                dl = 1.0 / n_inc
                res = _polish(newton, lambda: s._force(groups, ground, target), fixed, tol_k, s.log, where)
        if not ok:
            if newton.stalled:
                # NOT A LIMIT POINT: Newton's corrections had shrunk to the
                # round-off of the coordinates and the residual stood still
                # above the tolerance. Halving the increment cannot lower
                # the arithmetic's floor, and calling it a limit point sent
                # the analyst looking for a peak
                msg = stall_message(f"static step {k + 1}, load factor {target:.6g}"
                                    + amplitude_note(s, groups, target), res, tol_k)
                finish_diagnostics(s.diagnostics, "MAX_STEPS", step=s.step, residual=res,
                                   load_step=k + 1, message=msg)
                raise ConvergenceFailure(msg)
            cuts += 1
            if cuts > max_cuts or dl / 2.0 < dl_min * (1.0 - 1e-9):
                msg = (f"static step {k + 1}: no convergence at load factor {target:.6g}"
                       f"{amplitude_note(s, groups, target)}, the "
                       f"increment halved to {target - lam:.3g} (static.cutbacks: "
                       f"{int(spec.get('cutbacks', 10))}; residual {res:.3e}, tolerance {tol_k:.1e}). A limit "
                       "point -- the peak of a pushover, a snap-through -- cannot be passed "
                       "under load control; add static.arc_length, drive it by a displacement, "
                       "let a relaxation take the step from here (static.fallback: relaxation), "
                       "or use mode: relaxation")
                finish_diagnostics(s.diagnostics, "MAX_STEPS", step=s.step, residual=res,
                                   load_step=k + 1, message=msg)
                raise ConvergenceFailure(msg)
            s.log(f"  cutback       step {k + 1}, load factor {target:.4g}: residual {res:.2e} "
                  f"after {its} iterations; increment halved to {dl / 2:.4g}")
            dl /= 2.0
            continue
        newton.commit()
        if _DEBUG:
            F2 = s._force(groups, ground, target); f2 = newton.force(); r2 = F2 - f2; r2[fixed] = 0.0
            print(f"      after commit: residual {newton.measure(F2, f2, ~fixed, r2):.3e} (was {res:.3e})")
        where = f"step {k + 1}, load factor {target:.6g}" + amplitude_note(s, groups, target)
        _settle_ruptures(newton, lambda: s._force(groups, ground, target), fixed, tol_k, n_before,
                         s.log, where)
        _settle_breaks(newton, lambda: s._force(groups, ground, target), fixed, tol_k,
                       s._frame_time(k, target) if hasattr(s, "_frame_time") else k + target, s.log, where)
        # the budget of halvings is per increment -- and with the fallback
        # on, per STATED level: a path that converges only in increments at
        # the floor, failing every doubled one, is crawling along a jump,
        # and the relaxation takes the level instead (1.7)
        if spec.get("fallback", "none") == "none" or target >= level - 1e-9:
            cuts = 0
        lam = target; inc += 1; ctx["count"] += 1
        s.step += 1
        # grow back towards the stated increment after a success
        dl = min(1.0 / n_inc, dl * 2.0)
        s._static_state = {"kind": 0, "lam": lam, "dl": dl, "inc": inc, "count": ctx["count"],
                           "relaxing": 1 if relaxing else 0}
        s.residuals.append(res); s.kinetic.append(0.0)
        # MONITORS AT EVERY INCREMENT, frames every `output.every`. Both
        # were thinned together, so the default `every: 200` left a
        # ten-level curve as one point -- and `monitors/residual` (every
        # increment) no longer matched `monitors/time` in length
        # (found in review). Relaxation records its
        # monitors at every level too.
        s.frame_equilibrium = True
        F = s._force(groups, ground, lam)
        if getattr(s, "time_dependent", False):
            # the increment is a TIME STATION: the elastic
            # deformation it settled at goes into the creep history, dated as
            # `_station_phi` dated it while it was solved -- the staged
            # relaxation's own step, `_record_time_station`
            s._record_time_station(s._current_age)
        frame = ctx["count"] % every == 0 or lam >= 1.0 - 1e-12
        s._record(s._frame_time(k, lam) if hasattr(s, "_frame_time") else k + lam,
                  F, on_output if frame else None)
        # asked after the frame is written: a stop before it lost that
        # load level from both the stopped and the resumed result
        if on_progress and on_progress(int(round(lam * n_inc)), n_inc, res, 0.0, inc) is False:
            from .core.diagnostics import StopAnalysis
            raise StopAnalysis()
    s._static_state = None

def static_step(solver, k, r, on_output=None, output_every=1, on_progress=None):
    """A `procedure: static` step inside a relaxation or dynamic run (1.2):
    the same Newton as `run`, built once and kept for the run's later static
    steps. The state it starts from is what the step before left, at rest:
    a static step after a dynamic one solves for the equilibrium under its
    loads and discards the velocities."""
    check(solver)
    s = solver
    sol = s.p.solver
    if getattr(s, "_newton", None) is None:
        spec = sol.static.model_dump()            # (Newton reads it as a dict still)
        s._newton = Newton(s, spec)
        count0 = int(r.static.get("count", 0)) if (r is not None and getattr(r, "static", None)) else 0
        s._static_ctx = {"count": count0, "spec": spec, "tol": float(sol.tolerance or 1e-4),
                         "every": max(1, int(output_every or 1)),
                         "on_output": on_output, "on_progress": on_progress}
        s.log(f"  static        Newton for the static steps, finite-difference tangent over "
              f"{len(s._newton.groups)} node colours")
    s.v[:] = 0.0
    s._half = True
    _static_step(s, s._newton, k, r, s._static_ctx)


def run(solver, on_output=None, output_every=1, on_progress=None, progress_every=500):
    """`solver.run` for `solver.mode: static`: every step's load applied in
    `load_steps` increments (10 by default), each solved by Newton; an
    increment that does not converge is halved, up to `cutbacks` times."""
    check(solver)
    s = solver
    sol = s.p.solver
    spec = sol.static.model_dump()                # (Newton reads it as a dict still)
    tol = float(sol.tolerance or 1e-4)
    every = max(1, int(output_every or 1))
    started = time.perf_counter()
    s.times, s.residuals, s.kinetic = [], [], []
    s.diagnostics = start_diagnostics("static", getattr(s, "backend", "numpy"))
    from .core.diagnostics import resume_ledger
    resume_ledger(s)      # a restart carries the ledger on
    led = s.diagnostics.get("energy")
    if led is not None:
        led.available = False
        led.reason = "an implicit static solution has no time history to audit"
    s.frame_equilibrium = None
    newton = Newton(s, spec)
    s._newton = newton
    s.log(f"  static        Newton, {'full' if spec.get('tangent', 'full') == 'full' else 'modified'} tangent "
          f"by finite differences over {len(newton.groups)} node colours")
    # a restart resumes the increment it was at
    from .core.resume import resume_position, check_resumed
    resume = resume_position(s, len(s.steps))
    count = int(resume.static.get("count", 0)) if resume is not None and resume.static else 0
    ctx = {"count": count, "spec": spec, "tol": tol, "every": every,
           "on_output": on_output, "on_progress": on_progress}
    plan = s.plan
    if plan is not None:
        from .core import analysis_steps
    prev = None
    for k in range(resume.k if resume else 0, len(s.steps)):
        r = resume if (resume is not None and k == resume.k) else None
        if plan is not None:
            if prev is not None:
                analysis_steps.end_step(s, prev)
            analysis_steps.begin_step(s, k, resumed=r is not None)
            if plan.procedure[k] != "static":
                raise ValueError(f"steps[{k}] is {plan.procedure[k]}: in a model whose solver.mode is "
                                 "static every step is static; give the model the mode of its first "
                                 "time-stepping procedure (relaxation or dynamic) and the static steps "
                                 "`procedure: static`")
        prev = k
        _static_step(s, newton, k, r, ctx)
    if plan is not None and prev is not None:
        analysis_steps.end_step(s, prev)
    count = ctx["count"]
    check_resumed(s, resume)
    s.wall_time = time.perf_counter() - started
    s.log(f"  static        {count} increments, {newton.iterations} Newton iterations, "
          f"{newton.tangents} tangents, {newton.evaluations} force evaluations"
          + (f"; {newton.stabilised} increment(s) by the stabilised Newton" if newton.stabilised else ""))
    finish_diagnostics(s.diagnostics, "CONVERGED", step=s.step,
                       residual=s.residuals[-1] if s.residuals else None, kinetic=0.0,
                       load_step=len(s.steps))
    return s
