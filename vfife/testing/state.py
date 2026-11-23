"""One state vector and one force for every solver family.

The nine solver classes do not agree on anything an outside question needs.
A solid keeps its displacement in `u` (nn, 3) and its rotations per element in
`blocks[i].quat`; a frame keeps `q` (6 nn,) flat with the points' triads
beside it in `orient`, and its `u` is a read-only copy. A solid's internal
force is a loop over blocks plus the reinforcement; a frame's is one method
that also writes `end_actions`; a truss's is a third spelling. Half of them
integrate the material forward as a side effect of being asked. The external
force is a method on the first two and a two-stage `step_data` / `_force` on
the rest.

None of that is wrong -- they are different solvers -- and all of it is why
the questions of `invariants.py` (is this force a gradient, does a rigid
motion move it, does the mass bound the stiffest mode) were only ever asked
of the one family whose spelling the person asking happened to know.

This module is the adapter. `probe(solver)` returns a `Probe`:

    p.n                     the number of degrees of freedom
    p.get() -> (n,)         the current state, flat
    p.set(x)                write it (orientations resynchronised)
    p.internal(x) -> (n,)   the internal force, WITHOUT side effects
    p.external(x) -> (n,)   the external force at that state
    p.fixed -> (n,) bool    the held dofs
    p.mass -> (n,)          the mass the integrator actually uses
    p.nodes -> (nn, 3)      reference coordinates
    p.translation -> (n,)   the indices of the translational dofs, node-major
    p.rotation -> (n,)      the rotational ones, or an empty array

Everything downstream -- the invariants, `vfife probe`, a future arc-length
continuation -- is written once against that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class Probe:
    solver: object
    n: int
    nn: int
    get: Callable[[], np.ndarray]
    set: Callable[[np.ndarray], None]
    internal: Callable[[np.ndarray], np.ndarray]
    external: Callable[[np.ndarray], np.ndarray]
    fixed: np.ndarray
    mass: np.ndarray
    nodes: np.ndarray
    translation: np.ndarray
    rotation: np.ndarray
    #: rigid motions are applied to translational dofs and, where the family
    #: has them, composed onto the points' orientations
    family: str

    @property
    def free(self) -> np.ndarray:
        return ~self.fixed

    def residual(self, x: np.ndarray) -> np.ndarray:
        """External minus internal, zeroed on the held dofs."""
        r = self.external(x) - self.internal(x)
        r[self.fixed] = 0.0
        return r

    def rigid(self, x: np.ndarray, R: np.ndarray, c: np.ndarray,
              t: np.ndarray) -> np.ndarray:
        """The state x moved rigidly: every material point of the deformed
        body taken to R (p - c) + c + t, with the orientation state, where
        there is one, pre-multiplied by R.

        This is the motion the internal force must not notice. Doing it on
        the STATE and not on the mesh is the point: a co-rotational element
        that recovers the frame from the chord alone passes a rigid motion of
        the reference body and fails one of a deformed body, which is the
        defect v0.1 found in the frame solver.
        """
        x = np.asarray(x, float).copy()
        p = self.nodes + x[self.translation].reshape(self.nn, 3)
        moved = (R @ (p - c).T).T + c + t
        x[self.translation] = (moved - self.nodes).ravel()
        if self.rotation.size:
            self._rotate(x, R, None)
        return x

    def perturb(self, x: np.ndarray, j: int, h: float) -> np.ndarray:
        """The state x with degree of freedom j moved by h -- ADDITIVELY for a
        translation, MULTIPLICATIVELY for a rotation.

        This is the whole reason the harness has its own perturbation. Since
        v0.1 a structural point's orientation is a matrix advanced by
        R <- exp(omega dt) R, and the moment the solver assembles is the
        SPATIAL moment, conjugate to that spin. The rotation vector log(R) is
        derived from the matrix for output and monitors; it is not the
        coordinate the solver integrates in. Differencing along log(R)
        therefore returns dm/dspin composed with the tangent map T(phi), which
        is not symmetric even when the force is a perfect gradient -- a
        cantilever bent 1.2 rad scores 9e-2 that way and 1e-10 this way. Ask
        the question in the coordinates the equations of motion use.
        """
        x = np.asarray(x, float).copy()
        if self.rotation.size and self._is_rotation[j]:
            self._spin(x, self._rot_node[j], self._rot_comp[j], h)
        else:
            x[j] += h
        return x

    def bent(self, angle: float = 1.2) -> np.ndarray:
        """The model rolled into an arc of `angle` radians about its own
        longest axis: every material point turned about the transverse axis
        through the origin end by an angle proportional to how far along it
        lies, with the orientation state turned with it.

        One state, and it is the state the library keeps finding defects in:
        finite rotations that differ from point to point (so a co-rotational
        frame has to be right and not merely consistent), a real strain in
        every element (so the tangent is the deformed one, not K_0), and no
        symmetry left to hide an out-of-plane term. It costs one line and no
        solve, which is what makes it affordable to ask of every element.
        """
        from ..core.rotations import exp_map

        X = self.nodes
        span = X.max(axis=0) - X.min(axis=0)
        long = int(np.argmax(span))
        axis = int(np.argmin(span)) if span.min() > 0 else (long + 1) % 3
        if axis == long:
            axis = (long + 1) % 3
        L = span[long] or 1.0
        s = (X[:, long] - X[:, long].min()) / L                 # 0 .. 1
        e = np.zeros(3); e[axis] = 1.0
        theta = angle * s
        R = exp_map(theta[:, None] * e[None, :])                # (nn, 3, 3)
        c = X.copy(); c[:, long] = X[:, long].min()
        moved = np.einsum("nij,nj->ni", R, X - c) + c
        x = np.zeros(self.n)
        x[self.translation] = (moved - X).ravel()
        if self.rotation.size:
            self._rotate(x, None, R)
        return x

    #: set by the builders that carry an orientation state
    _rotate: Callable[..., None] = lambda *a: None
    _spin: Callable[..., None] = lambda *a: None
    _is_rotation: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    _rot_node: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    _rot_comp: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    _scale: Optional[float] = None


def _no_rotation(x, R=None, per_node=None):
    raise NotImplementedError("this family carries no orientation state")


def probe(solver) -> Probe:
    """The adapter for whichever solver this is."""
    from ..core.coupled_solver import CoupledSolver
    from ..core.orientation import OrientedPoints
    from ..core.solver import Solver
    from ..core.truss_solver import TrussSolver

    if isinstance(solver, Solver):
        return _solid(solver)
    if isinstance(solver, TrussSolver):
        return _truss(solver)
    # everything that carries an orientation per point: beams, fibre beams,
    # shells, mixed lines, connectors, coupled and shell_line models
    if isinstance(solver, OrientedPoints):
        return _framed(solver, coupled=isinstance(solver, CoupledSolver))
    raise TypeError(f"no probe for {type(solver).__name__}")


def _solid(s) -> Probe:
    from ..core.mesh import scatter_add

    nn = s.nn
    n = 3 * nn

    def internal(x):
        u = np.asarray(x, float).reshape(nn, 3)
        F = np.zeros((nn, 3))
        for b in s.blocks:
            # copies: the kernels advance the element rotation and the
            # material state in place, and a question must not be a step
            f_e, _ = b.force(u, b.quat.copy(), b.state.copy(), s.dt,
                             rot_iters=_ROT_ITERS, want_stress=False)
            F += b.force.assemble(f_e)
        for emb in s.rebar:
            # the bar law advances its state in place too: a probe of a
            # yielded bar moved its plastic strain (found in review)
            keep = emb.state
            emb.state = keep.copy()
            try:
                fb, _ = emb.force(u, s.dt, want_stress=False)
            finally:
                emb.state = keep
            F += scatter_add(fb, emb.conn, nn)
        for bt in s.bonded:
            _, f_host, _ = bt.force(u, dt=s.dt, want_stress=False)
            F += bt.scatter_host(f_host, nn)
        return F.ravel()

    def external(x):
        keep = s.u.copy()
        s.u[:] = np.asarray(x, float).reshape(nn, 3)
        try:
            return s.external_force(1.0).ravel()
        finally:
            s.u[:] = keep

    def setter(x):
        s.u[:] = np.asarray(x, float).reshape(nn, 3)

    p = Probe(solver=s, n=n, nn=nn,
              get=lambda: s.u.ravel().copy(), set=setter,
              internal=internal, external=external,
              fixed=s.fixed.ravel().copy(), mass=s.mass.ravel().copy(),
              nodes=s.mesh.nodes, translation=np.arange(n),
              rotation=np.empty(0, int), family="solid")
    p._rotate = _no_rotation
    return p


def _truss(s) -> Probe:
    nn = s.nn
    n = 3 * nn

    def internal(x):
        keep_u, keep_state = s.u.copy(), s.state.copy()
        s.u[:] = np.asarray(x, float).reshape(nn, 3)
        try:
            F, _ = s._force(want_stress=False)
            return F.ravel()
        finally:
            s.u[:] = keep_u
            s.state[...] = keep_state

    def external(x):
        keep = s.u.copy()
        s.u[:] = np.asarray(x, float).reshape(nn, 3)
        try:
            return s.external_force(1.0).ravel()
        finally:
            s.u[:] = keep

    def setter(x):
        s.u[:] = np.asarray(x, float).reshape(nn, 3)

    p = Probe(solver=s, n=n, nn=nn,
              get=lambda: s.u.ravel().copy(), set=setter,
              internal=internal, external=external,
              fixed=s.fixed.ravel().copy(), mass=s.mass.ravel().copy(),
              nodes=s.mesh.nodes, translation=np.arange(n),
              rotation=np.empty(0, int), family="truss")
    p._rotate = _no_rotation
    return p


def _framed(s, coupled=False) -> Probe:
    """Beams, fibre beams, shells, mixed lines, connectors, coupled models:
    everything that keeps `q` flat with an orientation matrix per point."""
    from ..core.dof import ROTATION, TRANSLATION
    from ..core.rotations import exp_map, log_map, orthonormalize

    nn = s.nn
    n = s.ndof
    every = np.arange(nn)
    tr = s.dofs.node_dofs(every, TRANSLATION).ravel()
    rot = (s.dofs.node_dofs(every, ROTATION).ravel()
           if s.dofs.stride == 6 else np.empty(0, int))

    trial = hasattr(s, "_trial_mode")

    def internal(x):
        keep_q, keep_orient = s.q.copy(), getattr(s, "orient", None)
        if keep_orient is not None:
            keep_orient = keep_orient.copy()
        was_trial = getattr(s, "_trial_mode", None)
        if trial:
            # a fibre section integrates its history forward when asked; in
            # trial mode every call restores from the last committed point
            s._trial_mode = True
        s.q[:] = np.asarray(x, float)
        if hasattr(s, "sync_orientation"):
            s.sync_orientation()
        try:
            if coupled:
                keep_quat = s.solid.blocks[0].quat.copy()
                keep_state = s.solid.blocks[0].state.copy()
                # the solid inside a coupled model warm-starts its polar
                # decomposition from the last step's quaternion and takes the
                # solver's two iterations; from an arbitrary state that is not
                # a converged rotation, so ask for the full solve
                keep_iters = s.solid.rot_iters
                s.solid.rot_iters = _ROT_ITERS
            # a probe, not a step: path-dependent member state (plastic
            # hinges) is read and not advanced -- the modal and spectrum
            # tangents committed hinge rotation at every evaluation
            if hasattr(s, "_probe"):
                with s._probe():
                    f, _ = s.internal_force()
            else:
                f, _ = s.internal_force()
            return np.asarray(f, float).copy()
        finally:
            if coupled:
                s.solid.blocks[0].quat[...] = keep_quat
                s.solid.blocks[0].state[...] = keep_state
                s.solid.rot_iters = keep_iters
            s.q[:] = keep_q
            if keep_orient is not None:
                s.orient[...] = keep_orient
            if trial:
                s._trial_mode = was_trial

    # the step the solver is at -- after a run, its LAST step's supports,
    # not the first's
    fixed, _, _, groups, ground = s.step_data(int(getattr(s, "step_index", 0) or 0))
    if getattr(s, "current_fixed", None) is not None:
        fixed = np.asarray(s.current_fixed, bool)

    def external(x):
        keep_q, keep_orient = s.q.copy(), getattr(s, "orient", None)
        if keep_orient is not None:
            keep_orient = keep_orient.copy()
        s.q[:] = np.asarray(x, float)
        if hasattr(s, "sync_orientation"):
            s.sync_orientation()
        try:
            f = s._force(groups, ground, 1.0)
            if coupled:
                # CoupledSolver._force adds the solid's own external force
                return np.asarray(f, float).copy()
            return np.asarray(f, float).copy()
        finally:
            s.q[:] = keep_q
            if keep_orient is not None:
                s.orient[...] = keep_orient

    def setter(x):
        s.q[:] = np.asarray(x, float)
        if hasattr(s, "sync_orientation"):
            s.sync_orientation()

    def rotate(x, R=None, per_node=None):
        """Pre-multiply every point's triad, by one R (a rigid rotation of the
        whole body) or by one R per node (`Probe.bent`). The state's rotation
        dofs are log(orient), so this goes through the matrices and back."""
        orient = _orient_of(s, x)
        turn = per_node if per_node is not None else R[None, :, :]
        x[rot] = log_map(orthonormalize(np.einsum("nij,njk->nik", turn, orient))).ravel()

    p = Probe(solver=s, n=n, nn=nn,
              get=lambda: s.q.copy(), set=setter,
              internal=internal, external=external,
              fixed=fixed.copy(), mass=np.asarray(s.mass, float).ravel().copy(),
              nodes=s.mesh.nodes, translation=tr, rotation=rot,
              family="framed")
    p._rotate = rotate

    def spin(x, node, comp, h):
        w = np.zeros(3); w[comp] = h
        orient = _orient_of(s, x)
        orient[node] = exp_map(w) @ orient[node]
        x[s.dofs.node_dofs(node, ROTATION).ravel()] = log_map(
            orthonormalize(orient[node][None]))[0]

    p._spin = spin
    p._is_rotation = np.zeros(n, bool)
    p._is_rotation[rot] = True
    p._rot_node = np.zeros(n, int)
    p._rot_comp = np.zeros(n, int)
    p._rot_node[rot] = np.repeat(every, 3)
    p._rot_comp[rot] = np.tile(np.arange(3), nn)
    return p


def _orient_of(s, x):
    """The orientation matrices a state vector stands for."""
    from ..core.dof import ROTATION
    from ..core.rotations import exp_map

    return exp_map(s.dofs.field(np.asarray(x, float), ROTATION))


#: from a cold quaternion at an arbitrary state the polar decomposition needs
#: the full solve, not the solver's warm-started two
_ROT_ITERS = 60
