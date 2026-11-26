"""Work-conjugate six-component co-rotational connector mechanics.

The Gmsh Line2 chord defines the reference separation and, for a local
connector, its local x axis.  The two points' orientations remove the rigid
motion; the remaining relative translation and rotation drive six independent
or coupled spring/dashpot components.  A numerical 6x12 kinematic Jacobian
keeps the generalized nodal forces work-conjugate without embedding a
small-angle shortcut in the element force law.

THE ORIENTATION IS A MATRIX. Until this stage the connector
took the mean of the two nodes' ROTATION VECTORS as its frame and their
DIFFERENCE as its rotational deformation:

    Q = rotation_matrix(0.5 * (q[0, 3:] + q[1, 3:]))
    pure_rotation = q[1, 3:] - q[0, 3:]

Rotation vectors do not average and do not subtract. Both expressions are
exact when the two rotations are equal or coaxial -- which is every case the
library had, and every earlier test in tests/test_connector2.py -- and
neither is a rotation when they are not: a connector whose two ends have
turned differently, moved rigidly, reported 7.5% of the model's own
stiffness scale in force that was not there. That is the defect v0.1 fixed
in the frame solver and, in the same version, carried into the shell and the
fibre beam by giving every structural point an orientation MATRIX; the
connector was the one element not migrated with them, and the invariant
harness of v0.1 is what found it still sitting there.

Now the frame is the geodesic mean triad `T1 exp(0.5 log(T1^T T2))`, the
same one `rotations.beam_frames_from_triads` uses, and the rotational
deformation is the spatial relative rotation `log(T2 T1^T)`. Both are
equivariant -- turn the whole connector by R and the deformation does not
change -- which the earlier pair is not.

AND THE JACOBIAN IS TAKEN ALONG SPINS. Since v0.1 a point's orientation is
advanced by `R <- exp(omega dt) R` and the moment the solver assembles is
conjugate to that spin, not to an increment of `log R`. So the rotational
columns of the Jacobian difference `exp(h e_k) T` rather than `q + h e_k`,
which is also what makes the dashpot term `J v` correct: the rotational
components of the solver's velocity ARE spins.
"""
from __future__ import annotations

import numpy as np

from ..core.rotations import exp_map, log_map


COMPONENTS = ("ux", "uy", "uz", "rx", "ry", "rz")


def matrix(value) -> np.ndarray:
    """Return a validated-schema matrix as a dense 6x6 array."""
    if value is None:
        return np.zeros((6, 6))
    a = np.asarray(value, dtype=float)
    return np.diag(a) if a.shape == (6,) else a.copy()


def _exp(v) -> np.ndarray:
    return exp_map(np.asarray(v, dtype=float)[None])[0]


def _log(M) -> np.ndarray:
    return log_map(np.asarray(M, dtype=float)[None])[0]


def rotation_matrix(vector) -> np.ndarray:
    """Exponential map of an engineering rotation vector."""
    return _exp(vector)


def identity_triads() -> np.ndarray:
    """The two points' orientations in the reference configuration."""
    return np.tile(np.eye(3), (2, 1, 1))


def mean_triad(triads) -> np.ndarray:
    """The connector's frame: the geodesic mean of its two ends' triads,
    `T1 exp(0.5 log(T1^T T2))`, which is what `rotations.beam_frames_from_triads`
    calls the mean nodal triad of a member. Equivariant, unlike the mean of
    the two rotation VECTORS this replaced."""
    T = np.asarray(triads, dtype=float).reshape(2, 3, 3)
    return T[0] @ _exp(0.5 * _log(T[0].T @ T[1]))


def _swap(M):
    return np.swapaxes(M, -1, -2)


def _deformations(u, T, x0, R0, props) -> np.ndarray:
    """`deformation` of a whole STACK of configurations at once: u (B, 2, 3)
    and T (B, 2, 3, 3) -> (B, 6). Every operation is the per-configuration
    one broadcast over the leading axis, so row b is exactly
    `deformation(u[b], T[b], ...)` -- which is how that function is now
    defined, so there is one formula and not two."""
    x0 = np.asarray(x0, dtype=float).reshape(2, 3)
    T0 = T[:, 0]; T1 = T[:, 1]
    # mean triad, T1 exp(0.5 log(T1^T T2)), stacked
    Q = T0 @ exp_map(0.5 * log_map(_swap(T0) @ T1))
    current = x0 + u
    chord = x0[1] - x0[0]
    pure_translation = (current[:, 1] - current[:, 0]) - Q @ chord
    pure_rotation = log_map(T1 @ _swap(T0))
    if props.get("coordinate_system", "local") == "local":
        axes = np.asarray(R0, dtype=float) @ _swap(Q)
        return np.concatenate([np.einsum("bij,bj->bi", axes, pure_translation),
                               np.einsum("bij,bj->bi", axes, pure_rotation)], axis=1)
    return np.concatenate([pure_translation, pure_rotation], axis=1)


def deformation(u, triads, x0, R0, props) -> np.ndarray:
    """Pure connector deformation after removing the rigid motion.

    `u` is the two ends' translations, (2, 3); `triads` their orientation
    matrices, (2, 3, 3), or None for the reference configuration.
    """
    u = np.asarray(u, dtype=float).reshape(1, 2, 3)
    T = identity_triads() if triads is None else np.asarray(triads, float).reshape(2, 3, 3)
    return _deformations(u, T[None], x0, R0, props)[0]


#: the spin increment the rotational columns of the Jacobian are taken along,
#: in radians -- a rotation has no length scale to borrow one from
_SPIN_STEP = 2.0e-7

#: exp(+h e_k) for k = 0..2, then exp(-h e_k): the six multiplicative spin
#: perturbations, built once rather than 12 times a call
_SPINS = exp_map(np.r_[np.eye(3), -np.eye(3)] * _SPIN_STEP)


def _deformation_and_jacobian(u, triads, x0, R0, props):
    """The deformation and d(deformation)/d(u1, omega1, u2, omega2) from ONE
    stacked evaluation of 25 configurations -- the state itself and the 24
    central-difference perturbations.

    THIS WAS THE COST OF A BEARING (v0.4). The
    Jacobian used to be 24 separate calls of `deformation`, each a dozen
    3x3 numpy operations with two log maps and an exp map in them: small
    arrays, so all of it Python overhead. On the Hangzhou Bay seismic model
    that was 80% of every step for two connectors. Stacking changes the
    overhead from per-configuration to per-call and nothing else: the
    perturbations, the step sizes and the differences are the ones below.
    """
    u = np.asarray(u, dtype=float).reshape(2, 3)
    T = identity_triads() if triads is None else np.asarray(triads, float).reshape(2, 3, 3)
    h_u = 2.0e-7 * max(float(np.linalg.norm(u)), 1.0)
    # rows: 0 = the state; then for node n, component k (i = 3n + k):
    #   1 + i      translation +h     7 + i   translation -h
    #   13 + i     spin +h            19 + i  spin -h
    U = np.broadcast_to(u, (25, 2, 3)).copy()
    TT = np.broadcast_to(T, (25, 2, 3, 3)).copy()
    for n in range(2):
        for k in range(3):
            i = 3 * n + k
            U[1 + i, n, k] += h_u
            U[7 + i, n, k] -= h_u
            TT[13 + i, n] = _SPINS[k] @ T[n]
            TT[19 + i, n] = _SPINS[3 + k] @ T[n]
    D = _deformations(U, TT, x0, R0, props)
    J = np.empty((6, 12))
    for n in range(2):
        for k in range(3):
            i = 3 * n + k
            J[:, 6 * n + k] = (D[1 + i] - D[7 + i]) / (2.0 * h_u)
            J[:, 6 * n + 3 + k] = (D[13 + i] - D[19 + i]) / (2.0 * _SPIN_STEP)
    return D[0], J


def jacobian(u, triads, x0, R0, props) -> np.ndarray:
    """d(connector deformation) / d(u1, omega1, u2, omega2), numerically.

    The translational columns are an ordinary central difference; the
    rotational ones perturb the triad MULTIPLICATIVELY, `exp(h e_k) T`, so
    the columns are derivatives with respect to the spins the solver
    integrates and `J^T action` is the force conjugate to them.
    """
    return _deformation_and_jacobian(u, triads, x0, R0, props)[1]


def curve_force(value: float, table) -> float:
    """Piecewise-linear interpolation with terminal-slope extrapolation."""
    a = np.asarray(table, dtype=float)
    x, y = a[:, 0], a[:, 1]
    if value <= x[0]:
        return float(y[0] + (value-x[0])*(y[1]-y[0])/(x[1]-x[0]))
    if value >= x[-1]:
        return float(y[-1] + (value-x[-1])*(y[-1]-y[-2])/(x[-1]-x[-2]))
    return float(np.interp(value, x, y))


def effective_stiffness(props) -> np.ndarray:
    """Conservative reference matrix for assembly and stable-step estimates."""
    K = matrix(props.get("stiffness"))
    for name, table in (props.get("curves") or {}).items():
        a = np.asarray(table, dtype=float)
        slopes = np.diff(a[:, 1])/np.diff(a[:, 0])
        K[COMPONENTS.index(name), COMPONENTS.index(name)] = float(np.max(slopes))
    for name, law in (props.get("laws") or {}).items():
        K[COMPONENTS.index(name), COMPONENTS.index(name)] = law_stiffness(law)
    return K


# --------------------------------------------------------------------------- #
#  Path-dependent one-dimensional laws on a component
# --------------------------------------------------------------------------- #
#
# Three laws, each a force F(d, state) on ONE component of the deformation d,
# and each carrying at most ONE scalar of history -- the plastic (slipped)
# part of that component's deformation, `slip`:
#
#   gap        F = k (d - hi) for d > hi, k (d - lo) for d < lo, else 0.
#              No history. A stop, an expansion joint, pounding against an
#              abutment, a compression-only seat on the local ux, which is
#              the elongation: {lo: 0} (the docs said {lo: None, hi: 0}
#              until v1.0, which is a TENSION-only tie).
#   bilinear   elastic k1 up to fy, then k2, kinematic hardening: the
#              Bouc-Wen-free skeleton of a lead-rubber or a friction-pendulum
#              isolator, and of any steel yielding device. Return mapping on
#              the back stress alpha = H slip with H = k1 k2 / (k1 - k2).
#   friction   Coulomb: sticks with stiffness k until |F| = mu N, then slides
#              at mu N. N is the COMPRESSION currently carried by the local
#              ux of the same connector (`normal: ux`, the only component
#              whose sign does not depend on the node order), or a number.
#
# ON uy AND uz TOGETHER, the same law twice is ONE law in the plane:
# a bearing slides, or yields, in the direction its
# resultant points, with a circular yield surface |F| = f_y. Run as two
# independent laws the surface was a square: a 45-degree drag of a friction
# bearing carried 1.414 mu N.
#              bilinear with k2 = 0 and a yield force that follows N; with N
#              at zero (uplift) the component carries nothing and its slip
#              follows the deformation.
#
# The state is advanced from the COMMITTED slip and the current deformation
# (a return mapping), so evaluating the same deformation twice gives the same
# answer: within a static stage every evaluation is a trial re-driven from
# the last equilibrium, exactly as the fibre materials are handled
# (fiber_beam_solver); the solver commits at equilibrium, and
# at every step of a dynamic run.

LAWS = ("gap", "bilinear", "friction", "damper", "pendulum", "material")

# AND ONE THAT IS NOT A LAW OF ITS OWN (1.21):
#
#   material   a one-dimensional law from the case's material table on the
#              component: F = area sigma(d / length), the tangent area E_t /
#              length. With area = length = 1 (the defaults) the table is the
#              component's force-deformation relation itself; with a core's
#              area and its yielding length it is a buckling-restrained
#              brace. Its history is the law's own state, which the solver
#              keeps beside the slips (`MaterialStates`); every evaluation
#              starts from the committed state, as the laws above do from
#              the committed slip. The slip reported for the component is
#              the material's strain, d / length. `uy` and `uz` stay two
#              laws (a square interaction), and the compiled connector
#              kernel does not take it (the solver runs the connector in
#              Python and says so).


#: one Material per law spec, built once (keyed on the spec dict's id, held
#: with it so the id cannot be reused)
_MATERIALS: dict = {}


def law_material(law):
    """The Material of a `material` law, from its resolved spec."""
    spec = law["material"]
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError(f"a connector `material` law names {spec!r}: in a case it is a name in `materials`, "
                         "resolved when the case is read; here it must be the material's spec")
    hit = _MATERIALS.get(id(spec))
    if hit is not None and hit[0] is spec:
        return hit[1]
    from ..materials.factory import create_material
    m = create_material(dict(spec))
    if not getattr(m, "uniaxial", False) and m.name != "linear_elastic":
        raise ValueError(f"a connector `material` law takes a one-dimensional law; {m.name!r} is not")
    _MATERIALS[id(spec)] = (spec, m)
    return m


def _scale(law):
    """(area, length) of a `material` law."""
    a = law.get("area"); L = law.get("length")
    return (1.0 if a is None else float(a)), (1.0 if L is None else float(L))


def has_material_law(props) -> bool:
    return any(l.get("type") == "material" for l in (props.get("laws") or {}).values())


class MaterialStates:
    """The history of every connector component that carries a `material`
    law: a committed state and a working one, (1, n_state) each, per
    (connector, component). `response` takes `of(e)` and writes the working
    state from the committed one; the solver commits where it commits the
    slips, clears a connector's at its birth, and saves the committed ones
    in its checkpoint."""

    def __init__(self, ids, props_of):
        self.rows = {}
        for e in ids:
            for name, law in ((props_of(e) or {}).get("laws") or {}).items():
                if law.get("type") == "material":
                    m = law_material(law)
                    c = m.init_state(1)
                    self.rows.setdefault(int(e), {})[COMPONENTS.index(name)] = [m, c, c.copy()]

    def __bool__(self):
        return bool(self.rows)

    def of(self, e):
        return self.rows.get(int(e))

    def commit(self):
        for comps in self.rows.values():
            for row in comps.values():
                row[1][...] = row[2]

    def reset(self, ids):
        for e in ids:
            for row in (self.rows.get(int(e)) or {}).values():
                row[1][...] = row[0].init_state(1); row[2][...] = row[1]

    def _order(self):
        return [(e, i) for e in sorted(self.rows) for i in sorted(self.rows[e])]

    def checkpoint(self):
        parts = [self.rows[e][i][1].ravel() for e, i in self._order()]
        return np.concatenate(parts) if parts else np.zeros(0)

    def restore(self, flat):
        flat = np.asarray(flat, float).ravel()
        need = sum(self.rows[e][i][1].size for e, i in self._order())
        if flat.size != need:
            raise ValueError(f"the checkpoint's connector material state has {flat.size} values; "
                             f"this model's connector materials have {need}")
        at = 0
        for e, i in self._order():
            row = self.rows[e][i]; n = row[1].size
            row[1][...] = flat[at:at + n].reshape(row[1].shape); row[2][...] = row[1]
            at += n

# TWO MORE (1.10), the devices of a bridge's seismic design:
#
#   damper     a fluid viscous damper, F = c sign(v) |v|^alpha on the
#              component's deformation RATE (0 < alpha <= 1; alpha = 1 is the
#              linear dashpot `connector_damping` already is), beside an
#              optional spring `stiffness`. Acts in a dynamic step only, as
#              every dashpot does. BELOW `linear_below` THE LAW IS LINEAR,
#              F = c v0^(alpha-1) v: at v -> 0 the power law's slope is
#              infinite, and an explicit step chatters about v = 0 with the
#              force flipping sign every step. The linear piece is also what
#              gives the explicit bound dt < 2 m / c a coefficient to read.
#              No history.
#   pendulum   a friction pendulum (sliding isolation) bearing: Coulomb
#              friction mu N, as `friction`, plus the restoring force of the
#              spherical surface, (N / radius) d -- both follow the
#              compression N the bearing carries at that instant (`normal:
#              ux`, or a number), which is what a bilinear law fitted to the
#              dead load cannot do under a vertical motion or an overturning
#              couple. With N at zero (uplift) it carries nothing. On uy and
#              uz together it is one law in the plane, as friction is.


def law_stiffness(law) -> float:
    """The elastic stiffness a law contributes to the reference matrix."""
    t = law["type"]
    if t == "material":
        area, length = _scale(law)
        return area * float(law_material(law).E) / length
    if t == "gap":
        return float(law["stiffness"]) if (law.get("lo") is not None or law.get("hi") is not None) else 0.0
    if t == "bilinear":
        return float(law["k1"])
    if t == "damper":
        return float(law.get("stiffness") or 0.0)
    return float(law["stiffness"])


def damper_linear(law) -> float:
    """The coefficient of a damper law's linear piece, c v0^(alpha-1) (c
    itself at alpha = 1): the largest slope dF/dv the law has."""
    c, a = float(law["c"]), float(law["exponent"])
    if a >= 1.0:
        return c
    return c * float(law["linear_below"]) ** (a - 1.0)


def damper_force(law, rate: float) -> float:
    """F = c sign(v) |v|^alpha, linear below `linear_below`."""
    c, a = float(law["c"]), float(law["exponent"])
    if a >= 1.0:
        return c * rate
    v0 = float(law["linear_below"])
    av = abs(rate)
    if av < v0:
        return c * v0 ** (a - 1.0) * rate
    return (c * av ** a) if rate > 0.0 else -(c * av ** a)


def damping_matrix(props) -> np.ndarray:
    """The dashpot matrix the explicit bound and a rank check read: the
    linear `connector_damping`, and every damper law's linear piece (its
    largest slope) on its own component."""
    C = matrix(props.get("connector_damping"))
    for name, law in (props.get("laws") or {}).items():
        if law["type"] == "damper":
            i = COMPONENTS.index(name)
            C[i, i] += damper_linear(law)
    return C


def _plastic(d, slip, k, fy, H):
    """Return mapping of a 1-D elastic-plastic law with linear kinematic
    hardening: force and the new slip, from the committed slip."""
    alpha = H * slip
    trial = k * (d - slip)
    xi = trial - alpha
    over = abs(xi) - fy
    if over <= 0.0:
        return float(trial), float(slip)
    dg = over / (k + H)
    sgn = 1.0 if xi > 0.0 else -1.0
    slip_new = slip + dg * sgn
    return float(k * (d - slip_new)), float(slip_new)


def _plastic_2d(d, slip, k, fy, H):
    """`_plastic` with a vector deformation: the circular yield surface."""
    alpha = H * slip
    trial = k * (d - slip)
    xi = trial - alpha
    nx = float(np.linalg.norm(xi))
    over = nx - fy
    if over <= 0.0:
        return trial, slip.copy()
    dg = over / (k + H)
    slip_new = slip + dg * xi / nx
    return k * (d - slip_new), slip_new


def _same_law(a, b) -> bool:
    return dict(a) == dict(b)


def law_force(law, d: float, slip: float, normal_force: float):
    """Force on the component and the updated slip, from the committed slip."""
    t = law["type"]
    if t == "gap":
        k = float(law["stiffness"])
        lo, hi = law.get("lo"), law.get("hi")
        if hi is not None and d > hi:
            return k * (d - float(hi)), slip
        if lo is not None and d < lo:
            return k * (d - float(lo)), slip
        return 0.0, slip
    if t == "bilinear":
        k1, k2, fy = float(law["k1"]), float(law["k2"]), float(law["fy"])
        H = k1 * k2 / (k1 - k2) if k1 > k2 else 0.0
        return _plastic(d, slip, k1, fy, H)
    if t == "damper":
        # the spring beside it; the rate-dependent part is added with the dashpots
        return float(law.get("stiffness") or 0.0) * d, slip
    # friction, and the pendulum's friction
    k = float(law["stiffness"])
    N = max(float(normal_force), 0.0)
    fy = float(law["mu"]) * N
    if fy <= 0.0:
        return 0.0, float(d)          # nothing to press against: it just slides
    f, s_new = _plastic(d, slip, k, fy, 0.0)
    if t == "pendulum":
        f += N / float(law["radius"]) * d
    return f, s_new


def _normal_force(law, K, d):
    """The compression the friction law presses against: minus the linear
    action of the named component (compression is negative deformation on
    the spring), or a constant. The named component is the local ux (the
    schema and `check_geometry` allow no other): the elongation, whose sign
    is the same whichever end the Line2 starts at."""
    n = law.get("normal")
    if isinstance(n, str):
        i = COMPONENTS.index(n)
        return -K[i, i] * d[i]
    return float(n)


def check_geometry(x0, props) -> None:
    """What a connector's settings cannot mean on this geometry.

    A friction law pressing against a NAMED component reads that component's
    sign as compression. On global axes the sign of a translation component
    flips with the order the Line2 lists its nodes in (Gmsh orders them
    arbitrarily): the same bearing written top-to-bottom carried 0 N of
    friction instead of mu W."""
    laws = props.get("laws") or {}
    if props.get("coordinate_system", "local") == "global":
        for name, law in laws.items():
            law = law if isinstance(law, dict) else dict(law)
            if law.get("type") in ("friction", "pendulum") and isinstance(law.get("normal"), str):
                raise ValueError(
                    f"friction on {name} presses against `normal: {law['normal']}`, a component of a "
                    "connector in GLOBAL axes, whose sign flips with the order of the Line2's nodes; "
                    "use coordinate_system: local (normal: ux is then the elongation) or give the "
                    "normal force as a number")


def reference_stiffness(x0, R0, props) -> np.ndarray:
    check_geometry(x0, props)
    J0 = jacobian(np.zeros((2, 3)), None, x0, R0, props)
    return J0.T@effective_stiffness(props)@J0


def dashpot_rows(ids, elements, nodes, R0, props_of, dofs, ndof) -> np.ndarray:
    """Row sums of |J^T C J| of the connector dashpots, assembled: the
    explicit bound dt < 2 m / c reads them (v1.0: the fibre solver
    had no such bound, and a stiff dashpot went non-finite)."""
    rows = np.zeros(ndof)
    for e in ids:
        props = props_of(e)
        C = damping_matrix(props)
        if not np.any(C):
            continue
        conn = elements[e]
        J0 = jacobian(np.zeros((2, 3)), None, nodes[conn], R0[e], props)
        rows[dofs.element_dofs(conn)] += np.sum(np.abs(J0.T @ C @ J0), axis=1)
    return rows


def response(u, triads, ve, x0, R0, props, *, include_damping=False,
             slip_committed=None, slip_out=None, offset=None, materials=None):
    """Work-conjugate 12-DOF force, six connector actions and deformation.

    `u` is the two ends' translations, `triads` their orientation matrices.
    `ve` is the 12 velocity components in the solver's own order, whose
    rotational entries are spins -- which is what the Jacobian's rotational
    columns are now taken along, so `J v` is the connector's deformation
    rate. `slip_committed` (6,) is the history the `laws` start from and
    `slip_out` (6,) receives the history they arrive at; both are ignored
    when the connector has no laws. `offset` (6,) is the deformation the
    connector was born with (a connector built by a later analysis step is
    stress-free where it stands): its springs, curves and laws
    read the deformation less that, and so does what is returned.
    `materials` is `MaterialStates.of(e)` for a connector with `material`
    laws: {component: [Material, committed state, working state]}."""
    d, J = _deformation_and_jacobian(u, triads, x0, R0, props)
    if offset is not None:
        d = d - np.asarray(offset, dtype=float)
    K = matrix(props.get("stiffness"))
    action = K@d
    for name, table in (props.get("curves") or {}).items():
        i = COMPONENTS.index(name)
        action[i] += curve_force(d[i], table) - K[i, i]*d[i]
    laws = props.get("laws") or {}
    open_gaps = []
    if laws:
        sc = np.zeros(6) if slip_committed is None else np.asarray(slip_committed, float)
        planar = ("uy" in laws and "uz" in laws and laws["uy"]["type"] in ("friction", "bilinear", "pendulum")
                  and _same_law(laws["uy"], laws["uz"]))
        for name, law in laws.items():
            i = COMPONENTS.index(name)
            if planar and name in ("uy", "uz"):
                continue
            if law["type"] == "material":
                from ..materials.composite import uniaxial_stress
                area, length = _scale(law)
                strain = float(d[i]) / length
                if materials is not None and i in materials:
                    m, committed, work = materials[i]
                    work[...] = committed
                else:                          # a probe with no history kept: from the virgin state
                    m = law_material(law); work = m.init_state(1)
                action[i] = area * float(uniaxial_stress(m, np.array([strain]), work, 1.0)[0])
                if slip_out is not None:
                    slip_out[i] = strain
                continue
            N = _normal_force(law, K, d) if law["type"] in ("friction", "pendulum") else 0.0
            f, s_new = law_force(law, float(d[i]), float(sc[i]), N)
            action[i] = f
            if slip_out is not None:
                slip_out[i] = s_new
            if law["type"] == "gap" and f == 0.0:
                open_gaps.append(i)
        if planar:
            law = laws["uy"]
            dp = d[1:3]; sp = sc[1:3]
            if law["type"] == "bilinear":
                k1, k2, fy = float(law["k1"]), float(law["k2"]), float(law["fy"])
                f2, s2 = _plastic_2d(dp, sp, k1, fy, k1 * k2 / (k1 - k2) if k1 > k2 else 0.0)
            else:
                N = max(float(_normal_force(law, K, d)), 0.0)
                fy = float(law["mu"]) * N
                if fy <= 0.0:
                    f2, s2 = np.zeros(2), dp.copy()
                else:
                    f2, s2 = _plastic_2d(dp, sp, float(law["stiffness"]), fy, 0.0)
                    if law["type"] == "pendulum":
                        f2 = f2 + N / float(law["radius"]) * dp
            action[1:3] = f2
            if slip_out is not None:
                slip_out[1:3] = s2
    if include_damping:
        rate = J@np.asarray(ve, dtype=float).reshape(12)
        damp = matrix(props.get("connector_damping"))@rate
        for name, law in laws.items():
            if law["type"] == "damper":
                i = COMPONENTS.index(name)
                damp[i] += damper_force(law, float(rate[i]))
        # A DASHPOT ON A GAP ACTS IN CONTACT ONLY: the
        # Kelvin pounding model. It acted with the gap open: 2 mm into a
        # 6 mm gap, closing at 0.5 m/s, the stop already pushed 5000 kN.
        damp[open_gaps] = 0.0
        action += damp
    fe = J.T@action
    if props.get("coordinate_system", "local") == "global":
        # GLOBAL AXES ARE NOT OBJECTIVE for a connector with length: a rigid
        # rotation strains springs fixed in space, and the forces conjugate
        # to that energy do not balance moments -- a 0.05 m bearing under
        # 10 MN at an offset of 0.2 m put the whole P-Delta moment, 2 MN m,
        # into the structure. The couple the two ends'
        # forces leave is returned half to each end, as a link element's
        # P-Delta split does.
        x = np.asarray(x0, float).reshape(2, 3) + np.asarray(u, float).reshape(2, 3)
        net = np.cross(x[0], fe[0:3]) + np.cross(x[1], fe[6:9]) + fe[3:6] + fe[9:12]
        fe[3:6] -= 0.5 * net
        fe[9:12] -= 0.5 * net
    return fe, action, d
