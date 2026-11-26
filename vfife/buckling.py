"""Linear buckling: the load factors at which a frame loses its stiffness (1.1).

    (K_b + lambda K_G) phi = 0

* K_b, the stiffness at the base state, is the finite-difference tangent
  `vfife modal` uses (`testing.invariants.tangent`): whatever the families
  put into their internal force -- connectors, bearings, fibre sections, the
  stress stiffness the base state already carries -- is in it.
* K_G is the stress stiffness of the step's loads: the members' axial forces
  and bending moments in the linear increment u = K_b^-1 F, each member's
  consistent geometric matrix, assembled. For a beam it is the second
  variation of the work of the section's stresses on the nonlinear part of
  the axial strain (1.5; 1.1 had the axial force alone),

      U_G = int [ N/2 (v'^2 + w'^2) + N Ip/(2A) theta'^2
                  + M_z theta w'' + M_y theta v'' ] dx,

  v, w the deflections along local y and z (cubic), theta the twist
  (linear), N the axial force (tension +), M_y, M_z the internal moments
  `bending_y` and `bending_z` (linear between the ends, core/member_actions)
  and Ip/A the section's (Iy + Iz)/A: the axial force across each bending
  plane (N / 30 L [36, 3L, 4L^2, ...]), the Wagner term that buckles a
  column in twist, and the moments' coupling of twist with bending out of
  their plane that buckles a beam laterally. A bar or cable has the string
  stiffness N / L (I - e e^T).

WHY NOT THE DIFFERENCE OF TWO TANGENTS. The first version took K_G as
(K(e u) - K_b) / e, the tangent's own rate along the increment, so that no
element matrix would have to be written. A tangent taken at a deformed state
also holds every member's stiffness ROTATED with it, a first-order term of
size EA times the rotation, and under any load that bends a member before it
buckles the eigenproblem read those rotations as softening: a cantilever with
a lateral load of 2% of its axial one "buckled" at a third of its Euler load,
a portal loaded on its beam at a tenth, and the error grew with the mesh and
with EA (review of 1.1). Taking the tangent along the path instead removed
that, and with it the buckling of any column that carried a lateral load:
an imperfect column has no bifurcation, and its tangent never becomes
singular. Linear buckling is the eigenproblem of the stress stiffness alone
-- which is what this is.

What is in it: flexural buckling of columns, piers, towers, frames and
arches (in and out of plane); torsional buckling of a column (the Wagner
term); lateral-torsional buckling of beams and girders (1.5). What is not:

* WARPING. A member has Saint-Venant torsion (G J) only, so an open
  thin-walled section -- an I girder -- buckles laterally at the critical
  moment of its St Venant stiffness, below the true one by
  sqrt(1 + pi^2 E Iw / (G J L^2)): on the safe side, and much on it for a
  short, deep I beam. A box, a solid or a narrow rectangle is not affected.
* Doubly symmetric sections: the shear centre is the centroid, and a
  monosymmetric section's Wagner coefficient (a T girder) is not in it.
* Loads at the centroid: the height of a load above it (a deck on top of
  the girder) is not in it.
* The torque enters through the co-rotational part only: a member under
  torque is turned as the solver turns it (its moments are spatial,
  conjugate to the spins), and a shaft under torque alone buckles where
  the nonlinear model does -- not at a textbook Greenhill load, which
  depends on how the torque is applied.
* An applied moment is treated as the end of a member (no load stiffness
  of its own): at a fork support, where the twist is held, every kind of
  moment gives the same answer.

THE BENDING BEFORE BUCKLING. The linear theory ignores it, and a member
whose two second moments are close -- a round pier, a square or box
section -- or one bent about its minor axis then "buckles laterally" where
it does not: the 1.5 draft found the Bathe 45-degree bend (a square) at
0.86 of the load it carries, and a girder bent about its minor axis at
7800 times its load, both ways, while the nonlinear model's tangent
stiffened. Each plane's moments are therefore scaled by Kirchhoff's
factor sqrt((1 - E Iz / E Iy)(1 - G J / E Iy)) (bent about y; zero when
Iy <= Iz or G J >= E Iy, each bracket clamped), exact for a uniform moment
on forks and close otherwise (Pi and Trahair; a cantilever 2-4%). Where it
is zero the plane's moments and shears leave K_G altogether, the
co-rotational part's included: the critical moment of a narrow rectangle rises by 0.3%,
of a section with Iz = Iy / 4 by 17%, and the nonlinear model buckles
within 2% of it (it was 1.05 to 1.3 times the linear theory's). The
textbook value, which design codes use, is the reported one times that
factor.

Shells and solids are refused (no member forces to build K_G from).

The base state is the reference configuration (the default), which must be
at rest -- a prestress, an imposed strain or a temperature there is an
internal force no load balances (`BUCKLING_BASE_NOT_AT_REST`) -- or the
state a run reached (`state=`, `vfife buckling --result`), whose histories
(cracking, yielding, slipped bearings) and existing axial forces are in K_b;
lambda is then the factor on the step's loads ADDED to it. lambda multiplies
every load of the step, self-weight included, at the amplitude's value at
the step's end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core.diagnostics import UserError
from .testing import invariants
from .testing.state import probe as make_probe

#: the finite-difference step of the tangent, as a fraction of the model's
#: size (not of its largest coordinate: a bridge in survey coordinates,
#: 5e5 m east, differenced at 0.05 m and was 21 times off; review of 1.1)
STEP = 1.0e-7
#: |mu| below this fraction of the largest is zero
NOISE = 1.0e-8
#: the axial forces are differenced at a state whose largest translation is
#: this fraction of the model's size (and at its mirror image)
PROBE = 1.0e-6
#: ... and at most this fraction of the loads
LOAD_PROBE = 1.0e-4
#: the buckling file's layout version
BUCKLING_FORMAT = 1


@dataclass
class BucklingResult:
    factors: np.ndarray              #: load factors, ascending, positive
    shapes: np.ndarray               #: (n_modes, ndof), zero on held dofs, unit largest translation
    method: str = "reference"        #: "reference" or "state" (the base)
    n_free: int = 0
    n_negative: int = 0              #: factors < 0 (the loads reversed)
    first_negative: float = float("nan")
    load: np.ndarray = field(default_factory=lambda: np.zeros(3))   #: the step's load on the free dofs
    axial_force: np.ndarray = field(default_factory=lambda: np.zeros(0))   #: per member, tension +
    note: str = ""
    rotational: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    def table(self) -> str:
        lines = ["  mode    load factor"]
        rot = self.rotational if len(self.rotational) == len(self.factors) else [False] * len(self.factors)
        for i, (lam, r) in enumerate(zip(self.factors, rot), 1):
            lines.append(f"  {i:4d}  {lam:13.6g}" + ("   rotation only" if r else ""))
        return "\n".join(lines)


def _size(p) -> float:
    nodes = np.asarray(p.nodes, float)
    return float(np.ptp(nodes, axis=0).max()) or 1.0


def _family(solver) -> str:
    from .core.coupled_solver import CoupledSolver
    from .core.shell_solver import ShellSolver
    from .core.solver import Solver
    from .core.truss_solver import TrussSolver
    if isinstance(solver, (Solver, ShellSolver, CoupledSolver)):
        raise UserError("BUCKLING_FAMILY",
                f"buckling is for line models (frames, fibre beams, bars and cables); this is a "
                f"{type(solver).__name__}: its stress stiffness is built from member axial forces, "
                "which shells and solids do not have",
                "a shell or solid's stability is asked with a nonlinear static run (`mode: static`, "
                "arc length) to its limit point")
    return "truss" if isinstance(solver, TrussSolver) else "framed"


def _axial_forces(solver, p, x, family):
    """Each member's mean axial force at state x, tension positive: the
    average of its two ends' internal forces (equal without a load along the
    member, and the mean with one)."""
    s = solver
    if family == "truss":
        keep_u, keep_state = s.u.copy(), s.state.copy()
        s.u[:] = np.asarray(x, float).reshape(-1, 3)
        try:
            _F, sig = s._force(want_stress=True)
            fields = s.nodal_fields(sig)
        finally:
            s.u[:] = keep_u
            s.state[...] = keep_state
        return np.asarray(fields["axial_force"], float)
    keep_q = s.q.copy()
    keep_orient = s.orient.copy() if getattr(s, "orient", None) is not None else None
    was_trial = getattr(s, "_trial_mode", None)
    if hasattr(s, "_trial_mode"):
        s._trial_mode = True
    s.q[:] = np.asarray(x, float)
    if hasattr(s, "sync_orientation"):
        s.sync_orientation()
    try:
        if hasattr(s, "_probe"):
            with s._probe():
                _f, actions = s.internal_force()
                fields = s.nodal_fields(actions)
        else:
            _f, actions = s.internal_force()
            fields = s.nodal_fields(actions)
    finally:
        s.q[:] = keep_q
        if keep_orient is not None:
            s.orient[...] = keep_orient
        if hasattr(s, "sync_orientation"):
            s.sync_orientation()
        if was_trial is not None:
            s._trial_mode = was_trial
    if all(f"{n}_{e}" in fields for n in _END_NAMES for e in "ij"):
        # the end actions (the forces the nodes exert on the member, local
        # axes, f = k d: core/member_actions), all twelve
        i = np.column_stack([np.asarray(fields[f"{n}_i"], float) for n in _END_NAMES])
        j = np.column_stack([np.asarray(fields[f"{n}_j"], float) for n in _END_NAMES])
        return np.hstack([i, j])
    N = np.asarray(fields["axial_force"], float)
    z = 0 * N
    return np.column_stack([-N, z, z, z, z, z, N, z, z, z, z, z])


#: the end actions K_G is built from, in their order in `f = k d`
_END_NAMES = ("axial_force", "shear_y", "shear_z", "torsion", "bending_y", "bending_z")


def _beam_kg(N, L, My=(0.0, 0.0), Mz=(0.0, 0.0), rp2=0.0):
    """The consistent geometric matrix of a beam, 12 x 12, local axes
    [u v w rx ry rz] at each end: the second variation of U_G (the module's
    docstring) with the axial force N (tension +), the internal moments
    My = (M_y at end i, at end j) and Mz likewise, linear between, and
    rp2 = Ip / A. Cubic Hermite deflections (ry = -w'), linear twist,
    integrated exactly (four Gauss points; the integrands are of degree 4)."""
    k = np.zeros((12, 12))
    xg, wg = np.polynomial.legendre.leggauss(4)
    iv, iw, it = [1, 5, 7, 11], [2, 4, 8, 10], [3, 9]
    for x, wt in zip(0.5 * (xg + 1.0), 0.5 * wg):
        d1 = np.array([-6 * x + 6 * x * x, L * (1 - 4 * x + 3 * x * x), 6 * x - 6 * x * x,
                       L * (-2 * x + 3 * x * x)]) / L
        d2 = np.array([-6 + 12 * x, L * (-4 + 6 * x), 6 - 12 * x, L * (-2 + 6 * x)]) / L ** 2
        sw = np.array([1.0, -1.0, 1.0, -1.0])           # w's rotations are -ry
        v1, v2 = np.zeros(12), np.zeros(12)
        w1, w2 = np.zeros(12), np.zeros(12)
        t0, t1 = np.zeros(12), np.zeros(12)
        v1[iv], v2[iv] = d1, d2
        w1[iw], w2[iw] = d1 * sw, d2 * sw
        t0[it] = (1.0 - x, x)
        t1[it] = (-1.0 / L, 1.0 / L)
        my = My[0] + (My[1] - My[0]) * x
        mz = Mz[0] + (Mz[1] - Mz[0]) * x
        g = N * (np.outer(v1, v1) + np.outer(w1, w1)) + N * rp2 * np.outer(t1, t1)
        g += mz * (np.outer(t0, w2) + np.outer(w2, t0)) + my * (np.outer(t0, v2) + np.outer(v2, t0))
        k += wt * L * g
    return k


def _has_kinked_laws(solver) -> bool:
    """Whether the model has a response with a kink at its base state: a
    connector with a law or a curve (a gap at its stop, a curve's origin), a
    tension-only member, soil springs."""
    if np.any(np.asarray(getattr(solver, "member_tension_only", False), bool)):
        return True
    if bool(getattr(getattr(solver, "pre", None), "tension_only", False)):
        return True
    soil = getattr(solver, "soil", None)
    if soil is not None and (not hasattr(soil, "__len__") or len(soil)):
        return True
    kinds = getattr(solver, "member_kind", None)
    ids = getattr(solver, "_connector_ids", None)
    ids = list(np.asarray(ids).ravel()) if ids is not None else []
    if kinds is not None:
        ids += [int(e) for e in np.flatnonzero(np.asarray(kinds) == "connector2")]
    el = (getattr(solver, "p", None) or {}).get("element") or {}
    if el.get("type") == "connector2" and (el.get("laws") or el.get("curves")):
        return True
    for e in set(int(e) for e in ids):
        if getattr(solver, "_connector_props", None) is not None and hasattr(solver, "_connector_of"):
            pr = solver._connector_props[int(solver._connector_of[e])]
        elif getattr(solver, "member_props", None) is not None:
            pr = solver.member_props[e]
        else:
            pr = el
        if (pr or {}).get("laws") or (pr or {}).get("curves"):
            return True
    return False


def _kirchhoff(solver, e):
    """(c_y, c_z): the factors on member e's moments in K_G for the bending
    the moments themselves cause before it buckles (Kirchhoff; Pi and
    Trahair): bent about y by M_y, the member's critical moment is the
    linear one over sqrt((1 - E Iz / E Iy)(1 - G J / E Iy)). Exact for a
    uniform moment on forks, close for other moment diagrams; 0 when the
    bending is about the minor axis (or Iz = Iy), where a member does not
    buckle laterally at all."""
    if hasattr(solver, "total_area") and hasattr(solver, "Iy"):              # fibre sections
        Iy, Iz, J = float(solver.Iy[e]), float(solver.Iz[e]), float(solver.J[e])
        g = float(solver.member_G[e]) / float(solver.member_E[e])
    else:
        props = solver.member_props[e] if hasattr(solver, "member_props") else solver.p["element"]
        mat = solver.member_material[e] if hasattr(solver, "member_material") else solver.material
        try:
            Iy, Iz, J = float(props["iy"]), float(props["iz"]), float(props["j"])
        except (KeyError, TypeError):
            return 1.0, 1.0
        g = 1.0 / (2.0 * (1.0 + float(getattr(mat, "nu", 0.3))))

    def c(major, minor):
        # each bracket clamped, not their product: bent about its minor axis
        # a solid rectangle narrower than 0.56 has G J > E I there too, and
        # two negative brackets made a factor of 14 (review of 1.5)
        if major <= 0.0 or minor >= major or g * J >= major:
            return 0.0
        return float(np.sqrt((1.0 - minor / major) * (1.0 - g * J / major)))
    return c(Iy, Iz), c(Iz, Iy)


def _no_tension_laws(solver) -> list:
    """The names of the fibre materials whose stiffness is a kink at zero
    strain -- no tension at all (Concrete01, a Concrete02 with ft = 0): at
    the reference configuration every fibre of theirs sits on it, and the
    central-difference K_b is the mean of the compression stiffness and
    zero (a column under axial load alone buckles at half its Euler load,
    review of 1.5). From a loaded state the fibres are off it."""
    from .materials.concrete01 import Concrete01
    mats = getattr(solver, "materials", None)
    if mats is None or not hasattr(solver, "fiber_area"):
        return []
    items = mats.items() if isinstance(mats, dict) else enumerate(mats)
    out = []
    for name, m in items:
        if isinstance(m, Concrete01) or (hasattr(m, "ft") and hasattr(m, "fc") and float(m.ft) == 0.0):
            out.append(type(m).__name__.lower() if not isinstance(mats, dict) else str(name))
    return sorted(set(out))


def _polar_ratio(solver, e) -> float:
    """(Iy + Iz) / A of member e, the Wagner term's radius squared."""
    if hasattr(solver, "total_area") and hasattr(solver, "Iy"):            # fibre sections
        A = float(solver.total_area[e])
        return (float(solver.Iy[e]) + float(solver.Iz[e])) / A if A > 0 else 0.0
    props = solver.member_props[e] if hasattr(solver, "member_props") else solver.p["element"]
    try:
        A = float(props["area"])
        return (float(props["iy"]) + float(props["iz"])) / A if A > 0 else 0.0
    except (KeyError, TypeError):
        return 0.0


def increment_forces(solver, p, xb, u, family):
    """The members' end actions of the LINEAR increment u from xb, per
    unit of it -- (nel, 12), f = k d in local axes (core/member_actions);
    a bar model's axial forces (nel,) -- a central difference at a small multiple of u. Read at xb + u
    itself, a member that the increment bends grows longer by L theta^2 / 2,
    and that second-order tension swamped the compression it carries: a
    cantilever with a lateral load of 10% of its axial one read 44 times its
    Euler load (review of 1.1). The difference also cancels whatever the
    fields subtract that does not scale (the fixed-end forces of a load
    inside a member)."""
    t = float(np.abs(u[p.translation]).max()) if p.translation.size else float(np.abs(u).max())
    # small in displacement AND in load: an axial increment moves little,
    # and 1e-6 of the size there was a fifth of the load
    e = min(PROBE * _size(p) / t, LOAD_PROBE) if t > 0 else LOAD_PROBE
    Np = _axial_forces(solver, p, xb + e * u, family)
    Nm = _axial_forces(solver, p, xb - e * u, family)
    return (Np - Nm) / (2.0 * e)


def _at_ends(xx, TT, off):
    """The elastic ends of members whose nodes are at xx (ne, 2, 3) with
    triads TT, and the offsets now (Q e), for offsets `off` (ne, 2, 3)
    (core/member_geometry); (xx, None) without them."""
    if off is None:
        return xx, None
    r = np.einsum("enij,enj->eni", TT, off)
    return xx + r, r


def _cr_stress_stiffness(x, T, R0, f, scale, off=None):
    """The stress part of the co-rotational tangent, per member: d(R^T f)/d
    (state) with the end actions f (ne, 12) held, R the member frames of
    `rotations.beam_frames_from_triads` at end positions x (ne, 2, 3) and
    triads T (ne, 2, 3, 3) -- the frame every line solver's member forces
    are rotated back with. Differenced along the probe's coordinates:
    translations, and spatial spins T <- exp(h e) T. (ne, 12, 12).

    With end offsets `off` the state is the NODES': the frame is the
    elastic ends', and the moments at the nodes take the levers r x f with
    r = Q e turned by the spin -- so the derivative carries B^T (.) B and
    the lever's own term (d theta x r) x f, the rigid stretch's string
    stiffness."""
    from .core.rotations import beam_frames_from_triads, exp_map
    ne = len(f)
    fb = f.reshape(ne, 4, 3)
    K = np.zeros((ne, 12, 12))
    ht, hr = 1e-6 * scale, 1e-6
    for col in range(12):
        node, kind, comp = col // 6, (col % 6) // 3, col % 3
        out = []
        for sgn in (1.0, -1.0):
            xx, TT = x.copy(), T.copy()
            if kind == 0:
                xx[:, node, comp] += sgn * ht
            else:
                w = np.zeros(3); w[comp] = sgn * hr
                TT[:, node] = np.einsum("ij,ejk->eik", exp_map(w), T[:, node])
            xe, r = _at_ends(xx, TT, off)
            R, _theta, _L = beam_frames_from_triads(xe, TT, R0)
            g = np.einsum("eji,ekj->eki", R, fb).reshape(ne, 12)
            if r is not None:
                g[:, 3:6] += np.cross(r[:, 0], g[:, 0:3])
                g[:, 9:12] += np.cross(r[:, 1], g[:, 6:9])
            out.append(g)
        K[:, :, col] = (out[0] - out[1]) / (2.0 * (ht if kind == 0 else hr))
    return K


def _cr_local_jacobian(x, T, R0, L0, scale, off=None):
    """d(local deformation)/d(state) of each member at (x, T): the twelve
    local quantities of `f = k d` -- the elongation at index 6, the two
    ends' local rotations at 3:6 and 9:12, the rest zero (the frame passes
    through both ends). (ne, 12, 12)."""
    from .core.rotations import beam_frames_from_triads, exp_map
    ne = len(x)
    B = np.zeros((ne, 12, 12))
    ht, hr = 1e-6 * scale, 1e-6
    for col in range(12):
        node, kind, comp = col // 6, (col % 6) // 3, col % 3
        out = []
        for sgn in (1.0, -1.0):
            xx, TT = x.copy(), T.copy()
            if kind == 0:
                xx[:, node, comp] += sgn * ht
            else:
                w = np.zeros(3); w[comp] = sgn * hr
                TT[:, node] = np.einsum("ij,ejk->eik", exp_map(w), T[:, node])
            _R, theta, L = beam_frames_from_triads(_at_ends(xx, TT, off)[0], TT, R0)
            d = np.zeros((ne, 12))
            d[:, 6] = L - L0
            d[:, 3:6], d[:, 9:12] = theta[:, 0], theta[:, 1]
            out.append(d)
        B[:, :, col] = (out[0] - out[1]) / (2.0 * (ht if kind == 0 else hr))
    return B


def stress_stiffness(solver, p, N, family=None, x=None):
    """K_G (n, n) of the members' increment forces: a (nel,) array of axial
    forces (tension +), or `increment_forces`' (nel, 12) end actions, at the
    state x (the probe's; the reference configuration when None). Returns
    (K_G, the mean axial forces).

    A beam's K_G has two parts (1.5):

    * the stress part of the co-rotational tangent, d(R^T f)/d(state) with
      the end actions f held: what turning the member frame does to the
      forces it carries -- the string stiffness N/L of the chord, and the
      end moments and shears turned with the frame. This is the members'
      own kinematics, the one their forces are rotated back with, so a
      moment carried round a joint is turned there as the solver turns it.
      With the continuum terms alone (the 1.5 draft: U_G on the whole
      member, end deflections included) Argyris' right-angle frame, a
      cantilever L in its plane pushed at its tip, buckled out of its plane
      4.5% above the load at which the nonlinear model's own tangent turns
      singular, and 7.7% below it pushed the other way; with this part,
      0.2% and 0.4% (the rest is the pre-buckling deflection);
    * the second-order terms INSIDE the member (U_G of the module's
      docstring) on its local end rotations: the curvature between the
      chord and the ends, the twist's Wagner term and the moments' coupling
      of local twist with local bending, which the linear local element
      does not have and a mesh would otherwise need many elements to find.

    For a straight member under an axial force alone the two parts are the
    consistent geometric matrix of 1.1, to round-off."""
    family = family or _family(solver)
    ends = np.asarray(N, float)
    if ends.ndim == 1:
        z = 0 * ends
        ends = np.column_stack([-ends, z, z, z, z, z, ends, z, z, z, z, z])
    N = 0.5 * (ends[:, 6] - ends[:, 0])
    n = p.n
    KG = np.zeros((n, n))
    nodes = np.asarray(solver.mesh.nodes, float)
    conn = np.asarray(solver.mesh.elements)[:, :2]
    if family == "truss":
        for e, (a, b) in enumerate(conn):
            d = nodes[b] - nodes[a]
            L = float(np.linalg.norm(d))
            if L <= 0.0 or N[e] == 0.0:
                continue
            ex = d / L
            G = N[e] / L * (np.eye(3) - np.outer(ex, ex))
            ia, ib = np.arange(3 * a, 3 * a + 3), np.arange(3 * b, 3 * b + 3)
            KG[np.ix_(ia, ia)] += G; KG[np.ix_(ib, ib)] += G
            KG[np.ix_(ia, ib)] -= G; KG[np.ix_(ib, ia)] -= G
        return KG, N
    kinds = getattr(solver, "member_kind", None)
    ids = getattr(solver, "_connector_ids", None)
    skip = set(int(i) for i in (np.asarray(ids).ravel() if ids is not None else []))
    R0 = np.asarray(solver.R0, float)
    lengths = np.asarray(getattr(solver, "length", None) if getattr(solver, "length", None) is not None
                         else np.linalg.norm(nodes[conn[:, 1]] - nodes[conn[:, 0]], axis=1), float)
    beams = []
    for e, (a, b) in enumerate(conn):
        kind = kinds[e] if kinds is not None else "beam2"
        if kind == "connector2" or e in skip or not np.any(ends[e]) or float(lengths[e]) <= 0.0:
            continue
        dof = solver.dofs.element_dofs(np.array([a, b]))
        if kind == "truss2":
            L = float(lengths[e])
            ex = R0[e][0]
            G = N[e] / L * (np.eye(3) - np.outer(ex, ex))
            ta, tb = dof[:3], dof[6:9]
            KG[np.ix_(ta, ta)] += G; KG[np.ix_(tb, tb)] += G
            KG[np.ix_(ta, tb)] -= G; KG[np.ix_(tb, ta)] -= G
            continue
        beams.append(e)
    if not beams:
        return KG, N
    from .core.dof import TRANSLATION
    from .testing.state import _orient_of
    beams = np.asarray(beams)
    x = p.get() if x is None else np.asarray(x, float)
    pos = nodes[:solver.nn] + solver.dofs.field(x, TRANSLATION)
    orient = _orient_of(solver, x) if solver.dofs.stride == 6 else np.repeat(np.eye(3)[None], solver.nn, 0)
    c = conn[beams]
    xe, Te = pos[c], orient[c]
    from .core import member_geometry
    geometry = member_geometry.of(solver)
    off = None if not geometry.any else np.asarray(geometry.offset, float)[beams]
    scale = float(np.median(lengths[beams]))
    # the end actions in four groups, K_G being linear in them: the axial
    # force, the torque, and the two planes' bending (shear and moment) --
    # each plane's scaled by its Kirchhoff factor
    fe = ends[beams]
    cyz = np.array([_kirchhoff(solver, e) for e in beams])
    groups = {"axial": [0, 6], "torque": [3, 9], "y": [2, 4, 8, 10], "z": [1, 5, 7, 11]}
    Kcr = np.zeros((len(beams), 12, 12))
    for name, cols in groups.items():
        part = np.zeros_like(fe)
        part[:, cols] = fe[:, cols]
        if name == "y":
            part *= cyz[:, :1]
        elif name == "z":
            part *= cyz[:, 1:]
        if np.any(part):
            Kcr += _cr_stress_stiffness(xe, Te, R0[beams], part, scale, off)
    Bl = _cr_local_jacobian(xe, Te, R0[beams], lengths[beams], scale, off)
    for k, e in enumerate(beams):
        L = float(lengths[e])
        f = ends[e]
        cy, cz = cyz[k]
        # the internal moments at the two ends: -f_i at i, +f_j at j
        loc = _beam_kg(N[e], L, My=(-cy * f[4], cy * f[10]), Mz=(-cz * f[5], cz * f[11]),
                       rp2=_polar_ratio(solver, e))
        # on the local deformation: v = w = 0 at the ends (the frame passes
        # through both), so the chord's share of N -- the string stiffness
        # of the co-rotational part -- is not counted twice, and what is
        # left is the curvature between the chord and the ends, the twist
        # relative to the frame, and the moments' coupling of the two
        Kloc = Bl[k].T @ loc @ Bl[k]
        dof = solver.dofs.element_dofs(np.array(conn[e]))
        KG[np.ix_(dof, dof)] += Kcr[k] + Kloc
    return KG, N


def analyse(solver, n_modes=6, state=None, max_dofs=6000, h=None) -> BucklingResult:
    """The lowest `n_modes` buckling factors of `solver`'s current step's
    loads, from its reference configuration -- or, with `state` (the solver
    already holding it, as `modal.restore_state` leaves it), from that state."""
    import scipy.linalg
    from .modal import physical_mass
    from .core import constraints
    ties = constraints.of(solver)
    family = _family(solver)
    p = make_probe(solver)
    if p.n > max_dofs:
        raise ValueError(f"{p.n} dofs is past max_dofs = {max_dofs}: the tangent costs {2 * p.n} "
                         "force evaluations")
    size = _size(p)
    h = STEP * size if h is None else h
    xb = p.get() if state is None else np.asarray(state, float)
    free = np.flatnonzero(~p.fixed)
    Kb, _ = invariants.tangent(p, xb, h=h, external=False)
    # the tangent's asymmetry: round-off of the finite difference, and at a
    # loaded state of a path-dependent section the two branches a fibre
    # straddles (3e7 for concrete01 under its load). A stiffness under the
    # relative threshold is a mechanism only if it is under the asymmetry
    # too: the weak bending of a cantilever on a fine mesh of a narrow,
    # deep section (3e-11 of the largest stiffness at 64 elements, an
    # asymmetry of 3 against an eigenvalue of 41) was refused as a
    # mechanism by the relative threshold alone (review of 1.5)
    noise = float(np.abs(Kb - Kb.T).max())
    Kb = 0.5 * (Kb + Kb.T)
    F = np.asarray(p.external(xb), float)
    comp = np.full(p.n, -1)
    comp[p.translation] = np.tile(np.arange(3), p.nn)
    on = np.zeros(p.n, bool); on[free] = True
    load = np.array([F[(comp == k) & on].sum() for k in range(3)])
    scale = float(np.abs(Kb[np.ix_(free, free)]).max()) or 1.0
    # degrees of freedom that carry no stiffness at all (a spare particle
    # kept for a split, a node no member reaches) are no part of it
    live = free[np.abs(Kb[np.ix_(free, free)]).max(axis=1) > 1e-12 * scale]
    T = None
    if ties is not None and not ties.empty():
        # WITH MULTI-POINT CONSTRAINTS LIVENESS IS THE REDUCED SYSTEM'S: a
        # link's master reached only through its slaves has no stiffness of
        # its own, and dropping its dof locked the slaves in it and lost their
        # loads -- a cantilever whose tip is a slave "had no positive factor"
        # (found in review, 1.21). The map is taken over every free dof, and
        # the masters that carry nothing in T^T K T are dropped there.
        live = free
        T = ties.free_map(p, live, physical_mass(solver, p))
        Kr = np.asarray(T.T @ np.asarray((T.T @ Kb[np.ix_(live, live)]).T))
        T = T[:, np.flatnonzero(np.abs(Kr).max(axis=1) > 1e-12 * scale)].tocsr()
    if not np.any(F[live] if T is None else T.T @ F[live]):
        if np.any(F[free]):
            raise ValueError("the loads act where the model has no stiffness at its base state (a slack "
                             "cable, a tension-only member): ask at a loaded state with --result")
        raise ValueError("the step has no load on the free degrees of freedom: a buckling factor "
                         "multiplies the step's loads")
    if state is None:
        # THE BASE STATE MUST BE AT REST. A tendon's prestress, a shrinkage or
        # a temperature is an internal force at the reference configuration
        # with no load to balance it; from there case_24f's prestressed girder
        # "buckled" at 1.7 times its dead load
        fi = np.asarray(p.internal(xb), float)
        if float(np.abs(fi[live]).max()) > 1e-6 * float(np.abs(F[live]).max()):
            raise UserError("BUCKLING_BASE_NOT_AT_REST",
                    f"the reference configuration carries internal forces of its own (up to "
                    f"{float(np.abs(fi[live]).max()):.4g}, against loads of {float(np.abs(F[live]).max()):.4g}): "
                    "prestress, an imposed strain or a temperature",
                    "run the case to the state it is in under them and ask there: `vfife buckling <case> "
                    "--result <result.h5>` (the factor is then on the loads added to that state)")

    def reduce(M):
        M = M[np.ix_(live, live)]
        if T is not None:
            M = np.asarray(T.T @ np.asarray((T.T @ M).T))
        return 0.5 * (M + M.T)

    B = reduce(Kb)
    ev = np.linalg.eigvalsh(B)
    unstable = lambda ev: ev[0] <= 1e-10 * ev[-1] and ev[0] <= 2.0 * noise
    if not unstable(ev) and _has_kinked_laws(solver):
        # ON THE SIDE THE LOAD PUSHES: K_b again, a hair along the loads'
        # increment. A central difference at a law's kink -- a gap at its
        # stop, a bearing that sticks one way -- is the mean of its two
        # branches, and a strut braced by a gap at its stop buckled at the
        # brace's half stiffness however hard the load pushed into it.
        # ONLY WHERE THERE IS A LAW TO KINK (1.5): the tangent a hair along
        # the increment also turns every member's stiffness with the
        # increment's rotations, EA times about 1e-6, and against a narrow
        # girder's lateral and torsional stiffness that moved an L-shaped
        # cantilever's lateral buckling factor from 7.17 to 2.79 -- and
        # broke the +/- symmetry its mirror image demands
        Fr0 = F[live] if T is None else np.asarray(T.T @ F[live]).ravel()
        ur0 = scipy.linalg.solve(B, Fr0, assume_a="pos")
        u0 = np.zeros(p.n)
        u0[live] = ur0 if T is None else np.asarray(T @ ur0).ravel()
        t0 = float(np.abs(u0[p.translation]).max()) if p.translation.size else float(np.abs(u0).max())
        if t0 > 0.0:
            e0 = min(PROBE * size / t0, 1e-2 * LOAD_PROBE)
            Kf, _ = invariants.tangent(p, xb + e0 * u0, h=h, external=False)
            noise = max(noise, float(np.abs(Kf - Kf.T).max()))
            Kb = 0.5 * (Kf + Kf.T)
            B = reduce(Kb)
    # A MECHANISM IS REFUSED BY NAME: neither the solve nor the eigensolver
    # raises on a singular matrix, and a pinned leaning column came back with
    # factors of 5e-8 and no word (review of 1.1)
    ev = np.linalg.eigvalsh(B)
    if unstable(ev):
        raise UserError("BUCKLING_BASE_NOT_STABLE",
                "the stiffness at the base state is not positive definite on the free degrees of freedom: "
                + ("a mechanism (check the supports)" if state is None else
                   "the state has already lost its stability, or the model is a mechanism"),
                "`vfife probe <case>` names the free rigid motions; a state past its limit has no "
                "buckling factor left")
    Fr = F[live] if T is None else np.asarray(T.T @ F[live]).ravel()
    ur = scipy.linalg.solve(B, Fr, assume_a="pos")
    u = np.zeros(p.n)
    u[live] = ur if T is None else np.asarray(T @ ur).ravel()
    N = increment_forces(solver, p, xb, u, family)
    KG, N = stress_stiffness(solver, p, N, family, x=xb)
    A = -reduce(KG)
    mu, v = scipy.linalg.eigh(A, B)
    top = float(np.abs(mu).max()) or 1.0
    pos = np.flatnonzero(mu > NOISE * top)
    negative = mu[mu < -NOISE * top]
    order = pos[np.argsort(-mu[pos])][:n_modes]          # largest mu = smallest lambda
    factors = 1.0 / mu[order]
    red = v[:, order]
    full = np.zeros((p.n, len(order)))
    full[live] = red if T is None else np.asarray(T @ red)
    shapes = full.T.copy()
    rotational = np.zeros(len(order), bool)
    for i in range(len(order)):
        sh = shapes[i]
        tr = sh[p.translation] if p.translation.size else sh
        rotational[i] = bool(p.rotation.size) and float(np.abs(tr).max()) <= 1e-6 * float(np.abs(sh).max())
        pick = sh if rotational[i] else tr
        amp = pick[int(np.argmax(np.abs(pick)))] if pick.size else 1.0
        shapes[i] = sh / (amp or 1.0)
    note = []
    if not len(order):
        note.append("no positive factor: the step's loads put no member in compression that can buckle "
                    "(in tension they stiffen it)")
    first_neg = 1.0 / float(negative.min()) if len(negative) else None
    if first_neg is not None and (not len(order) or abs(first_neg) <= 100.0 * float(factors[0])):
        # said where it matters: a reversed load that buckles the model
        # within a hundred times the first factor (a tie that is a strut the
        # other way, a load whose sign is a choice)
        note.append(f"{len(negative)} negative factor(s): the model also buckles under the loads REVERSED, "
                    f"first at {1.0 / float(negative.min()):.6g}")
    try:
        from .modal import connector_kinks
        kinks = connector_kinks(solver)
    except Exception:
        kinks = []
    if kinks:
        note.append(f"{len(kinks)} connector law(s) sit on a kink at the base state (a gap at its stop, a "
                    "bearing sliding): their stiffness in K_b is the mean of the two sides")
    if state is None:
        nt = _no_tension_laws(solver)
        if nt:
            note.append("the reference configuration sits on the kink of the no-tension fibre law(s) "
                        + ", ".join(nt) + ": K_b takes the mean of their compression stiffness and zero, "
                        "and a column under its axial load alone comes out at about half its Euler load; "
                        "ask from the loaded state (`--result`), where the fibres are in compression")
    return BucklingResult(factors=np.asarray(factors), shapes=shapes,
                          method="reference" if state is None else "state", n_free=len(live),
                          n_negative=int(len(negative)),
                          first_negative=float(1.0 / negative.min()) if len(negative) else float("nan"),
                          load=load, axial_force=np.asarray(N), note="; ".join(note), rotational=rotational)


def write(path, result: BucklingResult, nn, stride):
    """The factors and shapes into an HDF5 file (shape: mode, node, dof)."""
    import h5py
    from . import __version__
    with h5py.File(path, "w") as hf:
        hf.attrs["buckling_format"] = BUCKLING_FORMAT
        hf.attrs["vfife_version"] = __version__
        hf.attrs["method"] = result.method
        hf.attrs["shape_layout"] = "mode,node,dof (dof: ux uy uz [rx ry rz])"
        hf.attrs["note"] = result.note
        hf.attrs["n_negative"] = result.n_negative
        hf.create_dataset("load_factor", data=result.factors)
        hf.create_dataset("reference_load", data=result.load)
        hf.create_dataset("axial_force", data=result.axial_force)
        hf.create_dataset("shape", data=result.shapes.reshape(len(result.factors), nn, stride))
        hf.create_dataset("rotation_only", data=np.asarray(result.rotational, bool))
    return path
