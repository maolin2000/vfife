"""Two-node truss, and the embedding that makes it reinforcement.

A truss carries axial force only, and it is naturally co-rotational: the current
axis IS the rotation, so there is no polar decomposition to do.  Engineering
strain against the reference length, one integration point, one uniaxial
material.

EMBEDDED REINFORCEMENT
----------------------
A bar node is not a degree of freedom.  It is a point inside a host continuum
element, and its displacement is whatever the host interpolates there:

    u_bar = sum_a N_a(xi) u_host_a

so the bar's internal force distributes back the same way,

    f_host_a += N_a(xi) f_bar

That is the whole embedded formulation.  No constraint equations, no Lagrange
multipliers, no extra unknowns, and -- the practical point -- the concrete mesh
never has to line up with the bars.  Move the reinforcement, keep the mesh.

Mechanically the bar is then just an element whose connectivity is the union of
its two hosts' nodes and whose force carries shape-function weights, so it goes
through the same gather assembly as everything else.  Perfect bond is implied:
the bar and the surrounding concrete share displacement at that point.  A
bond-slip law would need the bar node back as an independent DOF plus an
interface element, which is a different and much larger piece of work.

THE OVERLAP
-----------
An embedded bar occupies space that the host element also fills, so E_c A_s is
counted twice and the section comes out stiffer than the transformed-section
value by E_c A_s.  That is inherent to embedded reinforcement -- Abaqus does the
same -- and it is small at ordinary reinforcement ratios: at rho = 1% it is
0.2% of EA, at the rho = 3.1% of the verification case it is 2.6%, measured.

`host_deduction` subtracts the duplicate, and then the answer IS the transformed
section (verified to 4e-4).  It defaults off because the deduction is only
strictly right while the surrounding concrete is elastic: once the host has
cracked it is no longer carrying E_c eps there, and continuing to subtract it
would remove stiffness that was never added.  Turn it on for service-load and
elastic work at high rho; leave it off when the concrete is expected to damage.

PRESTRESS
---------
A pretensioned bar is a bar whose material is already strained when the model's
displacement is zero.  `pre.prestrain` is that offset: the constitutive law sees

    eps_total = (L - L0)/L0 + prestrain

so at u = 0 the bar carries E_p eps_p A and the structure is out of balance by
exactly the jacking force.  Relaxing it IS transfer -- the concrete shortens,
the bar shortens with it, eps_total falls, and the elastic-shortening loss comes
out of the equilibrium rather than out of a formula.  There is nothing else to
the model: no separate "initial stress" state, no bar force held fixed.

Two consequences worth stating, because they are what makes this the right
primitive rather than a stress offset bolted onto the residual:

  * yield is judged on the TOTAL strain, so a strand jacked to 0.7 f_py is
    elastic and hardens from there, which is what a strand does. Offsetting the
    stress instead would have let the bar reach f_py plus the prestress.
  * host_deduction is applied to the MECHANICAL strain only. The concrete the
    bar displaces is strained by the structure's deformation, not by the
    tensioning the bar had before it was cast in, and deducting E_c eps_p A
    would have removed concrete force that never existed.

Perfectly bonded bars transfer the force at the bar node, so the whole force
appears within one host element of each end. A bonded bar (elements/bonded.py)
has its own dofs and transfers through the interface law, which gives the bar's
force a real transfer length -- that is the pretensioning model, and it costs
nothing extra here because both formulations call `internal_force`.
"""

from __future__ import annotations

import numpy as np

from . import stability

from . import library
from ..materials.points import initialise_points


class TrussPrecomputed:
    __slots__ = ("n_elem", "n_gp", "n_nodes", "L0", "area", "volume",
                 "char_len", "formulation", "name", "host_deduction",
                 "prestrain", "tension_only")

    def __init__(self):
        self.n_gp = 1
        self.n_nodes = 2
        self.formulation = "truss"
        self.name = "truss2"
        self.host_deduction = 0.0
        #: strain the bar already carries at u = 0 (see PRESTRESS above).
        #: A scalar when the analyst states the EFFECTIVE prestress; one value
        #: PER ELEMENT when a `friction` block asks for the profile along a
        #: post-tensioned tendon (`tendon_profile` below).
        self.prestrain = 0.0
        #: cables: no compression is carried
        self.tension_only = False


def precompute(nodes: np.ndarray, conn: np.ndarray, area, dtype=np.float64):
    """Reference length and volume of every bar.  `area` is scalar or per-bar."""
    if conn.shape[1] != 2:
        raise ValueError("a truss element has exactly 2 nodes")
    x = nodes[conn]
    L0 = np.linalg.norm(x[:, 1] - x[:, 0], axis=1)
    if np.any(L0 <= 0):
        raise ValueError("a truss element has zero length")
    A = np.broadcast_to(np.asarray(area, dtype=float), (conn.shape[0],))

    pre = TrussPrecomputed()
    pre.n_elem = conn.shape[0]
    pre.L0 = np.ascontiguousarray(L0, dtype=dtype)
    pre.area = np.ascontiguousarray(A, dtype=dtype)
    pre.volume = np.ascontiguousarray(A * L0, dtype=dtype)
    # for a bar the stable time step is set by its own length
    pre.char_len = pre.L0
    return pre


def internal_force(pre, x0_e, u_e, material, state, dt, want_stress=False):
    """Axial force of every bar. Returns (f_e (nel, 2, 3), stress or None).

    Sign convention matches the continuum elements: from the virtual work
    dW = N n . (du1 - du0), so the element contributes -N n at node 0 and +N n
    at node 1, and the residual is F_ext - F_int as everywhere else.
    """
    x = x0_e + u_e
    d = x[:, 1] - x[:, 0]
    L = np.linalg.norm(d, axis=1)
    n = d / np.maximum(L, 1e-30)[:, None]

    eps_mech = (L - pre.L0) / pre.L0
    eps = np.zeros((pre.n_elem, 6))
    # the material sees the TOTAL strain, prestrain included, so yield is
    # judged where the strand actually is on its curve (see PRESTRESS)
    eps[:, 0] = eps_mech + pre.prestrain
    # A standalone truss is uniaxial: using the 3-D elastic matrix here would
    # incorrectly impose plane strain and return (lambda + 2G) eps instead of
    # E eps.  Rebar keeps its existing uniaxial constitutive update.
    if material.name == "linear_elastic":
        sig = np.zeros_like(eps)
        sig[:, 0] = material.E * eps[:, 0]
    else:
        sig = material.update(eps, state, dt)
    # remove the host material the bar duplicates, if asked (see THE OVERLAP).
    # The MECHANICAL strain: the concrete the bar displaces was never jacked.
    N = (sig[:, 0] - pre.host_deduction * eps_mech) * pre.area
    if getattr(pre, "tension_only", False):
        N = np.maximum(N, 0.0)

    f_e = np.empty((pre.n_elem, 2, 3))
    f_e[:, 1] = N[:, None] * n
    f_e[:, 0] = -f_e[:, 1]
    return f_e, (sig.reshape(pre.n_elem, 1, 6) if want_stress else None)


# --------------------------------------------------------------------------- #
#  Point location: which host element contains a bar node, and where
# --------------------------------------------------------------------------- #


def _inverse_map(elem, X, target, max_iter=20, tol=1e-12):
    """Natural coordinates of `target` inside one element. Newton on x(xi)."""
    xi = elem.centroid.astype(float).copy()
    for _ in range(max_iter):
        d = elem.dN_dxi(xi[None, :])[0]                 # (3, nn)
        N = _shape(elem, xi)
        r = N @ X - target
        if np.abs(r).max() < tol:
            break
        J = d @ X                                        # (3, 3): dx_j / dxi_i
        xi -= np.linalg.solve(J.T, r)
    return xi


def _shape(elem, xi):
    """Shape functions by integrating dN/dxi is fragile; build them directly."""
    x, y, z = xi
    name = elem.name
    if name == "hexa8":
        s = np.array([-1.0, 1, 1, -1, -1, 1, 1, -1])
        t = np.array([-1.0, -1, 1, 1, -1, -1, 1, 1])
        u = np.array([-1.0, -1, -1, -1, 1, 1, 1, 1])
        return 0.125 * (1 + x * s) * (1 + y * t) * (1 + z * u)
    if name == "tet4":
        return np.array([1 - x - y - z, x, y, z])
    if name == "tet10":
        L = np.array([1 - x - y - z, x, y, z])
        out = np.empty(10)
        out[:4] = L * (2 * L - 1)
        for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))):
            out[4 + k] = 4 * L[i] * L[j]
        return out
    if name == "wedge6":
        L = np.array([1 - x - y, x, y])
        return np.concatenate([L * 0.5 * (1 - z), L * 0.5 * (1 + z)])
    raise ValueError(f"no shape functions for {name}")


def _inside(elem, xi, tol=1e-8):
    x, y, z = xi
    if elem.name == "hexa8":
        return max(abs(x), abs(y), abs(z)) <= 1.0 + tol
    if elem.name.startswith("tet"):
        return (min(x, y, z) >= -tol) and (x + y + z <= 1.0 + tol)
    if elem.name == "wedge6":
        return (min(x, y) >= -tol and x + y <= 1.0 + tol and abs(z) <= 1.0 + tol)
    return False


def locate(nodes, conn, element, points, k=12):
    """Find the host element and shape functions of every point.

    Returns (host (n_pts,), weights (n_pts, n_nodes_host)).

    Candidates come from a KD-tree on element centroids; each is inverse-mapped
    and tested against its own reference domain.  A point outside the mesh is an
    error with its coordinates, not a silent nearest-element fallback -- a bar
    that has quietly slipped outside the concrete is exactly the sort of thing
    that should stop a run.
    """
    from scipy.spatial import cKDTree

    elem = library.get(element)
    centroids = nodes[conn].mean(axis=1)
    tree = cKDTree(centroids)
    k = min(k, len(centroids))
    _, cand = tree.query(np.atleast_2d(points), k=k)
    cand = np.atleast_2d(cand)

    host = np.full(len(points), -1, dtype=np.int64)
    w = np.zeros((len(points), elem.n_nodes))
    for p, target in enumerate(np.atleast_2d(points)):
        best = None
        for c in cand[p]:
            X = nodes[conn[c]]
            xi = _inverse_map(elem, X, target)
            if _inside(elem, xi):
                best = (c, _shape(elem, xi))
                break
        if best is None:
            raise ValueError(
                f"reinforcement point {p} at {tuple(np.round(target, 6))} is not "
                f"inside any of the {k} nearest {element} elements -- it lies "
                "outside the mesh, or the mesh does not cover the bar"
            )
        host[p] = best[0]
        w[p] = best[1]
    return host, w



def locate_in_blocks(nodes, blocks, points, k=12):
    """Find each point's host across SEVERAL element blocks.

    `blocks` is a sequence of (element name, connectivity). Returns
    (host_nodes (n_pts, nnh_max), weights (n_pts, nnh_max)) -- the node ids the
    point interpolates from, and the shape-function value at each.

    Why node ids rather than an element index: in a mixed mesh a bar can run
    from a hexahedron into a tetrahedron, so "element 47" is meaningless without
    also saying which block, and the two blocks do not even agree on how many
    nodes an element has. Returning the nodes themselves makes the caller
    indifferent to which block a point landed in.

    Rows are padded to the widest block by repeating the first host node with a
    weight of zero. A zero weight contributes no displacement, no force and no
    mass, and a repeated id only ever adds zero, so every downstream einsum and
    scatter keeps working unchanged on one uniform array.

    One KD-tree spans all blocks, so candidates come back in true distance
    order rather than best-per-block.
    """
    from scipy.spatial import cKDTree

    defs = [(name, library.get(name), np.asarray(conn)) for name, conn in blocks]
    nnh_max = max(d.n_nodes for _n, d, _c in defs)

    centroids, owner = [], []
    for b, (_name, _d, conn) in enumerate(defs):
        centroids.append(nodes[conn].mean(axis=1))
        owner.append(np.stack([np.full(len(conn), b), np.arange(len(conn))], axis=1))
    centroids = np.vstack(centroids)
    owner = np.vstack(owner)

    tree = cKDTree(centroids)
    k = min(k, len(centroids))
    _, cand = tree.query(np.atleast_2d(points), k=k)
    # k = 1 (a one-element host) comes back as a vector, not a column
    cand = np.asarray(cand).reshape(len(np.atleast_2d(points)), -1)

    host_nodes = np.zeros((len(points), nnh_max), dtype=np.int64)
    w = np.zeros((len(points), nnh_max))
    for p, target in enumerate(np.atleast_2d(points)):
        placed = False
        for c in cand[p]:
            b, e = owner[c]
            _name, elem, conn = defs[b]
            ids = conn[e]
            xi = _inverse_map(elem, nodes[ids], target)
            if _inside(elem, xi):
                n = elem.n_nodes
                host_nodes[p, :n] = ids
                host_nodes[p, n:] = ids[0]        # padding, weight stays 0
                w[p, :n] = _shape(elem, xi)
                placed = True
                break
        if not placed:
            raise ValueError(
                f"reinforcement point {p} at {tuple(np.round(target, 6))} is not "
                f"inside any of the {k} nearest elements of "
                + " / ".join(n for n, _d, _c in defs)
                + " -- it lies outside the mesh, or the mesh does not cover the bar"
            )
    return host_nodes, w


class EmbeddedTruss:
    """A bar set whose displacement is interpolated from a host mesh.

    Presents the same (force, assemble) contract as a continuum block: its
    effective connectivity is the union of both hosts' nodes, and its force
    carries the shape-function weights, so the ordinary gather assembly applies.
    """

    def __init__(self, nodes, host_conn, host_element, bar_nodes, bar_conn,
                 material, area, host_deduction=0.0, prestrain=0.0):
        """`host_conn` is one connectivity array with `host_element` naming its
        type, or a {name: connectivity} mapping for a mixed mesh -- in which
        case `host_element` is ignored, because the bar decides per node which
        block it is in."""
        self.material = material
        if not getattr(material, "uniaxial", False):
            raise ValueError(
                f"reinforcement needs a one-dimensional material; {material.name!r} "
                "is a three-dimensional law. Use `rebar`."
            )
        self.pre = precompute(bar_nodes, bar_conn, area)
        self.pre.host_deduction = float(host_deduction)
        # scalar for a stated effective prestress, one value per element
        # for a computed friction profile (truss.tendon_profile)
        self.pre.prestrain = (np.asarray(prestrain, dtype=float)
                              if np.ndim(prestrain) else float(prestrain))
        self.bar_conn = np.ascontiguousarray(bar_conn, dtype=np.int32)
        self.n_elem = self.pre.n_elem

        blocks = (list(host_conn.items()) if isinstance(host_conn, dict)
                  else [(host_element, host_conn)])
        host_nodes, w = locate_in_blocks(nodes, blocks, bar_nodes)
        self.host_nodes, self.w = host_nodes, w
        nnh = w.shape[1]
        self.n_host_nodes = nnh

        # effective connectivity: node ids each bar's two ends interpolate from
        self.conn = np.concatenate(
            [host_nodes[bar_conn[:, 0]], host_nodes[bar_conn[:, 1]]],
            axis=1,
        ).astype(np.int32)
        self.weights = np.stack([w[bar_conn[:, 0]], w[bar_conn[:, 1]]], axis=1)

        # reference positions of the bar ends, as the HOST interpolates them:
        # if a bar node is not exactly on the interpolated point the bar would
        # start pre-strained
        self.x0_e = np.stack(
            [np.einsum("ea,eai->ei", w[bar_conn[:, 0]], nodes[self.conn[:, :nnh]]),
             np.einsum("ea,eai->ei", w[bar_conn[:, 1]], nodes[self.conn[:, nnh:]])],
            axis=1,
        )
        self.state, self.material_points = initialise_points(
            material, self.n_elem, kinematics="uniaxial", backend="numpy",
            char_length=self.pre.char_len,
        )

    def gather(self, u):
        """Bar-end displacement, interpolated from the host nodes."""
        nnh = self.n_host_nodes
        return np.stack(
            [np.einsum("ea,eai->ei", self.weights[:, 0], u[self.conn[:, :nnh]]),
             np.einsum("ea,eai->ei", self.weights[:, 1], u[self.conn[:, nnh:]])],
            axis=1,
        )

    def force(self, u, dt, want_stress=False):
        u_e = self.gather(u)
        f_e, sig = internal_force(self.pre, self.x0_e, u_e, self.material,
                                  self.state, dt, want_stress=want_stress)
        # distribute each end's force onto its host nodes with the same weights
        nnh = self.n_host_nodes
        out = np.empty((self.n_elem, 2 * nnh, 3))
        out[:, :nnh] = self.weights[:, 0][:, :, None] * f_e[:, 0][:, None, :]
        out[:, nnh:] = self.weights[:, 1][:, :, None] * f_e[:, 1][:, None, :]
        return out, sig

    def stress(self, u):
        """Axial stress WITHOUT advancing the plastic state.

        `force()` updates the state in place, which is right inside the time
        loop and wrong everywhere else.  Writing an output frame must not move
        the material: an extra constitutive update per frame would make the
        answer depend on how often you asked to look at it.
        """
        scratch = self.state.copy()
        _f, sig = internal_force(self.pre, self.x0_e, self.gather(u),
                                 self.material, scratch, 1.0, want_stress=True)
        return sig[:, 0, 0]

    def lumped_mass(self, n_nodes):
        """Bar mass, distributed to the host nodes the same way the force is."""
        from ..core.mesh import scatter_add

        m = self.material.rho * self.pre.volume / 2.0
        nnh = self.n_host_nodes
        me = np.empty((self.n_elem, 2 * nnh, 3))
        me[:, :nnh] = (self.weights[:, 0] * m[:, None])[:, :, None]
        me[:, nnh:] = (self.weights[:, 1] * m[:, None])[:, :, None]
        return scatter_add(np.abs(me), self.conn, n_nodes)


def bar_chains(conn) -> list:
    """`[(first, last+1)]` element ranges, one per connected bar.

    `bars_from_spec` lays each line entry's elements consecutively and gives
    every entry its own nodes, so a bar is a maximal run of elements where one
    ends on the node the next starts from. A closed route is one run too: its
    last element returns to the first node without breaking the run.
    """
    conn = np.asarray(conn)
    breaks = [0]
    for i in range(1, len(conn)):
        if conn[i - 1, 1] != conn[i, 0]:
            breaks.append(i)
    breaks.append(len(conn))
    return [(breaks[i], breaks[i + 1]) for i in range(len(breaks) - 1)]


def is_closed(conn, lo, hi) -> bool:
    """Does this bar return to where it started?  (a hoop; no jacking end)"""
    return bool(np.asarray(conn)[hi - 1, 1] == np.asarray(conn)[lo, 0])


def tendon_profile(nodes, conn, sigma_con, mu, k, jack="start", anchor_set=None,
                   E=None):
    """Tendon stress in every bar element, after friction and anchor set.

        sigma(s) = sigma_con * exp(-(mu * theta(s) + k * s))

    the post-tensioning friction law of GB50010, ACI 318 and EN 1992 alike,
    with `s` the arc length from the jacking end and `theta` the accumulated
    ANGLE CHANGE along the way -- which is why this lives next to the bar
    geometry: a tendon has a theta only because it is one connected bar whose
    segments are laid in order (`bar_chains`).

    The returned value is the stress at each element's MIDPOINT. Within one
    element the direction is constant, so the mu term is exact there; only the
    wobble term k*s varies along it.

    `jack`: which end the jack is at, or `both` (each point takes the smaller
    of the two losses, which is what jacking from both ends buys).

    ANCHOR SET, `anchor_set` metres of draw-in as the wedges seat. The tendon
    slips back at the live end until friction -- acting the other way now --
    has absorbed the movement, so over a draw-in length L_set the profile is
    REFLECTED about sigma(L_set):

        sigma_after(s) = 2 sigma(L_set) - sigma(s),    s < L_set

    and L_set is where the area between the two curves equals the elongation
    lost, E_p * delta.  That is the standard construction and it is solved
    here by walking the elements rather than by assuming a straight profile,
    so it is right for a draped tendon as well as a flat one.  It needs `E`.

    Nothing here is a loss the analyst has to look up: given mu, k and the
    draw-in, this IS the friction calculation, and what remains outside the
    program is relaxation and shrinkage/creep.
    """
    nodes = np.asarray(nodes, dtype=float)
    conn = np.asarray(conn)
    sigma = np.empty(len(conn), dtype=float)
    for lo, hi in bar_chains(conn):
        c = conn[lo:hi]
        d = nodes[c[:, 1]] - nodes[c[:, 0]]
        L = np.linalg.norm(d, axis=1)
        t = d / L[:, None]
        s = np.cumsum(L) - 0.5 * L                 # arc length to each midpoint
        if len(t) > 1:
            turn = np.arccos(np.clip(np.einsum("ij,ij->i", t[:-1], t[1:]), -1.0, 1.0))
            theta = np.concatenate([[0.0], np.cumsum(turn)])
        else:
            theta = np.zeros(1)
        f_start = mu * theta + k * s
        f_end = mu * (theta[-1] - theta) + k * (float(L.sum()) - s)
        if jack == "start":
            f = f_start
        elif jack == "end":
            f = f_end
        elif jack == "both":
            f = np.minimum(f_start, f_end)
        else:
            raise ValueError(f"jack must be start, end or both, not {jack!r}")
        sig = sigma_con * np.exp(-f)
        if anchor_set:
            if E is None:
                raise ValueError("anchor set needs the tendon modulus")
            ends = ([slice(None)] if jack == "start" else
                    [slice(None, None, -1)] if jack == "end" else
                    [slice(None), slice(None, None, -1)])
            for sl in ends:
                sig[sl] = _seat_anchor(sig[sl], L[sl], float(anchor_set), float(E))
        sigma[lo:hi] = sig
    return sigma


def _seat_anchor(sig, L, draw_in, E):
    """Reflect the profile over the draw-in length. `sig`/`L` start at the jack.

    The seated length is where 2 * integral(sigma(s) - sigma(L_set)) ds equals
    E * draw_in -- the shortening the wedges take up. Walking the elements and
    stopping at the first one where the integral is reached keeps this exact
    for any profile, and a draw-in bigger than the whole tendon can absorb is
    an error rather than a silently flat tendon.
    """
    target = E * draw_in
    for j in range(1, len(sig) + 1):
        area = 2.0 * float(np.sum((sig[:j] - sig[j - 1]) * L[:j]))
        if area >= target:
            out = sig.copy()
            if j == 1 or area == target:
                level = sig[j - 1]
            else:
                # THE SEATED LENGTH ENDS INSIDE ELEMENT j-1, not at its
                # midpoint: between the midpoints j-2 and j-1 the area is
                # linear in the reflection level, 2 (S - level * Lsum) over
                # the elements still above it, and the level that gives
                # exactly E * draw_in is solved for. Stopping at the first
                # midpoint past the target always over-shot -- a 6 mm set
                # drew in 6.66 mm on a 20-element 20 m tendon (found in
                # review)
                S = float(np.sum(sig[:j - 1] * L[:j - 1]))
                Lsum = float(np.sum(L[:j - 1]))
                level = (S - 0.5 * target) / Lsum
            above = sig[:j] > level
            out[:j] = np.where(above, np.maximum(2.0 * level - sig[:j], 0.0), sig[:j])
            return out
    # WHAT THE WALK COULD ABSORB AT MOST: the largest reflected area over
    # every seated length. The whole-profile area against the far end went
    # negative when the far end is the higher one -- `jack: both`, whose
    # profile rises again past the middle -- and the message read "recovers
    # only -11.73 mm" (found in review)
    cap = max(2.0 * float(np.sum((sig[:j] - sig[j - 1]) * L[:j])) for j in range(1, len(sig) + 1))
    raise ValueError(
        f"anchor set of {draw_in * 1e3:.1f} mm draws in more than this tendon can "
        f"absorb: reflecting the profile from this end recovers at most "
        f"{cap / E * 1e3:.2f} mm (with `jack: both`, up to the tendon's middle). The tendon is "
        "too short, or too straight, for that much seating -- state the draw-in "
        "the anchorage actually has")


def bars_from_spec(spec) -> tuple:
    """Bar geometry from a case file. Returns (nodes (n,3), conn (m,2)).

    Two forms, because the two ways people describe reinforcement are different:

        lines:  - {start: [x,y,z], end: [x,y,z], divisions: 40}
                - {start: ..., end: ..., segment_length: 0.209}
                - {start: ..., end: ..., divisions: 6,
                   repeat: {count: 58, offset: [0.2, 0, 0]}}

    lays a straight bar and cuts it into segments -- and `repeat` steps the whole
    line along a vector, which is how transverse reinforcement is described:
    one stirrup leg plus a spacing, rather than sixty near-identical entries.
    That is not a convenience only.  Stirrups are what stop a crushed region
    from becoming a mechanism, and a format that makes them tedious to write is
    a format that quietly encourages leaving them out.

    Subdividing -- which is how longitudinal
    reinforcement is actually specified, and the subdivision matters because
    each segment carries one constitutive point, so a single element per bar
    would smear yielding over the whole span.

    WHY `segment_length` EXISTS, AND WHEN TO PREFER IT
      These bars are perfectly bonded: a crack of opening w crossing a segment
      of length L puts strain w/L into that segment, with no slip to spread it.
      The segment length is therefore a PHYSICAL length -- the length over
      which bar strain is smeared -- and not a discretisation choice.

      Measured on the solid RC column: refining the mesh from 100 to 50 mm
      while letting `divisions` follow it halved the drift the model survived,
      from 32.0 to 18.0 mm, at a bar plastic strain that stayed put (4.83% and
      4.67%). Holding the segments at 100 mm while refining the concrete to
      50 mm put the drift back to 32.0 mm exactly. The mesh was never the
      variable; the bar segment was.

      `divisions` invites exactly that mistake, because it is a count and a
      count is what people scale with the mesh. `segment_length` is a length in
      metres and does not move when the mesh does. For a bar in a cracking
      member, choose it from the bond transfer length -- Priestley's strain
      penetration Lsp = 0.022 fy db is the same quantity the fibre path uses in
      cases/case_11 -- not from the element size.

    Or, for a cage generated elsewhere:

        nodes: [[x,y,z], ...]
        elements: [[i,j], ...]
    """
    if spec.get("lines"):
        pts, conn, base = [], [], 0
        for ln in spec["lines"]:
            if ln.get("segment_length") is not None and ln.get("divisions") is not None:
                raise ValueError(
                    "a reinforcement line takes `divisions` or "
                    "`segment_length`, not both -- they are different "
                    "answers to the same question and one of them would "
                    "silently win")
            route = ln.get("points")
            if route is not None:
                corners = np.asarray(route, dtype=float)
                if ln.get("closed"):
                    corners = np.vstack([corners, corners[:1]])
            else:
                corners = np.stack([np.asarray(ln["start"], dtype=float),
                                    np.asarray(ln["end"], dtype=float)])

            # THE ROUTE IS LAID AS ONE BAR: the legs share nodes, so the
            # segments are connected. See BarLineSpec in io/schema.py for what
            # goes wrong when they are not.
            legs = []
            for a, b in zip(corners[:-1], corners[1:]):
                span = float(np.linalg.norm(b - a))
                if span <= 0.0:
                    raise ValueError("a reinforcement line has zero length")
                if ln.get("segment_length") is not None:
                    seg = float(ln["segment_length"])
                    if seg <= 0.0:
                        raise ValueError("segment_length must be positive")
                    n = max(1, int(round(span / seg)))
                else:
                    # `or 1`, not get(..., 1): the schema emits `divisions:
                    # None` for an unset field, so the key IS present and
                    # get()'s default never fires -- int(None) would raise. The
                    # same shape of bug once ran relaxation cases with physical
                    # mass (see _setup_mass_and_dt).
                    n = int(ln.get("divisions") or 1)
                    if n < 1:
                        raise ValueError(
                            "a reinforcement line needs divisions >= 1")
                # every leg but the first drops its start: that node is the
                # previous leg's end, and sharing it is what joins them
                t = np.linspace(0.0, 1.0, n + 1)[1:, None] if legs else \
                    np.linspace(0.0, 1.0, n + 1)[:, None]
                legs.append(a + t * (b - a))
            route_pts = np.vstack(legs)
            if route is not None and ln.get("closed"):
                # the closing point IS the first one: drop the duplicate and
                # wire the last segment back to node 0, so a hoop is a loop
                # with no free end -- which is what makes bonded.py's "a closed
                # hoop has none" refusal true rather than decorative
                route_pts = route_pts[:-1]
                m = len(route_pts)
                rows = np.column_stack([np.arange(m), np.arange(1, m + 1) % m])
            else:
                m = len(route_pts) - 1
                rows = np.column_stack([np.arange(m), np.arange(1, m + 1)])

            rep = ln.get("repeat") or {}
            count = int(rep.get("count", 1))
            step = np.asarray(rep.get("offset", (0.0, 0.0, 0.0)), dtype=float)
            for r in range(count):
                pts.append(route_pts + step * r)
                conn.append(base + rows)
                base += len(route_pts)
        return np.vstack(pts), np.vstack(conn).astype(np.int32)
    if spec.get("nodes") is not None and spec.get("elements") is not None:
        return (np.asarray(spec["nodes"], dtype=float),
                np.asarray(spec["elements"], dtype=np.int32))
    raise ValueError(
        "a reinforcement block needs either `lines` or `nodes` + `elements`"
    )


def _emb_row_sums(emb, n_nodes):
    """Underwood row sums for the bars, distributed to their host nodes.

    Not a bound -- the exact sum.  In host degrees of freedom one bar's
    stiffness is

        K[(a,i),(b,j)] = +- k w_a w_b n_i n_j,   k = EA/L0

    so the row sum of dof (a,i) is

        sum_b sum_j |K| = k |w_a| |n_i| (sum_b |w_b|) (sum_j |n_j|)

    with the w-sum running over BOTH ends.  An earlier version used the crude
    bound 2k|w_a|, dropping the |n_i| sum_j|n_j| factor, which is up to
    2 sqrt(3) = 3.46 -- harmless while the concrete dominated the nodal mass and
    fatal once the mesh was refined enough that it did not: the reinforced beam
    at 1920 elements diverged in the bars to eps_p ~ 1e140.  Underestimating a
    row sum underestimates the mass, and an under-massed dof in an explicit
    scheme is simply unstable.
    """
    from ..core.mesh import scatter_add

    k = ((emb.material.E - emb.pre.host_deduction) * emb.pre.area
         / emb.pre.L0)
    d = emb.x0_e[:, 1] - emb.x0_e[:, 0]
    # The closed form above, in the shared pieces (elements/stability.py):
    # `directional_row_sums` is the k |n_i| sum_j|n_j| factor and
    # `relative_coordinate_factor` is the sum over BOTH ends' weights. This is
    # numerically identical to the hand-written version it replaces -- asserted
    # in tests/test_stability_row_sums.py -- and the point of the rewrite is
    # that the next element to need this cannot derive it a fourth time.
    w_sum = stability.relative_coordinate_factor(emb.weights[:, 0],
                                                 emb.weights[:, 1])
    dir_fac = stability.directional_row_sums(d, k) * w_sum[:, None]

    nnh = emb.n_host_nodes
    rs = np.empty((emb.n_elem, 2 * nnh, 3))
    rs[:, :nnh] = np.abs(emb.weights[:, 0])[:, :, None] * dir_fac[:, None, :]
    rs[:, nnh:] = np.abs(emb.weights[:, 1])[:, :, None] * dir_fac[:, None, :]
    return scatter_add(rs, emb.conn, n_nodes)
