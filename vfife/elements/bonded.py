"""An embedded bar that can SLIP: its own degrees of freedom, plus an interface.

WHY, IN ONE MEASUREMENT
-----------------------
`EmbeddedTruss` interpolates the bar's ends from the host and is perfectly
bonded. A crack of opening w crossing a segment of length L therefore puts
strain w/L into that segment with nothing to spread it, and the drift the solid
RC column survived scaled with the SEGMENT length -- 32.0 / 18.0 / 11.0 mm for
100 / 50 / 25 mm segments, at a bar plastic strain that did not move. Holding
the segments at 100 mm while refining only the concrete put the drift back to
32.0 mm exactly.

Here the bar has its own nodes. The slip is a real unknown, the bond law
(materials/bond.py) resists it, and the transfer length comes out of tau(s) and
the bar perimeter instead of out of the mesh.

WHAT LIVES WHERE
    u_bar     (n_bar_nodes, 3)   the bar's own displacement, NOT part of the
                                 host array. Appending it to the host nodes
                                 would renumber `u`, `v`, `mass`, `fixed`,
                                 `frozen`, the regions, the constraints, the
                                 output mesh and the checkpoint -- a dozen
                                 places, each a regression. A parallel array
                                 costs one extra block in the leapfrog.
    slip      per bar node       u_bar - N(x_bar) . u_host
    bond      per bar node       tau(s_axial) * perimeter * tributary length

THE TRANSVERSE DIRECTION, AND WHY IT IS A PENALTY
    Only the AXIAL slip has a constitutive law: MC2010 describes ribs bearing
    along the bar. Transversely the bar cannot move relative to the concrete
    without crushing it, and leaving that direction free lets the bar drift
    sideways through the host on a spurious zero-energy mode.

    So the transverse relative displacement is restrained by a penalty, stiff
    enough to be effectively rigid and no stiffer -- an unnecessarily large one
    only shrinks the stable time step. The default is the bar's own axial
    stiffness over the length each node speaks for, E*A/L_trib, and
    `transverse_penalty` overrides it. It is a NUMERICAL device, not a dowel
    model: it carries no shear strength and does not fail. See the constructor
    for what an earlier, far larger default did to the Tanaka column.

SIGN CONVENTION
    Same as everywhere else here: `force()` returns the INTERNAL force, and the
    solver forms F_ext - F_int. Bond acts equal and opposite on the two sides,
    which is asserted in tests/test_bonded.py rather than left to the reader.
"""
from __future__ import annotations

import numpy as np

from . import stability
from .truss import internal_force, locate_in_blocks, precompute


class BondedTruss:
    """Embedded bar with independent DOFs and a bond-slip interface.

    `EmbeddedTruss` is not replaced: perfect bond stays available, is cheaper,
    and is right whenever the bar is not expected to slip -- a bar in a member
    that never cracks, or a first look at a model.
    """

    def __init__(self, nodes, host_conn, host_element, bar_nodes, bar_conn,
                 material, bond, area, diameter, transverse_penalty=None,
                 host_deduction=0.0, label="bonded", anchorage=None,
                 prestrain=0.0):
        self.label = label
        self.material = material
        self.bond = bond
        self.area = float(area)
        self.diameter = float(diameter)
        self.perimeter = np.pi * self.diameter

        self.bar_nodes = np.ascontiguousarray(bar_nodes, dtype=float)
        self.bar_conn = np.ascontiguousarray(bar_conn, dtype=np.int32)
        self.n_bar_nodes = int(self.bar_nodes.shape[0])
        self.n_elem = int(self.bar_conn.shape[0])

        # --- the bar as a truss in its own right -------------------------- #
        self.pre = precompute(self.bar_nodes, self.bar_conn, self.area)
        # THE OVERLAP, WHICH THIS CLASS USED TO IGNORE.
        #   A bar occupies concrete that the solid element is still integrating
        #   over, so its stiffness is counted twice; `host_deduction` subtracts
        #   the duplicate, exactly as it does for a perfectly bonded bar
        #   (elements/truss.py). `internal_force` reads it off `pre`.
        #
        #   It was accepted by the schema, never passed to this constructor and
        #   not an attribute of this class -- so every bonded model quietly ran
        #   with the concrete inside its bars counted twice. The Tanaka column
        #   asked for 2.53e10 and got nothing. Found by the field-consumption
        #   test in tests/test_reinforcement_fields.py.
        self.pre.host_deduction = float(host_deduction)
        # PRETENSIONING WITH A REAL TRANSFER LENGTH.
        #   The same offset a perfectly bonded bar gets, but here the force
        #   reaches the concrete through the interface law, so it builds up
        #   over the length the bond stress needs -- which is the transfer
        #   length, not an assumption. See PRESTRESS in elements/truss.py.
        # scalar for a stated effective prestress, one value per element
        # for a computed friction profile (truss.tendon_profile)
        self.pre.prestrain = (np.asarray(prestrain, dtype=float)
                              if np.ndim(prestrain) else float(prestrain))
        self.x0_e = self.bar_nodes[self.bar_conn]
        from ..materials.points import initialise_points
        self.state, self.material_points = initialise_points(
            material, self.n_elem, kinematics="uniaxial", backend="numpy",
            char_length=self.pre.char_len,
        )

        # --- where each BAR NODE sits in the host ------------------------- #
        blocks = (list(host_conn.items()) if isinstance(host_conn, dict)
                  else [(host_element, host_conn)])
        self.host_nodes, self.w = locate_in_blocks(nodes, blocks, self.bar_nodes)
        self.n_host_nodes = self.w.shape[1]

        # --- per bar node: axis direction and tributary length ------------ #
        # The axis is averaged from the segments meeting at the node, so a bent
        # bar (a stirrup corner) gets the bisector rather than one arm. A
        # tributary length is half of each adjacent segment, which makes the
        # tributary lengths sum to the bar's total length exactly -- if they did
        # not, the bond force integrated along the bar would depend on the
        # discretisation, which is the whole thing this class exists to remove.
        seg = self.bar_nodes[self.bar_conn[:, 1]] - self.bar_nodes[self.bar_conn[:, 0]]
        seg_len = np.linalg.norm(seg, axis=1)
        seg_dir = seg / np.maximum(seg_len, 1e-30)[:, None]

        axis = np.zeros((self.n_bar_nodes, 3))
        trib = np.zeros(self.n_bar_nodes)
        for e in range(self.n_elem):
            for k in (0, 1):
                node = self.bar_conn[e, k]
                axis[node] += seg_dir[e]
                trib[node] += 0.5 * seg_len[e]
        norm = np.linalg.norm(axis, axis=1)
        if np.any(norm < 1e-12):
            raise ValueError(
                "a bar node has no direction: two segments meeting head-on. "
                "Check the bar connectivity for a doubled-back line")
        self.axis = axis / norm[:, None]
        self.trib = trib
        #: the area of bar surface each node speaks for
        self.bond_area = self.perimeter * self.trib

        self.bond_state = bond.init_state(self.n_bar_nodes)

        # --- transverse penalty ------------------------------------------- #
        # SCALED TO THE BAR, NOT TO THE BOND, AND THE DIFFERENCE IS NOT SMALL.
        #   The first version used 100x the bond's own stiffness. On the Tanaka
        #   column that came to 5e11 N/m per node, a fictitious mass of 1.4e12
        #   kg, and a model so over-massed that the first 1 mm load step never
        #   converged in 6000 iterations. The push reported a tip displacement
        #   of exactly zero.
        #
        #   E*A/L_trib is the bar's own axial stiffness over the length each
        #   node speaks for -- the natural scale for "as stiff as the thing it
        #   is holding". On the same column it is ~6e8 N/m, three orders down,
        #   and a kilonewton of transverse force then buys under two microns of
        #   drift: rigid for any purpose, without paying for it in mass.
        #
        #   A penalty is a numerical device and its only job is to be stiff
        #   enough. Making it stiffer than that is not conservative, it is a
        #   smaller time step and a slower solve.
        k_bar = float(material.E) * self.area / np.maximum(self.trib, 1e-12)
        self.k_trans = (float(np.mean(k_bar)) if transverse_penalty is None
                        else float(transverse_penalty))

        #: filled by the solver; the bar's own displacement and velocity
        self.u_bar = np.zeros((self.n_bar_nodes, 3))
        self.v_bar = np.zeros((self.n_bar_nodes, 3))

        # --- anchorage: bar ends TIED to the host ------------------------- #
        # A bar that stops inside the mesh is, under the bond law alone, a
        # free end: the last node holds by its own half-segment of bond and
        # nothing else, and pulls out at the law's tau_max. Real bars do not
        # end like that -- they run past the support, hook, or bear on a
        # plate (the Alberta beam has 216 mm plates over the bars at each
        # support). `anchorage` makes the chosen end nodes move WITH the host:
        # their dofs are not integrated, their displacement is the host's
        # interpolated to them, and the segment force arriving at them goes
        # onto the host through the same weights -- exactly what a perfectly
        # bonded (EmbeddedTruss) end does. The interface is switched off
        # there: a tied node has no slip, so the bond law would only be
        # integrating round-off.
        self.anchored = self._anchored_nodes(anchorage)

    def _anchored_nodes(self, anchorage):
        anchored = np.zeros(self.n_bar_nodes, dtype=bool)
        if anchorage in (None, "none"):
            return anchored
        if anchorage not in ("ends", "start", "end"):
            raise ValueError(f"anchorage must be ends, start or end, not {anchorage!r}")
        degree = np.bincount(self.bar_conn.ravel(), minlength=self.n_bar_nodes)
        free_end = degree == 1
        if not free_end.any():
            raise ValueError(
                f"reinforcement {self.label!r} asks for anchorage: {anchorage} but has no "
                "free end (a closed hoop has none); drop `anchorage` for it")
        starts = np.zeros(self.n_bar_nodes, dtype=bool)
        starts[self.bar_conn[:, 0]] = True         # a segment starts here
        stops = np.zeros(self.n_bar_nodes, dtype=bool)
        stops[self.bar_conn[:, 1]] = True
        if anchorage == "ends":
            anchored[:] = free_end
        elif anchorage == "start":
            anchored[:] = free_end & starts & ~stops   # the first node of a line
        else:
            anchored[:] = free_end & stops & ~starts   # the last node of a line
        return anchored

    def tie(self, u):
        """Put every anchored bar node where its host is, at rest."""
        if self.anchored.any():
            self.u_bar[self.anchored] = self.host_displacement(u)[self.anchored]
            self.v_bar[self.anchored] = 0.0

    # ------------------------------------------------------------------ #

    def host_displacement(self, u):
        """The host's displacement interpolated to each bar node."""
        return np.einsum("na,nai->ni", self.w, u[self.host_nodes])

    def slip(self, u, u_bar=None):
        """Relative displacement of bar over host, and its axial component."""
        ub = self.u_bar if u_bar is None else u_bar
        rel = ub - self.host_displacement(u)
        s_axial = np.einsum("ni,ni->n", rel, self.axis)
        return rel, s_axial

    def interface_force(self, u, u_bar=None, dt=1.0, advance=True):
        """Bond + transverse penalty at every bar node.

        Returns the force ON THE BAR. The host takes the negative of it,
        distributed by the same interpolation weights -- so the pair is in
        equilibrium by construction, not by a separate calculation that could
        drift from this one.
        """
        rel, s_axial = self.slip(u, u_bar)
        state = self.bond_state if advance else self.bond_state.copy()
        strain = np.zeros((self.n_bar_nodes, 6))
        strain[:, 0] = s_axial
        tau = self.bond.update(strain, state, dt)[:, 0]

        # SIGN. The internal-force convention, the same one elements/truss.py
        # states: the solver forms F_ext - F_int, so an internal force is the
        # RESISTING one. A positive slip means the bar has moved ahead of its
        # host, the bond pulls it back, and that restoring pull is +tau along
        # the axis -- so the residual -F_int drags the bar back.
        #
        # The first version had both terms negated. Every element-level test
        # passed, because the tests asserted the same wrong sign, and the
        # solver's -F_int/m then turned the restoring force into a driving one:
        # positive feedback, and every bonded run diverged. The judge that
        # caught it is the one this stage exists for -- a rigid bond has to
        # reproduce perfect bond, and instead it reached 1e150.
        #
        # `f_axial` from internal_force() already follows this convention, and
        # the two are summed into one array, so they had to agree.
        axial = (tau * self.bond_area)[:, None] * self.axis
        transverse = self.k_trans * (rel - s_axial[:, None] * self.axis)
        return axial + transverse

    def stress(self, u=None):
        """Axial stress WITHOUT advancing the plastic state.

        The twin of `EmbeddedTruss.stress`, and it exists for the same reason:
        `force()` updates the state in place, so asking a monitor or an output
        frame for the stress must not go through it or the answer would depend
        on how often you looked. A bonded bar carries its own displacement
        (`u_bar`), so unlike the embedded one it needs no host argument; `u`
        is accepted and ignored to keep the two interchangeable.
        """
        scratch = self.state.copy()
        _f, sig = internal_force(self.pre, self.x0_e,
                                 self.u_bar[self.bar_conn],
                                 self.material, scratch, 1.0, want_stress=True)
        return sig[:, 0, 0]

    def force(self, u, u_bar=None, dt=1.0, want_stress=False):
        """(force on the bar nodes, force scattered onto the host nodes).

        The host term is `-f_interface` spread with the interpolation weights,
        which is the transpose of how the displacement was gathered -- the
        pairing that makes the coupling conservative.
        """
        ub = self.u_bar if u_bar is None else u_bar
        f_axial, sig = internal_force(self.pre, self.x0_e, ub[self.bar_conn],
                                      self.material, self.state, dt,
                                      want_stress=want_stress)
        f_bar = np.zeros((self.n_bar_nodes, 3))
        np.add.at(f_bar, self.bar_conn[:, 0], f_axial[:, 0])
        np.add.at(f_bar, self.bar_conn[:, 1], f_axial[:, 1])

        f_int = self.interface_force(u, ub, dt)
        if self.anchored.any():
            # a tied node hands its segment force to the host and takes no
            # interface force; its own residual is zero by construction
            a = self.anchored
            f_int[a] = 0.0
            f_host = -self.w[:, :, None] * f_int[:, None, :]
            f_host[a] += self.w[a, :, None] * f_bar[a][:, None, :]
            f_bar = f_bar + f_int
            f_bar[a] = 0.0
            return f_bar, f_host, sig
        f_bar += f_int
        f_host = -self.w[:, :, None] * f_int[:, None, :]
        return f_bar, f_host, sig

    def scatter_host(self, f_host, n_nodes):
        from ..core.mesh import scatter_add

        return scatter_add(f_host, self.host_nodes, n_nodes)

    # ------------------------------------------------------------------ #

    def bar_mass(self):
        """Physical mass of the bar, lumped on its own nodes."""
        m = np.zeros((self.n_bar_nodes, 3))
        half = self.material.rho * self.pre.volume / 2.0
        np.add.at(m, self.bar_conn[:, 0], half[:, None])
        np.add.at(m, self.bar_conn[:, 1], half[:, None])
        return m

    def _segment_axis_row_sums(self):
        """The bar's own axial stiffness, as a row sum on its own dofs.

        A segment is a two-point spring along its own axis, so each end gains
        `k |n_i| |n|_1 * (1 + 1)` -- the coupling factor for a relative
        coordinate with a single unit weight at the other end.
        """
        seg = self.bar_nodes[self.bar_conn[:, 1]] - self.bar_nodes[self.bar_conn[:, 0]]
        k_ax = ((self.material.E - self.pre.host_deduction)
                * self.area / self.pre.L0)
        per_seg = stability.directional_row_sums(seg, k_ax) * 2.0
        rows = np.zeros((self.n_bar_nodes, 3))
        np.add.at(rows, self.bar_conn[:, 0], per_seg)
        np.add.at(rows, self.bar_conn[:, 1], per_seg)
        return rows

    def _interface_row_sums(self):
        """The interface, as a row sum on ONE bar node's dofs, before coupling.

        Bond acts along the bar axis and the penalty across it, which is
        exactly the shape `directional_row_sums` is written for. `bond.E` is
        the largest tangent the law can present (materials/bond.py caps it at
        the origin), so this is a bound on the whole slip history, not a
        snapshot.
        """
        return stability.directional_row_sums(
            self.axis, float(self.bond.E) * self.bond_area, self.k_trans)

    def bar_row_sums(self):
        """Gershgorin row-sum bound for the bar's own dofs.

        WHAT THIS USED TO GET WRONG, TWICE OVER
            It added the DIAGONAL terms only -- `k_ax` twice, plus `bond.E *
            area`, plus `k_trans` -- and omitted the equal off-diagonal ones
            that the same stiffness assembles into. That is a factor of two in
            the row sum and sqrt(2) in the declared critical time step.

            It also spread the bond isotropically over the three dofs, when
            bond acts along the bar AXIS: the direction factor |n_i| |n|_1
            reaches 1.207, so the axial dof was short by that again.

            Both are now `elements/stability.py`, derived once, where the
            perfectly bonded path's exact closed form also lives.
        """
        return self._segment_axis_row_sums() + (
            self._interface_row_sums()
            * stability.coupling_factor(self.w)[:, None])

    def host_row_sums(self, n_nodes):
        """What the interface adds to the HOST nodes' bound.

        The interface is written on the RELATIVE coordinate
        `u_bar - sum_a w_a u_a`, so host dof a gains `k |w_a| (1 + sum_b
        |w_b|)` -- the same coupling factor the bar's own row carries, times
        this node's share. The previous version had `k |w_a|`, missing the
        factor entirely, and dropped the direction structure with it.
        """
        from ..core.mesh import scatter_add

        per_bar_node = (self._interface_row_sums()
                        * stability.coupling_factor(self.w)[:, None])
        if self.anchored.any():
            # a tied node has no interface; what reaches the host through
            # it is the SEGMENT stiffness, on the same relative coordinate
            per_bar_node = per_bar_node.copy()
            per_bar_node[self.anchored] = (
                self._segment_axis_row_sums()[self.anchored] / 2.0
                * stability.coupling_factor(self.w)[self.anchored, None])
        share = np.abs(self.w)[:, :, None] * per_bar_node[:, None, :]
        return scatter_add(share, self.host_nodes, n_nodes)
