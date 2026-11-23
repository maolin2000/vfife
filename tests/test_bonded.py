"""`BondedTruss`: the element, on its own, before any solver touches it.

Stage B1 of v0.1. The element computes forces from a host displacement
field and a bar displacement field; nothing here integrates anything, so every
property below is checkable by construction rather than by convergence.

WHAT IS DELIBERATELY *NOT* HERE
    "Make the bond rigid and recover perfect bond" is the strongest judge of
    this whole change, and it belongs to stage B2, not here. Zero slip is not an
    equilibrium state for the bar: a bar strained like its host carries axial
    force, and that force has to be balanced by bond, which needs a slip to
    exist. The rigid limit is slip -> 0 with stiffness -> infinity at finite
    force, and finding it requires solving the bar's equilibrium -- which is
    what the solver does. Asserting it here would mean writing a second solver
    inside a test.
"""
import numpy as np
import pytest

from vfife.core.mesh import box_mesh
from vfife.elements.bonded import BondedTruss
from vfife.materials import create

AREA, DIA = 3.1416e-4, 0.020


def _bar(n=11, z0=0.0, z1=1.0, x=0.2, y=0.2):
    nodes = np.stack([np.full(n, x), np.full(n, y), np.linspace(z0, z1, n)], 1)
    conn = np.column_stack([np.arange(n - 1), np.arange(1, n)]).astype(np.int32)
    return nodes, conn


def _build(**kw):
    mesh = box_mesh([0.4, 0.4, 1.0], [2, 2, 5])
    steel = create("rebar", E=2.0e11, rho=7850.0, yield_strength=4.74e8,
                   hardening=1.0e9)
    bond = create("bond_ceb", fc=25.6e6)
    bar_nodes, bar_conn = _bar(**{k: v for k, v in kw.items() if k == "n"})
    bt = BondedTruss(mesh.nodes, mesh.elements, "hexa8", bar_nodes, bar_conn,
                     steel, bond, area=AREA, diameter=DIA,
                     transverse_penalty=kw.get("transverse_penalty"))
    return mesh, bt


# --------------------------------------------------------------------------- #
#  Geometry the bond force is integrated over
# --------------------------------------------------------------------------- #

def test_the_tributary_lengths_add_up_to_the_bar():
    """If they did not, the bond force integrated along the bar would depend on
    how the bar was cut up -- which is the dependence this class exists to
    remove, so it would be a particularly bad one to introduce."""
    _mesh, bt = _build()
    assert bt.trib.sum() == pytest.approx(1.0, rel=1e-12)


def test_the_bond_area_is_the_bar_surface():
    _mesh, bt = _build()
    assert bt.bond_area.sum() == pytest.approx(np.pi * DIA * 1.0, rel=1e-12)


def test_refining_the_bar_does_not_change_the_surface_it_bonds_over():
    """The same physical bar, cut into 10 or 40 pieces, must present the same
    area to the concrete. This is the discretisation independence, at the level
    where it can be checked exactly."""
    areas = []
    for n in (11, 21, 41):
        _mesh, bt = _build(n=n)
        areas.append(bt.bond_area.sum())
    assert areas[0] == pytest.approx(areas[1], rel=1e-12)
    assert areas[0] == pytest.approx(areas[2], rel=1e-12)


def test_the_axis_at_a_bend_is_the_bisector():
    """A stirrup corner has two arms; taking one of them would put the bond
    force off-axis and leak a transverse component into a penalty that is
    supposed to be inactive."""
    mesh = box_mesh([0.4, 0.4, 1.0], [2, 2, 5])
    nodes = np.array([[0.1, 0.2, 0.5], [0.2, 0.2, 0.5], [0.2, 0.3, 0.5]])
    conn = np.array([[0, 1], [1, 2]], dtype=np.int32)
    bt = BondedTruss(mesh.nodes, mesh.elements, "hexa8", nodes, conn,
                     create("rebar", E=2.0e11, rho=7850.0,
                            yield_strength=4.74e8, hardening=1.0e9),
                     create("bond_ceb", fc=25.6e6), area=AREA, diameter=DIA)
    mid = bt.axis[1]
    assert mid[0] == pytest.approx(mid[1], rel=1e-9)     # 45 degrees
    assert mid[2] == pytest.approx(0.0, abs=1e-12)


def test_a_bar_that_doubles_back_is_refused():
    mesh = box_mesh([0.4, 0.4, 1.0], [2, 2, 5])
    nodes = np.array([[0.1, 0.2, 0.5], [0.2, 0.2, 0.5], [0.1, 0.2, 0.5]])
    conn = np.array([[0, 1], [1, 2]], dtype=np.int32)
    with pytest.raises(ValueError, match="doubled-back|no direction"):
        BondedTruss(mesh.nodes, mesh.elements, "hexa8", nodes, conn,
                    create("rebar", E=2.0e11, rho=7850.0,
                           yield_strength=4.74e8, hardening=1.0e9),
                    create("bond_ceb", fc=25.6e6), area=AREA, diameter=DIA)


# --------------------------------------------------------------------------- #
#  The coupling
# --------------------------------------------------------------------------- #

def test_the_bar_and_the_host_take_equal_and_opposite_force():
    """The coupling has to be conservative or the model gains momentum from
    nothing. It holds by construction here -- the host term is the transpose of
    the gather -- and that is exactly why it is worth asserting: a refactor that
    computed the two sides separately would break it silently."""
    mesh, bt = _build()
    u = np.zeros((mesh.n_nodes, 3))
    bt.u_bar[:, 2] = 1.0e-4
    f_bar, f_host, _ = bt.force(u, dt=1.0)
    total = f_bar.sum(axis=0) + bt.scatter_host(f_host, mesh.n_nodes).sum(axis=0)
    assert np.abs(total).max() < 1e-6


def test_moving_bar_and_host_together_produces_nothing():
    """Rigid-body translation of the whole model: no slip, no bond force, no
    axial force. A frame error in the interpolation shows up here and nowhere
    else, because every other test moves the two sides differently."""
    mesh, bt = _build()
    shift = np.array([1.0e-3, -2.0e-3, 5.0e-4])
    u = np.tile(shift, (mesh.n_nodes, 1))
    bt.u_bar[:] = shift
    _rel, s = bt.slip(u)
    assert np.abs(s).max() < 1e-15

    # Judged against the force a REAL slip produces, not against an absolute
    # tolerance. The first version of this test used < 1e-6 N and failed at
    # 0.043 N -- correctly, and for a reason worth keeping: alpha < 1 leaves the
    # MC2010 power law with an infinite tangent at the origin, so a slip of
    # 1e-16 still returned about 79 Pa. That is what put the `s_elastic`
    # cut-off into materials/bond.py, which the explicit time step needed
    # anyway.
    bt2 = _build()[1]
    bt2.u_bar[:, 2] = 0.5e-3
    reference = np.abs(bt2.force(np.zeros_like(u), dt=1.0)[0]).max()

    f_bar, f_host, _ = bt.force(u, dt=1.0)
    assert np.abs(f_bar).max() < 1e-9 * reference
    assert np.abs(f_host).max() < 1e-9 * reference


def test_the_bond_force_at_a_node_is_tau_times_the_area_it_speaks_for():
    """The one place the material and the geometry meet. Checked against the
    law directly rather than against a previous run of this code."""
    mesh, bt = _build()
    u = np.zeros((mesh.n_nodes, 3))
    slip = 0.4e-3
    bt.u_bar[:, 2] = slip

    probe = create("bond_ceb", fc=25.6e6)
    tau = float(probe.update(np.array([[slip, 0, 0, 0, 0, 0]]),
                             probe.init_state(1), 1.0)[0, 0])
    f_int = bt.interface_force(u, dt=1.0)
    node = 5                                    # an interior node
    # +tau: the internal-force convention, where an internal force is the
    # resisting one and the solver forms F_ext - F_int. This assertion had the
    # sign backwards in its first version and the element agreed with it, so
    # both were wrong together until the solver drove every bonded run to 1e150.
    expected = tau * bt.bond_area[node]
    assert f_int[node, 2] == pytest.approx(expected, rel=1e-10)


def test_the_transverse_penalty_resists_sideways_slip_and_not_axial():
    """Axial slip must reach the bond law untouched; the penalty exists only to
    stop the bar walking sideways through the concrete."""
    mesh, bt = _build(transverse_penalty=1.0e9)
    u = np.zeros((mesh.n_nodes, 3))

    bt.u_bar[:] = 0.0
    bt.u_bar[:, 0] = 1.0e-5                     # purely transverse
    f_t = bt.interface_force(u, dt=1.0, advance=False)
    assert f_t[5, 0] == pytest.approx(1.0e9 * 1.0e-5, rel=1e-9)
    assert abs(f_t[5, 2]) < 1e-9                # nothing leaked into the axis

    bt.u_bar[:] = 0.0
    bt.u_bar[:, 2] = 1.0e-5                     # purely axial
    f_a = bt.interface_force(u, dt=1.0, advance=False)
    assert abs(f_a[5, 0]) < 1e-9                # nothing leaked out of the axis
    assert f_a[5, 2] > 0.0                      # and the bond resists it


def test_the_host_moving_alone_slips_the_bar_the_other_way():
    """Sign check on the gather. Getting it backwards would make bond a driver
    rather than a resistance, and the model would gain energy every step."""
    mesh, bt = _build()
    u = np.zeros((mesh.n_nodes, 3))
    u[:, 2] = 1.0e-4                            # host up, bar still
    _rel, s = bt.slip(u)
    assert np.all(s < 0)                        # bar lags the host


def test_reading_the_interface_without_advancing_does_not_move_the_state():
    """Writing an output frame must not change the answer. `EmbeddedTruss.stress`
    exists for the same reason; this is that guarantee for the interface."""
    mesh, bt = _build()
    u = np.zeros((mesh.n_nodes, 3))
    bt.u_bar[:, 2] = 2.0e-3
    before = bt.bond_state.copy()
    bt.interface_force(u, dt=1.0, advance=False)
    assert np.array_equal(bt.bond_state, before)
    bt.interface_force(u, dt=1.0, advance=True)
    assert not np.array_equal(bt.bond_state, before)


# --------------------------------------------------------------------------- #
#  What the solver will need
# --------------------------------------------------------------------------- #

def test_the_bar_carries_its_own_mass():
    mesh, bt = _build()
    m = bt.bar_mass()
    assert m.shape == (bt.n_bar_nodes, 3)
    total = m[:, 0].sum()
    assert total == pytest.approx(7850.0 * AREA * 1.0, rel=1e-12)


def test_the_bar_row_sums_carry_axial_bond_and_penalty():
    """The Gershgorin bound for the bar's own dofs. Under-massing a dof in an
    explicit scheme is the defect `_emb_row_sums` still carries (the xfail in
    tests/test_mixed_mesh.py), and the reason it cannot recur here is that the
    bar now has its own dofs and its own bound instead of having its stiffness
    pushed onto host nodes it does not belong to."""
    mesh, bt = _build(transverse_penalty=1.0e9)
    rows = bt.bar_row_sums()
    assert rows.shape == (bt.n_bar_nodes, 3)
    assert np.all(rows > 0)
    # THE STRUCTURE IS DIRECTIONAL, and asserting that is the point.
    #   The bar runs along z. Its axial stiffness and the bond both act along
    #   z and nowhere else, so only the z row carries them; the x and y rows
    #   are the transverse penalty alone, carrying the relative-coordinate
    #   factor of 2 and nothing more.
    #
    #   The first version of this test asserted `rows[5, 0] > rows[0, 0]` --
    #   interior node beats end node -- which was true only because the old
    #   derivation spread everything isotropically over the three dofs. It
    #   was asserting the defect (elements/stability.py).
    assert np.allclose(rows[:, :2], 2.0 * 1.0e9, rtol=1e-12)
    # along the bar, an interior node has two segments and an end node one
    assert rows[5, 2] > rows[0, 2]
    assert np.all(rows[:, 2] > rows[:, 0])


def test_the_host_row_sums_only_carry_the_interface():
    """The host sees the bar only through the interface, so its bound gains the
    interface stiffness and nothing else. That is a two-point coupling and its
    row sum is elementary -- which is the whole reason this design dissolves
    the under-massing defect rather than inheriting it."""
    mesh, bt = _build(transverse_penalty=1.0e9)
    rows = bt.host_row_sums(mesh.n_nodes)
    assert rows.shape == (mesh.n_nodes, 3)
    assert np.all(rows >= 0)
    # Every bar node hands its row to host nodes with weights summing to one,
    # so the total handed over is conserved exactly -- whatever the row is.
    # Checked against the bar-side row rather than against a restatement of
    # the stiffness, because a restatement is where the old version went
    # wrong: it used `bond.E * area + k_trans`, the DIAGONAL only, isotropic,
    # and half the true relative-coordinate row.
    from vfife.elements import stability
    per_bar = (bt._interface_row_sums()
               * stability.coupling_factor(bt.w)[:, None])
    assert rows.sum(axis=0) == pytest.approx(per_bar.sum(axis=0), rel=1e-9)
    # and it is directional: the bond acts along the bar, the penalty across
    assert per_bar[:, 2].max() > per_bar[:, 0].max()
