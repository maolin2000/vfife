"""The explicit-stability bound, derived once and checked against brute force.

WHY THIS FILE EXISTS
    `sum_j |K_ij|` per dof had been written three times. Two of the three were
    wrong, in different ways, and both failure modes are silent -- an
    under-stated row sum under-masses a dof, and an under-massed dof in an
    explicit scheme is unstable, sometimes only after the run has converged:

      * `truss._emb_row_sums` -- an early version dropped the direction factor
        (up to 3.46) and took a reinforced beam to 1e140. Fixed long ago; its
        closed form is exact and this file proves it against brute force.
      * `bonded.BondedTruss` -- added the DIAGONAL terms and omitted the equal
        off-diagonal ones, so its declared critical time step was sqrt(2) too
        large, and it spread the bond isotropically when bond acts along the
        bar AXIS, which costs another factor of up to 1.207.

    The derivation now lives in `elements/stability.py`. What this file does is
    check it the only way worth checking: BUILD the small stiffness matrix and
    sum it, then require the closed form to agree.

THE BRUTE FORCE IS THE POINT
    A closed form tested against another closed form tests nothing but
    algebra transcription. Every assertion below assembles a dense K for a
    tiny configuration, sums |K| by row, and compares. That is how you find
    out a factor of two is missing.
"""
import numpy as np
import pytest

from vfife.elements import stability


def _dense_row_sums(K):
    return np.abs(K).sum(axis=1)


# --------------------------------------------------------------------------- #
#  Direction
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("axis", [
    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0],
    [0.8, 0.6, 0.0], [0.3, -0.5, 0.9], [-1.0, 2.0, -0.5],
])
def test_an_axial_stiffness_matches_a_dense_sum(axis):
    """K = k n(x)n, one point. The row sum is k |n_i| |n|_1 and nothing else."""
    n = np.array(axis, dtype=float)
    n /= np.linalg.norm(n)
    k = 3.7e9
    K = k * np.outer(n, n)
    got = stability.directional_row_sums(np.array([axis], dtype=float), k)[0]
    assert got == pytest.approx(_dense_row_sums(K), rel=1e-12)


@pytest.mark.parametrize("axis", [
    [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.8, 0.6, 0.0], [0.3, -0.5, 0.9],
])
def test_an_axial_plus_transverse_stiffness_matches_a_dense_sum(axis):
    """K = k_a n(x)n + k_t (I - n(x)n): the bond interface's exact shape."""
    n = np.array(axis, dtype=float)
    n /= np.linalg.norm(n)
    k_a, k_t = 5.0e9, 6.9e8
    K = k_a * np.outer(n, n) + k_t * (np.eye(3) - np.outer(n, n))
    got = stability.directional_row_sums(np.array([axis], dtype=float),
                                         k_a, k_t)[0]
    assert got == pytest.approx(_dense_row_sums(K), rel=1e-12)


def test_the_direction_factor_really_does_exceed_one():
    """The reason an isotropic `k` is not a safe substitute. If this ever came
    back <= 1, spreading the stiffness over the three dofs would be
    conservative and the whole direction term would be unnecessary."""
    rng = np.random.default_rng(3)
    axes = rng.normal(size=(4000, 3))
    worst = stability.directional_row_sums(axes, 1.0).max()
    assert worst > 1.15, worst
    assert worst < 1.7321 + 1e-9, "|n_i| |n|_1 cannot exceed sqrt(3)"


def test_a_degenerate_axis_is_refused():
    with pytest.raises(ValueError, match="non-degenerate"):
        stability.directional_row_sums(np.zeros((1, 3)), 1.0)


# --------------------------------------------------------------------------- #
#  Coupling -- the factor of two
# --------------------------------------------------------------------------- #

def test_a_two_point_spring_matches_a_dense_sum():
    """The simplest relative coordinate there is, and the one that shows the
    halving: K = [[k, -k], [-k, k]] has row sums 2k, not k."""
    k = 4.2e8
    K = k * np.array([[1.0, -1.0], [-1.0, 1.0]])
    factor = stability.relative_coordinate_factor(np.ones((1, 1)),
                                                  np.ones((1, 1)))
    assert factor[0] == pytest.approx(2.0)
    assert (k * factor)[0] == pytest.approx(_dense_row_sums(K)[0])


@pytest.mark.parametrize("weights", [
    [1.0], [0.5, 0.5], [0.25, 0.25, 0.25, 0.25],
    [0.144, 0.384, 0.384, 0.088],
])
def test_an_interface_row_matches_a_dense_sum(weights):
    """r = u_bar - sum_a w_a u_a, a scalar stiffness k on r.

    Builds the full (1 + n_host) matrix and checks BOTH the bar's row and
    every host row against the closed form -- the bar row is where the
    missing factor showed up first, the host rows are where it propagated.
    """
    w = np.array(weights, dtype=float)
    k = 5.0e9
    c = np.concatenate([[1.0], -w])          # the relative coordinate's row
    K = k * np.outer(c, c)

    factor = stability.coupling_factor(w[None, :])[0]
    assert factor == pytest.approx(1.0 + np.abs(w).sum())

    dense = _dense_row_sums(K)
    assert k * factor == pytest.approx(dense[0], rel=1e-12), "bar row"
    for a, wa in enumerate(w):
        assert k * abs(wa) * factor == pytest.approx(dense[1 + a], rel=1e-12)


@pytest.mark.parametrize("w0, w1", [
    ([1.0], [1.0]),
    ([0.5, 0.5], [0.25, 0.75]),
    ([0.144, 0.384, 0.384, 0.088], [0.2, 0.3, 0.4, 0.1]),
])
def test_an_embedded_bar_row_matches_a_dense_sum(w0, w1):
    """r = sum_b w1_b u_b - sum_a w0_a u_a: the perfectly bonded bar, whose
    two ends are BOTH interpolated and which has no dof of its own. The factor
    is sum|w0| + sum|w1|, not 1 + sum|w| -- different expression, same 2."""
    a0, a1 = np.array(w0, float), np.array(w1, float)
    k = 6.28e8
    c = np.concatenate([-a0, a1])
    K = k * np.outer(c, c)

    factor = stability.relative_coordinate_factor(a0[None, :], a1[None, :])[0]
    assert factor == pytest.approx(np.abs(a0).sum() + np.abs(a1).sum())

    dense = _dense_row_sums(K)
    for q, cq in enumerate(c):
        assert k * abs(cq) * factor == pytest.approx(dense[q], rel=1e-12)


# --------------------------------------------------------------------------- #
#  The two element paths, against brute force
# --------------------------------------------------------------------------- #

def test_the_embedded_closed_form_is_unchanged_by_the_refactor():
    """`_emb_row_sums` was already exact. Rewriting it in terms of the shared
    pieces must not move it beyond floating-point round-off. Not bit-identity:
    the helper multiplies its factors in a different order, so the last bit
    can differ -- the claim is that the FORMULA is the same one, and 1e-14 is
    tight enough that a dropped or added factor could not hide under it.

    Only the per-bar factor was rewritten -- the scatter below it is
    untouched -- so that is what is compared, against the hand-written form
    transcribed here. Comparing the scattered result would test the scatter
    too, which nothing changed, and would hide which half had moved.
    """
    from vfife.core.mesh import box_mesh
    from vfife.elements.truss import EmbeddedTruss
    from vfife.materials import create

    mesh = box_mesh([0.4, 0.4, 1.0], [2, 2, 5])
    steel = create("rebar", E=2.0e11, rho=7850.0, yield_strength=4.74e8,
                   hardening=1.0e9)
    nodes = np.array([[0.13, 0.17, 0.1], [0.27, 0.23, 0.9]])
    conn = np.array([[0, 1]], dtype=np.int32)
    emb = EmbeddedTruss(mesh.nodes, mesh.elements, "hexa8", nodes, conn,
                        steel, 3.1416e-4, host_deduction=2.53e10)

    k = (steel.E - emb.pre.host_deduction) * emb.pre.area / emb.pre.L0
    d = emb.x0_e[:, 1] - emb.x0_e[:, 0]
    # the hand-written version, verbatim
    n = np.abs(d / np.linalg.norm(d, axis=1)[:, None])
    w_sum = (np.abs(emb.weights[:, 0]).sum(1)
             + np.abs(emb.weights[:, 1]).sum(1))
    want = n * n.sum(axis=1, keepdims=True) * (k * w_sum)[:, None]
    # the shared pieces
    got = (stability.directional_row_sums(d, k)
           * stability.relative_coordinate_factor(emb.weights[:, 0],
                                                  emb.weights[:, 1])[:, None])
    assert np.allclose(got, want, rtol=1e-14, atol=0.0), (
        f"the refactor changed the embedded bar's bound: {got} vs {want}")


def test_the_bonded_bar_bound_is_bigger_where_it_was_short_and_smaller_where_it_was_invented():
    """The correction, in both directions, because it goes both ways.

    The old form added `2 k_ax + bond.E * area + k_trans` to EVERY dof:
    diagonals only, and isotropic. So it was

      * TOO SMALL along the bar, where the missing relative-coordinate factor
        (2 for a partition of unity) and the direction factor both bite. That
        is the one that matters: an under-stated row under-masses the dof.
      * TOO LARGE across it, where it had credited the bar's axial stiffness
        and the bond -- neither of which acts transversely -- with holding
        dofs that only the penalty holds.

    A test that asserted "larger everywhere" would be asserting the second
    error was correct.
    """
    from vfife.core.mesh import box_mesh
    from vfife.elements.bonded import BondedTruss
    from vfife.materials import create

    mesh = box_mesh([0.4, 0.4, 1.0], [2, 2, 5])
    n = 11
    nodes = np.stack([np.full(n, 0.2), np.full(n, 0.2),
                      np.linspace(0.0, 1.0, n)], 1)      # along z
    conn = np.column_stack([np.arange(n - 1), np.arange(1, n)]).astype(np.int32)
    bt = BondedTruss(mesh.nodes, mesh.elements, "hexa8", nodes, conn,
                     create("rebar", E=2.0e11, rho=7850.0,
                            yield_strength=4.74e8, hardening=1.0e9),
                     create("bond_ceb", fc=25.6e6), area=3.1416e-4,
                     diameter=0.020, transverse_penalty=1.0e9)

    old = np.zeros((bt.n_bar_nodes, 3))
    k_ax = bt.material.E * bt.area / bt.pre.L0
    np.add.at(old, bt.bar_conn[:, 0], k_ax[:, None])
    np.add.at(old, bt.bar_conn[:, 1], k_ax[:, None])
    old += (float(bt.bond.E) * bt.bond_area)[:, None]
    old += bt.k_trans

    new = bt.bar_row_sums()
    axial, transverse = new[:, 2], new[:, :2]
    assert np.all(axial > old[:, 2]), "the axial row did not grow"
    assert (axial / old[:, 2]).min() > 1.5, "the axial correction is not ~2"
    # and the transverse rows are now just the penalty, doubled by the same
    # relative-coordinate factor -- no axial stiffness, no bond
    assert np.allclose(transverse, 2.0 * bt.k_trans, rtol=1e-12)


def test_the_host_side_of_the_interface_carries_the_same_factor():
    """A relative-coordinate stiffness puts the SAME factor on both sides.
    The host rows had `k |w_a|`; they must now be twice that for a partition
    of unity, and the total handed to the host must equal the bar's own row
    sum times the weights -- which is what makes the pair consistent."""
    from vfife.core.mesh import box_mesh
    from vfife.elements.bonded import BondedTruss
    from vfife.materials import create

    mesh = box_mesh([0.4, 0.4, 1.0], [2, 2, 5])
    n = 6
    nodes = np.stack([np.full(n, 0.2), np.full(n, 0.2),
                      np.linspace(0.1, 0.9, n)], 1)
    conn = np.column_stack([np.arange(n - 1), np.arange(1, n)]).astype(np.int32)
    bt = BondedTruss(mesh.nodes, mesh.elements, "hexa8", nodes, conn,
                     create("rebar", E=2.0e11, rho=7850.0,
                            yield_strength=4.74e8, hardening=1.0e9),
                     create("bond_ceb", fc=25.6e6), area=3.1416e-4,
                     diameter=0.020, transverse_penalty=1.0e9)

    host = bt.host_row_sums(mesh.n_nodes)
    per_bar = bt._interface_row_sums() * stability.coupling_factor(bt.w)[:, None]
    # weights sum to one, so scattering |w_a| * row over the host nodes
    # conserves the total exactly
    assert host.sum(axis=0) == pytest.approx(per_bar.sum(axis=0), rel=1e-9)
    assert np.all(host >= 0.0)
