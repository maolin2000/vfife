"""Stage B2: bonded bars inside the solver, judged against perfect bond.

THE JUDGE
    Stiffen the bond and the answer must walk back to the perfectly bonded one.
    Nothing else proves that the new path is a GENERALISATION of the old rather
    than a different model that happens to run, and it is not a formality: the
    first version of the interface had both force terms negated, every
    element-level test agreed with it because they asserted the same wrong
    sign, and the solver's -F_int/m then turned the bond's restoring force into
    a driving one. Every bonded run reached 1e150. This test is what found it.

    Measured, block in tension with one bar, host tip displacement:

        perfect bond          216.4978 um     slip     -     30 500 iterations
        MC2010 as published   218.6725        24.905 um      30 500
        bond x100 stiffer     216.5104         2.658         151 000

    Softer than perfect when the bar can slip, and converging back as it
    cannot -- in the displacement, in the slip, and monotonically in both.

    TWO THINGS ABOUT THAT TABLE
        The iteration counts are in it because they are the point: a bond 100
        times stiffer needs five times the budget, and giving every case the
        same 60 000 made the stiff one stop at residual 1.35e-7 against a
        tolerance of 1e-8 with a kinetic energy of 3e-15 -- at rest, and
        reported as a failure to converge. That is a budget, not a defect,
        but it is indistinguishable from one until the budget is raised.

        The perfect-bond number is unchanged from when this test was written;
        the other two are NOT. The published-MC2010 case was recorded here as
        217.7357 um with 20.837 um of slip and now gives 218.6725 / 24.905,
        and the x100 case moved from 216.5448 / 2.261 to 216.5104 / 2.658.
        Something in the bond path moved those by up to 20% in slip after
        they were recorded -- `s_elastic` and the transverse-penalty default
        both landed in that window -- and it has NOT been chased down. The
        assertions below are all orderings and limits and they still hold, so
        the test is honest; the table is now the current measurement rather
        than a corroborated one.
"""
import numpy as np
import pytest

from vfife.api import prepare_case
from vfife.core.diagnostics import UserError
from vfife.core.solver import Solver

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _case(bond=None, n_steps=60000, tol=1.0e-8, backend="numpy"):
    case = {
        "name": "bonded_vs_perfect",
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"dimensions": [0.2, 0.2, 1.0], "divisions": [2, 2, 10]},
        "materials": {
            "concrete": {"type": "linear_elastic", "E": 2.53e10, "nu": 0.2, "rho": 2400.0},
            "steel": {"type": "rebar", "E": 2.0e11, "rho": 7850.0,
                      "yield_strength": 4.74e8, "hardening": 1.0e9},
        },
        "sections": {
            "body": {"type": "solid", "material": "concrete", "formulation": "b-bar"},
            "bar20": {"type": "bar", "material": "steel", "shape": "circle",
                      "diameter": 0.020, "area": 3.1416e-4},
        },
        "reinforcement": {"bar": {"line": {"start": [0.1, 0.1, 0.0], "end": [0.1, 0.1, 1.0],
                                           "divisions": 20}}},
        "assign": [{"reinforcement": "bar", "section": "bar20", "name": "bar"}],
        "regions": {
            "base": {"plane": {"point": [0, 0, 0], "normal": [0, 0, 1]}},
            "top": {"plane": {"point": [0, 0, 1.0], "normal": [0, 0, 1]}},
        },
        "constraints": [{"region": "base", "dofs": [1, 1, 1]}],
        "loads": [{"region": "top", "vector": [0.0, 0.0, 2.0e5]}],
        "solver": {"backend": backend, "mode": "relaxation",
                   "mass": "underwood", "n_steps": n_steps, "tolerance": tol,
                   "damping": {"type": "kinetic"}},
        "output": {"path": "unused.h5", "monitors": {}},
    }
    if bond is not None:
        case["assign"][0]["bond"] = bond
    return case


def _solve(tmp_path, case, name):
    import yaml

    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(case, allow_unicode=True), encoding="utf-8")
    prep = prepare_case(str(path))
    solver = Solver(prep.case.model_dump(mode="python"), prep.mesh,
                    prep.material, log=lambda *_: None)
    try:
        solver.run()
        converged = True
    except Exception:
        converged = False
    return solver, converged


# --------------------------------------------------------------------------- #
#  Wiring, cheap enough to run every time
# --------------------------------------------------------------------------- #

def test_a_bond_block_gives_the_bar_its_own_dofs(tmp_path):
    """Without a `bond` block the bar goes down the perfectly bonded path and
    has no dofs at all; with one it becomes a separate object with its own
    displacement array."""
    plain, _ = _solve(tmp_path, _case(n_steps=10), "plain")
    assert plain.rebar and not plain.bonded

    bonded, _ = _solve(tmp_path, _case({"type": "bond_ceb", "fc": 25.6e6},
                                       n_steps=10), "bonded")
    assert bonded.bonded and not bonded.rebar
    bt = bonded.bonded[0]
    assert bt.u_bar.shape == (bt.n_bar_nodes, 3)
    assert bt.v_bar.shape == (bt.n_bar_nodes, 3)


def test_a_bond_block_without_a_diameter_is_refused(tmp_path):
    """Bond acts on the bar SURFACE, so the area alone does not determine it.
    Silently assuming a diameter would put a factor of pi*d into every result."""
    case = _case({"type": "bond_ceb", "fc": 25.6e6}, n_steps=10)
    # a `general` bar section knows its area and nothing about its surface
    case["sections"]["bar20"] = {"type": "bar", "material": "steel",
                                 "shape": "general", "area": 3.1416e-4}
    with pytest.raises(ValueError, match="diameter"):
        _solve(tmp_path, case, "nodia")


def test_bond_slip_is_refused_on_warp(tmp_path):
    """Warp keeps its whole time loop on the device and never brings `u` back,
    so a host-side bar would slip against a displacement field that stopped
    updating.

    Runs everywhere, INCLUDING where warp is not installed, and it asserts the
    guard's own CODE rather than any words. The guard sits in `Solver.__init__`
    before the kernel is built, so it fires ahead of the "warp is not
    installed" error -- and checking the code is what makes that a claim
    rather than a coincidence: if the order ever changed, this fails loudly
    instead of passing on the wrong exception.

    The code is an attribute, not part of `str(exc)`, so `pytest.raises(match=)`
    cannot see it -- which is how the first version of this assertion came to
    match the loose `"warp|BONDED_BAR_BACKEND"` and would have been satisfied
    by the missing-module error.
    """
    case = _case({"type": "bond_ceb", "fc": 25.6e6}, n_steps=10,
                 backend="warp")
    with pytest.raises(UserError) as exc:
        _solve(tmp_path, case, "warp")
    assert exc.value.code == "BONDED_BAR_BACKEND", (
        f"raised {exc.value.code!r} -- the bonded guard did not fire first")


def test_bond_slip_is_allowed_on_numba(tmp_path):
    """The other half, and the one that matters: numba is NOT resident -- only
    warp defines `force_resident` -- so it goes through the host arrays every
    step and the bar joins it exactly as under numpy.

    The first version of the guard refused every backend but numpy, which was
    wrong by inspection and cost the ability to use numba for no reason. This
    is what stops that coming back.

    SEPARATE FROM THE WARP TEST BECAUSE IT NEEDS NUMBA INSTALLED. It used to
    be the second half of one test, and CI -- which has no numba -- failed on
    `this case asks for the numba backend, which is not installed here`. The
    guard was fine; the test was asking the wrong environment.
    """
    pytest.importorskip("numba")
    solver, _ = _solve(tmp_path, _case({"type": "bond_ceb", "fc": 25.6e6},
                                       n_steps=10, backend="numba"), "numba")
    assert solver.bonded


def test_the_bar_carries_its_own_fictitious_mass(tmp_path):
    """One entry per bonded block, sized from that bar's own row sums. Under
    relaxation the host mass is fictitious too, and the two have to be built
    the same way or the coupled pair is not stable."""
    solver, _ = _solve(tmp_path, _case({"type": "bond_ceb", "fc": 25.6e6},
                                       n_steps=10), "mass")
    assert len(solver.mass_bar) == 1
    m = solver.mass_bar[0]
    assert m.shape == (solver.bonded[0].n_bar_nodes, 3)
    assert np.all(m > 0)


def test_a_checkpoint_carries_the_slip(tmp_path):
    """u_bar, v_bar and both state arrays. A restart that rebuilt the bar at
    zero slip would silently heal every bond in the model."""
    solver, _ = _solve(tmp_path, _case({"type": "bond_ceb", "fc": 25.6e6},
                                       n_steps=200), "ck")
    ck = solver.checkpoint()
    for suffix in ("u", "v", "s", "b"):
        assert f"bonded0_{suffix}" in ck

    bt = solver.bonded[0]
    assert np.abs(ck["bonded0_b"]).max() > 0.0          # there is slip to lose
    bt.u_bar[:] = 1.0
    bt.bond_state[:] = 0.0
    # the checkpoint is a copy: it used to BE these arrays, and restoring it
    # onto itself could not fail
    assert not np.any(ck["bonded0_u"] == 1.0) and np.abs(ck["bonded0_b"]).max() > 0.0
    solver.restore(ck)
    assert np.array_equal(bt.u_bar, ck["bonded0_u"])
    assert np.array_equal(bt.bond_state, ck["bonded0_b"])


# --------------------------------------------------------------------------- #
#  The judge
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_stiffening_the_bond_walks_back_to_perfect_bond(tmp_path):
    """The whole of stage B, in one assertion.

    Three solves, so it is slow; there is no cheaper form of this check,
    because the rigid limit is slip -> 0 at finite force and only an
    equilibrium finds it.
    """
    perfect, ok = _solve(tmp_path, _case(), "perfect")
    assert ok
    tip_perfect = float(perfect.u[:, 2].max())

    results = []
    # the budget scales with the bond stiffness: see the docstring. 60 000 is
    # plenty for the soft case and leaves the stiff one at rest but short of
    # tolerance, which reads as a failure and is not one.
    for name, s1, n_steps in (("soft", 1.0e-3, 60000),
                              ("stiff", 1.0e-5, 300000)):
        bond = {"type": "bond_ceb", "fc": 25.6e6, "s1": s1,
                "s2": 2.0 * s1, "s3": 10.0 * s1}
        solver, ok = _solve(tmp_path, _case(bond, n_steps=n_steps), name)
        assert ok, f"{name} did not converge"
        assert np.isfinite(solver.u).all()
        bt = solver.bonded[0]
        results.append((float(solver.u[:, 2].max()),
                        float(np.abs(bt.slip(solver.u)[1]).max())))

    (tip_soft, slip_soft), (tip_stiff, slip_stiff) = results

    # a bar that can slip makes the member softer, never stiffer
    assert tip_soft > tip_perfect
    assert tip_stiff > tip_perfect

    # and stiffening the bond moves both the displacement and the slip back
    assert abs(tip_stiff - tip_perfect) < abs(tip_soft - tip_perfect)
    assert slip_stiff < slip_soft

    # Close enough that the limit is not in doubt. The threshold is 3e-3, and
    # the reason it is not tighter is worth writing down rather than tuning
    # away: this quantity is not pinned by the residual tolerance. Measured on
    # the stiff case, varying only the iteration budget --
    #
    #     300 000 steps   +0.1296 %
    #     600 000         +0.0092 %
    #   1 200 000         +0.1212 %
    #
    # -- non-monotonic, so picking 600 000 would be picking a lucky value. A
    # relative residual of 1e-8 on a bond a hundred times stiffer than the
    # block it is embedded in leaves about a part in a thousand of slack in
    # the displacement, and that is what this number is. The CLAIM being
    # tested is the limit, and 0.13% is comfortably a limit.
    assert abs(tip_stiff - tip_perfect) / tip_perfect < 3.0e-3


@pytest.mark.slow
@pytest.mark.parametrize("n_steps", [20000, 60000])
def test_a_cyclic_bond_law_leaves_a_monotonic_solve_exactly_alone(tmp_path,
                                                                  n_steps):
    """The defect this test exists for was found here, not in the material.

    `bond_cyclic` degrades from dissipated energy. Its first version banked
    that energy every time it was evaluated -- which, in a relaxation solve,
    is every explicit iteration. The slip overshoots its converged value on
    the way in (24.9 -> 33.9 um on this very case) and comes back, so every
    convergence wiggle read as a load reversal, and the feedback did the rest:

        monotonic tension, tolerance 1e-8, only n_steps changed
            n_steps  60 000    slip 106.0 um    damage 0.0201
            n_steps 200 000    slip  24.3 um    damage 0.0172
            bond_ceb, either   slip  24.9 / 28.9 um    no damage

    A factor of 4.4 in the answer from a solver setting. Every constitutive
    test in tests/test_bond_cyclic.py passed throughout, because a
    constitutive test drives the law along a load path -- which is exactly
    what a relaxation solve does not do.

    The fix is `BondCyclic.commit`, called from `Solver._commit_bonded` when a
    load level converges. What this test pins is its consequence: on a
    monotonic solve the cyclic law is the monotonic law, at ANY iteration
    budget. The two budgets are the point -- one alone would have passed
    against the broken version too.
    """
    ref, _ = _solve(tmp_path, _case({"type": "bond_ceb", "fc": 25.6e6},
                                    n_steps=n_steps), f"ceb{n_steps}")
    cyc, _ = _solve(tmp_path, _case({"type": "bond_cyclic", "fc": 25.6e6},
                                    n_steps=n_steps), f"cyc{n_steps}")

    assert np.array_equal(ref.u, cyc.u), "a monotonic solve moved"
    a, b = ref.bonded[0], cyc.bonded[0]
    assert np.array_equal(a.u_bar, b.u_bar)
    assert float(np.max(b.bond.damage(b.bond_state[:, 3]))) == 0.0


@pytest.mark.slow
def test_numpy_and_numba_agree_with_a_bonded_bar(tmp_path):
    """The bar is host-side under both; what changes is the concrete kernel.

    A disagreement would mean the coupling reads something the backend does not
    keep up to date -- a silent divergence between two lines that differ only
    in a setting the analyst thinks is about speed, which is the most expensive
    class of error this project has met.

    Measured: host displacement 1.4e-10, bar displacement 2.1e-9, bond state
    2.0e-8, with numba about twice as fast.
    """
    bond = {"type": "bond_ceb", "fc": 25.6e6}
    out = {}
    for backend in ("numpy", "numba"):
        solver, _ = _solve(tmp_path, _case(bond, n_steps=40000, tol=1.0e-8,
                                           backend=backend), backend)
        bt = solver.bonded[0]
        out[backend] = (solver.u.copy(), bt.u_bar.copy(), bt.bond_state.copy())

    for i, what in enumerate(("host displacement", "bar displacement",
                              "bond state")):
        a, b = out["numpy"][i], out["numba"][i]
        rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
        assert rel < 1e-6, f"{what} split by {rel:.2e}"
