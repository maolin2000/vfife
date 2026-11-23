"""`core/staging.py`: the load path, not the iteration path.

WHY THERE IS A MODULE TO TEST
    Staged relaxation reaches ONE equilibrium point by thousands of explicit
    iterations whose path is not physical. Path-dependent state that advances
    on every one of them records wiggles as load reversals. It cost this
    project twice, in two materials, and was fixed twice, separately:

        fibre sections   absorbed-energy ratio 0.32 -> 1.77 (verification/case_11)
        bond damage      a MONOTONIC case moved 106.0 -> 24.3 um on n_steps alone

    The second happened because the first fix lived inside one solver with no
    name anyone could find. `core/staging.py` is the name.

WHAT IS DELIBERATELY NOT WIRED
    The continuum state, and the last test in this file is the measurement
    that says so rather than a shrug.
"""
import numpy as np
import pytest

from vfife.core import staging


def test_a_trial_state_starts_every_evaluation_from_the_committed_point():
    live = np.zeros(3)
    ts = staging.TrialState(live)

    live[:] = [1.0, 2.0, 3.0]          # an iteration wandered
    ts.restore_trial()
    assert np.array_equal(live, np.zeros(3)), "the wander survived a restore"

    live[:] = [4.0, 5.0, 6.0]          # the converged value
    ts.commit()
    live[:] = [9.9, 9.9, 9.9]          # another wander
    ts.restore_trial()
    assert np.array_equal(live, [4.0, 5.0, 6.0])


def test_a_trial_state_writes_through_rather_than_rebinding():
    """The kernels were handed this array at setup. Rebinding it here would
    leave them writing into an orphan while the solver read a different one --
    a divergence with no error anywhere."""
    live = np.zeros(3)
    ts = staging.TrialState(live)
    live[:] = [1.0, 2.0, 3.0]
    ts.commit()
    ts.restore_trial()
    assert ts.live is live


def test_an_empty_state_is_harmless():
    """Materials with no state at all still get registered by the loops that
    build these; `if size` is the guard the fibre path has always had."""
    ts = staging.TrialState(np.zeros((0, 4)))
    ts.restore_trial()
    ts.commit()


def test_a_callback_state_commits_the_way_its_owner_wants():
    """`bond_cyclic` banks an ENERGY INCREMENT between committed points rather
    than snapshotting an array, so it cannot be a TrialState. The protocol
    requires `commit` and nothing else for exactly this reason."""
    seen = []
    state = np.arange(4.0)
    cs = staging.CallbackState(seen.append, state)
    cs.commit()
    assert len(seen) == 1 and seen[0] is state


def test_the_registry_commits_everyone_and_restores_only_those_who_can():
    """`restore_trial` is optional: a member with the accumulate policy must
    not be forced into the restore one, and must not crash the loop either."""
    a = staging.TrialState(np.zeros(2))
    calls = []
    b = staging.CallbackState(lambda s: calls.append("commit"), None)

    reg = staging.Registry()
    reg.add(a)
    reg.add(b)
    assert len(reg) == 2

    a.live[:] = [7.0, 8.0]
    reg.restore_trial()                # a restores, b is skipped, no error
    assert np.array_equal(a.live, np.zeros(2))
    assert calls == []

    a.live[:] = [1.0, 2.0]
    reg.commit()
    assert calls == ["commit"]
    assert np.array_equal(a.committed, [1.0, 2.0])


def test_registering_something_with_no_commit_is_refused():
    reg = staging.Registry()
    with pytest.raises(TypeError, match="commit"):
        reg.add(object())


# --------------------------------------------------------------------------- #
#  Who is wired, and who is not
# --------------------------------------------------------------------------- #

def _solver(bond, tmp_path):
    import yaml

    from vfife.api import prepare_case
    from vfife.core.solver import Solver

    case = {
        "name": "staging", "units": {"length": "m", "force": "N",
                                     "stress": "Pa"},
        "mesh": {"dimensions": [0.2, 0.2, 1.0], "divisions": [2, 2, 4]},
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
                                           "divisions": 8}}},
        "assign": [{"reinforcement": "bar", "section": "bar20", "name": "bar",
                    **({"bond": bond} if bond else {})}],
        "regions": {"base": {"plane": {"point": [0, 0, 0],
                                       "normal": [0, 0, 1]}}},
        "constraints": [{"region": "base", "dofs": [1, 1, 1]}],
        "loads": [{"region": "base", "vector": [0.0, 0.0, 0.0]}],
        "solver": {"backend": "numpy", "mode": "relaxation",
                   "mass": "underwood", "n_steps": 10},
        "output": {"path": "unused.h5", "monitors": {}},
    }
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(case, allow_unicode=True), encoding="utf-8")
    prepared = prepare_case(str(path))
    return Solver(prepared.case.model_dump(mode="python"), prepared.mesh,
                  prepared.material, log=lambda *_: None)


@pytest.mark.parametrize("bond, expected", [
    (None, 0),
    ({"type": "bond_ceb", "fc": 25.6e6}, 1),
    ({"type": "bond_cyclic", "fc": 25.6e6}, 2),
])
def test_only_what_needs_staging_pays_for_it(tmp_path, bond, expected):
    """A perfectly bonded bar registers nothing and costs nothing.

    THIS TEST USED TO SAY `bond_ceb` REGISTERS NOTHING, AND IT WAS WRONG.
    The docstring read "bond_ceb has no irreversible accumulator". It has two:
    `s_max`, which only grows, and the unloading anchor (s_prev, tau_prev),
    which every evaluation moves. On the relaxation transient those record
    the overshoots -- a bar started 20 um off inside a rigid host came to rest
    at 300 um of permanent slip with zero force (tests/test_bonded_convergence.py).
    So every bond law is a TrialState (restore-then-integrate), and the
    cyclic law is additionally the CallbackState that banks its energy.
    """
    assert len(_solver(bond, tmp_path).staged) == expected


def test_the_continuum_state_is_excluded_on_purpose_and_here_is_the_reason():
    """NOT an oversight, and this test is where the reason lives.

    The continuum path keeps its material state on the iteration path, like
    every solver here did before the fibre fix. It was measured rather than
    assumed, on the Tanaka column with CDP:

      * the composite `damage_measure` FALLS inside a held load stage, so it
        is not a monotone accumulator at all and measuring it proves nothing;
      * the genuinely monotone equivalent plastic tensile strain `ept` DOES
        grow inside a held stage -- about 15-20% of that stage's total -- but
        the growth does NOT track the iteration count:

            stage 5    775 iterations    +3.3e-5
            stage 6   1275 iterations    +1.7e-6

        which is settling towards equilibrium, not accumulation driven by
        wiggles. CDP's return map is far less sensitive to direction flips
        than an anchored uniaxial rule such as GMP or concrete02, which is
        the difference that matters.

    Switching it to restore-then-integrate would change every CDP result that
    has been verified, so it is a decision to take on its own evidence rather
    than as a side effect of tidying. This test records the state of that
    decision; it will fail the day someone registers the continuum state, at
    which point the reasoning above has to be revisited rather than deleted.
    """
    import inspect

    from vfife.core.solver import Solver

    src = inspect.getsource(Solver._register_staged)
    assert "self.blocks" not in src and "self.state" not in src, (
        "the continuum state is now registered for staging -- see the "
        "measurements in this test's docstring and in core/staging.py before "
        "deciding that is right")
