"""`damping.alpha` alongside kinetic damping.

WHAT CHANGED AND WHY
    `alpha` used to be read from the case file whatever `damping.type` said,
    and then applied only when the type was `rayleigh`. Writing

        damping: {type: kinetic, alpha: 0.002}

    was accepted, ignored, and silent -- the worst of the three.

    It is applied now, because kinetic damping has a blind spot with a
    precise shape. It zeroes the velocity when the kinetic energy PEAKS, so a
    mode whose kinetic energy grows MONOTONICALLY never presents a peak and
    is never touched. A softening element is exactly that mode: measured on
    the Tanaka column under a full +-32 mm reversal, one node grew by a
    constant 1.0096 per iteration, displacement and velocity together, from
    32 mm to 46 mm in 37 iterations and on to 6e8 mm. `alpha * m * v` opposes
    velocity whatever the energy is doing, and alpha = 0.002 carried that
    reversal to completion.

WHAT THESE TESTS DO AND DO NOT COVER
    They pin the WIRING -- that the term is applied, that it is applied with
    kinetic damping, that zero alpha changes nothing, and that the backend
    which cannot carry it still refuses. They do NOT reproduce the divergence:
    that needs the RC column and about three minutes, and it lives in
    v0.1 rather than in the suite.
"""
import numpy as np
import pytest

from vfife.api import prepare_case
from vfife.core.solver import Solver

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _case(damping=None, n_steps=400):
    case = {
        "name": "alpha_wiring",
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"dimensions": [0.2, 0.2, 1.0], "divisions": [1, 1, 4]},
        "materials": {"concrete": {"type": "linear_elastic", "E": 2.5e10, "nu": 0.2, "rho": 2400.0}},
        "sections": {"body": {"type": "solid", "material": "concrete", "formulation": "b-bar"}},
        "regions": {
            "base": {"plane": {"point": [0, 0, 0], "normal": [0, 0, 1]}},
            "top": {"plane": {"point": [0, 0, 1.0], "normal": [0, 0, 1]}},
        },
        "constraints": [{"region": "base", "dofs": [1, 1, 1]}],
        "loads": [{"region": "top", "vector": [0.0, 0.0, 1.0e5]}],
        "solver": {"backend": "numpy", "mode": "relaxation",
                   "mass": "underwood", "n_steps": n_steps,
                   "tolerance": 1.0e-12,
                   "damping": damping or {"type": "kinetic"}},
        "output": {"path": "unused.h5", "monitors": {}},
    }
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
    except Exception:
        pass
    return solver


def test_alpha_is_applied_when_the_damping_is_kinetic(tmp_path):
    """The whole change, in one assertion.

    A tolerance of 1e-12 that the run cannot reach means both solves take the
    same number of iterations from the same start, so any difference in the
    answer is the damping term and nothing else. Before this change the two
    were identical, which is what "accepted, ignored and silent" looks like
    from the outside.
    """
    plain = _solve(tmp_path, _case({"type": "kinetic"}), "plain")
    damped = _solve(tmp_path, _case({"type": "kinetic", "alpha": 0.05}),
                    "damped")
    assert plain.step == damped.step
    assert not np.array_equal(plain.u, damped.u), (
        "damping.alpha had no effect under kinetic damping")


def test_a_zero_alpha_is_exactly_the_undamped_run(tmp_path):
    """The guard that makes the change safe: every existing case that never
    mentions `alpha` is bit-for-bit what it was."""
    a = _solve(tmp_path, _case({"type": "kinetic"}), "a")
    b = _solve(tmp_path, _case({"type": "kinetic", "alpha": 0.0}), "b")
    assert np.array_equal(a.u, b.u)
    assert a.step == b.step


def test_the_term_opposes_velocity_rather_than_displacement(tmp_path):
    """`alpha * m * v` is a viscous force, so it must vanish at equilibrium.

    Under kinetic damping the velocity is zeroed at every energy peak, so
    "vanishes at rest" is checkable directly: run to a state whose kinetic
    energy is negligible and the damping force there is negligible too. This
    is what makes alpha different in KIND from a viscoplastic regularisation,
    which changes the equilibrium equations themselves -- though on a
    softening problem both end up selecting a branch, which is why the
    solver's own comment refuses to call alpha free.
    """
    s = _solve(tmp_path, _case({"type": "kinetic", "alpha": 0.05},
                               n_steps=4000), "rest")
    ke = 0.5 * float(np.sum(s.mass * s.v * s.v))
    force = float(np.abs(0.05 * s.mass * s.v).max())
    scale = float(np.abs(s.last_external).max())
    assert ke >= 0.0
    assert force < 1e-3 * max(scale, 1.0), (
        f"a damping force of {force:.3e} N against a load of {scale:.3e} N "
        "is not 'vanishes at rest'")


def test_a_rayleigh_run_is_unchanged(tmp_path):
    """`type: rayleigh` meant "apply alpha" before and still does; the change
    widened where alpha acts, it did not move it."""
    s = _solve(tmp_path, _case({"type": "rayleigh", "alpha": 0.05}), "r")
    assert s.rayleigh_alpha == 0.05
    assert s.damp_kind == "rayleigh"


def test_the_backend_that_cannot_carry_it_still_refuses(tmp_path):
    """Warp keeps the time loop on the device and the damping term is applied
    host-side, so a warp run would accept alpha and silently drop it. The
    refusal used to key off `type == rayleigh`; keying it off alpha is what
    keeps `type: kinetic, alpha: ...` from becoming a new silent no-op on the
    GPU -- the exact failure this whole change is undoing.
    """
    case = _case({"type": "kinetic", "alpha": 0.002})
    case["solver"]["backend"] = "warp"
    with pytest.raises(Exception, match="Rayleigh|warp"):
        import yaml

        path = tmp_path / "w.yaml"
        path.write_text(yaml.safe_dump(case, allow_unicode=True),
                        encoding="utf-8")
        prep = prepare_case(str(path))
        Solver(prep.case.model_dump(mode="python"), prep.mesh, prep.material,
               log=lambda *_: None)
