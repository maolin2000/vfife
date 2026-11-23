"""M02 qualification tests for the general one-dimensional material."""

from pathlib import Path

import numpy as np
import pytest

from vfife.cli import _with_element, build, make_solver
from vfife.materials import create, definition
from vfife.io import schema


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "cases" / "case_02_truss_bar.yaml"


def material(**overrides):
    args = dict(
        E=200_000.0, rho=7850.0,
        tension=[[0.0, 200.0], [0.01, 300.0], [0.03, 340.0]],
        compression=[[0.0, 150.0], [0.01, 230.0], [0.03, 260.0]],
        hardening_mode="isotropic",
    )
    args.update(overrides)
    return create("piecewise_uniaxial", **args)


def follow(mat, history, state=None):
    state = mat.init_state(1) if state is None else state
    stress = []
    for value in history:
        eps = np.zeros((1, 6)); eps[0, 0] = value
        stress.append(mat.update(eps, state, 1.0)[0, 0])
    return np.asarray(stress), state


def test_metadata_declares_uniaxial_numpy_material_point_only():
    meta = definition("piecewise_uniaxial")
    assert meta.kinematics == ("uniaxial",)
    assert meta.backends == ("numpy",)
    assert meta.nonlinear
    assert "hysteretic_energy" in meta.output_fields


def test_monotonic_tension_and_compression_follow_separate_tables():
    mat = material()
    # At alpha=0.01 the tensile and compressive yield magnitudes are 300/230.
    # Total strain is eps_p + sigma/E.
    tension, _ = follow(mat, [0.0, 200/200_000, 0.01 + 300/200_000])
    compression, _ = follow(material(), [0.0, -150/200_000, -(0.01 + 230/200_000)])
    assert tension == pytest.approx([0.0, 200.0, 300.0], abs=2e-8)
    assert compression == pytest.approx([0.0, -150.0, -230.0], abs=2e-8)


def test_step_partition_does_not_change_rate_independent_monotonic_result():
    target = 0.018
    coarse, coarse_state = follow(material(), [target])
    fine, fine_state = follow(material(), np.linspace(0, target, 101))
    assert coarse[-1] == pytest.approx(fine[-1], rel=1e-11)
    assert coarse_state[0, :3] == pytest.approx(fine_state[0, :3], rel=1e-11)


def test_unload_reverse_distinguishes_isotropic_and_kinematic_hardening():
    history = [0.0, 0.012, 0.0, -0.001]
    iso, _ = follow(material(hardening_mode="isotropic"), history)
    kin, state = follow(material(hardening_mode="kinematic"), history)
    # Positive backstress causes earlier reverse yielding (Bauschinger effect).
    assert state[0, 2] > 0
    assert kin[-1] > iso[-1]


def test_closed_plastic_cycle_has_positive_loop_energy_and_elastic_cycle_has_zero():
    elastic, elastic_state = follow(material(), [0.0, 0.0005, 0.0])
    plastic, plastic_state = follow(
        material(hardening_mode="kinematic"),
        [0.0, 0.006, 0.0, -0.006, 0.0, 0.006, 0.0],
    )
    elastic_energy = material().output_fields(elastic_state)["hysteretic_energy"][0]
    plastic_energy = material(hardening_mode="kinematic").output_fields(plastic_state)["hysteretic_energy"][0]
    assert elastic[-1] == pytest.approx(0.0, abs=1e-12)
    assert elastic_energy == pytest.approx(0.0, abs=1e-12)
    assert plastic_energy > 0.0


def test_invalid_tables_and_hardening_options_fail_before_analysis():
    with pytest.raises(ValueError, match="start at zero"):
        material(tension=[[0.1, 200], [0.2, 250]])
    with pytest.raises(ValueError, match="non-decreasing"):
        material(tension=[[0, 200], [0.1, 190]])
    with pytest.raises(ValueError, match="hardening_mode"):
        material(hardening_mode="magic")
    with pytest.raises(ValueError, match="kinematic_fraction"):
        material(hardening_mode="combined", kinematic_fraction=1.2)


def test_truss2_accepts_material_and_checkpoint_restores_all_history():
    case, _ = schema.load(str(CASE))
    raw = case.model_dump()
    raw["material"] = {
        "type": "piecewise_uniaxial", "E": 2e11, "rho": 7850.0,
        "tension": [[0.0, 2e8], [0.01, 3e8]],
        "compression": [[0.0, 1.5e8], [0.01, 2.4e8]],
        "hardening_mode": "combined", "kinematic_fraction": 0.4,
    }
    case = schema.Case.model_validate(raw)
    p = _with_element(case, "truss2")
    mesh, mat, _ = build(case, str(CASE))
    solver = make_solver(p, mesh, mat)
    solver.u[1, 0] = 0.005
    _force, stress = solver._force(True)
    assert 2e8 < stress[0, 0, 0] < 3e8
    saved = {key: np.asarray(value).copy() for key, value in solver.checkpoint().items()}
    solver.state.fill(0); solver.u.fill(0)
    solver.restore(saved)
    assert np.array_equal(solver.state, saved["state"])
    fields = solver.nodal_fields(stress)
    assert fields["hysteretic_energy"].shape == (1,)


def test_the_return_map_is_exact_not_bisected():
    """v0.1 replaced 80 bisection steps with the closed-form root, segment by
    segment. The consistency equation is linear inside a table segment, so
    'converged to 1e-13' was never the best available answer -- and the cost
    of getting it that way was 465 us a material point.

    Checked against a 200-step bisection of the same residual over random
    states, including states that start part way up the table and states whose
    root crosses a breakpoint.
    """
    import numpy as np
    from vfife.materials.base import create

    m = create("piecewise_uniaxial", E=2.0e11, rho=7850.0,
               tension=[[0.0, 4.0e8], [0.002, 4.6e8], [0.01, 5.2e8], [0.05, 6.0e8]],
               compression=[[0.0, 3.6e8], [0.004, 4.4e8], [0.02, 5.0e8]],
               hardening_mode="combined", kinematic_fraction=0.4)

    def bisected(trial, alpha, X, d, table):
        b = m.kinematic_fraction
        old = m._value(table, alpha)

        def residual(dp):
            new = m._value(table, alpha + dp)
            return d * (trial - d * m.E * dp - (X + d * b * (new - old))) \
                - m._bound(table, alpha + dp)

        lo, hi = 0.0, max(abs(trial) / m.E, 1e-14)
        while residual(hi) > 0:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if residual(mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    rng = np.random.default_rng(7)
    crossings = 0
    for _ in range(2000):
        alpha = float(rng.uniform(0.0, 0.06))
        X = float(rng.normal(0.0, 5.0e7))
        d = 1.0 if rng.random() < 0.5 else -1.0
        table = m._tension if d > 0 else m._compression
        trial = float(d * rng.uniform(3.5e8, 9.0e8) + X)
        if d * (trial - X) <= m._bound(table, alpha):
            continue
        dp, _ = m._plastic_increment(trial, 0.0, alpha, X, d, table)
        assert dp == pytest.approx(bisected(trial, alpha, X, d, table), rel=1e-9)
        if any(alpha < x < alpha + dp for x, _ in table):
            crossings += 1
    assert crossings > 50, "the random states never crossed a breakpoint"
