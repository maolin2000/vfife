"""Solver-level verification: convergence, stability, reproducibility.

Meshes here are deliberately small so the suite stays in CI range; the accuracy
claims are made as CONVERGENCE statements (refining the mesh must move the
answer monotonically towards beam theory) rather than absolute tolerances on a
coarse mesh, because trilinear hexahedra lock in bending and approach the beam
solution from below -- that is discretisation, not a defect.
"""

import copy

import numpy as np
import pytest

from vfife.core.mesh import box_mesh
from vfife.core.solver import Solver
from vfife.core.diagnostics import ConvergenceFailure
from vfife.materials import create

L, B, H = 1.0, 0.05, 0.05
E, NU, RHO = 210e9, 0.3, 7800.0
TIP_LOAD = 200.0


def cantilever_case(formulation="b-bar", nx=16, nz=2, mode="relaxation",
                    n_steps=40000, damping=None, load=TIP_LOAD, tol=1e-6):
    return {
        "mesh": {"dimensions": [L, B, H], "divisions": [nx, 1, nz]},
        "element": {"formulation": formulation, "hourglass_coeff": 0.05},
        "regions": {
            "clamp": {"box": [-1e-9, 1e-9, -1e9, 1e9, -1e9, 1e9]},
            "tip": {"box": [L - 1e-9, L + 1e-9, -1e9, 1e9, -1e9, 1e9]},
        },
        "constraints": [{"region": "clamp", "dofs": [1, 1, 1], "value": None}],
        "loads": [{"region": "tip", "vector": [0.0, 0.0, -load]}],
        "solver": {
            "mode": mode,
            "n_steps": n_steps,
            "total_time": None,
            "tolerance": tol,
            "damping": damping or {"type": "kinetic", "coeff": 0.0},

        },
        "output": {
            "monitors": {"tip_w": {"kind": "displacement", "region": "tip", "dof": "z"}}
        },
    }


def solve(case, nu=NU, mat=None, on_output=None, output_every=10 ** 9,
          allow_max_steps=False):
    mesh = box_mesh(case["mesh"]["dimensions"], case["mesh"]["divisions"])
    mat = mat or create("linear_elastic", E=E, nu=nu, rho=RHO)
    s = Solver(case, mesh, mat, log=lambda *_: None)
    try:
        s.run(on_output=on_output, output_every=output_every, progress_every=100)
    except ConvergenceFailure:
        if not allow_max_steps:
            raise
        assert s.diagnostics["status"] == "MAX_STEPS"
    return s


def tip_w(s):
    return float(np.abs(s.u[s.mesh.regions["tip"], 2]).mean())


def timoshenko_tip():
    I = B * H ** 3 / 12.0
    Gs = E / (2 * (1 + NU))
    return TIP_LOAD * L ** 3 / (3 * E * I) + TIP_LOAD * L / ((5.0 / 6.0) * Gs * B * H)


def test_relaxation_converges_and_is_stable():
    """Underwood masses + kinetic damping must reach the residual tolerance."""
    s = solve(cantilever_case(n_steps=40000, tol=1e-6))
    assert np.isfinite(s.u).all(), "relaxation diverged"
    assert s.residuals[-1] < 1e-6
    assert s.step < 40000, "did not hit the tolerance before the step limit"
    assert 5 < s.n_resets < 500, f"kinetic resets look wrong: {s.n_resets}"


@pytest.mark.slow
# 网格加密收敛研究。同一个解析答案由 case_01 的契约每晚核对。
def test_cantilever_converges_to_beam_theory():
    """Refining must approach Timoshenko -- from below for full integration.

    Trilinear hexahedra lock in bending, so b-bar (full integration) is too
    stiff and converges upward; reduced integration with hourglass control is
    slightly too soft and converges downward.  Bracketing the beam solution
    from both sides is much stronger evidence than either sequence alone.
    """
    ref = timoshenko_tip()
    full = [tip_w(solve(cantilever_case("b-bar", nx=nx, nz=nz, n_steps=60000)))
            for nx, nz in ((16, 2), (32, 4))]
    red = [tip_w(solve(cantilever_case("reduced", nx=nx, nz=nz, n_steps=60000)))
           for nx, nz in ((16, 2), (32, 4))]

    assert full[0] < full[1] < ref, f"b-bar not converging from below: {full}"
    assert red[0] > red[1] > ref, f"reduced not converging from above: {red}"
    assert full[1] / ref > 0.85
    assert red[1] / ref < 1.10


@pytest.mark.slow
# 两种阻尼策略的优劣比较，是质量论断而非正确性闸门。
def test_kinetic_damping_beats_fixed_viscous():
    """Why adaptive relaxation beats a hand-tuned damping coefficient.

    In relaxation mode dt = 1, so the useful viscous coefficients are O(1e-3),
    not the O(1e2) of a physical-time run -- one of the traps of carrying a
    hand-picked zeta across problems.  Even at its best setting the fixed
    coefficient needs more steps than kinetic damping needs with no tuning at
    all, and one decade off it does not converge within the step limit.
    """
    kin = solve(cantilever_case(n_steps=40000))
    target = tip_w(kin)

    reached, missed = [], []
    for zeta in (1e-3, 1e-2, 3e-2, 1e-1):
        try:
            v = solve(cantilever_case(
                n_steps=40000, damping={"type": "viscous", "coeff": zeta}))
        except ConvergenceFailure:
            missed.append((zeta, 40000))
            continue
        ok = np.isfinite(v.u).all() and abs(tip_w(v) / target - 1.0) < 0.02
        (reached if ok else missed).append((zeta, v.step))

    assert reached, "no viscous coefficient reached the relaxation answer"
    assert missed, "expected most viscous coefficients to miss"
    best = min(step for _, step in reached)
    assert kin.step < best, f"kinetic {kin.step} steps vs best viscous {best}"


@pytest.mark.slow
# 长时程动力行为。
def test_undamped_step_response_is_bounded():
    """A suddenly applied load oscillates about the static solution and does not grow."""
    case = cantilever_case(mode="dynamic", damping={"type": "none"})
    case["solver"]["total_time"] = 0.06        # several natural periods
    case["solver"]["mass"] = "physical"
    # a STEP load, which is now spelled by naming the `hold` amplitude -- the
    # default ramps the load on, and a ramped load does not produce the
    # factor-of-two dynamic amplification this test is about
    for ld in case["loads"]:
        ld["amplitude"] = "hold"

    mesh = box_mesh(case["mesh"]["dimensions"], case["mesh"]["divisions"])
    mat = create("linear_elastic", E=E, nu=NU, rho=RHO)
    s = Solver(case, mesh, mat, log=lambda *_: None)
    peaks = []
    s.run(on_output=lambda sol, *_: peaks.append(tip_w(sol)),
          output_every=25, progress_every=10 ** 9)

    static = tip_w(solve(cantilever_case()))
    assert 1.6 * static < max(peaks) < 2.4 * static, (
        f"peak {max(peaks):.3e} should be about twice static {static:.3e}"
    )
    third = len(peaks) // 3
    assert max(peaks[-third:]) < 1.05 * max(peaks[:third]), (
        "undamped response is growing: energy is being injected"
    )


def test_bitwise_reproducible():
    case = cantilever_case(n_steps=600, tol=0.0)
    a = solve(copy.deepcopy(case), allow_max_steps=True)
    b = solve(copy.deepcopy(case), allow_max_steps=True)
    assert np.array_equal(a.u, b.u), "solver is not bit-for-bit reproducible"


def test_restart_matches_uninterrupted_run():
    """Checkpoint mid-run, restore, continue -- and land in the same place.

    The checkpoint has to come from the SAME configuration, stopped part way,
    not from a separately configured shorter run. Amplitudes are evaluated on
    the step's own progress, so a 400-iteration step and a 1000-iteration step
    are at different points of their loading at iteration 400, and comparing
    them would be comparing two different analyses.
    """
    case = cantilever_case(n_steps=1000, tol=0.0)
    full = solve(copy.deepcopy(case), allow_max_steps=True)

    grabbed = {}

    def grab(s, step, t, sig):
        if step == 400:
            grabbed["ck"] = {k: np.array(v, copy=True)
                             for k, v in s.checkpoint().items()}

    solve(copy.deepcopy(case), on_output=grab, output_every=400,
          allow_max_steps=True)
    ck = grabbed["ck"]

    mesh = box_mesh(case["mesh"]["dimensions"], case["mesh"]["divisions"])
    mat = create("linear_elastic", E=E, nu=NU, rho=RHO)
    b = Solver(copy.deepcopy(case), mesh, mat, log=lambda *_: None)
    b.restore(ck)
    with pytest.raises(ConvergenceFailure, match="without convergence"):
        b.run(output_every=10 ** 9, progress_every=100)
    assert b.diagnostics["status"] == "MAX_STEPS"

    assert np.abs(full.u - b.u).max() / np.abs(full.u).max() < 1e-12


def test_chaboche_beam_yields():
    """End to end with the full constitutive model."""
    mat = create(
        "chaboche", E=E, nu=NU, rho=RHO, yield_strength=250e6,
        Q_inf=50e6, b=20.0, H_iso=1e9, C1=20e9, gamma1=300.0,
        C2=4e9, gamma2=30.0, n_newton=2,
    )
    case = cantilever_case(nx=24, nz=4, n_steps=12000, load=20000.0, tol=0.0)
    s = solve(case, mat=mat, allow_max_steps=True)
    assert np.isfinite(s.u).all()
    assert s.state[:, 6].max() > 1e-4, "expected plastic yielding at the root"
    assert s.state[:, 6].min() == 0.0, "the tip region should stay elastic"


def test_staged_relaxation_gives_equilibrium_points():
    """Every output frame of a staged run must be a real equilibrium point.

    For a linear elastic cantilever that means the tip deflection is exactly
    proportional to the load level.  A continuous ramp cannot pass this: its
    intermediate frames are still ringing.
    """
    # The 8k/stage budget reaches about 5.6e-5 on this deliberately tiny
    # reduced-integration mesh.  Use a tolerance compatible with that budget;
    # the solver must no longer label budget exhaustion at 1e-6 as convergence.
    case = cantilever_case(n_steps=200000, tol=1e-4)
    case["solver"]["load_steps"] = 5
    case["solver"]["max_steps_per_load"] = 8000
    case["solver"]["stage_ramp_iterations"] = 200


    mesh = box_mesh(case["mesh"]["dimensions"], case["mesh"]["divisions"])
    mat = create("linear_elastic", E=E, nu=NU, rho=RHO)
    s = Solver(case, mesh, mat, log=lambda *_: None)
    frames = []
    s.run(on_output=lambda sol, step, t, sig: frames.append((t, tip_w(sol))),
          output_every=1, progress_every=100)

    assert len(frames) == 5, frames
    levels = np.array([f[0] for f in frames])
    w = np.array([f[1] for f in frames])
    assert np.allclose(levels, [1, 2, 3, 4, 5])
    ratio = w / (levels / 5.0)
    # Consistent with the 1e-4 force-residual acceptance above; tighter
    # proportionality belongs to a larger per-stage budget.
    assert np.abs(ratio / ratio[-1] - 1.0).max() < 5e-3, ratio


def test_staged_relaxation_needs_relaxation_mode():
    # refused by the schema before the solver's own check (a solver's case is
    # validated since v1.19, core/problem.py)
    case = cantilever_case(mode="dynamic", damping={"type": "viscous", "coeff": 0.0})
    case["solver"]["total_time"] = 0.01
    case["solver"]["load_steps"] = 5
    mesh = box_mesh(case["mesh"]["dimensions"], case["mesh"]["divisions"])
    mat = create("linear_elastic", E=E, nu=NU, rho=RHO)
    with pytest.raises(ValueError, match="load_steps holds load levels to equilibrium in a relaxation"):
        Solver(case, mesh, mat, log=lambda *_: None)


def test_a_load_kind_solids_do_not_take_is_refused_not_turned_into_nan():
    """The solid load dispatch ended in a bare `else` that assumed `vector`,
    and `np.asarray(None, dtype=float)` is `nan` -- so a `temperature` load on
    a solid model poisoned the whole force vector instead of being refused.
    It was the only one of the six dispatches with no closing
    sentence."""
    case = cantilever_case(n_steps=1)
    case["loads"] = case["loads"] + [{"region": "tip", "temperature": {"uniform": 50.0}}]
    mesh = box_mesh(case["mesh"]["dimensions"], case["mesh"]["divisions"])
    mat = create("linear_elastic", E=E, nu=NU, rho=RHO, thermal_expansion=1.2e-5)
    with pytest.raises(ValueError, match="do not take a 'temperature' load"):
        Solver(case, mesh, mat, log=lambda *_: None)


def test_the_loads_solids_do_take_are_untouched():
    case = cantilever_case(n_steps=1)
    mesh = box_mesh(case["mesh"]["dimensions"], case["mesh"]["divisions"])
    mat = create("linear_elastic", E=E, nu=NU, rho=RHO)
    s = Solver(case, mesh, mat, log=lambda *_: None)
    total = sum(np.asarray(F).sum() for _, F in s.load_groups)
    assert total == pytest.approx(-TIP_LOAD)
