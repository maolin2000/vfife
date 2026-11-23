"""A prescribed displacement blends in from where the node already is.

THE DEFECT, AND WHY IT TOOK FOURTEEN CASES TO SHOW UP
    A prescribed level is ABSOLUTE: `value: -0.0564` means -56.4 mm, not
    56.4 mm further down. The staging inside a step already ramps INTO each
    load level for exactly that reason -- the comment in `Solver.run` records
    that 4 mm imposed in one iteration diverged to 1e100. What it could not
    cover is a step that STARTS somewhere: the first level's ramp begins at
    `0/load_steps`, so a node an earlier step left at +3.57 mm is dragged to
    zero in a single iteration.

    Every case in this project until the post-tensioned T-beam either
    prescribed a displacement in its FIRST step, from rest, or prescribed one
    on nodes nothing had moved. cases/case_15 is the first with a transfer
    step that cambers the bearings 3.6 mm before the jacks touch the beam, and
    its loading step reached a kinetic energy of 1e269 within 200 iterations.

    `apply_bc(..., approach)` blends the target in from the node's position at
    the start of the step, over the same `stage_ramp` iterations the level
    ramp uses. After those iterations it is exactly the old behaviour, and for
    a step whose prescribed nodes start at zero -- which is every other case
    here -- it is the old behaviour throughout. That is why the fix is a blend
    and not a change of what `value` means.
"""
import numpy as np
import pytest
import yaml

from vfife.api import prepare_case
from vfife.core.solver import Solver


def _case(tmp_path, target, n_steps=4000):
    """A block hung on gravity, then pushed to an absolute level.

    Step 1 moves the top face down under its own weight; step 2 prescribes a
    displacement there. The prescribed nodes therefore do NOT start at zero,
    which is the whole point.
    """
    case = {
        "name": "moved", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"dimensions": [0.4, 0.2, 0.2], "divisions": [4, 2, 2]},
        "materials": {"m": {"type": "linear_elastic", "E": 3.0e10, "nu": 0.2,
                            "rho": 2400.0}},
        "sections": {"body": {"type": "solid", "material": "m",
                              "formulation": "b-bar"}},
        "assign": [{"section": "body", "region": "all"}],
        "regions": {
            "left": {"box": [-1e-3, 1e-3, -0.01, 0.21, -0.01, 0.21]},
            "right": {"box": [0.399, 0.401, -0.01, 0.21, -0.01, 0.21]},
        },
        "constraints": [{"region": "left", "dofs": [1, 1, 1]}],
        "loads": [{"region": "all", "gravity": [0.0, 0.0, -9.81]}],
        "steps": [
            {"name": "weight", "load_steps": 1, "max_steps_per_load": n_steps,
             "tolerance": 1.0e-8},
            {"name": "push", "load_steps": 2, "max_steps_per_load": n_steps,
             "tolerance": 1.0e-8,
             "constraints": [{"region": "right", "dofs": [0, 0, 1],
                              "value": target}]},
        ],
        "solver": {"backend": "numpy", "mode": "relaxation", "mass": "underwood",
                   "n_steps": 4 * n_steps, "tolerance": 1.0e-8,
                   "relaxation_safety": 0.3, "damping": {"type": "kinetic"}},
        "output": {"path": str(tmp_path / "m.h5"), "every": 100000,
                   "monitors": {"tip": {"kind": "displacement", "region": "right",
                                        "dof": "z"}}},
    }
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(case), encoding="utf-8")
    prepared = prepare_case(str(p))
    return Solver(prepared.case.model_dump(mode="python"), prepared.mesh,
                  prepared.material, log=lambda *_: None)


# --------------------------------------------------------------------------- #
#  The mechanism
# --------------------------------------------------------------------------- #

def test_the_blend_starts_from_the_position_the_step_begins_at(tmp_path):
    """`approach` 0 holds the node still, 1 is the raw absolute target, and
    the way between them is a straight line. This is the whole fix, and it is
    asserted here rather than inferred from a run that did not diverge."""
    s = _case(tmp_path, -0.002)
    s.u[:] = 0.0
    right = s.mesh.regions["right"]
    s.u[right, 2] = 0.0035                      # where an earlier step left it
    s.apply_step(1)                             # the step records that position

    u = np.zeros_like(s.u)
    u[right, 2] = 0.0035
    s.apply_bc(u, 1.0, approach=0.0)
    assert u[right, 2] == pytest.approx(0.0035), "approach 0 must not move it"

    u[right, 2] = 0.0035
    s.apply_bc(u, 1.0, approach=1.0)
    assert u[right, 2] == pytest.approx(-0.002), "approach 1 is the target"

    u[right, 2] = 0.0035
    s.apply_bc(u, 1.0, approach=0.5)
    assert u[right, 2] == pytest.approx(0.5 * (0.0035 - 0.002))


def test_a_step_whose_nodes_start_at_zero_is_untouched(tmp_path):
    """Every case in this project before case_15. The blend has to be exactly
    the old behaviour there, or the fix is a change of meaning rather than a
    shock absorber."""
    s = _case(tmp_path, -0.002)
    s.u[:] = 0.0
    s.apply_step(1)
    right = s.mesh.regions["right"]
    for approach in (0.0, 0.25, 1.0):
        u = np.zeros_like(s.u)
        s.apply_bc(u, 1.0, approach=approach)
        assert u[right, 2] == pytest.approx(-0.002 * approach if approach < 1.0
                                            else -0.002)


def test_the_blend_is_recorded_per_prescribed_block(tmp_path):
    s = _case(tmp_path, -0.002)
    s.u[:] = 0.0
    s.u[s.mesh.regions["right"], 2] = 0.0035
    s.apply_step(1)
    assert len(s._bc_start) == len(s.prescribed)
    assert np.allclose(s._bc_start[0], 0.0035)


# --------------------------------------------------------------------------- #
#  The run it was found by
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_a_second_step_can_prescribe_a_displacement_on_moved_nodes(tmp_path):
    """The behavioural half: the block sags under its own weight, then the
    right face is pushed to an absolute level 5.5 mm away from where it sat.
    Before the blend this diverged; the assertion is that it arrives, and
    arrives at the level asked for."""
    s = _case(tmp_path, -0.002, n_steps=60000)
    s.run(on_progress=lambda *_: None, progress_every=10 ** 9)

    tip = np.array(s.monitors["tip"].values)
    assert np.isfinite(tip).all()
    assert tip[0] < 0.0, "the block should sag under gravity in step 1"
    assert tip[-1] == pytest.approx(-0.002, abs=1e-9), \
        "the prescribed level is absolute and has to be reached exactly"
    # and nothing ran away on the road there
    assert np.abs(tip).max() < 0.01


# --------------------------------------------------------------------------- #
#  BOTH paths, because the last defect of this shape was one path fixed
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_the_device_loop_blends_the_same_way_the_cpu_loop_does(tmp_path):
    """The warp backend keeps the whole time loop -- integrator AND boundary
    conditions -- on the device, so `apply_bc` is not the only place a
    prescribed displacement is written. Fixing the shock on the CPU and not on
    the device would leave a GPU run diverging where the CPU one no longer
    does, and the divergence takes an hour of card time to discover.

    That is exactly the shape of `host_deduction` reaching one reinforcement
    path and not the other (tests/test_reinforcement_fields.py), so it gets
    the same treatment: the two paths are required to agree, on a case whose
    prescribed nodes DO start somewhere.
    """
    pytest.importorskip("warp", reason="warp-lang not installed")
    out = {}
    for backend in ("numpy", "warp"):
        d = tmp_path / backend
        d.mkdir(parents=True, exist_ok=True)
        s = _case(d, -0.002, n_steps=8000)
        # a solver's case is changed by making a new one (core/problem.py)
        extra = {"device": "cpu", "precision": "fp64"} if backend == "warp" else {}
        s.p = s.p.derive(solver={**s.p["solver"], "backend": backend, **extra})
        s.run(on_progress=lambda *_: None, progress_every=10 ** 9)
        out[backend] = np.array(s.monitors["tip"].values)

    a, b = out["numpy"], out["warp"]
    assert a.shape == b.shape
    assert np.isfinite(b).all(), "the device loop diverged where the CPU one did not"
    scale = max(np.abs(a).max(), 1e-30)
    assert np.abs(a - b).max() / scale < 1e-6, (
        f"device loop differs by {np.abs(a - b).max() / scale:.2e} of the "
        "displacement; the blend has to reach both paths")
