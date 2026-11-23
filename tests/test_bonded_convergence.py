"""A bonded bar's own dofs are part of the state the solver judges.

WHY
    The convergence residual was `norm(F_un[free])` over HOST dofs, and the
    kinetic energy that drives kinetic damping was `sum(m v^2)` over HOST dofs.
    A bonded bar's nodes are free dofs too -- that is the whole reason it has
    them -- and neither number saw them. So the host could settle against a
    bar that was still moving and the load step was declared converged
    mid-transient.

    It was found on case_12b (the Alberta beam with bond-slip). The first load
    step reported 37.1 kN with a 71.7% left-right reaction split on a symmetric
    beam. Run to a real equilibrium the same step is 66.8 kN and 1.1%. Every
    later step inherited the unfinished one, and the run went non-finite at
    the fifth -- which was written up, wrongly, as a stability-bound defect.
    Cutting the time step by three made it fail EARLIER, which is how that
    story was ruled out.

THE TEST
    Fix every host dof, so the host residual is identically zero and the host
    kinetic energy identically zero, then start the bar displaced along its
    axis. The only thing that can be out of equilibrium is the bar. The old
    code declared convergence at the first check and left the bar where it
    was; the solver now has to run until the bar has slid back.
"""
import numpy as np
import pytest
import yaml

from vfife.api import prepare_case
from vfife.core.solver import Solver


def _fixed_host_with_one_bar(tmp_path, offset):
    case = {
        "name": "bar_alone", "units": {"length": "m", "force": "N",
                                       "stress": "Pa"},
        "mesh": {"dimensions": [0.2, 0.2, 1.0], "divisions": [2, 2, 5]},
        "materials": {
            "concrete": {"type": "linear_elastic", "E": 3.0e10, "nu": 0.2, "rho": 2400.0},
            "steel": {"type": "rebar", "E": 2.0e11, "rho": 7850.0,
                      "yield_strength": 4.5e8, "hardening": 2.0e9},
        },
        "sections": {
            "body": {"type": "solid", "material": "concrete", "formulation": "b-bar"},
            "bar16": {"type": "bar", "material": "steel", "shape": "circle",
                      "diameter": 0.016, "area": 2.0e-4},
        },
        "reinforcement": {"bar": {"line": {"start": [0.1, 0.1, 0.05], "end": [0.1, 0.1, 0.95],
                                           "divisions": 6}}},
        "assign": [{"reinforcement": "bar", "section": "bar16", "name": "bar",
                    "bond": {"type": "bond_ceb", "fc": 40.0e6}}],
        "regions": {"all": {"box": [-1.0, 1.0, -1.0, 1.0, -1.0, 2.0]}},
        "constraints": [{"region": "all", "dofs": [1, 1, 1]}],
        # a load on FIXED dofs does nothing to the mechanics; it is here so
        # the residual has a finite scale to be relative to. This system
        # relaxes to a zero-force state, and 0/0 is not a convergence test.
        "loads": [{"region": "all", "vector": [0.0, 0.0, 1.0e3]}],
        "solver": {"backend": "numpy", "mode": "relaxation",
                   "mass": "underwood", "n_steps": 40000, "tolerance": 1.0e-4,
                   "relaxation_safety": 0.3, "damping": {"type": "kinetic"}},
        "output": {"path": str(tmp_path / "unused.h5"), "monitors": {}},
    }
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(case, allow_unicode=True), encoding="utf-8")
    prepared = prepare_case(str(path))
    solver = Solver(prepared.case.model_dump(mode="python"), prepared.mesh,
                    prepared.material, log=lambda *_: None)
    assert len(solver.bonded) == 1
    bar = solver.bonded[0]
    bar.u_bar[:, 2] += offset
    return solver, bar


def test_a_displaced_bar_in_a_rigid_host_is_not_converged_until_it_has_slid_back(tmp_path):
    """Host residual is zero by construction here. If the bar's residual is
    not in the norm, this converges at the first check with the slip intact."""
    offset = 2.0e-5                       # 20 um, well inside the bond's rise
    solver, bar = _fixed_host_with_one_bar(tmp_path, offset)
    solver.run(on_progress=lambda *_: None, progress_every=200)

    _, s_axial = bar.slip(solver.u)
    assert np.abs(s_axial).max() < 0.02 * offset, (
        f"the bar is still displaced by {np.abs(s_axial).max():.3e} m of a "
        f"{offset:.1e} m start: the host converged and the bar was not asked")
    assert solver.diagnostics["converged"]


def test_the_kinetic_energy_sees_the_bar(tmp_path):
    """With the host pinned, ALL the kinetic energy is the bar's. Kinetic
    damping resets on the peak of that energy; with the bar excluded the
    energy is identically zero and there is never a reset."""
    solver, bar = _fixed_host_with_one_bar(tmp_path, 2.0e-5)
    solver.run(on_progress=lambda *_: None, progress_every=200)
    assert solver.n_resets > 0, "kinetic damping never fired: the bar's motion is invisible to it"
    assert max(solver.kinetic) > 0.0
