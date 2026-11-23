"""`host_deduction` on a cracking host is refused, and here is what it did.

WHY
    `host_deduction` subtracts E_c * eps * A_s from an embedded or bonded bar
    so the concrete volume the bar displaces is not counted twice. That is
    right while the concrete carries E_c * eps and wrong once it has cracked:
    the bar is then short by E_c * eps * A_s that nobody else carries -- 15%
    of its force at yield in case_12 -- and past yield its net tangent is
    E_hard - E_c = 7.68 - 29.8 = -22 GPa, a bar that LOSES force as it
    hardens.

    elements/truss.py said so from the day the option was written: "leave it
    off when the concrete is expected to damage". case_12 copied the setting
    from the README's perfect-bond example into a beam whose whole point was
    to crack, and six stages of work chased the ceiling it
    built -- immune to erosion, damping, load steps, hardening, the damage
    cap, shear retention, a ductile compression curve, the bond law, the
    yield surface and viscosity, because none of those was the bar's
    negative tangent.

    It was found in one minute by the smallest thing that could show it:
    the composite tension test below. An embedded bar in a CDP prism, pulled
    in displacement control: with the deduction the total force fell from
    80 kN at bar yield to 2 kN; without it the force keeps rising.

    A sentence in a docstring is not a guard. The solver now refuses the
    combination, and this file pins both the refusal and the physics it
    protects.
"""
import numpy as np
import pytest
import yaml

from vfife.api import prepare_case
from vfife.core.diagnostics import UserError
from vfife.core.solver import Solver

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

EC = 4500.0 * 44.0 ** 0.5 * 1e6


def _prism(tmp_path, host_deduction, host="cdp", steps=12):
    L = 0.5
    material = ({"type": "linear_elastic", "E": EC, "nu": 0.2, "rho": 2400.0}
                if host == "linear" else
                {"type": "cdp", "E": EC, "nu": 0.2, "rho": 2400.0,
                 "fb0_fc0": 1.16, "Kc": 0.6667, "dilation_angle": 36.0,
                 "eccentricity": 0.1, "wt": 0.0, "wc": 1.0, "viscosity": 1.0,
                 "n_return": 5, "crack_band": "both", "l_ref": "auto",
                 "damage_max": 0.80,
                 "compression": {"source": "gb50010", "fc": 44e6,
                                 "beta_c": 0.7, "sigma_c0_ratio": 0.4},
                 "tension": {"source": "fracture_energy", "ft": 3.42e6,
                             "Gf": 140.0, "beta_t": 0.9}})
    steel = {"type": "rebar", "E": 192.0e9, "rho": 7850.0,
             "yield_strength": 449.0e6, "hardening": 8.0e9, "kinematic_fraction": 0.5}
    assign = {"reinforcement": "bar", "section": "bar16", "name": "bar",
              "segment_length": 0.025}
    if host_deduction:
        assign["host_deduction"] = host_deduction
    case = {
        "name": "ct", "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"dimensions": [L, 0.1, 0.1], "divisions": [6, 2, 2]},
        "materials": {"host": material, "steel": steel},
        "sections": {
            "body": {"type": "solid", "material": "host", "formulation": "b-bar"},
            "bar16": {"type": "bar", "material": "steel", "shape": "circle",
                      "diameter": 0.016, "area": 200.0e-6}},
        "reinforcement": {"bar": {"line": {"start": [0.0, 0.05, 0.05], "end": [L, 0.05, 0.05]}}},
        "assign": [assign],
        "regions": {
            "left": {"box": [-0.001, 0.001, -0.01, 0.11, -0.01, 0.11]},
            "right": {"box": [L - 0.001, L + 0.001, -0.01, 0.11, -0.01, 0.11]},
            "A": {"box": [-0.001, 0.001, 0.04, 0.06, -0.001, 0.001]},
            "Bn": {"box": [-0.001, 0.001, 0.04, 0.06, 0.099, 0.101]}},
        "constraints": [{"region": "left", "dofs": [1, 0, 0]},
                        {"region": "A", "dofs": [0, 1, 1]},
                        {"region": "Bn", "dofs": [0, 1, 0]}],
        "steps": [{"name": "pull",
                   "constraints": [{"region": "right", "dofs": [1, 0, 0],
                                    "value": 0.006 * L}],
                   "load_steps": steps, "max_steps_per_load": 20000,
                   "tolerance": 1.0e-3}],
        "solver": {"backend": "numpy", "mode": "relaxation",
                   "mass": "underwood", "n_steps": steps * 20000 + 10,
                   "tolerance": 1.0e-3, "relaxation_safety": 0.3,
                   "damping": {"type": "kinetic"}},
        "output": {"path": str(tmp_path / "ct.h5"), "every": 500,
                   "monitors": {"R": {"kind": "reaction", "region": "left",
                                      "dof": "x"}}},
    }
    path = tmp_path / "ct.yaml"
    path.write_text(yaml.safe_dump(case, allow_unicode=True), encoding="utf-8")
    prepared = prepare_case(str(path))
    return Solver(prepared.case.model_dump(mode="python"), prepared.mesh,
                  prepared.material, log=lambda *_: None)


def test_host_deduction_with_a_damaging_host_is_refused(tmp_path):
    with pytest.raises(UserError) as exc:
        _prism(tmp_path, host_deduction=EC)
    assert exc.value.code == "HOST_DEDUCTION_DAMAGING"


def test_host_deduction_with_an_elastic_host_is_still_allowed(tmp_path):
    """The option exists for exactly this: elastic work at high rho, where
    tests/test_rebar.py shows it recovering the transformed section."""
    _prism(tmp_path, host_deduction=EC, host="linear", steps=2)


@pytest.mark.slow
def test_an_embedded_bar_in_a_cracking_prism_gains_force_after_yield(tmp_path):
    """The composite tension test that found it, kept as the physics guard.

    The bar yields at eps = 449/192000 = 2.3e-3. From there to 6e-3 the bar
    alone gains 200 mm2 * 7.68 GPa * 3.7e-3 = 5.7 kN; the cracked concrete's
    residual (the damage cap's 20% floor) loses a little. The composite must
    still RISE. With the deduction it fell to a fortieth of the bar force.
    """
    solver = _prism(tmp_path, host_deduction=0.0)
    solver.run(on_progress=lambda *_: None, progress_every=2000)
    R = np.abs(np.array(solver.monitors["R"].values))
    eps = 0.006 * np.arange(1, len(R) + 1) / 12
    at_yield = R[np.searchsorted(eps, 0.0025)]
    at_end = R[-1]
    assert at_end > at_yield, (
        f"composite force fell from {at_yield/1e3:.1f} to {at_end/1e3:.1f} kN "
        "after bar yield")
    assert at_end > 0.9 * 200e-6 * (449e6 + 7.68e9 * (0.006 - 449e6 / 192e9)), (
        "the composite carries less than the bar alone could")
