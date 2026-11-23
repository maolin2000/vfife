"""The YAML in README.md has to still be a valid case.

WHY
    Documentation rots silently and in exactly one direction: the code moves,
    the example stays, and the first person to trust it loses an afternoon
    finding out that a key was renamed. Everything else in this suite checks
    that the code does what it says; this checks that what it SAYS is still
    something the code accepts.

    It is not hypothetical here. The README's own limitations section carried
    "Perfect bond only -- bond-slip needs the bar node back as an independent
    DOF plus an interface element" for as long as it took to build exactly
    that, and a reader would have concluded the feature did not exist.

WHAT IS CHECKED, AND WHAT IS NOT
    That the blocks PARSE and BUILD -- the schema accepts them and a solver
    can be constructed. Not that they solve to anything: a README fragment is
    not a verification case, and the eleven that are live in `cases/`.
"""
import pathlib
import re

import pytest
import yaml

from vfife.api import prepare_case
from vfife.core.solver import Solver

#: the long description, where the examples below live: the README until
#: 1.12, `docs/technical_notes.md` since (the README is the front page now,
#: and `tests/test_front_page.py` holds it to its own examples)
README = pathlib.Path("docs/technical_notes.md")


def _yaml_blocks(text):
    return re.findall(r"```yaml\n(.*?)```", text, flags=re.S)


#: the beam the perfect-bond example was written against: bars at y = -0.15,
#: z = 0.06..0.94, running x = 0.05..11.95
BEAM_MESH = {"dimensions": [12.0, 0.3, 1.0], "divisions": [12, 1, 2],
             "origin": [0.0, -0.3, 0.0]}


def _wrap(fragment, mesh=None):
    """The smallest complete case a README fragment can live in.

    A fragment carries `sections` / `reinforcement` / `assign` (and possibly
    `materials`); this adds the mesh, a host section and material, and the
    solver. The mesh has to COVER the bars -- `locate_in_blocks` refuses a bar
    point outside the host, which is the right refusal and means each example
    needs the geometry it was written for. The bond-slip block is a 1.9 m
    column, the perfect-bond block a 12 m beam.
    """
    case = {
        "name": "readme", "units": {"length": "m", "force": "N",
                                    "stress": "Pa"},
        "mesh": mesh or {"dimensions": [0.4, 0.4, 1.9],
                         "divisions": [2, 2, 8]},
        "materials": {"concrete": {"type": "linear_elastic", "E": 2.53e10, "nu": 0.2,
                                   "rho": 2400.0},
                      "steel": {"type": "rebar", "E": 2.0e11, "rho": 7850.0,
                                "yield_strength": 4.74e8, "hardening": 1.0e9}},
        "sections": {"body": {"type": "solid", "material": "concrete", "formulation": "b-bar"}},
        "regions": {"base": {"plane": {"point": [0, 0, 0],
                                       "normal": [0, 0, 1]}}},
        "constraints": [{"region": "base", "dofs": [1, 1, 1]}],
        "loads": [{"region": "base", "vector": [0.0, 0.0, 0.0]}],
        "solver": {"backend": "numpy", "mode": "relaxation",
                   "mass": "underwood", "n_steps": 10},
        "output": {"path": "unused.h5", "monitors": {}},
    }
    case["materials"].update(fragment.get("materials", {}))
    case["sections"].update(fragment["sections"])
    case["reinforcement"] = fragment["reinforcement"]
    case["assign"] = list(fragment["assign"])
    return case


def _build(case, tmp_path, name):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(case, allow_unicode=True), encoding="utf-8")
    prepared = prepare_case(str(path))
    return Solver(prepared.case.model_dump(mode="python"), prepared.mesh,
                  prepared.material, log=lambda *_: None)


def test_the_readme_is_where_this_test_thinks_it_is():
    """A missing file would make every parametrised test below collect zero
    cases and pass, which is the standard way a static check stops checking."""
    assert README.is_file(), "run pytest from the repository root"
    assert len(_yaml_blocks(README.read_text(encoding="utf-8"))) >= 2


def test_the_model_example_is_a_complete_case_that_loads():
    """The block under 'The model: materials, sections, assign' is case_12's
    head. It must validate as a case on its own -- the README is the first
    thing a new user copies from."""
    from vfife.io import schema
    text = README.read_text(encoding="utf-8")
    block = next(b for b in _yaml_blocks(text) if "sections:" in b and "hoop:" in b)
    d = yaml.safe_load(block)
    d["solver"] = {"backend": "numpy", "mode": "relaxation", "n_steps": 10}
    case = schema.from_dict(d)
    assert case.material["type"] == "cdp"
    assert [r.name for r in case.reinforcement] == ["tension", "hangers", "stirrups"]


def test_the_bond_slip_example_builds_the_thing_it_describes(tmp_path):
    """The block under 'Bond-slip: the bar gets its own degrees of freedom',
    with the numbers the prose quotes around it.

    `segment_length: 0.1` on a 1.9 m bar is 19 segments and 20 nodes -- the
    section says the length is what you should specify and the count is what
    invites the mistake, so the count is what this asserts.
    """
    text = README.read_text(encoding="utf-8")
    block = next(b for b in _yaml_blocks(text) if "bond:" in b and "longitudinal" in b)
    fragment = yaml.safe_load(block)

    solver = _build(_wrap(fragment), tmp_path, "bond")
    assert solver.bonded and not solver.rebar
    bar = solver.bonded[0]
    assert bar.n_elem == 19 and bar.n_bar_nodes == 20

    # the default the prose names: the bar's own E*A over the tributary length
    k_bar = bar.material.E * bar.area / bar.trib
    assert bar.k_trans == pytest.approx(float(k_bar.mean()), rel=1e-9)


def test_the_perfect_bond_example_still_builds(tmp_path):
    """The older block, with the stirrup `repeat`. It has no `bond`, so it
    must go down the embedded path and stay there.

    It used to carry `host_deduction` too, and this test asserted the value
    arrived. The example is a beam with yielding steel -- one that cracks --
    and the deduction is exactly wrong there (tests/test_host_deduction_guard.py
    has the measurement). The README no longer shows it in that block, and
    the solver would refuse it with a cracking host; this test now pins that
    the example does NOT carry it."""
    text = README.read_text(encoding="utf-8")
    block = next(b for b in _yaml_blocks(text)
                 if "repeat:" in b and "bond:" not in b and "bottom" in b)
    assert "host_deduction" not in block, (
        "the perfect-bond example shows host_deduction on a beam that yields")
    fragment = yaml.safe_load(block)

    solver = _build(_wrap(fragment, BEAM_MESH), tmp_path, "perfect")
    assert solver.rebar and not solver.bonded
    assert solver.rebar[0].pre.host_deduction == 0.0
    # 4.909e-4 derived from the 25 mm diameter, as the comment says
    assert solver.rebar[0].pre.area[0] == pytest.approx(4.909e-4, rel=2e-3)


def test_the_readme_no_longer_claims_perfect_bond_is_the_only_option():
    """The specific stale sentence, named so it cannot come back by accident.

    It described bond-slip precisely -- 'the bar node back as an independent
    DOF plus an interface element' -- and stayed in the limitations section
    after exactly that was built.
    """
    text = README.read_text(encoding="utf-8")
    assert "Perfect bond only" not in text
    assert "bond_cyclic" in text, "the cyclic law is not mentioned anywhere"


def test_a_dynamic_run_is_told_to_name_its_damping():
    """The README sends the reader to `mode: dynamic` for near-incompressible
    problems. Without an explicit `damping`, that used to mean kinetic damping
    -- a device for finding a STATIC answer, which removes the dynamics the
    run exists to have."""
    text = README.read_text(encoding="utf-8")
    assert "damping: {type: none}" in text or "type: none" in text
