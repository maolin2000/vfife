"""Every reinforcement field is consumed, or refused, or named with a reason.

THE DEFECT THIS FILE EXISTS FOR
    `ReinforcementSpec.host_deduction` was accepted by the schema, honoured by
    `EmbeddedTruss`, and silently dropped by `BondedTruss` -- the constructor
    was never passed it and the class had no such attribute. So every
    bond-slip model ran with the concrete inside its bars counted twice: the
    solid element integrates over that volume AND the bar adds its own E A.
    The Tanaka column asked for 2.53e10 and got nothing.

    Nothing failed. Two element types consume the same spec, one of them
    stopped reading a field, and there was no place where that could show up.

THE RULE
    A field is CONSUMED (changing it changes the answer), or REFUSED (it
    cannot act on this path and says so), or NAMED below with the reason it is
    neither. What is not allowed is silence.

    The third category is not a loophole as long as it is explicit: `name` is
    a label, and `diameter` on a perfectly bonded bar is a real property of
    the bar that the formulation does not need. Writing that down is the
    opposite of dropping it on the floor.
"""
import numpy as np
import pytest
import yaml

from vfife.api import prepare_case
from vfife.core.solver import Solver
from vfife.io.schema import ReinforcementSpec

BOND = {"type": "bond_ceb", "fc": 25.6e6}

#: Why each field is not checked by a behavioural probe below. Every field of
#: ReinforcementSpec must be either in a probe or in here, and the
#: completeness test at the bottom is what makes that true.
NAMED_EXCEPTIONS = {
    "name": "a label for output and diagnostics; it cannot change an answer",
    "material": "required; a case without it does not validate",
    "lines": "one of three mutually exclusive geometry inputs, covered by "
             "tests/test_bonded.py and tests/test_bar_length_scale.py; a line "
             "entry's own `points`/`closed` route is pinned by "
             "tests/test_path_reinforcement.py, which asserts that a route is "
             "laid as ONE connected bar",
    "nodes": "geometry input, as `lines`",
    "elements": "geometry input, as `lines`",
    "area": "covered by the element tests, which check the bar force and the "
            "bonded surface against area directly",
    "bond": "its presence IS the choice of path; tests/test_bonded_solver.py "
            "asserts the two paths are built differently",
    "diameter": "ACCEPTED, not refused, on a perfectly bonded bar: it is a "
                "real property of the bar that the formulation does not "
                "happen to need, and refusing it would forbid running one "
                "reinforcement spec with and without a `bond` block, which is "
                "how the judge test in tests/test_bonded_solver.py is written",
}


def _case(bond, host_deduction=0.0, n_steps=3000, **rebar):
    """A block in tension with one bar.

    `tolerance` is unreachable on purpose: both runs of a comparison then take
    exactly the same number of iterations from the same start, so a difference
    in the answer is the field under test and cannot be a difference in how
    far each one converged.
    """
    section = {"type": "bar", "material": "steel", "shape": "circle",
               "diameter": rebar.pop("diameter", 0.020), "area": 3.1416e-4}
    assign = {"reinforcement": "bar", "section": "bar20", "name": "bar"}
    if host_deduction:
        assign["host_deduction"] = host_deduction
    if bond is not None:
        assign["bond"] = bond
    assign.update(rebar)                      # transverse_penalty and the like
    return {
        "name": "field_consumption",
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"dimensions": [0.2, 0.2, 1.0], "divisions": [2, 2, 10]},
        "materials": {
            "concrete": {"type": "linear_elastic", "E": 2.53e10, "nu": 0.2, "rho": 2400.0},
            "steel": {"type": "rebar", "E": 2.0e11, "rho": 7850.0,
                      "yield_strength": 4.74e8, "hardening": 1.0e9}},
        "sections": {"body": {"type": "solid", "material": "concrete", "formulation": "b-bar"},
                     "bar20": section},
        "reinforcement": {"bar": {"line": {"start": [0.1, 0.1, 0.0], "end": [0.1, 0.1, 1.0],
                                           "divisions": 20}}},
        "assign": [assign],
        "regions": {
            "base": {"plane": {"point": [0, 0, 0], "normal": [0, 0, 1]}},
            "top": {"plane": {"point": [0, 0, 1.0], "normal": [0, 0, 1]}},
        },
        "constraints": [{"region": "base", "dofs": [1, 1, 1]}],
        "loads": [{"region": "top", "vector": [0.0, 0.0, 2.0e5]}],
        "solver": {"backend": "numpy", "mode": "relaxation",
                   "mass": "underwood", "n_steps": n_steps,
                   "tolerance": 1.0e-30, "damping": {"type": "kinetic"}},
        "output": {"path": "unused.h5", "monitors": {}},
    }


def _tip(tmp_path, case, name):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(case, allow_unicode=True), encoding="utf-8")
    prepared = prepare_case(str(path))
    solver = Solver(prepared.case.model_dump(mode="python"), prepared.mesh,
                    prepared.material, log=lambda *_: None)
    try:
        solver.run()
    except Exception:
        pass
    return float(np.abs(solver.u[:, 2]).max())


# --------------------------------------------------------------------------- #
#  The field that was dropped
# --------------------------------------------------------------------------- #

@pytest.mark.slow
@pytest.mark.parametrize("bond, path", [(None, "perfect"), (BOND, "bonded")])
def test_host_deduction_reaches_both_reinforcement_paths(tmp_path, bond, path):
    """The regression, and the reason this file exists.

    Measured at convergence, tip displacement of the block:

        perfect bond   216.4978 -> 217.6753 um   (+0.54%)
        bonded bar     220.9799 -> 219.4027 um   (-0.71%)

    The two move in opposite directions and that is not asserted -- the bar's
    stiffness, the slip it develops and the load it draws are coupled, so the
    sign is not obvious and nothing here depends on it. What is asserted is
    that the number is READ. Before the fix the bonded row of that table did
    not move at all.
    """
    plain = _tip(tmp_path, _case(bond, host_deduction=0.0), f"{path}0")
    deducted = _tip(tmp_path, _case(bond, host_deduction=2.53e10), f"{path}1")
    change = abs(deducted - plain) / plain
    assert change > 1e-3, (
        f"{path} bond ignored host_deduction: {plain * 1e6:.4f} um either way")


def test_bonded_truss_carries_the_deduction_onto_its_precompute(tmp_path):
    """The cheap half of the check above, without a solve.

    `internal_force` reads `pre.host_deduction`, so this is the single
    attribute the whole thing turns on -- and the one that was left at its
    0.0 default.
    """
    case = _case(BOND, host_deduction=2.53e10)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(case, allow_unicode=True), encoding="utf-8")
    prepared = prepare_case(str(path))
    solver = Solver(prepared.case.model_dump(mode="python"), prepared.mesh,
                    prepared.material, log=lambda *_: None)
    assert solver.bonded[0].pre.host_deduction == pytest.approx(2.53e10)


@pytest.mark.parametrize("bond, path", [(None, "perfect"), (BOND, "bonded")])
def test_prestress_reaches_both_reinforcement_paths(tmp_path, bond, path):
    """`prestress` is lowered to a strain offset on `pre`, which both element
    types read through the same `internal_force`. The probe is the offset
    itself rather than a solve: it is the single attribute the whole feature
    turns on, and it is the attribute `host_deduction` was missing on one of
    these two paths for the life of the bonded element."""
    case = _case(bond, prestress=3.0e8)
    p = tmp_path / f"{path}.yaml"
    p.write_text(yaml.safe_dump(case, allow_unicode=True), encoding="utf-8")
    prepared = prepare_case(str(p))
    solver = Solver(prepared.case.model_dump(mode="python"), prepared.mesh,
                    prepared.material, log=lambda *_: None)
    bar = (solver.bonded if bond else solver.rebar)[0]
    assert bar.pre.prestrain == pytest.approx(3.0e8 / 2.0e11)


def test_the_stability_bound_is_net_of_the_deduction(tmp_path):
    """The bar's row sum has to describe the stiffness the bar actually
    assembles, which is (E - host_deduction) A / L. Leaving the deduction out
    would over-state it -- safe for the time step, but inconsistent with
    `_emb_row_sums`, which has always been net of it."""
    case = _case(BOND, host_deduction=1.0e11)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(case, allow_unicode=True), encoding="utf-8")
    prepared = prepare_case(str(path))
    solver = Solver(prepared.case.model_dump(mode="python"), prepared.mesh,
                    prepared.material, log=lambda *_: None)
    bar = solver.bonded[0]
    axial = (bar.material.E - bar.pre.host_deduction) * bar.area / bar.pre.L0
    assert np.all(bar.bar_row_sums() >= axial.min())
    # and it is genuinely smaller than the undeducted version
    undeducted = bar.material.E * bar.area / bar.pre.L0
    assert axial.max() < undeducted.max()


# --------------------------------------------------------------------------- #
#  Fields that cannot act, and say so
# --------------------------------------------------------------------------- #

def test_transverse_penalty_without_a_bond_block_is_refused(tmp_path):
    """It restrains slip ACROSS the bar. A perfectly bonded bar has no slip,
    so nothing would read the number -- and a number that was read and a
    number that was ignored look identical from the outside."""
    case = _case(None, transverse_penalty=1.0e9)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(case, allow_unicode=True), encoding="utf-8")
    # refused where the assignment is read, before any solver exists
    with pytest.raises(ValueError, match="transverse_penalty"):
        prepare_case(str(path))


def test_transverse_penalty_is_consumed_when_there_IS_a_bond_block(tmp_path):
    """The other half: refusing it on one path only counts for something if it
    does its job on the other."""
    case = _case(BOND, transverse_penalty=5.0e8)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(case, allow_unicode=True), encoding="utf-8")
    prepared = prepare_case(str(path))
    solver = Solver(prepared.case.model_dump(mode="python"), prepared.mesh,
                    prepared.material, log=lambda *_: None)
    assert solver.bonded[0].k_trans == pytest.approx(5.0e8)


def test_diameter_is_consumed_by_the_bonded_path(tmp_path):
    """Bond acts on the bar SURFACE, so the diameter sets the area every bond
    stress is integrated over. Doubling it must double that area."""
    areas = []
    for dia in (0.020, 0.040):
        case = _case(BOND, diameter=dia)
        path = tmp_path / f"d{dia}.yaml"
        path.write_text(yaml.safe_dump(case, allow_unicode=True),
                        encoding="utf-8")
        prepared = prepare_case(str(path))
        solver = Solver(prepared.case.model_dump(mode="python"), prepared.mesh,
                        prepared.material, log=lambda *_: None)
        areas.append(float(solver.bonded[0].bond_area.sum()))
    assert areas[1] == pytest.approx(2.0 * areas[0], rel=1e-12)


# --------------------------------------------------------------------------- #
#  Completeness
# --------------------------------------------------------------------------- #

def test_every_reinforcement_field_is_consumed_or_named():
    """A new field on `ReinforcementSpec` cannot be added without deciding.

    The probes above cover `host_deduction`, `diameter` and
    `transverse_penalty`; everything else has to be listed with a reason. The
    list is a human declaration and this test does not verify the reasons --
    what it makes impossible is adding a field to the schema, wiring it into
    one of the two element paths, and having nothing anywhere notice that the
    other path ignores it. That is exactly what happened.
    """
    probed = {"host_deduction", "diameter", "transverse_penalty", "prestress",
              # tests/test_tendon_friction.py: the profile reaches every
              # element's prestrain, and is refused without `prestress`, on a
              # split tendon and on a closed hoop
              "friction",
              # tests/test_bonded_anchorage.py: a tied end carries the host
              # strain, a free one does not, and it is refused without `bond`
              "anchorage"}
    declared = set(ReinforcementSpec.model_fields)
    unaccounted = declared - probed - set(NAMED_EXCEPTIONS)
    assert not unaccounted, (
        f"reinforcement fields with no consumer and no reason: "
        f"{sorted(unaccounted)}. Add a probe above, or a line in "
        "NAMED_EXCEPTIONS saying why it cannot be probed.")
    stale = set(NAMED_EXCEPTIONS) - declared
    assert not stale, f"NAMED_EXCEPTIONS mentions fields that no longer exist: {sorted(stale)}"
