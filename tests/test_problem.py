"""The internal case a solver is built from (vfife/core/problem.py,
v1.19, P2 of the architecture review): typed and as its dict, the
two the same data; read-only; a composite solver's halves derived in both
forms at once."""
import copy
import os
import re

import pytest

from vfife.core.problem import Problem
from vfife.io import schema

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = sorted(f[:-5] for f in os.listdir(os.path.join(ROOT, "cases"))
               if f.startswith("case_") and f.endswith(".yaml"))


def _typed_as_dict(problem):
    """The typed case as the dict a solver reads: a static run's solver
    section says relaxation and `static_run` (cli._with_element)."""
    d = problem.case.as_dict()
    if problem.static_run:
        d["solver"] = dict(d["solver"], static_run=True)
    return d


def _identical(a, b, where="case"):
    """Equal to the type of every value (1, 1.0 and True compare equal and
    are three different numpy dtypes)."""
    assert type(a) is type(b), f"{where}: {type(a).__name__} {a!r} against {type(b).__name__} {b!r}"
    if isinstance(a, dict):
        assert list(a) == list(b), f"{where}: keys {list(a)} against {list(b)}"
        for k in a:
            _identical(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{where}: {len(a)} items against {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _identical(x, y, f"{where}[{i}]")
    else:
        assert a == b or (a != a and b != b), f"{where}: {a!r} against {b!r}"


@pytest.mark.parametrize("stem", CASES)
def test_every_shipped_case_is_the_same_data_typed_and_as_its_dict(stem):
    from vfife.cli import _with_element, build
    path = os.path.join(ROOT, "cases", stem + ".yaml")
    case, _raw = schema.load(path)
    _mesh, _material, element = build(case, path)
    d = _with_element(case, element, path)
    p = Problem.of(d)
    _identical(_typed_as_dict(p), d)
    assert p.element == element and p.static_run == (case.solver.mode == "static")
    assert p.solver.mode == d["solver"]["mode"]
    assert p.given == frozenset(d)


def _beam(**extra):
    d = {"element": {"type": "mixed_line", "formulation": "full"},
         "solver": {"mode": "relaxation", "backend": "numpy", "n_steps": 10},
         "members": [{"region": "all", "type": "beam2", "area": 1e-2, "iy": 1e-4, "iz": 1e-4, "j": 2e-4,
                      "orientation": [0.0, 0.0, 1.0]}]}
    d.update(extra)
    return d


def test_a_hand_built_dict_is_validated_and_keeps_what_it_did_not_say():
    d = _beam()
    p = Problem.of(d)
    assert p.given == {"element", "solver", "members"}
    assert "mesh" not in p and "material" not in p          # the dict as it was
    assert p.solver.backend == "numpy" and p.solver.tolerance == schema.SolverSpec().tolerance
    assert Problem.of(p) is p
    with pytest.raises(ValueError):
        Problem.of(_beam(solver={"mode": "relaxation", "n_stepz": 10}))


def test_the_dict_form_is_read_only_and_a_copy_is_to_change():
    p = Problem.of(_beam())
    with pytest.raises(TypeError):
        p["loads"] = []
    plain = p.plain()
    plain["solver"]["n_steps"] = 99
    assert p["solver"]["n_steps"] == 10
    assert copy.deepcopy(p)["solver"] == p["solver"]


def test_a_static_run_is_typed_as_the_static_case_it_was():
    d = _beam(solver={"mode": "relaxation", "static_run": True, "backend": "numpy",
                      "n_steps": 1, "load_steps": 10})
    p = Problem.of(d)
    assert p.static_run and p.solver.mode == "relaxation"
    assert "static_run" not in p.case.solver.model_fields_set
    assert _typed_as_dict(p) == {**p.case.as_dict(), "solver": dict(p.case.as_dict()["solver"], static_run=True)}


def test_a_static_run_s_solver_derived_is_validated_as_static():
    # a relaxation refuses alpha under viscous damping; a static run's own
    # settings, derived for a half with another backend, are the static case's
    d = _beam(solver={"mode": "relaxation", "static_run": True, "backend": "numpy", "n_steps": 1,
                      "load_steps": 10, "damping": {"type": "viscous", "alpha": 3.0}})
    p = Problem.of(d)
    q = p.derive(solver={**p["solver"], "backend": "numba"})
    assert q.static_run and q.solver.mode == "relaxation" and q.solver.backend == "numba"


#: where a composite solver's half is told less in the dict than typed
#: (core/problem.py): the parts that may differ
HALVES = {"coupled": [("solid", {"output"})], "shell_line": [("line", {"output", "element"})], "connector2": [(None, {"members"})]}


@pytest.mark.parametrize("kind", sorted(HALVES))
def test_a_composite_solver_s_halves_are_the_same_data_but_where_listed(kind):
    from vfife.testing import models
    s = models.make(models.build(kind))
    for attr, may_differ in HALVES[kind]:
        half = getattr(s, attr) if attr else s
        typed = _typed_as_dict(half.p)
        for key in half.p:
            if key not in may_differ:
                _identical(typed[key], half.p[key], f"{kind}.{attr}.{key}")


def test_a_derived_half_changes_both_forms():
    p = Problem.of(_beam(loads=[{"region": "all", "vector": [0.0, 0.0, -1.0]}]))
    q = p.derive(loads=[], output={"monitors": {}})
    assert q["loads"] == [] and q.case.loads == [] and p.case.loads
    plain = p.plain()
    plain["element"] = {"type": "hexa8"}
    r = p.derived(plain)
    assert r.element == "hexa8" and r["element"] == plain["element"]
    assert r.case.members == p.case.members
    # the connector model's members are the element's own settings: the
    # typed part keeps what a member group has
    props = dict(p["members"][0], formulation="full", thickness=None)
    s = p.derive(members=[props])
    assert s["members"][0]["formulation"] == "full" and s.case.members[0].region == "all"


#: reads of the case as a dict (`self.p["solver"]`, `problem.get('loads')`),
#: which P2 moves to the typed case; none may be added. Read
#: on the names a solver's case goes by, in vfife/core and the drivers; a
#: write (`p["element"] = ...`, on the dict `_with_element` builds) is not one
DICT_READS = re.compile(r"\b(?:self\.p|solver\.p|s\.p|problem|prob)(?:\[\s*[\"']|\.get\(\s*[\"'])("
                        + "|".join(schema.Case.model_fields) + r")[\"'](?!\s*\]\s*=[^=])")
DICT_READS_LEFT = 66
#: the drivers that EDIT an input case's plain dict between the solvers they
#: build on it (formfind; `Problem` keeps the very dict it is given), or before
#: any solver holds it (the test fixtures): their reads are of that dict as
#: the input it is, not of a solver's typed case
DICT_EDITORS = {"formfind.py", os.path.join("testing", "models.py")}

#: WHY THE REST CAN STAY. The dict a solver reads is the
#: validated case dumped whole (`schema.Case.as_dict`, every field with the
#: schema's value -- `test_every_shipped_case_is_the_same_data_typed_and_as_
#: its_dict`), so a literal default written beside a read of a schema field
#: is never taken in a run: the silent defect P2 set out to remove (a default
#: in the code that disagrees with the schema's) cannot reach one. The free-
#: form parts (`Dict[str, Any]`: a ground motion, a section's mesh, a tendon's
#: friction) have their defaults in the code alone, and those agree across
#: the places that read them (the scan of v1.19). Two kinds are the code's
#: own and were checked by hand: an Optional field whose schema default is
#: None, read as `... or LIT` (the age 28, `load_steps or 1`), where LIT is
#: the live default and agrees across its readers; and `.get(k, LIT)` on a
#: key the dump carries as None, where LIT never fires -- three did (an
#: unnamed step reported as "step None", unnamed rebar sets merged) and read
#: `.get(k) or LIT` now. This ratchet keeps new dict reads out; the ones left
#: are not moved for their own sake.


def _dict_reads():
    n = 0
    for root, _dirs, files in os.walk(os.path.join(ROOT, "vfife")):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), os.path.join(ROOT, "vfife"))
            if f.endswith(".py") and f != "problem.py" and rel not in DICT_EDITORS:
                n += len(DICT_READS.findall(open(os.path.join(root, f), encoding="utf-8").read()))
    return n


def test_the_dict_reads_of_a_case_do_not_grow():
    n = _dict_reads()
    assert n <= DICT_READS_LEFT, (
        f"{n} reads of a case as a dict, {DICT_READS_LEFT} before: read the typed case "
        "(`problem.case...`, core/problem.py) instead")
    if n < DICT_READS_LEFT:
        pytest.fail(f"fewer dict reads ({n}): lower DICT_READS_LEFT to {n}")
