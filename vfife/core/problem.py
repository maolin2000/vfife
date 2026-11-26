"""The internal case a solver is built from, typed (P2 of the architecture
review of 1.17).

Until 1.18 a solver was handed `case.as_dict()` -- the validated case
dumped to a plain dict, with the element type the mesh holds and, for a
static run, the solver section rewritten -- and read it by string keys with
literal defaults, in 466 places across the package. A misspelt key read as
absent; a default written beside a read could disagree with the schema's
(the cause of two silent defects, `solver_setting`'s docstring tells them).

A `Problem` holds the same case twice:

    problem.case   the validated `schema.Case`, typed: `problem.case.solver.backend`
    problem[...]   the plain dict every reader read until now (a read-only
                   Mapping: `problem["solver"]["backend"]`), while P2 moves
                   the readers over to `case`, one module at a time

For a case that came through `assemble._with_element` (every run, every
shipped case) the two are the same data, to the type of every value
(tests/test_problem.py). They differ, and a reader moved to `case` must
mind it, where:

  * a hand-built dict left a part out: the dict lacks it, the typed case
    has the schema's default (the solvers' `solver_setting` gives the same);
  * a composite solver's half (`derive`) was given a part in a short form:
    the coupled model's solid half and the shell-line model's frame half
    get `output: {monitors: {}}` in the dict and every OutputSpec default
    typed (and the frame half its `element` as two keys, typed in full); the connector model's members are the element's settings whole
    in the dict and only a member group's keys typed.

`Problem.of(x)` makes one from a dict (validated against the schema, so
every solver now has a valid case) or hands a `Problem` back as it is.
`problem.derive(...)` is a composite solver's view for one of its halves
(the coupled model's solid, the shell-line model's frame, the connector
model's members), changed in both forms at once.

Neither form is to be changed once a solver holds it: the dict is
read-only at the top level only (a nested write still goes through, and
the typed case would not see it), and the typed parts are shared between
a problem and the halves derived from it. To change a solver's case, make
a new one (`s.p = s.p.derive(...)`).

What the dict says that the case cannot:

  * a static run is solved by a solver built as for relaxation
    (`assemble._with_element`): the dict's solver section says `mode:
    relaxation` and `static_run: true`. `problem.static_run` says it; the
    typed case is validated as the static case it was (a relaxation may not
    hold members that fail, a static run may) and then carries the mode the
    solver runs in, `relaxation`, exactly as the dict does;
  * a hand-built dict -- most unit tests build one -- need not say `mesh`
    or `material`: the solver is handed both, as objects. The typed case
    then holds a placeholder for each (`given` says which keys the dict
    stated), never read: the solvers read the mesh and the material they
    are handed.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Dict, FrozenSet, Iterator

#: what the typed case holds for a dict that did not say them (never read)
PLACEHOLDER_MESH = {"type": "gmsh", "path": "(the mesh the solver is handed)"}
PLACEHOLDER_MATERIAL = {"type": "linear_elastic", "E": 1.0, "nu": 0.0, "rho": 1.0}


class Problem(Mapping):
    """A validated case for a solver: typed (`case`) and as the plain dict
    it has always been (`problem[key]`, read-only)."""

    __slots__ = ("case", "static_run", "given", "_d")

    def __init__(self, d: Dict[str, Any], case, static_run: bool, given: FrozenSet[str]):
        # the caller's own dict, not a copy: the solvers read the very object
        # they read before (the bitwise guarantee of v1.19)
        self._d = d if type(d) is dict else dict(d)
        self.case = case
        self.static_run = bool(static_run)
        self.given = frozenset(given)

    # -- making one
    @classmethod
    def of(cls, x) -> "Problem":
        """`x` as a Problem: a Problem as it is, a dict validated."""
        if isinstance(x, Problem):
            return x
        if not isinstance(x, Mapping):
            raise TypeError(f"a solver is built from a case (a Problem or its dict), not {type(x).__name__}")
        return cls._validated(x)

    @classmethod
    def _validated(cls, d) -> "Problem":
        from ..io import schema
        v = copy.deepcopy(dict(d))
        given = frozenset(v)
        sol = dict(v.get("solver") or {})
        static_run = bool(sol.pop("static_run", False))
        if static_run:
            sol["mode"] = "static"
        v["solver"] = sol
        v.setdefault("mesh", PLACEHOLDER_MESH)
        v.setdefault("material", PLACEHOLDER_MATERIAL)
        case = schema.Case.model_validate(v)
        if static_run:
            case = case.model_copy(update={"solver": case.solver.model_copy(update={"mode": "relaxation"})})
        return cls(d, case, static_run, given)

    def derive(self, **changes) -> "Problem":
        """This problem with some top-level parts replaced, in both forms: a
        composite solver's view for one of its halves. Each value is given as
        the dict form; the typed part is validated against its field's type
        alone (the halves are views, not cases: a coupled model's solid half
        keeps the members of the whole). For `members`, the keys a member
        group does not have (the element's own settings, which the connector
        model hands its members whole) are left out of the typed part."""
        from pydantic import TypeAdapter
        from ..io import schema
        d = copy.deepcopy(self._d)
        typed = {}
        fields = schema.Case.model_fields
        for key, value in changes.items():
            if key not in fields:
                raise KeyError(f"{key!r} is not a part of a case")
            d[key] = copy.deepcopy(value)
            v = copy.deepcopy(value)
            if key == "members":
                keep = set(schema.MemberGroupSpec.model_fields)
                v = [{k: x for k, x in m.items() if k in keep} for m in v]
            if key == "solver" and self.static_run:
                # validated as the static case it is, run as relaxation (as `_validated`)
                v = {k: x for k, x in v.items() if k != "static_run"}
                v["mode"] = "static"
            part = TypeAdapter(fields[key].annotation).validate_python(v)
            if key == "solver" and self.static_run:
                part = part.model_copy(update={"mode": "relaxation"})
            typed[key] = part
        return Problem(d, self.case.model_copy(update=typed), self.static_run, self.given | set(changes))

    def derived(self, plain: Dict[str, Any]) -> "Problem":
        """`derive` with the parts in which `plain` -- a changed copy of
        this problem's dict (`plain()`) -- differs from it."""
        gone = set(self._d) - set(plain)
        if gone:
            raise KeyError(f"a derived problem keeps every part: {sorted(gone)} left out")
        return self.derive(**{k: v for k, v in plain.items() if k not in self._d or not _same(self._d[k], v)})

    # -- what the solver runs
    @property
    def element(self) -> str:
        """The element type the mesh holds (a mixed or coupled model's kind)."""
        return self.case.element.type

    @property
    def solver(self):
        """The solver settings, typed (`mode` as the solver runs it:
        `relaxation` for a static run, `static_run` says which)."""
        return self.case.solver

    # -- the dict form, read-only
    def __getitem__(self, key):
        return self._d[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    # the dict's own, not Mapping's (which go through a KeyError)
    def get(self, key, default=None):
        return self._d.get(key, default)

    def __contains__(self, key) -> bool:
        return key in self._d

    def keys(self):
        return self._d.keys()

    def items(self):
        return self._d.items()

    def values(self):
        return self._d.values()

    def plain(self) -> Dict[str, Any]:
        """A copy of the dict form, to change (a half of a composite
        solver is made with `derive` instead)."""
        return copy.deepcopy(self._d)

    def __reduce__(self):                  # pickle and deepcopy, on every Python (slots)
        return (Problem, (self._d, self.case, self.static_run, self.given))

    def __repr__(self):
        return f"Problem({self.case.name!r}, element={self.element!r}, static_run={self.static_run})"


def _same(a, b) -> bool:
    try:
        return bool(a == b)
    except (ValueError, TypeError):           # arrays in a hand-built dict
        return False


def as_dict(problem) -> Dict[str, Any]:
    """The dict form of a problem, as `case.as_dict()` gave it (for the
    writer, the fingerprint, the checkpoint): a copy."""
    return problem.plain() if isinstance(problem, Problem) else copy.deepcopy(dict(problem))
