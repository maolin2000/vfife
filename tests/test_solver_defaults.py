"""One default per setting, and it lives in the schema.

THE DEFECT THIS FILE MAKES IMPOSSIBLE
    A pydantic model always emits every field, so a validated case dict has
    the key PRESENT even when the analyst never wrote it. That makes

        sol.get("dt_safety", 0.6)

    dead code inside a solver: the key is there, the literal never fires, and
    the schema's value silently wins. Two places each declaring a default,
    with one of them永远 unreachable.

    It has cost this project twice:

      * `mass` -- the schema emits `mass: None`, so the solver's fallback
        never fired and relaxation cases silently ran with PHYSICAL mass and
        a CFL time step: real dynamics, with n_steps counted as relaxation
        iterations, and no error anywhere.

      * `damping.type` -- the schema defaulted it to "kinetic", so the
        mode-dependent expression in SIX solvers was dead and every
        `mode: dynamic` run got kinetic damping, which zeroes the velocity at
        every kinetic-energy peak and removes the dynamics it is applied to.

    The second one is why the test below scans EVERY solver rather than the
    one that happened to be under the microscope: the same line had been
    written six times, and fixing one of them would have fixed nothing.
"""
import pathlib
import re

import pytest

from vfife.io.schema import SolverSpec, solver_setting

SOLVERS = sorted(pathlib.Path("vfife/core").glob("*solver*.py"))
#: settings the schema gives a value for -- these must not be defaulted again
WITH_DEFAULT = {k: f.default for k, f in SolverSpec.model_fields.items()
                if f.default is not None}


def test_there_are_solvers_to_scan():
    """A guard on the guard: a glob that matched nothing would make every
    assertion below vacuously true, which is the classic way a static check
    stops checking."""
    assert len(SOLVERS) >= 6, [p.name for p in SOLVERS]
    assert WITH_DEFAULT, "SolverSpec declares no defaults at all?"


@pytest.mark.parametrize("path", SOLVERS, ids=lambda p: p.name)
def test_no_solver_restates_a_schema_default(path):
    """The whole rule, as a static check.

    `.get("key", <anything>)` or `_step_solver("key", <anything>)` for a key
    the schema already defaults is the defect, whatever the literal says --
    even when it AGREES with the schema, because agreement today is what
    makes tomorrow's silent divergence possible.
    """
    src = path.read_text(encoding="utf-8")
    bad = []
    for key in WITH_DEFAULT:
        for pattern in (r'\.get\(\s*"' + key + r'"\s*,',
                        r'_step_solver\(\s*"' + key + r'"\s*,'):
            for m in re.finditer(pattern, src):
                bad.append(f"{path.name}:{src[:m.start()].count(chr(10)) + 1} "
                           f"restates the default for {key!r}")
    assert not bad, (
        "read it from the typed case (`self.p.solver.<key>`, core/problem.py) or "
        "with io.schema.solver_setting, which take the default from the schema:\n  " + "\n  ".join(bad))


def test_the_reader_returns_the_schema_default_for_an_absent_key():
    assert solver_setting({}, "dt_safety") == WITH_DEFAULT["dt_safety"]
    assert solver_setting({}, "backend") == WITH_DEFAULT["backend"]
    assert solver_setting(None, "tolerance") == WITH_DEFAULT["tolerance"]


def test_a_present_but_none_key_still_gets_the_schema_default():
    """The exact shape of the original defect: the key IS there, carrying
    None, so a `get` default walks straight past it."""
    assert solver_setting({"dt_safety": None}, "dt_safety") == \
        WITH_DEFAULT["dt_safety"]


def test_an_explicit_value_wins():
    assert solver_setting({"dt_safety": 0.2}, "dt_safety") == 0.2


def test_a_setting_the_schema_leaves_open_comes_back_none():
    """`load_steps`, `mass`, `erosion`, `target_dt` and `damping.type` have no
    schema default because the right value depends on something the schema
    cannot see. Those stay the solver's business -- resolved with `or`, never
    with a `get` default."""
    for key in ("load_steps", "mass", "erosion", "target_dt"):
        assert key in SolverSpec.model_fields, key
        assert solver_setting({}, key) is None, key


def test_damping_type_has_no_schema_default_so_the_mode_can_decide():
    """The one field this whole exercise turned on. If a default comes back
    here, six solvers' `damp.get("type") or (...)` go dead again and every
    dynamic run silently gets kinetic damping."""
    from vfife.io.schema import DampingSpec

    assert DampingSpec().type is None
    assert DampingSpec(type="kinetic").type == "kinetic"


@pytest.mark.parametrize("path", SOLVERS, ids=lambda p: p.name)
def test_the_mode_dependent_damping_default_uses_or_not_a_get_default(path):
    """`damp.get("type", ...)` would be dead the moment anyone gives
    DampingSpec.type a default again -- and someone will, because a default
    looks like an improvement. `or` survives that."""
    src = path.read_text(encoding="utf-8")
    assert 'damp.get("type",' not in src, (
        f"{path.name}: a present-but-None key walks past a get() default; "
        'write `damp.get("type") or (...)`')
