"""Each solver the drivers step implements `core.protocol.StepSolver`:
the methods are there, public, and every solver takes the
same arguments, defaults included; a script that still wraps a method under
its old private name is told so instead of being silently skipped."""
import inspect

import pytest

from vfife.core.beam_solver import BeamSolver
from vfife.core.connector_solver import ConnectorSolver
from vfife.core.coupled_solver import CoupledSolver
from vfife.core.fiber_beam_solver import FiberBeamSolver
from vfife.core.mixed_structural_solver import MixedStructuralSolver
from vfife.core.protocol import RENAMED, StepSolver, refuse_old_names
from vfife.core.shell_line_solver import ShellLineSolver
from vfife.core.shell_solver import ShellSolver

KINDS = [BeamSolver, MixedStructuralSolver, ConnectorSolver, CoupledSolver, ShellSolver, ShellLineSolver,
         FiberBeamSolver]
METHODS = [n for n, v in vars(StepSolver).items() if callable(v) and not n.startswith("_")]


def _sig(kind, name):
    return [(p.name, p.default) for p in inspect.signature(getattr(kind, name)).parameters.values()]


@pytest.mark.parametrize("kind", KINDS)
def test_the_stepped_solvers_provide_the_protocol(kind):
    for name in METHODS:
        assert callable(getattr(kind, name, None)), f"{kind.__name__} has no {name}"
    assert isinstance(kind.components, tuple)
    obj = kind.__new__(kind)                       # the class's attributes, no model needed
    assert isinstance(obj, StepSolver)
    for old in RENAMED:
        assert not hasattr(kind, old), f"{kind.__name__} still has {old}"


@pytest.mark.parametrize("kind", KINDS[1:])
def test_every_solver_takes_the_frame_solvers_arguments(kind):
    for name in METHODS:
        if name == "apply_pending_splits":
            # the fibre solver needs its `t`; the frame one gives it a default
            assert [p for p, _ in _sig(kind, name)] == ["self", "t"]
            continue
        assert _sig(kind, name) == _sig(BeamSolver, name), (kind.__name__, name)


def test_a_wrapper_on_an_old_name_is_refused():
    s = BeamSolver.__new__(BeamSolver)
    refuse_old_names(s)
    s._internal_force = lambda: None
    with pytest.raises(ValueError, match="_internal_force -> internal_force"):
        refuse_old_names(s)
