"""What the drivers may call on a solver.

The drivers -- the step loop and the compiled step (core/compiled_step), the
analysis steps (core/analysis_steps), the implicit solver (implicit), the
restart (core/resume), modal and buckling analyses -- were written against
the solvers' private methods (the architecture review of 1.17).
The ones every driver uses are public now and listed here; the boundary test
(tests/test_solver_boundary.py) counts what is still reached privately.

`StepSolver` is a `typing.Protocol`: nothing inherits it, it only says what a
frame, mixed-line or fibre solver provides (`tests/test_solver_protocol.py`
checks that each one does). The fibre solver also has `compiled_event`,
for the step its kernel finished with an event (a fibre point that
failed); the frame kernel has no such status.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

#: the private names of 1.17 and before: a script that still
#: replaces one of them on a solver sets an attribute nothing calls any more
RENAMED = {"_internal_force": "internal_force", "_commit_state": "commit_state", "_step_data": "step_data",
           "_one_dynamic_step": "one_dynamic_step", "_after_dynamic_step": "after_dynamic_step",
           "_compiled_pending": "compiled_pending", "_compiled_event": "compiled_event",
           "_compiled_event_stop": "compiled_event_stop", "_apply_pending_splits": "apply_pending_splits"}


def refuse_old_names(solver) -> None:
    """Refuse a solver on which a script replaced a method under its old
    private name: the wrapper would never run, and nothing would say so."""
    old = [n for n in RENAMED if n in vars(solver) or any(n in vars(k) for k in type(solver).__mro__)]
    if old:
        raise ValueError("these solver methods were renamed in 1.18 and the wrappers put on the old names "
                         "would not run: " + ", ".join(f"{n} -> {RENAMED[n]}" for n in old))


@dataclass
class KernelMembers:
    """What a frame or mixed-line solver hands the compiled step for its
    members (`compiled_step` read the solver's private arrays
    and chose between them by the solver's type until then)."""
    #: the frame members: (ids, conn, R0, k, L0, dofs, imposed deformation)
    f: tuple
    #: the message when a frame member collapses
    collapse: str
    #: the axial members (bars, cables): ids, (n, 2) ends, EA
    t_ids: Any
    t_conn: Any
    t_EA: Any
    #: the frame kernel's threads
    threads: int
    #: the compiled connectors (`_connector_kernel()`), or None
    connectors: Optional[dict] = None


@runtime_checkable
class StepSolver(Protocol):
    #: the components attached to the solver (core/components)
    components: tuple

    def internal_force(self) -> tuple[Any, Any]:
        """(f, actions): the members' internal force at the current state and
        what the components add to it; probes and trials go through it too."""

    def commit_state(self) -> Any:
        """Commit the history of an accepted step: the members', then the
        components'."""

    def step_data(self, k: int) -> list:
        """[fixed, moving, prescribed, groups, ground] of analysis step `k`."""

    def one_dynamic_step(self, k: int, local: int, data: list) -> Any:
        """Advance one explicit step `local` of analysis step `k`; returns its
        external force."""

    def after_dynamic_step(self, k, local, F, fixed, plan, on_output, output_every,
                           on_progress, progress_every) -> Any:
        """Book, write and check what one explicit step left."""

    def compiled_pending(self) -> bool:
        """True when the loop must take the next step (an event the compiled
        step does not run is waiting)."""

    def compiled_event_stop(self, k, local, last, plan) -> Any:
        """The step at which the compiled step must stop for the loop."""

    def apply_pending_splits(self, t) -> Any:
        """Apply the splits and removals queued during the step, stamped `t`
        (the frame solver stamps them with its own step's time)."""

    def nodal_fields(self, *a, **kw) -> dict:
        """The output fields of the current state, the components' included."""

    def checkpoint(self, *a, **kw) -> dict:
        """The state a restart needs, the components' included."""

    def restore(self, ck: dict, *a, **kw) -> Any:
        """Restore a checkpoint written by `checkpoint`."""
