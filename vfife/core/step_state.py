"""The state the analysis steps keep on a solver.

`core/analysis_steps` wrote it onto the solver as some fifteen private
attributes (`solver._clock`, `solver._birth_d`, ...), and every reader asked
first whether it was there (`getattr(solver, "_age_shift", None)`). It is one
object now: `solver.staging`. A solver whose case has no analysis steps that
change the model holds `NO_STAGING`, whose fields are the "not staged"
answers the probes used to supply, and which cannot be written to -- a write
there would have been a bug (state of a plan that does not exist).
`analysis_steps.install` gives a staged solver its own `Staging`.

`solver.plan` (the analysis plan, `analysis_steps.Plan`, or None) sits beside
it on the solver.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Staging:
    #: the model's own clock across the steps (s), the start of the current one
    clock: float = 0.0
    #: the solver mode the case asked for (a static step switches it per step)
    base_mode: Optional[str] = None
    #: "time" when any step is dynamic, else "load_factor"; None when unstaged
    time_axis: Optional[str] = None
    #: True when the steps mix procedures (the ledger then audits part of the run)
    mixed_procedures: bool = False
    #: (nel, 12) the deformation each member was born with, or None
    birth_d: Optional[np.ndarray] = None
    #: the local and the creep reference stiffness of every member as built
    k_built: Optional[np.ndarray] = None
    k_ref_built: Optional[np.ndarray] = None
    #: (nn,) particles no built member reaches
    dormant: Optional[np.ndarray] = None
    #: per step: the extra force vectors it carries (step index -> list)
    step_extra: Dict[int, list] = field(default_factory=dict)
    #: the members born in the current step
    born_now: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    #: each member's concrete clock: age = model age + shift, never before floor
    age_shift: Optional[np.ndarray] = None
    age_floor: Optional[np.ndarray] = None
    #: the fibre solver's first section states, for a member built again
    state0: Dict[str, np.ndarray] = field(default_factory=dict)
    fstate0: List[np.ndarray] = field(default_factory=list)
    fcom0: List[np.ndarray] = field(default_factory=list)
    #: a restart in a dynamic step: the (dt, n_steps, total_time) the step
    #: was running with, which `begin_step` puts back instead of taking a
    #: new step from the masses a split or a park has changed since
    resumed_time: Optional[tuple] = None


class _Unstaged(Staging):
    """`NO_STAGING`: the defaults, read-only."""

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(f"solver.staging.{name}: this solver has no analysis steps that change it")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"solver.staging.{name}: this solver has no analysis steps that change it")


def _unstaged() -> Staging:
    s = _Unstaged()
    empty = np.zeros(0, np.int64)
    empty.flags.writeable = False
    object.__setattr__(s, "born_now", empty)
    object.__setattr__(s, "step_extra", MappingProxyType({}))
    object.__setattr__(s, "state0", MappingProxyType({}))
    object.__setattr__(s, "fstate0", ())
    object.__setattr__(s, "fcom0", ())
    object.__setattr__(s, "_frozen", True)
    return s


NO_STAGING: Staging = _unstaged()
