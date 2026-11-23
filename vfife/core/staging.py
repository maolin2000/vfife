"""Path-dependent state and the load path: one protocol, two policies.

THE PROBLEM, MEASURED TWICE IN THIS PROJECT
-------------------------------------------
Staged relaxation reaches ONE equilibrium point by taking thousands of
explicit iterations whose path is not physical: kinetic damping zeroes the
velocity whenever the energy dips, so a strain wanders up and down on the way
in. A material whose state depends on the path it took therefore records every
one of those wiggles.

It has cost this project twice, in two different materials, and each time the
symptom looked constitutive and was not:

  * FIBRE SECTIONS. ~50 000 update calls and 4 000-6 500 direction flips in a
    SINGLE load stage, which re-anchors the Giuffre-Menegotto-Pinto curve away
    from the real excursion and collapses the hysteresis loops. Absorbed
    energy came out 16x too small: 1.13 kJ against 17.92 kJ for one +-42 mm
    cycle, measured 14.7 kJ. The ratio against the experiment went 0.32 ->
    1.77 when the state was committed once per converged stage
    (verification/case_11).

  * BOND DAMAGE. Energy banked per iteration made a MONOTONIC tension case
    depend on the iteration budget: same tolerance, n_steps 60 000 -> 200 000
    moved the slip from 106.0 um to 24.3 um against a perfectly bonded 24.9.

Both were fixed. They were fixed SEPARATELY, with two different mechanisms in
two different files, and the continuum path still has neither -- which is how
the second one came to be written at all. This module is the third time it is
written and the last.

TWO POLICIES, AND THEY ARE NOT THE SAME
---------------------------------------
`restore-then-integrate` (what the fibre sections need)
    Every force evaluation starts from the last COMMITTED state, so the state
    is a function of (committed, current strain) and the iterations in between
    leave nothing behind. This is what an implicit solver computes anyway. It
    is the right policy whenever the constitutive rule ANCHORS on reversals --
    GMP, concrete02 -- because a spurious reversal moves the anchor.

`accumulate-between-commits` (what bond damage needs, ON TOP of the first)
    Only the DRIVER of the irreversible part is advanced at committed points.
    `bond_cyclic` banks its damage energy between equilibrium points this way.

    An earlier version of this note said the bond laws' `s_max` and unloading
    anchor "track the iteration path" deliberately. They must not, and the
    measurement that settled it is in tests/test_bonded_convergence.py: a bar
    started 20 um off inside a rigid host came to rest at 300 um of PERMANENT
    slip with zero force, each overshoot of the relaxation transient having
    moved the anchor a little further. `bond_ceb` and `bond_cyclic` are both
    anchored laws and both are now TrialStates as well; the cyclic one is
    additionally a CallbackState, committed FIRST so the energy is banked from
    the old anchor before the snapshot moves it.

A protocol that forced one of these would have to break the other, so this one
requires `commit` and leaves `restore_trial` optional.

WHAT THE SOLID/CONTINUUM PATH DOES
    Neither, deliberately, and the reason is a measurement rather than
    an oversight -- see `tests/test_staging.py`. CDP's return map is far less
    sensitive to direction flips than an anchored uniaxial rule: the
    equivalent plastic strain does grow inside a held load stage, but the
    growth does NOT track the iteration count (stage 6 grew 1.7e-6 over 1275
    iterations while stage 5 grew 3.3e-5 over 775), so it is settling rather
    than accumulation. Switching it to restore-then-integrate would change
    every CDP result that has ever been verified, and that is a decision to
    take on its own evidence, not as a side effect of tidying.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class StagedState(Protocol):
    """Something whose history advances on CONVERGED points, not iterations."""

    def commit(self) -> None:
        """This state is an equilibrium point. Advance whatever it drives."""

    # `restore_trial` is optional: implement it only for the
    # restore-then-integrate policy. `Registry` treats its absence as a no-op.


class TrialState:
    """The restore-then-integrate policy, over a plain state array.

    Holds the live array the material mutates and a committed copy. Every
    force evaluation restores, so the material integrates from the last
    equilibrium point; `commit` makes the current state that point.

    The arrays are written THROUGH (`[...] =`) rather than rebound, because
    the live array is the one the kernels were handed at setup and rebinding
    it here would leave them writing into an orphan.
    """

    __slots__ = ("live", "committed")

    def __init__(self, array: np.ndarray):
        self.live = array
        self.committed = np.array(array, copy=True)

    def restore_trial(self) -> None:
        if self.live.size:
            self.live[...] = self.committed

    def commit(self) -> None:
        if self.live.size:
            self.committed[...] = self.live


class CallbackState:
    """A `StagedState` for something that commits itself its own way.

    `bond_cyclic` banks an energy increment between committed points rather
    than snapshotting an array, so it cannot be a `TrialState`; it is a
    callable plus the array it works on.
    """

    __slots__ = ("_commit", "_state")

    def __init__(self, commit, state):
        self._commit = commit
        self._state = state

    def commit(self) -> None:
        self._commit(self._state)


class Registry:
    """The staged states of one solver, committed together.

    A list rather than a set of attributes so that a solver gains the
    behaviour by REGISTERING, and a path that registers nothing pays nothing
    -- which is what keeps this out of the way of the continuum path while
    the decision above stands.
    """

    __slots__ = ("_items",)

    def __init__(self):
        self._items = []

    def add(self, item) -> None:
        if not hasattr(item, "commit"):
            raise TypeError(f"{item!r} has no commit(); see core/staging.py")
        self._items.append(item)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def restore_trial(self) -> None:
        """Called before a force evaluation, for whoever wants it."""
        for item in self._items:
            restore = getattr(item, "restore_trial", None)
            if restore is not None:
                restore()

    def commit(self) -> None:
        """Called where a load level has CONVERGED, and nowhere else.

        Nowhere else is the whole point: a commit inside the iteration loop
        would put the iteration path back into the history and undo both of
        the fixes this module exists to hold.
        """
        for item in self._items:
            item.commit()
