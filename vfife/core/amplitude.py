"""Amplitude curves: how each boundary condition varies over its step.

    amplitudes:
      ramp:  [[0, 0], [1, 1]]                    # the default
      hold:  [[0, 1], [1, 1]]                    # already there, stays there
      push:  {cyclic: {peaks: [0.25, 0.5, 1.0], repeats: 2}}

    steps:
      - name: gravity
        loads: [{region: all, vector: [0, 0, -1], amplitude: ramp}]
      - name: pushover
        loads: [{region: all, vector: [0, 0, -1], amplitude: hold}]
        constraints:
          - {region: top, dofs: [1, 0, 0], value: 0.05, amplitude: push}

THE ABSCISSA IS THE STEP'S OWN PROGRESS, 0 TO 1
-----------------------------------------------
Not physical time, and not the iteration count.  A relaxation run has no
physical time at all -- its "time" is a pseudo-time whose only meaning is the
order things happen in -- so a curve written against seconds cannot be shared
between a relaxation and a dynamic run of the same structure.  Against
normalised progress it can: the same `push` curve above means the same loading
protocol whether the step is 40 staged equilibrium points or 200 000 explicit
iterations.

Under staged relaxation, stage k of N sits at p = k/N, so a cyclic curve makes
each stage one equilibrium point on the hysteresis loop -- which is exactly what
the reaction-displacement history needs to be a curve rather than a smear.

WHY THIS REPLACED `solver.load_curve`
-------------------------------------
There was one curve, and it multiplied every load and every prescribed
displacement in the model together.  So an axial load could not be held constant
while a lateral displacement was pushed, and reversed cyclic loading could not
be expressed at all -- which meant the Chaboche/rebar kinematic hardening and
the CDP `wt`/`wc` stiffness recovery, both of which exist for exactly that
loading, had no way to be driven.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class Amplitude:
    """A piecewise-linear factor on [0, 1]. Constant outside its own range."""

    __slots__ = ("name", "p", "a")

    def __init__(self, name: str, table):
        t = np.asarray(table, dtype=float)
        if t.ndim != 2 or t.shape[1] != 2 or t.shape[0] < 2:
            raise ValueError(
                f"amplitude {name!r}: expected at least two [progress, factor] "
                f"pairs, got shape {t.shape}"
            )
        if np.any(np.diff(t[:, 0]) < 0):
            raise ValueError(
                f"amplitude {name!r}: the progress column must not decrease. "
                "To go back down, add a point with a lower FACTOR, not a lower "
                "progress."
            )
        self.name = name
        self.p = np.ascontiguousarray(t[:, 0])
        self.a = np.ascontiguousarray(t[:, 1])

    def __call__(self, progress: float) -> float:
        return float(np.interp(progress, self.p, self.a))

    def settled(self, progress: float, tol: float = 1e-12) -> bool:
        """Has this curve reached the value it will end the step at?

        The convergence test needs this: a residual below tolerance means
        nothing while the load that produced it is still moving.
        """
        return abs(self(progress) - float(self.a[-1])) <= tol * max(
            1.0, abs(float(self.a[-1])))

    def describe(self) -> str:
        return (f"{self.name}: {len(self.p)} points, "
                f"factor {self.a.min():+.3g} to {self.a.max():+.3g}")


RAMP = Amplitude("ramp", [[0.0, 0.0], [1.0, 1.0]])
HOLD = Amplitude("hold", [[0.0, 1.0], [1.0, 1.0]])
#: the default: on over the first half of the step, then steady.
#:
#: `ramp` on its own is the wrong default for a relaxation run, and not for a
#: subtle reason: if the load is still changing on the last iteration then the
#: solve is still chasing it, and "converged" cannot honestly be declared at
#: any point in the step.  Ramping over the first half leaves the second half
#: to relax at a fixed load, which is what a quasi-static step is for.  It also
#: avoids applying the whole load as a step change, which `hold` would.
RAMP_HOLD = Amplitude("ramp_hold", [[0.0, 0.0], [0.5, 1.0], [1.0, 1.0]])

#: available without being declared
#: the `hold` of a support that holds its node where it found it
#: (orientation._hold_in_place): the same curve under its own identity, so
#: that `_apply_bc` can place all of them in one assignment
IN_PLACE = Amplitude("hold", [[0.0, 1.0], [1.0, 1.0]])

BUILTIN = {"ramp": RAMP, "hold": HOLD, "ramp_hold": RAMP_HOLD}


def cyclic_table(peaks, repeats: int = 1, start_at_zero: bool = True):
    """The standard reversed-cyclic protocol, as a table.

    `peaks` are the successive amplitudes of a displacement-controlled test --
    the usual thing is a few cycles at each of a growing set of drift levels.
    Each cycle is 0 -> +peak -> 0 -> -peak -> 0, `repeats` times per peak, and
    the whole protocol is laid out evenly over the step's progress.

    Writing that by hand is 4 x len(peaks) x repeats + 1 pairs of numbers, and
    getting one of them wrong shifts every subsequent point -- which is a
    tedious way to discover that a hysteresis loop is asymmetric for a reason
    that has nothing to do with the material.
    """
    vals = [0.0] if start_at_zero else []
    for pk in peaks:
        for _ in range(int(repeats)):
            vals += [float(pk), 0.0, -float(pk), 0.0]
    if len(vals) < 2:
        raise ValueError("a cyclic amplitude needs at least one peak")
    p = np.linspace(0.0, 1.0, len(vals))
    return np.column_stack([p, np.asarray(vals, dtype=float)])


def build(specs: Dict[str, object] | None) -> Dict[str, Amplitude]:
    """Named amplitudes from the case file, plus the built-ins."""
    out = dict(BUILTIN)
    for name, spec in (specs or {}).items():
        if isinstance(spec, dict) and "cyclic" in spec:
            c = spec["cyclic"]
            out[name] = Amplitude(name, cyclic_table(
                c["peaks"], c.get("repeats", 1),
                c.get("start_at_zero", True)))
        elif isinstance(spec, dict) and "table" in spec:
            out[name] = Amplitude(name, spec["table"])
        else:
            out[name] = Amplitude(name, spec)
    return out


def get(amps: Dict[str, Amplitude], name: str | None, used_by: str,
        staged: bool = False) -> Amplitude:
    """Look up an amplitude, or supply the default for this kind of step.

    The default differs, and it has to.  A STAGED step holds each level until
    it converges, so its stages should sample the loading evenly: stage k of N
    at k/N, which is `ramp`.  Give a staged step `ramp_hold` instead and the
    load reaches full at the halfway stage and sits there -- the second half of
    the load-deflection curve becomes the same point measured repeatedly.

    A CONTINUOUS step has no such structure, and there `ramp` is the wrong
    default for the opposite reason: the load is still moving on the last
    iteration, so equilibrium can never honestly be declared.
    """
    if name is None:
        return RAMP if staged else RAMP_HOLD
    try:
        return amps[name]
    except KeyError:
        raise ValueError(
            f"{used_by} uses amplitude {name!r}, which is not defined. "
            f"Available: {', '.join(sorted(amps))}"
        ) from None


def timed_hold(step, constraints, loads):
    """In a step that is a stretch of TIME (`to_age`), every load and every
    prescribed displacement without its own `amplitude` is `hold`: applied in
    full at the first time station and held.

    Its `load_steps` are time stations, not load levels. Under the staged
    default (`ramp`) a superimposed dead load repeated in a creep step dropped
    from full to 1/n at the first station and climbed back over the years --
    a 30-year deflection of +1.1 mm instead of -3.2 mm, and nothing said so.
    Ramping over time needs an amplitude written on purpose."""
    if not (step or {}).get("to_age"):
        return constraints, loads
    def held(items):
        return [{**x, "amplitude": "hold"} if x.get("amplitude") is None else x for x in items]
    return held(constraints), held(loads)
