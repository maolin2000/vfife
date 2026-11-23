"""Fracture in three layers.

THE QUESTION THIS FILE ANSWERS is whether a fracture criterion has to be
written once per material times once per element. It does not, and the reason
is worth stating before any code: the three things that have to be decided are
decided by three different parties, and each of them can only decide one.

    the MATERIAL knows when its own point is finished
        -> `Material.failure(state)`, a monotone fraction per point, and
           nothing else. It does not know how many points the element has,
           where they sit, or what happens next.

    the ELEMENT knows how its points add up
        -> a `Layout` here: how many points, and how they are grouped. A
           hexahedron has eight Gauss points and fails when NUMFIP of them do;
           a fibre section has twenty fibres, each spanning the Gauss stations
           along the member, and the paper's FAC is the fraction of fibres
           gone (Cai et al. 2023, eq. 19). Same arithmetic, different shape.

    the MODEL knows what to do with a finished element
        -> an `Action` here: stop it carrying load, and (later) split the
           particle so the fragment can fall.

M materials and E element families then cost M + E implementations rather
than M x E, and a new material inherits fracture on every element on the day
it defines one method.

WHY NOT JUST DELETE THE ELEMENT, which is what this library already does for
solids and what `*MAT_ADD_EROSION` does in LS-DYNA: because the mass goes with
it. LS-DYNA's own debris work says it plainly -- with erosion "the failed
elements are deleted" and the mass is gone, while "the existence of debris
cannot be ignored since the impact force is loaded on the structure through
the debris", which is why they added solid-to-SPH and solid-to-DES conversion.
In a collapse analysis a deleted member cannot fall on anything. So the first
action here is DEACTIVATION, not deletion: the element stops carrying load and
its mass stays on the nodes it was assigned to at setup. That is still not a
fragment that detaches -- for that the particle has to split, which is the
next stage -- but it is not a model that quietly loses weight either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Layout:
    """How an element family's material points are arranged.

    `points_per_element` is the total, and `groups_per_element` is how many
    INDEPENDENT things those points describe -- which is the number the
    fraction is taken over.

    They differ exactly where the paper's fibre beam differs from a solid: a
    fibre runs the length of the member and is sampled at every Gauss station,
    so twenty fibres at three stations are sixty points but twenty things that
    can break. A hexahedron's eight Gauss points are eight things.
    """
    name: str
    points_per_element: int
    groups_per_element: int
    #: (n_elements, points_per_element) -> (n_elements, groups_per_element)
    #: `any` for a fibre (broken anywhere is broken), `mean` for a Gauss point
    within_group: str = "max"
    #: what the fraction is taken OVER, when that is more than the number of
    #: groups tracked. A section of 64 concrete fibres and 4 bars can only
    #: ever fail in the bars -- concrete01 declares no criterion, so those 64
    #: points are structurally zero -- and tracking them costs 17x the memory
    #: and the time for a number that cannot change. The tracked groups are
    #: then the 4 bars and the denominator stays 68, so FAC means exactly what
    #: it meant before: the fraction of the SECTION that is gone, not the
    #: fraction of the fibres that were being watched.
    #: 0 means "the groups themselves", which is the ordinary case.
    denominator: int = 0

    def __post_init__(self):
        if self.points_per_element < 1 or self.groups_per_element < 1:
            raise ValueError(f"layout {self.name!r} needs at least one point")
        if self.points_per_element % self.groups_per_element:
            raise ValueError(
                f"layout {self.name!r}: {self.points_per_element} points do not "
                f"divide into {self.groups_per_element} groups")
        if self.within_group not in ("max", "mean"):
            raise ValueError("within_group is max or mean")
        if self.denominator and self.denominator < self.groups_per_element:
            raise ValueError(
                f"layout {self.name!r}: the fraction cannot be taken over "
                f"{self.denominator} things when {self.groups_per_element} are "
                f"tracked")

    @property
    def over(self) -> int:
        """How many things the fraction is out of."""
        return self.denominator or self.groups_per_element

    @property
    def per_group(self) -> int:
        return self.points_per_element // self.groups_per_element


def solid_layout(n_gauss: int) -> Layout:
    """A solid: each Gauss point is its own thing."""
    return Layout("solid", n_gauss, n_gauss)


def fiber_layout(n_fibers: int, n_stations: int, tracked: int = 0) -> Layout:
    """A fibre section: `n_fibers` things, each sampled at `n_stations`.

    `max` within the group is Cai et al.'s eq. (20) exactly: fibre i has
    failed when the strain at ANY of its integration points has passed the
    fracture strain -- not when the average has.

    `tracked` is how many of those fibres carry a law that can fail at all.
    Fewer means a smaller array to gather and reduce every step; the fraction
    is still taken over `n_fibers`, so the number means the same thing.
    """
    live = tracked or n_fibers
    return Layout("fiber", live * n_stations, live, within_group="max",
                  denominator=n_fibers)


def bar_layout() -> Layout:
    """A truss or a bar: one point, one thing."""
    return Layout("bar", 1, 1)


def fraction(point_failure: np.ndarray, layout: Layout,
             over: Optional[np.ndarray] = None) -> np.ndarray:
    """FAC per element: the fraction of the element's things that are gone.

    `point_failure` is (n_elements, layout.points_per_element) in [0, 1], with
    the points of one group contiguous. Returns (n_elements,) in [0, 1].

    `over`, when given, is (n_elements,) and replaces the layout's scalar
    denominator element by element. It exists for a model whose members
    carry DIFFERENT fibre sections: the arrays are padded to
    the widest section with fibres of zero area, and a padded fibre is not a
    thing that can break -- the fraction on a 12-fibre member is out of 12,
    not out of the 20 the widest section has.

    0 is intact, 1 is fully fractured, and in between is Cai et al.'s
    partially fractured state (their Fig. 8) -- which is not a curiosity: it
    is what a section looks like for most of the time between first yield and
    separation, and it is the number worth recording.
    """
    f = np.asarray(point_failure, dtype=float)
    if f.ndim != 2 or f.shape[1] != layout.points_per_element:
        raise ValueError(
            f"point failure is {f.shape}, expected (n, {layout.points_per_element}) "
            f"for layout {layout.name!r}")
    if f.size and (f.min() < -1e-12 or f.max() > 1.0 + 1e-12):
        raise ValueError("a failure fraction lies in [0, 1]")
    grouped = f.reshape(len(f), layout.groups_per_element, layout.per_group)
    per_group = _within(grouped, layout.within_group)
    if over is not None:
        over = np.asarray(over, dtype=float)
        if over.shape != (len(f),):
            raise ValueError(f"`over` is {over.shape}, expected ({len(f)},)")
        if np.any(over < 1.0):
            raise ValueError("every element is out of at least one thing")
        return per_group.sum(axis=1) / over
    if layout.denominator and layout.denominator != layout.groups_per_element:
        return per_group.sum(axis=1) / layout.denominator
    return per_group.mean(axis=1)


def _within(grouped, how):
    """Reduce over a group's points -- NOT with `grouped.max(axis=2)`.

    THE LAST AXIS IS TWO OR THREE LONG (the Gauss stations of a fibre), and
    numpy's reduction over a short trailing axis is one call per group rather
    than one pass over memory: measured on (40, 68, 2), `max(axis=2)` costs
    120 us against 5 us for the same answer written as a running `np.maximum`
    over the stations. That is 90% of what asking about fracture used to cost
    per step -- it made a fibre-beam step 1.9x, of which only a fifth was the
    question itself.

    The same shape appears in `mean(axis=1)` just above and does NOT have the
    problem: that axis is the fibres, and there are 68 of them.
    """
    n = grouped.shape[2]
    out = grouped[:, :, 0].copy()
    if how == "mean":
        for k in range(1, n):
            out += grouped[:, :, k]
        return out / n if n > 1 else out
    for k in range(1, n):
        np.maximum(out, grouped[:, :, k], out=out)
    return out


class Latch:
    """The state of the fracture: what has failed, and it never un-fails.

    A LATCH RATHER THAN A TEST EVALUATED EACH STEP, because the underlying
    measures are not all monotone in practice even when they are in theory: a
    strain-based criterion on a vibrating member is above its threshold on one
    half-cycle and below it on the next. `cdp`'s `d` really does fall when a
    crack closes, which is why `Material.failure` is built from `dt` and `dc`
    instead -- but a latch is the belt to that braces, and it costs one array.
    """

    def __init__(self, n_elements: int, layout: Layout, threshold: float = 1.0,
                 over: Optional[np.ndarray] = None):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("the fracture threshold lies in (0, 1]")
        self.layout = layout
        #: per-element denominator, or None for the layout's own (see `fraction`)
        self.over = None if over is None else np.asarray(over, dtype=float)
        self.threshold = float(threshold)
        self.point = np.zeros((n_elements, layout.points_per_element))
        self.fractured = np.zeros(n_elements, dtype=bool)
        self.n_fractured = 0

    @property
    def fac(self) -> np.ndarray:
        return fraction(self.point, self.layout, over=self.over)

    def update(self, point_failure: np.ndarray, kinematic: Optional[np.ndarray] = None):
        """Take the worst of what has ever been seen, and latch the elements
        that have crossed the threshold. Returns the newly fractured indices.

        `kinematic` is the element-level half of the criterion -- buckling, a
        collapsed volume, a rotation past a limit -- which is not a material
        property and never was. The solid erosion of v0.1 already had one
        (an inverted element goes whatever the constitutive model says), and
        it is the one criterion that no material can provide.
        """
        f = np.asarray(point_failure, dtype=float)
        if f.shape != self.point.shape:
            raise ValueError(f"point failure is {f.shape}, expected {self.point.shape}")
        np.maximum(self.point, f, out=self.point)
        now = self.fac >= self.threshold - 1e-12
        if kinematic is not None:
            now |= np.asarray(kinematic, dtype=bool)
        newly = np.flatnonzero(now & ~self.fractured)
        self.fractured[newly] = True
        self.n_fractured = int(self.fractured.sum())
        return newly


# --------------------------------------------------------------------------- #
#  layer 3: what the model does with a finished element
# --------------------------------------------------------------------------- #

class Fracture:
    """The latch plus the action, and the record of what happened when.

    THE ACTION IS DEACTIVATION, and only that: a fractured element stops
    contributing internal force, and everything else about it is left alone.
    In particular ITS MASS STAYS ON ITS NODES, because the mass was lumped
    onto the particles at setup and this class never touches it. That is the
    whole reason not to reach for the element deletion the solid path already
    has: in a collapse analysis the interesting question is what the
    debris does to what is underneath it, and a deleted member weighs nothing.

    WHAT DEACTIVATION IS NOT is a separation. The two particles at the ends of
    a cut member are still the same two particles, so a beam cut in the middle
    of a frame does not fall out of it -- it hangs from whatever else those
    nodes are attached to, and if nothing else is, the nodes keep the member's
    mass and move under gravity as one piece. The fragment detaches only when
    the PARTICLE is duplicated and the surviving elements are re-pointed at
    the copy (Cai et al. 2023 §4.2), which is the next stage and the reason
    `on_fracture` exists: it is the seam that stage plugs into, not a
    convenience for logging.
    """

    def __init__(self, n_elements: int, layout: Layout, threshold: float = 1.0,
                 on_fracture: Optional[Callable[[np.ndarray, float], None]] = None,
                 over: Optional[np.ndarray] = None):
        self.latch = Latch(n_elements, layout, threshold, over=over)
        self.active = np.ones(n_elements, dtype=bool)
        self.on_fracture = on_fracture
        #: (time, element) of every fracture, in the order they happened
        self.events: List[Tuple[float, int]] = []

    @property
    def layout(self) -> Layout:
        return self.latch.layout

    @property
    def fac(self) -> np.ndarray:
        return self.latch.fac

    @property
    def n_fractured(self) -> int:
        return self.latch.n_fractured

    def update(self, point_failure: np.ndarray,
               kinematic: Optional[np.ndarray] = None,
               t: float = 0.0) -> np.ndarray:
        newly = self.latch.update(point_failure, kinematic)
        if len(newly):
            self.active[newly] = False
            self.events.extend((float(t), int(e)) for e in newly)
            if self.on_fracture is not None:
                self.on_fracture(newly, float(t))
        return newly
