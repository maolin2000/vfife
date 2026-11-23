"""The one monitor record every solver keeps.

A monitor is a scalar time history extracted during the run; `output.monitors`
in the case file, io/schema.MonitorSpec, says what each kind reads. Until
1.17 each of the four solver modules had its own copy of this dataclass, with
the fields its own monitors used; one record with all of them means a field
added for one solver is there for the others, and nothing reads a field one
copy lacked.

    reaction, displacement   reduce over the NODES of `regions`
    element                  one per-member (or per-element) field over the
                             members a region owns, reduced by `reduce`
    bar                      one per-segment field over a reinforcement set
    fiber                    one fibre's strain or stress, at `at`
    vehicle                  a vehicle's field (`of`, `field`, `axle`)
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class Monitor:
    kind: str
    regions: List[str]
    dof: int
    # everything else by keyword: the four copies had their fields in four
    # orders, and a positional call written for one would fill another's
    # field silently
    _: dataclasses.KW_ONLY
    # `field` is the case file's own key, so it keeps the name here too --
    # which shadows dataclasses.field, hence the qualified calls
    values: List[float] = dataclasses.field(default_factory=list)
    #: element, bar, fiber, vehicle monitors: the field read
    field: str = ""
    #: element and bar monitors: how the values reduce (`max`, `min`, `mean`,
    #: `absmax`; each solver passes its own default)
    reduce: str = ""
    #: displacement monitors: the region whose mean is subtracted. Only the
    #: solid solver read it until v1.0
    relative_to: str = ""
    #: element monitors of the solid solver: the stress or strain component
    component: Optional[str] = None
    #: vehicle monitors: the vehicle's name and the axle;
    #: bar monitors: the reinforcement set
    of: str = ""
    axle: int = 0
    #: fiber monitors: the fibre at [y, z], "concrete" or
    #: "steel" fibres only, and what was read -- element, station, the
    #: fibre's coordinates, kind and law, written beside the series
    at: Optional[List[float]] = None
    fiber_kind: str = ""
    read: Optional[dict] = None


def from_spec(spec, reduce_default="") -> Monitor:
    """A monitor from its `output.monitors` entry. Element and bar monitors
    may name no region at all: then they reduce over the whole model."""
    return Monitor(spec["kind"],
                   spec.get("regions") or ([spec["region"]] if spec.get("region") else []),
                   {"x": 0, "y": 1, "z": 2}[spec.get("dof", "z")],
                   field=spec.get("field") or "",
                   reduce=spec.get("reduce") or reduce_default,
                   relative_to=spec.get("relative_to") or "",
                   component=spec.get("component"),
                   of=spec.get("of") or "",
                   axle=int(spec.get("axle") or 0),
                   at=spec.get("at"),
                   fiber_kind=spec.get("fiber_kind") or "")
