# Moving loads

A load that travels along the structure: a vehicle crossing a bridge, a crane
running along a girder, an axle group walked across for an influence line.
Added in v0.3; `vfife/core/moving_load.py`.

```yaml
loads:
- region: deck
  moving_load:
    axles: [[0.0, 1.4e+5], [-4.2, 1.4e+5], [-5.6, 1.0e+5]]
    start: -6.0
    speed: 22.2                 # m/s -- a crossing in physical time
    direction: [0, 0, -1]       # the default
```

## What it is

Every other load in the library is a fixed distribution of nodal forces that
an amplitude scales. A moving load is not: its magnitude never changes, and
what changes is **where** it acts — which between two nodes is nowhere the
mesh has a degree of freedom. So it is two things:

* a **path** — the members of `region`, ordered end to end, with arc length
  `s` along the undeformed geometry; and
* the **consistent distribution** of a point force at `s` onto the one member
  that contains it: the Hermite shape functions, which give the exact nodal
  deflection of a prismatic beam for a load anywhere between its nodes, and
  which integrate to the `line_load` end actions (`qL/2`, `±qL²/12`) when a
  uniform load is summed over the member.

The cheap alternative — the lever rule, `1-ξ` and `ξ` with no end moments — is
a different structure. With the load in the middle of a member it loses 27% of
the midspan deflection on one element per half span, and it is never exact at
any refinement. It is not offered.

## The fields

| key | meaning |
|---|---|
| `axles` | `[[offset, force], ...]`. `offset` is the axle's position along the path **relative to the group's reference point**, so a trailing axle 4.2 m behind a lead axle is `-4.2` whichever way the group travels. `force` is the magnitude along `direction`. |
| `start` | where the reference point is at the start of the step. Negative puts the group off the near end, so it drives on. |
| `speed` | m/s: `s = start + speed·t`. A crossing in physical time; needs `solver.mode: dynamic` to mean anything. |
| `stop` | instead of a speed: `s` sweeps `start → stop` over the step's progress. With `load_steps: n` under relaxation that is `n` static positions — an influence line, with no dynamics at all. |
| `direction` | unit vector the force pushes along (normalised on read). Default `[0, 0, -1]`. |
| `region` | the members the load travels. They must form **one open chain**; a fork, a ring or two disconnected runs is refused by name rather than traversed in mesh order. |
| `amplitude` | as for any load, but the default is `hold` rather than the step's `ramp`/`ramp_hold`: a vehicle that fades in over the first half of the crossing is not what anyone means. |

`s = 0` is the end of the chain that comes first in `(x, y, z)`, so a bridge
laid out along `+x` is entered at its left end by a positive `speed`. An axle
whose station is off the path carries nothing, which is how a group drives on
at one end and off at the other.

Exactly one of `speed` and `stop` is required; `speed: 0` is refused (use a
`vector` load). In a multi-step dynamic run each step's clock starts at zero,
so `start` is per step.

## What it is not

A moving **force**, not a moving **mass**. The vehicle has no inertia of its
own, does not bounce on its suspension, and does not follow the deck down as
it deflects: the force keeps its direction and its station on the undeformed
path. That is the standard first model of a vehicle on a bridge — and the one
whose closed-form solution exists to check against. Vehicle–bridge
interaction needs the vehicle as a model of its own and is not here.

It is available on `beam2`, `beam2_timoshenko` and `beam2_fiber`, which is to
say wherever a line of members can be walked along.

## Verification

* `cases/case_26_moving_load_beam.yaml` — 100 kN at 40 m/s over a 20 m beam
  with `f1 = 2.000 Hz` (speed parameter `α = 0.50`), against Frýba's modal
  series. Inside 0.3% over the whole crossing, 0.05% at the peak. The peak is
  **1.70×** the static `PL³/48EI` and arrives a third of the crossing *after*
  the load passed midspan: the beam is vibrating, not tracking the load.
* `cases/case_26b_moving_load_influence.yaml` — the same axle walked across
  in 39 quasi-static stations, half of them between nodes, against
  `P a (3L² − 4a²)/48EI`. Worst relative error over the line: `1.3e-4`.
* `cases/case_28_confederation_trucks.yaml` — the same machinery on a real
  bridge: two trucks (379 and 346 kN) walked the 440 m girder of the
  Confederation Bridge's P31-P32 portal frame in 22 stations, against the
  deflection profile measured in the 1997 load test. 17.0 mm at midspan
  against 17.1 measured.
* `tests/test_moving_load.py` — the distribution against the `line_load` end
  actions, the influence line station by station to `1e-9`, the path's
  refusals, and what the lever rule would have cost.

```
python benchmarks/generate_moving_load_beam.py --check
python -m vfife run cases/case_26_moving_load_beam.yaml
python benchmarks/plot_moving_load.py
```
