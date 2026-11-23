# Temperature

A temperature change over a member's section. Added in v0.4;
`vfife/core/temperature.py`.

```yaml
materials:
  concrete: {type: concrete02, ..., thermal_expansion: 1.0e-5}   # 1/K
loads:
- {region: deck, temperature: {uniform: 15.0}}
- {region: deck, temperature: {gradient: [12.5, 0.0]}}          # K per metre of depth
- {region: deck, temperature: {profile: [[-0.9, 0], [-0.5, 0], [0.5, 21]], axis: y}}
```

## Temperature is not a force

It is a **free strain** the section would take on if nothing held it,
`eps_th(y, z) = alpha T(y, z)`. What it produces depends entirely on what is
in the way:

| what is in the way | what happens |
|---|---|
| nothing (a determinate member, a linear `T`) | the member changes shape and carries **no stress at all**: it grows by `alpha dT L` and bows to `kappa = alpha dT/dy` |
| a restraint or a redundant support | the part of the free strain the structure refuses becomes force: `N = EA alpha dT` held at both ends, secondary moments in a continuous beam |
| a **nonlinear** `T` through the depth (a code's design profile) | even a simply supported beam cannot take that shape, because plane sections stay plane. The section keeps the linear part and turns the rest into **self-equilibrated stress** — zero net force, zero net moment, real stress |

That third row is why a bridge code specifies a curved gradient, and why it
needs a fibre section: a `general` section has no fibres to hold a stress
whose resultants are zero.

## How it is implemented, and why it matters

The member force is `f = k (d - d_th)`: the free thermal deformation is taken
**out of the deformation** rather than added as a load.

The alternative — equivalent nodal forces, `F = ∫ Bᵀ D eps_th dV` — gives the
same displacements and the **wrong member actions**. A free beam under a
uniform `dT` comes out the right length, and the stretch read back through the
section law reports `N = EA alpha dT` on a member that carries nothing. The
fix in that form is to remember to subtract the thermal part again at stress
recovery, everywhere, forever. Taking it out of the deformation instead leaves
nothing to remember: the actions a member reports are the ones it carries.
(`benchmarks/generate_thermal_beam.two_span_solution` fell into exactly that
trap while being written as an independent check, and read `EI kappa / 2`
instead of `3 EI kappa / 2`.)

On a fibre section the same idea goes one level down: each fibre's material is
handed its **mechanical** strain, the total less `alpha T` at that fibre. A
nonlinear profile then produces its eigenstresses by itself, with no special
case anywhere.

## The fields

| key | meaning |
|---|---|
| `uniform` | K over the whole section |
| `gradient` | `[dT/dy, dT/dz]` in **K per metre** along the section's own axes (a scalar is the gradient along `axis`: `[g, 0]` by default, `[0, g]` with `axis: z`). Not "top minus bottom": a `general` section states area and second moments and has no depth to divide by. |
| `profile` | `[[s, T], ...]` through the section, linear between the points and constant beyond its ends — a code's design profile written literally. Mutually exclusive with `uniform`/`gradient`. |
| `axis` | which section axis `profile` -- or a scalar `gradient` -- runs along, `y` (default) or `z`. `rect`/`box` have their depth on **z**; a `patches` section on whichever axis its corners put it. Refused with a two-component `gradient`. |
| `thermal_expansion` | on the **material**, not the load: `1/K`, default 0. Spelled out because `alpha` is already `bond_ceb`'s curve exponent. Concrete ≈ 1.0e-5, steel ≈ 1.2e-5. A fibre section takes each fibre's own. |

`region` selects the members, as for `line_load`. The amplitude is the step's
own default (`ramp` staged, `ramp_hold` continuous), because a temperature
really is applied gradually — and because dropping a whole field onto a
structure at rest leaves kinetic damping resetting every iteration and
creeping towards the answer.

A `profile` on a section without fibres is **refused**, not reduced: recovering
its mean and slope needs the section's area distribution, and inventing one
would be inventing the section.

## What it does not do

No coupling back the other way: nothing in the library generates heat, conducts
it, or computes a section's temperature from an environment. The field is an
input. Material properties do not change with temperature (no `E(T)`, no fire).

A sustained temperature on a material with `creep` **does** relax: the thermal
strain and the creep strain ride in the same imposed-strain slot (v0.4),
so a restrained fibre bar heated by 20 K and held from 28 to 1028 days loses
about 70% of its restraint force with φ → 2.

A temperature acts in the step that states it and, if it is at model level, in
every step after it; a step-level temperature ends with its step (the fibre
solver kept it until v1.0). Fibres whose law states no `thermal_expansion`
expand by zero; preflight warns (`THERMAL_EXPANSION_PARTIAL`) when a heated
fibre section mixes laws with and without it.

## Verification

* `cases/case_27_thermal_two_span.yaml` — two 20 m spans. A uniform +40 K with
  the far end free: grows 19.198 mm (exact 19.200) and the axial reaction stays
  at 0.05 N against the 10 080 kN it would have been if held. A 15 K/m gradient:
  `M = 3EI kappa/2 = 55.0 kN m` at the middle support, the ends lifting 2.750 kN,
  midspan 2.2511 mm (exact 2.2500) against the 9.0 mm a single span would bow.
* `cases/case_27b_thermal_eigenstress.yaml` — a code-shaped nonlinear profile
  on a determinate fibre beam: −2.204 MPa at the heated top, +1.835 MPa just
  under the heated band, −1.504 MPa at the soffit, all within 0.05% of the
  closed form, with the net force and moment at 1e-5 and 4e-4 of what it would
  take to suppress the field.
* `cases/case_28b_confederation_thermal.yaml` — a real bridge under a recorded
  summer profile: the Confederation Bridge's 250 m portal frame rises 58.2 mm
  at midspan against the 60.6 published and drops 43.9 at x = -180 m against
  43.7, with the field resolved on each section's own depth (a `general`
  section cannot hold the nonlinear remainder, so it is reduced outside the
  solver by `benchmarks/generate_confederation.py`). `case_28c` takes the
  gradient back out and says what it was worth: 9 mm at midspan, 31 at every
  cantilever tip.
* `tests/test_temperature.py` — the free member's own reported actions, the
  restrained `EA alpha dT` to 1e-9, the curvature read back through the
  element's own shape-derivative rows, the refusals, and both kernels against
  their reference paths.

```
python benchmarks/generate_thermal_beam.py --check
python -m vfife run cases/case_27_thermal_two_span.yaml
```
