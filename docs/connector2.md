# Six-component connector (`connector2`)

Implemented 2026-08-27 as E02 of the common element-library roadmap.
`connector2` is a two-node, six-DOF/node spring and dashpot. It uses the same
Gmsh `Line2`, named regions, amplitudes, constraints, loads, monitors and HDF5/
XDMF workflow as the other structural members.

## Mechanical definition

The component order is always:

```text
[ux, uy, uz, rx, ry, rz]
```

`stiffness` and `connector_damping` accept either a six-value diagonal or a
symmetric positive-semidefinite 6x6 matrix. Translational rows produce force;
rotational rows produce moment. A local connector takes local x from the
Line2 chord and local y from `orientation`. A global connector retains global
component directions while still removing the mean rigid motion.

The element removes the rigid motion before measuring relative translation and
rotation. Its frame is the **geodesic mean of the two ends' orientation
matrices**, `T₁ exp(½ log(T₁ᵀT₂))` — the same mean nodal triad a beam member
uses (`rotations.beam_frames_from_triads`) — and the rotational deformation is
the spatial relative rotation `log(T₂T₁ᵀ)`. Until v0.1 both were
written on the rotation VECTORS, as their mean and their difference; those are
exact when the two ends have turned by equal or coaxial amounts, which is every
case the library had, and neither is a rotation otherwise. Generalized forces
are formed with the deformation Jacobian, taken along the **spins** the solver
integrates (`R ← exp(ω dt) R`) rather than along increments of
`log R`, so they are work-conjugate to what the equations of motion advance --
which is also what makes the dashpot term `C J v` the connector's real
deformation rate. A common finite rigid translation/rotation produces
zero connector force. This preserves the VFIFE compatible-description idea
instead of adding a small-angle global spring beside it.

## YAML examples

Independent diagonal components:

```yaml
sections:
  link:
    type: connector
    stiffness: [1.0e8, 2.0e8, 2.0e8, 5.0e5, 6.0e5, 6.0e5]
    connector_damping: [2.0e4, 3.0e4, 3.0e4, 100, 120, 120]
assign:
  - {region: all, section: link, coordinate_system: local, orientation: [0, 0, 1]}
```

A curve replaces the diagonal stiffness of its named component while retaining
any off-diagonal matrix coupling. Tables must include `[0, 0]`, use strictly
increasing deformation coordinates, and are linearly extrapolated with their
terminal slopes:

```yaml
sections:
  support:
    type: connector
    stiffness: [0, 0, 0, 0, 0, 0]
    curves:
      ux: [[-3.0e-5, -2000], [0, 0], [1.0e-5, 1000], [3.0e-5, 2000]]
assign:
  - {region: all, section: support}
```

A `curve` is reversible: it does not claim gap, friction, degradation or
hysteretic unloading. Those are `laws`.

## Path-dependent laws: gap, bilinear, friction

```yaml
sections:
  sliding:
    type: connector
    stiffness: [1.0e+10, 0, 0, 0, 0, 0]              # ux: the bearing's vertical spring
    laws:
      uy: {type: friction, mu: 0.05, normal: ux, stiffness: 1.0e+8}
  stop:
    type: connector
    laws:
      ux: {type: gap, lo: -0.006, stiffness: 2.0e+9}
  isolator:
    type: connector
    laws:
      uy: {type: bilinear, k1: 2.0e+7, k2: 2.0e+6, fy: 5.0e+4}
```

Each component carries either a stiffness, a curve or a law. A law owns one
scalar of history, the slipped (plastic) part of that component's
deformation, written as `connector_slip_<component>` and carried in the
checkpoint. `elements/connector.py` has the three return mappings:

* `gap`: no force inside `[lo, hi]` (either bound may be absent), `stiffness`
  times the overrun outside. A stop, an expansion joint, pounding, a
  compression-only seat on the local `ux` (`lo: 0`: `ux` is the elongation,
  so pressing the seat is `ux < 0`; `hi: 0` is a tension-only tie -- the
  docs said the opposite until v1.0).
* `bilinear`: `k1` to `fy`, then `k2`, kinematic hardening. The skeleton of
  a lead-rubber or friction-pendulum isolator and of a yielding device; one
  closed loop between `+-u_max` dissipates `4 (fy - k2 u_y) (u_max - u_y)`.
* `friction`: sticks with `stiffness` until the force reaches `mu N`, then
  slides at `mu N`. `normal: ux` takes N from the compression the local
  `ux` spring of the same connector currently carries -- so the sliding
  force follows the bearing's vertical load, and an uplifted bearing slides
  for free -- or `normal` gives a constant force. Only `ux`: every other
  component's sign flips with the order of the Line2's nodes, and a global
  connector cannot take a named normal at all.
* The SAME law on `uy` and `uz` is one law in the plane: it yields or
  slides in the direction of the resultant, on a circular surface
  (two independent laws made a square one, 1.414 mu N at 45
  degrees). Different laws on `uy` and `uz` stay independent.
* A `connector_damping` term on a `gap` component acts only while the gap is
  closed (the Kelvin pounding model).

The history is advanced from the COMMITTED slip and the current deformation,
so two evaluations of one deformation agree. In a static run the solvers
commit at equilibrium (every evaluation inside a load level is a trial
re-driven from the last converged state, the discipline the fibre materials
already follow); in a dynamic run they commit every step. `case_32` is the
shipped example: a girder on a fixed and a sliding bearing warming into an
abutment stop, every number against its closed form. tests/test_connector_laws.py
has the stick-slide-return of a Coulomb bearing, the parallelogram of a
bilinear loop, a gap's free travel, and a pushed block's stopping distance.

## Ground connections, mixed models and dynamics

A node-to-ground connection uses a Line2 anchor node constrained in all six
DOFs; case 09 is the supplied example. Keeping the anchor explicit makes its
reaction auditable and avoids hidden equations or implicit supports.

Within `mixed_line` or `coupled`, assign `type: connector2` to a Physical Curve
group in `members`. Connector nodes then share the same translations and,
where available, rotations as trusses, beams and solids. A point solid node
still cannot receive a rotational moment without a physical rotational
constraint/MPC; preflight preserves that warning.

`connector_damping` is physical force per physical relative velocity and is
active only for `solver.mode: dynamic`. It is rejected in relaxation because
relaxation velocity is numerical pseudo-time, not material time. Connectors
carry no mass: dynamic standalone models must supply `nodal_masses`. A pure
dashpot has no stiffness-based time scale and therefore also requires
`solver.target_dt`; the solver further limits that value using `2m/c`.

## Results and verification

Available cell fields include `connector_force_x/y/z`,
`connector_moment_x/y/z`, and
`connector_deformation_ux/uy/uz/rx/ry/rz`. Mixed/coupled output also carries
`connector_flag`.

```powershell
python -m vfife run cases/case_09_connector_support.yaml
python -m vfife verify cases/case_09_connector_support.yaml
python -m vfife qualify check qualification/connector2.yaml
```

Case 09 checks the inverse bilinear skeleton and support equilibrium. Unit
qualification additionally covers six components, full matrices, rigid motion,
local/global axes, terminal extrapolation, physical dashpots, fixed-anchor
ground modelling, and mixed/coupled assembly.
