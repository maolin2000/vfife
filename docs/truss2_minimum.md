# Standalone `truss2` engineering workflow

`truss2` is now an independent two-node axial element, distinct from the
existing embedded reinforcement.  It imports Gmsh `Line2` elements and takes
its support/load regions from Gmsh Physical Point or Physical Curve groups.

The current engineering version supports a constant cross-sectional area, a
`linear_elastic` material, three translational DOFs at each node, total or
per-node `vector` loads, displacement constraints, named table/cyclic
amplitudes, multiple analysis steps, NumPy relaxation/dynamic execution, and
XDMF output. `body_force` is integrated as `b A L / 2` at each end of a
complete selected bar; `gravity` uses `rho g A L / 2`. `axial_stress`,
`axial_force`, and `von_mises` (the absolute axial stress for this uniaxial
element) are cell fields.

Additional equipment, deck, or payload inertia is written at model level:

```yaml
nodal_masses:
  - {region: apex, mass: 5000.0, per_node: true} # kg at each selected node
  - {region: floor_nodes, mass: 12000.0}         # kg total, shared by the region
```

These masses enter dynamic inertia, `ground_motion` equivalent force, and the
relaxation mass. They are translational lumped masses, repeated in x/y/z.

`truss2` deliberately does not yet support Warp, explicit line traction/pressure,
nonlinear bar laws, or mixed solid/truss meshes.  The separate `beam2` section
below documents the currently supported linear frame path.

## 3-D frame member (`beam2`)

`beam2` uses the same Gmsh `Line2` mesh and named Physical Point/Curve groups
as `truss2`, but creates a linear Euler--Bernoulli space-frame member with six
DOFs per node: `[ux, uy, uz, rx, ry, rz]`.  Select it in YAML and provide the
section properties `area`, `iy`, `iz`, `j`, plus an `orientation` vector that
defines the member local-y direction.  Nodal `vector` forces and `moment`
loads, six-flag constraints, end rotational releases such as `[i_rz, j_rz]`,
linear static solution, and explicit dynamic motion are supported on NumPy.
Uniform beam loads use `line_load: [qx, qy, qz]` in N/m and optionally
`coordinate_system: local`; the default is global. Output may request signed
end actions such as `bending_z_i`, `bending_z_j`, `shear_y_i` and `shear_y_j`.

```powershell
python -m vfife run cases/case_01_beam_cantilever.yaml
```

The included cantilever is checked against `u = PL^3/(3 E Iz)`.  `truss2` and
`beam2` now enter through `core.structural.make_structural_solver`, sharing
their case/YAML, regions, amplitudes, nodal mass, earthquake and output
contracts. Mixed truss--beam meshes assign bar and beam sections to their line groups (the `mixed_line` path), one
assignment for every Gmsh Line2 group; see numbered case 03.
Plastic hinges, nonlinear bars and GPU frame kernels are not claimed yet.

## 3-D shear-flexible frame member (`beam2_timoshenko`)

Use `beam2_timoshenko` for short/deep members where transverse shear is not
negligible. It has the same mesh, six-DOF constraints, orientation, releases,
loads and outputs as `beam2`, plus:

```yaml
sections:
  short_beam:
    type: beam
    material: steel
    theory: timoshenko
    shape: general               # or {shape: rect, width: 0.1, height: 0.1}: area, iy, iz, j, ay, az derived
    area: 1.0e-2
    iy: 1.0e-4
    iz: 1.0e-4
    j: 2.0e-4
    ay: 8.0e-3
    az: 8.0e-3
    kappa_y: 0.8333333333333333
    kappa_z: 0.8333333333333333
assign:
  - {region: all, section: short_beam, orientation: [0, 0, 1]}
```

`ay`/`az` are shear areas before correction. The effective rigidities are
`kappa_y*G*ay` and `kappa_z*G*az`; do not include the correction twice.
Run and independently verify the supplied short-beam case with:

```powershell
python -m vfife run cases/case_08_timoshenko_short_beam.yaml
python -m vfife verify cases/case_08_timoshenko_short_beam.yaml
```

Run the analytical one-bar verification:

```powershell
python -m vfife run cases/case_02_truss_bar.yaml
python benchmarks/evaluate_truss_bar.py results/case_02_truss_bar.h5
```

The 1 m bar has `E=200 GPa`, `A=100 mm²`, and `P=1 kN`; the target is
`u_x=PL/(EA)=5.0e-5 m`, axial stress `10 MPa`, and support reaction `-1 kN`.

## Space truss under recorded ground motion

`cases/case_05_space_truss_seismic.yaml` is a four-leg three-dimensional pyramid
truss. Its base is fixed and its relative displacement is driven by the public
El Centro 1940 N--S acceleration record in
`benchmarks/assets/elcentro_1940_ns_g.dat`. Run it with:

```powershell
python -m vfife run cases/case_05_space_truss_seismic.yaml
python benchmarks/evaluate_space_truss_elcentro.py --report results/case_05_space_truss_seismic_report.json
```

`ground_motion` is dynamic-only. It expects a two-column `(seconds,
acceleration)` text file, a direction vector, and a scale; its force is
`-M a_g(t)` in relative coordinates. The same input is available for solid and
truss models on the CPU paths. The record asset is in g, hence its
`scale: 9.80665`.

The record is the publicly posted [El Centro 1940 N--S history](https://www.vibrationdata.com/elcentro.dat);
its provider notes the two columns are seconds and g, and that different El
Centro versions exist. The model uses this supplied version unchanged and
compares its peak apex response against an independent linear Newmark-beta
calculation.

## Unified kinematic boundary conditions

Constraints default to displacement, preserving existing cases. Dynamic cases
may instead prescribe velocity or acceleration; the value may be scalar or a
three-vector and can use every existing amplitude table/cyclic history:

```yaml
constraints:
  - {region: actuator, dofs: [1, 0, 0], motion: displacement, value: 0.02}
  - {region: shaker, dofs: [1, 0, 0], motion: velocity, value: 0.15, amplitude: drive}
  - {region: table, dofs: [0, 0, 1], motion: acceleration, value: 9.80665, amplitude: pulse}
```

One DOF must have one kinematic prescription in a step. A velocity constraint
is integrated once, acceleration twice, from the state at the step start.
Velocity and acceleration are currently CPU-only; `backend: warp` rejects them
explicitly rather than producing an inconsistent device boundary condition.

## Rayleigh damping

Dynamic `truss2` supports the initial-stiffness Rayleigh model
`C = alpha*M + beta*K0` on CPU:

```yaml
solver:
  mode: dynamic
  damping: {type: rayleigh, alpha: 0.35, beta: 1.0e-4}
```

`alpha` has units 1/s and `beta` seconds. It is especially useful for the
space-truss seismic model. Solid CPU models currently support the mass term
(`alpha`) only; a non-zero solid `beta` and all Warp Rayleigh requests are
rejected explicitly until a matrix-free solid initial-stiffness action is
available.
