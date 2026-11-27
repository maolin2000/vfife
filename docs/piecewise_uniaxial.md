# M02 General uniaxial piecewise-linear material

`piecewise_uniaxial` is the one-dimensional material point shared by bars, embedded reinforcement and fibre beams. It is not a three-dimensional continuum material and cannot be assigned to solid elements.

## Input

```yaml
materials:
  steel:
    type: piecewise_uniaxial
    E: 2.0e11
    rho: 7850
    tension: [[0.0, 2.0e8], [0.005, 2.6e8], [0.02, 3.2e8]]
    compression: [[0.0, 1.8e8], [0.005, 2.4e8], [0.02, 3.0e8]]
    hardening_mode: combined       # isotropic | kinematic | combined
    kinematic_fraction: 0.5        # combined only
```

Both tables are `[accumulated plastic strain, magnitude of the yield stress]`: the first plastic strain must be zero, the abscissae strictly increasing, the stresses positive and non-decreasing. `compression` may be left out, and the tension table is then used for it. Past the last point the last segment's slope is extrapolated.

`E` is both the initial elastic modulus and the unloading stiffness. `isotropic` expands the tension and compression yield limits, without a Bauschinger effect; `kinematic` translates the yield surface; `combined` shares the two by `kinematic_fraction`. The model is rate-independent plasticity: no viscosity, strength degradation, pinching, fracture or negative-stiffness softening.

## State, energy and restart

Each material point keeps the plastic strain, the accumulated plastic strain, the back stress, the previous strain and stress, and the external work density. Output fields:

- `axial_stress`, `axial_strain`;
- `eps_p`, `alpha`, `backstress`;
- `external_work`, `elastic_energy`, `hysteretic_energy`.

`hysteretic_energy = max(external_work - elastic_energy, 0)`, in J/m³; at the end of a closed cycle it equals the area of the stress-strain loop. All six state quantities are saved with the `truss2` checkpoint, and a restore checks the material point's signature and the array layout.

## Where it is used

- the stand-alone co-rotational `truss2` (NumPy, dynamic/relaxation);
- the underlying `EmbeddedTruss` calls it through the common uniaxial material-point protocol, for embedded reinforcement;
- `mixed_line` still has one global material block and cannot yet express "elastic beam sections, with different bar groups nonlinear each"; until members are given their own materials this is not let through for particular cases;
- springs, connectors and fibre beams can reuse the model, each once its own material-point ownership and state output are in place.

The qualification list is `qualification/piecewise_uniaxial.yaml`; the level is `verified`.

The cyclic paths of the three hardening rules, with their CSV and SVG, can be regenerated in the development environment:

```powershell
python benchmarks/m02_path.py
```
