# The material-point protocol

`vfife/materials/points.py` manages the life a material shares on its way from the user's parameters to an element's integration points. The constitutive equations stay in each material's module; the protocol unifies the kinematics, the state, the binding to the mesh, the backend and the restart contract.

## Initialisation

```python
state, contract = initialise_points(
    material,
    n_points,
    kinematics="solid_3d",       # or uniaxial
    backend="numpy",
    char_length=point_lengths,
)
```

Before a backend is set up it checks:

- that the material supports the kinematics and the backend;
- that the number of points is valid;
- that the state fields' slices are complete, do not overlap and stay in range;
- that `init_state()` returns exactly `(n_points, n_state)`;
- that the characteristic length is per point, finite and positive;
- that a regularised material does get its characteristic length;
- that the material gives a finite, positive wave speed.

## Components

Solids use engineering Voigt strains

```text
[exx, eyy, ezz, gxy, gyz, gxz]
```

and the symmetric Cauchy stress

```text
[sxx, syy, szz, sxy, syz, sxz]
```

Bars and reinforcement declare `uniaxial`, with the physical components `axial_strain` and `axial_stress`. The device buffers underneath may still pack six components, but that implementation detail is no longer taken for a three-dimensional material capability.

## State and restart

`MaterialPointContract` records:

- the material's name and kinematics;
- the stress and strain components;
- the state width and fields;
- the backend;
- the characteristic-length policy;
- the wave speed and the stable time step it gives;
- a signature of the state layout.

The checkpoints of solids and stand-alone bars save `material_signature`. A restore checks the signature, the size of the state arrays and that they are finite, so the state of another material layout cannot be loaded into the model silently. An old checkpoint without a signature is still read, but must pass the size check.

## Where it is used

- every solid integration point is initialised through it;
- stand-alone bars' points with `uniaxial` kinematics;
- embedded reinforcement's points with `uniaxial` kinematics;
- the crack-band characteristic length of CDP is bound through it;
- the capabilities of NumPy, Numba and Warp are checked before state is allocated or a device started;
- `stable_time_step(length)` gives the local limit from the material's wave speed.

The elastic beam uses a linear elastic section stiffness and calls no three-dimensional material update; the registry restricts its materials to `linear_elastic`. A fibre section holds many `uniaxial` material points at the section level, rather than letting the beam call a three-dimensional law.

## A new material

A new material must:

1. declare `solid_3d`, `uniaxial` or a future kinematics;
2. name every state column;
3. for a regularised model, declare `regularised=True` and bind the characteristic length point by point;
4. give a wave speed or equivalent stability information;
5. pass the parity tests on the same path and step for every backend it declares;
6. verify the monotonic, unloading, reversal, multiaxial, energy and restart paths;
7. not claim a capability it lacks by swallowing parameters with `**ignored`.
