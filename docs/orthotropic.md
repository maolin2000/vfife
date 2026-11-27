# M01 Orthotropic linear elastic

`orthotropic` is a three-dimensional orthotropic linear elastic material, for directional members such as timber, rolled plate and single-ply composites. Continuum elements only (`hexa8/tet4/tet10/wedge6/pyramid5/hexa20/wedge15`), `numpy` backend.

## Input

```yaml
materials:
  laminate:
    type: orthotropic
    E1: 12.0e9
    E2: 8.0e9
    E3: 10.0e9
    G12: 4.5e9
    G23: 3.8e9
    G13: 4.0e9
    nu12: 0.3
    nu13: 0.25
    nu23: 0.28
    rho: 800
```

Nine independent constants (`E1-3, G12/G23/G13, nu12/13/23`); the symmetric compliance matrix is completed automatically, and it is checked for positive definiteness before it is inverted (`E>0, G>0, |nu|<0.5`, determinant > 0). The material axes are the global axes for now; rotating them through `orientation` is planned.

In the isotropic limit (`E1=E2=E3, G=E/2(1+nu), nu12=nu`) it agrees with `linear_elastic` to machine precision (`test_isotropic_limit_matches_linear_elastic`).

Qualified `verified` (`qualification/orthotropic.yaml`), `numpy` only; no failure criterion and no laminated shells yet.
