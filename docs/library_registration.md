# Registering elements and materials

The element library, the material library, preflight and the command line share one machine-readable statement of what each model can do. It does not replace the numerical implementation; it lets a new model answer, before it reaches a solver, what it combines with, what parameters it needs, what results it gives and how far it has been verified.

## Materials

A material is registered with `@register`, with its engineering metadata:

```python
@register(
    "my_material",
    kinematics=("solid_3d",),
    backends=("numpy",),
    analysis_modes=("dynamic", "relaxation"),
    output_fields=("stress", "strain", "history"),
    nonlinear=True,
    regularised=False,
    verification="constitutive_path",
    caveat="...",
)
class MyMaterial(Material):
    ...
```

The constructor's parameters go into `MaterialDef.parameters` and `state_names` into the state fields automatically. No other list of material names is kept, in the command line or anywhere else.

## Elements

`ElementCapability` describes solid and line elements alike:

- family and mesh topology;
- nodal degrees of freedom;
- formulation;
- material kinematics and the compatible materials;
- backends and analysis modes;
- loads and output fields;
- the geometric nonlinearity;
- verification level and known limitations.

A continuum element's interpolation and integration are defined by its `ElementDef`, from which its capability is generated; bars, beams, mixed and coupled elements use the same `ElementCapability` protocol.

Every element declares `geometric_nonlinearity: co_rotational_vfife`, so that a new element that departs from VFIFE's removal of rigid-body motion does not go unnoticed.

## Access and compatibility

```python
from vfife.catalog import library, compatibility

catalog = library()  # JSON-serialisable, schema_version = 1
ok, reasons = compatibility("beam2", "j2", backend="warp")
```

`preflight.check()` calls the same `compatibility()`: an element-material-backend-analysis combination that is not supported gives `CAPABILITY_MISMATCH` without building a solver or initialising CUDA.

The command line shows the same catalogue:

```powershell
vfife list
vfife list materials --format json
vfife describe beam2
vfife describe cdp --format json
vfife capabilities --element hexa8 --material cdp --backend warp
vfife capabilities --element beam2 --backend warp --show-unsupported
```

`capabilities` lists the supported combinations; `--show-unsupported` adds each refusal with its reason. All three commands read the registry only: they import no mesh, build no solver and do not initialise CUDA. `--format json` carries the catalogue's `schema_version` at the top, for scripts.

## A new model

A new element or material must:

1. register its full metadata (changing a `Literal` in the schema is not enough);
2. provide an implementation and a parity test for every backend it declares;
3. state its verification level (a model with unit tests alone is not `benchmarked`);
4. state its output fields and state fields and where they live;
5. record the combinations it does not support among its known limitations;
6. add catalogue-completeness and invalid-combination tests.

The compatible materials are listed explicitly by each element. This is deliberately conservative: a new material is not supported by every solid or line element on registration, but is let through explicitly once the combination is verified.

How a material lives at an element's integration points is in [material_points.md](material_points.md): the registry says which combinations are allowed, the integration-point protocol how such a combination gets a consistent state, characteristic length, stability information and restart identity.

How a model moves from registered to verified, benchmarked or production is in [qualification_gates.md](qualification_gates.md); a label in the registry is not, by itself, evidence.
