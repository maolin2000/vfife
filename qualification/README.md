# Library qualification

Registration and qualification are deliberately separate. A registered model
can be selected by the schema; a qualified model has machine-checkable evidence
for a declared use level.

## Levels

| Level | Meaning |
|---|---|
| `registered` | Metadata and implementation exist; no engineering accuracy claim |
| `verified` | Required mechanics/constitutive gates have project test evidence |
| `benchmarked` | Verified plus convergence, dynamics/restart where applicable, and an external reference |
| `production` | Benchmarked plus every declared backend, controlled failure paths and restart coverage |

The checker never silently lowers the requested level. Missing evidence returns
`INCOMPLETE` and exit code 2.

## Current machine-checked manifests

| Model | Kind | Current manifest level | Status |
|---|---|---:|:---:|
| hexa8 | element | verified | PASS |
| beam2 | element | verified | PASS |
| beam2_timoshenko | element | verified | PASS |
| connector2 | element | verified | PASS |
| Chaboche | material | verified | PASS |
| CDP | material | verified | PASS |
| piecewise_uniaxial (M02) | material | verified | PASS |
| shell4 (E03) | element | verified | PASS |

The remaining registered models (`tet4`, `tet10`, `wedge6`, `truss2`,
`mixed_line`, `coupled`, `linear_elastic`, `j2`, `rebar`) retain their existing
tests and verification cases, but have not yet been curated into complete
qualification manifests. They must not be presented as having passed this new
admission process until those manifests exist and pass.

Generate and check evidence manifests with:

```powershell
vfife qualify template element truss2 --level verified -o qualification/truss2.yaml
vfife qualify check qualification/truss2.yaml
vfife qualify check qualification/truss2.yaml --format json
```

Use `--force` only when intentionally replacing an existing template.
