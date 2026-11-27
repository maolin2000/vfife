# vfife

[![CI](https://github.com/maolin2000/vfife/actions/workflows/ci.yml/badge.svg)](https://github.com/maolin2000/vfife/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/maolin2000/vfife)](https://github.com/maolin2000/vfife/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24594822.svg)](https://doi.org/10.5281/zenodo.24594822)

vfife is a structural analysis program for bridges, based on the vector form intrinsic finite element method (VFIFE). It integrates the motion of every point explicitly, so large displacements, yielding, member fracture, falling debris and contact need no special treatment. The same model also gives the static dead-load state, construction stages, modes and buckling loads.

A model is one YAML file, run from the command line or the Python API. Meshes come from the case file, Gmsh or a converted Midas Civil model; results are written as XDMF/HDF5 for ParaView.

**Version 1.23.2** · MIT licence · [CHANGELOG](CHANGELOG.md). The case, result and checkpoint formats and the Python API have been stable since 1.0.

## Features

| | |
|---|---|
| Elements | Euler–Bernoulli and Timoshenko beams; fibre beams with concrete and steel laws; bars and cables; connectors and bearings; shells; solids with embedded or bonded reinforcement and tendons |
| Statics | dynamic relaxation, Newton statics, arc length, cable form-finding, construction stages with members built and removed |
| Time-dependent | creep, shrinkage and tendon relaxation with each segment's own age; temperature |
| Dynamics | uniform and multi-support ground motion, response spectrum, moving loads and sprung vehicles on rough decks, friction bearings and dampers, p-y soil springs, added water mass |
| Collapse | member and fibre fracture with separation, falling spans, contact between members and with the ground |
| Stability | modes about a loaded state, linear and lateral-torsional buckling |
| Interfaces | Midas Civil conversion, Gmsh meshes, ParaView output, two read-only web viewers (`vfife view`) |

## Installation

```bash
pip install "vfife-1.23.2-py3-none-any.whl[cpu]"     # from the Releases page; [cpu] adds numba
```

From source:

```bash
git clone https://github.com/maolin2000/vfife.git
cd vfife
pip install -e ".[dev,plot,cpu]"
```

Python 3.10 or newer. Optional extras: `[cpu]` (numba, needed for models of any size), `[gpu]` (Warp/CUDA, for large solid models), `[plot]` (matplotlib). `environment.yml` defines the equivalent Conda environment.

## Quick start

```bash
vfife new girder                # write a three-span girder case
vfife info girder.yaml          # check the model before solving
vfife run girder.yaml           # dead-load state
vfife summary results/girder.h5
vfife modal girder.yaml         # natural modes
vfife view results/girder.h5    # replay the result in a web page
```

The case written by `vfife new girder`:

```yaml
version: 1
name: girder
units: {length: m, force: N, stress: Pa}
mesh:
  type: lines
  points:
    A: [0.0, 0.0, 0.0]
    B: [30.0, 0.0, 0.0]
    C: [70.0, 0.0, 0.0]
    D: [100.0, 0.0, 0.0]
    mid_main: [50.0, 0.0, 0.0]
  lines:
    side_1: {through: [A, B], size: 2.0}
    main: {through: [B, mid_main, C], size: 2.0}
    side_2: {through: [C, D], size: 2.0}
  groups:
    girder: [side_1, main, side_2]
materials:
  steel: {type: linear_elastic, E: 2.06e+11, nu: 0.3, rho: 7850.0}
sections:
  box: {type: beam, material: steel, shape: general, area: 0.35, iy: 1.2, iz: 6.0, j: 2.0}
assign:
  - {region: girder, section: box, orientation: [0, 1, 0]}
constraints:
  - {region: A, dofs: [1, 1, 1, 1, 0, 0]}
  - {region: B, dofs: [0, 1, 1, 1, 0, 0]}
  - {region: C, dofs: [0, 1, 1, 1, 0, 0]}
  - {region: D, dofs: [0, 1, 1, 1, 0, 0]}
loads:
  - {region: all, gravity: [0.0, 0.0, -9.80665]}
  - {region: girder, line_load: [0.0, 0.0, -6.0e+4], with_mass: true}
solver: {mode: static, backend: numpy}
output:
  path: results/girder.h5
  monitors:
    w_main: {kind: displacement, region: mid_main, dof: z}
    R_B: {kind: reaction, region: B, dof: z}
```

For a Gmsh mesh, `vfife mesh file.msh` lists its contents and `vfife init` writes a first case around it; `vfife schema` gives editors completion for case files. From Python:

```python
from vfife.api import run_case
result = run_case("girder.yaml")
print(result.status, result.monitors["w_main"][-1])
```

## Documentation

- [User manual](docs/user_manual.md): modelling, the case file, the command line, a complete worked bridge
- [Tutorial](docs/tutorial.md): one analysis from Gmsh mesh to hand check
- [YAML reference](docs/yaml_reference.md), [API reference](docs/api_reference.md), [error codes](docs/error_codes.md): generated from the code
- [Reference cases](cases/README.md): The 67 reference cases and what each is checked against
- [Technical notes](docs/technical_notes.md): design decisions and verification records

## Verification

Every reference case has a contract under `verification/` with the values it must reproduce and their tolerances, checked before each release. The references include closed-form solutions (elastica, Irvine's catenary, Frýba's moving load, Yang and Yau's sprung mass, Lee's frame, the 45° bend), laboratory tests (Tanaka and Park's column, Vecchio and Emara's frame, the Alberta and Chalmers beams, hollow-core slabs loaded to failure), measured deflections of the Confederation Bridge and published analyses of the Hangzhou Bay Bridge's 70 m girders. Compiled paths are tested to give the same results as the reference loop to the bit, and a restart the same as an uninterrupted run. `vfife probe case.yaml` checks a user's model for symmetry of the tangent, free rigid-body motion and stability of the time step.

## Limitations

- Beams have six degrees of freedom per node: no warping, so open thin-walled sections in non-uniform torsion and distortion of box girders are not modelled.
- Shear in fibre sections is a member-level law, not a field theory.
- Soil springs are monotonic p-y curves.
- Contact between members uses capsules of a given radius.
- Explicit time steps are small; static and long-duration problems use relaxation, Newton statics or analysis steps.
- No aerodynamics, fluids, soil continua or design-code checks.

The full list is in section 16 of the user manual.

## Citation

Cite the version used: each release has its own DOI, listed on the [Zenodo record](https://doi.org/10.5281/zenodo.24594822); the DOI below stands for all versions. `CITATION.cff` has the metadata.

```bibtex
@software{tian_vfife,
  author  = {Tian, Maolin},
  title   = {vfife: vector form intrinsic finite element analysis of bridges},
  version = {1.23.2},
  year    = {2026},
  doi     = {10.5281/zenodo.24594822},
  url     = {https://github.com/maolin2000/vfife}
}
```

The method: E. C. Ting, C. Shih and Y.-K. Wang, "Fundamentals of a vector form intrinsic finite element: Part I. Basic procedure and a plane frame element", *Journal of Mechanics* 20(2), 113–122, 2004.

## Licence

MIT; see [LICENSE](LICENSE).
