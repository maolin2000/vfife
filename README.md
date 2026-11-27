# vfife

A structural solver for **bridges, from construction to collapse**. It
integrates the equations of motion of every point explicitly (the vector
form intrinsic finite element method, VFIFE), so large displacements, members
that yield, break and fall, and the contact that follows are the ordinary
case rather than a special one -- and the same model also gives the static
dead-load state, the modes and the buckling loads.

A case is one YAML file. There is no GUI, by design: the command line, the
Python API and the case file are one workflow.

Pre- and post-processing are left to other programs. A mesh is written in
the case file, made in Gmsh, or converted from a Midas Civil model; results
go to ParaView (XDMF/HDF5, and VTU for sweeps and mode shapes). vfife's own
pictures are limited to two read-only web pages, each one file:
`vfife view case.yaml` shows a frame model before it is run -- what each
support, connector, tendon and analysis step means to the solver, which no
mesher knows -- and `vfife view result.h5` replays a result with members at
their section size, breaks and monitors, for someone without ParaView. The
pages do not edit a model and are not meant to grow into a pre- or
post-processor.

**Version 1.23.0** ([CHANGELOG](CHANGELOG.md), MIT licence). The case
format, the result format, the checkpoint format and the Python API have
been frozen since 1.0: a case written for 1.0 runs unchanged.

## What it is for

| Question | What the model uses |
|---|---|
| The dead-load state of a cable-supported bridge | form-finding of the cables' unstressed lengths, relaxation or Newton statics, construction steps with members born and removed |
| Service: vehicles, temperature, time | moving loads and sprung vehicles on a rough deck, temperature as a free strain, creep, shrinkage and tendon relaxation with each segment's own age |
| An earthquake | uniform or multi-support ground motion, response spectrum, fibre-section piers that crack and yield, bearings with friction, gaps, dampers and friction pendulums, piles in p-y soil, added water mass |
| Collapse | members that reach their capacity and break off with their mass, fibre fracture, falling spans, contact between members and with the ground |
| Is the answer right | modes at a loaded state, linear buckling, lateral-torsional buckling, an energy ledger, and a contract per case that is checked on every release |

Beams (Euler-Bernoulli, Timoshenko, fibre sections with concrete and steel
laws), bars and cables, connectors and bearings, shells, and solids with
embedded or bonded reinforcement and prestressing tendons. The line elements
are where the work has gone; shells and solids are there for the member that
needs them (a hollow slab, a D-region, a reinforced-concrete test specimen).

## Install

```bash
pip install "vfife-1.23.0-py3-none-any.whl[cpu]"     # from the Releases page; [cpu] is numba
vfife --version
```

or from a checkout, for development:

```bash
git clone https://github.com/maolin2000/vfife.git
cd vfife
pip install -e ".[dev,plot,cpu]"
```

Python 3.10 or newer. `[cpu]` (numba) is what makes it fast and is assumed
by everything dynamic of any size; `[gpu]` (Warp/CUDA) serves large solid
models; `[plot]` is matplotlib for `vfife plot`. `environment.yml` is the
equivalent Conda definition.

## A first run

```bash
vfife new girder            # a three-span girder that runs as it stands
vfife info girder.yaml      # what the model is: members, axes, mass, supports
vfife run girder.yaml       # the dead-load state
vfife summary results/girder.h5
vfife modal girder.yaml     # its modes
vfife view results/girder.h5    # one web page that replays the result in 3-D
```

`vfife new` lists the other templates (a pier on piles, an arch, a hanging
cable). The case it writes is this, and every line of it is yours to change:

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

Meshes can also come from Gmsh (`mesh: {type: gmsh, path: ...}`; `vfife mesh
file.msh` says what is in one, `vfife init` writes a first case around it).
From Python:

```python
from vfife.api import run_case
result = run_case("girder.yaml")
print(result.status, result.monitors["w_main"][-1])
```

## Where things are written down

| | |
|---|---|
| [`docs/用户手册.md`](docs/用户手册.md), [`docs/user_manual.md`](docs/user_manual.md) | The user manual, Chinese and English, section for section: a complete worked bridge, every part of the case file, the command line |
| [`docs/tutorial.md`](docs/tutorial.md) | One analysis end to end in fifteen minutes, with the hand check |
| [`docs/YAML参考.md`](docs/YAML参考.md) | Every block and field of a case, generated from the code; `vfife schema` gives your editor completion |
| [`docs/API参考.md`](docs/API参考.md), [`docs/error_codes.md`](docs/error_codes.md) | The Python API and every error code, generated |
| [`cases/README.md`](cases/README.md) | The 67 reference cases and what each is checked against |
| [`docs/technical_notes.md`](docs/technical_notes.md) | Why each feature is built the way it is, and its verification record |
| [`CHANGELOG.md`](CHANGELOG.md) | What changed |

## What has been checked

Every case under `cases/` has a contract under `verification/` -- the numbers
it must reproduce and their tolerances -- and all of them are solved and
checked before each release. Among them: closed forms (the elastica, Irvine's
catenary, Fryba's moving load, Yang & Yau's sprung mass, Lee's frame, the
45-degree bend, free fall onto the ground), measured records (Tanaka &
Park's column, Vecchio & Emara's frame, the Alberta and Chalmers beams, Chen's
hollow-core slabs loaded to failure, the Confederation Bridge's deflections),
and published analyses of the Hangzhou Bay Bridge's 70 m girder.

The fast paths are held to the slow ones: the compiled step gives the bits of
the step-by-step loop, a restart gives the bits of the uninterrupted run, and
tests assert both. `vfife probe case.yaml` asks your own model the questions
that do not depend on the element -- is the force a gradient, is a rigid
motion free, does the mass bound the step.

## What it does not do

Written down rather than discovered; Section 16 of the manual is the full
list. The ones that matter most:

- **No warping.** Beams have six degrees of freedom a point: open thin-walled
  sections in non-uniform torsion and distortion of box girders are outside it.
- **Shear in a fibre section is a member-level law**, not a field theory
  (no MCFT): shear-critical members need judgement or a solid model.
- **Soil springs are monotonic** p-y curves: no cyclic degradation or gapping.
- **Contact between members is between capsules** of a stated radius, not
  between section outlines, and not with a solid mesh.
- **Explicit steps are small.** Static and long-duration questions go through
  relaxation, Newton statics or analysis steps; a stiff, fine solid mesh over
  a long record is not what this is for.
- No aerodynamics (flutter, buffeting), no fluid, no soil continuum, no
  design-code checks.

## Citing

`CITATION.cff` has the entry. The method: Ting, Shih and Wang, *Fundamentals
of a vector form intrinsic finite element*, J. Mech. 20 (2004).

## Licence

MIT; see [LICENSE](LICENSE).
