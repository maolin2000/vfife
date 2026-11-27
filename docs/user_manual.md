# vfife Vector Form Intrinsic Finite Element Structural Analysis Software — User Manual

**Software name**: vfife Vector Form Intrinsic Finite Element structural analysis software (vfife for short)
**Version**: 1.23.2
**Runtime environment**: Python 3.10 or later; Windows, Linux, macOS; the GPU backend requires NVIDIA CUDA
**License**: MIT

This manual follows the order in which a user actually works: it first introduces what the software can do and how to install it, then walks through one complete case from input
to results, then explains chapter by chapter every part of the input file, elements and materials, solution control, result output and checking, and finally
lists the command-line reference, frequently asked questions and the software's scope of application.

Relation to the other documents: `README.md` is the front page; `docs/technical_notes.md` (the README until 1.12) is the complete description for developers, including the rationale and verification record for every feature;
this manual is for users and covers only how to use the software and what to watch out for when using it. Topic documents are under `docs/`, and this manual
links to them at the relevant places.

---

## Contents

1. Software overview
2. Installation and runtime environment
3. A first case: a 20 m post-tensioned prestressed concrete simply supported beam
4. Input file overview
5. Mesh
6. Materials
7. Sections and assignment
8. Regions
9. Constraints, loads and amplitudes
10. Analysis steps and solvers
11. Bridge-specific features
12. Result output and post-processing
13. Model checking and result verification
14. Command-line reference
15. Frequently asked questions
16. Scope of application and known limitations
Appendix A List of shipped cases

---

## 1 Software overview

### 1.1 Purpose

vfife is a structural analysis program built around the **Vector Form Intrinsic Finite Element (VFIFE)** method. It discretises the structure into a set of
particles, whose motion is integrated directly from Newton's second law; elements are only responsible, at each time
step, for computing internal forces from the relative displacements of the particles, and large rotations are handled by a "co-rotational" treatment (removing the rigid-body motion step by step). Geometric nonlinearity,
material nonlinearity, and member fracture and separation are therefore all handled naturally in the same explicit framework, with no need to assemble and solve a global stiffness matrix.

Static problems are solved by **dynamic relaxation**: the particles are given fictitious masses and kinetic damping is applied, so that the structure "falls" into its equilibrium position; with
incremental loading every increment converges to the residual tolerance, so every point on the load–displacement curve is a true equilibrium point.

Typical problems the software targets:

* The full process of reinforced concrete and prestressed concrete members and bridges from cracking, yielding and crushing to wire rupture and collapse
* Degradation of load-carrying capacity under corrosion, prestress loss, temperature, creep and shrinkage, and similar actions
* Dynamic problems such as seismic time histories and moving loads; modes in the loaded state
* Form finding of cable structures; large-displacement analysis of suspension and cable-stayed bridges
* Structural models mixing solids, shells, beams, trusses and connectors

### 1.2 Main features

| Category | Features |
|---|---|
| Elements | Solids tet4/tet10/hexa8/hexa20/wedge6/wedge15/pyramid5; shells shell3/shell4; beams beam2 (Euler–Bernoulli), beam2_timoshenko; fibre beam beam2_fiber; truss/cable truss2; six-component connector connector2; mixed models mixed_line, shell_line, coupled |
| Materials | Linear elastic, orthotropic, viscoelastic, J2 plasticity, Chaboche cyclic plasticity and damage, concrete damaged plasticity CDP; uniaxial concrete01/concrete02, steel02, steel_cyclic, rebar, piecewise linear; bond–slip bond_ceb, bond_cyclic |
| Reinforcement and prestress | Embedded reinforcement placed by drawing coordinates (independent of the mesh), bond–slip, pre-tensioned and post-tensioned prestress (friction, anchorage set, elastic shortening), tendons located element by element within fibre sections |
| Loads | Concentrated forces, couples, line loads, surface tractions, pressure (optionally follower), body forces, gravity, temperature (uniform/gradient/arbitrary profile), moving loads, seismic ground acceleration |
| Time effects | Concrete creep, shrinkage, age; relaxation of prestressing steel |
| Analysis | Dynamic-relaxation static analysis (continuous or incremental), explicit dynamic analysis, sequential execution of multiple analysis steps, modes in the loaded state, cable form finding |
| Failure | Fibre fracture and member separation, element deletion, strand wire rupture |
| Backends | NumPy (reference implementation), Numba (CPU compiled kernels), Warp/CUDA (GPU) |
| Output | HDF5 + XDMF (opens directly in ParaView), monitor curves as plots and CSV, fibre sections swept along the deformed axis into surfaces, mode-shape animations |
| Quality assurance | Pre-run model checking (preflight), mechanical-invariant probes (probe), 61 verification contracts and the `verify` command |

### 1.3 How it is used

vfife has no graphical interface, and this is a deliberate design choice: **the command line, the Python interface and YAML case files share one and the same workflow**.
A case is a single YAML text file that can be written with any editor, tracked in version control, and generated in batches by scripts.
Meshes are generated with Gmsh and results are viewed with ParaView.

**This manual covers concepts and modelling methods; for what can be written in a case file, consult `docs/yaml_reference.md`.** That file is
generated from the code by `python -m vfife reference`: every field of every block, its type, default value and description, and
the numbers of the shipped cases that use it; at the end it lists the fields that have no case. Tests guarantee that it is consistent with the code, so it does not go out of date.

---

## 2 Installation and runtime environment

### 2.1 Installation

```bash
git clone https://github.com/maolin2000/vfife.git
cd vfife
pip install -e ".[dev,plot,cpu]"      # NumPy + Numba; add ",gpu" when a GPU is needed, ",paraview" for sweep/mode-shape export
python -m vfife --version
```

Or with Conda (which also installs Gmsh):

```bash
conda env create -f environment.yml
conda activate vfife
```

Dependencies: numpy, scipy, pydantic, pyyaml, h5py; optional matplotlib (plotting), numba (CPU compiled kernels),
warp-lang (GPU), meshio (sweep and mode-shape export; the `paraview` extra installs it).

### 2.2 Companion software

* **Gmsh**: builds geometry and generates meshes. vfife reads `.msh` (4.1 ASCII / 4.1 binary / 2.2)
* **ParaView**: open the `.xdmf` file next to the results to view deformation, stress and damage

### 2.3 Checking the installation

```bash
python -m vfife list
```

Output:

```
elements: beam2, beam2_fiber, beam2_timoshenko, connector2, coupled, hexa20, hexa8, mixed_line, pyramid5, shell3, shell4, shell_line, tet10, tet4, truss2, wedge15, wedge6
materials: bond_ceb, bond_cyclic, bouc_wen, cdp, chaboche, concrete01, concrete02, gap, j2, linear_elastic, orthotropic, parallel, piecewise_uniaxial, rebar, series, steel02, steel_cyclic, viscoelastic
```

If the elements and materials are listed, the installation is fine. Commands are always written as `python -m vfife ...`; after `pip install -e .`
you can also write `vfife ...` directly.

### 2.4 Editor completion

Writing a case in VS Code can come with field completion, hover help and a red underline on a misspelt key:

1. Install the YAML extension by Red Hat (search the marketplace for "YAML", publisher Red Hat).
2. In the case's folder run `vfife schema`: it writes `vfife.schema.json` (`-o` puts it elsewhere) and prints the line to add.
3. Put that line at the top of the case (the path is relative to the case file):

```bash
python -m vfife schema       # writes vfife.schema.json
```

```yaml
# yaml-language-server: $schema=vfife.schema.json
```

Typing a key then lists the fields and enum values that level accepts (`mode: relaxation | static | dynamic`, a material's
`type`); hovering over a field shows its description -- the same code comment as docs/yaml_reference.md; a misspelt key, a value outside an enum and a missing required key are underlined. `vfife init` writes this line as
the first line of the case it creates and puts `vfife.schema.json` beside it (an existing one is kept, with a hint to rerun
`vfife schema` when it differs from this version). The schema changes with the version: rerun it after upgrading vfife. Notes:

* It checks structure (keys, types, enums); of a material it checks only `type`, and the model itself is still `vfife check`'s job.
* Old spellings (`b`, `h`, `torsion_constant`, ...) are underlined: `vfife migrate` renames them.
* The editor reads YAML 1.2, where `2e11` is a number; so does vfife. To validate with Python's `jsonschema`, note that
  PyYAML (YAML 1.1) reads `2e11` without a decimal point as a string: convert first with `vfife.io.schema.coerce_numbers`,
  or write `2.0e+11`.

---

## 3 A first case: a 20 m post-tensioned prestressed concrete simply supported beam

This chapter uses one complete case to go through "check the mesh → write the case → check the model → run → look at the results".
**All outputs in the text were obtained from actual runs.** For an introductory case with solid members see `docs/tutorial.md` (a steel cantilever bracket,
including Gmsh modelling and how to deal with tetrahedral locking).

### 3.1 Mesh

The beam is represented by a string of two-node line elements. The repository ships a straight-line mesh of 20 m with 20 elements,
`benchmarks/assets/thermal_span_20.msh`; first look at what is in it:

```bash
python -m vfife mesh benchmarks/assets/thermal_span_20.msh
```

```
  element       truss2 (line), 20 elements, 21 nodes
  extent        x [0, 20]  y [0, 0]  z [0, 0]
  physical groups (use these names in `region:`)
    beam                 21 nodes   x [0, 20], y=0, z=0
    left                  1 nodes   x=0, y=0, z=0
    mid                   1 nodes   x=10, y=0, z=0
    right                 1 nodes   x=20, y=0, z=0
```

The four physical groups `beam`, `left`, `mid` and `right` are the names that can be used directly in `region:` in the case.
(truss2 is shown here because a line mesh carries no section by itself; the section is assigned in the case, and once a fibre section is assigned it is a fibre beam.)

### 3.2 Case file

Copy the mesh next to the case and create `pc_beam.yaml`:

```yaml
version: 1
name: pc_beam
units: {length: m, force: N, stress: Pa}
mesh: {type: gmsh, path: thermal_span_20.msh}

materials:
  concrete: {type: concrete02, fpc: 40.0e+6, epsc0: 0.002, fcu: 8.0e+6, epscu: 0.0035,
             ft: 2.6e+6, Ets: 1.5e+9, rat: 0.1, rho: 2500}
  rebar:    {type: steel02, E: 2.0e+11, Fy: 4.0e+8, b: 0.01, R0: 20.0, rho: 7850}
  strand:   {type: steel02, E: 1.95e+11, Fy: 1.58e+9, b: 0.02, R0: 20.0, rho: 7850,
             rupture_strain: 0.035}

sections:
  phi16:  {type: bar, material: rebar, shape: circle, diameter: 0.016}
  duct:   {type: bar, material: strand, shape: general, area: 2.8e-3}
  girder:
    type: fiber
    material: concrete
    shape: rect
    width: 0.5
    height: 1.2
    ny: 4
    nz: 24
    integration: lobatto3
    bars:
      - {y: 0.0, z: -0.55, section: phi16, count: 4}
    tendons:
      - section: duct
        profile: [[0.0, 0.0, 0.0], [10.0, 0.0, -0.45], [20.0, 0.0, 0.0]]
        interpolation: quadratic
        jack: 1.30e+9
        friction: {mu: 0.17, k: 0.0015, jack: start, anchor_set: 0.006}

assign:
  - {region: beam, section: girder, orientation: [0, 1, 0]}

constraints:
  - {region: beam,  dofs: [0, 1, 0, 1, 0, 1]}   # plane problem: lock the out-of-plane DOFs
  - {region: left,  dofs: [1, 1, 1, 0, 0, 0]}
  - {region: right, dofs: [0, 1, 1, 0, 0, 0]}

loads:
  - {region: beam, line_load: [0.0, 0.0, -15.0e+3]}   # dead load 15 kN/m

amplitudes:
  live: [[0.0, 0.0], [1.0, 1.0]]

steps:
  - name: dead
    load_steps: 1
  - name: live
    loads:
      - {region: beam, line_load: [0.0, 0.0, -60.0e+3], amplitude: live}
    load_steps: 12

solver:
  backend: numpy
  mode: relaxation
  damping: {type: kinetic}
  tolerance: 1.0e-5
  max_steps_per_load: 60000

output:
  path: results/pc_beam.h5
  fields: [displacement, fiber_stress, fiber_damage]
  monitors:
    w_mid: {kind: displacement, region: mid, dof: z}
    R_left: {kind: reaction, region: left, dof: z}
```

Explanation, block by block:

* **materials**: C40 concrete (`concrete02`, with tension softening), ordinary reinforcement and prestressing strand (`steel02`,
  strand rupture strain 3.5%)
* **sections**: `girder` is a 0.5 m × 1.2 m rectangular fibre section with 4 × 24 concrete fibres; 4 Φ16 bars at the bottom;
  one parabolic tendon with a 0.45 m sag at midspan, jacking stress 1300 MPa, jacked from one end, with losses computed from a friction coefficient of 0.17,
  a wobble coefficient of 0.0015 and a 6 mm anchorage set
* **assign**: assigns the section to the `beam` physical group; `orientation` gives the direction of the section's local y axis
* **constraints**: the six flags of `dofs` correspond in turn to ux, uy, uz, rx, ry, rz, with 1 meaning constrained.
  The first line locks the whole beam in the x–z plane; the next two are a pin at one end and a roller at the other
* **loads / steps**: the dead load is written at model level and acts from the first step onwards; the second step `live` applies the 60 kN/m live load in
  12 increments, and one frame is output after each increment converges
* **solver**: dynamic relaxation. `max_steps_per_load` is the iteration limit per increment; the total number of iterations is summed automatically from each step's
  `load_steps × max_steps_per_load`, so `n_steps` need not be written
  (in versions 0.4.2 and earlier, models with line and shell elements still had to specify `solver.n_steps`, see Section 10.3)
* **output**: two monitor curves, midspan deflection and left support reaction; fibre stress and damage are written to the result file

### 3.3 Pre-run check

```bash
python -m vfife check pc_beam.yaml
```

```
case: pc_beam
  preflight     PASS (0 errors, 0 warnings)
```

`check` checks whether regions select nodes, whether there is any unconstrained rigid-body motion, whether unit magnitudes are reasonable, and so on.
Once preflight passes, it always builds the model on the numpy backend (no GPU, no solve): a model that cannot be built is an ERROR
`MODEL_BUILD_FAILED` (exit code 2); when the case lists `output.fields`, a field the model cannot produce is refused (`UNKNOWN_OUTPUT_FIELD`).

`info` goes further: it builds the model and prints a summary, but does not solve:

```bash
python -m vfife info pc_beam.yaml
```

```
  mesh          21 nodes, 20 Line2 elements
  element       beam2_fiber / fiber Euler-Bernoulli, 6 DOF/node
  section       rect 0.375 x 1.15 m, 96 fibers + 1 rebar + 1 tendon, A=6.036e-01 Iy=7.212e-02 Iz=1.172e-02
  tendon        section 0 slot 1: A=2.800e-03 m2, y +0.000..+0.000 m, z -0.449..-0.044 m, sigma 1168..1234 MPa, P 3272..3454 kN
  mass          30565.9 kg, of it members 30565.9 kg (each fibre at its material's density); weight 299749 N at g = 9.80665
  fiber         concrete02, steel02, steel02; 98 x 3 points (3pt lobatto)
  mode          relaxation; dt=1.0000e+00, 780000 steps
  axes          beam (beam2_fiber, 20 members): local x = +X, local y = +Y, local z = +Z
```

Read these lines before running: section area 0.6036 m² (0.6 m² of concrete plus reinforcement), the tendon sags to −0.449 m at midspan,
tendon stress after friction and anchorage set 1168–1234 MPa. **When the numbers do not match, the problem is in the input, not in the solver.**
(`rect 0.375 x 1.15` is the extent of the fibre centres, not the outer outline. The `axes` line: local z is +Z, so the section's `height` and the tendon's `z` are vertical; see 7.5.)

The program rejects physically impossible input. For example, if this tendon were written as jacked from both ends (`jack: both`), the 6 mm set would exceed
what this short, flat tendon can absorb, and the run would report:

```
error: anchor set of 6.0 mm draws in more than this tendon can absorb: reflecting the profile
from this end recovers at most 1.79 mm (with `jack: both`, up to the tendon's middle). The
tendon is too short, or too straight, for that much seating -- state the draw-in the
anchorage actually has
```

### 3.4 Running

```bash
python -m vfife run pc_beam.yaml
```

```
  analysis      CONVERGED (residual 7.58e-06, analysis step 2 of 2)
  max |u|       5.901875e-02
  w_mid         final -5.901875e-02
  R_left        final 7.500285e+05
  written       results/pc_beam.h5 (+ .xdmf time series, _last.xdmf fallback for ParaView; .ckpt.h5 checkpoint)
```

**The first check is always the reaction**: left support 750.03 kN, corresponding to (15 + 60) kN/m × 20 m / 2 = 750 kN.
During an incremental relaxation the progress line reads `increment k/n … iterations/s`.

### 3.5 Results

Midspan deflection read out increment by increment (upward positive):

| Live load kN/m | 0 | 10 | 20 | 30 | 35 | 40 | 45 | 50 | 55 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|
| Midspan deflection mm | +10.6 | +3.6 | −3.6 | −11.1 | −14.9 | −19.3 | −25.0 | −33.0 | −44.1 | −59.0 |

* Under dead load plus prestress the midspan **cambers upward by 10.6 mm**
* Between 35 and 40 kN/m of live load the bottom fibres begin to crack (damage category 1), after which the curve becomes markedly softer — the number of cracked fibres grows from
  240 integration points at 40 kN/m to 1568 at 60 kN/m

Plot the load–deflection curve and export CSV:

```bash
python -m vfife plot results/pc_beam.h5 --x w_mid --y R_left     # -> results/pc_beam.png
python -m vfife export results/pc_beam.h5                          # -> results/pc_beam.csv
```

Open `results/pc_beam.xdmf` in ParaView to see the deformation; to sweep the fibre section along the deformed axis into a solid
coloured by fibre state, use `vfife sweep` (Section 12.4) -- it needs the particles' orientations, so add `rotation` to
`output.fields` first (`[displacement, rotation, fiber_stress, fiber_damage]`; without it `sweep` refuses the result);
for a 3D playback web page of any result, use `vfife view` (Section 12.5).

---

## 4 Input file overview

A case is one YAML file, with the following top-level keys (in any order):

| Key | Required | Purpose | Section |
|---|---|---|---|
| `version` | No | Case format version, write `1`; if omitted, the file is read by the same rules as `version: 1` | 4 |
| `name` | No | Case name, written to the result file | |
| `units` | No | Declares the unit system, e.g. `{length: m, force: N, stress: Pa}`; used for magnitude checks, result attributes and plot labels; **numbers you write are not converted** (defaults that carry a unit are, see Units below) | 4, 16 |
| `mesh` | Yes | Mesh source | 5 |
| `materials` | Yes | Constitutive models, defined by name | 6 |
| `sections` | Yes | Section = geometry + material | 7 |
| `reinforcement` | No | Reinforcement geometry placed by drawing coordinates | 7.4 |
| `assign` | Yes (except a solid mesh with one `solid` section, which is assigned to the whole mesh, as `vfife init` writes it) | Assigns sections to regions or reinforcement | 7.5 |
| `regions` | No | User-defined node sets | 8 |
| `constraints` | No | Displacement constraints and prescribed displacements | 9.1 |
| `loads` | No | Loads | 9.2 |
| `amplitudes` | No | Amplitude curves | 9.3 |
| `nodal_masses` | No | Additional lumped masses (directional, weightless, with rotary inertia) | 9.5 |
| `hydrodynamic_mass` | No | Hydrodynamic added mass of submerged members | 9.5 |
| `diaphragms` | No | Rigid floor diaphragms | 9.6 |
| `rigid_links` | No | Master-slave rigid links (1.21) | 9.7 |
| `steps` | No | Sequence of analysis steps | 10.1 |
| `removals` | No | Members cut at a given time | 11.10 |
| `contact` | No | Rigid contact surfaces | 11.12 |
| `response_spectrum` | No | Response-spectrum analysis at the end state | 11.8 |
| `solver` | Yes | Solver settings | 10 |
| `output` | No | Result file, fields, monitors | 12 |

**One modelling rule**: things that have physics are sections (`sections`), things that have position are geometry or regions
(`mesh`, `reinforcement`, `regions`), and the two meet only in `assign`. This is the same "material–section–assignment" structure as Abaqus and OpenSees;
reinforcement is independent geometry, so changing the mesh density does not affect the reinforcement.

**Format version and legacy syntax**:

* `version: 1` declares that the case uses format version 1. vfife 1.x only adds fields and never removes them or changes their meaning,
  so a case that says `version: 1` can be read by every later 1.x. A case without `version` is read by the same rules as `version: 1`;
  a number larger than the program knows is rejected with an explanation (it was written by a newer vfife).
* Each quantity has only one name. Several legacy spellings on sections are still read, with a `DEPRECATED_FIELD` warning from preflight:
  `b` → `width`, `h` → `height`, `d` → `diameter`, and `torsion_constant` of fibre sections → `j`.
  Materials still read a few other names of their parameters (`fc` for `fpc`, `E0` for `E`, `yield_strength` for `Fy`, ...;
  the list is in docs/yaml_reference.md, part 3); written beside the parameter's own name with a different value, the case is
  refused (`MATERIAL_ALIAS_CONFLICT`, v1.0 -- the law used to take its own name and drop the other without a word).
* `python -m vfife migrate case.yaml --in-place` converts a 0.x case to version 1:
  * it renames in place in the original text and adds `version: 1`, leaving comments, layout and number formatting untouched;
  * it lowers the case once before and once after rewriting and compares field by field, refusing to write if the models differ.
* Older formats (top-level `element:` / `material:` / `members:`) are no longer read directly; `migrate` also rewrites them into
  `materials / sections / assign`, likewise checking that the model is the same before and after rewriting.
* A case written with materials / sections / assign may not also carry `element`, `material`, `members` or `blocks` (keys of the old form):
  it is refused (`INTERNAL_KEY_IN_INPUT_FORM`); the element comes from the sections and the mesh.
* Result files (HDF5) carry `result_schema_version` (currently 3). The checkpoint is a separate
  `<result>.ckpt.h5` beside the result, carrying `checkpoint_format` (currently 1). A result or checkpoint newer than the
  program knows is rejected (`RESULT_TOO_NEW`, `CHECKPOINT_TOO_NEW`); older ones without the attribute are read as before.
  `tests/data/result_format_2/` holds a format-2 result, and every future version must be able to read it. The full
  layout is in docs/result_format.md.

**Units**: the program does not convert units; all values must belong to the same consistent unit system (m, N, Pa, kg, s recommended).
Once `units` is declared, preflight checks magnitudes using the material wave speeds, so mm/t/MPa numbers mixed into an m/N/Pa case are caught
before the run (`UNIT_MAGNITUDE`). The declaration itself is checked too: a misspelt key (`UNITS_UNKNOWN_KEY`, an error from v1.0), a stress
unit that is not force / length^2 (`UNITS_INCONSISTENT`, e.g. `{length: mm, force: N, stress: Pa}`), and a gravity that is
not g in the declared length unit (`GRAVITY_MAGNITUDE`: in millimetres g is 9810).

**Defaults that carry a unit** are given in SI and **converted to the declared units** when a case declares
non-SI units; a case with no `units`, or SI, uses them as SI. They are a vehicle's `gravity` (9.80665 m/s², 9806.65 in mm),
the water `density` of `hydrodynamic_mass` (1025 kg/m³, 1.025e-9 t/mm³), a stirrup modulus `E` (2e11 Pa, 2e5 MPa), a
roughness PSD's `n0`/`n_min`/`n_max` (cycles per length unit), and the materials' defaults: `rho`, `fpc` of
`concrete01`/`concrete02`, `E` and `Fy` of `steel02`, `E` and a grade's `fy`/`fu` of `steel_cyclic`, and the
`tau_max` and slips of `bond_ceb` and `bond_cyclic` (the same MC2010 envelope). The consistent mass unit is force × time² / length (a tonne in mm / N / s) and the stress unit
force / length². A declaration with `stress` but without `length` or `force` names the third through the other two:
`{force: N, stress: MPa}` is read as mm and `{length: m, stress: kPa}` as kN (`UNITS_INFERRED`, INFO), and the
magnitude checks (`GRAVITY_MAGNITUDE`, `UNITS_INCONSISTENT`, `UNIT_MAGNITUDE`) use the completed system. When the two
imply no unit it knows (`{force: kN, stress: MPa}` would be a 31.6 mm length), or `stress` stands alone (other than
`Pa`), preflight refuses the case: `UNITS_INCOMPLETE` (ERROR) names the missing key. Otherwise a key the declaration
leaves out is taken as SI (m, N, s). Preflight lists every default it converted
(`UNIT_DEFAULT_CONVERTED`, INFO) and warns when a declared unit is one it cannot convert to, leaving that default SI
(`UNIT_DEFAULT_SI`). Before v1.0 they were SI
whatever the case declared: a mm case's vehicle weighed a thousandth of its weight.

---

## 5 Mesh

### 5.1 Four sources

```yaml
mesh: {type: box, dimensions: [2.0, 0.2, 0.3], divisions: [24, 4, 8]}   # structured hexahedra
mesh: {type: gmsh, path: beam.msh}                                      # Gmsh file
mesh: {type: abaqus, path: part.inp}                                    # subset of Abaqus .inp
mesh: {type: lines, points: {...}, lines: {...}}                        # a line mesh written in the case (5.4)
```

* `path` is relative to the case file
* `scale` multiplies all coordinates: `scale: 0.001` converts a mesh built in millimetres to metres
* The element type is determined by the mesh: Gmsh tetrahedra, hexahedra, wedges, pyramids (including quadratic elements), quadrilateral/triangular
  surface elements, two-node line elements
* From Abaqus `.inp` only nodes, supported solid and line elements, and NSET/ELSET are read; materials, sections and loads are still written in the YAML

### 5.2 Physical groups

Gmsh `Physical` groups (volumes, surfaces, lines, points) are read in as regions of the same name and can be used directly in `region:`.
**A mesh without physical groups cannot have boundary conditions applied**; do not guess surface numbers — use `vfife mesh` to see which nodes each group
selects and what its coordinate range is.

A physical group also remembers the **elements** it contains: when the two nodes of a column also belong to a beam, the column and the beam cannot be told apart by nodes, but they can by the
physical group's elements. Groups are told apart by (dimension, tag): `Physical Volume(1)` and `Physical Surface(1)` are two groups.

Files refused on reading, with the reason: parametric node coordinates (`MESH_PARAMETRIC_NODES`), a repeated node tag
(`MESH_DUPLICATE_NODE_TAG`), an element naming a node that is not there (`MESH_UNKNOWN_NODE_TAG`), and shells mixing
triangles and quads (`MIXED_SHELL_ELEMENTS`). A triangle-only mesh reads as a shell. A name under `regions:` may not also
be a physical group's (`REGION_NAME_CLASH`), and a `type: box` mesh takes no `scale`.

### 5.3 Generating a case skeleton from a mesh

```bash
python -m vfife init beam.yaml --mesh beam.msh --fixed-region left --load-region tip --load 0 0 -5000
```

The generated case comes with a displacement monitor on the loaded region and a support-reaction monitor; after changing the material and loads it is ready to run.

**A Midas Civil stage model** (`.mct`) comes in here too: `vfife init bridge.yaml --mesh model.mct` writes the case, the line mesh
`bridge.msh` and a map `bridge.map.json` (which Midas node and element became which, and what each region is). Beam elements get the
area, second moments and torsion constant of their Midas shape (pipe, solid round, box, H, double channel), truss elements become `bar`s
with their pretension, and supports, rigid links (as stiff weightless beams), self weight, nodal and uniform member loads come with them,
with one analysis step per construction stage. Everything is converted to m and N. It lists what it did not carry over (plates,
temperatures, the later parts of composite sections) and what it approximated. A cable's pretension is translated as the model's
construction-stage control says (`*STAGE-CTRL`, CPFC, from 1.21): External (and a model that does not say) is written as `jack`, the
force it is stressed to in the step that builds it; Internal as `prestress`, each element cut to the length of its force and then
sharing what the structure gives; the report's notes say which. `vfife init` always follows the switch; `convert(..., jack_which="one")` from Python asks for the rule before
1.21 whatever the switch says: of a cable modelled as several elements over sliding saddles only the one that reaches the newly built
structure is jacked, the others are cut to length (`prestress`) and counted. An elastic link
becomes a connector whose springs are written in global axes (as Midas keeps them: they do not turn with its ends), built in the first
stage in which both its ends are held by something and born free of force there; where a node no beam reaches hangs on a link, the
link's rotational stiffness is raised so that the node turns with it. `vfife mesh model.mct` only reads, and gives the model's summary.

Since 1.15 the translation also takes: the section properties the file states (`*SECT-PSCVALUE`, VALUE and tapered sections, a
tapered section over a `*TS-GROUP`), with their shear areas -- a beam is a Timoshenko beam where they are known, as Midas' is
(`convert(..., shear=False)` for Euler's); a section set off its nodes (offset CT/CB: from 1.21 the member stays on its drawn nodes, its
elastic part on its centroid's line by end offsets, `end_offsets`, as in Midas; before, a centroid node and a rigid arm); beam end offsets
(`*OFFSET`, 1.21: TYPE=ELEMENT rigid zones along the member, TYPE=GLOBAL vectors; a boundary condition: in a model with stages it is there when a stage activates its
boundary group, as a support is; a zone weighs as its member and is rigid in bending and shear only, `axial: full`); tension-only elements (TENSTR, from 1.21 `tension_only` bars; a
cable's, iSUB 3, own unstressed length or tension as `unstressed_lengths` or `prestress`; a hook's gap not followed); section stiffness scale
factors (`*SECT-SCALE`, from 1.21 multiplied into the section's properties, the weight factor by a material of its own density; a boundary
condition too, on the same rule); member loads varying linearly, projected, or in the member's own axes (from 1.21; split to the two ends by
the lever rule); master-slave rigid links (`*RIGIDLINK`, from 1.21 the solver's `rigid_links` with the flags as written -- a saddle
that slides on its tower, 011111, and a stay hung by its translations, 111000, too; made in the stage their boundary group comes on, or
that builds the last of their nodes, and released where it goes off; before, those tying all six were stiff weightless beams and the
others were left out. Left out and counted: a slave no member reaches, one a support holds in a tied direction, a chain whose links do not
both tie all six and live as long); point springs
(`*SPRING`); a node's own axes (`*LOCALAXIS`, turned about Z: a guided bearing in them); and **tendons** (`*TDN-PROPERTY`,
`*TDN-PROFILE` with 2-D profiles on a straight or an element reference line, `*TDN-PRESTRESS`) as bars bonded to the girder on rigid
arms, each jacked to the force after friction and anchorage draw-in in the stage that stresses it (its load group on and its members
built) and an ordinary bar after: later stages shorten it with the girder, and it adds its stiffness to the section. A tendon jacked at
both ends loses to both anchorages' draw-in together (the point that does not move is where the two draw-ins meet, JTG D62's l_f > l/2
on a short tendon). Since 1.16 a tendon whose `*TDN-PROPERTY` says it relaxes (the code's method) relaxes as that code says: each bar gets
the relaxation curve of its own stress after anchoring (from `vfife.codes`, Section 11.3), and a staged model with such tendons runs each
stage's duration after the stage as a step of time (`to_age`; the map's `stage_steps` gives the step each stage ends with, which is the
one Midas' stage results are at). **Creep and shrinkage (1.22)**: the concrete `*TDM-LINK` ties to a model of JTG D62-2004 or JTG 3362-2018
(`*TDM-TYPE`) creeps and shrinks by MC90, which those codes' annexes are, at fcm = 0.8 fcu,k + 8 MPa, the humidity, notional size, β_sc and
drying age the file gives; an element's own notional size (`*ELEM-DEPMATL`, NSM) is a material of its own; the modulus is the material's
as the 28-day one, or grows by CEB-FIP 2010 where `*TDM-ELAST` says so and the stage control takes the strength's growth (`ageing`, s by
its cement type iCTYPE: 1 rapid hardening high strength 0.20, 2 normal or rapid hardening 0.25, 3 slowly hardening 0.38); β_sc is the shrinkage's alone, as in JTG. It follows the stage control
(`*STAGE-CTRL` line 4: time-dependent effects on, and its "Creep & Shrinkage" box, bCNS, with CREEP, SHRINKAGE or BOTH); a model without a stage control takes none. Each stage
builds its concrete at the age the stage gives it (`activate_age`; concrete of another age in the same stage in a step of its own just
after, the first stage's too), the clock starts at the age of the concrete standing from the first stage, and every stage's duration runs after it as a step
of time. Other codes, 3362's fly-ash adjustment and other modulus laws are counted and left out. A stub (a member shorter than 0.05 of its radius of gyration and stiffer than 1e13 N/m) becomes a stiff
connector. `convert(..., cases=("wind",))` adds a load case of the finished bridge as one more step; a case of imposed
support displacements (`*SPDISP`), or a settlement group by its name (`*SM-GROUP`, 1.23), is such a step too, its
supports brought to the stated displacement and kept (Midas' settlement load cases are envelopes over combinations of
groups, which this does not do: one group a step). A case's temperatures are taken the same way (1.23): the whole
model's (`*SYSTEMPER`) and an element's (`*ELTEMPER`) as a uniform change, one over a section's depth (`*BSTEMPER`, layer
by layer) as the uniform change and the gradient along local z that give the same axial force and moment -- what a frame
member carries; the self-equilibrated rest of a nonlinear profile is not followed. The materials' thermal coefficients are
written only when a case taken heats the model (a grade the file names without its values takes the codes', 1.0e-5/K
for concrete and 1.2e-5/K for steel, noted); a temperature across the width (DIR LY), one in the PSC form, a stage's own
temperature load and a material with no coefficient are counted. The map's `tendons` gives each
tendon's stage, jacking force and its bars' regions and forces. Against Midas on six real bridges, the displacements and member forces
agree to about 1% where the two models are the same; on a rigid frame built by balanced cantilevers with 132
tendons the tendons' force agrees to 1% after relaxation and the finished bridge's dead-load moments to 0.2%. Its deflections, compared
on Midas' dead load alone, are 6% larger than Midas' when the comparison leaves the tendons out and 3% larger with them in as bars that
carry no prestress (a translated model always has them in, as bonded bars). Midas evidently counts the grouted tendons in the girder's
section (a transformed section): integrating Midas' own moments and shears by hand over its own gross-section properties gives
deflections that the tendon-free translation matches to 0-3%, while Midas itself is 1-6% stiffer than that, more so as the tendons
accumulate along the cantilever. Of the 3% left, about 1% is a heavier load here (reactions 0.4-0.7% and cantilever moments about 1%
larger) and about 2% is the sections' ordinary bars. With `bCALC-CSP` on in its construction stage control (`*STAGE-CTRL`), Midas varies the
section properties stage by stage: it counts the ordinary bars (a transformed section), takes the ducts out before grouting and the
tendons in after it. Its table of each stage's section properties (FRAME ELEMENT SECTIONAL PROPERTY FOR EACH STAGE in the .OUT) gives
this bridge's girder A and I 0-4% over the gross section when cast and 4-11% once all is grouted; the piers, with no bars, stay gross; and
a model without the switch (the curved bridge) is gross throughout (found in 1.21). This translation reads neither the ordinary bars nor
the ducts: gross sections, the tendons as bonded bars. To match Midas, work out the transformed `area`, `iy` and `iz` and write them in. The tendons' eccentricity, 1-3% larger here, is most likely the same cause seen from the other side: a transformed section's
centroid lies nearer the tendons (estimated, not checked on its own).

`init` writes a solid section, so it takes only a mesh of volume elements: a shell or line mesh is refused with the section to write by hand
(a `shell` section with `thickness`; `bar`/`beam`/`fiber`/`connector` for lines). A volume mesh that also holds Line2 elements gets a warning:
they are read only as node sets of their physical groups, and to compute them as members add a `bar` section and an `assign` entry for their group.
An existing target file is refused unless `--overwrite` is given; the mesh file itself is always refused. On an interactive run, `--fix` replaces
the support prompt and `--force` the load prompt. The prompts offer only boundary-sized physical groups; `--fix`, `--force`, `--fixed-region` and
`--load-region` may name any physical group, the whole volume included (`--fix body 010` holds y at every node). A mesh whose only group is the
volume needs these options.

---

### 5.4 A line mesh written in the case: `type: lines` (1.1)

A bridge's frame model is a few dozen points and the members between them. It no longer needs a Gmsh script and a .msh;
it can be written in the case:

```yaml
mesh:
  type: lines
  points:                                  # named points; each name is a region of that one node
    A: [0.0, 0.0, 0.0]
    B: [40.0, 0.0, 0.0]
    C: [80.0, 0.0, 0.0]
    P: [40.0, 0.0, -15.0]                  # the pier's foot
  lines:                                   # named groups of members; each name is a region of them and their nodes
    girder: {through: [A, B, C], size: 2.0}          # each segment in elements of about 2 m
    pier:   {through: [P, B], divisions: 6}
    arch:   {through: [A, C], divisions: 24, arc: {rise: 12.0}}    # a parabolic arch, 12 m rise
assign:
  - {region: girder, section: box_girder, orientation: [0, 0, 1]}
  - {region: pier,   section: pier_rc,    orientation: [1, 0, 0]}
constraints:
  - {region: P, dofs: [1, 1, 1, 1, 1, 1]}
```

* `through` lists the points the line passes through in order (names from `points`, or coordinates); each pair of
  neighbours is a segment, divided into `divisions` elements, or by `size` (a target element length: ceil(length / size)
  elements, along the curve for an arc); neither gives 1.
* **It builds what a Gmsh Line2 file reads as**: one Line2 block, a region per point and per line, and an element group per
  line (so `assign` tells girders from piers by their members). `assign`, constraints, loads, monitors and `regions:`
  use them as they use Gmsh physical groups; case_03 and case_22 rewritten from their .msh files to `lines` give the
  same results.
* **Named points are nodes**: lines through the same point join there, and nowhere else. Two nodes at one place are refused
  (two named points together; a line laid over part of another; a line attached by coordinates at another's division node),
  and so are two members on one pair of nodes. A bearing's two sides are two points a little apart with a connector between
  them. A point written as coordinates is the point already at that place.
* **Lines joined into one group**: `groups: {main_girder: [span1, span2, span3]}` is an **element** region, as a Gmsh
  physical group of several pieces is, for `assign` and `removals`. A `regions:` union is a set of nodes only: every member
  whose two nodes are in it belongs to it whichever line it is on -- a removal on a union of hangers removes the girder with
  them. To give one span of a girder its own section, write the spans as lines and join them with `groups`.
* `arc: {rise, shape, up}` draws the line between its two points as a curve: `shape: parabola` (the default; nodes evenly
  along the chord, the usual arch axis) or `circle` (evenly by angle), rising `rise` at mid-chord along `up` (default +z).
* Coordinates are in the case's length unit; `scale` is refused. A name used twice among points, lines and groups, the names
  `all` and `surface`, a `through` naming an undefined point and a point on no line are refused (`LINE_MESH`); a case
  region may not take a point's, line's or group's name (`REGION_NAME_CLASH`). `size` sets how many elements a segment has;
  a parabola's nodes stay evenly along the chord (its end elements a little longer). This mesh holds line members only; a model with shells or solids keeps Gmsh.

### 5.5 Starting from a scenario template: `vfife new` (1.2)

```bash
python -m vfife new --list                     # the templates
python -m vfife new girder my_bridge.yaml      # a case that runs as it stands
python -m vfife info my_bridge.yaml
python -m vfife run my_bridge.yaml
```

| Template | What it is | Then |
|---|---|---|
| `girder` | a 30 + 40 + 30 m continuous steel box girder, its weight and 60 kN/m superimposed dead load (with its mass) | `run` (static), `summary`, `modal` |
| `pier_on_piles` | a 12 m pier on a 25 m pile in p–y soil, the deck's reaction and a 500 kN push at its top | `run`, `summary`, `buckling` |
| `arch` | an 80 m parabolic steel arch rib, 16 m rise, fixed springings, a 50 kN/m deck load | `run`, `buckling`, `modal` |
| `cable` | a cable of 102 m between supports 100 m apart, hanging under its weight (dynamic relaxation) | `run`, `summary` |

* The templates use a line mesh written in the case (5.4) and are commented; the first line points at `vfife.schema.json`,
  which is written beside the case (editor completion).
* Sections, loads and soil curves are placeholders to replace with your own. Every template is tested: written, `check`ed,
  run, and its `summary` reaction check is 1.
* An existing file is not written over without `--overwrite`.

## 6 Materials

Materials are defined by name under `materials:`, and `type` selects the constitutive model. `python -m vfife describe <material name>` lists
all parameters, state variables, output fields, supported backends and analysis modes of that material; `python -m vfife capabilities`
queries the available element–material–backend combinations.

### 6.1 Three-dimensional materials (solids, shells, beams)

| type | Use | Main parameters |
|---|---|---|
| `linear_elastic` | Linear elastic | `E, nu, rho` |
| `orthotropic` | Orthotropic elastic | `E1..E3, nu12, nu13, nu23, G12..G23, rho` |
| `viscoelastic` | Linear viscoelastic (Prony series, dynamic only) | `E, nu, rho, g, tau` |
| `j2` | von Mises plasticity, isotropic/kinematic hardening | `E, nu, rho, yield_strength, hardening, kinematic` |
| `chaboche` | Cyclic plasticity (nonlinear kinematic + isotropic) with damage | `yield_strength, Q_inf, b, C1, gamma1, C2, gamma2`, damage parameters |
| `cdp` | Concrete damaged plasticity (the same model as Abaqus CDP) | `E, nu, rho, compression, tension, dilation_angle, Kc, wt, wc, crack_band` |

The compression and tension curves of `cdp` can be generated from codes (e.g. `compression: {source: gb50010, fc: 44.0e6}`,
`tension: {source: fracture_energy, ft: 3.42e6, Gf: 140}`), or given as Abaqus-format tables;
`crack_band` applies crack-band regularisation so that the softening branch is independent of mesh size. See `docs/concrete.md` for details.

### 6.2 Uniaxial materials (fibres, reinforcement, trusses)

| type | Use | Main parameters |
|---|---|---|
| `concrete01` | Kent–Scott–Park concrete with Karsan–Jirsa unloading, tension ignored | `fpc, epsc0, fpcu, epscu` |
| `concrete02` | Yassin's concrete with tension softening (the law of OpenSees' Concrete02) | `fpc, epsc0, fcu, epscu, rat, ft, Ets` |
| `steel02` | Menegotto–Pinto steel with Filippou et al.'s isotropic hardening (the law of OpenSees' Steel02), optional rupture strain | `E, Fy` (or `fy`)`, b, R0, cR1, cR2, a1..a4, rupture_strain` |
| `steel_cyclic` | Cyclic steel with a low-cycle fatigue fracture criterion | `grade, fy, fu, ...` |
| `rebar` | Bilinear reinforcing steel, isotropic/kinematic hardening | `E, yield_strength` (or `fy`)`, hardening, kinematic_fraction, rupture_strain` |
| `piecewise_uniaxial` | Piecewise-linear tension–compression curve | `tension, compression, hardening_mode` |
| `gap` | Free travel between two strains, linear elastic past them (1.21) | `E, rho, lo, hi` |
| `parallel` | Side by side: the parts share the strain, their stresses add (1.21) | `of, factors` |
| `series` | End to end: the parts share the stress, their strains add; the split is iterated at each point (1.21) | `of, max_iterations, tolerance` |

Concrete strength parameters are entered as positive values (e.g. `fpc: 40.0e6`), consistent with the shipped cases. When `rupture_strain` of `steel02` is given, a fibre whose strain reaches it ruptures and
no longer carries force (damage category 4); this is how wire rupture is modelled.

**Which steel law** (the table measured on one ±1% cyclic path is in `docs/steel02.md`):

* **Monotonic loading, pushover**: `rebar`. Bilinear and the simplest, with numba and warp kernels on the solid and truss paths; but its peak keeps growing under constant-amplitude cycles (+36% after 6), so do not use it for time histories.
* **Cyclic and seismic, reinforcement and strand**: `steel02`. Rounded reversals, stable loops; most reinforcement and strand in the shipped cases use it.
* **Structural steel cycled to fracture (collapse, ultra-low-cycle fatigue)**: `steel_cyclic` (`docs/steel_cyclic.md`). The only law with degradation and a fatigue fracture criterion.
* **A measured curve in hand**: `piecewise_uniaxial`. Slow (about 3.7 µs a point), for small models or single-point studies.

A fibre section runs the compiled kernel only when every bar is `steel02`, or every bar is `steel_cyclic`, and the concrete is one type of `concrete01` or `concrete02`; bars of `rebar` or `piecewise_uniaxial` send the whole section to the NumPy path, about four times slower.

**Composite materials** (1.21): the `of` of `parallel` and `series` lists two or more names in the material table (or material specs written in place); each part is a one-dimensional law, or another composite. `parallel`'s `factors` weight the parts' stresses, 1 each by default; `series` finds the parts' strains at each point until their stresses differ by no more than `tolerance` (1e-10 by default) of the stress, and refuses with `SERIES_NOT_CONVERGED` when it cannot. A `linear_elastic` part is E·ε (a uniaxial stress), not the 3-D constrained modulus. No part may carry `creep`, `shrinkage`, `relaxation` or `thermal_expansion`. Density: `parallel` adds the parts' densities with the `factors`, `series` takes the mean over all its parts; `rho` gives it directly. Poisson's ratio is the first part's (a fibre section's torsion reads it). Failure: a `series` has failed when any part has, a `parallel` when every part has; once a part has broken a series carries next to nothing and the run goes on. A part written inside `of` takes the case's `units` defaults as a named one does. The composites and `gap` are taken by a standalone `truss2`, by embedded bars and by fibre sections (on the NumPy path); a bar of a frame model (`mixed_line`) still takes `linear_elastic` only. A buckling-restrained brace's core beside a spring, and a stop with slack in it:

```yaml
materials:
  core: {type: steel02, E: 2.0e11, Fy: 2.35e8, b: 0.01}
  slack: {type: gap, E: 2.0e11, rho: 7850.0, hi: 0.002}
  spring: {type: linear_elastic, E: 2.0e9, nu: 0.3, rho: 7850.0}
  brace: {type: parallel, of: [core, spring]}
  stop: {type: series, of: [slack, core]}
```

**Bouc–Wen** (`bouc_wen`, 1.22): σ = α E ε + (1 − α) f_y z, with ε_y dz = [A − |z|^n (β sgn(dε z) + γ)] dε (Wen 1976). `E` the
modulus (the initial tangent with the default A = 1; ε_y = fy/E), `fy` the yield stress, `alpha` the post-yield ratio (0), `n` the sharpness of the knee (1; large n is bilinear), `beta`,
`gamma`, `A` (0.5, 0.5, 1: |z| tends to 1). It needs β > 0 and −β < γ ≤ β. Integrated by the implicit midpoint rule in sub-increments
of at most ε_y/50, rate-independent; state z and the energy the loop has dissipated (`hysteretic_energy`). On a connector with area =
length = 1 it is the force–deformation law itself (E the stiffness, fy the yield force), and on `uy` and `uz` together one law in the
plane (Section 11.11). The explicit step and the reference stiffness take its largest tangent, E[α + (1 − α) A max(1, 2β/(β + γ))]
(E itself at the defaults). It is meant for connectors and bars (`truss2` takes it); fibre sections do not, though a `parallel` or
`series` holding it runs there on the NumPy path.

```yaml
materials:
  lrb: {type: bouc_wen, E: 2.0e7, fy: 5.0e4, alpha: 0.1, n: 2}
sections:
  bearing: {type: connector, stiffness: [2.0e9, 0, 0, 0, 0, 0],
            laws: {uy: {type: material, material: lrb}, uz: {type: material, material: lrb}}}
```

### 6.3 Bond materials

`bond_ceb` (CEB-FIP/MC2010 bond–slip curve) and `bond_cyclic` (cyclic bond with damage accumulation) are used for
bond–slip of embedded reinforcement (Section 7.4, `bond`).

### 6.4 Time-dependent properties

These are written in the material, independently of the constitutive model:

```yaml
materials:
  concrete:
    type: linear_elastic
    E: 3.87e+10
    nu: 0.2
    rho: 2450.0
    creep:     {model: mc90, notional_size: 480.0, rh: 75.0, fcm: 63.0e+6}
    shrinkage: {model: mc90, notional_size: 480.0, rh: 75.0, drying_start: 7.0}
    ageing:    {s: 0.25, of: modulus, reference_age: 28.0}
    thermal_expansion: 1.0e-5
```

* `creep`, `shrinkage`, `ageing`: creep, shrinkage, and growth of the elastic modulus with age; see `docs/creep.md`
* `relaxation` of prestressing steel: in either a power-law form or a measured-curve form; **the user must give the source of the coefficients** —
  the program does not build in code formulas (see `docs/creep.md` for why)
* Every key of a time-dependent spec is checked when the material is made; a key its model does not read (e.g. `RH` for
  `rh`) is refused by name.
* `mc90` creep is referenced to the 28-day modulus, J(t,t0) = 1/Ec(t0) + φ/Ec28. With `ageing` on the material,
  Ec(t0)/Ec28 follows from it; without it give `E28` (Pa) in `creep`, `E` then being the modulus at loading. Neither, or
  both, is refused by name. **Known limit:** with `ageing` instead of `E28` the result is still about 2% short, because
  the ageing scheme scales the whole stiffness by one factor and the elastic part drifts from 1/Ec(t0) towards
  1/Ec(t); `E28` gives the MC90 value exactly. `cement` (4 slow, 5 normal, 8 rapid high-strength) adjusts the loading
  age as in MC90; `rh` is 40–100, `fcm` is in Pa and 20–88 MPa (MC90's C12–C80; outside it is refused).
* `aci209` creep is multiplied by the loading-age factor (1.25 t0^-0.118 for `curing: moist`, 1.13 t0^-0.094 for
  `steam`) at each increment's own loading age; `ultimate` excludes it.
* `mc90` shrinkage at RH ≥ 99% is swelling (β_RH = +0.25; the member lengthens).
* A `curve` table has an implicit origin (0,0) and rises linearly to its first row.
* Relaxation `rho_1000` and curve losses are fractions (0.031 for 3.1%, range 0–0.2); `tensioned_at` may not be later
  than the analysis start, and defaults to `solver.age` (or the first step's age if that is earlier); a continuation
  from `initial_state` keeps the original stressing age. Besides fibre tendons, a bar section on a named region with
  `prestress` or `unstressed_length` relaxes; a material used by both prestressed bars and beams relaxes only the
  prestressed bars, and relaxation on a material none of whose members carries a prestrain is refused.
* `thermal_expansion`: coefficient of linear thermal expansion (1/K), required by temperature loads (Section 11.4)

---

## 7 Sections and assignment

A section is "geometry + material", and it is the object that gets assigned.

### 7.1 Section types

| type | Requires | Corresponding element |
|---|---|---|
| `solid` | `material`; optional `formulation` (`b-bar`, `full`, `reduced`, `selective`), `hourglass_coeff` | The solid element determined by the mesh |
| `shell` | `material`, `thickness`; optional `shear_correction`, `drilling_factor` | `shell4` / `shell3` |
| `beam` | `material`, `shape` or directly `area, iy, iz, j`; `theory: timoshenko` additionally needs `ay, az` | `beam2` / `beam2_timoshenko` |
| `bar` | `material`; `shape: circle` (`diameter`) or `shape: general` (`area`) | `truss2` on a line region; embedded reinforcement on `reinforcement` |
| `fiber` | See 7.2 | `beam2_fiber` |
| `connector` | `stiffness` (six components) and/or `connector_damping`, `curves` | `connector2` |

Beam section shapes: `circle` (`diameter`, optionally with wall thickness `t`), `rect` (`width`, `height`, optionally with `t` for a box),
`I` (`bf, tf, hw, tw`), `box_girder` and `t_girder` (the drawing's dimensions, 7.2 below), `general` (section properties given directly). Explicitly given section properties override the values computed from the shape.

A solid mesh can have several `solid` sections (material zones): assignments are executed in order, later ones overriding earlier ones, so assign
the whole first, then the local parts. An element belongs to the section whose region contains "all of its nodes".

### 7.2 Fibre sections

```yaml
sections:
  rc:
    type: fiber
    material: concrete          # default fibre material
    shape: rect                 # rect | tube | box | patches | box_girder | t_girder
    width: 0.25
    height: 0.35
    ny: 8
    nz: 8
    integration: lobatto3       # gauss2 (default) | lobatto3 | lobatto4 | lobatto5
    j: 1.2e-3                   # Saint-Venant torsion constant, always give it for box girders
    bars:
      - {y: 0.09, z: 0.14, section: bar20}
      - {y: -0.09, z: 0.14, section: bar20, count: 2}   # count: several identical bars lumped at one point
```

* `rect`: uniform `ny × nz` division; `tube` (circular tube: `diameter, t, nt`); `box` (rectangular tube: `width, height, t, ny, nz`)
* `patches`: a section assembled from arbitrary quadrilaterals, each block `{corners: [[y, z] × 4], ny, nz}`, optionally with circular holes
  `holes: [[y, z, r], ...]`. Box girders, T-beams and hollow slabs all use it. **The coordinate origin is the member axis**;
  to place it at the centroid, compute the centroid yourself first
* `j` (legacy spelling `torsion_constant`): if not given, the polar moment Iy + Iz is used, which for box girders is several times too large (Hangzhou Bay box girder: Bredt formula
  29.5 m⁴ versus polar moment 150 m⁴), making torsion and mode shapes noticeably too stiff
* `integration`: integration points along the member. Lobatto includes the end points and is more accurate when the member end sections yield first
* `cover`, `core_material`: different materials for the cover and the confined core (`cover` needs `core_material`)
* A fibre section refuses keys its shape does not read (e.g. `nt` or `t` on `rect`, `width` on `tube`); a patch reads `{corners, ny, nz, material?, holes?}`
  and a bar `{y, z, section, count?}` — a bar's area and law come from the bar section it names
* A model can have several fibre sections, each assigned to its own region (e.g. each segment of a variable-section pier)
* Mass and weight: every fibre weighs and vibrates at its own material's density — concrete fibres, bars and tendons, in each section and each patch. Before, a fibre member used the first fibre section's cover density for its whole area. `vfife info` prints the model's total mass and weight; compare the weight with the sum of the reactions.
* `shear_stiffness`: member shear flexibility (elastic; the fibres do not see shear)
* `shear`: **cracked shear law**, mutually exclusive with `shear_stiffness`:

  ```yaml
      shear:
        direction: z                 # shear force along the section depth (plane of bending about y); or y
        stiffness: 1.25e9            # uncracked κGA (N)
        stirrups: {area: 2.0e-4, spacing: 0.125, depth: 0.35, E: 2.0e+11}   # or cracked_stiffness: 9.8e7
        cracking: flexural           # shear cracks when the member concrete first cracks in flexure; or give a shear force (N)
        # yield_force: 3.0e+5        # optional: stirrup yielding, after which stiffness is multiplied by hardening (default 0.01)
  ```

  * Shear acts **in series** with the flexure of the fibres: the member gains one shear rotation γ satisfying M_i + M_j = L·V(γ).
    In the elastic range this is exactly a Timoshenko member (matches PL³/3EI + PL/GA to 1e-8).
  * The post-cracking stiffness can be given directly, or stirrups can be given and it is computed from the 45° truss model:
    GA_cr = ρ_v/(1 + 4nρ_v)·E_s·b_w·d (Park & Paulay 1975).
  * After cracking, unloading and reloading both follow the secant pointing to the origin.
  * NumPy backend only (`backend: numba` is rejected).
  * This is not MCFT: shear is not coupled to the axial and flexural strain fields, and the fibres do not see shear stress.

For the remaining options of fibre sections (prestressing tendons, fracture, strain penetration) see Chapter 11 and `docs/fiber_beam.md`.

#### Box girders, T girders and bars in layers (1.2)

A girder is written by the dimensions on its drawing. The same `shape` works on a `beam` section (the properties are integrated from the outline) and on a `fiber` section (the outline becomes `patches`), so a beam and a fibre model of one girder agree on A, Iy and Iz; a 20 m cantilever under a tip load deflects 0.3–0.5% more in the fibre model than in the beam (the fibre grid's Iy is that much low), and twists the same. The lateral modes of a wide deck differ more (a 12 m deck: 15.0 against 15.8 Hz) because the fibre element carries the section's rotary inertia and the beam element does not.

```yaml
sections:
  deck:
    type: fiber                  # or beam
    material: c50
    shape: box_girder
    height: 2.5                  # top of deck to bottom of the bottom slab
    top_width: 12.0              # tip to tip
    bottom_width: 6.0            # bottom slab, outside to outside
    cantilever: 2.5              # tip to the web's outer face, at the top of the deck (0: no cantilevers)
    top_thickness: 0.25
    bottom_thickness: 0.22
    web_thickness: 0.45          # measured horizontally on an inclined web
    cantilever_thickness: [0.5, 0.2]   # at the web, at the tip (default top_thickness)
    top_haunch: [0.6, 0.2]       # [width, height] in each inside top corner (optional)
    bottom_haunch: [0.3, 0.15]   # optional
    # cells: 2                   # interior webs, vertical, evenly between the outer webs
    # layers: 3                  # fibre section: fibres through each wall (default 3)
    # fiber_size: 0.3            # fibre section: longest fibre along a wall (default height / 8)
    # origin: centroid           # fibre section: centroid (default) | top | bottom
    bar_layers:
      - {section: d16, from: [-5.9, 0.05], to: [5.9, 0.05], spacing: 0.15, z_from: top}
      - {section: d25, from: [-2.8, 0.06], to: [2.8, 0.06], count: 24, z_from: bottom}
```

* The webs run straight from the bottom slab's corners to the points `cantilever` in from the tips; a `bottom_width` smaller than `top_width − 2·cantilever` makes them lean out.
* `shape: t_girder` reads `height, top_width, top_thickness` (the flange at the web), `web_thickness`, `cantilever_thickness: [top_thickness, tip]` (a tapered flange; a root other than `top_thickness` is refused), and `bottom_width` + `bottom_thickness` for a bottom flange (an I or bulb-T).
* **Origin.** The section's origin — where the member line passes — is the centroid unless a fibre section says `origin: top` or `bottom`. Bars and tendons are written about it; `z_from` places bars from a face without knowing where the centroid is. Bars must lie in the walls (the girder's outline is exact, not the 1% a hand-drawn patch section is allowed); a tendon may also run in a box's cell (external prestress). A beam section is always about its centroid and refuses `origin`. An axis off the centroid couples a member's linear axial and cubic bending fields: under a tip load, 0.11% stiff with 2 m members, 0.02% with 1 m.
* **Torsion and shear (beam).** For a box, `j` is Bredt on the wall centre lines — a multi-cell box by its cells' shear-flow equations — plus ∫t³ dy/3 for the cantilevers; thick walls make it a few percent low (a finite-difference solution of the 12 m box gives 12.5–12.8 m⁴ against 12.19). For a T or an I, `j` is Saint-Venant's problem solved on the outline itself (Prandtl's stress function by finite differences, within 1%); the sum of its rectangles was 11–26% low. A fibre girder gets the same `j` (not the polar moment). `az` is the webs' area times cos² of their lean, `ay` the slabs' area (5/6 of the flanges' for a T). These are hand-check values; give `j`, `ay`, `az` (or `area`, `iy`, `iz`) to override.
* **Fibres.** The default is about two hundred fibres for a single-cell box (the Hangzhou Bay girder in case_24d ran with 166).
* **`bar_layers`** (any fibre section that takes bars: rect, patches, the girders): each layer is `{section, from: [y, z], to: [y, z], count | spacing, z_from}`. Both ends get a bar; `count: 1` puts one bar at the middle; `spacing` puts as many bars as fit at that spacing or closer (d16 @ 150 over 11.8 m is 80 bars). `z_from: top` measures z downward from the top face, `bottom` upward from the bottom face, `origin` (default) is the section axes. The layers become `bars`, and the bars' outline check applies to them.
* A girder's keys on another shape, or a key the shape does not read (`width` on a box girder, `layers` on a beam), are refused by name.

### 7.3 Connector sections

```yaml
sections:
  fixed_pot:
    type: connector
    material: concrete
    stiffness: [5.0e+10, 1.0e+9, 1.0e+10, 1.0e+10, 1.0e+10, 1.0e+7]
```

The six components are, in order, the stiffnesses in local ux, uy, uz, rx, ry, rz. A connector is a spring (optionally with damping) between two nodes;
its typical use is a bearing. The local directions are determined by `orientation` and `coordinate_system: local` in the assignment.
A component may also carry a nonlinear law (gap, bilinear hysteresis, friction, a nonlinear viscous damper, a friction pendulum, or a one-dimensional law of the material table); see 11.11 and `docs/connector2.md`. A connector section's `material` takes no part in the computation (a connector's law is its stiffness and its laws). A `coordinate_system: global` connector with length returns the couple its end forces leave half to each end (its springs are fixed in space, so a rigid rotation strains them).

### 7.4 Reinforcement geometry

Reinforcement in solid members is placed by drawing coordinates, independent of the mesh:

```yaml
reinforcement:
  main:     {layer: {from: [0, 0.0443, 0.0443], to: [2.0, 0.0443, 0.0443], count: 3, spacing: [0, 0.0557, 0]}}
  stirrups: {hoop: {corners: [[0.0443, 0.0443], [0.1557, 0.258]], axis: x, from: 0.05, to: 1.95, spacing: 0.075, divisions: 8}}
  tendons:
    path:
      points: [[0.0, 0.09, 0.30], [4.0, 0.09, 0.10], [8.0, 0.09, 0.30]]
      shape: parabola
      segment_length: 0.25
      repeat: {count: 4, offset: [0.0, 0.04, 0.0]}
```

Primitives: `line` (start and end points, optionally `repeat`), `layer` (a row of `count` parallel bars), `hoop` (rectangular stirrups repeated along
an axis), `lines` (a list), `path` (polyline or parabola, used for curved prestressing tendons).

### 7.5 Assignment

```yaml
assign:
  - {region: all, section: body}
  - {reinforcement: [main], section: bar15, segment_length: 0.025, name: tension}
  - {reinforcement: tendons, section: strand12, name: tendons, prestress: 1000.0e+6}
  - {region: beam, section: girder, orientation: [0, 1, 0]}
  - {region: bearing_1, section: fixed_pot, orientation: [1, 0, 0], coordinate_system: local}
```

Each assignment targets either a `region` (a node set on the mesh) or `reinforcement` (one or several groups of bars).
Options on an assignment belong to "this pair":

* Beams: `orientation` (direction of the section's local y axis), `releases`, `end_offsets` (see the end of this section); connectors: `orientation`, `coordinate_system` (a connector has no `releases`)
* Trusses/cables: `prestress`, `unstressed_length(s)` (written by `vfife formfind`), `tension_only`; `jack` (1.13, a force in N): in the analysis step that builds the cable (the first, when it is built from the start) the jack holds `jack` times the step's progress whatever the structure gives; at the step's end the cable carries exactly `jack`, and from then on it is a bar of the unstressed length it ended with. `prestress` states a length (the one that gives that stress in the mesh geometry) and loses force as the structure comes to it; `jack` states the force. One of the four. Frame models (beams, bars, connectors); refused in a fibre model
* Embedded reinforcement: `bond` (bond–slip), `segment_length`, `host_deduction`, `anchorage`, `prestress`
  * Embedded bars carry inertia from their material's `rho` but **no weight**: `gravity` integrates the solid blocks' densities only. With the customary 2500 kg/m³ for reinforced concrete the steel's weight is already in the concrete, so nothing is counted twice. If the concrete is given a plain-concrete density (about 2400), add the bars' weight as a load.
* Fibre beams: `anchorage: {region, bar_diameter}` (strain penetration); fibre beams have no `releases`

A section's `width` runs along local y, its `height` and a fibre's `z` along local z; local y follows `orientation`, by default
global +Z, so a horizontal member's local y is vertical and its local z horizontal unless told otherwise. `vfife info` prints
each line-member group's local axes in global coordinates, e.g.
`axes  girder (beam2_fiber, 42 members): local x = +X, local y = +Z, local z = -Y`. When a rect or box section (width not
equal to height) lies with its `height` horizontal on a horizontal member, a `note` line gives the `orientation` that turns
local z up, e.g. `note  beam: section height 1.2 m lies along local z = -Y (horizontal); if the depth should be vertical, set
orientation so local z is up (e.g. orientation: [0, 1, 0])`. Patches sections (whose y and z the user drew), circles, tubes,
sections given by A/Iy/Iz, bars and connectors get the `axes` line only.

A key the pair does not read (`releases` on a fibre section, `prestress` on a beam, `friction` on a cable, ...) is
refused by name; these were accepted and dropped before v1.0. Sections likewise: `tendons` on a beam section and
`theory` on a fibre section are refused. A single fibre section's region must contain every fibre member; several
sections take several assignments.

Elements not covered by any assignment, or sections that receive no element at all, are rejected with the element numbers given
(`ELEMENTS_UNASSIGNED`, `SECTION_EMPTY`).

#### End offsets `end_offsets` (1.21)

A member's elastic part need not run from node to node: `end_offsets` gives the vectors from the nodes to the ends of the
elastic part, and the stretch between a node and its elastic end is rigid and turns with the node.

```yaml
assign:
  # a rigid end zone: the 0.4 m of the beam inside each column is rigid
  - {region: cross_beam, section: girder, orientation: [0, 0, 1], end_offsets: {i: [0.4, 0, 0], j: [-0.4, 0, 0]}}   # 0.4 m at each end of the line
  # a section set off its nodes: the nodes on the top line, the centroid 1.2 m below (a tapered girder gives each end its own)
  - {region: girder, section: box, orientation: [0, 1, 0], end_offsets: {i: [0, 0, -1.2], j: [0, 0, -1.2]}}
```

* The offsets are the **line's**: a region meshed into several members must be one chain without branches; `i` is at the
  line's start (the point its mesh is drawn from), `j` at its end. Along the members (x) an offset is a rigid zone, at the
  chain's two ends only; across them (y, z) it is the section set off its nodes, linear in the arc length from `i` to `j`
  at every node (the same everywhere when the two are equal). A zone longer than the line's end member is refused; mesh finer.
* `axes: local` (the default): components along each member's own axes, x in the chain's direction, y from `orientation`,
  z = x cross y (on a member drawn against the chain, z is minus its section's z). `axes: global`: one vector in global
  axes, linear in the arc length from `i` to `j` at every node, the ends included, with no rigid zone of its own; give a
  rigid zone in `local`. An end not given has no offset.
* A model with one section assigned to `all` must then be one chain; give a frame's lines their own assignments.
* `per: member`: every member of the region has the two offsets, `i` at its first node and `j` at its second, along
  and across alike; the region need not be a line (the Midas converter gives each element's offsets this way).
* The elastic part's length, local axes and stiffness are those of its two elastic ends; a node gets the elastic end's force
  and the moment r x f. The reference stiffness (the time step, the relaxation masses, the modes) holds the rigid stretch.
* **Loads are on the drawn line** (the line between the nodes): a `line_load` is per unit length of it, and a moving
  load's or a vehicle's station is measured along it. What falls on a rigid zone goes to the zone's node rigidly, force
  and the moment of its lever; what falls beside the elastic part goes to the elastic axis at the same place along it,
  with the moment of the lever across (a sideways load on a girder drawn on its top line twists it). The elastic part's
  share is the consistent one, so a vehicle reads the deck's motion under its wheel back through the same vector.
* **The mass is where the section is**: the elastic part's mass, lumped at its two ends, is carried there by the
  nodes -- the parallel-axis inertia, the coupling of translation and rotation and the centripetal terms are in the
  explicit step and the modes; its weight and a ground motion's inertia act there (a girder drawn on its top line
  weighs on its centroid and does not twist). The explicit step shrinks where that coupling needs it (logged). The
  rigid zone weighs as the member it is part of -- its mass per length times the zone's length along it -- carried at
  the zone's middle; an offset across the member (the arm to a section set off its nodes) has no mass.
  `end_actions` are at the elastic ends. The mass of a `line_load` with `with_mass` and a `hydrodynamic_mass` are
  still over the length between the nodes, at the nodes.
* Frame beams (`beam2`, `beam2_timoshenko`, the beams of `mixed_line`) and fibre beams (a `type: fiber` assignment);
  refused on bars and connectors. A member with offsets breaks, splits off and is removed as any other: its rigid
  stretches follow its ends onto their new particles. In contact between members its capsule is on the elastic axis.
  A buckling analysis takes the rigid stretches and their levers (checked against a closed form for rigid zones; a
  section set across the member under compression, 1.23: on forks it is the beam-column of its eccentricity, M = P e, to
  5e-4, and with an end free to twist -- where the lever turning with the node makes it 0.8% from fixed end moments -- the
  model's own nonlinear tangent turns singular within 0.5% of the factor). The moment of the lever, r x N, couples twist
  with bending as a moment does, but without the member's Kirchhoff factor: on a section with equal second moments (whose
  own moments carry none) set off sideways by 1.5 radii of gyration a column buckles 3.5% below its Euler load, where the
  same column of stiff arms gives 1.000 -- conservative, and the reference between the two is not settled. The compiled step and the compiled relaxation take
  such a model step by step (see the capability table).
* `axial: elastic` (the default) keeps a rigid zone rigid in every direction (OpenSees, Nastran); `axial: full` takes
  a frame beam's axial and torsional stiffness over the length between the nodes, the zone rigid in bending and shear
  only (Midas, ETABS; the Midas converter writes it). Refused on fibre beams. A temperature's or shrinkage's free
  strain still acts over the elastic part.
* Offsets that bring the elastic ends together, or turn the elastic part round, are refused.

---

## 8 Regions

Regions are node sets; constraints, loads, assignments and monitors all refer to them by region name. There are three sources: Gmsh physical groups, the built-in
`all` and `surface`, and user-defined selectors under `regions:`:

```yaml
regions:
  base:      {plane: {point: [0, 0, 0], normal: [0, 0, 1]}}
  support:   {where: {x: 0.10, z: 0}}                  # a line: two coordinates fixed
  pin:       {where: {x: 0.10, y: 0.10, z: 0}}         # a point
  web:       {where: {y: {min: -0.05, max: 0.05}}}     # a coordinate range
  box1:      {box: [0.35, 0.45, 0.0, 0.2, 0.2625, 0.3]}  # xmin xmax ymin ymax zmin zmax
  diagonal:  {line: {from: [0, 0, 0], to: [2, 0, 0.3]}}
  bolts:     {cylinder: {point: [1, 0.5, 0], axis: [0, 0, 1], radius: 0.05}}
  shear:     {intersect: [base, web]}
  ends:      {union: [left_face, right_face]}
  interior:  {difference: [all, surface]}
```

There are also `sphere: {centre: [...], radius: r, rmin: r0}` (`center` is read too), `group`, `node_ids` and `nearest`. A
region may refer to regions defined after it. A misspelt or missing key in `plane`, `line`, `cylinder` or `sphere` is
refused by name, as is a `node_ids` entry past the last node.

Two rules to remember:

* **A region that selects no nodes is an error**, not an empty set — a model whose tolerance is slightly too tight, so that a support just misses, looks the same as a correct one
* A single number in `where` **snaps to the nearest line of nodes**, and preflight reports the coordinate actually used
  (`REGION_SNAPPED`); two or three pinned coordinates snap together, to one node

The default tolerance is 1e-6 of the mesh diagonal, so the same file works in both millimetres and metres.

---

## 9 Constraints, loads and amplitudes

### 9.1 Constraints

```yaml
constraints:
  - {region: base, dofs: [1, 1, 1]}                                  # solid: ux uy uz
  - {region: left, dofs: [1, 1, 1, 0, 0, 0]}                         # beam: ux uy uz rx ry rz
  - {region: head, dofs: [1, 0, 0], value: 0.010, amplitude: drift}  # prescribed displacement
```

The length of `dofs` equals the number of nodal degrees of freedom (3 for solids, 6 for beams and shells). `value` gives a prescribed displacement, which can vary with an amplitude;
in dynamic analysis `motion: velocity | acceleration` means that a velocity or acceleration is prescribed.

Under dynamic relaxation, unconstrained rigid-body motion of the structure is an error (a free body has no equilibrium position); preflight points out which
motion it is and which constraint needs to be added; in dynamic analysis it is only a warning.

### 9.2 Loads

Each load has one `region` (default `all`) and gives only one kind of load quantity:

| Key | Meaning | Unit |
|---|---|---|
| `vector` | Resultant force on the region, divided equally among the nodes (with `per_node: true` each node receives this full amount) | N |
| `moment` | Couple (beam nodes; or a fixed-axis couple on a solid face, with `follower: true` rotating with the face) | N·m |
| `line_load` | Uniformly distributed line load along the member, global coordinates by default, local with `coordinate_system: local` | N/m |
| `traction` | Traction vector on the outer surface | N/m² |
| `pressure` | Pressure on the outer surface (positive inward); `follower: true` updates the direction with deformation | Pa |
| `body_force` | Body force | N/m³ |
| `gravity` | Gravitational acceleration, multiplied by the material density | m/s² |
| `temperature` | Temperature field over the section (Section 11.4) | K |
| `moving_load` | Axle group moving along members (Section 11.5) | N |
| `ground_motion` | Ground acceleration record (Section 11.6) | m/s² |

Surface loads are integrated over the actual outer surface area and body loads over the element volume, independent of node density.
The units column is for an SI case; in another declared system each quantity is in that system's unit (N/mm, MPa, mm/s², ...).

### 9.3 Amplitudes

An amplitude gives how a load or prescribed displacement varies with **analysis-step progress** (0 to 1, not physical time):

```yaml
amplitudes:
  live:    [[0.0, 0.0], [1.0, 1.0]]
  service: {table: [[0.0, 0.0], [0.2, 1.0], [0.8, 1.0], [1.0, 0.0]]}
  drift:   {cyclic: {peaks: [0.333, 0.667, 1.0], repeats: 2}}
```

Built in are `ramp` (0→1), `hold` (full value from the start) and `ramp_hold` (ramp to full in the first half of the step, hold in the second half).
Defaults: `ramp` for incremental steps, `ramp_hold` for continuous steps. In a time step (`to_age`) loads and prescribed displacements without an amplitude default to `hold` (full at the first time station, held); ramping over time needs an amplitude written explicitly. A table may exceed 1 — `[[0, 0], [1, 20]]` pushes the load
to 20 times; this is the usual way to write a pushover analysis.

### 9.4 Loads across analysis steps

* Model-level constraints and loads are the "base": they are applied in the first step and held thereafter
* A step's own constraints and loads are active only in that step (with `keep: true`, 1.2, in every later step too, Section 10.1); where they conflict with the base on the same DOF, the step's take precedence
* Displacements, material state and fracture are passed on completely from step to step. Velocities carry on in a dynamic run (a dynamic
  step is a stretch of one time history, Section 10.2); a relaxation or static step starts from rest
* A `line_load` (or any load) is by default a force, not a mass: superimposed dead load written as a `line_load` weighs but does not
  vibrate. When the load is the weight of something (superimposed dead load, surfacing, parapets), write `with_mass: true` (1.1): it
  carries the mass |q|/g too, lumped to the members' end nodes in all three translations, with inertia in a time history, the modes and
  a spectrum. That mass weighs nothing (the load is its weight), so `gravity` does not count it twice; g is 9.80665 m/s² in the declared
  units (9806.65 in mm/N). Model-level `loads` only; translational mass only, no rotary inertia.
  ```yaml
  loads:
    - {region: deck, line_load: [0.0, 0.0, -1.2e+5], with_mass: true}   # 120 kN/m of SDL, with its 12.2 t/m
  ```
  An added mass that is not uniform along the members is still `nodal_masses` (`weight: false` if a load already carries the weight),
  or folded into the section's density (Section 9.5)

### 9.5 Additional mass

```yaml
nodal_masses:
  - {region: pier_top, mass: 4.0e+4}          # divided equally among the region's nodes; with per_node: true, per node
  - {region: deck, mass: 2.0e+5, components: [1, 1, 0]}   # only in x, y: added mass, no weight
  - {region: tower_top, mass: 1.0e+4, inertia: [0, 0, 5.0e+6]}   # rotary inertia about z, kg·m²
hydrodynamic_mass:
  - {region: piles, diameter: 4.0, coefficient: 1.0, density: 1025, water_level: 0.0}
```

From v1.0:

* **By default** this mass acts in all three translational directions, and gravity sees it too (as before).
* `components` makes the mass act only in some translational directions. Directional mass is **added mass** (water, soil, a deck that follows
  in only one direction) and **has no weight**: `weight` defaults to false, and writing `weight: true` is rejected — an object that has weight
  has inertia in every direction. Three-directional mass may also be written with `weight: false`.
* `inertia` is the rotary inertia about the global axes, used only for beams and shells (six DOFs); trusses and solids reject it.
* `hydrodynamic_mass` is the Morison added mass of submerged members: C_a·ρ_w·A per metre (`diameter` for circular sections,
  `area` for other sections), counting only the part whose vertical coordinate is below `water_level`, distributed to the two end nodes by the lever rule.
  * It acts only in the two directions **perpendicular to the member** and has no weight.
  * The member's true angle to the nearest global axis must be within 5° (piers, piles): lumped mass is diagonal, so "perpendicular to the axis" of an inclined member cannot be expressed, and it is rejected. From v1.0 the true angle is checked (previously each coordinate separately, so a member 4.9° off in two planes -- 6.9° in all -- was accepted).
  * During an earthquake it participates in −M·a_g together with the structural mass, and it is also included in the effective mass of the response spectrum.
* Nodal masses on trusses and solids now also have weight under `gravity` (previously only the beam family did).

Case case_37: a 20 m circular pier (D = 2 m) in water carrying a 500 t deck on top; under the response spectrum it differs from the closed-form solution by 0.02%–0.3%.
The water lengthens the first period by only 1.4% (the deck mass dominates) and the second period by 18%, and increases the base shear by 6%; the axial period is unchanged.

### 9.6 Rigid floor diaphragms

```yaml
diaphragms:
  - {region: floor_1, normal: z}
```

The nodes in the region move as a rigid body in the plane perpendicular to `normal` (two translations plus rotation about the normal), and remain free out of plane.

Modal analysis and the response spectrum also apply rigid floors (from v1.0; previously they were ignored): a floor's two in-plane translations and its rotation about the normal follow three master DOFs at the floor's centre of mass. Since 1.21 the static Newton solver takes its three masters at the same point (it took the nodes' mean point, the same constraint: the answers differ by round-off), and fibre-beam models take rigid floors too (frame, fibre-beam, shell and shell-line models; not the warp shell backend).

### 9.7 Master-slave rigid links (1.21)

```yaml
rigid_links:
  - {master: tower_top, slaves: saddle, dofs: [0, 1, 1, 1, 1, 1]}   # a saddle: free along x
  - {master: N12, slaves: anchors_12, dofs: [1, 1, 1, 0, 0, 0]}     # translations only: a hinge
  - {master: deck_a, slaves: deck_b, dofs: [1, 1, 1, 0, 0, 0], arm: false}   # equalDOF
```

`master` is a region of exactly one node, `slaves` a region of one or more; `dofs` are six flags in the global axes ux,
uy, uz, rx, ry, rz, 1 where the component follows the master (Midas' `*RIGIDLINK` flags; OpenSees' `equalDOF` and
`rigidLink`). The constraint is in **large rotation**: a slave's translation is the master's plus the arm turned with
the master (without the arm when `arm: false`, equalDOF); with the three rotations tied the slave's orientation is the
master's times their relative orientation when the link was made, with some of them tied those components of the
relative rotation vector are zero. The arms and relative orientations are taken where the nodes stand when the run
starts, so a link is made free of force.

**In analysis steps** (1.21): a step's own `rigid_links` (written as the model's, with a `name` to release them by)
are made at the step's start, after its members are built, where their nodes then stand -- free of force, as a member
born in a step is. A step's `release` may name a link: it is freed at the step's start, and the force it carried at
the end of the step before (on each slave's tied components, and the opposite on the master through the arm) is a
load taken off over the step, as a released support's reaction is (the force is f_int - F at the slave, exact at
rest; after a dynamic step that ends in motion it leaves out the slave's inertia, as a released support's reaction
does). A link is made once and released once; a name that is both a region and a link is refused, and a slave's
tied dof may not be one a support of the model or of any step names. In a dynamic step a link made between nodes in
relative motion sets the slave's tied velocity to the master's at once (not a momentum-conserving merge). `vfife modal` and `vfife buckling` at a result's frame read which links
tie there from the checkpoint, so a model whose steps make or release links is asked at the result's last state.

How: on the acceleration -- the master accelerates under its own force and its slaves' tied forces (with the moments of
their arms), through its own mass and theirs, and the slaves' tied components follow; after each position update the
slaves' tied components are set from the constraint again, removing round-off. The relaxation's residual is the
master's resultant (a slave's tied residual is a constraint force). A master and a slave at one position (equalDOF)
are not reported as duplicate nodes.

The scope of this version: frame, fibre-beam, shell and shell-line models (`beam2`, `beam2_timoshenko`, `mixed_line`,
`beam2_fiber`, `shell3`, `shell4`, `shell_line`), in the explicit dynamic step, the dynamic relaxation and the modal and
buckling analyses, and in the static Newton solver on the models it runs (frame and fibre-beam; 1.21: one constraint
layer, `core/constraints`). Newton solves the
master system Tᵀ K T with T at the current configuration, made again as the masters turn; the constraint's curvature is
not in the tangent, which can slow Newton where the links carry large forces, not change its answer. A model with
`rigid_links` keeps the step-by-step loop instead of the compiled step and relaxation ("master-slave rigid links" in the
capability table). Refused: the warp shell backend, solid, truss and coupled models, links in an analysis step of
a shell model (shells take no analysis steps that change the model), links in a chain (a slave that is another link's master), a node that is the slave of two links, a slave in a rigid
diaphragm, and a tied component a support also holds. The checkpoint carries each link's arms and relative
orientations; a state written without them (an `initial_state` relaxed before the links were added) is used, the links
made where the run starts. A partial tie of the rotations wants small relative rotations: the spins are tied component
by component, and at tens of degrees of relative rotation the orientation and the spin drift apart. A support on a
master (held or driven) acts as usual and its slaves follow; a master in a rigid diaphragm is refused.

---

## 10 Analysis steps and solvers

### 10.1 Analysis steps

```yaml
steps:
  - name: axial                      # model-level loads are applied in this step
  - name: cyclic
    constraints:
      - {region: head, dofs: [1, 0, 0], value: 0.010, amplitude: drift}
    load_steps: 8
    max_steps_per_load: 20000
```

Each step can override `n_steps`, `load_steps`, `max_steps_per_load`, `stage_ramp_iterations` and
`tolerance`; `mass`, `backend` and `damping` belong to the model and cannot be changed per step. Steps of creep cases also have
`age` and `to_age` (Section 11.3).

#### Steps with their own procedure, members built and removed (1.2)

On a line model (`beam2`, `beam2_timoshenko`, `mixed_line`, `beam2_fiber`) a step can change the procedure and the
model, the way an Abaqus step does:

```yaml
steps:
  - name: span_1
    activate: [span1]                  # members born in this step
    load_steps: 4
  - name: span_2
    activate: [span2]
    birth: tangent                     # tangent (default) | design
    procedure: static                  # relaxation | dynamic | static; default solver.mode
    load_steps: 4
  - name: strike_falsework
    deactivate: [falsework]            # members removed: their forces come off over the step
    release: [temporary_pier]          # inherited supports freed: their reaction comes off over the step
                                       # (a rigid link's name frees the link, 9.7)
    load_steps: 4
  - name: earthquake
    procedure: dynamic
    total_time: 20.0                   # a dynamic step's duration; default solver.total_time
    loads: [{ground_motion: {path: elcentro.txt, direction: [1, 0, 0]}}]
```

* **`procedure`**: a relaxation, static or dynamic step in one run. A dead-load state relaxed or solved statically and handed to a
  dynamic step starts at rest and in equilibrium: with nothing new on it, it does not move (tested to 1e-9 of the deflection). Each
  procedure keeps its own settings: a relaxation step its fictitious mass and kinetic damping, a dynamic step the physical mass, the
  stable step and `solver.damping` (which must not be `kinetic` then). In a model whose `solver.mode` is `static`, every step is static.
* **`activate`**: the members of the regions (for a region that is a group of lines -- a Gmsh physical group of lines, a `lines` group -- that group's own members, as `assign` means it; for a region given by a box or by points, the members with both nodes in it) are not there before this step -- no stiffness, no mass, no weight, no line
  load -- and are born in it **without stress, where the structure has moved to**: their local deformation at that moment (stretch and
  both ends' rotations) is their reference. Their weight and the model-level line loads on them come on over the step (Abaqus'
  reactivation), the rest stays held. A node that no built member reaches is held in every direction and reports no reaction.
* **`birth`**: where the nodes that only the new members reach are put. `tangent` carries each of them rigidly -- displacement and
  rotation -- with the standing node it is reached from first: a cantilever segment set on the tangent of the tip. A translation a
  support holds stays at the support (a roller at the far pier; under a plane restraint the other directions still follow the tangent),
  so a span cast between a deflected pier head and the next pier is born stress-free with a kink at the far end -- for a span cast on
  falsework to the drawing, write `design`, which leaves the nodes at their coordinates. A new group touching nothing that stands is
  always at its design coordinates. A bar with a designed `prestress` or `unstressed_length` is installed to that length: it pulls as
  designed from the moment it is born (a stay tensioned as it is put in), while every other member is born stress-free.
* **`deactivate`**: the members are removed, and what they did to the rest -- the loads on them (their weight and line loads, at
  the values the step before ended with) less their internal force on their nodes -- is applied as a load taken off over the step
  (Abaqus' removal), so the structure does not feel a jump. Their mass and rotary inertia leave with them, and so does their share of
  stiffness-proportional damping.
* **`release`**: frees the constraints the regions' nodes inherited (model-level, and ones kept by earlier steps); their reaction at
  the end of the step before is taken off over this step. The step's own constraints still apply, and a constraint kept by a LATER step
  holds again. A release of nodes no inherited constraint holds is refused.
* **`keep: true`** on a step's load or constraint: it stays on in every later step, at the value it ended its own step with (a kept
  constraint until a `release`). Without it a step's load is that step's alone, as before.
* **A support put under a node that has moved holds it where it stands** (1.14; beams, bars, connectors and fibre beams): a
  constraint without a `value` that takes a degree of freedom which was free at the end of the step before holds it at the
  displacement it has then -- an arch foot hinged for the erection and then cast in, a prop put under a deflected girder -- and
  goes on holding it there in every later step that constrains it (`keep`). It does not bring the node back to its coordinates:
  to jack a point to a position, give the constraint that `value`. A rotation is held as it is when all three of the node's are
  constrained. Unchanged: a degree of freedom constrained from the first step, or from the step that builds its node, is at its
  coordinates (a support is where it is: the new node is put there, and with all three rotations held it is put there unturned,
  so the new member is born free of stress between the standing end and the support); and one that a `value` has moved and a
  plain constraint then takes returns to zero -- it was never free. This holds in any run of several steps, with or without
  `activate`, `release` or `keep`. Before 1.14 every such constraint held zero: the foot was turned back through all it had turned.
* **The time axis**: in a run with a dynamic step, the result's `time` is the clock -- a dynamic step moves it by its duration, a
  relaxation or static step does not (its frames sit at the clock). Each frame records `analysis_step` and `procedure`, and the
  members built in it (`member_built`, 1 or 0).
* **Each segment's own concrete age** (frame models: `beam2`, `beam2_timoshenko`, `mixed_line`; fibre beams since 1.8): a member born in a step is
  `activate_age` days old then (default `solver.age`), whatever the model's clock reads; from then on it creeps, shrinks and ages on
  its own clock -- phi(t + shift, t0 + shift) with nothing dated before its birth -- and its shrinkage in the formwork is not a load.
  Checked: a bar cast at clock 100 at age 7 and loaded then stretches N/EA (1 + phi(907, 7)) by clock 1000, beside one loaded at 28
  days with (1 + phi(1000, 28)); a free bar cast then shortens by (eps_sh(907) - eps_sh(7)) L; a held stretch on a bar born late
  relaxes station for station as on a bar loaded at its age. Listing a member in the FIRST step's `activate` makes it born then, free of
  its shrinkage before `activate_age`; a member built from the start carries its shrinkage from the drying start (the rule before 1.2).
* **The fibre beam** takes every model change and procedure. A fibre member born late is stress-free at its birth's mechanical
  state, and one built again after a removal starts from its fibres' first state (new concrete); a temperature already acting on it
  then is not subtracted (it is born carrying that temperature's restraint), nor is a tendon's prestrain (it acts at birth).
  Since 1.8 a fibre model with creep or shrinkage changes its model too: each segment creeps and shrinks on its own
  clock, the rule above, with the creep scale on the law's input set member by member (on the NumPy reference path the law's
  `strain_scale` per member, in the fused kernel its per-member parameters). The frames' checks hold on concrete02 fibres
  (tests/test_fiber_segment_ages.py): phi(907, 7) beside phi(1000, 28); (eps_sh(907) - eps_sh(7)) L; a held stretch on a late segment
  relaxing station for station as a segment loaded at its age in a model of its own, to 1e-6; the kernel and the reference to 1e-8;
  a run stopped while the late segment creeps resumes to the bit; a segment born in a step that also moves the clock is a bar built at
  its age in a model of its own, and one taken out and cast again is one cast once. A member not built does not size the residual.
  Ageing is refused on a fibre section whatever the steps (a nonlinear law's modulus is not one number).
* **A connector born in a later step** (frame models, 1.13; fibre models, 1.22): like a beam or a bar it is free of force where its ends stand when it is
  built -- its six deformations then are its reference -- and its laws start from nothing; removed and built again it is a new one.
  A connector carries no new node under `birth: tangent`, and neither does a cable being jacked: a node only they reach starts where the model has it.
* **Not yet** (refused by name): the shell and solid models; model
  changes with relaxing fibre tendons; `keep` on a moving load, a vehicle or a ground motion. (Model changes with relaxing prestressed
  or jacked bars run since 1.16: each bar relaxes from its birth or its lock-off, Section 11.3.)
* In a model with dynamic steps, `solver.damping` is the dynamic steps' (not `kinetic`; Rayleigh `beta` is allowed in a relaxation
  model then), and its relaxation steps take kinetic damping. The energy ledger is not audited in a run that mixes procedures.

Checked against closed forms: two equal spans cast one after the other give R_A = 7/16 wL and R_B = 9/8 wL (3/8 and 5/4 cast at once);
removing the second span leaves R_A = R_B = wL/2; releasing the middle support gives the single span's wL at each end; a cantilever cast
on its tangent ends where it would have cast at once, and one cast to its drawing differs by w_B + theta_B L.

### 10.2 Solution modes

```yaml
solver:
  mode: relaxation          # or static (implicit Newton), dynamic
  backend: numpy            # numpy | numba | warp
  damping: {type: kinetic}  # kinetic damping for relaxation; viscous / rayleigh / none for dynamics
  tolerance: 1.0e-5
```

**Dynamic relaxation (`mode: relaxation`)** solves for static equilibrium:

* The mass is fictitious (taken from the stiffness row sums, controlled by `relaxation_safety`), and the time step is 1
* **Continuous**: loading within one step, only the final equilibrium is reported
* **Incremental** (`load_steps: n`): the load is divided into n increments, each converged to `tolerance` with one frame output — incremental loading is required
  to obtain a load–displacement curve. The iteration limit per increment is `max_steps_per_load`, and at the start of each increment
  `stage_ramp_iterations` iterations are used to apply the load smoothly

**Implicit static (`mode: static`)** solves equilibrium increment by increment with Newton's method:

```yaml
solver:
  mode: static
  load_steps: 20            # increments per step (default 10); an analysis step's own load_steps takes precedence
  tolerance: 1.0e-5
  static: {max_iterations: 30, cutbacks: 10, line_search: true, tangent: full}
```

* The tangent is a finite difference of **the same internal force**: there is no second set of element stiffnesses.
  * Nodes that are neither adjacent nor share a common neighbour are perturbed together (distance-2 colouring of the node graph).
  * One tangent needs only 2 × DOFs per node × number of colours internal-force evaluations, typically 36–100 for frames, independent of model size.
  * Rotations are perturbed and updated as spins, consistent with the co-rotational formulation of the explicit solver.
  * Since 1.15, in a model of beams, bars and connectors under `backend: numba` (without plastic hinges, soil springs,
    breaking members or a rigid contact surface) a colour's columns are read from the members that touch its nodes alone -- no other member's force changes
    -- so a tangent evaluates each member 24 times whatever the number of colours, and the matrix is the whole force's to round-off.
    The run's log says which is used (`VFIFE_TANGENT_MEMBERS=0`, which differenced the whole force, went in 1.18).
* Each increment starts with a tangent predictor (prescribed displacements are spread through the whole structure by K_fc rather than all piled into one element next to the driven node).
  Then follows Newton with backtracking line search; if it does not converge the increment is halved and retried, but output frames always land on the increments you specified.
  * Monitors are written at every increment; `output.every` thins only the frames.
  * An increment is never halved below `cutbacks` halvings of the stated increment; still failing there, it is reported as a limit point (this is how load control past a peak ends).
  * The residual scale includes a millionth of the largest force (fibre forces included) any accepted state has carried; only accepted states raise it, and a restart carries it.
  * A step's own `tolerance` takes precedence over the solver's.
  * A tolerance below round-off is reported as "Newton stalled … round-off", not as a limit point: loosen the tolerance to the value it names.
  * Equilibria with no external force (a free thermal bow, a simply supported girder under a temperature gradient, prestress alone, a support released in a later step) solve by Newton.
* Path-dependent materials are always trial-computed from the last converged point and committed once after each increment converges (the same as incremental relaxation).
* Scope: frames and fibre beams (beam2, beam2_timoshenko, mixed_line, beam2_fiber).
  truss2 members inside a mixed_line model run as well; a stand-alone truss2 model (a single `bar` section on `all`) is refused (`STATIC_UNSUPPORTED`): use `relaxation`.
  Shells, solids, as well as fibre fracture, plastic hinges, the cracked shear law, timed removals and a failure with
  `split: true` are rejected with an explanation. Creep, shrinkage, ageing and relaxation run on beam and mixed-line models since 1.16
  (every converged increment of a step with `to_age` is a time station, Section 11.3), not on fibre models or under `arc_length`.
* **Rigid diaphragms (`diaphragms`)** (1.7) and **rigid links (`rigid_links`)** (1.21): a multi-point constraint,
  and Newton solves the master system Tᵀ K T (three masters per floor: the two in-plane translations and the rotation
  about the normal at the floor's centre of mass; a link's slave's tied components follow its master's dofs, through the
  arm as it is now -- the constraint relaxation applies to the accelerations). The residual is measured on the master
  system: the constraint forces at a floor's or a slave's nodes are not an imbalance.
* **Member failure (`failure`)** (1.7): a member that reaches its capacity in an **accepted equilibrium** is taken
  out whole (never split off: a fragment has no static equilibrium), the same level is solved again without it, and again
  while that cascades, until nothing new fails -- the alternate-path analysis, one member at a time; the failures are
  recorded in `member_failures` with the step plus its progress as their time. No equilibrium after a break is reported as
  "the structure has lost its load". Timed `removals` have no clock here: use a step's `deactivate`.
* **Stabilised Newton (`static.stabilise: true`)** (1.23): an increment the plain Newton cannot take is tried again,
  before it is halved (once per stated level; load control only, not `arc_length`), by pseudo-time: every dof is tied by a spring of `stabilise_mu` (1e-6) times the tangent's own
  diagonal to where the round began; each round is solved by Newton, then the springs' anchors move up and mu is quartered,
  until the residual without the springs is within the tolerance. The answer is the plain equilibrium; the springs are
  only the way to it. It is for a tangent that is right and ill-conditioned: a suspension bridge's deck whose end bays
  barely resist twisting (J = 1e-4 m4), whose tangent is accurate to four digits and whose softest directions turned the
  first correction into a hundred metres; stabilised, the bridge's dead load converged in ten increments. **Off by
  default, because it crosses limit points**: a snap-through Newton refuses under load control is carried onto the far
  branch, the jump a static analysis must report. The Midas converter turns it on for the bridges it writes. The log
  counts the increments it took (`… increment(s) by the stabilised Newton`).
* **Fallback to relaxation (`static.fallback: relaxation`)** (1.7): when Newton has halved the increment
  `cutbacks` times without converging -- a limit point under load control, or a jump in the equilibrium path -- the level
  is taken by a dynamic relaxation from the last converged state to the **stated** level (fictitious mass, kinetic
  damping, the algorithm of `mode: relaxation`, `max_steps_per_load` or `static.fallback_iterations` iterations per
  level), **and so is the rest of the step**; the next step starts with Newton again. The default is `none`, the former
  error (which now names the option). A shallow arch pushed past its limit point: Newton without a line search cannot
  take the snap, the relaxation reaches the inverted arch in 784 iterations, and Newton goes on from there
  (tests/test_static_fallback.py). **Which branch it lands on past a jump is the fictitious transient's choice**: on a
  softening section not necessarily the one `mode: relaxation` follows -- case_25 relaxed from Newton's post-peak state
  landed on a branch whose load went on rising to 335 kN while the strands unloaded, where `mode: relaxation` crushes and
  collapses at the same level. Both are equilibria of the discrete system; the contracts were written against the latter,
  so **the four reinforced-concrete pushovers (case_25, 25b, 30, 31) keep relaxation** (case_30 and 31 finished through
  the fallback within 1.6% of relaxation's load, but 12-16% apart in local strains). With the fallback on, the budget of
  halvings counts per stated level rather than being reset by every small success: a path that crawls at the smallest
  increment is on a jump, and falls back at once.
* **Which cases use it** (1.5; 1.7): every shipped static case that can -- elastic frames, a beam on
  bearings with laws, temperatures, a moving-load influence line, a reinforced-concrete frame short of its peak (1.5),
  the Confederation Bridge under its trucks and temperatures and the Hangzhou Bay girder under dead and live load (1.7:
  case_24 74 s → 10 s, case_28 102 s → 1.4 s) -- to the same contracts. The reinforced-concrete fibre beams pushed past
  their ultimate state (the pushovers of case_25, 25b, 30 and 31) keep relaxation, for the reason above; case_16 keeps it
  too, as the twin of case_16b's arc length.
* **Limit points**: under load control, a peak or a jump has no equilibrium to find, and the failing load factor is reported.
  To pass a peak, use displacement control, or the arc-length method:

  ```yaml
    static:
      arc_length:
        initial: 0.05              # load factor of the first increment, from which the arc length is set
        max_increments: 400
        target_iterations: 4       # arc length scaled by sqrt(target/actual iterations)
        stop: {region: tip, dof: z, value: -0.5}   # where to stop if the load factor cannot reach 1
        steps: [live]              # use arc length only for these analysis steps; other steps are load-controlled (if omitted, every step uses it)
        control: {region: tip, dof: z}   # 1.3: measure the arc on this one displacement (displacement control, load factor unknown)
  ```

  * Version 1.0. The loads and prescribed displacements of each step are all treated as one pattern, and the load factor λ becomes an unknown.
    The constraint is Crisfield's cylindrical arc length ‖W Δx‖ = Δl, with rotations weighted by model size.
  * It can pass limit points, snap-through and snap-back, and λ may become negative.
  * A step ends normally when λ reaches 1 (the last increment lands on 1 under load control) or when `stop` is reached.
    Running out of `max_increments` before either is a failure (MAX_STEPS), not an answer: raise `max_increments` or `initial`, or give a `stop`.
  * The arc-length corrector has no line search; `line_search` acts in the load-controlled increments and in each arc-length step's λ = 0 equilibrium and last increment landing on λ = 1.
    `tangent: modified` computes one tangent per increment and renews it when convergence slows; on strongly nonlinear paths `full` is usually faster.
    An increment within one iteration of `target_iterations` keeps its arc; an arc below a quarter of the first that converges three increments running without a cutback is doubled.
    The monitor `time` is the number of previously completed steps + λ (with only one step, it is just λ).
  * Arc length starts from equilibrium at λ = 0: if a step is not in equilibrium at its start (for example, in-step loads of the previous step are no longer applied in this step),
    equilibrium at λ = 0 is first found with Newton. **The `loads` in an analysis step belong to that step only**; loads that should always be present (dead load)
    are written in the top-level `loads`.
  * Common use: dead load + prestress in one Newton step, live load in one arc-length step (`steps: [live]`).
  * **`control` (1.3)**: the arc is measured on one node's displacement along x, y or z instead of the whole state:
    displacement control with the load factor as the unknown (OpenSees' DisplacementControl). Every increment moves that point on by Δl,
    so an equilibrium that would bring it back (the elastic unloading from the last state, for one) is not on the arc.
    * It cannot follow a snap-back of the controlled displacement itself. There an increment can converge on an equilibrium of another
      branch far away; such an increment is now refused: when the state moves more than 1.5 times as fast per unit of the controlled
      displacement as over the last increment, it is a jump and the arc is halved. When the cutbacks run out the run stops, and the message says
      this may be a snap-back. To pass one, leave out `control` and measure the whole state.
    * The region must hold one node; the dof must not be held in that step; the load pattern must move it.
    * The Hangzhou Bay pushover's second peak (case_24g, −0.38 m) is not passed with it either: 2 mm increments stop near −0.386 m, which looks
      like a snap-back, and the whole-state arc stalls there too. Coarser increments "pass" it by jumping to another branch, which is not the path.
  * Case: Lee frame (case_16b) traverses the whole path in 61 increments and 6 s.
    Limit load 18.577 EI/L² (NAFEMS 18.552), end point v/L = 0.781 (NAFEMS 0.784, Abaqus 0.781).
    The snap-back branch is on the path, whereas the displacement-controlled case_16 can only jump past it dynamically.
* The finite-difference step is `perturbation` × median element length (from v1.0; before that it was multiplied by the model's maximum coordinate,
  giving a step of 3e-5 m on a 580 m bridge, which straddled the cracking kink of the concrete fibres, so that near the peak Newton degraded to linear convergence).
* Performance:
  * Vecchio & Emara frame (case_21c) 8 s, versus 22 min for the same model with dynamic relaxation; the two differ by less than 0.3% at every increment read by the contract.
  * Cyclic loading of a fibre column (case_11) 17 s.
  * Pushover of one Hangzhou Bay unit with piers (case_24g): dead load in one Newton step, live load by arc length in 18 increments, about 30 s in a 2-core container (29 s at v1.0);
    the explicit dynamic version (case_24d) takes 50 min. After passing the peak k = 14.69 (at −0.186 m) it goes into the softening branch (k falls to 12.76 at −0.275 m) and then rises again to the stop at −0.38 m (k = 14.08).
    The dead-load midspan deflection is −23.47 mm (−23.18 mm before v1.0, whose anchor-set fix raised the girder's prestress by 0.8%; dynamic relaxation agreed with that earlier value to 3e-6); the −27.77 mm of case_24d is cracking left behind by the dead load being applied suddenly at t = 0
    and overshooting to −42.7 mm.

**Multithreading (`threads`, v1.0)**: the fibre-beam kernel and the frame (beam2) kernel under `backend: numba`
can distribute elements over several cores.

```yaml
solver:
  backend: numba
  threads: 0                # 1 serial (default); 0 uses all cores; n uses n
```

* Each element writes its own row, and after computation the rows are added to the nodes in element order: results are bit-for-bit identical to serial, independent of the number of threads.
* Measured on the seismic time history of one Hangzhou Bay unit with bearings (`_hzb_seismic`, 1 s computed and scaled to the full 31.18 s):

  | Threads | 1 | 4 | 8 | 16 (`threads: 0`) |
  |---|---|---|---|---|
  | Full run | 22.0 min | 11.4 min | **9.0 min** | 10.9 min |
  | Speed-up | 1.00 | 1.92 | 2.44 | 2.02 |

  Core Ultra X9 388H (16 cores, hybrid performance/efficiency cores); the results for the four thread counts are bit-for-bit identical.
* **More threads is not always better**:
  * Only about two thirds of a step can be parallelised (the fibre kernel); bearing connectors and integration remain serial,
    so the ceiling is about 3 times.
  * On a CPU with hybrid performance/efficiency cores, 16 threads have to wait for the slowest core and are actually slower than 8.
  * First measure on your own machine with `python benchmarks/bench_threads.py <case> --time 1.0 --threads 1 4 8 0`,
    take the fastest number, and do not simply write `threads: 0`.
* On a 2-core container: one internal-force evaluation 3.4 → 2.1 ms, explicit dynamics 347 → 550 steps/s.
* Frame kernel: elements are handed to threads in contiguous blocks, and each block allocates its temporary arrays only once.
  Xihoumen (851 beams, 356 cables and hangers): one internal-force evaluation 0.62 → 0.33 ms (2 cores),
  per simulated second 42.5 → 30.5 s (2-core container). The remaining ~0.3 ms of a step was vector updates, rotation advance and the energy account, then still serial.
* **The compiled whole step (1.3)**: for explicit dynamics of a frame or a frame with cables, the steps between two
  progress checks or output points are taken in one compiled call: external loads, member forces, damping, the energy ledger,
  velocities, positions, rotations, supports, with the points divided between the threads as the members are. The results are
  bit-for-bit those of the step-by-step loop (positions, velocities, rotations, energy ledger, monitors, reactions), whatever the
  number of threads. Xihoumen (2-core container, 1 s computed):

  | | 1.2.1 | step by step (this version's kernels) | compiled step, 1 thread | compiled step, 2 threads |
  |---|---|---|---|---|
  | per simulated second | 30.1 s | 27.5 s | 29.0 s | **14.9 s** |

  * **Fibre beams and bearings**: connectors and bearings are compiled (every law, the dashpots and the slip
    history), and fibre-beam models and frame models with bearings take the compiled step too, again bit-for-bit the
    step-by-step loop's answer whatever the number of threads. Hangzhou Bay seismic (fibre beams on two pot bearings,
    2-core container, 2 threads), per step:

    | | 1.3.0 | connectors compiled, step by step | compiled step |
    |---|---|---|---|
    | per step | 2.33 ms | 1.16 ms | **0.88 ms** |

    * The compiled connectors agree with the NumPy reference to round-off (about 1e-8 in force; the finite-difference
      Jacobian amplifies the last place), not to the bit. A frame model's reference is `backend: numpy`; a fibre model runs the compiled connectors with its
      fused kernel whenever numba is installed (`VFIFE_COMPILED_CONNECTORS=0`, which switched them off, went in 1.18).
      Near a relative end rotation of π neither is reliable; no bearing turns that far.
    * The fibre step is compiled once per process (about 3 s), so a fibre run of fewer than 20000 steps keeps the
      step-by-step loop (the answer is the same); `VFIFE_COMPILED_STEP=1` forces the compiled step.
  * **Prescribed motions, plastic hinges, breaking and splitting, moving loads and vehicles (1.6)**: these
    features, each with code of its own inside a step, take the compiled step too. Each one's per-step code is written
    ONCE and compiled into a function that both the step-by-step loop and the compiled step call (the hinge update, the
    carrying of parked particles and appendages, the prescribed motions, the axles' consistent nodal forces, a vehicle's
    step), so the two paths agree to the bit **by construction** rather than by comparison. A member reaching its
    capacity is an **event**: the kernel stops before that step, the loop takes it (records the failure, deactivates or
    splits the member, parks what has no body left) and the compiled step is rebuilt on what remains; a fibre
    section's fracture likewise (that step is finished first); a timed removal cuts the compiled run before its step.
    A vehicle's state advances inside the compiled run; its contact losses are recorded in step order on the run's clock.
    2-core container: Xihoumen with hangers removed and failing in turn (1.5 s) 43.6 → 29.0 s (1 thread); case_34
    (blast demolition) 12.1 → 1.8 s; case_38 (sprung mass crossing) 3.7 → 0.6 s and case_26 (moving load) 1.1 → 0.2 s
    (both timed with the output interval widened to 500 steps: writing a frame every two steps costs most of a step
    by itself, and the compiled run then has little to save).
    * Vehicles and PSD road roughness now go through the compiled functions and differ from the 1.5 NumPy version in the
      last place (summation order), not to the bit; moving loads are unchanged to the bit.
    * What still keeps the step-by-step loop is in the table below ("What takes the compiled step").
  * **The cracking shear law in the fused kernel (1.7)**: a fibre section's `shear` law ran on the NumPy
    reference path alone (`backend: numba` was refused); it is in the compiled fibre kernel now, and the compiled step
    takes it with it. The NumPy path stays the reference, the two agree to round-off (tests/test_fiber_shear.py).
    case_21b (the Vecchio-Emara frame with cracked shear) goes from 20 minutes to half a minute.
  * **The frames' staged relaxation (1.9)**: the iterations of every level of a `mode: relaxation` with
    `load_steps`, and the static solver's `static.fallback: relaxation`, run compiled too under `backend: numba`: the
    member forces, the energy ledger, the kinetic damping and its resets, the positions and rotations, the supports and
    prescribed displacements (blended in at the start of a level), and the convergence measures are one compiled
    function, which returns to Python at every progress check (the report, perhaps a checkpoint). The answer is the
    step-by-step loop's to the bit, whatever the number of threads; under numba the loop's kinetic energy and norms
    go through the same compiled functions. 2-core container: case_16 (Lee's frame, now `backend: numba`) 87 → 27 s;
    Xihoumen dead load 0.69 → 0.50 ms an iteration (1 thread).
    * Frames, bars and connectors only; a prescribed rotation, velocity or acceleration, rigid diaphragms,
      temperatures and creep, plastic hinges, soil springs and analysis steps keep the loop, and
      `solver.compiled_relax_refusal` says why.
    * **Fibre models' relaxation is not compiled this way**: its time is the fibre kernel's, and Python's overhead is a
      small part of it; to speed it up, set `solver.threads` (the fused kernel threads to the serial answer to the bit).
  * Which path is taken needs no setting; to compare, set the environment variable `VFIFE_COMPILED_STEP=0` to turn it off.
  * Frame models need `backend: numba`; a fibre model takes the compiled step whenever its fused kernel runs.
  * **What takes the compiled step and what keeps the loop**: the reasons the solver gives in
    `solver.compiled_step_refusal` and `solver.compiled_relax_refusal`, from `vfife/core/capabilities.py` (the same
    table the code refuses from; "—" where the case does not arise):

<!-- capabilities:compiled (generated by tools/capability_tables.py; do not edit) -->

| what | frames, dynamic | fibre beams, dynamic | staged relaxation |
|---|---|---|---|
| numba not installed | loop | loop | loop |
| `backend: numpy` (a fibre model takes its fused kernel all the same) | loop | compiled | loop |
| a fibre run of fewer than 20000 steps (`VFIFE_COMPILED_STEP=1` forces the compiled step) | — | loop | — |
| shell, solid, coupled and shell-line models, a standalone `truss2` or `connector2` model (no compiled step at all) | loop | — | loop |
| a fibre model's relaxation (`solver.threads` speeds its kernel up) | — | compiled | loop |
| members other than frames, bars and connectors | loop | — | loop |
| bars that creep | loop | — | loop |
| connectors whose law has no compiled path | loop | loop | loop |
| rigid diaphragms | loop | loop | loop |
| soil springs | loop | loop | loop |
| master-slave rigid links (`rigid_links`, 1.21) | loop | loop | loop |
| Rayleigh damping by region (`damping.regions`, 1.22) | loop | loop | — |
| members with end offsets (`end_offsets`, 1.21) | loop | loop | loop |
| plastic hinges | compiled | — | loop |
| a fibre section the fused kernel does not take (`solver.kernel_status` says why) | — | loop | — |
| an imposed fibre strain other than the prestrain | — | loop | — |
| `fracture` on a law other than steel02 bars | — | loop | — |
| a prescribed rotation | loop | loop | loop |
| a prescribed velocity or acceleration | compiled | compiled | loop |
| a script's own callable in a load group (a relaxation level takes the load it gives once) | loop | loop | compiled |
| temperatures, creep, shrinkage and the relaxation of steel | loop | loop | loop |
| an imposed deformation of the frame members | loop | — | loop |
| analysis steps (`steps` with procedures, 1.2) | loop | loop | loop |
| members born in a step | loop | loop | loop |
| a script's hook on the solver; rigid-surface contact (`contact.rigid`); in the relaxation any contact | loop | loop | loop |

<!-- /capabilities:compiled -->

* Cables and hangers (`truss2`) under `backend: numba` go through a compiled loop,
  rather than temporary arrays plus two `np.add.at` calls; the difference from the NumPy path is of round-off magnitude.
* Embedded reinforcement in solids under `backend: numba`: a bar set whose law has a compiled twin
  (`rebar`) is one compiled pass -- the ends gathered from the host nodes, the law, the force spread back -- instead of
  NumPy temporaries and a scatter each iteration; it agrees with the NumPy path to round-off. Other laws (`steel02`,
  `steel_cyclic`, ...) keep the NumPy path; `backend: numpy` is the reference (`VFIFE_COMPILED_BARS=0` went in 1.18). Two cores: case_12 3.15 → 1.95 ms an
  iteration, case_13 0.68 → 0.36 ms. The solid kernels themselves run on all the cores numba sees (environment variable `NUMBA_NUM_THREADS=n`);
  `solver.threads` is the frames' and fibres' setting.

**Explicit dynamics (`mode: dynamic`)**:

```yaml
solver:
  mode: dynamic
  total_time: 60.0
  damping: {type: viscous, coeff: 2.0}      # mass-proportional damping; or alpha/beta for rayleigh
  dt_safety: 0.6
```

The time step is determined automatically from the stability condition (`dt_safety` is the safety factor). **A dynamic run without `damping` is undamped** (that is the default, `type: none`);
state the damping you mean. Kinetic damping
zeroes the velocities at every kinetic-energy peak, which is a way of finding a static solution; used in a dynamic analysis it would wipe out the dynamic response, so `kinetic` is refused there. Each step's loads and ground acceleration are taken at the start of the step (second order since v1.0). A dynamic analysis step is a stretch of the time history: it takes no `load_steps` and no step-level `n_steps`, and the velocity carries across steps.

Where each damping quantity acts (other combinations are refused by name):

| | relaxation | dynamic |
|---|---|---|
| `type: kinetic` | velocities zeroed at energy peaks; may add `alpha` (softening problems, below) | refused |
| `type: viscous, coeff` | mass-proportional | mass-proportional |
| `type: rayleigh, alpha` | mass-proportional | mass-proportional |
| `beta` | refused | only under `type: rayleigh` |

`kinetic` with a small `alpha` is the combination made for softening problems: kinetic damping does nothing while the kinetic energy grows monotonically, and `alpha·M·v` always opposes motion. It selects a solution branch; use the smallest value that runs, and report it.

Rayleigh damping can be given directly as `alpha`, `beta`, or, following engineering practice, as **damping ratios at two frequencies**:

```yaml
  damping: {ratio: 0.02, frequencies: [0.3, 3.0]}       # 2% at both 0.3 Hz and 3 Hz, alpha and beta computed from this
  damping: {ratio: [0.02, 0.05], frequencies: [0.3, 3.0]}   # different damping ratios at the two points
```

The stiffness-proportional term (`beta`) reduces the stable time step — the damping ratio β·ω/2 is largest in the highest mode. This term is computed from the model's **true** highest
frequency (power iteration on the reference stiffness and mass, milliseconds), not the Gershgorin upper bound: the bound is 1.5–4 times the true value
(single beam 1.0, Lee frame 3.4, shell 4.1), so the time step of a computation with `beta` can be larger by the same factor. The time step of computations without damping or with only
`alpha` is unchanged and still taken from the bound. The log's `damping` lines give what β costs in time step and both frequencies (`damping  step x… for beta at omega_max … (power iteration; Gershgorin …)`) and, on fibre models since v1.0, the α and β a `ratio` became (`damping  rayleigh alpha=… 1/s, beta=… s`).

**Rayleigh damping by region (1.22).** `damping.regions` gives the members of some regions their own coefficients -- a steel deck and
concrete piers, an isolation layer and the structure on it (Midas's group damping, OpenSees' `region -rayleigh`):

```yaml
  damping:
    type: rayleigh
    ratio: 0.05
    frequencies: [0.5, 5.0]       # the model's: members outside the regions, concentrated masses
    regions:
      - regions: [piers]          # read as an element monitor's regions
        ratio: 0.02               # at the two frequencies above, unless the entry gives its own
      - regions: [isolators]
        beta: 0.004
```

The damping force is α₀Mv + β₀Kv with each region member's own α_e M_e + β_e K_e in place of the model's: M_e is the member's lumped mass at
its two ends (with its rigid zones' mass), K_e its reference stiffness. Where two entries overlap the later one holds; an entry's `ratio` without `frequencies` takes the model's, so a model written with
`alpha`/`beta` gives each region its own. The mass that is not a member's (concentrated masses, mass scaling, the rotary floor) keeps the
model's α. A member removed or
broken no longer damps. The explicit step is sized with the largest α and β of the model. Line models only (`beam2`, `beam2_timoshenko`,
`beam2_fiber`, `mixed_line`), `type: rayleigh` in a dynamic run or dynamic steps; a run with regions takes the step-by-step loop rather
than the compiled step (same results).

Ultimate-capacity analysis often uses a "slow dynamic pushover": the load is ramped at a very slow rate, is quasi-static before the peak, and past
the limit point the deflection develops by itself — static relaxation cannot pass a limit point and can only report the last converged load increment.

### 10.3 Iteration budget

Under dynamic relaxation every step must have an iteration budget: `solver.n_steps`, or the step's own `n_steps`, or for incremental steps
`load_steps × max_steps_per_load`. When all steps are incremental and give the latter, `solver.n_steps` can be omitted and the program adds them up.

> **Version note**: in 0.4.2 and earlier, this summation was done only in the solid (continuum) solver; for line-element and shell-element models, omitting
> `solver.n_steps` raised a `TypeError` at solver initialisation, so several shipped fibre-beam cases wrote a placeholder
> `n_steps: 100`. Later versions compute it uniformly during input validation, and all solvers read the same number.

### 10.4 Backends

| Backend | Description |
|---|---|
| `numpy` | Reference implementation, supported by all elements |
| `numba` | CPU compiled kernels, results agree with numpy to 1e-12; the first run must compile |
| `warp` | GPU (`device: cuda`, `precision: mixed` or `fp64`); solids and shells |

Which elements, materials and analysis modes support which backend is authoritatively given by `python -m vfife capabilities`. Fibre beams under
`numpy` automatically enable the compiled kernel (when the conditions are met); `solver.kernel_status` states whether it is enabled and why.

### 10.5 Convergence

Each increment must satisfy both the residual `tolerance` and the kinetic-energy ratio `kinetic_tolerance`. When an increment does not converge, the program stops and
reports the node and DOF with the largest residual. For debugging methods see `docs/convergence_debugging.md`: common causes are load increments that are
too large (a single increment close to the cracking or yield displacement), insufficient constraints, or elements or materials softening in that state.

`tolerance` bounds the out-of-balance force relative to the model's force scale, not the sum of reactions. On a 30+40+30 m bridge at 1e-5 the
reaction sum moved by up to 25 kN (0.16% of the weight) between creep stations and the 10-year deflection was 0.6% off; at 1e-7 it held to
±400 N. Use 1e-6 to 1e-7 for a dead state that later steps or a dynamic run start from.

---

### 10.6 Mass scaling (the explicit dynamic time step)

The stable time step of explicit dynamics is set by the DOF with the largest ratio of stiffness to mass, which in real bridge models is often an
unimportant node — for example the rotation of an anchor point hanging on a rigid arm with almost no mass of its own. `mass_scaling` adds just enough mass only to the DOFs whose time step is below the target,
and leaves the rest untouched:

```yaml
solver:
  mode: dynamic
  mass_scaling: {target_dt: 2.0e-5}      # optional max_added_fraction, default 0.01
```

* What is added is **real mass**: gravity weighs it and ground motion produces inertial forces on it; the run summary reports how much was added
* Exceeding `max_added_fraction` (default 1% of the model's translational mass) is rejected outright — the target time step is then so large that it changes the structure's
  dynamic characteristics, and it is no longer a "speed setting"
* The same case under dynamic relaxation adds exactly the same mass, so the dead-load state found by relaxation is consistent with the
  weight carried in the subsequent dynamic computation
* Applies to beam2, beam2_timoshenko, beam2_fiber, mixed_line; other elements are rejected
* Xihoumen model: time step 1.07×10⁻⁷ s → 2.0×10⁻⁵ s, 757.5 t added (0.176%), the first 40 frequencies change by no more than
  0.60% (`benchmarks/xihoumen_dynamic.py`)

**Rotary inertia**: a line member's nodes carry its true rotary inertia -- beam2 ρ·I_p·L/2 about its own axis and ρAL³/24 about the two transverse axes; Timoshenko and fibre beams ρ·[I_p, I_y, I_z]·L/2 (I_p = I_y + I_z) -- rotated from the member's local axes to global and taken as the diagonal (for a member along a global axis, the tensor itself). Modes, the response spectrum and dynamic runs all use it. Previously beam2's torsional inertia was ρAL³/24, so its torsional frequency grew with the element count, and Timoshenko/fibre members added their local inertia to the global rotations unrotated, so frequencies depended on member orientation. The explicit step is still computed by the old rule, so no case runs slower; where a rotation's true inertia is too small to be stable at that step it is raised to the least stable value (the stage-113 rule, weightless). One theoretical gap: with Rayleigh β damping and a skew Timoshenko or fibre member, the damped-step check still uses the old mass.

### 10.7 Starting from an existing state

Time histories such as earthquakes and vehicles usually have to start from the dead-load equilibrium state. `initial_state` makes one computation start from the final state of another
computation — positions, rotations and material state are all taken over, velocities are set to zero, and time starts from 0:

```yaml
solver:
  mode: dynamic
  initial_state: ../results/bridge_dead.h5    # relative to the case file
loads:
  - {region: all, gravity: [0, 0, -9.80665], amplitude: hold}   # dead load already present, hold it
```

The typical use is to compute the dead-load state with dynamic relaxation first (incremental, with a tightened tolerance) and then start the dynamic computation from it. The difference from `--restart`:
restart **continues** the same computation (same mode, same step counter, velocities kept), whereas
`initial_state` **starts** a new computation **from** the end point of another computation. The result file must have been computed on the same mesh,
otherwise it is rejected. Without this, starting directly from the mesh geometry, gravity makes the structure oscillate widely about the equilibrium position (Xihoumen: 1.68 m);
starting from the relaxed state, the deviation within 5 s is 0.023 mm.

A restart continues from where its checkpoint was taken: the same analysis step, the same load level, the same instant; a
ground motion, a vehicle or a staged load does not start again from its beginning, and the material state, fracture and
splits, creep history and energy ledger all carry on (before v1.0 a restart of a frame, fibre, truss or implicit static
run computed the whole analysis again from step 0). A restart from a checkpoint at the end of the analysis is refused
(`RESTART_BUDGET_EXHAUSTED`). `initial_state` carries the creep history too -- it is the concrete's memory, not the
run's. A shell run on the warp backend cannot be restarted; use the numba backend.

A restart writes to a **new file**: `vfife run case.yaml --restart part.h5 -o rest.h5`. The checkpoint is
`part.ckpt.h5` beside the result; copy the two together. Writing back into the file holding the checkpoint is refused
(`RESTART_OVERWRITES`), and so is an `initial_state` that is the output (`INITIAL_STATE_OVERWRITES`).

A checkpoint records the model it belongs to and the run that wrote it (v1.0). A restart requires the same
element family, mode, mesh, section layout, materials, failure settings, loads, constraints, masses, damping and units
(only the step budget, total time, tolerances, backend, threads and output may change, and `solver.initial_state` is
ignored); otherwise it is refused, naming what differs (`CHECKPOINT_MISMATCH`). An `initial_state` may change the mode,
the loads, constraints, masses, damping, material constants and what may break (`fracture`, `failure`), but the
elements, mesh, section layout and state shapes must agree (the nodes may add spare particles). The names of materials,
sections and regions do not count. A checkpoint beside a result that is not that run's is refused
(`CHECKPOINT_NOT_THIS_RUN`); a result may not be named `*.ckpt.h5`.

Ctrl-C or SIGTERM stops at the next progress check, so the checkpoint is exactly at a step boundary; a second signal
stops at once and keeps the previous checkpoint. The exit code is 130 after Ctrl-C and 143 after SIGTERM; in Python,
`AnalysisInterrupted.signal` gives the signal. A run whose final state is not finite keeps its last finite checkpoint.
A stopped run records every frame due before it
stopped, so the frames of `part.h5` followed by those of `rest.h5` are the frames of one whole run. A killed process
(SIGKILL) leaves a readable result up to its last frame and its last checkpoint; frames written after that checkpoint
are in both files. A continuation carries the attribute `restarted_from` and holds only the tail of the run.

Checkpoints written before v1.0 lived inside the result with no record of their model and cannot be resumed
(`RESTART_OLD_CHECKPOINT`); they still work as an `initial_state`. A checkpoint written before v1.0 has no loop
position and is refused too.

## 11 Bridge-specific features

### 11.1 Prestressing tendons in fibre sections

```yaml
sections:
  strand: {type: bar, material: strand_steel, shape: general, area: 1.4e-3}
  girder:
    type: fiber
    ...
    tendons:
      - section: strand
        profile: [[0, 1.6, 0], [35, -1.5, 0], [70, 1.6, 0]]   # [s, y, z]
        interpolation: quadratic
        jack: 1.395e9
        friction: {mu: 0.17, k: 0.0015, jack: both, anchor_set: 0.006}
        elastic_shortening: compensate
        extent: [0.0, 70.0]
      - section: strand
        profile: [[0, -1.7, 0], [70, -1.7, 0]]
        prestress: 1.2e9
      - section: strand
        profile: [[70, -2.36, 2.5], [140, -2.36, 2.5]]
        prestress: [[70, 1.15e9], [105, 1.10e9], [140, 1.15e9]]
```

* A tendon is "one steel fibre that has its own position in each element": `profile` is evaluated at each element's midpoint, (y, z) are section coordinates,
  `s` is the distance along the member from its start (a fraction of member length when `normalised: true`); `extent` limits the range of the tendon
* Three ways to give the effective stress: `prestress` as a constant; `prestress` as a `[[s, σ], ...]` table along the tendon (it must cover all elements the tendon
  passes through); `jack` + `friction` (σ = σ_con·exp(−(μθ + ks)), with anchorage-set reflection; the anchor set draws in exactly the stated length;
  `friction` reads only `mu, k, jack, anchor_set`)
* `interpolation: quadratic` is one parabola through each three consecutive points (0-1-2, 2-3-4, …): give an odd number of points. The placed curve, and every bar, must lie
  inside the section's outer outline (the rectangle of a `rect`/`box`, the circle of a `tube`, the union of the patches to within 1% of the section size); a box
  cell, a tube bore and patch `holes` count as inside
* Elastic shortening is computed automatically by relaxation; when the post-tensioning jack reading already includes it, write `elastic_shortening: compensate`
  (it uses the transformed section's 3×3 stiffness about the origin, so a T or box girder need not be written about its centroid)
* **The prestrain is built into the fibres at initialisation and cannot be ramped**: self-weight and prestress should act together from the start
* When a model has several fibre sections, the tendon `s` of each section is measured from **the start of that section's region**, and friction is solved separately per region.
  When tendon properties need to change along the beam (e.g. a corroded segment), keep one section, cut the tendon into several segments, and give each segment, via a `prestress` table,
  the friction result of the whole tendon

### 11.2 Wire rupture and fracture

* Prestressing strand: `rupture_strain` of `steel02`; when the fibre strain reaches it the fibre breaks, damage category 4
* Ordinary reinforcement: `rupture_strain` of `rebar`, or the fatigue criterion of `steel_cyclic`
* Member separation: `fracture: {threshold: 0.5, split: true}` on a fibre section makes the two sides of a fractured section come apart and the fragments fall
  under gravity (`split` requires `mode: dynamic`; there is no contact, see Chapter 16)

Fibre damage categories: 0 intact, 1 cracked, 2 crushed, 3 yielded, 4 ruptured (`fiber_damage` field).

### 11.3 Creep, shrinkage, age and relaxation

Give `creep`, `shrinkage` and `ageing` on the material (Section 6.4), give the starting age on the solver, and express
time in analysis steps as age:

```yaml
steps:
  - {name: build,   age: 28.0,      load_steps: 1}
  - {name: 3_years, to_age: 1123.0, load_steps: 12}    # logarithmically spaced time stations
solver:
  age: 28.0
```

In a fibre section each concrete fibre has its own creep history; `relaxation` of prestressing steel appears as a reduction of the prestrain over
time. See `docs/creep.md`. Fibre creep is read once per element at its middle; an even rule (`gauss2`, the default, and `lobatto4`) uses the mean
of the two stations straddling it.

Each member uses its own material's `creep`/`shrinkage`/`ageing`; members without them do not creep. The clock only runs
forward: a later step whose `age` is earlier than the age already reached, or a `to_age` earlier than the step's `age`,
is refused by name. A relaxation step without `load_steps` is one time station at `to_age`. In a time step (`to_age`) loads and prescribed displacements without an amplitude default to `hold` (full at the first time station, held); ramping over time needs an amplitude written explicitly. With `creep`, a dynamic
step may not move the concrete clock (`to_age`, or an `age` other than the current one; refused by name). Run the creep
in relaxation steps and the dynamics from its result with `solver.initial_state`; the creep reached is held as a frozen
imposed strain. `creep`/`shrinkage`/`ageing`/`relaxation` on solids, shells, coupled models, a whole-model
`truss2` (bar section on `all`) or embedded reinforcement are refused by name.

**In static steps too (1.16).** `solver.mode: static` runs the same clock on beam and mixed-line models: every converged increment of a
step with `to_age` is a time station, as every level of a staged relaxation is, and the answers are the staged relaxation's (to 1e-9 on
the test cases); fibre models too from 1.22 (to the relaxation's tolerance on case_29d and case_29g). Steps of time under
`static.arc_length` are refused by name.

**Each prestressed bar on its own clock (1.16).** A bar with `prestress` relaxes from `tensioned_at` (by default the analysis start); one
born in a later step from the age it is born at; one `jack`ed in a step from the age its step locks it off, from the strain the jack
left. A `curve` relaxation may be read `interpolation: linear` (linear in t, as a table against days is meant) instead of the default,
linear in log t.

**Design-code curves are not in the solver.** The solver reads a relaxation as a curve (or a power law) stated in the case file, and a
case gives the same answer whatever becomes of the formulas that once produced it. The formulas live in `vfife.codes`, one module per
code, which the solver never imports: `vfife.codes.jtg3362.relaxation_curve(sigma_pe, fpk, zeta, psi)` gives JTG 3362's (and JTG D62's)
relaxation of strand as a `curve`, checked against Midas' tendon loss table to four figures; the rows of its time table after two days
are not yet checked against an independent source.

### 11.4 Temperature

```yaml
loads:
  - region: girder
    temperature:
      profile: [[-2.4785, 0.0], [0.5215, 0.0], [1.5215, 14.0]]   # [section coordinate, temperature K]
      axis: y
```

* `uniform`: uniform temperature change over the whole section; `gradient`: `[dT/dy, dT/dz]` (K/m); `profile`: an arbitrary distribution
  over the section (the nonlinear temperature gradients of design codes can be written as they are)
* A scalar `gradient` runs along `axis` (default y; `axis: z` for a rect deck's depth)
* Every fibre law should state `thermal_expansion`; preflight warns (`THERMAL_EXPANSION_PARTIAL`) when a heated section mixes laws with and without it
* A step-level temperature ends with its step; with `creep` a sustained temperature relaxes
* Temperature is an imposed strain, not a force: in a statically indeterminate structure the restrained part becomes internal force, and the part that cannot remain plane
  within the section becomes self-equilibrating stress

### 11.5 Moving loads

```yaml
loads:
  - region: deck
    moving_load:
      axles: [[0.0, 3.0e+5], [-4.2, 3.0e+5], [-5.6, 3.0e+5]]    # [position relative to reference point, axle load]
      start: -10.0
      speed: 20.0            # dynamic analysis: s = start + speed·t
      # stop: 220.0          # or: under incremental relaxation, load_steps positions between start→stop = influence line
      direction: [0, 0, -1]
```

The region must be **an unbranched chain of members**. Axle positions are measured along **the direction of travel**: rear axles are written as negative values, and when travelling backwards with a negative `speed` (or `stop < start`) the same axle group still has the rear axles behind. A queue of vehicles can be written as one load group, a string of axles (each vehicle's axles plus the vehicle spacing), and all axles are computed together, which is much faster than one load group per vehicle. It is a moving **force**: the vehicle has no mass or suspension and does not follow the deck deformation,
and the dynamic amplification factor cannot be computed with it. See `docs/moving_load.md`.

A moving_load with `speed` is dynamic only; in static or relaxation runs sweep it with `stop`. Moving loads cannot be combined with `static.arc_length`. `moving_load`/`vehicle` act on beam members only; shell models are refused. With an axle or a line load between nodes, the written member end actions (`*_i/_j`) and `bending_y` etc. include the fixed-end moments; on `beam2_fiber` the per-member values are section resultants, so refine the mesh under point loads between nodes. An axle or a vehicle on a member with end releases is condensed through them, as a `line_load` is.

**Vehicles with mass**:

```yaml
loads:
  - region: deck
    vehicle:
      name: truck
      speed: 20.0
      start: -30.0                      # starting chainage of the reference point; negative values are on the approach
      body: {mass: 2.4e+4, pitch_inertia: 1.5e+5, cg: -2.0}
      axles:
        - {offset: 0.0,  unsprung_mass: 700, suspension: {stiffness: 1.0e+6, damping: 2.0e+4},
           tyre: {stiffness: 3.5e+6, damping: 2.0e+3}}
        - {offset: -4.5, unsprung_mass: 1100, suspension: {stiffness: 2.0e+6, damping: 3.0e+4},
           tyre: {stiffness: 7.0e+6, damping: 4.0e+3}}
      roughness: {psd: {g0: 64.0e-6, n0: 0.1, w: 2, n_min: 0.011, n_max: 2.83, seed: 1}}
output:
  monitors:
    truck_acc: {kind: vehicle, of: truck, field: acceleration}
    front_P:   {kind: vehicle, of: truck, field: contact_force, axle: 0}
```

* The vehicle body is a rigid body: vertical translation, plus pitch when there is more than one axle (in which case `pitch_inertia` must be given).
* Each axle: suspension (spring + damper) → unsprung mass → tyre (spring + damper) → road surface.
  * Tyres act in compression only: when a wheel leaves the road (the tyre is no longer compressed) the contact force is zero and its damper no longer acts; the number of airborne steps and of rigid-axle tension steps, with the first time, go to the run report (`vehicle_contact`; `vehicle_contact_json` in the result's `diagnostics`) and the log.
  * An axle given no tyre and no unsprung mass sticks rigidly to the road surface, which is the "moving sprung mass" of the literature.
* Road surface = deck deflection at the contact point (read back with the same Hermite shape functions as `moving_load`) + roughness.
  * Roughness can be given as a `profile` file (chainage, elevation).
  * It can also be given as a displacement power spectral density G_d(n) = g0 (n/n0)^−w, generated by harmonic superposition from a seed.
  * The library does **not store** the code tables of roughness classes: you supply g0.
  * A `psd` roughness takes only g0, n0, w, n_min, n_max, count and seed (a misspelt key is refused); `scale` multiplies a profile or a psd; a psd road is evaluated as the exact harmonic sum.
* The vehicle and the structure are advanced with the same central difference and the same time step; the vehicle starts from static equilibrium at its starting chainage.
  If the vehicle's own stiffest mode (the wheel on the tyre) requires a smaller time step, this is reported directly.
* Dynamic mode only; available for frames and fibre beams.
* A monitor with `kind: vehicle` reads: `heave`, `pitch`, `acceleration` (vertical body acceleration),
  `contact_force`, `wheel`, `station`; `axle` selects which axle.
* An unnamed vehicle is called `vehicle`, `vehicle_2`, … within each step (the first unnamed vehicle of every step is `vehicle`); two vehicles in one step may not share a name. A vehicle monitor records NaN in a step that does not contain that vehicle. A vehicle takes no `amplitude`, and `direction` may not be the zero vector. A monitor's `of` must name a vehicle and `axle` must be below its number of axles, or the case is refused when read.
* Limitations: planar model, no roll, one wheel track per axle; the contact point is taken at the chainage on the undeformed path.

Case case_38: the sprung mass of Yang & Yau (1997) crossing a 25 m simply supported beam at 100 km/h, compared with an independent modal solution
(20 sine modes + high-accuracy scipy integration). The peak midspan deflection differs by 0.03% and the peak body acceleration by 0.2%.

### 11.6 Earthquakes

```yaml
loads:
  - ground_motion: {path: ../benchmarks/assets/elcentro_1940_ns_g.dat, scale: 9.80665, direction: [1, 0, 0]}
```

A two-column record (time s, acceleration); `scale` converts it to m/s², `direction` is a unit vector, and the inertial force −M·a_g(t) is applied in relative coordinates. Only these three keys are taken; not `amplitude` (the record is the time history) and not `region` (it is the whole model's base; a support moving on its own record is the multi-support excitation below). In a multi-step dynamic run a model-level `ground_motion` replays from the record's start in every step (preflight warns `GROUND_MOTION_REPEATS`); put it in the step it belongs to.
This is **uniform excitation**: all supports move at the same instant with the same motion. For the response spectrum see 11.8. When reading reactions from a damped explicit analysis,
filter them or add a little stiffness-proportional damping (high-frequency oscillation).

**Multi-support excitation**. On a kilometre-scale bridge two piers may be several hundred metres apart, and the seismic wave takes a few tenths of a second to travel between them: the motions of the two piers
have a phase difference, and members are pulled by the differential motion of their ends, a part that uniform input cannot capture. Add `record` to a constraint and that region moves
according to its own record (absolute coordinates):

```yaml
constraints:
  - {region: piers, dofs: [1, 1, 1, 1, 1, 1]}     # fixed as usual ...
  - region: piers                                 # ... of which x follows the record
    dofs: [1, 0, 0, 0, 0, 0]
    record:
      path: ground_x.dat                          # two columns: time s, value
      quantity: displacement                      # or velocity / acceleration
      scale: 1.0
      delay: 0.0                                  # this support's own time delay, s
      wave: {speed: 1000.0, direction: [1, 0, 0], origin: [0, 0, 0]}
```

* Records drive only translations; rotations are fixed by the ordinary constraint beside it, and the record overrides that constraint's zero value in the flagged components.
* **Displacement** records (baseline-corrected) are recommended: a displacement record also applies its velocity. It is read as a C1 cubic (a monotone Hermite through the samples, Fritsch–Carlson/PCHIP slopes, so it never leaves the range of the two samples it joins -- the earlier three-point slopes overshot a coarse step by 7%), so the support velocity is continuous and is the exact derivative of the displacement (read linearly the velocity jumped at every sample, and the pier base shear came out 3.2× that of the same motion given as an acceleration record, 2266 against 708 kN; now 698 against 700 kN). Before the record starts the support is at rest; after it ends
  the displacement stays at its last value. Velocity and acceleration records are integrated exactly from the record's piecewise-linear interpolation (from v1.0; previously first order), independent of the step like a displacement record; after an acceleration record ends the support coasts at its last velocity, and baseline drift is the user's responsibility.
* `wave` adds to each node in the region a further delay of (x − origin)·direction / speed; for different records at different piers, write several constraints.
* Dynamic mode only; available for frames, fibre beams and shells; solids and stand-alone trusses are rejected.
* Displacements are **absolute**. Rayleigh α and `type: viscous` (both mass-proportional, `·M·v`) act on absolute velocities and damp the
  rigid-body motion imposed by the ground. Use β, or be aware of the extra damping. A monitor's `relative_to` can read displacement relative to a given support.
* The energy account records the work done by the ground through support reactions as external work, and the kinetic energy of the support particles themselves is attributed to the ground.
* A support `record` and a global `ground_motion` along the same direction are refused by preflight (`GROUND_MOTION_AND_RECORD`): both would apply and the response would double. Use `record` on every support (multiple-support excitation) or one `ground_motion` (uniform), not both.

Case case_36: a 100 t deck mass between two piers 400 m apart, wave speed 1000 m/s, the east pier 0.4 s later than the west pier
(nearly half a period). The peak deck displacement is 0.1019 m, a little over one third of the uniform-input value (0.285 m), yet the member force
is about the same (1.50 MN versus 1.46 MN): the two piers move in antiphase, holding the deck in place and pulling the member apart. The difference from the closed-form solution is 0.02%.

Soil springs on piles (p–y curves) are in Section 11.13; radiation damping is not available, and a record is applied where it is written.

### 11.7 Bearings

Bearings are placed with a `connector` section between a girder node and a pier-top node (in the mesh, a vertical line element of about 0.5 m):
fixed pot bearings get a large longitudinal stiffness, sliding pot bearings a small longitudinal stiffness, and rotational stiffness follows the actual bearing (pot bearings release
rotation about the transverse axis). Fibre beams can share one model directly with connectors.

### 11.8 Modes

```bash
python -m vfife modal cases/case_23_suspension_bridge.yaml --result results/case_23_suspension_bridge.h5
python -m vfife modal cases/_xihoumen.yaml --modes 40 --paraview
```

Natural frequencies and mode shapes are computed in the given loaded state using the finite-difference tangent stiffness and the physical mass; `--paraview` writes each mode shape
as a ParaView time series (requires meshio). With `--result` the supports are those of the analysis step the frame belongs to (from v1.0; previously always the first step's). A mode with no translation (a column's torsion) is scaled to a unit largest rotation, marked `rotation only` in the table and listed in the modes file's `rotation_only` dataset (previously it was divided by numerical noise and rotations reached 1e13). Models with `diaphragms` are solved with rigid floors.

* With `--result`, when the frame is the result's last (the one its checkpoint holds), the whole state is restored as `solver.initial_state` does — fibre and connector-law histories, creep history, relaxed prestrain — and the summary says `full state from its checkpoint`. Other frames are positions only, with a note. `probe --result` restores the same way.
* After a creep run, modes and the spectrum hold the creep reached at the instantaneous modulus, as a dynamic run from that state does. Before, the last station's E/(1+φ) was used: periods 20–35% long, depending on the station count.
* A connector law on its kink (friction sliding, bilinear at yield, gap at its stop) is linearised at the mean of its two branches. `modal` and the spectrum warn by name, and the spectrum records `connector_kinks`; give such a bearing an effective linear stiffness for a spectrum.

**Response spectrum**: add a `response_spectrum` block to the case, and `vfife run` performs a modal analysis in the state at the end of the computation
(the relaxed dead-load equilibrium; with no loads, the drawing geometry), finds the peak of each mode from the spectrum and combines them:

```yaml
response_spectrum:
  spectra:
    - {direction: [1, 0, 0], path: spectrum_x.txt, scale: 9.80665}   # period s and spectral acceleration (here in g) in the first two columns; further columns are not read
    - {direction: [0, 1, 0], table: [[0.0, 0.9], [0.1, 2.25], [0.5, 2.25], [5.0, 0.3]]}
  combination: cqc          # or srss
  damping: 0.05             # modal damping ratio for CQC (the spectrum itself should be given for the same damping ratio)
  mass_fraction: 0.9        # when modes is not given, take modes until the effective mass in each excited direction reaches 90%
```

* Peak of each mode U_i = φ_i Γ_i S_a(T_i)/ω_i²; displacements, member end forces and support reactions are combined component by component by CQC (Der Kiureghian,
  one damping ratio) or SRSS, and the directions are then combined by SRSS
* Member end forces and reactions come from the model itself: the internal-force difference from moving ±h along the mode shape, using the same tangent as for the mode shapes (cable tension stiffening
  and girder axial force are both included). **Base resultants are summed per mode first and then combined**
* The spectrum table is supplied by the user; the library stores no code spectra. A period outside the spectrum table is rejected by default, reporting which modes; write
  `beyond_table: hold` to take the end value
* No residual-mass correction: when the modes taken do not reach `mass_fraction`, the fraction actually reached is reported as it is
* Equal or closely spaced frequencies (bending in the two directions of a square column, symmetric structures) require CQC: SRSS is wrong for degenerate modes
* Works with `mode: relaxation` or `mode: static` (linearisation must be done at an equilibrium state)
* The spectrum uses the supports of the run's last analysis step
* `modes` and a non-default `mass_fraction` cannot both be given (previously `modes` silently overrode it)
* A spectrum `path` is read and checked before the run: two numbers per row (spaces, tabs or commas; `#` comments), periods from 0 or more and increasing, no negative values; otherwise `SPECTRUM_FILE`
* A truss model writes its bar axial forces as `axial_force`
* A contract reading the vector quantities `displacement`, `rotation`, `reaction` or `base_force` with a `final`/`first`/`delta` reduction must name a `component`; `nearest` on `reaction` finds the nearest node
* Results are written to the `response_spectrum` group of the result file: `period`, `frequency`, `effective_mass` (ratio),
  `modal_mass` (kg), `spectral_acceleration`, `displacement`, `rotation`, `reaction`,
  `base_force`, `end_actions` and `axial_force`/`bending_y`/… (the larger of the two ends). Verification contracts read them with
  `source: spectrum`. Case case_35

**Linear buckling** (1.1): `vfife buckling <case>` finds the buckling load factors λ of a step's loads and the buckled shapes, the
smallest positive roots of (K_b + λ K_G) φ = 0.

```bash
python -m vfife buckling cases/pier.yaml                       # the first step's loads, from the reference configuration
python -m vfife buckling cases/pier.yaml --step 2 --modes 4    # the second step's loads
python -m vfife buckling cases/pier.yaml --result results/pier_dead.h5 --paraview   # loads added to a state a run reached
```

* **K_b**, the stiffness at the base state, is the same finite-difference tangent `modal` uses: connectors, bearings, fibre
  sections and the stress stiffness the base state already carries are all in it.
* **K_G** is the stress stiffness of the step's loads, assembled from the members' **axial forces and bending moments** in
  the linear increment u = K_b⁻¹F (since 1.5; 1.1 had the axial forces only). A bar or cable has N/L (I − e eᵀ). A beam has
  two parts:
  * the co-rotational part: the end forces a member carries, turned with its frame (the chord's string stiffness N/L, the
    end moments and shears), in the solver's own kinematics, so a moment carried round a joint between members at an angle
    (one member's bending becoming the next one's torsion) is turned as the nonlinear run turns it;
  * the second-order terms inside the member: the end rotations relative to the chord (the axial force's term), the
    twist's Wagner term N·Ip/A·θ′², and the moments' coupling of twist with bending out of their plane.

  It finds the flexural buckling of columns, piers, towers, frames and arches (in and out of plane), the **torsional
  buckling** of columns and the **lateral-torsional buckling** of beams and girders. Shells and solids are refused
  (`BUCKLING_FAMILY`): their stability is found with a nonlinear static run (arc length) to the limit point.
* **The reach of lateral-torsional buckling**:
  * **No warping**: a member has Saint-Venant torsion GJ only. An open thin-walled section (an I girder) therefore buckles
    at a critical moment that is too low, by the factor √(1 + π²EI_w/(GJL²)). That is on the safe side, and much on it for
    a short, deep I beam. Box, solid and narrow rectangular sections are not affected.
  * Sections are taken as doubly symmetric: the shear centre is the centroid, and a monosymmetric section's Wagner
    coefficient (a T girder) is not in it.
  * Loads act at the centroid: a load applied above it (a deck on top of the girder) is not in it.
  * The torque enters through the co-rotational part only. A member under torque is turned as the solver turns it (its
    moments are spatial, conjugate to the spins). A shaft under torque alone buckles where the nonlinear run does, not at
    a textbook Greenhill load, which depends on how the torque is applied.
  * An applied moment is treated as a member end, with no load stiffness of its own. At a fork support, where the twist
    is held, every kind of moment gives the same answer.
  * **The bending before buckling is included.** Each plane's moments are scaled by Kirchhoff's factor
    √((1 − I_z/I_y)(1 − GJ/EI_y)) (bent about y; 0 when I_y ≤ I_z or GJ ≥ EI_y, each bracket clamped at 0). It is exact for a uniform moment on forks and close
    for other moment diagrams (Pi and Trahair; a cantilever lands 2–4% off). Where the factor is 0, that plane's moments
    and shears are dropped from K_G altogether, co-rotational part included; the axial force and the torque stay.
    * A member whose second moments are close (a round pier, a square or box section), or one bent about its minor axis,
      therefore does not report a lateral-torsional mode that does not exist.
    * A narrow rectangle's critical moment rises by 0.3%, and that of a section with I_z = I_y/4 by 17%. The nonlinear run
      buckles within 2% of it.
    * The textbook value that design codes use is the reported one times this factor.
* **λ multiplies every load of the step**, self-weight included, at the amplitude's value at the step's end.
* **By default the base is the reference configuration**, which must be at rest: a model with internal forces there
  (prestress, creep, temperature) is refused (`BUCKLING_BASE_NOT_AT_REST`); run it to the state they put it in and ask from
  there with `--result`. A slack cable has no stiffness at the reference configuration and needs `--result` too.
* **`--result`**: the base is that frame's state (the last frame, which the checkpoint holds, with every material, creep and
  bearing history), and λ is the factor on the step's loads **added** to it -- a pier run to equilibrium under its dead load
  and asked there gives how many more dead loads it takes.
* A mechanism (a stiffness singular on the free dofs) is refused (`BUCKLING_BASE_NOT_STABLE`): since 1.5 only when the
  smallest stiffness is both under 1e-10 of the largest and under the finite-difference tangent's own asymmetry (a narrow,
  deep section on a very fine mesh had its weakest bending refused as a mechanism).
* A fibre law without tension (concrete01, a concrete02 with ft = 0) has every fibre on its zero-strain kink at the
  reference configuration: K_b averages the compression stiffness and zero, and a column under axial load alone comes out
  at about half its Euler load. The note says so; ask from the loaded state with `--result`, where the fibres are in
  compression. Negative factors (buckling
  under the reversed load) are given in the note when they are not far beyond the first positive one. Where connectors have
  laws or curves (a gap at its stop, a bearing sliding), K_b is taken on the side the load pushes, a hair along the
  increment. A model without such connectors is not shifted (1.5): the shift also turns every member's stiffness with the
  increment's rotations, and a narrow girder's lateral-torsional buckling is sensitive to that.
* Written to `results/<case>_buckling.h5` (`load_factor`, `shape` (mode, node, dof), `axial_force`, `reference_load`);
  `--paraview` also writes the shapes as animations (as `modal --paraview`). Dense, like `modal`: for up to a few thousand dofs.
* Lateral-torsional buckling against closed forms (no warping, loads at the centroid, 32 elements, each closed form
  times Kirchhoff's factor):
  * a simply supported beam on forks under a uniform moment, M_cr = π/L·√(EI_z GJ): 0.04%, with its second mode at twice
    that;
  * a central point load, a uniform load over the span and a cantilever end load: γ = 16.94, 28.31 and 4.013, each within
    0.1%;
  * a column whose torsional load GJ·A/Ip is below its Euler load;
  * a beam-column under moment and thrust, M² = (Ip/A)(P_E − P)(P_T − P);
  * the same answers from a mixed line model, a beam along y and a fibre section.

  Argyris' right-angle frame (pushed in its plane, buckling out of it) has no closed form. It is checked against the
  solver's own nonlinear run: the co-rotational tangent at the static equilibrium turns singular within 1% of the linear
  factor, pushed and pulled. A cantilever with I_z = I_y/4 is within 5%. A square section, and a member bent about its
  minor axis, have no lateral-torsional mode.
* Flexural buckling against closed forms: a cantilever's Euler load (3e-5 high on 4 elements, unchanged with a lateral load of 30% of
  the axial one beside it), a pinned column, the sway of a fixed-base portal (stability functions, 0.01%), a cantilever under
  its own weight (Greenhill 7.837, 0.2% on 16 elements), a Timoshenko column (Engesser, < 0.1%; 3% when very shear-soft), a
  fibre column, a strut on a spring, a leaning column held by a rigid floor.

### 11.9 Cable form finding### 11.9 Cable form finding

```bash
python -m vfife formfind cases/case_23_suspension_bridge.yaml --groups main_cable,hangers -o results/found.yaml
```

Finds the unstressed lengths of the cables so that the structure rests at the mesh geometry under dead load; outputs a new
case with `unstressed_lengths`.

Form finding needs `solver.mode` `relaxation` or `static`; every inner relaxation must converge, and one that does not stops the search with
NOT CONVERGED (exit code 3) and writes nothing. If the outer iterations run out before the tolerance is reached, the written file's first header
line is `# NOT CONVERGED` and its lengths are only the last estimate. `--step` is the dead-load step's index (from 0). Written with `-o` into another
folder, the relative paths of the mesh, records and `initial_state` are rewritten relative to the new file; `output.path` is left as it is (it is
relative to the working directory). The folder of `-o` is created if it does not exist. In `static` mode the inner solves are Newton solutions and
the header says "static solutions".

---

### 11.10 Elastic–brittle fracture and timed removal of line members (cables, hangers, steel box girders)

In dynamic computations line-member models can let members break: a member is linear elastic until it reaches its capacity, and is then removed (no longer carries force,
its mass stays at the nodes). This is written on the assignment — per member group (mixed line-element models), or, when one section is assigned to `all`, for
all members:

```yaml
assign:
  - {region: hang_pure, section: hanger, failure: {tensile_strength: 1.77e9}}
  - {region: rail_box, section: rail_box, orientation: [0, 0, 1],
     failure: {yield_strength: 3.7e8, extreme_fibre: [2.5, 6.5], shape_factor: 1.12}}
removals:
  - {region: cut_0, time: 3.0}        # the members in this region are cut 3 s after the run starts (accumulated across steps)
```

* **Cables (trusses)**: break when the tension reaches `tensile_strength × area`
* **Beams**: break when at either end |N|/(A f_y) + |M_y|/M_y,cap + |M_z|/M_z,cap ≥ 1,
  with M_cap = shape_factor · f_y · I / c; `extreme_fibre` is the distance to the outermost fibre in the section's local y and z directions.
  Shear force and torque are not checked
* The criterion is **capacity**, not strain: when the strain of an elastic member reaches the true fracture strain of steel (about 15%), the stress is 30,000 MPa,
  which would never be reached. Capacity needs only the area, moment of inertia and section dimensions
* There is no plastic redistribution or energy dissipation, so collapse is earlier and more abrupt than in a real structure — conservative. Moreover, reaching capacity releases the
  whole moment within one step, and the unloading shock drives adjacent sections over their limit too: in progressive collapse members break in a "zipper" fashion, and the fragment size equals the element length
* **Plastic hinges** (mixed_line): add `rotation_capacity: 0.03` (rad) to the beam's `failure`, and both ends of every
  member become plastic hinges — end moments are capped by |N|/N_p + |M_y|/M_y,p + |M_z|/M_z,p ≤ 1 (M_p is the M_cap above),
  the excess becomes plastic rotation and dissipates energy, and unloading is elastic; an end breaks only when its accumulated plastic rotation reaches `rotation_capacity`,
  while an axial force reaching N_p still breaks immediately. Ideal elastic–plastic, no hardening; the moment exceeds the yield surface by at most one time step. Plastic rotation accumulates only in each step's own force evaluation, independent of the output frequency.
  The output field `hinge_rotation` is the accumulated plastic rotation of the worse end (rad), and `hinge_usage` is its ratio to
  the rotation capacity (0 elastic, 1 broken); before v1.0 `hinge_rotation` held the ratio. Progressive collapse of steel members
  should use it
* **After breaking there are two approaches** (`failure.split` / `removals[].split`). **Particle splitting** (the default for beams and
  mixed_line): a beam leaves the particle at the end that broke and moves onto the spare particle reserved for that end from the start (two per breakable member,
  `core/topology`); it stays welded at the other end and may break there later; a bar (cable), a timed removal and a member that is deactivated rather than split part at both ends. Its share of mass goes with it, and the member continues as a free segment with stiffness and self-weight — a box-girder segment broken at both ends
  is a fragment that flies, falls and comes to rest on the seabed, the remaining structure loses exactly its weight, and is no longer dragged by fragments lying on
  the seabed. Particles at supports (constrained) are always duplicated, so fragments do not stay attached to the support they broke from (only particles constrained in the current step count as supports; one released in an earlier step is free); particles touched by splitting
  are topped up, by the rule of v0.4, with the mass needed for stability at the current time step: one translational mass per particle, the same on x, y and z (the largest any direction needs), with the velocity scaled by old/new mass (momentum conserved, the kinetic energy lost booked as dissipated). The added mass is **weightless** (like hydrodynamic added mass), so a fragment that received it falls somewhat slower than g (by m/(m+Δm)); the log states the kg actually added.
  **Massless members** (crossbeams, rigid arms with ρ≈0 — their mass smeared into other members) are only removed, not split, when they break; a cluster of massless members
  connected at neither end to a massive member is removed; one hanging on only one massive particle, when that particle is still connected to a support, becomes an **appendage**
  (an outrigger crossbeam after its hanger is cut) and follows the rigid chord motion of one massive member there (not the roll of the node about its own axis),
  while one hanging on a fragment is removed; one hanging on two or more carries force as usual — these are connectors, not bodies. Particles left behind when all massless members
  are broken (not at a support, no nodal mass) stay where they are rather than sliding on forever with the velocity they had at the moment of breaking. **Remove only**
  (`split: false`, the default for stand-alone trusses): no longer transmits force, mass stays at the nodes — sufficient for a cut hanger rope.
  The connectivity after splitting is written frame by frame in the `connectivity` field and `vfife view` draws it; `diagnostics/member_splits`
  has one row per event [time, element, original particle, new particle]. Particle splitting requires `mode: dynamic`; fibre beams use
  `fracture.split`; fragment particles belong to no region: gravity reaches them through the region their parent particle is in, and a nodal force written on a region is not spread onto them (before v1.0 a fibre fragment only joined `all`)
* Without `contact.members` (11.12, since 1.11) a falling member passes through the members below it; with a rigid surface it comes to rest on the surface
* Which models: mixed_line (per member group), single-section beam2 / beam2_timoshenko / truss2 (`failure`
  written on the assignment to `all`, applying to all members); `removals` is additionally used for fibre-beam models. **Fibre beams do not accept
  `failure`** — their section knows the strength of each fibre and breaks in the material law (`fracture`, see 11.2), or they are
  cut at a given time with `removals`; `fracture` events of fibre beams are also written into the same `member_failures` table below.
  Checked only in dynamic computations; dynamic relaxation rejects it (states within relaxation iterations are not equilibrium states)
* `removals[].time` is measured from the start of the run and accumulates across dynamic steps (before v1.0 the fibre beam used the time within the step, and a removal due in step 2 never happened); a removal after the run ends (number of steps × `solver.total_time`) gets the preflight warning `REMOVAL_AFTER_END`.
* When only removed, a member no longer carries force: mixed_line takes it out of the batch, beam2 sets its local stiffness to zero, truss2 sets its area
  to zero (stress is still reported from strain), and fibre beams use the same switch as `fracture`. When Rayleigh stiffness-proportional damping (β) is used,
  broken members are also subtracted from the reference stiffness matrix (split members are moved to their new particles) — otherwise a falling fragment would be
  tethered to the deck by a damper
* Results: element fields `member_failed` and `capacity_ratio` (demand/capacity), and `diagnostics/member_failures`
  with one row per fracture [time, element number, capacity ratio at fracture, cause 0=capacity 1=timed removal 2=what the massless member connected is gone (on splitting)]; broken members are recorded in checkpoints

### 11.11 Nonlinear bearings: sliding, restrainers and isolation hysteresis

Each component of a connector (`type: connector`) can use, instead of a linear stiffness, a **path-dependent law** `laws`:

```yaml
sections:
  sliding_bearing:                      # the chord is vertical: ux is vertical, orientation [1,0,0] makes uy longitudinal
    type: connector
    stiffness: [1.0e+10, 0, 0, 0, 0, 0]
    laws:
      uy: {type: friction, mu: 0.05, normal: ux, stiffness: 1.0e+8}   # Coulomb friction: sticks up to mu·N, then slides
  expansion_joint:                      # the chord is longitudinal: ux is the joint opening/closing
    type: connector
    laws:
      ux: {type: gap, lo: -0.006, stiffness: 2.0e+9}   # free within 6 mm, restrainer stiffness once closed (pounding)
  lead_rubber:
    type: connector
    stiffness: [2.0e+9, 0, 0, 0, 0, 0]
    laws:
      uy: {type: bilinear, k1: 2.0e+7, k2: 2.0e+6, fy: 5.0e+4}      # bilinear hysteresis (kinematic hardening)
      uz: {type: bilinear, k1: 2.0e+7, k2: 2.0e+6, fy: 5.0e+4}
  viscous_damper:                       # the chord is longitudinal: ux is the damper's stroke (1.10)
    type: connector
    laws:
      ux: {type: damper, c: 2.5e+6, exponent: 0.3, linear_below: 0.005}   # F = c·sign(v)·|v|^0.3
  friction_pendulum:                    # vertical chord, as the sliding bearing (1.10)
    type: connector
    stiffness: [1.0e+10, 0, 0, 0, 0, 0]
    laws:
      uy: {type: pendulum, mu: 0.05, radius: 3.0, normal: ux, stiffness: 1.0e+8}
      uz: {type: pendulum, mu: 0.05, radius: 3.0, normal: ux, stiffness: 1.0e+8}
```

* `damper` (1.10): a fluid viscous damper, F = c·sign(v)·|v|^α on the component's deformation rate, 0 < α ≤ 1
  (α = 1 is the linear dashpot `connector_damping` is). `c` is in force per velocity^α in the case's units. It acts in
  **dynamic** steps only (a static or relaxation step sees only its optional parallel spring `stiffness`).
  **Below `linear_below` the law is linear**, F = c·v₀^(α−1)·v: the power law's slope is infinite at v = 0 and an explicit
  step would chatter there. `linear_below` is required when α < 1; take about 1% of the design velocity (the energy of a
  cycle it leaves out is about 0.1·(v₀/v_max)^(2+α) of the cycle's: 3e-6 at 1%, 3e-3 at 20%). A smaller value is closer to the power law and holds the stable step
  down (dt < 2m / (c·v₀^(α−1)), which the solver applies). Checked against a fine Runge–Kutta integration of a mass on
  a spring and a damper (tests/test_seismic_devices.py)
* `pendulum` (1.10): a friction pendulum bearing — Coulomb friction μN as `friction`, plus the restoring force of the
  spherical surface, (N/`radius`)·d. Both follow the compression N the bearing carries at that instant (`normal: ux`, or a
  number): a `bilinear` law fitted to the dead load cannot follow a vertical motion or an overturning couple, this
  does. The period of the sliding structure is 2π√(R/g), whatever the mass. With N at zero (uplift) the bearing carries
  nothing horizontally (the ux spring that measures N is linear and still pulls: give ux a `gap` law only by taking N as a number). On `uy` and `uz` together it is one law in the plane. A double or triple pendulum is written with its
  effective radius; a velocity-dependent friction coefficient is not modelled
* `friction`: `normal: ux` takes the normal force from the **current** compression in the same connector's local ux linear spring (only `ux`: every other component's sign flips with the order of the Line2's nodes, and a global connector cannot name a component) —
  when the girder lifts off (zero compression) the sliding force is also zero; a constant can also be given. `stiffness` is the stick stiffness: the larger it is, the closer to
  ideal Coulomb friction, and the smaller the stable time step
* `gap`: `lo`/`hi` give the two ends of the free travel; omitting one side means no restrainer on that side; the local ux is the elongation, so `{lo: 0}` is a compression-only support and `{hi: 0}` a tension-only tie (the manual had it the other way round before v1.0). `connector_damping` on a gap component acts only while the gap is closed
* `material` (1.21): a one-dimensional law of the material table (`parallel`, `series` and `gap` included, Section 6.2), the force
  F = `area`·σ(d/`length`). `area` and `length` are 1 when left out, and the table is then the component's force–deformation
  relation itself (as OpenSees' zeroLength reads a material); with a core's area and its yielding length a steel's σ–ε is used
  as it is, e.g. a buckling-restrained brace `ux: {type: material, material: core, area: 4.0e-3, length: 3.2}`. The reference
  stiffness is `area`·E/`length`. The component's `connector_slip_*` field reports the material's strain d/`length`. The law's
  state is kept in the checkpoint and starts afresh when an analysis step builds the connector. `uy` and `uz` stay two laws
  (a square yield surface) -- except a `bouc_wen` material given to both as the same law (1.22): that is one law in the plane,
  the yield surface a circle (Park, Wen & Ang 1986), a lead-rubber or high-damping rubber bearing. The material's own rupture does not break the connector. The compiled connector kernel does not
  take this law: in a model with it the connectors are evaluated in Python and the compiled step keeps the step-by-step loop
  ("connectors whose law has no compiled path" in the capability table)
* the SAME law (friction, bilinear or pendulum) on `uy` and `uz` is one law in the plane: it yields or slides along the resultant, on a circular surface (before v1.0 the two were independent and the surface was square: 1.414 mu N at 45 degrees)
* `bilinear`: initial stiffness `k1`, yield force `fy`, post-yield stiffness `k2`, kinematic hardening — the backbone of lead–rubber bearings, friction pendulums and
  yielding energy-dissipation devices; one hysteresis loop dissipates 4(f_y − k₂u_y)(u_max − u_y)
* Each component has either one law or one stiffness, not both; the law's history (slip) goes into checkpoints, output fields `connector_slip_*`
* Under **static** analysis every evaluation within a load increment starts from the slip of the previous equilibrium state (the same treatment as fibre materials), and is committed only
  at equilibrium; under **dynamic** analysis it is committed every step. So static friction and hysteresis follow the loading path increment by increment, and there must be enough increments to resolve the path
* Most "pounding" in bridges is one of these three: girder end hitting the abutment (gap), bearing sliding (friction), isolation (bilinear);
  they occur between two known points and need no contact search. Case case_32
* **Build bearing connectors short.** When the two ends rotate differently, the connector's translational deformation is measured in the average frame of the two ends: a
  1 m long vertical bearing reads a girder-end rotation θ as a longitudinal slide of θ/2 × 1 m (the rotation under dead load alone gives several millimetres).
  Placing the anchor node 10–50 mm below the girder end is enough

### 11.12 Rigid-surface contact: ground, seabed, pile caps

```yaml
contact:
  rigid:
    - {name: seabed, plane: {point: [0, 0, -40], normal: [0, 0, 1]}, damping: 0.5}
    - {plane: {point: [0, 0, 0], normal: [0, 0, 1]}, region: deck_nodes, friction: 0.3}
```

A rigid, fixed surface: nodes in `region` (default all) that touch it receive a penalty force, a normal spring times the penetration;
under dynamics Coulomb friction (`friction`, on the tangential velocity) and contact damping (`damping`, so that impacts are no longer perfectly elastic and fragments
come to rest) can be added. The normal points to the permitted side. No pairing, no master/slave, no search — a few dot products per node per surface.

* `stiffness: auto` (default) takes k = m(2/dt)²/4 from each node's mass and the time step: the contact spring's own ω·dt = 1,
  which together with the structure's ω·dt ≤ 1.2 is still within the central-difference stability limit of 2, so **the time step is not reduced by contact**;
  a node at rest penetrates g·dt², of the order of micrometres. A numeric `stiffness` is capped at that same value (larger is unstable; before v1.0 it was used as given). A free-flying fragment has no stiffness of its own to borrow a scale from, and `auto` is exactly for that
* An impact lasts only a few steps, and the coefficient of restitution is not a controllable quantity; to look at rebound height, give a softer `stiffness`
* Friction is a velocity law and applies only in dynamic computations; giving `friction` or `damping` under relaxation is rejected. A node resting on a rough surface
  creeps at the regularisation velocity (of the order of 0.1 mm/s) under a tangential force below μN, rather than sticking strictly; to stick, use the connector's
  `friction` law (Section 11.11)
* Nodes that are behind the surface at the start are rejected by name — contact is not responsible for pushing the model back into place
* A surface restricted to a `region` also stops the fragments split off that region's nodes (before v1.0 they passed through)
* A rigid surface knows nothing of the members: it does not know the section thickness, and it is the nodes that come to rest. Contact BETWEEN members is `contact.members`, below
* Results: nodal fields `contact_force`, `contact_penetration`. When there is a contact surface, preflight's mechanism check for free bodies
  is downgraded to a warning — the surface is its support. Case case_33 (a beam falling to the ground after its hanger ropes are cut)

**Contact between members (`contact.members`, 1.11).** A falling span landing on the deck below, a girder end striking
its neighbour, a broken hanger whipping across the others, the pile a collapse leaves:

```yaml
contact:
  members:
    bodies:
      - {region: upper_deck, radius: 1.2}    # each member is a capsule of this radius about its axis
      - {region: lower_deck, radius: 1.0}
      - {region: hangers}                    # bars and cables: `auto`, the circle of their area
    friction: 0.4
    damping: 0.3
```

* A line member has no volume: in contact it is a **capsule**, the points within `radius` of its axis. Two members touch when
  their axes come closer than the sum of their radii, and push each other apart along the common normal, with a penalty spring
  on the overlap and, in dynamic steps, a dashpot (`damping`) and Coulomb friction (`friction`, regularised as the rigid
  surface's). Every member of every body is tested against every other, its own body's included, except the ones it shares a
  point with; connectors take no part
* **`radius` is the modeller's**: for a beam it must be written (a section does not say how far its surface is from its axis in
  every direction). A box girder 12 m wide and 2 m deep landing flat is a capsule of radius 1; where its edges are is not
  modelled. `auto` is for bars and cables alone
* The force goes to the two members' points with the weights of the closest point, statically equivalent to the force at the
  contact point; friction acts at the contact point, off the axes, so it also twists the members (a rolling log slows down)
* `stiffness: auto` (default): k = m/dt² from the pair's reduced mass (a point held in every step counts as infinite), divided by
  the number of contacts on the busiest point — **the time step is not reduced by contact**, and a member resting on another
  overlaps by tens of micrometres. A number is capped there. Friction's regularisation takes the levers into account, so a
  member that rolls is inside the step too. Bars and cables take no couple (the contact point is on their axis)
* **How fast is too fast**: two members must not close more than about half the sum of their radii in one step
  (closing speed × dt < 0.5 (r₁ + r₂)), or they pass through each other. A girder is nowhere near it; a thin cable under
  mass scaling can be (r = 0.07 m, dt = 1e-3 s: 70 m/s)
* **What overlaps at the start is let be.** A girder sits 30 mm above its cap on a bearing; with honest radii the two capsules
  overlap by a metre before anything has happened. Each pair that overlaps, or is within `slack` (default 5% of the smaller
  radius) of touching, in the reference configuration keeps that overlap plus the slack as an **allowance**: the contact acts on
  what exceeds it. The slack is what the bearing between them needs to work (without it the contact spring, far stiffer than any
  bearing, would take the dead load off it). The allowance only shrinks, following the pair apart, and is gone once they are
  more than the slack apart. So a deck whose bearing fails drops by the slack and lands on the cap from where it stood. A
  member that breaks off its neighbour gets its allowance at that step (the two sides of the break only), and so does a member
  born in a later analysis step. A pair that starts within the slack of touching may therefore overlap by up to the slack
  before the contact acts. The log says how many pairs started this way; the allowances are in the checkpoint
* `mode: static` refuses it (use a relaxation, which takes it; `friction` and `damping` act in dynamic steps only). Two long
  nearly parallel members touch at ONE point, which jumps from the middle of their overlap to an end as they tilt
* Not modelled: a section's shape (see above), contact with a solid mesh, a debris particle's contact (a parked particle has no
  member; `contact.rigid` still stops it at the ground). A member at rest on a slope creeps at the regularisation velocity
* **Speed**: since 1.12 `contact.members` runs inside the compiled step (frames, bars and connectors under `backend: numba`,
  and fibre beams), to the same bits as the step-by-step loop -- one evaluation of the contact is one compiled function that
  both call. 2-core container: Xihoumen (1207 members, all in contact bodies) 20.4 → 9.8 s for 18758 steps, against 9.3 s
  without contact; case_34 with every member a contact body 33.1 → 3.4 s (2.6 s without). The pair search is a neighbour
  list, remade when a point has moved a quarter of the smallest radius. A model with rigid surfaces (`contact.rigid`) still
  runs the loop, and so does a relaxation with contact. A fibre model with `fracture` kept the loop in 1.12.0 and takes
  the compiled step since (the step in which a fibre point fails is finished by the loop, which takes the fractured
  member out before that step's contact; a girder breaking onto a beam below, 18.8 → 2.4 s)
* Results: `contact_force`, `contact_penetration` (with the rigid surfaces', if any). Checked: a beam dropped across another
  touches at t = √(2(h − r₁ − r₂)/g) and rests with the axes r₁ + r₂ apart, the supports carrying its weight; a collision of two
  free beams conserves momentum to round-off; a beam sliding on two rails stops in v²/(2μg) (tests/test_member_contact.py)

### 11.13 Soil springs on piles: p–y curves by depth (1.2)

```yaml
soil_springs:
  - region: pile                     # the pile's members
    ground: 0.0                      # the ground surface (a coordinate on `vertical`)
    vertical: z
    direction: horizontal            # radial in the horizontal plane; x or y for a plane model
    curves:                          # p: resistance per unit length of pile; y: the pile's displacement (in the case's units: N/m and m in SI)
      - {depth: 0.0,  p_y: [[0, 0], [0.005, 4.0e+4], [0.02, 9.0e+4], [0.05, 1.2e+5]]}
      - {depth: 10.0, p_y: [[0, 0], [0.005, 1.6e+5], [0.02, 4.0e+5], [0.05, 5.2e+5]]}
```

* Each node of the region below the ground gets a horizontal spring: the curve at its depth times the length of pile the
  node stands for (the buried part of each member next to it, shared by its linear shape functions; the end above ground
  of a member that crosses the ground gets its share, at the curve of the surface). Between depths the curves are interpolated
  linearly at the same y; above the first depth the first curve, below the last the last; past a curve's last point p
  stays at its last value (the ultimate resistance). A curve starts at [0, 0], y increases, p is not negative.
* **The curves are the user's**: the code formulas (API, JTG 3363) are not in the library; work them out from the soil
  layers and write them in.
* **Nonlinear elastic**: loading and unloading follow the same curve, with no hysteresis and no gap behind the pile. It is
  for pushovers, dead-load states, modes and buckling; in a time history it dissipates nothing itself (Rayleigh β damping
  gives it some, at the springs' steepest slope), and radiation damping has to be accounted for otherwise. The springs
  resist horizontal displacement only: a battered pile's axial motion is resisted by its horizontal part. A support
  `record` (multi-support excitation) is refused with them -- the springs act on the absolute displacement and the soil
  would not move with the record; a uniform `ground_motion` (relative motion) is fine. With `direction: horizontal` the resistance follows the resultant of the two horizontal
  displacements (a push at 45 degrees meets the same soil) and has a potential, so the tangent is symmetric.
* The soil is a support: its force on the pile is in the `reaction_force` field and in `kind: reaction` monitors, and in
  `vfife summary`'s sum of reactions; preflight counts the pile's nodes below ground as held sideways. The springs'
  steepest slope joins the reference stiffness of the explicit step and the relaxation masses.
* For beam2, Timoshenko, fibre and mixed line models; bars (truss2), shells and solids are refused. There is no t–z (shaft
  friction) or q–z (tip) spring: hold the pile vertically with a constraint or a connector.
* Checked against a long pile with a lateral load at its head (Hetenyi's beam on an elastic foundation, y0 = 2Hβ/k): free
  head 0.48% low on 60 elements, converging as the square of the element length (also with the ground inside an
  element); a fixed head, a rigid short pile (4H/kL, H/kL), a fibre pile, the modes (√(k/ρA)) and buckling on a Winkler
  foundation likewise. A curve's second point has p > 0 (soil with no initial stiffness does not hold the pile).

### 11.14 The tendon block of a frame model, `tendons:`

A frame model's (beams, bars) post-tensioned bonded tendons can be written one entry each and are expanded on reading:

```yaml
tendons:
  - name: T1                         # its name; the regions it expands to are T1/bar1, T1/arm1, ...
    nodes: girder                    # the girder nodes it is bonded to: a group of the mesh
    section: strand                  # a bar section: the area and material of one strand
    count: 12                        # strands in the duct
    line: {origin: [0, 0, 0], direction: [1, 0, 0]}   # its reference line; `turn` (degrees) optional
    profile_z:                       # its drape: [x, z, radius], each corner rounded to its radius
      - [0.0, 0.0, 0]
      - [15.0, -0.8, 40]
      - [30.0, 0.0, 0]
    stress: {step: stressing, ends: both, stress: [1.395e9, 1.395e9]}   # or force: [the whole tendon's force, ...]
    friction: {mu: 0.17, k: 0.0015}
    draw_in: {begin: 0.006, end: 0.006}
    relaxation: {code: jtg3362, fpk: 1.86e9, zeta: 0.3}                # optional
```

* It expands to what the Midas converter has written since 1.15 (v1.15 and v1.16):
  * a tendon node at each node of the group along the tendon's length, and at the node nearest each anchor;
  * each tendon node joined to its girder node by a rigid arm;
  * a bar between each two tendon nodes, jacked to the force at its middle after friction and draw-in (`jack`).
* A bar is born in the step that stresses it. An arm and its tendon node are born with the girder there, so a tendon is not set off by a girder that has already deflected.
* The tendon's material is the section's, without weight (the girder's weight has the duct in it). With `relaxation`, each bar gets the curve of its own stress after anchoring, as in 11.3.
* What it adds comes after the mesh's own nodes and elements, so no existing node changes its row.
* What cannot be a tendon is refused by name (HIGH_LEVEL_BLOCK): a group that does not exist, a section that is not a bar, a step name that does not exist, profile points that do not run forward. What is followed only roughly is a preflight warning, TENDON_APPROXIMATE: an anchor beyond the members, nodes far off the reference line, stress above fpk somewhere.
* For frame models on a Gmsh mesh only. A fibre beam's tendons are still written in its section (11.1).
* `vfife expand model.yaml -o flat.yaml` writes the expanded case and its mesh (MSH 2.2), to look at and check. The two run to the same state, bit for bit.
* The result records what each tendon expanded to: under the `tendons` group, one subgroup per tendon, with its name, strands, area, the jacking force at each end and the step that stresses it, and its bars in order along it -- their element rows in the result (the rows of `axial_force`), their regions and the force each was jacked to. `vfife tendons result.h5` prints the tendon force table from it (12.3).
* From v1.20 the Midas converter (5.3) writes tendons as this block, writes `groups` from Midas structure groups and puts short lists on one line, writes `tendon_defaults` and the profiles' CSV, and writes the offset arms as one rule (11.16). A rigid-frame case drops from 68,000 lines to about 1,300, a continuous girder from 36,000 to about 900, and a continuous T-girder bridge from 91,000 to about 2,400 (with the tendon profiles in CSV). Most of what remains is each tendon's name, node group and offset, and one section per element of a tapered girder. `convert(..., expanded=True)` still writes the expanded form.

**What every tendon shares, `tendon_defaults:`.** A key that is the same for the tendons is written once:

```yaml
tendon_defaults:
  nodes: girder
  section: strand
  count: 12
  profile_z: {table: tendons_z.csv}  # with a name column: each tendon takes the rows of its own name
  stress: {ends: both, stress: [1.395e9, 1.395e9]}
  friction: {mu: 0.17, k: 0.0015}
tendons:
  - {name: T1, line: {origin: [0, 0, 0], direction: [1, 0, 0]}, stress: {step: stress1}}
  - {name: T2, line: {origin: [0, 0, 0], direction: [1, 0, 0]}, stress: {step: stress2, ends: begin}, relaxation: null}
```

* A key a tendon does not write comes from here. A mapping such as `stress` or `friction` is merged key by key, and the tendon's own keys win: above, T2 is jacked at its beginning only and otherwise as shared. When a tendon's `stress` writes `stress` or `force`, neither is taken from the shared one (only one of the two can be given).
* `null` in a tendon is its own value: `relaxation: null` means that tendon does not relax.
* Any tendon key but `name` can be written here; a key that is not one is refused (HIGH_LEVEL_BLOCK).
* A profile written as `{table: file.csv}`, where the file has a `name` column, gives each tendon the rows of its own name (11.15). A tendon's own `profile_z: {name: T9}` merges with the shared table and takes T9's rows.
* It can be in an `include`d file; a table it names is then relative to that file. A key in both files is compared as a whole, and `override: true` takes the case's whole key (not merged key by key).
* From v1.20 the Midas converter writes this: a key every tendon has, at the value most of them have, goes into `tendon_defaults`, and the profiles go to `model_tendon_z.csv` (`_y` across the line). The short form runs to the same state as the long form of `expanded=True`, bit for bit.

### 11.15 `include:`, `groups:` and `table:`

None of the three needs the mesh. They are expanded on reading, before the tendon block, and `vfife expand` writes what they become.

```yaml
include: [lib/materials.yaml, {path: lib/sections.yaml, override: true}]
groups:
  cantilevers: [L1, L2, R1, R2]      # mesh groups or other groups
  girder: [cantilevers, closure]
steps:
  - {name: cantilevers, activate: [cantilevers]}
  - {name: closure, activate: [closure], loads: [{table: sdl.csv, keep: true}]}
tendons:
  - {name: T1, ..., profile_z: {table: tendons.csv, name: T1}}
```

* **`include`**:
  * Merges a file's `materials`, `sections`, `reinforcement`, `groups`, `tendons` and `tendon_defaults`. Paths are relative to the including file, and an included file may include others.
  * A name defined in both files is refused; with `{path: ..., override: true}` the case's own definition wins.
  * An included file that holds anything else (`steps`, `solver`, ...) is refused, so each parameter lives in one place.
* **`groups`**: name → a list of mesh groups or other groups.
  * Once read, it is a group of the mesh holding its parts' elements and nodes, as a Gmsh physical group of several pieces would. It can be used, with the same meaning, wherever a region name is written. For example, a cable assigned to a group has an `unstressed_length` that is the whole group's.
  * It joins mesh groups only. A region written in the case (a box, a plane) has no elements and cannot be a part.
  * A group name must not be a region, a mesh group or `all`.
  * It lowers to `mesh.groups`, which is what `vfife expand` writes.
* **`table`**: a CSV file with a header row; lines starting with `#` are comments, and the units are the case's.
  * A tendon profile has the columns `x, z, radius` (or `x, y, radius` across the line). With a `name` column, `name:` picks one tendon's rows out of a shared file; without `name:` a tendon takes the rows of its own name (a table holding one name that is not the tendon's is still read whole, as before). A `name` cell is stripped of surrounding spaces, and a line whose first cell starts with `#` is a comment, which is why the Midas converter writes the `name` column last.
  * Nodal loads have `region`, or `x, y, z` (the nearest node), then `fx, fy, fz, mx, my, mz`; a missing column is 0. Each row becomes one `per_node` load, which can take `keep` and `amplitude`.
  * Nodal masses have `region` or `x, y, z`, then `mass`; they can take `weight` and `components`.
* The Midas converter writes these from v1.20:
  * a stage that builds all of a structure group's regions names the group in its `activate`, and the group is written once in `groups`;
  * sections, assignments and a tendon's line, which hold only numbers, are written one per line.

### 11.16 The arms of members set off their nodes, `offset_arms:`

A Midas girder is often drawn by the top of its section (offset CT), while the member is on its centroid's line. The converter puts a centroid node under each drawn node and joins the two with a rigid arm. The arm's stiffness is a multiple of what the members it carries do, so each arm used to have a section of its own: a converted continuous T-girder had 846. It is now written as one rule:

```yaml
offset_arms:
  material: rigid          # the arms' material (its E)
  factor: 30.0             # times what the members set off carry
  limit: 1.0e13            # the most one member hands an arm (force/length; none: 1e13 N/m in the case's units)
  regions: [E266, E267, ...]   # the arms: one member per region, from the drawn node to the centroid node
```

* Each arm's section is worked out on reading, from the mesh. The beams at its centroid node whose other end is the centroid of another listed arm are the members set off there; a pier drawn to the centroid is not one.
* A member set off hands over the larger of EA/L and 12EIy/L³ along (at most `limit`) and 4EIy/L in turning (at most limit·L²/3), L its drawn length. The arm takes the largest of each, and over its own length r: area = factor·along·r/E, iy = iz = factor·max(turning·r, along·r³/12)/E, j = 2iy. Its orientation is global Z, or global X for an arm nearer the vertical than 45°.
* The beams' stiffness is read from `assign`: the region must be a group of the mesh (or of `groups:`, or a node group of the mesh), the section must write `area` and `iy`, and the material `E`. A member set off whose section is given by its sizes, or beams assigned to a region written in the case (a box, a plane), are refused rather than skipped.
* Sections that come out the same are written once, named `offset_arm_1`, `offset_arm_2`, ...; so the case must not have sections of those names, and an arm's region must not have an `assign` of its own. A region that is not one member, one listed twice, an arm with no member set off at its centroid end, or a material that is not there is refused (HIGH_LEVEL_BLOCK).
* The Midas converter wrote this in 1.20; from 1.21 it writes end offsets, `end_offsets` (7.5), with no centroid nodes or arms, and the rule stays for the cases there are. In 1.20: The mesh holds coordinates to 12 significant digits, so the rule's arm stiffness differs from what the converter wrote before by about 1e-11 at most. An arm the rule does not give (one whose member set off is a stub, say) keeps its own section, renamed `offset_arm_own_1`, ...; the other arms still use the rule. `vfife expand` writes each arm's section.

## 12 Result output and post-processing

### 12.1 Output settings

```yaml
output:
  path: results/pc_beam.h5
  every: 1                     # write one frame every this many steps (incremental relaxation: one frame per increment)
  fields: [displacement, rotation, reaction_force, fiber_stress, fiber_damage]
  monitors:
    w_mid:  {kind: displacement, region: mid, dof: z}
    R_left: {kind: reaction, region: left, dof: z}
    sigma_p: {kind: bar, of: tendons, field: axial_stress, reduce: mean}
    dc_span: {kind: element, field: dc, reduce: max, region: span_block}
    eps_bot: {kind: fiber, region: mid, field: strain, at: [0.0, -0.6]}      # [y, z]: the soffit of a 1.2 m deep girder
    sig_bar: {kind: fiber, region: mid, field: stress, at: [0.0, -0.55], fiber_kind: steel}
```

A relative `path` is relative to the **working directory** (where the command is run), not to the case file; the mesh, earthquake records,
`initial_state` and the other input paths are the ones relative to the case file.

**Fields** (`fields`) are whole-model field quantities written in every output frame. Which fields an element can output is found with
`python -m vfife describe <element>` (`output_fields`); a field that does not exist is rejected before the run.
Left out, `fields` defaults by model (1.1): `displacement`, `rotation`, `von_mises` (solids, bars), a shell's `von_mises_top/bottom`,
`reaction_force`, `reaction_moment`, the members' signed forces `axial_force`, `shear_y`, `shear_z`, `torsion`, `bending_y`, `bending_z`,
`hinge_rotation` and `hinge_usage` with plastic hinges, `connector_force`, `connector_moment` and `connector_deformation` with connectors,
and `member_failed` and `capacity_ratio` when members can fail; a default name the model does not produce is skipped without a word.
Not in the default: the end actions (`end_actions`), the envelopes (`*_max_abs`), fibre stresses and damage (a number per fibre per frame,
asked for by name). The 1.0 default was `[displacement, von_mises]`, which wrote a beam or bar model's displacements and nothing else.
A name written in `fields` that the model cannot produce is refused (`UNKNOWN_OUTPUT_FIELD`); a written `fields` is written as given.
Some names stand for a group: `end_actions` writes the six actions at both ends (`axial_force_i` ... `bending_z_j`),
`connector_force`, `connector_moment`, `connector_deformation` and `connector_slip` write their components (the forces as
`connector_force_x/y/z`, `connector_moment_x/y/z`; the deformations and slips per component, `_ux` ... `_rz`), and a shell's
`membrane_resultants`, `bending_resultants` and `shear_resultants` write each resultant.

**Monitors** (`monitors`) are time-history curves:

| kind | What it reads |
|---|---|
| `displacement` | A displacement component of the region's nodes (`relative_to: <region>` reads relative displacement) |
| `reaction` | Resultant reaction of the region's nodes |
| `element` | an output of the region's elements (solids: material outputs; line models: any member field, e.g. `axial_force`, `bending_y_max_abs`), reduced by `reduce` (max/min/mean/absmax) |
| `bar` | Axial stress, axial force or plastic strain of a group of bars (the `name` given at assignment) |
| `fiber` | Strain or stress at a specified position `at: [y, z]` on a fibre section (y across the width, z through the depth) |

`bar` only on solid models, `fiber` only on fibre beams, `element` on solids and line models (not shells); `reduce` only applies
to `element`/`bar`. A static run that does not write `every` writes a frame at every increment. A `fiber` point outside the section
outline is refused (`FIBER_MONITOR_OUTSIDE`, v1.0: `at: [-0.6, 0]`, an example this manual printed, read the side of a 0.5 m wide
section at mid-depth -- 4e-5 against the soffit's 1e-3). A `field: stress` reading takes the nearest fibre, among the concrete or the
steel fibres only with `fiber_kind: concrete | steel`; which fibre it read (element, station, coordinates, kind, law) is written on the
monitor in the result (`read_*` attributes, docs/result_format.md). A monitor the model cannot record is reported by preflight as
`MONITOR_UNSUPPORTED` (a beam model used to record it as a displacement, or as 0). In dynamic and relaxation runs the
monitors are recorded at the output frames; a static run records them at every increment and `every` thins only the
frames. Either way `time` is as long as each monitor. `time`, `residual`, `kinetic_energy` and `age` are reserved and cannot name a monitor.

**Checkpoints**: `checkpoint_every: N` writes the checkpoint to `<result>.ckpt.h5` beside the result about every N steps
(taken at a frame or progress check; the progress interval is lowered to N when larger; before v1.0 it was written
only at an output frame whose step number N divided). One is also written at the end of a run and when it is stopped, except when the final state is not finite.

### 12.2 Result file

The result is an HDF5 file with an XDMF description file next to it:

* `monitors/`: every monitor and `time`
* `steps/<step number>/`: field quantities of each frame, e.g. `displacement (node, 3)`, `fiber_stress (element, integration point, fibre)`
* `diagnostics/`: the energy account (external work, internal energy, kinetic energy, dissipation) and the convergence record
* Attribute `case_yaml`: the complete case at run time, so results can always be traced back to the input

Open the `.xdmf` in ParaView; if an older ParaView reports an error reading the time series, open `_last.xdmf` (which has only the last
frame). A six-component field (a solid's stress or strain) is written as a `Matrix`, with each component also written
as a scalar of its own: `stress_xx`, `stress_yy`, `stress_zz`, `stress_xy`, `stress_yz`, `stress_zx` (the stored order and
the names monitors and contracts use; `stress_xz` before v1.0). Colour by those;
ParaView's `Tensor6` ordering is not this program's (it was labelled wrongly before v1.0). Node quantities sit on the
nodes and element quantities on the elements; a member split into particles is drawn on its own connectivity at every
frame.

`residual` and `kinetic_energy` under `monitors/` are **convergence series**, sampled at every progress check rather than
at the output frames, so they are not as long as `time`. Incremental relaxation records them at every progress check within
each increment.

Member forces are signed: `axial_force` tension-positive; shears, torque and moments by the right-hand rule in the
member's local axes on the +x face (with local z up a sagging moment is negative); the larger end with its sign, a fibre member's
largest station. The old unsigned values are `*_max_abs`. `*_i/_j` are the end forces on the member (internal force = −end i,
+end j). A failed member reads 0. A continuation's `member_failures`/`member_splits` hold the whole history; the first
`rows_before_restart` rows are already in the earlier file. The full statement is in docs/result_format.md.

The full layout (groups, datasets, shapes, node and cell fields, units, frame naming, what `time` means in each mode,
attributes and versioning) is in docs/result_format.md. Result format 3: the checkpoint is a separate
`.ckpt.h5`; the case's declared units are root attributes (`unit_length`, `unit_force`, ...); `equilibrium` is always the
string `"true"`, `"false"` or `"not_applicable"`; tables carry `columns` / `dims` attributes; the XDMF clock of static
and relaxation results is the record index 0, 1, 2, ... (an arc-length λ runs backwards), with the load factor λ and the
age as field data `load_factor` and `age`; `_last.xdmf` carries only the six stress component scalars, no `Matrix`.

### 12.3 Curves and tables

```bash
python -m vfife plot results/pc_beam.h5 --x w_mid --y R_left    # monitor curve -> PNG
python -m vfife export results/pc_beam.h5                        # all monitors -> CSV
```

The CSV has `time` as its first column and leaves out the convergence series; name them (`residual`, `kinetic_energy`)
to get a table whose first column is `check`, the progress check. Naming both kinds together is refused
(`MIXED_SAMPLING`); a plot gives the convergence series their own axis. Axis units are the case's declared `units`
(m and N when none are declared; before v1.0 they were m and N whatever the case used). Stacked plot panels carry
name and unit on the axis and the description as the title; the time axis reads `time [s]` or `load factor (completed
steps + λ)`. A series of one or two records is drawn with markers (a one-frame result used to give empty axes).

**Result summary** (1.1): `vfife summary <result.h5>` prints one page to read before any plot:

* whether the run finished and is an answer (status, convergence, frames, residual);
* the model's mass, the weight its `gravity` loads apply, the resultants of its `line_load`s and nodal forces (the last
  step's, each at its amplitude's value at the step's end), and the sum of the reactions at the last frame; for a static or relaxation result also "reaction / (weight + line and point loads)",
  which is 1 when these are the only loads and the frame is in equilibrium -- a model whose sections weigh 2.8 times
  what they should shows it on this line;
* over every frame, the largest displacement (magnitude and components), rotation, signed member force (largest and
  smallest), stress and connector force, each with its node or member, position and time; the first members to fail
  and how many have failed at the last frame;
* the energy balance's relative error (final and largest), and each monitor's final value and range.

```bash
python -m vfife summary results/pc_beam.h5            # one page of text
python -m vfife summary results/pc_beam.h5 --json     # the same, as JSON
```

The mass, weight and line load are written into the result's root attributes at run time (`model_mass`,
`model_weight`, `model_line_load`). A result written before 1.1 does not have them; `--case <case>` rebuilds the model
from its case for them. Only the fields the result holds are read: member forces must be in `output.fields` (they
are in the default, 12.1).

**Tendon force table**: for a case with a top-level `tendons:` (11.14), `vfife tendons <result.h5>` lists, step by step, the force in each tendon at the step's last frame -- the least, mean and largest axial force over its bars, and the mean against the jacking force:

* a step before the tendon is stressed shows it as not yet stressed; in the step that stresses it each bar holds the force it was jacked to (ratio 1), and what changes later is the losses -- elastic shortening, relaxation and so on;
* `--bars` gives one row per bar; `--csv file` writes the rows as CSV. The table's columns are `tendon, step, time, least, mean, largest, of_jack` (the last is the mean against the jacking force); with `--bars` they are `tendon, bar, region, step, time, force, jack`;
* a step's row is its last frame: the end of the step, or, for a run that stopped part-way, where it stopped;
* the result must hold `axial_force` (the default does). A result written before v1.20 has no `tendons` group and must be run again; a case written out by `vfife expand` has no `tendons:` any more, and its result has no such group either. The group's layout is in result_format.md.

```bash
python -m vfife tendons results/model.h5
python -m vfife tendons results/model.h5 --bars --csv bars.csv
```

### 12.4 Fibre-section sweep

```bash
python -m vfife sweep benchmarks/break_sweep_demo.yaml --result results/break_sweep_demo.h5
```

Sweeps the fibre section along the deformed member axis into a surface, written frame by frame as `.pvd` + `.vtu`, coloured by fibre state (cracked, crushed,
yielded, ruptured), for looking at the damage distribution and collapse process of box girders (requires meshio). Without `--result` it reads the
result the case's `output.path` names; `--every` must be at least 1.

### 12.5 3D playback web page: `vfife view`

Meshes are made in Gmsh and large-scale post-processing is done in ParaView (1.3, 12.2, 12.4). The two pages here are read-only. The model page shows what only vfife knows: supports, connectors, tendons and analysis steps. The result page replays a run with members at their section size, breaks and monitors, in one file that opens in a browser. They do not edit a model and are not meant to grow into a pre- or post-processor.

```bash
python -m vfife view results/case_33_falling_beam.h5            # -> results/case_33_falling_beam.html
python -m vfife view results/case_23_suspension_bridge.h5 --case cases/case_23_suspension_bridge.yaml
python -m vfife view results/d1_act1_typhoon.h5 results/d1_act2_quake.h5 results/d1_act3_cut_k3.h5 \
    --case cases/d1_act3_cut_k3.yaml --names "typhoon,traffic+earthquake,3 hangers cut" --max-frames 300 -o demo.html
```

Writes a result file as **a single web page carrying its own data**: results of any line elements, shells or solids (including coupled models and their reinforcement)
can be rotated, zoomed and played back in time in the browser. Given several result files of **the same model**, they become several acts in one page (switched by buttons,
with `--names` giving the act names; results of one model may differ in the spare particles reserved for splitting and still share a page). Given `--case`, line members are drawn at their section size: sections with a shape are drawn from b/h or d, `general` sections are back-calculated from
A and I (steel as thin-walled h = √(4I/A), concrete as solid √(12I/A), distinguished by E), and cables at the equivalent diameter of their area;
`--hide rigid,arm` omits the rigid arms and links used for modelling; `--fields a,b` includes only these fields (for when the page has a size limit).
A `--case` that cannot be read or is not the model of the result, a `--hide` region that does not exist or holds no members, a `--fields` entry
the result does not hold, and a `--names` count different from the number of results are all refused, not silently ignored. `--fields` accepts a
group name (`end_actions` expands to the components the result holds); a field the result holds but the page cannot draw (e.g. `rotation`) is refused
with the reason (one value per node or element; a vector by its magnitude). `--hide` works on models of line members only; for a coupled or
shell_line model it is refused as not supported. Empty entries in `--names` are dropped.
To draw the **true section outline** (for example a streamlined box girder or tower legs), call `vfife.post.view.write(...,
profiles={region: {"outline": [[y, z], ...], "outboard": True}})` from Python: give the outer section outline per region (metres; y along the member width,
z along the member depth — girder depth vertical, while for near-vertical tower legs z is along the bridge axis), and `outboard` makes the outline's +y face away from the model centreline
(the wind fairings at the outer edge of the side boxes). This affects only the drawing, and neither reads nor changes the computation; the D1 demonstration (nsfc `scripts/d1_xihoumen.py`) was drawn this way.
The page contains:

* **Continuous colouring** (on by default): each member is shaded along its axis from the colour at one end to that at the other. Nodal fields (displacement) use the values at the two end nodes;
  element fields (capacity ratio, internal forces) are first averaged at the nodes over members of **the same section** (girder values are not mixed with cables meeting at the same
  node), and the two sides of a break are not averaged with each other. Turned off, each member gets one colour. Shells and solids are likewise averaged at nodes
* **Global coordinate axes** (on by default): X, Y and Z axes with ticks drawn on three edges of the model's bounding box (unit taken from the case's length unit), and in the lower left corner
  an orientation indicator that rotates with the view; very short axes (e.g. the bridge width) are labelled only with ticks at their two ends
* Members coloured by any scalar field (drop-down menu: `|u|`, `u_x/u_y/u_z`, every nodal and element field in the result file, e.g.
  `capacity_ratio`, `axial_force`, `von_mises`, `|contact_force|`), with the legend giving the field's range over all frames
* Broken members (`member_failed`) are drawn grey and can be hidden: while the two ends have not yet separated they remain one segment; after separation each node carries
  half a segment (oriented along the neighbouring segment it is still connected to), and broken cables disappear; fracture events are tick marks on the time axis, counted in the upper right corner
* Members without a capacity criterion (towers, crossbeams) are neutral grey in the `capacity_ratio` contour, not the lowest end of the colour scale;
  the colour scale of internal-force fields takes the 0.5%–99.5% quantiles (tower moments are three orders of magnitude larger than those of the box girder; scaled to the maximum, the box girder would be all one colour),
  values beyond them are drawn in the end colours, and the legend marks ≤/≥
* Displacement magnification (automatic or ×1…×1000), by default **relative to the first frame** — for a bridge starting from the dead-load state, what is magnified is the vibration relative to dead load,
  not the dead-load deflection itself
* Monitor curves with a time cursor; for relaxation and static (including arc-length) results the axis is the load factor λ, not seconds, and the
  curves are drawn against the record index, so an arc-length path that goes back and below zero stays on the chart; a creep, shrinkage or ageing
  relaxation is likewise drawn against the record index and read off in the concrete's age, in days; play, pause, drag, speed
* Solids are drawn with outer surfaces only (the interior faces are found once by sorting), shells as surfaces, and line members as square bars (without `--case`, a single thickness,
  `--thickness`, default 1/300 of the model diagonal)

Data compression: displacements are quantised per frame to 16 bits (1/32000 of the maximum displacement), and only particles used during the run are stored (spare
particles reserved for splitting are not stored if unused); displacement components and magnitude are computed in the page from positions; `member_failed` is reconstructed from the fracture event table;
connectivity after splitting stores only changes; other fields are quantised to 8 bits (256 colours, finer than a contour plot can distinguish), and constant fields store just one value;
everything is embedded in the HTML as base64. The number of frames is at most 400 by default (sampled uniformly, the last frame
always kept); `--every k` takes one frame every k frames. Xihoumen, four acts of 240 frames each with four contour fields: 14.6 MB. Only
one thing on the page needs the network: three.js is fetched once from two CDNs (cdnjs.cloudflare.com, cdn.jsdelivr.net). Where
those are not reachable the page stays dark with a note; `--scripts DIR` writes three.js into the page instead, from a folder holding
three.js r128's `three.min.js` and `OrbitControls.js` (as in the npm package `three@0.128.0`: `build/` and `examples/js/controls/`,
or both directly in the folder), and the page then needs no network (1.16; a page made by an earlier version without real section
outlines stopped on its first frame, black, whatever the network -- v1.16). Modal files from `vfife modal` are not run results and are explicitly rejected.

**The model before it is run**: given a case file (`.yaml`), `vfife view` writes the model page:

```bash
python -m vfife view bridge.yaml                 # -> bridge.model.html
python -m vfife view bridge.yaml --scripts three_r128 -o model.html
```

* The page shows the model as the solver reads it: the high-level blocks (11.14–11.16) expanded, with the tendon nodes, bars and arms `tendons:` adds, and the regions resolved as the solver resolves them. Nothing is run and no solver is built.
* It goes step by step. The slider at the bottom picks the step, and the page draws the members standing in it. Members not built yet (or removed) are faint, or hidden; those the step builds are magenta.
* Members are thin lines of a fixed width on the screen (`--thickness`: the beams' width in pixels, 2 by default), every member's nodes small dots, those only the rigid arms reach included; a tendon's own nodes are not drawn. Colour by member kind (beam, cable or bar, tendon, rigid arm in red, connector), section, build step or region. Rigid arms can be hidden (what stands on them stays). The region list (the regions `assign` names) shows one region's members, the rest faint -- or hidden, with the faint members off.
* With "sections at size" ticked (1.21), beams and bars are boxes of their sections' size: depth along local z, width along local y (the axes `orientation` gives), the size taken as the result page's `--case` takes it (from the shape, or for a `general` section the depth A and I give), at most half the member's length. A member with end offsets (`end_offsets`) is drawn between its two elastic ends, and the rigid stretch from a node to an elastic end as a thin red line. Rigid arms, connectors and sections with no size stay lines. The switch cannot be ticked when no section has a size.
* Supports, connectors and soil springs are drawn the way the usual programs draw them -- one symbol each, what each degree of freedom or component is in a short code beside it -- and each kind has its own switch (v1.20):
  * a support: a triangle under the node, filled where a translation is held, open where only rotations are, with a bar under it where a rotation is held. Its code is the six dofs ux uy uz rx ry rz, `1` held and `0` free (`111000`), a held translation red and a held rotation magenta; a dof that moves is cyan, and so is the triangle (a nonzero `value` for it, a `motion` other than displacement, or the translations a `record` drives). The model's constraints, the step's own and those kept from earlier steps count; those `release`d do not.
  * a connector: an open square at its first node, an open triangle at its second (the nodes' dots show through), its axes as a small triad at its middle (x red, y green, z blue: x along it; y the `orientation` made normal to it, or the global axis most nearly normal to it; z = x × y; the global axes with `coordinate_system: global`). Its code is the six components: `K` a stiffness, `N` a `curves` table, a `bilinear` law or a `material` law, `G` a `gap` law, `F` a `friction` or `pendulum` law, `D` a damper alone, `0` free; a component with a `connector_damping` or a `damper` law beside it is blue (`KK0000`). A friction, pendulum or bilinear law the same on uy and uz -- one law in the y–z plane -- is in lower case (`Kff000`); a trailing `*` marks a coupled 6×6 stiffness or damping.
  * soil springs (`soil_springs`): a short spring to the ground at each node they act on.
  * The symbols keep their size on the screen; the size slider scales the symbols and codes (not the clicked one's full drawing). The supports' codes are placed before the connectors'. A code that would overlap one already written is left out (its symbol is still drawn); zoom in, or turn the codes off.
  * Click a support or connector to see it in full: each held dof as a bar along its axis (a ring about the axis for a rotation), each component on its axis -- a spring, a yellow spring, two plates (gap), a block on a track (friction, pendulum), a spiral (rotation), a dashpot beside it; `direction: horizontal` soil springs as one radial spring on the two horizontal axes, tied by a dashed arc; a friction or pendulum law the same on uy and uz as one law in the y–z plane, its two symbols tied the same way.
* Loads: the step's loads -- the model's, its own and those kept from earlier steps -- are drawn as arrows.
  * Nodal forces are red, pointing at the node; couples are purple, along the couple's vector.
  * Line loads are blue, at the members' middles; a line load in the members' local axes is listed, not drawn.
  * Each arrow is scaled to the largest of its kind.
  * Gravity, temperature, moving loads and the rest are listed in the left panel.
* Click a member to see its number and nodes, kind, region, section (name and values), length, and the steps that build and remove it; a tendon's bar also shows its tendon, a connector its axes and what each component is. Click a support (or a node with soil springs) to see the node, the degrees of freedom held in the step and those that move.
* For frame models only (beams, bars, cables, connectors); a solid or shell case is refused (VIEW_MODEL). `--scripts`, `-o` and `--title` work as for results.
* For three real bridges (6,600, 4,100 and 9,300 members) the page takes 6–17 s to write and is 1–3 MB.

### 12.6 Python interface

`vfife.api` is the interface for scripts, notebooks and front ends (API version 1).
The names in `vfife.api.__all__` are the promise: within 1.x their signatures and meanings do not change, and only parameters with default values are added.
Name-by-name documentation is in `docs/api_reference.md`, generated from the code's docstrings
(`python -m vfife reference --api`); tests guarantee the two are consistent.

**Running a case**:

```python
from vfife import api
outcome = api.run_case("pc_beam.yaml", output="results/pc_beam.h5")
print(outcome.status, outcome.converged, outcome.message)
print(outcome.monitors["w_mid"][-1], outcome.times[-1])
```

`run_case` does not exit the process. A numerical failure also returns a result, with the reason for failure in `status` and `message`,
and the diagnostics, the monitors recorded before it and the checkpoint are still written; the frames written before it stay, but the
failed state itself is not written as a frame (it is in `<result>.ckpt.h5`).
A mistake in the input raises `api.UserError` (a `ValueError` with `code`, `message` and `hint`; codes in docs/error_codes.md); a case
that does not fit the schema raises pydantic's `ValidationError` (also a `ValueError`) and a missing file `FileNotFoundError`, so catching
`ValueError` and `OSError` catches every input mistake. A model that fails preflight is not raised: the outcome's status is `PREFLIGHT_FAILED`. To stop a run part-way, return False from `on_progress` (or press
Ctrl-C): once the result and checkpoint are written, `api.AnalysisInterrupted` is raised; its `result_path`, `sidecars` and `report` say what was
kept, and `restart=` resumes from it.

**Changing a parameter and running again**:

```python
case = api.load_case("pc_beam.yaml")
case.get("materials.concrete.fpc")                    # read
stronger = case.with_changes({"materials.concrete.fpc": 5.0e7,
                              "loads[0].line_load[2]": -3.0e4})
api.run_case(stronger, output="results/pc_beam_fpc50.h5")
```

* Keys are dotted paths through the YAML, with `[i]` (or `.i`) for lists.
* Only the last segment may be new (for example, giving `solver.threads` that was not written before).
  A misspelled section or material name raises an error listing the names that actually exist; a misspelled field name is rejected by the schema before the run.
* A modified copy remembers the folder of the original case, so meshes, earthquake records and `initial_state` are still found via the relative paths written in the original file.
  Passing a dictionary directly to `run_case` does not do this: relative paths are resolved against the current directory.

**Parametric studies**:

```python
rows = api.study("pc_beam.yaml",
                 {"materials.concrete.fpc": [3.0e7, 4.0e7, 5.0e7],
                  "loads[0].line_load[2]": [-2.0e4, -3.0e4]},
                 output_dir="results/fpc_study",
                 extract=lambda o: {"w": o.monitors["w_mid"][-1]})
```

* By default all combinations are run (6 runs in the example above); with `combine="zip"` the items are paired one by one.
  A list `[{key: value}, ...]` can also be given directly, one run per item.
* Each run writes `output_dir/<case name>_<index>.h5`.
* One row per run: parameter values, status, whether converged, result path, time taken, plus the quantities returned by `extract`.
* If one run fails, only the reason is recorded and the study continues; if `extract` fails, the run's status and result path are still recorded
  and the `extract` error goes in the `extract_error` column.
* The list of rows is also written as `output_dir/study.csv`; `pandas.DataFrame(rows)` is a table.
* Misspelled keys and field names (material parameters, keys inside a `creep`/`shrinkage`/`relaxation`/`ageing` block and `units` keys included)
  are reported before the first run; an invalid value is recorded on that run's row only.

There are also `prepare_case` (validate, read the mesh, preflight, without building a solver), `read_monitors` (read the curves in a result),
`verify(contract, result)` (check a result against a contract; the arguments the other way round are refused as `VERIFY_ARGUMENTS`) and `case_schema` (the JSON Schema of the case file itself — materials / sections / assign and the rest — for user interfaces and external
generators; every shipped case conforms).

---

## 13 Model checking and result verification

### 13.1 Pre-run checks

```bash
python -m vfife check case.yaml     # model-quality JSON: regions, constraints, unit magnitudes, mesh quality, output fields
python -m vfife info case.yaml      # build the model and print a summary, without solving
```

Preflight is also executed at the start of every run, and stops immediately when it finds an error. Common reports: unconstrained rigid-body motion (with the
constraint needed: on a six-dof model the rotational dof itself), region snapping, unassigned elements, empty sections, inconsistent unit magnitudes, poor mesh quality.
`info` also lists each connector section (region, element count, each component's stiffness or law) and the model's total mass and weight,
in the declared units.

### 13.2 Invariant probes

```bash
python -m vfife probe case.yaml
```

Asks the model four element-independent questions: whether the internal force is the gradient of some potential, whether rigid-body motion produces no internal force, whether the mass can
bound the time step, and whether the current state is stable. Used to diagnose new elements, new materials or suspect models.
For prestressed, thermal, creep/shrinkage/relaxation or initial-state models the force-free and stability checks are skipped at the
reference and bent states (with the reason); use `--result` on a converged frame.

### 13.3 Verification contracts

Every numbered case under `verification/` has its own independent verification contract (YAML), stating what class of evidence it is, what it is compared
against, and how large a difference is allowed:

| Class | Meaning |
|---|---|
| `analytical` | Closed-form solution |
| `public_benchmark` | Published test or benchmark (external target) |
| `independent_reference` | Reference solution obtained by independent assembly or integration |
| `regression` | This project's baseline or a physical invariant, not a public benchmark |

```bash
python -m vfife run    cases/case_01_beam_cantilever.yaml
python -m vfife verify cases/case_01_beam_cantilever.yaml
```

`verify` returns 0 for pass and 2 for fail, and writes JSON and Markdown reports to `results/verification/`. Without `--contract` it
looks for `verification/<case name>.yaml` beside the case's folder (the shipped layout, `cases/` and `verification/` side by side),
then inside it, then under the working directory (before, only under the working directory); without `--result` it reads
the file the case's `output.path` names, else `results/<case name>.h5`.
A metric consists of a source (`monitor`, `diagnostic`, `field`), a name, a reduction and an expected value; besides
`final`, `max` and the like, reductions include `at` (the value of this curve when another curve reaches a given value), `energy` (area under the curve) and
`at_peak_of` (the value of this curve at the peak of another curve), so a contract can constrain the shape of the whole load–displacement curve, not only
its peak. A vector field read with `source: field` at a `location` needs a `component` for `final`, `first` and
`delta` (from v1.0; it silently returned z).

Users can also write contracts for their own cases, following `verification/README.md`.

### 13.4 Tests

**Three promises about a result, each held by tests:**

* the compiled step gives the result of the step-by-step loop, to the bit;
* a run stopped and resumed gives the result of the uninterrupted run, to the bit;
* **a result does not depend on where its output frames and progress checks fall** (`output.every`, `--progress-every`): a
  frame after every step and no frame at all end in the same state, to the bit. Before 1.12.1 a fibre beam, a frame or a truss
  with a break did not (a force evaluation that should only read the state broke the member a step early, Section 11.10);
  `tests/test_reading_is_not_a_step.py` runs the comparison for every solver family and every kind of history a material
  keeps, and a new solver or law belongs in its table.

```bash
python -m pytest -q            # default tier (about 2200 tests; about half an hour on a two-core machine)
python -m pytest -q -m slow    # solves every shipped case one by one and checks against its contract (several hours)
python tools/check_all.py      # all checks, run in parallel on the local machine, see below
```

**All checks (`tools/check_all.py`)**: the nightly verification tier on GitHub (sixty jobs each night, each up to two hours)
has been switched off, and the same checks are run on the local machine. There are four parts:

* `fast`: the default test tier;
* `slow`: the slow tests other than the case contracts;
* `cases`: every shipped case solved and checked against its contract. One process per case, single-threaded, `--jobs` running at the same time,
  defaulting to half the number of cores;
* `package`: builds a wheel, installs it into a brand-new virtual environment, and runs a case outside the repository (requires `pip install build`).

The report is written to `results/check_all/<time>.md`, and the exit code is non-zero if anything fails. Common usage:

```bash
python tools/check_all.py --jobs 8                        # eight cases at the same time
python tools/check_all.py --only cases --cases case_24g case_36
python tools/check_all.py --include-slow-cases            # also run case_15 (one to two hours)
```

What remains on CI: the default tier runs on every push; the verification tier runs only when triggered manually
from the Actions page (Run workflow).

---

## 14 Command-line reference

| Command | Purpose |
|---|---|
| `run <case>` | Solve a case |
| `mesh <msh>` | Report on a mesh: element types, available formulations, physical groups and their node ranges |
| `info <case>` | Build the model and print a summary, without solving |
| `check <case>` | Pre-run model check, JSON output |
| `probe <case>` | Mechanical-invariant probes |
| `list [materials|elements]` | List the registered materials and elements |
| `describe <name>` | Parameters, state variables, output fields and backends of an element or material |
| `capabilities` | Query element–material–backend–analysis-mode combinations (`--element --material --backend --mode`, mode dynamic / relaxation / static) |
| `new <template> [case]`, `new --list` | Write a case that runs as it stands from a scenario template (5.5, 1.2) |
| `init <case> --mesh <msh>` | Generate a case skeleton from a volume mesh (an existing file needs `--overwrite`) |
| `schema [-o FILE]` | Write the case file's JSON Schema (default `vfife.schema.json`) for completion and checking in an editor (2.4) |
| `reference [--api\|--codes] [--check]` | Generate `docs/yaml_reference.md` and `docs/vfife.schema.json` from the code (`--api`: `docs/api_reference.md`; `--codes`: `docs/error_codes.md`; `--check` only checks whether it is out of date) |
| `migrate <old> -o <new>` | Rewrite an old-format case in the new format |
| `expand <case> -o <flat>` | Write the case and mesh the high-level blocks (`tendons:`, `groups:`, `offset_arms:`, ...) expand to, to look at (11.14–11.16) |
| `verify <case>` | Check results against the verification contract |
| `plot <h5> --x A --y B` | Plot monitor curves |
| `export <h5>` | Export monitors to CSV |
| `formfind <case> --groups ...` | Cable form finding |
| `modal <case> [--result h5] [--modes n] [--paraview]` | Modes in the loaded state |
| `sweep <case> --result <h5>` | Fibre-section sweep export |
| `view <h5> [-o page.html]` | Write results as a self-contained 3D playback web page (12.5) |
| `view <case.yaml> [-o page.html]` | The model before it is run: members, supports, connectors, springs and loads step by step, a member's section on a click (12.5, v1.20) |
| `buckling <case> [--step k] [--result h5] [--modes n] [--paraview]` | Linear buckling: load factors and buckled shapes (11.8, 1.1) |
| `tendons <h5> [--bars] [--csv <file>]` | The force in each tendon of `tendons:` at the end of each step, against its jacking force (12.3) |
| `summary <h5> [--case <case>] [--json]` | One page on a result: status, mass and weight against the reactions, the largest values and where, energy, monitors (12.3, 1.1) |
| `qualify` | Admission evidence for new elements/materials |
| `--version`, `--debug` | Version; show the full call stack on errors. `--debug` is the program's option and goes before the command: `vfife --debug run case.yaml` |

`--frame` of `probe` and `modal` takes the step number, padded or not (`3338` or `00003338`).

A YAML syntax error, a result file that is not HDF5, and input ending at an interactive prompt each print one error line (exit code 1), not a traceback.
A YAML syntax error names the file it is in (case, contract or manifest). A file error names the path that failed ("cannot read or write <path>"),
with a hint when another program holds a result locked. When the output pipe closes (`| head`) the command exits quietly with code 141.
`vfife run` stopped by Ctrl-C (or a GUI's stop) exits with code 130, and stopped by SIGTERM (a cluster's time limit) with 143,
after writing the partial result and its checkpoint.

**Exit codes**: 0 success; 1 an input mistake (a refusal, a file that cannot be read, invalid YAML or schema); 2 a preflight FAIL
(`check`, `info`, `run`) or a `verify` FAIL; 3 an analysis that did not converge (`run`: MAX_STEPS or NONFINITE, the failed
result is still written; `formfind`: an inner relaxation that did not converge); 130 Ctrl-C; 143 SIGTERM; 141 a closed pipe.
A refusal prints its code: `error [CHECKPOINT_MISMATCH]: …`. Every code a refusal or a preflight issue can carry
is listed in docs/error_codes.md, generated from the code (`python -m vfife reference --codes`).

Use `python -m vfife <command> --help` to see the full arguments of each command.

---

## 15 Frequently asked questions

**Q: The run reports "did not converge".**
First look at the reported node and DOF with the largest residual. If the load increment is too large, increase `load_steps`; if it gets stuck right at cracking or yielding,
reduce the increment size in that range or increase `max_steps_per_load`; a static analysis beyond a limit point will never converge anyway, so switch to a slow
dynamic pushover (Section 10.2). See `docs/convergence_debugging.md`.

**Q: The results look reasonable, but the reactions do not match the applied loads.**
Almost always a region has selected the wrong nodes (wrongly guessed surface number, tolerance, snapping). Use `vfife mesh` to look at the physical groups and `vfife check`
to look at the region report.

**Q: The response of a dynamic analysis has been "smoothed out".**
Check the damping type: do not use `kinetic` in dynamic analysis.

**Q: A static problem converges extremely slowly.**
For nearly incompressible materials (ν → 0.5) the fictitious mass is governed by the volumetric term, and bending modes are very soft; switch to `mode: dynamic` with mass
scaling, or expect more iterations. For a box girder modelled as a fibre beam, first make sure `j` is given.

**Q: The torsional frequency of a box girder is about twice too high.**
`j` was not given, and the program used the polar moment of inertia.

**Q: A simply supported beam has stress under a temperature load.**
This is normal: the part of a nonlinear temperature distribution that cannot keep plane sections plane becomes self-equilibrating stress.

**Q: The numbers are off by a factor of 1000.**
Inconsistent units. Declare `units:` so that preflight checks magnitudes; for a mesh drawn in millimetres use `mesh.scale: 0.001`.

**Q: The camber from the prestressing tendons does not match expectations.**
First use `vfife info` to look at the stress range and position of each tendon; check whether post-tensioned tendons need `elastic_shortening:
compensate`; make sure self-weight and prestress act together from the start (prestress cannot be ramped).

---

## 16 Scope of application and known limitations

vfife is **an analysis engine, not design software**. The following are not included and must be handled separately by the engineer:

* Code load combinations and envelopes, code checking procedures
* Automatic loading by lane loads and influence surfaces, transverse distribution (moving loads travel along member axes)
* Radiation damping of the soil, t–z and q–z springs (shaft friction and the pile tip), hysteresis and gapping of p–y springs (p–y is nonlinear elastic only, Section 11.13); earthquake records are applied at the supports where they are written
* Shear lag, distortion and restrained-torsion warping when a box girder is modelled as a beam (fibre sections obey the plane-sections assumption)
* Construction stages are for line-element models only (`beam2`, `beam2_timoshenko`, `mixed_line`, `beam2_fiber`): members are built and removed step by step and supports released (Section 10.1, since 1.2). Shell and solid models are complete from the first step, and a fibre model's connectors must be built from the start; the other limits are listed in Section 10.1
* Velocity-dependent friction and stiffness degradation of bearings; connector nonlinearity is limited to Coulomb friction, gaps and bilinear hysteresis (Section 11.11)
* Contact with a section's real shape or with a solid mesh: members touch as capsules of a stated radius (`contact.members`, Section 11.12)
* Shear cracking of fibre beams is only a member-level stiffness reduction (`shear`), not coupled to flexure (not MCFT),
  and gives no shear failure; when the stress field of diagonal web cracks is needed, use a local solid model
* Temperature is only an applied field; there is no heat conduction, and material properties do not vary with temperature
* Vehicles are planar models (vertical + pitch), with no roll or transverse distribution; lane loads and dynamic amplification factors are code content and not in the library
* **Shells (shell3, shell4, shell_line) are experimental and not covered by 1.0**: the triangle shell3 locks in shear -- a thin plate deflects 0.2-20% of the true value, so mesh with quads; shell4's default hourglass stabilisation stiffens bending by 2-30% (more on elongated elements, not removed by refinement) and distorted quads lock. Preflight gives every shell model a `SHELL_EXPERIMENTAL` warning. Resultants are in the element's local axes, whose x axis follows the element's first edge and so the node order; the top/bottom stresses carry no transverse shear (`shear_x`, `shear_y` are Q; the mid-surface peak is 1.5 Q/t)
* A shell run on the warp (GPU) backend cannot be restarted from a checkpoint (`--restart`); use the numba backend
* Comparisons with commercial software are with Midas Civil, on the course models the project keeps outside the repository; the Midas translation follows tendon relaxation (1.16) and creep and shrinkage by the JTG codes (1.22), but no temperature, no moving loads and no plate stiffness
* A model whose only load is bars being `jack`ed does not converge as one continuous relaxation step (`mode: relaxation` with `n_steps` and no
  `load_steps`; its residual stalls near 0.5): give the step `load_steps`, or use `mode: static`, which the Midas translation writes
* Debris after a progressive collapse: up to 1.16, fragments could gain energy without bound once on the ground (a real steel truss
  losing an end post threw a dozen pieces off at 1e13 J after about 1.2 s). The causes were found on a generic truss
  (`benchmarks/debris_truss.py`) and fixed in 1.17:
  - the rotary inertia (its stability floor included) is a diagonal fixed in global axes and the step has no gyroscopic term, which a
    tumbling fragment makes wrong: on debris (particles joined to no support) a particle's three rotary inertias are made the largest
    of them, with the angular momentum kept. The structure still joined to its supports (`constraints`) is unchanged: modal analyses
    and dynamic runs in which no member breaks give what 1.16 gave, to the bit. A model standing on a contact surface with no
    constraint at all is all debris at its first break. A long span that falls loose can have its torsional inertia raised up to
    about 250 times, which slows its twisting. It applies only when some member can split (`split`);
  - a breakable beam twisted a quarter turn end to end is removed whole (cause 3 in `member_failures`). That is the range in which the
    co-rotational frame can read a twist (under half a turn), not a material criterion: torque is not checked against a strength, and
    a real steel member yields in shear well before;
  - a piece broken at both ends is removed once its demand reaches twice its capacity. Twice is a judgement: the piece is elastic once
    free and dissipates nothing plastically.
  The three are in the frame and mixed-line solvers; fibre-beam debris is unchanged from 1.16.
  The strain energy a removed member takes is not booked, so the energy error is on the low side: 0.3% on the generic truss with
  structural damping 1.0, 8% with 0.3, 6% with none. The same real truss, run again, went the whole 5 s with its kinetic energy no
  longer growing; but at about 0.8 s three bearing joints, left on nothing but their bearing connector once every truss member had
  split off them, were thrown off (1e14 m, and an energy error of 50% from then on). Why has not been found; leave such joints out
  when reading the results. Still read `diagnostics/relative_energy_error`

Accuracy and efficiency:

* 61 verification contracts, 402 metrics (classified by contract: 31 closed-form solutions, 17 published tests or benchmarks, 10 independent reference solutions,
  3 regression baselines); each states what it is compared against, by how much it differs, and why. On 0.4.2 every contract was re-solved and re-run, and all 50 existing at the time passed
* The descending branch of fibre beams after compression softening is sensitive to element length; the effect is secondary when capacity is controlled by reinforcement or strand
* Static solutions come from dynamic relaxation by default: seconds for linear elastic beam models; from several minutes to an hour for fibre models of a 3 × 70 m unit.
  Frames and fibre beams can use implicit static analysis instead (`mode: static`): the Vecchio & Emara frame goes from 22 min to 8 s,
  and the pushover of one Hangzhou Bay unit with piers from 50 min (explicit dynamics) to 30 s

---

## Appendix A List of shipped cases

Every case under `cases/` has a contract of the same name under `verification/`. The table below is generated from the contract files: the class is the evidence level declared by the contract
itself, and the number of metrics is the number of entries in the contract that can be judged pass or fail.

| Case | Content | Evidence class | Metrics |
|---|---|---|---|
| case_01_beam_cantilever | Cantilever beam with a concentrated tip load | Closed-form solution | 3 |
| case_02_truss_bar | Bar in axial tension | Closed-form solution | 3 |
| case_03_mixed_frame | Mixed beam–truss frame | Closed-form solution | 3 |
| case_04_nafems_le10 | NAFEMS LE10 thick plate | Published test/benchmark | 2 |
| case_05_space_truss_seismic | Space truss under El Centro earthquake | Independent reference solution | 2 |
| case_06_rc_column_cyclic | Cyclic loading of a reinforced concrete column (solid CDP) | Regression baseline | 7 |
| case_07_coupled_solid_truss | Solid–truss coupling with shared nodes | Independent reference solution | 3 |
| case_08_timoshenko_short_beam | Timoshenko short beam | Closed-form solution | 3 |
| case_09_connector_support | Connector support | Closed-form solution | 3 |
| case_10_clamped_square_shell | Square plate clamped on four edges (shell) | Closed-form solution | 3 |
| case_11_fiber_rc_column_cyclic | Tanaka & Park reinforced concrete column, cyclic (fibre beam) | Published test/benchmark | 3 |
| case_12_ualberta_sf0b | University of Alberta reinforced concrete beam in four-point bending (CDP) | Published test/benchmark | 10 |
| case_12b_ualberta_sf0b_bond | As above, with reinforcement bond–slip | Published test/benchmark | 7 |
| case_12c_ualberta_sf0b_confined | As above, with confined core | Published test/benchmark | 7 |
| case_13_pretensioned_beam | Pre-tensioned prestressed beam | Closed-form solution | 10 |
| case_14_draped_tendon_balance | Load balancing by a parabolic tendon | Closed-form solution | 6 |
| case_15_chalmers_prc | Chalmers post-tensioned prestressed beam in four-point bending | Published test/benchmark | 12 |
| case_16_lee_frame | Lee frame buckling (NAFEMS NL7) | Independent reference solution | 6 |
| case_16b_lee_frame_arc_length | Lee frame under load control, whole path traversed by arc length (including snap-back and negative-load branch): two NAFEMS points | Independent reference solution | 6 |
| case_17_hinged_cylindrical_roof | Hinged cylindrical shell roof (Sze 2004) | Independent reference solution | 7 |
| case_18_bend45_cantilever | 45° curved cantilever, large displacement (Bathe) | Independent reference solution | 12 |
| case_19_floating_beam | Free-flying beam (Wu & Ting) | Independent reference solution | 8 |
| case_20_rolled_cantilever | Cantilever rolled into a full circle (elastica) | Independent reference solution | 8 |
| case_21_vecchio_emara_frame | Vecchio & Emara reinforced concrete frame | Published test/benchmark | 7 |
| case_21b_vecchio_emara_cracked_shear | The same frame, member shear with the cracked shear law (stirrup truss model): ascending branch 1.15 → 1.08 | Published test/benchmark | 7 |
| case_21c_vecchio_emara_static | The same frame solved with implicit static (Newton): 8 s versus 22 min with dynamic relaxation, contract identical to case_21 | Published test/benchmark | 7 |
| case_22_cable_catenary | Elastic catenary cable | Closed-form solution | 5 |
| case_23_suspension_bridge | Suspension bridge (Irvine) | Closed-form solution | 8 |
| case_24_hzb_girder | Hangzhou Bay 3×70 m box girder, static (against Chen Hengzhi 2005) | Published test/benchmark | 11 |
| case_24b_hzb_girder_modal | Hangzhou Bay 70 m simply supported box girder, modes | Closed-form solution | 3 |
| case_24c_hzb_girder_collapse | One Hangzhou Bay unit pushed to collapse (against the ultimate load of Chen Hengzhi 2005) | Published test/benchmark | 7 |
| case_24d_hzb_girder_piers | One Hangzhou Bay unit with piers | Regression baseline | 7 |
| case_24g_hzb_static_pushover | One Hangzhou Bay unit with piers, implicit static pushover: dead load by Newton, live load by arc length, past the peak into the softening branch | Regression baseline | 9 |
| case_24e_hzb_girder_bearings | One Hangzhou Bay unit with pot bearings (elastic girder) | Closed-form solution | 7 |
| case_24f_hzb_girder_bearings | One Hangzhou Bay unit with piers and bearings (fibre beam) | Closed-form solution | 9 |
| case_25_chen_hollow_slab | Chen Hengzhi pre-tensioned hollow-slab beam No. 1 failure test | Published test/benchmark | 9 |
| case_25b_chen_hollow_slab | Chen Hengzhi pre-tensioned hollow-slab beam No. 2 failure test | Published test/benchmark | 8 |
| case_26_moving_load_beam | Moving load crossing a simply supported beam (Frýba) | Closed-form solution | 7 |
| case_26b_moving_load_influence | Deflection influence line | Closed-form solution | 7 |
| case_27_thermal_two_span | Two-span continuous beam under temperature action | Closed-form solution | 8 |
| case_27b_thermal_eigenstress | Nonlinear temperature self-equilibrating stress | Closed-form solution | 8 |
| case_28_confederation_trucks | Confederation Bridge truck load test | Published test/benchmark | 11 |
| case_28b_confederation_thermal | Confederation Bridge thermal deflection | Published test/benchmark | 6 |
| case_28c_confederation_uniform | Confederation Bridge uniform temperature | Closed-form solution | 5 |
| case_28d_confederation_sdl | Confederation Bridge long-term deflection under superimposed dead load | Independent reference solution | 10 |
| case_29_creep_bar | Creep bar | Closed-form solution | 5 |
| case_29b_relaxation | Stress relaxation (the dual of creep) | Closed-form solution | 6 |
| case_29c_shrinkage_bars | Shrinkage | Closed-form solution | 6 |
| case_29d_fiber_creep_bar | Creep in a fibre section | Closed-form solution | 6 |
| case_29e_fiber_relaxation | Relaxation in a fibre section | Closed-form solution | 6 |
| case_29f_tendon_relaxation | Relaxation of prestressing steel | Closed-form solution | 5 |
| case_29g_prestress_loss | Prestress loss (age-adjusted effective modulus method) | Closed-form solution | 5 |
| case_30_kuang_control_beam | Kuang 2023 uncorroded prestressed box-girder control beam | Published test/benchmark | 7 |
| case_31_li_control_beam | Li 2024 uncorroded prestressed box-girder control beam | Published test/benchmark | 8 |
| case_32_bearing_laws | Beam under temperature rise with a fixed bearing, a sliding bearing (Coulomb friction) and an expansion-joint restrainer (gap) | Closed-form solution | 8 |
| case_33_falling_beam | A beam whose hanger ropes are cut falls freely, hits the ground and comes to rest on it | Closed-form solution | 7 |
| case_34_brittle_cantilever | Single-section beam2 cantilever, breaking at the root at M_cap/L | Closed-form solution | 5 |
| case_35_spectrum_cantilever | Uniform cantilever column under a flat response spectrum: first four periods, effective masses and SRSS peaks | Closed-form solution | 8 |
| case_36_multi_support_wave | Deck mass between two piers, a travelling wave at 1000 m/s puts the two piers nearly in antiphase: peaks of deck and pier forces, supports following their records | Closed-form solution | 6 |
| case_37_pier_in_water | Circular pier in water carrying a deck on top, response spectrum with hydrodynamic added mass: periods, effective mass, pier-top displacement, base shear and moment; axial period unaffected by the water | Closed-form solution | 7 |
| case_38_sprung_mass_crossing | Sprung mass crossing a 25 m simply supported beam at 100 km/h: midspan deflection, body acceleration and contact force against an independent modal solution | Independent reference solution | 7 |

The comment at the head of each case file states where it comes from, what the model does, and how far the results differ from the reference.
