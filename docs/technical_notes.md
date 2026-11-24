# vfife: technical notes

What each feature is, why it is built the way it is, and what it was checked
against -- the long description for people who work ON the code or want the
reasoning behind a number. Until 1.12 this was the repository's README; the
[README](../README.md) is now the short front page, and the user manuals
([中文](用户手册.md), [English](user_manual.md)) are how to USE it.

Paths in the text (`cases/...`, `tests/...`, `benchmarks/...`) are from the
repository root.

## Install

```bash
git clone https://github.com/maolin2000/vfife.git
cd vfife
pip install -e ".[dev,plot,cpu]"      # NumPy + Numba CPU kernels; add ",gpu" for Warp/CUDA
python -m vfife --version
```

Python 3.10 or newer. `environment.yml` is the equivalent Conda definition
(`conda env create -f environment.yml`, then `conda activate vfife`), which
also installs Gmsh.

**What can a case file say?** [`docs/YAML参考.md`](YAML参考.md) is the
generated reference: every block, every field with its type, default and doc
comment, and the shipped case that uses it (`python -m vfife reference`;
a test keeps it in step with the code).

**New here?** [`docs/tutorial.md`](tutorial.md) walks one complete analysis
end to end — Gmsh mesh, YAML case, run, ParaView, and the hand check that tells
you whether to believe the answer. Every command and every number in it was run
to produce the text. About 15 minutes.

**中文用户手册**: [`docs/用户手册.md`](用户手册.md) — installation, a
complete post-tensioned beam from case file to result, every part of the case
file, the bridge features and the command line, in Chinese.
**User manual in English**: [`docs/user_manual.md`](user_manual.md) —
the same manual, section for section (the Chinese one is the
reference, and `tests/test_manuals.py` keeps their sections in step).

## Quick start

**From a mesh with no case beside it, three commands.** Each one names the
next, so nothing has to be known in advance:

```powershell
python -m vfife mesh beam.msh          # what is in the file: element type,
                                       # legal formulations, Physical groups
python -m vfife init beam.yaml --mesh beam.msh --fixed-region left --load-region tip --load 0 0 -5000
python -m vfife run beam.yaml
#   analysis      CONVERGED (residual 7.64e-07, analysis step 1 of 1)
#   u             final -6.623198e-06
#   R             final  5.000032e+03     <- against the 5000 N applied
python -m vfife plot results/beam.h5 --x u --y R      # the load-deflection curve -> results/beam.png
```

The scaffolded case carries a displacement monitor on the loaded region and a
reaction monitor on the support, in the direction the load acts, so that last
command works on a case nobody edited. `R` against the applied load is the
first sanity check of any model and it is free.

The rest of the command line, from the repository root:

```powershell
python -m vfife list                                  # registered materials/elements
python -m vfife describe beam2                        # parameters and capabilities
python -m vfife describe beam2_timoshenko             # shear-flexible 3-D frame member
python -m vfife describe connector2                   # six-component spring/dashpot/connector
python -m vfife capabilities --element hexa8 --backend warp
python -m vfife info cases/case_01_beam_cantilever.yaml # set up, print summary (and each member group's local axes), don't solve
python -m vfife schema                                # vfife.schema.json: completion and hover help in VS Code's YAML extension
python -m vfife check cases/case_01_beam_cantilever.yaml # model-quality JSON (and output.fields), no CUDA init
python -m vfife run  cases/case_01_beam_cantilever.yaml
python -m vfife verify cases/case_01_beam_cantilever.yaml # JSON + Markdown verification report
python -m vfife formfind cases/case_23_suspension_bridge.yaml --groups main_cable,hangers -o results/found.yaml  # cable lengths for the dead-load geometry
#   without -o it writes cases/case_23_suspension_bridge_found.yaml, a case with no contract beside it
python -m vfife sweep benchmarks/break_sweep_demo.yaml --result results/break_sweep_demo.h5  # fibre sections swept along the deformed members -> .pvd for ParaView (needs meshio)
python -m vfife view results/case_33_falling_beam.h5  # one self-contained web page that replays the result in 3-D: fields, breaks, monitors (three.js from a CDN)
python -m vfife summary results/case_33_falling_beam.h5  # one page: status, mass and weight against the reactions, the largest values and where, energy
python -m vfife buckling my_tower.yaml                  # linear buckling: the load factors and buckled shapes of a step's loads
python -m vfife modal cases/case_23_suspension_bridge.yaml --result results/case_23_suspension_bridge.h5  # natural frequencies at a state (FD tangent, physical mass)
python -m vfife modal cases/_xihoumen.yaml --modes 40 --paraview       # ... and the mode shapes as ParaView time series (one .pvd per mode, needs meshio)
python -m vfife init cases/my_part.yaml --mesh my_part.msh
python -m pytest -q                                   # default tier (slow solves deselected)
python -m pytest -q -m slow                           # the full-solve verification tier
python benchmarks/throughput.py                       # numpy vs numba, vs MATLAB
python benchmarks/gpu.py                              # GPU: correctness, throughput, 10^6 projection
```

`environment.yml` installs the editable project plus the development, Numba
CPU, Warp GPU and Gmsh dependencies; the `pip install` line under *Install*
is the same thing without Conda and without the GPU extra.

Results go to HDF5 with an XDMF sidecar — open the `.xdmf` in ParaView.  If an
older ParaView XDMF reader warns about a temporal `vtkMultiBlockDataSet`, open
the generated `_last.xdmf` sidecar instead; it is the final equilibrium frame
as one unstructured grid and references the same HDF5 file.

Requested field arrays such as `von_mises` are written at an early converged
relaxation frame as well as at a nominal final step, so a short successful run
does not lose its stress output. Without `output.fields` a run writes the
displacements, rotations and reactions, and the stresses or the members' signed
forces its model has (1.1; the manual's Section 12.1 lists them); ask for more
under `output.fields`. A `cdp` host also offers `dt` and `dc`, the
tensile and compressive damage variables on their own (Abaqus's DAMAGET and
DAMAGEC — monotone, the ones a crack map wants), `d`, the degradation the
stress actually carries (with recovery, so it falls where a crack closes), and
the equivalent plastic strains `ept`, `epc`. An unknown name under
`output.fields` is refused at run time.

Every numbered reference case has an independent contract in `verification/`.
`vfife verify` checks fields, monitors and convergence diagnostics without
feeding reference values back into the solver. Reports record backend, device
and precision and are written to `results/verification/` as JSON and Markdown.

A contract reads monitors, whole-model field extrema and diagnostics, so what
a monitor can reach is what a contract can assert. Besides `reaction` and
`displacement`, which read the nodes of a region, there are two that read the
model itself:

```yaml
  monitors:
    dc_span:   {kind: element, field: dc, reduce: max, region: span_block}
    sigma_bot: {kind: element, field: stress, component: xx, reduce: mean,
                region: span_soffit}
    sigma_p:   {kind: bar, of: tendons, field: axial_stress, reduce: mean,
                region: tendon_flat}
```

`element` reduces any per-element output of the material (`d`, `dt`, `dc`,
...) plus `stress` and `von_mises` over the elements a region owns — every
node of the element inside it, the rule a section assignment already uses —
and skips eroded ones. `bar` reduces `axial_stress`, `axial_force` or
`plastic_strain` over one reinforcement set, named by the `name` on its
assignment; a bar carries its own nodes, so its region test is geometric and
must be a `box`. Reductions are `max`, `min`, `mean`, `absmax`; the region may
be left out, meaning the whole model. An unknown field or a region that owns
no elements fails at setup, not with an empty series. A fibre beam adds
`fiber` (`{kind: fiber, region: node, field: strain | stress, at: [y, z]}`):
the plane-section strain, or the nearest fibre's stress, at a point of the
section at the station nearest the node -- a strain gauge on the deck.
`docs/tutorial.md` has the longer version.

Every run also embeds a unified termination status, per-frame equilibrium flag,
failure location and (on CPU paths) an energy ledger. A maximum-step exit is a
failed partial result rather than a silently accepted equilibrium solution; it
is preserved for diagnosis and rejected by `vfife verify`. See
[`docs/diagnostics.md`](diagnostics.md).

Frame members accept consistently integrated global/local uniform line loads,
including static condensation through rotational end releases, and a
`moving_load`: a group of axles that TRAVELS along a chain of members, either
as a crossing in physical time (`speed`) or as a quasi-static sweep whose
stations are an influence line (`stop`). The point force is distributed onto
the containing member by the element's own Hermite shape functions, so it is
exact at the nodes wherever between them the axle stands; see
[`docs/moving_load.md`](moving_load.md) and cases 26/26b, which hold it
against Fryba's closed form (0.3% over a crossing at alpha = 0.50) and against
the analytic influence line (1.3e-4). A `vehicle` is a
vehicle with mass: a rigid body (heave, and pitch on more than one axle) on
per-axle suspensions, unsprung masses and tyres that push and never pull,
riding the deck's deflection plus a road `roughness` (a profile file, or a
displacement PSD `g0 (n/n0)^-w` sampled with a seed -- no roughness class
table lives here). It is integrated with the structure by the same central
difference and step; an axle without a tyre rides the road rigidly (the
moving sprung mass). `kind: vehicle` monitors read its heave, pitch,
acceleration, contact forces, wheels and stations. `case_38` is Yang & Yau's
sprung mass crossing a 25 m beam at 100 km/h, held to an independent modal
solution (peak deflection 0.03%, the mass's acceleration 0.2%). A
`temperature` load is a free strain
rather than a force -- `uniform`, a `gradient` in K/m, or a code's nonlinear
`profile` sampled at each fibre -- so a member free to take it carries nothing
and reports nothing, a restrained one carries `EA alpha dT`, and a nonlinear
profile leaves self-equilibrated stress behind on a beam that nothing is
holding; see [`docs/temperature.md`](temperature.md) and cases 27/27b.
Concrete's `creep`, `shrinkage` and `ageing` are the same mechanism with a
clock attached: a step declares the concrete `age` it sits at or the `to_age`
it runs to, the stations are log-spaced because that is the axis creep happens
on, and the imposed strain is the superposition over the elastic increments
the member has actually taken. A determinate member then creeps and
redistributes nothing; a restrained one relaxes, to the closed-form
`R(t, t0)` and not to `1/(1 + phi)`. See [`docs/creep.md`](creep.md) and
cases 29/29b/29c. ParaView output
contains both member action envelopes and signed local i/j-end actions.
For short/deep members, `beam2_timoshenko` adds independent local-y/local-z
shear areas and correction factors while preserving the same VFIFE
co-rotational rigid-motion removal. See
[`cases/case_08_timoshenko_short_beam.yaml`](../cases/case_08_timoshenko_short_beam.yaml).

Frame members are evaluated for all elements at once: the
first version looped over them in Python and nobody noticed on one- and
two-member cases; forty members ran at 120 iterations/s and now run at 2700.
The same case, [`cases/case_16_lee_frame.yaml`](../cases/case_16_lee_frame.yaml)
— Lee's frame, NAFEMS NL7 — is the frame solver's first finite-rotation
benchmark: driven under displacement control through the limit point
(PL²/EI = 18.575 against NAFEMS's 18.552), a dynamic jump across the
snap-back, the negative-load region and the final rise (v/L = 0.779 where
NAFEMS reads 0.784). Three more things it made the frame solver do the way the
solid solver already did: normalise the residual by the internal force (under
displacement control the old `max(|F_ext|, 1 N)` was an absolute newton
count), honour a step's own `load_steps` and ramp a prescribed displacement
into each level, and rebuild the fictitious mass from the current geometry at
every level — a rotated member spreads its axial stiffness across global rows,
and a mass taken once from the reference stiffness stops bounding the time
step (the frame stalled at v/L = 0.13 at the default safety, at 0.5 with a
larger one; with the mass following the geometry it converges throughout).

Frame members are exact under finite rotations in space.
Until then a member's rigid rotation was removed by subtracting rotation
vectors, with the element frame following the chord alone — exact in a
plane, which is all the earlier benchmarks were, and wrong in space, where
rotation vectors do not add and a member whose nodes have twisted has a
frame the chord cannot see: a member rotated rigidly with a twist carried a
spurious moment. The frame now follows the mean nodal triad as well as the
chord and the local rotation is the relative rotation of the node's triad
(Crisfield 1990; `core/rotations.py`). Each structural point carries its
orientation as a rotation matrix, advanced multiplicatively by the step's spin
— R ← exp(ω dt) R, driven by the spatial moment directly, the
particle equation of Ting & Wu (2020) as written; the rotation vector log R is
derived from it for output, monitors and prescribed values. A held rotational
dof holds the spin component: a clamped point stays at the identity and a
symmetry-plane point (two components held) stays exactly in its one-parameter
subgroup. The case that found the kinematics defect is
[`cases/case_18_bend45_cantilever.yaml`](../cases/case_18_bend45_cantilever.yaml),
the 45° bend of Bathe & Bolourchi (1979, NAFEMS 3DNLG-5): a curved cantilever
loaded out of its plane, which bends about two axes and twists at once. The
old kinematics gave a tip of (−12.8, −24.2, 52.4) against the 1979 answer of
(−13.4, −23.5, 53.4); the new one gives (−13.64, −23.81, 53.64) with the
square section's own torsion constant, against Abaqus's (−13.74, −23.85,
53.63), and reproduces the Abaqus table at 300 and 450 as well. The planar
cases kept their numbers to round-off, as the reduction to the plane
predicts. The fibre beam (`beam2_fiber`) shares the orientation state and the
member frame: its numpy path and its numba kernel take the
local rotations from the points' triads, and a rigidly moved and twisted
member is force-free on both. Until then it rotated the global displacement
into the chord frame and left the Hermite functions to cancel the rigid part,
which they do to third order in the member's rotation; case_11 moved by
1e-5.

Supports and equipment links can use `connector2`: six local/global spring and
dashpot components, symmetric coupling matrices, and reversible piecewise-linear
skeletons. See [`docs/connector2.md`](connector2.md) and numbered case 09.

Nonlinear bars can use the M02 `piecewise_uniaxial` material with independent
tension/compression tables, isotropic/kinematic/combined hardening, reversal
history, loop-energy outputs and checkpoint-safe state. It remains explicitly
one-dimensional; see [`docs/piecewise_uniaxial.md`](piecewise_uniaxial.md).

Surface structures can use the E03 co-rotational `shell4`: homogeneous
Mindlin--Reissner membrane/bending/shear response, locking-resistant selective
integration, physically constrained drilling rotation, pressure/gravity/mass,
top/bottom stress output and conforming selected shell--line members. See
[`docs/shell4.md`](shell4.md) and numbered case 10.

`shell4` runs on all three backends. The element-at-a-time
loop was replaced by a batched evaluation (44× on 256 elements); `numba`
fuses that into one scalar pass per element, serial — on two cores `prange`
was 20× slower than the serial loop, because each element is a few
microseconds of work — and runs the 256-element roof at 2500 iterations/s
against numpy's 900; `warp` keeps the whole staged relaxation on the device
with the six-dof state viewed as a (2·nodes, 3) array, so the solid backend's
nodal integrator, boundary conditions and reductions are reused unchanged and
only six scalars per iteration cross the bus (kinetic energy, the four norms
of the convergence check, and the count of refused faces). `precision: mixed`
keeps the 24×24 local stiffness in fp32 and everything geometric in fp64. The
three are pinned to each other in `tests/test_shell_at_scale.py`: numba and
warp-fp64 equal the numpy force to 1e-9, warp-mixed to 1e-6, and the device
run converges at the same iteration as the host run with the same curve. A
node's local rotation is the relative rotation of its orientation with
respect to the face frame, log(R T_n R₀ᵀ), on all three;
until then it was the difference of two rotation vectors, exact only when
coaxial, and a rigidly rotated roof was not quite force-free.
The warp backend takes displacement constraints and loads under
`mode: relaxation`; ground motion, velocity/acceleration constraints and
`mode: dynamic` name the numba backend. `shell3` stays on the reference loop.

Measured on an RTX 5060 Ti (`benchmarks/shell_gpu.py`, one full iteration
of the staged relaxation, minimum of repeats): 256 elements numba 151 µs /
warp 367 µs; 1024 elements 711 / 425; 4096 elements 3583 / 532; 16384
elements 13469 / 1388 — the crossover is near a thousand elements, as it is
for the solid backend, and at 16k elements the card is still not full. The
floor is about 300 µs an iteration, nine launches and one synchronising
readback. case_17 (256 elements) is therefore a numba case; run on the
device it gives the same seven contract values to 1e-9.

The first benchmark that needed the speed is
[`cases/case_17_hinged_cylindrical_roof.yaml`](../cases/case_17_hinged_cylindrical_roof.yaml),
the hinged cylindrical roof of Sze, Liu & Lo (2004, their 3.8, 12.7 mm thick):
a quarter model on 256 `shell4`, pushed 30 mm at the crown through a full
snap-through under displacement control. The limit load comes out at
0.7485 P_max against their 0.7420 (0.9%), the load minimum at 0.191 against
0.173 (the descending branch is where the coarse quarter mesh shows), and the
final rise at 1.0224 against 1.0208; seven contract checks read the curve at
the six deflections of their Table 9a. Their pinched hemisphere (3.4, R/h =
250) is parked in `cases/scratch/`: the element is not the problem — its
linear regime is 94% of the MacNeal–Harder value on the mesh used — but the
dynamic relaxation of a shell that thin converges slowly, and the case is not
in the library until it converges within a budget a user would accept.

Conforming solid–beam–truss models assign a solid section to the volume and bar or
beam sections to the mesh's line groups (the `coupled` path). Shared Gmsh node
tags share translations exactly, and results are exposed as a solid `.xdmf` plus
`_members.xdmf`. See [`docs/coupled_shared_nodes.md`](coupled_shared_nodes.md)
and `cases/case_07_coupled_solid_truss.yaml` for the supported scope and the
important point-connection moment-transfer limitation.

Mesh input also accepts the engineering subset of Abaqus `.inp` (nodes,
supported solid/Line2 elements, NSET and ELSET). Applications and the future GUI
can call `vfife.api.prepare_case` and `vfife.api.run_case` without parsing CLI
text or intercepting process exits. See
[`docs/engineering_api.md`](engineering_api.md). The API is versioned
(`vfife.api.__all__`, version 1 since v1.0, frozen for 1.x), and a
parameter study is three lines:

```python
from vfife import api
case = api.load_case("cases/case_24g_hzb_static_pushover.yaml")
rows = api.study(case, {"sections.girder.j": [20.0, 29.5, 40.0]}, output_dir="results/j_study",
                 extract=lambda o: {"k_peak": 20 * (max(o.times) - 1)})
```

Every name, signature and docstring: [`docs/API参考.md`](API参考.md),
generated by `python -m vfife reference --api`. Every code a refusal or a
preflight issue carries, with the command line's exit codes:
[`docs/error_codes.md`](error_codes.md) (`python -m vfife reference
--codes`); the command line prints a refusal as `error [CODE]: …`.

## The model: materials, sections, assign

Five nouns and one rule. **Materials** are constitutive laws, by name.
**Sections** are geometry plus a material — the thing that gets assigned.
Geometry is the **mesh** (the meshed continuum) and **reinforcement** (bars
placed by drawing coordinates, located in whatever mesh is there). **Regions**
name node sets on the mesh. **Assign** is the only place a section meets
geometry. The rule that decides where a new feature goes: anything with
physics is a section; anything with a place is geometry or a region; the two
meet only in `assign`; anything between two things is an interaction
(reserved: `interactions`); anything kinematic is a constraint.

It is the Abaqus / OpenSees / Code_Aster structure — material, section,
assignment — with reinforcement treated the way DIANA and ANSYS treat it: as
its own geometry, independent of the mesh, so a change of mesh density
changes nothing about the bars.

```yaml
name: case_12_ualberta_sf0b
units: {length: m, force: N, stress: Pa}
mesh: {dimensions: [2.0, 0.2, 0.3], divisions: [24, 4, 8]}     # or {type: gmsh, path: beam.msh}

materials:
  concrete: {type: cdp, E: 2.985e10, nu: 0.2, rho: 2400, crack_band: both, l_ref: auto,
             compression: {source: gb50010, fc: 44.0e6},
             tension: {source: fracture_energy, ft: 3.42e6, Gf: 140}}
  steel15:  {type: rebar, E: 192.0e9, rho: 7850, yield_strength: 449.0e6, hardening: 8.0e9}
  steel10:  {type: rebar, E: 195.0e9, rho: 7850, yield_strength: 404.0e6, hardening: 7.0e9}

sections:
  body:  {type: solid, material: concrete, formulation: b-bar}
  bar15: {type: bar, material: steel15, shape: circle, diameter: 0.016, area: 200.0e-6}
  bar10: {type: bar, material: steel10, shape: circle, diameter: 0.0113, area: 100.0e-6}

reinforcement:                     # geometry only, read the way a drawing reads
  main:     {layer: {from: [0, 0.0443, 0.0443], to: [2.0, 0.0443, 0.0443], count: 3, spacing: [0, 0.0557, 0]}}
  main_top: {layer: {from: [0, 0.0443, 0.0853], to: [2.0, 0.0443, 0.0853], count: 2, spacing: [0, 0.1114, 0]}}
  hangers:  {layer: {from: [0, 0.0443, 0.258],  to: [2.0, 0.0443, 0.258],  count: 2, spacing: [0, 0.1114, 0]}}
  stirrups: {hoop: {corners: [[0.0443, 0.0443], [0.1557, 0.258]], axis: x, from: 0.05, to: 1.95, spacing: 0.075, divisions: 8}}

assign:                            # a lone solid section goes to `all` by itself
  - {reinforcement: [main, main_top], section: bar15, segment_length: 0.025, name: tension}
  - {reinforcement: hangers,  section: bar10, segment_length: 0.025}
  - {reinforcement: stirrups, section: bar10}
```

Section types and what they take:

| type | needs | gives the solver |
|---|---|---|
| `solid` | `material`; optional `formulation` (b-bar, full, reduced, selective), `hourglass_coeff` | the continuum; the element type comes from the mesh |
| `shell` | `material`, `thickness`; optional `shear_correction`, `drilling_factor`, `warpage_tolerance` | `shell4` |
| `beam` | `material`, a `shape` or `area, iy, iz, j`; `theory: timoshenko` adds `ay, az` | `beam2` / `beam2_timoshenko` |
| `bar` | `material`, `shape: circle` with `diameter` (perimeter for bond) or `shape: general` with `area` | `truss2` on a line region; an embedded bar on `reinforcement` |
| `fiber` | `material` (cover), `shape: rect` (`width, height, ny, nz`), `tube` (`diameter, t, nt`), `box` (`width, height, t, ny, nz`) or `patches` (`patches: [{corners: [[y, z] x 4], ny, nz}]`, any quadrilateral assembly -- a box girder); optional `cover`, `core_material`, `integration`, `j` (the torsion constant), `fracture`, `bars: [{y, z, section, count?}]`, `tendons: [{section, profile, extent?, prestress | jack + friction}]` (prestressing, v0.3 and v0.4); several fibre sections may share one model, each on its own region (docs/fiber_beam.md) | `beam2_fiber` |
| `connector` | `stiffness` and/or `connector_damping`, `curves` | `connector2` |

Shapes: `circle` (`diameter`, optional wall `t`), `rect` (`width`, `height`,
optional `t` for a box), `I` (`bf`, `tf`, `hw`, `tw`), `general` (the
integrated values themselves). Any integrated value given explicitly overrides
the derived one, so a better torsion constant does not force `general`.
`vfife/sections.py` is the one place the arithmetic lives.

Reinforcement primitives: `line` (start, end, optional `repeat`), `layer` (a
row of `count` parallel bars, OpenSees' word), `hoop` (a closed rectangular tie
repeated along an axis), `lines` (a plain list). `divisions` or
`segment_length` go on the primitive or on the assignment; `path` (a Gmsh
curve as a bar route) is reserved.

An assignment targets one of `region` (a mesh node set: the whole solid, a
Gmsh line group) or `reinforcement` (one or more bar sets). Options ride on
the assignment because they belong to the pairing, not to either side:
`orientation`, `releases`, `coordinate_system` for beams and connectors;
`prestress`, `unstressed_length` or per-element `unstressed_lengths` (what
`vfife formfind` writes), and `tension_only`, for a bar on a line region
that is a cable (docs/structural_framework.md); `bond`,
`segment_length`, `transverse_penalty`, `host_deduction`, `anchorage`
(`ends` / `start` / `end`, bonded bars) for embedded bars; `anchorage:
{region, bar_diameter}` for a fiber beam's strain penetration.

Several solid sections on one mesh are material blocks. An element belongs to
the section whose region holds **every one of its nodes** — the rule that lets a
Gmsh physical volume select exactly its own elements — and assignments apply in
order, a later one taking what an earlier one also covers (Abaqus' rule), so the
background goes first and the local sections after it:

```yaml
sections:
  cover:    {type: solid, material: concrete}
  core:     {type: solid, material: confined}            # a second cdp, higher fc and strain
  plate:    {type: solid, material: steel_elastic}
regions:
  core_box: {box: [0.0, 2.0, 0.0443, 0.1557, 0.0443, 0.258]}
  pad_L:    {box: [0.35, 0.45, 0.0, 0.2, 0.2625, 0.3]}    # the top element layer under the load
assign:
  - {region: all,      section: cover}
  - {region: core_box, section: core}
  - {region: pad_L,    section: plate}
```

Each block runs its own material and formulation; the fused (numba) and
device-resident (warp) backends take several blocks, on one node set. An
element no assignment covers, or a section that takes no whole element, is
refused with the element ids (`ELEMENTS_UNASSIGNED`, `SECTION_EMPTY`), and
preflight lists every section's element count. Output fields are the union
over blocks — an elastic plate reports zero damage, not a missing field — and
`block` says which section each element is in, for ParaView's threshold.

A line section names its own material, so one model can carry a steel girder,
a concrete tower and two cable steels at once; the first one is the model's
and every other travels down on its member group.

Limits, refused with a message rather than silently narrowed: a shell section
is the whole mesh, a coupled model (solid + line members) carries one solid
material, a line member's material must be `linear_elastic`, no `path`.

Files in the older form (`element:` / `material:` / `reinforcement:` /
`members:`) are refused at load; `vfife migrate old.yaml -o new.yaml` rewrites
one and checks that the result lowers back to the same model before writing.

## Backends

One contract, three implementations:

```yaml
solver:
  backend: warp        # numpy | numba | warp
  device: cuda         # warp only: cuda, cuda:0, cpu
  precision: mixed     # warp only: mixed (fp32 elements) | fp64 (verification)
```

- **`numpy`** — batched tensor operations. The readable reference, and where the
  physics is verified (rigid rotation, patch test, closed-form constitutive
  solutions).
- **`numba`** — the same algebra fused into one scalar pass per element: gather,
  polar decomposition, strain, constitutive update, internal force, rotate back,
  with nothing intermediate reaching memory.
- **`warp`** — that same kernel on a GPU, with the whole time loop resident on
  the device.

`test_backends.py` runs all three through the same load histories on distorted
meshes and requires they agree to **1e-12** on force, stress, rotation and
material state, for every formulation and every material. That is what lets the
fast paths be trusted: the reference stays honest and the kernels stay correct.

Each material carries all three implementations, side by side, and the test
above fails if they drift.

Measured on **2 cores** (force evaluation + nodal assembly, per step):

| elements | formulation | material | numpy | numba | gain | vs MATLAB |
|---|---|---|---|---|---|---|
| 19 200 | reduced | linear elastic | 24.3 ms | **2.6 ms** | 9.5× | 1391× |
| 19 200 | reduced | chaboche | 27.8 ms | **3.0 ms** | 9.4× | 1212× |
| 19 200 | reduced | cdp | 61.1 ms | **7.5 ms** | 8.2× | 478× |
| 19 200 | b-bar | cdp | 460 ms | **44.8 ms** | 10.3× | 80× |

`prange` gives 4× on 2 cores, so expect this to keep scaling on your machine.

**Two things had to change together.** Fusing the element loop alone gave ~6×;
the assembly then became 44% of a reduced-integration step, because `np.bincount`
stood still while everything around it got faster. Replacing it with a
gather over the inverted connectivity — each node sums its own contributions,
no atomics, fixed index order — made it 5.7× faster and bit-for-bit
reproducible, which is also exactly what the GPU backend needs.

**One-time cost:** ~4–7 s of JIT per (material, formulation) pair on the first
call. Numba's disk cache does not help, because the kernel closes over a freshly
built material device function. On the CDP beam case that is 7 s spent against
47 s saved.

## The GPU backend

**Measured on an RTX 5060 Ti** (Blackwell, sm_120, 16 GiB), 10^6 elements,
reduced integration, mixed precision, measured directly — not extrapolated:

| material | ms/step | M element-steps/s | vs the MATLAB kernel | GB | 70 000 steps |
|---|---|---|---|---|---|
| linear elastic | 2.77 | 361 | 67 000× | 0.71 | **3.2 min** |
| Chaboche | 3.06 | 327 | 61 000× | 0.79 | **3.6 min** |
| CDP | 3.51 | 285 | 53 000× | 0.77 | **4.1 min** |

The same job — 10^6 elements for 70 000 steps — is 7×10^10 element-steps. At the
original MATLAB kernel's rate (5 372 element-steps/s, from its own 659 s run)
that is **151 days**.

3.51 ms/step for CDP lands at the fast end of the 3–8 ms the plan's §05 predicted
from a pure bandwidth argument, before any of this existed.

Agreement with the NumPy reference in fp64 across the six GPU verification
material–formulation combinations: **2.5e-14 to 6.5e-12** (Chaboche is the loose one — three Newton
iterations over `exp()`, whose last ulp differs between libm and the GPU's
transcendentals).

**Do not size a GPU run from a small mesh.** At 19 200 elements the card is
idle — one thread per element is ~4 waves on 4608 cores — and every material
comes out at the same time regardless of arithmetic cost (elastic 0.130 ms,
Chaboche 0.131, CDP 0.126). Extrapolating 10^6 from there predicted 6.6 ms/step
against 3.51 measured: **1.9× pessimistic**. `benchmarks/gpu.py` prints a
throughput-vs-size column that says `still climbing` until the card is full, and
`--million` measures the real thing.

**Mixed precision.** Element data, material state and the element kernel in
fp32; nodal displacement, velocity and assembled force in fp64. A consumer
GeForce runs fp64 at 1/64 of fp32, so a double-precision element kernel throws
away 98% of the card — but the nodal integration is a tiny fraction of the
arithmetic and bandwidth-bound anyway, and it is where precision actually
matters, because a 70 000-step run accumulates displacement increments that are
small next to the coordinates they sit on. Measured cost: about four significant
digits of stress, far more than the constitutive parameters are known to.

**And it has a resolution that depends on how long your model is.** The element
REFERENCE COORDINATES are in fp32 too, and a segment vector is a difference of
two of them, so what survives the cancellation scales with (model extent) /
(element size) — a number nobody thinks about when choosing a mesh. On the
post-tensioned T-beam of `cases/case_15`, 8.5 m long with 50 mm elements, that
ratio is 170 and it leaves 1.5e-5 of strain error on the tendon, an eighth of
the cracking strain: at 195 GPa, 2.9 MPa of stress noise. Relaxation cannot
remove it, because it is not out of balance — it is the arithmetic. The symptom
is kinetic damping resetting every few iterations and a residual that stops
falling: that run spent 150 000 iterations with 44 581 resets and ended at a
relative residual of 2.2e-1, where the same model reaches 1e-5 in 34 000 on the
CPU. At `precision: fp64` it converged.

Preflight computes this before the run (`MIXED_PRECISION_RESOLUTION`), names
whether the mesh or which bar set carries it, and warns above 5% of the
material's cracking strain. So `precision: fp64` is not only for verification:
it is what a long model with small elements needs.

**The time loop is on the device, and that is not an optimisation.** With
displacement, velocity, material state and element force copied across the bus
every step, a 10^6-element CDP case moves about **280 MB per iteration** — on
PCIe 5.0 x8 that is ~14 ms against a 3–8 ms kernel, so the transfers would cost
more than the compute they exist to feed. So the integrator and boundary
conditions run there too, and the host sees only scalars: kinetic energy each
step for the damping decision, residual norms at progress intervals. Fields come
back when a frame is written.

That means a second implementation of the integrator, so it has its own
equivalence test against the CPU loop — including a displacement-controlled case,
because the load-controlled one never touches the prescribed-BC path, and that
path had the device kernels applying the value before zeroing the constrained
dofs. The symptom was a run where nothing moved at all.

**Device memory**, measured, mixed precision:

| formulation | material | B/element | at 10^6 elements |
|---|---|---|---|
| reduced | linear elastic | 613 | **0.61 GB** |
| reduced | cdp | 669 | **0.67 GB** |
| b-bar | cdp | 1 933 | 1.93 GB |

Comfortable on a 16 GB card either way. Two arrays the CPU backends cache for
locality — the transposed `dNdx` and the centroid operator — are not uploaded at
all; on a device they are a second full-size array to hold *and stream*, and
indexing the original transposed costs nothing.

**First-run compile:** Warp compiles one module per (material, formulation,
precision) — tens of seconds — then caches it under `~/.cache/warp`.

## Measured against the MATLAB original

Same mesh (600 elements, 1098 nodes), same `b-bar` formulation, same problem,
on a 2-core cloud container (your 5060 Ti machine has more cores, so treat
these as a floor):

| | steps | wall time |
|---|---|---|
| MATLAB `solver.m` (its own damage model) | 5 900 | **659 s** |
| `vfife` dynamic mode, numpy backend | 6 211 | **25.7 s** |
| `vfife` relaxation mode, numpy backend | 22 000 | 91.8 s |

The same case with the CDP material (a heavier model than the original's, and a
correct one) runs in **60.3 s** on the numpy backend and **12.7 s** on numba.

**But the more important number is the residual.** The dynamic run — the one
that mirrors the original's settings — finishes with an out-of-balance residual
of **0.96**, i.e. inertia and damping forces are still the same order as the
internal forces. It never reaches equilibrium. The relaxation run converges to
`8e-7`, and the two disagree on the support reaction by **19%**.

So the original 659 s was not only slow, it was not producing a static answer.
Fixed viscous damping at `zeta = 100` over 0.2 s of physical time leaves the
beam ringing. For a quasi-static problem use `mode: relaxation`.

## What changed, and why

### Structure

**Cold/hot split.** `elements/hexa8.py::precompute()` runs once and stores
`dN/dx`, `w*detJ`, the centroid operator and the hourglass base vectors. The
original recomputed all of it every step for every element and every Gauss
point, though it depends only on the reference configuration.

**No B matrix in the time loop.** Working with the displacement gradient
instead of a 6×24 matrix is algebraically identical and removes most of the
arithmetic:

```
grad[i,j] = dN/dx[i,k] q[k,j]                    # 3x3
eps       = [g00, g11, g22, g01+g10, g12+g21, g20+g02]
f[k,j]    = w detJ dN/dx[i,k] T[i,j]             # T = stress as a 3x3 tensor
```

A 6×24 B matrix is built in exactly one place, `stiffness_row_sums`, which runs
once when Underwood relaxation masses are requested.

**Flat state arrays.** `state[n_points, n_state]` replaces
`state_vars{i,j}.backstress_1`. This is the change that makes everything else
possible: batch vectorisation, a block copy to GPU memory, checkpointing as one
HDF5 write, and one indexing scheme shared by NumPy, Numba and (next) Warp.

**Warm-started quaternion polar decomposition** instead of SVD, with a 3/2
over-relaxation that turns Müller's linear iteration (ratio 1/3) into a Newton
step for near-spherical stretch — two warm iterations leave an error of 0.08%
of the per-step rotation increment, against 11% without it.

**Named regions**, resolved to node indices once at setup. The original called
`find_nodes_in_region` — a full sweep over every node — inside the time loop,
for every constraint and every load, on all 5 900 steps.

**Leapfrog integration on displacement**, not absolute coordinates. On a 12 m
beam with 0.08 m of displacement the original lost about two significant digits
to cancellation on every step; that becomes fatal in fp32.

**Adaptive dynamic relaxation** (`mode: relaxation`): fictitious Underwood
masses with `dt = 1`, plus kinetic damping. No damping coefficient to tune. On
the cantilever benchmark it reaches the answer in 14 800 steps against 30 300
for the best of four hand-tuned viscous coefficients — and one decade away from
that best value, viscous damping does not converge at all. `mass: eigenvalue`
is the rotation-invariant alternative to the Underwood row sums: each element
gives its nodes λ_max(K_e)/4 (plus the hourglass stiffness of a reduced
element), which by Hughes' element-eigenvalue inequality bounds the stiffest
assembled mode in every configuration, where the row sums only measure the
reference one — at the price of a step typically 1.5–2 times smaller. It was
written on suspicion that the row sums had stalled the rolled cantilever
(case_20); they had not (its stiffest mode sat at half the limit), and it is
kept as the guarantee it is.

### Physics fixed

Each of these has a test that fails if it regresses.

**P1 — Voigt convention** (`materials/chaboche.py`). The flow rule produces a
*tensor* strain increment; the original stored it straight into `epsilon_p_vec`,
which is then subtracted from an *engineering* total strain and multiplied by an
engineering-shear elasticity matrix. Plastic shear was off by a factor of two.
Masked in the beam case because the plasticity there is mostly axial.
→ `test_chaboche.py::test_voigt_convention_engineering_shear`

**P2 — plastic multiplier** (`materials/chaboche.py`). The denominator was
`E/(1+nu) + C1 + C2 + H'` = `2G + ...`; consistency gives
`3G + sum_i (C_i - gamma_i X_i:n) + H'`. Both the `3G` and the
`gamma_i (X_i:n)` term were wrong or missing, and only one linearisation step
was taken. Now a proper Newton solve on the scalar residual, `n_newton`
configurable.
→ `test_chaboche.py::test_plastic_multiplier_denominator_is_3G`

**P3 — damage applied twice** (`materials/chaboche.py`). The elastic predictor
was multiplied by `(1 - D_prev)` *and* the final stress by `(1 - D_new)`, giving
`(1-D)^2` degradation. Yield is now checked on the effective (undamaged) stress
and the Cauchy stress is `(1-D) sigma_eff`, once.
→ `test_chaboche.py::test_damage_applied_once`

**P4 — crack-band softening modulus** (the old `material_damage.m`). It used
`H = 2 E L_c / (Gf - 0.5 ft eps0 L_c)`; the derivation gives `H = ft L_c / (...)`.
Since `ft = E eps0`, the original was a factor `2/eps0` too large — with the
beam's own numbers the softening branch dissipated about 1/20000 of the intended
energy, and `Gf` never entered the answer. That model has since been deleted and
replaced by CDP, whose crack band is tested directly.
→ `test_cdp.py::test_crack_band_makes_softening_mesh_independent`

**P5 — hourglass stiffness from the bulk modulus** (`elements/hexa8.py`). Scaling
the hourglass stiffness with `rho c^2` (the P-wave modulus) makes it diverge as
`nu -> 1/2`. Measured on a cantilever at `nu = 0.4999`, the tip deflection came
out 64× too small: hourglass control was reintroducing exactly the locking that
reduced integration exists to remove. Now scaled with `G`.

**P6 — hourglass stiffness proportional to 1/dt** (`elements/hexa8.py`). The
original's `hg_stiffness = alpha_h/dt` meant that changing the time step — or
mass scaling — silently changed the hourglass stiffness, i.e. changed the
answer. Now `dt`-independent.

**Hourglass base vectors** are now the four true hex8 modes with the
Flanagan–Belytschko projection `gamma_a = h_a - (h_a . x_j) dN/dx_j`. The
original used three of four vectors, unprojected.

**Viscous (Duvaut–Lions) regularisation** on both damage models, kept as a
separate `d_bar` state variable so the rate-independent solution is untouched —
folding the viscosity back into `d_bar` merely rescales the damage *rate* and
does not restore mesh objectivity.

**True characteristic length** `V/A_max` per element for the CFL limit, instead
of the nominal `L/n` — unsafe on distorted meshes, over-conservative on regular
ones.

## How short a case file is

`cases/case_12_ualberta_sf0b.yaml` — a 2 m RC beam with five main bars, two
hangers, 26 stirrups, four-point loading, three monitors — is 77 lines, and
every line is an engineering decision: where the bars are, where the supports
are, how far to push. Nothing in it is there to satisfy the solver. Three
things keep it that way: a bare number in `where` names a support line
without a bounding box; the preflight names any rigid-body mode still free
and prints the constraint that removes it; a staged step's iteration budget is
derived rather than copied. When the solver refuses a file it says what it
needs; when it snaps a coordinate it says which.

## Boundary conditions

Three pieces: regions say *where*, amplitudes say *how it varies*, steps say
*in what order*.

### Regions are composable

```yaml
regions:
  base:      {plane: {point: [0, 0, 0], normal: [0, 0, 1]}}
  support:   {where: {x: 0.10, z: 0}}                  # a line: two axes pinned
  pin:       {where: {x: 0.10, y: 0.10, z: 0}}         # a point: three
  web:       {where: {y: {min: -0.05, max: 0.05}}}     # a slab: a range on any subset of axes
  diagonal:  {line: {from: [0, 0, 0], to: [2, 0, 0.3]}}
  bolts:     {cylinder: {point: [1, 0.5, 0], axis: [0, 0, 1], radius: 0.05}}
  shear:     {intersect: [base, web]}
  ends:      {union: [left_face, right_face]}          # gmsh physical groups
  interior:  {difference: [all, surface]}
```

Also `sphere`, `box`, `group`, `node_ids`, `nearest`, and `all`. `all` and
`surface` exist without being declared, and order does not matter — references
resolve by dependency, so a region can be built from parts declared below it. A
region that selects nothing is an error, because a tolerance slightly too tight
selects slightly the wrong nodes and an almost-right support looks exactly like
a right one until you read the reactions. Tolerances default to 1e-6 of the mesh
diagonal, so the same file works in millimetres and in metres.

A bare number in `where` pins that axis to a **node line**. A support at
`x: 0.10` on a mesh whose nearest node line is at 0.0833 snaps to that line and
the preflight report says so (`REGION_SNAPPED`, with the coordinate actually
used). That is what every beam case needs and what a six-number `box` with a
hand-tuned ±26 mm used to do silently.

### A physical group selects elements, not just nodes

A region has always been a set of nodes, and an element belonged to it when
all of its nodes did. On the first real building this library ran — an eleven
storey frame-and-wall block imported from an OpenSees model, 286 columns, 473
beams and 55 walls — that rule cannot say which member is which: **every
column's two nodes are also beam nodes**, so both groups claim every column
and the model is refused as over-assigned. A Gmsh physical group knows exactly
which elements it holds, and since v0.1 the reader keeps that
(`mesh.element_regions`) instead of throwing it away. Regions written as a box
or a plane have no elements of their own and keep the node rule, so nothing
that worked before depends on the change.

### Rigid floors

```yaml
diaphragms:
- {region: floor_1, normal: z}
```

The nodes of that region move together as one rigid body **in the plane
normal to `normal`** — two translations and the rotation about the normal.
The out-of-plane translation and the two bending rotations stay free, which is
what makes it a diaphragm and not a rigid body: the floor may still deflect and
its beams may still bend. It is how a slab that is not meshed is written down,
and it is what an OpenSees model means by `rigidDiaphragm`.

In an explicit particle scheme the constraint is applied where it is cheapest
and exact — to the **acceleration**. The floor's nodal forces are condensed
onto the three degrees of freedom a rigid floor really has, those three are
accelerated by the floor's own mass and polar inertia, and the nodal
accelerations are read back off the rigid motion. Velocities and positions stay
consistent for ever after, because they start consistent and every increment
they receive is consistent. `tests/test_diaphragm.py` asserts the four things
that makes it: the floor moves rigidly in plane, it does **not** out of plane,
the condensation conserves the force and torque it condenses, and a motion that
is already rigid passes through unchanged.

A floor that is both held by a support and moved by a diaphragm is refused —
by `vfife check` before anything is assembled, and by the solver at setup.

### Amplitudes are per boundary condition

```yaml
amplitudes:
  drift: {cyclic: {peaks: [0.333, 0.667, 1.0], repeats: 2}}
```

Built in: `ramp` (0→1), `hold` (already on), and `ramp_hold` — on over the
first half of the step, then steady — which is the default. The abscissa is the
step's own progress from 0 to 1, not physical time: a relaxation run has no
physical time, so a curve written against seconds could not be shared between a
relaxation and a dynamic run of the same structure. Against progress it can.

The default differs by step type, and has to. A **staged** step (`load_steps`)
holds each level until it converges, so its stages should sample the loading
evenly — `ramp`, stage k of N at k/N. Give a staged step `ramp_hold` and the
load reaches full at the halfway stage and sits there, turning the second half
of the load-deflection curve into the same point measured repeatedly. A
**continuous** step needs the opposite: with `ramp` the load is still moving on
the last iteration, so equilibrium can never honestly be declared. The
convergence test waits for every amplitude to settle, not merely for the step
to end.

### Steps run in sequence and share the model

```yaml
constraints: [{region: base, dofs: [1, 1, 1]}]
loads:       [{region: head, vector: [0, 0, -800.0e+3]}]   # the base

steps:
  - name: axial                      # 800 kN ramps on here
  - name: cyclic                     # ...and stays on through this
    constraints:
      - {region: head, dofs: [1, 0, 0], value: 0.010, amplitude: drift}
    load_steps: 8
```

For an explicit surface traction, supply its vector in Pa.  VFIFE identifies
the exterior faces wholly contained in the region and integrates the load with
that face's interpolation functions; it does not divide the traction by the
count of surface nodes:

```yaml
loads:
  - {region: upper_surface, traction: [0, 0, -1.0e6]}  # 1 MPa downward
  - {region: upper_surface, pressure: 1.0e6}           # 1 MPa inward
```

Six-component cell stress is written to XDMF as `Tensor6` in the order
`[xx, yy, zz, xy, yz, xz]`, so ParaView can probe or calculate individual
components after **Cell Data to Point Data**.

### General loads and time amplitudes

Every load has a `region` (default `all`) and may reference a named amplitude.
Exactly one loading quantity is supplied:

```yaml
amplitudes:
  service: {table: [[0.0, 0.0], [0.2, 1.0], [0.8, 1.0], [1.0, 0.0]]}
loads:
  - {region: lifting_lug, vector: [0, 0, -25000], amplitude: service}
  - {region: side_face, traction: [0, 0, -1.0e6], amplitude: service}
  - {region: upper_face, pressure: 1.0e6, amplitude: service}
  - {region: bulk, body_force: [0, 0, -24000], amplitude: service}
  - {region: bulk, gravity: [0, 0, -9.80665], amplitude: service}
```

`vector` is a total nodal resultant (N); `traction` is an explicit vector in
N/m²; positive `pressure` acts inward in Pa; `body_force` is N/m³; and
`gravity` is m/s² multiplied by material density. Surface loads integrate over
actual exterior faces and volume loads over element volumes, independent of
node density. A pressure is integrated once over the reference faces, which
fixes its direction and area for the analysis; `follower: true` re-integrates
it over the faces at their current positions every step, so it stays normal
to the surface it acts on (solids, numpy/numba). On a body that turns the
two are different loads: the end pulses of the floating beam below spin it at
126 rad/s as follower pressures and at 365 as fixed-direction forces, whose
lever arm about the centre grows with L/2 sin θ. `table` abscissae are the step's normalised progress from 0 to 1:
each staged-relaxation point is equilibrated, and a dynamic step maps it
monotonically over its duration. Use the existing `cyclic` generator for
reversed histories.

In dynamic analyses, `constraints.motion` may be `displacement` (the default),
`velocity`, or `acceleration`; the latter two integrate the specified
`value * amplitude` in time. A `ground_motion` load reads a physical
two-column acceleration record and applies the relative-coordinate inertia
force `-M a_g(t)` on CPU backends for solids and standalone trusses.

**Implicit static solver**. `solver.mode: static` finds each
load level's equilibrium by Newton's method instead of by dynamic relaxation.
The tangent is a finite difference of the same internal force the explicit
solvers use -- no element stiffness written twice -- and costs 2 x (dofs per
node) x (colours) evaluations, the nodes grouped by a distance-2 colouring so
that the ones sharing no element and no neighbour move together (36-100
evaluations for a frame, whatever its size). Each level starts from a tangent
predictor, Newton runs with a backtracking line search, and a level that does
not converge is halved and retried, the frames still landing on the levels
asked for. Path-dependent sections are evaluated in trial mode and committed
once per level. Frames and fibre beams; `case_21c` is Vecchio & Emara's frame
solved this way in 8 s (22 minutes by relaxation) against case_21's contract.
With `static.arc_length` the load factor becomes an unknown
and Crisfield's cylindrical arc length carries the path over limit points,
snap-throughs and snap-backs; the step ends at the full load, at a `stop`
displacement, or after `max_increments`, and the monitors' `time` is the load
factor. `case_16b` traces Lee's frame end to end in 61 increments (limit
load 18.577 EI/L^2 against NAFEMS's 18.552; v/L = 0.781 at the second point,
Abaqus's value, against NAFEMS's 0.784). `arc_length.steps`
names the steps that take it, the rest going by load control: `case_24g`
settles the Hangzhou Bay unit's dead load and prestress by one Newton level
and pushes the live load by arc length through the peak (k = 14.69 at
-0.186 m) onto the softening branch (k = 12.76) and up to the stop at
-0.38 m (k = 14.08) in 18 arc-length increments and about 30 s, where
case_24d's explicit ramp takes 50 minutes and can only jump the dip. Its
dead state is -23.47 mm (-23.18 before v1.0's anchor-set fix,
which the relaxation matched to 3e-6); case_24d's -27.77 is what its t = 0 load's
overshoot cracked. Since v1.7 (1.7) the static solver takes rigid
`diaphragms` (a transformation of the Newton system to the floors' master
dofs) and member `failure` (a member that reaches its capacity in an accepted
equilibrium is taken out and the level solved again, the alternate-path
analysis one member at a time), and `static.fallback: relaxation` lets a
level Newton cannot take -- a limit point under load control, a jump in the
path -- be relaxed from the last converged state with the staged relaxation's
own iteration, the rest of the step with it; past a softening jump the branch
it lands on is the fictitious transient's, so the reinforced-concrete
pushovers keep `mode: relaxation`, while case_24, 28, 28b and 28c run static
(case_28: 102 s → 1.4 s).

**Threads**. `solver.threads` (1 by default, 0 for every
core) runs the numba fibre-beam kernel's elements in parallel. Each element
writes its own row and the rows are summed in element order afterwards, so
the answer does not depend on the thread count, to the bit. The Hangzhou Bay
seismic run (`_hzb_seismic`, 31.18 s of El Centro) on a 16-core Core Ultra
X9 388H: 22.0 min serial, 11.4 with 4 threads, 9.0 with 8, and 10.9 with all
16 -- about two thirds of a step is the fibre kernel (the bearing
connectors and the integration stay serial), and on a hybrid CPU the extra
threads wait for the slowest cores. Measure first:
`python benchmarks/bench_threads.py <case> --time 1.0 --threads 1 4 8 0`.
Since v1.0 the frame (beam2) kernel is threaded the same way, the
members cut into contiguous runs that allocate their scratch once, and the
axial members (`truss2`) under `backend: numba` are one compiled loop instead
of temporaries and two `np.add.at` scatters. On the Xihoumen model (851
beams, 356 cables and hangers) a simulated second goes from 42.5 to 30.5 s
on two cores; the rest of a step, about 0.3 ms of vector updates, rotation
updates and the energy ledger, was still serial. Since v1.3 (1.3)
the whole explicit step of a frame or frame-and-cable model is one compiled
call between two progress or output points -- loads, member forces,
damping, ledger, velocities, positions, orientations, supports, the points
split between the threads as the members are -- and gives the step-by-step
loop's answer to the bit: on the same two cores a simulated second of
Xihoumen takes 14.9 s (27.5 s step by step). Since v1.4 the
connectors and bearings are compiled too (laws, dashpots, slip history), and
fibre-beam models and frame models on bearings take the compiled step, to
the bit as well: a step of the Hangzhou Bay seismic model (fibre beams on two
pot bearings, two cores) goes from 2.33 to 0.88 ms. A fibre run under 20000
steps keeps the loop, because compiling its step takes about 3 s per
process (`VFIFE_COMPILED_STEP=1` forces it). Since v1.7 (1.6)
prescribed motions, plastic hinges, breaking and splitting, moving loads and
vehicles take the compiled step too: each is one compiled function that the
loop and the compiled step both call, so the two agree to the bit by
construction; a member reaching its capacity is an event the loop takes
before the compiled step is rebuilt on what is left (Xihoumen with its
hangers removed and failing, 43.6 → 29.0 s on one thread; case_34 12.1 →
1.8 s). Since v1.12 (1.12) the contact between members
(`contact.members`) is in the compiled step as well, frames and fibre beams
(Xihoumen with every member a contact body 20.4 → 9.8 s). What keeps the
step-by-step loop: a prescribed rotation, rigid-surface contact,
temperatures and creep, rigid diaphragms, soil springs,
members born in a step, analysis steps, and a script's own callable in a
load group; `VFIFE_COMPILED_STEP=0` turns the compiled step off.
Since v1.9 (1.9) the frames' staged relaxation is compiled the
same way: the iterations of a level (and of the static solver's relaxation
fallback) run in one compiled function that returns at every progress
check, to the bit (Lee's frame 87 → 27 s); fibre models keep their loop
and are sped up by `solver.threads`.

**Added mass**. `nodal_masses` may be directional
(`components: [1, 1, 0]`), weightless (`weight: false`, the default for a
directional mass) and rotary (`inertia: [Ix, Iy, Iz]`, beams and shells);
`hydrodynamic_mass` gives the members of a region below `water_level` the
Morison added mass C_a rho_w A per metre, across the member and weightless
(piers and piles: members within 5 degrees of a global axis). `case_37` is a
pier in 20 m of water under a response spectrum, held to the closed form.

**Multiple-support excitation**. A constraint with a
`record` makes its region follow its own ground motion, in absolute
coordinates, so that every support of a long bridge can move differently and
the members between them are strained by the difference:

```yaml
constraints:
  - {region: piers, dofs: [1, 1, 1, 1, 1, 1]}     # clamped as usual ...
  - region: piers                                 # ... and x follows the record
    dofs: [1, 0, 0, 0, 0, 0]
    record:
      path: ground_x.dat                          # two columns: time, value
      quantity: displacement                      # or velocity / acceleration
      delay: 0.0                                  # this support's own lag, s
      wave: {speed: 1000.0, direction: [1, 0, 0], origin: [0, 0, 0]}
```

A record drives translations only (the rotations are held by the plain
constraint beside it, whose zero the record overrides). A displacement record
also imposes its velocity; before the record starts the support is at rest,
after it ends a displacement stays where it went. `wave` adds a delay of
`(x - origin) . direction / speed` to every node of the region. Dynamic only,
frame, fibre-beam and shell models. The displacement is ABSOLUTE: Rayleigh
`alpha` acts on the absolute velocity and so damps the rigid motion the ground
imposes -- prefer `beta` or viscous damping on the relative problem, or
accept the extra damping knowingly. The energy ledger books the reaction
work the ground does. `case_36` (a deck between two piers 400 m apart, a
wave at 1000 m/s) is held to a closed form.

A model with a free rigid-body mode is an error under relaxation — a free body
has no equilibrium to relax to — and a warning under `mode: dynamic` with a
physical mass, where the particle equations of motion need no support.
[`cases/case_19_floating_beam.yaml`](../cases/case_19_floating_beam.yaml)
is Wu & Ting's unconstrained beam (2020, part 2, example 1): two pressure
pulses on its end faces, then 40 ms of free flight, translating and spinning.
Its contract is what the physics fixes without the paper's patch geometry,
which the paper does not give: the energy audit closes to 1e-8 (the flexible
companion, exchanging 6 J of strain energy against 85 J kinetic, to 3e-5),
the momentum is the impulse, the ends' relative motion is a rotation of
radius L/2 about the width axis (amplitude L to 5e-6, the chord shortens to
zero at half a turn), the mass centre coasts at constant velocity, and
nothing moves across the width. Two defects it found: the energy ledger
paired step n's forces with step n+1's displacement increment and drifted to
20% on the flexible flight, and a solver-block `n_steps` — a relaxation
budget — silently truncated a dynamic run and compressed its load history
with it. The monitor option `relative_to: <region>` reports one region's
displacement relative to another's, which is how the spin is read off a body
whose mass centre is moving.

A `moment` on a solid face is a couple about a fixed axis with a potential
(`mesh.FaceMoment`). A solid has no rotational dof to hang a
couple on; what the load is is the potential −|M|·θ(x), θ the angle the
face's best-fit rigid rotation has turned about M's axis, whose gradient in
the reference configuration is the textbook bending traction M z / I (and,
for a component along the face normal, the linear shear of Saint-Venant
torsion) as tributary-area nodal forces, and on a turned face that
distribution rigidly rotated. With `follower: true` it stays a couple of
magnitude M about the face's own section through any rotation — smooth
through a full turn, no state; without it the reference forces are a dead
load whose couple decays with the cosine of the rotation. Both are
conservative. (A fixed couple is not conservative in three dimensions — a
body turned a full circle about a slightly different axis has received
different work — and this potential pays for that only where the face turns
edge-on to M, which a bending roll-up never approaches. The rotation-vector
potential −M·φ, tried first, paid at 180 degrees with a branch jump and at
360 with a singular tangent map.)
[`cases/case_20_rolled_cantilever.yaml`](../cases/case_20_rolled_cantilever.yaml)
is Wu & Ting's example 2 (2020, part 2): a solid cantilever under
M = 2πEI/L rolled into a complete circle in forty levels, its tip on its
root, against the closed-form elastica at every quarter turn — within 3% of
L, the solid's own thickness (a third of the circle's radius at the full
turn) accounting for most of it. The load was first written as the textbook
traction re-evaluated on the current face, which is not the gradient of
anything: a finite-difference tangent (`vfife probe`) showed
the element's part symmetric to 1e-9 at 10% strain and the load's part
almost purely skew — a thousandth of the element stiffness and a thousand
times the beam's softest mode. Kinetic damping, a descent method, thrashed
on it from a 150-degree tip rotation on and stalled at 230; a potential load
converges in 850 000 iterations flat.

Displacement, velocity, material state, rotations and any erosion carry
straight over — a structure that has already cracked under its own weight is
the structure the next step pushes on. Each step starts from rest, since the
kinetic energy the previous one ended with belongs to its loading.

Model-level BCs are the base and apply in every step; a step's own are added,
and where both prescribe the same (node, dof) the step wins — so a step can
drive a DOF the base merely held. **A base BC ramps on during the first step and
holds thereafter**, which is what "the column carries this throughout" means.
Steps may override `n_steps`, `load_steps`, `max_steps_per_load`,
`stage_ramp_iterations` and `tolerance`; `mode`, `mass`, `backend` and `damping`
are properties of the model, not of a step.

**What this replaced.** A six-number bounding box or a list of node ids, and one
global `load_curve` that multiplied every load and every prescribed displacement
together. So an axial load could not be held while a lateral displacement was
cycled — which meant the Chaboche and rebar kinematic hardening and the CDP
`wt`/`wc` stiffness recovery, all of which exist for exactly that loading, had
nothing to drive them. `cases/case_06_rc_column_cyclic.yaml` is that analysis:
800 kN held on the head while the head is cycled to ±1% drift, with a pinched
hysteresis loop whose area its contract asserts.

On the GPU, loads are grouped by amplitude curve and each group keeps its own
unit vector on the device. The alternative — re-uploading the external force
whenever a factor changes — is 24 MB per iteration at 10^6 nodes, more traffic
than the element kernel it feeds. Real cases have one or two distinct curves.

## Line meshes written in the case (1.1)

A frame, bridge or cable model does not need a Gmsh file: `mesh: {type: lines}`
names its points and its groups of members in the case, and each name is a
region the sections, supports and loads refer to as they would to a Gmsh
physical group (user manual, Section 5.4):

```yaml
mesh:
  type: lines
  points: {A: [0, 0, 0], B: [40, 0, 0], C: [80, 0, 0], P: [40, 0, -15]}
  lines:
    girder: {through: [A, B, C], size: 2.0}
    pier:   {through: [P, B], divisions: 6}
    arch:   {through: [A, C], divisions: 24, arc: {rise: 12.0}}
```

Gmsh meshes stay as they are, and are what shells and solids need.
`vfife new --list` lists scenario templates written this way (a continuous
girder, a pier on piles in p-y soil, an arch, a cable) that run as they
stand: `vfife new girder my_bridge.yaml` (1.2).

## Meshes from Gmsh

### Start a generic analysis from any model

`vfife init` is mesh-driven, not a catalogue of member shapes.  First define
Physical groups in any 3-D Gmsh model, then run:

```bash
python -m vfife init cases/bracket.yaml --mesh geometry/bracket.msh
```

The command imports the mesh, lists its Physical groups, asks which group is
fully fixed and which receives the initial total load, and generates a valid
linear-elastic YAML scaffold.  It preserves a relative mesh path and leaves
the generated file ready for `vfife info` and then `vfife run`.  It writes a
solid section, so it takes a mesh of volume elements: a shell or a line mesh
is refused with the section to write by hand instead.  It does not write over
an existing file unless given `--overwrite`, and never over the mesh itself;
`--fix` / `--force` given on an interactive run replace the corresponding
prompts.  For automation:

```bash
python -m vfife init cases/bracket.yaml --mesh geometry/bracket.msh \
  --fixed-region bolt_holes --load-region load_pad --load 0 0 -25000 \
  --non-interactive
```

For arbitrary support and loading layouts, add repeatable boundary entries;
the three-character mask is `[x, y, z]`, where `1` constrains that direction:

```bash
python -m vfife init cases/frame.yaml --mesh geometry/frame.msh \
  --fix left_bearing 111 --fix guide 010 \
  --force top_pad 0 0 -25000 --force wind_face 12000 0 0 \
  --non-interactive
```

The GUI integration boundary is `vfife.scaffold.GenericGmshDraft`: it holds a
mesh reference, lists of public YAML-shaped constraints and loads, plus the
material, solver, regions, amplitudes, steps, reinforcement and output
dictionaries.  A GUI should edit this draft, call `draft.validate()`, render
YAML, then invoke the existing solver; it must not duplicate the solver's case
rules.

The scaffold is deliberately a starting point: add named constraints and loads,
steps/amplitudes, reinforcement, nonlinear material data, and output monitors
for the actual model.  Geometry remains arbitrary; VFIFE derives selectable
regions from the Gmsh Physical groups rather than assuming a beam or column.

```yaml
mesh:
  type: gmsh
  path: cantilever.msh     # relative to the case file
  scale: 1.0               # 0.001 if the mesh was built in millimetres
```

**Physical groups become regions.** Name a group in Gmsh and use it directly:

```yaml
constraints: [{region: left_face, dofs: [1, 1, 1]}]
loads:       [{region: right_face, vector: [0.0, 0.0, -1.0e+4]}]
```

No bounding boxes, which is the point — on a real geometry a `box:` selector is
guesswork. The element type comes from the file too; a section never names one,
so there is nothing to disagree with.

Formats: **MSH 4.1 ASCII and binary, and MSH 2.2 ASCII** (including Windows
CRLF line endings). (MSH 2.2 binary is refused with a message saying so.) Measured on 837 k tetrahedra: 2.1 s binary,
2.7 s ASCII, both about 35 MB — binary is somewhat faster to parse but not
smaller for linear tets, where the node tags cost 8 bytes each in binary and
two or three characters in text. The gap widens on higher-order elements and on
coordinates, which are exact in binary and rounded in text.

**Why not `meshio`.** Two things it would not give us. First, node ordering is
not delegable: Gmsh numbers a tet10's mid-edge nodes (0,1) (1,2) **(0,2)**
(0,3) **(2,3) (1,3)** where the VTK convention `library.py` is built on ends
**(1,3) (2,3)** — slots 8 and 9 swap. `verify_ordering()` checks that mapping
from the node positions rather than trusting a table; it permits a curved
second-order CAD edge, whose mid-side node is intentionally off its chord. The
imported meshes are held to the same patch test as the generated ones (1e-13). Second,
Gmsh files carry nodes belonging to curves and points that no volume element
uses; left in, each is a node with no stiffness and no mass. They are pruned and
renumbered at import, and inverted elements are flipped there too, rather than
surfacing later as a message about Jacobians.

## Elements and materials

Elements are **data**, not code: `elements/library.py` holds each topology's
`dN/dxi`, quadrature and faces, and one precompute plus one kernel serve all of
them. Adding a topology does not add a kernel.

| element | nodes | formulations | note |
|---|---|---|---|
| `hexa8` | 8 | full / reduced+hourglass / b-bar | the workhorse |
| `tet4` | 4 | full | constant strain: **locks**, use for fill only |
| `tet10` | 10 | full / b-bar | 4-point rule; what to use when tets are needed |
| `wedge6` | 6 | full / b-bar | 3 tri x 2 line = 6 points |
| `truss2` | 2 | full | independent axial bar (Gmsh Line2, NumPy; vector/body/gravity loads); also used by embedded reinforcement |

| material | dim | state | for |
|---|---|---|---|
| `linear_elastic` | 3-D | 0 | reference, elastic runs |
| `j2` | 3-D | 13 | steel; isotropic `Curve` + linear kinematic |
| `chaboche` | 3-D | 21 | steel with several backstresses and damage |
| `cdp` | 3-D | 15 | concrete damaged plasticity (Lee-Fenves) |
| `rebar` | 1-D | 3 | reinforcing bar; bilinear, combined hardening |
| `steel02` | 1-D | 7 | reinforcing bar; OpenSees' Steel02 (Giuffre-Menegotto-Pinto), the default for cycling |
| `steel_cyclic` | 1-D | 11 | cyclic steel with degradation and ULCF fracture (Cai et al. 2023) |

**Which steel law** is a question with a measured answer, not a preference: `docs/steel02.md` has the table (reversal roundness, how much the peak grows over six cycles at one amplitude, fracture, whether the fibre kernel can inline it, cost per point), and `tests/test_material_selection.py` asserts every ordering in it so the guidance cannot drift from the laws. The short version: `rebar` for monotonic, `steel02` for cyclic, `steel_cyclic` when it has to break, `piecewise_uniaxial` when the table itself is the model. `rebar_gmp` was deleted in v0.1 — it was `steel02` uncompiled. Since v1.0 `steel02` is the same model as OpenSees' Steel02 (it was a variant whose envelope ran 3.45% over its own hardening asymptote); since v1.20.1 it is written from Filippou et al.'s equations rather than ported (docs/steel02.md).


`rebar` sets `uniaxial = True`, and the continuum element **refuses** it: a bar
law inside a solid element would produce a material with no shear stiffness at
all, which looks like a converged answer rather than an error.

## Embedded reinforcement

A bar node is not a degree of freedom. It is a point inside a host element, and
its displacement is whatever the host interpolates there:

    u_bar = sum_a N_a(xi) u_host_a      ->      f_host_a += N_a(xi) f_bar

so the mesh never has to line up with the bars. Move the reinforcement, change
the cover, add a layer: nothing about the mesh changes. Perfect bond is implied.

```yaml
materials:
  steel: {type: rebar, E: 2.0e+11, rho: 7850.0, yield_strength: 400.0e+6, hardening: 2.0e+9}
sections:
  bar25: {type: bar, material: steel, shape: circle, diameter: 0.025}     # area 4.909e-4 derived
reinforcement:
  bottom: {line: {start: [0.05, -0.15, 0.06], end: [11.95, -0.15, 0.06], divisions: 60}}
  # stirrups: one leg plus a spacing, not sixty near-identical entries
  legs:   {line: {start: [0.2, -0.15, 0.06], end: [0.2, -0.15, 0.94], divisions: 7,
                  repeat: {count: 59, offset: [0.2, 0.0, 0.0]}}}
assign:
  - {reinforcement: [bottom, legs], section: bar25}
```

`divisions` matters: each segment carries one constitutive point, so a bar
modelled as a single element yields over its whole length at once.

### Bars that are not straight: `path`

`line`, `layer` and `hoop` draw straight bars, which is enough for a cage and
not enough for a bent-up bar, a hook, or a draped tendon. `path` takes a route:

```yaml
reinforcement:
  # a harped tendon, or a bent-up bar: exact, through the points you give
  harped: {path: {points: [[0.0, 0.09, 0.50], [3.0, 0.09, 0.06], [5.0, 0.09, 0.06],
                           [8.0, 0.09, 0.50]], segment_length: 0.2}}
  # a draped tendon: the unique parabola through three points
  draped: {path: {points: [[0.0, 0.09, 0.30], [4.0, 0.09, 0.10], [8.0, 0.09, 0.30]],
                  shape: parabola, segment_length: 0.25,
                  repeat: {count: 4, offset: [0.0, 0.04, 0.0]}}}
```

`shape: polyline` (the default) lays a straight leg between each pair of
points and subdivides each one the way an ordinary `line` is subdivided. It is
exact: the bar goes through the points and nowhere else.

`shape: parabola` takes **exactly three** points and draws the unique quadratic
through them, with the middle point at the parameter midpoint — so for the
usual symmetric profile (anchorage, low point at midspan, anchorage) the middle
point is the low point, and for an asymmetric one it is not what you meant and
`polyline` is the honest way to say it. The curve is sampled at **equal arc
length**, so `segment_length` means the same physical thing here as on a
straight bar; each chord becomes one bar segment, and the curve therefore needs
its sampling length up front (from the path or from the assignment) where a
straight leg could take it later.

`repeat` works as it does on a line: four parallel tendons on one profile is
one entry and an offset.

**What a curved tendon buys.** The prestress acts *along* the bar, so the
uplift of a draped tendon comes out of the angle change between consecutive
segments — `w = 8 P a / L²` for a parabola of sag `a`, Lin's load balancing.
`cases/case_14` runs exactly that: transfer cambers the beam, and a downward
uniform load of `8 P a / L²` takes the camber back to 3% of itself. It is the
test a straight tendon cannot fail, because a straight tendon has no curvature
to get wrong.

**And it is what friction runs along** — see the next section: a tendon has an
arc length and an accumulated angle only because a route is one connected bar.

`path: {group: ...}` — a curve read out of the Gmsh file as a bar route — is
still refused by name. It is a different feature from drawing one here.

**A route is one bar, and that is not a detail.** Every `lines:` entry gets its
own nodes, so two entries whose ends merely touch are two bars. With perfect
bond that is harmless — coincident bar nodes take the same host displacement —
and with a `bond` block it is not: a bar written as two touching entries is two
pieces, each free to slip and each pulling out at `tau_max`, and
`anchorage: ends` ties the ends of both pieces instead of the two ends of the
bar. `path` and `hoop` therefore emit a single route; you can write one
directly as `{points: [[x,y,z], ...], closed: true}`; and preflight raises
`SPLIT_BAR_ROUTE` on a bonded set drawn as touching pieces.

**Looking at the bars.** A run with reinforcement writes a second sidecar,
`<name>_rebar.xdmf`, holding the bars as a polyline grid with `displacement`
(host-interpolated, so Warp By Vector moves them with the concrete),
`axial_stress`, `plastic_strain` and `block`. Open it alongside the main
`.xdmf` and apply Tube. They are two files rather than one spatial collection
because ParaView's handling of spatial collections nested inside a temporal one
is unreliable — some builds show only the first block, others lose the time
steps.

**The overlap.** An embedded bar occupies space the host element also fills, so
`Ec*As` is counted twice — inherent to the method, and what Abaqus does too. It
is 0.2% of EA at rho = 1% and 2.6% at rho = 3.1% (measured). `host_deduction`
on the assignment removes it exactly, and is right while the concrete around
the bar is elastic.
With a host that can crack (`cdp`, or anything with a damage measure) the
solver **refuses** it: once the concrete has cracked it no longer carries the
`Ec*eps` being subtracted, the bar is short by that much, and past yield its
net tangent is `E_hard - Ec` — negative. case_12 carried the setting into a
cracking beam and six rounds of work chased the ceiling it built before a
one-minute composite tension test showed the bar losing force as it hardened
(`tests/test_host_deduction_guard.py`). Use it for elastic work; leave it out
otherwise — the overlap it corrects is under 3% of EA at rho = 3%.

The bars run **on the device** on the Warp backend. Evaluating a few thousand
bars on the host instead would mean pulling `u` down and pushing a force back
every step — precisely the transfer the device-resident loop exists to avoid.

**Segment length, not division count.** A bar primitive, or its assignment,
takes `divisions` OR `segment_length` (metres), never both — writing both is refused rather than
letting one silently win. Prefer the length. A perfectly bonded bar smears a
crack of opening `w` over one segment, so the strain it sees is `w/L` with
nothing to spread it, and the drift a solid RC column survived scaled with the
SEGMENT length rather than with anything physical: 32.0 / 18.0 / 11.0 mm for
100 / 50 / 25 mm segments, at a bar plastic strain that did not move. Holding
the segments at 100 mm while refining only the concrete put the drift back to
32.0 mm exactly. `divisions` invites that mistake because it is a count, and a
count is what people scale with the mesh; `segment_length` pins the smearing
length to a number you chose on purpose.

**Rupture.** `rebar` takes an optional `rupture_strain`, on ACCUMULATED plastic
strain rather than on total strain, so it is monotone and therefore
irreversible — a bar that has ruptured stays ruptured through any subsequent
unloading, without a fourth state column to carry the flag. Omitted, it is off.

### Bond-slip: the bar gets its own degrees of freedom

Omitting `bond` keeps perfect bond, which is cheaper, runs on every backend and
is right whenever the member is not expected to crack. A `bond` block replaces
the interpolated bar with one that has its own nodes and an interface:

```yaml
sections:
  bar20: {type: bar, material: steel, shape: circle, diameter: 0.020}   # bond acts on the SURFACE
reinforcement:
  longitudinal: {line: {start: [0.062, 0.062, 0.0], end: [0.062, 0.062, 1.9], segment_length: 0.1}}
assign:
  - {reinforcement: longitudinal, section: bar20,
     bond: {type: bond_ceb, fc: 25.6e+6, condition: good}}
```

| law | what it is |
|---|---|
| `bond_ceb` | fib Model Code 2010 envelope, pull-out or splitting, with a linear unloading rule. Right for a pushover. Under a long cyclic history it overstates the bond that survives, which is the unsafe direction. |
| `bond_cyclic` | the same envelope with Eligehausen-type degradation and a pinched reversal. On a monotonic path it reproduces `bond_ceb` **bit for bit**, so switching costs nothing but the extra state. Its `damage_coeff`, `damage_exponent`, `friction_ratio` and `s_reload` are exposed defaults to CALIBRATE, not citations. |

**Why it exists.** Perfect bond has no length scale of its own, so the transfer
length is whatever the discretisation says (see `segment_length` above). With
bond, the transfer length comes out of `tau(s)` and the bar perimeter and the
analyst does not choose it at all.

**What it needs.** A bar section with a surface — `shape: circle` with a
`diameter` — because a bond force is a stress times the bar surface and the
area alone does not give it. `transverse_penalty` on the assignment is
optional — a numerical restraint on slip ACROSS the bar, defaulting to the
bar's own `E*A/L_trib`; it is refused on an assignment with no `bond`, where
nothing would read it. Bond-slip runs on **numpy and numba**; warp keeps its whole time loop
on the device and refuses it by name rather than silently slipping the bar
against a displacement field that stopped updating.

**Anchorage is not automatic, and it is a choice.** A bar that stops at the
member face has no development length, and bond will tell you so: on the
Tanaka column the bars ended at z = 0, bond could deliver at most `tau_max * pi
* d` = 795 kN/m against the 149 kN a bar needs to yield, and the column
plateaued at 51% of its measured peak with the bars never passing 338 MPa.
Perfect bond hid that completely — its bar ends were interpolated straight from
fixed host nodes, so no anchorage length was ever needed. Two honest answers:
run the bars into a stub, or into whatever the real detail is; or, when the
real detail is a hook, a plate or a bar running past the support, say so:

```yaml
  - {reinforcement: longitudinal, section: bar20, anchorage: ends,
     bond: {type: bond_ceb, fc: 25.6e+6}}
```

`anchorage: ends` (or `start` / `end`, each line's first or last node) ties
those bar nodes to the host: no dofs of their own, the segment force handed to
the host through the interpolation weights, no interface — exactly what a
perfectly bonded end does, on a bar whose length is otherwise bonded. It is
refused on a perfectly bonded assignment (nothing to tie) and on a closed hoop
(no free end).

### Prestress: a bar that is already strained when the model is not

```yaml
sections:
  strand: {type: bar, material: prestressing_steel, shape: circle, diameter: 0.0158, area: 150.0e-6}
reinforcement:
  tendon: {line: {start: [0.0, 0.10, 0.06], end: [8.0, 0.10, 0.06], segment_length: 0.05,
                  repeat: {count: 4, offset: [0.0, 0.06, 0.0]}}}
assign:
  - {reinforcement: tendon, section: strand, prestress: 1000.0e+6}
```

`prestress` is a stress in the bar, positive in tension, and it is lowered to a
strain offset `eps_p = sigma / E` that the constitutive law sees on top of the
mechanical strain. At `u = 0` the bar therefore carries `E A eps_p` and the
structure is out of balance by exactly the jacking force; the first
equilibrium **is** transfer. Nothing is applied as an external load, so a step
with no loads and no imposed displacement is a transfer stage.

The prestress is **not** ramped by `load_steps` — it is present in full from
the first iteration, the way a release is — so a transfer step wants
`load_steps: 1` and nothing else. Splitting it into ten does not stage the
transfer; it runs one transfer and nine empty steps.

**What the program computes.** The elastic-shortening loss, and it computes it
by equilibrating rather than by applying a formula. On a 0.5 x 0.1 x 0.1 m
prism with a 200 mm² tendon at 1000 MPa the closed form
`eps = -E_p A_p eps_p / (E_p A_p + E_c A_g)` gives -5.8997e-4 and 884.96 MPa
left in the tendon; the model gives -5.8997e-4 and 884.92 MPa. An eccentric
tendon gives `P e L^2 / (8 EI)` of camber to 1.3%.

**What it does not compute, and what you must therefore write.** Friction
along a duct, anchor set, relaxation and shrinkage/creep are not modelled —
`prestress` is the **effective** stress the tendon has after them, the
sigma_pe of a design calculation, not sigma_con. Preflight says so on every
case that uses it, together with the force per tendon and the total.

**Pretensioning is what this models.** The tendon is strained before the
concrete exists, and everything after that is one system finding equilibrium.
Post-tensioning — jacking against concrete that is already there, as a stage
of its own — is not expressible: there is no way to raise the prestress
part-way through a run. Curved and harped tendons *are* expressible: see
`path` under Embedded reinforcement, and `cases/case_14` for the load
balancing a draped one produces.

**Perfect bond or bond-slip, and the difference is the transfer length.** A
perfectly bonded tendon hands its whole force to one host element at each end,
which puts a St Venant zone about one section depth long at each end and no
transfer length at all. Add a `bond` block and the force builds through the
interface instead. Measured on the prism above with a 16 mm strand and CEB
bond at fc = 40 MPa, tendon stress along the bar:

| x (m) | 0.013 | 0.063 | 0.113 | 0.163 | 0.213 | 0.263 | … | 0.488 |
|---|---|---|---|---|---|---|---|---|
| MPa | 45 | 206 | 343 | 455 | 539 | 567 | | 45 |

The member is shorter than its own transfer length and never develops the
strand — 567 MPa in the middle against the 885 MPa the perfectly bonded model
reports everywhere. That is the right answer for a 0.5 m pretensioned member
(fib MC2010 puts the transfer length near 50 diameters, twice this specimen),
and it is the reason to pay for the bonded element. Do **not** add
`anchorage: ends` to a pretensioned strand: a tied end is an anchorage plate,
which is a post-tensioning detail, and it removes the transfer length by
construction.

### Friction, and the losses the program does compute

```yaml
assign:
  - {reinforcement: tendons, section: strand15, prestress: 1395.0e+6,
     friction: {mu: 0.25, k: 0.0015, jack: start, anchor_set: 0.006}}
```

With a `friction` block, `prestress` is the stress **at the jack** and the
solver computes the stress in every segment of the tendon:

    sigma(s) = sigma_con exp(-(mu theta(s) + k s))

`s` is the arc length from the jacking end and `theta` the angle accumulated
along the way — which is why this needs `path`: a tendon has a `theta` only
because it is one connected bar whose segments are laid in order. A tendon
drawn as touching pieces is refused (`FRICTION_SPLIT_TENDON`) rather than given
a profile that restarts at each piece, and so is a closed hoop, which has no
jacking end.

| key | what it is |
|---|---|
| `mu` | curvature coefficient, per radian |
| `k` | wobble coefficient, per metre |
| `jack` | `start`, `end`, or `both` — each point then takes the smaller of the two losses, which is what jacking from both ends buys |
| `anchor_set` | draw-in as the wedges seat, in **metres** (6 mm is `0.006`) |

Anchor set reflects the profile about `sigma(L_set)` over the draw-in length,
where `L_set` is found from the area between the two curves: `E_p * delta`.
It is solved by walking the elements, so it is right for a draped tendon and
not only for a flat one, and a draw-in larger than the tendon can absorb is an
error that tells you how much it can.

Preflight reads the result back as engineering numbers:

```
[INFO] PRESTRESS_PROFILE: reinforcement 'tendons': 1395 MPa AT THE JACK (start),
       20.1 m long, turning 0.2379 rad; friction alone leaves 1272 MPa at the
       far end, an 8.8% loss, before 6 mm of anchor set
```

**So the loss ledger now reads:** elastic shortening, friction and anchor set
are computed; relaxation and shrinkage/creep are still yours to deduct before
writing `prestress`. Preflight says exactly that, and changes what it says
depending on whether a `friction` block is present.

**Refused:** a `prestress` at or past the bar material's `yield_strength`.
The conversion `sigma / E` is only the right strain while the bar is elastic,
and a tendon jacked to yield has no elastic range left to carry load — so it
is an error rather than a warning. Give the effective prestress (0.6–0.75
f_py is the usual range), or a strand `yield_strength` that is actually the
strand's.

## Element deletion (erosion)

A softening model has no answer for an element that has reached total
degradation: it still has nodes, still carries whatever residual force those
nodes carry, and has no stiffness left to resist it. Both commercial codes
answer the same way — take the element out of the calculation.

| | Abaqus | LS-DYNA | here |
|---|---|---|---|
| keyword | `*SECTION CONTROLS, ELEMENT DELETION=YES` | `*MAT_CSCM` `ERODE` | `solver.erosion` |
| material criterion | MAX DEGRADATION (0.99 kept / 1.0 deleted) | d > 0.99 **and** ε₁ > ERODE−1 | `damage` **and** `strain` |
| points needed | all section points at one integration location (Explicit) | NUMFIP, default 1 | `points`, default half |
| criterion-free net | distortion control | `*CONTROL_TIMESTEP` ERODE + DTMIN | volume ratio + non-finite |
| deleted element's nodes | not documented | retained as free mass | **frozen** if unsupported |

```yaml
solver:
  erosion:
    damage: 0.98
    strain: 0.01            # max principal strain, not a plastic strain
    points: 4               # of 8
    every: 50
    min_volume_ratio: 0.2
    max_volume_ratio: 5.0
```

Three things about this implementation are worth knowing:

**Erosion is zeroed quadrature weights, not compaction.** Every force path —
the deviatoric assembly, the b-bar volumetric term, the hourglass term — carries
the element's weight as a factor, so zeroing it makes the internal force
*identically* zero on all three backends with no kernel change. The element
keeps its node ids and costs its share of the kernel. Compacting would break the
deterministic gather (which is what makes CPU and GPU comparable bit for bit)
and would force a full re-upload every time one element died.

**`strain` is kinematic on purpose.** LS-DYNA erodes on max principal strain,
and the first version here used the material's own equivalent plastic strain
instead. That silently never fires: under crack-band regularisation the plastic
strain at full damage is O(2·Gf/(ft·l)), which on the RC beam is 6e-3 against
FHWA's threshold of 0.05 — 56 elements sat at d = 0.99 and not one eroded.

**Unsupported nodes are frozen, not left free.** LS-DYNA keeps a deleted
element's nodes as free mass, which is right for a transient analysis — a freed
fragment flies off harmlessly. Under dynamic relaxation the mass is fictitious
and the time is pseudo-time, so such a node does not fly off; it coasts for the
rest of the run and sits in the residual norm the whole time. A node no element
and no bar still touches is held where it is. A *driven* node is never frozen,
however unsupported it becomes — freezing the loaded end of the bar test held
the displacement at 0.00056 against a prescribed 0.020.

**What it buys, measured.** On a concrete bar pulled past cracking it is exact:
the bar severs, the reaction goes to zero, and the residual drops four orders of
magnitude. On the RC beam it moves the onset of divergence from ~26 mm to
~34 mm of midspan deflection, and costs some smoothness in the load-deflection
curve. It does not cure the underlying problem — see Known limits.

## Fracture: three layers, M + E rather than M × E

Erosion above is the solid path and it is a *deletion*. v0.1 answers a
different question — "does a fracture criterion have to be written once per
material times once per element?" — and the answer is no, because the three
things that have to be decided are decided by three different parties and each
of them can only decide one:

| who | decides | interface |
|---|---|---|
| the material | whether **its own point** is finished | `Material.failure(state)` → a monotone fraction per point, `None` if the law has no such notion |
| the element | how its points **add up** | `fracture.Layout`: how many points, how they group |
| the model | what to do with a **finished element** | `fracture.Fracture`: stop it carrying load, and with `split: true` cut its bonds so it falls |

Fourteen laws and nine element families cost 14 + 9 implementations, and a new
law inherits fracture on every element the day it defines one method. The
material never learns how many integration points the element has; the element
never learns why a point failed.

`Layout` is where a fibre section differs from a solid: a fibre runs the length
of the member and is sampled at every station, so 20 fibres × 3 stations are 60
points but **20 things that can break**, and a fibre broken at any one station
is broken (Cai et al. 2023, eq. 20). FAC — the fraction of the section that is
gone (their eq. 19) — is an output field alongside the latched `fractured`.

```yaml
sections:
  rc: {type: fiber, material: conc, width: 0.4, height: 0.4, ny: 8, nz: 8,
       bars: [...],
       fracture: {threshold: 1.0}}      # the whole input
```

One number, because everything else is decided elsewhere on purpose: *when* a
point is finished is in the material block (a bar's `rupture_strain`), and how
this element's points add up is not a choice at all.

**The action is deactivation, and the mass stays.** LS-DYNA's own debris work
says why deletion is not enough — with erosion "the failed elements are
deleted" while "the existence of debris cannot be ignored since the impact
force is loaded on the structure through the debris", which is what
solid-to-SPH and solid-to-DES conversion were added for. A deleted member
cannot fall on anything. So a fractured element stops contributing internal
force and its fibres stop being driven, and its mass sits on the nodes it was
lumped onto at setup, untouched.

**Deactivation is not separation — `split: true` is.** On its own,
the two ends of a cut member are still the same two particles, so a beam cut in
the middle of a frame hangs from whatever else those nodes touch rather than
falling out of it. Turning the split on makes the topology change too:

```yaml
       fracture: {threshold: 1.0, split: true}   # + solver.mode: dynamic
```

The model keeps a **bond graph** — particles are vertices, an element *end* tied
to a particle is an edge (`core/topology.py`, the shape NVIDIA's Blast uses).
A fracture cuts that member's two edges; an end whose particle still carries
other members moves onto a particle of its own, and the member's share of the
mass moves with it. Connected components then say what has come away.

| what is asserted | why it is the assertion |
|---|---|
| total mass unchanged | the same row leaves one particle and lands on another |
| total momentum unchanged | the mass that moves keeps the velocity it had — anything else is an invented impulse |
| component count never falls | a split separates; it never rejoins |
| untouched members bit-identical | a renumbering that touched anything else would show here and nowhere else |
| the fragment's centre of mass falls at **9.806 m/s²** | it is genuinely free (`tests/test_particle_split.py`) |

Two particles per element are **reserved before the run**, parked on the ends
they may replace, so nothing is sized while the time loop runs and there is no
capacity to run out of — an end can separate once. They are reserved *after*
the regions are resolved, so a spare sitting on a fixed base is in no region
and the fragment cannot be born clamped to the support it broke away from.

**What it costs to ask**, measured on 40 members with a 64-fibre concrete
section and 4 bars:

| per step | fused kernel (0.21 ms/step) | NumPy path (8.6 ms/step) |
|---|---|---|
| no `fracture:` block | 1.00× | 1.00× |
| `threshold` (the latch) | **1.20×** | 1.00× (below noise) |
| `+ split: true` | **1.30×** | 1.12× |

Asking about fracture is a *fixed* amount per step (0.03–0.04 ms here), so its
weight is decided by how fast your step already is. The split's extra is not
about breaking anything — it is the reserved particles making the model three
times as many points for the orientation update and the DOF vectors, from the
first step. One split *event* (rebuild K, re-measure the bound, rebuild the
step's loads) costs 3.6–3.8 ms, i.e. under half a step of the NumPy model.

Both numbers used to be twice that, and the reason is worth repeating because
it is not specific to fracture: `grouped.max(axis=2)` over a **two-long**
trailing axis is one numpy call per group rather than one pass over memory —
120 µs against 5 µs for the same bits — and the gather was building 68 columns
of which 64 belonged to a law that declares no criterion and can never change.

**The step bound is re-measured after every split**, because a split takes mass
off a particle while the bound is `sqrt(min m/k)`. Rebuilding K over the
surviving members restores the axial ratio exactly and the rotational one only
to 0.56 — at a shared node the two members' bending couplings largely cancel in
the assembled row sum, so removing one takes half the mass with a tenth of the
stiffness. dt is not adapted mid-run; a bound that stops holding is logged.

`element.fracture` is accepted on `beam2_fiber` only and **refused** elsewhere
rather than silently ignored: a collapse run that reports a building still
standing is the most expensive wrong answer here.

**What breaks a fibre** is `steel_cyclic`, the law of the same paper's
§3 — see `docs/steel_cyclic.md`. A bare rupture latch takes the stress from
full to zero in one step, which is a finite jump in the constitutive function
and the shape that trapped a bond node in a limit cycle in v0.1; this law
falls to `f_uf` along a degradation curve first, so what the latch finally
removes is a residual. Its fracture strain is not a constant either: the DSPS
model shrinks it as compressive plastic strain accumulates, which is the
ultra-low-cycle fatigue that decides when a member goes in a collapse. It
compiles into the fused fibre kernel, so the fast path fractures on its own —
when v0.1 was written, none of the laws that kernel could inline had a
criterion at all.

## What a GPU iteration is made of

The warp shell backend measured 367 µs an iteration at 256 elements and 532 µs
at 4096 — sixteen times the work for 1.4 times the time, which is a solver
paying a fixed toll rather than doing arithmetic.
[`benchmarks/warp_loop_cost.py`](../benchmarks/warp_loop_cost.py) separates the
four candidate tolls in one run — five arms of the same problem, so the gaps
name the cost — and `--per-kernel` times every launch of the iteration on its
own inside a CUDA graph. It found the toll three times, and three times it was
not what was expected:

| what it found | fix | result |
|---|---|---|
| the convergence measures gave every dof a `wp.atomic_add` onto the same five addresses — 250 000 serialised atomics, 555 µs at 16384 elements, more than the element force and the integrator together | a three-level fold, quantity-major so each level reads contiguously | 555 → 25 µs, and the reduction became **bit-reproducible** |
| the element force read K[24,24] and B[8,24] — 768 values — in 32 dependent chains, **one thread per element**: 128 µs at 256 elements and 130 at 4096, and unchanged by switching to fp64, which is what ruled out the arithmetic | three launches, each with as many threads as its work has independent pieces: the frame per element, the four node rotations per node, the matrix products per node block | 128 → 39 µs |
| two `zero_()` calls costing 16 µs of API round trips | one kernel | 16 → 1.3 µs |
| the fold's first level ran 1024 threads whatever the problem | group count grows with the dof count | 62 → 13 µs at 16384 |

Per iteration, summed over the kernels: **194 → 60 µs** at 256 elements,
606 → 489 at 16384. None of it changes a number: the force is bit-for-bit what
the single kernel produced (7e-15 against the numpy path in fp64), and case_17
passes its contract on `warp`/`cuda`/`mixed` agreeing with the `numba` run to
3e-9. [`benchmarks/verify_backend.py`](../benchmarks/verify_backend.py) is how
that check is run without editing a tracked case file — a backend under repair
is held to the same contract as the one that works.

What is left is measured and written down in v0.1: the
per-iteration synchronising readback (84–130 µs, and the reason the numbers
above are a *potential* rather than a speedup yet), CUDA graph capture of the
block, and — the only thing that helps a large mesh — not storing K per element
at all.

## Invariants: the questions that do not depend on the element

Every numbered case in this library took days to make converge, and at the
end of most of them was a defect that had nothing to do with the case: a load
that was not the gradient of anything, a frame that was exact in a
plane and wrong in space, an energy ledger a step out of phase,
a fictitious mass that bounded the reference configuration and not
the current one. Each is a property that holds for *every* element,
costs a fraction of a second to check, and nobody was checking.

`vfife.testing` asks them. It carries one minimal model per registered
element — seventeen of them, a cantilever of some kind each — and one adapter
(`testing/state.py`) that gives a solid, a truss, a frame, a shell, a
connector and a coupled model the same three methods: the state as a flat
vector, the internal force at an arbitrary state without side effects, and
the external force. On top of that sit the checks:

| question | what a failure means |
|---|---|
| is the internal force the gradient of something (its tangent symmetric) | near equilibrium, a force field kinetic damping cannot descend: no iteration budget will help |
| is the **load** the gradient of something | the v0.1 defect. A load is a function of the state alone, so this is a verdict at any state |
| is a rigid motion of the *deformed* body free | the co-rotational frame, the polar decomposition or the orientation state is wrong |
| does the fictitious mass still bound the stiffest mode | ω² against the 4s² stability limit: at or above 1 the explicit step is what is stalling the run |
| is this state stable at all | a negative eigenvalue, with the direction split of its mode — the solver cannot find an equilibrium that is not there |

The perturbation is the one the equations of motion use: additive for a
translation and **multiplicative for a rotation**, because since v0.1 a
structural point's orientation is a matrix advanced by `R ← exp(ω dt) R` and
the assembled moment is conjugate to that spin, not to an increment of
`log R`. Differencing along `log R` scores 9e-2 on a beam bent 1.2 rad where
the right question scores 1e-10.

`tests/test_invariants.py` runs all of it over every element on every commit,
including a round trip between the load kinds each element *declares* in
`vfife capabilities` and the ones that actually work. **The user-facing half
is `vfife probe`**, which asks the same questions of the user's own case —
and, with `--result`, at the state a failed run stalled in:

```powershell
python -m vfife probe cases/case_20_rolled_cantilever.yaml
python -m vfife probe my_case.yaml --result results/my_case.h5
```

What the first run found, in one afternoon: a pressure on a
triangular exterior face crashed inside a bincount, on every tet4 model since
the beginning; a pressure on any *distorted* hexahedral mesh flipped the
normals of some faces and put a net 19% resultant into a load that must have
none, because the "interior point" that decides which way a face points was
the centre of one of that element's own faces; the same decision was being
made per quadrature point on quadratic faces, which made a follower pressure
on a tet10 body fully circulatory; three solids advertised a pressure they
had no facet table for and raised `KeyError` when asked; the line members of
a `shell_line` model were never given the shared orientation state, so a
finite rigid rotation put 37% of the model's own stiffness scale of spurious
force into them; and `connector2` still averaged rotation *vectors* where
every other family had been migrated to orientation matrices in v0.1.
That last one was carried for two stages as an assertion that it still
failed — so that fixing it would set off an alarm rather than pass quietly —
and v0.1 fixed it: the frame is now the geodesic mean triad and the
rotational deformation `log(T₂T₁ᵀ)`, and the spurious force went from 7.5%
of the model's own stiffness scale to 2e-10.

### The same, one layer down: the constitutive laws

v0.1 asks the material half. Each law has its own deep test file — the CDP
meridian ratio, the MC2010 bond envelope branch by branch — and none of those
would notice a *new* law arriving with an old defect. So `testing/materials.py`
drives every registered law along a strain path and asks what is true of all
of them: the virgin state carries no stress; the stress is **continuous in the
strain at a frozen state** (the `bond_ceb` jump of v0.1 sat a node in a limit
cycle for 8 000 kinetic resets); the work round a closed cycle is non-negative;
unloading is never stiffer than first loading; four points driven together give
what each gives alone (one `np.max` without an `axis=` and the worst-damaged
point in a mesh sets the response of all of them); `dt` does not enter a law
that declared it rate-independent; a rotated strain path gives the rotated
stress; damage does not heal on a monotonic path; a declared **failure
fraction never falls, on any path including a closed cycle** (v0.1 — damage
may recover, `cdp` restores stiffness when a crack closes, but what is dead
stays dead); **an evaluation at zero strain costs nothing**; and a
regularised law dissipates the same energy per unit crack **area** at every
element size.

That last-but-one is the newest and the one worth reading twice, because a
single-point test bench cannot fail it and a solver always can. Every solver
here assembles the internal force once *before* the first step — to size the
time step, to write the t = 0 frame — so every material point in every run is
asked for its stress at zero strain first. `steel_cyclic` answered that
question by banking a loading direction of +1, and the first step that then
went **downwards** looked like a reversal: the point left the monotonic curve
for the hysteresis skeleton before it had ever been loaded, and (because the
old branch logic could not leave the elastic line from a reversal at exactly
zero stress) ran up that line to **14 times f_u**. Found by Cai et al.'s Tube
A carrying 147 kN against its own squash load of 100.8; after the fix, 99.2.
The check drives the path **and its mirror**, which is not thoroughness but
the whole test: a path that goes up first agrees with itself no matter what.

Fourteen laws × ten questions, and each check is also shown *failing* against
a deliberately broken law in the same test file — a check nobody has seen fail
is a check nobody has tested. `vfife probe` runs them on the analyst's own
material block, with their numbers:

```powershell
python -m vfife probe my_case.yaml            # model questions, then material
python -m vfife probe my_case.yaml --no-materials
```

One open finding, recorded rather than fixed: `cdp` dissipates **8.6% more
fracture energy per unit crack area on an element twice the size** (measured
at `l_ref` and `2 l_ref`, confirmed on the material's own uniaxial-stress
driver). Turning damage off collapses the spread from 13.5% to 2.0%, which
locates it: the crack band scales the *plastic* strain by `l_e/l_ref`, and the
crack opening also carries the damage part of the inelastic strain, which is
not scaled. It is asserted as a *failure* in `tests/test_material_invariants.py`
so that fixing it sets off an alarm.

## Verification

**Where the full check runs**. CI runs the default layer on
every push and the solving layers only when started by hand: the nightly
matrix -- every shipped case every night, sixty jobs of up to two hours --
used a private repository's month of Actions minutes in days. The same
checks run locally, in parallel: `python tools/check_all.py` (the default
layer, the slow tests, every case against its contract, and the wheel
installed into a fresh environment and run outside the repository), with a
report in `results/check_all/`.

2395 collected tests (2301 in the default layer, 94 `slow`). The ones worth knowing about:

| test | what it pins down |
|---|---|
| `test_rigid_body_rotation` | 30° rigid rotation → zero internal force, all three formulations, distorted meshes. Catches any co-rotational or hourglass defect. |
| `test_patch_test` | linear displacement field → exact constant stress to 1e-9 |
| `test_volumetric_locking_spurious_pressure` | at ν=0.4999, full integration shows the spurious pressure that causes locking; b-bar and reduced show none |
| `test_pure_shear_monotonic_matches_closed_form` | Chaboche against its closed-form solution; error halves with step size (backward Euler is first order) |
| `test_bauschinger_reversal` | reversed loading against an independent 1-D integration |
| `test_equibiaxial_compressive_strength` | CDP yields in equibiaxial compression at exactly `fb0/fc0 * fc0` |
| `test_meridian_ratio_is_Kc` | CDP tensile/compressive meridian ratio equals `Kc` |
| `test_uniaxial_*_reproduces_input_curve` | CDP reproduces the curves it was built from |
| `test_compressive_stiffness_recovery` | `wc = 1` restores the compressive modulus after cracking |
| `test_crack_band_makes_softening_mesh_independent` | sigma vs crack opening is the same curve for every element size |
| `test_staged_relaxation_gives_equilibrium_points` | every frame of a staged run is a real equilibrium point |
| `test_numba_matches_numpy` | the fused kernel equals the reference to 1e-12, all formulations × all materials |
| `test_gather_assembly_is_reproducible` | same input, same bits — what makes a CPU/GPU check meaningful |
| `test_warp_fp64_matches_numpy` | the GPU port equals the reference to 1e-12 in double precision |
| `test_warp_mixed_precision_stays_within_fp32` | production precision stays finite and within 1e-3 |
| `test_warp_device_loop_matches_cpu_loop` | the device-resident integrator equals the CPU loop, load- and displacement-controlled |
| `test_device_memory_fits_a_million_elements` | resident buffers leave room on a 16 GB card |
| `test_cantilever_converges_to_beam_theory` | b-bar converges to Timoshenko from below (shear locking), reduced from above — bracketing it is stronger evidence than either alone |
| `test_undamped_step_response_is_bounded` | step load peaks at ~2× static and does not grow |
| `test_bitwise_reproducible` | two runs identical bit for bit |
| `test_restart_matches_uninterrupted_run` | checkpoint/restart to 1e-12 |
| `test_all_elements_agree_across_backends` | tet4/tet10/wedge6/hexa8 x elastic/J2 x numpy/numba/warp, all to 1e-11 |
| `test_host_interpolation_is_exact` | the embedding puts the bar where it was asked to, on every element type, to 1e-12 |
| `test_node_coincident_bar_degenerates_to_shared_nodes` | a bar on mesh nodes loads those nodes and **identically zero** elsewhere |
| `test_uniform_stretch_gives_the_exact_bar_stress` | and the assembled bar force sums to zero — the embedding adds no net momentum |
| `test_host_deduction_recovers_the_transformed_section` | RC prism to 4.7e-4 of the closed-form composite section |
| `test_warp_truss_matches_numpy_through_yield_and_reversal` | device bars track the reference to 5.7e-16 through yield and Bauschinger |
| `test_reinforcement_enters_the_time_step_and_the_mass` | a stiff light bar binds the CFL limit; forgetting it is silently unstable |
| `test_eroded_elements_contribute_exactly_zero` | zero, not small — and the survivors are bit-identical, so erosion is local |
| `test_erosion_severs_a_fully_cracked_bar` | the bar carries nothing afterwards and the residual falls 100x |
| `test_a_driven_node_is_never_frozen` | freezing the loaded end would cancel the loading and still look like a result |
| `test_the_safety_net_survives_a_non_finite_element` | `eigvalsh` does not return from an `inf`; the net must outlive what it catches |
| `test_relaxation_defaults_to_underwood_mass_through_the_schema` | the schema emits `mass: None`, which used to defeat `get()`'s default and silently run real dynamics at a CFL step |
| `test_patch_test_on_imported_meshes` | linear field → exactly constant stress on real Gmsh output, every element type and formulation — the test a wrong mid-side ordering cannot survive |
| `test_every_encoding_gives_the_same_mesh` | MSH 4.1 ASCII, 4.1 binary and 2.2 are three spellings of one mesh |
| `test_physical_groups_become_regions_on_the_right_nodes` | `left_face` really is every node at x = 0 |
| `test_the_gmsh_to_vtk_permutation_is_not_the_identity` | and applying Gmsh's order unchanged is detected, not tolerated |
| `test_splitting_one_analysis_into_two_steps_reaches_the_same_answer` | to 1e-6 — if not, step state is not really carrying over |
| `test_a_model_level_load_ramps_on_once_and_then_holds` | the base load is neither re-ramped nor lost when the next step starts |
| `test_cyclic_loading_shows_the_bauschinger_effect` | reverse yield arrives early, the loop has area, and the protocol is followed to 1e-6 |
| `test_surface_and_interior_partition_the_mesh` | a selector property with no tolerance in it |
| `test_an_empty_region_is_an_error_not_an_empty_set` | a tolerance slightly too tight is caught, not silently obeyed |

## Concrete damaged plasticity

`materials/cdp.py` is the Lee-Fenves / Lubliner model Abaqus calls CDP: Lubliner
yield surface with `alpha`, `beta(ep)`, `gamma` and `Kc`, two hardening variables
(tension and compression, split by the stress-state factor `r`), non-associated
Drucker-Prager hyperbolic flow, stiffness recovery `wt`/`wc`, and Duvaut-Lions
viscous regularisation.

**No eigenvectors anywhere.** `dF/dsig_bar` contains `d<sig_max>/dsig_bar =
n1 (x) n1`, which normally forces a closed-form eigenvector. But it only ever
appears inside the scalar denominator of the cutting-plane step, and `D0` is
isotropic, so

```
(n1 (x) n1) : D0 : m  =  2G (m_max - tan(psi)/3) + K tan(psi)
```

— built from principal *values*, which Cardano gives in closed form. Exact, not
an approximation, and the reason this model ports to a GPU kernel unchanged.

**Curves in, table out.** Three input routes converge on one runtime path:

```yaml
compression: {source: gb50010, fc: 30.0e+6, beta_c: 0.7}
tension:     {source: fracture_energy, ft: 3.0e+6, Gf: 120.0, beta_t: 0.9}
# also: source: table (Abaqus stress/inelastic_strain/damage)
#       source: stress_displacement (softening vs crack opening)
```

GB 50010-2010 appendix C is a pure *damage* model (`sigma = (1-d) E eps`, so no
plastic strain at all). Feeding it to CDP needs a declared plastic fraction
`beta`; `beta = 0` is rejected with a reason rather than producing a model with
no hardening variable. The formulas are reproduced in full in the
`curves.py` docstring so they can be checked against the standard.

**The compression band needs a length too.** A code curve is a *strain* curve,
and crushing localises into one element row whatever its size — so the energy
that row dissipates per unit crushed area is the curve's energy per volume
times the element size: 9.6 N/mm at 75 mm and 2.4 N/mm at 18.75 mm for 44 MPa
concrete, against 35–58 N/mm in the literature. Measured on the Alberta beam,
refining only the depth: the peak fell 334 / 330 / 321 / 300 kN while the
pre-peak branch did not move. `Gc` (J/m², the crushing energy) is the missing
length scale — the post-peak branch is stretched so a band `l_ref` thick
dissipates `Gc`, and `k = l_e/l_ref` rescales it per element, exactly as the
tension side has always worked:

```yaml
compression: {source: gb50010, fc: 44.0e+6, Gc: 35.0e+3}   # 35 N/mm
```

With it the same refinement gives 350 / 352 / 352 / 344 kN. The value is a
modelling choice and there is no default: `8.8 sqrt(f_c[MPa])` N/mm (Nakamura
& Higai) or `~250 Gf` (Feenstra) are the usual estimates — 58 and 35 N/mm here.
Only the post-peak branch moves; `Gc` requires `crack_band: both`, since
nothing else would rescale it per element.

**Two guards worth knowing about:**

- *No surface expansion.* Substituting the beta-derived damage back gives
  `sigma_bar = sigma + (1-beta) e_in E0`, so for any `beta < 1` the effective
  yield stress grows without bound on a softening branch — with the beam's own
  numbers the tensile yield stress reached 4.8 MPa against `ft` = 3 MPa, a yield
  surface *expanding* while the material cracks. The effective stress is capped
  non-increasing after the peak; where the cap binds, more of the softening is
  carried by plasticity than `beta` asked for, and the run summary prints the
  fraction actually achieved.
- *Snap-back.* Along the softening branch `eps = sigma_bar/E + ep/k`, so once
  `-d(sigma_bar)/d(ep) * k > E` the strain stops increasing and no strain-driven
  solver can follow. `bind_points` refuses meshes coarser than that and names
  the limit (0.28 m for ft = 3 MPa, Gf = 120 J/m2, E = 30 GPa).
- *The compression band factor.* `crack_band: both` multiplies the same
  `k = l_e/l_ref` onto the compression curve, and that curve is a code curve on
  a fixed strain axis with no reference length the solver knows: `l_ref` cancels
  out of the tension curve (which is built from it) and does not cancel out of
  the compression one. On the Alberta beam `l_ref: 0.15` over a 15 mm mesh
  (k = 0.10) stretched the crushing branch tenfold, the compression zone never
  softened, and the curve went 3% *over* the measured peak with no descent — a
  wrong answer that looked better than the right one. So under `both`, k must
  lie in [1/2, 2]; outside it the run is refused (`COMPRESSION_BAND_RATIO`,
  also reported by preflight) with the ratio and the fix: `l_ref: auto`, the
  element size, or `crack_band: tension` on a mesh graded more than 2:1.

## Staged relaxation

For a load-displacement curve, set `load_steps` in relaxation mode. Each level
is relaxed to the residual tolerance and emits one output frame, so every point
on the curve is a genuine equilibrium point. A staged step's iteration budget
is `load_steps × max_steps_per_load`, and that is all the solver needs:
`solver.n_steps` may be left out (it is derived), and is required only for a
continuous step that has no `n_steps` of its own.

This is not cosmetic. In a continuous ramp the intermediate frames are still
ringing — on the beam case, successive frames read 2341, 287, 2151, 258 kN. The
level is also ramped into over `stage_ramp_iterations` rather than stepped: a
4 mm prescribed displacement applied in one iteration on an Underwood-mass
system diverged to 1e100 within the first load level.

## Layout

```
vfife/
  voigt.py               conventions (read this first)
  core/
    mesh.py              Mesh, box generator, deterministic scatter
    selectors.py         composable region expressions (box/plane/cylinder/boolean)
    amplitude.py         per-BC amplitude curves, incl. the cyclic generator
    solver.py            leapfrog loop, dynamic + relaxation modes, monitors
  elements/
    library.py           element definitions as DATA: dN/dxi, quadrature, faces
    continuum.py         precompute (cold) + internal_force (hot), any topology
    truss.py             2-node bar + EMBEDDED reinforcement (host interpolation)
  materials/
    base.py              Material protocol + registry
    elastic.py           linear_elastic
    chaboche.py          chaboche (P1-P3 fixed)
    j2.py                von Mises, isotropic (Curve) + linear kinematic
    rebar.py             uniaxial steel, combined hardening (1-D, for bars)
    cdp.py               concrete damaged plasticity (Lee-Fenves)
    curves.py            GB 50010 / Abaqus tables / Gf -> one curve type
  tensor.py              closed-form principal values, contractions
  kernels/
    __init__.py          backend selection; all three satisfy one force() contract
    scalar.py            njit device functions: quaternion, Cardano, curve lookup
    numba_hexa8.py       the fused element kernel (any node count)
    assembly.py          deterministic gather over the inverted connectivity
    warp_scalar.py       the same device functions for Warp, generic over precision
    warp_materials.py    Warp device twins of the material models
    warp_backend.py      GPU element kernel, device-resident integrator, reductions
    numba_shell4.py      the shell4 co-rotational force, one element at a time, serial
    warp_shell4.py       the same on a device, plus the six-dof resident relaxation
  io/
    gmsh.py              .msh import: MSH 4.1 ascii/binary + 2.2, physical groups
    schema.py            pydantic case schema  <- add a parameter here, not in 5 places
    writer.py            HDF5 + XDMF (+ a separate _rebar.xdmf) + the .ckpt.h5 checkpoint (docs/result_format.md)
  cli.py
cases/                   YAML case files
tests/                   416 tests (+ real Gmsh fixtures in tests/data/)
qualification/           machine-checked element/material admission evidence
benchmarks/              timing regression
```

## Adding an element or a material

Materials: subclass `Material`, set `n_state` and `state_names`, implement
`update(strain, state, dt) -> stress` operating on `(n_points, ...)` batches,
decorate with `@register(...)` including its kinematics, backends, outputs and
verification metadata, then import it in `materials/__init__.py`. Registration
does not itself grant engineering qualification; add a manifest under
`qualification/`. See `docs/library_registration.md`,
`docs/material_points.md` and `docs/qualification_gates.md`.

For the numba backend, also implement `numba_point_fn()`: an `njit` device
function `point(strain, state, dt, k, stress)` doing the same thing one
integration point at a time, closed over the material's own parameters. It lives
in the same file on purpose — the same model written for a different machine.

For the GPU, add `warp_params(dtype)` returning two flat buffers (scalars and
packed curve tables) and `warp_point_fn()` returning the `wp.func` that reads
them. Two buffers whatever the model, so the element kernel signature is
identical for CDP and linear elasticity. The Warp device functions live in
`kernels/warp_materials.py` rather than beside their siblings, for one
mechanical reason: Warp evaluates a `wp.func`'s annotations against the defining
module's globals, and the material files use `from __future__ import
annotations`, which turns those into strings that fail to resolve.

Two Warp traps worth knowing, both of which cost real accuracy before they were
found:

- A bare Python literal in device code is **fp32**, whatever the surrounding
  precision. An fp64 kernel built on `sqrt(3/2)` written as a literal lands at
  1e-7 against the reference instead of 1e-13. Irrational constants and
  tolerances travel in the material's `prm` buffer, packed by NumPy at the target
  precision; `principal()` gets pi from `wp.acos(-1)` rather than a literal.
- `type(x)(v)` works only in call position — binding it to a name fails codegen.

`test_backends.py` fails if any of the three drift. Without the hooks the
material still works on `backend: numpy`, with a clear error if the others are
requested.

## Known limits / next steps

- **Backend scope differs by element family.** Continuum elements support
  NumPy, Numba and Warp; `shell4` has NumPy, Numba and Warp; `beam2` has
  NumPy and Numba; `beam2_fiber` has NumPy (promoted to its compiled Numba
  kernel when the model qualifies -- `solver.kernel_status` says whether it
  did and why not) and Numba (the kernel is required, refused by name when
  it cannot be built); standalone `truss2` and `connector2` are NumPy; the
  mixed-line, shell-line and coupled solvers take Numba for the block that
  has a kernel. `vfife capabilities` is the authoritative table and, since
  0.1.1, is asserted against what the solvers accept
  (`tests/test_capability_alignment.py`).
- **The explicit step bound is still Gershgorin's.** Since v0.1
  every solver shrinks `dt` by `sqrt(1+zeta^2) - zeta` under Rayleigh
  `beta`; since v0.5 the `zeta` in that factor uses a power-iteration
  estimate of the top frequency (1.7x to 4x the step on a fibre column, a
  frame and a shell), but the undamped base step is the row-sum bound,
  which over-estimates `omega_max` by up to 4x on a shell. Undamped and
  alpha-only runs are unchanged to the bit.
- **The fibre beam's shear cracks at the member level, not in the fibres.**
  A fibre section may carry `shear_stiffness` (elastic kappa G A_s) or
  a `shear` law: uncracked kappa G A until the member's concrete
  cracks in flexure, then the 45-degree truss stiffness of its stirrups (Park
  & Paulay), unloading to the origin, optionally a stirrup-yield plateau --
  in series with the bending, so elastic it is the Timoshenko member. It is
  not MCFT: the shear does not interact with the axial and flexural strain
  field, the fibres never see a shear stress, and there is no shear failure.
  On Vecchio & Emara's frame it takes the rising branch from 1.15x to 1.08x
  the measurement with nothing fitted (case_21b; case_21 keeps the elastic
  shear and its band). In the fused numba kernel since v1.7 (1.7),
  the NumPy path the reference (case_21b: 20 minutes → half a minute).
- **Near-incompressible relaxation converges slowly.** At ν→0.5 the Underwood
  masses are set by the bulk terms while the bending modes stay soft. Use
  `mode: dynamic` with mass scaling there, or expect many more steps. **A
  dynamic run should name its damping** — `damping: {type: none}` or a
  Rayleigh alpha. Kinetic damping zeroes the velocity at every kinetic-energy
  peak, which is how a relaxation run finds a STATIC answer and removes the
  dynamics a dynamic run exists to have.
- **Mesh imports are intentionally bounded.** Structured boxes, Gmsh `.msh`
  and an engineering subset of Abaqus `.inp` are supported. The INP reader
  imports nodes, supported solid/Line2 elements and NSET/ELSET; materials,
  sections, loads, surfaces and includes remain in YAML. Nastran and VTU are
  not direct inputs.
- **A `moving_load` is a moving FORCE; a `vehicle` is planar.** The first has
  no inertia of its own and does not follow the deck (Fryba's model,
  case_26 to 0.3%). The `vehicle` brings mass, suspension, tyres and
  road roughness, but in the vertical plane of its path: heave and pitch, no
  roll, one wheel line per axle, a tyre as a point contact, and the contact
  on the undeformed station. What neither does is
  lateral distribution across a deck, since the axles ride on the member axis
  and an eccentric wheel line has to be written as an added moment; and the
  code impact factors, which are not this library's business.
- **Temperature goes one way only.** A `temperature` load is an imposed field:
  nothing in the library generates heat, conducts it, or works out a section's
  temperature from an environment, and no material property changes with it
  (no `E(T)`, no fire). Combined with the absence of creep, a sustained
  temperature never relaxes.
- **Creep, shrinkage and relaxation are there; construction staging is not.**
  Since the 0.4 line `creep` and `shrinkage` on beam and fibre concrete
  and `relaxation` on prestressing steel follow a
  member's history on a concrete-age axis (`docs/creep.md`; case_29 to 29g,
  case_28d). What is still missing is the SEQUENCE: a model is complete from
  its first step, so a girder built by cantilevering, or simply supported and
  then made continuous, cannot be erected in the order it was built, and its
  dead-load state is the one-stage state. That is the difference behind
  case_24's pier strains and case_24c's 15 mm against Chen.
- **Contact is with rigid surfaces only**: `contact.rigid`
  stops a node at a plane -- the ground, the sea floor, a pile cap -- with a
  penalty spring (`stiffness: auto`, sized from the node's mass and the step,
  or a number for a soft floor), Coulomb friction and a contact dashpot. The
  surface knows nothing of a section's depth. Contact BETWEEN line members
  is `contact.members` (since 1.11): each member a capsule
  of a stated radius, a compiled pair search kept as a neighbour list, a
  penalty sized from the masses and the step, friction with its couples,
  and an allowance for what overlaps at the start (a girder over its cap). `case_33` drops a beam on the ground. The energy ledger
  books the contact force by the trapezoid rule with the structure's
  forces; on an `auto` penalty an impact lasts two or three steps and that
  rule is not accurate, so many impacts accumulate an error that is the
  ledger's, not the motion's: 1.2 of the external work on the Xihoumen
  collapse with `auto`, 0.18 with a soft floor.
- **Only line members break by a capacity** (122–124):
  a bar at f_t A, a frame member at |N|/N_p + |M_y|/M_y,p + |M_z|/M_z,p = 1,
  on mixed_line, beam2, beam2_timoshenko and truss2, plus scheduled
  `removals` on those and on the fibre beam (whose members break through
  their own fibres, `fracture`). A broken frame member parts at the end that
  failed and splits off with its share of the mass. By default it breaks
  elastic-brittle, which in a collapse unzips a girder element by element;
  `rotation_capacity` makes a mixed_line member's ends perfectly
  plastic hinges that tear on accumulated rotation instead, with no
  hardening, no local buckling softening, and on mixed_line only. Shells and
  solids erode by material damage, not by a member capacity.
- **For bridge engineering this is an analysis engine, not a design
  package** (the assessment is v0.4). What a design office
  would reach for and will not find: load combinations and envelopes, code
  checks, lane or influence-SURFACE loading of a deck (a moving load rides the
  member axis, so transverse distribution has to be modelled by hand),
  soil-structure interaction beyond nonlinear-elastic p-y springs (no
  hysteresis, no t-z/q-z, no radiation damping at the supports: a support
  record is imposed where it is written), construction stages (planned for
  1.2) and a comparison
  of one bridge against commercial software are planned after 1.0 -- and
  shear lag, distortion and warping of a
  box girder modelled as a beam. Bearings carry Coulomb friction, a gap and
  a bilinear hysteresis since v0.5 (`docs/connector2.md`), but no
  velocity-dependent friction and no degradation. Static answers come from
  dynamic relaxation by default, so a fibre model of a 3 x 70 m unit takes
  minutes to an hour; frames and fibre beams can use `mode: static` (Newton,
  with Crisfield's arc length, both v1.0), which solves Vecchio &
  Emara's frame in 8 s against 22 minutes, pushes the Hangzhou Bay unit past
  its peak in 30 s against 50 minutes, and traces Lee's frame through its
  snap-back, but refuses fracture, hinges, creep, diaphragms, shells and
  solids. Past a second peak where many fibres sit on their cracking and
  crushing kinks, the arc can turn back down the unloading line (case_24g
  stops before it).
- **Rayleigh `alpha` alone leaves the high modes undamped.** Mass-proportional
  damping gives 0.03% at 80 Hz where it gives 5% at 0.5 Hz, and cracking
  excites exactly those modes: on the Hangzhou Bay seismic runs the
  displacements and the base shear in the shaking direction are unchanged by a
  5 Hz low-pass, while the pier axial force swings 20 MN of which 1.2 MN
  survives it. Read reactions from a damped explicit run
  filtered, or add a small `beta` -- which shrinks the step (the bullet on the
  step bound above).
- **Fracture reaches separation on frames only.** Layers 1–3 are in place on
  the fibre beam (material latch → FAC → the element stops carrying load,
  mass kept), `steel_cyclic` supplies the criterion on both the NumPy and the
  compiled path, and `split: true` cuts the bonds so the member comes away
  and falls at g. What is still missing on the fibre beam: a
  fragment joins the region `all` and nothing else, so a gravity load
  written for a named region does not reach the debris, and `dt` is not
  adapted when a split lowers the stability bound — it is measured and
  logged, not acted on (the frame solvers' split, v0.5, carries gravity to
  the fragment and adds the stability mass it needs); and anything at all
  on the solid path, which still erodes and loses the mass with the
  element.
- **Units are checked, never converted.** Every number in a case must be in
  one consistent system; `units:` declares which, and preflight compares the
  materials' wave speed against it (`UNIT_MAGNITUDE`) so mm/tonne/MPa numbers
  in a case declaring m/N/Pa are caught before the run rather than after it.
  No quantity is rescaled.
- **CDP is still the expensive material**: 2.6 M element-steps/s against 7.5 for
  linear elastic (reduced integration, numba), because of the return map plus two
  closed-form eigenvalue solves per integration point per iteration. Reducing
  `n_return` from 5 helps.
- **First-run compile is slow**: Warp builds one module per (material,
  formulation, precision). On the 5060 Ti, 11 s to 7 minutes (b-bar Chaboche was
  the worst); then it caches and loads in ~20 ms.
- **Throughput saturates around 150k elements** on this card. Below that the
  kernel is launch- and latency-bound and timings say little.
- **Dynamic relaxation can lose a rapidly softening region**, and there is no
  arc-length control to catch it. Measured directly on Lee's frame (case_16):
  under displacement control the solver traces the limit point and the
  descending branch, and at the snap-back — where the controlled displacement
  itself folds — it jumps dynamically to the far branch and settles there,
  which is what the structure would do and not what an arc-length method
  would trace. Both NAFEMS targets lie off the fold and are met; the fold is
  what arc-length would add, and case_16 is the test it will have to pass. The RC-beam evidence this bullet used to
  carry (an onset at 26–34 mm, later load steps sitting at residuals of 1e-2
  to 3e-1) was measured on a model with two defects since removed —
  `host_deduction` on cracking concrete and an unregularised compression band
  — and does not survive them: case_12 now converges to its 1e-3 tolerance at
  every one of 20 load steps out to 19 mm. The limit is still real where the
  softening is fast enough (a snap-back, a shear failure, a column pushed past
  its hinge), and it is a property of explicit relaxation, not of this
  implementation. This is structural, and
  LS-DYNA has the same blind spot: its dynamic relaxation judges convergence on
  one global distortional-kinetic-energy ratio (`DRTOL`, default 0.001) and
  damps by scaling every nodal velocity by `DRFCTR` (default 0.995), so it
  cannot see a handful of runaway nodes either — which is exactly why the
  LSTC/FHWA concrete workflow leans on erosion rather than on better damping.
  Two attempted cures were implemented, measured, and removed rather than
  shipped: rebuilding the Underwood masses from the secant stiffness (CDP with
  `wc = 1` recovers stiffness on crack closure, so the mass bound must use the
  *stiffest* state an element can reach — the secant version went NaN within
  three load steps), and nodal kinetic damping (zeroing individual velocities
  breaks the leapfrog's momentum consistency and was worse). Arc-length or
  line-search control is the real answer and is the most valuable next piece of
  solver work.
- **`tet4` locks** and there is no nodal-pressure averaging yet; use `tet10`.
- **Bond-slip is available and is NOT the default.** Omitting the `bond` block
  keeps perfect bond, which is cheaper, runs on every backend and is right
  whenever the member is not expected to crack. Adding one gives the bar its
  own degrees of freedom and an interface: `bond_ceb` is the MC2010 monotonic
  envelope, `bond_cyclic` adds Eligehausen-type degradation and reproduces
  `bond_ceb` bit for bit on a monotonic path. A bonded bar needs `diameter`
  (bond acts on the bar SURFACE) and runs on numpy and numba; warp keeps its
  time loop on the device and refuses it by name.
- **`precision: mixed` is not safe on every model.** Its element reference
  coordinates are fp32, so the strain resolution it leaves scales with (model
  extent)/(element size). At 170 — an 8.5 m beam with 50 mm elements — that is
  1.5e-5, an eighth of a concrete cracking strain, and relaxation cannot remove
  it. Preflight measures it and warns; the fix is `precision: fp64`. The proper
  fix would be storing element-LOCAL reference coordinates on the device, which
  would make mixed precision size-independent, and that is a kernel change
  nobody has made yet.
- **A prescribed displacement is an absolute level**, and until v0.1
  a step that began with those nodes already displaced dragged them to zero in
  one iteration. They are now blended in from where the step finds them, over
  the same `stage_ramp` the level ramp uses. What is still true is the meaning:
  `value: -0.0564` is -56.4 mm from the undeformed position, never 56.4 mm
  further than wherever the model happens to be.
- **Prestress is pretensioning only, and computes one loss.** `prestress` on
  a bar assignment is an effective stress that becomes a strain offset, so the
  first equilibrium is transfer and the elastic-shortening loss falls out of
  it (verified against the closed form to 4e-5). Everything else about
  prestressed design is absent: friction along a duct, anchor set, relaxation
  is computed from equilibrium, and friction and anchor set from `mu`, `k` and
  the draw-in along a drawn tendon — but a POST-tensioned tendon is jacked
  against concrete that is already shortening, so its jack force is already net
  of the elastic shortening and applying it as a prestrain deducts that loss a
  second time; `cases/case_15` pre-compensates by the 6% it measured, and says
  so. Relaxation, shrinkage and creep are computed when a case declares them
  (the bullet above); the shipped girder cases still state an effective
  prestress, because their sources do not give the ages and environment the
  laws need (route decision D4). What is
  still missing for post-tensioning proper is **staging** (there is no way to
  raise the prestress part-way through a run, so jacking against concrete that
  is already loaded cannot be sequenced) and the **unbonded tendon**, whose
  force is governed by the elongation of the whole tendon rather than by the
  strain at a section and so needs a single global unknown this element set
  does not have. A bonded post-tensioned member's FINAL state — which is how
  design offices usually model one — is expressible today.
- **Solid and structural meshes do not mix in one file.** A `.msh` holding
  both solid and beam/shell elements is refused by name — they are solved by
  different solvers — and the `coupled` path is how a solid block takes
  shared-node Line2 members. *Several solid topologies in one file are fine*:
  each element type becomes a block, and since v0.1 the fused and GPU
  backends take them too. `hexa8`, `tet4`, `tet10`, `wedge6`, `wedge15`,
  `pyramid5` and `hexa20` are all registered (`vfife list`); the higher-order
  and transition elements support `full` only and carry less verification
  than `hexa8`.
- **Near-incompressible relaxation converges slowly** (see above), and CDP with a
  high dilation angle pushes in the same direction.
