# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses
[Semantic Versioning](https://semver.org/) from 0.1.0 on, with the usual
0.x caveat that minor versions may change the case-file schema.

## [Unreleased]

## [1.23.0] — 2026-11-27

A feature release on the 1.0 formats, and the last before a stable period:
a case written for 1.22 runs unchanged and gives the same answers (the
shipped cases and the six course bridges are the same to the bit). The
static Newton can take an increment by pseudo-time continuation when the
plain one cannot (`static.stabilise`), which brings a suspension bridge's
finished state through. The Midas converter takes more of a file: imposed
support displacements and settlement groups, and a load case's
temperatures, as steps after the bridge is built; and it reads two fields of
the creep translation as Midas' quick reference has them (a converted model
whose file uses them may change). A buckling analysis with sections set off
sideways has its reference solutions.

### Added

- **The stabilised Newton** (`static.stabilise: true`): an increment the plain
  Newton cannot take is tried again by pseudo-time -- springs of a small
  multiple of the tangent's diagonal to where each round began, slackened
  round by round until the residual without them is within the tolerance --
  before it is halved. For a tangent that is right and ill-conditioned (a
  suspension deck whose end bays barely resist twisting). Off by default
  (it would carry a snap-through onto the far branch); the Midas converter
  turns it on.
- **Midas: imposed support displacements** -- a load case of `*SPDISP`, or a
  settlement group (`*SM-GROUP`) by its name, through `convert(...,
  cases=(...))`: one more step with the supports brought to the stated
  displacement and kept. Midas' settlement cases are envelopes over
  combinations of groups; this does one group a step.
- **A reference for buckling with a section set off sideways** (two tests):
  on forks the column is the beam-column of its eccentricity to 5e-4; with
  an end free to twist its own nonlinear tangent turns singular within 0.5%
  of the factor. The manual says what is left (sections with equal second
  moments, 3.5% conservative against stiff arms).
- **Midas: temperatures of a load case** through the same `cases=(...)`:
  `*SYSTEMPER` and `*ELTEMPER` as a uniform change, `*BSTEMPER` (a section's
  layers) as the uniform change and the gradient along local z of the same
  axial force and moment. The thermal coefficients are written only for a
  model a case taken heats; other models convert to the same bytes.

### Fixed

- **Midas: the cement type of CEB-FIP 2010's modulus growth** (`*TDM-ELAST`
  iCTYPE) is numbered from 1, as Midas' MCT quick reference has it (1 rapid
  hardening high strength, 2 normal or rapid hardening, 3 slowly hardening);
  1.22 read it from 0, every type one step too slow. And the stage control's
  "Creep & Shrinkage" box (`*STAGE-CTRL` line 4, bCNS) is read: off, no
  creep or shrinkage whatever its TYPE.

## [1.22.0] — 2026-11-26

A feature release on the 1.0 formats: a case written for 1.21 runs
unchanged, and a model that uses none of the new features gives the same
answers (Newton statics to round-off, from the faster factorisation). Fibre
models take creep in static steps and connectors born in later steps, as
frame models do; Rayleigh damping can be given region by region; Bouc-Wen
is a material of the table and, on a connector's two shear components, one
law in the plane. The Midas converter follows creep and shrinkage by the JTG
codes: a bridge converted again from a file that has them is a different
model, and its results change accordingly (on the course models checked,
as close to Midas' as the elastic part is). Under the hood: one table of
what each element's models take, the bottom layer `vfife.base`,
`vfife.assemble` for building a model, and the fibre solver in four modules.

### Added

- **Fibre models creep, shrink and relax in static steps**: `solver.mode:
  static` with `to_age` now runs the clock on fibre-beam models too (one time
  station per converged increment, as in the staged relaxation, which it
  matches on case_29d and case_29g to the relaxation's tolerance). Steps of
  time under `static.arc_length` stay refused.
- **A connector born in a later step of a fibre-beam model** is born free of
  force, as in the frame models: the six deformations at that moment are its
  reference. It was refused by name.
- **Rayleigh damping by region** (`damping.regions`): the members of some
  regions take their own alpha and beta (or ratio at two frequencies), the
  rest and the concentrated masses the model's -- Midas's group damping,
  OpenSees' `region -rayleigh`. Line models, dynamic runs; such a run takes
  the step-by-step loop (core/damping.py).
- **The Midas translation follows creep and shrinkage** (`*TDM-TYPE`,
  `*TDM-LINK`, `*TDM-ELAST`, `*ELEM-DEPMATL`, `*STAGE-CTRL`): concrete of
  JTG D62-2004 or JTG 3362-2018 creeps and shrinks by MC90 at fcm = 0.8 fcu,k
  + 8 MPa, at the file's humidity, notional sizes (an element's own a
  material of its own), beta_sc and drying age; each stage builds its
  concrete at its age and runs its duration as a step of time, as the stage
  control says. Bridges converted again from such files change accordingly.
- **Bouc-Wen** (`bouc_wen`): the smooth hysteretic law of Wen (1976), a
  one-dimensional material of the table -- a connector's force-deformation
  law through `type: material`, and on a connector's `uy` and `uz` together
  one law in the plane with a circular yield surface (Park, Wen & Ang 1986):
  lead-rubber and high-damping rubber bearings.

### Changed

- **Newton statics factor the tangent about ten times faster** on large
  line models: SuperLU in its symmetric mode, on the tangent scaled to a unit
  diagonal (so the units of the case do not matter), fills about a tenth of
  what the default did. A converted rigid-frame bridge of 16,000 dofs ran its
  48 stages in 130 s instead of 314 s. Static results change at round-off.
- **One table of what each element's models take** (`elements.library.FEATURES`:
  `static`, `diaphragms`, `rigid_links`, `step_changes`, `buckling`), read by
  the schema, preflight, the analyses' refusals, `vfife describe` (a
  `features` key) and the YAML reference (a column). It replaces six lists
  kept by hand. A `connector2` model run straight through `implicit.run`,
  past preflight, is now refused as preflight and the command line refused
  it.
- **`vfife.assemble`**: `build`, `make_solver` and `_with_element` moved there
  from `vfife.cli` (which still exports them), so the Python API no longer
  imports the command line.
- **`vfife.base`**: the bottom layer -- the errors (`UserError` and the
  analysis failures), the degrees of freedom, finite rotations, `scatter_add`,
  the Voigt and tensor notations -- that elements, materials and kernels
  import instead of reaching up into `core`. The old module paths
  (`vfife.core.dof`, `vfife.core.rotations`, `vfife.voigt`, `vfife.tensor`,
  `vfife.core.diagnostics.UserError`, `vfife.core.mesh.scatter_add`) are the
  same modules and objects.

## [1.21.0] — 2026-11-26

A feature release on the 1.0 formats: a case written for 1.20 runs
unchanged. Multi-point constraints -- master-slave rigid links and the rigid
floors -- are one constraint layer that every procedure takes (the explicit
step, the relaxation, Newton statics, modes, buckling, analysis steps).
Member offsets -- rigid end zones and sections set off their nodes
(`end_offsets`) -- have their geometry and their mass each written once:
loads on the drawn line, the mass where the section is, a zone's weight,
and moving loads, vehicles, contact, buckling, breaking and removal taking
them. One-dimensional composite materials, and connector laws from the
material table. The Midas converter translates rigid links, beam end
offsets, section offsets and the cable-pretension switch. Results of models
that use none of these are unchanged, apart from round-off in Newton runs
with rigid floors (their masters are at the floor's centre of mass now);
converted Midas models change.

### Added

- **Master-slave rigid links `rigid_links: [{master, slaves, dofs, arm}]`**
  (Midas' `*RIGIDLINK`, OpenSees' `rigidLink` and, with `arm: false`,
  `equalDOF`): each slave's tied components, six flags in the global axes,
  follow the master as a rigid body in large rotation, the arms taken where
  the nodes stand. Imposed on the acceleration (the master accelerates under
  its slaves' tied forces and the moments of their arms, through their
  masses) and set exactly after each position update; the relaxation's
  residual is the master's. Frame, fibre-beam, shell and shell-line models;
  explicit dynamics, dynamic relaxation and the modal and buckling
  analyses, and the static Newton solver on the models it runs (the master
  system T^T K T, T at the current configuration). The compiled step and relaxation decline it;
  solid, truss, coupled and warp-shell models refuse it for now. Preflight
  takes a link as joining its nodes (a slave on its master's position is no
  duplicate node). **In analysis steps**: `steps[k].rigid_links` are made at
  the step's start, after its members are built, free of force where their
  nodes then stand; a step's `release` may name a link, which is freed with
  the force it carried at the end of the step before taken off over the
  step, as a released support's reaction is. The checkpoint carries which
  links tie and their forces; `vfife modal` and `vfife buckling` ask such a
  model at the result's last state only.
- **Composite one-dimensional materials `parallel` and `series`, and `gap`**:
  `{type: parallel, of: [a, b], factors: [...]}` puts laws side by side
  (one strain, the stresses added), `{type: series, of: [a, b], max_iterations,
  tolerance}` end to end (one stress, the strains added; the split is found
  at each point by a bracketed Newton iteration, and a split that does not
  converge is refused as `SERIES_NOT_CONVERGED`); `{type: gap, E, lo, hi}` is
  free between two strains and linear elastic past them. The parts are
  names in `materials` (or specs written in place), any one-dimensional law
  including another composite; a `linear_elastic` part is E eps. They are
  taken by a standalone `truss2`, embedded bars and fibre sections (on the
  NumPy path). A part written in place takes the case's unit defaults; the
  density adds with the factors side by side and is the parts' mean end to
  end (`rho` sets it); a series has failed when a part has, a parallel when
  every part has, and a series whose part ruptures goes on carrying next to
  nothing. Checked against the closed forms, the connector's `bilinear`
  law, and OpenSees' Parallel, Series and ElasticPPGap
  (`tests/data/opensees_composite.npz`, `tools/opensees_reference.py
  --composite`).
- **A connector component takes a law of the material table**:
  `laws: {ux: {type: material, material: <name>, area, length}}`, the force
  `area` sigma(d / `length`) (both 1 when left out: the table is then the
  component's force-deformation relation). Any one-dimensional law, the
  composites included: a buckling-restrained brace's core, a stop with
  slack. The law's state is the solver's (committed at equilibrium or every
  step, cleared at the connector's birth, in the checkpoint as
  `connector_material_state`); its `connector_slip_*` reports the material
  strain; the reference stiffness is `area` E / `length`. The compiled
  connector kernel declines it, so such a model's connectors run in Python
  and its compiled step keeps the step-by-step loop (the fibre model's
  compiled step now refuses a connector with no compiled form, which it
  would have carried as nothing).
- **`end_offsets` on beams** (`beam2`, `beam2_timoshenko`, the beams of
  `mixed_line`, `beam2_fiber`): `{i: [x, y, z], j: [x, y, z], axes: local|global}` puts
  the ends of a member's elastic part at fixed vectors from its line's first
  and last nodes; the stretch between is rigid and turns with its node. An
  offset along the line is a rigid end zone, one across it a section set off
  its nodes (linear along the line from `i` to `j`). The forces reach the
  nodes with the moment r x f, and the reference stiffness (the time step,
  the relaxation masses, the modes) holds the rigid stretch. The rigid
  stretch across the member has no mass; a rigid zone weighs as the member it
  is part of, at its middle; the elastic part's mass is carried where
  it is, at its ends (`CarriedMass`: the parallel-axis inertia, the coupling
  of translation and rotation, in the explicit step and the modes), and
  its weight and a ground motion's inertia act there. Loads are on the drawn line (a
  `line_load` per unit length between the nodes, a moving load's or a
  vehicle's station along it): a rigid zone's share goes to its node rigidly,
  the elastic part's with the moment of the lever across. Members with
  offsets break, split off and are removed as any other (the rigid stretches
  follow their ends), their contact capsules are on the elastic axis, and a
  buckling analysis takes the stretches and their levers. The compiled step
  and relaxation take such a model step by step. Models without offsets are
  unchanged to the bit. `per: member` gives every member of a region the two
  offsets. `axial: full` takes a frame beam's axial and torsional stiffness
  over the length between the nodes (Midas' rigid zone, rigid in bending and
  shear only; the converter writes it); `elastic`, the default, over the
  elastic part. All of it goes through one object, the solver's `geometry`
  (`core/member_geometry`).
- **The model page (`vfife view case.yaml`) draws sections at their size**
  with a switch: boxes of each section's depth and width on the member's
  local axes, between its elastic ends; the rigid stretches of end offsets as
  lines.
- **The Midas converter translates more of what a model says** (each to
  vfife's own, general way of saying it): a TENSTR element is a
  `tension_only` bar, and a cable (iSUB 3) keeps the unstressed length or
  tension on its own row (`unstressed_lengths`, `prestress`); `*SECT-SCALE`
  multiplies the section's properties (the weight factor by a material of
  the section's own); member loads varying along the member, projected, or
  in the member's own axes are split to its two ends by the lever rule
  (they were averaged, read as unprojected, or left out). Partial and
  eccentric member loads are counted as not read (they were read as
  uniform).
- **The Midas converter reads `*OFFSET`** (beam end offsets: TYPE=ELEMENT
  rigid zones along the member, TYPE=GLOBAL vectors from the nodes) as end
  offsets. In a model with stages an offset is there when a stage activates
  its boundary group, as a support is (one in no group is not). A zone
  carries no weight of its own; different zones for the two bending planes
  take the one for My, and say so.

### Changed

- **The Midas converter writes `*RIGIDLINK` as `rigid_links`** (the flags
  as Midas writes them): a saddle that slides on its tower (011111) and a
  stay hung on its tower or deck by its translations (111000, 111010) are
  translated, where they were left out, and a link tying all six is no
  longer a stiff weightless beam (the "a multiple of the model's modulus"
  rule stays for the elastic links of kind RIGID). Made in the stage their
  boundary group comes on, or that builds the last of their nodes, and
  released where the group goes off (by name, in the step's `release`); a
  link written again in the group that follows is one link kept. Left out
  and counted: a slave no member reaches, one a support holds in a tied
  direction, and a chain whose links do not both tie all six and live as
  long. On the course models: the suspension bridge's equilibrium model
  (2.5) now converges, its staged model (6.2) passes the main cable's stage,
  and the cable-stayed bridge's 96 steps converge with the pretensions as
  the stage control says (`jack_which="one"` no longer needed there). The
  preflight takes a step's links as joining the model, and a link of some
  directions as holding its slave in them in the rigid-mode check.
- **One constraint layer for the multi-point constraints** (`core/constraints`):
  the rigid diaphragms were written three times -- the acceleration
  projection of the explicit step and the relaxation, the Newton system's
  transformation, the modal and buckling analyses' -- each with its own
  check that a floor is not also held, and they hung on the frame solver,
  where the fibre-beam solver (a sibling, not a child) could not reach them.
  They and the rigid links are now one class each behind one interface
  (`project`, `correct`, `residual_norm`, `newton_map`, `free_map`,
  `check_free`, checkpoint), held by the beam family's solvers as `ties`
  (`constraint_layer()`); each procedure takes the interface, not the kind.
  No answer changed in a bit (the floors' arithmetic is each procedure's
  own, as it was). A floor's in-plane dof held in a step is refused with
  one message in every procedure.
- **Rigid diaphragms on fibre-beam models, and one map of the multi-point
  constraints for Newton, modal and buckling** (`ConstraintLayer.map`):
  the fibre-beam solver takes the floors (they were refused by the
  preflight, and ignored by `vfife modal` and `vfife buckling`, which do not
  run it). The Newton system's floor masters sit at the floor's centre of
  mass as the projection's and the modal analysis' do (they were at the
  nodes' mean point: the same constraint, round-off -- 1.4e-13 on a storey
  with unequal corner masses). A reduced mass T^T M T with a massless
  direction -- a point mass on a rigid arm gives its master m [[1, a],
  [a, a^2]], singular with no zero on its diagonal -- is taken in its own
  axes, where that direction is condensed as any massless dof (handed to
  the generalised eigenproblem whole, the factorisation failed); one with
  mass in every direction is solved as before, to the bit. The buckling
  analysis decides which dofs are live on the reduced system T^T K T: a
  link master with no stiffness of its own was dropped, locking its slaves
  (found in review).
- **The Midas converter holds a cable's pretension as the model's
  construction-stage control says** (`*STAGE-CTRL`, CPFC): External, and a
  model that does not say, is a `jack` on every element given one;
  Internal is each element cut to the length of its force (`prestress`).
  The report's notes say which. Before, every model took one rule: of a
  cable over sliding saddles one element was jacked and the rest cut to
  length. That rule is now asked for by name, `convert(...,
  jack_which="one")` (or `"ends"`), whatever the switch says, and the notes
  say so; `vfife init` always follows the switch. Models without a pretension convert as before. Two
  with External control change: the real arch, whose cables run over soft
  saddles, and the cable-stayed bridge, whose side- and main-span stays
  meet at tower nodes held only by partial rigid links, which are not
  translated yet; `jack_which="one"` gives them what they had.
- The manuals' Midas section says what the last 3% between this translation
  and Midas on the rigid frame is: Midas' `bCALC-CSP` switch counts the
  ordinary bars, the ducts and the grouted tendons in the section
  properties stage by stage; the translation uses gross sections.
- **The Midas converter writes a section set off its nodes (CT/CB) as end
  offsets** of each member, not as a centroid node under each drawn node
  and a rigid arm (`offset_arms`). The girder stays on its drawn nodes, as
  in Midas. Results of converted models change: no arm's own give, no
  centroid node's mass, and a pier drawn at the girder's centroid is joined
  to it only where the model joins them (it used to be merged). `offset_arms`
  is still read. The compiled step and relaxation take such a model step by
  step.
- The manuals' known limits no longer say construction stages are planned
  after 1.0 (they came in 1.2); they say what stages cannot do. Section 6.2
  says which steel law to use for what, names `concrete01`'s unloading,
  and gives `fy` beside `Fy` and `yield_strength`.
- `tools/state_baseline.py --tolerance` judges a case loaded to collapse
  (`case_24c`, `case_24d`) on every monitor at every frame up to the
  collapse, not on the state after it. `--write` keeps every case's
  monitors; `--backfill-monitors` adds them to an older baseline after
  checking that the baseline's own commit gives its answer to the bit.
  Strains are now a class of their own: they were held to a floor set by
  the largest stress and so were not judged at all.

### Fixed

- **Removing a line-loaded fibre member in an analysis step** no longer stops
  with an AttributeError: the load it takes with it is put together by the
  same rule as the step's own line loads (`member_line_load`).

## [1.20.1] — 2026-11-24

A maintenance release on the frozen 1.0 formats, so that the repository
holds no third-party code: the three laws that were ports of OpenSees are
rewritten from their papers. No new solver features; one new spelling
(`fy`). Results of cases that use these laws change at round-off. One case
loaded to collapse (`case_24d_hzb_girder_piers`) agrees with 1.20.0 to
1e-11 up to the collapse at t = 54 s and departs from it after that, as any
post-collapse motion does under a change of round-off (about 1% in the
midspan deflection at the end of the run).

### Added

- **`fy` is the yield strength in every law that has one**: `steel02`,
  `rebar`, `j2` and `chaboche` take it beside their own name (`Fy`,
  `yield_strength`), as `steel_cyclic` already did. Giving both with
  different values is refused (`MATERIAL_ALIAS_CONFLICT`).

### Changed

- **`steel02`, `concrete01` and `concrete02` are written from their papers.**
  They were ports of the OpenSees source, whose licence does not allow its
  code in an MIT repository. Now: Menegotto-Pinto with Filippou, Popov &
  Bertero's isotropic hardening (UCB/EERC-83/19, Eqs. 3.4-3.8); Kent-Scott-
  Park with Karsan-Jirsa unloading; Yassin's concrete with tension softening.
  Where a paper leaves a detail open it was fixed against the OpenSees
  program as a black box. Names, parameters and state layout are unchanged,
  so cases and checkpoints carry over; results differ only at round-off
  (the laws give OpenSees' curves to 1e-14 of the strength over random
  strain histories). A relaxation can therefore come to rest at a slightly
  different residual: a concrete02 fibre where its tension branch meets the
  compression lines makes a small hysteresis loop, and one staged fibre-beam
  test now rests at 4.4e-5 instead of below 1e-5. `tools/opensees_reference.py`
  writes the reference curves, `tests/test_opensees_reference.py` checks
  against them, and the test's own transcription of the OpenSees routine is
  gone.
- The repository is public at https://github.com/maolin2000/vfife
  (CITATION.cff follows). It no longer ships `results/` or `docs/archive/`,
  and earlier entries name versions instead of internal development stages.
- README and both manuals state the division of labour: meshes come from
  the case file, Gmsh or the Midas converter, heavy post-processing is left
  to ParaView, and `vfife view` stays two read-only pages.

### Fixed

- `vfife view case.yaml` draws a dot at every member's nodes: a node only
  rigid arms reach (an offset arm's drawing node, a cap beam's end) had none,
  and a connector's filled square and triangle covered its ends' dots; they
  are open now. A tendon's own nodes are still not drawn.
- tests/test_backends.py skipped the whole file without warp-lang, so its
  numba checks never ran on CI; only the tests that need warp are skipped
  now, and the cross-element comparisons run numba alone.

## [1.20.0] — 2026-11-23

### Added (cases people can write and read)

- High-level blocks of the input format, expanded on reading into what the
  solver reads (`vfife expand case.yaml` writes the expansion). Format 1
  only gains optional keys: every case of 1.19 reads as before.
  - `tendons:`: a frame model's tendons, one entry each --
    line, area, jacking, friction, draw-in, relaxation -- in place of the
    tendon nodes, arms and jacked bars written out one by one.
  - `groups:`, `include:` and tables: a group made of mesh
    groups, usable wherever a region is named (construction stages name
    them in `activate`, `deactivate` and `release`); materials, sections,
    tendons and groups merged from other files; tendon lines, loads and
    nodal masses read from CSV files. schema.json accepts the table entries.
  - `tendon_defaults:`: what every tendon has unless it
    says otherwise, merged key by key; a tendon profile table may hold all
    tendons, one row set per tendon name.
  - `offset_arms:`: one rule for the rigid arms of members
    set off their nodes; each arm's section is worked out on reading from
    the members it carries.
- The Midas converter writes these blocks: the cast-in-place continuous
  rigid frame of the tests went from 68,122 lines to 1,317 (plus its
  tendon profile CSVs), with the same state to the bit as the long form
  (results of the converted cases change at round-off, from the arms'
  sections now worked out by rule).
- A result's tendons: the result file records each tendon
  of `tendons:` and its bars (`tendons/` group), and `vfife tendons
  result.h5 [--bars] [--csv file]` tabulates the force in each tendon at
  the end of each step against its jacking force.
- `vfife view case.yaml`: the model before it is run,
  as one web page -- the high-level blocks expanded, step by step: the
  members standing (coloured by kind, section, build step or region), the
  supports, connectors and soil springs, the loads as arrows, a member's
  section and steps on a click. A support or connector is one symbol with
  a six-character code of what each dof or component is (`111000`;
  `KKG000`: a spring, a gap, a free component), drawn dof by dof and
  component by component when clicked; switches for each kind, a symbol
  size, a region filter. Frame models only.

### Changed (results of converted Midas models with tendons)

- The converter's tendons are expanded on reading by the code of the
  `tendons:` block, with these differences from 1.19:
  a tendon's arms are built in the step that builds the first member at
  their point (they followed the tendon's own elements); a tendon without
  `stress.step` is stressed in the step its group's last member is built;
  the arms are weightless tendon steel; the jacking force is not rounded;
  the length thresholds follow the case's units. On the continuous rigid
  frame of the tests the side-span closure's vertical displacement against
  Midas went from 1.0275 to 1.0352; the other stages, and the short and
  long forms of each converted bridge, are unchanged to the bit.

### Changed (results: shell_line staged relaxation)

- The shell-line solver rebuilds its relaxation mass from the current
  geometry at each relaxed load level, as the frame, mixed and shell
  solvers do: it never did, so a shell with beams relaxed
  in stages kept the reference rows and could stop converging (a shell
  cantilever with two beams failed at the 8th of 10 levels, and now
  converges). Only shell_line cases relaxed in stages change; no shipped
  case is one.

### Fixed

- The title of a result page (`vfife view result.h5 --title`) is written
  as text: a title with markup in it was inserted into the page.

## [1.19.0] — 2026-11-21

### Fixed (restarts and staged runs)

- A frame-solver checkpoint (beam2, Timoshenko, and the mixed, shell and
  coupled solvers through them) now holds the parked particles and the
  appendages -- the massless members a split leaves carried rigidly by one
  particle. A restart after a hanger was cut let the
  hanger's joint and the crossbeam's tip go free: the girder of the tests
  ended 3 to 6 cm from where the uninterrupted run left it on a 10 cm
  swing, and a joint parked without spare particles (`split: false`) went
  to 1e18 m. Both now resume to the bit.
- The fibre-beam checkpoint holds `weightless_mass` (a split's stability
  floor, which a park zeroes) where a split or a park can have changed it:
  a fibre girder with a `with_mass` line load restarted across a split
  ended up to 5 mm from the uninterrupted run.
- A dynamic analysis step that began after a park or an appendage took its
  stable step from those particles, which have no mass of their own and
  are not integrated: 1.8e14 steps for 0.07 s, a run that never ended. They
  are out of the bound (the stiffness's and the connector dashpots'). A
  restart inside a dynamic step now keeps the step the run was on (`dt`,
  `n_steps`), saved in the checkpoint as `step_time`, unless the restart
  raises the step's duration.
- A fibre-beam run stopped inside the staged level a strand breaks at, past
  the rupture strain in the trial, and restarted -- or started from that
  state with `initial_state`, or a static run stopped inside its relaxation
  fallback -- did not solve the level again: the commit found no new
  failure, and the frame paired the intact positions with the cut forces
  (372 kN where the uninterrupted run reads 294 kN on the chain of the
  tests). The failure count is of the accepted (committed) state now.
- An unnamed step is no longer reported as "step None" in two error
  messages, and two unnamed sets of embedded reinforcement are no longer
  merged under one key (the input format always names them).

### Changed (results: staged relaxation)

- The frame (beam2, Timoshenko) and fibre-beam solvers rebuild their
  relaxation mass from the current geometry at each relaxed load level, as
  the mixed and shell solvers have since v0.1. The
  same pure-beam model relaxed in stages took a different path through
  `beam2` and through `mixed_line` (case_27 relaxed in 10 levels: 1.2e-4 in
  q, now 4e-15), and a beam2 cantilever that turns far in a staged
  relaxation now converges where it stopped at its iteration limit. Only
  relaxation changes (staged levels, relaxation steps, the static run's
  fallback, the creep stations); dynamic and Newton runs are the same to
  the bit. Of the shipped cases case_24f, case_11 and case_21b change, each
  within the tolerance it relaxes to and passing its contract; one test's
  bound (a dead state released into a dynamic step) was a peak to peak
  held to one tolerance and is now held to two.

- One relaxation-mass refresh for the frame, fibre-beam and mixed solvers:
  each member's block in the reference K is replaced by
  the same block turned into its current frame, and everything else K
  holds stays. The mixed solver rebuilt the rows from its members alone,
  so a soil spring's diagonal dropped out of its nodes' mass after the
  first staged level; K's members are now tracked block by block, so a
  member built while stiffness damping was off, or split while it was off,
  gets its rows (both made a staged relaxation non-finite on the frame
  solver, and on the mixed one once it took this refresh). The coupled
  solver kept its solid's relaxation mass only until the first refresh: a
  staged relaxation of case_07 went non-finite after 7 iterations, and now
  converges. Of the shipped cases case_16, 23, 24e, 29 and 29b change, at
  round-off.

### Changed (results: the energy ledger only)

- The frame solvers book the viscous or Rayleigh damping work of a step on
  the step's own velocity, before the step's splits and appendages change
  it, as the fibre-beam solver and the compiled step do.
  Only damped frame runs with splits change, and only in the ledger
  (case_34: 3e-4 J of 1.9e5 J dissipated); the motion is the same to the bit.

### Added

- `tests/test_restart_across_versions.py`: checkpoints written by 1.18.0
  (a frame dynamic run, a mixed-line dynamic and a Newton run, stopped half
  way) resume on this version to the state 1.18.0 ended in.
- `tools/state_baseline.py --tolerance`: a case that differs from the
  baseline is judged on its final state, quantity by quantity (a table of
  tolerances per class of quantity, floors set by the size of the class,
  a relaxed case held to ten times what it relaxes to), and reported
  WITHIN instead of DIFFERS; every compare reports the cases more than 20 %
  and 5 s slower than the baseline (`--perf-gate` fails on them).

- `tools/state_baseline.py`: every shipped case solved as
  `tools/check_all.py` solves it, and the SHA-256 of every dataset and
  attribute of its result file and checkpoint kept (`--write NAME`), to
  compare a refactoring with to the bit (`--compare NAME`, which says by how
  much a dataset that differs differs). Attributes that name the run (its
  id, date, version, paths) are not compared.

### Changed (internal; no result changes)

- P2 of the architecture review: the frame and fibre-beam solvers share one
  body (`core/beam_family`) for the step data, the moving loads and the
  age groups, the convergence measures, the probes, the dynamic step and
  its end, the event stop, the checkpoint and the restore, the run loop
  and the output frame. What differs between the two
  is a hook one of them overrides; the methods both define went from 32 to
  20, and the 20 left are the two section models' own physics or rules
  that differ on purpose.
- More of the solvers' reads of the case go to the typed case: units,
  amplitudes, removals and four presence checks. The rest
  stay: the dict a solver reads is the validated case dumped whole, so a
  default written beside a read of a schema field is never taken in a run.
  `tests/test_problem.py` counts the reads of a solver's
  case (68) and fails if one is added.

- P2 of the architecture review, first step: a solver is
  built from a `Problem` (`vfife/core/problem.py`), the validated case typed
  (`problem.case`, a `schema.Case`) beside the plain dict every reader read
  until now (`problem[...]`, read-only). `Problem.of` validates a dict
  handed to a solver directly, so every solver has a valid case: a
  hand-built dict the schema refuses is refused when the solver is built,
  with the schema's message (a script building solvers from such a dict
  sees pydantic's `ValidationError`, a `ValueError`, instead of the
  solver's own). The composite solvers (coupled, shell-line, connector)
  derive their halves in both forms. The readers move to the typed case in
  the steps that follow; `tests/test_problem.py` counts the dict reads
  (213) and fails if one is added. Same answers to the bit.
- Rayleigh damping stated by `ratio` and `frequencies`: alpha and beta
  beside them are accepted when they are the ones computed from them (a
  validated case validated again); any other alpha or beta is refused as
  before.

## [1.18.0] — 2026-11-17

### Changed (results of fibre-beam runs whose members can split)

- A fibre member that fractures with `split: true` now separates through
  the frame solvers' code (`core/beam_family`, P1 of the review):
  - the energy ledger is relabelled, not charged, when an end moves onto its
    spare particle: the spare's displacement jumping from zero to its
    parent's was booked as work. On the RC girder of the tests the energy
    error at the end of a 1.5 s fall was 1.7 %, now 0.1 %; on the generic
    debris truss with fibre members (60 m, 38 splits, 10 s) 11 %, now 0.4 %.
    Diagnostics only: the motion does not change for it;
  - a piece the fracture (or a removal, where members can split) set loose
    from every support gets one rotary inertia on all three axes, the
    largest, with the angular momentum kept -- 1.17's rule for frame
    debris, which the fibre beam did not have. This changes the motion of
    the debris; the structure still on its supports is unchanged;
  - a particle's mass never goes below zero when a share leaves it;
  - the viscous or Rayleigh damping work of the step a split happens in is
    booked on the step's own velocity, as the compiled step books it.
  - a connector (a bearing) left holding nothing up -- one of its ends no
    support and with no fibre member left on it -- is removed (cause 2 in
    `member_failures`), and a particle left with no member, no mass and
    nothing but connectors on it is parked. A girder member
    fracturing over its bearing left the bearing's particle on the
    connector alone with no mass: the step bound fell to 5e-16 s and the
    run went non-finite. The frame solvers' rule since v0.5;
  - a parked particle is held where it stopped (a load written on its node
    pushed it to 1e22 m/s), and the parked particles are in the checkpoint;
  - a massless member is not taken for holding nothing up while its end
    carries a nodal mass (both beam solvers; the frame solvers removed the
    bearing under a lumped mass once the girder member there broke);
  - in a staged run the spare particle a member's end moves onto is no
    longer held as a node nothing built reaches: the fragment's end stayed
    clamped where it broke while the rest fell (both beam solvers, since
    analysis steps came in, 1.2);
  - a node a staged run has not built yet is no debris (both beam solvers:
    the frame solvers' rule of 1.17 counted it as one when a split or a
    removal came during a staged run, and raised its floor inertia);
  - the warning that dt exceeds the stability bound after a split is given
    only when the bound fell: it came at every split, the run's dt being
    the bound itself to within a rounding.
  The capacity rules of 1.17 (a member twisted a quarter turn, a free piece
  at twice its capacity) stay with the frame solvers: a fibre member has no
  capacity criterion. Runs without `fracture.split` are unchanged to the
  bit; with it, a run changes where a member splits, or where a removal
  sets a piece of the frame loose.

### Removed

- The environment switches `VFIFE_COMPILED_BARS`, `VFIFE_COMPILED_CONNECTORS`
  and `VFIFE_TANGENT_MEMBERS` (P1 of the review): each turned a
  path checked against its reference back to the old one. The reference is
  `backend: numpy` for the bars and the connectors, and the whole-force
  tangent for every model the members' tangent does not take; the tests
  compare them as before. `run_case` warns when one of them is still set.
  `VFIFE_COMPILED_STEP=0` stays, for diagnosing.

### Changed (internal; no result changes)

- The architecture review after 1.17 found features
  attached to a solver by replacing its methods on the instance, and the
  drivers reading the solvers' private state. Its P0, first two steps:
  - contact between members and contact with rigid surfaces are solver
    *components* (`vfife/core/components.py`) the solver
    calls after its own part (each class's own force is `_member_forces`,
    the truss solver's `_bar_forces`); no method is replaced on a solver
    any more, and the compiled step tells the contact it runs from a
    component it would step past by type;
  - the methods every driver calls are public:
    `internal_force`, `commit_state`, `step_data`, `one_dynamic_step`,
    `after_dynamic_step`, `compiled_pending`, `compiled_event`,
    `compiled_event_stop`, `apply_pending_splits`, listed in
    `vfife/core/protocol.py`. **A script that replaced one of them under its
    old private name (`solver._internal_force = ...`) is refused by
    `run_case` with the new name**: the wrapper would no longer run.
  - the state the analysis steps keep is one object, `solver.staging`
    (`vfife/core/step_state.py`), beside `solver.plan`;
    a solver whose steps change nothing holds a read-only `NO_STAGING`
    whose fields are what the old probes answered, so no reader asks
    whether the state exists (70 `getattr`/`hasattr` probes fewer).
  - the compiled step no longer assembles its kernel arguments from the
    solvers' private arrays: the solvers give them (`kernel_members()`
    returning `core/protocol.KernelMembers`, the fibre solver's
    `fused_kernel_section()`, `kernel_connectors()`,
    `fracture_kernel_arrays()` and others). The private
    attributes the compiled step reads went from 81 to 29.
  Same answers to the bit (independent reviews compared every contact
  combination with 1.17.0).
- P1 of the review, first step: what the frame solver and
  the fibre-beam solver share is in `vfife/core/beam_family.py`
  (`BeamFamily`, the base of both): the regions and monitors, the member
  frame, vehicle loads, the frame time. The four copies of the monitor
  record are one, `vfife/core/monitors.py` (each solver module still
  exports the name `Monitor`; its fields after `dof` are keyword-only, as
  the four copies had them in four orders). An `element` monitor on a
  standalone truss with no region now reduces over every bar, as on the
  other solvers, where it failed at setup. A vehicle on fibre members now reads end
  releases as one on frame members does; fibre members have none, so
  nothing changes. Same answers to the bit.
- The solid solver's `_run_step` is split the same way: the
  step's preparation and iteration budget, the loading's progress, the host
  iteration (forces, ledger, leapfrog of the host and the bonded bars), the
  residual at a progress check, kinetic damping -- complexity 123 down to
  57. Bitwise the same on cases 04, 10, 12b, 13 and 19.
- `InputCase.lower` (the case file's input form lowered to the internal
  one) is split into the parts it branches into -- the continuum, the line
  sections, one or several fibre sections, the plain line sections, the
  reinforcement: complexity 135 down to 27 at most. The
  lowered case and its provenance are the same for the 1028 input forms the
  tests write and the 69 case files in the repository, errors included.
- What takes the compiled step, and what keeps the step-by-step loop, is
  one table, `vfife/core/capabilities.py` (P1 of the review):
  every reason `solver.compiled_step_refusal` and
  `solver.compiled_relax_refusal` give (40 of them) is a row there, the code
  asks it for its words, and the two manuals' tables are generated from it
  (`python tools/capability_tables.py`; `tests/test_capabilities.py` fails
  while a manual is stale, or while a reason is written in place). The
  words are as they were.
- Tests: on a Linux machine with AVX-512, `conftest.py` pins OpenBLAS to
  its AVX2 kernel before NumPy loads. The SkylakeX kernel rounds the NumPy
  reference's matrix products differently (1e-15), and four tests failed
  there: a numba-against-NumPy bitwise check, and two fibre creep runs in
  which a relaxation stage then stalls at residual 0.7 -- a convergence
  that depends on round-off, noted for a look of its own.
- `tools/check_all.py --log FILE` prints its progress to the console (each
  pytest layer every 5 percent and once a minute, each case as it ends,
  with the time) while keeping everything in the file.
- `tests/test_solver_boundary.py` holds the boundary as a ratchet: private
  attributes of a solver used from other modules (216 in 1.17, now 133),
  methods replaced on a solver (5, now 0), probes of private names (447,
  now 370), tests reading a solver's private state (181, now 136), and
  methods the two beam solvers each define under the same name (38, now
  32) -- none may grow, and each must be written down when it shrinks.

## [1.17.0] — 2026-11-15

### Fixed

- Debris after a progressive collapse gained energy without bound once on
  the ground: a real steel truss losing an end post threw a dozen pieces
  off at 1e13 J after about 1.2 s. Found on a generic 60 m
  truss (`benchmarks/debris_truss.py`):
  - the rotary inertia, its stability floor included, is a diagonal fixed
    in global axes and the step has no gyroscopic term; a tumbling fragment
    carried its stiff bending onto the axis its small torsional inertia had
    set (omega dt 5.5). On debris -- particles joined to no support -- a
    particle's three rotary inertias are now the largest of them, raised
    with the angular momentum kept, when some member can split. The
    structure still joined to a constraint is unchanged (bitwise, for runs
    in which nothing breaks); a model held up by contact alone is all
    debris at its first break;
  - a member twisted past half a turn end to end, which the co-rotational
    frame cannot follow: a breakable beam twisted a quarter turn is now
    removed (cause 3 in `member_failures`) -- a guard, torsion is still not
    checked against a strength;
  - a piece broken at both ends was never judged again: it is removed once
    its demand reaches twice its capacity (a judgement, not a criterion).
  The generic truss that ended at 1e14 J now keeps its energy to 0.3%
  (structural damping 1.0), 8% (0.3) and 6% (none), the misses being the
  strain energy of removed members, which is not booked. The real truss
  now runs its 5 s with bounded kinetic energy; three bearing joints left
  on nothing but their connector are still thrown off there (known, not
  yet explained; manual chapter 16). Frame and mixed-line solvers only:
  fibre-beam debris is unchanged. CHANGES THE RESULTS of dynamic runs in
  which breakable beams twist a quarter turn or members split or are
  removed: a first version made the floor the same on all three axes
  everywhere, which halved a beam's torsional frequency in a modal
  analysis (caught by the full check) and was taken back.

## [1.16.0] — 2026-11-15

The end of the Midas translation's checks on the course's real bridges
(kept outside the repository), and the first of time: the implicit static
solver runs the concrete clock, prestressed and jacked bars relax each on
their own clock, and design-code formulas live outside the solver, in
`vfife.codes`, as the curves a case file carries. Tendons translated from
Midas lose to both anchorages' draw-in together and relax as the model's
code says; on a rigid frame of 132 tendons their force now agrees with
Midas' to 1% (it was 3.5%). Several corrections CHANGE THE RESULTS of a
translated model with tendons, and one of a staged model whose new segments
are reached through a stay; each is below. That rigid frame's deflections
on Midas' dead load alone, 6% larger than Midas' when the comparison leaves
the tendons out, are 3% larger with them in as unstressed bars: Midas
evidently counts the grouted tendons in the girder's section (by hand, its
own moments over its own gross sections give deflections the tendon-free
translation matches to 0-3%, and Midas is 1-6% stiffer than that). About 1%
of what is left is a heavier load here; about 2% remains, most likely the
sections' ordinary bars, which Midas may count (not confirmed). The tendons'
1-3% larger eccentricity is most likely the same transformed section. Creep
and shrinkage are still not translated. A progressive-collapse
run of a real steel truss found an open problem, now in the manuals'
limitations: free debris can gain energy without bound after the pieces
land (after about 1.2 s there); read `relative_energy_error`. The case-file
and result formats are unchanged; a checkpoint's `relax_t0` is now per bar
(older checkpoints still restore).

### Added

- The implicit static solver runs the concrete clock on the frame path:
  every converged increment of a step with `to_age` is a time station, as
  every level of a staged relaxation is, so creep, shrinkage, ageing and
  the relaxation of prestressed bars run in `mode: static` (the same
  answers as the staged relaxation to 1e-9, and the closed forms). Fibre
  models, and arc-length steps of time, are still refused.
- Relaxation on each bar's own clock: a bar `jack`ed in a step relaxes from
  the age its step locks it off, from the strain the jack left; a bar with
  `prestress` born in a later step relaxes from its birth. Both were refused.
  Checkpoints keep each bar's clock.
- A relaxation `curve` may be read `interpolation: linear` (in t) as well as
  in log t.
- `vfife.codes`: design-code formulas as the curves the solver reads, one
  module per code (`jtg3362`: the relaxation of strand). The solver imports
  none of it; a case file carries the curves.
- Midas translation: tendons whose `*TDN-PROPERTY` says they relax (the
  JTG method) get, bar by bar, the relaxation curve of their stress after
  anchoring, and a staged model with them runs each stage's duration after
  the stage as a step of time (the map's `stage_steps` gives the step each
  stage ends with). Checked against Midas' tendon loss table to four figures.
  CHANGES THE RESULTS of a translated model with relaxing tendons.

- `vfife view --scripts DIR` writes three.js (r128's `three.min.js` and
  `OrbitControls.js`, from DIR) into the page, which then needs no network:
  where the two CDNs are not reachable the page stayed dark.

### Changed

- `vfife.io.midas_case` is a package now (`convert`, `prepare`, `tendons`,
  `shapes`, `geometry`) instead of one 1,900-line module. Imports and
  output are unchanged: the converted case files of every Midas test are
  byte-for-byte the same.

### Fixed

- `vfife view`: a page drawn without real section outlines (every page the
  command line makes; outlines come only from the Python API's
  `profiles`) stopped on its first frame with a script
  error and stayed black, whatever the network: every member part carries
  an outline array, empty there, and the page read the outline table it
  did not have. Found opening a collapse; a test now runs
  the page in a browser when Playwright and three.js are at hand
  (`VFIFE_THREE_DIR`).
- Midas translation: a tendon's arm bends as the cantilever it is (its node
  turns freely on its bars): it was sized as an arm held at both ends and
  gave a seventh of the bar's stretch at an anchor. CHANGES THE RESULTS of
  a translated model with tendons, slightly.

- Midas translation: a tendon jacked at BOTH ends loses to its anchorages'
  draw-in on the force both jacks leave, up to the point where the two
  friction curves meet, which does not move; a draw-in not taken up by
  then drops that half evenly (JTG D62's l_f > l/2). Each end was drawn in
  on its own friction curve over the whole tendon and the larger of the
  two kept, which lost too little on a short tendon: a rigid frame's 17 m
  closure tendon now has the stresses Midas and a hand calculation give
  (within 0.4%), and the cantilevers' prestress force went from 3.5% over
  Midas' to 0.7%. CHANGES THE RESULTS of a translated model with tendons
  jacked at both ends. The two draw-ins are solved
  together: each moves the tendon from its end back to the point that stays
  still, so unequal slips or friction on one side move that point instead
  of leaving a step in the force (12 of the rigid frame's 132 tendons and
  34 of the continuous girder's 116 had one).
- A segment born with `birth: tangent` is no longer placed by a cable
  that is cut to length (`prestress:`) or tension-only and reaches it from
  a part already standing. Such a cable carried the new nodes along to
  where it would be unstrained, so a segment hung from a stay was born
  off the deck it joins; now it is placed on the deck like any segment
  and the stay takes up its strain from there; a stay meshed in several
  bars is born straight between its two ends. Tension-only bracing places
  nothing either: a new node reached only through it starts where it was
  designed. CHANGES THE RESULTS of a staged model with such members.

## [1.15.0] — 2026-11-14

The Midas translation, checked against Midas on six real bridges of a
course's models (kept outside the repository): a steel box girder, a
steel truss (7 stages), a curved girder on skew bearings (8), a rigid
frame built by balanced cantilevers (24), a continuous girder (16) and a
continuous T-beam (6) now run to their last stage, and where the two
models are the same the displacements, reactions and member forces agree
to about 1% (the curved girder's deflection 1.000, 0.1% over 178 nodes;
the truss's 0.993; the rigid frame's member forces 1.002). Several
corrections CHANGE THE RESULTS of a model translated before -- the
weight and the stubs of the old softening, the rigid arms' stiffness,
shear deformation, concrete's Poisson's ratio; each is below. Tendons
come in as bars bonded to the girder. The implicit static solver's
tangent costs an eighth of what it did. The case-file and result formats
are unchanged.

### Changed

- A jacked bar's force joins the implicit static solver's residual scale:
  a girder under its own tendons and nothing else has no external load,
  and at the answer the bars' pull and the girder's push cancel at every
  node -- the relative residual was round-off over round-off, 1.000, and
  the step failed.
- **The implicit static solver's tangent costs an eighth of the member
  evaluations, and its assembly nothing**. Measured
  on the real arch of 1.13 (17 000 members, 66 stages, 3684 tangents in
  2 h 50 min): nearly every force evaluation of a static run is spent on
  tangents, each tangent was 204 evaluations of the WHOLE force (2 x 6
  x 17 node colours), and a quarter of a tangent's time was a Python loop
  rebuilding the same sparsity one small array a node. Two changes:
  - the sparsity is built once and the values are placed by a stored
    permutation. The same matrix to the bit, on every model.
  - in a model of beams, bars and connectors under `backend: numba`
    (no plastic hinges, soil springs, breaking members or rigid contact surface -- those keep
    the whole force), a colour's columns are read from the members that
    touch its nodes alone: moving those nodes changes no other member's
    force. Each member is then evaluated 24 times a tangent (its two
    ends, six components, both signs) whatever the number of colours. The matrix is the whole force's to
    round-off (1e-12 of its largest entry on an 8000-member lattice) and
    a little cleaner -- a stiff member far from the nudge no longer
    leaves the round-off of its force in the column -- so a run's
    iterations, and the last digits of its answer, can differ within
    the tolerance. `VFIFE_TANGENT_MEMBERS=0` differences the whole force,
    as before. An 8000-member lattice: 1.48 -> 0.22 s a tangent, a
    whole run of 33 tangents 67 -> 24 s (2-core container). Found in
    review and fixed before release: a bar being jacked has no axial
    stiffness in its step, so its length is read at every nudge, not
    once a tangent; and a force wrapped on the solver (a rigid contact
    surface) keeps the whole force.

- **In the Midas translation a stub is a joint, and a length of girder
  is left as stiff as it is. THIS CHANGES THE RESULTS of a model
  translated before** (found on nine real
  bridge models). A member stiffer than `stiffest` (1e13 N/m) had its
  section divided by a power of two. Three faults in that:
  - most of the members over the limit were lengths of girder, not
    stubs: a steel box girder whose every 1 m segment was softened by 8
    stood on an eighth of its weight (the weight went with the area), 24
    short members of a steel truss softened by 64 moved a node by a
    third, and sixteen 37 mm and 146 mm pieces of the real arch's chords
    (1.13) were softened by 128 and by 2. Only a member shorter than
    `stub` (0.05) times its section's radius of gyration is a stub now;
    the others are left as they are, counted
    (`members_over_the_limit_left_as_they_are`) and said.
  - a stub divided until 12EI/L^3 is under the limit has an EI/L of the
    limit times L^2/12 -- 3 N m/rad for 2 mm: a hinge, with next to no
    EA/L either. Two spans of a real curved bridge met in a 2 mm stub and
    came apart (the far span turned in plan: 50 mm where Midas has 0.2).
    A stub is now a connector: `stiffest` in each direction, and in
    turning what its section has over `stub` radii of gyration. The
    report's `softened_members` is `stubs_made_joints`.
- **A rigid link's arm in the Midas translation is rigid whatever its
  length. This changes the results of a translated model with long
  arms**. It was one section for every arm, a
  half-metre round: 4 m of it from a pile cap's node to its piles bends
  at 5e8 N/m where the pile carries 2e9, and the cap's weight went down
  the nearest pile -- a real rigid frame's piles settled 0.45 of Midas'
  under the cap, its pier tops six times Midas'. An arm now has
  `stiffest`/2 along and across at its own length (area with the
  length, second moment with its cube). The same frame: 1.001 and 1.000.
- The Midas translation's H section has its two flanges (the fifth and
  sixth sizes are the bottom one's where it differs), and the steel of
  JTG D64 weighs 78.5 kN/m3 as Midas has it, not 76.98: a real truss's
  girder weighed 3.5% less than in Midas' reactions, and now weighs the
  same to four digits.

### Added

- **More of a Midas model is translated**. Of ten
  real models, none could be translated before; nine can, and five run
  to their last stage.
  - sections by the properties the file states: `*SECT-PSCVALUE` (PSC,
    general and composite sections, and both ends of a tapered one), a
    VALUE section, a solid rectangle, a tapered section of sizes; a
    tapered section over a `*TS-GROUP` gives each element the properties
    at its own place along the run (stated properties: linear between
    the two ends, and said);
  - `*RIGIDLINK`: a master and its slaves are rigid arms (all six
    directions; one that leaves a direction free is counted);
  - an elastic link's "rigid in this direction" flags: all six, a rigid
    link; some, `link_rigid` (1e10) in those;
  - the same link written in the boundary group that takes over from
    another is one link, kept (a closure's bearings were removed with the
    closure's group, and the finished bridge had none); two on together,
    or a different one, are counted;
  - a node's force and moment are two loads (they were one, which no
    case accepts).
  - a section set off its nodes (offset CT, CB: a girder drawn by its
    top). The member is on its centroid's line -- a node of its own under
    each end, by the centroid's depth there, so a tapered girder's line
    bends -- joined to the model's node by an arm `offset_rigid` (30)
    times as stiff as what it carries, built and removed with the
    member; Midas' displacement of the node, at the top, is still the
    node's row (`members_set_off_their_nodes`). On a real rigid frame
    (24 stages) the lengthwise displacement at the node went from 0.02
    to 0.85 of Midas' to 0.96 to 1.03. Sideways (L, R) is counted, not
    followed;
  - **tendons** (`*TDN-PROPERTY`, `*TDN-PROFILE` -- 2-D profiles on a
    straight or an element reference line, corners rounded to their
    radii -- and `*TDN-PRESTRESS`) as bars bonded to the girder on rigid
    arms: a node on the tendon's line at each node of its members and at
    its anchors, each bar jacked to the tendon's force there after
    friction (mu per radian, k per length) and the anchorage's draw-in,
    in the stage its load group is on and its members are built, and an
    ordinary bar after -- it shortens with the girder in the stages that
    follow and stiffens its section. Relaxation, creep and shrinkage are
    not followed. The map's `tendons` gives each tendon's bars and forces.
    On a real rigid frame (24 stages, 132 tendons) the tendons' stiffness
    alone brings the dead-load deflection from 1.067 to 1.035 of Midas'
    (Midas counts them in the section); the tendons' own effect is 1.05
    to 1.15 of Midas' "钢束一次", growing with the stages as Midas' time
    losses do;
  - shear: Midas' beam is a Timoshenko beam with the section's ASy, ASz,
    and so is the translated one where they are known (stated, or a
    pipe's, a round's, a rectangle's, a box's, an H's) -- `shear=False`
    for Euler's. A real curved bridge on bearings: its pier bearings
    carried 18% less than Midas' without it and the girder deflected
    0.90 of Midas'; with it 1.009, 1% apart over 178 nodes;
  - a concrete grade the file names without values has Poisson's ratio
    0.2, as in Midas' database (it was 0.3): its shear modulus sets the
    shear deformation and the twist;
  - `*LOCALAXIS` (a node's own axes, turned about Z: a curved or a skew
    bridge's bearings): a point spring's ground node is set along the
    node's own x, and a support that holds one of x and y alone -- a
    guided bearing -- is a link `link_rigid` stiff in that direction;
  - a rigid link's arm between members that carry more in turning than
    an arm of the common stiffness (a pier fixed to a girder) has a
    section of its own, `offset_rigid` times what its ends carry;
  - `*SPRING` (linear point springs: soil on a pile): a ground node of
    the spring's own, held, and a link to it, with the spring's boundary
    group;
  - two nodes at one place joined by a link (a bearing drawn at a point,
    a point spring's ground): the second is set 1 mm along X from the
    first -- a mesh has no two nodes at one place -- and a rigid link
    between such two is six stiff springs. Counted in the notes;
  - a part on pins in one line (a side span cast on falsework: beams in
    a line, held in translation only) has its rotation about that line
    held at one pin, as each stage's own constraint while nothing else
    holds it; Midas solves such a part as it is, the Newton iteration
    stopped on it (`parts_held_from_spinning`);
  - `convert(..., cases=("wind",))`: a load case of the finished bridge
    as one more step after the last stage. On a real steel truss bridge
    (1266 beams, 7 stages, 496 soil springs) the wind case's transverse
    displacement is 1.009 of Midas' at its largest, 1.7% apart in the
    root mean square over 976 nodes.

## [1.14.0] — 2026-11-13

One correction that CHANGES RESULTS, on the frozen 1.0 formats: a support
put under a node that has already moved, in a later step and without a
`value`, now holds the node where it stands; it used to put it back at
its coordinates. A new member's end at a clamped support is born
unturned. Both are what a staged erection means by a support, and both
were found by following the real arch of 1.13 through its stages: with
them its buckle cables' forces are all within 5% of Midas' (82% within
2%; 66% within 5% before), every stage's displacement increment within
4 mm, and what remains between the two -- the rib 0.2 to 0.5 m higher
here -- is geometric nonlinearity, which Midas' staged analysis of that
model leaves out (shown by scaling every load down: at a tenth of the
loads the two agree to 1 to 4%, and to 0 to 3% extrapolated to none).
Every shipped case is unchanged. A minor version, not a patch, because a
case that props a deflected node in a later step gets another answer.

### Fixed

- **A support put under a node that has moved holds it where it stands**
  (beams, bars, connectors, fibre beams). THIS CHANGES
  RESULTS: a constraint without a `value` that takes a degree of freedom
  which was free at the end of the step before used to hold ZERO -- it
  put the node back at its coordinates, with nothing in the case asking
  for it. It now holds the displacement the node has
  then, and keeps holding it there while plain constraints hold it; a
  clamp (all three rotations) holds the rotation too. To bring a point to
  a position is a `value`. Found on the real arch of 1.13: its feet are
  hinged for the erection and cast in at the 35th stage, and each half
  rib was turned back about its foot by 1.5 and 1.3 mrad there -- 85 mm
  at the control points, where Midas, and a foot cast in place, move
  nothing. (It had also hidden most of the difference that
  geometric nonlinearity makes: see above.)
  Unchanged: a degree of freedom held from the first step or from the
  step that builds its node (or builds it again) is at its coordinates; one that a `value`
  moved and a plain constraint then takes returns to zero; every shipped
  case (their later-step constraints all carry a `value`). A case that
  props or fixes a node that has already moved, in a later step and
  without a `value`, changes -- to what it described; so does one with
  no `keep`, `release` or `activate` in it (a step's own plain constraint
  on a node the step before left free). The held values
  are in the checkpoint (`hold_at`, `hold_on`); a checkpoint from before
  resumes, holding zero in the step it resumes in.
  The Midas conversion follows the stage's own word: a boundary group
  activated DEFORMED is a plain constraint, one activated ORIGINAL is
  given `value: 0` (counted in the report).
- **The Midas conversion's map gave another node's row when a node was in
  no element**: `<name>.map.json` numbers the nodes for
  reading results, and gave each its place in the mesh file; the mesh
  reader leaves out a node no element uses, so every row after it was one
  off. A model converted whole with every node in a member read right
  (the real arch did); one cut at `stages`, or with a node only left-out
  elements reach (a plate's own), did not -- a control point of the arch
  cut at five stages "moved" -14 mm that had moved +1. The map now gives
  the row in the results; the report counts the nodes no member reaches.
  The case and the mesh were right throughout: only the map.
- **A new member's end at a clamped support is born unturned.** THIS
  CHANGES RESULTS for a member built in a later step from a node that has
  turned to a node whose three rotations a support holds: under `birth:
  tangent` the new node was born with the standing node's rotation and
  the clamp turned it back over the step, the member stressed by a
  rotation it was to be born free of (a connector's new node has been
  put there unturned since 1.13). The member is now born free of stress
  between the standing end as it stands and the support as it is, and
  the support carries nothing until something is added.

## [1.13.0] — 2026-11-12

Staged erection of a real bridge, on the frozen 1.0 formats. Three things
are new: a Midas Civil model (`.mct`) is converted into a vfife case with
its construction stages as analysis steps (`vfife init case.yaml --mesh
model.mct`), a connector can be born in a later analysis step, and a cable
can be jacked to a force (`jack`) in the step that builds it. They were
written to follow a 508 m steel-truss arch through its 66 erection stages.
What it changes in a RESULT: a step that activates
or removes a named group of LINES now takes that group's members, not
every member between two of its points, and a truss bar that ruptures is
no longer cut a step early by an output frame (see Fixed). Every shipped
case is unchanged.

### Fixed

- **A truss bar that ruptures was cut a step early by an output frame**:
  the standalone truss solver's frame evaluated the bars
  again, and a steel02 law accepts its last strain -- which is what cuts a
  ruptured bar -- when it is evaluated again. The frame now leaves the
  bars' state as it found it (the fibre beam's fault of 1.12.1, here in
  the truss). Found by running every shipped case and every law that
  keeps a history with a frame after every step, a check after every
  step, both at odd intervals and neither: the 61 shipped cases, the
  solid solver with j2, cdp, erosion, embedded and bonded bars that
  rupture, the frames' hinges and the fibre beam hold the same state to
  the bit under all four; the truss with a rupturing bar did not.

- **A step activates the members of a group of lines, not every member
  between two of the group's points**: `activate` and
  `deactivate` took a region's members by its points -- a member was in
  the region when both its nodes were -- so a group of chords also held
  every brace between two chord nodes, and the step that built the brace
  later was refused ("activates member N, which is already built"); a
  brace that no later step named was built with the chords, silently.
  Now the members the mesh recorded for the group are taken (a Gmsh
  physical group of lines, a `lines` group -- what `assign` already
  meant by the name); a region given by a box or by points keeps the old
  rule. No shipped case and no test changes: with both rules compared in
  every step test, they named the same members everywhere.

### Added

- **A Midas Civil stage model as a case** (`vfife/io/midas_case.py`,
  v1.13): `vfife init case.yaml --mesh model.mct` writes the
  case, its line mesh and a map of which Midas node and element became
  which. Line members with the section properties of the Midas shapes
  (pipe, solid round, box, H, double channel), bars with their
  pretension, supports, rigid links as stiff arms, self weight, nodal and
  uniform member loads, and one analysis step per construction stage --
  what it builds, removes, holds, releases and loads. In SI whatever the
  file's units. It says what it left out and what it approximated (an
  independent review found five ways the first version built a wrong
  structure without a word -- a link, a pretension and a self weight
  that ignored their groups' stages, a release that freed a permanent
  support, two members on the same nodes merged -- all now followed or
  counted). An elastic link becomes a connector with its springs in
  global axes (they do not turn with its ends: in turning axes a 3 cm
  slide along a saddle's soft direction times a thousandth of a radian
  was 2e7 N in its stiff one), built in the first stage in which both
  its ends are held and born free of force there; a link to a node no
  beam reaches has its rotational stiffness raised, so the node turns
  with it instead of leaning. A cable's pretension is written as
  `jack` (below); of a cable modelled as several elements over sliding
  saddles, each given the same force, only the one that reaches the
  newly built structure is jacked and the others are cut to length --
  their horizontal parts do not balance at a saddle, which a
  geometrically linear solution never notices and this one answers with
  metres of slide. A boundary group that is activated again after its
  removal (an arch foot fixed, hinged for the erection, fixed again) is
  followed. A plate's weight is put on its nodes (its stiffness is
  left out): the gusset plates at an arch foot were 4% of the first
  segment's moment about its hinge, which the buckle cable's force is
  designed to balance. A stage that builds members of the model's
  tangent-assembly group builds along the tangent, the others at the
  design position; under `birth: tangent` a connector and a cable being
  jacked carry no new node (a rib segment was reached first through its
  buckle cable from the tower top and put 0.27 m off its foundation).
  All 66 stages of the real arch's erection are then solved; against
  Midas' stage-by-stage results the buckle cables' forces have a median
  ratio of 1.004 over 972 cable-stages (66% within 5%, 98% within
  10%), a cable carries exactly its jack force in its own stage in
  both, and the first segment's rise when its foot is hinged is 0.9 mm
  against -0.1 mm. The rib's vertical displacement at the 36 control
  points follows Midas' with a correlation of 0.95 to 0.997 in every
  stage from the 41st to the last before the cables are released, a
  median difference of 42 mm and none above 71 mm (the reference
  workbook's stage names are out of order from the 41st stage on; its
  rows are in the model's order, which is how they are compared --
  v1.13).
  What remains is mostly that Midas' staged analysis of this model is
  geometrically linear and this one is not: with the rib 0.24 m up, the
  moment of its weight about the hinge is 0.6% less, which is 4% of the
  first cable's force (hand statics on both geometries: 540 kN m).
  With `prestress` in place of `jack`, 34 of the 66 stages of
  a real arch's erection were solved without one cutback and the buckle
  cables' forces were within a few per cent of Midas' (median ratio
  1.007 over 184 cable-stages); the displacements were not, which is
  what `jack` is for.
- **A cable stressed to a force: `jack`** (frame models;
  v1.13): `jack: T` on a bar assignment. In the analysis step
  that builds the bar (the first, for one built from the start) the jack
  holds T times the step's progress whatever the structure gives; at
  the step's end the bar carries exactly T, and from then on it is a
  bar of the unstressed length it ended with. `prestress` states a
  length and loses force as the structure comes to it: a cantilever's
  stay jacked to 50 kN carries 50 kN, the same force as a prestress
  carries 20. Done through the unstressed length the force law already
  reads, set before every force evaluation of that step, so no kernel
  changes; the lengths are in the checkpoint (`jack_length`). A jack
  is worked in a static or a relaxation step (a dynamic one is refused),
  once (a bar removed and built again keeps the length its jack left);
  outside a run -- a modal or buckling analysis of the model as built --
  the cable is the bar it is until jacked. Refused in any model that is
  not of beams, bars and connectors, and in form-finding.
- **A connector can be born in a later analysis step** (frame models;
  v1.13): `activate` on a region of connectors was refused. Like
  a beam or a bar, it is free of force where its ends stand when it is
  built -- its six deformations then are its reference -- and its laws
  start from nothing; removed and built again it is a new one. A
  saddle's links wait for the cable they carry, a bearing for its
  girder. A new node that only connectors reach and whose rotations a
  support holds is put at the support unturned (found in review: a prop
  to a fixed point was born with the tip's rotation in its springs).
  The offsets are in the checkpoint (`connector_birth`). Fibre
  models still refuse it. The compiled connector kernel takes the
  offsets as one more argument; the compiled step is not used with
  analysis steps, as before.
- **`midas.with_stage_order(model, names)`**: the model
  with its construction stages in another order, every stage named
  once.
- **A reader for Midas Civil models** (`vfife/io/midas.py`):
  nodes, line members, materials, sections, structure / boundary / load
  groups, supports, elastic and rigid links, load cases with their loads
  (self weight, pretension, nodal and uniform member loads, temperature)
  and the construction stages of a `.mct` file, as plain data; `vfife mesh
  model.mct` says what is in one, what was not read, and whether it hangs
  together. The first step towards following a real bridge through its
  erection.
- **`tests/test_reading_is_not_a_step.py`**: that comparison, for every
  solver family and every kind of history a material keeps, as a
  permanent test; the manual's Section 13.4 states it as a promise beside
  loop = compiled and restart = uninterrupted.

## [1.12.1] — 2026-11-11

A maintenance release on the frozen 1.0 formats. What it changes in a
RESULT: a dynamic run in which a member breaks, a fibre bar ruptures or a
section fractures no longer depends on where its output frames and progress
checks fall (it did, by one step of the broken member's force; see Fixed).
A run without a break is unchanged to the bit, and so is every shipped
case. Otherwise: the README is a front page, the fibre-fracture-with-
contact combination takes the compiled step, and three loops that were
written twice are written once.

### Changed

- **The README is a front page**: what the solver is for
  (bridges, from construction to collapse), install, a first run with
  `vfife new girder`, where each thing is written down, what has been
  checked and what it does not do -- 170 lines. The 2300-line description
  it was is `docs/technical_notes.md`, unchanged but for its links.
  `tests/test_front_page.py` holds the front page to itself: its case is the
  `girder` template and runs, every command it names exists, every link in
  it and in the notes leads to a file.
- **Package metadata**: the description and keywords say what it is; the
  classifier is Production/Stable (it was Alpha at 1.12); Python 3.13 is
  listed (the development container has run it since 1.3).
- **`vfife run` in `mode: static`** prints `increment k/n  resid ...` instead
  of the relaxation's line with `KE 0.000e+00  resets k`.
- Six audits and plans from before the 0.5 input model moved to
  `docs/archive/` (removed before the code was open-sourced).
- **One driver for the compiled step** (internal): the loop
  that hands runs of steps to the compiled kernel and takes back the steps
  it cannot take was written twice, in the frame and in the fibre solver;
  it is `compiled_step.drive` now, with the solver's own three questions as
  methods. A step the frame kernel does not take is reported as HANDBACK,
  as the fibre kernel's; EVENT is a step the kernel finished.

- **One relaxation loop** (internal): the one-level
  relaxation and the level of a staged one were written twice as well;
  they are `OrientedPoints._relax_one_level` and `._relax_stage`, with what
  differs between the frames and the fibre beam as methods. 32 runs on
  both versions -- frames, bars, connectors, fibre sections, creep, the
  static solver's fallback, stops and resumes, runs that hit their budget
  -- give the same bits.

### Added

- **Fibre fracture with contact between members in the compiled step**:
  the combination a collapse analysis is made of kept the
  step-by-step loop in 1.12.0, because the loop takes a fractured member
  out of the contact in the step it fractures and the compiled step learned
  of the fracture a step later. The fibre chunk now stops in the step a
  fibre point fails, with that step's member forces made, and the loop's
  force evaluation goes on from them: the latch, then the contact. To the
  bits of the loop, the count of neighbour-list sweeps included; a girder
  breaking onto a beam below, 38868 steps, 18.8 → 2.4 s.
- **A lint gate** (`[tool.ruff]` in pyproject.toml, `tests/test_lint.py`,
  ruff in the `dev` extra): no undefined name, no unused import in `vfife/`.
  26 unused imports removed to get there.

### Fixed

- **A reading of the state broke members and cut bars**.
  After a dynamic step the forces are evaluated again for a progress check
  and for an output frame, at the state the next step starts from. That
  reading (a) broke a frame member at its capacity and took out a removal
  now due, (b) latched a fibre fracture, and (c) made a ruptured steel02
  bar's law accept its last strain, which is what cuts the bar -- each one
  step before the member's own step would have, so it carried nothing in
  the step it should have carried its last force in. The answer depended
  on `output.every` and `progress_every`: a fibre girder breaking under a
  point load ended 52 mm apart between a frame after every step and none;
  a cable breaking on its overshoot, 0.1 mm. A reading only reads now, in
  the frame, the mixed, the fibre and the truss solvers: the step breaks,
  latches and removes; a frame's `capacity_ratio` is still that of its
  own forces. The last frame of the dynamic history, after which no step
  comes, still finds a removal due at its very end. Tests run each case
  with a frame after every step, every seventh and none, in the loop and
  in the compiled step, and compare the bits. No shipped case changes
  (cases 05, 19, 24c, 26, 33, 34, 36, 38 and twelve relaxation and static
  cases compared before and after: bit-identical).
- **A restarted run reported 0 for the fibre stresses of a broken member**
  (the uninterrupted run: the stress it broke at, 404 MPa in the test):
  the array was not in the checkpoint. It is; a checkpoint written before
  this restores as before.
- **Fracture with two fibre sections and a bearing stopped at its first
  step** ("every element is out of at least one thing"): a connector has no
  fibres, and its row of the latch's denominators was 0. Found in review.

## [1.12.0] — 2026-11-11

The twelfth feature release on the frozen 1.0 formats, and the one that
closes the 1.x line of features: nothing that 1.0 to 1.11 reads or writes
changes meaning (case format 1, result format 3, checkpoint format 1,
API v1). One item: the contact between members of 1.11 runs inside the
compiled step, to the bits of the step-by-step loop. An independent review
found one defect in the first version (a fractured fibre member stayed in
the contact one step longer), handled here; a 1.11.0 defect found on the
way is fixed. Every contract is unchanged.

### Changed

- **Contact between members in the compiled step**: a
  model with `contact.members` ran the step-by-step loop. One evaluation
  of the contact -- the neighbour list, the pairs that overlap, their
  forces, the allowances -- is now one compiled function
  (`kernels/numba_member_contact.contact_step`) that the loop and the
  compiled step both call, frames, bars and connectors under `backend:
  numba` and fibre beams alike, so the two agree to the bit by
  construction and every 1.11 result is unchanged. A neighbour list that
  outgrows its arrays hands that one step back to the loop. 2-core
  container: Xihoumen with every member a contact body 20.4 → 9.8 s for
  18758 steps (9.3 s without contact); case_34 likewise 33.1 → 3.4 s. A
  model with rigid surfaces, a relaxation with contact and a fibre model
  with `fracture` and contact still run the loop.

### Fixed

- **A probe's allowances written by the step after it** (1.11.0): an
  evaluation that is a question (a probe, a trial state) keeps the
  contact allowances it arrived at aside, for a relaxation's level to
  commit; a dynamic step that followed wrote its own and then, committing,
  the probe's over them. A probe of the state the run is in asks the same
  question as the next step and nothing changed; a probe of another state
  (`testing.state.probe` with the girder lifted) left that state's
  allowances behind. A step's evaluation now discards them.

## [1.11.0] — 2026-11-11

The eleventh feature release on the frozen 1.0 formats: nothing that 1.0
to 1.10 reads or writes changes meaning (case format 1, result format 3,
checkpoint format 1, API v1; one new block, `contact.members`). One item:
contact between line members, each a capsule about its axis -- a falling
span on the deck below, pounding girders, the pile a collapse leaves. An
independent review found eight defects in the first version (bars spun
up by friction's couple, a member born later launched from its overlap,
a probe that changed the contact's state ...), all fixed here with a test
each. Every contract is unchanged.

### Added

- **Contact between line members** (`contact.members`):
  each member of the `bodies` a capsule of its `radius` about its axis;
  two that overlap push each other apart with a penalty spring sized from
  their masses and the step (the step never shrinks for it), and in a
  dynamic step a dashpot and Coulomb friction, which acts at the contact
  point and so twists the members too. Pairs that overlap or touch in the
  reference configuration (a girder over its cap on a bearing, neighbours
  at a joint) and a member that breaks off its neighbour keep that overlap
  plus a `slack` as an allowance that only shrinks: a deck whose bearing
  fails lands on the cap from where it stood. The allowances are in the
  checkpoint. The pair search is a compiled sweep kept as a neighbour
  list. Checked: a dropped beam touches at sqrt(2 (h - r1 - r2) / g) and
  rests r1 + r2 above the other, whose supports carry its weight; a
  collision conserves momentum to round-off; a beam sliding on rails stops
  in v^2 / (2 mu g). A model with contact runs the step-by-step loop.
  Capsules only (no section shape), line members only (no solid mesh).

## [1.10.0] — 2026-11-10

The tenth feature release on the frozen 1.0 formats: nothing that 1.0 to
1.9 reads or writes changes meaning (case format 1, result format 3,
checkpoint format 1, API v1). Two new connector laws for a bridge's
seismic devices -- a nonlinear viscous damper and a friction pendulum
bearing, new `type`s inside `laws` -- and the solid path's embedded bars
compiled; each reviewed by an independent agent. The reviews and the full
check on the user's machine found two defects that could give a wrong
answer without an error, both fixed here: a dynamic step after a
relaxation ignored the dashpots' step bound, and a stale numba cache could
run old kernel code after a source update. Every contract is unchanged.

### Added

- **A nonlinear viscous damper as a connector law**:
  `{type: damper, c, exponent, linear_below}`, F = c sign(v) |v|^alpha on a
  component's deformation rate, 0 < alpha <= 1, with an optional parallel
  spring. Linear below `linear_below` (required when alpha < 1): the power
  law's slope is infinite at zero velocity and an explicit step chatters
  there; the linear piece is also what the explicit bound dt < 2m/c reads.
  Acts in dynamic steps only. Checked against a Runge-Kutta integration of
  a mass on a spring and a damper, for alpha = 0.3 and 1.
- **A friction pendulum bearing as a connector law**: `{type: pendulum, mu,
  radius, normal, stiffness}`, Coulomb friction mu N plus the restoring
  force (N / radius) d, both following the compression the bearing carries
  at that instant; one law in the plane on uy and uz together; nothing
  when lifted off. Checked: slides at mu W, then d = (H - mu W) R / W, the
  same displacement for twice the weight at twice the push.
- Both laws are in the reference and in the compiled connectors (which
  agree to round-off), and the compiled step takes them to the loop's
  answer to the bit. New case keys inside `laws` only: every 1.x case reads
  as before.

### Fixed

- **A stale numba cache could run old kernel code** (numba before 0.68):
  numba keys a cached function on its own source file, so a cached kernel
  that calls a function of another file -- the compiled step calls the
  connectors', the frames', the fibres' -- kept that function's old machine
  code when only the other file changed (a checkout that moves; an install
  replaces every file and is not affected). Found when the damper and the
  pendulum went into the connectors: the compiled step ran yesterday's,
  0.1 m off, with no error. The package now keeps a stamp of every kernel
  and material source beside the cache and removes the cache files when it
  does not match, before a kernel is imported (`kernels/_cache.py`;
  `VFIFE_KEEP_NUMBA_CACHE=1` turns that off).
- A `procedure: dynamic` step under a relaxation base mode took its time
  step from the stiffness alone: the connector dashpots' bound dt < 2m/c
  was applied under a dynamic base mode only. `connector_damping` is
  refused under a relaxation base, so nothing reached it until the damper
  law, which is not: a stiff damper after a gravity relaxation went
  non-finite (found in the review of the damper). The
  step takes the bound now, frames and fibres.
- A model whose only free dof carries dashpots alone (its stiff components
  all held) crashed on an empty minimum when its step was taken; it is the
  pure-dashpot model `solver.target_dt` is for.
- `modal` and the spectrum warn of a sliding `pendulum` bearing as they do
  of a sliding friction one.
- `testing.state.probe` of a solid with embedded bars advanced the bars'
  plastic state at every question (found in the review of v1.10).

### Changed

- **Embedded reinforcement is compiled** under `backend: numba`:
  a bar set whose law has a compiled twin (`rebar`)
  gathers its ends from the host nodes, updates the law, and spreads its
  force back in one compiled pass (`kernels/numba_embedded`), instead of a
  dozen NumPy temporaries and a scatter per iteration. The reference path
  stays for the other laws and for `backend: numpy`; the two agree to
  round-off (the law's compiled twin), so a numba run with bars moves in the
  last places. `VFIFE_COMPILED_BARS=0` turns it off. Two cores: an
  iteration of the U of Alberta beam (case_12) 3.15 → 1.95 ms, of the
  pretensioned beam (case_13) 0.68 → 0.36 ms.
- case_13 and case_14 run on `backend: numba` (2.5 → 0.36 ms an iteration
  for case_13; they were the longest cases of the full check on numpy);
  the full check runs them in 87 and 47 s (386 and 186 on numpy), and the
  three U of Alberta beams in 150-190 s (280). Every contract is unchanged.

## [1.9.0] — 2026-11-10

The ninth feature release on the frozen 1.0 formats: nothing that 1.0 to
1.8 reads or writes changes meaning (case format 1, result format 3,
checkpoint format 1, API v1; no new case key). Three items, reviewed by an
independent agent: a relaxation with rigid diaphragms converges; the
frames' staged relaxation is compiled, to the loop's answer to the bit;
and the small findings left from earlier reviews. Every contract is
unchanged; the full check takes 22.9 min (25.9 for 1.8.0).

### Changed

- **The frames' staged relaxation is compiled** under `backend: numba`:
  every level of a `mode: relaxation` with `load_steps`,
  and the static solver's relaxation fallback, run their iterations in one
  compiled function (`kernels/numba_step.relax_chunk`) that returns at each
  progress check -- the step-by-step loop's answer to the bit, on any number
  of threads, through a stop and a restart (the energy ledger's dot
  products are NumPy's BLAS in the loop and SciPy's in the kernel, as in
  the compiled dynamic step: the same numbers where the two libraries agree,
  as on the machines checked). Lee's frame 87 → 27 s on two
  cores. Frames, bars and connectors; the rest keeps the loop and
  `solver.compiled_relax_refusal` says why. Fibre models keep their loop
  (their kernel is the cost; `solver.threads` speeds it up).
- Under `backend: numba` a relaxation's kinetic energy and convergence norms
  are the compiled functions the compiled relaxation uses (NumPy's pairwise
  sum, sqrt(x.x)): the same numbers on the machines checked.
- case_16 runs on `backend: numba` (444 s of the full check on numpy);
  `tools/check_all.py` gives the four fibre pushovers (case_25, 25b, 30,
  31) two threads each.

### Fixed

- **A relaxation with rigid diaphragms could not converge** (since
  v0.1; found in the review of 1.7). Its residual was the nodal
  one, and at a floor's nodes that holds the constraint forces that keep
  the floor rigid, which never vanish: a one-storey frame sat at a
  residual of 0.5 after 200 000 iterations. It is measured on the floors'
  own equations now -- the in-plane resultants and the torque about the
  normal, the three equations of a floor's three dofs, the split
  `_project_diaphragms` makes -- as the static solver has measured it since
  1.7. The storey converges to 1e-7 and agrees with the
  static solution, for a floor normal to z and to y; a model without
  diaphragms is unchanged to the bit. The static solver's relaxation
  fallback takes diaphragms in consequence (it refused them in 1.7).
- A fibre model with a cracking shear law gave its shear rotation one more
  Newton step at every output frame and every progress check, so its
  answer depended on `output.every` (1.2e-6 in gamma between frames every
  7 and every 50 steps of a cracking cantilever; found in the review of
  1.7). Those evaluations read the state now, as the frames' plastic
  hinges have since v1.0. A continuous (not staged) fibre
  relaxation with that law evaluates its convergence measure every
  iteration, so its numbers change at every iteration (by the step the
  reading no longer takes).
- The static solver's relaxation fallback reports to the progress callback
  every 500 iterations and can be stopped; it ran its whole budget deaf.
  Its samples do not enter the static run's residual series, which is
  Newton's, one per increment.
  No checkpoint is written inside a fallback, and a stop there puts the
  last converged increment back: a checkpoint of the half-relaxed state,
  its fictitious mass and its iteration count resumed Newton from the
  wrong place (found in the review of 1.9).
- A staged relaxation stopped at the progress check of the iteration that
  converged a level ran on past the level's end when restarted (the
  checkpoint carried the iteration before's measures); the restart now
  ends where the uninterrupted run ends, frames and fibres.

## [1.8.0] — 2026-11-10

The eighth feature release on the frozen 1.0 formats: nothing that 1.0 to
1.7 reads or writes changes meaning (case format 1, result format 3,
checkpoint format 1, API v1; no new case key). One item, reviewed by an
independent agent: a fibre model with creep or shrinkage may change its
model in analysis steps, each segment on its own concrete clock. Every
contract is unchanged. This closes the 1.x route; 2.0 waits for a change
that must break a frozen format (warping, a shell rewrite).

### Added

- **Each fibre segment on its own concrete clock** (item
  19 of the 1.x route). A fibre model with `creep` or `shrinkage` could not
  activate, deactivate or release in an analysis step (1.2): the creep
  scale on the law's input was one per material, so a segment born late
  would have shrunk and crept from the model's first day. It is per member
  now, by the frames' rule of 1.2 (`_age_shift`, `_age_floor`): a member
  born at its `activate_age` creeps as phi(t + shift, t0 + shift) with
  nothing dated before its birth, and its shrinkage in the formwork is no
  load. The reference path sets the law's `strain_scale` member by member,
  the fused kernel takes it in its per-member parameters; the station's
  intervals are filed (`history.windows`) as the frames file them. Checked
  in tests/test_fiber_segment_ages.py on concrete02 fibres: a bar cast at
  clock 100 at age 7 creeps by phi(907, 7) beside one loaded at 28 with
  phi(1000, 28), a free late bar shortens by (eps_sh(907) - eps_sh(7)) L, a
  held stretch on a late segment relaxes as a segment loaded at its own
  age in a model of its own to 1e-6, the kernel agrees with the reference
  to 1e-8, and a run stopped mid-creep resumes to the bit. A fibre model
  whose tendons relax is still refused when it changes (a tendon born late
  would relax from the model's stressing age). A model with no birth is
  unchanged to the bit (case_29d, 29e). Reviewed by an independent agent,
  who found no wrong number and confirmed a born-and-creeping step against
  a model of its own to 1.7e-8 and a member taken out and cast again
  against one cast once, to the bit; on its finding a member not built no
  longer sizes the residual with its shrinkage and creep (the scale was
  twice too large with half the members out, so a segmental model
  converged loosely).

## [1.7.0] — 2026-11-07

The sixth and seventh feature releases on the frozen 1.0 formats in one
(1.6 was held back to ship with 1.7): nothing that 1.0 to 1.5 reads or
writes changes meaning (case format 1, result format 3, checkpoint format
1, API v1; the one new case key is `solver.static.fallback`). Three items,
each reviewed by an independent agent: the compiled whole step extended
to prescribed motions, plastic hinges and breaking, fibre fracture, moving
loads and vehicles (1.6, item 16 of the route); the implicit static solver
extended to rigid diaphragms and member failure, with a relaxation
fallback (item 17); and the cracking shear law in the fused fibre kernel
(item 18). Four shipped cases run static and one on numba; every contract
is unchanged; one old diaphragm bug (floors normal to y) is fixed.

### Changed

- **The compiled whole step takes prescribed motions, plastic hinges,
  breaking and splitting, moving loads and vehicles** (item
  16 of the 1.x route). Since 1.3 the compiled step (kernels/numba_step)
  refused every model with code of its own inside a step and left it to
  the step-by-step loop. Each of these features is now written once as a
  compiled function that BOTH the loop and the compiled step call --
  `hinge_update` and `carry_appendages` (the hinges and the parked
  particles and appendages), `_prescribe` (displacement, velocity,
  acceleration and record-driven supports), `kernels/numba_loads.
  axle_actions` and `vehicle_step` (the axles' consistent nodal forces,
  a vehicle's step) -- so the two paths agree to the bit by construction.
  A member reaching its capacity is an EVENT: the kernel stops before that
  step, the loop takes it (the failure record, deactivation or split, the
  parking of particles without a body) and the compiled step is rebuilt
  on the model that is left; a fibre section's fracture likewise, with the
  step finished first; a timed removal cuts the compiled run before its
  step. A vehicle's state advances inside the run and its contact losses
  are recorded on the run's clock in step order; a progress probe that
  asks for the force again at the same instant is handed the force, as
  before. Checked bitwise, loop against compiled step, in
  tests/test_compiled_prescribed.py, test_compiled_breaking.py and
  test_compiled_moving.py (a cable breaking on its overshoot and removed
  at its time, a hinged cantilever tearing and falling, a hanger cut
  leaving an appendage and a parked joint, a fibre girder fracturing and
  splitting, a platoon on released members, a pitching truck losing a tyre
  on a PSD road, threads, restarts). 2-core container: Xihoumen with its
  hangers removed and failing (1.5 s) 43.6 → 29.0 s on one thread, case_34
  12.1 → 1.8 s, case_38 3.7 → 0.6 s and case_26 1.1 → 0.2 s with the
  output interval at 500 steps.

  Still left to the loop: a prescribed rotation (NumPy's exp_map rebuilds
  the point's orientation), a fibre shear law, temperatures and creep,
  rigid diaphragms, soil springs, members born in a step, analysis steps,
  and a load group that is a script's own callable.

- `Vehicle.__call__` and the PSD road of `roughness` run through the
  compiled functions above whenever numba is present. Their numbers
  differ from 1.5's NumPy arithmetic in the last place (the order of the
  sums); the contracts of case_38 and the vehicle tests are unchanged.
  Moving loads are unchanged to the bit.

### Added

- **The implicit static solver takes rigid diaphragms and member failure,
  and can fall back to relaxation** (item 17 of the 1.x
  route). A `diaphragms` floor is a linear multi-point constraint: Newton
  solves the master system T^T K T (three masters per floor, the two
  in-plane translations and the rotation about the normal, the constraint
  `_project_diaphragms` applies to a relaxation's accelerations), and the
  residual is measured on the masters, so a floor's constraint forces are
  not an imbalance. A member with a `failure` capacity that reaches it in
  an ACCEPTED equilibrium is taken out whole and the level solved again
  without it, and again while that cascades (the alternate-path analysis,
  one member at a time); the events go to `member_failures` with the
  step plus its progress as their time; no equilibrium after a break is
  "the structure has lost its load". Timed `removals` and `split: true`
  are refused under `mode: static` by name (no clock; a fragment has no
  static equilibrium). `static.fallback: relaxation` lets a level Newton
  cannot take -- the increment halved `cutbacks` times -- be taken by a
  dynamic relaxation from the last converged state to the stated level,
  with the solver's own staged-relaxation iteration (`_relax_stage`, now a
  method of the frame and fibre solvers), and the rest of the step with it;
  the next step starts with Newton again. The default stays `none`, the
  former error, which now names the option. Checked in
  tests/test_static_fallback.py: a one-storey frame's floor moves as one
  rigid body in plan and bends out of it, with the reactions balancing the
  load; a girder on two hangers loses the weak one at the level its force
  passes its strength and the other takes the load, or, both too weak,
  ends hanging from its pin; a shallow arch pushed past its limit point
  snaps through in the fallback (784 iterations) and Newton goes on from
  the inverted arch, and `fallback: none` reports the limit point.

  Which branch the fallback lands on past a jump is the fictitious
  transient's choice, and on a softening section it need not be the one
  `mode: relaxation` follows: case_25 relaxed from Newton's post-peak state
  found a branch whose load rose to 335 kN while the strands unloaded,
  where `mode: relaxation` crushes and collapses at the same level. The
  four reinforced-concrete pushovers (case_25, 25b, 30, 31) therefore keep
  relaxation (case_30 and 31 do finish through the fallback, within 1.6%
  of relaxation's load and 12-16% apart in local strains).

- **The cracking shear law in the fused fibre kernel**. A fibre section's `shear` law (v1.0: the shear
  rotation in series with the bending, cracking at a stated shear or with
  the concrete's flexural cracking, the stirrups' truss stiffness once
  cracked, unloading on the secant) ran on the NumPy reference path alone
  and refused `backend: numba`. It is in kernels/numba_fiber_beam now
  (`shear_law_point`, the same arithmetic in the same order), with the
  state and the history (gamma, the crack strain, the largest excursion)
  advanced in place; the compiled whole step takes it, committing the
  history every step as the loop does. The NumPy path stays the reference
  and the two agree to round-off (tests/test_fiber_shear.py, uncracked and
  cracked); loop and compiled step agree to the bit
  (tests/test_compiled_shear.py).

### Changed

- `cases/case_24`, `case_28`, `case_28b`, `case_28c` run on `mode: static`
  (case_24 74 s → 10 s, case_28 102 s → 1.4 s on the reference machine and
  a 2-core container respectively), and `case_21b` on `backend: numba`
  (20 minutes → half a minute); every contract is unchanged. The
  `statics_at_the_piers` figure of case_28 reads 361 kN where the
  relaxation's residual had left it at 347 (the 3% "shortfall" the
  verification file described), within its tolerance either way.
- `tools/check_all.py` runs case_24d, the case that sets the length of
  the cases part (1142 s of 35 minutes in 1.5.0), with threads of its own:
  the cores the other jobs leave, at most 8 (`--case-threads` sets it, 1
  runs it serially). Its threaded fibre kernel gives the serial answer to
  the bit (checked again over 3 s of its ramp, 1 against 2 threads); the
  contract test takes the count from `VFIFE_CASE_THREADS`.
- The staged relaxation's iteration loop is a method of the frame and
  fibre solvers (`_relax_stage`), the body of `run`'s staged branch moved
  without a change in arithmetic: case_01 and case_11 are unchanged to the
  bit.

### Fixed

- A rigid diaphragm with `normal: y` tied the floor's nodal rotation about
  y to the NEGATIVE of its plan rotation: the in-plane axes were taken as
  (x, z), which is not cyclic with y, so u = U - alpha r_b, v = V + alpha
  r_a was a rotation of -alpha while theta_y carried +alpha (since
  v0.1; floors normal to z, the usual ones, were right). The axes
  are (z, x) now, in `_setup_diaphragms`, which the relaxation, the dynamic
  loop, the modal and buckling reductions and the static solver all read.
  Found in the review of 1.7; tests/test_static_fallback.py
  checks the y-normal building against its z-normal twin.

## [1.5.0] — 2026-11-06

The fifth feature release on the frozen 1.0 formats: nothing that 1.0 to
1.4 reads or writes changes meaning (case format 1, result format 3,
checkpoint format 1, API v1; no new case key). Two items, each reviewed
by an independent agent: lateral-torsional and torsional buckling in
`vfife buckling`, and six shipped static cases moved to the implicit
solver. Every contract is unchanged.

### Added

- **Lateral-torsional and torsional buckling in `vfife buckling`**.
  K_G of a beam is now built from the members' axial
  forces and bending moments, in two parts. The first is the stress part of
  the co-rotational tangent: the end forces turned with the member frame,
  in the solver's own kinematics. The second is the second-order terms
  inside the member: the end rotations relative to the chord, the Wagner
  term, and the moments' coupling of twist with out-of-plane bending.
  Checked against closed forms (Saint-Venant torsion, no warping, loads at
  the centroid): uniform moment on forks pi/L sqrt(E Iz G J), central point
  load 16.94, uniform load 28.31, cantilever end load 4.013, a column's
  torsional load G J A / Ip, and a beam-column. Argyris' right-angle frame
  was checked against the nonlinear model's own tangent, to within 1% both
  ways; the continuum terms alone were 4.5% high and 7.7% low. Flexural
  buckling under axial forces alone is unchanged to round-off.

  The bending before buckling is included through Kirchhoff's factor on
  each plane's moments and shears, sqrt((1 - Iz/Iy)(1 - GJ/EIy)) with each
  bracket clamped at 0, which is 0 for bending about the minor axis. It is
  exact for a uniform moment on forks and within 2-4% for a cantilever. Without it, round, square and box members
  and members bent about their minor axis reported lateral modes that the
  nonlinear model does not have; the Bathe bend (case_18) came out at 0.86
  of its load. With it, a cantilever with Iz = Iy/4 buckles within 2% of
  where the nonlinear model does. The textbook value that codes use is the
  reported one times the factor.

  Not modelled: warping, so an I girder's critical moment is low by
  sqrt(1 + pi^2 E Iw / (G J L^2)), on the safe side; monosymmetric
  sections; loads above the centroid. The torque enters only as the solver
  turns it.

### Fixed

- `vfife buckling` refused a narrow, deep section on a fine mesh (a 6 m
  cantilever of 0.02 x 0.4 m on 64 elements, or 0.05 m elements beside
  0.4 m ones) as a mechanism: its weakest bending stiffness was under
  1e-10 of its largest. A mechanism now has to be under the finite
  difference's own asymmetry as well; a pinned column is still refused.
- `vfife buckling` says when the reference configuration sits on the kink
  of a no-tension fibre law (concrete01, a concrete02 with ft = 0), where
  K_b averages the compression stiffness and zero and a column under
  axial load alone comes out at about half its Euler load; the note points
  to `--result`, where the fibres are in compression and the full
  stiffness is read (the Euler load within 3% in the test).

### Changed

- Six shipped static cases run under `mode: static` instead of dynamic
  relaxation, to the same contracts: case_18 (72 s on the
  reference machine to about 1 s), case_32 (98 to 2), case_27b (118 to 4),
  case_26b (55 to 1), case_27 (30 to 1), case_21 (28 to 6). The
  reinforced-concrete pushovers (case_25, 25b, 30, 31) stay on relaxation:
  near the ultimate state the concrete's crushing localises in one element
  and the equilibrium path jumps there, where Newton sits at a residual of
  1e-3 with the increment cut to 1e-5; relaxation crosses it as a
  transient. case_16 stays as the relaxation twin of case_16b.
- `vfife buckling` takes K_b on the side the load pushes only where the
  model has a response with a kink at its base state: a connector with a
  law or a curve, a tension-only member, soil springs. Elsewhere the
  shifted tangent turned every member's stiffness with the increment's
  rotations, which moved a narrow frame's lateral-torsional factor by a
  factor of 2.6 and broke its mirror symmetry. Flexural factors are
  unchanged: the 1.1 tests stand as they were, the pier template is the
  same to all digits, and the arch template's first factor moves by 0.2%
  (28.90 to 28.84).

## [1.4.1] — 2026-11-06

A patch release: the same input gives the same or a more correct answer;
formats and API unchanged. `vfife modal` and `spectrum` on a model far from
the origin were wrong and are fixed; every contract is unchanged (the
modal, spectrum and moving-load cases rerun).

### Fixed

- `vfife modal` and `spectrum` took the finite-difference tangent's step
  from the model's largest coordinate. A model in survey coordinates (5e5 m
  from the origin) was differenced at 0.05 m and its frequencies came out
  1.8 to 3.7 times too high. The step is now the model's size, as
  `buckling`'s has been since 1.1. A model near the origin is unchanged to
  the last few digits.
- `vfife new girder g1` wrote a file named `g1`; a case name without an
  extension now gets `.yaml`.

### Documented

- Embedded bars in a solid model have mass but no weight (gravity
  integrates the solid's densities only). The customary 2500 kg/m3 for
  reinforced concrete already holds the steel's weight; with a plain
  concrete density the bars' weight has to be added as a load. Behaviour
  unchanged.

## [1.4.0] — 2026-11-06

The fourth feature release on the frozen 1.0 formats: nothing that 1.0 to
1.3 reads or writes changes meaning (case format 1, result format 3,
checkpoint format 1, API v1; no new case key). One item, reviewed by an
independent agent before it went in: the compiled whole step extended to
fibre beams and to connectors and bearings, the step-by-step loop's answer
to the bit. The whole Hangzhou Bay seismic record (fibre beams on pot
bearings) on the reference machine with 8 threads: 150.9 s (1.1: 23.9 min).
One fix rides along: a fibre model's bearings are now held to the
connector rules. Every contract is unchanged.

### Added

- **The compiled step for fibre beams and bearings**.
  Connectors and bearings are compiled (`vfife/kernels/numba_connector.py`:
  the deformation, the finite-difference Jacobian, curves, gap, bilinear,
  friction and the planar pair, dashpots, slip); they replace the NumPy
  path in fibre models and, under `backend: numba`, in mixed line models
  (`VFIFE_COMPILED_CONNECTORS=0` switches back). It agrees with the NumPy
  reference to round-off (about 1e-8 in force), not to the bit. The
  compiled whole step now takes connectors, and fibre models get one of
  their own around the fused kernel; both are the step-by-step loop's
  answer to the bit, whatever the thread count, restarts included. Hangzhou
  Bay seismic on two cores: 2.33 to 0.88 ms a step. A fibre run under
  20000 steps keeps the loop, since compiling its step costs about 3 s per
  process; `VFIFE_COMPILED_STEP=1` forces it.

### Fixed

- The bearings of a fibre model (`element.fiber_connectors`) were not held
  to the connector rules: a law on a component that also had a stiffness,
  or friction pressed by a component with no stiffness, was accepted and
  ran (the friction carried nothing). They are refused now, as in a mixed
  line model.

## [1.3.0] — 2026-11-06

The third feature release on the frozen 1.0 formats: added, nothing that
1.0, 1.1 or 1.2 reads or writes changes meaning (case format 1, result
format 3, checkpoint format 1, API v1; the one new case key,
`static.arc_length.control`, is read only when written). Two items, each
reviewed by an independent agent before it went in:

- the explicit step of frame and frame-and-bar models as one compiled call,
  the step-by-step loop's answer to the bit (Xihoumen on two cores: 30.1 to
  14.9 s per simulated second);
- displacement control in the arc length. It does not take case_24g past
  its second peak, which the item was for: that peak appears to be a
  snap-back of the deflection, and an increment that would jump it is now
  refused. case_24g and every contract are unchanged.

### Added

- **The compiled whole step** (1.3, item 10). Explicit dynamics of a frame
  (`beam2`, `beam2_timoshenko`) or a frame-and-bar mixed line model under
  `backend: numba` takes the steps between two progress or output points
  in one compiled call (`vfife/kernels/numba_step.py`,
  `vfife/core/compiled_step.py`): loads (fixed vectors with their
  amplitudes, stacked groups, a uniform ground motion), member forces,
  no/Rayleigh/viscous damping, the energy ledger, the velocity, position
  and orientation updates and the held dofs, with the points divided
  between threads as the members are. The state, monitors and reactions
  are the step-by-step loop's to the bit, whatever the thread count; the
  energy ledger is too wherever numba's `np.dot` and NumPy's are the same
  BLAS routine (they are on the reference machine and in CI's container;
  elsewhere the ledger may differ in the last place). Xihoumen on two cores: 14.9 s per simulated second (1.2.1: 30.1 s).
  Steps with code of their own keep the loop; `VFIFE_COMPILED_STEP=0`
  turns it off.

- **Displacement control in the arc length** (1.3, item 11).
  `static.arc_length.control: {region, dof}` measures each increment's arc
  on one node's translation instead of the whole state: displacement
  control with the load factor as the unknown, as OpenSees'
  DisplacementControl. The controlled point moves on at every increment,
  so the elastic unloading from the last state is not on the arc. A
  snap-back of the controlled displacement cannot be followed: an increment
  that converges on another branch (the state moving more than 1.5 times
  as fast per unit of the control as over the last increment) is refused
  and the arc halved, and when the cutbacks run out the message says it
  may be a snap-back. The region must be one node, not held, and moved by
  the pattern. case_24g is unchanged: its second peak is not passed this
  way either.

### Changed

- **The member frame and the orientation update hold their 3x3 products in
  local variables** (`kernels/numba_rotations`): the same operations in the
  same order, so the same numbers to the bit, with no per-member scratch
  arrays. The frame kernel is about a quarter faster; the step-by-step
  Xihoumen loop goes from 30.1 to 27.5 s per simulated second.
- **The rotation, frame and step kernels divide as IEEE does**
  (`error_model="numpy"`). A frame that turns inside out (a step far above
  the stable one) used to end in a ZeroDivisionError traceback from inside
  the kernel -- from a threaded kernel a SystemError; it now gives
  non-finite forces, as the NumPy path does, and the run stops as
  NONFINITE with the location. The same holds wherever these kernels run:
  the fibre beam (`member_frame`), relaxation, implicit static, arc length
  and buckling, each of which already stops on a non-finite state. No
  finite result changes.

## [1.2.1] — 2026-11-06

### Fixed

- **1.2.0 did not import on Python 3.10 and 3.11.** Two error messages in
  the fibre solver reused their f-string's own quote inside the braces,
  which only Python 3.12 accepts (PEP 701); on 3.10 and 3.11 every import
  of `vfife.core.structural` -- every analysis -- raised a SyntaxError.
  The reference machine and the development container run 3.12 and 3.13,
  so only CI (3.11) saw it. A test now reads every module's tokens and
  refuses such an f-string, and the fast test layer was run on 3.11 (all
  passed). No result changes.

## [1.2.0] — 2026-11-05

The second feature release on the frozen 1.0 formats: added, nothing that
1.0 or 1.1 reads or writes changes meaning (case format 1; result format 3
with a new root attribute `model_point_load`, a field `member_built` and
two frame attributes `analysis_step` and `procedure` written only by runs
whose steps use them; checkpoint format 1, the new keys only in such runs;
API v1; a 1.0 or 1.1 checkpoint still resumes). Four items, each reviewed by
an independent agent before it went in:

- soil springs on piles (`soil_springs`, p-y curves by depth);
- `vfife new`, scenario templates that run as they stand;
- box and T girders from the drawing's dimensions, bars in layers;
- analysis steps: a procedure per step, members built and removed,
  supports released, loads and constraints kept, each segment's own
  concrete age.

One behaviour changed (below): `vfife summary` reads each load at its
amplitude's value at the step's end. Local full check 20261105_2055
(79.7 min, 16 cores): all passed.

### Added

- **`vfife new <template> [case]`** (1.2): scenario templates that run as
  they stand -- `girder` (a three-span continuous girder), `pier_on_piles`
  (a pier on a pile in p-y soil), `arch` (a parabolic arch rib), `cable`
  (a hanging cable by its unstressed length) -- written with a line mesh in
  the case, commented, with the schema file beside them. Each is tested:
  written, checked, run, and its summary's reaction check is 1.
- `vfife summary` also counts the nodal forces (`vector` loads) in its
  reaction check (`model_point_load`, a new root attribute).
- **Box and T girders by their drawing's dimensions** (1.2): `shape:
  box_girder` (height, widths, cantilever, slab and web thicknesses,
  tapered cantilevers, haunches, several cells) and `shape: t_girder`
  (optionally with a bottom flange), on a `beam` section (A, Iy, Iz from
  the outline; J by Bredt, multi-cell by the cells' shear flows) and on a
  `fiber` section (the outline as patches, the same J, `origin: centroid |
  top | bottom`). **`bar_layers`** on fibre sections: straight rows of bars
  by `count` or `spacing`, measured from the top or bottom face with
  `z_from`.

- **Analysis steps** (1.2, line models: beam, Timoshenko, mixed_line and
  the fibre beam, numpy and the fused kernel): a step may have its own
  `procedure` (relaxation, static or dynamic, with `total_time`), build
  members (`activate`, born stress-free where the structure has moved to,
  `birth: tangent | design`), remove them (`deactivate`, their forces taken
  off over the step) and free inherited supports (`release`, the reaction
  taken off over the step); a step's load or constraint with `keep: true`
  stays on afterwards. Results carry `member_built` and each frame's
  `analysis_step` and `procedure`; a run with a dynamic step records the
  clock as its time. Checked on staged two-span and cantilever
  construction against closed forms, and a relaxed state handed to a
  dynamic step stays still to 1e-9. On the frame models a segment born in
  a step has its own concrete age (`activate_age`): it creeps, shrinks and
  ages on its own clock, checked against phi(t, t0) of each bar.

### Changed

- `vfife summary` takes each load at its amplitude's value at the step's
  end (1.1 took full amplitude: a pushover on a cyclic table ending at zero
  read 0.73 in the reaction check of a correct run).

- **Soil springs on piles** (1.2): `soil_springs: [{region, ground, vertical,
  direction, curves: [{depth, p_y}]}]` -- p-y curves the user gives by
  depth (the code formulas are not in the library), interpolated between
  depths, lumped at the pile's nodes below ground by the length each stands
  for, radial in the horizontal plane (or along one axis). Nonlinear
  elastic: no hysteresis, no gap. The soil's force is a reaction (the
  `reaction_force` field, reaction monitors, `vfife summary`); the springs
  are in the internal force, so modes, buckling and the static solver see
  them. Beam, Timoshenko, fibre and mixed line models. Checked against
  Hetenyi's beam on an elastic foundation (0.48% on 60 elements, second
  order).

## [1.1.0] — 2026-11-05

The first feature release on the frozen 1.0 formats: everything here is
added, nothing that 1.0 reads or writes changes meaning (case format 1,
result format 3 -- three new root attributes --, checkpoint format 1, API
v1; a 1.0 checkpoint still resumes). Five items, each reviewed by an
independent agent before it went in:

- `line_load` with its mass (`with_mass: true`), and default output fields
  that follow the model;
- `vfife summary`, one page on a result;
- `vfife buckling`, linear buckling of line models;
- `mesh: {type: lines}`, a line mesh written in the case (Gmsh unchanged);
- the seismic template and the thread benchmark start from `initial_state`
  (the whole El Centro run of `_hzb_seismic` on 8 threads, measured: 23.9
  min -- the 9.0 min of 0.8 was scaled from the first second, and the
  record's nonlinear second half costs about twice as much per step).

Checked on the reference machine (16 cores, Windows, Python 3.12.13) on
2026-11-05 without `--quick`: all 60 case contracts, the default and slow
layers, and the package -- results/check_all 20261105_1057 (41.5 min).

### Added

- **A `line_load` that carries its mass** (1.1): `with_mass: true` on a
  model-level `line_load` adds the mass |q|/g of what the load is the weight
  of (superimposed dead load, surfacing, parapets), half of each member's to
  each end node, in all three translations. The mass weighs nothing -- the
  load is its weight -- so `gravity` does not count it twice; g is 9.80665
  m/s² in the declared units. Until now a superimposed dead load written as
  a `line_load` weighed and had no inertia, and a seismic run, the modes and
  a response spectrum left it out (the bridge scenario).
  Refused on a step's `loads` (a mass is the model's) and on any load but a
  `line_load`. Beam, Timoshenko, fibre, mixed and coupled models; a member
  that splits off takes its share of the mass with it, as it takes the
  load. Preflight says `LINE_LOAD_WITHOUT_MASS` (INFO) for a dynamic run's
  line loads without it, and `UNIT_DEFAULT_SI` when the declared units
  cannot convert g.

- **`vfife summary <result.h5>`** (1.1): one page on a result -- status and
  convergence; the model's mass, the weight its `gravity` loads apply and
  the resultant of its line loads beside the reactions at the last frame
  (with "reaction / (weight + line loads)" for a static result, 1 when those
  are the only loads); the largest displacement, rotation, signed member
  force, stress and connector force over every frame with where and when;
  failures; energy balance; the monitors' ranges. `--json` gives the same
  as data, `--case` the mass of a result written before 1.1. The mass,
  weight and line load are new root attributes of the result
  (`model_mass`, `model_weight`, `model_line_load`; result format 3,
  additive).

- **`vfife buckling <case>`** (1.1): linear buckling -- the load factors of
  a step's loads at which the model loses its stiffness, and the buckled
  shapes, (K_b + λ K_G) φ = 0. K_b is the finite-difference tangent `modal`
  uses; K_G is assembled from the members' axial forces in the linear
  increment (consistent beam and string geometric matrices), so it finds
  the flexural buckling of columns, frames, towers and arches, not
  lateral-torsional or torsional buckling. From the reference configuration
  by default (refused with internal forces there: `BUCKLING_BASE_NOT_AT_REST`),
  or `--result` from a state a run reached, the factor then being the load
  added to it. Mechanisms and shells/solids are refused by name. Written to
  `<case>_buckling.h5` (`buckling_format` 1), `--paraview` as in `modal`.
  Checked against Euler, Greenhill, Engesser and the stability functions of
  a sway portal (0.01%-0.2%). A first version that took K_G as the
  difference of two tangents was rejected in review: it read the member
  rotations of any bending load as softening (factors 3-300 times low).

- **A line mesh written in the case** (1.1): `mesh: {type: lines, points:
  {...}, lines: {...}}` -- named points and named groups of members, each
  segment divided by `divisions` or `size`, straight or as an `arc`
  (parabola or circle, by its `rise`). It builds what a Gmsh Line2 file
  reads as (a region per point and per line, an element group per line for
  `assign`): case_03 and case_22 rewritten this way give their Gmsh results.
  Named points are nodes and lines meet only there (two nodes at one place,
  or two members on one pair of nodes, are refused); `groups` joins lines
  into one element region, as a Gmsh group of several pieces. Gmsh meshes
  are unchanged; a model with shells or solids keeps them.

### Changed

- **The default `output.fields` follows the model** (1.1). A case that
  names no fields now writes `displacement`, `rotation`, the reactions,
  `von_mises` on solids and bars (`von_mises_top/bottom` on shells), the
  members' signed forces (`axial_force`, `shear_y`, `shear_z`, `torsion`,
  `bending_y`, `bending_z`), and, where the model has them, its plastic
  hinges', connectors' and breakable members' fields. The 1.0 default,
  `[displacement, von_mises]`, wrote a frame's or a truss's displacements
  and nothing else. A case that names its fields writes exactly those, as
  before. Results of cases that relied on the default are larger: line
  models gain rotations, forces and reactions; solids gain
  `reaction_force` (an array the size of `displacement` per frame).
  The result format is unchanged (3): only which fields are present.
- A checkpoint written by 1.0 still resumes: `with_mass: false`, the
  default, is left out of the model fingerprint.

## [1.0.0] — 2026-11-04

The first stable release. It is 1.0.0rc1 unchanged: nothing was found
between the candidate and this release that needed a fix. What 1.0 freezes
is listed under 1.0.0rc1 below and holds for every 1.x release:

- the case-file format 1, the result format 3 and the checkpoint format 1,
- the Python API v1 (`vfife.api.__all__`),
- the command-line commands and options, and the error codes.

A 1.x release may add (a new element, material, mesh type, command or
field) but does not change what an existing case means or what an existing
result file contains; Gmsh meshes (`mesh: {type: gmsh}`) stay as they are.
Shells (shell3, shell4, shell_line) remain experimental and outside these
promises.

Checked on the reference machine (16 cores, Windows, Python 3.12.13) on
2026-11-04 with no `--quick`: all 60 case contracts (case_24d and case_21b
included), the default layer, the slow layer, and the package (the wheel
builds, installs into a fresh environment and runs a case outside the
repository) -- results/check_all 20261104_1622.

## [1.0.0rc1] — 2026-10-13

The first release candidate of 1.0. From here to 1.0.0 only fixes go in;
what 1.0 freezes is fixed as of this candidate:

- the case-file format 1 (`version: 1`; `docs/YAML参考.md`, `docs/vfife.schema.json`),
- the result format 3 and the checkpoint format 1 (`docs/result_format.md`),
- the Python API v1 (`vfife.api.__all__`, `docs/API参考.md`),
- the command-line commands and options, and the error codes (`docs/error_codes.md`).

Shells (shell3, shell4, shell_line) stay experimental and outside these
promises. Everything below was reviewed in five area rounds, two reviews of
the fixes themselves and one round of end-to-end scenarios;
the whole default layer, the slow layer and every case contract pass
(results/check_all 20261013_1552 on the reference machine).

### Added

- **Editor completion and member axes in `info`**: `vfife schema
  [-o FILE]` writes the case file's JSON Schema (`vfife.schema.json`,
  draft 2020-12, titled "vfife case file, format 1") and prints the line
  `# yaml-language-server: $schema=...` that points VS Code's YAML
  extension (Red Hat) at it: completion of keys and enum values, hover
  help, a red underline on a misspelt key. Every field carries its `#:`
  comment as `description` (the text of docs/YAML参考.md, Chinese and
  English), a material's `type` is an enum of the catalogue, and the
  `fracture` / `bond` / `friction` / `failure` sub-blocks have their own
  fields; `api.case_schema()` returns the same schema. A generated copy
  ships as `docs/vfife.schema.json`, kept current by `python -m vfife
  reference` (and `--check`). `vfife init` writes the modeline as the
  case's first line and `vfife.schema.json` beside it (an existing one is
  kept). `vfife info` and the run summary print each line-member group's
  local axes in global coordinates (`axes  girder (beam2_fiber, 42
  members): local x = +X, local y = +Z, local z = -Y`; a curved or mixed
  group as "within N deg of" or "varies"; groups with the same axes on one
  line), and a `note` when a rect/box section (width not equal to height,
  beam or fibre) on a horizontal member has its `height` lying flat --
  local z off the vertical (opposite to the gravity load, else +Z) -- with
  the `orientation` that turns local z up. Patches, circles, tubes, A/Iy/Iz
  sections, bars and connectors get the axes line only. Manual 2.4, 7.5 and 14;
  tests/test_review6_ux.py.
- **Response-spectrum analysis**: a `response_spectrum`
  block makes `vfife run` take the modes of the tangent at the state the run
  ends in and combine each mode's peak, phi Gamma S_a(T) / omega^2, by CQC
  (Der Kiureghian, one damping ratio) or SRSS, directions by SRSS. The
  spectra are the user's tables (`table` or `path`, `scale`); periods off
  the table are refused unless `beyond_table: hold`; the modes are the
  lowest `modes` or as many as reach `mass_fraction`. Displacements, member
  end actions and support reactions come from the model moved along each
  mode (the tangent the modes were found with), the base force is combined
  from the modal sums. Results in the `response_spectrum` group; contracts
  read them with `source: spectrum`. `case_35` (a uniform cantilever against
  the continuum's closed form) and tests/test_response_spectrum.py (a
  two-mass chain against its 2 x 2 eigenproblem, CQC and SRSS).
- **Multiple-support excitation**: a constraint with a
  `record` (`path`, `quantity` displacement / velocity / acceleration,
  `scale`, `delay`, and a travelling `wave` with `speed`, `direction`,
  `origin`) makes its region follow its own ground motion, in absolute
  coordinates; translations only, dynamic only, frame, fibre-beam and shell
  models (solids and standalone trusses refuse it). A displacement record
  imposes its velocity too; the energy ledger books the work the ground does
  through the reactions. `case_36` (a deck between two piers 400 m apart, a
  wave at 1000 m/s putting them nearly in anti-phase) against its closed
  form; tests/test_multi_support.py (the same record at every support is the
  uniform `ground_motion`, a slow settlement gives 12 E I d / L^3, the wave's
  delays, the ledger).
- **Added mass**: `nodal_masses` takes `components` (which
  translations carry the mass), `weight` (whether gravity sees it; a
  directional mass is added mass and does not weigh) and `inertia` (a rotary
  inertia about the global axes, beams and shells). `hydrodynamic_mass` is
  the Morison added mass of members under water, C_a rho_w A per metre of the
  part below `water_level`, across the member, weightless; for members within
  5 degrees of a global axis. `case_37` (a pier in 20 m of water under a
  response spectrum) against the cantilever-with-tip-mass closed form;
  tests/test_added_mass.py.
- **Vehicles with mass**: a `vehicle` load is a rigid body
  (heave, and pitch on more than one axle) on per-axle suspensions, unsprung
  masses and tyres that push and never pull, riding the deck's deflection
  (read back through the same Hermite shape functions `moving_load` pushes
  with) plus a road `roughness` -- a profile file, or a displacement PSD
  sampled with a seed (no class table in the library). Integrated with the
  structure by the same central difference and step, from static
  equilibrium; an axle without a tyre rides the road rigidly (the moving
  sprung mass). `kind: vehicle` monitors. Frame and fibre-beam models,
  dynamic only. `case_38` (Yang & Yau's sprung mass crossing a 25 m beam)
  against an independent modal solution; tests/test_vehicle.py (a quarter
  car against its frequency response, a truck's axle loads by the lever
  rule, a PSD road's variance).
- **Cracked shear for fibre sections**: a section's `shear`
  law -- uncracked kappa G A, then (from the member's first flexural crack,
  or a stated shear) a cracked stiffness given or taken from the 45-degree
  truss of its `stirrups` (Park & Paulay), origin-oriented unloading, an
  optional stirrup-yield plateau -- in series with the fibres' bending
  through one shear rotation per member, relaxed with the solution. Elastic,
  it is the Timoshenko cantilever to 1e-8. NumPy backend. `case_21b`
  (Vecchio & Emara's frame with it: 30 mm load 1.15 -> 1.08 of the test,
  nothing fitted); tests/test_fiber_shear.py.
- **Implicit static solver**: `solver.mode: static` --
  Newton per load level on a tangent built by finite differences of the
  internal force, with a distance-2 node colouring so that a tangent costs a
  few dozen force evaluations whatever the model's size; tangent predictor,
  backtracking line search, cutbacks that climb back to the stated levels,
  trial/commit of path-dependent state per level. Frames and fibre beams.
  `case_21c` (case_21 by Newton: 8 s against 22 minutes, same contract);
  tests/test_static.py.
- **Arc length**: `static.arc_length` makes the step's load
  (and the displacements its constraints impose) a pattern whose factor is an
  unknown; Crisfield's cylindrical constraint, rotations weighted by the
  model's size, a predictor that keeps the direction of the last increment,
  an arc that adapts to the iterations taken, landing on the full load by
  load control, and a `stop` displacement. `case_16b` (Lee's frame by arc
  length, both NAFEMS points and the snap-back); the snap-through truss
  against its exact path to 1e-9.
- **Arc length on chosen steps, and threads for the fibre kernel**:
  `arc_length.steps` names the steps that take it (the
  others take load-controlled Newton), and an arc-length step first settles
  any imbalance at lambda = 0. `solver.threads` (1 serial, 0 all cores, n)
  spreads the numba fibre-beam kernel's elements over cores; each element
  writes its own row and the rows are summed in element order, so the answer
  is the serial one to the bit (2 cores: 3.4 -> 2.1 ms per force on the
  Hangzhou Bay unit). `case_24g`: case_24d's 50-minute explicit pushover as a
  Newton dead step and an arc-length live step, 30 s, through the peak
  (k = 14.79) onto the softening branch; its dead state equals the dynamic
  relaxation's (-23.18 mm), where case_24d's t = 0 load leaves -27.77.

- **Threads for the frame kernel, and a compiled loop for the axial members**:
  `solver.threads` now also runs the numba beam2 kernel
  (pure frames and mixed meshes) in contiguous chunks, each member writing
  its own row, summed in member order -- the serial answer to the bit.
  `truss2` members under `backend: numba` go through one compiled loop
  (NumPy's arithmetic, to round-off, slack cables included). Xihoumen: 0.62
  -> 0.33 ms per frame force on two cores, 42.5 -> 30.5 s per simulated
  second. `benchmarks/bench_threads.py` measures it on a given machine.

- **Case format 1 and a kept result file** (route 0.9 item 1):
  a case may declare `version: 1`; absent, it is read as 0.x wrote it, and a
  larger number is refused as written by a newer vfife. `vfife migrate` now
  also brings an input-form 0.x case to format 1 IN ITS OWN TEXT -- the old
  spellings renamed at the positions the YAML composer reports, `version: 1`
  added, comments and layout untouched -- after proving that the two texts
  lower to the same model. Every reader of result files opens them through
  `open_result`, which refuses a newer `result_schema_version` by name
  (`RESULT_TOO_NEW`) and reads older files; `tests/data/result_format_2`
  keeps a result of today's format (monitors, frames, checkpoint) that the
  monitors reader, `verify`, the 3-D view, `--restart` and `initial_state`
  are tested against. Every shipped case is format 1.

- **Python API version 1** (route 0.9 item 2): `vfife.api.__all__`
  is the API, frozen for 1.x (names and parameters kept; new parameters only
  with defaults), with `API_VERSION = 1`. New: `load_case` and `CaseDraft`
  (a case as data, read and changed by dotted keys such as
  `loads[0].vector[2]`, remembering its folder so a changed copy still finds
  its mesh), `study` (every combination, or zipped lists, of parameter
  values; one result file and one row each, a failed run recorded and the
  study continued, rows written to `study.csv`), and `read_monitors` /
  `verify` re-exported. `AnalysisOutcome` gains `times`. `docs/API参考.md`
  is generated from the docstrings by `python -m vfife reference --api`, and
  tests/test_api_v1.py keeps it, and the frozen names and parameters, in step.

- **The full check runs locally; an English manual** (route 0.9
  item 5): `tools/check_all.py` runs what the nightly CI layer did -- the
  default tests, the slow tests, every shipped case against its contract
  (one single-threaded process per case, `--jobs` at a time) and the wheel
  built, installed into a fresh environment and run outside the repository
  -- and writes a report to `results/check_all/`. `docs/user_manual.md` is
  the user manual in English, section for section (tests/test_manuals.py).

### Changed

- **Shells are experimental and not covered by 1.0**: a
  third review found the triangle `shell3` locking in shear (a thin plate
  deflects 0.2-20% of the true value), `shell4`'s default hourglass term
  stiffening bending by 2-30% in a way refinement does not remove, and
  distorted quads locking. Preflight gives every shell model a
  `SHELL_EXPERIMENTAL` warning; `shell3` runs on numpy only and refuses
  `hourglass_coeff`. The element formulation (MITC4, MITC3) is after 1.0.
- **A dynamic step's load is read at the start of the step**,
  where the state is: the frame, fibre, truss and solid loops read
  loads and ground acceleration at t_(n+1), which made the time history
  first-order in dt (0.08% on a frame, 1.1% on a solid at the default step).
  Every dynamic answer moves by O(dt).
- **Damping says what it does or is refused**: `alpha` acts
  under `type: rayleigh`, and alongside `kinetic` in a relaxation (v0.1's
  combination for softening problems -- now on the frame, fibre and truss
  relaxations too, which ignored it); under `viscous` or `none` it is
  refused. `beta` needs `type: rayleigh` in a dynamic run. `kinetic` in a
  dynamic run is refused (it did nothing on frames and zeroed the velocity
  on trusses and solids).
- **Kinetic damping resets at the energy peaks, not on every step**:
  a continuous (unstaged) relaxation of a frame, fibre or
  truss model kept the pre-reset energy as the one to beat and so reset on
  every step -- a steepest descent, 73 941 resets in 74 043 iterations, 48
  times slower than the staged path on a cantilever. Continuous relaxations
  now converge faster, and to their tolerance rather than past it.
- A solid's dynamic results carry `time` in seconds (it was the step
  number plus the step's progress: a 2 ms run ended at 1.0; case_19's
  contract reads 0.020 s now).
- A dynamic run is refused `load_steps` (it staged nothing and changed the
  default amplitude: the response at 0.25 s halved) and a step's own
  `n_steps`; a relaxation step's own `n_steps` is honoured by the frame,
  fibre and truss solvers (they used the solver's).
- The truss's dynamic step is taken from the masses its nodes carry
  (Gershgorin), `nodal_masses` included: with rho = 1 and a lumped mass it
  was 500 times too small; `linear_elastic` refuses rho <= 0 by name (it
  divided by zero).

- **`steel02` is OpenSees' Steel02**: `Steel02::setTrialStrain`
  line for line -- each branch aims at the intersection of the elastic line
  with the other side's hardening asymptote, R = R0 (1 - cR1 xi / (cR2 +
  xi)), and `a3`/`a4` shift the tension side (they were read and not used).
  It agrees bit for bit with an independent transcription over a 3000-step
  random history. The law it replaces kept R near R0 (the Bauschinger
  rounding hardly showed: 0.58 eps_y after a 1% excursion, now 4.61), put
  the monotonic envelope 3.45% of Fy over its own asymptote (v0.4's open
  finding, now closed), forgot the hardening envelope at every reversal (a
  2e-4 wobble at 2% strain took the stress at 3% from 470 to 440 MPa) and
  could create energy on a small closed loop. **Every case with a `steel02`
  bar moves**: flexural capacity with bars on their plateau falls about
  2.5%, and the first branch is the rounded curve from the origin (0.45%
  under E eps at 0.89 Fy). State is 13 columns (`n_state` was 8); `cR1`
  must lie in [0, 1), `cR2`, `a2` and `a4` be positive. The shipped cases,
  re-run on both laws: peaks and plateaus 2.5-3% lower (case_21 346 -> 337.5
  kN, case_31 137.7 -> 133.7 kN, case_25 225 -> 219 kN), case_24c's girder
  no longer holds on its strands after the pier yields (727 mm at k = 14.8,
  was 17.3), the eight corroded beams 2.6-2.9% lower (mean error 4.1 ->
  6.2%). Contracts against measurements keep their targets: case_21b's
  plateau at 155 mm is held at 7% (model 6.4% low) and case_25's peak
  deflection at 14% (13% short). Regression contracts (case_24g, case_29f)
  take the new values. case_30's push ends at 0.8 x 45 mm, past the
  measured 36.5 mm failure deflection: the crushing now localises in the
  midspan element after the peak, and the last levels to 45 mm did not
  settle.
- **An arc-length increment that turns back is not accepted**:
  a corrector that wandered could settle on the elastic unloading
  from the last accepted state -- an equilibrium, and the path back down;
  case_24g did so at its limit point under the new `steel02` and unloaded to
  a negative load factor. An accepted increment must point along the
  previous one, as the predictor does; otherwise the arc is halved.

- **CI no longer runs on a timer**: the nightly matrix
  solved every case every night, sixty jobs of up to two hours, and spent a
  private repository's month of Actions minutes in days. The solving layers
  run when started by hand.
  tests/test_ci_layers.py holds both.
- The case-contract test (`test_a_case_solved_satisfies_its_contract`)
  loads each case with `api.load_case`, so every relative path a case
  names -- a support's own record, a roughness profile, an initial state --
  resolves against the case's folder, not only the mesh and the ground
  motion it used to fix up by hand.
- Packaging metadata: `license = "MIT"` with `license-files` (the table
  form and the licence classifier are deprecated by setuptools).

- **One name per section quantity**: `width` / `height`
  for a rectangle and a box (a beam's were `b` / `h`), `diameter` for a
  circle and a tube (a beam's was `d`), `j` for the torsion constant (a fibre
  section's was `torsion_constant`). The old spellings are still read, with a
  preflight `DEPRECATED_FIELD` warning; giving both with different values is
  refused. A `j` on a fibre section used to be dropped without a word; it now
  reaches it.

- **The static solver's difference step follows the element, not the
  model**: `perturbation` now multiplies the median element
  length instead of the largest coordinate. On a 580 m bridge the old step
  (3e-5 m) straddled the concrete's cracking kink, Newton went linear near
  the peak and the arc shrank for ever. Under Newton the fused fibre kernel
  restarts from the converged state with two array copies instead of one
  per fibre group (1.4 ms a force evaluation on that model). An arc-length
  corrector that has not halved its residual in eight iterations gives up,
  and an arc just halved does not grow straight back.

- **Nodal masses weigh on trusses and solids too**: under
  `gravity` a truss's or a solid's `nodal_masses` were inertia without
  weight, while every frame solver weighed them. They now weigh everywhere,
  unless `weight: false`.

- **A dynamic run's step divides its duration**: `dt` is
  the stable step shortened so that `n_steps * dt == total_time` (the bound
  is kept as `stable_dt`). The loops label step i with `i / n_steps *
  total_time` -- the instant records and amplitudes are read at -- while the
  particles advance by `dt`; with `n_steps = round(total_time / dt)` the two
  clocks differed by up to half a step over the run, a time-scale error that
  showed as a frequency error and did not converge with the step. Every
  dynamic answer moves by at most `dt / total_time`; the dynamic contracts
  (05, 19, 24c, 24d, 26, 33, 34) pass unchanged.

### Fixed

**Final review: whole-bridge seismic workflow** (tests in
tests/test_review6_bridge.py):

- A fibre member's mass and weight now take each fibre's own material density (concrete, bars, tendons, per section and per patch); it was the first fibre section's cover density times the whole area. A 30+40+30 m bridge was 1.4% light; a steel girder written first would have weighed concrete piers as steel. case_25 and case_31 values move 0.03–0.35%; the case_25 140 mm band is widened from 9% to 9.5%. A model whose fibres all share one density moves only at round-off: the mass is now summed fibre by fibre (Σ ρA rather than ρ ΣA), about 4e-16 relative, so the stable step is bit-identical but dynamic answers are no longer bit-identical to 0.5.0.
- `vfife modal --result` and `vfife probe --result` restore the full checkpoint state (fibre, connector-law and creep histories, relaxed prestrain) when the frame is the result's last; other frames are positions only and say so. On friction bearings T1 was 0.868 s against 0.508 s for the spectrum at the same state; both now give 0.5125 s.
- Modes and spectra at the end of a creep run hold the creep reached at the instantaneous modulus (`hold_time_strain`), instead of the last station's E/(1+φ): periods were 20–35% long and depended on the station count, and the spectrum base force was 19% below modal mass × Sa.
- Support displacement records are read as a C1 cubic Hermite, so the support velocity is continuous. Pier base shear from a displacement record was 3.2× that of the same motion as an acceleration record; now 698 against 700 kN.
- Spectrum and `modal` warn by name when a connector law sits on its kink (sliding, yielding, at a gap stop); the spectrum records `connector_kinks`.
- `vfife info` lists connector sections and the total mass and weight; fibre runs print the Rayleigh α/β and the β step factor.
- Relaxation convergence series are sampled at every progress check, as the result format states.
- `docs/YAML参考.md` documents `response_spectrum`; `describe beam2_fiber` lists `static`, the fibre fields and connector fields; `connector_slip` is accepted as a field group.
- Wording: box-with-bars message points to `patches`; `sweep --help` default; `mesh` on a line mesh no longer suggests `init`; run progress shows `increment … iterations/s` and the summary `analysis step k of n`; stacked plot panels no longer overlap, and the time axis states its unit or load factor.

**Final review: service life to collapse** (tests in
tests/test_review6_service.py):

- Element force fields are signed in every line family: `axial_force` tension-positive, shears/torque/moments by the right-hand rule in local axes on the +x face (larger end, with its sign; a fibre member's largest station). `*_max_abs` keeps the old unsigned envelope. A sagging moment with local z up is negative. Sign conventions are documented in `docs/result_format.md`. case_28d's pier-moment metrics now read `peak_abs` (value unchanged).
- A member that has failed (broken, removed, split off) reports zero force.
- `kind: element` monitors on frame, mixed_line, coupled, fibre and truss models (any element field over a region's members, with `reduce`); unknown fields are refused (`UNKNOWN_MONITOR_FIELD`).
- In `to_age` steps, loads and prescribed displacements default to `hold`.
- Static runs without `output.every` write a frame at every increment.
- Arc length: an increment within one iteration of the target keeps its arc; a small arc that keeps converging without cutbacks is doubled. The corroded pushover finishes without tuning. The max-increments message reports the stop position.
- Limit and rupture messages give the step loads reached relative to their written value.
- `vfife view` accepts runs of one model with different spare-particle counts.
- Restart event tables carry `rows_before_restart`; results carry `restarted_at_time`.
- `probe` skips the force-free and stability checks of self-stressed models at the reference geometry and arc.
- Preflight no longer reports a rigid-body mode for a straight bar chain in mixed_line.
- `run` reports unknown output or monitor fields as a preflight failure; the `--debug` hint names `vfife --debug <cmd>`; the unit hint only appears for magnitude errors; static removal refusals and restart mismatches name the mode and the changed key.

**Final review: the documentation as specification** (tests in
tests/test_review6_docs.py):

- **Behaviour change: a material parameter written under two of its names with different values is refused** (`MATERIAL_ALIAS_CONFLICT`), as a section's old and new spellings already were. `concrete01`/`concrete02` read `fc` for `fpc` (and `eps_c0`, `fcu`/`fpcu`/`fcu2`, `eps_cu`), `steel02` `E0` for `E` and `yield_strength` for `Fy`, and took their own name when both were given: the manual's parameter-study example, `{"materials.concrete.fc": [3e7, 4e7, 5e7]}` on a case with `fpc: 40e6`, ran three identical analyses at 40 MPa with no warning. A study now refuses it before the first run. The manual §12.6 example and the `CaseDraft` docstring use `fpc` and the chapter-3 case's `line_load`.
- **Behaviour change: defaults that carry a unit are written in the declared units.** A vehicle's `gravity` (9.80665 m/s²), the water `density` of `hydrodynamic_mass` (1025 kg/m³), a stirrup modulus (2e11 Pa), a roughness PSD's `n0`/`n_min`/`n_max`, and the materials' `rho`, `fpc` (concrete01/02), `E`/`Fy` (steel02), `E` and a grade's `fy`/`fu` (steel_cyclic) and `bond_ceb`'s `tau_max` and slips were SI whatever the case declared: case_38 in mm / N / MPa / t gave a contact force of 56.3 N against 56 306 N and a deflection a thousand times small, preflight PASS. They are converted from SI when a case declares non-SI units (`vfife/core/units.py`); preflight lists each (`UNIT_DEFAULT_CONVERTED`, INFO) a key the declaration leaves out is SI (m, N, s), and a unit it cannot convert to leaves the default SI with a warning (`UNIT_DEFAULT_SI`). SI cases are unchanged.
- **A `fiber` monitor's point is checked against the section outline** (`FIBER_MONITOR_OUTSIDE`): `at: [-0.6, 0.0]` in the manual's own §12.1 example read the side of a 0.5 m wide, 1.2 m deep girder at mid-depth (4.3e-5 against the soffit's 1.03e-3) without a word; the example is now `[0.0, -0.6]`. A `field: stress` reading can be held to `fiber_kind: concrete | steel`, and the monitor records what it read (`read_element`, `read_station`, `read_y`, `read_z`, `read_fiber`, `read_material`).
- The command line prints a refusal's code, `error [CHECKPOINT_MISMATCH]: …`, and `vfife check`'s `MODEL_BUILD_FAILED` names the code inside. `docs/error_codes.md` lists every refusal and preflight code with the exit codes and analysis statuses, generated from the source (`python -m vfife reference --codes`, kept in step by a test).
- `vfife verify <case>` finds `verification/<stem>.yaml` beside the case's folder, then in it, then under the working directory (it looked only under the working directory, so a case run from elsewhere had no contract), and reads the result the case's `output.path` names.
- `api.UserError` is exported (API version 1, additive) and the API module documents what its functions raise; `api.verify(contract, result)` with the arguments swapped is refused as `VERIFY_ARGUMENTS` (it was a `UnicodeDecodeError`).
- XDMF names the sixth stress component `stress_zx`, as the stored order, the monitors and the contracts do (it was `stress_xz`); monitors and contracts read either spelling.
- `RIGID_BODY_MODE` on a six-dof model proposes the rotational dof that holds a free rotation, six entries long (it proposed a three-entry translation at the node already pinned).
- `describe` and the YAML reference list every field a line model writes (the signed member forces, `*_max_abs`, `capacity_ratio`, `member_failed`, `von_mises` on bars) and `static` for the elements `mode: static` runs.
- A plotted series of one or two records is drawn with markers (a one-frame result gave empty axes). The `paraview` extra installs `meshio` for `vfife sweep` and `vfife modal --paraview`. Fibre `info` prints the mass and weight in the declared units.
- v1.0's fixes reviewed again (tests in tests/test_review6_followup.py):
  - `bond_cyclic`'s defaults are converted like `bond_ceb`'s (it inherits the same MC2010 `tau_max`, slips and `rho`): in a mm / N / MPa case `bond_cyclic` with `fc: 30` had `tau_max` 13 693 and `s1` 0.001 against `bond_ceb`'s 13.69 and 1.0, a bond a thousand times too strong after switching laws, with no message.
  - **A `units` declaration with `stress` but without `length` or `force` names the third through the other two**: `{force: N, stress: MPa}` is mm, `{length: m, stress: kPa}` is kN (`UNITS_INFERRED`, INFO), and the defaults, `GRAVITY_MAGNITUDE`, `UNITS_INCONSISTENT` and the wave-speed check use the completed system. It was read as m beside the declared MPa: a mm cantilever with `steel02` defaults deflected 311 137 instead of 0.96 and weighed 4.6e16 N, preflight PASS. When the two imply no unit (`{force: kN, stress: MPa}`) or `stress` stands alone (other than `Pa`), the case is refused: `UNITS_INCOMPLETE` (ERROR) names the missing key.
  - Displacement records are interpolated with Fritsch–Carlson (PCHIP) slopes: monotone between samples, still C1, and the support velocity still the exact derivative. The three-point slopes overshot a coarse step 0, 0, 1, 1 by 7.4% on each side (the support stepped back before it stepped forward).
  - A fibre member's `*_max_abs` is the station maximum of |value| by the old strictly-larger rule, so it is the pre-v1.0 unsigned field bit for bit; the signed field keeps its tie rule (a later station replaces the first only beyond 1e-9).
  - `vfife info` on a fibre model names the mass unit from the declared units (t for N / mm and kN / m; it said kg for kN / m). `vfife plot` labels element monitors of member forces with the force unit and moments with force × length (also `*_max_abs`, `*_i`, `*_j`).
  - docs/structural_framework.md no longer describes `bending_z` as an unsigned magnitude.
- Documentation: both manuals -- `version`, `assign` and `units` in chapter 4, the defaults converted, velocity carried across dynamic steps (§9.4 said each step starts from rest), an undamped dynamic default (§10.2 said damping was required), a stand-alone truss2 refused by `mode: static`, the failed run's state (§12.6), `sweep` needing `rotation` (§3.5), exit codes 2 and 3 and the error-code catalogue (§14), the `list` order, case_24g's numbers; the tutorial's restart (`-o`), plot paths and monitor names; `docs/result_format.md` (`stress_zx`, the fibre monitor's `read_*`, `load_step`/`location_json`/`units` on diagnostics, `effective_mass`'s `meaning`, `connector_kinks`, the mixed models' flags, `rebar/block`'s dtype); `docs/diagnostics.md` (`INTERRUPTED`, `PREFLIGHT_FAILED`); the YAML reference (`lo` is not friction's, every `units` key, `ground_motion` for every family, the default fields, `version` absent); README and cases/README case_24g numbers, and cases/README rows for case_30 and case_31.

**The v1.0 fixes, reviewed again** (tests in
tests/test_review5_*.py, sections "1.0 re-review"):

- Fibre creep under an even integration rule (`gauss2`, the default, and
  `lobatto4`) is read as the mean of the two stations straddling the element
  middle. v1.0 read the first of them, 0.21 L from each element's node 0:
  the same cantilever crept +6.7% fixed at one end and -5.9% fixed at the
  other. Shipped cases use `lobatto3` and are unchanged.
- `elastic_shortening: compensate` on a section with no stiffness about one
  axis (`rect` with `ny: 1`, everything at y = 0) solves the axes that have
  stiffness (a scaled pseudo-inverse). It raised "Singular matrix"; before
  v1.0 it was NaN.
- `THERMAL_EXPANSION_PARTIAL` is raised only for fibre sections on members a
  temperature load reaches (region overlap, `all`, model or step loads). A
  temperature on the deck warned about the piers.
- Bars and tendons are checked against the section's outer outline, not its
  bounding rectangle: the rectangle of a `rect`/`box`, the circle of a
  `tube`, the union of a `patches` section's patches (to within 1% of the
  section size). Voids, bores and `holes` count as inside. A bar or tendon
  beside a T's web, or in a tube's square corner, is now refused. The
  message gives `s` in the profile's own units.
- `info` names the member theory per section when sections differ, and
  counts "1 rebar" / "1 tendon" in the singular.
- An anchor set too large for the tendon reports the most it can absorb from
  that end. With `jack: both` it read a negative number ("recovers only
  -3.60 mm").
- mode: static raises the residual floor (1e-6 of the peak force) only from
  accepted equilibria, now including the forces the fibres carry. Every
  Newton iterate used to raise it, and the linear predictor of a string-like
  member pushed it to 1e12 N. A later 10 N load was then accepted with 82 N
  out of balance: R = 41.7 N instead of 5.0, twice the deflection, reported
  CONVERGED. At the default tolerance the first step was also less converged
  than asked.
- A restart carries the residual floors' peaks (`_force_scale`,
  `_motion_scale`) in the checkpoint; a restarted run judged later steps on
  another scale. Stopped and restarted static runs are bit-identical again.
- The round-off stall detector requires that the residual has stopped
  falling, and measures coordinate round-off from the distance to the origin
  rather than the model's extent. A modified tangent converging linearly to
  1e-10 was stopped as "stalled". A model at UTM coordinates (round-off floor
  about 3e-7) was called a limit point.
- Load control past a limit point ends: no increment smaller than `cutbacks`
  halvings of the stated one. The increment used to shrink across successes
  to 4e-17, where lambda + dlambda = lambda, and the run never ended.
- A round-off stall during a rupture re-solve, the lambda = 0 equilibrium of
  an arc-length step, or its landing on lambda = 1 is reported as such, not
  as a lost section or a failed landing.
- `ARC_NO_LINE_SEARCH` is an INFO, issued only when every step runs by arc
  length, and says where the line search still acts (the lambda = 0 and
  lambda = 1 solves). It warned on `line_search: true` beside a
  load-controlled dead step.
- `load_steps: 0` is refused (`ge=1`): it meant "continuous" in the solid
  solver and "use the solver's" in the frame solvers.
- `vfife capabilities`: the "no scalar twin" reason is given only when it is
  what decides.
- A result may not be named `*.ckpt.h5` (`OUTPUT_IS_CHECKPOINT_NAME`): that
  name was its own checkpoint path, so the writer deleted the open result
  and the run's end replaced it with the checkpoint. Only `.h5` drops its
  extension for the checkpoint name (`x.hdf5` -> `x.hdf5.ckpt.h5`); `x.h5`
  and `x.hdf5` shared one checkpoint.
- **The checkpoint fingerprint is the lowered model, not the YAML text.** A
  `schema.Case` source had no materials or sections in it, so a restart with
  a 3x wider section and 10x the density was accepted, and the same model
  given as a file was refused. Names, unused sections and file folders no
  longer count. Node coordinates are compared to 1e-9 of the model size
  (stored as `fingerprint/nodes`), so a generated mesh on another CPU is
  the same mesh.
- `initial_state` may add what can break (`fracture`, `failure`,
  `removals`, prestress). A fibre model with `fracture` could not start from
  its relaxed dead state ("section layout changed"; 0.5 ran it), and with
  `split: true` the spare particles failed with "boolean index did not
  match".
- **Behaviour change: `--restart` also refuses changed loads, constraints,
  nodal or added masses, damping, mass scaling and units**
  (`CHECKPOINT_MISMATCH` names them). A changed nodal mass was ignored
  without a word: the result was bit-identical.
- A restart ignores `solver.initial_state`. The stop message's own
  `--restart` advice failed for every scenario started from a dead state
  ("not both").
- A checkpoint beside a result that is not that run's is refused
  (`CHECKPOINT_NOT_THIS_RUN`, by a `run_id` in both files). On Windows a
  stale checkpoint that could not be deleted was resumed as the new run's
  state.
- A run that ends non-finite keeps its last finite checkpoint. The NaN
  state replaced it, and a restart resumed the NaN.
- **Behaviour change: `vfife run` exits 143 after SIGTERM** (130 after
  Ctrl-C). `AnalysisInterrupted.signal` says which signal stopped the run,
  for batch scripts that catch `Exception`.
- Results: `_members.xdmf` carries `load_factor` / `age`; the `vfife sweep`
  `.pvd` of a load-factor result runs on the frame index (lambda runs
  backwards on an arc-length path); a killed result is reported as
  `RESULT_NOT_CLOSED` rather than "add monitors"; `restarted_from` names the
  result even when the checkpoint file was given; energy `units` are absent
  when the case declares none. docs/result_format.md states that static runs
  record monitors at every increment while `output.every` thins only the
  frames.

**Fibre sections, cracked shear, temperature and tendons** (each
with a test in tests/test_review5_fibre.py):

- A step-level temperature on a fibre model kept acting in every later step;
  it now ends with its step, as on beam2.
- Fibre creep was read at the element END station under every Lobatto rule
  (a cantilever's creep came out 20% high); it is read at the middle station.
- The `fiber` strain monitor on a member with a cracking shear law counted
  the shear rotation as curvature (6x the fibre's strain).
- `interpolation: quadratic` was one least-squares parabola through all the
  points (a five-point draped tendon came out straight); it is one parabola
  per three consecutive points (0-1-2, 2-3-4, ...; an odd count), and the
  curve as placed is checked against the section outline, not only the
  listed points.
- `elastic_shortening: compensate` assumed the section origin was its
  centroid (+20% on a T written about its flange); it uses the transformed
  section's 3 x 3 stiffness about the origin and solves for the compensated
  tendons' own shortening.
- The anchor-set reflection stopped at the first element midpoint past the
  draw-in and so drew in more than stated (6.66 mm for 6 on 20 elements);
  the draw-in length is solved inside the element. The Hangzhou Bay cases'
  prestress rises 0.8%; case_24g and case_24f contracts updated.
- Refused by name instead of ignored: unknown `friction` keys, a bar's
  `area`/`material`, unknown patch keys, fibre-section keys the shape does
  not read (`nt`/`t`/`diameter` on rect, `width`/`cover` on tube, ...), and
  `cover` without `core_material`.
- A scalar temperature `gradient` follows `axis`; a two-component gradient
  with `axis: z` is refused.
- Preflight warns (`THERMAL_EXPANSION_PARTIAL`) when a heated fibre section
  mixes laws with and without `thermal_expansion`.
- A temperature on a bare truss2 names the load instead of "traction and
  pressure"; `info` counts concrete fibres, bars and tendons once each and
  names the member theory.

**Static solver, arc length and the numba kernels** (each with a
test in tests/test_review5_static.py):

- `solver.load_steps` stages a named step that has no `load_steps` of its
  own. The step dict always carries the key (as None), so the fallback never
  applied and the step took `ramp_hold`: full load at half the levels, the
  rest repeated (static, relaxation on beam and fibre models, truss). The
  shipped cases were not affected.
- The numba frame kernels (beam2, mixed_line, threaded) subtract every
  component of the imposed deformation, as NumPy does. Creep's end twists
  were dropped, and a held torque crept 16-27% short of 1 + phi on numba only.
- mode: static converges to equilibria that carry no force: a free thermal
  bow, a simply supported girder under a gradient, a prestressed cantilever
  under its own tendons, a support released in a later step. The residual
  scale takes the restraint force, the tendons' force and a 1e-6 peak floor,
  as relaxation does, and the line search also accepts a step that lowers
  the energy along it (|dx . r|), which lets a released member straighten.
- mode: static names a tolerance below round-off ("Newton stalled ...
  round-off ... not a limit point") instead of calling it a limit point.
- mode: static writes monitors at every increment whatever `output.every`
  says; only the frames are thinned, and `monitors/residual` matches
  `monitors/time` in length. Arc length honours `output.every` for frames,
  with the last increment of a step always written.
- mode: static honours a step's own `tolerance`.
- `static.tangent: modified` under arc length keeps the increment's tangent
  and renews it only when an iteration fails to halve the residual (it was a
  fresh tangent every iteration after the first). `full` is usually faster on
  strongly nonlinear paths.
- Preflight warns when a setting does nothing: `damping` under mode: static
  (`STATIC_IGNORES_DAMPING`), a `solver.static` block outside static mode
  (`STATIC_BLOCK_IGNORED`), an explicit `line_search` under arc length
  (`ARC_NO_LINE_SEARCH`; the corrector has none). A static run's summary
  reads "static" instead of "relaxation; dt=1".
- `vfife capabilities` and preflight refuse beam2_fiber on numba with
  linear_elastic, rebar or piecewise_uniaxial fibres (no scalar twin); the
  build refused them after preflight had passed.
- tests/test_static.py: the load-control limit-point test now pulls a
  perfectly plastic section past its plastic axial force. A tip load past
  the plastic moment has an equilibrium (the cantilever swings down until
  its arm is short enough), which Newton now finds.

**Results, checkpoints and restarts** (each with a test in
tests/test_review5_io.py):

- **Behaviour change: result format 3, the checkpoint is its own file**:
  `<result>.ckpt.h5` beside the result, written to a
  temporary file and renamed, so a killed run never leaves a broken one.
  Inside the result it shared the result's fate: after a SIGKILL neither was
  readable, `checkpoint_every` included. It carries `checkpoint_format` (1)
  and a fingerprint of the model (family, mode, counts, mesh and
  connectivity hashes, materials, section layout). `--restart` from another
  model is refused as `CHECKPOINT_MISMATCH`, naming what differs: a mesh of
  the same node count but another geometry was accepted and answered wrong,
  another count failed in numpy ("could not broadcast"), another mode went
  non-finite. `initial_state` may change the mode, the loads and the
  material constants, but not the family, the elements, the mesh, the
  section layout or a state array's shape (a 4 x 4 fibre state went into a
  6 x 6 section; a frame state into a fibre model failed on
  `KeyError: 'fiber_state'`). **Old in-result checkpoints cannot be
  resumed** (`RESTART_OLD_CHECKPOINT`); they still start a run as an
  `initial_state`. tests/data/result_format_2 is still read.
- **Ctrl-C and SIGTERM stop at a step boundary**: a flag
  honoured at the next progress check, so the checkpoint is exact (a Ctrl-C
  inside a step made 2 of 5 restarts differ from the uninterrupted run); a
  second signal stops at once and keeps the previous checkpoint. SIGTERM (a
  cluster's time limit) used to kill the process with the result unclosed;
  it now stops the same way (exit 130 here; v1.0 made it 143, see above). The result is flushed at every frame and
  checkpoint, so a SIGKILL leaves it readable up to its last frame.
- `output.checkpoint_every` counts steps: it was tested
  only at a frame whose step number divided by it, so with sparse frames or
  a staged relaxation nothing was written before the end.
- A restart keeps where each split end came from (the bond graph's parents)
  and the run's event tables: debris split off a region's
  nodes fell through a rigid surface restricted to that region after a
  restart (case_34: z = -0.183 under a plane at -0.12), and a restarted
  result had no `member_splits` / `member_failures`.
- **Behaviour change: `hinge_rotation` is the plastic rotation in rad**; the
  ratio to the rotation capacity it used to hold is the new `hinge_usage`.
- **Behaviour change: reserved monitor names** `time`, `residual`,
  `kinetic_energy`, `age`: the first three crashed the end
  of a full run ("name already exists"), `age` was silently replaced by the
  concrete age. An `initial_state` that is the output path is refused
  (`INITIAL_STATE_OVERWRITES`): the run erased the state it started from.
- Self-describing results: the declared units as root
  attributes `unit_length`, ... (`vfife plot` labelled a mm case's
  displacement [m]); `columns` / `dims` / `units` on every table
  (`member_splits`, `member_failures`, the energy ledger, the spectrum's
  `end_actions`, `effective_mass`, ...); `equilibrium` always a string
  ("true" / "false" / "not_applicable", it was a bool or a string by mode);
  frames carry `record`, `load_factor`, `age`; a continuation carries
  `restarted_from`; the modal file `modal_format` and `vfife_version`. The
  layout is specified in docs/result_format.md.
- **Behaviour change: the XDMF clock of static and relaxation results is the
  record index**, with lambda (and the concrete age) as field
  data `load_factor` / `age`: lambda runs backwards on an arc-length path
  (29 of case_16b's 60 frames), and ParaView orders frames by the clock.
  `_last.xdmf` carries the six stress components as scalars only: the older
  reader it is the fallback for read the `Matrix` attribute as one tuple.

Found by a second review round (mesh import and units, material laws at
their edges, reading results back), each with a test in
tests/test_mesh_review.py, tests/test_post_review.py or
tests/test_material_review.py.

**Meshes and units**:

- A Gmsh physical name is keyed by (dimension, tag): a point group and a
  line group with the same tag took one name; format 2.2 takes the
  dimension from the element type. `read_coupled` and `vfife mesh` likewise.
- Parametric node coordinates (`$Nodes` with parametric = 1, ASCII and
  binary) are refused (`MESH_PARAMETRIC_NODES`); they were read as the next
  node's tag. So are shells of mixed triangles and quads
  (`MIXED_SHELL_ELEMENTS`), a repeated node tag (`MESH_DUPLICATE_NODE_TAG`)
  and an element naming a node that is not there (`MESH_UNKNOWN_NODE_TAG`);
  a triangle-only mesh reads as a shell.
- A region's element list survives the merging of duplicate elements (it
  pointed at the rows before the merge); the pyramid5 orientation is
  corrected.
- A traction or pressure on a face two element blocks share is applied
  once, and summed over the blocks rather than refused when one block has
  none of it; a solid's reaction monitor over overlapping regions counts
  each node once.
- `node_ids` out of range are refused; a `where` axis is snapped the way the
  other axes are; a `plane`, `line`, `cylinder` or `sphere` region with a
  misspelt key is refused (`center` and `axis_point` are read as `centre` and
  `point`); the `box` order is documented as `[xmin, xmax, ymin, ymax, zmin,
  zmax]`.
- Preflight checks the `units` declaration itself (a misspelt key, a
  stress unit that is not force / length^2, a gravity that is not g in the
  declared length unit), finds duplicate nodes within 1e-8 of the model's
  size, and refuses a pressure on shells whose normals flip across an edge
  (`SHELL_NORMALS_INCONSISTENT`: two opposed Gmsh surfaces summed to zero). `mesh.scale` on a generated box and a
  region name that is also a mesh group are refused.
- A checkpoint without a loop position (older than v1.0) is refused
  (`RESTART_OLD_CHECKPOINT`) rather than replayed from step 0.

**Reading results back**:

- XDMF: a six-component field is a `Matrix` with one scalar per component
  (`stress_xx` ... `stress_xz`) -- `Tensor6` put ParaView's components in
  another order; a field is node- or cell-centred by what it is, not by
  whether its length happens to equal the node count; a member split into
  particles is drawn on its own connectivity at every frame.
- A run stopped by `on_progress` or Ctrl-C records the frame it stopped at,
  and `--restart` into the file it restarts from is refused
  (`RESTART_OVERWRITES`; write to a new `-o`). N steps = N/2 + restart +
  N/2 now holds frame for frame.
- `residual` and `kinetic_energy` are sampled at progress checks, not at
  output frames: `vfife csv`/`post.to_csv` leave them out by default, write
  them alone against a `check` column, and refuse mixing them with frame
  series (`MIXED_SAMPLING`); `plot` gives them their own axis.
- Contracts: `source: diagnostic` reads the energy audit, not the progress
  series of the same name; `location: nearest` on an element field finds the
  element by its centroid; a vector field under `final`/`first`/`delta`
  needs a `component`; a coupled model's members are read.
- A reaction monitor over overlapping regions counts each node once
  (beam, fibre and truss); `vfife sweep`'s `.pvd` refers to its frames by a
  relative path.

**Explicit dynamics, connectors, bearings and contact** (each with
a test in tests/test_review3.py):

- A connector dashpot to a clamped anchor node -- the ground connection
  docs/connector2.md recommends -- set the step to 1e-24 s (the held dofs'
  1e-20 mass floor was in the bound); the fibre solver had no dashpot bound
  at all and went non-finite.
- `removals` on a connector in a fibre model reported it failed and left it
  carrying load.
- The documented compression-only seat `{lo: null, hi: 0}` is tension-only
  (the local ux is the elongation): it is `{lo: 0}`; the docs are corrected.
- The same friction or bilinear law on `uy` and `uz` is one law in the
  plane with a circular surface (a square one gave 1.414 mu N at 45
  degrees); a friction `normal` is `ux` or a number (the sign of any other
  component flips with the node order), and a global connector cannot name
  one (a bearing written top-to-bottom had no friction).
- A global-axes connector with length returns the couple its end forces
  leave half to each end (the whole P-Delta moment went into the structure).
- A dashpot on a `gap` component acts only while the gap is closed.
- `connector_force_*` and `connector_moment_*` are signed (compression and
  uplift both came out positive).
- A zero-length connector is reported by preflight, not a traceback; a
  connector section's `material` is accepted whichever material it names.
- Rigid contact: a surface restricted to a region also stops the fragments
  split off that region's nodes (debris fell through); a numeric contact
  stiffness is capped at the explicit limit (a beam at rest was thrown
  1.4 mm); the docs' restitution, resting penetration and stick-creep
  figures are corrected.
- A shell pressure on a physical group loads that group's elements (the
  strip between two wheel patches was loaded too, +50%); the top and bottom
  shell stresses carry no transverse shear (it was Q/t on the free faces).
- Multi-step dynamic trusses and solids keep their velocity across steps;
  `solver.target_dt` is refused where nothing reads it; `ground_motion`
  takes `path`, `scale` and a unit `direction` and no `amplitude` or
  `region`; a model-level ground motion in a multi-step dynamic run is
  warned about (`GROUND_MOTION_REPEATS`); the YAML reference states the
  default amplitudes.

**The v1.0 fixes, reviewed again** (tests under "1.0 re-review" in
tests/test_review4_*.py):

- An axle or a vehicle on a member with end releases was distributed as if
  the member were fixed at both ends: the fixed-end moment went to the node
  and bent the next member (a hinge deflected 30% too far), and since
  v1.0 the written moment at the released end was -P L/8. The load is
  condensed through the member's releases, as a `line_load`'s is, in the
  nodal loads and in the end actions.
- A support constrained only in an earlier step counted as a support when a
  member split: the member left it, and the node stayed behind as a
  massless particle drifting at its last velocity. Splits, parking and
  "grounded" read the current step's constraints; a support released
  earlier is free.
- **Behaviour change:** unnamed vehicles are numbered per step -- the first
  unnamed vehicle of every step is `vehicle` (numbered over the whole case,
  `of: vehicle` read step 1's frozen vehicle in step 2). A vehicle monitor
  records NaN in a step without that vehicle.
- A removed or fractured member under a line load or an axle reported the
  fixed-end actions (-fl) as its end actions; it reports 0.
- `REMOVAL_AFTER_END` warned of a removal at exactly the run's end, which
  the last frame then logged; only one strictly after the end is warned of.
  The fibre beam stamps a frame's own force evaluation at the frame's time.
- The modal and response-spectrum finite-difference tangent advanced
  plastic-hinge rotation at every evaluation; it evaluates as a probe.
- A fibre member that fractured at a clamped support kept that particle:
  the dead member stayed pinned there with half its mass on the support.
  Its end leaves a particle held in the current step, as a frame member's
  does.
- A vehicle's tyre lift-off time (`vehicle_contact.first_time`, the log)
  was on the step's clock; it is on the run's clock.
- **Behaviour change:** `creep` in a `mode: dynamic` run is refused only
  when a dynamic step moves the concrete clock (`to_age`, or an `age` other
  than where the clock is). A dynamic run started from a crept
  `solver.initial_state` holds the restored creep and shrinkage as a frozen
  imposed strain; v1.0 refused it, and dropping `creep` to get past the
  refusal let the member spring back from 0.030 to 0.010.
- `tensioned_at` defaults to `solver.age` (or the first step's `age` when
  that is earlier), and the relaxation start rides in the checkpoint. A
  continuation from `initial_state` starting at 444.7 days restarted the
  relaxation there (fibre tendon 196.15 kN where 194.80 is right, bar 197.08
  where 195.73 is).
- **Behaviour change:** MC90 `fcm` outside C12-C80 (mean 20-88 MPa) is
  refused. 8-150 MPa was accepted, and above about 110 MPa the shrinkage
  base strain changed sign: the member swelled at 70% humidity.
- A fibre member that split gave its fragment the true, unfloored rotary
  inertia, and the stability bound fell to 0.90 dt ("the step is NOT
  adapted"). The particles a split touches keep the setup's weightless
  rotary floor.
- A material with `relaxation` used by prestressed bars and by beams was
  refused as having no prestressed member; its prestressed bars relax, and
  the refusal (no prestressed member at all) names the material's member
  groups. The static solver's refusal names relaxation too.
- A response-spectrum file with more than two columns was refused by
  v1.0; the first two (period, acceleration) are read.
- **Behaviour change:** a misspelt `units` key (`lenght`) is an error
  (`UNITS_UNKNOWN_KEY`); as a warning the case passed and ran with its
  length unit undeclared.
- A contract reading a multi-component field at a `location` without a
  `component` returned that node's z; it is refused by name for `final`,
  `first` and `delta`.
- `vfife init` checked a group named by `--fix` / `--force` /
  `--fixed-region` / `--load-region` against the prompt's boundary-sized
  list: `--fix body 010` (a plane constraint on every node) was refused as
  "not a Physical group". A named group may be any physical group; only a
  prompt needs a boundary-sized one, and only then is a mesh without one
  refused.
- `vfife view --case` refused the case of a mixed volume mesh (hexa8 +
  wedge6: "has 2 elements and the result 3"); the count is over every
  volume block, and the solid or shell alone for a coupled or shell_line
  model.
- **Behaviour change:** `vfife check` builds the model (numpy, no GPU)
  whenever preflight passes, not only when `output.fields` is written; a
  model that cannot be built is an ERROR `MODEL_BUILD_FAILED` in the report
  (exit 2). A build refusal was exit 1 with no report when fields were
  written, and a PASS when they were not.
- An I/O error named the case ("cannot read case.yaml: unable to lock
  file") whatever failed; it names the path in the error, says "cannot read
  or write", and suggests closing a program that holds a locked result.
  **Behaviour change:** a closed pipe (`vfife capabilities | head`) exits
  141 without a message.
- A YAML syntax error in a verification contract, qualification manifest or
  migrated file was reported as the case; the file the parser was reading
  is named.
- `api.study` let a misspelt key in a material's `creep` / `shrinkage` /
  `ageing` / `relaxation` block or in `units` through to every run; both are
  refused before the first run.
- `vfife mesh` listed a volume mesh's boundary faces and a shell mesh's edge
  lines as blocks a case must assign; only Line2 elements beside a volume
  are listed (as `init` warns).
- `vfife view`: `--hide` on a coupled or shell_line model said the model had
  no line members, and now says the option is not supported there;
  `--fields` accepts a group name (`end_actions`) and says "holds but does
  not draw" of a field such as `rotation`; `--names` drops empty entries
  ("a, b," is two names).
- `vfife view` labelled a creep / ageing relaxation by a load factor; it is
  labelled by the concrete's age in days.
- Arc-length progress ended at count / `max_increments` (15% for a finished
  Lee frame); a finished step reports done.
- Static `formfind` printed the Newton solver's lines whatever `log=` was,
  and its header and summary said "relaxations"; it is quiet and says
  "static solutions". `formfind -o` into a folder that does not exist
  failed after the search; the folder is made first.
- **Behaviour change:** an input-form case that also carries `element`,
  `material`, `members` or `blocks` is refused
  (`INTERNAL_KEY_IN_INPUT_FORM`); the key was overwritten in silence by the
  sections.

**Member failure, fracture and splitting** (each with a test in
tests/test_review4_fracture.py):

- The fibre beam's clock restarted at 0 in every dynamic step: a removal
  due in step 2 never fired and fractures and splits were logged at their
  time into the step. Removals, fractures and splits now run on the run's
  clock, as the frame and truss solvers do.
- A member that broke at an end on a driven support (a constraint with a
  `value`, or a record) kept that particle and went on carrying (reaction
  5.5x its capacity); every particle constrained in any step now counts as
  a support when a member splits.
- A fibre fragment only joined region `all`: gravity on a named region
  missed it (it hung in the air) and a nodal force on `all` was spread onto
  it. Gravity now reaches fragments through their parent's region, and
  nodal forces stay on the nodes they were written on.
- The stability mass a split adds was floored per dof (36.6 kg in x,
  7.9 kg in y and z), weighed, cut the velocity per dof (89% in x only) and
  was logged as x + y + z. It is now one translational mass per particle,
  the same on x, y and z, with momentum conserved and no weight -- so a
  fragment that received it falls at g m / (m + dm), not at g -- and the
  log gives the kg actually added.
- A split queued in the output frame's force evaluation was lost if the
  run was stopped before the next step: the checkpoint now carries the
  queue (frame and fibre solvers).
- A restart after a split kept the setup's reference K, so Rayleigh beta
  damped the fragment against the frame (0.37 |u| off at the end); the
  split members' blocks are moved on restore.
- A removal's `split` was overwritten by the same members' `failure.split`;
  each cause now uses its own.
- Plastic-hinge rotation also grew in `_record` and the convergence
  measures, so `output.every` changed when a hinge broke; it grows only in
  the step's own force evaluation.
- A removal at or after the run's end is warned about
  (`REMOVAL_AFTER_END`); it was accepted and nothing was removed.
- A fibre member that fractured was reported failed again at its removal
  time.

**Command line and Python API** (each with a test in
tests/test_review4_cli.py):

- `output.path` is resolved against the working directory, as it always
  was; the YAML reference and the manuals said "relative to the case file"
  and are corrected.
- `api.case_schema()` described the internal, lowered case every loader
  refuses; it is now the schema of the case file as written (materials /
  sections / assign and the rest), and every shipped case validates.
- `vfife init` wrote a solid section for any mesh: a shell mesh passed
  `check` and crashed in the run, a line mesh was refused by `info`. It now
  refuses a non-volume mesh and says which section to write, and a solid
  section on a shell mesh is refused before the run (`SECTION_MESH_MISMATCH`).
- **Behaviour change:** `vfife init` no longer writes over an existing file
  without `--overwrite`, and never over the mesh it reads.
- `vfife formfind` reported CONVERGED when the inner runs never reached
  equilibrium (`mode: dynamic`), and an exhausted relaxation was a
  traceback. It needs `relaxation` or `static`, checks every inner run, and
  one that did not converge ends with NOT CONVERGED, exit 3; an
  unconverged search writes a `# NOT CONVERGED` header instead of claiming
  the lengths reproduce the mesh.
- `formfind -o` into another folder copied the relative mesh, record and
  `initial_state` paths verbatim; they are rebased onto the new file.
- `api.study` recorded a successful run as ERROR when `extract` raised; the
  run's columns are kept and the error goes in `extract_error`.
- `api.study` let a misspelt field name (`solver.n_stepz`, a material's
  `EE`) through to every run; it is refused before the first.
- `vfife view` ignored a `--case` it could not use or that was not the
  result's model, an unknown `--hide` region and an unknown `--fields`
  entry; all are refused. **Behaviour change:** `--names` must give one
  name per result.
- `vfife view` labelled a load factor in seconds and drew an arc-length
  path off the chart; relaxation and static acts are labelled λ and drawn
  against the record index.
- **Behaviour change:** `vfife check` did not check `output.fields`,
  though the manual said it did; when the case lists its fields it builds
  the model on the numpy backend (no GPU) and refuses a field the model
  cannot produce (`UNKNOWN_OUTPUT_FIELD`); a model that cannot be built is
  reported (`MODEL_BUILD_FAILED`) rather than a traceback.
- A YAML syntax error, a result file that is not HDF5 and input ending at
  an interactive prompt were tracebacks; each is one line, exit 1.
- `vfife mesh` hid a Line2 block beside a volume; every block the reader
  keeps only as node sets is listed, and `init` warns that such Line2
  elements are not assigned.
- Interactive `init` still asked for a fixed support and a -1000 N load
  when `--fix` / `--force` were given, and offered whole-volume groups as
  boundaries; those prompts are skipped and those groups left out.
- `run_case`'s documented stop raised `AnalysisInterrupted`, which the API
  did not export; it is in `api.__all__`.
- Minor: `sweep --every 0` and `formfind --step 5` are refused by name;
  `probe` / `modal --frame` take an unpadded step number; `sweep` defaults
  to the case's `output.path`; `capabilities --mode static` is accepted;
  arc-length progress counts against `max_increments` (it showed 100%); a
  material value that is not a number names its key.

**Vehicles and moving loads** (each with a test in
tests/test_review4_vehicle.py):

- A member numbered against the travel mirrored the axle's distribution
  within it (midspan deflection 3.8x, 1.24x, 0.80x exact); `moving_load`
  and `vehicle` now measure the position from each member's own node i.
- Two vehicles with the same name -- two unnamed ones, both `vehicle` --
  were one vehicle counted twice. **Behaviour change:** unnamed vehicles
  are `vehicle`, `vehicle_2`, ...; two with one name in a step are refused.
- A `moving_load` under `static.arc_length` blended its first and last
  stations instead of moving the axles; it is refused by name.
- After a member split off, a vehicle went on pushing the particles the
  fragment had left; it takes the members' current particles.
- A moving load on a fibre deck whose member fractured and split: covered
  by a test (the path is the as-read connectivity).
- `moving_load` / `vehicle` on a shell model passed preflight and died with
  AttributeError; refused by name.
- With an axle or a `line_load` between nodes, the written end actions
  (`*_i/_j`) and `bending_y` etc. lacked the fixed-end moments (6.7% low);
  they are subtracted now. Known limits: on `beam2_fiber` only `*_i/_j`
  are corrected (the per-member values are section resultants; refine the
  mesh under an axle), and the member-failure check still reads k d.
- A vehicle's `amplitude` was accepted and ignored; refused.
- A vehicle monitor with a misspelt `of` recorded NaN, an `axle` out of
  range was an IndexError; both are refused when the case is read.
- Tyre lift-off and a rigid axle in tension were counted and not reported;
  counts and first time go to the run report (`vehicle_contact`, and
  `vehicle_contact_json` in the result's diagnostics), the log and the
  `vfife run` summary.
- **Behaviour change:** a `moving_load` with `speed` outside a dynamic run
  moved on the load factor (or crashed); it is refused -- use `stop`.
- A tyre's dashpot pushed a wheel that was above the road (31 kN at a
  0.08 mm gap); a tyre acts only while compressed.
- Since v1.0 read the loads at the step's start the vehicle was one step
  behind (`t - dt`); it acts at the step's start.
- The PSD road accepted misspelt keys, ignored `scale` and was
  interpolated from 4 points per shortest wavelength (slope variance 7.8%
  low); its keys are checked, `scale` applies, and the sum is evaluated
  exactly.
- `vehicle.direction: [0, 0, 0]` ran to NONFINITE; refused.

**Creep, shrinkage, relaxation and ageing** (each with a test in
tests/test_review4_creep.py):

- The frame path took the first assigned material's creep, shrinkage and
  ageing and applied them to every member (a steel bar crept 10 -> 30 mm
  beside the concrete, or the concrete did not creep); each member now uses
  its own material, and members without them (and connectors) do not creep.
- **Behaviour change:** `mc90` phi was taken against Ec(t0); MC90 references
  it to Ec28, J = 1/Ec(t0) + phi/Ec28 (creep part 17-22% low). An `mc90`
  creep spec now needs the material's `ageing` or a new `E28` key, and
  neither or both is refused. With `ageing` the result is still ~2% short
  (the ageing scheme scales the whole stiffness); `E28` is exact.
  case_28d gives `E28: 3.87e+10` and its creep targets move from
  1 + phi = 1.533 to 1 + phi Ec(400)/Ec28 = 1.64063.
- A `truss2` bar under a held load crept and came back to the elastic value
  (the bar force missed the station's 1/(1 + phi)); it keeps its creep.
- A relaxation step without `load_steps` stopped the clock where the
  residual converged (301 of 1028 days); it is one station at `to_age`.
- **Behaviour change:** `creep` in a `mode: dynamic` run advanced the age
  and crept nothing; it is refused by name.
- MC90 shrinkage at RH >= 99% shrank; it swells (beta_RH = +0.25), and `rh`
  outside 40-100 is refused.
- `aci209` creep ignored the loading age (late increments up to 81% high);
  it applies gamma_la = 1.25 t0^-0.118 (moist) or 1.13 t0^-0.094 (steam).
- A step at an earlier `age` un-crept the member, and `to_age < age` was
  ignored; the clock only runs forward, and both are refused by name.
- `creep` on solids, shells, coupled models, a standalone `truss2` and
  embedded reinforcement, and `relaxation` on prestressed bars, were read by
  nothing. Relaxation on a prestressed bar (mixed-line path) is now its
  prestrain falling by r(t); the rest is refused by name.
- **Behaviour change:** creep, shrinkage, ageing and relaxation spec keys
  are checked (`RH` for `rh` ran at the default); `cement` (mc90) now
  adjusts the loading age and `curing` (aci209) selects gamma_la.
- **Behaviour change:** `tensioned_at` only delayed the relaxation clock
  while the tendon force was there from the start; a `tensioned_at` later
  than the analysis start is refused by name.
- `rho_1000: 3.1` (a percentage) put the tendon in compression; it must be
  a fraction in [0, 0.2], `k` in (0, 1), and a loss reaching 100% stops.
- A `curve` creep table without a `[0, 0]` row read its first value at the
  instant of loading; the origin is implicit and the value rises linearly.

**Seismic input, modes and the response spectrum** (each with a test
in tests/test_review4_seismic.py):

- The response spectrum and `vfife modal --result` took the FIRST step's
  supports; they take the last step's (the frame's step for `--result`).
- Modal analysis and the response spectrum ignored `diaphragms`; a rigid
  floor's in-plane dofs now follow three master dofs at its centre of mass
  (a two-column floor: 2.96 / 5.24 Hz, the closed form).
- **Rotary inertia.** `beam2` had rho A L^3/24 about its own axis too, so
  its torsional frequency grew with the element count; that term is
  replaced by rho Ip L/2 (73.64 / 73.70 / 73.71 Hz at 10 / 20 / 40
  elements, closed form 73.65). Timoshenko and fibre beams added their
  local rho [Ip, Iy, Iz] L/2 to the global rotations unrotated, so a
  member's frequencies depended on its orientation; the tensor is rotated
  to global axes and its diagonal kept. The explicit step is still taken
  from the old rule (no case slows down); a rotation whose true inertia is
  below what that step needs is raised to the least stable value. No
  shipped contract moved.
- A mode without translation (a column's torsion) was scaled by numerical
  noise (rotations of 1e13); it is scaled to a unit largest rotation and
  flagged (`rotation only`, `rotation_only`).
- Velocity and acceleration support records were first order in dt; they
  are integrated exactly from the record's interpolation, like a
  displacement record.
- A contract reading the spectrum's `reaction` with `nearest` picked an
  element's row; it finds the nearest node.
- A spectrum vector (`displacement`, `rotation`, `reaction`, `base_force`)
  read without `component` silently returned z; refused by name.
- A truss model's spectrum wrote no bar forces; it writes `axial_force`.
- **Behaviour change:** a support `record` and a `ground_motion` along the
  same direction were both applied and the response doubled; preflight
  refuses them (`GROUND_MOTION_AND_RECORD`).
- `hydrodynamic_mass` checked its 5-degree rule per coordinate (a member
  6.9 degrees off passed); it checks the true angle.
- A spectrum `path` was read after the run, one column as an IndexError and
  negative values accepted; it is checked before the run (`SPECTRUM_FILE`),
  and comma-separated files are read.
- `modes` silently overrode a written `mass_fraction`; giving `modes` with
  a non-default `mass_fraction` is refused.

**Material laws at their edges**:

- `steel_cyclic`: the same strain twice returned a different stress on an
  unloading branch (340 -> 202 MPa); a strided state was updated in a copy;
  a 1e-5 wobble on the monotonic curve resumed on the hysteresis skeleton
  (461 -> 541 MPa in one step) -- it now resumes the curve it left, taking
  the skeleton only where that is lower; a reload whose secant is steeper
  than E stepped onto the skeleton at the peak strain (92 MPa in a 1e-4
  step) -- it stays on the line at E until it meets the curve. `f_uf` must
  lie in [0, fu].
- `rebar`: a ruptured point's plastic strain kept growing on the numpy path;
  E and the yield strength must be positive.
- `concrete01`/`concrete02`: `|epscu|` must exceed `|epsc0|`; E and `epsc0`
  cannot both be given to `concrete01`; `rat` in [0, 1).
- `chaboche`: `damage_max` in [0, 1).
- `hardening_linear` reaches a plastic strain of 10 by default and
  `hardening_power` has points past 1 (a power curve ended at 1).
- Strain penetration is refused, by name, on a section without bars, a bar
  law without a strain scale, a prestrained bar in an anchored element and
  a temperature load (each ran without it).

**Every text file is read and written as UTF-8**. On a
Chinese Windows the default encoding is GBK: a ground-motion record with a
non-ASCII comment (case_36) could not be read, and the case files the tests
read without an encoding -- Chinese comments in 23 of them -- failed 85
default-layer and 4 slow tests there. The records, spectra, road profiles,
XDMF and CSV writers name `encoding="utf-8"`, as do the 238 reads and
writes in the tests; the CLI prints a character its console cannot show as
an escape instead of stopping; `tools/check_all.py` runs its children with
UTF-8 pipes. The whole suite now passes under an ASCII locale as well.

**`--restart` resumes** (tests/test_restart_resumes.py). The
frame, fibre-beam, truss and implicit static solvers restored the state and
then ran the whole analysis again on top of it, from step 0, t = 0 and load
factor 0: a ground motion replayed from its first second, a staged push
reloaded from level 1 (unloading a plastic model), a vehicle started again at
the abutment. A checkpoint now carries the loop position -- step, the step's
start, load level and the count it began at, the level's prescribed starting
positions, the implicit solver's load factor and increment (and an arc's
length and direction) -- with the integrator's memory, the fictitious mass
and the vehicles' states, and every solver picks up there. N steps and N/2 +
restart + N/2 now end in the same state bit for bit, on the frame, mixed,
shell, coupled, fibre, truss, solid and implicit paths. Also:

- a solid restarted inside step k > 0 kept its velocity (it was zeroed) and
  a staged solid resumed its level (it went back to level 1);
- eroded elements stay eroded; a fibre fracture stays latched and is not
  logged again; a fibre split keeps its bonds, masses and particles; a
  coupled model keeps its continuum's material state; a shell_line model
  keeps its line members' slip, hinges and breaks;
- the converged fibre state (and a bonded bar's) is saved beside the live
  one, since inside a level every evaluation starts from it;
- the energy ledger carries on (a restarted audit reported the kinetic
  energy it resumed with as an imbalance of 100%);
- creep, shrinkage and ageing history is saved, and `initial_state` carries
  it into the next run (a continuation of case_29 went back to the elastic
  deformation and crept again);
- `initial_state` stops the bonded bars too, and starts a fresh ledger;
- `checkpoint()` returns copies (it returned the live arrays, and one test
  could therefore not fail);
- a restart with nothing left to run is `RESTART_BUDGET_EXHAUSTED`, and a
  warp shell run refuses `--restart` rather than replaying.

An error the internal form finds at a key the lowering made up
(`members.0.failure...`) is reported at the user's `assign[i]`.

Found by a targeted review of the older code for settings that validate and
are then ignored, each with a test in
tests/test_scope_review.py. **A case that ran before may now be refused**:
every refusal below names the key and says what to do instead.

- **`end_actions` in `output.fields` wrote no member force** (thirteen
  shipped cases asked for it); group names (`end_actions`,
  `connector_force/moment/deformation`, the shell `*_resultants`) now write
  every field of the group. The misspelt-field check was skipped on every
  beam, fibre, mixed, shell and coupled model because its probe raised; it
  runs on all of them. `velocity` and `acceleration`, listed as outputs of
  the line elements and solids, are not written by any; they are no longer
  listed.
- **Keys a section type or assignment does not read are refused** by name
  (fibre `releases`, beam `prestress`/`tension_only`/`tendons`/`bars`, cable
  `friction`, fibre `theory`, beam `shear_stiffness`, ...), as are a
  `core_material` without `cover`, fibre bars and tendon points outside the
  section, a tendon profile that does not cover its members, a misspelt
  section name (it was a KeyError traceback), and a single fibre section
  assigned to a region that is not every fibre member (it was applied to all
  of them).
- **Material parameters are checked**: every law took `**_ignored`, so
  `R00` for `R0` or `s1x` for `s1` ran on the default.
- **Preflight** refuses monitors the model cannot record (a frame recorded
  a `von_mises` or `fiber` monitor as a displacement, or 0), `reduce` where
  it is not applied, diaphragms outside the models that apply them,
  `follower` outside solids, `solver.erosion` outside solids, `mode: static`
  outside frame and fibre models, `load_steps` on a relaxed truss, an
  undefined or unreadable amplitude, and a non-linear member, reinforcement
  or fibre law wherever it sits in `assign` (the verdict depended on the
  order); it warns on an unused amplitude, one that runs past progress 1,
  and a model-level amplitude that re-applies the load in every step. A
  solid section on a Line2 mesh is refused in section terms (it ran as a
  truss with A = NaN). A displacement monitor on a solid averages all of its
  `regions` (it read the first).
- A step's `tolerance` is honoured by the frame, fibre and truss solvers
  outside staged levels; `rotation_iterations` is documented as the solid
  setting it is.
- A shell section written as the manual shows (`material`, `thickness`) is
  accepted.

Found by a second reading of v1.0, each with a
test in tests/test_static_review.py or tests/test_review_fixes.py:

- **Static without `load_steps` put the full load on at 0.5**: the step was
  unstaged to `_step_data`, its load took the ramp-hold amplitude, and the
  ten Newton levels repeated the full load five times, labelled 0.6 ... 1.0.
  The static mode now states its ten levels.
- **A steel fibre that ruptured at a commit was recorded unbalanced** by the
  static solver (70 kN out on a 97 kN cantilever); the level is now solved
  again, as staged relaxation does, until the ruptures stop cascading.
- **Arc length**: a temperature or imposed strain now follows the load
  factor (it was on in full from lambda = 0); a pattern that moves no free
  dof (held loads, a prescribed rotation) runs by load control instead of
  "converging" at lambda = 0 for max_increments; running out of
  max_increments is a MAX_STEPS failure, not CONVERGED; a misspelt name in
  `arc_length.steps` is refused (it silently ran every step by load
  control); `stop` means crossing the value from where the step starts, and
  may only apply to the last step; an increment accepted within 1e-9 of the
  full load ends the step.
- **A parameter study changed values it was not asked to**: `CaseDraft`
  copied with `deepcopy`, which kept YAML anchors shared -- one tendon's
  friction in case_24d changed all 66. Values are now copied without
  sharing, numpy numbers become plain ones (they failed every run), a `/`
  in a case name no longer moves the folder paths resolve in, a numeric
  mapping key is reachable, and Ctrl-C stops the study with its rows
  written.
- **`vfife migrate`** renamed an aliased section twice (`widthidth`), put
  `version` before a `---`, a `%YAML` directive or into a BOM, broke a
  flow-mapping case and an empty `version:`, and rewrote a newer format
  downwards; all fixed, and a file it cannot rewrite is reported, not a
  traceback.
- **A fibre section without rebars checkpointed zeros**: the checkpoint
  read an alias the kernel's arrays had replaced, so `--restart` and
  `initial_state` began from a virgin section; every fibre group is now
  saved from the live state (and a cover + core section is no longer group
  0 only).
- The compiled cable loop read stress-free lengths and tension-only flags
  once; it reads them live, as the NumPy path does.
- `solver.threads` set numba's process-wide thread count at setup, changing
  every later kernel's; it is now set around each threaded call and
  restored.
- `vfife probe --frame` with a missing frame raised a TypeError.

- **`motion: acceleration` constraints on frames never moved their node**:
  the dynamic loop zeroes a held dof's velocity before every position update
  and the prescribed acceleration was integrated from it, so the node crept
  by dt^2/2 a step instead of moving a t^2/2. The kinematic velocity is now
  kept separately (the truss solver, the only one tested, had it right).
- **`relative_to` on displacement monitors was ignored outside solids**: the
  frame, fibre-beam and truss solvers accepted it and reported the absolute
  displacement.
- The user manual's list of what is not there still named response-spectrum
  analysis after v1.0.

## [0.5.0] — 2026-09-27

The collapse line: what fails now falls, lands, and can be watched. Bearings
with friction, gaps and hysteresis; rigid surfaces to land on;
member breaking on every line solver, a broken member
that splits off with its own mass, breaks at the section that
failed and, for steel, bends through a plastic hinge before it
tears; a sharper top frequency for beta damping; and
`vfife view`, a result as one self-contained 3-D page. Minor, not
patch: a broken frame member now splits off by default (`split: false`
restores 0.4's deactivation), and a beta-damped model takes a longer step.
No shipped case's contract changed.

### Added

- **`vfife view`: continuous colouring and the global axes.** Each member
  is shaded from the colour at one end to the colour at the other (a second
  per-instance colour blended in the vertex shader); a nodal field uses its
  nodes' values, an element field is averaged onto the nodes over members
  of the same drawn section (and never across a break). An axis box on the
  model's bounding box, ticks in the case's length unit, and an orientation
  triad that turns with the view. Both on by default, both toggled on the
  page.

- **Plastic hinges at the ends of mixed_line beams**:
  `failure.rotation_capacity` (rad) makes every end of the group a
  perfectly plastic hinge -- its moments are returned to |N|/N_p +
  |M_y|/M_y,p + |M_z|/M_z,p <= 1 through the end's rotational stiffness,
  the difference becomes plastic rotation (in the eigen-deformation slot a
  temperature uses, so the NumPy law and the numba kernel both see it),
  unloading is elastic, and the end breaks when its accumulated plastic
  rotation reaches the capacity (or at once on N_p). Field
  `hinge_rotation` (accumulated / capacity), checkpoint entries
  `plastic_rotation` and `hinge_rotation`. tests/test_plastic_hinge.py: a
  cantilever's root moment stops at M_p and the root tears on rotation,
  not at M_p; nothing changes below it; the numba kernel agrees; the energy
  ledger closes.
- A massless bar (a bearing, a link) with no massive member left at one of
  its ends, and no support there, is removed (cause 2): it holds nothing
  up, and bearings and crossbeams left between a tower's corbels after the
  girders had gone were a pinned mechanism of stability masses.

- **A beam breaks at an end, not as a whole**: the
  capacity criterion is judged at each end and a split member leaves the
  particle only at the end that reached it, staying welded at the other
  (where it may break later). Cables, scheduled removals and members that
  are deactivated rather than split still go whole. `end_broken` is in the
  checkpoint. It does not change how a brittle collapse fragments -- the
  instantaneous release of the capacity overloads the next section (the
  note has the numbers); that needs a plastic hinge.

- **`vfife view` draws real section outlines** where it is given them:
  `write(..., profiles={region: {"outline": [[y, z], ...], "outboard":
  bool}})` extrudes each outline (metres, y across the member, z along its
  depth) over the members of that region instead of a box, at its real
  size; `outboard` turns it so its +y side faces away from the model's
  centreline (an edge girder's fairing) with a mirrored copy of the
  geometry rather than a mirrored matrix (which lights the faces black).
  Python API only; a drawing option, nothing the run reads.

- **A broken line member splits off onto its own particles**
  (`failure.split` / `removals[].split`, the default on beam2,
  beam2_timoshenko and mixed_line): two spare particles per breakable
  member are reserved at setup (core/topology.BondGraph, the fibre beam's
  device), at the break both ends move onto them with the member's share of
  the mass, and the member goes on carrying force between them -- a free
  elastic segment with its weight, instead of a dead element whose mass
  stayed on the frame. A constrained particle is always duplicated (a
  fragment never stays clamped to the support it broke off); the fragment's
  particles get the mass the current step needs to stay stable (v0.4's
  rule, logged); gravity on a region reaches the fragments that descend
  from it, nodal forces do not; the particle a member left gets the same
  floor as the fragment; a member with no mass of its own next to its
  stiffness (rho ~ 0, a modelling device) is deactivated rather than let go
  as a massless spring; a moving load's carriageway is the chain of members
  as the road was laid, and the axles over a member that fell ride the
  fragment; a cluster of massless members attached to nothing massive
  and no support is removed (cause 2 in member_failures), one attached at
  a single massive particle becomes an appendage carried rigidly by it
  (an outrigger whose hanger was cut),
  one attached at two or more stays elastic; the split itself happens at the end of the step
  (the forces of the step belong to the old connectivity); the stability
  mass is added with momentum conserved; the energy ledger does not book
  the relabelling of a spare as a displacement; the mass a split leaves
  on the parent is clipped at zero (a difference of two sums of shares is
  -1e-10 kg as often as +0, and a negative mass under gravity climbs), and
  a particle left with no member that carries force, no support, no
  nodal mass and only massless members on it is parked (zero mass, zero
  velocity, held where it is) and those members are marked removed; an
  appendage exists only on the standing structure (its anchor joined to
  a support through members that carry force) -- the massless clusters
  hanging from a single point of a piece of debris are removed, since a
  free segment's roll carries none of their rotary inertia; and an appendage turns with the chord of a massive member at its anchor (the least rotation, no roll), not with the anchor's nodal rotation -- a line member on a frictionless floor rolls about its axis without end. `split: false` keeps v0.4's
  deactivation (the standalone truss's only rule). The connectivity of
  every frame is written (`connectivity`), the checkpoint carries the
  moved bonds and masses, `diagnostics/member_splits` lists every moved
  end, and `vfife view` draws a split-off member where it is. Closed forms
  in tests/test_member_failure.py (a beam cut loose falls at g on both its
  particles; the cases 33 and 34 unchanged to 1e-3).

- **The top frequency for the stiffness-proportional damping bound by
  power iteration** (`elements/stability.power_omega_max`):
  the Rayleigh beta term shrinks the explicit step by sqrt(1 + zeta^2) -
  zeta with zeta = beta omega_max / 2, and omega_max was the Gershgorin
  bound -- paid twice, in omega and in the zeta built on it. Every solver
  on `OrientedPoints` (frames, mixed_line, fibre, shells, coupled) now
  evaluates that factor at the pair's true top frequency (power iteration
  on the reference K and the mass in force, a millisecond, capped by the
  bound it replaces); the undamped step is untouched and an undamped or
  alpha-only run does not move by a bit. Measured: a single beam 1.0x, a
  fibre column 1.7x, Lee's frame 3.4x, a shell 4.1x larger steps under
  beta, the kicked-model-stays-at-rest invariants unchanged. And Rayleigh
  stated the way engineers state it, `damping: {ratio: 0.02, frequencies:
  [f1, f2]}` (one ratio or [zeta_1, zeta_2] at the two frequencies in Hz),
  from which alpha and beta are computed.

- **`vfife view`**, a result file into one self-contained web page that
  replays it in three dimensions (post/view.py): line
  members as boxes, shells as faces, solids as their exterior faces (found
  once by sorting), a coupled model's solid and members together and its
  reinforcement as lines; every nodal and element scalar field the run
  wrote as a contour with its range over all frames, `member_failed`
  members grey or hidden, the breaks as ticks on the time axis, the
  monitors with a cursor, displacement amplification relative to the
  first frame (so a bridge started from its dead-load state amplifies its
  motion, not its sag). Displacements are quantised to 16 bits, fields to
  8 (the displacement's components are read off the positions), all
  base64 in the HTML; three.js from a CDN is the only thing the page
  fetches. Several results of one model are the acts of one page
  (`--names`), and with `--case` line members are drawn at their sections'
  size (shapes as given, `general` from A and I: thin-walled for steel,
  solid otherwise). The hand-built D1 demo of the nsfc project,
  generalised -- and now generated by it. A broken member whose ends have
  parted is drawn as two half-stubs riding their nodes (not a box
  stretched between them), a snapped cable disappears; `--hide` leaves
  named regions (rigid arms) out; a member with no capacity criterion is
  neutral in the `capacity_ratio` contour (its ratio is now NaN, not 0).
  Every run result on hand (89 files, 0.1 MB to 840 MB) makes a page; a
  modal file is refused by name. The page stores only the particles a run
  used, `member_failed` rebuilt from the events, a split member's
  connectivity as its changes, constant fields as one value, and
  `--fields` chooses which fields go in; member-force contours take a
  0.5-99.5 percentile range.

- **Member breaking on every line solver**: the elastic-
  brittle criterion and the scheduled `removals` of v0.4, until now
  mixed_line only, are one table (`core/member_failure.py`) shared by
  beam2, beam2_timoshenko and the standalone truss2 (`element.failure`, or
  `failure` on a section assigned to `all`) and by the fibre beam
  (`removals`; its members break through the section's own `fracture`,
  whose events now go to the same `diagnostics/member_failures` list and
  `member_failed` field). A broken frame member has zero local stiffness, a
  broken bar zero area, and under stiffness-proportional Rayleigh damping
  the member's reference stiffness is taken out of K as well, so a fallen
  fragment is not held to the deck by a dashpot. The same closed forms as
  v0.4 on each solver (tests/test_member_failure.py: the root of a
  beam2 cantilever at M_cap/L on both force paths, a standalone bar on the
  dynamic overshoot, a fibre member cut at its time and the rest in free
  fall at g).

- **Path-dependent connector laws**, `sections.<name>.laws: {<component>:
  {type: gap | bilinear | friction, ...}}`: a gap with a stop
  (an expansion joint, pounding, a compression-only seat), a bilinear
  hysteresis with kinematic hardening (an isolator, a yielding device), and
  Coulomb friction that presses on another component's linear spring (a
  sliding bearing whose sliding force follows its vertical load). One scalar
  of history per component, the slip, written as `connector_slip_*` and
  carried in the checkpoint; advanced from the committed slip so a static
  load level's trial evaluations re-drive it from the last equilibrium, as
  the fibre materials are, and committed every step of a dynamic run.
  mixed_line, standalone connectors and connectors in a fibre model.
  `case_32_bearing_laws`: a girder on a fixed and a sliding bearing warming
  into an abutment stop, every number against its closed form (friction
  0.2%, stop force 0.02%, travel 0.002%); tests/test_connector_laws.py holds
  the stick-slide-return of a Coulomb bearing, the bilinear loop's area, a
  gap's free travel and a pushed block's stopping distance (accelerations
  to 0.5%).

- **Rigid-surface contact**, `contact: {rigid: [{plane: {point, normal},
  region, stiffness, friction, damping}]}`: a node that
  crosses the plane is pushed back by a penalty spring; `stiffness: auto`
  sizes it per node as m (2/dt)^2 / 4, so the contact never shrinks the
  explicit step and a free particle -- the debris contact exists for --
  gets a scale from its own mass. `friction` is Coulomb on the tangential
  velocity and `damping` a dashpot beside the spring (both dynamic only,
  refused in relaxation); the total never pulls. No pairs, no search: every
  node of the region against every surface. Initial penetration is refused
  by name. Fields `contact_force` and `contact_penetration`; every solver
  family except the device-resident warp path. With a surface present the
  preflight mechanism checks warn instead of refuse, since the surface may
  be what holds the body. `case_33_falling_beam`: nine strings cut at once,
  a 4 m beam falls 1 m and lands at 0.1 + sqrt(2 x 0.99 / g) to 0.1%, and
  lies on the ground carrying its weight; tests/test_rigid_contact.py holds
  a propped cantilever's prop reaction, a rod's bounce back to its drop
  height, a block sliding out at mu g, and a solid box's weight on the floor.

## [0.4.3] — 2026-09-25

The step to time histories on a long-span bridge, and a case file that
documents itself. Selective mass scaling and a start from a relaxed dead
state took the Xihoumen model from 2.3 hours to 42 seconds per simulated
second; elastic-brittle breaking and scheduled removal
let a cut hanger run into a progressive collapse; a generated
YAML reference lists every field with its Chinese and English doc comment
and the shipped case that uses it. Two fixes: the relaxation
iteration budget the schema said could be omitted, and a moving
load driven backwards that ran with its axles reversed.

### Added

- **Selective mass scaling**, `solver.mass_scaling: {target_dt, max_added_fraction}`:
  only the DOFs whose own explicit step is below the target
  get mass, exactly as much as lifts them to it; translational mass is added
  isotropically per node, and it is real mass (gravity weighs it, a ground
  motion accelerates it). More than `max_added_fraction` (default 1%) of the
  model's mass is refused, and the run reports what was added. beam2,
  beam2_timoshenko, beam2_fiber and mixed_line; refused elsewhere. On the
  Xihoumen model the step goes from 1.07e-7 to 2.0e-5 s for 757.5 t (0.176%),
  and the lowest 40 frequencies move by at most 0.60%.
- **Starting a run from an earlier state**, `solver.initial_state: <result.h5>`:
  positions, rotations and material state of that result's
  final checkpoint, at rest, at t = 0 -- a dynamic run from a relaxed dead
  state. Unlike `--restart`, which resumes a run, it starts a new one; a result
  computed on another mesh is refused. On Xihoumen, 5 s at rest drift 0.023 mm
  from the dead state, against 1.68 m when started from the mesh.
- `cases/_xihoumen_dead.yaml`, `cases/_xihoumen_rest.yaml`
  (`benchmarks/xihoumen_dynamic.py`): the Xihoumen model ready for time
  histories, about 42 s of wall time per simulated second on two cores.
- **Elastic-brittle breaking of line members**, `assign[].failure`:
  a cable or bar breaks when its tension reaches
  `tensile_strength x A`; a beam when `|N|/N_p + |M_y|/M_y,cap + |M_z|/M_z,cap`
  reaches 1 at either end, with `M_cap = shape_factor x yield_strength x I / c`
  from the section's own I and the `extreme_fibre` distances. A broken member
  carries nothing from then on and its mass stays on its nodes. Checked every
  step of a dynamic run; mixed_line only (a single-kind model is refused with
  the hint), and relaxation is refused -- its trial iterations are not a
  history. Break times, members and ratios go to
  `diagnostics/member_failures`; the member states ride in the checkpoint;
  `member_failed` and `capacity_ratio` can be written as fields.
- **Scheduled removal**, `removals: [{region, time}]`: the
  members of a region taken out at a given time -- a cut cable, a lost
  hanger -- through the same path as a break.

- **A generated YAML reference**, `docs/YAML参考.md` and `vfife reference`:
  every block of a case file, every field with its type,
  default and `#:` doc comment, the materials and elements from the
  catalogue, and for each field the shipped cases that use it; the fields
  no case exercises are listed at the end. `tests/test_reference.py` fails
  when the file on disk differs from what the code generates, or when a
  field has no doc comment -- so the reference cannot go stale. Every
  field of the input model now carries a `#:` comment (136 were bare) and,
  as its first line, a one-line Chinese summary; the reference shows both.

### Changed

- **Many time-history load groups are summed as one sparse product**:
  with eight or more fixed-shape groups, their vectors are
  stacked once and their amplitudes interpolated together each step; same
  force to 1e-12. A buffeting case with 588 nodal load histories no longer
  pays 588 Python calls per step.
- **A moving load evaluates all its axles at once**: one
  search along the path and one scatter-add. A 30-axle platoon costs about
  what one axle did; the Xihoumen traffic case went from 135 to 725 steps/s.

### Fixed

- **A moving load driven towards decreasing `s` now trails its axles
  behind it**. docs/moving_load.md said that a trailing axle
  is `-4.2` whichever way the group travels, but the offsets were added to
  `s` as written, so a group with a negative `speed` (or `stop < start`) ran
  with its axle pattern reversed -- the lead axle at the back. Symmetric axle
  patterns (one axle, a uniform train) were unaffected.

- **`solver.n_steps` can really be left out on line and shell models**.
  The schema accepted a relaxation case with no
  `solver.n_steps` when every step carried its own budget (`load_steps x
  max_steps_per_load`, or its own `n_steps`), and the README said so -- but
  only the continuum solver added the budgets up. The beam, fibre-beam,
  mixed-line, shell and truss solvers read `solver.n_steps` and died in
  setup on `int(None)`, which is why the fibre-beam cases carried a
  placeholder `n_steps: 100`. The validator that allows the omission now
  writes the sum into the case, so every solver reads one number;
  `tests/test_relaxation_budget.py` builds one case per solver family
  without it (all five failed before). Cases that state `n_steps` are
  unchanged.

### Documented

- **A Chinese user manual**, `docs/用户手册.md`: installation, a complete
  post-tensioned fibre-beam example whose every output was run, each part of
  the case file, the elements and materials, solver control, the bridge
  features (tendons, rupture, creep and relaxation, temperature, moving loads,
  ground motion, bearings, modal, form-finding), output and post-processing,
  checking and verification, the command line, common problems, and the scope
  and limits; an appendix lists the 50 shipped cases and their contracts,
  generated from the contract files. It is also the manual for a software
  copyright registration.

### Verified

- **All 50 case contracts re-run on 0.4.2 and passed**: each shipped case
  solved and held to its contract (`tests/test_cases.py::
  test_a_case_solved_satisfies_its_contract`, the contract half of the `slow`
  layer), about 2.7 h on two cores.

## [0.4.2] — 2026-09-24

One addition, found while cutting a corroded stretch out of the Hangzhou
Bay unit's tendons: a tendon can carry its stress as a
profile along its length, so a post-tensioned tendon cut into pieces is
still the tendon it was. With it, the section-block behaviour of tendon
coordinates is documented rather than left to be discovered.

### Added

- **A stress profile along a fibre-beam tendon**:
  `prestress: [[s, sigma], ...]`, linear in `s` and sampled at the element
  midpoints; a table that does not cover every element the tendon is in is
  refused, not extrapolated. It is what a post-tensioned tendon cut into
  pieces needs -- a corroded stretch with its own steel, say: each piece
  written with `jack` + `friction` is jacked at its own ends and is a
  different tendon, while the one friction solution written as a profile on
  every piece is the original tendon (initial strains identical, midspan
  camber to 1e-6 in tests/test_fiber_tendons.py). Also the form a measured
  effective prestress takes.

### Documented

- `s` in a tendon profile is measured along EACH fibre-section block from
  that block's own start, and friction is solved over that block alone. A
  girder split into several section blocks therefore misplaces a tendon
  written in whole-girder coordinates on every block but the first
  (docs/fiber_beam.md). To change the steel over a stretch, keep one section
  and cut the tendon.

## [0.4.1] — 2026-09-24

A bug-fix release. A strand with a `rupture_strain` was cut by the
relaxation's own trial iterations rather than by the structure, which made
the corroded-beam capacities of v0.4 look mesh-dependent. With the latch
read against the accepted state, Li's eight corroded beams follow the section
method to within 1.5% on any mesh, and the degradation curve against the
tests is -1.3% to -11.3% (the eight-beam study itself lives
in the project that uses this library).

### Fixed

- **A strand cut by a trial, not by the structure**.
  `steel02.rupture_strain` now reads the peak strain of the last ACCEPTED
  state, before the current call raises it: the call that crosses the limit
  still carries, and the point is cut from the next one. A staged relaxation
  re-drives every point from the converged state on every iteration, and a
  stage imposes its whole displacement increment at once, so the element next
  to a load point transiently sees about increment / h. Read after the
  update, that transient cut the fibre inside the iteration and the stage
  converged on the broken branch although the intact one was in equilibrium
  too. On Li's corroded beam A7 it broke the strand at 19.73 kN m with 40 mm
  elements and 29.49 with 80 mm ones -- read in v0.4 as a mesh-dependent
  rupture criterion. It now reaches 39.48 / 39.51 kN m, against 40.14 by the
  section method; A4 goes 31.14 -> 52.65 / 52.71 (measured 53.72). In an
  explicit time history the deferral is one time step. Cases whose strands
  never reach `rupture_strain` (case_25) are bit-identical.
- **The stage that breaks a strand is solved again before it is recorded**.
  With the cut now made at the commit, recording straight
  away paired the intact positions with the cut forces (-169 kN of reaction
  on A7). The fibre-beam staged relaxation re-solves the same stage while a
  commit adds failed points (at most 50 passes), then records.

## [0.4.0] — 2026-09-24

The 0.4 line: time, and the substructure. Temperature, shrinkage, creep and
steel relaxation; the Hangzhou Bay unit given its piers and then its pot
bearings, shaken along and across; the Confederation Bridge rebuilt from the
literature and held to its measured truck, thermal and two-year records; a
member group with its own material, and with it a 1488 m road-rail
suspension bridge. The close-out is an assessment of what
this version can and cannot do on a real bridge's short-term effects.

The first half of the line is time: one mechanism at three difficulties -- a
strain the section takes on for reasons that are not force. Temperature depends on the
field you impose, shrinkage on the clock, creep on the member's whole stress
history; all three are subtracted from the deformation before the member law
sees it, and none of them is a load.

### Added

- **Temperature**: `temperature: {uniform, gradient, profile,
  axis}` on `beam2`, `beam2_timoshenko` and `beam2_fiber`, with `thermal_expansion`
  (1/K) on the material -- a property of the substance, so it is set once for every
  law and defaults to zero, and a fibre section takes each fibre's own.
  (Spelled out rather than `alpha`, which `bond_ceb` already uses for its own
  curve exponent -- one key with two meanings, decided by which material you
  happened to write.)
  A temperature is NOT a force: it is the free strain `alpha T(y, z)`, and
  the member force is `f = k (d - d_th)` with the free thermal deformation
  taken out of the DEFORMATION rather than added as a load. The
  equivalent-nodal-load form gets the same displacements and the wrong member
  actions -- a free beam under a uniform dT comes out the right length and
  reports `N = EA alpha dT` on a member that carries nothing -- and the fix
  there is to remember to subtract the thermal part again at every stress
  recovery, forever. On a fibre section the same idea goes one level down:
  each fibre's material is handed its MECHANICAL strain, so a code's
  nonlinear profile produces its self-equilibrated stresses by itself, with
  no special case anywhere. A `profile` on a section with no fibres is
  refused rather than reduced (the reduction needs the area distribution a
  `general` section does not have). See `docs/temperature.md`.
- **case_27: what is in the way decides.** Two 20 m spans: a uniform +40 K
  with the far end free grows 19.198 mm (exact 19.200) and holds its axial
  reaction at 0.05 N against the 10 080 kN it would have been if restrained;
  a 15 K/m gradient, which a single span would have bowed 9.0 mm for free,
  makes `3 E I kappa / 2` = 55.0 kN m at the middle support, lifts both ends
  2.750 kN, and leaves midspan at 2.2511 mm (exact 2.2500). case_27b puts a
  code-shaped NONLINEAR profile on a determinate fibre beam, where nothing
  can resist it and it is stressed anyway: -2.204 MPa at the heated top,
  +1.835 MPa just under the heated band, -1.504 MPa at the soffit, every one
  within 0.05% of the closed form, with the net force and moment at 1e-5 and
  4e-4 of what it would take to suppress the field.
- **case_28: the Confederation Bridge, reconstructed from the literature.**
  The P31-P32 portal frame -- 250 m between piers 60.98 and 68.58 m tall, two
  95 m cantilevers, a girder 14.0 m deep at the pier and 4.5 at midspan --
  built with nothing calibrated on a deflection. The outline is read off
  Dilger & Langohr's dimensioned section; its two undimensioned thicknesses
  are fitted on Butcher's published gross-section table, which then PREDICTS
  area, centroid and second moment within 4% at two depths and the midspan
  centroid within 1% with nothing fitted at all; the haunch exponent 1.380
  falls out of two published depths and reproduces Dilger's stated soffit
  slope (a parabola would not); the missing 15% of Ghali's 8000 t erection
  weight is carried as mass, not area.
  Against the measurements: the two-truck load test gives 17.0 mm at midspan
  against 17.1 measured (Ghali's own analysis reads 13.7), and the whole
  deflection profile to a few percent except the east tip, where the drop-in
  beam bears and no beam model of the frame reproduces the measured kink.
  case_28b puts the published summer temperature profile on it: midspan rises
  58.2 mm against 60.6 published and x = -180 m drops 43.9 against 43.7,
  while the cantilever tips go 22% too far -- the one station that reads how
  deep the hot band at the deck goes, which the source figure never
  dimensions. case_28c takes the gradient back out and isolates what it was
  worth: 9 mm at midspan, 31 at every tip.
- **Creep, shrinkage and ageing**: `creep`, `shrinkage` and
  `ageing` on the material, and `age` / `to_age` / `time_spacing` on a step.
  The same mechanism as a temperature, one step further out: a free strain
  that depends on the clock (shrinkage) or on the member's whole stress
  history (creep), handed to the same `f = k (d - d_imposed)` slot. Models:
  `mc90`, `aci209` and `curve` for both creep and shrinkage -- `curve` being
  measured data `[[t - t0, phi], ...]`, read linearly in log(t - t0), or a
  dict of them at several loading ages -- plus `Ec(t)/Ec(28) = exp[s(1 -
  sqrt(28/t))]`, with `of: strength` halving `s` because MC90's coefficient
  is a strength one and E follows its square root. Creep is the superposition
  integral over the elastic increments the member has actually taken,
  recomputed from the whole history at every station, so refining the time
  axis converges to the integral instead of drifting. Steps are log-spaced in
  age by default, because that is the axis creep happens on. `core.creep`
  also exposes `relaxation(spec, t, t0)`, the stress left in a member held at
  a strain -- which is NOT `1/(1 + phi)`, and not the age-adjusted formula
  either except in so far as its `chi` was fitted to this integral.
  See `docs/creep.md`.
- **The concrete age of every frame** is written beside the monitors as
  `age`, so a creep result can be plotted and checked on its own axis
  (`vfife plot --x age`, and `reduction: at, against: age` in a contract).
  Log-spaced stations make the step counter meaningless as a time axis.
- **case_29, 29b, 29c: the two closed forms creep has, and shrinkage's.** A
  bar loaded at 28 days and held creeps to exactly three times its elastic
  elongation and carries the same 300.0 kN from the first station to the last
  -- a determinate member redistributes nothing, which is the check that
  fails if creep is put in as a load. The same bar stretched 10 mm and HELD
  decays to 100.000 kN, which is `300 x R(1028, 28)` for a standard linear
  solid, whose relaxation modulus is closed form; six, twelve and twenty-four
  stations all land on it. And two drying bars side by side: the free one
  shortens `eps_sh L = -2.755 mm` and carries 2e-8 N against an 82.7 kN
  scale, the held one carries `E A eps_sh` at 28, 100, 365 and 1000 days.
- **case_28d: creep on a structure that can actually redistribute**.
  Ghali's two-year Confederation Bridge record has one increment that
  is fully specified -- the 54 kN/m superimposed dead load of April 1997 --
  and the paper states what its own analysis makes of it: about 70 mm at
  midspan and 80 at the cantilever tips. This model gives 66 and 89/90, with
  the aged modulus that Ghali's own fitted `s = 0.25` produces from the
  measured Ec(28); on the unaged 38.7 GPa the same frame reproduces the
  MEASURED truck deflection instead (17.0 against 17.1) while Ghali's analysis
  reads 13.7. Each overhang tip also carries the 1620 kN reaction of a simply
  supported 60 m drop-in beam; leaving it out moves the tips from 89 mm to 28.
  Held for 630 days of MC90 creep, every one of five displacement monitors
  grows by 1.532-1.534 against an exact `1 + phi = 1.53305`, and the largest
  bending moment in the model sits at 397.8 MN m from the instant of loading
  to the last station -- on a frame that is three times redundant, where
  case_29's bar could not have redistributed even had the implementation been
  wrong. Six, twelve and twenty-four stations agree to 4e-5. The three
  published long-term CURVES are deliberately NOT targets: they carry an
  erection sequence, tendon forces and steel relaxation this paper does not
  publish, and `verification/Ghali2000_confederation_longterm.txt` digitises
  them with that said out loud.
- **Fibre sections creep**: `creep` and `shrinkage` on the
  concrete material of a `type: fiber` section. The same mechanism as the beam
  path, one level down -- the free strain goes into the fibre's imposed-strain
  slot, the one a temperature and a tendon's prestrain already use, and the
  implicit half of the station step is a SCALE ON THE LAW'S OWN INPUT
  (`concrete02.strain_scale`) rather than a factor on a member stiffness,
  because a nonlinear law has no stiffness to scale: it has a stress at a
  strain, and what the station must hand it is the elastic strain. The elastic
  history is read straight out of the material's own committed strain, which
  has already been divided by `(1 + phi_cur)` in the one place that could not
  forget to. Creep is evaluated once per element, at the middle integration
  station, because the imposed-strain slot has no station axis -- the same
  resolution, and the same slot, a draped tendon's prestrain already uses.
  `case_29d` and `case_29e` are `case_29` and `case_29b` rebuilt out of
  sixteen `concrete02` fibres and land on the same thirteen numbers to a
  millionth; the middle of the relaxation decay comes within 2.8%, 1.0% and
  0.4% of the superposition integral on 6, 12 and 24 stations, moving towards
  it as the axis is refined. The clock itself (`read_step_ages`, `age_at`,
  `station_phi`, `station_date`) moved into `core.creep`, where both solvers
  read it, and a test asserts they put their stations at the same ages.
  Refused rather than approximated on a fibre section: `ageing` (it changes
  Ec, which on a nonlinear law is not one number), `creep` on reinforcement
  (steel relaxes, and that law is not here), `creep` on a material with no
  `strain_scale`, and two different creep laws in one model.
- **Relaxation of prestressing steel**: `relaxation` on a steel
  material, read by `vfife/core/relaxation.py`. It is a LOSS OF THE TENDON'S
  OWN PRESTRAIN rather than a strain added to it -- `sigma_p = E_p (eps_section
  + eps_pre)`, so a stress loss at held section strain is `eps_pre` falling --
  so it rides the imposed-strain slot a tendon already uses and does nothing at
  all on a fibre whose prestrain is zero. Unlike creep it keeps no history, so
  several tendon steels in one model cost nothing.
  **There is deliberately no `mc90` and no `ec2` model.** Those coefficients
  could not be sourced verbatim, and a recalled version of one was checked
  against its own definition and FAILED it: at the stress ratio where it must
  reduce to `rho_1000` it returned 0.96% for a declared 2.5%. What is here is
  the shape the codes share -- `r(t) = rho_1000 (t/1000 h)^k`, with both
  coefficients required and no default -- plus a measured `curve`, so a case
  has to say where its numbers came from. case_29f's come from a published
  MC90 curve for a real strand (Ghali 2000 Fig. 7, digitised), which
  independently puts the exponent at 0.198-0.221.
  `case_29f` holds both ends of a pretensioned bar and reads the law off a
  reaction: 208.320 kN to 197.846 = `A_p sigma_pi [1 - r(t)]`, with the ratio
  exact to 1.2e-4 (the absolute value sits 0.23% low at every station, which is
  `steel02`'s smooth curve already softening at 0.89 of yield strain, not the
  relaxation). `case_29g` frees one end and turns all three laws on: separately
  the losses are 200.7 + 53.6 + 82.4 = 336.8 MPa, together 320.8, and backing
  the ageing coefficient out of the answer station by station gives **0.22 at
  38 days rising to 0.99 at 1028** -- chi is not the textbook constant 0.8, and
  the long-term answer here is the plain effective-modulus method. The
  age-adjusted closed form it is bracketed against is DERIVED in
  `benchmarks/generate_creep_bar.py` from that bar's own numbers, not quoted.
- **case_24e: what a bearing is worth**. Version 0.4 measured a
  cost it could not price -- `case_24d`'s piers are MONOLITHIC with the girder,
  because a bearing is a `connector2` and the fibre-beam solver cannot host
  one, and that joint DOUBLES the middle-span dead-load deflection. This is the
  elastic twin that prices the fix, on `mixed_line`, which has hosted `beam2`
  and `connector2` together since case_03: the prestress shortening costs
  0.11 mm with pot bearings against 9.88 mm on the monolithic fibre model, so
  the bearing is worth 15 mm of a 23 mm deflection.
  TWO THINGS THE SWEEP SAYS THAT THE HEADLINE DOES NOT. Releasing the ROTATION
  alone removes the penalty as completely as releasing the slide -- it is a
  moment and thrust transfer, not a sliding question, and the transition sits
  at the pier's own EI/L -- so a FIXED pot fixes it too and the arrangement
  decides only where the girder's fixed point sits. And the sensitive
  assumption moves rather than going away: `case_24d`'s pier wall was worth
  10% of the deflection, and the bearing's VERTICAL stiffness, which does
  nothing at all to the penalty, is worth 230% over a plausible range.
  Checked against the classical three-span continuous reactions (1.1 qL and
  0.4 qL, to 0.5%) and against an independent direct-stiffness frame (the
  dead-load middle span to four figures).
  `write_line_mesh` grew a `bearing` option that puts the pier top on its own
  node; `bearing_price()` and its own reference solve are in
  `benchmarks/generate_hzb_girder.py --check`.
- **A fibre mesh can carry members that are not fibre beams**:
  `element.fiber_connectors`, reached from the input by assigning a
  `type: connector` section to a region of a fibre model. This is what
  case_24e priced, and `case_24f` is it done -- the Hangzhou Bay unit with its
  real tendons, its real piers and a pot bearing between them, whose middle-span
  dead-load deflection is **-7.68 mm against case_24d's -18.99** and the elastic
  twin's -7.59. The `case_24c/24d` headers' "not a comparison" can come back in.
  NO NEW MECHANICS. `elements/connector.py` answers with `FiberBeamSolver`'s own
  `q`, `orient`, `v` and `R0` unchanged, and was verified to before any of this
  was written. What was written is bookkeeping: an element with no section, no
  fibres and no mass of its own travels through the section rows, the reference
  stiffness, the internal-force loop, the section strains and the output writer,
  and the fused kernel is handed `_kernel_active = _active & _is_fiber` so that
  it never sees one. The two masks are deliberately separate -- `_active` is the
  fracture latch and the solver WRITES it, so one flag doing both jobs would
  mean the next fracture put a bearing back into the kernel. Kernel and NumPy
  reference agree to 1e-11 on case_24f with the connectors in the mesh.
  THE ONE COST, and it is not the one that was predicted. The evaluation
  measured a bearing-scale stiffness added to a girder node, where the box
  girder's own mass swamps it, and concluded the dynamic step bound was
  unchanged. A real bearing hangs off a PIER-TOP node carrying only pier mass:
  case_24f in dynamic mode takes 507 170 steps against case_24d's 231 089, a
  factor of 2.2. case_24f is therefore the static dead state; a collapse run on
  bearings is its own piece of work with its own budget.
  Preflight refuses a connector node that carries no other element and is not
  fully restrained (`CONNECTOR_NODE_UNHELD`): a pot bearing RELEASES components,
  a zero stiffness row has zero pseudo-mass, and the failure is a NaN at step
  one or -- worse -- one node quietly flying off.

- **A member group with its own material**: a `mixed_line`
  model's sections may name different materials, and each group of members
  carries its own E, nu, rho and thermal_expansion (`member_E`, `member_rho`,
  `_material_of`). Until this, `io/model.py` refused a second material on the
  line solvers by name -- which made a steel-cable, concrete-tower bridge
  impossible to write down. The E-scaling workaround it replaces was exact on
  axial, bending and translational mass and is now measured to be wrong in two
  places: rotary inertia (0.535 of the true value, because `nodal_masses` has
  no rotation) and torsion (7.7% soft, because scaling E cannot scale nu). A
  single-material model writes no `material` key and is unchanged to the bit.
- **The Xihoumen road-rail suspension bridge, from its design paper**
  (`benchmarks/generate_xihoumen.py`, `cases/_xihoumen.yaml`):
  1488 m main span, spatial main cables, three separated steel boxes with
  Vierendeel cross-beams, stays and hangers at 14 m, A-shaped towers -- a
  stated dead-load state, the one unknown deck load calibrated on the design's
  tower-base force and then checked independently against its girder axial
  force (+7.6%) and cable plan offset (-0.22 m). The three frequencies the
  designer published come out at -10.0%, -5.4% and +5.6% (first symmetric
  lateral, first symmetric and antisymmetric torsion). Not a verified case:
  the box plate thicknesses are estimated, and a 1.25x steel area would bring
  both lateral and torsion within 2.5%, which is written down and not folded
  in. The truss members of a `mixed_line` model are now evaluated in one
  vectorised pass (`_truss_forces_batched`), pinned to the per-element
  version, which took the modal tangent of this 346-cable model from over six
  minutes to about five seconds.
- **Mode shapes as ParaView time series**: `vfife modal
  <case> --paraview [DIR] [--frames N] [--mode-scale S]` writes one `.pvd`
  per mode (the points are the displaced positions, so Play animates it),
  `<stem>_all.pvd` with mode k on t in [k-1, k), and the table, with the
  mesh's own physical groups as cell data. Until this the only output of
  `vfife modal` was an HDF5 of arrays with no mesh in it.
- **Temperature on cables and bars**: an axial member of a
  `mixed_line` model takes its free thermal elongation `alpha dT Lu` off the
  stretch before E A sees it -- from its UNSTRESSED length, so a pretensioned
  stay loses exactly E A alpha dT of its tension. `tests/test_cable_temperature.py`
  holds both closed forms (held: -E A alpha dT; free: +alpha dT L, no force).
  A `profile` on a bar region is refused by name. (Until this it was a
  silent defect, see Fixed.)

### Fixed
- **A generated mesh its generator did not reproduce.**
  `benchmarks/generate_hzb_girder.py` writes `hzb_girder_3span_14_piers.msh`
  and `..._bearings.msh` from the same `--n-pier`, and the committed piers
  mesh had FOUR pier elements against the bearings mesh's six -- so no single
  run could produce both, and merely running the generator silently rewrote
  case_24d's mesh (4 elements to 6) and its answers with it. A container that
  had run it then generated case_24d's per-element pier regions against a mesh
  the repository does not contain, and the case failed on a machine that had
  the committed one: `region 'pier1_seg1' covers no complete Line2 element`,
  after preflight had passed (regions are resolved when the solver is built,
  not by `vfife check`).
  The piers mesh is now six elements like the bearings mesh, and
  `test_the_generator_reproduces_every_committed_mesh_asset` regenerates all
  four into a temp directory and compares them byte for byte. case_24d's
  numbers move for this as well as for the pier section, so its contract is
  re-cut from the next run of it.
- **The pier section was an assumption and it was wrong in the direction that
  mattered**. Version 0.4 had no published
  statement of the Hangzhou Bay piers and derived a SOLID 6.25 x 1.80 m from
  their weights. Three independent papers -- this unit's own
  黄燕庆《水中低墩区引桥总体设计》§3, 赵毅《高墩区引桥下部结构及基础设计》§3 and
  罗园辉《高墩区引桥墩身的施工》§1 -- state a HOLLOW rectangle, 6.25 m across
  the bridge by 2.60 m along it at the base growing to 4.00 m at the top, walls
  0.50 m. In the plane of the frame that is **4.4x stiffer** (I 3.04 -> 13.39
  m4 at mid-height); ACROSS the bridge the old assumption was right to 8%, by
  luck; the area is 24% smaller.
  v0.4's sensitivity sweep -- "the assumed thickness is worth 10%", 1.62 to
  2.08 m solid, I 2.2 to 4.7 m4 -- was run on a bracket that does not contain
  the answer.
  case_24d, case_24e, case_24f and the elastic twin now carry the stated
  section, tapered with one fibre section per pier element (`pier_properties`
  and `pier_depth` in `benchmarks/generate_hzb_girder.py`; the taper is linear,
  which is a reading and not a measurement -- this unit's paper gives only the
  two end widths).
  WHAT IT MOVED IS THE POINT. On the bearings (case_24f) the middle span goes
  -7.678 to -7.977 mm, 3.9%, and the elastic twin predicted +4% before the
  fibre model was re-run. On the MONOLITHIC joint the same correction takes the
  prestress penalty from -7.1 to -12.4 mm (+74%), so case_24d's "the joint
  doubles the dead-load deflection" is an understatement. **The bearing does
  not only remove the penalty -- it removes the model's sensitivity to the
  pier**, which is the assumption v0.4 was worried about, and that is now
  measured against a real section rather than argued.
  Also corrected by the same papers: the unit's basic length is 6 x 70 m (a
  note had 14 spans, which is the HIGH-pier zone), the girder segments weigh
  2140 t (middle span) and 2160 t (side span), and the tendons are jacked from
  ONE end with alternating anchorages -- `case_24f` still says `jack: both`,
  which is now a known discrepancy rather than an unexamined default.
- **A base acceleration along an axis nothing can move in was a silent
  no-op**. `ground_motion` enters in the relative frame as
  `-m a_g(t)` on the translation DOFs; if the direction lies in DOFs
  constrained at every node, all of it goes into the reactions and the
  structure does not move -- exactly, not approximately. The solve converges,
  every monitor is flat, and the reading is that the model is broken
  somewhere else. The case it was found on is a plane frame -- case_24f and
  its seismic twin lock `uy` at all 57 nodes -- where `direction: [0, 1, 0]`,
  the obvious second run to make with a real record, excites nothing.
  Preflight now refuses it by name (`GROUND_MOTION_FULLY_RESTRAINED`), and a
  mixed direction with any live component still passes, because half of one
  is a real analysis.
- **Four uniaxial laws said they could not do the thing they were doing.**
  `concrete01`, `concrete02`, `steel02` and `steel_cyclic` declared
  `backends=("numpy",)` while each carries `numba_uniaxial_fn` and
  `numba_params` -- exactly what the fused fibre kernel inlines -- and every
  shipped fibre case has been running on that kernel since v0.1.
  `compatibility()` checks the material's backends as well as the element's,
  so `backend: numba` on case_24f was refused at the front door:
  `CAPABILITY_MISMATCH: concrete02 does not support backend numba`, while
  `backend: numpy` on the same file was silently promoted to the very kernel
  the message denied.
  This is v0.1's defect one level down. That stage fixed the ELEMENT
  half (`beam2_fiber` declares numpy and numba) and wrote tests for it -- but
  every one of those tests builds the solver DIRECTLY, so none of them ever
  went through `compatibility()`, which is the door a case file uses. The new
  tests in `test_capability_alignment.py` go through it. Independently, both
  laws' own `qualification/*.yaml` had cited a `backend_parity` test for
  years.
  Corrected to `("numpy", "numba")` -- NOT warp: `kernels/warp_materials.py`
  carries elastic, j2, cdp, chaboche, rebar and the two bond laws, and none
  of these four.
  Genuinely numpy-only, and unchanged: `orthotropic`, `piecewise_uniaxial`,
  `viscoelastic` (no numba hook of any kind). `linear_elastic` and `rebar`
  keep their numba declaration, which is true for the SOLID kernels' point
  functions, but neither has a uniaxial twin, so a fibre model of them runs
  the NumPy path -- and that status line now NAMES the law instead of saying
  "a material", which sent a reader through a section with four of them in it.

- **The fused fibre kernel could not carry an imposed strain on a CONCRETE
  fibre.** It had `spre` for steel only, while the NumPy reference added
  `fiber_prestrain` to every fibre; the two agreed because nothing had ever
  set one on concrete. A temperature field does, so the kernel grew `cpre`
  and the two paths are asserted equal again.
- **The residual scale collapsed on a load that has no force.** A temperature
  field is the first load here whose converged state has neither an external
  force nor an internal one -- a beam free to expand ends up longer and
  carries nothing -- so the relative residual divided a small number by a
  smaller one and read 0.238 on a model converged to 2.4e-7 of its own
  thermal action. The restraint force (what it would take to suppress the
  field) now joins the force scale: it is the load's magnitude whether or not
  any of it survives to the answer.
- **Two step-by-step schemes for creep that do not work**, kept in the code
  as comments because both are the obvious thing to try. Dating each elastic
  increment to the END of its interval -- so that it has not crept when its
  own station is solved -- makes the scheme explicit, and on a held strain it
  is marginally UNSTABLE: twelve stations wobbled around the answer and
  landed 5% low, and twenty-four were worse (+8%), not better. Dating it to
  the middle of its interval WITHOUT putting the current increment's creep
  into the station's own stiffness is pure over-relaxation and diverges --
  the bar ended up carrying 332 kN, more than the 300 it was stretched to,
  which relaxation cannot do. The scheme that works is both halves together:
  the geometric mid-interval date and a `1/(1 + phi_cur)` factor on the
  station's local stiffness.
- **`benchmarks/sweep_confederation.py` and `plot_confederation.py` sorted
  the result file's step names as STRINGS**, which is exactly what
  `tests/test_step_ordering.py` exists to catch and what breaks a ParaView
  animation the moment the frame counter changes width. Both now sort with
  `key=int`.
- **An ageing or creeping model invalidated the wrong cache field.** The
  batched beam reference is guarded by its FIRST field (`_beam_ids` in the
  mixed solver), and the stiffness update cleared `_beam_dofs` instead, which
  left the guard set and the dof map `None`; the next element-action pass
  reshaped `q[None]`. Latent until a station changed the stiffness.
- `patch_fibers`'s `ny`/`nz` document what they actually count: the patch's
  own edges (corner0->corner1 and corner1->corner2), not the section's axes.
  Getting them the wrong way round changes neither area nor fibre count, only
  which direction is resolved finely -- invisible until a through-depth
  quantity needs the layers, where it is worth 15%.

- **The energy audit booked damping as error, and three solvers fed it a
  step late**. No solver passed the Rayleigh or viscous
  damping force's work to `EnergyLedger`, so on every damped dynamic run it
  came out as `energy_imbalance`: the Hangzhou Bay seismic runs reported a
  relative energy error of 0.54 and 0.77, and the damping work was 93-100% of
  it. Separately, `beam`, `beam2_fiber` and `truss2` called the ledger AFTER
  the position update, pairing the forces of step n with the increment of step
  n+1 -- the lag v0.1 had removed from the continuum solver only. On
  the 3-D seismic run it drifted to 15% of the external work by 6.5 s, after
  cracking put energy into modes near the step limit. Now every solver books
  `fdamp . du` into `dissipated` (for the c1/c2 viscous update the force is
  zeta m (v_old + v_new)/2) and feeds the ledger forces and state of one
  instant; the same 7 s of the 3-D run now stays within 4e-4 of the external
  work throughout and ends at 1e-7.
  NO ANSWER CHANGES -- displacements, forces and every contract are
  untouched; only `energy_imbalance`, `relative_energy_error` and
  `dissipated_energy` move. `tests/test_energy_damping.py`: a suddenly loaded
  cantilever or bar, damped to rest, on each of the three solvers.
- **A temperature on a region of beams and cables heated only the beams**
  (v0.4 scan item 8, fixed in v0.4). `_thermal_deformation`
  filtered the region to beam members and dropped the rest without a word;
  the run converged and reported success with the stays at their erection
  temperature. See Added.
- **Nodal masses were counted twice in the `coupled` and `shell_line`
  solvers**: both built their second half from a deep copy of
  the problem that kept `nodal_masses`, and both summed the halves' masses --
  exactly x2.0000 on case_07, so every period was sqrt(2) too long and the
  summary still printed the number the user wrote.
- **A fibre model's torsional stiffness came from whichever section was
  written first**. `G J` took E and nu from `self.material`,
  the first fibre section's cover material; swapping the order of two
  sections in the YAML changed every member's torque by 6.46x on a
  concrete-and-steel beam. Each element now takes its own section's G, in the
  kernel and in `_reference_k`. All thirteen shipped fibre cases have one E
  and are unchanged to the bit.
- **A load nobody routes has to say so**. The coupled solver
  split loads with two white-lists and silently dropped what was on neither --
  `temperature` and `moving_load`, and any `traction`/`pressure`/`body_force`
  written inside a step. It is now one partition with the remainder refused
  by name; the solid solver's load dispatch, which ended in a bare `else` and
  turned a `temperature` into NaN, now refuses it too.
- **Mode shapes left condensed DOFs at zero**. `modal.analyse`
  condenses massless-but-stiff DOFs out of the eigenproblem, correctly, and
  never recovered them: u_s = -K_ss^-1 K_sr u_r. Frequencies were unaffected,
  so only an animation showed it -- the node stayed where it started while
  its neighbours swung. A massless cantilever tip now follows u2 + theta2 L.
- **case_24d's contract predated the pier correction**.
  Re-cut from a run on the literature pier: middle-span dead deflection
  -18.99 -> -27.77 mm (+46%; the elastic twin had predicted +49%), side span
  -17.18 -> -12.89, pier-top sway 11.39 -> 10.89 mm -- 4% for a 4.4x stiffer
  shaft, because the girder imposes it. The bands on the two deflections
  tighten from 15% (which had to cover an assumed thickness) to 3%. case_24f
  on the same correction moves 3.9%.
- **A bearing cost 80% of a seismic step**: `connector.response`
  rebuilt its 6 x 12 Jacobian from 24 separate calls of `deformation`, all
  Python overhead on 3 x 3 arrays. The 25 configurations are now evaluated as
  one stack -- same perturbations, same differences, equal to the old columns
  at the difference's own rounding floor (~5e-10). A response is 0.20 ms
  instead of 3.41; the Hangzhou Bay seismic step 3.3 ms instead of 12.6.

- **The fused fibre kernel ignored a member deactivated from outside the
  solver** (found by the 0.4.0 release check). Since v0.4 the kernel
  is handed `_kernel_active = _active & _is_fiber` so that a bearing never
  enters it, and that product was refreshed only where the solver itself
  wrote `_active`. Anything else that wrote it -- `tests/test_fracture.py`
  deactivates a member by hand to check the kernel against the NumPy
  reference -- was honoured by the reference and not by the kernel, and that
  test had been failing since the bearings went in. The mask is now derived
  in place on every call; no shipped case writes `_active` from outside, so
  no answer changes (case_24f, 29e and 31 re-verified).

### Known limitations carried from 0.3

No contact; fracture separation on frames only; shells linear elastic; the
damped explicit step bound conservative under Rayleigh `beta`; no cracked
shear model on the fibre beam; dynamic relaxation without arc-length; a
moving load is a moving FORCE. Creep and shrinkage are now modelled, but
construction staging is not, so a girder built in stages is compared at its
one-stage state. New with this line: bearings (`connector2`) are linear
springs -- no friction, gap or isolator hysteresis; seismic input is uniform
base excitation only, with no response spectrum, multi-support excitation or
soil-structure interaction; with Rayleigh `alpha` alone the high modes are
undamped and reactions from a cracking explicit run ring above what a
monitor samples (v0.4 -- read them filtered, or add `beta`); a
`nodal_masses` entry has no direction and has weight, so hydrodynamic added
mass cannot yet be written. For bridge engineering the library is an
analysis engine, not a design package: no load combinations or envelopes, no
code checks, no lane or influence-surface loading, no linear buckling, and a
box girder modelled as a beam has no shear lag, distortion or warping
(README, *Known limits*; the assessment is v0.4).

## [0.3.0] — 2026-09-13

The 0.3 line: bridges. Cables and form-finding, prestressing tendons in the
fibre beam, an arbitrary fibre section with voids, modal analysis at a loaded
state, the deformed model swept out as a solid for ParaView, and loads that
travel. The anchors are a 300 m suspension bridge against the deflection
theory, Hangzhou Bay's 70 m box girder against two published analyses, two
13 m hollow-core slabs against the test that broke them, and a force crossing
a beam against Fryba.

### Added

- **Moving loads**: a load that TRAVELS along a line of
  members -- `moving_load: {axles: [[offset, force], ...], start, speed |
  stop, direction}` on `beam2`, `beam2_timoshenko` and `beam2_fiber`. The
  members of the region are ordered into one open chain (a fork, a ring or
  two disconnected runs is refused by name), `s` is arc length along it,
  and a point force at `s` is distributed onto the containing member by the
  element's own Hermite shape functions -- exact at the nodes for a
  prismatic beam wherever between them the axle stands, where the lever
  rule loses 27% on one element per half span. `speed` is a crossing in
  physical time; `stop` instead sweeps the group over the step's progress,
  which under staged relaxation is an influence line for the cost of n
  relaxations. Axles off the path carry nothing, so a group drives on at
  one end and off at the other. A moving FORCE, not a moving mass: no
  vehicle inertia, no suspension, and the station is on the undeformed
  path. See `docs/moving_load.md`.
- **case_26: a force crossing a beam** against Fryba's modal series --
  100 kN at 40 m/s over 20 m with f1 = 2.000 Hz (alpha = 0.50): inside
  0.3% over the crossing and 0.05% at the peak, which is 1.70x the static
  P L^3/48EI and arrives a third of the crossing AFTER the load passed
  midspan. case_26b walks the same axle across in 39 quasi-static
  stations, half of them between nodes, against the analytic influence
  line: worst relative error 1.3e-4.
- **Cables**: a `bar` section on a line region takes
  `prestress` or `unstressed_length` (the same statement, `L_u = L_0/(1 +
  sigma/E)`) and `tension_only`. A cable meshed on its design geometry
  carries its design tension at `u = 0`; a slack stay carries nothing.
  Cable members size their fictitious masses on the axial stiffness in all
  three translations, so a slack straight chord can be relaxed into shape.
  Plain trusses are unchanged to the bit.
- **case_22: Irvine's elastic catenary**, found from the straight chord by
  relaxation: H and sag to 0.03%, V exact. 45 s.
- **Form-finding**: `vfife formfind <case> --groups a,b`
  finds the unstressed lengths that leave the mesh at its dead-load
  geometry (relax, `L_u <- L_u L_0 / L`, repeat; geometric convergence) and
  writes the case back with per-element `unstressed_lengths`, the third
  way of stating a cable's length. `vfife.formfind.find` is the API.
- **case_23: a suspension bridge**, 300 m span, form-found dead load (the
  closed-form lengths to 4e-7, H_w = w l^2/8f) and a live load against the
  linearised deflection theory: h to 0.1%, deflections to 0.5%. 20 s.
- Contract reductions `first` and `delta` (final minus first): the built
  state and what the last step added.
- **Prestressing tendons in the fibre beam**: a fibre
  section's `tendons` entry -- a bar section, a `profile` [[s, y, z], ...]
  (linear or quadratic), and `prestress` or `jack` + `friction` (the
  post-tensioning loss profile the embedded tendons use) -- becomes a
  bonded steel fibre with its own eccentricity per element and an initial
  strain; `elastic_shortening: compensate` makes the stated stress the one
  the tendon carries after the concrete has shortened. Closed forms held:
  parabolic camber 5Pe L^2/48EI (16 elements 0.04%), uniform-moment shape
  of a straight tendon, chord shortening. Without tendons every model is
  bit-identical.
- **`vfife sweep`**: the fibre sections swept along the
  deformed members by the particles' exact triads into a solid of fibre
  hexahedra -- one cell per concrete fibre, so a Clip or Slice shows the
  section's damage fibre by fibre (`--surface` for the outline shell
  only); .pvd + one .vtu per frame, painted with each fibre's stress when
  the result carries the new `fiber_stress` field (nel x stations x
  fibres, both paths), plus the members' actions and fracture state. A split
  member is drawn on the particles it fell with: frames now carry
  `connectivity`. `fiber_damage` (new field) paints each face with the
  nearest fibre's damage class -- 0 intact, 1 cracked, 2 crushed,
  3 yielded, 4 failed, from the laws' new `damage_class(state)` -- and the
  rebars and tendons come as a second line series painted the same way.
- **`vfife modal`**: natural frequencies, periods, modal
  mass fractions and shapes at a state -- the finite-difference tangent of
  the internal force against the lumped physical mass, held dofs removed,
  massless ones condensed or dropped. Asked at a result's frame it sees
  the geometric stiffness of the loaded state: Irvine's cable modes at the
  catenary and case_23's vertical modes against the deflection theory's
  eigenproblem (0.3-1.1 %), the Euler beam to 0.1 %.
- Fibre-beam integration rules `lobatto4` and `lobatto5` beside gauss2 and
  lobatto3 (`INTEGRATION_RULES`); every path -- kernel, sweep, checkpoint --
  takes the station count from the rule.
- The Gmsh reader records which ELEMENTS a line group holds, as the coupled
  reader did; a hanger group no longer swallows the deck it hangs.
- **Voids in a `patches` fibre section**: a patch takes
  `holes: [[y, z, r], ...]`; a cell whose samples all fall inside one is
  dropped and a cut cell keeps the fraction of its area outside (0.3% of
  the net area on a unit square with a quarter-radius hole). A hollow-core
  slab is a rectangle with the circles taken out.
- **case_25: Chen's 13 m pretensioned hollow-core slabs**, the first
  MEASURED destructive test in this line of work -- 99 x 55 cm with two
  R19 voids, nine strands at 195 kN with four different bonded lengths
  (`extent`), pushed to failure by two jacks as the test was, displacement
  controlled. Against the measured points (digitised in
  `verification/Chen2005_hollow_slab_{1,2}.txt`): beam 1# within 1-5%
  over the whole curve with the peak 8% low and the failure deflection 8%
  early; beam 2# (the jacks 6 m apart instead of 4) WITHIN 2% from 20 to
  140 mm, its peak 5% low. Both fail with the strands past yield, which
  is the failure the test had. 2#'s recorded maximum, 370.6 kN, is one
  frame at the snap -- the displacement control catching a slab that has
  let go -- so the peak is read off the branch before it
  (`plot_chen_slab.stable`). The case
  file argues the two tails of the concrete curve (epscu, Ets) and gives
  the section's moment-curvature under each; with the code's 0.0035 the
  section cliffs and the beam localises into one element at half the
  test's deflection.
- **The `patches` fibre section**: `shape: patches` with
  `patches: [{corners: [[y, z] x 4], ny, nz}, ...]`, each quadrilateral
  subdivided bilinearly into fibres -- a box girder with inclined webs,
  fillets and tapered cantilevers in ten entries. `torsion_constant` on a
  fibre section replaces the fibres' polar moment (five times too stiff
  for a box) in the torsional stiffness. A bar entry's `count` lumps
  identical bars at one point.
- **Tendons anchored inside the member**: a tendon's `extent: [s0, s1]`
  gives it zero area outside its anchors -- pier-top cap tendons, one
  span's draped tendons when three spans share a section.
- **`fiber` monitors** (`kind: fiber, region, field: strain|stress, at:
  [y, z]`): the plane-section strain, or the nearest fibre's stress, at a
  point of the section at the station nearest the node -- what a strain
  gauge on the deck or the soffit reads. `FiberBeamSolver.section_strains()`
  is the API.
- **case_24: the Hangzhou Bay Bridge 70 m box girder**, three files.
  `case_24b` the simply supported elastic girder: P L^3/48EI to 0.01% and
  the modes against Xu's ANSYS (first vertical 1.77 vs 1.85 Hz, second
  6.84 vs 6.72, lateral and torsional ordered as his). `case_24` the
  3 x 70 m continuous unit with Chen's design tendons (29 tendons, 59 640
  mm2 per span, 36 960 per pier), dead load and the JTG lane load to
  k = 10 by staged relaxation: the midspan dead-load strains to 16% of
  his, the pier and the live-load stiffness not (the notes say what could
  be behind each). `case_24c` the same pushed to failure by a dynamic
  ramp: pier yield and a snap at k = 14.8, the strand-hardening branch,
  his 727 mm at k = 17.3 and the mechanism at 18.1 against his k_u = 16.8,
  then the span hanging on its strands. 76 s and 4.6 min. Chen's curve is
  digitised in `verification/Chen2005_hzb_girder_case1.txt` (it ends 1%
  from his stated ultimate) and `benchmarks/plot_hzb_k_delta.py` draws the
  comparison. Both curves start at their own dead-load deflection, not at
  the origin -- 5.9 mm here, 21.3 mm for Chen, the difference being that
  his girder is precast simply supported and made continuous -- so the
  comparison is of increments: the model is a flat 0.65-0.72 of his.

### Fixed

- The kinetic-ratio convergence measure no longer collapses on an
  UNDEFORMED stage. It divides the kinetic energy by
  `force x displacement`, and the displacement half hit a 1e-12 floor when
  a stage genuinely bends nothing -- a moving load standing on a support,
  with the whole axle going straight into the reaction -- so a model at
  rest to 4e-20 J reported a ratio of 8e-9 against a 1e-10 tolerance and
  failed at the one state it had exactly right. The scale now keeps a
  millionth of the largest displacement the analysis has reached, which is
  what the FORCE scale beside it has done since v0.1, and is inert
  wherever anything is deflected.
- A verification contract may read `time`, `residual` and
  `kinetic_energy`: the writer puts them beside the monitors in every
  result file, but nobody declares them in `output.monitors`, so the
  invariant that a contract only names monitors its case records was
  reporting case_24c's three by-the-instant metrics as missing.
- **case_24's tendons are one entry per duct**: 110 of them, each with its
  own (y, z) profile -- the web ducts inside the webs (and moving outboard
  as they rise, so the plan path and its friction come in), the bottom
  slab's 12 across the soffit, the pier's 22 across the deck at the
  spacing measured off Fig. 6.25b's plan panel. Lumping them on the
  centreline, as the case first did, is identical in the vertical plane to
  0.2% but draws the pier's 22 ducts as four coincident lines. A fibre
  beam's `summary()` reports more than twelve tendons as totals by area
  rather than a line each.
- **A draped tendon was drawn as a staircase**. The sweep's
  bars series put each element's steel row at the element's own (y, z) --
  for a tendon, the profile sampled at the midpoint -- at BOTH ends of the
  element, so a drape came out as a run of horizontal chords, which on a
  210 m girder reads as a straight tendon. The drawn polyline now runs
  between the profile's values at the element's two ends: continuous at
  the nodes and on the declared profile. The mechanics is unchanged (the
  section still integrates one eccentricity per element).
- `vfife sweep` clears the old frames out of its output directory before
  writing. The frames are numbered by kept-frame index, so re-sweeping a
  result with a different `--every` used to leave a mixture of two runs in
  the folder (the .pvd listed only the new ones; anything opened by hand
  was a lottery).
- **An elastic fibre answered with the constrained modulus.** The fibre
  beam handed `linear_elastic` the 3-D law with [eps, 0, 0, 0, 0, 0] and
  got C11 eps = E (1 - nu) / ((1 + nu)(1 - 2 nu)) eps back: 7% stiff at
  nu = 1/6, 35% at 0.3. Every fibre test had nu = 0 and never saw it; the
  Hangzhou Bay girder's static check did. Fibres now read E. Models with
  concrete/steel laws or nu = 0 are unchanged to the bit.
- `stage_ramp_iterations` in the `solver` block now reaches the frame and
  shell solvers (they read it from a step only); the fibre beam already did.

**Tested at this tag** (Linux, 2 threads): default tier 1779 passed / 13
skipped in 26 min; all 32 cases have a contract each and the nightly
matrix runs them one job per case. The slow tier was last run in full at
0.1.0. `pyflakes` is clean on everything but the re-export `__init__` files;
`vfife qualify check` passes on all 18 manifests.

### Known limitations carried from 0.2

No contact; fracture separation on frames only; shells linear elastic; the
damped explicit step bound conservative under Rayleigh `beta`; no cracked
shear model on the fibre beam; dynamic relaxation without arc-length. New
with the bridges: a moving load is a moving FORCE (no vehicle inertia, no
suspension, no road roughness -- which on a real bridge is the larger part of
the dynamic amplification), creep and shrinkage are not modelled, so a
staged-construction girder is compared at its as-built state only, and the
displacement-controlled push records one transient frame at the instant a
structure snaps (case_25b) because there is no arc-length control.

## [0.2.0] — 2026-09-12

Reinforced concrete frames: several reinforced sections on one model, a
published frame test as the anchor, and the two member-level mechanisms
that test asked for.

### Added

- **Several reinforced fibre sections on one model**. Each
  `fiber` section in `assign` carries its own cover law, confined core,
  bars and `anchorage`; the solver builds one material group per law per
  section, the fused kernel reads parameters per fibre as before, and a
  one-section model is bit-identical to what it was. Lifts the
  "no rebars, no cover/core, one law" limit of 0.1.
- **case_21: Vecchio & Emara (1992) one-bay two-storey RC frame** pushed to
  155 mm, with the measured curve digitised from Guner &
  Vecchio (2010) and checked against the energy that paper quotes. 36 s.
  Plateau within 3%, peak 1.04 (of which ~6% is the 8-per-member mesh),
  rising branch 1.15x stiff for the cracked shear the element lacks — held
  as a band by the contract.
- **Member shear flexibility on the fibre beam**: a fibre
  section may state `shear_stiffness` (kappa G A_s). The bending
  interpolation becomes Reddy's interdependent Timoshenko one and a
  constant member shear strain is carried; the fibres never see shear. A
  section without it is the Hermite element to the bit. Closed-form
  cantilever to 2e-4, kernel and reference to 1e-10.
- **Staged ramp-in on the fibre beam**: `stage_ramp_iterations`
  blends a prescribed displacement in over a level's first iterations and
  forbids convergence inside them, as `beam2` has done since v0.1. Opt-in:
  measured on case_11 and case_21 it changes the answer by 0.3% and costs
  up to 2.4x, so the default stays at one iteration.

### Changed

- Nightly CI installs numba and solves each case on the backend it
  declares (`warp` mapped to `numba`); the slow contract test is
  `test_a_case_solved_satisfies_its_contract`. case_15 stays
  out of the nightly matrix: its cost is the bonded-reinforcement path's
  per-step overhead, not the kernel (254 steps/s on numba too).

**Tested at this tag** (Linux, 2 threads): default tier 1648 passed / 13
skipped in 24 min; the slow tier was last run in full at 0.1.0 and the
nightly matrix now covers it per case.

### Known limitations carried from 0.1

No contact; fracture separation on frames only; shells linear elastic; the
damped explicit step bound conservative under Rayleigh `beta`; no cracked
shear model on the fibre beam; dynamic relaxation without arc-length.

## [0.1.1] — 2026-09-12

The first nightly after the tag showed what a 120-minute job cannot hold.

### Changed

- **Nightly CI is one job per case.** Twenty full solves on the NumPy
  path in a single job exceeded its budget, and a timed-out check is as
  green as a skipped one. The `verify` job is now a matrix over
  `cases/*.yaml` (`fail-fast: false`), with the remaining slow tests in a
  job of their own. `case_15_chalmers_prc` is left out of the matrix by
  name for now -- ~100 minutes on the NumPy path on its own -- until it
  has a cheaper verification form; `tests/test_ci_layers.py` asserts the
  matrix equals the case list minus the named skips, so a new case that
  nobody added to the matrix fails the fast tier.
- **`beam2_fiber` accepts `backend: numba`.** `numpy` remains the readable
  path, promoted to the fused kernel when the model qualifies; `numba`
  requires the kernel and is refused with the kernel's own reason
  (`solver.kernel_status`) when it cannot be built -- no more silently
  running the reference path 100x slower after asking for the kernel.
- **`truss2` declares every uniaxial law** (`steel02`, `steel_cyclic`,
  `concrete01`, `concrete02` added), which is what `TrussSolver` always
  accepted; preflight used to refuse a `steel02` bar the solver would run.
  `tests/test_capability_alignment.py` now asserts that every declared
  (element, material) and (element, backend) pair builds a solver.

### Fixed

- `concrete01`, `concrete02` and `bond_ceb` overrode `init_state` without
  the base class's `dtype` argument, so a material-point set of them (a
  concrete strut as a `truss2` bar) raised `TypeError`.
- `vfife capabilities` caveats for `truss2` and `beam2_fiber` describe the
  sections, stations and backends they actually have.

## [0.1.0] — 2026-09-11

First tagged release. Everything below has been on `master` for some time;
this entry describes what the tag contains rather than what changed since a
previous release.

### What it is

An explicit co-rotational finite element core driven by YAML case files, with
a CLI (`vfife run | verify | check | probe | info | init | mesh | plot |
export | qualify | migrate`), a Python API (`vfife.api`) and a test suite that treats published
experiments and benchmarks as contracts.

**Elements.** Continuum: `hexa8`, `hexa20`, `tet4`, `tet10`, `wedge6`,
`wedge15`, `pyramid5` (b-bar, full, reduced and selective integration,
hourglass control). Line: `truss2`, `beam2` (3-D co-rotational Euler-Bernoulli
frame), `beam2_timoshenko`, `beam2_fiber` (fibre section with rebars, cover
and confined core, hollow tube/box sections, several sections per model,
strain penetration, fracture with separation), `connector2` (six-component
spring/dashpot). Shell: `shell4`, `shell3`. Coupling: `coupled` (solid with
shared-node line members), `mixed_line`, `shell_line`; embedded reinforcement
in solids with perfect or bonded interfaces (`bond_ceb`, `bond_cyclic`) and
pretensioning prestress.

**Materials.** `linear_elastic`, `orthotropic`, `j2`, `chaboche`,
`viscoelastic`, `cdp` (concrete damaged plasticity with crack-band
regularisation), `concrete01`, `concrete02`, `steel02`, `steel_cyclic` (Cai et
al. 2023, with its fracture criterion), `rebar` (with rupture strain),
`piecewise_uniaxial`, `bond_ceb`, `bond_cyclic`.

**Solvers.** Dynamic relaxation (kinetic damping, staged load steps,
displacement control with ramp-in) and explicit dynamics (Rayleigh or viscous
damping, ground motion, amplitude tables) on every family; the explicit step
bound accounts for stiffness-proportional damping. Backends: NumPy everywhere;
Numba fused kernels for continuum, `shell4`, `beam2` and `beam2_fiber`;
Warp/CUDA for continuum and `shell4`.

**Fracture.** Three layers — the material declares its own point failure,
the element reduces it to a fraction, the model decides what to do: a
fractured frame member stops carrying load, keeps its mass, and with
`split: true` its bonds are cut so the fragment falls under gravity.

**Verification.** 20 shipped cases under `cases/` with contracts under
`verification/` (NAFEMS LE10, Lee's frame, Sze's hinged cylindrical roof,
Tanaka & Park column, University of Alberta and Chalmers beams, a 45° bent
cantilever, a floating beam, a rolled-up cantilever, …), `qualification/`
gates per element and material, and invariant tests (rest stays at rest,
dimensional consistency, material-layer invariants, kernel-vs-reference
bit-equality) that run on every push. Two-tier CI: fast on every push,
full-solve verification nightly.

**Tested at this tag** (Linux, 2 threads, `pip install -e ".[dev,plot,cpu]"`):
default tier 1599 passed / 13 skipped in 29 min; slow tier (`-m slow`, the
full-solve contracts) 48 passed in 3 h 55 min, of which case_15 alone is
105 min. Warp/CUDA tests were not run here and are skipped without a GPU.

**Input.** `materials → sections → assign` case files, Gmsh `.msh` (2.2 and
4.1) and an engineering subset of Abaqus `.inp`, unit-system declaration
checked by preflight, HDF5/XDMF output for ParaView.

### Known limitations (stated, not hidden)

- **No contact.** A severed member falls through whatever is below it.
- **Fracture reaches separation on frames only**; the solid path erodes
  elements and loses their mass. A fragment joins region `all` only, and `dt`
  is not adapted after a split.
- **Shells are linear elastic only.**
- **The explicit step bound is conservative under Rayleigh `beta`** (up to
  ~5x below the measured limit for heavily damped models): the Gershgorin
  `omega_max` estimate enters the damping factor twice.
- **Several fibre sections on one model carry no rebars** and no confined
  core; RC frames with different reinforcement per member need one section.
- **The fibre beam's staged relaxation does not forbid convergence during
  the prescribed-displacement ramp-in** the way `beam2` does (`it > ramp`);
  it only refuses the first iteration.
- **Dynamic relaxation can lose a rapidly softening region** (no arc-length
  control); see the README's Known limits for what was measured on Lee's frame.
- **`tet4` locks**; use `tet10`.
- **Near-incompressible relaxation converges slowly.**
- **Prestress is pretensioning only** (elastic-shortening loss); no staging,
  no unbonded tendon.
- **Solid and structural meshes do not mix in one file** except through the
  `coupled` / `shell_line` paths.
- `precision: mixed` on Warp is not safe on every model (preflight measures
  and warns).
- The `capabilities` declarations of `truss2` and `beam2_fiber` are known to
  disagree with what the solvers actually accept; `vfife capabilities` is
  the intent, the solver's own error message is the truth.
