# vfife result format 3 (and checkpoint format 1)

This page specifies what `vfife run` writes, so that a third party can read a result file without vfife.
It is the layout that is frozen at 1.0. Result format 3 was introduced in v1.0; format 2 is
described at the end, and vfife still reads it.

## Files of a run

`vfife run case.yaml -o run.h5` writes these files next to each other:

| file | what it is |
|---|---|
| `run.h5` | the result: mesh, frames, monitors, diagnostics, the embedded case (HDF5) |
| `run.ckpt.h5` | the checkpoint: the state `--restart` and `solver.initial_state` read (HDF5; see "Checkpoint file"). The name is the result's with `.h5` replaced by `.ckpt.h5`; any other extension is kept (`run.hdf5` -> `run.hdf5.ckpt.h5`). A result may not itself be named `*.ckpt.h5` (`OUTPUT_IS_CHECKPOINT_NAME`) |
| `run.xdmf` | XDMF 3 temporal collection over `run.h5`, for ParaView |
| `run_last.xdmf` | the last frame alone, as one Uniform grid (a fallback for older ParaView builds) |
| `run_members.xdmf` | coupled / shell_line models: the line members as their own grid |
| `run_rebar.xdmf` | models with embedded reinforcement: the bars as one polyline grid |

The `.xdmf` files are written when the run ends, and only refer to datasets inside `run.h5`.
If a run is killed, `run.h5` is still readable up to the last frame, because the file is flushed at every frame
and at every checkpoint. The monitors, the diagnostics, `analysis_status` and the `.xdmf` files are written at the end
of a run, so a killed run's result has frames but none of those (`vfife plot` says `RESULT_NOT_CLOSED`).
`run.ckpt.h5` is always either the previous checkpoint or the new one, never half of one:
it is written to `run.ckpt.h5.tmp` and then renamed over the old file.

## Conventions

* **Units.** vfife does not convert units. Every number is in the consistent system the case declares
  (`units:`), and the result records that system in the root attributes `unit_length`, `unit_force`, `unit_stress`,
  `unit_mass`, `unit_time`, and so on, one attribute per declared key. If the case declares no units, these
  attributes are absent; m, N, Pa, kg and s are recommended.
  Derived units follow from the declared ones: a moment is force x length, energy is force x length, and
  rotations are in radians. A default the case left out that carries a unit (a vehicle's gravity, the water density
  of `hydrodynamic_mass`, a material's `rho`, ...) is written in the declared units too (user manual
  chapter 4).
* **Indices** (node, element and particle numbers in connectivity and in tables) are 0-based.
* **Floating point.** Mesh coordinates, monitors, diagnostics, spectra and checkpoints are float64.
  Frame fields are float32, and connectivity is int32.
* **Strings** are UTF-8 HDF5 variable-length strings.

## Root attributes of `run.h5`

| attribute | type | meaning |
|---|---|---|
| `result_schema_version` | int | the layout, **3**. A reader refuses a larger number (`RESULT_TOO_NEW`). A file without this attribute predates format 2 |
| `vfife_version` | str | the version that wrote the file |
| `created_utc` | str | ISO 8601 creation time |
| `element_family` | str | the model's element family (`beam2`, `mixed_line`, `beam2_fiber`, `truss2`, `hexa8`, `coupled`, `shell_line`, ...) |
| `time_axis` | str | what a frame's `time` is: `time` (seconds, dynamic) or `load_factor` (relaxation and static, see "Time") |
| `unit_<key>` | str | the case's declared units, one attribute per key |
| `case_yaml` | str | the complete case as it was run |
| `analysis_status` | str | `CONVERGED`, `COMPLETED_DYNAMIC`, `MAX_STEPS`, `NONFINITE`, `INTERRUPTED`, ... |
| `analysis_converged` | bool | whether the result is an answer |
| `run_id` | str | a random id of this run (32 hex digits); every checkpoint the run writes carries the same id |
| `checkpoint_file` | str | the name of the checkpoint file next to it; present once the run has written one |
| `restarted_from` | str | only in a continuation (`--restart`): the absolute path of the result it resumed (also when the checkpoint file was named). Such a file holds **only the tail** of the run |
| `restarted_at_step` | int | only in a continuation: the step counter at which it resumed |
| `model_mass` | float64 (3,) | 1.1: the model's translational mass summed per global axis (the members', embedded bars', nodal, hydrodynamic and `with_mass` masses, and what selective mass scaling added) |
| `model_weight` | float64 (3,) | 1.1: the force the `gravity` loads of the last step (model-level and its own) apply at full amplitude: g times the mass that weighs. Absent where the lumped mass cannot say it: a truss or solid with gravity on part of it, a solid with embedded bars |
| `model_line_load` | float64 (3,) | 1.1: the same step's `line_load`s at full amplitude, q times the loaded length, global axes |
| `model_point_load` | float64 (3,) | 1.2: the same step's nodal forces (`vector` loads) at full amplitude |
| `restarted_at_time` | float | only in a continuation of a dynamic run: the time (s) at which it resumed, the clock of the event tables in `diagnostics/` |

## `mesh/`

| dataset | shape | meaning |
|---|---|---|
| `nodes` | (n_nodes, 3) float64 | reference coordinates. For models that can split, this includes the spare particles reserved for that (parked at their parent node) |
| `elements` | (n_elem, nodes_per_element) int32 | connectivity in the reference configuration |
| `elements` (mixed mesh) | (n,) int32 | an XDMF `Mixed` stream `[type code, node ids..., type code, ...]`; attribute `blocks` lists the block names in stream order |

## `steps/` — frames

There is one group per output frame. The group's name is the solver's **cumulative iteration counter** when the frame was
written, zero-padded to 8 digits (`00012345`). Sort the frames with `key=int`, not alphabetically.
A run longer than 10^8 iterations writes longer names.

Frame group attributes:

| attribute | type | meaning |
|---|---|---|
| `time` | float | see "Time" |
| `record` | int | the frame's index in this file: 0, 1, 2, ... |
| `equilibrium` | str | **always a string**: `"true"` (a converged equilibrium), `"false"` (a relaxation or static frame that is not converged), `"not_applicable"` (a dynamic frame) |
| `load_factor` | float | only in `load_factor` results; the same number as `time` |
| `age` | float | only in time-dependent (creep, shrinkage, relaxation) runs: the concrete age in days at the frame |

Frame datasets are the fields that `output.fields` asks for (`vfife describe <element>` lists them).
Whether a field lives on nodes or on elements is decided by its name, not by its length:

* **on the nodes** (first axis n_nodes): `displacement` (3), `rotation` (3, the rotational degrees of freedom in rad,
  global axes), `velocity`, `acceleration`, `reaction_force` (3, the support's force on the structure, which opposes
  the load), `reaction_moment`, `stress_nodal` (6), `temperature`;
* **on the elements** (first axis n_elem): everything else, for example `von_mises`, `stress` (6), `axial_force`,
  the envelopes `axial_force_max_abs` ..., the end actions `axial_force_i` ... `bending_z_j`, `member_failed`,
  `capacity_ratio`, `hinge_rotation`, `hinge_usage`, the connector components `connector_force_x/y/z`,
  `connector_moment_x/y/z`, `connector_deformation_ux` ... `_rz`, `connector_slip_ux` ... `_rz` (forces by axis,
  deformations and slips by degree of freedom), the mixed models' `member_type` (1 a beam member, 0
  otherwise), `beam_formulation` (1 Timoshenko) and `connector_flag` (1 a connector), and the material outputs (`dc`, `dt`, ...);
* **per fibre** (3-D): `fiber_stress`, `fiber_strain`, ... with shape (n_elem, n_stations, n_fibres). XDMF has no
  attribute for these; `vfife sweep` reads them.

Six-component tensors (`stress`, `stress_nodal`) are stored in the order **[xx, yy, zz, xy, yz, zx]** with the
true shear stress. In XDMF they appear as six scalars, `stress_xx`, `stress_yy`, `stress_zz`, `stress_xy`,
`stress_yz`, `stress_zx` (the last was named `stress_xz` before v1.0), and as a `Matrix`
attribute (not `Tensor6`, because VTK orders that differently). `_last.xdmf` carries only the six scalars.

`hinge_rotation` is the accumulated plastic rotation of the worse end, in rad. `hinge_usage` is that rotation divided by
the rotation capacity: 0 is elastic and 1 is a hinge about to tear. Both are NaN for a member without a hinge.
Before format 3, `hinge_rotation` held the ratio.

### Sign conventions of member forces, end actions and reactions

One convention for every line family -- beam2, beam2_timoshenko, mixed_line (its beams, bars and connectors),
the members of shell_line and coupled models, beam2_fiber, truss2. Local axes: x from the member's node i to its
node j, y the assignment's `orientation`, z = x × y.

* **Internal forces** -- `axial_force`, `shear_y`, `shear_z`, `torsion`, `bending_y`, `bending_z` -- are the
  stress resultants at a cross-section, in the local axes, acting on the face whose outward normal is +x (the face of
  the part on the node-i side):
  * `axial_force` is positive in **tension**;
  * `shear_y`, `shear_z` are positive along +y, +z;
  * `torsion`, `bending_y`, `bending_z` are positive by the right-hand rule about +x, +y, +z, so `bending_y` = ∫σ z dA
    and `bending_z` = −∫σ y dA. For a girder with local z up a **sagging** moment is a **negative** `bending_y`
    (bottom fibre in tension).
* One value per member: the end with the larger magnitude, **with its sign**. A fibre member gives its section
  resultants at the integration station with the largest magnitude (gauss stations are inside the member, so a
  fibre member's end moment can exceed it); its `shear_*` is the member shear of its `shear_stiffness` or `shear`
  law (0 for an Euler–Bernoulli fibre section).
* `axial_force_max_abs`, `shear_y_max_abs`, ..., `bending_z_max_abs` are max(|end i|, |end j|) (a fibre member:
  the station's magnitude) -- what the plain names held before v1.0, when they were unsigned.
* **End actions** `*_i`, `*_j` (the group `end_actions`) are the forces the nodes exert **on the member**, in the
  same local axes (f = k d, the member's equivalent nodal forces). By equilibrium of a short piece next to each end
  the internal force is −`*_i` at end i and +`*_j` at end j: a bar in tension has `axial_force_i` < 0 and
  `axial_force_j` > 0. The loads inside a member (a line load, an axle between the nodes) are taken off them.
* A member that has failed (`member_failed` = 1: broken, removed or split off) reports **0** in all of these: a
  fragment's own forces are not forces of the structure.
* `reaction_force` / `reaction_moment` (nodes, global axes) are the support's force on the structure: under a
  downward load the vertical reaction is positive. A `kind: reaction` monitor is their sum over the region.
* `connector_force_*` / `connector_moment_*` are the connector's signed force in its own axes (= the internal
  force above, at end j).
* The response spectrum's `axial_force`, `bending_y`, ... are combined peaks (CQC/SRSS), **magnitudes** by
  construction, the larger of the two ends.

`vfife view` draws a signed member force on a scale centred on zero.

`connectivity` (n_elem, 2) int32 appears in the frames of a model whose members can split. It is that frame's
connectivity: a fragment's ends sit on its own particles. The XDMF topology of such a frame points at it.

## `members/`, `member_steps/` (coupled, shell_line)

`members/elements` (n_members, 2) holds the line members' connectivity, using the same node numbering as `mesh/nodes`.
`member_steps/<frame>/` holds their fields (axial force, end actions, connector quantities, `member_type`), plus
`displacement` and `rotation` copied from the shared nodes. The groups and their `time` are the same as in `steps/`.

## `rebar/`, `rebar_steps/` (embedded reinforcement)

`rebar/nodes` (2 n_bar, 3) holds the bar ends' reference positions, interpolated from the host.
`rebar/elements` is (n_bar, 2), and `rebar/block` (n_bar, float32, whole numbers) gives each segment's reinforcement set. The group attributes
`block<k>` hold the set names. `rebar_steps/<frame>/` holds `displacement` (node), `axial_stress`,
`plastic_strain` and `block` (segment).

## `monitors/`

| dataset | shape | attributes | meaning |
|---|---|---|---|
| `time` | (n_records,) | `kind` = the `time_axis` | the record's time (see "Time") |
| `<name>` | (n_records,) | `kind`, `regions`, `dof`, `units` | one monitor of `output.monitors`, recorded at every record (see below) |
| `age` | (n_records,) | `kind`=`age`, `units`=`d` | time-dependent runs: the concrete age at each record |
| `residual` | (n_checks,) | `sampling`=`progress_check` | the convergence measure, sampled at the progress checks, **not** at the frames |
| `kinetic_energy` | (n_checks,) | `sampling`=`progress_check` | the same, for the kinetic energy |

A **record** is when the monitors are taken. In dynamic and relaxation runs the records are the output frames
(`n_records` = the number of groups in `steps/`). In static runs (`solver.mode: static`) the monitors are recorded at
**every increment**, and the frames are thinned by `output.every`, so `n_records` can be larger than the number of
frames; a frame's `time` (its load factor) is one of the records' `time`. A static run whose case does not write
`output.every` writes a frame at every increment (the default 200 counted increments and left a
90-increment sweep with its last frame only). Every monitor, `time` and `age` have the same length, `n_records`.

An `element` monitor on a line model is one element field (any name the frames can hold: `axial_force`,
`bending_y_max_abs`, `capacity_ratio`, ...) over the members its `region` owns -- those of the physical group, or every
member whose nodes are all in the region, on the connectivity as read -- reduced by `reduce` (`max` by default, `min`,
`mean`, `absmax` = the signed value of largest magnitude). It is taken at every record, from the same values a frame
would write; a member that has failed reads 0.

A `fiber` monitor also carries what it reads: `read_element` and `read_station` (the station, as a fraction of
the member from its node 0, nearest the region's node), `read_nodes` (how many nodes are averaged), `read_y`, `read_z`
(the point, or the coordinates of the fibre read), `read_fiber` (`concrete`, `bar`, `tendon`, or the plane section for a
strain) and, for a stress, `read_material` (the fibre's law).

`time`, `residual`, `kinetic_energy` and `age` are reserved: a case cannot name a monitor that.
The `units` attribute is missing when the case declared no units; it is `""` for a quantity that has none (a strain).

## `diagnostics/`

Attributes, each present when it is known (an attribute whose value is unknown is not written: a converged run has no
`message`, a run stopped before its first progress check no `residual`). `load_step` is the analysis step counter at the
end, 1-based (a dynamic run has it too). `location_json` is always written, `"{}"` when nothing failed:

* `status`, `converged`, `mode`, `backend`, `step`, `load_step`, `residual`, `kinetic_energy`, `message`;
* `energy_available`, `energy_unavailable_reason`;
* `location_json`: where a failure happened (JSON);
* `vehicle_contact_json` (vehicles only): per vehicle, the airborne-tyre and rigid-axle-tension step counts and the
  first time each happened.

Datasets:

| dataset | shape | attributes | meaning |
|---|---|---|---|
| `step` | (n,) | `units` = `""` when the case declared units | the iteration count of each energy snapshot |
| `external_work`, `internal_work`, `kinetic_energy`, `dissipated_energy`, `energy_imbalance` | (n,) | `units` = force*length (absent when the case declared no units, as on the monitors) | the energy ledger |
| `relative_energy_error` | (n,) | `units` = `""` when the case declared units | imbalance / scale |
| `member_splits` | (n, 4) | `columns` = `time,element,parent_particle,new_particle` | one row per element end moved onto a particle of its own |
| `member_failures` | (n, 4) | `columns` = `time,element,capacity_ratio,cause`; `cause_codes` | one row per break: cause 0 = capacity (for a piece broken at both ends, twice it), 1 = timed removal, 2 = what a massless member connected is gone, 3 = twisted a quarter turn end to end (1.17) |

A table that would be empty is not written.

**In a continuation** (`restarted_from`) the event tables `member_splits` and `member_failures` hold the **whole
history** of the run -- the checkpoint carries it -- while the frames and monitors hold only the tail. Each table has
the attribute `rows_before_restart`: its first that many rows (time at or before `restarted_at_time`) are already in
the file it resumed. Read the continuation's table alone for the run's events, or join the files with the earlier
table followed by this one's rows from `rows_before_restart` on; concatenating the two whole tables counts those
events twice. In a run that is not a continuation the attribute is 0.

## `response_spectrum/` (cases with a `response_spectrum` block)

Group attributes: `n_modes`, `combination`, `damping`, and `connector_kinks` (only when a connector law sat on
its kink at the linearisation: the connectors named, "; "-separated). Every dataset has a `dims` attribute naming its
axes; `effective_mass` also a `meaning` attribute saying what its numbers are.

| dataset | dims | meaning |
|---|---|---|
| `period`, `frequency` | mode | s, Hz |
| `spectral_acceleration` | mode,direction | the spectrum's ordinate at each mode, per direction |
| `effective_mass` | mode,direction | ratio of the model's mass in that direction |
| `modal_mass` | mode,direction | Γ², in the mass unit |
| `mass_reached` | direction | cumulative effective mass of the modes taken |
| `directions` | direction,xyz | the excitation directions |
| `displacement`, `rotation`, `reaction` | node,xyz | combined peaks |
| `base_force` | xyz | combined base resultants |
| `end_actions` | element,action | `columns` = `axial_force_i,shear_y_i,shear_z_i,torsion_i,bending_y_i,bending_z_i,axial_force_j,...` |
| `axial_force`, `shear_y`, ... | element | the larger of the two ends |

## `tendons/` (cases with a top-level `tendons:`)

What each tendon of the case's `tendons:` expanded to, for `vfife tendons`. Group attribute `step_names_json`: the
names of the case's steps (as expanded), a JSON list -- a frame's `analysis_step` indexes it. One subgroup per
tendon, `t00001`, `t00002`, ..., in the order of the case:

| attribute | meaning |
|---|---|
| `name` | the tendon's name |
| `count` | strands or bars in its duct |
| `area` | its area (all strands) |
| `jacking` | [beginning, end]: the jacking force at each end, in the case's force unit |
| `stress_step` | the index of the step that stresses it (in `step_names_json`) |
| `regions_json` | its bars' regions, in order along it (`name/bar1`, ...), a JSON list |

| dataset | dims | meaning |
|---|---|---|
| `elements` | bar | each bar's element row: the row of `steps/*/axial_force` (and of `mesh/elements`) |
| `jack` | bar | the force each bar was jacked to, after friction and draw-in, as the solver was given it |

A case written out by `vfife expand` (or `convert(..., expanded=True)`) has no `tendons:` any more: its result has
no `tendons/` group.

## Time

| run | frame `time`, `monitors/time` | XDMF `Time` |
|---|---|---|
| dynamic | seconds since the start, counted across steps (step k runs over (k T, (k+1) T]) | the same seconds |
| relaxation, static | completed analysis steps + the load factor λ of the current step (with one step, λ itself). On an arc-length path λ can decrease and become negative | **the frame's index** 0, 1, 2, ... (its attribute `record`); λ is a Grid-centred attribute `load_factor` (field data in ParaView), in `run.xdmf`, `run_last.xdmf` and `run_members.xdmf` |
| time-dependent (creep ...) | as relaxation; the concrete age is the frame attribute `age` and `monitors/age` | the frame's index, with `load_factor` and `age` as field data |

`vfife sweep` writes its `.pvd` on the same clock: seconds for a dynamic result, the frame's index for a load-factor one.

## Checkpoint file (`run.ckpt.h5`), format 1

The checkpoint file's root attributes say which run wrote it and which model it is the state of. The model is read
from the **lowered** case -- every section and material written into the elements that carry them -- so the names
of materials, sections and regions do not enter it, nor does a section or material no element uses; a case given as
a file, a dict or a `schema.Case` gives the same fingerprint.

| attribute | meaning |
|---|---|
| `checkpoint_format` | **1**; a larger number is refused (`CHECKPOINT_TOO_NEW`) |
| `vfife_version`, `created_utc`, `result` | the writer, the time, and the result file it sits next to |
| `run_id` | the `run_id` of the result that wrote it |
| `element_family`, `mode`, `backend` | as in the result (`mode` is `static` for `solver.mode: static`) |
| `n_nodes`, `n_elements` | counts, including spare particles |
| `nodes_sha256`, `elements_sha256` | hashes of the reference coordinates (float64; informative) and of the connectivity (int64 per block) |
| `layout_sha256`, `layout_json` | the section layout: each element's type, section shape and size, fibre arrangement, orientation and which elements carry it, without material laws and without the settings below; JSON {part: hash} names the parts (`section on <region>`, `block on <region>`, `reinforcement on <region>`, `element`) |
| `materials_sha256`, `materials_json` | the material laws and their constants (and connector laws and stiffnesses), wherever they act; JSON {where: hash} |
| `settings_sha256` | what may break and the initial strains: `failure`, `fracture`, `prestress`, `unstressed_length(s)`, `removals` |
| `conditions_json` | JSON {name: hash} of `units`, `constraints`, `loads`, `nodal_masses`, `hydrodynamic_mass`, `diaphragms`, `amplitudes`, `contact`, the steps' loads and constraints, `damping` and `mass` (`solver.mass`, `solver.mass_scaling`). File paths in them enter by file name only |

The dataset `fingerprint/nodes` (n_nodes, 3) float64 holds the reference coordinates; they are compared to a
billionth of the model's size, not bit for bit, so a generated mesh on another CPU is the same mesh.
The group `checkpoint/` holds the solver's state arrays. Their names and shapes belong to each solver; they are
read back only by vfife and are not part of this specification.

Which checkpoint is a result's: a result of format 3 names its checkpoint (`checkpoint_file`) and shares its
`run_id`. A checkpoint beside a result that names none, or with another `run_id`, belongs to another run (one that
could not be deleted when this run started: Windows keeps a file another program holds open) and is refused by name
(`CHECKPOINT_NOT_THIS_RUN`). A `.ckpt.h5` named directly, or found without its result, is read as it is.

How the fingerprint is used:

* `--restart` continues the same run. vfife refuses (`CHECKPOINT_MISMATCH`, naming what differs) a different element
  family, mode, node or element count, node coordinates, connectivity, section layout, material law or constant,
  setting (`settings_sha256`), any of the conditions (`conditions_json`), or any state array of another shape.
  Only the solver's budget (`n_steps`, `total_time`, the steps' budgets), its tolerances, the backend, the threads
  and `output` may change. A restart ignores `solver.initial_state` (the checkpoint is the state).
* `solver.initial_state` starts a new run from another run's end. The mode, the loads, the constraints, the masses,
  the damping, the material constants and the settings may change (a relaxed dead state without `fracture` or
  `failure` starts a run that may break). vfife refuses a different element family, element count, connectivity,
  section layout, any state array of another shape, or nodes that are not this mesh's first nodes (a state without
  spare particles can start a model that reserves them: they are parked at rest).
* A checkpoint stored *inside* a result of format 2 or older carries no fingerprint. `--restart` refuses it
  (`RESTART_OLD_CHECKPOINT`); `initial_state` still reads it, checking the result's `mesh/nodes`.

`output.checkpoint_every: N` writes a checkpoint about every N iterations: at the first frame or progress check at least N
iterations after the previous checkpoint. When N is smaller than the progress interval, the progress interval is
lowered to N (and `residual` / `kinetic_energy` are sampled that often). A run also writes a checkpoint at its end, and
when it is stopped -- except when its final positions or velocities are not finite (`NONFINITE`): then the previous
checkpoint, the last finite state, is kept. Ctrl-C and SIGTERM stop at the next progress check, so that checkpoint is
an exact step boundary. A second signal stops at once and keeps the previous checkpoint. `vfife run` exits 130 after
Ctrl-C and 143 after SIGTERM; through the Python API the stop is `AnalysisInterrupted`, whose `signal` attribute is the
signal (None for a stop by `on_progress`).

## Modal file (`vfife modal -o modes.h5`)

Root attributes are `modal_format` (1), `vfife_version`, `n_free` (free degrees of freedom), `note`, and `shape_layout`.
The datasets are:

* `frequency` (n_modes) in Hz;
* `period` (n_modes) in s;
* `participation` (n_modes, 3);
* `shape` (n_modes, n_nodes, stride), where stride is 3 (translations) or 6 (translations and rotations);
* `rotation_only` (n_modes, bool), marking modes with no translation, which are scaled to a unit largest rotation.

## Format 2 (vfife 0.5, before v1.0): what differs

* The checkpoint was the group `checkpoint/` inside the result, with no fingerprint.
* `equilibrium` was a bool for relaxation and static frames and the string `"not_applicable"` for dynamic frames.
* There were no `unit_*`, `time_axis`, `run_id`, `record`, `load_factor`, `age`, `restarted_from` attributes and no
  `columns` / `dims` / `units` attributes on tables.
* `hinge_rotation` held the usage ratio.
* The XDMF `Time` of a relaxation or static result was its λ.

`tests/data/result_format_2/` keeps a format-2 result that every later version must read.
