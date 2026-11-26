# vfife 错误代码 / error codes（自动生成，1.22.0）

> 由 `python -m vfife reference --codes` 从源码生成；`tests/test_review6_docs.py` 保证它与代码一致。**不要手改。**

Every code a refusal (`UserError`, raised by the Python API and printed by the command line as `error [CODE]: …`) or a preflight issue (`vfife check`, the start of every run; ERROR stops, WARNING and INFO do not) carries. A code is a stable name a script or a front end can match on; the message beside it is the start of the text, with `…` where a value is filled in.

Exit codes of the command line: 0 success; 1 an input mistake (a refusal, a file that cannot be read, invalid YAML or schema); 2 `check`/`info`/`run` preflight FAIL and `verify` FAIL; 3 an analysis that did not converge (`run`, `formfind`); 130 stopped by Ctrl-C; 143 by SIGTERM; 141 the output pipe closed.

Analysis statuses (`analysis_status` in the result, `AnalysisOutcome.status`; see docs/diagnostics.md): `CONVERGED`, `COMPLETED_DYNAMIC` (answers); `MAX_STEPS`, `NONFINITE`, `INTERRUPTED` (kept, not answers); `PREFLIGHT_FAILED` (an outcome of `api.run_case` only, nothing written).

| code | kind | first words of the message | source |
|---|---|---|---|
| `AMPLITUDE_INVALID` | ERROR | an amplitude does not read: … | vfife/preflight.py |
| `AMPLITUDE_PAST_PROGRESS` | WARNING | amplitude … runs to …, but its first column is the | vfife/preflight.py |
| `AMPLITUDE_REPEATS` | WARNING | a model-level load on … has amplitude …: it is applied | vfife/preflight.py |
| `AMPLITUDE_UNDEFINED` | ERROR | amplitude … is used but not defined (available: | vfife/preflight.py |
| `AMPLITUDE_UNUSED` | WARNING | amplitude … is defined but not used | vfife/preflight.py |
| `ANCHOR_SET` | refused | reinforcement …: … | vfife/core/solver.py |
| `ARC_NO_LINE_SEARCH` | INFO | every step here runs by arc length, whose corrector has no line search (its step | vfife/preflight.py |
| `BACKEND_UNAVAILABLE` | refused | this case asks for the warp backend on device …, | vfife/core/shell_solver.py, vfife/kernels/__init__.py |
| `BEAM_ROTATION_MECHANISM` | ERROR | coupled beam rotational subsystem has … mechanism DOF(s); add physical rotations/releases or use truss members… | vfife/preflight.py |
| `BEAM_SOLID_MOMENT_TRANSFER` | WARNING | a shared node transfers beam forces but a point solid node has no rotational DOF; restrain the beam rotation o… | vfife/preflight.py |
| `BONDED_BAR_BACKEND` | refused | bond-slip reinforcement cannot run on the warp backend yet | vfife/core/solver.py |
| `BUCKLING_BASE_NOT_AT_REST` | refused | the reference configuration carries internal forces of its own (up to | vfife/buckling.py |
| `BUCKLING_BASE_NOT_STABLE` | refused | the stiffness at the base state is not positive definite on the free degrees of freedom: | vfife/buckling.py |
| `BUCKLING_FAMILY` | refused | buckling is for line models (frames, fibre beams, bars and cables); this is a | vfife/buckling.py |
| `BUCKLING_STEP_AND_RESULT` | refused | --step and --result both choose the loads: the result's frame is in a step | vfife/cli.py |
| `BUCKLING_STEP_OUT_OF_RANGE` | refused | --step …: the case has | vfife/cli.py |
| `CAPABILITY_MISMATCH` | ERROR | member group … names material …; line members support | vfife/preflight.py |
| `CASE_INVALID` | refused | this case is not valid:\n | vfife/io/schema.py |
| `CHECKPOINT_MISMATCH` | refused | … … holds the state of a different model: | vfife/io/fingerprint.py |
| `CHECKPOINT_NOT_THIS_RUN` | refused | the checkpoint beside … (…) belongs to another | vfife/io/writer.py |
| `CHECKPOINT_TOO_NEW` | refused | … is checkpoint format …; this vfife reads format … | vfife/io/writer.py |
| `COMPRESSION_BAND_RATIO` | refused/ERROR | crack_band: both scales the compression curve by l_e/l_ref = | vfife/materials/cdp.py, vfife/preflight.py |
| `COMPRESSION_UNREGULARISED` | INFO | …: the compression curve has no `Gc`, so its crushing band is | vfife/preflight.py |
| `CONNECTIVITY_RANGE` | ERROR | element connectivity references an invalid node | vfife/preflight.py |
| `CONNECTOR_DYNAMIC_MASS` | ERROR | a standalone dynamic connector model needs nodal_masses; connectors carry no physical mass | vfife/preflight.py |
| `CONNECTOR_MECHANISM` | ERROR/WARNING | connector model has … unstiffened/unrestrained DOF(s); constrain inactive components or connect them to physic… | vfife/preflight.py |
| `CONNECTOR_NODE_UNHELD` | ERROR | … node(s) carry only connectors and are not fully | vfife/preflight.py |
| `CONNECTOR_PHYSICAL_DAMPING_IN_RELAXATION` | ERROR | connector_damping is physical-time damping and cannot act in relaxation pseudo-time | vfife/preflight.py |
| `CONNECTOR_RANK_SKIPPED` | WARNING | exact connector mechanism rank skipped above 2000 free DOFs | vfife/preflight.py |
| `COUPLED_TOPOLOGY` | ERROR | coupled model requires one solid topology and Line2 members | vfife/preflight.py |
| `CRACK_BAND` | INFO | …: l_e/l_ref = …..… on both curves | vfife/preflight.py |
| `DEPRECATED_FIELD` | WARNING | …: `…` is the old spelling of `…`; it is still read, and | vfife/preflight.py |
| `DIAPHRAGM_OVER_CONSTRAINED` | ERROR | diaphragm region … contains nodes whose in-plane degrees of | vfife/preflight.py |
| `DIAPHRAGM_TOO_SMALL` | ERROR | diaphragm region … has one node; a rigid floor needs at least two | vfife/preflight.py |
| `DIAPHRAGM_UNSUPPORTED` | ERROR | diaphragms are applied to beam, fibre-beam, shell and coupled models on the numpy/numba backends; | vfife/preflight.py |
| `DISTORTED_ELEMENT` | WARNING | shell4 Jacobian ratio is below 0.05 | vfife/preflight.py |
| `DUPLICATE_ELEMENT` | ERROR | duplicate elements have identical node sets | vfife/preflight.py |
| `DUPLICATE_NODE` | ERROR | distinct nodes share the same position (within …): | vfife/preflight.py |
| `ELEMENTS_UNASSIGNED` | refused/ERROR | … element(s) are covered by no `assign` entry; the first is | vfife/core/solver.py, vfife/preflight.py |
| `ELEMENT_TYPE_MISMATCH` | refused | the case says element.type = … but | vfife/assemble.py |
| `EMPTY_MESH` | ERROR | mesh contains no elements | vfife/preflight.py |
| `EMPTY_REGION` | ERROR | region … contains no nodes | vfife/preflight.py |
| `EROSION_UNSUPPORTED` | ERROR | solver.erosion removes solid elements; a … model loses members through | vfife/preflight.py |
| `EXPAND_MESH` | refused | — | vfife/cli.py |
| `FIBER_MONITOR_NO_FIBRE` | refused | monitor …: the section of element … has no | vfife/core/fiber_beam_solver.py |
| `FIBER_MONITOR_OUTSIDE` | refused | monitor …: at = […, …] is outside the section of element … | vfife/core/fiber_beam_solver.py |
| `FOLLOWER_UNSUPPORTED` | ERROR | `follower: true` is implemented for pressure on solid elements; on a … model | vfife/preflight.py |
| `FORMFIND_JACK` | refused | a cable with `jack` is stressed to a force in an analysis step; | vfife/formfind.py |
| `FORMFIND_MODE` | refused | form-finding relaxes the dead-load step to equilibrium; this case | vfife/formfind.py |
| `FORMFIND_NOT_RELAXED` | refused | … … did not reach equilibrium (…): … | vfife/formfind.py |
| `FORMFIND_STEP` | refused | step …: the case has … step(s), numbered 0 to | vfife/formfind.py |
| `FORMULATION_UNSUPPORTED` | ERROR | element … does not support formulation …; | vfife/preflight.py |
| `FRICTION_CLOSED_TENDON` | refused | reinforcement … asks for a friction profile on a closed | vfife/core/solver.py |
| `FRICTION_SPLIT_TENDON` | refused | reinforcement … asks for a friction profile but is laid as | vfife/core/solver.py |
| `FRICTION_WITHOUT_PRESTRESS` | refused | reinforcement … has a `friction` block but no `prestress` | vfife/core/solver.py |
| `FULL_ROTATIONAL_RELEASE` | WARNING | beam …-end releases all three rotations; verify this does not create a mechanism | vfife/preflight.py |
| `GRAVITY_MAGNITUDE` | WARNING | a gravity load of … with units.length = …: | vfife/preflight.py |
| `GROUND_MOTION_AND_RECORD` | ERROR | the support … follows its own `record` in … and a | vfife/preflight.py |
| `GROUND_MOTION_FULLY_RESTRAINED` | ERROR | ground_motion direction … lies entirely in | vfife/preflight.py |
| `GROUND_MOTION_NO_DIRECTION` | ERROR | ground_motion direction is the zero vector | vfife/preflight.py |
| `GROUND_MOTION_REPEATS` | WARNING | a model-level ground_motion is replayed from its start in every step; put it | vfife/preflight.py |
| `HIGH_ASPECT_RATIO` | WARNING | element corner-distance ratio exceeds 20 | vfife/preflight.py |
| `HIGH_LEVEL_BLOCK` | refused | — | vfife/api.py, vfife/cli.py, vfife/io/schema.py |
| `HOST_DEDUCTION_DAMAGING` | refused | reinforcement … sets host_deduction with the | vfife/core/solver.py |
| `INITIAL_STATE_OVERWRITES` | refused | output.path … is the result this run starts from | vfife/api.py |
| `INIT_EXISTS` | refused | … exists; init does not write over a file | vfife/cli.py |
| `INIT_INCOMPLETE` | refused | non-interactive init needs | vfife/cli.py |
| `INIT_NOT_A_VOLUME_MESH` | refused | … contains … elements; `vfife init` | vfife/cli.py |
| `INIT_OVERWRITES_MESH` | refused | … is the mesh file itself; the case would replace it | vfife/cli.py |
| `INTERNAL_KEY_IN_INPUT_FORM` | refused | this case has `materials` / `sections` / `assign` and also | vfife/io/schema.py |
| `INVALID_SHELL_FACE` | ERROR | shell4 faces are degenerate, inverted or excessively warped | vfife/preflight.py |
| `LINES_ARE_LINES` | refused | mesh type `lines` writes line members only; a model with shells or | vfife/assemble.py |
| `LINE_LENGTH_RANGE` | WARNING | line length ratio exceeds 100 | vfife/preflight.py |
| `LINE_LOAD_WITHOUT_MASS` | INFO | loads … … without | vfife/preflight.py |
| `LINE_MESH` | refused | — | vfife/assemble.py |
| `LOAD_STEPS_UNSUPPORTED` | ERROR | a truss2 model relaxes each step continuously: `load_steps` would only change the | vfife/preflight.py |
| `MATERIAL_ALIAS_CONFLICT` | refused | material …: … name the same parameter and differ | vfife/materials/base.py |
| `MATERIAL_NOT_A_NUMBER` | refused | material …: `…` is | vfife/assemble.py |
| `MERGED_SHARED_ELEMENTS` | INFO | … … cells were | vfife/preflight.py |
| `MESH_DUPLICATE_NODE_TAG` | refused | node tag(s) … appear more | vfife/io/gmsh.py |
| `MESH_GROUP` | refused | — | vfife/assemble.py |
| `MESH_PARAMETRIC_NODES` | refused | the mesh file stores parametric node coordinates, which this | vfife/io/gmsh.py |
| `MESH_SCALE_UNUSED` | refused | mesh.scale applies to a mesh read from a file; `lines` are written | vfife/assemble.py |
| `MESH_UNKNOWN_NODE_TAG` | refused | an element names node tag(s) …, | vfife/io/gmsh.py |
| `MIGRATE_FAILED` | refused | …: … | vfife/cli.py |
| `MIGRATE_MISMATCH` | refused | the rewritten case does not lower to the same model (… differences) | vfife/cli.py |
| `MIXED_ELEMENT_FAMILIES` | refused | the mesh mixes solid and structural elements: …. | vfife/io/gmsh.py |
| `MIXED_MESH_UNSUPPORTED` | refused | this model has … element blocks | vfife/core/solver.py |
| `MIXED_PRECISION_RESOLUTION` | WARNING/INFO | — | vfife/preflight.py |
| `MIXED_SAMPLING` | refused | … … sampled at the progress | vfife/post/__init__.py |
| `MIXED_SHELL_ELEMENTS` | refused | the mesh has … quadrilateral and … triangular shell | vfife/io/gmsh.py |
| `MODEL_BUILD_FAILED` | ERROR | the model could not be built (…): … | vfife/cli.py |
| `MONITOR_REDUCE_UNUSED` | ERROR | monitor …: `reduce` applies to element monitors (and bar monitors on | vfife/preflight.py |
| `MONITOR_UNSUPPORTED` | ERROR | monitor …: a … model records … monitors, | vfife/preflight.py |
| `NEW_EXISTS` | refused | … is a folder | vfife/cli.py |
| `NEW_UNKNOWN_TEMPLATE` | refused | no template … | vfife/cli.py |
| `NONFINITE_COORDINATE` | ERROR | mesh contains non-finite coordinates | vfife/preflight.py |
| `NONPOSITIVE_JACOBIAN` | ERROR | elements have a non-positive integration-point Jacobian | vfife/preflight.py |
| `NOTHING_TO_EXPAND` | refused | … has no high-level block (…, `table:`) | vfife/cli.py |
| `NOT_A_CASE` | refused | … is not a case (no mapping at the top level) | vfife/api.py |
| `NOT_A_RUN_RESULT` | refused | … is …: it has no mesh and frames | vfife/post/summary.py |
| `NO_CHECKPOINT` | refused | initial_state … holds no final state: there is no checkpoint | vfife/api.py |
| `NO_CONTRACT` | refused | no verification contract for …: looked for | vfife/cli.py |
| `NO_MATPLOTLIB` | refused | plotting needs matplotlib, which is not installed | vfife/post/__init__.py |
| `NO_MONITORS` | refused | … contains no monitors | vfife/post/__init__.py |
| `NO_PHYSICAL_GROUPS` | refused | the mesh has no boundary-sized Gmsh Physical groups to choose from | vfife/cli.py |
| `NO_SUCH_FILE` | refused | no such file: … | vfife/api.py |
| `NO_SUCH_FRAME` | refused | frame … is not in …; it has … frames, | vfife/modal.py |
| `NO_SUCH_RESULT` | refused | no such result file: … | vfife/post/__init__.py |
| `OLD_CASE_FORM` | refused | this case has no `materials` / `sections` / `assign`: it is written in the | vfife/io/schema.py |
| `ORPHAN_NODE` | ERROR | mesh contains nodes unused by every element | vfife/preflight.py |
| `OUTPUT_IS_CHECKPOINT_NAME` | refused | output path … ends in .ckpt.h5, the name of a checkpoint file | vfife/io/writer.py |
| `OVERLAPPING_MEMBER` | ERROR | mixed-line elements have multiple member assignments | vfife/preflight.py |
| `PLOT_NEEDS_Y` | refused | --x names the horizontal axis, so --y must | vfife/post/__init__.py |
| `PRESTRESS` | INFO | reinforcement …: … MPa… | vfife/preflight.py |
| `PRESTRESS_LOSSES` | INFO | … kN of prestress before losses; the solver computes | vfife/preflight.py |
| `PRESTRESS_MATERIAL` | refused | reinforcement … sets prestress but its material | vfife/core/solver.py |
| `PRESTRESS_PROFILE` | INFO | reinforcement …: … MPa AT THE JACK | vfife/preflight.py |
| `PRESTRESS_YIELD` | refused | reinforcement … is prestressed to | vfife/core/solver.py |
| `RAGGED_MONITORS` | refused | the monitors in this file have different lengths: | vfife/post/__init__.py |
| `REGION_NAME_CLASH` | refused | region … is both a point, line or group of mesh.lines and | vfife/assemble.py |
| `REGION_RESOLUTION` | ERROR | — | vfife/preflight.py |
| `REGION_SNAPPED` | INFO | — | vfife/preflight.py |
| `REMOVAL_AFTER_END` | WARNING | the removal of … is at t = … s, after the run ends | vfife/preflight.py |
| `REPEATED_ELEMENTS_IN_ONE_VOLUME` | WARNING | … … cells were | vfife/preflight.py |
| `REPEATED_ELEMENT_NODE` | ERROR | elements repeat a node locally | vfife/preflight.py |
| `RESTART_BUDGET_EXHAUSTED` | refused | this checkpoint is at the end of the analysis: there is nothing left to run | vfife/core/resume.py, vfife/core/solver.py, vfife/core/truss_solver.py |
| `RESTART_OLD_CHECKPOINT` | refused | … keeps its checkpoint inside the result (result format 2 or | vfife/api.py, vfife/core/resume.py |
| `RESTART_OVERWRITES` | refused | the restarted run would write over the result it restarts from, | vfife/api.py |
| `RESTART_UNSUPPORTED` | refused | a warp shell run cannot be resumed from a checkpoint | vfife/core/shell_solver.py |
| `RESULT_INCOMPLETE` | refused | — | vfife/cli.py |
| `RESULT_NOT_CLOSED` | refused | … was not closed (the run was killed?): its … frame(s) are | vfife/post/__init__.py |
| `RESULT_TOO_NEW` | refused | … is result format …; this vfife reads format … and older | vfife/io/writer.py |
| `RETIRED_MATERIAL` | refused | material … no longer exists | vfife/materials/base.py |
| `RIGID_BODY_MODE` | ERROR/WARNING | connected component … suppresses only … of … observable rigid-body modes | vfife/preflight.py |
| `SECTION_ELEMENTS` | INFO | section …: … element(s), material … | vfife/preflight.py |
| `SECTION_EMPTY` | refused/ERROR | section(s) … are assigned to a | vfife/core/solver.py, vfife/preflight.py |
| `SECTION_MESH_MISMATCH` | refused | the case's sections are solid or shell, but … contains only | vfife/assemble.py |
| `SECTION_OVERRIDE` | INFO | section … takes … element(s) also covered by | vfife/preflight.py |
| `SERIES_NOT_CONVERGED` | refused | material 'series' (…): the parts' stresses at point … | vfife/materials/composite.py |
| `SHELL_EXPERIMENTAL` | WARNING | — | vfife/preflight.py |
| `SHELL_LINE_TOPOLOGY` | ERROR | shell_line requires Shell4 and selected Line2 blocks | vfife/preflight.py |
| `SHELL_NORMALS_INCONSISTENT` | ERROR | the pressure on … acts on shells whose normals point different | vfife/preflight.py |
| `SPECTRUM_FILE` | refused | — | vfife/assemble.py |
| `SPLIT_BAR_ROUTE` | WARNING | bonded reinforcement … has … line(s) that | vfife/preflight.py |
| `SPLIT_CAPACITY` | refused | element … needs a new particle and the split capacity | vfife/core/topology.py |
| `SPLIT_NEEDS_DYNAMIC` | refused | element.fracture.split needs solver.mode = dynamic | vfife/core/fiber_fracture.py |
| `STATIC_BLOCK_IGNORED` | WARNING | solver.static (Newton, arc length) is read only under mode: static; this run is | vfife/preflight.py |
| `STATIC_IGNORES_DAMPING` | WARNING | solver.damping has no effect under mode: static (Newton has no time and no velocity); | vfife/preflight.py |
| `STATIC_UNSUPPORTED` | ERROR | solver.mode = static runs frame and fibre-beam models (…); this is a … | vfife/preflight.py |
| `STUDY_UNKNOWN_FIELD` | refused | …: the case has no such field (from the study's change | vfife/api.py |
| `SWEEP_EVERY` | refused | every = …: write every k-th frame, k >= 1 | vfife/sweep.py |
| `TENDON_APPROXIMATE` | WARNING | …: … | vfife/preflight.py |
| `THERMAL_EXPANSION_PARTIAL` | WARNING | …: a temperature load acts, and … state(s) | vfife/preflight.py |
| `TRUSS_MECHANISM` | ERROR/WARNING | truss rigidity matrix has … free mechanism DOFs | vfife/preflight.py |
| `TRUSS_RANK_SKIPPED` | WARNING | exact truss mechanism rank skipped above 2000 free DOFs | vfife/preflight.py |
| `UNASSIGNED_MEMBER` | ERROR | mixed-line elements lack a member assignment | vfife/preflight.py |
| `UNCONNECTED_MEMBER` | ERROR | a Line2 member network shares no node with the solid | vfife/preflight.py |
| `UNITS_INCOMPLETE` | ERROR | units.… is missing: units.stress = … is declared, and | vfife/preflight.py |
| `UNITS_INCONSISTENT` | WARNING | units.stress = … is not units.force / units.length^2 | vfife/preflight.py |
| `UNITS_INFERRED` | INFO | units.… is not declared: … is what units.stress = | vfife/preflight.py |
| `UNITS_UNKNOWN_KEY` | ERROR | units.… is not a key `units` takes | vfife/preflight.py |
| `UNIT_DEFAULT_CONVERTED` | INFO | … is left out: its default is written in the declared units as … | vfife/preflight.py |
| `UNIT_DEFAULT_SI` | WARNING | loads … carry their mass /q//g with g = 9.80665 (SI): the declared | vfife/preflight.py |
| `UNIT_MAGNITUDE` | WARNING | …E = … and rho = … give a wave speed of … m/s | vfife/preflight.py |
| `UNKNOWN_BACKEND` | refused | unknown solver.backend … | vfife/kernels/__init__.py |
| `UNKNOWN_GROUP` | refused | … region … is not a Physical group in this mesh | vfife/cli.py |
| `UNKNOWN_LENGTH_UNIT` | INFO | units.length = … is not one this checker knows, so the | vfife/preflight.py |
| `UNKNOWN_MATERIAL` | refused | unknown material … | vfife/materials/base.py |
| `UNKNOWN_MATERIAL_PARAMETER` | refused | material … does not read | vfife/materials/base.py |
| `UNKNOWN_MONITOR` | refused | no monitor called … in this result | vfife/post/__init__.py |
| `UNKNOWN_MONITOR_FIELD` | refused | ; | vfife/api.py |
| `UNKNOWN_OUTPUT_FIELD` | refused | output.fields names | vfife/api.py |
| `UNKNOWN_REGION` | ERROR | region … is referenced but unavailable | vfife/preflight.py |
| `VERIFY_ARGUMENTS` | refused | verify(contract, result) was given contract = … and result = | vfife/api.py |
| `VIEW_CASE` | refused | --case …: … | vfife/post/view.py |
| `VIEW_CASE_MISMATCH` | refused | --case … has … elements and the result | vfife/post/view.py |
| `VIEW_EVERY` | refused | --every …: take every k-th frame, k >= 1 | vfife/post/view.py |
| `VIEW_FIELDS` | refused | --fields names | vfife/post/view.py |
| `VIEW_HIDE` | refused | --hide is not supported for a … model | vfife/post/view.py |
| `VIEW_MODEL` | refused | give one case file (the model page shows one model), or result files | vfife/cli.py, vfife/post/model_view.py |
| `VIEW_NAMES` | refused | --names gives … name(s) for … | vfife/post/view.py |
| `ZERO_DYNAMIC_MASS` | ERROR | dynamic analysis has no positive material or nodal mass | vfife/preflight.py |
| `ZERO_LENGTH` | ERROR | line elements have zero length | vfife/preflight.py |
