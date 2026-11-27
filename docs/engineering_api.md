# Engineering input and application API

The YAML schema is the single analysis definition used by the CLI, Python and
a future GUI. Geometry can come from Gmsh `.msh`, Abaqus `.inp`, or a structured
box. The INP reader imports mesh information only: `*NODE`,
C3D4/C3D6/C3D8/C3D8R/C3D10, T3D2/B31, `*NSET` and `*ELSET`. Materials, loads,
constraints and steps stay in YAML so duplicate definitions cannot disagree.

```python
from vfife.api import case_schema, prepare_case, run_case

gui_form_definition = case_schema()  # standard JSON Schema
prepared = prepare_case("cases/my_model.yaml")
if prepared.preflight["status"] == "PASS":
    # Reuse the prepared object: the mesh is not imported and checked twice.
    outcome = run_case(prepared, on_progress=my_progress_callback)
    print(outcome.as_dict())
```

The API is versioned: `vfife.api.__all__` is API version 1,
and within 1.x those names keep their parameters and meaning. The reference
page `docs/api_reference.md` is generated from the docstrings
(`python -m vfife reference --api`) and a test keeps it in step. For a
parameter study use `load_case(path)` -> `CaseDraft.with_changes({"dotted.key":
value})` -> `run_case`, or `study(case, {key: [values]}, output_dir=...)`; a
draft keeps the folder its file came from, so its relative paths still
resolve, which a plain dict does not.

`prepare_case` performs validation, mesh import and preflight without creating a
solver or CUDA backend. An input mistake raises `vfife.api.UserError` (a
`ValueError` carrying a stable `code`, listed in `docs/error_codes.md`, a
`message` and a `hint`); a case that does not fit the schema raises pydantic's
`ValidationError`, also a `ValueError`. `run_case` returns an `AnalysisOutcome` rather than
exiting the process. It contains status, result/sidecar paths, diagnostics,
monitors, wall time and a failure message. A controlled numerical failure still
produces a diagnostic HDF5/checkpoint.

The CLI `run` command delegates to this same `prepare_case` / `run_case`
lifecycle. Solver construction, restart restoration, frame/checkpoint writing
and controlled numerical-failure handling therefore have one authoritative
implementation. Application front ends may use `on_ready(solver, prepared)` to
present the final solver summary after construction, plus `on_progress` and
`on_frame` for live updates.

Every result declares `result_schema_version` (3 since v1.0; layout in
docs/result_format.md), `vfife_version`, creation time and element family; its
checkpoint is a separate `<result>.ckpt.h5` (`checkpoint_format` 1) carrying a
fingerprint of the model, and a restart or initial state from another model is
refused (`CHECKPOINT_MISMATCH`). Every reader opens results through
`vfife.io.writer.open_result`, which refuses a file of a newer format by name
(`RESULT_TOO_NEW`) and reads older ones, including files without the
attribute; `tests/data/result_format_2` keeps a result of format 2 that
every later version must read. Case files declare
`version: 1`; see the user manual, chapter 4. Add `reaction_force` (and for beams, `reaction_moment`) to
`output.fields` for nodal support reactions. Energy histories remain under
`/diagnostics`; monitor reactions remain the compact choice for curves.
