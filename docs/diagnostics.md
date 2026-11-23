# Analysis termination and diagnostics

Every `vfife run` result now carries a machine-readable termination contract.
The root HDF5 attributes `analysis_status` and `analysis_converged` distinguish a
valid result from a file that was written only to preserve failure evidence.

| status | meaning | usable as a completed result |
|---|---|---|
| `CONVERGED` | quasi-static residual and kinetic criteria were satisfied | yes |
| `COMPLETED_DYNAMIC` | the requested physical time interval was completed | yes, as a dynamic history |
| `MAX_STEPS` | an iteration or load-level budget was exhausted | no |
| `NONFINITE` | a force, state, stress, displacement, velocity, or diagnostic became NaN/Inf | no |
| `UNSTABLE` | reserved for a detected numerical/structural instability | no |
| `INTERRUPTED` | stopped on purpose (Ctrl-C, SIGTERM, `on_progress` returning False); the frames due before the stop and a checkpoint at a step boundary are kept, `--restart` resumes | no (a partial run) |
| `PREFLIGHT_FAILED` | `api.run_case` only: preflight found an error, nothing was solved or written (`result_path` is None); `vfife run` exits 2 instead | no |

The command line's exit codes and every refusal and preflight code are in docs/error_codes.md.

`vfife run` exits with code 3 for a controlled analysis failure. It still closes
the HDF5/XDMF pair and stores the checkpoint (beside the result, `<result>.ckpt.h5`), monitors, residual history and the
first identified abnormal location. A `MAX_STEPS` report includes the load stage
and the node/free degree of freedom with the largest residual on CPU paths. On a
GPU-resident path, this location is explicitly marked unavailable because finding
it would require transferring the full residual field to the host.

Each HDF5 frame also has an `equilibrium` attribute. It is `true` only for a
converged quasi-static state, `false` for an intermediate relaxation frame, and
`not_applicable` for physical-time dynamic frames. ParaView does not use this
attribute for rendering, but scripts can use it to prevent accidental extraction
of a force-displacement curve from non-equilibrium frames.

## Energy audit

CPU solvers integrate external and internal work by the trapezoidal rule and
record kinetic energy, kinetic-damping removal, absolute imbalance and normalized
imbalance in `/diagnostics`. These values are an audit signal, not an additional
equilibrium criterion. The GPU-resident solid path marks this audit unavailable:
copying full nodal force and displacement fields every iteration would defeat the
resident execution model. Scalar convergence and kinetic diagnostics remain
available there.

`vfife verify` rejects any HDF5 result whose `analysis_converged` flag is false,
so a failed partial file cannot accidentally pass a benchmark contract.

