# Model preflight

`python -m vfife check cases/case_NN_topic.yaml` imports and validates a model
without constructing a solver or initializing CUDA. It writes a structured JSON
report to `results/preflight/`. The same check runs automatically before
`info` and `run`; deterministic errors stop the command with exit code 2.

Checks currently cover connectivity bounds, repeated element nodes, duplicate
elements/nodes, orphan nodes, zero-length members, continuum Jacobian sign,
Jacobian distortion and aspect-ratio warnings, named-region resolution,
constraint suppression of observable rigid-body modes, exact small/medium
truss rigidity rank, mixed-member coverage/overlap, beam orientation and full
rotational-release warnings, and absence of physical mass in dynamics.

Warnings do not block analysis because quality thresholds and intentional end
releases require engineering judgement. An exact truss rank test is limited to
2000 free translational DOFs; larger trusses receive an explicit warning rather
than a potentially expensive dense rank calculation.
