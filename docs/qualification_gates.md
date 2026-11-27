# Qualification gates for elements and materials

The capability registry answers "what does the implementation claim to support"; qualification answers "what evidence stands behind each claim". The two are not to be mixed.

## Elements

`verified` needs at least:

- `rigid_body`: rigid translation and rotation produce no internal force;
- `fundamental_response`: a patch test, a closed form or an equivalent basic response;
- `equilibrium`: nodal forces, reactions or global equilibrium;
- for line and mixed-dimension models also `mixed_assembly`.

`benchmarked` further needs:

- `mesh_convergence`;
- `dynamic`;
- `restart`;
- `external_reference`.

`production` also needs:

- `backend_parity` on every declared backend;
- `failure_path`: an invalid, unstable or non-finite response fails under control.

## Materials

The base gates of `verified`:

- `monotonic`;
- `step_convergence`;
- `energy`.

Conditional gates:

- a nonlinear material needs `unload_reverse`;
- a three-dimensional material needs `multiaxial`;
- a regularised material needs `mesh_objectivity`.

`benchmarked` adds `restart` and `external_reference`; `production` adds `backend_parity` on every declared backend and `failure_path`.

## Evidence

- Evidence cites a pytest node id, a verification contract or a public URL;
- a file cited must exist;
- one test may stand for several gates, but each must really be asserted by it;
- a published reference shows where a value comes from; it does not replace comparing this software's results with it;
- a GPU that merely runs is no evidence of backend parity;
- with evidence missing the status is `INCOMPLETE`; it is neither downgraded automatically nor carried over from an old description.

Verified lists exist for `hexa8`, `beam2`, `beam2_timoshenko`, `connector2`, E03 `shell4`, Chaboche, CDP and M02 `piecewise_uniaxial`. The other models are registered and have tests, but have not yet been put through the new lists; see [qualification/README.md](../qualification/README.md).
