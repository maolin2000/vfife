"""Test support that ships with the package.

Four modules, used by `tests/test_invariants.py`,
`tests/test_material_invariants.py` and by `vfife probe`:

    models       one minimal model per registered element -- a cantilever of
                 some kind each, built in memory, written in the input form
    state        `probe(solver)`: one state vector, one internal force and one
                 external force for all nine solver families, with the
                 multiplicative perturbation the rotational dofs need
    invariants   the questions that do not depend on the element, and what a
                 failing answer means
    materials    the same one layer down: the questions that do
                 not depend on the material, asked of every registered law on a
                 strain path, and of the analyst's own material block

It ships rather than living under `tests/` because the second half of its
purpose is the user's own model: `vfife probe <case> --result <h5>` asks the
same questions at the state a run of theirs stalled in, and the material
questions on the parameters their case declares.
"""
