"""Features attached to a solver as components.

Contact between members and contact with rigid surfaces add a force to the
solver's internal force, history to commit after a step, fields to the
output, and state to the checkpoint. They used to do it by replacing the
solver's methods on the instance (`solver._internal_force = wrapped`, 1.17), and
the compiled step had to inspect what it found there to tell a known
wrapper from a script's (the architecture review of 1.17).

A component is now an object the solver holds in `components`, in the
order attached, and the solver's own methods call it:

    line_force(solver, out) -> out   after the members' forces of a solver on
                                     dof vectors (`_member_forces`)
    bar_force(solver, f) -> f        after the bars' forces of the truss solver
                                     (`_bar_forces`, f is (nn, 3))
    commit()                         after the solver commits a step's history
    fields(out)                      adds its output fields to `nodal_fields`
    checkpoint(ck), restore(ck)      its state in the solver's checkpoint

A component implements the ones it needs; the solver calls only those it
has. Each is applied after the solver's own part and after the components
attached before it -- the order the nested wrappers ran in, so the answers
are the same to the bit.
"""


from .step_state import NO_STAGING


class HasComponents:
    """The solver side: `components` and the public methods that run them.
    A class attribute so a solver whose `__init__` does not reach a base
    class's still has it (the trap `BeamSolver._half` documents)."""

    components: tuple = ()
    #: the analysis plan (core/analysis_steps.Plan) and the state the steps
    #: keep (core/step_state): class defaults for a solver
    #: whose steps change nothing, so no reader has to ask whether they exist
    plan = None
    staging = NO_STAGING

    def attach_component(self, component) -> None:
        self.components = tuple(self.components) + (component,)

    def _each(self, hook):
        for c in self.components:
            fn = getattr(c, hook, None)
            if fn is not None:
                yield fn

    def nodal_fields(self, *a, **kw):
        out = self._nodal_fields(*a, **kw)
        for fn in self._each("fields"):
            fn(out)
        return out

    def checkpoint(self, *a, **kw):
        ck = self._checkpoint(*a, **kw)
        for fn in self._each("checkpoint"):
            fn(ck)
        return ck

    def restore(self, ck, *a, **kw):
        out = self._restore(ck, *a, **kw)
        for fn in self._each("restore"):
            fn(ck)
        return out
