"""Common entry point for line-based structural elements.

The solvers keep their mechanical kernels separate, but share the same case
schema, Line2 import path, named regions, loads, amplitudes, result writer and
named DOF/assembly protocol. Mixed-member and conforming solid-line assembly
are selected here without changing their VFIFE element kinematics.
"""
from .truss_solver import TrussSolver
from .beam_solver import BeamSolver
from .fiber_beam_solver import FiberBeamSolver
from .mixed_structural_solver import MixedStructuralSolver
from .coupled_solver import CoupledSolver
from .connector_solver import ConnectorSolver
from .shell_solver import ShellSolver
from .shell_line_solver import ShellLineSolver


#: the solvers that apply `solver.mass_scaling`
MASS_SCALING = {"beam2", "beam2_timoshenko", "beam2_fiber", "mixed_line"}


def check_mass_scaling(problem):
    """Refuse `mass_scaling` where no solver would read it: a speed setting
    that is silently ignored runs at the old step, which is only slow, but a
    case that SAYS its masses were scaled when they were not is a false
    statement about the model."""
    kind = problem["element"]["type"]          # (a case dict too: tests call it with one)
    sol = problem.get("solver") or {}
    if sol.get("mass_scaling") and kind not in MASS_SCALING:
        raise ValueError(f"solver.mass_scaling is implemented for {sorted(MASS_SCALING)}, "
                         f"not for {kind!r}")
    # `target_dt` is the continuum's uniform mass scaling (and a pure-dashpot
    # connector model's step); on the line solvers it was read by nothing
    from ..elements.library import families
    if (sol.get("target_dt") and families().get(kind) not in (None, "continuum")
            and kind not in ("mixed_line", "connector2")):
        raise ValueError(f"solver.target_dt raises the step of a continuum model by scaling its "
                         f"mass; the {kind!r} solver does not read it -- use solver.mass_scaling "
                         "(selective, frame models) instead")


def _refuse_soil(problem, kind):
    """Soil springs join the frame families' internal force;
    anywhere else they would be read and ignored."""
    if problem.case.soil_springs and kind not in ("beam2", "beam2_timoshenko", "beam2_fiber", "mixed_line"):
        raise ValueError(f"soil_springs act on frame members (beam, Timoshenko, fibre, mixed line models); "
                         f"this is a {kind} model -- a pile modelled with bars has no bending for the "
                         "soil to resist")


def _refuse_jack(problem, kind):
    """`jack` is the frame models' (1.13); anywhere else it
    would be read and ignored, and the cable would carry nothing."""
    specs = list(problem.get("members") or []) + [problem.get("element") or {}]
    if any(s.get("jack") is not None for s in specs) and kind not in ("mixed_line", "connector2"):
        raise ValueError(f"`jack` (a cable stressed to a force) is for models of beams with bars and "
                         f"connectors; this is a {kind} model -- give the cable `prestress` (the stress "
                         "of that force: the length it is cut to), or model it with the beams it pulls on")


def make_structural_solver(problem, mesh, material):
    from .problem import Problem
    problem = Problem.of(problem)          # validated once, typed
    _refuse_soil(problem, problem.case.element.type)
    _refuse_jack(problem, problem.case.element.type)
    solver = _make_structural_solver(problem, mesh, material)
    from .rigid_contact import attach_contact
    attach_contact(solver, problem, mesh)
    from .member_contact import attach_member_contact
    attach_member_contact(solver, problem)
    return solver


def _make_structural_solver(problem, mesh, material):
    kind = problem.case.element.type
    check_mass_scaling(problem)
    from .creep import refuse_unclocked
    refuse_unclocked(problem, material)      # v1.0
    # EMBEDDED REINFORCEMENT IS A SOLID-HOST CAPABILITY, AND SAYING SO IS THE
    # POINT. `BondedTruss` locates its bar nodes by inverse-
    # mapping into a 3-D element (`elements/truss.locate_in_blocks`), and the
    # element library has shape functions for hexa/tet/wedge/pyramid only -- no
    # line element can be a bond host. Until v0.4 a `reinforcement` block on
    # a line model was lowered by `io/model.py` exactly as usual and then
    # silently ignored, because only `core/solver.Solver` ever reads the key.
    # A model with bars in the input and no bars in the analysis is the worst
    # shape a defect can take: it runs, it converges, and it is stiffer than
    # the thing it claims to be. `CoupledSolver` refused it for its own
    # reasons; this refuses it for all of them.
    if problem.case.reinforcement and kind != "coupled":
        raise ValueError(
            f"element {kind!r} is a line element, and embedded `reinforcement` "
            "needs a SOLID host to be located in -- it would be read, lowered "
            "and then ignored. For a line model put the steel in the section "
            "instead: a fibre section's `bars` and `tendons` (perfectly "
            "bonded), or `anchorage` for the strain-penetration share of "
            "anchorage slip. A tendon with real bond-slip along a beam does "
            "not exist yet; see docs/fiber_beam.md.")
    if kind == "truss2":
        return TrussSolver(problem, mesh, material)
    if kind in {"beam2", "beam2_timoshenko"}:
        return BeamSolver(problem, mesh, material)
    if kind == "beam2_fiber":
        return FiberBeamSolver(problem, mesh, material)
    if kind == "connector2":
        return ConnectorSolver(problem, mesh, material)
    if kind == "mixed_line":
        return MixedStructuralSolver(problem, mesh, material)
    if kind == "coupled":
        return CoupledSolver(problem, mesh, material)
    if kind == "shell4":
        return ShellSolver(problem, mesh, material)
    if kind == "shell3":
        return ShellSolver(problem, mesh, material)
    if kind == "shell_line":
        return ShellLineSolver(problem, mesh, material)
    raise ValueError(f"not a structural line element: {kind!r}")
