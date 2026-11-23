"""The capability table must say what the solvers do.

`vfife capabilities` and preflight's CAPABILITY_MISMATCH read the element
table in elements/library.py; the solvers read nothing of the kind and refuse
or accept on their own. Two entries had drifted: `truss2` declared three
materials while TrussSolver takes every uniaxial law, so preflight refused a
steel02 bar the solver would have run; and `beam2_fiber` declared numpy only
while its solver had been dispatching to a numba kernel since v0.1 -- a user
who wrote `backend: numba` to ask for exactly that was told no.

The rule these tests enforce is one-directional and simple: EVERY declared
(element, material) and (element, backend) pair must build a solver. The
reverse -- everything undeclared is refused -- is asserted only where a
solver actually checks, because an over-declaration promises something that
fails loudly, while an under-declaration hides something that works.
"""
import pytest

from vfife import cli
from vfife.elements import element_definition
from vfife.testing import models

numba = pytest.importorskip("numba", reason="the numba backend is what is being aligned")


def _solver(element, **kw):
    m = models.build(element, **kw)
    return cli.make_solver(m.problem, m.mesh, m.material)


# --------------------------------------------------------------------------- #
#  truss2: every declared material builds AND runs
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("material", element_definition("truss2").compatible_materials)
def test_every_material_truss2_declares_is_one_its_solver_takes(material):
    from vfife.core.diagnostics import ConvergenceFailure
    s = _solver("truss2", material=material, solver={"n_steps": 20, "max_steps_per_load": 20})
    try:
        s.run(output_every=10 ** 9, progress_every=10 ** 9)   # a few steps: enough to hit the law
    except ConvergenceFailure:
        pass                                                   # 20 iterations is not a solve


def test_truss2_declares_every_uniaxial_law_the_solver_would_accept():
    """The solver asks `getattr(material, "uniaxial", False)`; the table must
    list every registered law that answers yes, or preflight refuses it."""
    from vfife.materials import available, definition
    uniaxial = {n for n in available() if definition(n).kinematics == ("uniaxial",)}
    declared = set(element_definition("truss2").compatible_materials)
    assert uniaxial <= declared, sorted(uniaxial - declared)


# --------------------------------------------------------------------------- #
#  beam2_fiber: numpy promotes to the kernel, numba requires it
# --------------------------------------------------------------------------- #

def test_beam2_fiber_declares_the_backends_its_solver_accepts():
    assert set(element_definition("beam2_fiber").backends) == {"numpy", "numba"}


#: the minimal fibre model is linear_elastic, which has no scalar twin and so
#: no kernel; the kernel questions are asked on a law that has one
FIBRE_LAW = "steel02"


def test_backend_numba_on_a_qualifying_fibre_section_gets_the_kernel():
    s = _solver("beam2_fiber", backend="numba", material=FIBRE_LAW)
    assert s._kernel is not None, s.kernel_status
    assert s.backend == "numba"


def test_backend_numpy_still_promotes_to_the_kernel():
    """Unchanged behaviour, now stated: numpy is the readable path plus the
    promotion, which is what every shipped fibre case relies on."""
    s = _solver("beam2_fiber", backend="numpy", material=FIBRE_LAW)
    assert s._kernel is not None, s.kernel_status


def test_backend_numba_is_refused_by_name_when_the_kernel_cannot_be_built(monkeypatch):
    """Asked for the kernel and it is not there: say so, in the kernel's own
    words, instead of running the reference path a hundred times slower."""
    from vfife.core import fiber_beam_solver as fbs

    def declined(self):
        self._kernel = None
        self.kernel_status = "rebars use more than one material type; NumPy path"

    monkeypatch.setattr(fbs.FiberBeamSolver, "_setup_kernel", declined)
    with pytest.raises(ValueError, match="rebars use more than one material type"):
        _solver("beam2_fiber", backend="numba")
    # and numpy accepts the same model on the reference path
    s = _solver("beam2_fiber", backend="numpy")
    assert s._kernel is None


def test_backend_numba_on_a_law_without_a_scalar_twin_is_refused_by_name():
    """linear_elastic has no compiled twin, so a fibre model of it has no
    kernel; `backend: numba` must say that rather than run the slow path."""
    with pytest.raises(ValueError, match="no scalar twin"):
        _solver("beam2_fiber", backend="numba", material="linear_elastic")


def test_beam2_fiber_refuses_warp_by_name():
    with pytest.raises(ValueError, match="numpy .* or numba"):
        _solver("beam2_fiber", backend="warp")


# --------------------------------------------------------------------------- #
#  every line element: every declared backend builds
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("element", ["truss2", "beam2", "beam2_timoshenko", "beam2_fiber",
                                     "connector2", "mixed_line", "shell4", "shell3",
                                     "shell_line", "coupled"])
def test_every_declared_cpu_backend_builds_a_solver(element):
    for backend in element_definition(element).backends:
        if backend == "warp":
            continue                    # needs hardware the CI does not have
        _solver(element, backend=backend,
                **({"material": FIBRE_LAW} if element == "beam2_fiber" else {}))


# --------------------------------------------------------------------------- #
#  embedded reinforcement is a solid-host capability, and says so
# --------------------------------------------------------------------------- #

#: one bar in the internal form (a solver's case is validated, v1.19:
#: the block these tests wrote was not one)
BARS = [{"name": "main", "area": 2e-4, "diameter": 0.016,
         "material": {"type": "rebar", "E": 2e11, "rho": 7850.0, "yield_strength": 4.5e8},
         "lines": [{"start": [0.0, 0.0, 0.0], "end": [1.0, 0.0, 0.0], "segment_length": 0.25}]}]


@pytest.mark.parametrize("element", ["beam2_fiber", "beam2", "truss2"])
def test_reinforcement_on_a_line_model_is_refused_rather_than_ignored(element):
    """A `reinforcement` block needs a SOLID host: `BondedTruss` finds its bar
    nodes by inverse-mapping into a 3-D element, and no line element has shape
    functions to be mapped into. Before v0.4 the block was lowered exactly as
    usual and then silently dropped, because only the continuum solver reads
    the key -- a model with bars in the input, no bars in the analysis, and no
    complaint. That is the worst shape a defect can take: it runs, it
    converges, and it is stiffer than the thing it claims to be.
    """
    m = models.build(element)
    m.problem["reinforcement"] = BARS
    with pytest.raises(ValueError, match="needs a SOLID host"):
        cli.make_solver(m.problem, m.mesh, m.material)


def test_the_refusal_names_what_to_use_instead():
    m = models.build("beam2_fiber")
    m.problem["reinforcement"] = BARS
    with pytest.raises(ValueError) as e:
        cli.make_solver(m.problem, m.mesh, m.material)
    for word in ("bars", "tendons", "anchorage"):
        assert word in str(e.value), f"the message does not point at `{word}`"


# --------------------------------------------------------------------------- #
#  the MATERIAL half of the same table
# --------------------------------------------------------------------------- #
#  v0.1 fixed the element half: `beam2_fiber` declares numpy AND numba, and
#  the tests above pin it. They all build the solver DIRECTLY -- which is why
#  they never saw that `compatibility()` checks the material's backends too,
#  and that four uniaxial laws still declared numpy only. So `backend: numba`
#  on the project's own flagship case was refused by PREFLIGHT, by name:
#
#      [ERROR] CAPABILITY_MISMATCH: concrete02 does not support backend numba
#
#  while `backend: numpy` on the same case was silently promoted to the very
#  kernel the message said did not exist. Same untruth as v0.1, one level
#  down, and it survived because every test of it skipped the front door.
UNIAXIAL_WITH_A_TWIN = ("concrete01", "concrete02", "steel02", "steel_cyclic")


@pytest.mark.parametrize("material", UNIAXIAL_WITH_A_TWIN)
def test_a_law_with_a_scalar_twin_declares_numba(material):
    from vfife.materials import definition
    assert "numba" in definition(material).backends


@pytest.mark.parametrize("material", UNIAXIAL_WITH_A_TWIN)
def test_and_the_solver_really_gives_it_the_kernel(material):
    """The declaration is only worth something if it is the solver's own
    answer. Each of these four has `numba_uniaxial_fn` and `numba_params`,
    which is exactly what the fused kernel inlines."""
    s = _solver("beam2_fiber", backend="numba", material=material)
    assert s._kernel is not None, s.kernel_status
    assert "numba fused kernel" in s.kernel_status


@pytest.mark.parametrize("material", UNIAXIAL_WITH_A_TWIN)
def test_preflight_lets_backend_numba_through_for_it(material):
    """THE CHECK THAT WAS MISSING. Everything above builds the solver
    directly; the refusal lived in `compatibility()`, which is the door a
    case file goes through."""
    from vfife.catalog import compatibility
    ok, reasons = compatibility("beam2_fiber", material, backend="numba", mode="dynamic")
    assert ok, reasons


def test_a_law_without_a_twin_is_still_numpy_only_and_says_which_one():
    """The other direction, and it must stay true: `linear_elastic` and
    `rebar` declare numba because the SOLID kernels inline their
    `numba_point_fn`, but neither has a uniaxial twin, so a fibre model of
    them has no kernel. The status line has to name the law -- "a material"
    sent someone reading a section with four of them in it."""
    from vfife.materials import definition
    for name in ("orthotropic", "piecewise_uniaxial", "viscoelastic"):
        assert definition(name).backends == ("numpy",)
    s = _solver("beam2_fiber", backend="numpy", material="linear_elastic")
    assert s._kernel is None
    assert "linear_elastic has no scalar twin" in s.kernel_status
