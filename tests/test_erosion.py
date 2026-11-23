"""Element deletion: the criteria, the kill switch, and what it leaves behind.

Erosion is what both commercial codes do with an element that has run out of
stiffness -- Abaqus `*SECTION CONTROLS, ELEMENT DELETION=YES`, LS-DYNA
`*MAT_CSCM` ERODE ("elements erode when damage exceeds 0.99 and the maximum
principal strain exceeds ERODE-1") -- and it is the only mechanism either offers
that removes a local mechanism rather than damping it.

The properties worth pinning are the ones that would let a wrong implementation
still look plausible:

  * an eroded element contributes *identically* zero, not merely a little;
  * its neighbours are unchanged bit for bit, so erosion is not a global
    perturbation dressed up as a local one;
  * erosion is one-way and idempotent;
  * the criterion-free volume net fires on its own;
  * a driven node is never frozen, however unsupported it becomes.
"""

import numpy as np
import pytest

from vfife.core.mesh import box_mesh
from vfife.core.solver import Solver
from vfife.elements import continuum
from vfife.materials import create

FORMULATIONS = ["full", "reduced", "b-bar"]


def setup(formulation):
    mesh = box_mesh((1.0, 0.4, 0.4), (4, 2, 2))
    mat = create("linear_elastic", E=3e10, nu=0.2, rho=2400.0)
    pre = continuum.precompute(mesh.nodes, mesh.elements,
                               formulation=formulation, material=mat)
    x0 = mesh.nodes[mesh.elements]
    x0_e = np.ascontiguousarray(x0 - x0.mean(axis=1, keepdims=True))
    u = np.zeros_like(mesh.nodes)
    u[:, 0] = 5e-4 * mesh.nodes[:, 0]
    u[:, 2] = 3e-4 * mesh.nodes[:, 1]
    return mesh, pre, mat, x0_e, u


def force(pre, mat, x0_e, u_e):
    state = mat.init_state(pre.n_elem * pre.n_gp)
    quat = np.zeros((pre.n_elem, 4))
    quat[:, 0] = 1.0
    f, _ = continuum.internal_force(pre, x0_e, u_e, quat, mat, state, 1.0,
                                    rot_iters=30)
    return f


@pytest.mark.parametrize("formulation", FORMULATIONS)
def test_eroded_elements_contribute_exactly_zero(formulation):
    """Zero, not small. A residual force from a dead element is a dead element."""
    mesh, pre, mat, x0_e, u = setup(formulation)
    u_e = u[mesh.elements]
    before = force(pre, mat, x0_e, u_e)

    dead = np.zeros(pre.n_elem, dtype=bool)
    dead[[1, 5]] = True
    assert continuum.erode(pre, dead) == 2

    after = force(pre, mat, x0_e, u_e)
    assert np.abs(after[dead]).max() == 0.0
    # and the survivors are bit-identical: erosion is local
    assert np.array_equal(after[~dead], before[~dead])
    assert pre.active[dead].sum() == 0
    assert pre.active[~dead].min() == 1


@pytest.mark.parametrize("formulation", FORMULATIONS)
def test_erosion_is_one_way_and_idempotent(formulation):
    mesh, pre, mat, x0_e, u = setup(formulation)
    dead = np.zeros(pre.n_elem, dtype=bool)
    dead[[0, 3]] = True
    assert continuum.erode(pre, dead) == 2
    # asking again reports nothing new
    assert continuum.erode(pre, dead) == 0
    # and a superset only counts the newcomers
    more = dead.copy()
    more[7] = True
    assert continuum.erode(pre, more) == 1


def test_strain_measures_match_a_known_deformation():
    """Both criteria read one deformation gradient; check it against algebra."""
    mesh, pre, mat, x0_e, u = setup("b-bar")
    a, b, c = 0.10, -0.20, 0.05
    v = np.zeros_like(mesh.nodes)
    for i, k in enumerate((a, b, c)):
        v[:, i] = k * mesh.nodes[:, i]
    J, eps1, bad = continuum.strain_measures(pre, v[mesh.elements])
    assert np.allclose(J, (1 + a) * (1 + b) * (1 + c))
    # max principal Green-Lagrange strain of a uniform stretch
    assert np.allclose(eps1, 0.5 * ((1 + a) ** 2 - 1))
    assert not bad.any()


def test_the_safety_net_survives_a_non_finite_element():
    """It has to outlive the thing it catches.

    `eigvalsh` does not return from an inf: it raises "Eigenvalues did not
    converge" and takes the run down -- which is a worse outcome than the
    divergence the net was called to stop. Measured on the RC beam: the run
    crashed in the erosion check instead of eroding.
    """
    mesh, pre, mat, x0_e, u = setup("b-bar")
    u_e = u[mesh.elements].copy()
    u_e[2, 3, 1] = np.inf
    u_e[5, 0, 0] = np.nan
    J, eps1, bad = continuum.strain_measures(pre, u_e)
    assert np.flatnonzero(bad).tolist() == [2, 5]
    assert np.isfinite(J).all() and np.isfinite(eps1).all()


def test_volume_ratio_detects_collapse_and_inversion():
    """The criterion-free safety net, on a deformation with a known Jacobian."""
    mesh, pre, mat, x0_e, u = setup("b-bar")
    # uniform stretch: det(F) = (1+a)(1+b)(1+c), exactly
    a, b, c = 0.10, -0.20, 0.05
    v = np.zeros_like(mesh.nodes)
    v[:, 0] = a * mesh.nodes[:, 0]
    v[:, 1] = b * mesh.nodes[:, 1]
    v[:, 2] = c * mesh.nodes[:, 2]
    J = continuum.volume_ratio(pre, v[mesh.elements])
    assert np.allclose(J, (1 + a) * (1 + b) * (1 + c))

    # squash flat past inversion
    w = np.zeros_like(mesh.nodes)
    w[:, 2] = -1.5 * mesh.nodes[:, 2]
    assert (continuum.volume_ratio(pre, w[mesh.elements]) < 0).all()


# --------------------------------------------------------------------------- #


CDP = dict(
    type="cdp", E=3.0e10, nu=0.2, rho=2400.0, l_ref=0.1,
    crack_band="tension", viscosity=0.0,
    compression={"source": "gb50010", "fc": 30.0e6},
    tension={"source": "fracture_energy", "ft": 3.0e6, "Gf": 40.0},
)


def bar_case(erosion, log=None):
    """A concrete bar pulled to 2% strain: it is finished long before that."""
    prob = {
        "element": {"type": "hexa8", "formulation": "b-bar"},
        "regions": {
            "x0": {"box": [-1e-3, 1e-3, -1, 1, -1, 1]},
            "xL": {"box": [0.999, 1.001, -1, 1, -1, 1]},
        },
        "constraints": [
            {"region": "x0", "dofs": [1, 1, 1]},
            {"region": "xL", "dofs": [1, 0, 0], "value": [0.020, 0.0, 0.0]},
        ],
        "solver": {"mode": "relaxation", "mass": "underwood", "backend": "numpy",
                   "n_steps": 60000, "load_steps": 12,
                   "max_steps_per_load": 3000, "tolerance": 1e-5,
                   "damping": {"type": "kinetic"}, "erosion": erosion},
    }
    mesh = box_mesh((1.0, 0.1, 0.1), (10, 1, 1))
    m = dict(CDP)
    mat = create(m.pop("type"), **m)
    s = Solver(prob, mesh, mat, log=(log.append if log is not None else (lambda *_: None)))
    s.run(progress_every=200, output_every=10**9)
    f_e, _ = s._force(s.u, s.quat, s.state, s.dt, rot_iters=30)
    R = -s._force.assemble(f_e)[mesh.regions["x0"], 0].sum()
    return s, abs(R)


def test_erosion_severs_a_fully_cracked_bar():
    """Without erosion the broken bar still carries its residual stiffness."""
    s_off, R_off = bar_case(None)
    s_on, R_on = bar_case({"damage": 0.9, "strain": 1e-3, "points": 4,
                           "every": 200})

    assert s_off.state[:, 14].max() == pytest.approx(0.99, abs=1e-6)
    assert s_off.n_eroded == 0
    assert R_off > 10.0                        # ~90 N of residual stiffness

    assert s_on.n_eroded >= 1
    assert R_on < 1e-6 * R_off                 # severed: it carries nothing
    # and the run is far better converged once the dead element is gone
    assert s_on.residuals[-1] < 0.01 * s_off.residuals[-1]


def test_a_driven_node_is_never_frozen():
    """Freezing the loaded end would cancel the loading and look like a result.

    Eroding the last element in the bar leaves the prescribed face unsupported.
    Freezing it held the displacement at 0.00056 against a prescribed 0.020 --
    a quiet, plausible, completely wrong answer.
    """
    s, _ = bar_case({"damage": 0.9, "strain": 1e-3, "points": 4, "every": 200})
    driven = s._driven
    assert driven.any()
    assert not (s.frozen & driven).any()
    assert s.u[driven, 0].max() == pytest.approx(0.020, rel=1e-9)


def test_erosion_on_a_material_that_cannot_fail_is_an_error():
    """Silently never eroding is worse than refusing."""
    mesh = box_mesh((1.0, 0.2, 0.2), (2, 1, 1))
    mat = create("linear_elastic", E=3e10, nu=0.2, rho=2400.0)
    prob = {
        "element": {"type": "hexa8"},
        "regions": {"x0": {"box": [-1e-3, 1e-3, -1, 1, -1, 1]}},
        "constraints": [{"region": "x0", "dofs": [1, 1, 1]}],
        "solver": {"mode": "relaxation", "n_steps": 10,
                   "erosion": {"damage": 0.99, "strain": 0.05, "every": 100}},
    }
    with pytest.raises(ValueError, match="no notion of failure"):
        Solver(prob, mesh, mat, log=lambda *_: None)


def test_relaxation_defaults_to_underwood_mass_through_the_schema():
    """The schema emits `mass: None`, which used to defeat get()'s default.

    A relaxation case that did not name its mass ran with PHYSICAL mass and a
    CFL time step -- real dynamics, with n_steps spent as if they were
    relaxation iterations, and nothing anywhere said so.
    """
    import yaml

    from vfife.cli import build
    from vfife.io.schema import Case

    case = Case(**yaml.safe_load("""
        name: t
        mesh: {dimensions: [1.0, 0.2, 0.2], divisions: [2, 1, 1]}
        material: {type: linear_elastic, E: 3.0e10, nu: 0.2, rho: 2400.0}
        regions: {x0: {box: [-0.001, 0.001, -1, 1, -1, 1]}}
        constraints: [{region: x0, dofs: [1, 1, 1]}]
        solver: {mode: relaxation, n_steps: 10}
    """))
    assert case.as_dict()["solver"]["mass"] is None      # the trap
    mesh, mat, _element = build(case)
    s = Solver(case.as_dict(), mesh, mat, log=lambda *_: None)
    assert s.mass_kind == "underwood"
    assert s.dt == 1.0


# --------------------------------------------------------------------------- #
#  Erosion is a change to the model, so it says so
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_erosion_says_what_it_removed_and_where():
    """It used to remove elements in silence.

    On case_06 an erosion block that looked inert -- the last frame's worst
    damage was 0.88, under its 0.98 threshold, because damage FALLS again
    when a crack closes -- had quietly deleted four elements at the loaded
    head, and moved the peak shear of the whole column by 2%. The block was
    read as doing nothing because nothing said otherwise: `check_erosion`
    counted into `n_eroded` and no line of output ever mentioned it, in the
    log, in the run summary or in the diagnostics.

    Deleting part of a model is not an iteration detail. All three now say
    it: the log as it happens with a location, the CLI summary at the end,
    and the diagnostics record that goes into the result file.
    """
    log = []
    s, _R = bar_case({"damage": 0.9, "strain": 0.002, "points": 4, "every": 200}, log=log)
    assert s.n_eroded > 0
    lines = [ln for ln in log if "eroded" in ln]
    assert lines, f"{s.n_eroded} elements were removed and nothing said so: {log}"
    assert "hexa8" in lines[0] and " m" in lines[0]       # what, and where
    assert s.diagnostics["eroded"] == s.n_eroded


def test_a_run_with_no_erosion_configured_reports_none():
    """`eroded` is in every record, so a reader never has to wonder whether
    the key is missing because nothing eroded or because nobody looked."""
    s, _R = bar_case(None)
    assert s.n_eroded == 0 and s.diagnostics["eroded"] == 0
