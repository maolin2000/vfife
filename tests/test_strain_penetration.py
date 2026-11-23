"""Bar anchorage: the steel02 parameter, and the case-level feature built on it.

A bar does not stop straining at the member face. It keeps going for a length
Lsp into the anchorage, so the elongation a section curvature imposes is shared
between the bar inside the element and the slip outside it. Smeared over an
element of length Le, the bar sees Le/(Le + Lsp) of the section strain.

Two layers are tested here because they can fail independently:

* `steel02(strain_scale=k)` -- the law itself. Its defining property is that
  it answers at `eps` whatever an unscaled bar answers at `k * eps`, and that
  has to hold along a HISTORY, not just at one point: steel02 carries reversal
  anchors, so two laws can agree for a while and diverge on the third cycle.

* `element.strain_penetration` -- which elements get which k. Lsp comes from
  0.022 fy db, so nothing here is fitted; what can go wrong is the plumbing:
  the wrong elements picked, the scale written into the wrong parameter rows,
  or the value silently not reaching the solver at all.
"""
import pathlib

import numpy as np
import pytest

from vfife.api import prepare_case
from vfife.core.fiber_beam_solver import FiberBeamSolver
from vfife.materials import create


# --------------------------------------------------------------------------- #
#  The law
# --------------------------------------------------------------------------- #

def _steel(scale=1.0):
    return create("steel02", E=200.0e9, Fy=474.0e6, b=0.005, R0=20.0,
                  rho=7850, strain_scale=scale)


def _cyclic_strain():
    """Growing symmetric cycles, well past yield."""
    path = [0.0]
    for peak in (0.002, 0.006, 0.015, 0.030):
        for _ in range(2):
            path += list(np.linspace(path[-1], peak, 12))[1:]
            path += list(np.linspace(peak, -peak, 24))[1:]
            path += list(np.linspace(-peak, 0.0, 12))[1:]
    return np.array(path)


@pytest.mark.parametrize("scale", [1.0, 0.657, 0.4])
def test_scaled_bar_answers_at_eps_what_a_plain_bar_answers_at_k_eps(scale):
    """The defining property, walked along a history.

    This is what `strain_scale` means, so it is worth asserting directly rather
    than inferring it from a beam result: if it held only at small strain the
    beam would still look plausible while the hinge was wrong.
    """
    scaled, plain = _steel(scale), _steel()
    st_a, st_b = scaled.init_state(1), plain.init_state(1)
    ea, eb = np.zeros((1, 6)), np.zeros((1, 6))
    worst = 0.0
    for eps in _cyclic_strain():
        ea[0, 0] = eps
        eb[0, 0] = eps * scale
        a = scaled.update(ea, st_a, 1.0)[0, 0]
        b = plain.update(eb, st_b, 1.0)[0, 0]
        worst = max(worst, abs(a - b) / max(abs(b), 1.0))
    assert worst < 1e-12, f"scale={scale} drifted from k*eps by {worst:.2e}"


def test_the_default_is_a_perfectly_bonded_bar():
    """Every case written before this parameter existed must be unaffected."""
    assert _steel().strain_scale == 1.0
    assert create("steel02", E=200e9, Fy=400e6, rho=7850).numba_params()[8] == 1.0


def test_scaling_softens_the_bar_in_proportion():
    """Below yield the bar's stiffness is E*k -- the whole point of the value."""
    for scale in (1.0, 0.657, 0.4):
        mat = _steel(scale)
        strain = np.zeros((1, 6))
        strain[0, 0] = 1e-4
        sig = mat.update(strain, mat.init_state(1), 1.0)[0, 0]
        assert sig == pytest.approx(200.0e9 * 1e-4 * scale, rel=1e-12)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_a_scale_outside_its_range_is_refused(bad):
    """A scale above 1 would mean the anchorage ADDS strain to the bar."""
    with pytest.raises(ValueError, match="strain_scale"):
        create("steel02", E=200e9, Fy=400e6, rho=7850, strain_scale=bad)


def test_the_compiled_and_reference_paths_agree_with_a_scale():
    """The scale is applied in two places and they must not drift apart."""
    pytest.importorskip("numba")
    mat = _steel(0.657)
    st_fast, st_ref = mat.init_state(8), mat.init_state(8)
    strain = np.zeros((8, 6))
    worst = 0.0
    for eps in _cyclic_strain():
        strain[:, 0] = eps * (1.0 + 0.01 * np.arange(8) / 8)
        fast = mat.update(strain, st_fast, 1.0)[:, 0].copy()
        ref = mat._update_reference(strain, st_ref, np.zeros((8, 6)))[:, 0]
        worst = max(worst, float(np.max(np.abs(fast - ref)))
                    / max(float(np.max(np.abs(ref))), 1.0))
    assert worst < 1e-12, f"compiled and reference disagree by {worst:.2e}"


# --------------------------------------------------------------------------- #
#  The case-level feature
# --------------------------------------------------------------------------- #

CASE = """
name: penetration_check
units: {length: m, force: N, stress: Pa}
mesh: {type: gmsh, path: MESH}
materials:
  concrete: {type: concrete02, fpc: 25.6e6, epsc0: 0.002, fcu: 15.0e6,
             epscu: 0.02, ft: 3.1e6, Ets: 1.2e9, rat: 0.1, rho: 2400}
  steel: {type: steel02, E: 200.0e9, Fy: 474.0e6, b: 0.005, R0: 20.0, rho: 7850}
sections:
  bar20: {type: bar, material: steel, shape: general, area: 3.1416e-4}
  rc:
    type: fiber
    material: concrete
    shape: rect
    width: 0.4
    height: 0.4
    ny: 8
    nz: 2
    integration: lobatto3
    bars:
      - {y:  0.138, z:  0.138, section: bar20}
      - {y: -0.138, z:  0.138, section: bar20}
      - {y:  0.138, z: -0.138, section: bar20}
      - {y: -0.138, z: -0.138, section: bar20}
assign:
  - {region: all, section: rc PENETRATION}
regions:
  base: {box: [-0.01, 0.01, -0.01, 0.01, -0.01, 0.01]}
  head: {box: [1.59, 1.61, -0.01, 0.01, -0.01, 0.01]}
constraints:
  - {region: base, dofs: [1, 1, 1, 1, 1, 1]}
steps:
  - name: push
    constraints:
      - {region: head, dofs: [0, 0, 1, 0, 0, 0], value: 0.01}
solver: {backend: numpy, mode: relaxation, load_steps: 2, tolerance: 1.0e-3, n_steps: 50}
output:
  path: out.h5
  monitors:
    tip: {kind: displacement, region: head, dof: z}
"""


def _case(tmpdir, penetration=True, region="base", db=0.020, n_elem=4):
    d = pathlib.Path(tmpdir)
    d.mkdir(parents=True, exist_ok=True)
    msh = d / "col.msh"
    nodes = "\n".join(f"{i+1} {i*1.6/n_elem:.10g} 0 0" for i in range(n_elem + 1))
    elems = "\n".join(f"{e+1} 1 2 1 1 {e+1} {e+2}" for e in range(n_elem))
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n"
                   f"{n_elem+1}\n{nodes}\n$EndNodes\n$Elements\n{n_elem}\n{elems}\n"
                   "$EndElements\n", encoding="utf-8")
    block = (f", anchorage: {{region: {region}, bar_diameter: {db}}}"
             if penetration else "")
    text = CASE.replace("MESH", str(msh)).replace("PENETRATION", block)
    p = d / "case.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def _solver(path):
    prep = prepare_case(path)
    return FiberBeamSolver(prep.case.model_dump(mode="python"),
                           prep.mesh, prep.material)


def test_lsp_comes_from_fy_and_db_and_nothing_else(tmp_path):
    """0.022 fy db, both off the material and the case -- no fitted length."""
    sol = _solver(_case(tmp_path))
    lsp = sol.strain_penetration["Lsp"]
    assert lsp == pytest.approx(0.022 * 474.0 * 20.0 / 1e3, rel=1e-12)
    le = 1.6 / 4
    assert sol.strain_penetration["scale"] == pytest.approx(le / (le + lsp), rel=1e-12)


def test_only_the_elements_touching_the_region_are_scaled(tmp_path):
    """A hinge at the footing is not a softer column everywhere."""
    sol = _solver(_case(tmp_path))
    scales = sol._spar[:, 0, 8]
    assert sol.strain_penetration["elements"] == [0]
    assert scales[0] < 0.7, f"the base element was not scaled: {scales}"
    assert np.allclose(scales[1:], 1.0), f"an interior element was scaled: {scales}"


def test_a_longer_element_dilutes_the_same_anchorage(tmp_path):
    """Le/(Le + Lsp): the same Lsp matters less the longer the element is."""
    coarse = _solver(_case(tmp_path / "a", n_elem=2))     # 800 mm elements
    fine = _solver(_case(tmp_path / "b", n_elem=8))       # 200 mm elements
    assert coarse._spar[0, 0, 8] > fine._spar[0, 0, 8]


def test_without_the_block_every_bar_is_perfectly_bonded(tmp_path):
    sol = _solver(_case(tmp_path, penetration=False))
    assert sol.strain_penetration is None
    assert np.allclose(sol._spar[:, :, 8], 1.0)


# NOTE: `_setup_strain_penetration` also refuses a region that touches no
# element. There is no test for it because there is no way to reach it: the
# region system rejects an empty region first, with a better message than the
# one that guard would give, and in a line mesh every node that exists belongs
# to an element. The guard stays as a defence for a mesh carrying stray nodes,
# and is marked here as untested rather than left to look covered.


def test_penetration_softens_before_yield_and_not_after(tmp_path):
    """The mechanism it is meant to add, and the one it must not.

    Slip makes the hinge softer and delays yield; it does not reduce what a
    fully plastic section can carry. If the base moment fell at large drift too,
    the model would be losing capacity for a reason the physics does not have.
    """
    with_sp = _solver(_case(tmp_path / "a"))
    without = _solver(_case(tmp_path / "b", penetration=False))
    x = with_sp.mesh.nodes[:, 0]
    length = x.max()

    def base_moment(sol, tip):
        sol.q[:] = 0.0
        for n in range(sol.nn):
            sol.q[6 * n + 2] = tip * (x[n] / length) ** 2
            sol.q[6 * n + 4] = -2 * tip * x[n] / length ** 2
        return sol.internal_force()[1][0, 5]

    small = base_moment(with_sp, 0.004) / base_moment(without, 0.004)
    large = base_moment(with_sp, 0.060) / base_moment(without, 0.060)
    assert small < 0.97, f"no softening before yield (ratio {small:.3f})"
    assert large > 0.97, f"capacity lost at large drift (ratio {large:.3f})"


def test_the_kernel_and_the_reference_agree_with_penetration_on(tmp_path):
    """Per-element parameters exist in two places; they must say the same thing.

    The kernel reads `spar[e, f]`; the NumPy reference reads the same rows
    through `update(..., params=)`. Before that argument existed the reference
    could not express a per-element parameter at all, so it would have been
    modelling a different structure while claiming to be the reference.
    """
    pytest.importorskip("numba")
    path = _case(tmp_path)
    fast, ref = _solver(path), _solver(path)
    ref._kernel = None
    assert fast._kernel is not None, fast.kernel_status

    rng = np.random.default_rng(31)
    worst = 0.0
    for _ in range(25):
        dq = rng.normal(0.0, 3e-4, fast.ndof)
        fast.q[:] += dq
        ref.q[:] += dq
        f1, _ = fast.internal_force()
        f2, _ = ref.internal_force()
        worst = max(worst, float(np.max(np.abs(f1 - f2)))
                    / max(float(np.max(np.abs(f2))), 1.0))
    assert worst < 1e-10, f"kernel and reference disagree by {worst:.2e}"
