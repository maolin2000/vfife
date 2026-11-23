"""The compiled uniaxial point loops must reproduce their reference exactly.

`steel02` and `concrete02` each keep a readable Python loop as the definition
of the law (`_update_reference`) and run a compiled twin in `update()`. These
tests pin the twin to the definition.

They walk a full cyclic strain history rather than calling once. Both laws are
path dependent -- steel02 carries the Giuffre-Menegotto-Pinto reversal anchors,
concrete02 carries the minimum strain reached and the tensile offset -- so two
implementations can agree on any single call and still diverge on the third
reversal. Agreement is asserted on the returned stress AND on the state array
at every step, because the state is what the next step reads.
"""
import numpy as np
import pytest

from vfife.materials import create
import vfife.materials.concrete02 as concrete02_module
import vfife.materials.steel02 as steel02_module


def _steel_path():
    """Growing symmetric cycles, well past yield, ten reversals."""
    path = [0.0]
    for peak in (0.0015, 0.004, 0.010, 0.020, 0.035):
        for _ in range(2):
            path += list(np.linspace(path[-1], peak, 20))[1:]
            path += list(np.linspace(peak, -peak, 40))[1:]
            path += list(np.linspace(-peak, 0.0, 20))[1:]
    return np.array(path)


def _concrete_path():
    """Compression into crushing, unload through zero into tension, repeat."""
    path = [0.0]
    for peak in (-0.0005, -0.002, -0.006, -0.015):
        for _ in range(2):
            path += list(np.linspace(path[-1], peak, 25))[1:]
            path += list(np.linspace(peak, abs(peak) * 0.5, 40))[1:]
            path += list(np.linspace(abs(peak) * 0.5, 0.0, 15))[1:]
    return np.array(path)


def _walk(material, path, n_points, spread):
    """Run both implementations along `path`, return worst relative deviations."""
    state_fast = material.init_state(n_points)
    state_ref = material.init_state(n_points)
    strain = np.zeros((n_points, 6))
    worst_stress = 0.0
    worst_state = 0.0
    for eps in path:
        # a spread across points so every fibre sits at a slightly different
        # place on the envelope, instead of testing one scalar n times
        strain[:, 0] = eps * (1.0 + spread * np.arange(n_points) / n_points)
        fast = material.update(strain, state_fast, 1.0)[:, 0].copy()
        ref = material._update_reference(
            strain, state_ref, np.zeros((n_points, 6)))[:, 0].copy()
        scale = max(float(np.max(np.abs(ref))), 1.0)
        worst_stress = max(worst_stress, float(np.max(np.abs(fast - ref))) / scale)
        s_scale = max(float(np.max(np.abs(state_ref))), 1e-30)
        worst_state = max(worst_state,
                          float(np.max(np.abs(state_fast - state_ref))) / s_scale)
    return worst_stress, worst_state


def test_steel02_compiled_loop_matches_its_reference():
    material = create("steel02", E=200.0e9, Fy=474.0e6, b=0.012, R0=20.0, rho=7850)
    stress, state = _walk(material, _steel_path(), 64, spread=0.01)
    assert stress < 1e-12, f"stress drifted from the reference by {stress:.2e}"
    assert state < 1e-12, f"state drifted from the reference by {state:.2e}"


def test_concrete02_compiled_loop_matches_its_reference():
    material = create("concrete02", fpc=25.6e6, epsc0=0.002, fcu=15.0e6,
                      epscu=0.02, ft=3.1e6, Ets=1.2e9, rat=0.1, rho=2400)
    stress, state = _walk(material, _concrete_path(), 48, spread=0.02)
    assert stress < 1e-12, f"stress drifted from the reference by {stress:.2e}"
    assert state < 1e-12, f"state drifted from the reference by {state:.2e}"


@pytest.mark.parametrize("module", [steel02_module, concrete02_module])
def test_the_compiled_path_is_the_one_actually_running(module):
    """Guard against the fallback quietly becoming the default.

    If numba stops importing, or the kernel raises during compilation, update()
    falls back to the Python loop and every result stays correct -- only ten
    times slower, with nothing to say so. Where numba is installed, assert the
    fast path exists, so a silent regression to the reference is a red test
    rather than a mystery about why a run took all afternoon.
    """
    pytest.importorskip("numba")
    assert module._kernel() is not None, (
        f"{module.__name__} fell back to the Python reference loop even though "
        "numba imports; the run will be ~10x slower than it should be")


def test_the_reference_loop_still_works_on_its_own():
    """The fallback has to stay correct, not merely present.

    It is what runs wherever numba is unavailable, and nothing else exercises
    it once the compiled path is the default.
    """
    material = create("steel02", E=200.0e9, Fy=474.0e6, b=0.012, R0=20.0, rho=7850)
    state = material.init_state(4)
    strain = np.zeros((4, 6))
    strain[:, 0] = 0.001                      # below yield strain 0.00237
    out = material._update_reference(strain, state, np.zeros((4, 6)))
    # OpenSees' first branch is the rounded curve from the origin, not a
    # line: 0.3 Pa under E eps at 0.42 eps_y
    assert out[:, 0] == pytest.approx(200.0e9 * 0.001, rel=1e-8)
