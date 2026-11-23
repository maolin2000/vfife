"""The Warp twins of the two bond laws, compiled and compared.

WHY THIS FILE EXISTS SEPARATELY
    Before it, `bond_ceb_point` and `bond_cyclic_point` in
    kernels/warp_materials.py had never been COMPILED by anything. The solver
    refuses bond-slip on the warp backend (the time loop stays on the device
    and a host-side bar would slip against a frozen displacement field), and
    no element kernel instantiates a bond law, so the two device functions
    were dead source that the parameter-block tests could not reach: those
    check the numbers handed to the kernel, not the kernel.

    A device function nobody has compiled is a device function nobody knows
    parses. Driving it from a one-thread kernel costs a few seconds and turns
    "the warp path is written" into "the warp path agrees to 2e-16".

    Warp compiles the same source for CPU and CUDA, so this is a real check
    without a card. What it cannot measure is throughput.

MEASURED
    bond_ceb     max relative difference   1.5e-16
    bond_cyclic                            2.2e-16
"""
import os

import numpy as np
import pytest

from vfife.materials import create

wp = pytest.importorskip("warp", reason="warp-lang not installed")

#: Same convention as tests/test_backends.py: CPU by default so the suite runs
#: anywhere, `VFIFE_WARP_DEVICE=cuda` on a machine with a card.
WARP_DEVICE = os.environ.get("VFIFE_WARP_DEVICE", "cpu")
FC = 25.6e6

wp.init()
WTYPE = wp.float64
VEC6 = wp.types.vector(length=6, dtype=WTYPE)


def _make_kernel(point_fn):
    """One thread, one slip value, one constitutive call.

    The kernel has to be built here rather than in the test body because Warp
    reads a kernel's SOURCE to compile it, and source defined in an `exec`d
    string has none -- which is the error you get for trying this from a
    heredoc.
    """

    @wp.kernel
    def k(slip: wp.array(dtype=WTYPE), prm: wp.array(dtype=WTYPE),
          tab: wp.array(dtype=WTYPE), state: wp.array2d(dtype=WTYPE),
          i: int, out: wp.array(dtype=WTYPE)):
        e = VEC6()
        e[0] = slip[i]
        sig = point_fn(prm, tab, e, state, 0, WTYPE(1.0), WTYPE(1.0))
        out[i] = sig[0]

    return k


def _on_device(material, path, commit_each):
    kernel = _make_kernel(material.warp_point_fn())
    prm_np, tab_np = material.warp_params(np.float64)
    with wp.ScopedDevice(WARP_DEVICE):
        prm = wp.array(prm_np, dtype=WTYPE)
        tab = wp.array(tab_np, dtype=WTYPE)
        state = wp.array(np.zeros((1, material.n_state)), dtype=WTYPE)
        slip = wp.array(np.asarray(path, dtype=float), dtype=WTYPE)
        out = wp.array(np.zeros(len(path)), dtype=WTYPE)
        for i in range(len(path)):
            wp.launch(kernel, dim=1, inputs=[slip, prm, tab, state, i, out])
            if commit_each:
                # `commit` is host-side by design: it runs once per CONVERGED
                # load level, not once per iteration, so there is nothing to
                # gain from putting it on the device and a host round-trip
                # here mirrors what a solver would do.
                host = state.numpy()
                material.commit(host)
                state = wp.array(host, dtype=WTYPE)
        return out.numpy()


def _reference(material, path, commit_each):
    state = material.init_state(1)
    x = np.zeros((1, 6))
    out = np.empty(len(path))
    for i, s in enumerate(path):
        x[0, 0] = s
        out[i] = material.update(x, state, 1.0)[0, 0]
        if commit_each:
            material.commit(state)
    return out


PATH = np.concatenate([np.linspace(0, 3e-3, 120),
                       np.linspace(3e-3, -3e-3, 240),
                       np.linspace(-3e-3, 3e-3, 240),
                       np.linspace(3e-3, -3e-3, 240),
                       np.linspace(-3e-3, 12e-3, 240),
                       np.linspace(12e-3, 0, 120)])


@pytest.mark.slow
@pytest.mark.parametrize("name, commit", [("bond_ceb", False),
                                          ("bond_cyclic", True)])
def test_the_warp_twin_follows_the_reference_through_a_cyclic_path(name,
                                                                   commit):
    """Reversals, a plateau, a push past the previous maximum, and a return.

    The path matters: a monotonic one would agree even if the reversal branch
    were missing entirely, which for `bond_cyclic` is most of the law.
    """
    ref = create(name, fc=FC)
    dev = create(name, fc=FC)
    a = _reference(ref, PATH, commit)
    b = _on_device(dev, PATH, commit)
    rel = np.abs(a - b).max() / max(np.abs(a).max(), 1.0e3)
    assert rel < 1e-12, f"{name}: warp and numpy split by {rel:.2e}"


@pytest.mark.slow
def test_the_two_device_functions_share_one_envelope():
    """`bond_cyclic_point` calls `bond_ceb_envelope`, deliberately: a copy of
    the envelope in the cyclic function would be free to drift from the
    monotonic one, and the splitting mode -- a different shape with a zero
    residual -- is where it would drift first.

    Checked on the device, on a monotonic path where the cyclic law's
    degradation is inert, so any difference is the envelope itself.
    """
    push = np.linspace(0.0, 8.0e-3, 400)
    a = _on_device(create("bond_ceb", fc=FC, failure_mode="splitting"),
                   push, False)
    b = _on_device(create("bond_cyclic", fc=FC, failure_mode="splitting"),
                   push, True)
    assert np.abs(a - b).max() < 1e-9
