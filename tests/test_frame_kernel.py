"""The compiled kernel for the elastic frame members.

The NumPy path in `core/beam_solver.py` and `core/mixed_structural_solver.py`
is the readable reference and stays the fallback; this asserts the kernel is
the same function of the same state, on every element family that reaches it.

It is the same contract the shell kernel was held to in v0.1, and the reason
either can be trusted: a second implementation of the same physics is only
worth having if something compares them on every commit.
"""

import numpy as np
import pytest

from vfife.core.rotations import exp_map
from vfife.kernels import numba_frame
from vfife.testing import models

pytestmark = pytest.mark.skipif(not numba_frame.available(), reason="numba is not installed")

#: every element whose solver reaches the frame kernel
FRAME_ELEMENTS = ("beam2", "beam2_timoshenko", "mixed_line", "coupled", "shell_line")


def _bent(solver, seed=0, scale=5.0e-4):
    """A state with finite rotations: the rotations live in `orient` and `q`
    carries its logarithm, which is what the solvers read.

    Gentle enough that a shell4 face does not warp past its own tolerance --
    this compares two evaluations of the same physics, and a model the NumPy
    path itself refuses is not a comparison.
    """
    rng = np.random.default_rng(seed)
    q = rng.normal(scale=scale, size=(solver.nn, 6))
    q[:, :3] *= 2.0                        # translations against radians
    solver.q = q.reshape(-1).copy()
    solver.orient = exp_map(np.ascontiguousarray(q[:, 3:]))
    return solver


@pytest.mark.parametrize("element", FRAME_ELEMENTS)
def test_the_kernel_is_the_numpy_path(element):
    out = {}
    for backend in ("numpy", "numba"):
        s = _bent(models.make(models.build(element, solver={"backend": backend})))
        out[backend] = s.internal_force()[0].copy()
    scale = np.abs(out["numpy"]).max()
    assert scale > 0.0
    error = np.abs(out["numba"] - out["numpy"]).max() / scale
    assert error < 1e-12, f"{element}: {error:.3e}"


@pytest.mark.parametrize("element", ("beam2", "mixed_line"))
def test_the_end_actions_agree_too(element):
    """Not only the assembled force: the member end actions are what the
    monitors and the result file carry, and they come out of the same matvec.
    """
    out = {}
    for backend in ("numpy", "numba"):
        s = _bent(models.make(models.build(element, solver={"backend": backend})))
        s.internal_force()
        out[backend] = np.asarray(s.end_actions).copy()
    scale = np.abs(out["numpy"]).max()
    assert np.abs(out["numba"] - out["numpy"]).max() / scale < 1e-12


def test_a_collapsed_member_is_refused_by_the_kernel_too():
    """The zero-length guard is in the NumPy path; a kernel that silently
    divided by zero instead would put a NaN in the force and the run would
    fail somewhere else entirely."""
    s = models.make(models.build("mixed_line", solver={"backend": "numba"}))
    conn = s.mesh.elements[0]
    q = np.zeros((s.nn, 6))
    q[conn[1], :3] = s.mesh.nodes[conn[0]] - s.mesh.nodes[conn[1]]
    s.q = q.reshape(-1)
    with pytest.raises(ValueError, match="zero length"):
        s.internal_force()


def test_the_shell_line_halves_each_take_their_own_backend():
    """`shell_line` is two solvers on one node set. Under `numba` both halves
    compile; in v0.1 the line half was pinned to numpy because it had no
    kernel, and leaving that in would have thrown v0.1 away silently.
    """
    s = models.make(models.build("shell_line", solver={"backend": "numba"}))
    assert s.backend == "numba"
    assert s.line.backend == "numba"
    plain = models.make(models.build("shell_line", solver={"backend": "numpy"}))
    assert plain.line.backend == "numpy"


def test_an_unknown_backend_still_says_so():
    with pytest.raises(ValueError, match="backend"):
        models.make(models.build("beam2", solver={"backend": "opencl"}))
