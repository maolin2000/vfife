"""Compute backends for the element force evaluation.

Both backends satisfy one contract:

    force(u, quat, state, dt, rot_iters, want_stress) -> (f_e, stress or None)

  numpy   batched tensor operations; the readable reference, and the one the
          physics tests are written against
  numba   the same algebra fused into one scalar pass per element
  warp    that same kernel on a GPU, with mixed precision (fp32 elements,
          fp64 nodal integration)

`test_backends.py` runs the same problems through both and requires they agree
to 1e-12, so the reference stays honest and the fast path stays correct.
"""

from __future__ import annotations

import numpy as np

# before any kernel is imported: a numba cache written from other sources
# is removed (kernels/_cache.py)
from ._cache import refresh as _refresh_numba_cache

_refresh_numba_cache()

from ..base.errors import UserError
from ..elements import continuum

AVAILABLE = ("numpy", "numba", "warp")


class NumpyForce:
    """Batched NumPy path. Gathers nodal displacements per call."""

    name = "numpy"

    def __init__(self, pre, conn, x0_e, material, n_nodes=None):
        self.pre, self.conn, self.x0_e = pre, conn, x0_e
        self.material = material
        # `n_nodes` is the MODEL's node count, not this block's.  Inferring it
        # from conn.max() was right while a model was one element type and
        # every node belonged to it; in a mixed mesh the hexahedra may touch
        # only the first half of the nodes, and each block's assembly has to
        # return a vector the whole model's length so the blocks can be summed.
        self.n_nodes = int(n_nodes if n_nodes is not None
                           else int(conn.max()) + 1)

    def assemble(self, f_e):
        from ..base.scatter import scatter_add
        return scatter_add(f_e, self.conn, self.n_nodes)

    def __call__(self, u, quat, state, dt, rot_iters=2, want_stress=False):
        return continuum.internal_force(
            self.pre, self.x0_e, u[self.conn], quat, self.material, state, dt,
            rot_iters=rot_iters, want_stress=want_stress,
        )


def make_force(pre, conn, x0_e, material, backend: str = "numpy",
               n_nodes=None, **kw):
    """Build the force evaluator, or say plainly why this machine cannot.

    A backend can be unavailable for reasons that are nobody's mistake: numba
    is an optional install, and warp needs an NVIDIA driver. Those used to
    surface as whatever the third-party library raised -- for a machine with no
    GPU, forty-odd lines of Warp banner followed by a bare

        RuntimeError: "cuda" device requested but CUDA is not supported by
                      the hardware or driver

    which is what anyone without an NVIDIA card met on their first run of the
    shipped RC-column case. The CLI's top-level handler only turns UserError
    into a sentence, so an environment problem escaped as a stack trace while
    an input problem got a hint. Both are things the user has to act on.
    """
    if backend == "numpy":
        return NumpyForce(pre, conn, x0_e, material, n_nodes=n_nodes)
    if backend == "numba":
        try:
            from .numba_hexa8 import NumbaForce
        except ImportError as exc:
            raise UserError(
                "BACKEND_UNAVAILABLE",
                "this case asks for the numba backend, which is not installed "
                f"here ({exc})",
                "pip install 'vfife[cpu]'  (or numba directly), or set "
                "solver.backend = numpy -- same answer, slower.",
            ) from None
        return NumbaForce(pre, conn, x0_e, material, n_nodes=n_nodes, **kw)
    if backend == "warp":
        device = kw.get("device", "cuda")
        try:
            from .warp_backend import WarpForce
        except ImportError as exc:
            raise UserError(
                "BACKEND_UNAVAILABLE",
                "this case asks for the warp backend, which is not installed "
                f"here ({exc})",
                "pip install 'vfife[gpu]'  (or warp-lang directly), or set "
                "solver.backend = numpy -- same answer, slower.",
            ) from None
        try:
            return WarpForce(pre, conn, x0_e, material, n_nodes=n_nodes, **kw)
        except UserError:
            raise
        except Exception as exc:
            # warp raises a plain RuntimeError for "no CUDA driver", and the
            # message is about its own device model rather than about the case
            raise UserError(
                "BACKEND_UNAVAILABLE",
                f"this case asks for the warp backend on device {device!r}, "
                f"which this machine cannot provide: {exc}",
                "set solver.backend = numpy to run it here -- the answer is "
                "the same, the run is slower. A GPU run needs an NVIDIA card "
                "with a working CUDA driver; `vfife capabilities` lists what "
                "each element supports.",
            ) from None
    raise UserError(
        "UNKNOWN_BACKEND",
        f"unknown solver.backend {backend!r}",
        "expected one of " + ", ".join(AVAILABLE),
    )
