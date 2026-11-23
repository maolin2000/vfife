"""Make `pytest` work from a clean checkout, without an install or PYTHONPATH.

pytest puts the rootdir on sys.path only when it finds a conftest.py there, so
this file existing is the whole point of it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------- OpenBLAS's kernel
# The NumPy reference path's matrix products go through OpenBLAS, which picks
# its kernel by the CPU. On an AVX-512 machine it takes the SkylakeX kernel,
# whose sums round differently (1e-15) from the AVX2 (Haswell) kernel; the
# tests that hold the numba kernels to the reference to the bit, and two fibre
# creep runs that sit near their convergence limit, were written on AVX2
# machines and failed on an AVX-512 one -- the four failures the cloud
# container showed since 1.8 (test_temperature, test_review5_fibre). The tests
# pin the AVX2 kernel there; only where AVX2 is present (forcing it on a CPU
# without would stop the process), only on Linux (NumPy on Windows from conda
# need not be OpenBLAS), never over a setting made by hand, and before NumPy
# is first imported. Not covered: Windows with a pip (OpenBLAS) NumPy on an
# AVX-512 CPU, and MKL builds (MKL_CBWR) -- the user's machine is neither.
# The two fibre creep runs are worth a look of their own: under the other
# kernel a relaxation stage stalls at residual 0.7 for 100000 iterations,
# a convergence that depends on round-off.
def _pin_openblas():
    if not sys.platform.startswith("linux") or "OPENBLAS_CORETYPE" in os.environ:
        return
    if "numpy" in sys.modules:
        import warnings
        warnings.warn("conftest: NumPy was imported before the OpenBLAS kernel could be pinned; on an "
                      "AVX-512 machine four bitwise/creep tests may fail (see conftest.py)")
        return
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            flags = next((l.split(":", 1)[1].split() for l in fh if l.startswith("flags")), [])
    except OSError:
        return
    if "avx512f" in flags and "avx2" in flags and "fma" in flags:
        os.environ["OPENBLAS_CORETYPE"] = "Haswell"


_pin_openblas()


# --------------------------------------------------------------- heavy tests
# Tests listed in tests/heavy_tests.txt (more than 3 s each) are marked
# `heavy`. They stay in the default layer -- `pytest` and tools/check_all.py
# run them -- but the GitHub fast tier deselects them with
# `-m "not slow and not heavy"`: on a two-core runner the whole default
# layer did not fit the job's time limit.
_HEAVY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "heavy_tests.txt")


def _heavy_ids():
    try:
        with open(_HEAVY_FILE, encoding="utf-8") as fh:
            return {l.strip() for l in fh if l.strip() and not l.startswith("#")}
    except OSError:
        return set()


def pytest_collection_modifyitems(config, items):
    import pytest
    heavy = _heavy_ids()
    if not heavy:
        return
    root = os.path.dirname(os.path.abspath(__file__))
    for item in items:
        rel = os.path.relpath(str(item.fspath), root).replace(os.sep, "/")
        if f"{rel}::{item.name}" in heavy or item.nodeid in heavy:
            item.add_marker(pytest.mark.heavy)
