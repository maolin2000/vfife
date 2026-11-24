"""steel02, concrete01 and concrete02 against the OpenSees program.

The three laws carry the names and parameters OpenSees users know, and they
are written from the papers behind them (vfife/materials/*.py say which).
Where a paper leaves a detail open, the choice was fixed against OpenSees as
a black box; tests/data/opensees_uniaxial.npz holds its stresses over random
strain histories, made by tools/opensees_reference.py, so this suite does not
need OpenSees installed. Agreement is to round-off.
"""
import os

import numpy as np
import pytest

from vfife.materials import create

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "opensees_uniaxial.npz")


def _material(name, a):
    """The vfife material for an OpenSees argument list."""
    if name.startswith("steel02"):
        kw = dict(Fy=a[0], E=a[1], b=a[2], R0=a[3], cR1=a[4], cR2=a[5], rho=7850.0)
        if len(a) > 6:
            kw.update(a1=a[6], a2=a[7], a3=a[8], a4=a[9])
        return create("steel02", **kw), abs(a[0])
    if name.startswith("concrete01"):
        return create("concrete01", fpc=a[0], epsc0=a[1], fpcu=a[2], epscu=a[3]), abs(a[0])
    return create("concrete02", fpc=a[0], epsc0=a[1], fcu=a[2], epscu=a[3], rat=a[4],
                  ft=a[5], Ets=a[6]), abs(a[0])


def _cases():
    with np.load(DATA) as d:
        return sorted({k.split("/")[0] for k in d.files if "/" in k})


@pytest.mark.parametrize("name", _cases())
@pytest.mark.parametrize("path", ["compiled", "reference"])
def test_the_law_gives_the_opensees_curve(name, path):
    with np.load(DATA) as d:
        args, strain, ref = d[name + "/args"], d[name + "/strain"], d[name + "/stress"]
    mat, scale = _material(name, args)
    if path == "compiled":
        pytest.importorskip("numba")
        assert type(mat).__module__ and __import__(type(mat).__module__, fromlist=["_kernel"])._kernel() is not None
    state = mat.init_state(1)
    out = np.zeros((1, 6))
    got = np.empty(len(strain))
    for i, e in enumerate(strain):
        eps = np.array([[e, 0.0, 0.0, 0.0, 0.0, 0.0]])
        if path == "compiled":
            got[i] = mat.update(eps, state, 0.0)[0, 0]
        else:
            got[i] = mat._update_reference(eps, state, out)[0, 0]
    assert np.max(np.abs(got - ref)) <= 1e-12 * scale, name
