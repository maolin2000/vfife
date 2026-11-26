"""steel02, concrete01 and concrete02 against the OpenSees program (and,
from 1.21, the composite laws against its Parallel and Series).

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


# ------------------------------------------------------------ the composites (1.21)

COMPOSITE = os.path.join(os.path.dirname(DATA), "opensees_composite.npz")


def _composite_cases():
    with np.load(COMPOSITE) as d:
        return sorted({k.split("/")[0] for k in d.files if "/" in k})


def _part(kind, a):
    if kind == "Steel02":
        return {"type": "steel02", "Fy": a[0], "E": a[1], "b": a[2], "R0": a[3], "cR1": a[4], "cR2": a[5],
                "rho": 7850.0}
    return {"type": "linear_elastic", "E": a[0], "nu": 0.3, "rho": 7850.0}


@pytest.mark.parametrize("name", _composite_cases())
def test_the_composite_gives_the_opensees_curve(name):
    """`parallel` and `series` against OpenSees' Parallel and Series of the
    same parts, `gap` against ElasticPPGap with a yield force out of reach.
    Parallel and the gap agree to round-off. The series to 1e-7 of the peak:
    that is OpenSees' own split, not vfife's -- its stress differs from what
    its Steel02 part gives at the split it implies by up to 30 Pa (its
    iteration stops on a strain), where vfife's parts agree to 2e-4 Pa."""
    with np.load(COMPOSITE) as d:
        strain, ref, args = d[name + "/strain"], d[name + "/stress"], d[name + "/args"]
        parts = [_part(str(d[f"{name}/part{i}"]), d[f"{name}/part{i}_args"])
                 for i in range(3) if f"{name}/part{i}" in d.files]
    if name.startswith("parallel"):
        mat = create("parallel", of=parts)
    elif name.startswith("series"):
        mat = create("series", of=parts, tolerance=1e-13)
    else:
        mat = create("gap", E=args[0], hi=args[2], rho=1.0)
    state = mat.init_state(1)
    got = np.empty(len(strain))
    for i, e in enumerate(strain):
        got[i] = mat.update(np.array([[e, 0.0, 0.0, 0.0, 0.0, 0.0]]), state, 0.0)[0, 0]
    tol = 1e-7 if name.startswith("series") else 1e-12
    assert np.max(np.abs(got - ref)) <= tol * np.max(np.abs(ref)), name
