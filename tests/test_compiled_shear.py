"""A fibre section's cracking shear law in the compiled step (1.7). The law went into the fused fibre kernel (the NumPy path
stays the reference, tests/test_fiber_shear.py checks the two agree), and
the compiled step runs that kernel: its shear rotation, crack strain and
largest excursion advance inside the chunk and are committed every step
as the loop commits them. Loop against compiled step, to the bit, on a
concrete cantilever with bars that cracks under a growing tip load.
"""
import os

import numpy as np
import pytest

from test_member_failure import _write_msh
from vfife.api import run_case

numba = pytest.importorskip("numba")

E = 2.0e11
CONC = {"type": "concrete02", "fpc": 3.0e7, "epsc0": 0.002, "fcu": 6.0e6, "epscu": 0.005,
        "ft": 2.0e6, "Ets": 1.0e9, "rat": 0.1, "rho": 2400}
STEEL02 = {"type": "steel02", "E": E, "Fy": 4.0e8, "b": 0.01, "R0": 20.0, "rho": 7850.0}
B, H, L, N_EL = 0.2, 0.4, 2.0, 4


def _case(tmp_path, name, P=2.0e4, total=0.05, every=50):
    nodes = [(L * i / N_EL, 0, 0) for i in range(N_EL + 1)]
    _write_msh(tmp_path / f"{name}.msh", nodes,
               {"root": (0, [0]), "tip": (0, [N_EL]), "beam": (1, [(i, i + 1) for i in range(N_EL)])})
    bars = [{"y": 0.05, "z": z, "section": "bar"} for z in (0.15, -0.15)] + \
           [{"y": -0.05, "z": z, "section": "bar"} for z in (0.15, -0.15)]
    sec = {"type": "fiber", "material": "c", "shape": "rect", "width": B, "height": H,
           "ny": 1, "nz": 10, "integration": "lobatto3", "bars": bars,
           "shear": {"stiffness": 1.0e9, "cracked_stiffness": 5.0e7}}
    return {"name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
            "mesh": {"type": "gmsh", "path": str(tmp_path / f"{name}.msh")},
            "materials": {"c": CONC, "s": STEEL02},
            "sections": {"f": sec, "bar": {"type": "bar", "material": "s", "shape": "general", "area": 3e-4}},
            "assign": [{"region": "beam", "section": "f", "orientation": [0, 1, 0]}],
            "constraints": [{"region": "root", "dofs": [1] * 6}, {"region": "all", "dofs": [0, 1, 0, 1, 0, 1]}],
            "amplitudes": {"ramp": [[0, 0], [1, 1]]},
            "loads": [{"region": "tip", "vector": [0, 0, -P], "amplitude": "ramp"}],
            "solver": {"backend": "numba", "mode": "dynamic", "total_time": total,
                       "damping": {"type": "viscous", "coeff": 5.0}},
            "output": {"path": str(tmp_path / f"{name}.h5"), "every": every, "fields": ["displacement"],
                       "monitors": {"w": {"kind": "displacement", "region": "tip", "dof": "z"},
                                    "V": {"kind": "element", "region": "beam", "field": "shear_y"}}}}


def _run(d, compiled):
    old = os.environ.get("VFIFE_COMPILED_STEP")
    os.environ["VFIFE_COMPILED_STEP"] = "1" if compiled else "0"       # "1": a short run, asked for
    box = {}
    try:
        out = run_case(d, on_ready=lambda s, p: box.setdefault("s", s))
    finally:
        if old is None:
            os.environ.pop("VFIFE_COMPILED_STEP", None)
        else:
            os.environ["VFIFE_COMPILED_STEP"] = old
    return out, box["s"]


def test_the_shear_law_advances_inside_the_compiled_step_to_the_bit(tmp_path):
    d = _case(tmp_path, "sl")
    o0, s0 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "sl0.h5"))), False)
    o1, s1 = _run(dict(d, output=dict(d["output"], path=str(tmp_path / "sl1.h5"))), True)
    assert o0.status == o1.status == "COMPLETED_DYNAMIC", (o0.message, o1.message)
    assert s1.compiled_step_refusal is None, s1.compiled_step_refusal
    assert s1._kernel is not None
    for k in ("q", "v", "orient", "_state_c", "_state_s", "_sl_gamma", "_sl_gcr_c", "_sl_gmax_c", "_sl_V",
              "end_actions"):
        assert np.array_equal(getattr(s0, k), getattr(s1, k)), k
    for k in o0.monitors:
        assert np.array_equal(o0.monitors[k], o1.monitors[k]), k
    # and it did crack, and the law carries the shear
    assert np.isfinite(s1._sl_gcr_c[0]) and s1._sl_gamma[0] != 0.0
    assert abs(float(np.asarray(o1.monitors["w"])[-1])) > 1e-4


@pytest.mark.parametrize("compiled", [False, True])
def test_the_output_interval_does_not_relax_the_shear(tmp_path, compiled):
    """An output frame reads the state: it gave the shear rotation one more
    Newton step at every frame, so the answer depended on `output.every`
    (found in the review of 1.7)."""
    ends = []
    for every in (7, 50):
        d = _case(tmp_path, f"ev{every}{int(compiled)}", every=every)
        _o, s = _run(d, compiled)
        ends.append((s.q.copy(), s._sl_gamma.copy()))
    assert np.array_equal(ends[0][0], ends[1][0])
    assert np.array_equal(ends[0][1], ends[1][1])
