"""Threads for the frame kernel and a compiled loop for the axial members,
on the Xihoumen model: 851 frame members and 356 cable
and hanger members, the model the two were written for.

The threaded kernel must give the serial one's answer to the bit -- every
member writes its own row and the rows are added in member order -- and the
compiled axial loop must give the NumPy path's answer to round-off, slack
cables included.
"""
from pathlib import Path

import numpy as np
import pytest

from vfife.cli import build, _with_element, make_solver
from vfife.io import schema

numba = pytest.importorskip("numba")
CASE = Path(__file__).resolve().parents[1] / "cases" / "_xihoumen.yaml"


def _force(backend, threads=1, slack=False):
    case, _ = schema.load(str(CASE))
    mesh, mat, element = build(case, str(CASE))
    p = _with_element(case, element, str(CASE))
    p["solver"]["backend"] = backend
    p["solver"]["threads"] = threads
    s = make_solver(p, mesh, mat)
    if slack:
        # every axial member a cable, and every other one given a stress-free
        # length 1% longer than it is: those are shortened and carry nothing
        truss = np.flatnonzero(s.member_kind == "truss2")
        s.member_tension_only[:] = False
        s.member_tension_only[truss] = True
        s.member_Lu[truss[::2]] *= 1.01
    s.q[:] += np.random.default_rng(1).standard_normal(s.ndof) * 1e-2
    s.sync_orientation()
    f, actions = s.internal_force()
    return f.copy(), actions.copy(), s.end_actions.copy(), s


def test_the_compiled_axial_loop_is_the_numpy_path():
    for slack in (False, True):
        f0, a0, e0, s = _force("numpy", slack=slack)
        f1, a1, e1, _ = _force("numba", slack=slack)
        assert np.abs(f1 - f0).max() <= 1e-14 * np.abs(f0).max()
        assert np.abs(e1 - e0).max() <= 1e-14 * np.abs(e0).max()
    truss = s.member_kind == "truss2"
    slack_members = truss & (e0[:, 6] == 0.0)
    assert slack_members.sum() >= (s.member_kind == "truss2").sum() // 2   # the slack branch ran


def test_threads_give_the_serial_frame_kernels_answer_to_the_bit():
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("one core")
    f1, a1, e1, _ = _force("numba", 1)
    f2, a2, e2, _ = _force("numba", 2)
    assert np.array_equal(f1, f2) and np.array_equal(a1, a2) and np.array_equal(e1, e2)


def test_threads_on_a_pure_frame_model(tmp_path):
    """beam_solver's path (every member a beam2), not the mixed one."""
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("one core")
    root = Path(__file__).resolve().parents[1]
    case_path = root / "cases" / "case_16_lee_frame.yaml"
    out = []
    for n in (1, 2):
        case, _ = schema.load(str(case_path))
        mesh, mat, element = build(case, str(case_path))
        p = _with_element(case, element, str(case_path))
        p["solver"]["backend"] = "numba"
        p["solver"]["threads"] = n
        s = make_solver(p, mesh, mat)
        s.q[:] += np.random.default_rng(2).standard_normal(s.ndof) * 1e-3
        s.sync_orientation()
        out.append(s.internal_force()[0].copy())
    assert np.array_equal(out[0], out[1])
