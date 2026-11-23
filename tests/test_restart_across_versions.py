"""A checkpoint written by an earlier release resumes on this one.

`test_restart_resumes` checks a restart with the version that wrote the
checkpoint. This file checks the other half of the promise: the
checkpoints under tests/data/restart_1_18 were written by vfife 1.18.0
(`make_fixtures.py` there, run on the v1.18.0 tag), each half way through
its run, beside the state 1.18.0 ended in when it ran the case without a
stop. Resumed on this version, the run ends in that state: to the bit in
the cloud container the fixtures were written in, and to round-off carried
through the run (1e-8 of each quantity's size) on another machine, whose
BLAS and NumPy sum in another order (conftest.py): on the
Windows machine 2800 dynamic steps drifted the energy ledger by 2e-9.
A change to what a step does moves it by far more.

The fixtures: `beam2_dynamic`, the frame solver (BeamSolver) in an explicit
dynamic run; `frame_dynamic` and `frame_static`, the same cantilever lowered
to mixed_line members (MixedStructuralSolver), dynamic and by Newton. Not a
relaxation of a frame or fibre beam: 1.19 rebuilds their relaxation mass
from the current geometry, so a 1.18 one would end in
another state, within the tolerance it relaxed to and not to the bit.

It fails when a change moves a checkpoint's layout (the layout fingerprint
refuses the restart: v1.19 found that routing a frame through
another solver by rewriting its case does) or changes what a step does.
A change that is meant to change the answer of one of these cases replaces
its fixture with one from the release that made the change, and says so.
"""
import pathlib
import shutil

import numpy as np
import pytest

from vfife import api

DATA = pathlib.Path(__file__).resolve().parent / "data" / "restart_1_18"
CASES = sorted(p.name for p in DATA.iterdir() if p.is_dir())


@pytest.mark.parametrize("name", CASES)
def test_a_1_18_checkpoint_resumes_to_the_1_18_answer(tmp_path, name):
    d = tmp_path / name
    shutil.copytree(DATA / name, d)
    got = {}
    api.run_case(d / "case.yaml", output=str(d / "rest.h5"), restart=str(d / "half.h5"),
                 progress_every=1, on_ready=lambda s, _p: got.__setitem__("rest", s))
    import h5py
    with h5py.File(d / "rest.h5", "r") as f:          # resumed, not solved again from the start
        assert "restarted_from" in f.attrs
    assert got["rest"].times and got["rest"].times[0] > 0.0
    want = np.load(d / "full_state.npz")
    have = got["rest"].checkpoint()
    assert set(want.files) <= set(have), sorted(set(want.files) - set(have))
    for k in want.files:
        a, b = np.asarray(have[k]), want[k]
        assert a.shape == b.shape, k
        if b.dtype.kind in "fc" and b.size:
            assert np.allclose(a, b, rtol=1e-8, atol=1e-8 * float(np.max(np.abs(b)))), k
        else:
            assert np.array_equal(a, b), k


def test_there_are_fixtures():
    assert CASES == ["beam2_dynamic", "frame_dynamic", "frame_static"]
