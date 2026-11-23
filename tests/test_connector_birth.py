"""A connector built by a later analysis step (1.13): it is
born free of force where its ends stand, as a beam or a bar is, and takes
only what comes after. A cantilever is loaded, then propped by a spring to
the ground, then loaded again: the prop carries its share of the second
load alone."""
import sys
from pathlib import Path

import numpy as np
import pytest

from vfife.api import run_case

E, I, L, KS = 2.0e11, 1.0e-4, 4.0, 3.0e6
KB = 3 * E * I / L ** 3


def _case(tmp_path, steps, backend="numpy", name="prop"):
    return {
        "version": 1, "name": name, "units": {"length": "m", "force": "N", "stress": "Pa"},
        "mesh": {"type": "lines", "points": {"A": [0, 0, 0], "B": [L, 0, 0], "G": [L, 0, -0.5]},
                 "lines": {"beam": {"through": ["A", "B"], "size": 1.0}, "prop": {"through": ["B", "G"], "size": 9.0}}},
        "materials": {"steel": {"type": "linear_elastic", "E": E, "nu": 0.3, "rho": 1.0}},
        "sections": {"s": {"type": "beam", "material": "steel", "shape": "general",
                           "area": 0.01, "iy": I, "iz": I, "j": 2 * I},
                     # its rotational springs make the free ground point turn with the tip:
                     # without them the point leans, and the prop with it
                     "k": {"type": "connector", "stiffness": [KS, 1e3, 1e3, 1e6, 1e6, 1e6]}},
        "assign": [{"region": "beam", "section": "s", "orientation": [0, 1, 0]},
                   {"region": "prop", "section": "k", "orientation": [0, 1, 0]}],
        # the ground point is held in position only: a new node is put where the
        # standing one carries it, rotation and all, and a HELD rotation would
        # then be brought back to zero over the step (as for a beam's new end)
        "constraints": [{"region": "A", "dofs": [1, 1, 1, 1, 1, 1]}, {"region": "G", "dofs": [1, 1, 1, 0, 0, 0]}],
        "loads": [], "steps": steps,
        "solver": {"mode": "static", "backend": backend, "tolerance": 1e-8},
        "output": {"path": str(tmp_path / f"{name}.h5"),
                   "monitors": {"w": {"kind": "displacement", "region": "B", "dof": "z"},
                                "R": {"kind": "reaction", "region": "G", "dof": "z"}}},
    }


def _load(p):
    return {"region": "B", "vector": [0, 0, -p], "keep": True}


STEPS = [{"name": "load", "loads": [_load(1.0e4)]}, {"name": "prop", "activate": ["prop"]},
         {"name": "more", "loads": [_load(2.0e4)]}]


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_a_prop_built_later_carries_only_what_comes_after(tmp_path, backend):
    if backend == "numba":
        pytest.importorskip("numba")
    out = run_case(_case(tmp_path, STEPS, backend))
    assert out.status == "CONVERGED", out.message
    w, R = np.asarray(out.monitors["w"]), np.asarray(out.monitors["R"])
    # before the prop, and with it built: the cantilever's own deflection, nothing in the prop
    assert w[9] == pytest.approx(-1.0e4 / KB, rel=2e-4) and w[19] == pytest.approx(w[9], rel=1e-6)
    assert abs(R[19]) < 1e-3 * 2.0e4
    # the second load is shared by the two stiffnesses
    assert w[-1] - w[19] == pytest.approx(-2.0e4 / (KB + KS), rel=2e-3)
    assert R[-1] == pytest.approx(2.0e4 * KS / (KB + KS), rel=2e-3)


def test_built_from_the_start_it_carries_its_share_of_everything(tmp_path):
    steps = [dict(s) for s in STEPS]; steps[1] = {"name": "prop"}
    out = run_case(_case(tmp_path, steps))
    R = np.asarray(out.monitors["R"])
    assert R[-1] == pytest.approx(3.0e4 * KS / (KB + KS), rel=2e-3)        # not 2e4's share: the test above is not vacuous


def test_removed_and_built_again_it_is_a_new_prop(tmp_path):
    steps = STEPS + [{"name": "out", "deactivate": ["prop"]}, {"name": "again", "activate": ["prop"]},
                     {"name": "third", "loads": [_load(1.0e4)]}]
    out = run_case(_case(tmp_path, steps))
    assert out.status == "CONVERGED", out.message
    w, R = np.asarray(out.monitors["w"]), np.asarray(out.monitors["R"])
    assert w[39] == pytest.approx(-3.0e4 / KB, rel=2e-4)                   # the prop gone: the cantilever alone
    assert abs(R[49]) < 1e-3 * 2.0e4                                       # built again: nothing in it
    assert R[-1] == pytest.approx(1.0e4 * KS / (KB + KS), rel=2e-3)


def test_a_run_stopped_after_the_prop_is_built_resumes(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_restart_resumes import _same, _split
    d = _case(tmp_path, STEPS)
    d.pop("name"); d["output"].pop("path")
    full, half, rest = _split(tmp_path, d, lambda s, local: s.step_index == 2 and local >= 3, progress_every=1)
    _same(full, rest)
    assert rest._conn_birth is not None and abs(rest._conn_birth).max() > 1e-3    # the 1 cm it was born with


def test_the_reference_and_the_batched_force_agree_with_an_offset(tmp_path):
    got = {}
    run_case(_case(tmp_path, STEPS), on_ready=lambda s, _p: got.__setitem__("s", s))
    s = got["s"]
    f, _ = s.internal_force()
    f_ref, _ = s._internal_force_reference()
    assert np.allclose(f, f_ref, rtol=0, atol=1e-6 * abs(f).max())


def test_a_prop_to_a_fully_fixed_point_is_born_free_too(tmp_path):
    """Found in review: the fixed point was given the tip's rotation when the
    prop was placed, the support turned it back over the step, and the prop's
    rotational springs took the whole rotation: -1054 N in the prop before
    the second load."""
    d = _case(tmp_path, STEPS)
    d["constraints"][1]["dofs"] = [1, 1, 1, 1, 1, 1]
    out = run_case(d)
    w, R = np.asarray(out.monitors["w"]), np.asarray(out.monitors["R"])
    assert abs(R[19]) < 1e-3 * 2.0e4 and w[19] == pytest.approx(w[9], rel=1e-6)
