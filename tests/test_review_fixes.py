"""The parameter-study API, the case migration and the kernels as a second
reader found them. One test per reproduced defect.
"""
import pathlib
import tempfile

import numpy as np
import pytest

from vfife import api
from vfife.core.diagnostics import AnalysisInterrupted
from vfife.io.migrate import upgrade_text

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "data" / "result_format_2" / "cantilever.yaml"


# ----------------------------------------------------------------- drafts
def test_changing_one_anchored_value_changes_only_that_one():
    """case_24d's 66 tendons share one `&fr` friction block; a deepcopy
    kept the sharing and one change changed all of them."""
    case = api.load_case(ROOT / "cases" / "case_24d_hzb_girder_piers.yaml")
    new = case.with_changes({"sections.girder.tendons[0].friction.mu": 0.4})
    mus = [t["friction"]["mu"] for t in new.get("sections.girder.tendons") if "friction" in t]
    assert mus.count(0.4) == 1 and len(mus) > 60
    assert case.get("sections.girder.tendons[0].friction.mu") != 0.4


def test_numpy_values_become_plain_numbers(tmp_path):
    case = api.load_case(FIX)
    rows = api.study(case, {"loads[0].vector[2]": np.linspace(-1e3, -2e3, 2)}, output_dir=tmp_path)
    assert [r["status"] for r in rows] == ["COMPLETED_DYNAMIC"] * 2
    assert type(case.with_changes({"loads[0].vector": np.array([0.0, 0.0, -5.0])}).get("loads[0].vector[2]")) is float


def test_a_slash_in_the_name_does_not_move_the_folder(tmp_path):
    out = api.run_case(api.load_case(FIX).with_changes({}, name="Run A/B"), output=str(tmp_path / "x.h5"))
    assert out.status == "COMPLETED_DYNAMIC"


def test_a_numeric_mapping_key_is_reachable():
    d = api.CaseDraft({"sections": {"100": {"type": "beam"}}})
    assert d.get("sections.100.type") == "beam"
    assert d.with_changes({"sections.100.type": "bar"}).get("sections.100.type") == "bar"


def test_ctrl_c_stops_the_study_and_keeps_the_rows(tmp_path, monkeypatch):
    calls = []
    real = api._run_quietly

    def interrupted_second(draft, output):
        calls.append(output)
        if len(calls) == 2:
            raise AnalysisInterrupted(output, [], {})
        return real(draft, output)
    monkeypatch.setattr(api, "_run_quietly", interrupted_second)
    with pytest.raises(AnalysisInterrupted):
        api.study(api.load_case(FIX), {"loads[0].vector[2]": [-1e3, -2e3, -3e3]}, output_dir=tmp_path)
    assert len(calls) == 2                                    # the third never started
    rows = (tmp_path / "study.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3 and "INTERRUPTED" in rows[-1]


# ---------------------------------------------------------------- migrate
@pytest.mark.parametrize("text, expect", [
    ("name: a\nsections:\n  s1: &x {type: beam, b: 0.3, h: 0.5}\n  s2: *x\n",
     "version: 1\nname: a\nsections:\n  s1: &x {type: beam, width: 0.3, height: 0.5}\n  s2: *x\n"),
    ("---\nname: a\n", "---\nversion: 1\nname: a\n"),
    ("﻿# c\nname: a\n", "﻿# c\nversion: 1\nname: a\n"),
    ("%YAML 1.1\n---\nname: a\n", "%YAML 1.1\n---\nversion: 1\nname: a\n"),
    ("{name: a, sections: {s1: {type: beam, b: 0.3}}}\n",
     "{version: 1, name: a, sections: {s1: {type: beam, width: 0.3}}}\n"),
    ("version:\nname: a\n", "version: 1\nname: a\n"),
])
def test_migrate_places_version_and_renames_where_they_belong(text, expect):
    assert upgrade_text(text)[0] == expect


def test_migrate_does_not_rewrite_a_newer_format_downwards():
    with pytest.raises(ValueError, match="newer case format"):
        upgrade_text("version: 2\nname: a\n")


# ---------------------------------------------------------------- kernels
def test_a_rebarless_fibre_section_checkpoints_its_live_state():
    """The checkpoint saved an alias the kernel's consolidated arrays had
    replaced: zeros, so a restart began from a virgin section."""
    import sys
    sys.path.insert(0, str(ROOT / "tests"))
    from test_fiber_beam_kernel import _write_homogeneous, _solver
    path = _write_homogeneous(tempfile.mkdtemp())
    for kernel in (True, False):
        sol = _solver(path, kernel=kernel)
        rng = np.random.default_rng(3)
        for _ in range(6):
            sol.q[:] += rng.normal(0.0, 2e-3, sol.ndof)
            sol.internal_force()
        ck = sol.checkpoint()
        before = sol.internal_force()[0].copy()
        assert np.abs(ck["fiber_state_0"]).max() > 0.0
        sol.q[:] += rng.normal(0.0, 2e-3, sol.ndof)
        sol.internal_force()
        sol.restore(ck)
        after = sol.internal_force()[0]
        assert np.abs(after - before).max() <= 1e-12 * np.abs(before).max()


def test_the_compiled_cable_loop_reads_its_lengths_live():
    """A stress-free length or tension-only flag changed after the first
    evaluation was ignored by a cached copy."""
    pytest.importorskip("numba")
    from vfife.cli import build, _with_element, make_solver
    from vfife.io import schema
    path = ROOT / "cases" / "case_22_cable_catenary.yaml"
    out = {}
    for backend in ("numpy", "numba"):
        case, _ = schema.load(str(path))
        mesh, mat, element = build(case, str(path))
        p = _with_element(case, element, str(path))
        p["solver"]["backend"] = backend
        s = make_solver(p, mesh, mat)
        s.internal_force()
        truss = s.member_kind == "truss2"
        s.member_Lu[truss] *= 0.97
        s.member_tension_only[:] = False
        out[backend] = s.internal_force()[0].copy()
    assert np.abs(out["numba"] - out["numpy"]).max() <= 1e-12 * np.abs(out["numpy"]).max()
    assert np.abs(out["numpy"]).max() > 0.0


def test_a_threaded_solver_leaves_numbas_thread_count_as_it_found_it():
    numba = pytest.importorskip("numba")
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("one core")
    from vfife.kernels.numba_frame import configure_threads, with_threads
    before = numba.get_num_threads()
    n = configure_threads(2)
    assert numba.get_num_threads() == before                  # nothing set at setup
    assert with_threads(n, numba.get_num_threads) == 2
    assert numba.get_num_threads() == before
