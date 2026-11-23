"""`ground_motion`: the record behind a seismic run, and what it says when the
record is not there.

The fibre path had never been walked -- `case_05` is the only case with a
`ground_motion` load and it is a `truss2`. These are the
checks that walked it.
"""
import pathlib

import numpy as np
import pytest

from vfife.core import ground


def _rec(tmp_path, text):
    p = tmp_path / "r.txt"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_the_scale_is_what_carries_the_unit(tmp_path):
    """The El Centro file shipped in `benchmarks/assets` is in **g**, so its
    cases say `scale: 9.80665`. Leaving it out is not an error anywhere: the
    structure is shaken an order of magnitude too gently and the answer looks
    reassuring, which is why the case header says so twice."""
    path = _rec(tmp_path, "0.0 0.0\n0.1 0.5\n0.2 -0.25\n")
    t, a, d = ground.read_record({"path": path, "scale": 9.80665})
    assert t == pytest.approx([0.0, 0.1, 0.2])
    assert a == pytest.approx([0.0, 4.903325, -2.4516625])
    assert list(d) == [1.0, 0.0, 0.0]                 # the default direction
    _t, a1, _d = ground.read_record({"path": path})
    assert a1 == pytest.approx([0.0, 0.5, -0.25])     # scale defaults to 1


def test_a_missing_record_says_which_file(tmp_path):
    """It used to say `no such file: None`, because `np.loadtxt` raises
    `FileNotFoundError` with `filename` unset and the CLI printed that
    attribute. A mesh in the same position names itself."""
    with pytest.raises(FileNotFoundError, match="ground_motion record not found"):
        ground.read_record({"path": str(tmp_path / "gone.dat")})


def test_a_one_column_record_is_refused(tmp_path):
    """Half the ways of exporting an accelerogram drop the time column, and a
    record with no time axis would otherwise be read as `t = a`."""
    with pytest.raises(ValueError, match="two columns"):
        ground.read_record({"path": _rec(tmp_path, "0.1\n0.2\n0.3\n")})


def test_a_time_column_that_does_not_increase_is_refused(tmp_path):
    """`np.interp` requires an increasing x and does not check: a record
    concatenated from two files, or sorted by acceleration, would be read
    silently and wrongly."""
    with pytest.raises(ValueError, match="does not increase"):
        ground.read_record({"path": _rec(tmp_path, "0.0 1.0\n0.2 2.0\n0.1 3.0\n")})


def test_the_cache_reads_the_file_once(tmp_path):
    """A fracture rebuilds the step's data, and a 20 000-line record re-read at
    every split is a cost with no answer attached."""
    path = _rec(tmp_path, "0.0 1.0\n0.1 2.0\n")
    cache = {}
    ground.read_record({"path": path}, cache)
    pathlib.Path(path).write_text("0.0 99.0\n0.1 99.0\n", encoding="utf-8")     # changed on disk
    _t, a, _d = ground.read_record({"path": path}, cache)
    assert a == pytest.approx([1.0, 2.0])                     # served from cache
    assert list(cache) == [path]


# ------------------------------------------------- the record this repo ships
ROOT = pathlib.Path(__file__).resolve().parents[1]
ELCENTRO = ROOT / "benchmarks" / "assets" / "elcentro_1940_ns_g.dat"


def test_the_shipped_el_centro_record_is_what_it_says_it_is():
    """1940 Imperial Valley, El Centro, NS. If this file were ever replaced by
    one in cm/s^2 or m/s^2, every case using it would carry on quietly with
    `scale: 9.80665` on top."""
    t, a, _d = ground.read_record({"path": str(ELCENTRO)})
    assert len(t) == 1560
    assert np.diff(t) == pytest.approx(0.02, abs=1e-9)
    assert t[-1] == pytest.approx(31.18, abs=1e-6)
    assert np.abs(a).max() == pytest.approx(0.3188, abs=5e-4)     # g, not m/s^2
    assert t[np.argmax(np.abs(a))] == pytest.approx(2.02, abs=1e-6)


# --------------------------------------------------------------- preflight
#  A base acceleration along an axis nothing is free to move in converges,
#  reports flat monitors, and takes as long as a real run to do it.
def _shaken(tmp_path, direction, plane_frame=True):
    """The miniature fibre bridge, optionally locked into its own plane, with
    a two-line accelerogram."""
    import test_fiber_connector as fc
    from vfife.cli import build
    from vfife.io import schema
    from vfife.preflight import check

    rec = tmp_path / "r.txt"
    rec.write_text("0.0 0.0\n1.0 -3.0\n100.0 -3.0\n", encoding="utf-8")
    path = fc._write(tmp_path)
    t = pathlib.Path(path).read_text(encoding="utf-8")
    plane = "  - {region: all, dofs: [0, 1, 0, 1, 0, 1]}\n" if plane_frame else ""
    t = t.replace("constraints:\n", "constraints:\n" + plane)
    t = t.replace("loads:\n", "loads:\n  - {region: all, ground_motion: "
                  f"{{path: {rec}, direction: {direction}}}}}\n")
    t = t.replace("solver: {backend: numpy, mode: relaxation",
                  "solver: {backend: numpy, mode: dynamic, total_time: 0.01")
    # a dynamic step has no load levels
    t = t.replace("  - {name: push, load_steps: 2}", "  - {name: push}")
    pathlib.Path(path).write_text(t, encoding="utf-8")
    case, _raw = schema.load(path)
    mesh, _material, element = build(case, path)
    return check(case, mesh, element)


def _codes(report):
    return {i["code"] for i in report["issues"]}


def test_shaking_a_plane_frame_out_of_its_plane_is_refused(tmp_path):
    """THE ONE THIS EXISTS FOR. `case_24f` and its seismic twin carry
    `{region: all, dofs: [0, 1, 0, 1, 0, 1]}` -- `uy` locked at every node --
    so `direction: [0, 1, 0]`, which is the obvious second run to make with a
    real record, puts the whole inertial force into the reactions. Nothing
    fails; every monitor is flat; the natural reading is that the model is
    broken somewhere else."""
    report = _shaken(tmp_path, "[0, 1, 0]")
    assert report["status"] == "FAIL"
    assert "GROUND_MOTION_FULLY_RESTRAINED" in _codes(report)
    issue = next(i for i in report["issues"]
                 if i["code"] == "GROUND_MOTION_FULLY_RESTRAINED")
    assert issue["details"]["locked"] == ["uy"]


@pytest.mark.parametrize("direction", ["[1, 0, 0]", "[0, 0, 1]",
                                       "[0.7071067811865476, 0.7071067811865476, 0]"])
def test_a_direction_with_anything_live_in_it_is_fine(tmp_path, direction):
    """Including a MIXED one: `[1, 1, 0]` (as a unit vector: since v1.0 the
    direction must be one) has a locked half and a live half,
    and the live half is a real analysis. Refusing it would be the check
    getting in the way."""
    assert "GROUND_MOTION_FULLY_RESTRAINED" not in _codes(_shaken(tmp_path, direction))


def test_without_the_plane_lock_the_transverse_direction_is_fine(tmp_path):
    """So the refusal is about THIS model's constraints, not about the y axis."""
    assert "GROUND_MOTION_FULLY_RESTRAINED" not in _codes(
        _shaken(tmp_path, "[0, 1, 0]", plane_frame=False))


def test_a_zero_direction_is_refused(tmp_path):
    # by the schema since v1.0 (a direction must be a unit vector)
    with pytest.raises(Exception, match="unit vector"):
        _shaken(tmp_path, "[0, 0, 0]")
